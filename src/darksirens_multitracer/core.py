"""Core multitracer dark-siren likelihood (fixed-z selection).

Model (working/gw_agn/GOAL.md section 3, adapted to K tracers):

    p(d_i | H0, w) = sum_k w_k * N_ik / beta_k,
    N_ik = mean_s [ p_k(z_s, pix_s) * 1{z_s <= z_max_gw} / (ddL/dz)_s ],
    beta_k = CDF_k(z_max_gw)   (H0-independent constants for a fixed-true-z cut),

with p_k the full-catalog-normalized per-pixel KDE field of tracer k and
w on the (K-1)-simplex. Internally the constant betas are absorbed by the
effective-weight rewrite w_eff_k = (w_k/beta_k)/s, s = sum_k w_k/beta_k,
adding N_gw*log(s) to the log-likelihood.

The two-tracer path reproduces code/run_inference.py bit-for-bit (gate G4):
the per-tracer kernels and the mixture op order are lifted verbatim. The
K-tracer path is the same algebra via logsumexp (equal within float
tolerance, not bitwise, for K=2).
"""
import numpy as np
import h5py

from jax import jit, vmap
from jax import numpy as jnp
from jax.scipy.special import logsumexp
from jax.scipy.stats import norm

import astropy.units as u
import astropy.constants as constants
from astropy.cosmology import Planck15, FlatLambdaCDM
from jaxinterp2d import interp2d
from tqdm import tqdm

try:
    import healpy as hp
except ImportError:  # pragma: no cover - healpy is required for pixel lookups
    hp = None


def enable_x64():
    """Switch JAX to 64-bit floats.

    The legacy pipeline runs MIXED precision by accident of import order:
    the pixelized catalogs and the cosmology tables are built in float32,
    then run_inference.load_gw_samples lazily imports generate_gwsamples,
    whose module-level jax_enable_x64 flips everything after that point
    (GW sample arrays, all likelihood arithmetic) to float64. For
    bit-parity with validated legacy results, call in this order:
    load_tracer / setup_cosmology / build_prior_functions -> enable_x64()
    -> load_gw_samples -> evaluate.
    """
    import jax
    jax.config.update('jax_enable_x64', True)


# ----------------------------------------------------------------------
# Cosmology (lifted verbatim from code/run_inference.py::setup_cosmology
# for bit-parity; Planck15 fiducial grids, Om0 interpolation table).
# ----------------------------------------------------------------------
def setup_cosmology(zMax_1=0.5, zMax_2=5, Om0_range=0.1, n_Om0=100):
    H0_fiducial = Planck15.H0.value
    Om0_fiducial = Planck15.Om0
    speed_of_light = constants.c.to('km/s').value

    zgrid_1 = np.expm1(np.linspace(np.log(1), np.log(zMax_1 + 1), 5000))
    zgrid_2 = np.expm1(np.linspace(np.log(zMax_1 + 1), np.log(zMax_2 + 1), 1000))
    zgrid = np.concatenate([zgrid_1, zgrid_2])

    Om0grid = jnp.linspace(Om0_fiducial - Om0_range, Om0_fiducial + Om0_range, n_Om0)
    rs = []
    for Om0 in tqdm(Om0grid):
        cosmo = FlatLambdaCDM(H0=H0_fiducial, Om0=Om0)
        rs.append(cosmo.comoving_distance(zgrid).to(u.Mpc).value)

    zgrid = jnp.array(zgrid)
    rs = jnp.asarray(rs)
    rs = rs.reshape(len(Om0grid), len(zgrid))

    @jit
    def E(z, Om0=Om0_fiducial):
        return jnp.sqrt(Om0 * (1 + z)**3 + (1.0 - Om0))

    @jit
    def r_of_z(z, H0, Om0=Om0_fiducial):
        return interp2d(Om0, z, Om0grid, zgrid, rs) * (H0_fiducial / H0)

    @jit
    def dL_of_z(z, H0, Om0=Om0_fiducial):
        return (1 + z) * r_of_z(z, H0, Om0)

    @jit
    def z_of_dL(dL, H0, Om0=Om0_fiducial):
        return jnp.interp(dL, dL_of_z(zgrid, H0, Om0), zgrid)

    @jit
    def ddL_of_z(z, dL, H0, Om0=Om0_fiducial):
        return dL / (1 + z) + speed_of_light * (1 + z) / (H0 * E(z, Om0))

    return {
        'zgrid': zgrid, 'Om0grid': Om0grid, 'rs': rs, 'E': E,
        'r_of_z': r_of_z, 'dL_of_z': dL_of_z, 'z_of_dL': z_of_dL,
        'ddL_of_z': ddL_of_z,
        'H0_fiducial': H0_fiducial, 'Om0_fiducial': Om0_fiducial,
    }


# ----------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------
def load_gw_samples(fn_gwsamples, N_gw_inf=None):
    """Load PE-sample clouds (host-type-blind), matching the legacy loader:
    concatenate the per-type blocks, shuffle events with np.random (seed the
    global RNG beforehand for reproducibility), optionally truncate, flatten.
    """
    def _blocks(f, name):
        a, b = f[name + '_gal'][:], f[name + '_agn'][:]
        # a host type with zero events is stored as an empty 1-D array
        if a.ndim != 2:
            a = np.empty((0, b.shape[1]))
        if b.ndim != 2:
            b = np.empty((0, a.shape[1]))
        return np.concatenate([a, b], axis=0)

    with h5py.File(fn_gwsamples, 'r') as f:
        ra = _blocks(f, 'ra')
        dec = _blocks(f, 'dec')
        dL = _blocks(f, 'dL')

    N_gw_total = ra.shape[0]
    shuffle_indices = np.random.permutation(N_gw_total)
    ra = ra[shuffle_indices]
    dec = dec[shuffle_indices]
    dL = dL[shuffle_indices]

    if N_gw_inf is not None:
        if N_gw_inf > N_gw_total:
            raise ValueError('N_gw_inf ({}) exceeds available events ({})'.format(
                N_gw_inf, N_gw_total))
        ra, dec, dL = ra[:N_gw_inf], dec[:N_gw_inf], dL[:N_gw_inf]

    ra_flat = ra.flatten()
    return {
        'ra': jnp.array(ra_flat),
        'dec': jnp.array(dec.flatten()),
        'dL': jnp.array(dL.flatten()),
        'p_pe': jnp.ones(len(ra_flat)),
        'N_samples_gw': ra.shape[1] if len(ra) > 0 else 0,
        'N_gw': ra.shape[0] if len(ra) > 0 else 0,
    }


def load_tracer(fn_pixelated, Dz, name=None):
    """Load one pixelized tracer catalog (pixelize_catalogs.py schema)."""
    with h5py.File(fn_pixelated, 'r') as f:
        z = jnp.asarray(f['z'])
        n = jnp.asarray(f['n_in_pixel'])
    return {
        'name': name or fn_pixelated,
        'z': z,
        'dz': Dz * (1 + z),
        'w': jnp.ones(z.shape),
        'n': n,
        'npix': len(z),
    }


def tracer_beta_fixed(tracer, z_max_gw):
    """beta_X = CDF_X(z_max_gw): H0-independent selection constant for a
    hard cut on TRUE host redshift (empirical CDF of the tracer catalog).
    The CDF array goes through jnp (float32 by default) to reproduce the
    legacy precompute_beta_cdf quantization bit-for-bit."""
    z = np.asarray(tracer['z']).ravel()
    z = np.sort(z[np.isfinite(z)])
    # The CDF goes through jnp exactly as legacy precompute_beta_cdf does.
    # NOTE (bit-parity): call this BEFORE enable_x64() — the legacy pipeline
    # builds its CDFs in the float32 phase, and the resulting quantization
    # (1e-7 relative, physically inert) is part of the parity contract.
    cdf = np.asarray(jnp.arange(1, len(z) + 1, dtype=float) / len(z))
    return float(np.interp(z_max_gw, z, cdf))


def compute_pixel_indices(ra, dec, nside):
    if hp is None:
        raise ImportError('healpy is required for pixel index computation')
    # Identical expression to the legacy pipeline (bit-parity).
    return hp.pixelfunc.ang2pix(nside, np.pi / 2 - dec, ra)


# ----------------------------------------------------------------------
# Per-tracer kernels and the mixture prior
# ----------------------------------------------------------------------
def _build_tracer_kernel(tracer):
    """Per-(z, pix) log density kernel — lifted verbatim from the legacy
    logpcatalog_* closures (same ops, same order, for bit-parity)."""
    Z = tracer['z']
    DZ = tracer['dz']
    W = tracer['w']

    @jit
    def logp(z, pix, Om0, gamma):
        zs = Z[pix]
        ddzs = DZ[pix]
        valid_mask = jnp.isfinite(zs)
        nobj = jnp.sum(valid_mask)
        wts = W[pix] * (1 + zs)**(gamma)
        wts_valid = jnp.where(valid_mask, wts, 0.0)
        wts_sum = jnp.sum(wts_valid)
        wts_normalized = jnp.where(wts_sum > 0, wts_valid / wts_sum, 0.0)
        log_wts = jnp.where(valid_mask & (wts_normalized > 0),
                            jnp.log(wts_normalized) + norm.logpdf(z, zs, ddzs),
                            -jnp.inf)
        log_prob = logsumexp(log_wts)
        log_prob = jnp.where(nobj > 0, log_prob, -1e10)
        # Return the integer COUNT (not wts_sum), exactly as the legacy kernel
        # does: downstream `count + 1e-10` promotes to f64 under x64, and that
        # dtype path is part of the bit-parity contract. Weighted (non-unit)
        # tracer weights would need wts_sum here — future work, as in legacy.
        return log_prob, nobj

    return jit(vmap(logp, in_axes=(0, 0, None, None), out_axes=0))


def build_prior_functions(tracers):
    """Mixture prior over K tracers.

    tracers: ordered list of tracer dicts (load_tracer). The mixture weight
    convention for K=2 follows the legacy pipeline: the scalar mixture
    parameter alpha is the weight of tracers[1] (the 'AGN-like' slot) and
    (1-alpha) that of tracers[0]; this path is bit-parity with legacy.
    For K>2 pass a weight vector (simplex) and the same algebra runs via
    logsumexp (allclose, not bitwise, for K=2).
    """
    kernels = [_build_tracer_kernel(t) for t in tracers]
    W_totals = [jnp.array(float(jnp.sum(t['n']))) for t in tracers]

    def log_prior_pair(z, pix, alpha, Om0, gammas):
        # Legacy op order: secondary (alpha slot) first, then primary.
        logp_b, n_b = kernels[1](z, pix, Om0, gammas[1])
        logp_a, n_a = kernels[0](z, pix, Om0, gammas[0])
        logp_b = jnp.where(n_b > 0, logp_b, -1e10)
        logp_a = jnp.where(n_a > 0, logp_a, -1e10)
        log_alpha = jnp.where(alpha > 1e-10, jnp.log(alpha), -1e10)
        log_1malpha = jnp.where(alpha < 1.0 - 1e-10, jnp.log1p(-alpha), -1e10)
        term_b = log_alpha + jnp.log(n_b + 1e-10) - jnp.log(W_totals[1] + 1e-10) + logp_b
        term_a = log_1malpha + jnp.log(n_a + 1e-10) - jnp.log(W_totals[0] + 1e-10) + logp_a
        return jnp.logaddexp(term_b, term_a)

    def log_prior_general(z, pix, weights, Om0, gammas):
        terms = []
        for k, kern in enumerate(kernels):
            logp_k, n_k = kern(z, pix, Om0, gammas[k])
            logp_k = jnp.where(n_k > 0, logp_k, -1e10)
            w = weights[k]
            log_w = jnp.where(w > 1e-10, jnp.log(w), -1e10)
            terms.append(log_w + jnp.log(n_k + 1e-10) - jnp.log(W_totals[k] + 1e-10) + logp_k)
        return logsumexp(jnp.stack(terms, axis=0), axis=0)

    return {'pair': log_prior_pair, 'general': log_prior_general, 'K': len(tracers)}


# ----------------------------------------------------------------------
# Likelihood
# ----------------------------------------------------------------------
def _effective_weights(weights, betas):
    a = np.asarray(weights, dtype=float) / np.asarray(betas, dtype=float)
    s = a.sum()
    return a / s, s


def compute_log_likelihood(gw_data, prior_funcs, cosmo_funcs, samples_ind,
                           weights, betas, z_max_gw,
                           H0, Om0=None, gammas=None, use_pair_path=None):
    """Fixed-z-selection log likelihood at one parameter point.

    weights: length-K simplex (K=2: [1-alpha, alpha]); betas: length-K
    CDF_k(z_max_gw) constants (tracer_beta_fixed).
    """
    if Om0 is None:
        Om0 = cosmo_funcs['Om0_fiducial']
    K = prior_funcs['K']
    if gammas is None:
        # Integer zeros on purpose: (1+z)**0 traces to integer_pow and keeps
        # the float32 weights; a float exponent would promote them to f64
        # and break bit-parity with the legacy pipeline.
        gammas = [0] * K
    if use_pair_path is None:
        use_pair_path = (K == 2)

    dL = gw_data['dL']
    p_pe = gw_data['p_pe']
    N_samples_gw = gw_data['N_samples_gw']
    N_gw = gw_data['N_gw']

    w_eff, mix_norm = _effective_weights(weights, betas)

    z = cosmo_funcs['z_of_dL'](dL, H0, Om0)
    if use_pair_path:
        logprior = prior_funcs['pair'](z, samples_ind, float(w_eff[1]), Om0, gammas)
    else:
        logprior = prior_funcs['general'](z, samples_ind, jnp.asarray(w_eff), Om0, gammas)

    log_weights = (
        -jnp.log(cosmo_funcs['ddL_of_z'](z, dL, H0, Om0))
        - jnp.log(p_pe)
        + logprior
    )
    log_weights = jnp.where(z <= z_max_gw, log_weights, -jnp.inf)
    log_weights = log_weights.reshape((N_gw, N_samples_gw))
    ll = jnp.sum(-jnp.log(N_samples_gw) + logsumexp(log_weights, axis=-1))
    return ll + N_gw * jnp.log(mix_norm)


def compute_likelihood_grid(gw_data, tracers, cosmo_funcs, nside, z_max_gw,
                            H0_grid, alpha_grid, Om0=None, gammas=None,
                            betas=None, progress=True):
    """2-D (H0, alpha) grid for the two-tracer case (alpha = weight of
    tracers[1]). Returns np.ndarray (len(H0_grid), len(alpha_grid)).

    betas: per-tracer CDF_X(z_max_gw) constants. Pass values computed with
    tracer_beta_fixed BEFORE enable_x64() for legacy bit-parity; if None
    they are recomputed here under the current precision state."""
    if len(tracers) != 2:
        raise ValueError('the 2-D grid driver is for tracer PAIRS; use '
                         'compute_log_likelihood directly for K>2')
    prior_funcs = build_prior_functions(tracers)
    if betas is None:
        betas = [tracer_beta_fixed(t, z_max_gw) for t in tracers]
    samples_ind = compute_pixel_indices(gw_data['ra'], gw_data['dec'], nside)

    H0_grid = jnp.asarray(H0_grid)
    alpha_grid = jnp.asarray(alpha_grid)
    out = jnp.zeros((len(H0_grid), len(alpha_grid)))
    pbar = tqdm(total=len(H0_grid) * len(alpha_grid),
                desc='multitracer grid') if progress else None
    for i, H0 in enumerate(H0_grid):
        for j, alpha in enumerate(alpha_grid):
            ll = compute_log_likelihood(
                gw_data, prior_funcs, cosmo_funcs, samples_ind,
                weights=[1.0 - float(alpha), float(alpha)], betas=betas,
                z_max_gw=z_max_gw, H0=float(H0), Om0=Om0, gammas=gammas)
            out = out.at[i, j].set(float(ll))
            if pbar:
                pbar.update(1)
    if pbar:
        pbar.close()
    return np.array(out)
