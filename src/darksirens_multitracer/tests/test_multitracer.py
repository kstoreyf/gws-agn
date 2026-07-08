"""G4 parity + unit tests for darksirens_multitracer.

Parity needs the working/gw_agn s0_smoke data on disk (produced by the gate
program); tests skip cleanly if it is absent. Run from the repo root:

    python -m pytest src/darksirens_multitracer/tests -q

Precision note: the legacy pipeline is mixed-precision by import order
(catalogs + cosmology tables float32, samples + likelihood float64 after
generate_gwsamples flips jax_enable_x64). The fixture reproduces that
order exactly for both sides — see core.enable_x64.
"""
import os
import sys

import numpy as np
import pytest

HERE = os.path.abspath(os.path.dirname(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
CODE = os.path.join(REPO, 'code')
S0 = os.path.join(REPO, 'working', 'gw_agn', 'data', 's0_uniform')

# The repo root carries a stale non-JAX-safe jaxinterp2d.py copy that shadows
# the installed package when pytest runs from the root (and pytest startup
# preloads it into sys.modules) — strip both so the installed one is used.
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != REPO]
sys.path.insert(0, os.path.join(REPO, 'src'))
sys.path.insert(0, CODE)
_j2d = sys.modules.get('jaxinterp2d')
if _j2d is not None and 'site-packages' not in getattr(_j2d, '__file__', ''):
    del sys.modules['jaxinterp2d']

from darksirens_multitracer import core  # noqa: E402

FN_GAL = os.path.join(S0, 'cat_gal_pixelated_nside32.h5')
FN_AGN = os.path.join(S0, 'cat_agn_pixelated_nside32.h5')
FN_GW = os.path.join(S0, 'gwsamples_fagn0.5_lam0.5_seedgw1007_dLunc0.1_obs.h5')
have_data = all(os.path.exists(p) for p in (FN_GAL, FN_AGN, FN_GW))

Z_MAX_GW = 1.0
DZ = 3e-3
SEED = 20260708


@pytest.fixture(scope='module')
def both_sides():
    """Build legacy and package objects in the production precision order."""
    import run_inference as legacy

    # ---- float32 phase (matches run_inference.main before load_gw_samples)
    cat = legacy.load_catalog_data(FN_GAL, FN_AGN, nside=32, Dz_gal=DZ, Dz_agn=DZ)
    zgs, zgc, zas, zac = legacy.precompute_beta_cdf(cat)
    cat['z_gal_sorted'], cat['z_gal_cdf'] = zgs, zgc
    cat['z_agn_sorted'], cat['z_agn_cdf'] = zas, zac
    cat['selection_mode'] = 'fixed_z'
    cat['z_max_gw'] = float(Z_MAX_GW)
    cat['beta_gal_fixed'] = float(np.interp(Z_MAX_GW, zgs, zgc))
    cat['beta_agn_fixed'] = float(np.interp(Z_MAX_GW, zas, zac))
    cat['dL_max'] = None
    prob = legacy.create_catalog_probability_functions(cat)
    cosmo_leg = legacy.setup_cosmology()

    tracers = [core.load_tracer(FN_GAL, DZ, 'gal'), core.load_tracer(FN_AGN, DZ, 'agn')]
    betas = [core.tracer_beta_fixed(t, Z_MAX_GW) for t in tracers]
    prior = core.build_prior_functions(tracers)
    cosmo_pkg = core.setup_cosmology()

    # ---- x64 phase (load_gw_samples' lazy generate_gwsamples import flips it)
    core.enable_x64()
    np.random.seed(SEED)
    gw_leg = legacy.load_gw_samples(FN_GW)
    np.random.seed(SEED)
    gw_pkg = core.load_gw_samples(FN_GW)

    return dict(legacy=legacy, cat=cat, prob=prob, cosmo_leg=cosmo_leg,
                gw_leg=gw_leg, tracers=tracers, betas=betas, prior=prior,
                cosmo_pkg=cosmo_pkg, gw_pkg=gw_pkg)


@pytest.mark.skipif(not have_data, reason='s0_smoke workspace data not on disk')
def test_g4_parity_bitforbit(both_sides):
    """Package 2-tracer grid == legacy run_inference.py grid, exact equality."""
    b = both_sides
    H0_grid = np.linspace(62.0, 74.0, 4)
    alpha_grid = np.array([0.0, 0.3, 0.7, 1.0])

    grid_leg = b['legacy'].compute_likelihood_grid(
        b['gw_leg'], b['cat'], b['cosmo_leg'], b['prob'],
        H0_grid, alpha_grid, progress=False)
    grid_pkg = core.compute_likelihood_grid(
        b['gw_pkg'], b['tracers'], b['cosmo_pkg'], 32, Z_MAX_GW,
        H0_grid, alpha_grid, betas=b['betas'], progress=False)

    assert b['betas'][0] == b['cat']['beta_gal_fixed']
    assert b['betas'][1] == b['cat']['beta_agn_fixed']
    assert np.array_equal(grid_leg, grid_pkg), (
        'max |diff| = {}'.format(np.nanmax(np.abs(grid_leg - grid_pkg))))


@pytest.mark.skipif(not have_data, reason='s0_smoke workspace data not on disk')
def test_pair_vs_general_path(both_sides):
    """K-general logsumexp path agrees with the pair path (float tolerance)."""
    b = both_sides
    ind = core.compute_pixel_indices(b['gw_pkg']['ra'], b['gw_pkg']['dec'], 32)
    for alpha in (0.0, 0.3, 1.0):
        args = (b['gw_pkg'], b['prior'], b['cosmo_pkg'], ind,
                [1 - alpha, alpha], b['betas'], Z_MAX_GW)
        lp = core.compute_log_likelihood(*args, H0=68.0, use_pair_path=True)
        lg = core.compute_log_likelihood(*args, H0=68.0, use_pair_path=False)
        assert np.isclose(float(lp), float(lg), rtol=1e-6, atol=1e-3)


@pytest.mark.skipif(not have_data, reason='s0_smoke workspace data not on disk')
def test_beta_monotone_and_bounded(both_sides):
    for t in both_sides['tracers']:
        b_lo = core.tracer_beta_fixed(t, 0.5)
        b_hi = core.tracer_beta_fixed(t, 1.4)
        assert 0.0 < b_lo < b_hi <= 1.0


def test_effective_weights_algebra():
    w_eff, s = core._effective_weights([0.7, 0.3], [0.5, 0.25])
    # a = [1.4, 1.2], s = 2.6
    assert np.isclose(s, 2.6)
    assert np.allclose(w_eff, [1.4 / 2.6, 1.2 / 2.6])
    assert np.isclose(w_eff.sum(), 1.0)
    # boundary weights stay exact
    w_eff, s = core._effective_weights([0.0, 1.0], [0.5, 0.25])
    assert w_eff[0] == 0.0 and np.isclose(w_eff[1], 1.0) and np.isclose(s, 4.0)
