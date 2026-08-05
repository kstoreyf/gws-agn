"""Isolate the gws-agn H0 bias mechanisms in a catalog-free toy.

Mirrors the pipeline's generative model: hosts uniform-in-comoving-volume
with a HARD z <= z_max_gw cut (selection acts on the TRUE redshift), PE
clouds Gaussian in dL with sigma = fac * dL, flat implied PE prior
(p_pe = 1), estimator = mean_s[ p_cat(z_s) / (ddL/dz)_s ] per event with a
global beta(H0) division, beta = CDF_cat(z_max_det(H0)),
z_max_det = z(dL_max | H0), dL_max = dL(z_max_gw | H0_true).

Variants:
  A  current pipeline: truth-centered PE clouds, NO detection indicator in
     the numerator (only the beta division)
  B  truth-centered + indicator 1{z_s <= z_max_det(H0)} in the numerator
  C  obs-centered PE clouds (data = truth + noise; samples around data),
     WITH the indicator  -> the statistically consistent estimator
  D  obs-centered, no indicator (isolates the indicator's role)
"""
import numpy as np

rng_global = np.random.default_rng(20260708)

H0_TRUE = 67.74
OM0 = 0.3075
C_LIGHT = 299792.458
Z_MAX_GW = 1.0
DL_UNC_FAC = 0.1          # sigma_dL = fac * dL (pipeline's linear convention)
N_GW = 200
N_SAMP = 256
N_REAL = 60
H0_GRID = np.linspace(50, 100, 201)

zf = np.linspace(1e-4, 3.0, 4000)
Ez = np.sqrt(OM0 * (1 + zf) ** 3 + (1 - OM0))
Dc_over_c = np.concatenate([[0], np.cumsum(0.5 * (1 / Ez[1:] + 1 / Ez[:-1]) * np.diff(zf))])


def dL_of_z(z, H0):
    return (1 + z) * (C_LIGHT / H0) * np.interp(z, zf, Dc_over_c)


def z_of_dL(dL, H0):
    grid = dL_of_z(zf, H0)
    return np.interp(dL, grid, zf)


def ddL_of_z(z, H0):
    Dc = (C_LIGHT / H0) * np.interp(z, zf, Dc_over_c)
    E = np.sqrt(OM0 * (1 + z) ** 3 + (1 - OM0))
    return Dc + (1 + z) * (C_LIGHT / H0) / E


# Uniform-in-comoving-volume catalog density (continuum): p_cat(z) ~ Dc^2 / E
p_cat_un = (np.interp(zf, zf, Dc_over_c)) ** 2 / Ez
p_cat_un /= np.trapz(p_cat_un, zf)
cdf_cat = np.concatenate([[0], np.cumsum(0.5 * (p_cat_un[1:] + p_cat_un[:-1]) * np.diff(zf))])
cdf_cat /= cdf_cat[-1]


def p_cat(z):
    return np.interp(z, zf, p_cat_un, left=0.0, right=0.0)


def draw_hosts(n, rng):
    """Hosts ~ catalog density truncated at Z_MAX_GW (mock's hard cut)."""
    u_max = np.interp(Z_MAX_GW, zf, cdf_cat)
    return np.interp(rng.uniform(0, u_max, n), cdf_cat, zf)


DL_MAX = dL_of_z(Z_MAX_GW, H0_TRUE)


def run_variant(obs_centered, use_indicator, rng):
    z_hosts = draw_hosts(N_GW, rng)
    dL_true = dL_of_z(z_hosts, H0_TRUE)
    sig = DL_UNC_FAC * dL_true
    if obs_centered:
        d_obs = rng.normal(dL_true, sig)
        # pipeline convention: sample std tied to the center used
        dL_s = rng.normal(d_obs[:, None], (DL_UNC_FAC * np.abs(d_obs))[:, None],
                          (N_GW, N_SAMP))
    else:
        dL_s = rng.normal(dL_true[:, None], sig[:, None], (N_GW, N_SAMP))
    dL_s = np.abs(dL_s)

    lls = np.empty_like(H0_GRID)
    for k, H0 in enumerate(H0_GRID):
        z_s = z_of_dL(dL_s, H0)
        w = p_cat(z_s) / ddL_of_z(z_s, H0)
        z_max_det = z_of_dL(DL_MAX, H0)
        if use_indicator:
            w = w * (z_s <= z_max_det)
        beta = np.interp(z_max_det, zf, cdf_cat)
        Li = w.mean(axis=1)
        lls[k] = np.sum(np.log(np.maximum(Li, 1e-300))) - N_GW * np.log(max(beta, 1e-10))
    return H0_GRID[np.argmax(lls)]


results = {"A (truth-ctr, no ind.)": [], "B (truth-ctr, +ind.)": [],
           "C (obs-ctr, +ind.)": [], "D (obs-ctr, no ind.)": []}
for r in range(N_REAL):
    rng = np.random.default_rng(1000 + r)
    zh = None
    results["A (truth-ctr, no ind.)"].append(run_variant(False, False, np.random.default_rng(1000 + r)))
    results["B (truth-ctr, +ind.)"].append(run_variant(False, True, np.random.default_rng(1000 + r)))
    results["C (obs-ctr, +ind.)"].append(run_variant(True, True, np.random.default_rng(1000 + r)))
    results["D (obs-ctr, no ind.)"].append(run_variant(True, False, np.random.default_rng(1000 + r)))

print(f"H0_true = {H0_TRUE}, N_gw = {N_GW}, dL_unc = {DL_UNC_FAC}*dL, {N_REAL} realizations")
for k, v in results.items():
    v = np.asarray(v)
    print(f"  {k:26s}: <H0_MAP> = {v.mean():6.2f} +- {v.std()/np.sqrt(len(v)):.2f}  (bias {v.mean()-H0_TRUE:+6.2f})")
