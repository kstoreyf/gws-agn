"""Analysis 15 shared definitions: the scan table, paths, posterior summaries, closed-form Z_k.

Imported by the GPU driver (core venv), the CPU validation and the aggregation (jax env); it
imports neither jax nor darksirens.

Z_k (LIKELIHOOD_MATH.md sections 4 and 8) for a field-weighted catalog k with the selection
completeness, no density field and no survey depth:

    Z_k(H0) = N_obs,k + n0_k * 4 pi * int_0^zmax [1 - C_k(z)] dV_c/(dz dOmega) dz,

C_k(z) = clip[Gamma(a+1, x_lim(z)) / Gamma(a+1, x_faint), 0, 1] (h-scaled, independent of H0),
n0 in Mpc^-3 at the evaluated H0, zmax the end of darksirens' internal redshift grid
(DARKSIRENS_ZMAX, default 5). F_k = N_obs,k / Z_k.
"""
import json
import math
from pathlib import Path

import numpy as np

WORK = Path("/hildafs/projects/phy230014p/magana/gws-agn/working")
A15 = WORK / "analyses" / "analysis_15_host_normalisation"
D = WORK / "data" / "seed100"
EVENTS = D / "events" / "events.h5"
INJECTIONS = D / "injections" / "injections_targeted.h5"
FITS = WORK / "analyses" / "selection_redo" / "scripts" / "selection_fits_truez"
A3R = WORK / "analyses" / "analysis_3_incomplete_catalog_H0_fagn" / "results"
A4R = WORK / "analyses" / "analysis_4_density_anchoring_H0_fagn" / "results"
FUR = WORK / "analyses" / "selection_redo" / "fu_probes" / "results"

CORE = "e7c3007"
CORE_PY = "/hildafs/projects/phy230014p/magana/envs/darksirens-core-e7c3007/bin/python"
OM0 = 0.3075
C_KMS = 299792.458
H0_REF = 67.74
F_INPUT = 0.30            # planted f_AGN of events.h5 (realised 295/1000 AGN hosts)
ZMAX_R = 1.10             # the restricted grid end (README: why 1.10 and not 1.014)
LOG2 = math.log10(2.0)

H0_GRID = np.round(64.0 + 0.25 * np.arange(41), 6)


def f_grid(lo, hi):
    n = int(round((hi - lo) / 0.025)) + 1
    return np.round(lo + 0.025 * np.arange(n), 6)


RUNGS = {r: {"surveys": [D / "surveys" / f"survey_{t}_{r}_ns32.h5" for t in ("gal", "agn")],
             "fits": [FITS / f"fit_{t}_{r}_schechter.json" for t in ("gal", "agn")]}
         for r in ("complete", "m21", "m18")}

# name, part, rung, zmax ("5" = default grid, "R" = DARKSIRENS_ZMAX=ZMAX_R), (log10n0, log10n0_c2),
# f range, stored legacy grid (gate / context), reference scan for the identity, gate it depends on
SCANS = [
    dict(name="p1_complete", part="1", rung="complete", zmax="5", n0=(-3.0, -5.0), f=(0.05, 0.55),
         stored=A3R / "joint_complete_s100.h5", gate=True, ref="ref_complete_n0m24", needs=None,
         label="(a) complete, z<5, input densities"),
    dict(name="p1_m21", part="1", rung="m21", zmax="5", n0=(-3.0, -5.0), f=(0.05, 0.55),
         stored=A3R / "joint_m21_s100.h5", gate=True, ref=None, needs=None,
         label="m<21, z<5, input densities"),
    dict(name="p1_m18", part="1", rung="m18", zmax="5", n0=(-3.0, -5.0), f=(0.05, 0.55),
         stored=A3R / "joint_m18_s100.h5", gate=True, ref=None, needs=None,
         label="m<18, z<5, input densities"),
    dict(name="b_complete_x1", part="2b", rung="complete", zmax="R", n0=(-3.0, -5.0), f=(0.05, 0.55),
         stored=None, gate=False, ref="ref_complete_n0m24", needs="p1_complete",
         label="(b) complete, z<1.10, input densities"),
    dict(name="b_complete_x2", part="2b", rung="complete", zmax="R", n0=(-3.0, -5.0 + LOG2),
         f=(0.05, 0.55), stored=None, gate=False, ref="ref_complete_n0m24", needs="p1_complete",
         label="(b) complete, z<1.10, AGN density x2"),
    dict(name="ref_complete_n0m24", part="ref", rung="complete", zmax="5", n0=(-24.0, -24.0),
         f=(0.025, 0.75), stored=FUR / "joint_complete_n0m24_s100.h5", gate=True, ref=None,
         needs="p1_complete", label="complete, zero density (reference)"),
    dict(name="c_complete_x2", part="2c", rung="complete", zmax="5", n0=(-3.0, -5.0 + LOG2),
         f=(0.05, 0.75), stored=None, gate=False, ref="ref_complete_n0m24", needs="p1_complete",
         label="(c) complete, z<5, AGN density x2"),
    dict(name="d_m18_x1", part="2d", rung="m18", zmax="R", n0=(-3.0, -5.0), f=(0.025, 0.65),
         stored=None, gate=False, ref="p1_m18", needs="p1_m18",
         label="(d) m<18, z<1.10, input densities"),
    dict(name="d_m18_x05", part="2d", rung="m18", zmax="R", n0=(-3.0, -5.0 - LOG2), f=(0.0, 0.45),
         stored=None, gate=False, ref="d_m18_x1", needs="p1_m18",
         label="(d) m<18, z<1.10, AGN density x0.5"),
    dict(name="d_m18_x2", part="2d", rung="m18", zmax="R", n0=(-3.0, -5.0 + LOG2), f=(0.10, 0.80),
         stored=None, gate=False, ref="d_m18_x1", needs="p1_m18",
         label="(d) m<18, z<1.10, AGN density x2"),
]
BY_NAME = {s["name"]: s for s in SCANS}
# job order (README): Part 1, then (b), then the reference and (c), then (d)
STEPS = [("5", ["p1_complete", "p1_m21", "p1_m18"]),
         ("R", ["b_complete_x1", "b_complete_x2"]),
         ("5", ["ref_complete_n0m24", "c_complete_x2"]),
         ("R", ["d_m18_x1", "d_m18_x05", "d_m18_x2"])]
# legacy (0c5b3db, grid to 5) results kept as context in the summary
LEGACY_CONTEXT = {"p1_complete": A3R / "joint_complete_s100.h5", "p1_m21": A3R / "joint_m21_s100.h5",
                  "p1_m18": A3R / "joint_m18_s100.h5",
                  "ref_complete_n0m24": FUR / "joint_complete_n0m24_s100.h5",
                  "m18_x05_grid5": A4R / "joint_m18_a05_s100.h5", "m18_x2_grid5": A4R / "joint_m18_a20_s100.h5"}
GATE = {"dmed_hw": 0.2, "wratio_lo": 0.9, "wratio_hi": 1.1}

# LIKELIHOOD_MATH.md section 4 table (H0 = 67.74, input densities, grid to 5)
NOTES_TABLE = {"complete": {"Z": (1.94e9, 1.63e7), "F": (0.0778, 0.0930)},
               "m21": {"Z": (2.12e9, 2.12e7), "F": (0.0114, 0.0115)},
               "m18": {"Z": (2.12e9, 2.12e7), "F": (0.00039, 0.00039)}}


# ----------------------------------------------------------------------------- posteriors
_trapz = getattr(np, "trapezoid", None) or np.trapz
QS = (0.05, 0.16, 0.5, 0.84, 0.95)


def _cdf(x, p):
    c = np.concatenate([[0.0], np.cumsum(0.5 * (p[1:] + p[:-1]) * np.diff(x))])
    return c / c[-1]


def summarize_1d(x, logp):
    """Median, 68% and 90% equal-tailed intervals of a gridded 1-D log density (as Analysis 14)."""
    x = np.asarray(x, float)
    logp = np.asarray(logp, float)
    ok = np.isfinite(logp)
    if x.size < 2 or ok.sum() < 2:
        nan = float("nan")
        return {"median": nan, "ci68": [nan, nan], "ci90": [nan, nan], "hw68": nan, "w68": nan,
                "w90": nan, "mean": nan, "sd": nan, "map": nan, "mass_at_edges": [nan, nan]}
    p = np.where(ok, np.exp(logp - np.max(logp[ok])), 0.0)
    c = _cdf(x, p)
    q = np.interp(QS, c, x)
    norm = _trapz(p, x)
    mean = float(_trapz(p * x, x) / norm)
    sd = float(np.sqrt(max(_trapz(p * (x - mean) ** 2, x) / norm, 0.0)))
    return {"median": float(q[2]), "ci68": [float(q[1]), float(q[3])], "ci90": [float(q[0]), float(q[4])],
            "hw68": float(0.5 * (q[3] - q[1])), "w68": float(q[3] - q[1]), "w90": float(q[4] - q[0]),
            "mean": mean, "sd": sd, "map": float(x[int(np.nanargmax(np.where(ok, logp, -np.inf)))]),
            "mass_at_edges": [float(p[0] / p.max()), float(p[-1] / p.max())]}


def marginals_2d(h, f, lnL):
    lnL = np.asarray(lnL, float)
    m = np.nanmax(np.where(np.isfinite(lnL), lnL, -np.inf))
    p = np.where(np.isfinite(lnL), np.exp(lnL - m), 0.0)
    with np.errstate(divide="ignore"):
        return np.log(_trapz(p, f, axis=1)), np.log(_trapz(p, h, axis=0))


def summarize_2d(h, f, lnL):
    lh, lf = marginals_2d(h, f, lnL)
    sh, sf = summarize_1d(h, lh), summarize_1d(f, lf)
    lnL = np.asarray(lnL, float)
    p = np.where(np.isfinite(lnL), np.exp(lnL - np.nanmax(np.where(np.isfinite(lnL), lnL, -np.inf))), 0.0)
    p /= p.sum()
    H, F = np.meshgrid(h, f, indexing="ij")
    eh, ef = (p * H).sum(), (p * F).sum()
    cov = (p * (H - eh) * (F - ef)).sum()
    rho = cov / np.sqrt((p * (H - eh) ** 2).sum() * (p * (F - ef) ** 2).sum())
    return {"H0": sh, "f": sf, "rho": float(rho)}


def stored_on_cells(path, hs, fs):
    """A stored legacy (201 x 41) grid restricted to the cells (hs x fs); None where absent."""
    import h5py
    with h5py.File(path, "r") as f:
        H = np.asarray(f["H0_grid"])
        F = np.asarray(f["f_grid"])
        L = np.asarray(f["log_likelihood"])
        attrs = {k: (f.attrs[k].item() if hasattr(f.attrs[k], "item") else str(f.attrs[k]))
                 for k in ("arg_log10n0", "arg_log10n0_c2", "darksirens_git_sha", "c_mode") if k in f.attrs}
    ih = [int(np.argmin(np.abs(H - h))) for h in hs]
    jf = [int(np.argmin(np.abs(F - x))) for x in fs]
    if not (np.allclose(H[ih], hs, atol=1e-9) and np.allclose(F[jf], fs, atol=1e-9)):
        return None
    return {"lnL": L[np.ix_(ih, jf)], "attrs": attrs}


# ----------------------------------------------------------------------------- closed-form Z_k
def schechter_fit(path):
    s = json.loads(Path(path).read_text())["strata"][0]
    if s["family"] != "schechter":
        raise SystemExit(f"[fatal] {path}: family {s['family']}")
    return {k: float(s[k]) for k in ("m_lim", "Mstar_hat", "alpha", "M_faint_offset")} | {"n_gal": int(s["n_gal"])}


def _upper_gamma(s, x):
    """Gamma(s, x) for s in (-1, 1), x > 0 (one downward recursion from s + 1 when s <= 0)."""
    from scipy.special import gamma, gammaincc
    x = np.asarray(x, float)
    if s > 0:
        return gamma(s) * gammaincc(s, x)
    return (gamma(s + 1.0) * gammaincc(s + 1.0, x) - x ** s * np.exp(-x)) / s


def comoving(z, H0, Om0=OM0, n=200001):
    """chi(z) [Mpc] and dV_c/(dz dOmega) [Mpc^3] for flat LCDM, by fine trapezoid."""
    zz = np.expm1(np.linspace(0.0, np.log1p(max(float(np.max(z)), 1e-6)), n))
    E = np.sqrt(Om0 * (1 + zz) ** 3 + 1 - Om0)
    chi_f = np.concatenate([[0.0], np.cumsum(0.5 * (1 / E[1:] + 1 / E[:-1]) * np.diff(zz))]) * C_KMS / H0
    chi = np.interp(z, zz, chi_f)
    Ez = np.sqrt(Om0 * (1 + np.asarray(z)) ** 3 + 1 - Om0)
    return chi, (C_KMS / H0) * chi ** 2 / Ez


def c_sel(z, fit, Om0=OM0):
    """The Schechter selection completeness C(z) (h-scaled: evaluated at H0 = 100, M* = Mstar_hat)."""
    chi, _ = comoving(z, 100.0, Om0)
    dl = np.maximum((1 + np.asarray(z)) * chi, 1e-300)
    dm = 5 * np.log10(dl) + 25.0
    x_lim = 10.0 ** (-0.4 * (fit["m_lim"] - dm - fit["Mstar_hat"]))
    x_faint = 10.0 ** (-0.4 * fit["M_faint_offset"])
    s = fit["alpha"] + 1.0
    return np.clip(_upper_gamma(s, x_lim) / _upper_gamma(s, x_faint), 0.0, 1.0)


def core_zgrid(zmax):
    """darksirens-core's redshift grid (cosmology/_grid.py) for DARKSIRENS_ZMAX = zmax."""
    n = max(1000, int(round(1000 * math.log(zmax + 1.0) / math.log(6.0))))
    return np.expm1(np.linspace(0.0, np.log(zmax + 1.0), n))


def missing_volume(fit, H0, zmax, grid="fine"):
    """4 pi int_0^zmax (1 - C) dV_c [Mpc^3] at H0 (trapezoid on core's grid or a fine one)."""
    z = core_zgrid(zmax) if grid == "core" else np.expm1(np.linspace(0.0, np.log1p(zmax), 40001))
    _, dv = comoving(z, H0)
    return 4 * np.pi * _trapz((1.0 - c_sel(z, fit)) * dv, z)


def z_closed(n_obs, log10n0, fit, H0, zmax, grid="fine"):
    return n_obs + 10.0 ** log10n0 * missing_volume(fit, H0, zmax, grid)


def survey_nobs(path):
    import h5py
    with h5py.File(path, "r") as f:
        return int(np.sum(np.asarray(f["ngals"], dtype=np.int64)))


def zmax_of(code):
    return 5.0 if code == "5" else ZMAX_R


def logit(f):
    f = np.asarray(f, float)
    with np.errstate(divide="ignore"):
        return np.log(f) - np.log1p(-f)


def expit(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, float)))
