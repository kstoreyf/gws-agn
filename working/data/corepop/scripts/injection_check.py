#!/usr/bin/env python
"""Do the injection sets need regenerating for core's population?  Seed 100, both lanes.

    export PYTHONPATH=/hildafs/projects/phy230014p/magana/src/darksirens-2b86a2d
    JAX_PLATFORMS=cpu python injection_check.py      (jax env; calls core_eval.py points
                                                      in the core venv for core's density)

The selection term is mu = (1/N_draw) sum_det pi(theta_k)/pdraw_k.  Swapping the population
code changes only the population factor of pi, so every injection's weight is multiplied
by R_k = p_core(m1, q, chi)/p_legacy(m1, q, chi) (the redshift, host and Jacobian factors
are common).  The estimator stays unbiased iff pdraw > 0 wherever core's density is, and
stays well conditioned iff R is bounded on the detected injections.  This script checks:

  1. pdraw_population is the legacy population density exactly (recomputed with gmd);
  2. support: pdraw > 0 and finite on every row, and the proposal reaches every
     (m1det, q, chi, dL) core gives density to (by construction of the branches; recorded);
  3. R_k on every detected injection: how many are zero (m2 < 5: they simply drop out),
     the distribution and maximum of the rest, and sup R over core's whole support on a
     grid (the bound, not just the realised maximum);
  4. the population-branch importance weight w = p_pop/pdraw under each code (z prior:
     uniform in comoving volume, as the population branch draws it): max, and the effective
     number (sum w)^2 / sum w^2 under legacy and core.

Writes results/injection_check.json.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
CP = HERE.parent
sys.path.insert(0, str(CP.parent))
import generate_dataset as G  # noqa: E402

PIN = Path("/hildafs/projects/phy230014p/magana/src/darksirens-2b86a2d")
DATA = Path("/hildafs/projects/phy230054p/magana/gws-agn-data-v3")
CPY = "/hildafs/projects/phy230014p/magana/envs/darksirens-core-e7c3007/bin/python"
SEED = 100
BRANCH_NAMES = {"targeted": ["population", "uniform", "targeted_agn"],
                "popuni": ["population", "uniform"]}


def chunked(fn, n, chunk=200_000):
    return np.concatenate([fn(slice(i, min(i + chunk, n))) for i in range(0, n, chunk)])


def main():
    gmd = G.import_gmd(PIN)
    pop = gmd.PopulationConfig(gamma=G.GAMMA)
    grids = gmd._cosmology_grids(gmd._build_cosmology(G.H0_FID, G.OM0_FID, G.W0_FID, G.WA_FID),
                                 G.ZMAX_GRID)
    T = np.load(CP / "cache" / "core_tables.npz")
    out = {"what": __doc__.split("\n\n")[1].strip(), "seed": SEED, "lanes": {}}

    # sup R over core's support on the 2-D grid (chi factor cancels: same TN in both codes)
    m1d, qd = T["m1d"], T["qd"]
    M1, Q = np.meshgrid(m1d, qd, indexing="ij")
    chi0 = np.zeros(M1.size)
    p_leg = chunked(lambda s: gmd._mass_spin_pdf(M1.ravel()[s], Q.ravel()[s], chi0[s], pop),
                    M1.size).reshape(M1.shape)
    p_leg_mq = p_leg / gmd._truncnorm_pdf(np.array([0.0]), pop.chi_mu, pop.chi_sigma, -1, 1)[0]
    p_core_mq = np.exp(T["lnp2d_gal"])
    with np.errstate(divide="ignore", invalid="ignore"):
        Rg = np.where(p_leg_mq > 0, p_core_mq / p_leg_mq, np.nan)
    on = np.isfinite(Rg) & (p_core_mq > 0)
    ia = np.unravel_index(np.nanargmax(np.where(on, Rg, -1)), Rg.shape)
    out["sup_R_over_core_support_grid"] = {
        "grid": "m1 in [5, 100] step 0.05, q in [0.04, 1] step 0.001",
        "sup_R": float(Rg[on].max()), "at_m1": float(m1d[ia[0]]), "at_q": float(qd[ia[1]]),
        "at_m2": float(m1d[ia[0]] * qd[ia[1]]),
        "inf_R_where_core_positive": float(Rg[on].min()),
        "R_at_typical_35_30": float(np.interp(30 / 35, qd, Rg[np.argmin(np.abs(m1d - 35))])),
        "core_positive_where_legacy_zero": int(((p_core_mq > 0) & ~(p_leg_mq > 0)).sum())}
    print("[sup R grid]", out["sup_R_over_core_support_grid"], flush=True)
    # The weight itself is what must be bounded.  Per (m1, q, chi, z), with the common
    # p_z(z) / (dL'(z) 4 pi) factor removed, a lane's pdraw is at least
    #   w_pop p_leg(m1,q) TN(chi)/(1+z)  +  w_unif * 0.5/(m1det_hi - m1det_lo)
    # (the uniform branch: flat in m1det on [2, 200], q on [0, 1], chi on [-1, 1]), so
    #   W = p_core(m1,q) TN(chi)/(1+z) / pdraw  <=  p_core TN / (w_pop p_leg TN + w_unif u (1+z))
    # and z >= 0, TN <= TN(mu_chi) give a bound over core's whole support (m1det <= 200;
    # above it only the Gaussian far tail lives, where W -> R / w_pop with R ~ 1).
    tn_max = float(gmd._truncnorm_pdf(np.array([pop.chi_mu]), pop.chi_mu, pop.chi_sigma, -1, 1)[0])
    u = 0.5 / (gmd._M1DET_RANGE[1] - gmd._M1DET_RANGE[0])
    out["sup_weight_bound_over_core_support_grid"] = {}
    for lane, (wp, wu) in {"targeted": (0.65, 0.10), "popuni": (0.9, 0.1)}.items():
        W = p_core_mq * tn_max / (wp * p_leg_mq * tn_max + wu * u)
        iw = np.unravel_index(np.argmax(W), W.shape)
        out["sup_weight_bound_over_core_support_grid"][lane] = {
            "sup_W_core": float(W.max()), "at_m1": float(m1d[iw[0]]), "at_q": float(qd[iw[1]]),
            "same_bound_legacy": float((p_leg_mq * tn_max / (wp * p_leg_mq * tn_max + wu * u)).max())}
    print("[sup W bound]", out["sup_weight_bound_over_core_support_grid"], flush=True)
    del M1, Q, p_leg, p_leg_mq, p_core_mq, Rg

    ddldz = np.gradient(grids["dl"], grids["z"])
    for lane in ("targeted", "popuni"):
        p = DATA / f"seed{SEED}" / "injections" / f"injections_{lane}.h5"
        with h5py.File(p, "r") as f:
            d = {k: np.asarray(f[k][:]) for k in ("m1src", "m2src", "chieff", "z", "pdraw",
                                                  "pdraw_population", "branch", "m1det")}
            ndraw = int(f.attrs["ndraw"])
        n = d["pdraw"].size
        q = d["m2src"] / d["m1src"]
        p_leg = chunked(lambda s: gmd._mass_spin_pdf(d["m1src"][s], q[s], d["chieff"][s], pop), n)
        # 1. pdraw_population = p_leg * pz / ((1+z) dL'(z)) / (4 pi)
        pz = np.interp(d["z"], grids["z"], grids["dvc_dz"]) / np.trapz(grids["dvc_dz"], grids["z"])
        jac = np.interp(d["z"], grids["z"], ddldz) * (1.0 + d["z"])
        zfac = pz / np.maximum(jac, 1e-300) / (4.0 * np.pi)
        rec_pop = np.maximum(p_leg * zfac, 1e-300)
        relerr = np.abs(rec_pop / d["pdraw_population"] - 1.0)
        # core's density at the same points, in core's venv
        tmp_in = CP / "cache" / f"inj_points_{lane}.npz"
        tmp_out = CP / "cache" / f"inj_lnp_core_{lane}.npz"
        np.savez(tmp_in, m1src=d["m1src"], q=q, chieff=d["chieff"], z=d["z"])
        env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "NPY_DISABLE_CPU_FEATURES")}
        env["JAX_PLATFORMS"] = "cpu"
        subprocess.run([CPY, str(HERE / "core_eval.py"), "points", str(tmp_in), str(tmp_out)],
                       check=True, env=env)
        lnp_core = np.load(tmp_out)["lnp_core_gal"] + np.log1p(d["z"])   # drop (1+z)^-1
        p_core = np.exp(lnp_core)
        with np.errstate(divide="ignore", invalid="ignore"):
            R = np.where(p_leg > 0, p_core / p_leg, np.nan)
        br = d["branch"].astype(int)
        m2 = d["m2src"]
        pos = np.isfinite(R) & (R > 0)
        i_max = int(np.flatnonzero(pos)[np.argmax(R[pos])])
        w_leg = d["pdraw_population"] / d["pdraw"]
        w_core = np.where(np.isfinite(R), R, 0.0) * w_leg
        neff = lambda w: float(w.sum() ** 2 / (w ** 2).sum())  # noqa: E731
        lr = {
            "n_detected": n, "ndraw": ndraw,
            "pdraw_population_recomputed_max_rel_err": float(relerr.max()),
            "support": {
                "pdraw_all_positive_finite": bool(np.all(d["pdraw"] > 0) and np.all(np.isfinite(d["pdraw"]))),
                "n_core_positive_where_pdraw_population_zero": int(((p_core > 0) & ~(p_leg > 0)).sum()),
                "m1det_range": [float(d["m1det"].min()), float(d["m1det"].max())],
                "m2src_min": float(m2.min())},
            "by_branch": {},
            "R": {"n_R_zero_m2_below_5": int((R == 0).sum()),
                  "frac_R_zero": float((R == 0).mean()),
                  "n_R_nan_legacy_zero": int((~np.isfinite(R)).sum()),
                  "max": float(R[pos].max()),
                  "argmax": {"m1src": float(d["m1src"][i_max]), "m2src": float(m2[i_max]),
                             "branch": BRANCH_NAMES[lane][br[i_max]]},
                  "quantiles_of_positive": dict(zip(["0.001", "0.5", "0.99", "0.9999"],
                                                    np.quantile(R[pos], [1e-3, 0.5, 0.99, 0.9999]).tolist())),
                  "mean_over_detected": float(np.nanmean(R))},
            "population_branch_weight_uniform_volume_z": {
                "max_w_legacy": float(w_leg.max()), "max_w_core": float(w_core.max()),
                "bound_over_core_support": out["sup_weight_bound_over_core_support_grid"][lane]["sup_W_core"],
                "neff_legacy": neff(w_leg), "neff_core": neff(w_core),
                "neff_ratio_core_over_legacy": neff(w_core) / neff(w_leg),
                "mu_legacy": float(w_leg.sum() / ndraw), "mu_core": float(w_core.sum() / ndraw)}}
        for b, name in enumerate(BRANCH_NAMES[lane]):
            sel = br == b
            lr["by_branch"][name] = {"n_detected": int(sel.sum()),
                                     "n_m2_below_5": int((sel & (m2 < 5)).sum()),
                                     "n_m2_5_to_8": int((sel & (m2 >= 5) & (m2 < 8)).sum()),
                                     "max_R": float(np.nanmax(np.where(sel & pos, R, np.nan)))
                                     if (sel & pos).any() else None}
        out["lanes"][lane] = lr
        print(f"[{lane}] {json.dumps(lr)[:900]}", flush=True)
        for t in (tmp_in, tmp_out):
            t.unlink()
    (CP / "results" / "injection_check.json").write_text(json.dumps(out, indent=1))
    print("wrote", CP / "results" / "injection_check.json")


if __name__ == "__main__":
    main()
