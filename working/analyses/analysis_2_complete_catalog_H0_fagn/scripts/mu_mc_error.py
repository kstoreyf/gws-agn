#!/usr/bin/env python3
"""Carry the selection estimator's own Monte-Carlo error into the H0 numbers.

`CLOSURE.md` §14.2 named it and §16.6 measured it: the injection estimator of
`mu(theta)` carries a COMMON-MODE Monte-Carlo error on `d ln mu / dH0` that does
not average down within a realisation, and analysis 1's convention is to carry it
rather than drop it.  A common-mode error `sigma_MC` on `d ln mu/dH0` displaces the
H0 estimate by

    delta_H0 = sigma_MC * N_obs / |d^2 lnL/dH0^2|_total = sigma_MC / |curvature per event|

so the conversion needs this analysis's OWN curvature, which is measured here from
the joint grid's H0 marginal (the analysis of record, marginalised over f) and from
the h0scan (f pinned at 0.30), by a parabolic fit around the peak.

`sigma_MC` itself is NOT re-measured here.  Analysis 1 measured it on the same v3
injection sets for the two single-tracer limits, and those limits BRACKET the K = 2
mixture: at f = 0 the mixture's selection integral is exactly the GAL one and at
f = 1 exactly the AGN one -- verified bit-for-bit on all four (tracer, lane)
combinations by the N_eff endpoints.  The bracket is what is quoted; a sharper
number would need a Poisson bootstrap of the mixture's own `d ln mu/dH0`, which is
not run here.
"""
import json
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
SEEDS = [100, 101, 102, 103, 105]
H0_TRUTH = 67.74
# analysis 1, CLOSURE.md 16.6, measured on the v3 injection sets (targeted lane)
SIGMA_MC = {"matched GAL, targeted": 1.197e-4, "matched AGN, targeted": 5.334e-4,
            "matched GAL, popuni": 1.063e-4, "matched AGN, popuni": 1.101e-3}


def curvature(x, logp):
    """d^2 lnL/dx^2 at the peak, from a parabola through the top of the curve."""
    logp = np.asarray(logp, float)
    i = int(np.nanargmax(logp))
    lo, hi = max(0, i - 8), min(logp.size, i + 9)
    c = np.polyfit(x[lo:hi] - x[i], logp[lo:hi], 2)
    return 2.0 * c[0]


def main():
    out = {"_what": __doc__.strip().splitlines()[0], "seeds": {},
           "sigma_MC_source": "analysis_1 CLOSURE.md 16.6, v3 injection sets",
           "sigma_MC": SIGMA_MC}
    curv_marg, curv_scan = [], []
    for s in SEEDS:
        row = {}
        p = RES / f"joint_s{s}.h5"
        if p.exists():
            with h5py.File(p) as f:
                H0 = f["H0_grid"][:]; fv = f["f_grid"][:]; ll = f["log_likelihood"][:]
            P = np.where(np.isfinite(ll), np.exp(ll - np.nanmax(ll[np.isfinite(ll)])), 0.0)
            lm = np.log(np.maximum(np.trapz(P, fv, axis=1), 1e-300))
            k = curvature(H0, lm)
            row["curvature_joint_marginal_total"] = k
            row["curvature_joint_marginal_per_event"] = k / 1000.0
            curv_marg.append(k / 1000.0)
        p = RES / f"h0scan_s{s}.h5"
        if p.exists():
            with h5py.File(p) as f:
                H0 = f["H0_grid"][:]; ll = f["log_likelihood"][:]
            k = curvature(H0, ll)
            row["curvature_h0scan_total"] = k
            row["curvature_h0scan_per_event"] = k / 1000.0
            curv_scan.append(k / 1000.0)
        out["seeds"][str(s)] = row

    km = float(np.mean(np.abs(curv_marg)))
    ks = float(np.mean(np.abs(curv_scan))) if curv_scan else float("nan")
    out["curvature_per_event_mean_abs"] = {"joint_marginal": km, "h0scan": ks}
    out["carried_km_s_Mpc"] = {
        k: {"per_realisation_joint_marginal": v / km,
            "per_realisation_h0scan": (v / ks if ks == ks else None),
            "on_the_five_seed_mean": v / km / np.sqrt(len(SEEDS))}
        for k, v in SIGMA_MC.items()}
    lim = [out["carried_km_s_Mpc"]["matched GAL, targeted"],
           out["carried_km_s_Mpc"]["matched AGN, targeted"]]
    out["bracket_targeted_lane"] = {
        "per_realisation": [lim[0]["per_realisation_joint_marginal"],
                            lim[1]["per_realisation_joint_marginal"]],
        "on_the_five_seed_mean": [lim[0]["on_the_five_seed_mean"],
                                  lim[1]["on_the_five_seed_mean"]],
        "note": "the f = 0 and f = 1 limits of the K = 2 mixture; the mixture's own "
                "sigma_MC is not measured here and lies between them"}
    (RES / "mu_mc_error.json").write_text(json.dumps(out, indent=2))
    print(f"Wrote {RES / 'mu_mc_error.json'}")
    print(f"per-event curvature |d2 lnL/dH0^2|: joint marginal {km:.3e}, h0scan {ks:.3e}")
    for k, v in out["carried_km_s_Mpc"].items():
        print(f"  {k:24s}: +- {v['per_realisation_joint_marginal']:.2f} per realisation, "
              f"+- {v['on_the_five_seed_mean']:.2f} on the five-seed mean")
    b = out["bracket_targeted_lane"]
    print(f"  bracket (targeted): +- {b['per_realisation'][0]:.2f} to "
          f"{b['per_realisation'][1]:.2f} per realisation; "
          f"+- {b['on_the_five_seed_mean'][0]:.2f} to "
          f"{b['on_the_five_seed_mean'][1]:.2f} on the mean")


if __name__ == "__main__":
    main()
