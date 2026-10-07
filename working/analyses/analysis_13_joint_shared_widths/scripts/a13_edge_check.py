#!/usr/bin/env python
"""Analysis 13 -- is seed 1's Delta mu_chi = 0.30 edge mode selection Monte-Carlo noise?

Seed 1's interim posterior (2026-10-06 03:01) has a separate clump at Delta mu_chi 0.29-0.30
(H0 ~ 71.6, dmu_G ~ 7, sigma_chi ~ 0.12) holding the run's highest logL, 1.1 nats above the main
mode's best, with nothing between 0.23 and 0.28. The closure (K3) measured N_eff falling from
~3e5 at the main mode to ~8e3 at Delta mu_chi 0.30, so sigma^2_sel = N_obs^2 / N_eff rises from
~3 to ~120 nats^2 there.

This evaluates logL, N_eff and the implied selection variance N_obs^2/N_eff at
  * every dead/live point with Delta mu_chi > 0.26 (the edge clump),
  * the 30 highest-logL points with Delta mu_chi < 0.24 (the main mode's peak),
  * 300 equal-weight posterior draws,
with the af896ca joint-width cell (the closure's machinery, which reports N_eff; validated
against core at the posterior level). It reports the N_eff / variance distribution over the
posterior so a variance cap can be chosen that removes the edge clump without cutting the main
mode. Uses results/a13core_dynesty_n200_s1.npz if seed 1 has finished, else the checkpoint.

    sbatch --job-name=a13_edge --dependency=afterany:1361503 \\
        --export=ALL,A13_SCRIPT=scripts/a13_edge_check.py scripts/submit_a13_gpu.sbatch
"""
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
A13 = HERE.parent
A11 = A13.parent / "analysis_11_free_common_population"
sys.path.insert(0, str(A11 / "scripts"))

import a11_likelihood as L                                  # noqa: E402

TAG = "a13core_dynesty_n200_s1"
OUT = A13 / "diagnostics" / "a13_edge_check.json"
NAMES = ["H0", "fcat_2", "mu_G", "dmu_G", "mu_chi", "dmu_chi", "sigma_G", "sigma_chi"]
N_OBS = 1000


def load_points():
    npz = A13 / "results" / f"{TAG}.npz"
    if npz.exists():
        z = np.load(npz)
        return "results", np.asarray(z["dead"]), np.asarray(z["logl"]), np.asarray(z["samples"])
    from dynesty.utils import resample_equal
    s = pickle.load(open(A13 / "queue" / f"{TAG}.save", "rb"))["sampler"]
    s.loglikelihood.loglikelihood = lambda x: 0.0
    s.prior_transform = lambda u: u
    s.add_final_live(print_progress=False)
    r = s.results
    w = np.exp(r.logwt - r.logz[-1]); w /= w.sum()
    eq = resample_equal(r.samples, w, rstate=np.random.default_rng(1))
    return f"checkpoint it {s.it}", np.asarray(r.samples), np.asarray(r.logl), eq


def main():
    t0 = time.time()
    src, dead, logl, eq = load_points()
    rng = np.random.default_rng(11)
    edge = np.where(dead[:, 5] > 0.26)[0]
    main_idx = np.where(dead[:, 5] < 0.24)[0]
    main_top = main_idx[np.argsort(-logl[main_idx])][:30]
    draws = eq[rng.choice(len(eq), size=min(300, len(eq)), replace=False)]
    cell = L.build_a11("A13", shared_width=("sigma_G", "sigma_chi"))

    def ev(x, group, run_logl=None):
        r = cell.evaluate_at(**dict(zip(NAMES, map(float, x))))
        ne = r.get("Neff")
        return {"group": group, "x": dict(zip(NAMES, map(float, x))), "logL": r.get("logL"),
                "run_logL": run_logl, "Neff": ne, "threshold": r.get("threshold"),
                "sel_variance": (N_OBS ** 2 / ne) if ne else None}

    rows = [ev(dead[i], "edge", float(logl[i])) for i in edge]
    rows += [ev(dead[i], "main_peak", float(logl[i])) for i in main_top]
    rows += [ev(x, "posterior_draw") for x in draws]

    def summ(group):
        v = np.array([r["sel_variance"] for r in rows if r["group"] == group and r["sel_variance"]])
        if not len(v):
            return None
        return {"n": int(len(v)), "sel_variance_percentiles_5_50_95_max":
                [float(q) for q in np.percentile(v, [5, 50, 95])] + [float(v.max())]}

    post = [r for r in rows if r["group"] == "posterior_draw" and r["sel_variance"]]
    caps = {}
    for cap in (5, 10, 20, 30, 50, 100):
        caps[str(cap)] = {
            "posterior_draws_cut": float(np.mean([r["sel_variance"] > cap for r in post])) if post else None,
            "edge_points_cut": float(np.mean([r["sel_variance"] > cap for r in rows
                                              if r["group"] == "edge" and r["sel_variance"]]))
            if len(edge) else None}
    out = {"what": __doc__.split("    sbatch")[0].strip(), "source": src,
           "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
           "summary": {g: summ(g) for g in ("edge", "main_peak", "posterior_draw")},
           "cap_scan_total_variance_selection_only": caps, "rows": rows,
           "wall_seconds": time.time() - t0}
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out["summary"], indent=1)); print(json.dumps(caps, indent=1))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
