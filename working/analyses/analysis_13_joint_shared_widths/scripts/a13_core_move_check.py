#!/usr/bin/env python
"""Analysis 13 -- core-move check (CORE_MOVE_PLAN.md): does a new core commit give seed 1's likelihood?

Run in the new commit's env with A13_CORE=<sha>, once per evaluation mode:

    A13_CORE=e7c3007 envs/darksirens-core-e7c3007/bin/python a13_core_move_check.py --settings historical
    A13_CORE=e7c3007 envs/darksirens-core-e7c3007/bin/python a13_core_move_check.py --settings defaults

Points, all evaluated at max_likelihood_variance 1e6 like seed 1:
  * the four core A11 reference cells (widths at their fiducials) -- the driver's own pre-flight;
  * 500 of seed 1's dead points drawn in proportion to their posterior weight (no repeats), plus the
    20 highest-logL points of the Delta mu_chi edge clump -- compared with the logL seed 1 recorded
    for them on core bf58aa6 (historical evaluation);
  * the closure's six f = 1 points with AGN spin mean >= 0.237 (rejected by the 5 N_obs floor):
    they must stay -inf.
Pass (historical): every |dlogL| <= 1e-8 and the six rejections hold.
Pass (defaults): sd(dlogL) over the 500 posterior points <= 0.05 nats (a constant offset only moves
logZ) and the six rejections hold. Also records seconds per call and device memory.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import a13_core_sampler as S                                 # noqa: E402

A13 = S.A13
RUN = A13 / "results" / "a13core_dynesty_n200_s1.npz"
CLOSURE = A13 / "diagnostics" / "a13_closure.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings", choices=["historical", "defaults"], required=True)
    ap.add_argument("--n-post", type=int, default=500)
    args = ap.parse_args()
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import jax
    import jax.numpy as jnp
    t0 = time.time()
    ds, b, settings = S.build(args.settings)
    build_s = time.time() - t0
    labels = list(b.labels)
    jb = jax.jit(b.__call__)

    def ll(theta):
        vals = S.absolute(theta)
        return float(jb(jnp.asarray([vals[l] for l in labels], dtype=jnp.float64)))

    out = {"core": S.CORE, "darksirens_file": ds.__file__, "settings_mode": args.settings,
           "catalog_evaluation": settings, "build_seconds": build_s,
           "slurm_job_id": os.environ.get("SLURM_JOB_ID")}

    # 1. the driver's reference cells
    ref = json.loads(S.REF_CELLS.read_text())
    cells = []
    for r in ref["rows"]:
        v = ll([r["H0"], r["fcat_2"], r["mu_G"], r["dmu_G"], r["mu_chi"], r["dmu_chi"], 5.0, 0.1])
        cells.append({"cell": r["name"], "new": v, "bf58aa6": r["logL"], "d": v - r["logL"]})
    out["a11_cells"] = cells

    # 2. seed-1 points
    Z = np.load(RUN)
    dead, logl, logwt = Z["dead"], Z["logl"], Z["logwt"]
    w = np.exp(logwt - logwt.max()); w /= w.sum()
    rng = np.random.default_rng(2026)
    post = rng.choice(len(w), size=args.n_post, replace=False, p=w)
    edge = np.where(dead[:, 5] > 0.26)[0]
    edge = edge[np.argsort(-logl[edge])][:20]
    rows, secs = [], []
    for group, idx in (("posterior", post), ("edge", edge)):
        for i in idx:
            t1 = time.time()
            v = ll(dead[i])
            secs.append(time.time() - t1)
            rows.append({"group": group, "i": int(i), "new": v, "bf58aa6": float(logl[i]),
                         "d": v - float(logl[i])})
    out["seed1_points"] = rows

    # 3. guard rejections from the closure (f = 1, AGN spin mean >= 0.237)
    k3 = [r for r in json.loads(CLOSURE.read_text())["K3"]["rows"] if not r["finite"]]
    med = json.loads(CLOSURE.read_text())["D11_medians"]
    rej = []
    for r in k3:
        v = ll([med["H0"], r["f"], r["mu_G"], r["dmu_G"], r["mu_chi"], r["dmu_chi"], r["sigma_G"],
                r["sigma_chi"]])
        rej.append({"f": r["f"], "dmu_chi": r["dmu_chi"], "sigma_chi": r["sigma_chi"], "new": v,
                    "still_rejected": not np.isfinite(v)})
    out["guard_rejections"] = rej

    dc = np.array([c["d"] for c in cells])
    dp = np.array([r["d"] for r in rows if r["group"] == "posterior"])
    de = np.array([r["d"] for r in rows if r["group"] == "edge"])
    try:
        mem = {k: int(v) for k, v in jax.devices()[0].memory_stats().items()
               if k in ("peak_bytes_in_use", "bytes_limit")}
    except Exception as e:  # noqa: BLE001
        mem = {"error": str(e)}
    summ = {"max_abs_d_cells": float(np.max(np.abs(dc))),
            "max_abs_d_posterior": float(np.max(np.abs(dp))), "mean_d_posterior": float(dp.mean()),
            "sd_d_posterior": float(dp.std()), "max_abs_d_edge": float(np.max(np.abs(de))),
            "n_bitwise_posterior": int(np.sum(dp == 0.0)), "n_posterior": int(len(dp)),
            "guard_rejections_held": f"{sum(r['still_rejected'] for r in rej)}/{len(rej)}",
            "seconds_per_call_median": float(np.median(secs[5:])), "device_memory": mem}
    if args.settings == "historical":
        summ["pass"] = bool(max(summ["max_abs_d_cells"], summ["max_abs_d_posterior"],
                                summ["max_abs_d_edge"]) <= 1e-8 and all(r["still_rejected"] for r in rej))
    else:
        summ["pass"] = bool(summ["sd_d_posterior"] <= 0.05 and all(r["still_rejected"] for r in rej))
    out["summary"] = summ
    path = A13 / "diagnostics" / f"a13_core_move_{S.CORE}_{args.settings}.json"
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(summ, indent=1)); print(f"wrote {path}")


if __name__ == "__main__":
    main()
