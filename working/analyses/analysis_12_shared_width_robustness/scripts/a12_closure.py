#!/usr/bin/env python
"""Analysis 12 -- closure and selection support for ONE shared width, before sampling.

  K1  reduction.  With the released width at its fiducial (sigma_G = 5 Msun,
      sigma_chi = 0.1) the 12M and 12chi cells equal the Analysis-11 cell in logL,
      logL_pe and logL_selection at several (H0, f, population) points: bitwise, or
      within 1e-8 absolute (a traced slot can move the last bit, as in A11's C1).
  K2  shared and live.  Moving the width moves logL at f = 0 (reference branch only),
      at f = 1 (AGN branch only) and at the 11D median f: the width reaches BOTH
      branches, so it is common to GAL and AGN.
  K3  selection support.  The live guard at the 11D posterior medians along the
      released width, at f in {0, 11D median, 1}. Rejections are recorded, not waived.

Exits non-zero if K1 or K2 fails, so dependent sampler jobs never start.

    sbatch --job-name=a12_closure --time=01:00:00 \\
        --export=ALL,A12_SCRIPT=scripts/a12_closure.py scripts/submit_a12_gpu.sbatch
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
A12 = HERE.parent
A11 = A12.parent / "analysis_11_free_common_population"
sys.path.insert(0, str(A11 / "scripts"))

import a11_likelihood as L                                  # noqa: E402

OUT = A12 / "diagnostics" / "a12_closure.json"
PARTS = ("logL", "logL_pe", "logL_selection")


def rec(r):
    return {k: r.get(k) for k in PARTS + ("Neff", "threshold", "finite", "seconds")}


def same(a, b):
    return all(a[k] is not None and float(a[k]).hex() == float(b[k]).hex() for k in PARTS)


def close(a, b, tol=1e-8):
    return all(a[k] is not None and b[k] is not None and abs(a[k] - b[k]) <= tol for k in PARTS)


def finite(x):
    return x is not None and bool(np.isfinite(x))


def main():
    t0 = time.time()
    d = json.loads((A11 / "results" / "a11_11D.json").read_text())["summary"]
    med = {k: d[k]["median"] for k in ("H0", "f_agn", "mu_G", "dmu_G", "mu_chi", "dmu_chi")}
    pop = {k: med[k] for k in ("mu_G", "dmu_G", "mu_chi", "dmu_chi")}
    a11 = L.build_a11("A12REF")
    cells = {"sigma_G": L.build_a11("A12M", data=a11.data, shared_width=("sigma_G",)),
             "sigma_chi": L.build_a11("A12CHI", data=a11.data, shared_width=("sigma_chi",))}
    out = {"what": __doc__.split("    sbatch")[0].strip(),
           "labels": {k: list(c.labels) for k, c in cells.items()},
           "provenance": L.A10.a8.provenance(gw_path=L.GW_PATH_A11,
                                             survey_paths=[L.SURVEY_GAL, L.SURVEY_AGN]),
           "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "D11_medians": med}

    # timing
    c = cells["sigma_G"]
    secs = [c.evaluate_at(H0=med["H0"], fcat_2=med["f_agn"], sigma_G=5.0, **pop)["seconds"]
            for _ in range(4)]
    out["timing"] = {"first": secs[0], "steady_median": float(np.median(secs[1:]))}

    # K1
    pts = [dict(H0=L.H0_FID, fcat_2=0.3, mu_G=35.0, dmu_G=5.0, mu_chi=0.0, dmu_chi=0.1),
           dict(H0=med["H0"], fcat_2=med["f_agn"], **pop),
           dict(H0=64.0, fcat_2=0.0, mu_G=33.0, dmu_G=2.0, mu_chi=-0.05, dmu_chi=0.15),
           dict(H0=71.0, fcat_2=1.0, mu_G=36.0, dmu_G=4.0, mu_chi=0.02, dmu_chi=0.10)]
    k1 = []
    for p in pts:
        ref = rec(a11.evaluate_at(**p))
        for name, cell in cells.items():
            fid = L.SHARED_WIDTH[name][1]
            r = rec(cell.evaluate_at(**p, **{name: fid}))
            k1.append({"point": p, "width": name, "a11": ref, "a12": r, "bitwise": same(ref, r),
                       "within_1e-8": close(ref, r),
                       "abs_diff_logL": abs(r["logL"] - ref["logL"])})
    out["K1"] = {"rows": k1, "n_bitwise": sum(x["bitwise"] for x in k1), "n": len(k1),
                 "max_abs_diff_logL": max(x["abs_diff_logL"] for x in k1),
                 "pass": all(x["within_1e-8"] for x in k1)}

    # K2
    steps = {"sigma_G": (4.0, 6.0), "sigma_chi": (0.08, 0.12)}
    k2 = []
    for name, cell in cells.items():
        fid = L.SHARED_WIDTH[name][1]
        for f in (0.0, med["f_agn"], 1.0):
            base = cell.evaluate_at(H0=med["H0"], fcat_2=f, **pop, **{name: fid})["logL"]
            for v in steps[name]:
                r = cell.evaluate_at(H0=med["H0"], fcat_2=f, **pop, **{name: v})["logL"]
                k2.append({"width": name, "f": f, "value": v, "dlogL": r - base,
                           "live": bool(finite(r) and finite(base) and r != base)})
    out["K2"] = {"rows": k2, "pass": all(x["live"] for x in k2)}

    # K3
    grid = {"sigma_G": (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.5, 10.0),
            "sigma_chi": (0.01, 0.03, 0.05, 0.08, 0.1, 0.13, 0.2, 0.35, 0.6, 1.0)}
    k3 = []
    for name, cell in cells.items():
        for f in (0.0, med["f_agn"], 1.0):
            for v in grid[name]:
                r = rec(cell.evaluate_at(H0=med["H0"], fcat_2=f, **pop, **{name: v}))
                r.update({"width": name, "f": f, "value": v,
                          "Neff_over_threshold": (r["Neff"] / r["threshold"]
                                                  if r.get("Neff") and r.get("threshold") else None)})
                k3.append(r)
                print(f"[K3] {name} {v:6.3f} f {f:.3f} logL {r['logL']} "
                      f"Neff/thr {r['Neff_over_threshold']}", flush=True)
    out["K3"] = {"rows": k3, "n_rejected": int(sum(not finite(r["logL"]) for r in k3))}
    out["wall_seconds"] = time.time() - t0
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print(f"K1 {out['K1']['pass']} ({out['K1']['n_bitwise']}/{out['K1']['n']} bitwise)  K2 {out['K2']['pass']}  K3 rejected {out['K3']['n_rejected']}")
    print(f"wrote {OUT}")
    if not (out["K1"]["pass"] and out["K2"]["pass"]):
        sys.exit("[fatal] Analysis-12 closure failed; do not sample")


if __name__ == "__main__":
    main()
