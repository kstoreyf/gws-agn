#!/usr/bin/env python
"""Test of scripts/plateau_correction.py (the ln Z of finished nested runs whose initial points partly sit on the
driver's floor -1e300), on CPU:

  A  two toy runs with the driver's sampler and loop (tests/test_plateau_evidence.py: 6 dimensions, 75% of the prior on
     the floor, seeds 1 and 2): n_floor = the initial live points on the floor; the stored ln X after the plateau is
     dynesty's ln[(N + 1 - n_f)/(N + 1)]; the corrected ln Z is ln Z_dyn - ln X_dyn + ln(k/N) + n_f/(2 N k) and its
     error sqrt(logzerr^2 + n_f/(N k)), k = N - n_f
  B  the two runs merged as prod_summary.py merges parts (dynesty.utils.merge_runs): N = 400, n_floor the sum, the stored
     ln X after the plateau dynesty's formula at N = 400
  C  the estimator over k ~ Binomial(N, q) (exact sums, k >= 1): |bias| < 0.01 nats for q >= 0.05 at N = 200; dynesty's
     own estimator is biased by about +(1 - q)/(2 N q)
  D  a run without floor points is returned unchanged
  E  (if present) the finished production part single_population/no_catalog/complete/host_dependent/realization_100/nested_1:
     its n_floor, shift and error (reported, not a pass criterion)

    <core venv python> tests/test_plateau_correction.py -> tests/results/plateau_correction_test.json
"""
from __future__ import annotations

import json
import math
import os
import pickle
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

A16 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(A16 / "scripts"))
sys.path.insert(0, str(A16 / "tests"))
os.environ.setdefault("A16_PLATEAU_SCRATCH", tempfile.gettempdir())
import plateau_correction as PC  # noqa: E402
import test_plateau_evidence as TPE  # noqa: E402

OUT = A16 / "tests" / "results" / "plateau_correction_test.json"


def main():
    from dynesty.utils import merge_runs
    res, checks = {}, {}
    runs = []
    for seed in (1, 2):
        row, r = TPE._run_one({"d": 6, "q": 0.25, "sample": "unif", "floor": "1e300", "seed": seed}, return_results=True)
        runs.append((row, r))
    # A
    rowsA = []
    okA = True
    for row, r in runs:
        c = PC.plateau_corrected_logz(r)
        N, nf = 200, row["n_init_floor_live"]
        k = N - nf
        want_lnx = math.log((N + 1 - nf) / (N + 1))
        want = row["logz"] - want_lnx + math.log(k / N) + nf / (2 * N * k)
        want_err = math.sqrt(row["logzerr"] ** 2 + nf / (N * k))
        ok = (c["n_floor"] == nf and c["n_live"] == N and abs(c["lnX_dynesty"] - want_lnx) < 1e-12
              and abs(c["logz"] - want) < 1e-12 and abs(c["logzerr"] - want_err) < 1e-12)
        okA &= ok
        rowsA.append({"seed": row["seed"], "n_floor": nf, "lnX_dynesty": c["lnX_dynesty"], "lnX_hat": c["lnX_hat"],
                      "lnq_true": row["lnq"], "logz_dynesty": row["logz"], "logz_corrected": c["logz"],
                      "logz_true": row["logz_true"], "logzerr_dynesty": row["logzerr"], "logzerr_corrected": c["logzerr"],
                      "pass": bool(ok)})
    checks["A_single_runs"] = {"pass": bool(okA), "rows": rowsA}
    # B
    m = merge_runs([runs[0][1], runs[1][1]], print_progress=False)
    cm = PC.plateau_corrected_logz(m)
    nf_tot = runs[0][0]["n_init_floor_live"] + runs[1][0]["n_init_floor_live"]
    want_lnx = math.log((401 - nf_tot) / 401)
    okB = cm["n_live"] == 400 and cm["n_floor"] == nf_tot and abs(cm["lnX_dynesty"] - want_lnx) < 1e-9
    checks["B_merged_pair"] = {"pass": bool(okB), "n_live": cm["n_live"], "n_floor": cm["n_floor"],
                               "lnX_dynesty": cm["lnX_dynesty"], "formula": want_lnx, "logz_merged": cm["logz_dynesty"],
                               "logz_corrected": cm["logz"], "logzerr_merged": cm["logzerr_dynesty"],
                               "logzerr_corrected": cm["logzerr"], "logz_true": runs[0][0]["logz_true"]}
    # C
    tab = PC.binomial_plateau_table(200, [0.02, 0.035, 0.05, 0.1, 0.25, 0.5, 0.75])
    okC = all(abs(r["hat_bias"]) < 0.01 for r in tab if r["q"] >= 0.05) and \
        all(abs(r["dynesty_bias"] - (1 - r["q"]) / (2 * 200 * r["q"])) < 0.3 * (1 - r["q"]) / (2 * 200 * r["q"]) + 0.002
            for r in tab if r["q"] >= 0.05)
    checks["C_binomial"] = {"pass": bool(okC), "table": tab}
    # D
    row0, r0 = TPE._run_one({"d": 6, "q": 1.0, "sample": "unif", "floor": "1e300", "seed": 1}, return_results=True)
    c0 = PC.plateau_corrected_logz(r0)
    okD = c0["n_floor"] == 0 and c0["logz"] == row0["logz"] and c0["logzerr"] == row0["logzerr"]
    checks["D_no_floor_unchanged"] = {"pass": bool(okD), "n_floor": c0["n_floor"], "shift": c0["shift"]}
    # E
    prod = A16 / "runs" / "single_population" / "no_catalog" / "complete" / "host_dependent" / "realization_100" / "nested_1"
    if (prod / "results.pkl").exists():
        with open(prod / "results.pkl", "rb") as fh:
            rp = pickle.load(fh)
        cp = PC.plateau_corrected_logz(rp)
        checks["E_production_part"] = {"part": str(prod.relative_to(A16)), **{k: cp[k] for k in (
            "n_floor", "n_live", "lnX_dynesty", "lnX_hat", "shift", "var_plateau", "logz_dynesty", "logz",
            "logzerr_dynesty", "logzerr")}}
    res = {"written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "pass": all(v.get("pass", True) for v in checks.values()), "checks": checks}
    OUT.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: (v.get("pass"), {kk: vv for kk, vv in v.items() if kk not in ("rows", "table")})
                      for k, v in checks.items()}, indent=1, default=str))
    print("PASS" if res["pass"] else "FAIL", "->", OUT)
    sys.exit(0 if res["pass"] else 1)


if __name__ == "__main__":
    main()
