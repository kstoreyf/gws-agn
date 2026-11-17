#!/usr/bin/env python
"""Test (a) comparison: the driver's closure grids against Analyses 14 and 15 (CPU, seconds).

    python tests/compare_closure.py      -> tests/results/closure.json (prints the verdicts)

Pairs and pass criteria (cells H0 64-74 x f_AGN 0.05-0.55, 41 x 21):
  a14_convention vs A14 b_pe_n8000_r00   same settings and samples: max |d lnL| <= 1e-6 over all cells,
                                         and the posterior summaries (median, 68%/90% bounds) equal to
                                         1e-4 in H0 and 1e-5 in f_AGN, uncorrected and corrected
  a15_convention vs A15 b_complete_x1    same criteria (skipped if A15's scan is not on disk yet)
  production_stored vs a15_convention    complete model (Z_k = N_obs) against the selection model at
                                         C = 1 on the grid to 1.10: max |d lnL| <= 1e-6
  production_stored vs A15 b_complete_x1 summary to rounding (as above)
  production_a14_redraw vs A14           the production convention (grid to 1.10, complete model,
                                         cap 20) against A14's (grid to 5, count ratio at n0 = 1e-24,
                                         cap 1e6) on the same redraw: medians within 0.02 of the 68%
                                         half-width, 68%/90% widths within 1%; d lnL reported
"""
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np

A16 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(A16 / "scripts"))
import prod_common as P  # noqa: E402

CR = A16 / "tests" / "closure_runs"
A14S = P.WORK / "analyses/analysis_14_monte_carlo_accuracy/results/scans/b_pe_n8000_r00"
A15S = P.WORK / "analyses/analysis_15_host_normalisation/results/scans/b_complete_x1"
TOL = {"lnL": 1e-6, "H0": 1e-4, "f": 1e-5}


def mine(name):
    d = CR / name / "grid"
    if not (d / "result.json").exists():
        return None
    with h5py.File(d / "fine.h5", "r") as f:
        g = {"H0": np.asarray(f["axis_H0"]), "f": np.asarray(f["axis_f_agn"]), "lnL": np.asarray(f["lnL"]),
             "pe_var": np.asarray(f["pe_var"]), "n_eff": np.asarray(f["n_eff"])}
    r = json.loads((d / "result.json").read_text())
    ren = lambda s: {"H0": s["H0"], "f": s["f_agn"]}  # noqa: E731
    g["summary_raw"] = ren(r["summary_uncorrected"])
    g["summary_corr"] = ren(r["summary"]) if r["half_variance_correction"] else None
    g["n_events"] = r["model_info"]["n_events"]
    return g


def ref(path):
    if not path.with_suffix(".h5").exists():
        return None
    with h5py.File(path.with_suffix(".h5"), "r") as f:
        g = {"H0": np.asarray(f["H0_grid"]), "f": np.asarray(f["f_grid"]), "lnL": np.asarray(f["lnL"]),
             "pe_var": np.asarray(f["pe_var"]), "n_eff": np.asarray(f["n_eff"])}
    j = json.loads(path.with_suffix(".json").read_text())
    g["summary_raw"] = {"H0": j["summary"]["H0"], "f": j["summary"]["f"]}
    g["summary_corr"] = ({"H0": j["summary_bias_corrected"]["H0"], "f": j["summary_bias_corrected"]["f"]}
                         if "summary_bias_corrected" in j else None)
    g["n_events"] = j.get("n_events", 1000)
    return g


def summ_diff(a, b):
    out = {}
    for p in ("H0", "f"):
        x, y = a[p], b[p]
        out[p] = {"median": [x["median"], y["median"]], "d_median": x["median"] - y["median"],
                  "d_ci68": [x["ci68"][0] - y["ci68"][0], x["ci68"][1] - y["ci68"][1]],
                  "d_ci90": [x["ci90"][0] - y["ci90"][0], x["ci90"][1] - y["ci90"][1]],
                  "d_median_over_hw68": (x["median"] - y["median"]) / y["hw68"],
                  "w68_ratio": x["w68"] / y["w68"], "w90_ratio": x["w90"] / y["w90"]}
    return out


def rounding_pass(sd):
    return all(abs(sd[p]["d_median"]) <= TOL[p] and max(map(abs, sd[p]["d_ci68"] + sd[p]["d_ci90"])) <= TOL[p]
               for p in ("H0", "f"))


def compare(a, b, kind):
    if a is None or b is None:
        return {"available": False, "pass": None}
    assert np.allclose(a["H0"], b["H0"]) and np.allclose(a["f"], b["f"]), "different cells"
    d = a["lnL"] - b["lnL"]
    w = np.exp(b["lnL"] - np.nanmax(b["lnL"]))
    bulk = w > 1e-3
    out = {"available": True, "max_abs_dlnL": float(np.nanmax(np.abs(d))),
           "dlnL_bulk_mean": float(np.mean(d[bulk])), "dlnL_bulk_sd": float(np.std(d[bulk])),
           "max_rel_d_pe_var": float(np.nanmax(np.abs(a["pe_var"] / b["pe_var"] - 1))),
           "max_rel_d_n_eff": float(np.nanmax(np.abs(a["n_eff"] / b["n_eff"] - 1))),
           "summary_uncorrected": summ_diff(a["summary_raw"], b["summary_raw"])}
    if a["summary_corr"] is not None and b["summary_corr"] is not None:
        out["summary_corrected"] = summ_diff(a["summary_corr"], b["summary_corr"])
    if kind == "exact":
        ok = out["max_abs_dlnL"] <= TOL["lnL"] and rounding_pass(out["summary_uncorrected"])
        if "summary_corrected" in out:
            ok &= rounding_pass(out["summary_corrected"])
    elif kind == "lnL":
        ok = out["max_abs_dlnL"] <= TOL["lnL"]
    elif kind == "summary":
        ok = rounding_pass(out["summary_uncorrected"])
    else:  # "convention": production vs the old convention
        s = out["summary_uncorrected"]
        ok = all(abs(s[p]["d_median_over_hw68"]) <= 0.02 and abs(s[p]["w68_ratio"] - 1) <= 0.01
                 and abs(s[p]["w90_ratio"] - 1) <= 0.01 for p in ("H0", "f"))
    out["criterion"] = kind
    out["pass"] = bool(ok)
    return out


def main():
    m = {n: mine(n) for n in ("a14_convention", "production_a14_redraw", "a15_convention", "production_stored")}
    r14, r15 = ref(A14S), ref(A15S)
    res = {
        "a14_convention_vs_A14": compare(m["a14_convention"], r14, "exact"),
        "a15_convention_vs_A15": compare(m["a15_convention"], r15, "exact"),
        "production_stored_vs_a15_convention": compare(m["production_stored"], m["a15_convention"], "lnL"),
        "production_stored_vs_A15": compare(m["production_stored"], r15, "summary"),
        "production_a14_redraw_vs_A14": compare(m["production_a14_redraw"], r14, "convention"),
        "references": {"A14": str(A14S), "A15": str(A15S), "A15_present": r15 is not None},
        "tolerances": TOL, "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out = A16 / "tests" / "results" / "closure.json"
    out.write_text(json.dumps(res, indent=1))
    # measured seconds per call of the production grid (population fixed, both complete catalogs)
    rj = CR / "production_a14_redraw" / "grid" / "result.json"
    if rj.exists():
        tp = A16 / "tests" / "results" / "timing.json"
        t = json.loads(tp.read_text()) if tp.exists() else {}
        t["spc_fixed"] = json.loads(rj.read_text())["seconds_per_call_median"]
        sj = CR / "production_stored" / "grid" / "result.json"
        if sj.exists():
            t["spc_fixed_2000_samples"] = json.loads(sj.read_text())["seconds_per_call_median"]
        tp.write_text(json.dumps(t, indent=1))
    for k, v in res.items():
        if isinstance(v, dict) and "pass" in v:
            extra = f"max|dlnL| {v['max_abs_dlnL']:.2e}" if v.get("available") else "not available"
            print(f"{k:40s} pass={v['pass']}  {extra}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
