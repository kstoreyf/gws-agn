#!/usr/bin/env python
"""Tables of the mixture validation (numpy only): scans (item 4), mergers (item 1), lines (items 2, 3).

    python tests/mixture_redraw/mix_summary.py   -> results/summary.json and a printout
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parent / "results"


def scans():
    out = {}
    for p in sorted(R.glob("scan_*.json")):
        if "test_" in p.name:
            continue
        d = json.loads(p.read_text())
        rows = d["realizations"]
        tot = {"n_realizations": len(rows), "nsamp": d["nsamp"],
               "triggered": [v["n_triggered"] if "n_triggered" in v else None for v in rows.values()],
               "refilled": [v["n_refilled"] for v in rows.values()],
               "capped": sum(v["summary"].get("n_capped", 0) for v in rows.values()),
               "untouched_identical_all": all(v["untouched_bitwise_identical"] for v in rows.values()),
               "seconds_refill": [round(v["seconds_refill"], 1) for v in rows.values()],
               "seconds_production_redraw_median": float(np.median([v["seconds_production_redraw"] for v in rows.values()])),
               "raw_draws_total_median": float(np.median([v["summary"].get("raw_draws_total", 0) for v in rows.values()])),
               "raw_draws_max": int(max(v["summary"].get("raw_draws_max", 0) for v in rows.values())),
               "P_A_min_by_realization": {k: round(v["summary"].get("P_A_min", 1.0), 4) for k, v in rows.items()},
               "lowest": {k: v["lowest_P_A"][:2] for k, v in rows.items()},
               "region_active": [v["region"]["active"] for v in rows.values()][0]}
        out[p.stem] = tot
        print(f"{p.stem}: {tot['n_realizations']} realizations, refilled {tot['refilled']}, capped {tot['capped']}, "
              f"raw draws median {tot['raw_draws_total_median']:.3g} (max per merger {tot['raw_draws_max']:.3g}), refill s "
              f"{tot['seconds_refill']}, identical {tot['untouched_identical_all']}, active {tot['region_active']}")
        print("    P_A min:", tot["P_A_min_by_realization"])
    return out


def mergers():
    out = {}
    for p in sorted(R.glob("merger_*.json")):
        if "_smoke" in p.name:
            continue
        d = json.loads(p.read_text())
        out[p.stem] = {k: d[k] for k in ("merger", "nsamp", "streams", "reference", "window", "ref_largest_step")}
        for lab in ("off", "on"):
            o = d[lab]
            out[p.stem][lab] = {k: o[k] for k in ("ess_mid_median", "rms_resid_median", "max_abs_resid_median",
                                                  "largest_step_median", "largest_step_max", "mean_resid_curve_max_abs",
                                                  "D_end_minus_start")}
        out[p.stem]["record"] = d["mixture_records"][0]
        z = np.load(p.with_suffix(".npz"))
        w, K = z["w"], d["streams"]
        for lab, sl in (("off", slice(0, K)), ("on", slice(K, 2 * K))):
            # half-variance corrected per stream (ln of a noisy sum is low by sigma^2/2), against the reference, constant removed
            e = (z["ev"] + 0.5 * z["var"])[:, sl][w]
            dd = e - z["ref"][w][:, None]
            dd = dd - dd.mean(axis=0, keepdims=True)
            mc = dd.mean(axis=1)
            sem = dd.std(axis=1, ddof=1) / np.sqrt(K)
            out[p.stem][lab]["corrected_mean_curve_max_abs"] = float(np.abs(mc).max())
            out[p.stem][lab]["corrected_mean_curve_rms"] = float(np.sqrt((mc ** 2).mean()))
            out[p.stem][lab]["corrected_mean_curve_rms_sem"] = float(np.sqrt((sem ** 2).mean()))
        print(p.stem, json.dumps(out[p.stem]))
    return out


def lines():
    out = {}
    for p in sorted(R.glob("line_*.json")):
        d = json.loads(p.read_text())
        out[p.stem] = {"lines": {k: {kk: v[kk] for kk in ("median", "ci90", "hw90")} | {"step": v["steps"]["largest_excess_step"],
                                                                                         "n_steps_gt_0.2": v["steps"]["n_excess_steps_gt_0.2"]}
                                 for k, v in d["lines"].items()},
                       "hybrid_references": d["hybrid_references"], "per_merger": d["per_merger_on_minus_off_at_peak"],
                       "ess_census": d["ess_census_at_peak"], "mergers_bound": d["mergers_bound"]}
        print(p.stem, json.dumps(out[p.stem], default=float))
    return out


if __name__ == "__main__":
    res = {"scans": scans(), "mergers": mergers(), "lines": lines()}
    (R / "summary.json").write_text(json.dumps(res, indent=1, default=float))
