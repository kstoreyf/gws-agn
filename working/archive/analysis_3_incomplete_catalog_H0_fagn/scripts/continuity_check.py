#!/usr/bin/env python
"""GATE (b) -- pin analysis 3's rung 0 to the analysis-2 record.

Analysis 2 measured the complete-catalog joint (H0, f_AGN) with the out-of-catalog
field term suppressed (log10n0 = log10n0_c2 = -24).  Analysis 3 switches that term
on at the mock's true densities.  On COMPLETE catalogs the term should have almost
nothing to do: there are no missing hosts, so the missing-host budget

    dN_miss/dz = n0 dV_c/dz (1+z)^delta - dN_obs/dz

should be ~0 up to the mock's own shot noise and the shape residual of the model
form.  If the two configurations disagree materially on the complete pair, then the
ladder's rungs cannot be read against analysis 2's rung 0 and the campaign stops.

This compares the two 1-D cuts through the peak -- the f = 0.30 H0 column and the
truth-H0 f column -- on seed 100, targeted lane, which is 302 evaluations instead of
the 8241 a full joint grid would need.  Shifts are reported in the parameter's own
units AND in units of analysis 2's own quoted 68 % half-width, which is the scale
that decides whether the ladder stays on one axis.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
A2 = HERE.parent / "analysis_2_complete_catalog_H0_fagn" / "results"

PAIRS = [
    ("h0scan", "H0", A2 / "h0scan_s100.json", HERE / "results" / "h0scan_complete_s100.json"),
    ("fscan", "f", A2 / "fscan_s100.json", HERE / "results" / "fscan_complete_s100.json"),
]


def block(d, key):
    b = d[key]
    lo, hi = b["ci68"]
    return {
        "median": b["median"],
        "ci68": [lo, hi],
        "ci90": b["ci90"],
        "half_width": 0.5 * (hi - lo),
        "map": b["map"],
        "logL_max": d["logL_max"],
        "n_rejected": d["n_rejected"],
        "Neff_min": d["guard"]["summary"]["Neff_min"],
        "Neff_max": d["guard"]["summary"]["Neff_max"],
        "pe_variance_sum_max": d["guard"]["summary"]["pe_variance_sum_max"],
    }


def main() -> None:
    out = {
        "what": "analysis-3 configuration (true-n0 field term) vs analysis-2 "
        "configuration (log10n0 = -24) on the SAME complete surveys, same events, "
        "same injections, seed 100, targeted lane",
        "analysis_2_config": {"log10n0": -24.0, "log10n0_c2": -24.0},
        "analysis_3_config": {"log10n0": -3.0, "log10n0_c2": -5.0},
        "scans": {},
    }
    verdict_ok = True
    missing = []
    for name, key, p2, p3 in PAIRS:
        if not p3.exists():
            print(f"[MISSING] {name}: {p3} was never written")
            missing.append(str(p3))
            verdict_ok = False
            continue
        d2, d3 = json.loads(p2.read_text()), json.loads(p3.read_text())
        b2, b3 = block(d2, key), block(d3, key)
        dmed = b3["median"] - b2["median"]
        dhw = b3["half_width"] - b2["half_width"]
        in_hw = dmed / b2["half_width"] if b2["half_width"] else float("nan")
        rec = {
            "parameter": key,
            "analysis_2": b2,
            "analysis_3": b3,
            "shift_median": dmed,
            "shift_median_in_a2_half_widths": in_hw,
            "shift_half_width": dhw,
            "half_width_ratio": b3["half_width"] / b2["half_width"],
            "delta_logL_max": b3["logL_max"] - b2["logL_max"],
        }
        out["scans"][name] = rec
        ok = abs(in_hw) <= 0.25 and 0.8 <= rec["half_width_ratio"] <= 1.25
        verdict_ok &= ok
        print(
            f"[{name}] {key}: a2 {b2['median']:.5g} +- {b2['half_width']:.4g}"
            f"   a3 {b3['median']:.5g} +- {b3['half_width']:.4g}"
            f"   shift {dmed:+.5g} ({in_hw:+.3f} a2-half-widths)"
            f"   width ratio {rec['half_width_ratio']:.4f}"
            f"   {'OK' if ok else 'LARGE'}"
        )
        print(
            f"          Neff a2 [{b2['Neff_min']:.4g}, {b2['Neff_max']:.4g}]"
            f"  a3 [{b3['Neff_min']:.4g}, {b3['Neff_max']:.4g}]"
            f"   rejected a2 {b2['n_rejected']}  a3 {b3['n_rejected']}"
        )
    out["verdict"] = {
        "criterion": "median shift <= 0.25 of analysis 2's own 68% half-width and "
        "half-width ratio within [0.80, 1.25] on BOTH cuts; a missing cut fails",
        "missing": missing,
        "pass": bool(verdict_ok and not missing),
    }
    p = HERE / "results" / "continuity_vs_analysis2.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"\ncontinuity gate: {'PASS' if verdict_ok else 'FAIL'}")
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
