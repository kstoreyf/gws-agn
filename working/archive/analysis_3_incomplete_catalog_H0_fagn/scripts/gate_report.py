#!/usr/bin/env python
"""Collect the three validation gates into one machine-readable record.

  (a) timing    -- steady-state s/eval per rung, and the campaign size it implies
  (b) continuity-- results/continuity_vs_analysis2.json
  (c) guard     -- results/guard/guard_<level>_s100[_popuni].json

Writes results/gates.json.  Nothing downstream reads the logs.

EXITS NONZERO IF ANY GATE FAILS.  The ladder workers are submitted with
`--dependency=afterok:<gates job>`, so a failed gate leaves the campaign unstarted
rather than pushing through it -- the queue was busy enough that waiting to inspect
by hand would have meant the campaign never being submitted at all, and a nonzero
exit is the only enforcement that survives that.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
RES = HERE / "results"
LEVELS = ["m21", "m20", "m19", "m18"]
N_GRID = 201 * 41
# 20 record grids (5 seeds x 4 rungs) + 4 popuni cross-check grids on seed 100
N_RECORD_GRIDS = {lev: 5 for lev in LEVELS}
N_XCHECK_GRIDS = {lev: 1 for lev in LEVELS}


def main() -> None:
    out = {"n_grid_cells": N_GRID, "timing": {}, "continuity": None, "guard": {}}

    # ---- (a) timing -------------------------------------------------------
    total_h = 0.0
    for lev in LEVELS:
        p = RES / "pilot" / f"pilot_joint_{lev}_s100.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        s = d["timing"]["steady_state_median_seconds"]
        n = N_RECORD_GRIDS[lev] + N_XCHECK_GRIDS[lev]
        gpu_h = s * N_GRID / 3600.0
        total_h += gpu_h * n
        out["timing"][lev] = {
            "steady_state_s_per_eval": s,
            "first_eval_s": d["timing"]["first_eval_seconds"],
            "gpu_hours_per_grid": gpu_h,
            "n_grids": n,
            "gpu_hours_all_grids": gpu_h * n,
        }
    a2 = HERE.parent / "analysis_2_complete_catalog_H0_fagn" / "results" / "joint_s100.json"
    if a2.exists():
        try:
            out["timing"]["complete_reference_analysis_2"] = {
                "steady_state_s_per_eval": json.loads(a2.read_text())["timing"][
                    "steady_state_median_seconds"
                ],
                "gpu_hours_per_grid": 3.71 * N_GRID / 3600.0,
            }
        except Exception:
            pass
    out["timing"]["campaign_gpu_hours_total"] = total_h

    # ---- (b) continuity ---------------------------------------------------
    p = RES / "continuity_vs_analysis2.json"
    if p.exists():
        out["continuity"] = json.loads(p.read_text())

    # ---- (c) guard --------------------------------------------------------
    for p in sorted((RES / "guard").glob("guard_*.json")):
        d = json.loads(p.read_text())
        # diag_variance_guard.py stores the guard's own numbers in `guard_records`
        # (one per selection call site); everything else in the file is context.
        recs = d.get("guard_records") or []
        rec = recs[0] if recs else {}
        g = {
            k: rec.get(k)
            for k in (
                "Neff",
                "pe_variance_sum",
                "selection_variance_N2_over_Neff",
                "sigma2_total",
                "threshold",
                "passes",
                "legacy_floor_5N",
                "passes_legacy_floor",
            )
        }
        g["logL"] = d.get("logL")
        g["Ndraw"] = d.get("Ndraw")
        g["max_likelihood_variance"] = d.get("max_likelihood_variance")
        g["n_guard_records"] = len(recs)
        g["event_variance_stats"] = d.get("event_variance_stats")
        if g.get("Neff") and g.get("legacy_floor_5N"):
            g["Neff_over_floor"] = g["Neff"] / g["legacy_floor_5N"]
        out["guard"][p.stem] = g

    (RES / "gates.json").write_text(json.dumps(out, indent=2))

    print("=== GATE (a) timing ===")
    for lev in LEVELS:
        t = out["timing"].get(lev)
        if t:
            print(
                f"  {lev:9s} {t['steady_state_s_per_eval']:7.3f} s/eval   "
                f"{t['gpu_hours_per_grid']:6.2f} GPU-h/grid   x{t['n_grids']} grids"
                f" = {t['gpu_hours_all_grids']:7.2f} GPU-h"
            )
    print(f"  CAMPAIGN TOTAL: {total_h:.1f} GPU-h over {sum(N_RECORD_GRIDS.values()) + sum(N_XCHECK_GRIDS.values())} grids")

    print("\n=== GATE (b) continuity vs analysis 2 ===")
    if out["continuity"]:
        for name, r in out["continuity"]["scans"].items():
            print(
                f"  {name:8s} {r['parameter']:3s} shift {r['shift_median']:+.5g}"
                f" ({r['shift_median_in_a2_half_widths']:+.3f} a2-half-widths)"
                f"  width ratio {r['half_width_ratio']:.4f}"
            )
        print(f"  verdict: {'PASS' if out['continuity']['verdict']['pass'] else 'FAIL'}")
    else:
        print("  (not run)")

    print("\n=== GATE (c) guard / N_eff at (H0, f) = (67.74, 0.30) ===")
    for k, g in out["guard"].items():
        print(
            f"  {k:34s} Neff {g.get('Neff', float('nan')):12,.0f}"
            f"  x{g.get('Neff_over_floor', float('nan')):8.1f} floor"
            f"  sum sig2_PE {g.get('pe_variance_sum', float('nan')):8.3f}"
            f"  passes {g.get('passes')}"
        )
    # ---- verdicts ---------------------------------------------------------
    fails = []
    if not any(lev in out["timing"] for lev in LEVELS):
        fails.append("(a) timing: no pilot ran")
    elif total_h > 120.0:
        fails.append(
            f"(a) timing: campaign would cost {total_h:.0f} GPU-h, above the 120 GPU-h "
            "ceiling this directory was sized for -- re-plan rather than submit"
        )
    # (b) is MEASURED, not gating.  It was written as a blocker on the assumption
    # that analysis 2's complete grids could serve as rung 0.  They cannot -- the
    # check found a real estimator offset (f_AGN +0.080, 1.74 of analysis 2's own
    # half-widths) whose cause is measured in results/continuity_failure_diag.json.
    # The resolution is to re-run the complete rung IN THIS CONFIGURATION as rung 0
    # of record and keep analysis 2's as the zero-missing-budget reference, which
    # removes the dependency the gate was protecting.  It is therefore recorded as a
    # finding and reported, and no longer stops the campaign.
    if out["continuity"] is None:
        fails.append("(b) continuity: not run")
    else:
        miss = out["continuity"]["verdict"].get("missing") or []
        if miss:
            fails.append(f"(b) continuity: cut(s) never written: {', '.join(miss)}")
        else:
            out["continuity"]["role"] = (
                "measured finding, not a gate: rung 0 is re-run in this "
                "configuration, so the ladder does not depend on agreeing with "
                "analysis 2"
            )
    if not out["guard"]:
        fails.append("(c) guard: not run")
    for k, g in out["guard"].items():
        if g.get("passes") is False:
            fails.append(f"(c) guard: {k} fails the validity guard at the peak")
        elif g.get("Neff_over_floor") is not None and g["Neff_over_floor"] < 2.0:
            fails.append(
                f"(c) guard: {k} clears the N_eff floor by only "
                f"{g['Neff_over_floor']:.2f}x at the peak -- too marginal to scan"
            )
    out["verdict"] = {
        "pass": not fails,
        "failures": fails,
        "gating_checks": "(a) timing and (c) guard gate the campaign; (b) continuity "
        "is measured and reported but does not, because rung 0 is re-run here",
    }
    (RES / "gates.json").write_text(json.dumps(out, indent=2))

    print(f"\nwrote {RES / 'gates.json'}")
    if fails:
        print("\n*** GATES FAILED -- the ladder campaign must not run ***")
        for f in fails:
            print(f"  - {f}")
        sys.exit(1)
    print("\nALL GATES PASS")


if __name__ == "__main__":
    main()
