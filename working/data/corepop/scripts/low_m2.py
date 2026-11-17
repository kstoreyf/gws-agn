#!/usr/bin/env python
"""Smallest true secondary mass and largest fraction of an event's posterior samples below
5 Msun, for every corepop file: Analysis 14's diag_low_m2.table(), imported unchanged and
pointed at the new files (its FILES tuple is the only thing set here).

    python low_m2.py          (any env with numpy + h5py)

The legacy rows of the same seeds come from Analysis 14's results/diag_low_m2.json, and
the legacy host-dependent set (seed 100 only) is run through the same table() here.
Writes results/low_m2.json.
"""
import importlib.util
import json
from pathlib import Path

CP = Path(__file__).resolve().parents[1]
A14 = Path("/hildafs/projects/phy230014p/magana/gws-agn/working/analyses/"
           "analysis_14_monte_carlo_accuracy")
SEEDS = (100, 101, 102, 103, 105)
NEW = ("events_corepop.h5", "events_pureagn_corepop.h5", "events_puregal_corepop.h5",
       "events_marked_dmu0p10_dmuG5_corepop.h5")
OLD = {"events_corepop.h5": "events.h5", "events_pureagn_corepop.h5": "events_pureagn.h5",
       "events_puregal_corepop.h5": "events_puregal.h5",
       "events_marked_dmu0p10_dmuG5_corepop.h5": "events_marked_dmu0p10_dmuG5.h5"}


def main():
    spec = importlib.util.spec_from_file_location("diag_low_m2", A14 / "scripts" / "diag_low_m2.py")
    D = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(D)
    D.FILES = NEW + ("events_marked_dmu0p10_dmuG5.h5",)
    rows = [r for r in D.table() if r["seed"] in SEEDS]
    a14 = json.loads((A14 / "results" / "diag_low_m2.json").read_text())["event_sets"]
    legacy = [r for r in a14 if r["seed"] in SEEDS] + \
             [r for r in rows if r["file"] == "events_marked_dmu0p10_dmuG5.h5"]
    core = [r for r in rows if r["file"] in NEW]
    for r in core:
        r["legacy_counterpart"] = next((x for x in legacy if x["seed"] == r["seed"]
                                        and x["file"] == OLD[r["file"]]), None)
    out = {"what": "smallest true m2src, events with true m2src < 5, events with > 95% of their "
                   "posterior samples at m2src < 5, and the largest such fraction, per file "
                   "(diag_low_m2.table from analysis 14)",
           "m_min": D.M_MIN, "corepop": core,
           "summary": {
               "corepop_min_true_m2src": min(r["min_true_m2src"] for r in core),
               "corepop_n_events_true_m2src_below_5": sum(r["n_true_m2src_below_5"] for r in core),
               "corepop_n_events_gt_0p95_samples_below_5":
                   sum(r["n_events_frac_samples_below_5_gt_0p95"] for r in core),
               "corepop_max_frac_samples_below_5": max(r["max_frac_samples_below_5"] for r in core),
               "legacy_same_seeds_n_events_true_m2src_below_5":
                   sum(r["n_true_m2src_below_5"] for r in legacy),
               "legacy_same_seeds_n_sets_with_event_gt_0p95_below_5":
                   sum(r["n_events_frac_samples_below_5_gt_0p95"] > 0 for r in legacy)}}
    (CP / "results" / "low_m2.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out["summary"], indent=1))
    print("wrote", CP / "results" / "low_m2.json")


if __name__ == "__main__":
    main()
