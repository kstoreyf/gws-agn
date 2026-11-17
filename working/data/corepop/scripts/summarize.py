#!/usr/bin/env python
"""One row per corepop file: the generator's own checks, the low-m2 numbers, core's density
at the truths, the detected-set KS, and the correspondence with the legacy file.

    python summarize.py          (any env with h5py; after every other script)

Writes results/corepop_summary.json and prints the README's validation table.
"""
import json
from pathlib import Path

import h5py

CP = Path(__file__).resolve().parents[1]
DATA = Path("/hildafs/projects/phy230054p/magana/gws-agn-data-v3")
SEEDS = (100, 101, 102, 103, 105)
KINDS = [("mixed", "_corepop"), ("pureagn", "_pureagn_corepop"), ("puregal", "_puregal_corepop"),
         ("marked", "_marked_dmu0p10_dmuG5_corepop")]


def main():
    low = {(r["seed"], r["file"]): r for r in json.loads((CP / "results/low_m2.json").read_text())["corepop"]}
    tru = {(r["seed"], r["file"]): r for r in
           json.loads((CP / "results/core_logp_truth.json").read_text())["files"]
           if r["population_code"] == "corepop"}
    gof = {(r["seed"], r["file"]): r for r in json.loads((CP / "results/detected_gof.json").read_text())["files"]}
    rows = []
    for kind, sfx in KINDS:
        for s in SEEDS:
            fn = f"events{sfx}.h5"
            p = DATA / f"seed{s}" / "events" / fn
            v = json.loads((DATA / f"seed{s}" / "validation" / f"validation_events{sfx}.json").read_text())
            with h5py.File(p, "r") as f:
                meta = json.loads(f.attrs["metadata_json"])
                attrs = {k: (f.attrs[k].decode() if isinstance(f.attrs[k], bytes) else f.attrs[k])
                         for k in ("population_code", "population_core_commit", "seed_events",
                                   "planted_f_agn", "truth_f_agn")}
            lw, tr, g = low[(s, fn)], tru[(s, fn)], gof[(s, fn)]
            row = {"seed": s, "kind": kind, "file": str(p), "bytes": p.stat().st_size,
                   "seed_events": int(attrs["seed_events"]),
                   "population_code_attr": attrs["population_code"],
                   "core_commit_attr": attrs["population_core_commit"],
                   "checks_passed": [k for k, c in v["checks"].items() if c["pass"]],
                   "checks_failed": v["failed"],
                   "realised_f_agn": float(attrs["truth_f_agn"]),
                   "min_true_m2src": lw["min_true_m2src"],
                   "n_true_m2src_below_5": lw["n_true_m2src_below_5"],
                   "max_frac_samples_below_5": lw["max_frac_samples_below_5"],
                   "n_events_gt_0p95_samples_below_5": lw["n_events_frac_samples_below_5_gt_0p95"],
                   "core_lnp_truth_all_finite_historical": tr["historical"]["all_finite"],
                   "core_lnp_truth_all_finite_default": tr["default"]["all_finite"],
                   "core_sum_lnp_truth": tr["historical"]["sum_lnp_finite"],
                   "core_min_lnp_truth": tr["historical"]["min_lnp"],
                   "core_loader_ok": tr.get("core_loader", {}).get("ok"),
                   "detected_ks_min_pvalue": g["min_ks_pvalue"],
                   "n_detected_with_q_redrawn": g["n_detected_with_q_redrawn"],
                   "core_route_tally": g["core_route_tally"],
                   "correspondence_with_legacy": g.get("correspondence_with_legacy"),
                   "V5_selection_corrected": g.get("V5_selection_corrected"),
                   "offsets_agn_minus_gal": g.get("offsets_agn_minus_gal"),
                   "n_proposed": meta["realised"]["n_proposed"],
                   "detected_fraction": meta["realised"]["detected_fraction"]}
            rows.append(row)
    summ = {"n_files": len(rows),
            "all_generator_checks_pass": all(not r["checks_failed"] for r in rows),
            "all_core_lnp_finite": all(r["core_lnp_truth_all_finite_historical"]
                                       and r["core_lnp_truth_all_finite_default"] for r in rows),
            "all_core_loader_ok": all(r["core_loader_ok"] for r in rows),
            "min_true_m2src": min(r["min_true_m2src"] for r in rows),
            "max_frac_samples_below_5": max(r["max_frac_samples_below_5"] for r in rows),
            "total_bytes_events_h5": sum(r["bytes"] for r in rows)}
    (CP / "results" / "corepop_summary.json").write_text(json.dumps({"summary": summ, "files": rows}, indent=1))
    print(json.dumps(summ, indent=1))
    ab = {"V1_detection_deterministic_in_data": "V1", "V2_widths_from_observed_snr": "V2",
          "V2b_ra_width_from_observed_dec": "V2b", "V3_pe_calibration": "V3",
          "V3c_p_pe_jacobian": "V3c", "V5_planted_f_bookkeeping": "V5",
          "V7_darksirens_format_contract": "V7"}
    print("\n| seed | set | MB | checks passed | min true m2 | max frac < 5 | core ln p finite | "
          "KS min p | q redrawn | shared w/ legacy |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        c = r["correspondence_with_legacy"]
        print(f"| {r['seed']} | {r['kind']} | {r['bytes'] / 1e6:.1f} | "
              f"{' '.join(ab[k] for k in r['checks_passed'])} | {r['min_true_m2src']:.2f} | "
              f"{r['max_frac_samples_below_5']:.3f} | "
              f"{'yes' if r['core_lnp_truth_all_finite_historical'] else 'NO'} | "
              f"{r['detected_ks_min_pvalue']:.3f} | {r['n_detected_with_q_redrawn']} | "
              f"{c['n_identical_latent_draw'] if c else '-'} |")


if __name__ == "__main__":
    main()
