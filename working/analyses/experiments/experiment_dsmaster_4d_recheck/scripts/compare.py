#!/usr/bin/env python
"""Diff this experiment's master-built 4D fit against analysis 5's 2b86a2d fit,
using analysis 5's own rstate-23 replicate as the sampling-noise floor.

    python scripts/compare.py                 # m18
    python scripts/compare.py --rung m20

Writes results/comparison_<rung>.json and prints the verdict table.
"""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
A5 = Path("/hildafs/projects/phy230014p/magana/gws-agn/working/analyses/"
          "analysis_5_free_anchors_H0_fagn/results")

PARAMS = ("H0", "log10n0", "log10n0_c2", "f_AGN")


def load(p):
    if not p.exists():
        raise SystemExit(f"[fatal] missing {p}")
    return json.loads(p.read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung", default="m18")
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--tag", default=None,
                    help="explicit result tag; defaults to fit_<rung>_<cmode>_s<seed>")
    ap.add_argument("--cmode", default="aggregate",
                    choices=["aggregate", "per_pixel"])
    a = ap.parse_args()

    tag = a.tag or f"fit_{a.rung}_{a.cmode}_s{a.seed}"
    new = load(EXP / "results" / f"{tag}.json")
    old = load(A5 / f"campaign_{a.rung}_dynesty_s{a.seed}.json")
    rep_path = A5 / f"campaign_{a.rung}_dynesty_r2_s{a.seed}.json"
    rep = load(rep_path) if rep_path.exists() else None

    rows = []
    for k in PARAMS:
        o, n = old["summary"][k], new["summary"][k]
        d_med = n["median"] - o["median"]
        d_sd = n["sd"] - o["sd"]
        floor = (abs(rep["summary"][k]["median"] - o["median"])
                 if rep else float("nan"))
        rows.append({
            "param": k,
            "old_median": o["median"], "new_median": n["median"],
            "delta_median": d_med,
            "old_sd": o["sd"], "new_sd": n["sd"], "delta_sd": d_sd,
            "replicate_floor": floor,
            # a shift is only meaningful if it clears BOTH the sampler's own
            # scatter and a decent fraction of the posterior width
            "ratio_to_floor": (abs(d_med) / floor) if floor else float("inf"),
            "delta_in_sigma": abs(d_med) / o["sd"] if o["sd"] else float("nan"),
        })

    d_logz = (new["sampler_meta"]["logz"] - old["sampler_meta"]["logz"])
    logz_floor = (abs(rep["sampler_meta"]["logz"] - old["sampler_meta"]["logz"])
                  if rep else float("nan"))

    verdict = ("UNCHANGED" if all(r["ratio_to_floor"] <= 5 for r in rows)
               else "CHANGED")

    out = {
        "_what": "darksirens master (this experiment) vs 2b86a2d (analysis 5), "
                 "same rung/seed/rstate. replicate_floor is analysis 5's own "
                 "rstate-23 twin, i.e. dynesty's scatter; a delta below it is "
                 "not a code change.",
        "rung": a.rung, "seed": a.seed,
        "new_sha": new.get("darksirens_git_sha"),
        "old_sha": old.get("darksirens_git_sha"),
        "new_wiring_max_abs_diff": new.get("wiring_check_max_abs_diff"),
        "old_wiring_max_abs_diff": old.get("wiring_check_max_abs_diff"),
        "params": rows,
        "delta_logz": d_logz, "logz_replicate_floor": logz_floor,
        "verdict": verdict,
    }
    out["c_mode_new"] = new.get("c_mode", "per_pixel (unrecorded)")
    (EXP / "results" / f"comparison_{tag}.json").write_text(
        json.dumps(out, indent=2))

    print(f"\n  darksirens {out['old_sha'][:7]} -> {out['new_sha'][:7]}   "
          f"rung {a.rung}, seed {a.seed}, c_mode={out['c_mode_new']}")
    print(f"  wiring max|diff|: {out['old_wiring_max_abs_diff']:.3e} -> "
          f"{out['new_wiring_max_abs_diff']:.3e}\n")
    hdr = f"  {'param':<12}{'2b86a2d':>12}{'master':>12}{'delta':>12}{'floor':>10}{'x floor':>9}"
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for r in rows:
        print(f"  {r['param']:<12}{r['old_median']:>12.4f}{r['new_median']:>12.4f}"
              f"{r['delta_median']:>12.4f}{r['replicate_floor']:>10.4f}"
              f"{r['ratio_to_floor']:>9.1f}")
    print(f"\n  log Z {old['sampler_meta']['logz']:.2f} -> "
          f"{new['sampler_meta']['logz']:.2f}  (delta {d_logz:+.3f}, "
          f"floor {logz_floor:.3f})")
    print(f"\n  VERDICT: {verdict}\n")


if __name__ == "__main__":
    main()
