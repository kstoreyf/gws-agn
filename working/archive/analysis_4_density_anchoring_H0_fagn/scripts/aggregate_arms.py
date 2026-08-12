#!/usr/bin/env python3
"""What mis-anchoring the AGN density costs, arm by arm, rung by rung.

Reads `results/joint_<glev>_<arm>_s100.json` for every arm that is on disk, plus
the EXACT arm — analysis_3's own `results/joint_<glev>_s100.json`, referenced and
never rerun — and writes

  results/arms_summary.json   the full table: per rung x per arm medians, 68/90 %
                              intervals, offsets against both f references, the
                              f_AGN detection significance, widths and offsets
                              ratioed to the exact arm, guard behaviour, and the
                              oracle probe against analysis_3's m18 rung

The sweep is one scalar.  `log10n0_c2 = -5 + log10(factor)` for
factor in {0.5, 0.7, 0.9, 1.1, 1.3, 2.0}; the exact arm is factor = 1.  The GAL
density stays at truth everywhere, so every difference below is attributable to
the AGN anchor alone.

TWO TRUTH REFERENCES FOR f, both reported, exactly as in analyses 2 and 3:
  * the REALISED seed-100 host fraction (295/1000 = 0.295) -- the closure
    reference, what the drawn events actually contain
  * the PLANTED 0.30, carrying the mock's own binomial term
    sqrt(0.3 x 0.7 / 1000) = 0.0145
For H0 there is one truth, 67.74.

THE DETECTION SIGNIFICANCE of f_AGN is quoted as median / halfwidth68 -- the
same statistic the prototype (experiments/experiment_completeness_free) used
when it reported that a factor-2 anchoring error halves it.  It measures
distance from f = 0, not closure; closure is the offset columns.

Safe to run mid-campaign: arms whose grids are not yet on disk are recorded as
`present: false` and dropped from every aggregate, never imputed.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
A3 = ROOT.parent / "analysis_3_incomplete_catalog_H0_fagn" / "results"
DATA_ROOT = Path("/hildafs/projects/phy230014p/magana/gws-agn/working/data")
H0_TRUTH = 67.74
F_PLANTED = 0.30
LOG10N0_C2_TRUE = -5.0

LEVELS = ["m21", "m20", "m19", "m18"]
# tag -> density factor applied to the true AGN comoving density
ARMS = {"a05": 0.5, "a07": 0.7, "a09": 0.9,
        "exact": 1.0,
        "a11": 1.1, "a13": 1.3, "a20": 2.0}
ARM_ORDER = ["a05", "a07", "a09", "exact", "a11", "a13", "a20"]


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--levels", nargs="+", default=LEVELS)
    ap.add_argument("--arms", nargs="+", default=ARM_ORDER)
    ap.add_argument("--resdir", default=str(ROOT / "results"))
    ap.add_argument("--a3_resdir", default=str(A3))
    ap.add_argument("--data_root", default=str(DATA_ROOT))
    return ap.parse_args(argv)


def jload(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def realised_f(data_root, seed):
    m = json.loads((Path(data_root) / f"seed{seed}" / "META.json").read_text())
    r = m["stages"]["events"]["realised"]
    n_agn, n_gal = int(r["n_host_agn"]), int(r["n_host_gal"])
    return {"n_host_agn": n_agn, "n_host_gal": n_gal,
            "n_events": n_agn + n_gal, "f_realised": n_agn / (n_agn + n_gal)}


def arm_path(resdir, a3_resdir, level, arm, seed):
    """The exact arm is analysis_3's grid; every other arm is ours."""
    if arm == "exact":
        return Path(a3_resdir) / f"joint_{level}_s{seed}.json"
    return Path(resdir) / f"joint_{level}_{arm}_s{seed}.json"


def block(b, truth):
    """median / CI / half-widths / offset for one marginal block."""
    if b is None:
        return None
    lo68, hi68 = b["ci68"]
    lo90, hi90 = b["ci90"]
    med = b["median"]
    hw = 0.5 * (hi68 - lo68)
    out = {"median": med, "map": b.get("map"),
           "ci68": [lo68, hi68], "ci90": [lo90, hi90],
           "minus68": med - lo68, "plus68": hi68 - med,
           "halfwidth68": hw, "width68": hi68 - lo68, "width90": hi90 - lo90}
    if truth is not None:
        out["truth"] = truth
        out["offset"] = med - truth
        out["truth_in_ci68"] = bool(lo68 <= truth <= hi68)
        out["truth_in_ci90"] = bool(lo90 <= truth <= hi90)
        out["pull"] = (med - truth) / hw if hw > 0 else float("nan")
    return out


def summarise_grid(j, f_real):
    """One scan's JSON -> the row payload."""
    H = block(j.get("H0"), H0_TRUTH)
    F = block(j.get("f"), f_real)
    Fp = block(j.get("f"), F_PLANTED)
    hw = F["halfwidth68"]
    g = j.get("guard", {}) or {}
    return {
        "H0": H, "f_vs_realised": F, "f_vs_planted": Fp,
        "significance_f": (F["median"] / hw) if hw > 0 else float("nan"),
        "rho": j.get("rho"),
        "map": j.get("map"),
        "moments": j.get("moments"),
        "n_rejected": j.get("n_rejected"),
        "logL_max": j.get("logL_max"),
        "guard": {k: v for k, v in (g.get("summary") or {}).items()
                  if k in ("Neff_min", "Neff_median", "sigma2_total_max",
                           "threshold_max", "n_guard_would_reject")},
    }


def ratios(row, ref):
    """This arm against the exact arm, the quantities the campaign is about."""
    if row is None or ref is None:
        return None
    out = {}
    for key, p in (("H0", "H0"), ("f", "f_vs_realised")):
        a, b = row[p], ref[p]
        out[key] = {
            "delta_median": a["median"] - b["median"],
            "delta_median_over_ref_halfwidth68":
                ((a["median"] - b["median"]) / b["halfwidth68"]
                 if b["halfwidth68"] > 0 else float("nan")),
            "width_ratio": (a["halfwidth68"] / b["halfwidth68"]
                            if b["halfwidth68"] > 0 else float("nan")),
            "offset_ratio": (a["offset"] / b["offset"]
                             if b["offset"] not in (0, None) else float("nan")),
        }
    out["significance_f_ratio"] = (
        row["significance_f"] / ref["significance_f"]
        if ref["significance_f"] not in (0, None)
        and np.isfinite(ref["significance_f"]) else float("nan"))
    return out


def main(argv=None):
    a = parse_args(argv)
    R, A3R = Path(a.resdir), Path(a.a3_resdir)
    rf = realised_f(a.data_root, a.seed)
    f_real = rf["f_realised"]

    rungs = {}
    for lev in a.levels:
        ref_j = jload(arm_path(R, A3R, lev, "exact", a.seed))
        ref = summarise_grid(ref_j, f_real) if ref_j else None
        arms = {}
        for arm in a.arms:
            p = arm_path(R, A3R, lev, arm, a.seed)
            j = jload(p)
            if j is None:
                arms[arm] = {"present": False, "factor": ARMS.get(arm),
                             "path": str(p)}
                continue
            row = summarise_grid(j, f_real)
            row.update({
                "present": True,
                "factor": ARMS.get(arm),
                "log10n0_c2": (LOG10N0_C2_TRUE + math.log10(ARMS[arm])
                               if arm in ARMS else None),
                "source": "analysis_3 (referenced, not rerun)" if arm == "exact"
                          else "analysis_4",
                "path": str(p),
                "vs_exact": ratios(row, ref) if arm != "exact" else None,
            })
            arms[arm] = row
        rungs[lev] = {"arms": arms,
                      "n_present": sum(1 for v in arms.values() if v.get("present")),
                      "n_arms": len(a.arms)}

    # ---- the oracle probe ------------------------------------------------------
    # GAL at m < 18, AGN survey COMPLETE, both densities at truth.  Its reference
    # is analysis_3's m18 rung: same galaxies, same events, sparse AGN completion.
    oracle = None
    oj = jload(R / f"joint_m18_oracle_s{a.seed}.json")
    if oj:
        orow = summarise_grid(oj, f_real)
        m18 = rungs.get("m18", {}).get("arms", {}).get("exact")
        oracle = {
            "present": True,
            "_what": "GAL m < 18, AGN survey complete, log10n0 = -3 and "
                     "log10n0_c2 = -5 both at truth; tests whether the f_AGN "
                     "bias at the faintest rung is manufactured by the sparse "
                     "AGN completion rather than by the galaxy incompleteness",
            **orow,
            "vs_m18_exact": ratios(orow, m18) if m18 and m18.get("present") else None,
            "bias_removed_fraction": (
                1 - abs(orow["f_vs_realised"]["offset"])
                / abs(m18["f_vs_realised"]["offset"])
                if m18 and m18.get("present")
                and m18["f_vs_realised"]["offset"] else None),
        }
    else:
        oracle = {"present": False,
                  "path": str(R / f"joint_m18_oracle_s{a.seed}.json")}

    # ---- the headline: how the anchor propagates ------------------------------
    # For each rung, the spread across arms of the f_AGN offset and significance,
    # and the factor-2 arms specifically (the prototype's reported case).
    sweep = {}
    for lev, d in rungs.items():
        got = {k: v for k, v in d["arms"].items() if v.get("present")}
        if not got:
            continue
        ex = got.get("exact")
        s = {"n_present": len(got), "arms_present": sorted(got)}
        for key, p in (("H0", "H0"), ("f", "f_vs_realised")):
            offs = [v[p]["offset"] for v in got.values()]
            hws = [v[p]["halfwidth68"] for v in got.values()]
            s[key] = {
                "offset_min": float(np.min(offs)), "offset_max": float(np.max(offs)),
                "offset_span": float(np.max(offs) - np.min(offs)),
                "offset_span_over_exact_halfwidth68":
                    (float(np.max(offs) - np.min(offs)) / ex[p]["halfwidth68"]
                     if ex and ex[p]["halfwidth68"] > 0 else None),
                "halfwidth68_min": float(np.min(hws)),
                "halfwidth68_max": float(np.max(hws)),
                "halfwidth68_ratio_max_over_min":
                    float(np.max(hws) / np.min(hws)) if np.min(hws) > 0 else None,
            }
        sig = {k: v["significance_f"] for k, v in got.items()}
        s["significance_f"] = sig
        if ex:
            s["significance_f_ratio_vs_exact"] = {
                k: (v / ex["significance_f"]) for k, v in sig.items()
                if ex["significance_f"]}
        sweep[lev] = s

    out = {
        "_what": "the joint (H0, f_AGN) measurement down the completeness ladder "
                 "with the completion's AGN density anchor log10n0_c2 set OFF the "
                 "mock's truth by a known factor, one scalar per arm; darksirens "
                 "K = 2, field sky weighting, survey order [GAL, AGN] so "
                 "fcat_2 = f_AGN, GAL density fixed at truth (-3), guard hard, "
                 "W = 4096, grid 201 x 41, seed 100 targeted-injection lane",
        "_partial_note": "arms with present = false are not yet on disk; every "
                         "aggregate below is computed from the present arms only",
        "seed": a.seed,
        "truth": {"H0": H0_TRUTH, "f_planted": F_PLANTED, **rf,
                  "log10n0_c2_true": LOG10N0_C2_TRUE,
                  "log10n0_gal_true": -3.0},
        "binomial_sd_per_realisation": math.sqrt(F_PLANTED * (1 - F_PLANTED) / 1000),
        "arm_factors": ARMS,
        "rungs": rungs,
        "sweep": sweep,
        "oracle": oracle,
        "progress": {"n_grids_present":
                     sum(d["n_present"] - (1 if d["arms"].get("exact", {}).get("present")
                                           else 0) for d in rungs.values())
                     + (1 if oracle.get("present") else 0),
                     "n_grids_expected": 25},
    }
    R.mkdir(parents=True, exist_ok=True)
    (R / "arms_summary.json").write_text(json.dumps(out, indent=2))
    print(f"Wrote {R / 'arms_summary.json'}")

    # ---- console table ---------------------------------------------------------
    print(f"\n=== AGN density anchoring, seed {a.seed} "
          f"(truth H0 = {H0_TRUTH}, f_realised = {f_real:.3f}) ===")
    print(f"{'rung':>5} {'arm':>6} {'factor':>6} {'log10n0_c2':>10} | "
          f"{'H0 med':>7} {'+-68':>5} {'off':>6} | {'f med':>6} {'+-68':>6} "
          f"{'off':>7} {'S/N':>5} | {'dH0/sig':>8} {'df/sig':>7} {'S/N x':>6}")
    for lev in a.levels:
        for arm in a.arms:
            v = rungs[lev]["arms"].get(arm, {})
            if not v.get("present"):
                continue
            H, F = v["H0"], v["f_vs_realised"]
            r = v.get("vs_exact")
            dh = f"{r['H0']['delta_median_over_ref_halfwidth68']:+8.2f}" if r else f"{'ref':>8}"
            df = f"{r['f']['delta_median_over_ref_halfwidth68']:+7.2f}" if r else f"{'ref':>7}"
            sr = f"{r['significance_f_ratio']:6.2f}" if r else f"{'1.00':>6}"
            print(f"{lev:>5} {arm:>6} {v['factor']:>6.1f} {v['log10n0_c2']:>10.3f} | "
                  f"{H['median']:>7.2f} {H['halfwidth68']:>5.2f} {H['offset']:>+6.2f} | "
                  f"{F['median']:>6.3f} {F['halfwidth68']:>6.3f} {F['offset']:>+7.3f} "
                  f"{v['significance_f']:>5.1f} | {dh} {df} {sr}")
        print()
    if oracle.get("present"):
        print("=== oracle probe: GAL m < 18, AGN complete, densities at truth ===")
        H, F = oracle["H0"], oracle["f_vs_realised"]
        print(f"  H0 = {H['median']:.2f} +- {H['halfwidth68']:.2f} "
              f"(offset {H['offset']:+.2f}); "
              f"f = {F['median']:.3f} +- {F['halfwidth68']:.3f} "
              f"(offset {F['offset']:+.3f})")
        if oracle.get("bias_removed_fraction") is not None:
            print(f"  f_AGN bias removed vs the m18 exact arm: "
                  f"{100 * oracle['bias_removed_fraction']:.0f} %")
    else:
        print("=== oracle probe: not on disk yet ===")
    p = out["progress"]
    print(f"\ngrids on disk: {p['n_grids_present']} / {p['n_grids_expected']}")


if __name__ == "__main__":
    main()
