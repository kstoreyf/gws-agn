#!/usr/bin/env python3
"""The completeness-ladder closure table for the joint (H0, f_AGN) measurement.

Reads, per rung and per seed, `results/joint_<level>_s<seed>.json`, plus the seed's
own `META.json` for the REALISED host fraction and that rung's realised
completeness, and writes

  results/ladder_summary.json   the full table: per-rung x per-seed medians,
                                68/90 % intervals, offsets against both f
                                references, coverage flags, rho, guard/N_eff
                                behaviour, lane agreement, the null
  results/h0_fagn_ladder.json   the compact hook (analysis_2 conventions)

RUNG 0 IS ANALYSIS 2'S OWN RESULT, read from
`../analysis_2_complete_catalog_H0_fagn/results/joint_s<seed>.json`.  It is not
re-run here: it is the same events, the same injections and the same grid, and the
one configuration difference (the out-of-catalog field term suppressed to
log10n0 = -24 rather than switched on at the true density) is measured directly by
the continuity gate, `results/continuity_vs_analysis2.json`.  The ladder is quoted
DIFFERENTIALLY against that rung, so the gate's shift is the systematic floor on
every "x rung 0" ratio below.

TWO TRUTH REFERENCES FOR f, both reported, exactly as in analysis 2:
  * the REALISED per-seed host fraction (n_AGN / 1000) -- the closure reference
  * the PLANTED 0.30 -- the population parameter, carrying the extra binomial
    term sqrt(0.3 x 0.7 / 1000) = 0.0145 per realisation
For H0 there is one truth, 67.74.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
A2 = ROOT.parent / "analysis_2_complete_catalog_H0_fagn" / "results"
DATA_ROOT = Path("/hildafs/projects/phy230014p/magana/gws-agn/working/data")
H0_TRUTH = 67.74
F_PLANTED = 0.30
LEVELS = ["complete", "m21", "m20", "m19", "m18"]


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--seeds", nargs="+", type=int, default=[100, 101, 102, 103, 105])
    ap.add_argument("--levels", nargs="+", default=LEVELS)
    ap.add_argument("--resdir", default=str(ROOT / "results"))
    ap.add_argument("--data_root", default=str(DATA_ROOT))
    ap.add_argument("--lane_seed", type=int, default=100)
    return ap.parse_args(argv)


def load(path):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None


def joint_path(resdir, level, seed, lane="targeted"):
    """Every rung, INCLUDING the complete one, comes from this directory.

    The complete rung used to be read from analysis 2.  It is not any more: the
    continuity check measured that analysis 2's log10n0 = -24 run is a different
    estimator on this data (f_AGN +0.080, 1.74 of its own 68 % half-widths, cause
    in results/continuity_failure_diag.json), so using it as rung 0 would fold that
    estimator offset into every completeness ratio.  Analysis 2 is still reported,
    by a2_joint_path below, as the zero-missing-budget REFERENCE."""
    suf = "_popuni" if lane == "popuni" else ""
    return Path(resdir) / f"joint_{level}_s{seed}{suf}.json"


def a2_joint_path(seed, lane="targeted"):
    """Analysis 2's complete-catalog grid: the zero-missing-budget reference."""
    suf = "_popuni" if lane == "popuni" else ""
    return A2 / f"joint_s{seed}{suf}.json"


def meta(data_root, seed):
    return json.loads((Path(data_root) / f"seed{seed}" / "META.json").read_text())


def realised_f(m):
    r = m["stages"]["events"]["realised"]
    n_agn, n_gal = int(r["n_host_agn"]), int(r["n_host_gal"])
    return {
        "n_host_agn": n_agn,
        "n_host_gal": n_gal,
        "n_events": n_agn + n_gal,
        "f_realised": n_agn / (n_agn + n_gal),
    }


def completeness(m, level):
    """Realised completeness of this rung's surveys, from the dataset's own META."""
    c = m["stages"]["surveys"].get("completeness") or m.get("completeness") or {}
    lev = c.get(level)
    if lev is None:
        return None
    out = {"horizon_z": m["stages"]["surveys"]["horizon_z"]}
    for t in ("gal", "agn"):
        if t in lev:
            out[t] = {
                "mag_limit": lev[t].get("mag_limit"),
                "n_kept": lev[t].get("n_kept"),
                "C_within_horizon": lev[t].get("C_within_horizon"),
                "C_all_z": lev[t].get("C_all_z"),
            }
    return out


def block(b, truth):
    if b is None:
        return None
    lo68, hi68 = b["ci68"]
    lo90, hi90 = b["ci90"]
    med = b["median"]
    out = {
        "median": med,
        "map": b.get("map"),
        "ci68": [lo68, hi68],
        "ci90": [lo90, hi90],
        "minus68": med - lo68,
        "plus68": hi68 - med,
        "halfwidth68": 0.5 * (hi68 - lo68),
        "width68": hi68 - lo68,
        "width90": hi90 - lo90,
    }
    if truth is not None:
        out["truth"] = truth
        out["offset"] = med - truth
        out["truth_in_ci68"] = bool(lo68 <= truth <= hi68)
        out["truth_in_ci90"] = bool(lo90 <= truth <= hi90)
        out["pull"] = (
            (med - truth) / out["halfwidth68"] if out["halfwidth68"] > 0 else float("nan")
        )
    return out


def mean_sem(v):
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], dtype=float)
    if v.size == 0:
        return {"n": 0}
    m = float(v.mean())
    sd = float(v.std(ddof=1)) if v.size > 1 else float("nan")
    sem = sd / math.sqrt(v.size) if v.size > 1 else float("nan")
    t = m / sem if sem and np.isfinite(sem) and sem > 0 else float("nan")
    return {"n": int(v.size), "mean": m, "sd": sd, "sem": sem, "t": t}


def main(argv=None):
    a = parse_args(argv)
    R = Path(a.resdir)
    metas = {s: meta(a.data_root, s) for s in a.seeds}

    rungs = {}
    for level in a.levels:
        seeds = []
        for s in a.seeds:
            j = load(joint_path(a.resdir, level, s))
            if not j:
                continue
            rf = realised_f(metas[s])
            g = j.get("guard", {}) or {}
            gs = g.get("summary") or {}
            seeds.append(
                {
                    "seed": s,
                    **rf,
                    "H0": block(j.get("H0"), H0_TRUTH),
                    "f_vs_realised": block(j.get("f"), rf["f_realised"]),
                    "f_vs_planted": block(j.get("f"), F_PLANTED),
                    "rho": j.get("rho"),
                    "moments": j.get("moments"),
                    "n_evals": j.get("n_evals"),
                    "n_rejected": j.get("n_rejected"),
                    "logL_max": j.get("logL_max"),
                    "guard_summary": gs,
                    "timing_s_per_eval": (j.get("timing") or {}).get(
                        "steady_state_median_seconds"
                    ),
                    "source": str(joint_path(a.resdir, level, s)),
                }
            )
        if not seeds:
            continue
        rec = {
            "level": level,
            "is_rung_0": level == "complete",
            "provenance": "this directory (true-n0 out-of-catalog field term); "
            + (
                "rung 0 of record, re-run here rather than taken from analysis 2 "
                "so every rung shares one estimator"
                if level == "complete"
                else "magnitude-limited rung"
            ),
            # The flux limit is the same at every rung, but each realisation has
            # its own GW horizon (z_hor = 0.311 - 0.387 across the five seeds), so
            # the completeness INSIDE THE HORIZON -- the only completeness the
            # likelihood feels -- is not the same number at a given rung.  Both the
            # per-seed values and their spread are carried.
            "completeness": {s: completeness(metas[s], level) for s in a.seeds},
            "completeness_within_horizon": {
                t: {
                    "per_seed": {
                        s: completeness(metas[s], level)[t]["C_within_horizon"]
                        for s in a.seeds
                    },
                    "mean": float(
                        np.mean(
                            [
                                completeness(metas[s], level)[t]["C_within_horizon"]
                                for s in a.seeds
                            ]
                        )
                    ),
                    "min": float(
                        min(
                            completeness(metas[s], level)[t]["C_within_horizon"]
                            for s in a.seeds
                        )
                    ),
                    "max": float(
                        max(
                            completeness(metas[s], level)[t]["C_within_horizon"]
                            for s in a.seeds
                        )
                    ),
                }
                for t in ("gal", "agn")
            },
            "seeds": seeds,
            "closure": {
                "H0": mean_sem([r["H0"]["offset"] for r in seeds]),
                "f_vs_realised": mean_sem([r["f_vs_realised"]["offset"] for r in seeds]),
                "f_vs_planted": mean_sem([r["f_vs_planted"]["offset"] for r in seeds]),
                "rho": mean_sem([r["rho"] for r in seeds]),
            },
            "width": {
                "sigma_H0_mean_halfwidth68": float(
                    np.mean([r["H0"]["halfwidth68"] for r in seeds])
                ),
                "sigma_f_mean_halfwidth68": float(
                    np.mean([r["f_vs_realised"]["halfwidth68"] for r in seeds])
                ),
                "sigma_H0_per_seed": [r["H0"]["halfwidth68"] for r in seeds],
                "sigma_f_per_seed": [r["f_vs_realised"]["halfwidth68"] for r in seeds],
            },
            "coverage": {
                "H0_in_68": sum(r["H0"]["truth_in_ci68"] for r in seeds),
                "H0_in_90": sum(r["H0"]["truth_in_ci90"] for r in seeds),
                "f_realised_in_68": sum(r["f_vs_realised"]["truth_in_ci68"] for r in seeds),
                "f_realised_in_90": sum(r["f_vs_realised"]["truth_in_ci90"] for r in seeds),
                "n": len(seeds),
            },
            "guard": {
                "cells_total": int(sum(r["n_evals"] or 0 for r in seeds)),
                "cells_rejected": int(sum(r["n_rejected"] or 0 for r in seeds)),
                "Neff_min": float(
                    min(
                        r["guard_summary"].get("Neff_min", np.inf)
                        for r in seeds
                        if r["guard_summary"]
                    )
                )
                if any(r["guard_summary"] for r in seeds)
                else None,
                "Neff_max": float(
                    max(
                        r["guard_summary"].get("Neff_max", -np.inf)
                        for r in seeds
                        if r["guard_summary"]
                    )
                )
                if any(r["guard_summary"] for r in seeds)
                else None,
                "pe_variance_sum_max": float(
                    max(
                        r["guard_summary"].get("pe_variance_sum_max", -np.inf)
                        for r in seeds
                        if r["guard_summary"]
                    )
                )
                if any(r["guard_summary"] for r in seeds)
                else None,
                # the joint grids record the floor as `threshold_*` (it IS the
                # legacy 5*N_obs floor: the total-variance criterion is inert at
                # max_likelihood_variance = 1e6); the 1-D scans also carry it as
                # `legacy_floor_5N`.
                "legacy_floor_5N": next(
                    (
                        r["guard_summary"].get("legacy_floor_5N")
                        or r["guard_summary"].get("threshold_max")
                        for r in seeds
                        if r["guard_summary"]
                    ),
                    None,
                ),
            },
        }
        rungs[level] = rec

    # ---- the analysis-2 reference: the same complete catalogs with the
    # ---- out-of-catalog budget suppressed (log10n0 = -24).  Not rung 0 -- a
    # ---- second reference, so that COMPLETENESS degradation (rung vs rung 0,
    # ---- one estimator) and the ESTIMATOR's own sparse-pixel offset (rung 0 vs
    # ---- analysis 2, one catalog) are separable rather than summed.
    a2_seeds = []
    for s_ in a.seeds:
        j = load(a2_joint_path(s_))
        if not j:
            continue
        rf = realised_f(metas[s_])
        a2_seeds.append(
            {
                "seed": s_,
                **rf,
                "H0": block(j.get("H0"), H0_TRUTH),
                "f_vs_realised": block(j.get("f"), rf["f_realised"]),
                "rho": j.get("rho"),
                "source": str(a2_joint_path(s_)),
            }
        )
    a2_ref = None
    if a2_seeds:
        a2_ref = {
            "_what": "analysis_2_complete_catalog_H0_fagn: the same complete "
            "catalogs, same events, same injections, same grid, with "
            "log10n0 = log10n0_c2 = -24 so the out-of-catalog term is inert",
            "config": {"log10n0": -24.0, "log10n0_c2": -24.0},
            "seeds": a2_seeds,
            "closure": {
                "H0": mean_sem([r["H0"]["offset"] for r in a2_seeds]),
                "f_vs_realised": mean_sem(
                    [r["f_vs_realised"]["offset"] for r in a2_seeds]
                ),
                "rho": mean_sem([r["rho"] for r in a2_seeds]),
            },
            "width": {
                "sigma_H0_mean_halfwidth68": float(
                    np.mean([r["H0"]["halfwidth68"] for r in a2_seeds])
                ),
                "sigma_f_mean_halfwidth68": float(
                    np.mean([r["f_vs_realised"]["halfwidth68"] for r in a2_seeds])
                ),
            },
        }

    # ---- like-for-like guard -------------------------------------------------
    # Every "x R0" ratio compares a rung's mean width to rung 0's.  While the
    # campaign is draining, different rungs can carry different SEED SETS, and a
    # ratio between means over different realisations is not a completeness
    # effect -- it is partly realisation scatter.  Flag it loudly rather than let
    # a partial table be read as a result.
    seed_sets = {lev: sorted(r["seed"] for r in rec["seeds"])
                 for lev, rec in rungs.items()}
    complete_set = set(seed_sets.get("complete", []))
    like_for_like = all(set(v) == complete_set for v in seed_sets.values()) and \
        len(complete_set) == len(a.seeds)
    ratios_comparable = {
        "like_for_like": bool(like_for_like),
        "seed_sets": seed_sets,
        "note": ("all rungs carry the same complete seed set; the x R0 ratios are "
                 "like-for-like" if like_for_like else
                 "RUNGS CARRY DIFFERENT SEED SETS -- the x R0 ratios below mix "
                 "completeness degradation with realisation scatter and are NOT a "
                 "result until every rung has all requested seeds"),
    }

    # ---- widths against rung 0, and the estimator offset against analysis 2 -----
    if "complete" in rungs:
        w0 = rungs["complete"]["width"]
        for level, rec in rungs.items():
            rec["width"]["sigma_H0_vs_rung0"] = (
                rec["width"]["sigma_H0_mean_halfwidth68"] / w0["sigma_H0_mean_halfwidth68"]
            )
            rec["width"]["sigma_f_vs_rung0"] = (
                rec["width"]["sigma_f_mean_halfwidth68"] / w0["sigma_f_mean_halfwidth68"]
            )
        if a2_ref:
            c0, w2 = rungs["complete"], a2_ref
            # per-seed, so the paired difference is not diluted by realisation scatter
            by_seed = {r["seed"]: r for r in w2["seeds"]}
            dH = [
                r["H0"]["median"] - by_seed[r["seed"]]["H0"]["median"]
                for r in c0["seeds"]
                if r["seed"] in by_seed
            ]
            df = [
                r["f_vs_realised"]["median"] - by_seed[r["seed"]]["f_vs_realised"]["median"]
                for r in c0["seeds"]
                if r["seed"] in by_seed
            ]
            a2_ref["estimator_offset_rung0_minus_analysis2"] = {
                "_what": "rung 0 (true-n0 completion) minus analysis 2 (no "
                "out-of-catalog budget), paired per realisation, on COMPLETE "
                "catalogs where a correct completion would have nothing to do",
                "H0": mean_sem(dH),
                "f": mean_sem(df),
                "H0_in_analysis2_halfwidths": (
                    float(np.mean(dH)) / w2["width"]["sigma_H0_mean_halfwidth68"]
                    if dH
                    else None
                ),
                "f_in_analysis2_halfwidths": (
                    float(np.mean(df)) / w2["width"]["sigma_f_mean_halfwidth68"]
                    if df
                    else None
                ),
                "sigma_H0_ratio_rung0_over_analysis2": (
                    c0["width"]["sigma_H0_mean_halfwidth68"]
                    / w2["width"]["sigma_H0_mean_halfwidth68"]
                ),
                "sigma_f_ratio_rung0_over_analysis2": (
                    c0["width"]["sigma_f_mean_halfwidth68"]
                    / w2["width"]["sigma_f_mean_halfwidth68"]
                ),
                "mechanism": "results/continuity_failure_diag.json; scaling test in "
                "results/nside_scaling.json",
            }

    # ---- lane agreement, per rung, on the cross-check seed ---------------------
    lanes = {}
    s = a.lane_seed
    for level in a.levels:
        rec = load(joint_path(a.resdir, level, s, "targeted"))
        xch = load(joint_path(a.resdir, level, s, "popuni"))
        if not (rec and xch):
            continue
        d = {}
        for p in ("H0", "f"):
            hw = 0.5 * (rec[p]["ci68"][1] - rec[p]["ci68"][0])
            d[p] = {
                "targeted": rec[p]["median"],
                "popuni": xch[p]["median"],
                "delta": xch[p]["median"] - rec[p]["median"],
                "delta_over_halfwidth68": (xch[p]["median"] - rec[p]["median"]) / hw
                if hw > 0
                else float("nan"),
            }
        d["Neff_min"] = {
            "targeted": (rec.get("guard", {}).get("summary") or {}).get("Neff_min"),
            "popuni": (xch.get("guard", {}).get("summary") or {}).get("Neff_min"),
        }
        lanes[level] = d

    # ---- the sky-shuffle null (m18, the worst rung) ---------------------------
    null = None
    nrec = load(R / f"fscan_null_m18_s{s}.json")
    if nrec:
        nb = nrec["f"]
        null = {
            "seed": s,
            "level": "m18",
            "median": nb["median"],
            "ci68": nb["ci68"],
            "ci90": nb["ci90"],
            "halfwidth68": 0.5 * (nb["ci68"][1] - nb["ci68"][0]),
            "map": nb.get("map"),
            "n_rejected": nrec.get("n_rejected"),
        }
        rec = load(joint_path(a.resdir, "m18", s))
        if rec:
            fb = rec["f"]
            null["record_median"] = fb["median"]
            null["record_halfwidth68"] = 0.5 * (fb["ci68"][1] - fb["ci68"][0])
            null["width_ratio_null_over_record"] = (
                null["halfwidth68"] / null["record_halfwidth68"]
                if null["record_halfwidth68"] > 0
                else float("nan")
            )
            # the prototype's combined statistic: displacement in null widths
            null["separation_in_null_widths"] = (
                (null["record_median"] - null["median"]) / null["halfwidth68"]
                if null["halfwidth68"] > 0
                else float("nan")
            )

    out = {
        "_what": "the joint (H0, f_AGN) measurement down a magnitude-limited host "
        "survey ladder; dark_sirens K = 2 mixture with the out-of-catalog field term "
        "ACTIVE at the mock's true comoving densities (log10n0 = -3 GAL, -5 AGN, "
        "delta = delta_c2 = 0), field sky weighting, survey order [GAL, AGN] so "
        "fcat_2 = f_AGN; population fixed at the mock fiducial, Om0 pinned; free "
        "parameters H0 and fcat_2 only.  Same events, same injections, same grid at "
        "every rung -- only the survey files change.",
        "config": {
            "log10n0": -3.0,
            "log10n0_c2": -5.0,
            "delta": 0.0,
            "delta_c2": 0.0,
            "density_provenance": "results/true_density.json",
            "gates": "results/gates.json",
            "rung0_continuity": "results/continuity_vs_analysis2.json",
        },
        "truth": {
            "H0": H0_TRUTH,
            "f_planted": F_PLANTED,
            "f_reference_note": "closure is quoted against the REALISED per-seed host "
            "fraction AND against the planted 0.30; the two differ by the mock's own "
            "binomial draw, sd = sqrt(0.3*0.7/1000) = 0.0145 per realisation",
        },
        "binomial_sd_per_realisation": math.sqrt(F_PLANTED * (1 - F_PLANTED) / 1000),
        "rungs": rungs,
        "ratios_comparable": ratios_comparable,
        "analysis_2_reference": a2_ref,
        "lane_agreement": lanes,
        "sky_shuffle_null": null,
    }
    (R / "ladder_summary.json").write_text(json.dumps(out, indent=2))
    print(f"Wrote {R / 'ladder_summary.json'}")

    # ---- console table --------------------------------------------------------
    if not ratios_comparable["like_for_like"]:
        print("\n*** PARTIAL CAMPAIGN: " + ratios_comparable["note"] + " ***")
        for lev, ss in seed_sets.items():
            print(f"      {lev:>9}: seeds {ss}")
    print("\n=== the ladder: per rung, five realisations ===")
    hdr = (
        f"{'rung':>9} {'C_gal':>7} {'C_agn':>7} {'n':>2} | "
        f"{'sig(H0)':>8} {'xR0':>5} {'off(H0)':>16} {'68/90':>6} | "
        f"{'sig(f)':>7} {'xR0':>5} {'off(f)':>17} {'68/90':>6} | {'rej':>6}"
    )
    print(hdr)
    for level in a.levels:
        r = rungs.get(level)
        if not r:
            continue
        cwh = r["completeness_within_horizon"]
        cg = cwh["gal"]["mean"]
        ca = cwh["agn"]["mean"]
        cl, w, cov, g = r["closure"], r["width"], r["coverage"], r["guard"]
        print(
            f"{level:>9} "
            f"{cg if cg is None else f'{cg:.3f}':>7} "
            f"{ca if ca is None else f'{ca:.3f}':>7} "
            f"{cov['n']:>2} | "
            f"{w['sigma_H0_mean_halfwidth68']:>8.3f} "
            f"{w.get('sigma_H0_vs_rung0', float('nan')):>5.2f} "
            f"{cl['H0'].get('mean', float('nan')):>+8.3f} +-"
            f"{cl['H0'].get('sem', float('nan')):>5.3f} "
            f"{cov['H0_in_68']}/{cov['H0_in_90']}    | "
            f"{w['sigma_f_mean_halfwidth68']:>7.4f} "
            f"{w.get('sigma_f_vs_rung0', float('nan')):>5.2f} "
            f"{cl['f_vs_realised'].get('mean', float('nan')):>+9.4f} +-"
            f"{cl['f_vs_realised'].get('sem', float('nan')):>6.4f} "
            f"{cov['f_realised_in_68']}/{cov['f_realised_in_90']}    | "
            f"{g['cells_rejected']:>6}"
        )

    print("\n=== N_eff along the ladder (across all five joint grids) ===")
    for level in a.levels:
        r = rungs.get(level)
        if not r:
            continue
        g = r["guard"]
        floor = g.get("legacy_floor_5N") or float("nan")
        print(
            f"  {level:>9}  Neff [{g['Neff_min']:,.0f}, {g['Neff_max']:,.0f}]"
            f"  = [{g['Neff_min']/floor:.0f}x, {g['Neff_max']/floor:.0f}x] floor"
            f"   max sum sig2_PE {g['pe_variance_sum_max']:.1f}"
            f"   rejected {g['cells_rejected']}/{g['cells_total']}"
        )

    if a2_ref and a2_ref.get("estimator_offset_rung0_minus_analysis2"):
        e = a2_ref["estimator_offset_rung0_minus_analysis2"]
        print("\n=== the estimator's own offset on COMPLETE catalogs ===")
        print("    rung 0 (true-n0 completion) minus analysis 2 (no missing-host "
              "budget), paired per realisation:")
        print(f"      H0  {e['H0'].get('mean', float('nan')):+.3f} +- "
              f"{e['H0'].get('sem', float('nan')):.3f}  "
              f"({e['H0_in_analysis2_halfwidths']:+.3f} a2 half-widths)   "
              f"sigma ratio {e['sigma_H0_ratio_rung0_over_analysis2']:.3f}")
        print(f"      f   {e['f'].get('mean', float('nan')):+.4f} +- "
              f"{e['f'].get('sem', float('nan')):.4f}  "
              f"({e['f_in_analysis2_halfwidths']:+.3f} a2 half-widths)   "
              f"sigma ratio {e['sigma_f_ratio_rung0_over_analysis2']:.3f}")
        print("    (completeness degradation is the x R0 columns above; this is the "
              "separate, completeness-independent estimator offset)")

    if lanes:
        print("\n=== lane agreement (seed %d), per rung ===" % s)
        for level, d in lanes.items():
            print(
                f"  {level:>9}  H0 {d['H0']['delta']:+.3f}"
                f" ({d['H0']['delta_over_halfwidth68']:+.3f} hw)"
                f"   f {d['f']['delta']:+.4f}"
                f" ({d['f']['delta_over_halfwidth68']:+.3f} hw)"
            )
    if null:
        print("\n=== sky-shuffle null (m18, seed %d) ===" % s)
        print(json.dumps(null, indent=2))

    # ---- the hook -------------------------------------------------------------
    hook = {
        "_provenance": "results/ladder_summary.json; rung 0 is analysis 2's own "
        "joint grids, every other rung is this directory's.",
        "rungs": {
            level: {
                "sigma_H0": r["width"]["sigma_H0_mean_halfwidth68"],
                "sigma_H0_vs_rung0": r["width"].get("sigma_H0_vs_rung0"),
                "sigma_f": r["width"]["sigma_f_mean_halfwidth68"],
                "sigma_f_vs_rung0": r["width"].get("sigma_f_vs_rung0"),
                "H0_offset_mean": r["closure"]["H0"].get("mean"),
                "H0_offset_sem": r["closure"]["H0"].get("sem"),
                "f_offset_mean": r["closure"]["f_vs_realised"].get("mean"),
                "f_offset_sem": r["closure"]["f_vs_realised"].get("sem"),
                "rho_mean": r["closure"]["rho"].get("mean"),
                "cells_rejected": r["guard"]["cells_rejected"],
                "n_seeds": r["coverage"]["n"],
            }
            for level, r in rungs.items()
        },
    }
    (R / "h0_fagn_ladder.json").write_text(json.dumps(hook, indent=2))
    print(f"\nWrote {R / 'h0_fagn_ladder.json'}")


if __name__ == "__main__":
    main()
