#!/usr/bin/env python3
"""analysis_0 -- the pure-tracer H0 aggregation.

Adapted from analysis_1_complete_catalog_H0/scripts/aggregate_closure.py (the
`load`, `t_sf`, `_stats` and `summarise` machinery are that script's, with the
jackknife and the seed-104 bookkeeping dropped and two blocks added).  What is new
here is the head-to-head: analysis_1's two tracers were unequal SUBSETS of one
mixture draw (705 GAL vs 295 AGN), so their widths could not be compared.  These
are independent full draws of N = 1000 each, so the ratio of widths IS the
constraining-power statement.

Three blocks, all written to results/h0_pure_tracer.json:

  closure_{gal,agn}   per realisation: median, 68/90% interval, offset from truth;
                      then mean offset +- s.e.m. over the five realisations, the
                      Student-t statistic on 4 dof, and the 68/90% coverage counts.
                      This is the bias check.

  constraining_power  GAL vs AGN at equal N = 1000: per-seed 68% half-widths, their
                      means, the ratio, and a paired comparison across realisations
                      (the seeds share catalogs, so the pairing is real).

  lanes               targeted vs popuni per tracer per realisation.  The lanes are
                      the same detection rule under different proposals, so a
                      difference large against the 68% half-width would mean the
                      selection integral is setting digits of the answer.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

TRUTH = 67.74
HERE = Path(__file__).resolve().parents[1]
SEEDS_DEFAULT = [100, 101, 102, 103, 105]
LANE_OF_RECORD = "targeted"


def load(tag, results=None):
    """analysis_1/scripts/aggregate_closure.py::load, unchanged."""
    p = (results or (HERE / "results")) / f"{tag}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    h = d["H0"]
    grid = np.asarray(h["grid"], float)
    lo, hi = h["ci68"]
    lo90, hi90 = h["ci90"]
    edge = 0.5 * (grid[1] - grid[0])
    cells = d.get("guard", {}).get("cells") or []
    # With max_likelihood_variance = 1e6 the guard threshold collapses to exactly
    # 5 * N_obs, so it is also the run's event count.
    return {
        "tag": tag,
        "n_events": int(round(cells[0]["threshold"] / 5)) if cells else None,
        "median": h["median"],
        "offset": h["median"] - TRUTH,
        "ci68": [lo, hi],
        "ci90": [lo90, hi90],
        "err_hi": hi - h["median"],
        "err_lo": h["median"] - lo,
        "half68": 0.5 * (hi - lo),
        "half90": 0.5 * (hi90 - lo90),
        "map": h["map"],
        "truth_in_ci68": bool(h["truth_in_ci68"]),
        "truth_in_ci90": bool(h["truth_in_ci90"]),
        "n_rejected": int(d["n_rejected"]),
        "n_evals": int(d["n_evals"]),
        "grid": [float(grid[0]), float(grid[-1]), int(grid.size)],
        # Railed = the likelihood has no interior maximum in the scanned range
        # (the MAP sits ON an endpoint), or the 90% interval reaches an endpoint.
        "railed": bool(h["map"] <= grid[0] + 1e-9 or h["map"] >= grid[-1] - 1e-9
                       or lo90 <= grid[0] + edge or hi90 >= grid[-1] - edge),
        "map_at_edge": bool(h["map"] <= grid[0] + 1e-9 or h["map"] >= grid[-1] - 1e-9),
        "min_Neff": min(c["Neff"] for c in cells) if cells else None,
        "n_cells_guard_rejected": int(d.get("guard", {}).get("n_rejected", 0)),
    }


def t_sf(t, dof):
    """Two-sided Student-t tail.  With five realisations the standard error has 4
    dof, where the normal approximation understates the tail by ~40%."""
    from scipy import stats
    return float(2.0 * stats.t.sf(abs(t), dof))


def _stats(rows):
    off = np.array([r["offset"] for r in rows], float)
    n = off.size
    mean = float(off.mean())
    sd = float(off.std(ddof=1)) if n > 1 else float("nan")
    sem = sd / math.sqrt(n) if n > 1 else float("nan")
    t = mean / sem if n > 1 and np.isfinite(sem) and sem > 0 else float("nan")
    half = float(np.mean([r["half68"] for r in rows])) if n else float("nan")
    return {
        "n_seeds": n,
        "seeds": [r["seed"] for r in rows],
        "mean_offset": mean,
        "sd_offset": sd,
        "sem_offset": sem,
        "t_statistic": t,
        "dof": n - 1,
        "p_two_sided": t_sf(t, n - 1) if n > 1 and np.isfinite(t) else None,
        "mean_quoted_half68": half,
        "seed_scatter_over_quoted_half68": sd / half if half else None,
        "consistent_with_truth_2sigma": bool(abs(t) < 2.0) if np.isfinite(t) else None,
        "inconsistent_beyond_2p5sigma": bool(abs(t) > 2.5) if np.isfinite(t) else None,
    }


def summarise(rows, label):
    """analysis_1's summarise(), plus explicit 68/90% coverage counts.

    A posterior piled up against an edge of the scanned range has no median in any
    useful sense -- the number reported is where the prior was cut, not where the
    likelihood peaked, and the true value is further out still.  Such a realisation
    is KEPT in the headline statistic (dropping it would bias the mean towards
    truth, since it can only ever be censored AWAY from truth) but it is flagged,
    and the same statistic without it is carried alongside.
    """
    railed = [r for r in rows if r["railed"]]
    out = {"case": label, "per_seed": rows,
           "n_railed": len(railed),
           "railed_seeds": [r["seed"] for r in railed],
           "railed_note": ("a railed median is censored at the edge of the scanned "
                           "range, so the offsets it enters are magnitude LOWER "
                           "BOUNDS" if railed else None)}
    out.update(_stats(rows))
    n = len(rows)
    c68 = sum(r["truth_in_ci68"] for r in rows)
    c90 = sum(r["truth_in_ci90"] for r in rows)
    out["coverage"] = {
        "n_realisations": n,
        "n_truth_in_ci68": c68, "frac_truth_in_ci68": c68 / n if n else None,
        "expected_ci68": 0.68,
        "n_truth_in_ci90": c90, "frac_truth_in_ci90": c90 / n if n else None,
        "expected_ci90": 0.90,
        "note": "five realisations resolve coverage only to +-1 count; this is a "
                "consistency check on the interval, not a calibration measurement",
    }
    out["widths"] = {
        "half68_per_seed": {str(r["seed"]): r["half68"] for r in rows},
        "mean_half68": float(np.mean([r["half68"] for r in rows])) if n else None,
        "sd_half68": float(np.std([r["half68"] for r in rows], ddof=1)) if n > 1 else None,
        "mean_half90": float(np.mean([r["half90"] for r in rows])) if n else None,
        "n_events_per_seed": {str(r["seed"]): r["n_events"] for r in rows},
    }
    unrailed = [r for r in rows if not r["railed"]]
    if railed and len(unrailed) > 1:
        out["excluding_railed"] = _stats(unrailed)
    return out


def head_to_head(gal, agn):
    """GAL vs AGN at equal N = 1000 -- the constraining-power headline.

    The two draws share the catalog realisation seed by seed, so the per-seed ratio
    is paired; the mean of the per-seed ratios and the ratio of the means are both
    reported because they answer slightly different questions (typical realisation
    vs the ensemble).  A railed posterior has no width, so it is excluded from the
    ratio and flagged.
    """
    if not gal or not agn:
        return None
    g = {r["seed"]: r for r in gal["per_seed"]}
    a = {r["seed"]: r for r in agn["per_seed"]}
    seeds = sorted(set(g) & set(a))
    rows, usable = [], []
    for s in seeds:
        row = {"seed": s,
               "n_events_gal": g[s]["n_events"], "n_events_agn": a[s]["n_events"],
               "half68_gal": g[s]["half68"], "half68_agn": a[s]["half68"],
               "ratio_agn_over_gal": a[s]["half68"] / g[s]["half68"],
               "railed_gal": g[s]["railed"], "railed_agn": a[s]["railed"]}
        rows.append(row)
        if not (g[s]["railed"] or a[s]["railed"]):
            usable.append(row)
    mg = float(np.mean([r["half68_gal"] for r in usable])) if usable else None
    ma = float(np.mean([r["half68_agn"] for r in usable])) if usable else None
    rat = np.array([r["ratio_agn_over_gal"] for r in usable], float)
    return {
        "what": "68% half-width of the H0 posterior, pure-GAL vs pure-AGN, at the "
                "same N = 1000 detected events on the same catalog realisation; "
                f"lane of record = {LANE_OF_RECORD}",
        "per_seed": rows,
        "n_seeds_usable": len(usable),
        "railed_excluded_seeds": [r["seed"] for r in rows
                                  if r["railed_gal"] or r["railed_agn"]],
        "mean_half68_gal": mg,
        "mean_half68_agn": ma,
        "ratio_of_means_agn_over_gal": (ma / mg) if (mg and ma) else None,
        "mean_of_per_seed_ratios": float(rat.mean()) if rat.size else None,
        "sd_of_per_seed_ratios": float(rat.std(ddof=1)) if rat.size > 1 else None,
        "sem_of_per_seed_ratios": (float(rat.std(ddof=1) / math.sqrt(rat.size))
                                   if rat.size > 1 else None),
        "n_events_equal": bool(all(r["n_events_gal"] == r["n_events_agn"]
                                   for r in rows)),
    }


def lane_block(seeds, results=None):
    out = {}
    for tracer, base in (("gal", "h0_puregal"), ("agn", "h0_pureagn")):
        rows = []
        for s in seeds:
            t = load(f"{base}_targeted_s{s}", results)
            p = load(f"{base}_popuni_s{s}", results)
            if not (t and p):
                continue
            rows.append({"seed": s,
                         "targeted_median": t["median"], "popuni_median": p["median"],
                         "difference": p["median"] - t["median"],
                         "targeted_half68": t["half68"], "popuni_half68": p["half68"],
                         "difference_over_targeted_half68":
                             (p["median"] - t["median"]) / t["half68"],
                         "targeted_railed": t["railed"], "popuni_railed": p["railed"]})
        if rows:
            d = np.array([r["difference_over_targeted_half68"] for r in rows], float)
            out[tracer] = {
                "per_seed": rows,
                "max_abs_difference_over_half68": float(np.abs(d).max()),
                "mean_difference_over_half68": float(d.mean()),
            }
    return out


def diagnostics(seeds, results=None):
    """Per-scan guard state and posterior shape -- the internal health record.

    Two things a median and an interval do not show.  (a) The selection guard: how
    far the smallest per-cell N_eff sat above the wall, and how many cells were
    rejected.  (b) The shape: a bimodal posterior has an honest median but its
    68% interval spans the gap between the modes, so a wide realisation should be
    identifiable as bimodal rather than merely noisy.  Modes are counted on the
    flat-prior posterior density; interior local maxima only, so a monotone
    posterior counts zero and is caught by the railed flag instead.
    """
    import h5py
    res = results or (HERE / "results")
    rows = []
    for tracer, base in (("gal", "h0_puregal"), ("agn", "h0_pureagn")):
        for lane in ("targeted", "popuni"):
            for s in seeds:
                tag = f"{base}_{lane}_s{s}"
                jp, hp = res / f"{tag}.json", res / f"{tag}.h5"
                if not (jp.exists() and hp.exists()):
                    continue
                d = json.loads(jp.read_text())
                g = d.get("guard", {})
                with h5py.File(hp, "r") as f:
                    grid = f["H0_grid"][:].ravel()
                    logL = f["log_likelihood"][:].ravel()
                p = np.exp(logL - logL.max())
                p /= np.trapz(p, grid)
                pk = p / p.max()
                modes = [i for i in range(1, len(p) - 1)
                         if p[i] > p[i - 1] and p[i] > p[i + 1]]
                rows.append({
                    "tag": tag, "tracer": tracer, "lane": lane, "seed": s,
                    "n_cells": int(d["n_evals"]),
                    "n_cells_rejected": int(d["n_rejected"]),
                    "n_cells_guard_rejected": int(g.get("n_rejected", 0)),
                    "n_neginf_cells": int(d.get("n_neginf_cells", 0)),
                    "Neff_min": g.get("summary", {}).get("Neff_min"),
                    "guard_threshold": g.get("summary", {}).get("threshold_max"),
                    "Neff_min_over_threshold":
                        (g["summary"]["Neff_min"] / g["summary"]["threshold_max"])
                        if g.get("summary", {}).get("threshold_max") else None,
                    "n_interior_modes": len(modes),
                    "mode_positions": [float(grid[i]) for i in modes],
                    "mode_relative_heights": [float(pk[i]) for i in modes],
                    "density_at_grid_lo": float(pk[0]),
                    "density_at_grid_hi": float(pk[-1]),
                    "total_eval_seconds": d["timing"]["total_eval_seconds"],
                    "steady_state_seconds_per_cell":
                        d["timing"]["steady_state_median_seconds"],
                })
    if not rows:
        return None
    return {
        "what": "per-scan selection-guard state and posterior shape; internal "
                "health record, not a result",
        "guard_convention": "hard N_eff wall at 5 * N_obs with the total-variance "
                            "criterion made inert (max_likelihood_variance = 1e6)",
        "per_scan": rows,
        "n_scans": len(rows),
        "all_cells_accepted": all(r["n_cells_rejected"] == 0
                                  and r["n_cells_guard_rejected"] == 0 for r in rows),
        "min_Neff_over_threshold_across_all_scans":
            min(r["Neff_min_over_threshold"] for r in rows
                if r["Neff_min_over_threshold"] is not None),
        "multimodal_scans": [r["tag"] for r in rows if r["n_interior_modes"] > 1],
        "max_density_at_a_grid_edge": max(max(r["density_at_grid_lo"],
                                              r["density_at_grid_hi"]) for r in rows),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS_DEFAULT)
    ap.add_argument("--results", default=None)
    ap.add_argument("--out", default=str(HERE / "results/h0_pure_tracer.json"))
    args = ap.parse_args(argv)
    res = Path(args.results) if args.results else None

    out = {
        "truth_H0": TRUTH,
        "what": "H0 constraining power of each tracer separately, and the bias "
                "check, on NEW independent 1000-event draws: for each of the five "
                "v3 catalog realisations one event set with every host a galaxy "
                "(f_agn = 0) and one with every host an AGN (f_agn = 1), event "
                "noise independent of each other and of the record's mixture.",
        "differs_from_analysis_1_how":
            "analysis_1's ctrl_{gal,agn}_matched split the record's ONE 1000-event "
            "mixture draw on host type, giving 705 GAL and 295 AGN events that are "
            "neither independent nor equal in size.  Here each tracer gets its own "
            "full N = 1000 draw, so the widths are directly comparable and the "
            "event noise is independent of the record.",
        "configuration": {
            "estimator": "dark_sirens at log10n0 = -24 (complete-catalog limit)",
            "catalog_sky_weighting": "field",
            "K": 1,
            "h0_grid": [50.0, 100.0, 201],
            "Om0": 0.3075,
            "guard": "hard N_eff wall, max_likelihood_variance = 1e6 (variance "
                     "criterion inert)",
            "kde_window": "4096 on the GAL survey; module default on the AGN survey",
            "population_and_nuisances": "fixed at truth",
            "source": "copied verbatim from analysis_1_complete_catalog_H0",
        },
        "injection_lane_of_record": LANE_OF_RECORD,
        "seeds_requested": list(args.seeds),
        "seed_105_is_replacement_for_104": True,
    }

    for tracer, base in (("gal", "h0_puregal"), ("agn", "h0_pureagn")):
        rows = []
        for s in args.seeds:
            r = load(f"{base}_{LANE_OF_RECORD}_s{s}", res)
            if r is None:
                continue
            r["seed"] = s
            rows.append(r)
        if rows:
            out[f"closure_{tracer}"] = summarise(rows, tracer)

    out["constraining_power"] = head_to_head(out.get("closure_gal"),
                                             out.get("closure_agn"))
    out["lanes"] = lane_block(args.seeds, res)
    out["diagnostics"] = diagnostics(args.seeds, res)
    out["n_scans_found"] = sum(len(out.get(f"closure_{t}", {}).get("per_seed", []))
                               for t in ("gal", "agn")) + sum(
        len(v["per_seed"]) for v in out["lanes"].values())

    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}")

    for tracer in ("gal", "agn"):
        c = out.get(f"closure_{tracer}")
        if not c:
            print(f"\n=== {tracer.upper()}: no results yet")
            continue
        print(f"\n=== PURE-{tracer.upper()} 1000-event draws, {c['n_seeds']} "
              f"realisations ({LANE_OF_RECORD} lane)")
        print(f"{'seed':>6} {'N_ev':>6} {'median':>9} {'-68%':>7} {'+68%':>7} "
              f"{'offset':>8} {'half68':>7} {'in68':>5} {'in90':>5} {'rej':>4}  flag")
        for r in c["per_seed"]:
            print(f"{r['seed']:>6} {r['n_events']:>6} {r['median']:>9.2f} "
                  f"{-r['err_lo']:>7.2f} {r['err_hi']:>7.2f} {r['offset']:>8.2f} "
                  f"{r['half68']:>7.2f} {str(r['truth_in_ci68']):>5} "
                  f"{str(r['truth_in_ci90']):>5} {r['n_rejected']:>4}  "
                  f"{'RAILED' if r['railed'] else ''}")
        print(f"  mean offset {c['mean_offset']:+.3f}  sd {c['sd_offset']:.3f}  "
              f"sem {c['sem_offset']:.3f}  t({c['dof']}) = {c['t_statistic']:+.2f}  "
              f"p = {c['p_two_sided']:.4f}")
        cv = c["coverage"]
        print(f"  coverage: 68% {cv['n_truth_in_ci68']}/{cv['n_realisations']}  "
              f"90% {cv['n_truth_in_ci90']}/{cv['n_realisations']}")
        print(f"  mean quoted 68% half-width {c['widths']['mean_half68']:.2f}  "
              f"(seed scatter / half-width = "
              f"{c['seed_scatter_over_quoted_half68']:.2f})")
        if c.get("excluding_railed"):
            e = c["excluding_railed"]
            print(f"  [{c['n_railed']} railed: seeds {c['railed_seeds']}] excluding "
                  f"them: mean {e['mean_offset']:+.3f} sem {e['sem_offset']:.3f} "
                  f"t({e['dof']}) = {e['t_statistic']:+.2f}")

    cp = out.get("constraining_power")
    if cp and cp["n_seeds_usable"]:
        print(f"\n=== CONSTRAINING POWER at equal N = 1000 ({LANE_OF_RECORD} lane)")
        print(f"{'seed':>6} {'half68 GAL':>11} {'half68 AGN':>11} {'AGN/GAL':>8}  flag")
        for r in cp["per_seed"]:
            fl = " ".join(x for x, b in (("GAL-railed", r["railed_gal"]),
                                         ("AGN-railed", r["railed_agn"])) if b)
            print(f"{r['seed']:>6} {r['half68_gal']:>11.2f} {r['half68_agn']:>11.2f} "
                  f"{r['ratio_agn_over_gal']:>8.2f}  {fl}")
        print(f"  mean half68: GAL {cp['mean_half68_gal']:.2f}   "
              f"AGN {cp['mean_half68_agn']:.2f}   "
              f"ratio of means AGN/GAL {cp['ratio_of_means_agn_over_gal']:.2f}")
        if cp["sem_of_per_seed_ratios"] is not None:
            print(f"  mean of per-seed ratios {cp['mean_of_per_seed_ratios']:.2f} "
                  f"+- {cp['sem_of_per_seed_ratios']:.2f}  "
                  f"(n = {cp['n_seeds_usable']})")

    for tracer, v in out["lanes"].items():
        print(f"\n=== {tracer.upper()} lane cross-check (targeted vs popuni)")
        for r in v["per_seed"]:
            print(f"  seed {r['seed']}: targeted {r['targeted_median']:.3f}  "
                  f"popuni {r['popuni_median']:.3f}  diff {r['difference']:+.3f} = "
                  f"{100 * r['difference_over_targeted_half68']:.1f}% of one "
                  f"half-width")
        print(f"  max |diff| / half-width = {v['max_abs_difference_over_half68']:.3f}")

    dg = out.get("diagnostics")
    if dg:
        print(f"\n=== GUARD AND SHAPE ({dg['n_scans']} scans)")
        print(f"  every cell accepted in every scan: {dg['all_cells_accepted']}")
        print(f"  smallest N_eff / threshold over all scans: "
              f"{dg['min_Neff_over_threshold_across_all_scans']:.1f}x")
        print(f"  largest posterior density at a grid edge (1 = the peak): "
              f"{dg['max_density_at_a_grid_edge']:.2e}")
        print(f"  multimodal scans: {dg['multimodal_scans'] or 'none'}")
        for r in dg["per_scan"]:
            if r["n_interior_modes"] > 1:
                print(f"    {r['tag']}: modes at "
                      + ", ".join(f"{m:.2f} (rel. height {h:.2f})" for m, h
                                  in zip(r["mode_positions"],
                                         r["mode_relative_heights"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
