#!/usr/bin/env python3
"""Collect the closure evidence for the two matched-host controls into one file.

Three independent pieces, all written to results/closure_seeds.json:

  lanes      -- each control on both injection lanes, seed 100.  The lanes are
                the same detection rule under different proposals, so any
                disagreement large against the 68% half-width would mean the
                selection integral is setting digits of the answer.

  jackknife  -- each host-type event set cut into 8 disjoint contiguous blocks
                and scanned separately (seed 100).  sd(block medians)/sqrt(8)
                estimates the standard error of the full-set median directly
                from the data; comparing it with the quoted 68% half-width says
                whether the likelihood width is an honest error bar.  Blocks
                whose posterior touches a grid edge are flagged and excluded
                from the scatter, since a railed median is not a measurement.

  seeds      -- the same matched control on five independent realisations of the
                whole mock (100-104).  This is the only piece that can separate a
                realisation fluctuation from a systematic: the mean offset over
                seeds has standard error sd/sqrt(5), and the verdict is whether
                the mean offset is consistent with zero.  With five points the
                standard error carries 4 degrees of freedom, so the test
                statistic is Student-t, not normal.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

TRUTH = 67.74
HERE = Path(__file__).resolve().parents[1]


def load(tag, results=None):
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
        "map": h["map"],
        "truth_in_ci68": bool(h["truth_in_ci68"]),
        "truth_in_ci90": bool(h["truth_in_ci90"]),
        "n_rejected": int(d["n_rejected"]),
        "n_evals": int(d["n_evals"]),
        "grid": [float(grid[0]), float(grid[-1]), int(grid.size)],
        # Railed = the likelihood has no interior maximum in the scanned range
        # (the MAP sits ON an endpoint), or the 90% interval reaches an endpoint.
        # The first test is the decisive one: it says the peak is outside the
        # range, so the median is set by where the prior was cut.
        "railed": bool(h["map"] <= grid[0] + 1e-9 or h["map"] >= grid[-1] - 1e-9
                       or lo90 <= grid[0] + edge or hi90 >= grid[-1] - edge),
        "map_at_edge": bool(h["map"] <= grid[0] + 1e-9 or h["map"] >= grid[-1] - 1e-9),
        "min_Neff": min(c["Neff"] for c in cells) if cells else None,
    }


def t_sf(t, dof):
    """Two-sided Student-t tail.  With five seeds the standard error has 4 dof,
    where the normal approximation understates the tail by ~40%."""
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
    """Summarise the per-realisation offsets.

    A posterior that piles up against an edge of the scanned range has no median
    in any useful sense -- the number reported is where the prior was cut, not
    where the likelihood peaked, and the true value is further out still.  Such a
    realisation is kept in the headline statistic (dropping it would bias the mean
    towards truth, since it can only ever be censored AWAY from truth) but it is
    flagged, and the same statistic recomputed without it is carried alongside so
    the reader can see the edge is not what produces the verdict.
    """
    railed = [r for r in rows if r["railed"]]
    out = {"case": label, "per_seed": rows,
           "n_railed": len(railed),
           "railed_seeds": [r["seed"] for r in railed],
           "railed_note": ("a railed median is censored at the edge of the scanned "
                           "range, so the offsets it enters are magnitude LOWER "
                           "BOUNDS" if railed else None)}
    out.update(_stats(rows))
    unrailed = [r for r in rows if not r["railed"]]
    if railed and len(unrailed) > 1:
        out["excluding_railed"] = _stats(unrailed)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=[100, 101, 102, 103, 104])
    ap.add_argument("--n_blocks", type=int, default=8)
    ap.add_argument("--out", default=str(HERE / "results/closure_seeds.json"))
    args = ap.parse_args(argv)

    out = {"truth_H0": TRUTH,
           "note": "matched-host controls: each complete catalog analysed against "
                   "only the events it actually hosts",
           "injection_lane_of_record": "targeted",
           "seeds_excluded": {
               "104": {
                   "reason": "generate_dataset.py --stage validation failed "
                             "V6_injections_and_detection_closure",
                   "statistic": "pdet_lane_comparison.max_binomial_z = 7.53, gate < 6.0",
                   "diagnosis": "the extremum is a bin holding ONE detected targeted "
                                "injection (1 vs 35 popuni) at z ~ 0.277 where "
                                "P_det ~ 2e-4; the gate applies a Gaussian binomial "
                                "error to a single-count Poisson bin. Every seed's "
                                "extremum is such a bin (seeds 100/101/102/103 gave "
                                "4.00/3.43/5.69/2.69 from ndet = 1, 8, 1, 1). Seed "
                                "104's own end-to-end closure is 0.088 sigma, the "
                                "best of the five, and its pdraw is exact to 8e-15.",
                   "action": "not used; seed 105 generated as the replacement. The "
                             "gate was NOT relaxed, and the dataset remains on disk "
                             "unlinked from working/data/.",
               }},
           "seed_105_is_replacement_for_104": True}

    # --- lane cross-check (seed 100) ------------------------------------------
    lanes = {}
    for case, base in (("gal", "ctrl_gal_matched"), ("agn", "ctrl_agn_matched")):
        tgt, pop = load(base), load(base + "_popuni")
        if tgt and pop:
            lanes[case] = {
                "targeted": tgt, "popuni": pop,
                "difference": pop["median"] - tgt["median"],
                "difference_over_half68": (pop["median"] - tgt["median"]) / tgt["half68"],
            }
    out["lanes_seed100"] = lanes

    # --- disjoint-block scatter (seed 100) ------------------------------------
    jk = {}
    for case in ("gal", "agn"):
        rows = [load(f"jk_{case}_b{k}") for k in range(args.n_blocks)]
        rows = [r for r in rows if r]
        if not rows:
            continue
        good = [r for r in rows if not r["railed"]]
        med = np.array([r["median"] for r in good], float)
        full = load(f"ctrl_{case}_matched")
        sd = float(med.std(ddof=1)) if med.size > 1 else float("nan")
        # Two different calibration questions.  (a) Are the BLOCK widths honest?
        # compare the scatter of block medians with the blocks' own mean
        # half-width -- a ratio near 1 means the likelihood width describes how
        # much the answer moves.  (b) Does that scatter, scaled to the full set,
        # match the full set's quoted width?
        block_half = float(np.mean([r["half68"] for r in good])) if good else float("nan")
        jk[case] = {
            "n_blocks": len(rows),
            "n_blocks_unrailed": len(good),
            "events_per_block": rows[0]["n_events"],
            "blocks": rows,
            "block_median_mean": float(med.mean()) if med.size else None,
            "block_median_sd": sd,
            "mean_block_half68": block_half,
            "ratio_block_scatter_over_block_half68":
                sd / block_half if block_half and med.size > 1 else None,
            "implied_sem_full_set": sd / math.sqrt(len(good)) if med.size > 1 else None,
            "full_set_median": full["median"] if full else None,
            "full_set_half68": full["half68"] if full else None,
            "ratio_empirical_sem_over_quoted_half68":
                (sd / math.sqrt(len(good))) / full["half68"]
                if full and med.size > 1 else None,
        }
    out["jackknife_seed100"] = jk

    # --- across realisations --------------------------------------------------
    for case in ("gal", "agn"):
        rows = []
        for s in args.seeds:
            tag = f"ctrl_{case}_matched" if s == 100 else f"ctrl_{case}_matched_s{s}"
            r = load(tag)
            if r is None:
                continue
            r["seed"] = s
            rows.append(r)
        if rows:
            out[f"closure_{case}"] = summarise(rows, case)

    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}")

    for case in ("gal", "agn"):
        c = out.get(f"closure_{case}")
        if not c:
            continue
        print(f"\n=== {case.upper()} matched control, {c['n_seeds']} realisations")
        print(f"{'seed':>6} {'N_ev':>6} {'median':>9} {'-68%':>7} {'+68%':>7} {'offset':>8} {'rej':>5}  flag")
        for r in c["per_seed"]:
            print(f"{r['seed']:>6} {r['n_events']:>6} {r['median']:>9.2f} "
                  f"{-r['err_lo']:>7.2f} {r['err_hi']:>7.2f} {r['offset']:>8.2f} "
                  f"{r['n_rejected']:>5}  {'RAILED' if r['railed'] else ''}")
        print(f"  mean offset {c['mean_offset']:+.3f}  sd {c['sd_offset']:.3f}  "
              f"sem {c['sem_offset']:.3f}  t({c['dof']}) = {c['t_statistic']:+.2f}  "
              f"p = {c['p_two_sided']:.4f}")
        print(f"  seed scatter / mean quoted half-width = "
              f"{c['seed_scatter_over_quoted_half68']:.2f}")
        print(f"  consistent with truth at <2 sigma: {c['consistent_with_truth_2sigma']}"
              f"   beyond 2.5 sigma: {c['inconsistent_beyond_2p5sigma']}")
        if c.get("excluding_railed"):
            e = c["excluding_railed"]
            print(f"  [{c['n_railed']} railed: seeds {c['railed_seeds']}] excluding them: "
                  f"mean {e['mean_offset']:+.3f} sem {e['sem_offset']:.3f} "
                  f"t({e['dof']}) = {e['t_statistic']:+.2f}")

    for case, v in jk.items():
        print(f"\n=== {case.upper()} disjoint blocks (seed 100): "
              f"{v['n_blocks']} x {v['events_per_block']} events, "
              f"{v['n_blocks_unrailed']} unrailed")
        print("  medians: " + " ".join(f"{b['median']:.2f}" + ("*" if b["railed"] else "")
                                       for b in v["blocks"]) + "   (* = railed)")
        print(f"  scatter of block medians sd = {v['block_median_sd']:.2f} vs the blocks' "
              f"own mean half-width {v['mean_block_half68']:.2f}  (ratio "
              f"{v['ratio_block_scatter_over_block_half68']:.2f})")
        print(f"  sd/sqrt(K) -> implied sem on the full set "
              f"{v['implied_sem_full_set']:.2f}  vs its quoted half-width "
              f"{v['full_set_half68']:.2f}  (ratio "
              f"{v['ratio_empirical_sem_over_quoted_half68']:.2f})")

    for case, v in lanes.items():
        print(f"\n=== {case.upper()} lane cross-check (seed 100): "
              f"targeted {v['targeted']['median']:.3f}  popuni {v['popuni']['median']:.3f}  "
              f"diff {v['difference']:+.3f} = "
              f"{100*v['difference_over_half68']:.1f}% of one half-width")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
