#!/usr/bin/env python3
"""fig_closure_after_fix -- the five-realisation matched-host closure, before/after.

One panel per catalog.  Each realisation contributes a pair of points: the
matched-host control BEFORE the 2026-08-01 generator fixes (conventions (b2) and
(c2)) and AFTER, with their own 68 % intervals, joined by a line so the direction of
the move is readable at a glance.  The realisation means are drawn as a band.

"Before" is the record of 2026-07-31: seed 100 from ``attic/results_prefix2/`` (the
`dark_sirens` scan of record) and the further realisations from
``attic/results_dsc_attic/`` (the same configuration under `dark_sirens_complete`, which
experiment_model_equivalence measured to be bitwise identical -- on seed 100 the two
estimators give 62.789 and 62.785).  "After" is ``results/``.

Reads: results/closure_seeds.json (after) + the per-seed scan JSONs (both).
Writes: figs/fig_closure_after_fix.{png,pdf}, results/closure_after_fix.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
PRE = ROOT / "attic" / "results_prefix2"      # 2026-08-01 reorg: was ROOT/results_prefix2
ATTIC = ROOT / "attic" / "results_dsc_attic"  # 2026-08-01 reorg: was ROOT/results_dsc_attic
FIGS = ROOT / "figs"
TRUTH = 67.74

SURFACE = "#FFFFFF"
INK = "#1A1A1A"
INK_2 = "#4A4A4A"
INK_MUTED = "#9A9A9A"
BLUE = "#2C6E9B"
ORANGE = "#C4622D"
GREY = "#8A8A8A"

mpl.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.size": 9, "axes.labelsize": 9.5,
    "axes.titlesize": 10.5, "axes.edgecolor": INK_MUTED, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": INK_2, "ytick.color": INK_2,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "axes.linewidth": 0.8,
    "legend.frameon": False, "pdf.fonttype": 42,
})


def read(path):
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    h = d["H0"]
    cells = d.get("guard", {}).get("cells") or []
    grid = np.asarray(h["grid"], float)
    edge = 0.5 * (grid[1] - grid[0])   # grid IS the full 201-point array
    return {"median": h["median"], "offset": h["median"] - TRUTH,
            "ci68": h["ci68"], "ci90": h["ci90"], "map": h["map"],
            "half68": 0.5 * (h["ci68"][1] - h["ci68"][0]),
            "truth_in_ci68": h["truth_in_ci68"],
            "truth_in_ci90": h["truth_in_ci90"],
            "railed": bool(h["map"] <= grid[0] + 1e-9 or h["map"] >= grid[-1] - 1e-9
                               or h["ci90"][0] <= grid[0] + edge
                               or h["ci90"][1] >= grid[-1] - edge),
            "n_events": int(round(cells[0]["threshold"] / 5)) if cells else None,
            "n_rejected": d.get("n_rejected")}


def tag_for(case, seed):
    return f"ctrl_{case}_matched" if seed == 100 else f"ctrl_{case}_matched_s{seed}"


BEFORE_DIRS = [PRE, ATTIC]


def before(case, seed):
    t = tag_for(case, seed)
    for d in BEFORE_DIRS:
        r = read(Path(d) / f"{t}.json")
        if r is not None:
            return r
    return None


def after(case, seed):
    return read(RES / f"{tag_for(case, seed)}.json")


def summarise(rows):
    off = np.array([r["offset"] for r in rows], float)
    n = off.size
    sd = float(off.std(ddof=1)) if n > 1 else float("nan")
    sem = sd / math.sqrt(n) if n > 1 else float("nan")
    t = float(off.mean() / sem) if n > 1 and sem else float("nan")
    from scipy import stats
    p = float(2 * stats.t.sf(abs(t), n - 1)) if n > 1 else float("nan")
    return {"n_seeds": n, "mean_offset": float(off.mean()), "sd_offset": sd,
            "sem_offset": sem, "t_statistic": t, "p_two_sided": p,
            "mean_half68": float(np.mean([r["half68"] for r in rows])),
            "n_truth_in_ci68": int(sum(r["truth_in_ci68"] for r in rows)),
            "n_truth_in_ci90": int(sum(r["truth_in_ci90"] for r in rows)),
            "n_railed": int(sum(r["railed"] for r in rows))}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=[100, 101, 102, 103, 105])
    ap.add_argument("--out_json", default=str(RES / "closure_after_fix.json"))
    ap.add_argument("--before_dir", nargs="+", default=None,
                    help="Directories searched, in order, for the BEFORE scans.  "
                         "Default: attic/results_prefix2 then attic/results_dsc_attic "
                         "(the pre-(b2)/(c2) record).  For the v3 comparison pass "
                         "attic/results_v2postfix.")
    ap.add_argument("--before_label", default="before the (b2)+(c2) fixes")
    ap.add_argument("--after_label", default="after")
    ap.add_argument("--what", default=None)
    ap.add_argument("--fig_tag", default="fig_closure_after_fix")
    args = ap.parse_args(argv)
    FIGS.mkdir(exist_ok=True)
    global BEFORE_DIRS
    if args.before_dir:
        BEFORE_DIRS = [Path(d) if Path(d).is_absolute() else ROOT / d
                       for d in args.before_dir]

    doc = {"truth_H0": TRUTH, "seeds": args.seeds,
           "what": "matched-host controls before and after the 2026-08-01 generator "
                   "fixes: (b2) the RA measurement width from the OBSERVED dec, and "
                   "(c2) the mass PE drawn from the exact flat-prior posterior of "
                   "obs ~ N(m, f m).  Identical configuration otherwise: dark_sirens "
                   "at log10n0 = -24, field weighting, K = 1, targeted injections, "
                   "H0 in [50, 100] x 201, W = 4096 (GAL), campaign guard convention. "
                   "The detected sets are bit-identical across the fix, so this is a "
                   "paired comparison on the same events.",
           "cases": {}}
    if args.what:
        doc["what"] = args.what
    doc["before_dirs"] = [str(d) for d in BEFORE_DIRS]

    for case in ("gal", "agn"):
        rows = []
        for s in args.seeds:
            b, a = before(case, s), after(case, s)
            if a is None:
                continue
            rows.append({"seed": s, "before": b, "after": a,
                         "delta": (a["offset"] - b["offset"]) if b else None})
        doc["cases"][case] = {
            "per_seed": rows,
            "before": summarise([r["before"] for r in rows if r["before"]]),
            "after": summarise([r["after"] for r in rows]),
        }
        d = np.array([r["delta"] for r in rows if r["delta"] is not None], float)
        doc["cases"][case]["mean_shift"] = float(d.mean()) if d.size else None
        doc["cases"][case]["sem_shift"] = (float(d.std(ddof=1) / math.sqrt(d.size))
                                           if d.size > 1 else None)

    Path(args.out_json).write_text(json.dumps(doc, indent=2))
    print(f"wrote {args.out_json}")

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.3), sharey=False)
    for ax, case, colour, title in ((axes[0], "gal", BLUE, "GAL catalog"),
                                    (axes[1], "agn", ORANGE, "AGN catalog")):
        C = doc["cases"][case]
        rows = C["per_seed"]
        x = np.arange(len(rows), dtype=float)
        ax.axhline(TRUTH, color=INK, lw=1.1, zorder=1)
        ax.text(len(rows) - 0.42, TRUTH, "  truth", va="center", ha="left",
                fontsize=8.5, color=INK)
        for k, r in enumerate(rows):
            b, a = r["before"], r["after"]
            if b:
                ax.plot([x[k] - 0.16, x[k] + 0.16], [b["median"], a["median"]],
                        color=INK_MUTED, lw=0.8, zorder=2)
                ax.errorbar(x[k] - 0.16, b["median"],
                            yerr=[[b["median"] - b["ci68"][0]],
                                  [b["ci68"][1] - b["median"]]],
                            fmt="o", ms=4.2, mfc=SURFACE, mec=GREY, ecolor=GREY,
                            lw=1.1, capsize=0, zorder=3)
                if b["railed"]:
                    ax.annotate("railed", (x[k] - 0.16, b["ci68"][0]),
                                textcoords="offset points", xytext=(0, -12),
                                ha="center", fontsize=7, color=GREY)
            ax.errorbar(x[k] + 0.16, a["median"],
                        yerr=[[a["median"] - a["ci68"][0]],
                              [a["ci68"][1] - a["median"]]],
                        fmt="o", ms=4.6, color=colour, ecolor=colour, lw=1.3,
                        capsize=0, zorder=4)
            if a["railed"]:
                ax.annotate("railed", (x[k] + 0.16, a["ci68"][0]),
                            textcoords="offset points", xytext=(0, -12),
                            ha="center", fontsize=7, color=colour)
        for S, col, lab in ((C["before"], GREY, "before"), (C["after"], colour, "after")):
            m = TRUTH + S["mean_offset"]
            ax.axhline(m, color=col, lw=0.9, ls="--", alpha=0.75, zorder=1)
            ax.axhspan(m - S["sem_offset"], m + S["sem_offset"], color=col,
                       alpha=0.10, lw=0, zorder=0)
        lo = min([min(r["before"]["ci68"][0] if r["before"] else 1e9,
                      r["after"]["ci68"][0]) for r in rows] + [TRUTH])
        hi = max([max(r["before"]["ci68"][1] if r["before"] else -1e9,
                      r["after"]["ci68"][1]) for r in rows] + [TRUTH])
        span = hi - lo
        ax.set_ylim(lo - 0.16 * span, hi + 0.07 * span)
        ax.set_xticks(x)
        ax.set_xticklabels([f"mock {k+1}\n(seed {r['seed']})"
                            for k, r in enumerate(rows)], fontsize=8)
        ax.set_xlim(-0.55, len(rows) - 0.45)
        ax.set_ylabel(r"$H_0$  [km s$^{-1}$ Mpc$^{-1}$]")
        B, A = C["before"], C["after"]
        ax.set_title(f"{title}\n"
                     f"before {B['mean_offset']:+.2f} $\\pm$ {B['sem_offset']:.2f}"
                     f"   $\\to$   after {A['mean_offset']:+.2f} $\\pm$ "
                     f"{A['sem_offset']:.2f}", fontsize=9.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    h = [plt.Line2D([], [], marker="o", ls="", ms=4.2, mfc=SURFACE, mec=GREY,
                    label=args.before_label),
         plt.Line2D([], [], marker="o", ls="", ms=4.6, color=INK_2,
                    label=args.after_label)]
    axes[0].legend(handles=h, loc="lower left", fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIGS / f"{args.fig_tag}.{ext}", dpi=220)
    plt.close(fig)
    print(f"wrote figs/{args.fig_tag}.{{png,pdf}}")
    for case in ("gal", "agn"):
        C = doc["cases"][case]
        print(f"  {case.upper()}: before {C['before']['mean_offset']:+.3f} "
              f"+- {C['before']['sem_offset']:.3f}  ->  after "
              f"{C['after']['mean_offset']:+.3f} +- {C['after']['sem_offset']:.3f} "
              f"(mean shift {C['mean_shift']:+.3f} +- "
              f"{C['sem_shift'] if C['sem_shift'] is None else round(C['sem_shift'],3)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
