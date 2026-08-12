#!/usr/bin/env python
"""Result figures for analysis 4 under c_mode=selection. Deterministic.

    python make_figures.py        # writes ../figs/*.pdf and *.png

fig1_anchor_response   f_AGN against the AGN-density mis-anchoring factor, one
                       panel per rung, archived per_pixel vs selection.

The archived finding was that mis-anchoring the AGN completion density lands
almost entirely on f_AGN, steepening as the catalog thins. This asks two
separate questions of that result: does the SENSITIVITY (the slope) survive the
estimator change, and does the OFFSET (where the correctly-anchored arm sits
relative to truth)?

Renders whatever rungs are present, so it is useful mid-campaign; absent rungs
are simply omitted and the panel count follows the data.

Intervals are 90% (project rule). Fixed Okabe-Ito order; truth is a neutral
black dashed rule, never a series hue.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
A4 = HERE.parent
ARCHIVE = (A4.parent.parent / "archive" /
           "analysis_4_density_anchoring_H0_fagn" / "results")
FIGS = A4 / "figs"

TRUTH_F = 0.295
RUNGS = ["m21", "m20", "m19", "m18"]
ARMS = {"a05": 0.5, "a07": 0.7, "a09": 0.9, "a11": 1.1, "a13": 1.3, "a20": 2.0}
SERIES = [("per_pixel (archived)", "#0072B2", ARCHIVE),
          ("selection (this work)", "#009E73", A4 / "results")]
INK, MUTED = "#1a1a1a", "#6b6b6b"


def load(resdir, rung):
    out = []
    for tag, fac in ARMS.items():
        p = Path(resdir) / f"joint_{rung}_{tag}_s100.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        out.append({"factor": fac, "x": np.log10(fac),
                    "y": d["f"]["median"], "ci90": d["f"]["ci90"]})
    return sorted(out, key=lambda r: r["factor"])


def main():
    present = [r for r in RUNGS
               if load(A4 / "results", r) or load(ARCHIVE, r)]
    have_new = [r for r in RUNGS if load(A4 / "results", r)]
    if not present:
        print("  no arms found yet"); return

    n = len(present)
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n + 0.6, 4.0), squeeze=False)
    axes = axes[0]
    summary = {}

    for ax, rung in zip(axes, present):
        ax.axhline(TRUTH_F, color="black", ls="--", lw=1.2, zorder=1)
        summary[rung] = {}
        for lbl, colour, d in SERIES:
            rows = load(d, rung)
            if not rows:
                continue
            x = [r["x"] for r in rows]; y = [r["y"] for r in rows]
            lo = [r["y"] - r["ci90"][0] for r in rows]
            hi = [r["ci90"][1] - r["y"] for r in rows]
            ax.errorbar(x, y, yerr=[lo, hi], fmt="o-", color=colour, ms=6,
                        lw=1.8, capsize=0, label=lbl, zorder=3)
            ax.plot(x, y, "o", color="white", ms=2.2, zorder=4)
            if len(rows) >= 2:
                s, i = np.polyfit(x, y, 1)
                # where the correctly-anchored arm sits (factor 1.0 by the fit)
                summary[rung][lbl] = {"slope": float(s),
                                      "at_factor_1": float(i),
                                      "offset_at_1": float(i - TRUTH_F),
                                      "n_arms": len(rows)}
        ax.set_xticks([np.log10(f) for f in ARMS.values()])
        ax.set_xticklabels([f"{f:g}" for f in ARMS.values()], fontsize=8)
        ax.set_xlabel("AGN density mis-anchoring factor", fontsize=9, color=INK)
        ax.set_title(f"m<{rung[1:]}", fontsize=11, color=INK)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        for s_ in ("left", "bottom"):
            ax.spines[s_].set_color(MUTED); ax.spines[s_].set_linewidth(0.8)
        ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=8)
        ax.grid(axis="y", color="#ececec", lw=0.7); ax.set_axisbelow(True)
    axes[0].set_ylabel(r"$f_{\rm AGN}$", fontsize=11, color=INK)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left", labelcolor=INK)
    axes[0].text(0.03, 0.03, "bars = 90% CI\ndashed = truth",
                 transform=axes[0].transAxes, fontsize=7.5, color=MUTED)
    fig.suptitle("Mis-anchoring the AGN completion density, by estimator"
                 f"   —   selection rungs present: {', '.join(have_new) or 'none yet'}",
                 fontsize=11.5, color=INK, y=1.04)
    fig.tight_layout()

    FIGS.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"fig1_anchor_response.{ext}", dpi=200,
                    bbox_inches="tight", facecolor="white")
    print(f"  wrote figs/fig1_anchor_response.pdf and .png  ({n} rungs)")
    plt.close(fig)

    (A4 / "results" / "arms_summary.json").write_text(json.dumps(
        {"_what": "f_AGN response to AGN-density mis-anchoring, by estimator. "
                  "'slope' is d(f_AGN)/dlog10(factor) -- the SENSITIVITY; "
                  "'offset_at_1' is where the correctly-anchored arm sits "
                  "relative to truth -- the BIAS. The archived result is a "
                  "large slope; the question is whether the estimator change "
                  "moves the slope, the offset, or both.",
         "truth_f_AGN": TRUTH_F, "rungs": summary}, indent=2))
    print("  wrote results/arms_summary.json")
    for rung, d in summary.items():
        for lbl, f in d.items():
            print(f"  {rung:<5} {lbl:<24} slope {f['slope']:+.4f}  "
                  f"offset@1.0 {f['offset_at_1']:+.4f}  (n={f['n_arms']})")


if __name__ == "__main__":
    main()
