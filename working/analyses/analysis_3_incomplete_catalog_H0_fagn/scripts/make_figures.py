#!/usr/bin/env python
"""Result figures for analysis 3 under c_mode=selection. Deterministic.

    python make_figures.py        # writes ../figs/*.pdf and *.png

fig1_ladder   f_AGN and H0 down the completeness ladder, the new selection-mode
              result against the archived per_pixel one, with truth marked.
              Intervals are 90% (project rule).

The ladder starts at the COMPLETE catalog, not at m<21. That rung is the
control: at C = 100% there is nothing to complete, so any offset it carries is
the estimator's own, not a completion error. Reading the four faint rungs
without it invites the wrong conclusion.

The archived campaign is shown alongside because the estimator IS the result
here: the same grids, the same anchors, the same seed, one changed estimator.

Colours: fixed categorical order (Okabe-Ito), checked for colour-vision
separation -- the pair clears normal-vision dE 18.7 and worst-case CVD dE 10.5,
above the 15/8 floors. Truth is a neutral black dashed rule, never a series
colour.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
A3 = HERE.parent
ARCHIVE = (A3.parent.parent / "archive" /
           "analysis_3_incomplete_catalog_H0_fagn" / "results")
FIGS = A3 / "figs"

RUNGS = ["complete", "m21", "m20", "m19", "m18"]
# in-horizon completeness, from the mock's surveys_meta (both tracers, diagonal)
COMPLETENESS = {"complete": 1.0, "m21": 0.997, "m20": 0.8143,
                "m19": 0.3151, "m18": 0.0954}
TRUTH = {"f": 0.295, "H0": 67.74}
SEED100_H0_DRAW = 69.22       # this seed's own complete-catalog H0

SERIES = [("per_pixel (archived)", "#0072B2", ARCHIVE),
          ("selection (this work)", "#009E73", A3 / "results")]
INK, MUTED = "#1a1a1a", "#6b6b6b"


def fmt_c(c):
    """99.7% and 100% must not both print as '100%' -- the complete rung and
    m<21 are different rungs and the tick labels are how a reader tells them
    apart."""
    return "100%" if c >= 0.9995 else f"{c:.1%}"


def rung_label(r):
    """Tick label: the magnitude cut, or 'complete', over its completeness."""
    head = "complete" if r == "complete" else f"m<{r[1:]}"
    return f"{head}\nC={fmt_c(COMPLETENESS[r])}"


def load(resdir):
    out = {}
    for r in RUNGS:
        p = Path(resdir) / f"joint_{r}_s100.json"
        if p.exists():
            d = json.loads(p.read_text())
            # 90%, per the project rule -- these JSONs carry both ci68 and ci90.
            out[r] = {k: (d[k]["median"], d[k]["ci90"]) for k in ("H0", "f")}
    return out


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED); ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=9)
    ax.grid(axis="y", color="#ececec", lw=0.7)
    ax.set_axisbelow(True)


def main():
    data = [(lbl, c, load(d)) for lbl, c, d in SERIES]
    x = np.arange(len(RUNGS))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.1))

    for ax, key, truth, ylab in (
            (axes[0], "f", TRUTH["f"], r"$f_{\rm AGN}$"),
            (axes[1], "H0", TRUTH["H0"], r"$H_0$  [km s$^{-1}$ Mpc$^{-1}$]")):
        ax.axhline(truth, color="black", ls="--", lw=1.2, zorder=1)
        for k, (lbl, colour, d) in enumerate(data):
            if not d:
                continue
            off = (k - 0.5) * 0.13
            xs, ys, lo, hi = [], [], [], []
            for i, r in enumerate(RUNGS):
                if r not in d:
                    continue
                m, ci = d[r][key]        # ci90
                xs.append(i + off); ys.append(m)
                lo.append(m - ci[0]); hi.append(ci[1] - m)
            ax.errorbar(xs, ys, yerr=[lo, hi], fmt="o-", color=colour,
                        ms=7, lw=2.0, capsize=0, label=lbl, zorder=3)
            ax.plot(xs, ys, "o", color="white", ms=2.5, zorder=4)
        if key == "H0":
            ax.axhline(SEED100_H0_DRAW, color=MUTED, ls=":", lw=1.2, zorder=1)
            ax.text(len(RUNGS) - 0.55, SEED100_H0_DRAW, "  seed 100's own\n  complete-catalog draw",
                    fontsize=7.5, color=MUTED, va="center")
        ax.set_xticks(x)
        ax.set_xticklabels([rung_label(r) for r in RUNGS], fontsize=8.5)
        ax.set_ylabel(ylab, fontsize=11, color=INK)
        style(ax)

    axes[0].legend(frameon=False, fontsize=9, loc="upper left", labelcolor=INK)
    axes[0].text(0.02, 0.02, "bars = 90% CI   ·   dashed rule = truth", transform=axes[0].transAxes,
                 fontsize=8, color=MUTED)
    fig.suptitle("The completeness ladder, by estimator — same grids, same anchors, "
                 "same seed\nseed 100, K=2 field mixture, anchors fixed at truth",
                 fontsize=11.5, color=INK, y=1.06)
    fig.tight_layout()

    FIGS.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"fig1_ladder.{ext}", dpi=200, bbox_inches="tight",
                    facecolor="white")
    print(f"  wrote figs/fig1_ladder.pdf and .png")
    plt.close(fig)

    # a compact numeric companion, so the figure never has to be re-measured
    rows = []
    for r in RUNGS:
        row = {"rung": r, "C_in_horizon": COMPLETENESS[r]}
        for lbl, _, d in data:
            if r in d:
                row[lbl] = {"f_AGN": d[r]["f"][0], "f_offset": d[r]["f"][0] - TRUTH["f"],
                            "H0": d[r]["H0"][0], "H0_offset": d[r]["H0"][0] - TRUTH["H0"]}
        rows.append(row)
    (A3 / "results" / "ladder_summary.json").write_text(
        json.dumps({"truth": TRUTH, "seed100_complete_H0_draw": SEED100_H0_DRAW,
                    "rows": rows}, indent=2))
    print("  wrote results/ladder_summary.json")


if __name__ == "__main__":
    main()
