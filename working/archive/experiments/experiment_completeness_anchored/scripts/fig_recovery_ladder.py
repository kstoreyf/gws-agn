#!/usr/bin/env python3
"""Explicit H0-recovery figure for the anchored completeness ladder.

Reads ONLY this experiment's results files:

  results/anch_{c100,m20,m19,m18}.h5    H0_grid + log_likelihood per rung
  results/anch_{c100,m20,m19,m18}.json  median / ci68 per rung
  results/summary.json                  completeness fraction per rung (labels)

and writes:

  figs/fig_h0_recovery_ladder.{pdf,png}  the four normalised H0 posteriors
      overlaid, one colour per completeness rung, truth line, medians ticked
      at the baseline

Pure post-processing: h5py/numpy/matplotlib, CPU, no darksirens import.
Posterior = exp(logL - max) normalised by the trapezoid rule (flat prior).

Same framing as fig_completeness_ladder: the comparison is DIFFERENTIAL
against the complete-catalog control — the common leftward displacement from
truth is the unresolved baseline bias measured in ../experiment_matched_mock.
"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent.parent
RESULTS, FIGS = BASE / "results", BASE / "figs"
FIGS.mkdir(parents=True, exist_ok=True)
H0_TRUTH = 67.74

BLUE, AQUA, YELLOW, GREEN = "#2a78d6", "#1baf7a", "#eda100", "#008300"
INK, INK2, INK3, GRIDCOL = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "font.family": "serif",
    "font.serif": ["DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 9, "axes.labelsize": 9.5, "axes.titlesize": 9.3,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 7.8,
    "axes.edgecolor": INK2, "axes.linewidth": 0.7, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "grid.color": GRIDCOL, "grid.linewidth": 0.6,
    "legend.frameon": False, "lines.solid_capstyle": "round",
})

# Same rung colours/markers as fig_completeness_ladder (colour follows the rung
# across figures); a distinct dash pattern per rung so identity never rests on
# colour alone.
LADDER = [
    ("c100", "complete", GREEN, (0, ())),
    ("m20", r"$m<20$", BLUE, (0, (5, 1.8))),
    ("m19", r"$m<19$", AQUA, (0, (3, 1.4, 1, 1.4))),
    ("m18", r"$m<18$", YELLOW, (0, (1, 1.2))),
]


def main():
    summ = json.loads((RESULTS / "summary.json").read_text())

    rungs = []
    for tag, lab, col, dash in LADDER:
        with h5py.File(RESULTS / f"anch_{tag}.h5", "r") as f:
            x, ll = f["H0_grid"][:], f["log_likelihood"][:]
        ok = np.isfinite(ll)
        p = np.zeros_like(ll)
        p[ok] = np.exp(ll[ok] - ll[ok].max())
        p /= np.trapz(p, x)
        j = json.loads((RESULTS / f"anch_{tag}.json").read_text())
        comp = summ["levels"][tag]["completeness_within_z_ref"]
        rungs.append({"tag": tag, "lab": lab, "col": col, "dash": dash,
                      "x": x, "pdf": p, "median": j["H0"]["median"],
                      "ci68": j["H0"]["ci68"], "comp": comp,
                      "n_rejected": j["n_neginf_cells"]})

    fig, ax = plt.subplots(figsize=(5.4, 3.4), dpi=300)
    ax.axvline(H0_TRUTH, color=INK, lw=1.1, ls=(0, (2, 2)), zorder=4)
    ax.annotate("truth", xy=(H0_TRUTH, 1.0), xycoords=("data", "axes fraction"),
                xytext=(3, -2), textcoords="offset points", ha="left", va="top",
                fontsize=7.6, color=INK)

    ymax = max(r["pdf"].max() for r in rungs)
    for r in rungs:
        ax.plot(r["x"], r["pdf"], color=r["col"], lw=1.4, ls=r["dash"], zorder=3,
                label=f"{r['lab']}  ({100 * r['comp']:.0f}%):  "
                      f"$H_0 = {r['median']:.2f}$")
        # median tick at the baseline
        ax.plot([r["median"], r["median"]], [0, 0.05 * ymax], color=r["col"],
                lw=2.0, solid_capstyle="butt", zorder=5)

    lo = min(r["x"][r["pdf"] > 1e-3 * r["pdf"].max()].min() for r in rungs)
    hi = max(r["x"][r["pdf"] > 1e-3 * r["pdf"].max()].max() for r in rungs)
    ax.set_xlim(min(lo, H0_TRUTH) - 0.5, max(hi, H0_TRUTH) + 0.5)
    ax.set_ylim(0, 1.28 * ymax)
    ax.set_xlabel(r"$H_0$  [km s$^{-1}$ Mpc$^{-1}$]")
    ax.set_ylabel(r"posterior density  $p(H_0\,|\,d)$")
    ax.set_title("Anchored completeness ladder: $H_0$ posteriors\n"
                 "(ticks: medians; common displacement = baseline bias)")
    ax.grid(True, alpha=0.55)
    ax.set_axisbelow(True)
    # Upper left: all four posteriors peak right of centre, leaving it empty.
    ax.legend(loc="upper left", title="flux limit (completeness within horizon)",
              title_fontsize=7.4)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"fig_h0_recovery_ladder.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("wrote figs/fig_h0_recovery_ladder.{pdf,png}")

    for r in rungs:
        print(f"  {r['tag']:5s} C={100*r['comp']:5.1f}%  "
              f"H0 = {r['median']:.3f}  ci68 [{r['ci68'][0]:.3f}, "
              f"{r['ci68'][1]:.3f}]  offset {r['median'] - H0_TRUTH:+.3f}  "
              f"rejected {r['n_rejected']}")


if __name__ == "__main__":
    main()
