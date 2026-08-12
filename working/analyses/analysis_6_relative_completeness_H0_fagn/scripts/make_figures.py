#!/usr/bin/env python
"""Result figures for analysis 6 under c_mode=selection. Deterministic.

    python make_figures.py        # writes ../figs/*.pdf and *.png

fig1_ratio_law   the f_AGN offset against the completeness ratio C_AGN/C_GAL,
                 archived per_pixel vs selection, with the archived headline
                 relation drawn.

The archived campaign's headline was that the f_AGN bias is SET by the relative
completeness of the two tracers, offset ~ 0.067 + 0.124 log10(C_AGN/C_GAL) at
R2 = 0.89. This figure asks whether that relation survives the estimator change.

Intervals are 90% (project rule). Colours are the fixed Okabe-Ito order used
across this campaign; truth is a neutral black dashed rule, never a series hue.
"""
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
A6 = HERE.parent
ARCHIVE = (A6.parent.parent / "archive" /
           "analysis_6_relative_completeness_H0_fagn" / "results")
FIGS = A6 / "figs"

TRUTH_F = 0.295
# in-horizon completeness of each depth (surveys_meta, both tracers)
C = {"complete": 1.0, "m20": 0.8143, "m19": 0.3160, "m18": 0.0957}
SERIES = [("per_pixel (archived)", "#0072B2", ARCHIVE),
          ("selection (this work)", "#009E73", A6 / "results")]
INK, MUTED = "#1a1a1a", "#6b6b6b"


def load(resdir):
    out = []
    for p in sorted(Path(resdir).glob("joint_g*_a*_s100.json")):
        tag = p.name.replace("joint_", "").replace("_s100.json", "")
        g, a = tag.split("_a", 1)
        g = g[1:]
        if g not in C or a not in C:
            continue
        d = json.loads(p.read_text())
        out.append({"cell": tag, "gal": g, "agn": a,
                    "x": np.log10(C[a] / C[g]),
                    "y": d["f"]["median"] - TRUTH_F,
                    "ci90": [d["f"]["ci90"][0] - TRUTH_F,
                             d["f"]["ci90"][1] - TRUTH_F]})
    return out


def fit(rows):
    if len(rows) < 2:
        return None
    x = np.array([r["x"] for r in rows]); y = np.array([r["y"] for r in rows])
    s, i = np.polyfit(x, y, 1)
    r2 = float(np.corrcoef(x, y)[0, 1] ** 2)
    return {"slope": float(s), "intercept": float(i), "r2": r2,
            "rms": float(np.std(y - (s * x + i))), "n": len(rows),
            "span": float(y.max() - y.min())}


def main():
    data = [(lbl, c, load(d)) for lbl, c, d in SERIES]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.axhline(0, color="black", ls="--", lw=1.2, zorder=1)

    fits = {}
    for lbl, colour, rows in data:
        if not rows:
            continue
        f = fit(rows)
        fits[lbl] = f
        x = [r["x"] for r in rows]; y = [r["y"] for r in rows]
        lo = [r["y"] - r["ci90"][0] for r in rows]
        hi = [r["ci90"][1] - r["y"] for r in rows]
        ax.errorbar(x, y, yerr=[lo, hi], fmt="o", color=colour, ms=7, lw=2.0,
                    capsize=0, label=lbl, zorder=3)
        ax.plot(x, y, "o", color="white", ms=2.5, zorder=4)
        if f:
            xs = np.linspace(min(x) - 0.1, max(x) + 0.1, 50)
            ax.plot(xs, f["slope"] * xs + f["intercept"], "-", color=colour,
                    lw=1.6, alpha=0.75, zorder=2)
            ax.annotate(
                f"slope {f['slope']:+.3f},  $R^2$ = {f['r2']:.3f}",
                xy=(xs[-1], f["slope"] * xs[-1] + f["intercept"]),
                fontsize=8.5, color=colour,
                xytext=(4, 4 if f["slope"] > 0 else -12),
                textcoords="offset points")

    ax.set_xlabel(r"$\log_{10}\ (C_{\rm AGN} / C_{\rm GAL})$   "
                  "(relative completeness of the two tracers)",
                  fontsize=10, color=INK)
    ax.set_ylabel(r"$f_{\rm AGN}$ offset from truth", fontsize=10, color=INK)
    ax.set_title("Is the f$_{\\rm AGN}$ bias set by RELATIVE completeness?\n"
                 "off-diagonal GAL$\\times$AGN depth cells, seed 100, anchors fixed at truth",
                 fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=9, loc="upper left", labelcolor=INK)
    ax.text(0.02, 0.03, "bars = 90% CI   ·   dashed rule = truth",
            transform=ax.transAxes, fontsize=8, color=MUTED)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED); ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=9)
    ax.grid(axis="y", color="#ececec", lw=0.7); ax.set_axisbelow(True)
    fig.tight_layout()

    FIGS.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"fig1_ratio_law.{ext}", dpi=200,
                    bbox_inches="tight", facecolor="white")
    print("  wrote figs/fig1_ratio_law.pdf and .png")
    plt.close(fig)

    (A6 / "results" / "surface_summary.json").write_text(json.dumps(
        {"_what": "f_AGN offset vs relative completeness, by estimator. The "
                  "archived per_pixel headline was offset ~ 0.067 + 0.124 "
                  "log10(C_AGN/C_GAL) at R2 0.89 over 12 cells; here 8 "
                  "off-diagonal cells (the 3 diagonal cells live in analysis 3 "
                  "and the oracle in analysis 4).",
         "truth_f_AGN": TRUTH_F, "completeness": C,
         "fits": fits,
         "cells": {lbl: rows for lbl, _, rows in data}}, indent=2, default=float))
    print("  wrote results/surface_summary.json")
    for lbl, f in fits.items():
        if f:
            print(f"  {lbl:<24} slope {f['slope']:+.4f}  R2 {f['r2']:.3f}  "
                  f"span {f['span']:.4f}  (n={f['n']})")


if __name__ == "__main__":
    main()
