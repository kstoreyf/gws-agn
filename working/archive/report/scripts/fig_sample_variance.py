#!/usr/bin/env python3
"""Host-catalog sample variance in the single-tracer closure test.

Panel (a): five independent realisations -- catalog, events and posterior
samples all reseeded together -- with the 68% interval each one quotes.  The
realisations scatter by more than those intervals allow.

Panel (b): the same statement as an error budget.  The quoted interval is
conditional on one catalog; quadrature-subtracting it from the realised scatter
leaves the contribution of the catalog realisation itself.

Reads experiment_matched_mock/results/summary.json (multi_seed block).
"""
from __future__ import annotations

import json

import numpy as np
import matplotlib.pyplot as plt

import figstyle as fs

RES = fs.EXP / "experiment_matched_mock" / "results"
H0_TRUTH = 67.74


def main():
    fs.use()
    S = json.loads((RES / "summary.json").read_text())
    ms = S["multi_seed"]
    seeds = ms["seeds"]

    fig, (axa, axb) = plt.subplots(
        1, 2, figsize=(fs.TWOCOL, 2.5),
        gridspec_kw={"width_ratios": [1.45, 1.0], "wspace": 0.32})

    # ---- (a) per-realisation offsets with their own intervals -------------
    x = np.arange(len(seeds))
    y = np.array([s["median"] - H0_TRUTH for s in seeds])
    e = np.array([s["hw"] for s in seeds])
    mean, sem, sd = ms["offset"], ms["sem"], ms["scatter_sd"]

    axa.axhspan(mean - sd, mean + sd, color=fs.C["orange"], alpha=0.13, lw=0,
                zorder=1)
    axa.axhline(mean, color=fs.C["orange"], lw=1.4, zorder=2)
    fs.truth_line(axa, 0.0, axis="y", label="no offset")
    axa.errorbar(x, y, yerr=e, fmt="o", ms=5.0, color=fs.C["blue"],
                 ecolor=fs.C["blue"], elinewidth=1.6, mfc=fs.C["blue"],
                 mec="#fcfcfb", mew=0.9, zorder=4)
    fs.label_line(axa, x[-1], mean, f"mean {mean:+.2f}", fs.C["orange"],
                  dx=6, dy=10, ha="right", va="bottom")
    fs.label_line(axa, x[0], mean + sd, "realised scatter", fs.C["orange"],
                  dx=-2, dy=3, ha="left", va="bottom")
    axa.set_xticks(x)
    axa.set_xticklabels([s["seed"] for s in seeds])
    axa.set_xlabel("realisation")
    axa.set_ylabel(r"$\Delta H_0$  (km s$^{-1}$ Mpc$^{-1}$)")
    axa.set_title("(a)  each realisation and the interval it quotes")
    axa.set_xlim(-0.5, len(seeds) - 0.5)
    axa.grid(axis="x", visible=False)

    # ---- (b) the budget --------------------------------------------------
    labels = ["quoted 68%\nhalf-width", "realised\nscatter",
              "catalog\ncontribution"]
    vals = [ms["mean_halfwidth"], sd, ms["catalog_variance_component"]]
    cols = [fs.C["blue"], fs.C["orange"], fs.C["aqua"]]
    xb = np.arange(3)
    axb.bar(xb, vals, width=0.62, color=cols, zorder=3,
            linewidth=1.2, edgecolor="#fcfcfb")
    for xi, v, c in zip(xb, vals, cols):
        axb.annotate(f"{v:.2f}", (xi, v), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom",
                     fontsize=7.5, color=fs.INK)
    axb.set_xticks(xb)
    axb.set_xticklabels(labels)
    axb.tick_params(axis="x", length=0)
    axb.set_ylabel(r"km s$^{-1}$ Mpc$^{-1}$")
    axb.set_title("(b)  intervals against the realised spread")
    axb.set_ylim(0, 1.22 * max(vals))
    axb.grid(axis="x", visible=False)
    axb.annotate(f"{ms['interval_underestimate_factor']:.1f}$\\times$ too narrow",
                 xy=(0.5, 0.5 * (vals[0] + vals[1])), xytext=(0.5, 1.14 * max(vals)),
                 ha="center", va="top", fontsize=7.0, color=fs.INK2)

    fs.save(fig, "fig_sample_variance")


if __name__ == "__main__":
    main()
