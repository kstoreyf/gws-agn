#!/usr/bin/env python3
"""Where the AGN-hosted fraction stops being measurable.

Detection significance of f_AGN, median/sigma, over survey completeness (rows)
and prior knowledge of the sparse tracer's comoving number density (columns).
Each column is a reweighting of one likelihood grid per rung, so the whole map
comes from five grids.

Reads experiment_completeness_free/results/n0_arms_summary_fix.json for the
values and experiment_twotracer_incomplete/results/summary_fix.json for the completeness
of each rung.
"""
from __future__ import annotations

import json

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize

import figstyle as fs

FREE = fs.EXP / "experiment_completeness_free" / "results"
INC = fs.EXP / "experiment_twotracer_incomplete" / "results"
RUNGS = ["complete", "m21.0", "m20.0", "m19.0", "m18.0"]
ARMS = ["fixed", "5%", "10%", "30%", "factor 2", "free"]
ARM_LAB = ["exact", "5%", "10%", "30%", "factor 2", "free"]

# single-hue sequential ramp, light -> dark (magnitude, so never a rainbow)
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def main():
    fs.use()
    D = json.loads((FREE / "n0_arms_summary_fix.json").read_text())
    I = json.loads((INC / "summary_fix.json").read_text())
    C = [I["levels"][k]["agn_completeness_within_horizon"] for k in RUNGS]
    M = np.array([[D["levels"][k]["arms"][a]["detection_sigma"] for a in ARMS]
                  for k in RUNGS])

    cmap = LinearSegmentedColormap.from_list("seqblue", SEQ)
    norm = Normalize(vmin=0.0, vmax=float(M.max()))

    fig, ax = plt.subplots(figsize=(fs.ONECOL * 1.42, 2.55))
    ax.set_axisbelow(False)
    ax.grid(False)
    ax.imshow(M, cmap=cmap, norm=norm, aspect="auto", origin="upper",
              interpolation="nearest")
    # 2px surface gap between cells, per the mark spec
    for x in np.arange(-0.5, len(ARMS)):
        ax.axvline(x, color="#fcfcfb", lw=1.6, zorder=3)
    for y in np.arange(-0.5, len(RUNGS)):
        ax.axhline(y, color="#fcfcfb", lw=1.6, zorder=3)

    # every cell carries its value: identity is never colour alone
    for i in range(len(RUNGS)):
        for j in range(len(ARMS)):
            v = M[i, j]
            ink = "#ffffff" if norm(v) > 0.55 else fs.INK
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7.6,
                    color=ink, zorder=4)

    ax.set_xticks(range(len(ARMS)))
    ax.set_xticklabels(ARM_LAB)
    ax.set_yticks(range(len(RUNGS)))
    ax.set_yticklabels([f"{c:.2f}" for c in C])
    ax.tick_params(length=0)
    ax.set_xlabel(r"knowledge of $n_{0,\rm AGN}$")
    ax.set_ylabel(r"completeness $C(z\leq%.1f)$" % I["z_ref"])
    ax.set_title("detection significance of $f_{\\rm AGN}$  (median/$\\sigma$)")
    for s in ax.spines.values():
        s.set_visible(False)

    fs.save(fig, "fig_n0_significance")


if __name__ == "__main__":
    main()
