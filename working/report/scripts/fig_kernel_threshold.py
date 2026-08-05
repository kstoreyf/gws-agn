#!/usr/bin/env python3
"""The catalog kernel width is a threshold lever, not a linear one.

Recovered expansion-rate offset against the effective catalog kernel width
sqrt(dzgals^2 + sigma_kde^2) when the kernels are broadened at fixed data, for
two independent realisations of the matched single-tracer mock.  The reference
band is the 16-84% range of the per-event PE redshift width
sigma_dL * dL / (d dL/dz): the offset is flat until the kernels approach it.

Reads experiment_matched_mock/results/skde_summary.json.
"""
from __future__ import annotations

import json
import math

import matplotlib.pyplot as plt

import figstyle as fs

RES = fs.EXP / "experiment_matched_mock" / "results"


def main():
    fs.use()
    S = json.loads((RES / "skde_summary.json").read_text())
    dz = S["dzgals"]
    pe = S["pe_redshift_width"]

    fig, ax = plt.subplots(figsize=(fs.ONECOL, 2.5))

    # PE redshift resolution band (16-84%) with its median
    ax.axvspan(pe["p16"], pe["p84"], color=fs.GRID, alpha=0.55, zorder=1)
    ax.axvline(pe["median"], color=fs.MUTED, lw=0.9, ls=(0, (3, 2)), zorder=1.6)
    ax.annotate("PE redshift\nwidth", (pe["p16"], 0.60),
                xycoords=("data", "axes fraction"), xytext=(-4, 0),
                textcoords="offset points", ha="right", va="top",
                fontsize=6.8, color=fs.INK2)

    series = [("b", S["realisations"]["b"]["rungs"], fs.C["blue"], "o",
               "realisation 1"),
              ("s4102", S["realisations"]["s4102"]["rungs"], fs.C["orange"], "s",
               "realisation 2")]
    for tag, rungs, col, mk, lab in series:
        ks = sorted(rungs, key=float)
        x = [math.hypot(dz, float(k)) for k in ks]
        y = [rungs[k]["offset"] for k in ks]
        e = [0.5 * (rungs[k]["ci68"][1] - rungs[k]["ci68"][0]) for k in ks]
        ax.errorbar(x, y, yerr=e, fmt=mk + "-", ms=3.8, lw=1.3, color=col,
                    ecolor=col, elinewidth=1.0, mfc=col, mec="#fcfcfb",
                    mew=0.7, label=lab, zorder=4)

    fs.truth_line(ax, 0.0, axis="y", label="no offset", pos=0.28)
    ax.set_xscale("log")
    ax.set_xticks([0.003, 0.01, 0.03, 0.07])
    ax.set_xticklabels(["0.003", "0.01", "0.03", "0.07"])
    ax.set_xlim(0.0026, 0.085)
    ax.set_xlabel(r"effective catalog kernel width $(\mathrm{d}z^2+\sigma_{\rm kde}^2)^{1/2}$")
    ax.set_ylabel(r"$\Delta H_0$  (km s$^{-1}$ Mpc$^{-1}$)")
    ax.legend(loc="lower left")

    fs.save(fig, "fig_kernel_threshold")


if __name__ == "__main__":
    main()
