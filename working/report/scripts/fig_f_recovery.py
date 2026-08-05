#!/usr/bin/env python3
"""Recovery of the AGN-hosted fraction across its full planted range.

Panel (a): the flat-prior posteriors themselves, one per planted fraction, from
the 41-point scans at the true expansion rate.  Panel (b): recovered against
planted, with the 68% interval on each point, against the identity line.

Reads experiment_h0f_baseline/results/fscan_fagn*.h5 and its summary.json.
"""
from __future__ import annotations

import json

import h5py
import numpy as np
import matplotlib.pyplot as plt

import figstyle as fs

BASE = fs.EXP / "experiment_h0f_baseline" / "results"
TAGS = ["0.0", "0.3", "0.7", "1.0"]


def main():
    fs.use()
    with open(BASE / "summary.json") as fh:
        S = json.load(fh)

    fig, (axa, axb) = plt.subplots(
        1, 2, figsize=(fs.TWOCOL, 2.5),
        gridspec_kw={"width_ratios": [1.25, 1.0], "wspace": 0.30})

    for i, tag in enumerate(TAGS):
        col = fs.SERIES[i]
        ftrue = S["f_truth"][tag]
        with h5py.File(BASE / f"fscan_fagn{tag}.h5", "r") as f:
            g, L = f["f_grid"][:], f["log_likelihood"][:]
        p = fs.posterior_1d(g, L)
        axa.plot(g, p, color=col, lw=1.6, zorder=3)
        axa.fill_between(g, 0, p, color=col, alpha=0.12, lw=0)
        axa.plot([ftrue], [0], marker="^", ms=4.5, color=fs.TRUTH, clip_on=False,
                 zorder=5)
        fs.label_line(axa, g[np.argmax(p)], p.max(), f"{ftrue:.3f}", col,
                      dy=4, ha="center", va="bottom", size=7.0)

    axa.set_xlabel(r"$f_{\rm AGN}$")
    axa.set_ylabel("posterior density")
    axa.set_title("(a)  posteriors at the true distance scale")
    axa.set_xlim(-0.02, 1.02)
    axa.set_ylim(0, 1.16 * axa.get_ylim()[1])
    axa.set_yticks([])
    axa.annotate("planted value", xy=(0.307, 0), xytext=(0.34, 0.16),
                 textcoords="axes fraction", fontsize=6.8, color=fs.INK2,
                 arrowprops=dict(arrowstyle="-", lw=0.6, color=fs.MUTED,
                                 shrinkA=1, shrinkB=2))

    x, y, lo, hi = [], [], [], []
    for tag in TAGS:
        s = S["f_scan_at_true_H0"][tag]
        ftrue = S["f_truth"][tag]
        x.append(ftrue)
        y.append(s["median"])
        if tag == "1.0":                      # truth on the prior boundary
            lo.append(s["median"] - s["onesided68_lo"])
            hi.append(0.0)
        else:
            lo.append(s["median"] - s["ci68"][0])
            hi.append(s["ci68"][1] - s["median"])
    axb.plot([-0.02, 1.02], [-0.02, 1.02], color=fs.MUTED, lw=0.8,
             ls=(0, (3, 2)), zorder=1)
    axb.errorbar(x, y, yerr=[lo, hi], fmt="o", ms=4.5, color=fs.C["blue"],
                 ecolor=fs.C["blue"], elinewidth=1.4, mfc=fs.C["blue"],
                 mec="#fcfcfb", mew=0.8, zorder=4)
    axb.annotate("one-sided", xy=(1.0, y[-1]), xytext=(-6, -12),
                 textcoords="offset points", ha="right", fontsize=6.8,
                 color=fs.INK2)
    axb.set_xlabel(r"planted $f_{\rm AGN}$")
    axb.set_ylabel(r"recovered $f_{\rm AGN}$")
    axb.set_title("(b)  recovered against planted")
    axb.set_xlim(-0.04, 1.06)
    axb.set_ylim(-0.04, 1.06)
    axb.set_aspect("equal", adjustable="box")

    fs.save(fig, "fig_f_recovery")


if __name__ == "__main__":
    main()
