#!/usr/bin/env python3
"""Single-tracer flux-limit ladder: what incompleteness costs when the missing
host budget is known.

Panel (a): the completeness the flux limits actually impose, as a function of
redshift across the detection horizon.  Panel (b): the recovered distance scale
at each level, against the complete-catalog control.  Panel (c): the width of
the credible interval, as a factor against that control.

Reads experiment_completeness_anchored/results/summary.json.
"""
from __future__ import annotations

import json

import numpy as np
import matplotlib.pyplot as plt

import figstyle as fs

RES = fs.EXP / "experiment_completeness_anchored" / "results"
ORDER = ["c100", "m20", "m19", "m18"]


def main():
    fs.use()
    S = json.loads((RES / "summary.json").read_text())
    lv = S["levels"]
    zref = S["anchor"]["shape_residual_within_z_ref"]["z_ref"]

    fig, axes = plt.subplots(1, 3, figsize=(fs.TWOCOL, 2.35),
                             gridspec_kw={"wspace": 0.34})
    axa, axb, axc = axes

    # ---- (a) the imposed completeness -----------------------------------
    ramp = [fs.RAMP[0], fs.RAMP[2], fs.RAMP[3], fs.RAMP[4]]
    for col, key in zip(ramp, ORDER):
        L = lv[key]
        lab = "complete" if L["mag_limit"] is None else f"$m<{L['mag_limit']:.0f}$"
        if L["C_of_z_bins"] is None:
            axa.plot([0, zref], [1, 1], color=col, lw=1.8, zorder=4)
            fs.label_line(axa, zref, 1.0, lab, col, dx=-3, dy=-8, ha="right",
                          va="top")
            continue
        edges = np.asarray(L["C_of_z_bins"]["edges"], float)
        C = np.asarray(L["C_of_z_bins"]["C"], float)
        zc = 0.5 * (edges[:-1] + edges[1:])
        axa.plot(zc, C, color=col, lw=1.8, marker="o", ms=3.0, mfc=col,
                 mec="#fcfcfb", mew=0.5, zorder=4)
        fs.label_line(axa, zc[-1], C[-1], lab, col, dx=4, ha="left", va="center")
    axa.set_xlabel(r"host redshift $z$")
    axa.set_ylabel(r"completeness $C(z)$")
    axa.set_title("(a)  imposed by a flux limit")
    axa.set_xlim(0, zref * 1.30)
    axa.set_ylim(0, 1.08)

    # ---- (b) recovery, differential against the control -------------------
    Cs = np.array([lv[k]["completeness_within_z_ref"] for k in ORDER])
    off = np.array([lv[k]["offset"] for k in ORDER])
    hwd = np.array([lv[k]["hw"] for k in ORDER])
    ctl, ctl_hw = lv["c100"]["offset"], lv["c100"]["hw"]
    axb.axhspan(ctl - ctl_hw, ctl + ctl_hw, color=fs.C["blue"], alpha=0.13, lw=0,
                zorder=1)
    axb.axhline(ctl, color=fs.C["blue"], lw=1.3, zorder=2)
    axb.errorbar(Cs[1:], off[1:], yerr=hwd[1:], fmt="o", ms=5.0,
                 color=fs.C["orange"], ecolor=fs.C["orange"], elinewidth=1.6,
                 mfc=fs.C["orange"], mec="#fcfcfb", mew=0.9, zorder=4)
    axb.plot([Cs[0]], [off[0]], "o", ms=5.0, color=fs.C["blue"],
             mfc=fs.C["blue"], mec="#fcfcfb", mew=0.9, zorder=5)
    axb.annotate("complete-catalog control", (0.03, 0.04), xycoords="axes fraction",
                 ha="left", va="bottom", fontsize=6.8, color=fs.C["blue"])
    fs.truth_line(axb, 0.0, axis="y", label="no offset")
    axb.invert_xaxis()
    axb.set_xlabel(r"completeness $C(z\leq%.2f)$" % zref)
    axb.set_ylabel(r"$\Delta H_0$  (km s$^{-1}$ Mpc$^{-1}$)")
    axb.set_title("(b)  no added offset")

    # ---- (c) the width ---------------------------------------------------
    grow = np.array([S["verdict"]["interval_growth"][k] for k in ORDER])
    # discrete levels, one realisation each: dots, not a trend line
    axc.vlines(Cs, 1.0, grow, color=fs.C["aqua"], lw=1.4, alpha=0.45, zorder=3)
    axc.plot(Cs, grow, "o", color=fs.C["aqua"], ms=5.5, mfc=fs.C["aqua"],
             mec="#fcfcfb", mew=0.9, zorder=4, ls="none")
    for c, g in zip(Cs, grow):
        axc.annotate(f"{g:.2f}", (c, g), xytext=(0, 6),
                     textcoords="offset points", ha="center", fontsize=7.0,
                     color=fs.INK2)
    axc.axhline(1.0, color=fs.TRUTH, lw=0.9, ls=(0, (3, 2)), alpha=0.75, zorder=1.5)
    axc.invert_xaxis()
    axc.set_xlabel(r"completeness $C(z\leq%.2f)$" % zref)
    axc.set_ylabel(r"$\sigma(H_0)$ / control")
    axc.set_title("(c)  the price is width")
    axc.set_ylim(0.8, 1.15 * grow.max())

    fs.save(fig, "fig_completeness_anchored")


if __name__ == "__main__":
    main()
