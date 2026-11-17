#!/usr/bin/env python3
"""The clustered mocks' H0 deficit decomposed into its two levers.

Panel (a): the log-likelihood terms against H0 at planted f_AGN = 0.307, each
normalised to its own peak: per-event (numerator) term, selection term
(-N_obs ln mu), their total, and the repaired estimator (host prior truncated
at the detection horizon, H0-independent selection normalisation).

Panel (b): the budget at both planted fractions -- numerator pull, selection
pull, net offset, repaired estimator -- as peak shifts from the truth.

Reads experiment_h0f_baseline/results/tilt_terms_fagn0.{3,7}.h5,
tilt_budget.json and tilt_repaired_estimator.json.
"""
from __future__ import annotations

import json

import h5py
import numpy as np
import matplotlib.pyplot as plt

import figstyle as fs

RES = fs.EXP / "experiment_h0f_baseline" / "results"
H0_TRUTH = 67.74
NOBS = 1000


def main():
    fs.use()
    B = json.loads((RES / "tilt_budget.json").read_text())
    R = json.loads((RES / "tilt_repaired_estimator.json").read_text())

    fig, (axa, axb) = plt.subplots(
        1, 2, figsize=(fs.TWOCOL, 2.6),
        gridspec_kw={"width_ratios": [1.15, 1.0], "wspace": 0.30})

    # ---- (a) the terms at f = 0.307 --------------------------------------
    with h5py.File(RES / "tilt_terms_fagn0.3.h5", "r") as f:
        H0 = f["H0_grid"][:]
        num = f["numerator/full"][:]
        lnmu = f["lnmu/full"][:]
        rep = f["numerator/zcut_1"][:]     # repaired: truncated prior, flat beta
    sel = -NOBS * lnmu
    tot = num + sel

    # The selection term itself has no interior peak: it is a near-linear tilt
    # of ~ -24 nats per km/s/Mpc, far outside this axis range.  Its effect is
    # the displacement between the per-event term and the total.
    series = [(tot, "total", fs.C["blue"]),
              (num, "per-event term", fs.C["orange"]),
              (rep, "repaired estimator", fs.C["magenta"])]
    for y, lab, col in series:
        dy = y - y.max()
        axa.plot(H0, dy, color=col, lw=1.5, label=lab, zorder=3)
        i = int(np.argmax(y))
        axa.plot([H0[i]], [0.0], "v", ms=4.5, color=col, mec="#fcfcfb",
                 mew=0.6, zorder=4, clip_on=False)
    fs.truth_line(axa, H0_TRUTH, axis="x", label="truth", pos=0.985)
    axa.set_xlim(58, 80)
    axa.set_ylim(-38, 3.6)
    axa.set_xlabel(r"$H_0$  (km s$^{-1}$ Mpc$^{-1}$)")
    axa.set_ylabel(r"$\Delta\ln\mathcal{L}$  (nats)")
    axa.set_title(r"(a)  terms at $f_{\rm AGN} = 0.307$, each to its own peak")
    axa.legend(loc="lower right", fontsize=6.8)

    # ---- (b) the budget at both fractions --------------------------------
    entries = [
        ("numerator pull", "numerator_offset", fs.C["orange"]),
        ("selection pull", "selection_shift_full", fs.C["aqua"]),
        ("net offset", "total_offset", fs.C["blue"]),
    ]
    fr = [("fagn0.3", "0.307"), ("fagn0.7", "0.703")]
    ny = len(entries) + 1
    for j, (fk, flab) in enumerate(fr):
        base = -j * (ny + 0.8)
        for i, (lab, key, col) in enumerate(entries):
            v = B[fk]["budget"][key]
            y = base - i
            axb.barh(y, v, height=0.62, color=col, zorder=3)
            axb.annotate(f"{v:+.2f}", (v, y),
                         xytext=(4 if v > 0 else -4, 0),
                         textcoords="offset points",
                         ha="left" if v > 0 else "right", va="center",
                         fontsize=6.8, color=fs.INK2)
        v = R[fk]["repaired_offset"]
        err = R[fk]["sigma_stat"]
        y = base - len(entries)
        axb.barh(y, v, height=0.62, color=fs.C["magenta"], zorder=3)
        if err:
            axb.errorbar([v], [y], xerr=[err], fmt="none", ecolor=fs.INK,
                         elinewidth=0.9, capsize=1.8, zorder=4)
        axb.annotate(f"{v:+.2f}", (v - (err or 0), y), xytext=(-4, 0),
                     textcoords="offset points", ha="right", va="center",
                     fontsize=6.8, color=fs.INK2)
        axb.annotate(rf"$f_{{\rm AGN}} = {flab}$", (0.99, base + 0.85),
                     xycoords=("axes fraction", "data"), ha="right",
                     va="bottom", fontsize=7.2, color=fs.INK)

    yticks, ylabels = [], []
    for j in range(len(fr)):
        base = -j * (ny + 0.8)
        for i, (lab, _, _) in enumerate(entries):
            yticks.append(base - i)
            ylabels.append(lab)
        yticks.append(base - len(entries))
        ylabels.append("repaired")
    axb.set_yticks(yticks)
    axb.set_yticklabels(ylabels, fontsize=7.0)
    axb.tick_params(axis="y", length=0)
    axb.axvline(0.0, color=fs.TRUTH, lw=0.9, ls=(0, (3, 2)), alpha=0.75,
                zorder=1.5)
    axb.set_xlim(-6.6, 5.6)
    axb.set_ylim(min(yticks) - 0.9, 1.3)
    axb.set_xlabel(r"peak shift from truth  (km s$^{-1}$ Mpc$^{-1}$)")
    axb.set_title("(b)  the two levers and their sum")
    axb.grid(axis="y", visible=False)

    fs.save(fig, "fig_tilt_decomposition")


if __name__ == "__main__":
    main()
