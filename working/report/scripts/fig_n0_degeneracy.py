#!/usr/bin/env python3
"""The degeneracy between the AGN-hosted fraction and the AGN number density.

Panel (a): the joint likelihood in (f_AGN, log10 n0_AGN) at the complete rung
and at the shallowest one.  A missing AGN host and an AGN-hosted event explain
the same observation, so the two parameters trade off along a curve whose upper
tip is the truth.  Panel (b): the flat-prior density marginal at each rung,
which recovers below truth by an amount that shrinks as the survey empties.

Reads experiment_completeness_free/results/fn0_*_fix.h5 and
n0_arms_summary_fix.json (the post-fix grids).
"""
from __future__ import annotations

import json

import h5py
import numpy as np
import matplotlib.pyplot as plt

import figstyle as fs

FREE = fs.EXP / "experiment_completeness_free" / "results"
INC = fs.EXP / "experiment_twotracer_incomplete" / "results"
RUNGS = ["complete", "m21.0", "m20.0", "m19.0", "m18.0"]
TAG = {"complete": "complete", "m21.0": "m21.0", "m20.0": "m20.0",
       "m19.0": "m19.0", "m18.0": "m18.0"}
SHOW = ["complete", "m18.0"]


def grid(rung):
    with h5py.File(FREE / f"fn0_{TAG[rung]}_fix.h5", "r") as f:
        return f["f_grid"][:], f["n0c2_grid"][:], f["log_likelihood"][:]


def main():
    fs.use()
    D = json.loads((FREE / "n0_arms_summary_fix.json").read_text())
    I = json.loads((INC / "summary_fix.json").read_text())
    gtrue, ftrue = D["g_true"], D["truth_f"]

    fig, (axa, axb) = plt.subplots(
        1, 2, figsize=(fs.TWOCOL, 2.6),
        gridspec_kw={"width_ratios": [1.15, 1.0], "wspace": 0.30})

    for rung, col in zip(SHOW, (fs.C["blue"], fs.C["orange"])):
        F, G, L = grid(rung)
        levels, pn = fs.hpd_levels(L)
        C = I["levels"][rung]["agn_completeness_within_horizon"]
        axa.contourf(F, G, pn.T, levels=[levels[1], levels[0], 1.0001],
                     colors=[col, col], alpha=0.18)
        axa.contour(F, G, pn.T, levels=[levels[1]], colors=[col], linewidths=0.9)
        axa.contour(F, G, pn.T, levels=[levels[0]], colors=[col], linewidths=1.6)
        rho = D["levels"][rung]["rho_f_n0_flat_prior"]
        axa.plot([], [], color=col, lw=1.8,
                 label=rf"$C={C:.2f}$,  $\rho={rho:+.2f}$")
    axa.plot([ftrue], [gtrue], marker="+", ms=9, mew=1.6, color=fs.TRUTH, zorder=6)
    axa.annotate("truth", (ftrue, gtrue), xytext=(6, -9),
                 textcoords="offset points", fontsize=6.8, color=fs.INK2)
    axa.set_xlabel(r"$f_{\rm AGN}$")
    axa.set_ylabel(r"$\log_{10} n_{0,\rm AGN}$  (Mpc$^{-3}$)")
    axa.set_title("(a)  68% and 90% credible regions")
    axa.set_xlim(0, 0.85)
    axa.set_ylim(-9.3, -7.15)
    axa.legend(loc="lower right")

    # ---- (b) the density marginal at every rung --------------------------
    for rung, col in zip(RUNGS, [fs.RAMP[0], fs.RAMP[1], fs.RAMP[2], fs.RAMP[3],
                                 fs.RAMP[4]]):
        F, G, L = grid(rung)
        pg = np.nansum(np.exp(L - np.nanmax(L)), axis=0)
        pg /= np.trapz(pg, G)
        C = I["levels"][rung]["agn_completeness_within_horizon"]
        axb.plot(G, pg, color=col, lw=1.6, zorder=3,
                 label=rf"$C={C:.2f}$")
    fs.truth_line(axb, gtrue, axis="x", label="truth", pos=0.985)
    # the two shallowest rungs keep some mass against the low edge of the
    # scanned range; the amount is quoted in the caption rather than annotated
    # on top of the curves.
    axb.set_xlabel(r"$\log_{10} n_{0,\rm AGN}$  (Mpc$^{-3}$)")
    axb.set_ylabel("marginal density (flat prior)")
    axb.set_title("(b)  the density recovers low")
    axb.set_xlim(-9.6, -7.1)
    axb.set_ylim(bottom=0)
    axb.set_yticks([])
    axb.legend(loc="upper left", ncol=1)

    fs.save(fig, "fig_n0_degeneracy")


if __name__ == "__main__":
    main()
