#!/usr/bin/env python3
"""Joint (H0, f_AGN) constraint from the clustered two-tracer mock.

Panel (a): 68% and 90% credible regions at two planted AGN-hosted fractions, on
shared axes, so the fraction's recovery and the expansion rate's deficit are
read off the same plane.  Panel (b): the marginal distance-scale posteriors,
where that deficit is the quantity of interest.

Reads experiment_h0f_baseline/results/jointzoom_fagn{0.3,0.7}.h5 (the refined
grids) -- the same files summary.json is built from.
"""
from __future__ import annotations


import h5py
import numpy as np
import matplotlib.pyplot as plt

import figstyle as fs

BASE = fs.EXP / "experiment_h0f_baseline" / "results"
H0_TRUTH = 67.74
CASES = [("0.3", 0.307, fs.C["blue"]), ("0.7", 0.703, fs.C["orange"])]


def load(tag):
    with h5py.File(BASE / f"jointzoom_fagn{tag}.h5", "r") as f:
        return f["H0_grid"][:], f["f_grid"][:], f["log_likelihood"][:]


def main():
    fs.use()
    fig, (axa, axb) = plt.subplots(
        1, 2, figsize=(fs.TWOCOL, 2.55),
        gridspec_kw={"width_ratios": [1.35, 1.0], "wspace": 0.30})

    for tag, ftrue, col in CASES:
        H, F, L = load(tag)
        levels, pn = fs.hpd_levels(L)
        # contourf wants increasing levels; the two HPD levels are decreasing
        axa.contourf(H, F, pn.T, levels=[levels[1], levels[0], 1.0001],
                     colors=[col, col], alpha=0.20, antialiased=True)
        axa.contour(H, F, pn.T, levels=[levels[1]], colors=[col],
                    linewidths=0.9, linestyles="solid")
        axa.contour(H, F, pn.T, levels=[levels[0]], colors=[col],
                    linewidths=1.6, linestyles="solid")
        axa.plot([H0_TRUTH], [ftrue], marker="+", ms=8, mew=1.4, color=fs.TRUTH,
                 zorder=6, clip_on=False)
        # direct label rather than a legend box: two regions, far apart
        pf = np.nansum(np.exp(L - np.nanmax(L)), axis=0)
        ph0 = np.nansum(np.exp(L - np.nanmax(L)), axis=1)
        fs.label_line(axa, H[np.argmax(ph0)], F[np.argmax(pf)],
                      rf"planted $f_{{\rm AGN}}={ftrue:.3f}$", col,
                      dy=13, ha="center", va="bottom")

        # marginal in H0
        ph = np.nansum(np.exp(L - np.nanmax(L)), axis=1)
        ph /= np.trapz(ph, H)
        axb.plot(H, ph, color=col, lw=1.6,
                 label=rf"$f_{{\rm AGN}}={ftrue:.3f}$")
        axb.fill_between(H, 0, ph, color=col, alpha=0.14, lw=0)

    axa.set_xlabel(r"$H_0$  (km s$^{-1}$ Mpc$^{-1}$)")
    axa.set_ylabel(r"$f_{\rm AGN}$")
    axa.set_title("(a)  joint credible regions, 68% and 90%")
    axa.set_xlim(63.2, 69.0)
    axa.set_ylim(0.25, 0.80)
    fs.truth_line(axa, H0_TRUTH, axis="x", label="truth")

    axb.set_xlabel(r"$H_0$  (km s$^{-1}$ Mpc$^{-1}$)")
    axb.set_ylabel("marginal posterior density")
    axb.set_title("(b)  the distance-scale deficit")
    axb.set_xlim(62.5, 68.6)
    axb.set_ylim(bottom=0)
    fs.truth_line(axb, H0_TRUTH, axis="x", label="truth")
    axb.legend(loc="upper left")
    axb.set_yticks([])

    fs.save(fig, "fig_joint_h0f")


if __name__ == "__main__":
    main()
