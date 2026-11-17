#!/usr/bin/env python3
"""Why a sparse tracer needs its own selection proposal.

Panel (a): the effective sample size of the catalog-conditioned selection
integral against the AGN-hosted fraction, for a proposal drawn from the source
population and for one that also places a quarter of its draws on catalogued
AGN.  The population proposal decays as the mixture leans on the sparse tracer;
the targeted one climbs.

Panel (b): what that does to the inference.  With the same events and only the
proposal changed, the AGN-hosted fraction's posterior moves off the edge of the
admissible region and lands below the planted value.

Reads experiment_twotracer_deep/results/{summary.json,tgt_fscan_n80.h5,
deep_fscan_n80.h5}.
"""
from __future__ import annotations

import json

import h5py
import numpy as np
import matplotlib.pyplot as plt

import figstyle as fs

RES = fs.EXP / "experiment_twotracer_deep" / "results"


def scan(tag):
    with h5py.File(RES / f"{tag}.h5", "r") as f:
        return f["f_grid"][:], f["log_likelihood"][:]


def main():
    fs.use()
    S = json.loads((RES / "summary.json").read_text())
    nobs = S["meta"]["nobs"]
    tbl = S[f"neff_vs_f_at_N{nobs}"]
    floor = tbl["popuni"][0]["threshold"]

    fig, (axa, axb) = plt.subplots(
        1, 2, figsize=(fs.TWOCOL, 2.5),
        gridspec_kw={"width_ratios": [1.0, 1.12], "wspace": 0.28})

    # ---- (a) the selection integral's resolution -------------------------
    for key, lab, col, mk in (("popuni", "from the source population", fs.C["blue"], "o"),
                              ("targeted", "with a catalogue-targeted branch",
                               fs.C["orange"], "s")):
        f = np.array([r["f"] for r in tbl[key]])
        n = np.array([r["Neff"] for r in tbl[key]])
        axa.plot(f, n, mk + "-", color=col, ms=4.5, mfc=col, mec="#fcfcfb",
                 mew=0.8, zorder=4, label=lab)
    axa.axhline(floor, color=fs.BAD, lw=1.1, ls=(0, (4, 2)), zorder=2)
    axa.annotate("validity floor", (0.02, floor), xytext=(0, -4),
                 textcoords="offset points", ha="left", va="top", fontsize=6.8,
                 color=fs.BAD)
    axa.set_yscale("log")
    axa.set_xlabel(r"$f_{\rm AGN}$ at which the integral is evaluated")
    axa.set_ylabel(r"$N_{\rm eff}$ of the selection integral")
    axa.set_title("(a)  a proposal that follows the tracer")
    axa.set_xlim(-0.04, 1.04)
    axa.legend(loc="lower left", title="injections drawn", title_fontsize=7.0)
    axa.get_legend().get_title().set_color(fs.INK2)

    # ---- (b) the posterior it was hiding ---------------------------------
    for tag, key, lab, col in (("deep_fscan_n80", "deep_popuni",
                                "from the source population", fs.C["blue"]),
                               ("tgt_fscan_n80", "deep_targeted",
                                "with a catalogue-targeted branch", fs.C["orange"])):
        g, L = scan(tag)
        p = fs.posterior_1d(g, L)
        ok = np.isfinite(L)
        axb.plot(g[ok], p[ok], color=col, lw=1.8, zorder=4, label=lab)
        axb.fill_between(g[ok], 0, p[ok], color=col, alpha=0.14, lw=0)
        if (~ok).any():
            edge = g[ok].max()
            axb.axvspan(edge, g.max(), color=fs.BAD, alpha=0.09, lw=0, zorder=1)
            axb.annotate("inadmissible", (edge, 0.55), xycoords=("data",
                         "axes fraction"), xytext=(4, 0),
                         textcoords="offset points", ha="left", va="center",
                         fontsize=6.8, color=fs.BAD, rotation=90)
    fs.truth_line(axb, S["meta"]["truth_f_agn"], axis="x", label="planted",
                  pos=0.62)
    axb.set_xlabel(r"$f_{\rm AGN}$")
    axb.set_ylabel("posterior density")
    axb.set_title("(b)  same events, different proposal")
    axb.set_xlim(0.0, 0.62)
    axb.set_ylim(0, None)
    axb.set_yticks([])
    axb.legend(loc="upper left")

    fs.save(fig, "fig_selection_lane")


if __name__ == "__main__":
    main()
