#!/usr/bin/env python3
"""Two-tracer flux-limit ladder at fixed data.

Panel (a): how the joint 68% half-widths on the distance scale and on the
AGN-hosted fraction change as the host survey is thinned, as factors against
the complete-catalog rung.  Panel (b): the measured fraction against the
sky-shuffled null, which breaks the pairing between each event's distance and
its own host's redshift while preserving sky patches, distances and
localisation areas.  Panel (c): the separation between the two, which is the
degradation statistic for the AGN identification itself.

Reads experiment_twotracer_incomplete/results/{summary_fix.json,fscan_*_fix.h5}
(the post-fix ladder: repaired generator, same events and catalogs).
"""
from __future__ import annotations

import json

import h5py
import numpy as np
import matplotlib.pyplot as plt

import figstyle as fs

RES = fs.EXP / "experiment_twotracer_incomplete" / "results"
ORDER = ["complete", "m21.0", "m20.0", "m19.0", "m18.0"]
NULLS = ["complete", "m20.0", "m18.0"]


def scan(tag):
    with h5py.File(RES / f"fscan_{tag}_fix.h5", "r") as f:
        return f["f_grid"][:], f["log_likelihood"][:]


def main():
    fs.use()
    S = json.loads((RES / "summary_fix.json").read_text())
    lv = S["levels"]
    zref = S["z_ref"]
    C = np.array([lv[k]["agn_completeness_within_horizon"] for k in ORDER])

    fig, (axa, axb, axc) = plt.subplots(1, 3, figsize=(fs.TWOCOL, 2.35),
                                        gridspec_kw={"wspace": 0.36})

    # ---- (a) width factors ----------------------------------------------
    fH = np.array([lv[k]["width_degradation_vs_complete"]["joint_H0"] for k in ORDER])
    fF = np.array([lv[k]["width_degradation_vs_complete"]["joint_f"] for k in ORDER])
    axa.plot(C, fH, "-o", color=fs.C["blue"], ms=4.5, mfc=fs.C["blue"],
             mec="#fcfcfb", mew=0.8, zorder=4, label=r"$\sigma(H_0)$")
    axa.plot(C, fF, "-s", color=fs.C["orange"], ms=4.5, mfc=fs.C["orange"],
             mec="#fcfcfb", mew=0.8, zorder=4, label=r"$\sigma(f_{\rm AGN})$")
    fs.label_line(axa, C[-1], fH[-1], f"{fH[-1]:.2f}$\\times$", fs.C["blue"],
                  dx=2, dy=7, ha="center", va="bottom")
    fs.label_line(axa, C[-1], fF[-1], f"{fF[-1]:.2f}$\\times$", fs.C["orange"],
                  dx=-5, dy=7, ha="right", va="bottom")
    axa.axhline(1.0, color=fs.TRUTH, lw=0.9, ls=(0, (3, 2)), alpha=0.75, zorder=1.5)
    axa.invert_xaxis()
    axa.set_xlabel(r"completeness $C(z\leq%.1f)$" % zref)
    axa.set_ylabel("width / complete rung")
    axa.set_title("(a)  the fraction barely notices")
    axa.set_ylim(0.7, 2.05)
    axa.legend(loc="upper left")

    # ---- (b) the null ---------------------------------------------------
    g, L = scan("complete")
    p = fs.posterior_1d(g, L)
    gn, Ln = scan("null_complete")
    pn = fs.posterior_1d(gn, Ln)
    axb.plot(g, p, color=fs.C["blue"], lw=1.8, zorder=4, label="measured")
    axb.fill_between(g, 0, p, color=fs.C["blue"], alpha=0.14, lw=0)
    axb.plot(gn, pn, color=fs.MUTED, lw=1.6, ls=(0, (4, 2)), zorder=3,
             label="sky-shuffled null")
    axb.fill_between(gn, 0, pn, color=fs.MUTED, alpha=0.10, lw=0)
    nl = lv["complete"]["sky_shuffle_null"]
    ymax = max(p.max(), pn.max())
    axb.annotate("", xy=(nl["median"], 1.07 * ymax),
                 xytext=(lv["complete"]["fscan"]["median"], 1.07 * ymax),
                 arrowprops=dict(arrowstyle="<->", lw=0.7, color=fs.INK2))
    axb.annotate(f"{nl['displacement_in_widths']:.1f} widths",
                 (0.5 * (nl["median"] + lv["complete"]["fscan"]["median"]),
                  1.10 * ymax), ha="center", va="bottom", fontsize=7.0,
                 color=fs.INK2)
    fs.truth_line(axb, S["truth"]["f_AGN"], axis="x", label="planted", pos=0.60)
    axb.set_xlabel(r"$f_{\rm AGN}$")
    axb.set_ylabel("posterior density")
    axb.set_title("(b)  what the null looks like")
    axb.set_xlim(0, 0.72)
    axb.set_ylim(0, 1.62 * ymax)
    axb.set_yticks([])
    axb.legend(loc="upper right", bbox_to_anchor=(1.0, 1.0))

    # ---- (c) separation vs completeness ---------------------------------
    Cn = np.array([lv[k]["agn_completeness_within_horizon"] for k in NULLS])
    sep = np.array([lv[k]["sky_shuffle_null"]["displacement_in_widths"] for k in NULLS])
    axc.vlines(Cn, 0, sep, color=fs.C["aqua"], lw=1.4, alpha=0.45, zorder=3)
    axc.plot(Cn, sep, "o", color=fs.C["aqua"], ms=5.5, mfc=fs.C["aqua"],
             mec="#fcfcfb", mew=0.9, ls="none", zorder=4)
    for c, s in zip(Cn, sep):
        axc.annotate(f"{s:.2f}", (c, s), xytext=(0, 6),
                     textcoords="offset points", ha="center", fontsize=7.0,
                     color=fs.INK2)
    axc.invert_xaxis()
    axc.set_xlabel(r"completeness $C(z\leq%.1f)$" % zref)
    axc.set_ylabel("peak-to-null separation (widths)")
    axc.set_title("(c)  the honest degradation")
    axc.set_ylim(0, 1.28 * sep.max())

    fs.save(fig, "fig_completeness_twotracer")


if __name__ == "__main__":
    main()
