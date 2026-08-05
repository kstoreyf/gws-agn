#!/usr/bin/env python3
"""Error budget of the single-tracer closure test.

Panel (a): the distance-scale offset against the fractional distance
uncertainty, for posterior samples built about the true parameters and about a
noisy measurement, with a quadratic reference.  The offset scales with the noise
level, and correcting the posterior construction does not remove it.

Panel (b): the paired detection-rule experiment over 20 host-catalog and event
realisations.  Both arms share the catalog, the event seed and every ancillary
uncertainty model, so the difference between them is the detection rule alone.

Reads experiment_matched_mock/results/{summary,obsdet_summary}.json.
"""
from __future__ import annotations

import json

import numpy as np
import matplotlib.pyplot as plt

import figstyle as fs

RES = fs.EXP / "experiment_matched_mock" / "results"
H0_TRUTH = 67.74
SIGMAS = ["0.01", "0.03", "0.1"]


def main():
    fs.use()
    S = json.loads((RES / "summary.json").read_text())
    O = json.loads((RES / "obsdet_summary.json").read_text())

    fig, (axa, axb) = plt.subplots(
        1, 2, figsize=(fs.TWOCOL, 2.6),
        gridspec_kw={"width_ratios": [1.0, 1.25], "wspace": 0.28})

    # ---- (a) noise-level scaling -----------------------------------------
    arms = [("truth_centred", "about the true parameters", fs.C["blue"], "o"),
            ("corrected", "about a noisy measurement", fs.C["orange"], "s")]
    for key, lab, col, mk in arms:
        x = np.array([float(s) for s in SIGMAS])
        y = np.array([S["sigma_ladder"][key][s]["median"] - H0_TRUTH for s in SIGMAS])
        e = np.array([S["sigma_ladder"][key][s]["hw"] for s in SIGMAS])
        axa.errorbar(x, y, yerr=e, fmt=mk + "-", ms=4.5, color=col, ecolor=col,
                     elinewidth=1.3, mfc=col, mec="#fcfcfb", mew=0.8, label=lab,
                     zorder=4)
    # quadratic reference anchored on the corrected arm's deepest point
    xr = np.geomspace(0.01, 0.115, 60)
    anchor = S["sigma_ladder"]["corrected"]["0.1"]["median"] - H0_TRUTH
    axa.plot(xr, anchor * (xr / 0.10) ** 2, color=fs.MUTED, lw=0.9, ls=(0, (3, 2)),
             zorder=2)
    fs.label_line(axa, 0.055, anchor * (0.055 / 0.10) ** 2, r"$\propto\sigma^2$",
                  fs.MUTED, dy=-9, ha="center", va="top")
    axa.axhline(0.0, color=fs.TRUTH, lw=0.9, ls=(0, (3, 2)), alpha=0.75, zorder=1.5)
    axa.set_xscale("log")
    axa.set_xlabel(r"fractional distance uncertainty $\sigma_{d_L}$")
    axa.set_ylabel(r"$\Delta H_0$  (km s$^{-1}$ Mpc$^{-1}$)")
    axa.set_title("(a)  the offset scales with the noise")
    axa.set_xticks([0.01, 0.03, 0.10])
    axa.set_xticklabels(["0.01", "0.03", "0.10"])
    axa.set_xlim(0.008, 0.14)
    axa.legend(loc="lower left", title="posterior samples drawn",
               title_fontsize=7.0)
    axa.get_legend().get_title().set_color(fs.INK2)

    # ---- (b) the paired detection-rule experiment ------------------------
    ctrl = {r["tag"]: r["offset"] for r in O["arms"]["ctrl"]["per_seed"]}
    obs = {r["tag"]: r["offset"] for r in O["arms"]["obs"]["per_seed"]}
    tags = [t for t in ctrl if t in obs]
    xc, xo = 0.0, 1.0
    for t in tags:
        axb.plot([xc, xo], [ctrl[t], obs[t]], color=fs.MUTED, lw=0.6, alpha=0.55,
                 zorder=2)
    axb.plot([xc] * len(tags), [ctrl[t] for t in tags], "o", ms=3.6,
             color=fs.C["blue"], mfc=fs.C["blue"], mec="#fcfcfb", mew=0.6, zorder=3)
    axb.plot([xo] * len(tags), [obs[t] for t in tags], "s", ms=3.6,
             color=fs.C["orange"], mfc=fs.C["orange"], mec="#fcfcfb", mew=0.6,
             zorder=3)
    for x, key, col in ((xc, "ctrl", fs.C["blue"]), (xo, "obs", fs.C["orange"])):
        st = O["arms"][key]["offset_stats"]
        axb.errorbar([x + 0.22], [st["mean"]], yerr=[st["sem"]], fmt="D", ms=5.5,
                     color=col, ecolor=col, elinewidth=2.0, mfc=col,
                     mec="#fcfcfb", mew=0.9, zorder=5)
        fs.label_line(axb, x + 0.22, st["mean"],
                      f"{st['mean']:+.2f} $\\pm$ {st['sem']:.2f}", col,
                      dx=8, ha="left", va="center", size=7.0)
    fs.truth_line(axb, 0.0, axis="y", label="no offset")
    axb.set_xticks([xc + 0.11, xo + 0.11])
    axb.set_xticklabels(["detection on a\nlatent projection",
                         "detection on the\nobserved data"])
    axb.tick_params(axis="x", length=0)
    axb.set_xlim(-0.28, 1.72)
    axb.set_ylabel(r"$\Delta H_0$  (km s$^{-1}$ Mpc$^{-1}$)")
    axb.set_title("(b)  paired over 20 realisations")
    axb.grid(axis="x", visible=False)

    fs.save(fig, "fig_bias_budget")


if __name__ == "__main__":
    main()
