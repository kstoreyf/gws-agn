#!/usr/bin/env python3
"""The K=1 distance-scale budget as a waterfall: two generator defects, the
estimator overhead, and the closure endpoint.

Reads experiment_matched_mock/results/{obsdet_summary,oracle_summary}.json.
The steps chain the campaign's measured decomposition:

  as generated (ctrl arm)          -1.57 +- 0.18
  detection onto observed data     +0.77 (paired A/B)  -> obs arm -0.80 +- 0.16
  sky width from observables       +0.49 (exact-likelihood attribution)
  estimator overhead               -0.31 +- 0.13 (darksirens - exact)
  closure: exact likelihood on repaired-recipe events  -0.06 +- 0.07
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt

import figstyle as fs

RES = fs.EXP / "experiment_matched_mock" / "results"


def main():
    fs.use()
    O = json.loads((RES / "obsdet_summary.json").read_text())
    R = json.loads((RES / "oracle_summary.json").read_text())

    ctrl = O["arms"]["ctrl"]["offset_stats"]           # as generated
    pd = O["paired_difference_obs_minus_ctrl"]         # detection defect share
    ex = R["offset_exact_oracle"]                      # sky-width defect share
    pr = R["paired_ds_minus_oracle"]                   # estimator overhead
    fx = R["bootstrap_fix"]["offset"]                  # closure endpoint

    # components chain from zero and sum to the measured total
    steps = [
        ("detection acting\non latent data", -pd["mean"], pd["sem"], fs.RAMP[2]),
        ("sky width drawn from\nlatent parameters", ex["mean"], ex["sem"],
         fs.RAMP[2]),
        ("estimator overhead\n(darksirens $-$ exact)", pr["mean"], pr["sem"],
         fs.C["orange"]),
    ]

    fig, ax = plt.subplots(figsize=(fs.ONECOL, 2.6))
    ntot = len(steps) + 2
    left = 0.0
    for i, (lab, val, err, col) in enumerate(steps):
        y = ntot - i
        ax.barh(y, val, left=left, height=0.52, color=col, zorder=3)
        ax.annotate(f"{val:+.2f}", (left + val / 2, y), xytext=(0, 8),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=7.0, color=fs.INK2, zorder=5)
        left += val
        # connector down to the next row
        ax.plot([left, left], [y - 1 + 0.26, y - 0.26], color=fs.AXIS, lw=0.7,
                zorder=2)

    # the measured total, with its own uncertainty
    y = 2
    ax.barh(y, ctrl["mean"], height=0.52, color=fs.RAMP[4], zorder=3)
    ax.errorbar([ctrl["mean"]], [y], xerr=[ctrl["sem"]], fmt="none",
                ecolor=fs.INK, elinewidth=1.0, capsize=2.0, zorder=4)
    ax.annotate(f"{ctrl['mean']:+.2f}", (ctrl["mean"] / 2, y), xytext=(0, 8),
                textcoords="offset points", ha="center", va="bottom",
                fontsize=7.0, color="#fcfcfb", zorder=5)

    # regenerated campaign: production estimator on repaired-generator mocks;
    # lands on the estimator-overhead landing point if the attribution is right
    fj = RES / "obsdet_fix_summary.json"
    if fj.exists():
        F = json.loads(fj.read_text())["arms"]["fix"]["offset_stats"]
        yov = ntot - 2               # the estimator-overhead row
        ax.errorbar([F["mean"]], [yov - 0.42], xerr=[F["sem"]], fmt="o",
                    ms=4.2, color=fs.INK, mfc="#fcfcfb", mec=fs.INK, mew=1.0,
                    elinewidth=0.9, capsize=1.8, zorder=5)
        ax.annotate("repaired mocks,\nproduction estimator",
                    (F["mean"] - F["sem"], yov - 0.42), xytext=(-5, 0),
                    textcoords="offset points", ha="right", va="center",
                    fontsize=6.2, color=fs.INK2)

    # closure endpoint: exact likelihood on repaired-recipe events
    y0 = 1
    ax.errorbar([fx["mean"]], [y0], xerr=[fx["sem"]], fmt="D", ms=5.0,
                color=fs.INK, mfc="#fcfcfb", mec=fs.INK, mew=1.1,
                elinewidth=1.0, capsize=2.0, zorder=5)
    ax.annotate(f"{fx['mean']:+.2f}", (fx["mean"], y0), xytext=(-8, 0),
                textcoords="offset points", ha="right", va="center",
                fontsize=7.0, color=fs.INK2)

    ax.axvline(0.0, color=fs.TRUTH, lw=0.9, ls=(0, (3, 2)), alpha=0.75,
               zorder=1.5)
    ax.annotate("truth", (0.0, ntot + 0.62), xytext=(3, 0),
                textcoords="offset points", ha="left", va="bottom",
                fontsize=6.8, color=fs.INK2, annotation_clip=False)
    labels = [s[0] for s in steps] + ["total: mock\nas generated",
                                      "closure: exact likelihood,\nrepaired recipe"]
    ax.set_yticks(list(range(ntot, ntot - len(steps), -1)) + [2, 1])
    ax.set_yticklabels(labels, fontsize=7.0)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(0.35, ntot + 1.05)
    ax.set_xlim(-2.05, 0.30)
    ax.set_xlabel(r"$\Delta H_0$  (km s$^{-1}$ Mpc$^{-1}$)")
    ax.grid(axis="y", visible=False)

    fs.save(fig, "fig_closure_waterfall")


if __name__ == "__main__":
    main()
