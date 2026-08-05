"""Both parameters against truth, realisation by realisation.

    analyses/analysis_2_complete_catalog_H0_fagn/results/joint_summary.json
    analyses/analysis_2_complete_catalog_H0_fagn/results/h0_fagn_joint.json

Each point is one end-to-end realisation of the mock -- its own density field,
its own catalogs, its own 1000 events -- analysed with the same joint fit; the
bar is that realisation's own 68 % interval, read from the summary json
(`seeds[].joint.H0` and `seeds[].joint.f_vs_realised`).  The band is the mean
offset over the five realisations plus/minus its standard error, from
`closure.joint_H0` and `closure.joint_f_vs_realised`; the same two numbers are
quoted in h0_fagn_joint.json and in the text.

The right panel's reference is not a single line: each realisation's mock drew
its own AGN host count, so the value the fit should return is that
realisation's realised fraction (the tick), and the input 0.30 is drawn as
one dotted line behind them for scale.  The mean offset is quoted against the
realised fractions, which is what the closure statement is about.
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import figstyle as fs

ACC = fs.C["blue"]


def build():
    fs.use()
    summ = json.loads((fs.A2 / "joint_summary.json").read_text())
    rows = [r for r in summ["seeds"] if "joint" in r]
    x = np.arange(len(rows))

    fig, axes = plt.subplots(1, 2, figsize=(fs.TWOCOL, 2.5))
    for ax in axes:
        ax.grid(axis="x", visible=False)
        ax.set_xticks(x)
        ax.set_xticklabels([str(r["seed"]) for r in rows])
        # exactly the span the step-drawn mean band covers, so it meets both spines
        ax.set_xlim(-0.5, len(rows) - 0.5)
        ax.set_xlabel("realisation")

    def panel(ax, key, ref, closure, fmt, centre=True):
        """One panel: per-realisation medians, and the five-seed mean offset.

        `ref` is the value the fit should return for each realisation -- one
        number repeated for H0, the realisation's own realised fraction for f
        -- so the mean-offset band tracks it rather than floating free.
        """
        med = np.array([r["joint"][key]["median"] for r in rows])
        lo = med - np.array([r["joint"][key]["ci68"][0] for r in rows])
        hi = np.array([r["joint"][key]["ci68"][1] for r in rows]) - med
        c = summ["closure"][closure]
        ref = np.asarray(ref, float)
        ax.fill_between(x, ref + c["mean"] - c["sem"], ref + c["mean"] + c["sem"],
                        step="mid", color=ACC, alpha=0.11, lw=0, zorder=2)
        if centre:                    # a straight mean line only where the
            ax.step(x, ref + c["mean"], where="mid", color=ACC, lw=0.9,
                    zorder=3)         # reference is one number for every seed
        ax.errorbar(x, med, yerr=[lo, hi], fmt="o", ms=4.0, color=ACC,
                    ecolor=ACC, elinewidth=1.3, capsize=0, zorder=5,
                    markeredgecolor="white", markeredgewidth=0.7)
        ax.annotate(f"mean offset  {c['mean']:{fmt}}  $\\pm$  {abs(c['sem']):{fmt[1:]}}",
                    (0.5, 0.02), xycoords="axes fraction", ha="center",
                    va="bottom", fontsize=7.2, color=fs.INK2)
        return med

    # ---- H0: one truth for every realisation --------------------------------
    ax = axes[0]
    fs.truth_line(ax, fs.H0_TRUTH, axis="y")
    panel(ax, "H0", np.full(len(rows), fs.H0_TRUTH), "joint_H0", "+.2f")
    ax.annotate("truth 67.74", (0.015, fs.H0_TRUTH),
                xycoords=("axes fraction", "data"),
                textcoords="offset points", xytext=(0, -3), ha="left",
                va="top", fontsize=7.0, color=fs.INK2)
    ax.set_ylabel(r"$H_0$  [km s$^{-1}$ Mpc$^{-1}$]")
    ax.margins(y=0.20)

    # ---- f: one truth per realisation ---------------------------------------
    ax = axes[1]
    frs = np.array([r["f_realised"] for r in rows])
    ax.axhline(fs.F_PLANTED, color=fs.MUTED, lw=0.8, ls=(0, (1, 2)), zorder=3)
    ax.plot(x, frs, marker="_", ms=13, mew=1.5, ls="none", color=fs.INK,
            zorder=6)
    panel(ax, "f_vs_realised", frs, "joint_f_vs_realised", "+.3f",
          centre=False)
    ax.set_ylabel(r"$f_{\rm AGN}$")
    ax.margins(y=0.20)
    ax.legend(handles=[
        Line2D([], [], color=ACC, lw=1.3, marker="o", ms=4.0,
               markeredgecolor="white", markeredgewidth=0.7,
               label=r"joint fit, median $\pm$ 68 %"),
        Line2D([], [], color=fs.INK, lw=0, marker="_", ms=11, mew=1.5,
               label="realised fraction"),
        Line2D([], [], color=fs.MUTED, lw=0.8, ls=(0, (1, 2)),
               label="input 0.30"),
    ], loc="upper left", fontsize=7.0, handlelength=1.4, labelspacing=0.35,
        borderaxespad=0.3)

    fig.tight_layout(pad=0.3, w_pad=1.4)
    return fig


def main():
    fig = build()
    fs.save(fig, "fig_closure")
    plt.close(fig)


if __name__ == "__main__":
    main()
