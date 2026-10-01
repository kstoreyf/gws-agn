#!/usr/bin/env python3
"""Figures for experiment_completeness_free.

  figs/fig_n0_degeneracy.{pdf,png}  -- the (f_AGN, n0_AGN) plane across the ladder
  figs/fig_n0_arms.{pdf,png}        -- what knowing n0 to X% buys you

The question is not whether f_AGN can be measured when the density is known --
../experiment_twotracer_incomplete answered that -- but how much of the answer
was the assumption.
"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent.parent
RESULTS, FIGS = BASE / "results", BASE / "figs"
FIGS.mkdir(parents=True, exist_ok=True)
INC = BASE.parent / "experiment_twotracer_incomplete"

SFX = ""
LEVELS = ["complete", "m21.0", "m20.0", "m19.0", "m18.0"]
LABELS = {"complete": "complete", "m21.0": "$m<21$", "m20.0": "$m<20$",
          "m19.0": "$m<19$", "m18.0": "$m<18$"}
G_TRUE, TRUTH_F = -7.720033, 0.30
ARMS = ["fixed", "5%", "10%", "30%", "factor 2", "free"]
ARM_LAB = {"fixed": r"$n_0$ known exactly", "5%": r"$n_0$ to 5%",
           "10%": r"$n_0$ to 10%", "30%": r"$n_0$ to 30%",
           "factor 2": r"$n_0$ to a factor 2", "free": r"$n_0$ free"}

BLUE, AQUA, YELLOW, RED, INK, INK2, INK3 = ("#2a78d6", "#1baf7a", "#eda100",
                                            "#e34948", "#0b0b0b", "#52514e", "#898781")
GRIDCOL = "#e1e0d9"
RAMP = ["#0b0b0b", "#2a78d6", "#1baf7a", "#eda100", "#e34948"]
ARM_COL = {"fixed": "#0b0b0b", "5%": "#2a78d6", "10%": "#1baf7a",
           "30%": "#eda100", "factor 2": "#e34948", "free": "#898781"}
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "font.family": "serif",
    "font.serif": ["DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 9, "axes.labelsize": 9.5, "axes.titlesize": 9.2,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 7.4,
    "axes.edgecolor": INK2, "axes.linewidth": 0.7, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "grid.color": GRIDCOL, "grid.linewidth": 0.6,
    "legend.frameon": False, "lines.solid_capstyle": "round",
})


def load_grid(lev):
    with h5py.File(RESULTS / f"fn0_{lev}{SFX}.h5", "r") as f:
        return f["f_grid"][:], f["n0c2_grid"][:], f["log_likelihood"][:]


def hpd_levels(pw, fracs=(0.68, 0.90)):
    w = pw.ravel()
    o = np.argsort(w)[::-1]
    cs = np.cumsum(w[o])
    return [float(w[o[min(np.searchsorted(cs, fr), o.size - 1)]]) for fr in fracs]


def completeness():
    p = INC / "results/summary.json"
    if not p.exists():
        return {}
    d = json.loads(p.read_text())["completeness"]
    return {k: v["agn"]["completeness_within_horizon"] for k, v in d.items()}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="", help="_fix reads *_fix results and writes *_fix figures")
    global SFX
    SFX = ap.parse_args().suffix
    S = json.loads((RESULTS / f"n0_arms_summary{SFX}.json").read_text())
    C = completeness()
    have = [l for l in LEVELS if (RESULTS / f"fn0_{l}{SFX}.h5").exists()]

    # ------------------------------------------------------------ degeneracy
    fig, axes = plt.subplots(1, len(have), figsize=(2.05 * len(have), 2.9), dpi=300,
                             sharey=True)
    axes = np.atleast_1d(axes)
    for ax, lev, col in zip(axes, have, RAMP):
        fv, gv, ll = load_grid(lev)
        ok = np.isfinite(ll)
        L = np.where(ok, np.exp(ll - np.nanmax(ll[ok])), 0.0)
        pw = L * np.outer(np.gradient(fv), np.gradient(gv))
        pw = pw / pw.sum()
        Fg, Gg = np.meshgrid(fv, gv, indexing="ij")
        lv = hpd_levels(pw)
        ax.contourf(Fg, Gg, pw, levels=[lv[1], lv[0], pw.max()],
                    colors=[col, col], alpha=0.22, zorder=3)
        ax.contour(Fg, Gg, pw, levels=sorted(lv), colors=[col], linewidths=1.0,
                   zorder=4)
        ax.axhline(G_TRUE, color=INK2, lw=0.7, ls=(0, (1, 2.5)), zorder=2)
        ax.plot([TRUTH_F], [G_TRUE], marker="*", ms=9, color=YELLOW, mec=INK,
                mew=0.5, ls="none", zorder=6)
        cw = C.get(lev)
        ax.set_title(LABELS[lev] + (f"\n$C\\simeq{cw:.2f}$" if cw else ""),
                     fontsize=8.6)
        ax.set_xlim(0, 1)
        ax.set_ylim(gv[0], gv[-1])
        ax.set_xticks([0, 0.5, 1.0])
        ax.set_xlabel(r"$f_{\rm AGN}$")
        ax.grid(True, alpha=0.4)
        ax.set_axisbelow(True)
    axes[0].set_ylabel(r"$\log_{10} n_{0,\rm AGN}$")
    fig.suptitle(r"The $(f_{\rm AGN},\, n_{0,\rm AGN})$ degeneracy "
                 "(star = truth, dotted = true density)", fontsize=9)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"fig_n0_degeneracy{SFX}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("wrote figs/fig_n0_degeneracy.{pdf,png}")

    # ----------------------------------------------------------------- arms
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.6, 3.2), dpi=300)
    xs = [C.get(l, np.nan) for l in have]
    for arm in ARMS:
        y = [S["levels"][l]["arms"][arm]["detection_sigma"]
             if S["levels"].get(l, {}).get("arms", {}).get(arm) else None
             for l in have]
        pts = [(a, b) for a, b in zip(xs, y) if b is not None]
        if pts:
            axL.plot(*zip(*pts), color=ARM_COL[arm], lw=1.6, marker="o", ms=3.6,
                     zorder=4, label=ARM_LAB[arm])
    axL.axhline(3.0, color=INK3, lw=0.9, ls=(0, (1, 2)), zorder=2)
    axL.annotate(r"$3\sigma$", xy=(0.13, 3.05), fontsize=7.2, color=INK2,
                 va="bottom")
    axL.set_xlim(1.03, 0.10)
    axL.set_ylim(bottom=0)
    axL.set_xlabel(r"completeness within the horizon  $C(z\leq0.30)$")
    axL.set_ylabel(r"detection significance of $f_{\rm AGN}$   (median$/\sigma$)")
    axL.set_title("What knowing the AGN density buys")
    axL.grid(True, alpha=0.55)
    axL.set_axisbelow(True)
    axL.legend(loc="lower left", ncol=2)

    for k, (lev, col) in enumerate(zip(have, RAMP)):
        r = S["levels"][lev]["arms"]
        xi = np.arange(len(ARMS), dtype=float)
        m = [i for i, a in enumerate(ARMS) if r.get(a)]
        cw = C.get(lev)
        axR.errorbar(xi[m] + 0.09 * (k - (len(have) - 1) / 2),
                     [r[ARMS[i]]["median"] for i in m],
                     yerr=[r[ARMS[i]]["half_width68"] for i in m],
                     fmt="o", ms=3.2, lw=0, elinewidth=1.0, capsize=2.0,
                     color=col, ecolor=col, zorder=4,
                     label=LABELS[lev] + (f" ($C\\simeq{cw:.2f}$)" if cw else ""))
    axR.axhline(TRUTH_F, color=INK2, lw=1.0, ls=(0, (1, 2)), zorder=2)
    axR.annotate("planted", xy=(0.02, TRUTH_F + 0.008), fontsize=7.2,
                 color=INK2, va="bottom", ha="left")
    axR.set_xticks(np.arange(len(ARMS)))
    axR.set_xticklabels(["exact", "5%", "10%", "30%", r"$\times2$", "free"])
    axR.set_xlim(-0.5, len(ARMS) - 0.5)
    axR.set_xlabel(r"prior knowledge of $n_{0,\rm AGN}$")
    axR.set_ylabel(r"$f_{\rm AGN}$")
    axR.set_title("The recovered value moves with the assumption")
    axR.grid(True, alpha=0.55, axis="y")
    axR.set_axisbelow(True)
    axR.legend(loc="lower left", ncol=2)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"fig_n0_arms{SFX}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("wrote figs/fig_n0_arms.{pdf,png}")


if __name__ == "__main__":
    main()
