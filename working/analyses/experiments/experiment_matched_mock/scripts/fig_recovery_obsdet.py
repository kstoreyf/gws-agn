#!/usr/bin/env python3
"""Explicit H0-recovery figures for the K=1 three-arm closure (obsdet family).

Reads ONLY this experiment's results files:

  results/obsdet_{ctrl,obs,fix}_<tag>.h5    H0_grid + log_likelihood per realisation
  results/obsdet_{ctrl,obs,fix}_<tag>.json  median / ci68 per realisation

and writes:

  figs/fig_h0_recovery_arms.{pdf,png}     one panel per arm, all 20 normalised H0
                                          posteriors overlaid, truth line, arm
                                          mean-of-medians marked; shared x range
  figs/fig_h0_recovery_medians.{pdf,png}  per-realisation median +-68% strip, three
                                          arms side by side, ensemble mean bands

Arms (paired by realisation; see analyze_obsdet_fix.py):
  ctrl  detection on true params (gmd original rule)         -- pre-fix
  obs   detection on observed data (PR #334), latent width   -- pre-fix
  fix   + observed sky width (PR #335)                       -- post-fix,
                                                                measurement of record

Pure post-processing: h5py/numpy/matplotlib, CPU, no darksirens import.
Posterior = exp(logL - max) normalised by the trapezoid rule (flat prior);
non-finite (guard-rejected) cells carry zero posterior mass.
"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BASE = Path(__file__).resolve().parent.parent
RESULTS, FIGS = BASE / "results", BASE / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

H0_TRUTH = 67.74
TAGS = (("b", "s4102", "s4103", "s4104", "s4105")
        + tuple(f"n{s}" for s in range(4201, 4216)))
# Same colours as fig_obsfix_closure (colour follows the arm across figures);
# markers give each arm a non-colour identity in the strip figure.
ARMS = (("ctrl", "obsdet_ctrl", "detection on true params (gmd rule)", "#e34948", "o"),
        ("obs", "obsdet_obs", "detection on observed data (PR #334)", "#2a78d6", "s"),
        ("fix", "obsdet_fix", "+ observed sky width (PR #335)", "#1baf7a", "D"))

INK, INK2, INK3 = "#0b0b0b", "#52514e", "#898781"
GRIDCOL = "#e1e0d9"
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "font.family": "serif",
    "font.serif": ["DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 9, "axes.labelsize": 9.5, "axes.titlesize": 9.3,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 7.8,
    "axes.edgecolor": INK2, "axes.linewidth": 0.7, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "grid.color": GRIDCOL, "grid.linewidth": 0.6,
    "legend.frameon": False, "lines.solid_capstyle": "round",
})


def load_arm(prefix):
    """Per-realisation posteriors (from .h5) and interval summaries (from .json)."""
    out = []
    for tag in TAGS:
        with h5py.File(RESULTS / f"{prefix}_{tag}.h5", "r") as f:
            x, ll = f["H0_grid"][:], f["log_likelihood"][:]
        ok = np.isfinite(ll)
        p = np.zeros_like(ll)
        p[ok] = np.exp(ll[ok] - ll[ok].max())
        p /= np.trapz(p, x)
        j = json.loads((RESULTS / f"{prefix}_{tag}.json").read_text())
        h = j["H0"]
        out.append({"tag": tag, "x": x, "pdf": p,
                    "median": h["median"], "ci68": h["ci68"],
                    "n_rejected": j["n_neginf_cells"], "n_evals": j["n_evals"]})
    return out


# --------------------------------------------------------------------------- #
# Figure A — the 20 posteriors per arm, overlaid
# --------------------------------------------------------------------------- #
def fig_arms(D):
    # shared x range: union of the >0.1%-of-peak support over all 60 posteriors
    lo, hi = np.inf, -np.inf
    for arm, _, _, _, _ in ARMS:
        for e in D[arm]:
            m = e["pdf"] > 1e-3 * e["pdf"].max()
            lo, hi = min(lo, e["x"][m].min()), max(hi, e["x"][m].max())
    lo, hi = min(lo, H0_TRUTH) - 0.4, max(hi, H0_TRUTH) + 0.4

    # one shared y limit; a guard-truncated realisation can spike far above the
    # rest, so cap at the 95th percentile of peaks and annotate anything clipped
    peaks = np.array([e["pdf"].max() for a, _, _, _, _ in ARMS for e in D[a]])
    ytop = 1.18 * np.percentile(peaks, 95)

    fig, axes = plt.subplots(3, 1, figsize=(7.1, 5.6), dpi=300,
                             sharex=True, sharey=True)
    for ax, (arm, _, lab, col, _) in zip(axes, ARMS):
        med = np.array([e["median"] for e in D[arm]])
        mean, sem = med.mean(), med.std(ddof=1) / np.sqrt(med.size)
        ax.axvline(H0_TRUTH, color=INK, lw=1.0, ls=(0, (2, 2)), zorder=4)
        ax.axvline(mean, color=col, lw=1.4, zorder=5)
        for e in D[arm]:
            ax.plot(e["x"], e["pdf"], color=col, lw=0.7, alpha=0.45, zorder=3)
            if e["pdf"].max() > ytop:      # clipped spike: say which and why
                ax.annotate(
                    f"{e['tag']}: peak {e['pdf'].max():.1f}, "
                    f"{e['n_rejected']}/{e['n_evals']} cells rejected",
                    xy=(e["median"], ytop * 0.97), xytext=(6, -10),
                    textcoords="offset points", fontsize=6.6, color=INK2,
                    ha="left", va="top",
                    arrowprops=dict(arrowstyle="-", color=INK3, lw=0.6))
        # arm identity + ensemble numbers, in the empty left half of the panel
        ax.annotate(f"{arm}: {lab}\nmean of medians "
                    f"${mean:.2f}\\pm{sem:.2f}$  "
                    f"(offset ${mean - H0_TRUTH:+.2f}$)",
                    xy=(0.015, 0.93), xycoords="axes fraction",
                    ha="left", va="top", fontsize=7.8, color=col)
        ax.set_ylim(0, ytop)
        ax.set_ylabel("density")
        ax.grid(True, alpha=0.55)
        ax.set_axisbelow(True)
    axes[0].set_xlim(lo, hi)
    axes[-1].set_xlabel(r"$H_0$  [km s$^{-1}$ Mpc$^{-1}$]")
    axes[0].set_title("K=1 closure: all 20 realisations per arm  "
                      "(dashed line = truth, solid = arm mean of medians)",
                      fontsize=9.3)
    fig.tight_layout(h_pad=0.6)
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"fig_h0_recovery_arms.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("wrote figs/fig_h0_recovery_arms.{pdf,png}")


# --------------------------------------------------------------------------- #
# Figure B — per-realisation medians +- 68%, arms side by side
# --------------------------------------------------------------------------- #
def fig_medians(D):
    fig, ax = plt.subplots(figsize=(7.1, 3.6), dpi=300)
    n = len(TAGS)
    dx = {"ctrl": -0.28, "obs": 0.0, "fix": 0.28}

    # ensemble mean +- SEM bands first (recessive), then the truth line
    for arm, _, _, col, _ in ARMS:
        med = np.array([e["median"] for e in D[arm]])
        mean, sem = med.mean(), med.std(ddof=1) / np.sqrt(med.size)
        ax.axhspan(mean - sem, mean + sem, color=col, alpha=0.13, lw=0, zorder=1)
        ax.axhline(mean, color=col, lw=1.1, alpha=0.9, zorder=2)
    ax.axhline(H0_TRUTH, color=INK, lw=1.1, ls=(0, (2, 2)), zorder=3)

    for arm, _, _, col, mk in ARMS:
        xs = np.arange(n) + dx[arm]
        med = np.array([e["median"] for e in D[arm]])
        lo = med - np.array([e["ci68"][0] for e in D[arm]])
        hi = np.array([e["ci68"][1] for e in D[arm]]) - med
        ax.errorbar(xs, med, yerr=[lo, hi], fmt=mk, ms=3.4, color=col,
                    mfc=col, mec="white", mew=0.5, lw=0, elinewidth=0.9,
                    capsize=0, zorder=4)

    for i in range(n - 1):                 # thin separators between realisations
        ax.axvline(i + 0.5, color=GRIDCOL, lw=0.6, zorder=0)

    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(TAGS, rotation=90, fontsize=6.8)
    ax.set_xlim(-0.7, n - 0.3)
    ax.set_xlabel("catalog realisation")
    ax.set_ylabel(r"$H_0$  median $\pm$ 68%  [km s$^{-1}$ Mpc$^{-1}$]")
    ax.set_title("K=1 closure per realisation  (arms paired by catalog; "
                 "bands: arm mean $\\pm$ SEM)", fontsize=9.3)
    ax.grid(True, axis="y", alpha=0.55)
    ax.set_axisbelow(True)

    handles = [Line2D([0], [0], color=INK, lw=1.1, ls=(0, (2, 2)), label="truth")]
    for arm, _, lab, col, mk in ARMS:
        med = np.array([e["median"] for e in D[arm]])
        handles.append(Line2D([0], [0], marker=mk, color=col, mfc=col, mec="white",
                              mew=0.5, ms=4.5, lw=1.1,
                              label=f"{arm}: {lab} "
                                    f"({med.mean() - H0_TRUTH:+.2f})"))
    # Lower right: the fix arm's n4215 interval tops out well below the corner,
    # and no ensemble band reaches it.
    ax.legend(handles=handles, loc="lower right", fontsize=6.8, ncol=1)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"fig_h0_recovery_medians.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("wrote figs/fig_h0_recovery_medians.{pdf,png}")


# --------------------------------------------------------------------------- #
def main():
    D = {arm: load_arm(prefix) for arm, prefix, _, _, _ in ARMS}
    for arm, _, lab, _, _ in ARMS:
        med = np.array([e["median"] for e in D[arm]])
        print(f"{arm:5s} mean of medians {med.mean():.3f} "
              f"(offset {med.mean() - H0_TRUTH:+.3f}, "
              f"sem {med.std(ddof=1)/np.sqrt(med.size):.3f})  -- {lab}")
    fig_arms(D)
    fig_medians(D)


if __name__ == "__main__":
    main()
