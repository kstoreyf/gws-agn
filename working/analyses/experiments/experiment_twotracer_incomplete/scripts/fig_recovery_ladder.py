#!/usr/bin/env python3
"""Recovery figures along the completeness ladder (post-fix measurement of record).

  figs/fig_f_recovery_ladder_fix.{pdf,png}   -- f_AGN posteriors per rung, with
                                                the sky-shuffle nulls dashed
  figs/fig_h0_recovery_ladder_fix.{pdf,png}  -- H0 marginals of the joint grids
                                                per rung (the non-monotonic width)
  figs/fig_joint_h0f_ladder_fix.{pdf,png}    -- per-rung 68/90% credible regions,
                                                small multiples, zoomed

Reads ONLY results/*.h5 and results/*.json.  Default suffix `_fix` = the
post-repair generator (measurement of record); `--suffix ""` reproduces the
panels from the pre-fix scans for comparison.  Truth: H0 = 67.74, f_AGN = 0.30
(subject to the inherited absolute offset discussed in DESIGN.md -- the ladder
story is differential, but the truth lines are drawn always).

Rung colors follow this experiment's established assignment (make_figures.py
RAMP); one color per rung, identical across all three figures.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline

BASE = Path(__file__).resolve().parent.parent
RESULTS, FIGS = BASE / "results", BASE / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

TRUTH_H0, TRUTH_F = 67.74, 0.30
LEVELS = ["complete", "m21.0", "m20.0", "m19.0", "m18.0"]
LABELS = {"complete": "complete", "m21.0": "$m<21$", "m20.0": "$m<20$",
          "m19.0": "$m<19$", "m18.0": "$m<18$"}
RAMP = ["#0b0b0b", "#2a78d6", "#1baf7a", "#eda100", "#e34948"]
YELLOW, RED = "#eda100", "#e34948"
INK, INK2, INK3, GRIDCOL = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "font.family": "serif",
    "font.serif": ["DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 9, "axes.labelsize": 9.5, "axes.titlesize": 9.2,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 7.6,
    "axes.edgecolor": INK2, "axes.linewidth": 0.7, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "grid.color": GRIDCOL, "grid.linewidth": 0.6,
    "legend.frameon": False, "lines.solid_capstyle": "round",
})


def posterior_1d(h5):
    with h5py.File(h5, "r") as f:
        x, ll = f["f_grid"][:], f["log_likelihood"][:]
    ok = np.isfinite(ll)
    p = np.where(ok, np.exp(ll - np.nanmax(ll[ok])), 0.0)
    return x, p / np.trapz(p, x)


def h0_marginal(h5):
    with h5py.File(h5, "r") as f:
        H, F, ll = f["H0_grid"][:], f["f_grid"][:], f["log_likelihood"][:]
    ok = np.isfinite(ll)
    p = np.where(ok, np.exp(ll - ll[ok].max()), 0.0)
    m = (p * np.gradient(F)[None, :]).sum(axis=1)
    return H, m / np.trapz(m, H)


def refined_posterior_2d(h5, factor=6):
    """Normalized (H0, f) density, spline-refined in probability space for
    display (bounded ringing, clipped at zero; guard-rejected cells enter as
    p = 0)."""
    with h5py.File(h5, "r") as f:
        H, F, ll = f["H0_grid"][:], f["f_grid"][:], f["log_likelihood"][:]
    ok = np.isfinite(ll)
    p0 = np.where(ok, np.exp(ll - ll[ok].max()), 0.0)
    Hf = np.linspace(H[0], H[-1], factor * (H.size - 1) + 1)
    Ff = np.linspace(F[0], F[-1], factor * (F.size - 1) + 1)
    p = np.clip(RectBivariateSpline(H, F, p0, kx=3, ky=3)(Hf, Ff), 0.0, None)
    cell = np.outer(np.gradient(Hf), np.gradient(Ff))
    p /= (p * cell).sum()
    return Hf, Ff, p, cell, H, F, ok


def hpd_levels(p, cell, fracs=(0.68, 0.90)):
    w = (p * cell).ravel()
    order = np.argsort(w)[::-1]
    csum = np.cumsum(w[order])
    return [float(p.ravel()[order[min(np.searchsorted(csum, fr), order.size - 1)]])
            for fr in fracs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="_fix",
                    help="results suffix; default _fix (measurement of record)")
    sfx = ap.parse_args().suffix

    # -------------------------------------------- fig_f_recovery_ladder{sfx}
    fig, ax = plt.subplots(figsize=(5.1, 3.5), dpi=300)
    pmax, null_peak = 0.0, (0.0, 0.0)
    for lev, col in zip(LEVELS, RAMP):
        h5 = RESULTS / f"fscan_{lev}{sfx}.h5"
        if not h5.exists():
            print(f"  [skip] {h5.name} missing")
            continue
        x, p = posterior_1d(h5)
        pmax = max(pmax, float(p.max()))
        ax.plot(x, p, color=col, lw=1.6, zorder=4, label=LABELS[lev])
        nh5 = RESULTS / f"fscan_null_{lev}{sfx}.h5"
        if nh5.exists():
            xn, pn = posterior_1d(nh5)
            pmax = max(pmax, float(pn.max()))
            if pn.max() > null_peak[1]:
                null_peak = (float(xn[np.argmax(pn)]), float(pn.max()))
            ax.plot(xn, pn, color=col, lw=1.0, ls=(0, (2.5, 1.8)), alpha=0.65,
                    zorder=3)
    if null_peak[1] > 0:
        ax.annotate("sky-shuffled nulls\n(dashed)",
                    xy=(null_peak[0] + 0.07, 1.08 * null_peak[1]),
                    ha="center", va="bottom", fontsize=7.2, color=INK2)
    ax.axvline(TRUTH_F, color=INK2, lw=0.9, ls=(0, (1, 2)), zorder=2)
    ax.annotate("planted", xy=(TRUTH_F - 0.008, 0.965),
                xycoords=("data", "axes fraction"), fontsize=7.0, color=INK2,
                ha="right", va="top")
    ax.set_xlim(0, 0.8)
    ax.set_ylim(0, 1.32 * pmax)
    ax.set_xlabel(r"AGN-hosted fraction  $f_{\rm AGN}$")
    ax.set_ylabel("posterior density")
    ax.set_title("$f_{\\rm AGN}$ recovery along the completeness ladder"
                 + ("  (post-fix)" if sfx == "_fix" else f"  ({sfx or 'pre-fix'})"),
                 fontsize=9.0)
    ax.grid(True, alpha=0.55)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", title="host survey depth", title_fontsize=7.6)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"fig_f_recovery_ladder{sfx}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote figs/fig_f_recovery_ladder{sfx}.{{pdf,png}}")

    # -------------------------------------------- fig_h0_recovery_ladder{sfx}
    fig, ax = plt.subplots(figsize=(5.1, 3.5), dpi=300)
    pmax, xlo, xhi = 0.0, np.inf, -np.inf
    for lev, col in zip(LEVELS, RAMP):
        h5 = RESULTS / f"joint_{lev}{sfx}.h5"
        if not h5.exists():
            print(f"  [skip] {h5.name} missing")
            continue
        H, m = h0_marginal(h5)
        j = json.loads((RESULTS / f"joint_{lev}{sfx}.json").read_text())
        hw = 0.5 * (j["H0"]["ci68"][1] - j["H0"]["ci68"][0])
        pmax = max(pmax, float(m.max()))
        sup = H[m > 2e-3 * m.max()]
        xlo, xhi = min(xlo, sup.min()), max(xhi, sup.max())
        ax.plot(H, m, color=col, lw=1.6, zorder=4,
                label=f"{LABELS[lev]}   $\\pm{hw:.2f}$")
    ax.axvline(TRUTH_H0, color=INK2, lw=0.9, ls=(0, (1, 2)), zorder=2)
    ax.annotate("planted", xy=(TRUTH_H0 + 0.12, 0.965),
                xycoords=("data", "axes fraction"), fontsize=7.0, color=INK2,
                ha="left", va="top")
    pad = 0.10 * (xhi - xlo)
    ax.set_xlim(min(xlo - pad, TRUTH_H0 - pad), max(xhi + pad, TRUTH_H0 + pad))
    ax.set_ylim(0, 1.32 * pmax)
    ax.set_xlabel(r"$H_0$  [km s$^{-1}$ Mpc$^{-1}$]")
    ax.set_ylabel("marginal posterior density")
    ax.set_title("$H_0$ recovery along the completeness ladder"
                 + ("  (post-fix)" if sfx == "_fix" else f"  ({sfx or 'pre-fix'})"),
                 fontsize=9.0)
    ax.grid(True, alpha=0.55)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", title="depth   (68% half-width)",
              title_fontsize=7.6)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"fig_h0_recovery_ladder{sfx}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote figs/fig_h0_recovery_ladder{sfx}.{{pdf,png}}")

    # -------------------------------------------- fig_joint_h0f_ladder{sfx}
    have = [l for l in LEVELS if (RESULTS / f"joint_{l}{sfx}.h5").exists()]
    if not have:
        print("  [skip] no joint grids found; joint small multiples not made")
        return
    packs, ext_union = {}, None
    for lev in have:
        Hf, Ff, p, cell, H, F, ok = refined_posterior_2d(RESULTS / f"joint_{lev}{sfx}.h5")
        lv68, lv90 = hpd_levels(p, cell)
        m = p >= lv90
        e = (float(Hf[m.any(axis=1)].min()), float(Hf[m.any(axis=1)].max()),
             float(Ff[m.any(axis=0)].min()), float(Ff[m.any(axis=0)].max()))
        ext_union = e if ext_union is None else (
            min(ext_union[0], e[0]), max(ext_union[1], e[1]),
            min(ext_union[2], e[2]), max(ext_union[3], e[3]))
        packs[lev] = (Hf, Ff, p, lv68, lv90, H, F, ok)
    padH = 0.10 * (ext_union[1] - ext_union[0])
    padF = 0.12 * (ext_union[3] - ext_union[2])
    xlim = (min(ext_union[0] - padH, TRUTH_H0 - padH),
            max(ext_union[1] + padH, TRUTH_H0 + padH))
    ylim = (min(ext_union[2] - padF, TRUTH_F - padF),
            max(ext_union[3] + padF, TRUTH_F + padF))

    fig, axes = plt.subplots(1, len(have), figsize=(2.05 * len(have), 2.75),
                             dpi=300, sharey=True, sharex=True)
    axes = np.atleast_1d(axes)
    for ax, lev, col in zip(axes, have, RAMP):
        Hf, Ff, p, lv68, lv90, H, F, ok = packs[lev]
        Hm, Fm = np.meshgrid(Hf, Ff, indexing="ij")
        if (~ok).any():
            Hc, Fc = np.meshgrid(H, F, indexing="ij")
            ax.contourf(Hc, Fc, np.where(ok, np.nan, 1.0), levels=[0.5, 1.5],
                        colors=[RED], alpha=0.08, zorder=1)
        ax.contourf(Hm, Fm, p, levels=[lv68, p.max()], colors=[col],
                    alpha=0.20, zorder=3)
        ax.contour(Hm, Fm, p, levels=[lv90, lv68], colors=[col],
                   linewidths=[0.8, 1.15], zorder=4)
        ax.axvline(TRUTH_H0, color=INK3, lw=0.7, ls=(0, (1, 2)), zorder=2)
        ax.axhline(TRUTH_F, color=INK3, lw=0.7, ls=(0, (1, 2)), zorder=2)
        ax.plot([TRUTH_H0], [TRUTH_F], marker="*", ms=9, color=YELLOW, mec=INK,
                mew=0.5, ls="none", zorder=6)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_title(LABELS[lev])
        ax.set_xlabel(r"$H_0$")
        ax.grid(True, alpha=0.4)
        ax.set_axisbelow(True)
    axes[0].set_ylabel(r"$f_{\rm AGN}$")
    fig.suptitle("68/90% credible regions along the ladder"
                 + ("  (post-fix; " if sfx == "_fix" else "  (")
                 + "star + dotted cross = planted values)", fontsize=9)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"fig_joint_h0f_ladder{sfx}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote figs/fig_joint_h0f_ladder{sfx}.{{pdf,png}}")


if __name__ == "__main__":
    main()
