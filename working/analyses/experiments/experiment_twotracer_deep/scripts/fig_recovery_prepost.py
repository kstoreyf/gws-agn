#!/usr/bin/env python3
"""Recovery figures for the deep two-tracer mock, pre- vs post-fix generator.

  figs/fig_f_recovery_prepost.{pdf,png}    -- f_AGN posteriors (N=200 and N=80),
                                              pre-fix vs sigma_ang-fixed generator
  figs/fig_joint_h0f_prepost.{pdf,png}     -- 68/90% credible regions in the
                                              (H0, f_AGN) plane, pre vs post fix

Reads ONLY results/*.h5 and results/*.json.  The `_fix` scans are the
measurement of record (RESULTS_FIX.md); the untagged scans are the pre-fix
generator kept for comparison.  Truth: H0 = 67.74, f_AGN = 0.30.

Colors follow the family convention (experiment_twotracer_seeds/scripts/
make_fix_figures.py): pre-fix = documented-palette slot 1 (#2a78d6), post-fix =
slot 2 (#eb6834); the pair passes the adjacent-pair CVD gates per the palette's
validation record.
"""
from __future__ import annotations

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
PRE, POST = "#2a78d6", "#eb6834"          # slots 1-2, light mode
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


def refined_posterior_2d(h5, factor=6):
    """Normalized (H0, f) posterior density on a spline-refined grid.

    Display refinement only: the 81x41 evaluation grid makes chunky contours at
    the zoom this figure needs.  The spline is taken in probability space (not
    log space), where cubic ringing is bounded by the local value range; small
    negative lobes are clipped to zero.  Guard-rejected (-inf) cells enter as
    p = 0, and the recorded posterior mass adjacent to the rejected region is
    ~0 (see summary_fix.json), so this cannot move a contour.
    """
    with h5py.File(h5, "r") as f:
        H, F, ll = f["H0_grid"][:], f["f_grid"][:], f["log_likelihood"][:]
    ok = np.isfinite(ll)
    p0 = np.where(ok, np.exp(ll - ll[ok].max()), 0.0)
    Hf = np.linspace(H[0], H[-1], factor * (H.size - 1) + 1)
    Ff = np.linspace(F[0], F[-1], factor * (F.size - 1) + 1)
    p = np.clip(RectBivariateSpline(H, F, p0, kx=3, ky=3)(Hf, Ff), 0.0, None)
    cell = np.outer(np.gradient(Hf), np.gradient(Ff))
    p /= (p * cell).sum()
    return Hf, Ff, p, cell, ok, H, F


def hpd_levels(p, cell, fracs=(0.68, 0.90)):
    w = (p * cell).ravel()
    order = np.argsort(w)[::-1]
    csum = np.cumsum(w[order])
    return [float(p.ravel()[order[min(np.searchsorted(csum, fr), order.size - 1)]])
            for fr in fracs]


def region_extent(H, F, p, lev90):
    m = p >= lev90
    return (float(H[m.any(axis=1)].min()), float(H[m.any(axis=1)].max()),
            float(F[m.any(axis=0)].min()), float(F[m.any(axis=0)].max()))


def main():
    # ------------------------------------------------- fig_f_recovery_prepost
    specs = [
        ("tgt_fscan_n80",      PRE,  (0, (4, 1.8)), 1.2, 0.8,
         "pre-fix,  $N{=}80$"),
        ("tgt_fscan_n200",     PRE,  "-",           1.7, 1.0,
         "pre-fix,  $N{=}200$"),
        ("tgt_fscan_n80_fix",  POST, (0, (4, 1.8)), 1.2, 0.8,
         "post-fix, $N{=}80$"),
        ("tgt_fscan_n200_fix", POST, "-",           1.9, 1.0,
         "post-fix, $N{=}200$  (record)"),
    ]
    fig, ax = plt.subplots(figsize=(4.7, 3.3), dpi=300)
    pmax = 0.0
    for tag, col, ls, lw, al, lab in specs:
        h5 = RESULTS / f"{tag}.h5"
        if not h5.exists():
            print(f"  [skip] {h5.name} missing")
            continue
        med = json.loads((RESULTS / f"{tag}.json").read_text())["f"]["median"]
        x, p = posterior_1d(h5)
        pmax = max(pmax, float(p.max()))
        ax.plot(x, p, color=col, ls=ls, lw=lw, alpha=al, zorder=4,
                label=f"{lab}   $\\hat f={med:.3f}$")
    # keep the truth line clear of the legend block at top right
    ax.axvline(TRUTH_F, ymax=0.72, color=INK2, lw=0.9, ls=(0, (1, 2)), zorder=2)
    ax.annotate("planted", xy=(TRUTH_F - 0.008, 0.045),
                xycoords=("data", "axes fraction"), fontsize=7.0, color=INK2,
                ha="right")
    ax.set_xlim(0, 0.6)
    ax.set_ylim(0, 1.30 * pmax)
    ax.set_xlabel(r"AGN-hosted fraction  $f_{\rm AGN}$")
    ax.set_ylabel("posterior density")
    ax.set_title("Deep two-tracer mock: $f_{\\rm AGN}$ recovery,\n"
                 "pre-fix vs $\\sigma_{\\rm ang}$-fixed generator", fontsize=9.0)
    ax.grid(True, alpha=0.55)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", handlelength=2.4)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"fig_f_recovery_prepost.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("wrote figs/fig_f_recovery_prepost.{pdf,png}")

    # ------------------------------------------------ fig_joint_h0f_prepost
    fig, ax = plt.subplots(figsize=(4.7, 3.6), dpi=300)
    ext_union = None
    for tag, col, lab in (("tgt_joint_n200", PRE, "pre-fix"),
                          ("tgt_joint_n200_fix", POST, "post-fix (record)")):
        h5 = RESULTS / f"{tag}.h5"
        if not h5.exists():
            print(f"  [skip] {h5.name} missing")
            continue
        Hf, Ff, p, cell, ok, H, F = refined_posterior_2d(h5)
        lv68, lv90 = hpd_levels(p, cell)
        Hm, Fm = np.meshgrid(Hf, Ff, indexing="ij")
        ax.contourf(Hm, Fm, p, levels=[lv68, p.max()], colors=[col],
                    alpha=0.20, zorder=3)
        ax.contour(Hm, Fm, p, levels=[lv90, lv68], colors=[col],
                   linewidths=[0.9, 1.3], zorder=4)
        j = json.loads((RESULTS / f"{tag}.json").read_text())
        ax.plot([j["map"]["H0"]], [j["map"]["f"]], marker="P", ms=6.0,
                color=col, mec="white", mew=0.7, ls="none", zorder=6)
        e = region_extent(Hf, Ff, p, lv90)
        ext_union = e if ext_union is None else (
            min(ext_union[0], e[0]), max(ext_union[1], e[1]),
            min(ext_union[2], e[2]), max(ext_union[3], e[3]))
        # direct label at the region's high edge
        ax.annotate(lab, xy=(0.5 * (e[0] + e[1]), e[3]), xytext=(0, 5),
                    textcoords="offset points", ha="center", fontsize=7.8,
                    color=col, fontweight="bold")
    ax.axvline(TRUTH_H0, color=INK3, lw=0.8, ls=(0, (1, 2)), zorder=2)
    ax.axhline(TRUTH_F, color=INK3, lw=0.8, ls=(0, (1, 2)), zorder=2)
    ax.plot([TRUTH_H0], [TRUTH_F], marker="*", ms=11, color=YELLOW, mec=INK,
            mew=0.6, ls="none", zorder=6)
    ax.annotate("truth", xy=(TRUTH_H0, TRUTH_F), xytext=(6, -10),
                textcoords="offset points", fontsize=7.4, color=INK2)
    if ext_union is not None:
        padH = 0.13 * (ext_union[1] - ext_union[0])
        padF = 0.16 * (ext_union[3] - ext_union[2])
        ax.set_xlim(min(ext_union[0] - padH, TRUTH_H0 - padH),
                    max(ext_union[1] + padH, TRUTH_H0 + padH))
        ax.set_ylim(min(ext_union[2] - padF, TRUTH_F - padF),
                    max(ext_union[3] + 2.2 * padF, TRUTH_F + padF))
    ax.set_xlabel(r"$H_0$  [km s$^{-1}$ Mpc$^{-1}$]")
    ax.set_ylabel(r"$f_{\rm AGN}$")
    ax.set_title("Joint 68/90% credible regions, $N{=}200$\n"
                 "(star + dotted cross = planted values, $+$ = MAP)",
                 fontsize=9.0)
    ax.grid(True, alpha=0.45)
    ax.set_axisbelow(True)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"fig_joint_h0f_prepost.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("wrote figs/fig_joint_h0f_prepost.{pdf,png}")


if __name__ == "__main__":
    main()
