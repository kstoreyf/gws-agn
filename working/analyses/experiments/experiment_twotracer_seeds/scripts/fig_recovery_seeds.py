#!/usr/bin/env python3
"""Recovery figures for the 12-seed two-tracer ensemble, pre- vs post-fix.

  figs/fig_f_recovery_seeds.{pdf,png}     -- all 12 f_AGN posteriors overlaid,
                                             post-fix solid vs pre-fix faded,
                                             ensemble mean-of-medians markers
  figs/fig_h0_recovery_seeds.{pdf,png}    -- same for the joint-grid H0
                                             marginals (the -3.2 -> +0.4 plot)
  figs/fig_joint_medians_seeds.{pdf,png}  -- the 12 joint (H0, f) medians, pre
                                             (faded) and post (solid), with the
                                             mean quoted-interval ellipses

Reads ONLY results/*.h5, results/joint_*.json and results/seeds_summary*.json.
`fscan_fix_s73xx` / `joint_fix_s73xx` are the post-repair measurement of
record; `fscan_s73xx` / `joint_s73xx` are the pre-fix generator kept for the
comparison.  Truth: H0 = 67.74, f_AGN = 0.30.

Colors follow this experiment's convention (make_fix_figures.py): pre-fix =
documented-palette slot 1 (#2a78d6), post-fix = slot 2 (#eb6834); the pair
passes the adjacent-pair CVD gates per the palette's validation record.
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
from matplotlib.patches import Ellipse
from matplotlib.transforms import blended_transform_factory

BASE = Path(__file__).resolve().parent.parent
RESULTS, FIGS = BASE / "results", BASE / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

TRUTH_H0, TRUTH_F = 67.74, 0.30
SEEDS = [f"73{i:02d}" for i in range(1, 13)]
PRE, POST = "#2a78d6", "#eb6834"          # slots 1-2, light mode
YELLOW = "#eda100"
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

SUM_PRE = json.loads((RESULTS / "seeds_summary.json").read_text())
SUM_POST = json.loads((RESULTS / "seeds_summary_fix.json").read_text())


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


def ens_stats(summary, key):
    """(mean value, sem) with the summary's truth offset undone."""
    st = summary[key]
    truth = TRUTH_F if key.endswith("_f") else TRUTH_H0
    return truth + st["mean"], st["sem"], st["mean"]


def overlay_panel(ax, curve_fn, stem_pre, stem_post, key):
    """Draw the two 12-curve ensembles; return (n_pre, n_post, peak)."""
    counts, pmax = {"pre": 0, "post": 0}, 0.0
    for tag, stem, col, al, lw in (("pre", stem_pre, PRE, 0.32, 1.0),
                                   ("post", stem_post, POST, 0.62, 1.25)):
        for s in SEEDS:
            h5 = RESULTS / stem.format(s=s)
            if not h5.exists():
                print(f"  [skip] {h5.name} missing")
                continue
            x, p = curve_fn(h5)
            pmax = max(pmax, float(p.max()))
            ax.plot(x, p, color=col, alpha=al, lw=lw,
                    zorder=4 if tag == "post" else 3)
            counts[tag] += 1
    # ensemble mean-of-medians +- s.e.m., on a data-x / axes-y transform
    tr = blended_transform_factory(ax.transData, ax.transAxes)
    for summ, col, y in ((SUM_PRE, PRE, 0.72), (SUM_POST, POST, 0.78)):
        mean, sem, _ = ens_stats(summ, key)
        ax.errorbar([mean], [y], xerr=[sem], transform=tr, fmt="D", ms=5.5,
                    color=col, mec="white", mew=0.7, elinewidth=1.6, capsize=3,
                    zorder=6)
    return counts["pre"], counts["post"], pmax


def legend_handles(key):
    out = []
    for summ, col, tag in ((SUM_PRE, PRE, "pre-fix"), (SUM_POST, POST, "post-fix (record)")):
        _, _, off = ens_stats(summ, key)
        sem = summ[key]["sem"]
        sign = "+" if off >= 0 else "−"
        out.append(Line2D([], [], color=col, lw=1.6,
                          alpha=0.9 if tag.startswith("post") else 0.6,
                          label=f"{tag}:  mean offset {sign}{abs(off):.3g}"
                                f" $\\pm$ {sem:.2g}"))
    out.append(Line2D([], [], marker="D", ms=5.5, color=INK3, mec="white",
                      mew=0.7, ls="none",
                      label="ensemble mean of medians $\\pm$ s.e.m."))
    return out


def main():
    # ------------------------------------------------- fig_f_recovery_seeds
    fig, ax = plt.subplots(figsize=(5.1, 3.5), dpi=300)
    npre, npost, pmax = overlay_panel(ax, posterior_1d, "fscan_s{s}.h5",
                                      "fscan_fix_s{s}.h5", "fscan_f")
    ax.axvline(TRUTH_F, ymax=0.655, color=INK2, lw=0.9, ls=(0, (1, 2)), zorder=2)
    ax.annotate("planted", xy=(TRUTH_F, 0.665), xycoords=("data", "axes fraction"),
                fontsize=7.0, color=INK2, ha="center", va="bottom")
    ax.set_xlim(0.05, 0.55)
    ax.set_ylim(0, 1.55 * pmax)
    ax.set_xlabel(r"AGN-hosted fraction  $f_{\rm AGN}$")
    ax.set_ylabel("posterior density")
    ax.set_title(f"$f_{{\\rm AGN}}$ recovery across {npost} seeds,\n"
                 "pre-fix vs $\\sigma_{\\rm ang}$-fixed generator", fontsize=9.0)
    ax.grid(True, alpha=0.55)
    ax.set_axisbelow(True)
    ax.legend(handles=legend_handles("fscan_f"), loc="upper right")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"fig_f_recovery_seeds.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote figs/fig_f_recovery_seeds.{{pdf,png}}  ({npre} pre / {npost} post)")

    # ------------------------------------------------ fig_h0_recovery_seeds
    fig, ax = plt.subplots(figsize=(5.4, 3.5), dpi=300)
    npre, npost, pmax = overlay_panel(ax, h0_marginal, "joint_s{s}.h5",
                                      "joint_fix_s{s}.h5", "joint_H0")
    ax.axvline(TRUTH_H0, ymax=0.655, color=INK2, lw=0.9, ls=(0, (1, 2)), zorder=2)
    ax.annotate("planted", xy=(TRUTH_H0, 0.665), xycoords=("data", "axes fraction"),
                fontsize=7.0, color=INK2, ha="center", va="bottom")
    ax.set_xlim(60.5, 73.5)
    ax.set_ylim(0, 1.55 * pmax)
    ax.set_xlabel(r"$H_0$  [km s$^{-1}$ Mpc$^{-1}$]")
    ax.set_ylabel("marginal posterior density")
    ax.set_title(f"$H_0$ recovery across {npost} seeds: the generator fix\n"
                 "removes the ensemble bias", fontsize=9.0)
    ax.grid(True, alpha=0.55)
    ax.set_axisbelow(True)
    ax.legend(handles=legend_handles("joint_H0"), loc="upper left")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"fig_h0_recovery_seeds.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote figs/fig_h0_recovery_seeds.{{pdf,png}}  ({npre} pre / {npost} post)")

    # ---------------------------------------------- fig_joint_medians_seeds
    pre_rows = {r["seed"]: r for r in SUM_PRE["per_seed"]}
    post_rows = {r["seed"]: r for r in SUM_POST["per_seed"]}
    seeds = [s for s in SEEDS if s in pre_rows and s in post_rows]
    Hp = np.array([pre_rows[s]["joint_H0"] for s in seeds])
    Fp = np.array([pre_rows[s]["joint_f"] for s in seeds])
    Hq = np.array([post_rows[s]["joint_H0"] for s in seeds])
    Fq = np.array([post_rows[s]["joint_f"] for s in seeds])

    fig, ax = plt.subplots(figsize=(5.1, 3.7), dpi=300)
    for a, b, c, d in zip(Hp, Fp, Hq, Fq):
        ax.plot([a, c], [b, d], color=GRIDCOL, lw=0.8, zorder=1)
    ax.scatter(Hp, Fp, s=26, color=PRE, alpha=0.45, edgecolors="white",
               linewidths=0.6, zorder=3, label="pre-fix medians")
    ax.scatter(Hq, Fq, s=30, color=POST, edgecolors="white", linewidths=0.6,
               zorder=4, label="post-fix medians (record)")
    for summ, col, H, F in ((SUM_PRE, PRE, Hp, Fp), (SUM_POST, POST, Hq, Fq)):
        cH, semH, _ = ens_stats(summ, "joint_H0")
        cF, semF, _ = ens_stats(summ, "joint_f")
        ax.add_patch(Ellipse((cH, cF),
                             2 * summ["joint_H0"]["mean_quoted_half_width"],
                             2 * summ["joint_f"]["mean_quoted_half_width"],
                             fill=False, edgecolor=col, lw=1.1,
                             ls=(0, (4, 2)), zorder=5))
        ax.errorbar([cH], [cF], xerr=[semH], yerr=[semF], fmt="D", ms=5.5,
                    color=col, mec="white", mew=0.7, elinewidth=1.5, capsize=2.5,
                    zorder=6)
    ax.axvline(TRUTH_H0, color=INK3, lw=0.8, ls=(0, (1, 2)), zorder=2)
    ax.axhline(TRUTH_F, color=INK3, lw=0.8, ls=(0, (1, 2)), zorder=2)
    ax.plot([TRUTH_H0], [TRUTH_F], marker="*", ms=11, color=YELLOW, mec=INK,
            mew=0.6, ls="none", zorder=6)
    ax.annotate("truth", xy=(TRUTH_H0, TRUTH_F), xytext=(6, -11),
                textcoords="offset points", fontsize=7.4, color=INK2)
    handles, labels = ax.get_legend_handles_labels()
    handles += [Line2D([], [], color=INK3, lw=1.1, ls=(0, (4, 2)),
                       label="mean quoted 68% interval"),
                Line2D([], [], marker="D", ms=5.5, color=INK3, mec="white",
                       mew=0.7, ls="none",
                       label="ensemble mean $\\pm$ s.e.m.")]
    ax.legend(handles=handles, loc="upper left", fontsize=7.2)
    allH = np.concatenate([Hp, Hq, [TRUTH_H0]])
    allF = np.concatenate([Fp, Fq, [TRUTH_F]])
    padH, padF = 0.10 * np.ptp(allH), 0.16 * np.ptp(allF)
    ax.set_xlim(allH.min() - padH, allH.max() + padH)
    ax.set_ylim(allF.min() - padF, allF.max() + 2.6 * padF)
    ax.set_xlabel(r"$H_0$  [km s$^{-1}$ Mpc$^{-1}$]")
    ax.set_ylabel(r"$f_{\rm AGN}$")
    ax.set_title(f"Joint medians of the {len(seeds)} seeds, pre "
                 r"$\rightarrow$ post fix", fontsize=9.0)
    ax.grid(True, alpha=0.45)
    ax.set_axisbelow(True)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"fig_joint_medians_seeds.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote figs/fig_joint_medians_seeds.{{pdf,png}}  ({len(seeds)} seeds)")


if __name__ == "__main__":
    main()
