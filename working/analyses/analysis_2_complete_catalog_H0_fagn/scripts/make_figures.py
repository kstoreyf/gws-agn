#!/usr/bin/env python3
"""Figures for analysis_2_complete_catalog_H0_fagn — the joint (H0, f_AGN) fit.

  fig_joint_h0fagn.{pdf,png}  the 2-D 68 / 90 % credible regions: seed 100 filled,
                              the other four realisations as faint outlines, the
                              truth cross, zoomed on the union of the 90 % regions
  fig_marginals.{pdf,png}     the H0 and f marginal posteriors overlaid across the
                              five realisations, zoomed on where the mass is
  fig_closure_joint.{pdf,png} per-seed medians +- 90 % for BOTH parameters against
                              truth -- the closing exhibit
  fig_neff_f.{pdf,png}        the selection integral's N_eff and PE variance sum
                              against f at truth H0, both injection lanes -- the
                              operational exhibit

Colours are the dataviz reference palette (light surface), same slots and same
rcParams as analysis_1/scripts/make_figures.py so the two directories' figures read
as one system.  The colour choices were validated with scripts/validate_palette.py
(a faithful Python port of the skill's node validator -- this cluster has no node):

  * five seeds as overlaid lines -> categorical slots 1-5 on the ADJACENT pairlist:
    worst CVD dE 9.1 (protan), worst normal-vision dE 19.6, both PASS.  Slots 3-5
    sit below 3:1 against the light surface, so the relief rule applies: the panel
    carries a legend and the per-seed numbers are tabulated in the README and in
    results/joint_summary.json -- identity is never colour alone.
  * two injection lanes -> slots 1-2 on the ALL-PAIRS list: CVD dE 24.7,
    normal-vision dE 33.6, contrast PASS.
  * the 2-D region figure uses one accent (seed 100) and folds the other four into
    a single muted neutral, the documented "fold to Other" pattern.

Print medium: light surface only, PDF + PNG.  Every annotation is computed from the
result files; nothing is hard-coded.
"""
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent
RES = Path(os.environ.get("A2_RESDIR", ROOT / "results"))
FIGS = Path(os.environ.get("A2_FIGDIR", ROOT / "figs"))
DATA_ROOT = Path("/hildafs/projects/phy230014p/magana/gws-agn/working/data")
H0_TRUTH = 67.74
F_PLANTED = 0.30
SEEDS = [100, 101, 102, 103, 105]
REF_SEED = 100

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#8a8a85"
GRID = "#e6e5e1"
BLUE = "#2a78d6"      # slot 1
ORANGE = "#eb6834"    # slot 2
AQUA = "#1baf7a"      # slot 3
YELLOW = "#eda100"    # slot 4
MAGENTA = "#e87ba4"   # slot 5
SLOTS = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA]
OTHER = "#a9a8a2"     # the folded "other realisations" neutral
REJECT = "#d03b3b"

mpl.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.size": 9, "axes.labelsize": 9.5,
    "axes.titlesize": 10.5, "axes.edgecolor": INK_MUTED, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": INK_2, "ytick.color": INK_2,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "axes.linewidth": 0.8,
    "xtick.direction": "out", "ytick.direction": "out", "legend.frameon": False,
    "pdf.fonttype": 42,
})


def save(fig, name):
    FIGS.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"{name}.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {FIGS / name}.pdf/.png")


def tidy(ax, grid_axis="both"):
    ax.grid(True, axis=grid_axis, color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def jload(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def realised_f(seed):
    m = json.loads((DATA_ROOT / f"seed{seed}" / "META.json").read_text())
    r = m["stages"]["events"]["realised"]
    return r["n_host_agn"] / (r["n_host_agn"] + r["n_host_gal"])


def joint_grid(tag):
    """(H0, f, normalized 2-D posterior) from a joint scan h5."""
    p = RES / f"{tag}.h5"
    if not p.exists():
        return None
    with h5py.File(p, "r") as f:
        H0 = np.asarray(f["H0_grid"][:], float)
        fv = np.asarray(f["f_grid"][:], float)
        ll = np.asarray(f["log_likelihood"][:], float)
    fin = np.isfinite(ll)
    P = np.where(fin, np.exp(ll - ll[fin].max()), 0.0)
    P /= np.trapz(np.trapz(P, fv, axis=1), H0, axis=0)
    return H0, fv, P


def hpd_levels(H0, fv, P, levels=(0.68, 0.90)):
    """Density thresholds enclosing the requested posterior mass."""
    dA = np.outer(np.gradient(H0), np.gradient(fv))
    flat = P.ravel()
    w = (flat * dA.ravel())
    order = np.argsort(flat)[::-1]
    cum = np.cumsum(w[order]) / w.sum()
    out = []
    for lev in levels:
        k = int(np.searchsorted(cum, lev))
        k = min(k, flat.size - 1)
        out.append(float(flat[order][k]))
    return out  # descending mass -> ascending level order is (68, 90) -> hi, lo


# --------------------------------------------------------------------------- #
def fig_joint():
    grids = {}
    for s in SEEDS:
        g = joint_grid(f"joint_s{s}")
        if g:
            grids[s] = g
    if REF_SEED not in grids:
        print("[skip] fig_joint_h0fagn: no seed-100 joint grid yet")
        return
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    tidy(ax)

    # faint outlines for the other realisations
    for s in SEEDS:
        if s == REF_SEED or s not in grids:
            continue
        H0, fv, P = grids[s]
        l68, l90 = hpd_levels(H0, fv, P)
        ax.contour(H0, fv, P.T, levels=[l90, l68], colors=OTHER,
                   linewidths=0.9, zorder=2)

    H0, fv, P = grids[REF_SEED]
    l68, l90 = hpd_levels(H0, fv, P)
    ax.contourf(H0, fv, P.T, levels=[l90, l68, P.max() * 1.01],
                colors=[BLUE, BLUE], alpha=0.18, zorder=3)
    ax.contourf(H0, fv, P.T, levels=[l68, P.max() * 1.01],
                colors=[BLUE], alpha=0.22, zorder=3)
    ax.contour(H0, fv, P.T, levels=[l90, l68], colors=BLUE,
               linewidths=[1.1, 1.7], zorder=4)

    # truth
    ax.axvline(H0_TRUTH, color=INK_2, lw=0.9, ls=(0, (5, 3)), zorder=5)
    ax.axhline(F_PLANTED, color=INK_2, lw=0.9, ls=(0, (5, 3)), zorder=5)
    ax.plot([H0_TRUTH], [F_PLANTED], marker="+", ms=13, mew=2.0,
            color=INK, zorder=7)
    fr = [realised_f(s) for s in grids]
    ax.plot([H0_TRUTH] * len(fr), fr, marker="_", ms=9, mew=1.3, ls="none",
            color=INK_MUTED, zorder=6)

    # zoom on the union of the 90 % regions, with padding
    xs, ys = [], []
    for s, (H0s, fvs, Ps) in grids.items():
        _, l90s = hpd_levels(H0s, fvs, Ps)
        m = Ps >= l90s
        xs += [H0s[m.any(axis=1)].min(), H0s[m.any(axis=1)].max()]
        ys += [fvs[m.any(axis=0)].min(), fvs[m.any(axis=0)].max()]
    xlo, xhi = min(xs + [H0_TRUTH]), max(xs + [H0_TRUTH])
    ylo, yhi = min(ys + [F_PLANTED]), max(ys + [F_PLANTED])
    px, py = 0.12 * (xhi - xlo), 0.20 * (yhi - ylo)
    ax.set_xlim(xlo - px, xhi + px)
    ax.set_ylim(max(0.0, ylo - py), min(1.0, yhi + py))

    ax.set_xlabel(r"$H_0$  [km s$^{-1}$ Mpc$^{-1}$]")
    ax.set_ylabel(r"$f_{\rm AGN}$")
    j = jload(RES / f"joint_s{REF_SEED}.json")
    sub = ""
    if j:
        sub = (f"seed {REF_SEED}: "
               f"$H_0$ = {j['H0']['median']:.1f}"
               f"$^{{+{j['H0']['ci90'][1] - j['H0']['median']:.1f}}}"
               f"_{{-{j['H0']['median'] - j['H0']['ci90'][0]:.1f}}}$, "
               f"$f_{{\\rm AGN}}$ = {j['f']['median']:.3f}"
               f"$^{{+{j['f']['ci90'][1] - j['f']['median']:.3f}}}"
               f"_{{-{j['f']['median'] - j['f']['ci90'][0]:.3f}}}$, "
               f"correlation $\\rho$ = {j['rho']:+.2f}")
    ax.set_title(sub, loc="left", fontsize=8.5, color=INK_2, pad=8)
    fig.suptitle("The joint fit on the complete catalogs: both parameters at once",
                 x=0.005, ha="left", fontsize=10.5, y=1.035)
    handles = [Line2D([], [], color=BLUE, lw=1.7,
                      label=f"seed {REF_SEED}  90 % / 90 %")]
    if len(grids) > 1:
        handles.append(Line2D([], [], color=OTHER, lw=0.9,
                              label=f"the other {len(grids) - 1} realisations"))
    handles += [
        Line2D([], [], color=INK, lw=0, marker="+", ms=10, mew=1.8,
               label=r"truth  (67.74,  0.30 planted)"),
        Line2D([], [], color=INK_MUTED, lw=0, marker="_", ms=9, mew=1.3,
               label=r"realised $f_{\rm AGN}$ per realisation"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.16),
              ncol=2, fontsize=8, labelcolor=INK_2, handlelength=1.6,
              columnspacing=1.6)
    save(fig, "fig_joint_h0fagn")


# --------------------------------------------------------------------------- #
def fig_marginals():
    grids = {s: joint_grid(f"joint_s{s}") for s in SEEDS}
    grids = {s: g for s, g in grids.items() if g}
    if not grids:
        print("[skip] fig_marginals: no joint grids yet")
        return
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.9))
    for ax in axes:
        tidy(ax)

    order = [x for x in SEEDS if x in grids]
    handles = []
    spanH0, spanf = [], []
    for k, s in enumerate(order):
        H0, fv, P = grids[s]
        c = SLOTS[k % len(SLOTS)]
        pH0 = np.trapz(P, fv, axis=1); pH0 /= np.trapz(pH0, H0)
        pf = np.trapz(P, H0, axis=0); pf /= np.trapz(pf, fv)
        axes[0].plot(H0, pH0, color=c, lw=1.6, zorder=4)
        axes[1].plot(fv, pf, color=c, lw=1.6, zorder=4)
        handles.append(Line2D([], [], color=c, lw=1.6, label=f"seed {s}"))
        # x-range worth showing: where the curve carries 99.5 % of its mass
        for span, x, p in ((spanH0, H0, pH0), (spanf, fv, pf)):
            cdf = np.concatenate([[0.0], np.cumsum(0.5 * (p[1:] + p[:-1]) * np.diff(x))])
            cdf /= cdf[-1]
            span += [float(np.interp(0.0025, cdf, x)), float(np.interp(0.9975, cdf, x))]

    for ax, span, anchor in ((axes[0], spanH0, H0_TRUTH), (axes[1], spanf, F_PLANTED)):
        lo, hi = min(span + [anchor]), max(span + [anchor])
        pad = 0.10 * (hi - lo)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(0.0, 1.22 * max(l.get_ydata().max() for l in ax.get_lines()
                                    if len(l.get_ydata()) > 2))

    axes[0].axvline(H0_TRUTH, color=INK_2, lw=1.0, ls=(0, (5, 3)), zorder=5)
    axes[0].annotate("truth 67.74", (H0_TRUTH, 1.0), xycoords=("data", "axes fraction"),
                     textcoords="offset points", xytext=(4, -11), fontsize=8,
                     color=INK_2)
    axes[0].set_xlabel(r"$H_0$  [km s$^{-1}$ Mpc$^{-1}$]")
    axes[0].set_ylabel("marginal posterior density")
    axes[0].set_title("Expansion rate", loc="left")
    axes[0].legend(handles=handles, loc="upper left", fontsize=7.5,
                   labelcolor=INK_2, handlelength=1.4)

    axes[1].axvline(F_PLANTED, color=INK_2, lw=1.0, ls=(0, (5, 3)), zorder=5)
    for s in order:
        axes[1].axvline(realised_f(s), color=INK_MUTED, lw=0.7, ls=":", zorder=3)
    axes[1].annotate("planted 0.30", (F_PLANTED, 1.0), xycoords=("data", "axes fraction"),
                     textcoords="offset points", xytext=(-4, -11), fontsize=8,
                     color=INK_2, ha="right")
    axes[1].set_xlabel(r"$f_{\rm AGN}$")
    axes[1].set_ylabel("marginal posterior density")
    axes[1].set_title("AGN host fraction", loc="left")
    axes[1].legend(handles=[Line2D([], [], color=INK_MUTED, lw=0.7, ls=":",
                                   label="realised fraction")],
                   loc="upper right", fontsize=7.5, labelcolor=INK_2,
                   handlelength=1.4)
    fig.suptitle("The joint fit's marginals, five independent realisations of the mock",
                 x=0.005, ha="left", fontsize=10.5, y=1.02)
    save(fig, "fig_marginals")


# --------------------------------------------------------------------------- #
def fig_closure():
    summ = jload(RES / "joint_summary.json")
    if not summ:
        print("[skip] fig_closure_joint: no joint_summary.json yet")
        return
    rows = [r for r in summ["seeds"] if "joint" in r]
    if not rows:
        print("[skip] fig_closure_joint: no joint rows")
        return
    x = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.9))
    for ax in axes:
        tidy(ax, grid_axis="y")
        ax.set_xticks(x)
        ax.set_xticklabels([f"seed {r['seed']}" for r in rows])
        ax.set_xlim(-0.6, len(rows) - 0.4)
        ax.margins(y=0.16)

    # --- H0 -------------------------------------------------------------------
    ax = axes[0]
    med = np.array([r["joint"]["H0"]["median"] for r in rows])
    lo = np.array([r["joint"]["H0"]["median"] - r["joint"]["H0"]["ci90"][0] for r in rows])
    hi = np.array([r["joint"]["H0"]["ci90"][1] - r["joint"]["H0"]["median"] for r in rows])
    ax.axhline(H0_TRUTH, color=INK_2, lw=1.0, ls=(0, (5, 3)), zorder=3)
    c = summ["closure"]["joint_H0"]
    ax.axhspan(H0_TRUTH + c["mean"] - c["sem"], H0_TRUTH + c["mean"] + c["sem"],
               color=BLUE, alpha=0.10, zorder=2)
    ax.axhline(H0_TRUTH + c["mean"], color=BLUE, lw=1.0, zorder=3)
    ax.errorbar(x, med, yerr=[lo, hi], fmt="o", ms=5.5, color=BLUE,
                ecolor=BLUE, elinewidth=1.5, capsize=3, zorder=5)
    ax.set_ylabel(r"$H_0$  [km s$^{-1}$ Mpc$^{-1}$]")
    ax.set_title("Expansion rate", loc="left")
    ax.annotate(f"mean offset  {c['mean']:+.2f} $\\pm$ {c['sem']:.2f}",
                (0.5, 0.015), xycoords="axes fraction", ha="center", fontsize=8.5,
                color=INK_2)
    ax.annotate("truth 67.74", (-0.55, H0_TRUTH), fontsize=8,
                color=INK_2, va="bottom", ha="left",
                textcoords="offset points", xytext=(2, 3))

    # --- f --------------------------------------------------------------------
    ax = axes[1]
    med = np.array([r["joint"]["f_vs_realised"]["median"] for r in rows])
    lo = np.array([r["joint"]["f_vs_realised"]["median"] - r["joint"]["f_vs_realised"]["ci90"][0]
                   for r in rows])
    hi = np.array([r["joint"]["f_vs_realised"]["ci90"][1] - r["joint"]["f_vs_realised"]["median"]
                   for r in rows])
    fr = np.array([r["f_realised"] for r in rows])
    ax.axhline(F_PLANTED, color=INK_2, lw=1.0, ls=(0, (5, 3)), zorder=3)
    ax.plot(x, fr, marker="_", ms=16, mew=1.6, ls="none", color=INK_MUTED, zorder=4)
    ax.errorbar(x, med, yerr=[lo, hi], fmt="o", ms=5.5, color=BLUE,
                ecolor=BLUE, elinewidth=1.5, capsize=3, zorder=5)
    cr = summ["closure"]["joint_f_vs_realised"]
    ax.set_ylabel(r"$f_{\rm AGN}$")
    ax.set_title("AGN host fraction", loc="left")
    ax.annotate(f"mean offset vs realised  {cr['mean']:+.3f} $\\pm$ {cr['sem']:.3f}",
                (0.5, 0.015), xycoords="axes fraction", ha="center", fontsize=8.5,
                color=INK_2)
    ax.legend(handles=[
        Line2D([], [], color=BLUE, lw=1.5, marker="o", ms=5.5,
               label="joint fit, median $\\pm$ 90 %"),
        Line2D([], [], color=INK_MUTED, lw=0, marker="_", ms=12, mew=1.6,
               label="realised fraction"),
        Line2D([], [], color=INK_2, lw=1.0, ls=(0, (5, 3)), label="planted 0.30"),
    ], loc="best", fontsize=7.5, labelcolor=INK_2)
    fig.suptitle("Five realisations, both parameters, one fit",
                 x=0.007, ha="left", fontsize=10.5, y=1.02)
    save(fig, "fig_closure_joint")


# --------------------------------------------------------------------------- #
def fig_neff():
    lanes = [("targeted", "", BLUE, "-"), ("popuni", "_popuni", ORANGE, "--")]
    have = []
    for lane, suf, c, ls in lanes:
        j = jload(RES / f"joint_s{REF_SEED}{suf}.json")
        blk = (j or {}).get("guard", {}).get("neff_vs_f_at_truth_H0")
        if blk is None:                      # fall back to the 1-D f scan
            j = jload(RES / f"fscan_s{REF_SEED}{suf}.json")
            if j and j.get("guard", {}).get("cells"):
                cells = j["guard"]["cells"]
                blk = {"f": [c_["f"] for c_ in cells],
                       "Neff": [c_["Neff"] for c_ in cells],
                       "pe_variance_sum": [c_["pe_variance_sum"] for c_ in cells],
                       "threshold": [c_["threshold"] for c_ in cells],
                       "H0": j.get("h0_fixed")}
        if blk:
            have.append((lane, c, ls, blk))
    if not have:
        print("[skip] fig_neff_f: no guard records yet")
        return
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 5.4), sharex=True,
                             gridspec_kw={"height_ratios": [1.25, 1.0]})
    for ax in axes:
        tidy(ax)
    handles = []
    label_at = (0.30, 0.72)
    for k, (lane, c, ls, blk) in enumerate(have):
        f = np.asarray(blk["f"], float)
        neff = np.asarray(blk["Neff"], float)
        axes[0].plot(f, neff, color=c, lw=1.7, ls=ls, zorder=4)
        axes[1].plot(f, np.asarray(blk["pe_variance_sum"], float), color=c,
                     lw=1.7, ls=ls, zorder=4)
        i = int(len(f) * label_at[k % 2])
        axes[0].annotate(lane, (f[i], neff[i]), textcoords="offset points",
                         xytext=(0, 9 if k == 0 else -16), fontsize=8, color=c,
                         ha="center", zorder=6)
        handles.append(Line2D([], [], color=c, lw=1.7, ls=ls,
                              label=f"{lane} injections"))
    axes[0].legend(handles=handles, loc="center left", bbox_to_anchor=(0.02, 0.33),
                   fontsize=7.5, labelcolor=INK_2, handlelength=2.2)
    thr = float(np.max(have[0][3]["threshold"]))
    axes[0].axhline(thr, color=REJECT, lw=1.0, ls=(0, (4, 3)), zorder=3)
    axes[0].annotate(f"guard floor  $5N_{{\\rm obs}}$ = {thr:,.0f}", (0.02, thr),
                     xycoords=("axes fraction", "data"),
                     textcoords="offset points", xytext=(0, 5), fontsize=8,
                     color=REJECT)
    axes[0].set_yscale("log")
    axes[0].set_ylim(top=float(np.nanmax([np.max(b["Neff"]) for *_, b in have])) * 3)
    axes[0].set_ylabel(r"selection-integral $N_{\rm eff}$")
    axes[0].set_title(
        f"seed {REF_SEED}, at truth $H_0$ = {have[0][3].get('H0', H0_TRUTH):.2f}; "
        "solid = targeted lane (record), dashed = population+uniform (cross-check)",
        loc="left", fontsize=8, color=INK_2, pad=7)
    fig.suptitle("What the mixture weight costs the selection integral",
                 x=0.005, ha="left", fontsize=10.5, y=0.985)
    axes[1].set_ylabel(r"$\sum_i \sigma^2_{{\rm PE},i}$")
    axes[1].set_xlabel(r"$f_{\rm AGN}$")
    axes[1].axvline(F_PLANTED, color=INK_2, lw=0.9, ls=(0, (5, 3)), zorder=3)
    axes[1].annotate("planted 0.30", (F_PLANTED, axes[1].get_ylim()[1]),
                     textcoords="offset points", xytext=(4, -11), fontsize=8,
                     color=INK_2)
    save(fig, "fig_neff_f")


if __name__ == "__main__":
    which = sys.argv[1:] or ["joint", "marginals", "closure", "neff"]
    if "joint" in which:
        fig_joint()
    if "marginals" in which:
        fig_marginals()
    if "closure" in which:
        fig_closure()
    if "neff" in which:
        fig_neff()
