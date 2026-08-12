#!/usr/bin/env python3
"""Figures for analysis_4_density_anchoring_H0_fagn — the AGN density anchor.

  fig_anchor_response.{pdf,png}    THE HEADLINE: f_AGN and H0 against the density
                                   the completion is anchored to, one line per
                                   survey depth, truth marked
  fig_anchor_budget.{pdf,png}      the same shifts in units of the exact arm's own
                                   68 % half-width -- when the anchoring
                                   systematic overtakes the statistical error
  fig_anchor_significance.{pdf,png} the f_AGN detection significance and the
                                   posterior width against the anchor
  fig_anchor_posteriors.{pdf,png}  the marginal f_AGN posteriors themselves, one
                                   panel per rung
  fig_oracle_m18.{pdf,png}         the oracle probe: f_AGN at the faintest rung
                                   with the AGN survey handed over complete

Colours, rcParams and the light print surface are analyses 2 and 3's, unchanged,
so the four directories' figures read as one system.  Identity is never colour
alone: the anchor sweep also carries a monotone x axis and direct labels, and
every number is tabulated in the README and in results/arms_summary.json.

Every figure is a pure function of results/arms_summary.json (+ the scan .h5 for
posterior curves).  Nothing is hard-coded; a figure whose inputs are not on disk
is SKIPPED with a message rather than drawn from partial data, and panels for
arms still in the queue are simply absent -- so this is safe to run at any point
in the campaign, which is how the finalizer uses it.
"""
import json
import os
from pathlib import Path

import h5py
import numpy as np
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent
RES = Path(os.environ.get("A4_RESDIR", ROOT / "results"))
FIGS = Path(os.environ.get("A4_FIGDIR", ROOT / "figs"))
A3 = ROOT.parent / "analysis_3_incomplete_catalog_H0_fagn" / "results"
H0_TRUTH = 67.74
LEVELS = ["m21", "m20", "m19", "m18"]
PRETTY = {"m21": "m < 21", "m20": "m < 20", "m19": "m < 19", "m18": "m < 18"}
ARM_ORDER = ["a05", "a07", "a09", "exact", "a11", "a13", "a20"]

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
OTHER = "#a9a8a2"
REJECT = "#d03b3b"
LEVEL_COLOR = dict(zip(LEVELS, SLOTS))

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


def factor_axis(ax, factors):
    """A log x axis labelled with the arm factors and nothing else."""
    ax.set_xscale("log")
    ax.set_xticks(sorted(factors))
    ax.set_xticklabels([f"{v:g}" for v in sorted(factors)])
    ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())


def all_factors(S):
    return sorted({r["factor"] for l in levels_present(S)
                   for _, r in present_arms(S, l)})


def arm_color(factor, factors, cmap=None):
    """Monotone colour for one arm: dark = under-anchored, light = over."""
    cmap = cmap or plt.get_cmap("viridis")
    lo, hi = np.log10(min(factors)), np.log10(max(factors))
    t = 0.0 if hi == lo else (np.log10(factor) - lo) / (hi - lo)
    return cmap(0.12 + 0.76 * t)


def jload(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def summary():
    return jload(RES / "arms_summary.json")


def present_arms(S, lev):
    """(factor, row) for every arm of this rung that is on disk, in factor order."""
    arms = S["rungs"].get(lev, {}).get("arms", {})
    got = [(a, arms[a]) for a in ARM_ORDER
           if arms.get(a, {}).get("present")]
    return sorted(got, key=lambda t: t[1]["factor"])


def levels_present(S):
    return [l for l in LEVELS if present_arms(S, l)]


def partial_note(fig, S):
    """Say so on the figure itself whenever the campaign is not complete."""
    p = S.get("progress", {})
    n, N = p.get("n_grids_present"), p.get("n_grids_expected")
    if n is not None and N and n < N:
        fig.text(0.995, -0.01, f"partial campaign: {n} / {N} grids on disk",
                 ha="right", va="top", color=INK_MUTED, fontsize=7.5)


def f_marginal(path):
    """(f grid, normalised marginal posterior) from a joint scan .h5."""
    p = Path(path)
    if not p.exists():
        return None
    with h5py.File(p, "r") as h:
        f = np.asarray(h["f_grid"][:], float)
        H0 = np.asarray(h["H0_grid"][:], float)
        ll = np.asarray(h["log_likelihood"][:], float)
    fin = np.isfinite(ll)
    if not fin.any():
        return None
    L = np.where(fin, np.exp(ll - ll[fin].max()), 0.0)
    P = np.trapz(L, H0, axis=0)          # marginalise H0, flat prior
    area = np.trapz(P, f)
    return f, (P / area if area > 0 else P)


# --------------------------------------------------------------------------
def fig_anchor_response():
    """f_AGN and H0 against the anchored AGN density, per survey depth."""
    S = summary()
    if not S:
        print("skip fig_anchor_response: no arms_summary.json")
        return
    levs = levels_present(S)
    if not levs:
        print("skip fig_anchor_response: no arms on disk")
        return
    f_real = S["truth"]["f_realised"]

    fig, axes = plt.subplots(2, 1, figsize=(6.4, 6.2), sharex=True,
                             gridspec_kw={"hspace": 0.13})
    for ax, key, truth, lab in (
            (axes[0], "f_vs_realised", f_real, r"$f_{\rm AGN}$"),
            (axes[1], "H0", H0_TRUTH, r"$H_0$  [km s$^{-1}$ Mpc$^{-1}$]")):
        ax.axhline(truth, color=INK_MUTED, lw=1.0, ls="--", zorder=1)
        for lev in levs:
            got = present_arms(S, lev)
            x = np.array([r["factor"] for _, r in got])
            y = np.array([r[key]["median"] for _, r in got])
            lo = np.array([r[key]["median"] - r[key]["minus68"] for _, r in got])
            hi = np.array([r[key]["median"] + r[key]["plus68"] for _, r in got])
            c = LEVEL_COLOR[lev]
            ax.fill_between(x, lo, hi, color=c, alpha=0.13, lw=0, zorder=2)
            ax.plot(x, y, "-o", color=c, ms=3.6, lw=1.4, zorder=3,
                    label=PRETTY[lev])
            # the exact arm, marked
            for a, r in got:
                if a == "exact":
                    ax.plot([r["factor"]], [r[key]["median"]], "o", ms=7.5,
                            mfc="none", mec=c, mew=1.4, zorder=4)
        ax.set_ylabel(lab)
        factor_axis(ax, all_factors(S))
        tidy(ax)
    facs = all_factors(S)
    axes[0].annotate("truth", xy=(min(facs), f_real), xytext=(2, 2),
                     textcoords="offset points", color=INK_MUTED, fontsize=8)
    axes[1].annotate("truth", xy=(min(facs), H0_TRUTH), xytext=(2, 2),
                     textcoords="offset points", color=INK_MUTED, fontsize=8)
    axes[1].set_xlabel(r"assumed AGN density / true AGN density"
                       "\n"
                       r"($\log_{10} n_{0,\rm AGN} = -5 + \log_{10}$ factor;"
                       r"  open circle = anchored at truth)")
    axes[0].legend(title="galaxy survey depth", ncol=2, loc="upper left",
                   fontsize=8.5, title_fontsize=8.5)
    axes[0].set_title("The completion's AGN density anchor sets the recovered "
                      "AGN fraction", loc="left")
    partial_note(fig, S)
    save(fig, "fig_anchor_response")


def fig_anchor_budget():
    """Shift from the exact arm in units of that arm's own 68 % half-width."""
    S = summary()
    if not S:
        print("skip fig_anchor_budget: no arms_summary.json")
        return
    levs = [l for l in levels_present(S)
            if any(r.get("vs_exact") for _, r in present_arms(S, l))]
    if not levs:
        print("skip fig_anchor_budget: no non-exact arms with a reference")
        return

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.5), sharex=True)
    for ax, key, lab in ((axes[0], "f", r"$f_{\rm AGN}$"),
                         (axes[1], "H0", r"$H_0$")):
        ax.axhline(0, color=INK_MUTED, lw=1.0, zorder=1)
        for s, ls in ((1, ":"), (-1, ":")):
            ax.axhline(s, color=INK_MUTED, lw=0.8, ls=ls, zorder=1)
        for lev in levs:
            pts = [(r["factor"], r["vs_exact"][key]["delta_median_over_ref_halfwidth68"])
                   for _, r in present_arms(S, lev) if r.get("vs_exact")]
            if not pts:
                continue
            x, y = np.array([p[0] for p in pts]), np.array([p[1] for p in pts])
            ax.plot(x, y, "-o", color=LEVEL_COLOR[lev], ms=3.6, lw=1.4,
                    zorder=3, label=PRETTY[lev])
        factor_axis(ax, all_factors(S))
        ax.set_title(lab + " shift", loc="left")
        ax.set_xlabel("assumed / true AGN density")
        tidy(ax)
    axes[0].set_ylabel("shift from the exact anchor\n"
                       r"[units of that arm's own $68\%$ half-width]")
    axes[0].legend(fontsize=8.5, loc="upper left")
    fig.text(0.5, 1.02, "Anchoring error against the statistical error it has to "
             r"beat ($\pm 1$ = the whole $68\%$ half-width)",
             ha="center", fontsize=10.5)
    fig.subplots_adjust(wspace=0.28)
    partial_note(fig, S)
    save(fig, "fig_anchor_budget")


def fig_anchor_significance():
    """Detection significance of f_AGN, and the posterior width, vs the anchor."""
    S = summary()
    if not S:
        print("skip fig_anchor_significance: no arms_summary.json")
        return
    levs = levels_present(S)
    if not levs:
        print("skip fig_anchor_significance: no arms on disk")
        return

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.5))
    for lev in levs:
        got = present_arms(S, lev)
        x = np.array([r["factor"] for _, r in got])
        sig = np.array([r["significance_f"] for _, r in got])
        wid = np.array([r["f_vs_realised"]["halfwidth68"] for _, r in got])
        c = LEVEL_COLOR[lev]
        axes[0].plot(x, sig, "-o", color=c, ms=3.6, lw=1.4, label=PRETTY[lev])
        axes[1].plot(x, wid, "-o", color=c, ms=3.6, lw=1.4, label=PRETTY[lev])
        for a, r in got:
            if a == "exact":
                axes[0].plot([r["factor"]], [r["significance_f"]], "o", ms=7.5,
                             mfc="none", mec=c, mew=1.4)
                axes[1].plot([r["factor"]],
                             [r["f_vs_realised"]["halfwidth68"]], "o", ms=7.5,
                             mfc="none", mec=c, mew=1.4)
    axes[0].set_ylabel(r"$f_{\rm AGN}$ significance"
                       "\n"
                       r"[median / $68\%$ half-width]")
    axes[0].set_ylim(bottom=0)
    axes[1].set_ylabel(r"$\sigma(f_{\rm AGN})$  [$68\%$ half-width]")
    axes[1].set_ylim(bottom=0)
    for ax in axes:
        factor_axis(ax, all_factors(S))
        ax.set_xlabel("assumed / true AGN density")
        tidy(ax)
    axes[0].legend(fontsize=8.5, loc="lower right")
    fig.text(0.5, 1.02, "The anchor moves the AGN fraction and its error together, "
             "so the significance barely moves", ha="center", fontsize=10.5)
    fig.subplots_adjust(wspace=0.32)
    partial_note(fig, S)
    save(fig, "fig_anchor_significance")


def fig_anchor_posteriors():
    """The marginal f_AGN posteriors themselves, one panel per rung."""
    S = summary()
    if not S:
        print("skip fig_anchor_posteriors: no arms_summary.json")
        return
    levs = levels_present(S)
    if not levs:
        print("skip fig_anchor_posteriors: no arms on disk")
        return
    f_real = S["truth"]["f_realised"]

    # one colour scale, and one legend, across ALL panels -- so a rung whose arms
    # are still in the queue is not silently given its own scale
    facs = all_factors(S)
    n = len(levs)
    fig, axes = plt.subplots(1, n, figsize=(2.7 * n + 0.6, 3.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, lev in zip(axes, levs):
        for a, r in present_arms(S, lev):
            cur = f_marginal(Path(r["path"]).with_suffix(".h5"))
            if cur is None:
                continue
            ax.plot(*cur, color=arm_color(r["factor"], facs),
                    lw=2.0 if a == "exact" else 1.2,
                    zorder=4 if a == "exact" else 3)
        ax.axvline(f_real, color=INK_MUTED, lw=1.0, ls="--", zorder=1)
        ax.set_title(PRETTY[lev], loc="left")
        ax.set_xlabel(r"$f_{\rm AGN}$")
        ax.set_xlim(0, 0.8)
        tidy(ax)
    axes[0].set_ylabel(r"posterior density")
    handles = [Line2D([], [], color=arm_color(v, facs),
                      lw=2.0 if v == 1.0 else 1.2,
                      label=("truth anchor" if v == 1.0 else rf"$\times\,{v:g}$"))
               for v in facs]
    axes[-1].legend(handles=handles, fontsize=7.5, loc="upper right",
                    title="assumed AGN density", title_fontsize=7.5,
                    labelspacing=0.35)
    fig.text(0.5, 1.03, r"Marginal $f_{\rm AGN}$ posteriors under a mis-anchored "
             "AGN density (dashed: the realised host fraction)",
             ha="center", fontsize=10.5)
    fig.subplots_adjust(wspace=0.12)
    partial_note(fig, S)
    save(fig, "fig_anchor_posteriors")


def fig_oracle_m18():
    """The oracle probe: hand the model every AGN host at the faintest rung."""
    S = summary()
    if not S:
        print("skip fig_oracle_m18: no arms_summary.json")
        return
    O = S.get("oracle") or {}
    if not O.get("present"):
        print("skip fig_oracle_m18: the oracle grid is not on disk yet")
        return
    f_real = S["truth"]["f_realised"]
    m18 = S["rungs"].get("m18", {}).get("arms", {}).get("exact")

    curves = []
    if m18 and m18.get("present"):
        c = f_marginal(Path(m18["path"]).with_suffix(".h5"))
        if c:
            curves.append((r"$m<18$, AGN survey $m<18$ too", ORANGE, c, m18))
    c = f_marginal(Path(O.get("path", RES / "joint_m18_oracle_s100.h5")).with_suffix(".h5"))
    if c:
        curves.append((r"$m<18$, AGN survey complete", BLUE, c, O))
    if not curves:
        print("skip fig_oracle_m18: no posterior curves")
        return

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4),
                             gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    for lab, col, (x, P), row in curves:
        ax.plot(x, P, color=col, lw=1.8, label=lab)
        ax.fill_between(x, 0, P,
                        where=(x >= row["f_vs_realised"]["ci68"][0])
                        & (x <= row["f_vs_realised"]["ci68"][1]),
                        color=col, alpha=0.13, lw=0)
    ax.axvline(f_real, color=INK_MUTED, lw=1.0, ls="--")
    ax.annotate("realised", xy=(f_real, ax.get_ylim()[1]), xytext=(3, -10),
                textcoords="offset points", color=INK_MUTED, fontsize=8)
    ax.set_xlabel(r"$f_{\rm AGN}$")
    ax.set_ylabel("posterior density")
    ax.set_xlim(0, 0.8)
    ax.legend(fontsize=8.5, loc="upper right")
    ax.set_title("Faintest rung, AGN completion removed", loc="left")
    tidy(ax)

    ax = axes[1]
    labs, offs, errs, cols = [], [], [], []
    for lab, col, _, row in curves:
        labs.append("sparse AGN" if col == ORANGE else "AGN complete")
        offs.append(row["f_vs_realised"]["offset"])
        errs.append(row["f_vs_realised"]["halfwidth68"])
        cols.append(col)
    y = np.arange(len(labs))[::-1]
    ax.errorbar(offs, y, xerr=errs, fmt="o", ms=5, lw=1.4,
                ecolor=INK_2, mfc=SURFACE, mec=INK, zorder=3)
    for yi, o, c in zip(y, offs, cols):
        ax.plot([o], [yi], "o", ms=5, color=c, zorder=4)
    ax.axvline(0, color=INK_MUTED, lw=1.0, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(labs)
    ax.set_ylim(-0.6, len(labs) - 0.4)
    ax.set_xlabel(r"$f_{\rm AGN}$ offset from the realised fraction")
    ax.set_title("Is the bias the sparse completion?", loc="left")
    tidy(ax, grid_axis="x")
    if O.get("bias_removed_fraction") is not None:
        ax.annotate(f"{100 * O['bias_removed_fraction']:.0f} % of the bias removed",
                    xy=(0.02, 0.06), xycoords="axes fraction",
                    color=INK_2, fontsize=8.5)
    fig.subplots_adjust(wspace=0.35)
    partial_note(fig, S)
    save(fig, "fig_oracle_m18")


def main():
    fig_anchor_response()
    fig_anchor_budget()
    fig_anchor_significance()
    fig_anchor_posteriors()
    fig_oracle_m18()


if __name__ == "__main__":
    main()
