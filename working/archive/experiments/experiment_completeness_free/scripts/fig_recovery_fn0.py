#!/usr/bin/env python3
"""Explicit recovery figures for the density-knowledge family.

  figs/fig_joint_fn0_ladder_fix.{pdf,png}   -- 68/90% credible regions in the
      (f_AGN, log10 n0_AGN) plane per rung; truth cross at the banana's upper tip
  figs/fig_f_recovery_arms_fix.{pdf,png}    -- f_AGN marginals per rung under
      four density-knowledge arms (exact / 10% / factor 2 / free)
  figs/fig_n0_recovery_fix.{pdf,png}        -- flat-prior log10 n0_AGN marginals
      per rung; the density recovers low everywhere, and rails on the scan edge
      at the two most complete rungs
  figs/fig_f_recovery_prepost.{pdf,png}     -- detection significance per
      rung/arm, pre- vs post-repair generator (bar pairs)

Everything is computed from results/fn0_<lev>{,_fix}.h5 and
results/n0_arms_summary{,_fix}.json -- no hand-typed numbers.  Arm reweighting
reproduces analyze_n0_arms.py exactly: p(f) propto int dg L(f,g) pi(g), with
pi a Gaussian of width log10(1+eps) dex about the truth, a delta (nearest
grid slice) for the exact arm, or flat over the scanned range.
"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

BASE = Path(__file__).resolve().parent.parent
RESULTS, FIGS = BASE / "results", BASE / "figs"
FIGS.mkdir(parents=True, exist_ok=True)
INC = BASE.parent / "experiment_twotracer_incomplete"

SFX = "_fix"  # the measurement of record
LEVELS = ["complete", "m21.0", "m20.0", "m19.0", "m18.0"]
LABELS = {"complete": "complete", "m21.0": "$m<21$", "m20.0": "$m<20$",
          "m19.0": "$m<19$", "m18.0": "$m<18$"}
G_TRUE, TRUTH_F = -7.720033, 0.30
EDGE_FLAG = 0.05          # flat-prior n0 edge mass above this => range-dependent

BLUE, AQUA, YELLOW, RED, INK, INK2, INK3 = ("#2a78d6", "#1baf7a", "#eda100",
                                            "#e34948", "#0b0b0b", "#52514e", "#898781")
GRIDCOL = "#e1e0d9"
RAMP = ["#0b0b0b", "#2a78d6", "#1baf7a", "#eda100", "#e34948"]
ARMS_ALL = ["fixed", "5%", "10%", "30%", "factor 2", "free"]
ARM_COL = {"fixed": "#0b0b0b", "5%": "#2a78d6", "10%": "#1baf7a",
           "30%": "#eda100", "factor 2": "#e34948", "free": "#898781"}
ARM_LAB = {"fixed": r"$n_0$ known exactly", "5%": r"$n_0$ to 5%",
           "10%": r"$n_0$ to 10%", "30%": r"$n_0$ to 30%",
           "factor 2": r"$n_0$ to a factor 2", "free": r"$n_0$ free"}
# fractional uncertainty per arm (dex width = log10(1+eps)); None = flat
ARM_FRAC = {"fixed": 0.0, "5%": 0.05, "10%": 0.10, "30%": 0.30,
            "factor 2": 1.0, "free": None}
SHOW_ARMS = ["fixed", "10%", "factor 2", "free"]   # the explicit-recovery subset

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


def load_grid(lev, sfx=None):
    with h5py.File(RESULTS / f"fn0_{lev}{SFX if sfx is None else sfx}.h5", "r") as f:
        return f["f_grid"][:], f["n0c2_grid"][:], f["log_likelihood"][:]


def like(ll):
    ok = np.isfinite(ll)
    return np.where(ok, np.exp(ll - np.nanmax(ll[ok])), 0.0)


def arm_marginal(fv, gv, L, arm):
    """f_AGN marginal under one density-knowledge arm (analyze_n0_arms.py logic)."""
    frac = ARM_FRAC[arm]
    if frac is None:
        pf = np.trapz(L, gv, axis=1)
    elif frac == 0.0:
        pf = L[:, int(np.argmin(np.abs(gv - G_TRUE)))]
    else:
        sg = float(np.log10(1.0 + frac))
        pf = np.trapz(L * np.exp(-0.5 * ((gv - G_TRUE) / sg) ** 2)[None, :], gv, axis=1)
    return pf / np.trapz(pf, fv)


def n0_marginal(fv, gv, L):
    """Flat-prior log10 n0 marginal and its low-edge mass (5% of the span)."""
    pg = np.trapz(L, fv, axis=0)
    pg = pg / np.trapz(pg, gv)
    lo = gv <= gv[0] + 0.05 * (gv[-1] - gv[0])
    return pg, float(np.trapz(pg[lo], gv[lo]))


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


def save(fig, stem):
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"{stem}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote figs/{stem}.{{pdf,png}}")


# --------------------------------------------------------------- joint plane
def fig_joint(have, C):
    fig, axes = plt.subplots(1, len(have), figsize=(2.05 * len(have), 2.9),
                             dpi=300, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, lev, col in zip(axes, have, RAMP):
        fv, gv, ll = load_grid(lev)
        L = like(ll)
        pw = L * np.outer(np.gradient(fv), np.gradient(gv))
        pw = pw / pw.sum()
        Fg, Gg = np.meshgrid(fv, gv, indexing="ij")
        lv = hpd_levels(pw)
        ax.contourf(Fg, Gg, pw, levels=[lv[1], lv[0], pw.max()],
                    colors=[col, col], alpha=0.22, zorder=3)
        ax.contour(Fg, Gg, pw, levels=sorted(lv), colors=[col], linewidths=1.0,
                   zorder=4)
        ax.axvline(TRUTH_F, color=INK2, lw=0.7, ls=(0, (1, 2.5)), zorder=2)
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
    axes[-1].annotate("truth sits at the\nbanana's upper tip",
                      xy=(TRUTH_F, G_TRUE), xytext=(0.48, G_TRUE - 0.95),
                      fontsize=7.2, color=INK2, ha="left",
                      arrowprops=dict(arrowstyle="-", color=INK3, lw=0.7,
                                      shrinkA=2, shrinkB=4))
    fig.suptitle(r"Joint $(f_{\rm AGN},\, \log_{10} n_{0,\rm AGN})$ recovery, "
                 "flat density prior — 68/90% credible regions "
                 "(cross = planted values)", fontsize=9)
    fig.tight_layout()
    save(fig, f"fig_joint_fn0_ladder{SFX}")


# ------------------------------------------------------------- f marginals
def fig_f_arms(have, C):
    fig, axes = plt.subplots(1, len(have), figsize=(2.05 * len(have), 2.7),
                             dpi=300, sharex=True, sharey=False)
    axes = np.atleast_1d(axes)
    for ax, lev in zip(axes, have):
        fv, gv, ll = load_grid(lev)
        L = like(ll)
        for arm in SHOW_ARMS:
            pf = arm_marginal(fv, gv, L, arm)
            ax.plot(fv, pf, color=ARM_COL[arm], lw=1.6, zorder=4,
                    label=ARM_LAB[arm])
            ax.fill_between(fv, 0, pf, color=ARM_COL[arm], alpha=0.10,
                            lw=0, zorder=3)
        ax.axvline(TRUTH_F, color=INK2, lw=1.0, ls=(0, (1, 2)), zorder=2)
        cw = C.get(lev)
        ax.set_title(LABELS[lev] + (f"\n$C\\simeq{cw:.2f}$" if cw else ""),
                     fontsize=8.6)
        ax.set_xlim(0, 0.8)
        ax.set_xticks([0, 0.3, 0.6])
        ax.set_ylim(bottom=0)
        ax.set_yticks([])
        ax.set_xlabel(r"$f_{\rm AGN}$")
        ax.grid(True, axis="x", alpha=0.4)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("posterior density")
    axes[0].legend(loc="upper right", fontsize=6.8, handlelength=1.4,
                   borderaxespad=0.25)
    fig.suptitle(r"$f_{\rm AGN}$ recovery vs. density knowledge "
                 "(dotted = planted value; exact/10% recover high, "
                 "free recovers low)", fontsize=9)
    fig.tight_layout()
    save(fig, f"fig_f_recovery_arms{SFX}")


# ------------------------------------------------------------ n0 marginals
def fig_n0(have, C):
    fig, ax = plt.subplots(figsize=(5.4, 3.2), dpi=300)
    flagged, pmax = [], 0.0
    for lev, col in zip(have, RAMP):
        fv, gv, ll = load_grid(lev)
        pg, edge_lo = n0_marginal(fv, gv, like(ll))
        pmax = max(pmax, float(pg.max()))
        cw = C.get(lev)
        lab = LABELS[lev] + (f" ($C\\simeq{cw:.2f}$)" if cw else "")
        if edge_lo > EDGE_FLAG:
            flagged.append((lev, col, edge_lo, pg[0]))
            lab += r"$^{\dagger}$"
        ax.plot(gv, pg, color=col, lw=1.6, zorder=4, label=lab)
        ax.fill_between(gv, 0, pg, color=col, alpha=0.08, lw=0, zorder=3)
        med = float(np.interp(0.5, np.concatenate(
            [[0.0], np.cumsum(0.5 * (pg[1:] + pg[:-1]) * np.diff(gv))]), gv))
        ax.plot([med], [np.interp(med, gv, pg)], marker="o", ms=3.0, color=col,
                mec="white", mew=0.5, zorder=5)
    ax.axvline(G_TRUE, color=INK2, lw=1.0, ls=(0, (1, 2)), zorder=2)
    ax.annotate(f"truth\n{G_TRUE:.2f}", xy=(G_TRUE, 1.0),
                xycoords=("data", "axes fraction"), xytext=(G_TRUE + 0.03, 0.97),
                textcoords=("data", "axes fraction"), fontsize=7.2, color=INK2,
                ha="left", va="top")
    if flagged:
        pct = " / ".join(f"{100 * e:.0f}%" for _, _, e, _ in flagged)
        who = " / ".join(LABELS[l] for l, _, _, _ in flagged)
        ax.annotate(rf"$\dagger$ {who}: {pct} of the mass piles on the low"
                    "\nscan edge — the flat-prior width is range-dependent",
                    xy=(0.02, 0.985), xycoords="axes fraction",
                    fontsize=7.0, color=INK2, ha="left", va="top")
    ax.set_xlim(-9.6, -7.1)
    ax.set_ylim(0, 1.24 * pmax)
    ax.set_xlabel(r"$\log_{10} n_{0,\rm AGN}$   (flat prior, $f_{\rm AGN}$ marginalised)")
    ax.set_ylabel("posterior density")
    ax.set_title("The density recovers low at every rung (dot = median)")
    ax.grid(True, axis="x", alpha=0.45)
    ax.set_axisbelow(True)
    ax.legend(loc="center right", fontsize=6.8, handlelength=1.3,
              borderaxespad=0.3)
    fig.tight_layout()
    save(fig, f"fig_n0_recovery{SFX}")


# ------------------------------------------------------- pre/post comparison
def fig_prepost(C):
    pre_p = RESULTS / "n0_arms_summary.json"
    post_p = RESULTS / f"n0_arms_summary{SFX}.json"
    if not (pre_p.exists() and post_p.exists()):
        print("skipping fig_f_recovery_prepost (need both summaries)")
        return
    pre, post = (json.loads(p.read_text()) for p in (pre_p, post_p))
    have = [l for l in LEVELS if l in pre["levels"] and l in post["levels"]]
    xt = [f"{C[l]:.2f}" if l in C else LABELS[l] for l in have]

    fig, axes = plt.subplots(2, 3, figsize=(7.6, 4.4), dpi=300,
                             sharex=True, sharey=True)
    xi = np.arange(len(have), dtype=float)
    for ax, arm in zip(axes.ravel(), ARMS_ALL):
        col = ARM_COL[arm]
        for src, d, off, solid in (("pre", pre, -0.2, False),
                                   ("post", post, +0.2, True)):
            for i, lev in enumerate(have):
                b = d["levels"][lev]["arms"].get(arm)
                if not b or b["detection_sigma"] is None:
                    continue
                y = b["detection_sigma"]
                ax.bar(xi[i] + off, y, width=0.38,
                       facecolor=col if solid else "white",
                       edgecolor=col, linewidth=0.9,
                       hatch=None if solid else "///", zorder=4)
                g = d["levels"][lev].get("log10n0_agn_flat_prior") or {}
                if arm == "free" and g.get("edge_mass_low", 0.0) > EDGE_FLAG:
                    ax.annotate(r"$\dagger$", xy=(xi[i] + off, y), fontsize=7.5,
                                color=INK2, ha="center", va="bottom", zorder=6)
        ax.axhline(3.0, color=INK3, lw=0.9, ls=(0, (1, 2)), zorder=2)
        ax.set_title(ARM_LAB[arm], fontsize=8.8)
        ax.set_xticks(xi)
        ax.set_xticklabels(xt)
        ax.grid(True, axis="y", alpha=0.5)
        ax.set_axisbelow(True)
    axes[1, 2].annotate(r"$3\sigma$", xy=(0.985, 3.10),
                        xycoords=("axes fraction", "data"), fontsize=7.0,
                        color=INK2, ha="right", va="bottom")
    for ax in axes[1, :]:
        ax.set_xlabel(r"completeness $C(z\leq0.30)$")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"significance of $f_{\rm AGN}$" "\n" r"(median$/\sigma$)")
    fig.legend(handles=[
        Patch(facecolor="white", edgecolor=INK2, hatch="///", linewidth=0.9,
              label="pre-repair generator"),
        Patch(facecolor=INK2, edgecolor=INK2,
              label="post-repair generator (measurement of record)")],
        loc="upper center", ncol=2, fontsize=7.2, frameon=False,
        bbox_to_anchor=(0.5, 0.925))
    fig.suptitle("Detection significance per rung and density-knowledge arm, "
                 "before vs. after the generator repair\n"
                 r"($\dagger$: flat-prior $n_0$ rails on the scan edge — "
                 "significance is range-dependent)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    save(fig, "fig_f_recovery_prepost")


def main():
    C = completeness()
    have = [l for l in LEVELS if (RESULTS / f"fn0_{l}{SFX}.h5").exists()]
    fig_joint(have, C)
    fig_f_arms(have, C)
    fig_n0(have, C)
    fig_prepost(C)


if __name__ == "__main__":
    main()
