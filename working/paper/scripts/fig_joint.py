"""The joint (H0, f_AGN) posterior: 2-D credible regions and both marginals.

    analyses/analysis_2_complete_catalog_H0_fagn/results/joint_s{100,101,102,103,105}.h5
    analyses/analysis_2_complete_catalog_H0_fagn/results/joint_s100.json
    analyses/analysis_2_complete_catalog_H0_fagn/results/joint_summary.json

Each h5 holds the log-likelihood on the same (H0, f) grid -- 201 x 41 cells on
[50, 100] x [0, 1] -- evaluated with the K = 2 mixture on the complete galaxy
and AGN catalogs.  Priors are flat on both axes, so the posterior is the
exponentiated log-likelihood; the 68 % and 90 % regions are the
highest-posterior-density contours (fs.hpd_levels_2d).  The marginals are the
same grid integrated along the other axis.  Quoted intervals and the realised
AGN host fraction of each realisation are read from the result json rather than
recomputed.

The reduction (grid -> posterior -> HPD level -> zoom on the union of the 90 %
regions) is adapted from the analysis's own figure script,
analyses/analysis_2_complete_catalog_H0_fagn/scripts/make_figures.py, and
restyled onto the paper's visual system.

Two drawing decisions worth recording:

  * the reference realisation carries the one accent and the other four are
    folded into a single neutral ("Other"), rather than cycling four more hues
    that no reader needs to tell apart;
  * the horizontal reference is the realised AGN host fraction of the
    reference realisation, 0.295, not the input 0.30.  The two differ by a
    third of the mock's own binomial scatter and would draw as one line at this
    scale, so only the realised value is drawn and the input value is named
    in the same label.
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import figstyle as fs

REF = fs.C["blue"]

# labels that must stay legible where they cross a contour
LBLBOX = dict(facecolor="white", edgecolor="none", alpha=0.72, pad=1.0)


def load():
    grids, meta = {}, {}
    summ = json.loads((fs.A2 / "joint_summary.json").read_text())
    for row in summ["seeds"]:
        p = fs.A2 / f"joint_s{row['seed']}.h5"
        if p.exists():
            grids[row["seed"]] = fs.joint_grid(p)
            meta[row["seed"]] = row
    return grids, meta, summ


def build():
    fs.use()
    grids, meta, summ = load()
    ref = json.loads((fs.A2 / f"joint_s{fs.REF_SEED}.json").read_text())
    f_ref = meta[fs.REF_SEED]["f_realised"]

    fig = plt.figure(figsize=(fs.TWOCOL, 3.9))
    gs = fig.add_gridspec(2, 2, width_ratios=[3.4, 1.0],
                          height_ratios=[1.2, 2.8], wspace=0.045, hspace=0.045,
                          left=0.068, right=0.995, bottom=0.115, top=0.99)
    axm = fig.add_subplot(gs[1, 0])
    axt = fig.add_subplot(gs[0, 0], sharex=axm)
    axr = fig.add_subplot(gs[1, 1], sharey=axm)
    axk = fig.add_subplot(gs[0, 1])
    axk.axis("off")

    # ---- 2-D regions --------------------------------------------------------
    for s, (H0, fv, P) in grids.items():
        if s == fs.REF_SEED:
            continue
        _, l90 = fs.hpd_levels_2d(H0, fv, P)
        axm.contour(H0, fv, P.T, levels=[l90], colors=fs.OTHER,
                    linewidths=0.8, zorder=2)

    H0, fv, P = grids[fs.REF_SEED]
    l68, l90 = fs.hpd_levels_2d(H0, fv, P)
    axm.contourf(H0, fv, P.T, levels=[l90, l68], colors=[REF], alpha=0.13,
                 zorder=3)
    axm.contourf(H0, fv, P.T, levels=[l68, P.max() * 1.01], colors=[REF],
                 alpha=0.26, zorder=3)
    axm.contour(H0, fv, P.T, levels=[l90, l68], colors=REF,
                linewidths=[0.9, 1.5], zorder=4)

    # ---- truth --------------------------------------------------------------
    axm.axvline(fs.H0_TRUTH, color=fs.TRUTH, lw=0.8, ls=(0, (3, 2)),
                alpha=0.75, zorder=5)
    axm.axhline(f_ref, color=fs.TRUTH, lw=0.8, ls=(0, (3, 2)), alpha=0.75,
                zorder=5)
    axm.plot([fs.H0_TRUTH], [f_ref], marker="+", ms=9, mew=1.6, color=fs.INK,
             zorder=7)
    axm.plot([fs.H0_TRUTH] * (len(grids) - 1),
             [meta[s]["f_realised"] for s in grids if s != fs.REF_SEED],
             marker="_", ms=7, mew=1.1, ls="none", color=fs.OTHER, zorder=6)

    # ---- marginals ----------------------------------------------------------
    for s, (H0s, fvs, Ps) in grids.items():
        c, lw, z = (REF, 1.6, 5) if s == fs.REF_SEED else (fs.OTHER, 0.8, 3)
        pH0 = np.trapz(Ps, fvs, axis=1); pH0 /= np.trapz(pH0, H0s)
        pf = np.trapz(Ps, H0s, axis=0); pf /= np.trapz(pf, fvs)
        axt.plot(H0s, pH0, color=c, lw=lw, zorder=z)
        axr.plot(pf, fvs, color=c, lw=lw, zorder=z)
    axt.axvline(fs.H0_TRUTH, color=fs.TRUTH, lw=0.8, ls=(0, (3, 2)),
                alpha=0.75, zorder=6)
    axr.axhline(f_ref, color=fs.TRUTH, lw=0.8, ls=(0, (3, 2)), alpha=0.75,
                zorder=6)

    # ---- frame: zoom on the union of the 90 % regions ------------------------
    xs, ys = [], []
    for s, (H0s, fvs, Ps) in grids.items():
        _, l90s = fs.hpd_levels_2d(H0s, fvs, Ps)
        m = Ps >= l90s
        xs += [H0s[m.any(axis=1)].min(), H0s[m.any(axis=1)].max()]
        ys += [fvs[m.any(axis=0)].min(), fvs[m.any(axis=0)].max()]
    xlo, xhi = min(xs + [fs.H0_TRUTH]), max(xs + [fs.H0_TRUTH])
    ylo, yhi = min(ys + [fs.F_PLANTED]), max(ys + [fs.F_PLANTED])
    px, py = 0.14 * (xhi - xlo), 0.16 * (yhi - ylo)
    axm.set_xlim(xlo - px, xhi + px)
    axm.set_ylim(max(0.0, ylo - py), min(1.0, yhi + py))
    axm.set_xlabel(r"$H_0$  [km s$^{-1}$ Mpc$^{-1}$]")
    axm.set_ylabel(r"$f_{\rm AGN}$")

    # marginal panels are shape only: no frame, no scale, no ticks
    for ax in (axt, axr):
        ax.grid(visible=False)
        for side in ("top", "right", "left", "bottom"):
            ax.spines[side].set_visible(False)
        ax.tick_params(left=False, right=False, top=False, bottom=False,
                       labelleft=False, labelbottom=False)
    axt.set_ylim(bottom=0)
    axr.set_xlim(left=0)

    axm.annotate("truth 67.74", (fs.H0_TRUTH, 0.015),
                 xycoords=("data", "axes fraction"),
                 textcoords="offset points", xytext=(-4, 0), ha="right",
                 va="bottom", fontsize=7.0, color=fs.INK2, bbox=LBLBOX)
    axm.annotate(f"realised {f_ref:.3f}   (input {fs.F_PLANTED:.2f})",
                 (0.985, f_ref), xycoords=("axes fraction", "data"),
                 textcoords="offset points", xytext=(0, 3), ha="right",
                 va="bottom", fontsize=7.0, color=fs.INK2, bbox=LBLBOX)

    # ---- key, in the corner the corner plot leaves empty ---------------------
    axk.legend(handles=[
        Line2D([], [], color=REF, lw=1.5,
               label=f"seed {fs.REF_SEED}   68 %, 90 %"),
        Line2D([], [], color=fs.OTHER, lw=0.8,
               label=f"{len(grids) - 1} further realisations"),
        Line2D([], [], color=fs.INK, lw=0, marker="+", ms=8, mew=1.6,
               label="truth"),
    ], loc="upper left", fontsize=7.2, bbox_to_anchor=(-0.03, 1.04),
        handlelength=1.5, labelspacing=0.42, borderaxespad=0.0)

    h0, ff = ref["H0"], ref["f"]
    axk.text(-0.03, 0.30,
             f"$H_0 = {h0['median']:.1f}"
             f"^{{+{h0['ci68'][1] - h0['median']:.1f}}}"
             f"_{{-{h0['median'] - h0['ci68'][0]:.1f}}}$\n"
             f"$f_{{\\rm AGN}} = {ff['median']:.3f}"
             f"^{{+{ff['ci68'][1] - ff['median']:.3f}}}"
             f"_{{-{ff['median'] - ff['ci68'][0]:.3f}}}$",
             transform=axk.transAxes, ha="left", va="top", fontsize=7.4,
             color=fs.INK2, linespacing=1.9)
    return fig


def main():
    fig = build()
    fs.save(fig, "fig_joint")
    plt.close(fig)


if __name__ == "__main__":
    main()
