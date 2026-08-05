"""One tracer at a time, at matched event count (appendix figure).

    analyses/analysis_0_pure_tracer_H0/results/h0_pure{gal,agn}_targeted_s{S}.h5
    analyses/analysis_0_pure_tracer_H0/results/h0_pure{gal,agn}_targeted_s{S}.json
    analyses/analysis_0_pure_tracer_H0/results/h0_pure_tracer.json

Ten event sets, two per realisation: one in which every host is a galaxy and
one in which every host is an AGN, each of 1000 detected events, each analysed
against its own catalog alone.  The pair is what makes the two tracers'
constraining power comparable -- the single-catalog fits of the main text split
one mixed event set, so their arms differ in size and share their noise.

Left panel: the ten posteriors as normalised densities with units on the y
axis, the field convention.  The AGN densities are ~5x narrower and
correspondingly taller (peaks 0.8-1.1 against 0.10-0.26 for the galaxies), a
ratio one shared linear axis still resolves, so no stacking and no rescaling.
Hue carries the tracer (blue galaxies, orange AGN, the same assignment as
fig_single_tracer); within a tracer the reference realisation is drawn at full
strength and the other four lighter and thinner, so the family is one colour
rather than five.

    One galaxy realisation is genuinely bimodal -- its posterior has a second
    mode near H0 = 62 at 0.70 of the peak.  It is drawn as it is: nothing is
    smoothed, and the x range is chosen to contain it.

X range.  [50, 100] is scanned, but every curve's density outside [56, 80] is
below 3e-4 of its own peak and each curve keeps >= 99.99 % of its mass inside,
so the window crops no support and spends the width on the part of the range
the data occupy.

Right panel: the same ten measurements as medians with their 68 % intervals,
against truth, with each tracer's five-realisation mean offset and standard
error as a band.  The wide galaxy bar on the bimodal realisation is the honest
rendering of an interval that has to span the gap between two modes.

Colour.  Slots #2a78d6 / #eb6834 and their 55 %-strength composites on the page
(#8ab5e8 / #f4ac8f) were checked all-pairs with the palette validator's
conventions: worst normal-vision OKLab dE 15.8, worst min(protan, deutan) 13.2,
both above the 15 / 8 gates.  The two light steps sit at 2.1:1 and 1.9:1
against the page, below the 3:1 relief line, so they never carry identity
alone -- the legend names them, the full-strength curve of the same hue is
beside them, and the bimodal one is annotated directly.
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import figstyle as fs

GAL = fs.C["blue"]
AGN = fs.C["orange"]
FADE = 0.55           # strength of the four non-reference realisations
BIMODAL_SEED = 105    # the galaxy realisation with a second mode

XLO, XHI = 56.0, 80.0


def scan(tracer, seed):
    """(grid, normalised posterior density, quoted H0 summary) for one scan."""
    tag = f"h0_pure{tracer}_targeted_s{seed}"
    grid, logl = fs.scan_1d(fs.A0 / f"{tag}.h5", "H0_grid")
    p = fs.posterior_1d(grid, logl)
    meta = json.loads((fs.A0 / f"{tag}.json").read_text())["H0"]
    return grid, p, meta


def panel_posteriors(ax, curves):
    """The ten posteriors, as normalised densities on one shared axis."""
    for tracer, colour in (("gal", GAL), ("agn", AGN)):
        for seed in fs.SEEDS:                       # others first, reference on top
            if seed == fs.REF_SEED:
                continue
            x, y, _ = curves[tracer, seed]
            ax.plot(x, y, color=colour, lw=0.9, alpha=FADE, zorder=3)
        x, y, _ = curves[tracer, fs.REF_SEED]
        ax.plot(x, y, color=colour, lw=1.8, zorder=5)

    fs.truth_line(ax, fs.H0_TRUTH, axis="x")
    ax.annotate("truth 67.74", (fs.H0_TRUTH, 0.55),
                xycoords=("data", "axes fraction"), textcoords="offset points",
                xytext=(-4, 0), ha="right", va="center", fontsize=7.0,
                color=fs.INK2)

    # the second mode, named on the curve that has it
    x, y, _ = curves["gal", BIMODAL_SEED]
    k = int(np.argmin(np.abs(x - 62.0)))
    ax.annotate("one galaxy\nrealisation is\nbimodal", (x[k], y[k]),
                textcoords="offset points", xytext=(-6, 22), ha="right",
                va="bottom", fontsize=7.0, color=fs.INK2, linespacing=1.35,
                arrowprops=dict(arrowstyle="-", color=fs.MUTED, lw=0.7,
                                shrinkA=1, shrinkB=2), zorder=6)

    ax.set_xlim(XLO, XHI)
    ax.set_ylim(0, 1.52)
    ax.set_yticks([0, 0.5, 1.0, 1.5])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(r"$H_0$  [km s$^{-1}$ Mpc$^{-1}$]")
    ax.set_ylabel(r"$p(H_0 \mid d)$  [km$^{-1}$ s Mpc]")
    ax.legend(handles=[
        Line2D([], [], color=GAL, lw=1.8, label="galaxies, reference"),
        Line2D([], [], color=AGN, lw=1.8, label="AGN, reference"),
        Line2D([], [], color=GAL, lw=0.9, alpha=FADE, label="four others"),
        Line2D([], [], color=AGN, lw=0.9, alpha=FADE, label="four others"),
    ], loc="upper left", ncol=2, columnspacing=1.1, fontsize=7.0,
        handlelength=1.4, labelspacing=0.3, borderaxespad=0.2)


def panel_recovery(ax, curves, summary):
    """Medians and 68 % intervals per realisation, with the mean-offset bands."""
    x = np.arange(len(fs.SEEDS))
    for tracer, colour, dx, block in (("gal", GAL, -0.13, "closure_gal"),
                                      ("agn", AGN, +0.13, "closure_agn")):
        med = np.array([curves[tracer, s][2]["median"] for s in fs.SEEDS])
        lo = med - np.array([curves[tracer, s][2]["ci68"][0] for s in fs.SEEDS])
        hi = np.array([curves[tracer, s][2]["ci68"][1] for s in fs.SEEDS]) - med
        c = summary[block]
        ax.axhspan(fs.H0_TRUTH + c["mean_offset"] - c["sem_offset"],
                   fs.H0_TRUTH + c["mean_offset"] + c["sem_offset"],
                   color=colour, alpha=0.16, lw=0, zorder=2)
        ax.axhline(fs.H0_TRUTH + c["mean_offset"], color=colour, lw=0.9,
                   zorder=3)
        ax.errorbar(x + dx, med, yerr=[lo, hi], fmt="o", ms=4.0, color=colour,
                    ecolor=colour, elinewidth=1.3, capsize=0, zorder=5,
                    markeredgecolor="white", markeredgewidth=0.7)

    # truth is drawn over the two mean lines: the AGN mean offset is -0.001, so
    # its line coincides with truth and must not be able to hide it
    fs.truth_line(ax, fs.H0_TRUTH, axis="y")
    ax.lines[-1].set_zorder(4)

    ax.grid(axis="x", visible=False)
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in fs.SEEDS])
    ax.set_xlim(-0.5, len(fs.SEEDS) - 0.5)
    ax.set_xlabel("realisation")
    ax.set_ylabel(r"$H_0$  [km s$^{-1}$ Mpc$^{-1}$]")
    ax.margins(y=0.10)
    ylo, yhi = ax.get_ylim()
    ax.set_ylim(ylo, yhi + 2.1)          # headroom for the legend block
    ax.legend(handles=[
        Line2D([], [], color=GAL, lw=1.3, marker="o", ms=4.0,
               markeredgecolor="white", markeredgewidth=0.7,
               label=r"galaxies, median $\pm$ 68 %"),
        Line2D([], [], color=AGN, lw=1.3, marker="o", ms=4.0,
               markeredgecolor="white", markeredgewidth=0.7,
               label=r"AGN, median $\pm$ 68 %"),
        Line2D([], [], color=fs.TRUTH, lw=0.9, ls=(0, (3, 2)), alpha=0.75,
               label="truth 67.74"),
        # matplotlib fills a multi-column legend column-major, so the two band
        # swatches are listed last to land opposite their own marker rows
        Patch(facecolor=GAL, alpha=0.16, edgecolor="none",
              label=r"mean offset $\pm$ s.e."),
        Patch(facecolor=AGN, alpha=0.16, edgecolor="none",
              label=r"mean offset $\pm$ s.e."),
    ], loc="upper left", ncol=2, columnspacing=1.0, fontsize=7.0,
        handlelength=1.4, labelspacing=0.28, borderaxespad=0.2)


def build():
    fs.use()
    summary = json.loads((fs.A0 / "h0_pure_tracer.json").read_text())
    curves = {(t, s): scan(t, s) for t in ("gal", "agn") for s in fs.SEEDS}

    fig, axes = plt.subplots(1, 2, figsize=(fs.TWOCOL, 2.7),
                             gridspec_kw={"width_ratios": [1.18, 1.0]})
    panel_posteriors(axes[0], curves)
    panel_recovery(axes[1], curves, summary)
    fig.tight_layout(pad=0.3, w_pad=1.6)
    return fig


def main():
    fig = build()
    fs.save(fig, "fig_pure_tracer")
    plt.close(fig)


if __name__ == "__main__":
    main()
