"""One tracer at a time, at matched event count, over the full realisation set.

    analyses/analysis_0_pure_tracer_H0/results/h0_pure{gal,agn}_targeted_s{S}.h5
    analyses/analysis_0_pure_tracer_H0/results/h0_pure_tracer_ens93.json

For every realisation of the simulated universe, two further event sets were
drawn on the same catalogs, one with every host a galaxy and one with every
host an AGN, each of 1000 detected events, each analysed against its own
catalog alone.  The pair is what makes the two tracers' constraining power
comparable: the single-catalog fits of the main text split one mixed event set,
so their arms differ in size and share their noise.

Left panel: every posterior as a normalised density with units on the y axis,
the field convention.  Hue carries the tracer (blue galaxies, orange AGN, the
same assignment as fig_single_tracer); the reference realisation is drawn at
full strength and the rest of the family light and thin in the same hue, so the
ensemble reads as one colour per tracer rather than as a hairball.  The AGN
densities are several times narrower and correspondingly taller, a ratio one
shared linear axis still resolves, so nothing is stacked and nothing rescaled.
Bimodal realisations are drawn as they are; nothing is smoothed.

Right panel: coverage against credible level, the calibration this many
realisations buys and five could not.  For each realisation the posterior's
cumulative probability at the input value is evaluated, and the curve is the
fraction of realisations whose equal-tailed interval at level x contains the
input value.  A method whose intervals are the right width lies on the
diagonal.  The grey band is the 90 % binomial range for the number of
realisations drawn, so a curve leaving it is a deficit the counting noise does
not explain.  The galaxy curve tracks the diagonal; the AGN curve sits below
it, which is the statement that the sparse tracer's intervals are too narrow.

X range of the left panel is set from the data rather than fixed, so no curve's
support is cropped by a window chosen for a smaller family.

Colour.  Slots #2a78d6 / #eb6834 and their light composites on the page were
checked all-pairs with the palette validator's conventions (worst normal-vision
OKLab dE 15.8, worst min(protan, deutan) 13.2, above the 15 / 8 gates).  The
light steps never carry identity alone: the legend names them and the
full-strength curve of the same hue sits beside them.
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
ENS_FILE = "h0_pure_tracer_ens93.json"
FAMILY_ALPHA = 0.13       # strength of a single non-reference realisation
FAMILY_LW = 0.55


def ensemble():
    """The aggregation of record for this figure, and the seeds behind it."""
    d = json.loads((fs.A0 / ENS_FILE).read_text())
    seeds = list(d["closure_gal"]["seeds"])
    assert list(d["closure_agn"]["seeds"]) == seeds, \
        "the two tracers must be compared on the same realisations"
    return d, seeds


def scan(tracer, seed):
    """(grid, normalised posterior density) for one realisation."""
    tag = f"h0_pure{tracer}_targeted_s{seed}"
    grid, logl = fs.scan_1d(fs.A0 / f"{tag}.h5", "H0_grid")
    return grid, fs.posterior_1d(grid, logl)


def truth_quantile(grid, p):
    """Posterior cumulative probability at the input value.

    This is the quantity a coverage curve is built from: if the intervals are
    the right width these are uniform on (0, 1) across realisations.
    """
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (p[1:] + p[:-1]) * np.diff(grid))])
    cdf /= cdf[-1]
    return float(np.interp(fs.H0_TRUTH, grid, cdf))


def panel_posteriors(ax, curves, seeds):
    """Every posterior, tracer by tracer, with the reference on top."""
    for tracer, colour in (("gal", GAL), ("agn", AGN)):
        for s in seeds:
            if s == fs.REF_SEED:
                continue
            x, y = curves[tracer, s]
            ax.plot(x, y, color=colour, lw=FAMILY_LW, alpha=FAMILY_ALPHA,
                    zorder=3, solid_capstyle="butt")
        x, y = curves[tracer, fs.REF_SEED]
        ax.plot(x, y, color=colour, lw=1.7, zorder=5)

    fs.truth_line(ax, fs.H0_TRUTH, axis="x")
    ax.annotate("input 67.74", (fs.H0_TRUTH, 0.60),
                xycoords=("data", "axes fraction"), textcoords="offset points",
                xytext=(-4, 0), ha="right", va="center", fontsize=7.0,
                color=fs.INK2)

    # window from the data: the smallest range holding 99.9 % of every curve
    los, his = [], []
    for (_, _), (x, y) in curves.items():
        c = np.concatenate([[0.0], np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(x))])
        c /= c[-1]
        los.append(np.interp(0.001, c, x))
        his.append(np.interp(0.999, c, x))
    ax.set_xlim(np.floor(min(los)), np.ceil(max(his)))
    ax.set_ylim(0, max(y.max() for _, y in curves.values()) * 1.32)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(r"$H_0$  [km s$^{-1}$ Mpc$^{-1}$]")
    ax.set_ylabel(r"$p(H_0 \mid d)$  [km$^{-1}$ s Mpc]")
    ax.legend(handles=[
        Line2D([], [], color=GAL, lw=1.7, label="galaxies, reference"),
        Line2D([], [], color=AGN, lw=1.7, label="AGN, reference"),
        Line2D([], [], color=GAL, lw=1.1, alpha=0.45, label="galaxies, all others"),
        Line2D([], [], color=AGN, lw=1.1, alpha=0.45, label="AGN, all others"),
    ], loc="upper left", ncol=2, columnspacing=1.0, fontsize=7.0,
        handlelength=1.4, labelspacing=0.3, borderaxespad=0.2)


def panel_coverage(ax, quant, n):
    """Coverage against credible level, with the binomial range for n draws."""
    lev = np.linspace(0.0, 1.0, 201)
    band_lo, band_hi = [], []
    for a in lev:
        sd = np.sqrt(max(a * (1.0 - a), 0.0) / n)
        band_lo.append(max(a - 1.645 * sd, 0.0))
        band_hi.append(min(a + 1.645 * sd, 1.0))
    ax.fill_between(lev, band_lo, band_hi, color=fs.OTHER, alpha=0.30, lw=0,
                    zorder=1.5)
    ax.plot(lev, lev, color=fs.TRUTH, lw=0.9, ls=(0, (3, 2)), alpha=0.75,
            zorder=2)

    for tracer, colour in (("gal", GAL), ("agn", AGN)):
        q = np.asarray(quant[tracer], float)
        cov = [(np.abs(q - 0.5) <= a / 2.0).mean() for a in lev]
        ax.plot(lev, cov, color=colour, lw=1.7, zorder=4)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.68, 0.90, 1.0])
    ax.set_xticklabels(["0", "0.25", "0.5", "0.68", "0.90", "1"])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("credible level")
    ax.set_ylabel("coverage of the input value")
    ax.legend(handles=[
        Line2D([], [], color=GAL, lw=1.7, label="galaxies"),
        Line2D([], [], color=AGN, lw=1.7, label="AGN"),
        Line2D([], [], color=fs.TRUTH, lw=0.9, ls=(0, (3, 2)), alpha=0.75,
               label="intervals the right width"),
        Patch(facecolor=fs.OTHER, alpha=0.30, edgecolor="none",
              label=r"90 % binomial range"),
    ], loc="upper left", fontsize=7.0, handlelength=1.4, labelspacing=0.3,
        borderaxespad=0.2)


def build():
    fs.use()
    d, seeds = ensemble()
    curves = {(t, s): scan(t, s) for t in ("gal", "agn") for s in seeds}
    quant = {t: [truth_quantile(*curves[t, s]) for s in seeds]
             for t in ("gal", "agn")}

    # the figure must show exactly the realisations the text counts
    n = len(seeds)
    assert n == d["closure_gal"]["n_seeds"] == d["closure_agn"]["n_seeds"]
    for t, blk in (("gal", "closure_gal"), ("agn", "closure_agn")):
        q = np.asarray(quant[t])
        for lvl, key in ((0.68, "n_truth_in_ci68"), (0.90, "n_truth_in_ci90")):
            drawn = int((np.abs(q - 0.5) <= lvl / 2.0).sum())
            quoted = int(d[blk]["coverage"][key])
            assert abs(drawn - quoted) <= 1, (
                f"{t} at {lvl}: figure shows {drawn}/{n}, "
                f"the aggregation says {quoted}/{n}")
        print(f"  {t}: {n} realisations, "
              f"{int((np.abs(q - 0.5) <= 0.34).sum())} inside 68 %, "
              f"{int((np.abs(q - 0.5) <= 0.45).sum())} inside 90 %")

    fig, axes = plt.subplots(1, 2, figsize=(fs.TWOCOL, 2.9),
                             gridspec_kw={"width_ratios": [1.20, 1.0]})
    panel_posteriors(axes[0], curves, seeds)
    panel_coverage(axes[1], quant, n)
    fig.tight_layout(pad=0.3, w_pad=1.8)
    return fig


def main():
    fig = build()
    fs.save(fig, "fig_pure_tracer")
    plt.close(fig)


if __name__ == "__main__":
    main()
