"""Single-catalog H0 posteriors on the mixed universe (seed 100).

Both curves are the same 1000 events analysed twice: once against the complete
galaxy catalog alone, once against the complete AGN catalog alone.  Neither is
the universe the events came from, and the figure's job is to show what that
mis-specification does.

    analyses/analysis_1_complete_catalog_H0/results/h0_gal_targeted.{h5,json}
    analyses/analysis_1_complete_catalog_H0/results/h0_agn_targeted.{h5,json}

The h5 carries `H0_grid` and `log_likelihood` on a flat prior, so the posterior
is exp(logL) renormalised; the json carries the quantiles the analysis quotes,
and they are read rather than recomputed so the figure cannot drift from the
numbers in the text.

Two honesty rules are wired into the drawing:

  * the AGN posterior is still rising at H0 = 100, the top of the scanned
    range.  Its 68 % interval is therefore an artefact of where the scan was
    stopped, so no band and no interval is drawn for it -- only the curve, the
    boundary rule, and a label saying so.
  * both curves are normalised posterior densities with units on the y axis,
    the field convention.  Their peaks differ by a factor of 14, so a shared
    linear axis would flatten the galaxy curve to a bump; instead the two
    tracers get vertically stacked panels sharing the x axis, each with its
    own density scale, and the truth line runs through both.
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt

import figstyle as fs

GAL = fs.C["blue"]
AGN = fs.C["orange"]

YLABEL = r"$p(H_0 \mid d)$  [km$^{-1}$ s Mpc]"


def curve(tag):
    grid, logl = fs.scan_1d(fs.A1 / f"{tag}.h5", "H0_grid")
    p = fs.posterior_1d(grid, logl)
    meta = json.loads((fs.A1 / f"{tag}.json").read_text())["H0"]
    return grid, p, meta


def build():
    fs.use()
    gal_x, gal_p, gal = curve("h0_gal_targeted")
    agn_x, agn_p, agn = curve("h0_agn_targeted")
    single = json.loads((fs.A1 / "h0_single_tracer.json").read_text())

    fig, (axg, axa) = plt.subplots(2, 1, sharex=True,
                                   figsize=(fs.ONECOL, 3.35),
                                   layout="constrained",
                                   gridspec_kw={"hspace": 0.06})
    for ax in (axg, axa):
        ax.grid(axis="y", visible=False)
        fs.truth_line(ax, fs.H0_TRUTH, axis="x", label=None)

    # ---- galaxies: a density with an interior maximum and a 68 % interval ---
    bx, by = fs.ci_band(gal_x, gal_p, *gal["ci68"])
    axg.fill_between(bx, 0, by, color=GAL, alpha=0.16, lw=0, zorder=2)
    axg.plot(gal_x, gal_p, color=GAL, lw=1.6, zorder=4)

    axg.annotate("truth 67.74", (fs.H0_TRUTH, 0.86),
                 xycoords=("data", "axes fraction"), textcoords="offset points",
                 xytext=(-4, 0), ha="right", va="center", fontsize=7.0,
                 color=fs.INK2)
    axg.annotate("galaxies", (0.965, 0.86), xycoords="axes fraction",
                 ha="right", va="center", fontsize=7.4, color=fs.INK2)

    # the one number the galaxy-only curve supports, labelled on the mark
    gal_lbl = (f"${gal['median']:.1f}"
               f"^{{+{gal['ci68'][1] - gal['median']:.1f}}}"
               f"_{{-{gal['median'] - gal['ci68'][0]:.1f}}}$")
    axg.annotate(gal_lbl, (gal_x[gal_p.argmax()], gal_p.max()),
                 textcoords="offset points", xytext=(6, 0), ha="left",
                 va="center", fontsize=7.4, color=fs.INK2)

    axg.set_ylim(0, 0.30)
    axg.set_yticks([0, 0.1, 0.2, 0.3])

    # ---- AGN: still rising at the top of the scanned range ------------------
    axa.plot(agn_x, agn_p, color=AGN, lw=1.6, zorder=4)
    axa.annotate("AGN", (0.04, 0.86), xycoords="axes fraction",
                 ha="left", va="center", fontsize=7.4, color=fs.INK2)

    # the curve runs into the edge of the scanned range: say so there
    assert single["agn_h0_ci"] is None and single["agn_railed_at_grid_top"]
    axa.annotate("rails at the\nedge of the\nscanned range",
                 (100.0, 0.52), xycoords=("data", "axes fraction"),
                 textcoords="offset points", xytext=(-6, 0), ha="right",
                 va="center", fontsize=7.0, color=fs.INK2, linespacing=1.3,
                 zorder=6)
    axa.annotate("", xy=(100.0, 0.95), xytext=(97.6, 0.95),
                 xycoords=("data", "axes fraction"),
                 arrowprops=dict(arrowstyle="-|>", color=AGN, lw=1.1,
                                 shrinkA=0, shrinkB=0), zorder=6)

    axa.set_ylim(0, 4.0)
    axa.set_yticks([0, 1, 2, 3, 4])
    axa.set_xlim(50, 100)
    axa.set_xlabel(r"$H_0$  [km s$^{-1}$ Mpc$^{-1}$]")

    fig.supylabel(YLABEL, fontsize=8.5)
    return fig


def main():
    fig = build()
    fs.save(fig, "fig_single_tracer")
    plt.close(fig)


if __name__ == "__main__":
    main()
