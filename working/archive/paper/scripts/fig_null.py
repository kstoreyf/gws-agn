"""The sky-shuffle null: what the AGN weight is actually measuring.

    analyses/analysis_2_complete_catalog_H0_fagn/results/fscan_s100.{h5,json}
    analyses/analysis_2_complete_catalog_H0_fagn/results/fscan_null_s100.{h5,json}

Both curves are the same one-dimensional scan of f_AGN at the true H0, on the
same 101-point grid, with every other parameter held fixed.  The null curve is
the identical analysis run on a copy of the events in which the per-event sky
samples (and the bookkeeping host columns that travel with them) have been
permuted between events, so each event keeps its own distance, masses, spin and
localisation area but no longer sits in its own host's patch of sky
(scripts/shuffle_event_sky.py in that analysis).

If the AGN weight came from anything other than host association -- the two
tracers' different global normalisations, say -- the permutation would leave it
untouched.  It does not: the record curve peaks at 0.27 and the shuffled curve
collapses onto zero.  Medians and intervals are read from the jsons.
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import figstyle as fs

REC = fs.C["blue"]
NUL = fs.C["orange"]


def curve(tag):
    grid, logl = fs.scan_1d(fs.A2 / f"{tag}.h5", "f_grid")
    meta = json.loads((fs.A2 / f"{tag}.json").read_text())["f"]
    return grid, fs.posterior_1d(grid, logl), meta


def build():
    fs.use()
    fx, fp, fm = curve("fscan_s100")
    nx, np_, nm = curve("fscan_null_s100")

    fig, ax = plt.subplots(figsize=(fs.ONECOL, 2.55))
    ax.grid(axis="y", visible=False)

    for x, p, meta, c, off in ((fx, fp, fm, REC, (0, 5, "center")),
                               (nx, np_, nm, NUL, (5, -1, "left"))):
        y = p / p.max()
        bx, by = fs.ci_band(x, y, *meta["ci68"])
        ax.fill_between(bx, 0, by, color=c, alpha=0.16, lw=0, zorder=2)
        ax.plot(x, y, color=c, lw=1.6, zorder=4)
        # the median, marked on its own curve so the number cannot drift
        ym = float(np.interp(meta["median"], x, y))
        ax.plot([meta["median"]], [ym], marker="o", ms=3.4, color=c,
                markeredgecolor="white", markeredgewidth=0.7, zorder=6)
        ax.annotate(f"{meta['median']:.3f}", (meta["median"], ym),
                    textcoords="offset points", xytext=off[:2], ha=off[2],
                    va="bottom", fontsize=7.4, color=fs.INK2, zorder=6)

    fs.truth_line(ax, fs.F_PLANTED, axis="x")
    ax.annotate("planted 0.30", (fs.F_PLANTED, 0.995),
                xycoords=("data", "axes fraction"),
                textcoords="offset points", xytext=(4, 0), ha="left",
                va="top", fontsize=7.0, color=fs.INK2)

    ax.set_xlim(0, 0.6)
    ax.set_ylim(0, 1.16)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_xlabel(r"$f_{\rm AGN}$")
    ax.set_ylabel("posterior, scaled to peak")
    ax.legend(handles=[Line2D([], [], color=REC, lw=1.6, label="as recorded"),
                       Line2D([], [], color=NUL, lw=1.6,
                              label="sky shuffled")],
              loc="upper right", fontsize=7.2, borderaxespad=0.3,
              handlelength=1.5, labelspacing=0.35)
    fig.tight_layout(pad=0.2)
    return fig


def main():
    fig = build()
    fs.save(fig, "fig_null")
    plt.close(fig)


if __name__ == "__main__":
    main()
