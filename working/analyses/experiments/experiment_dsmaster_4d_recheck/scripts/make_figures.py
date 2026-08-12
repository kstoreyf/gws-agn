#!/usr/bin/env python
"""Result figures for the completeness-estimator re-check. Deterministic.

    python make_figures.py            # writes figs/*.pdf and figs/*.png

fig1_corner   the three arms' joint posteriors over the four sampled
              parameters, with truth marked
fig2_offsets  the same result as offsets from truth in units of each arm's own
              68% half-width -- the "is it accurate" view the corner cannot show

Colours are a fixed categorical order (Okabe-Ito blue/orange/green), checked for
colour-vision separation before use: every pair clears normal-vision dE 15 and
CVD dE 8 (worst pair per_pixel/selection under tritanopia, 10.5). Truth is a
neutral black dashed rule, never a series colour, so it can never be mistaken
for a fourth arm.
"""
import json
from pathlib import Path

import corner
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
FIGS = EXP / "figs"

# fixed categorical order -- never cycled, never reassigned by rank
ARMS = [
    ("per_pixel", "#0072B2", "per_pixel (legacy)"),
    ("aggregate", "#E69F00", "aggregate"),
    ("selection", "#009E73", "selection"),
]
LABELS = [r"$H_0$", r"$\log_{10} n_0$ (GAL)",
          r"$\log_{10} n_0$ (AGN)", r"$f_{\rm AGN}$"]
KEYS = ["H0", "log10n0", "log10n0_c2", "f_AGN"]
TRUTHS = [67.74, -3.0, -5.0, 0.295]

INK = "#1a1a1a"
MUTED = "#6b6b6b"


def load():
    out = {}
    for tag, colour, disp in ARMS:
        h5 = EXP / "results" / f"fit_m18_{tag}_s100.h5"
        js = EXP / "results" / f"fit_m18_{tag}_s100.json"
        with h5py.File(h5, "r") as f:
            s = np.asarray(f["samples"])
        out[tag] = {"samples": s, "colour": colour, "disp": disp,
                    "summary": json.loads(js.read_text())["summary"],
                    "logz": json.loads(js.read_text())["sampler_meta"]["logz"]}
    return out


def style_axes(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelcolor=INK, width=0.8, labelsize=8)


def fig_corner(data):
    # widest arm first so later contours draw on top of it
    order = ["per_pixel", "aggregate", "selection"]
    ranges = []
    for j in range(4):
        allx = np.concatenate([data[t]["samples"][:, j] for t in order])
        lo, hi = np.percentile(allx, [0.5, 99.5])
        pad = 0.06 * (hi - lo)
        ranges.append((min(lo - pad, TRUTHS[j] - pad),
                       max(hi + pad, TRUTHS[j] + pad)))

    figure = None
    for tag in order:
        d = data[tag]
        figure = corner.corner(
            d["samples"], labels=LABELS, range=ranges, fig=figure,
            color=d["colour"], levels=(0.68, 0.90), bins=40,
            smooth=1.0, smooth1d=1.0,
            plot_datapoints=False, plot_density=False, fill_contours=False,
            # with smooth1d set, corner draws the 1-D panels with ax.plot, so
            # hist_kwargs reaches Line2D -- no hist-only keys (e.g. density).
            hist_kwargs=dict(lw=2.0),
            contour_kwargs=dict(linewidths=2.0),
            label_kwargs=dict(fontsize=11, color=INK),
        )

    axes = np.array(figure.axes).reshape((4, 4))
    for j in range(4):                       # truth: neutral, never a series hue
        for i in range(j, 4):
            axes[i, j].axvline(TRUTHS[j], color="black", ls="--", lw=1.2,
                               zorder=10)
        for i in range(j):
            axes[j, i].axhline(TRUTHS[j], color="black", ls="--", lw=1.2,
                               zorder=10)
        axes[j, j].axvline(TRUTHS[j], color="black", ls="--", lw=1.2, zorder=10)
    for ax in axes.ravel():
        if ax.get_visible():
            style_axes(ax)

    handles = [Line2D([], [], color=d[1], lw=2.0,
                      label=f"{d[2]}   ln Z = {data[d[0]]['logz']:.1f}")
               for d in ARMS]
    handles.append(Line2D([], [], color="black", ls="--", lw=1.2, label="truth"))
    figure.legend(handles=handles, loc="upper right",
                  bbox_to_anchor=(0.98, 0.98), frameon=False,
                  fontsize=10, labelcolor=INK)
    figure.suptitle(
        "Completeness estimator vs the 4-parameter posterior\n"
        "m18, seed 100, K=2 field mixture, darksirens 0c5b3db",
        fontsize=12, color=INK, x=0.5, y=1.005, ha="center")
    return figure


def fig_offsets(data):
    fig, axes = plt.subplots(1, 4, figsize=(11.5, 3.4))
    for j, (key, ax) in enumerate(zip(KEYS, axes)):
        ax.axvline(0, color="black", ls="--", lw=1.2, zorder=1)
        for i, (tag, colour, disp) in enumerate(ARMS):
            v = data[tag]["summary"][key]
            y = len(ARMS) - 1 - i
            hw = v["halfwidth68"]
            ax.errorbar(v["offset"], y, xerr=hw, fmt="o", color=colour,
                        ms=8, lw=2.0, capsize=0, zorder=3)
            ax.plot([v["offset"]], [y], "o", color="white", ms=3, zorder=4)
        ax.set_yticks(range(len(ARMS)))
        ax.set_yticklabels([a[2].split(" ")[0] for a in ARMS][::-1]
                           if j == 0 else [""] * len(ARMS), fontsize=9)
        ax.set_ylim(-0.6, len(ARMS) - 0.4)
        ax.set_title(LABELS[j], fontsize=11, color=INK)
        ax.set_xlabel("offset from truth", fontsize=9, color=MUTED)
        ax.grid(axis="x", color="#e6e6e6", lw=0.7, zorder=0)
        ax.set_axisbelow(True)
        style_axes(ax)
    fig.suptitle("Accuracy by estimator — offset from truth with the 68% "
                 "half-width; the dashed rule is truth",
                 fontsize=11.5, color=INK, y=1.04)
    fig.tight_layout()
    return fig


def save(fig, stem):
    FIGS.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"{stem}.{ext}", dpi=200, bbox_inches="tight",
                    facecolor="white")
    print(f"  wrote figs/{stem}.pdf and figs/{stem}.png")
    plt.close(fig)


if __name__ == "__main__":
    d = load()
    save(fig_corner(d), "fig1_corner_three_estimators")
    save(fig_offsets(d), "fig2_offsets_from_truth")
