#!/usr/bin/env python
"""Result figures for analysis 5 under c_mode=selection. Deterministic.

    python make_figures.py        # writes ../figs/*.pdf and *.png

fig1_free_anchors   H0, f_AGN and the two completion densities down the
                    completeness ladder with BOTH anchors free, archived
                    per_pixel against selection. Intervals are 90%.
fig2_degeneracy     the f_AGN-vs-GAL-anchor plane, one panel per rung, 90%
                    credible contours for both estimators, truth crossed.

Analyses 3, 4 and 6 fix the completion densities at truth so the estimator is
the only moving part. This arm removes that support: both densities are free
under flat priors, which is the only configuration in the campaign where the
data alone have to identify them. The question it answers is not "is f_AGN
biased" -- with anchors free the interval is far too wide for that -- but
"where does the information about f_AGN actually come from", and the answer is
in the correlations, not the medians.

The archived twin is the same four tags on the same seed, so the ladder is
directly comparable. It is NOT a one-variable comparison: the archived campaign
ran on darksirens 2b86a2d and this one on 0c5b3db. The SHA-controlled version
of the same contrast lives in experiments/experiment_dsmaster_4d_recheck, and
agrees with the m18 rung here. Both provenances are written into the summary
JSON rather than left to memory.

Colours: fixed categorical order (Okabe-Ito), matching analyses 3, 4, 6 and 7 --
#0072B2 archived per_pixel, #009E73 this campaign's selection. Truth is a
neutral black rule, never a series hue.
"""
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

HERE = Path(__file__).resolve().parent
A5 = HERE.parent
ARCHIVE = (A5.parent.parent / "archive" /
           "analysis_5_free_anchors_H0_fagn" / "results")
FIGS = A5 / "figs"

RUNGS = ["m21", "m20", "m19", "m18"]
COMPLETENESS = {"m21": 0.997, "m20": 0.8143, "m19": 0.3151, "m18": 0.0954}
TRUTH = {"H0": 67.74, "f_AGN": 0.295, "log10n0": -3.0, "log10n0_c2": -5.0}
SEED100_H0_DRAW = 69.22       # this seed's own complete-catalog H0

SERIES = [("per_pixel (archived)", "#0072B2", ARCHIVE),
          ("selection (this work)", "#009E73", A5 / "results")]
INK, MUTED = "#1a1a1a", "#6b6b6b"

PANELS = [("H0", r"$H_0$  [km s$^{-1}$ Mpc$^{-1}$]"),
          ("f_AGN", r"$f_{\rm AGN}$"),
          ("log10n0", r"$\log_{10} n_0$  (GAL anchor)"),
          ("log10n0_c2", r"$\log_{10} n_0^{c2}$  (AGN anchor)")]

# h5 column order; 'fcat_2' is f_AGN because the survey order is GAL then AGN
CHAIN_LABELS = {"H0": 0, "log10n0": 1, "log10n0_c2": 2, "f_AGN": 3}


def load(resdir):
    """Per-rung summaries and equal-weight chains, whatever is present."""
    out = {}
    for r in RUNGS:
        pj = Path(resdir) / f"campaign_{r}_dynesty_s100.json"
        ph = Path(resdir) / f"campaign_{r}_dynesty_s100.h5"
        if not (pj.exists() and ph.exists()):
            continue
        d = json.loads(pj.read_text())
        with h5py.File(ph, "r") as f:
            chain = f["samples"][:]
            sha = str(f.attrs.get("darksirens_git_sha", "unknown"))
        s = d["summary"]
        out[r] = {
            "summary": {k: {"median": v["median"], "ci90": v["ci90"],
                            "offset": v["offset"], "pull": v["pull"],
                            "sd": v["sd"]}
                        for k, v in s.items() if k != "corr"},
            "corr": dict(zip(s["corr"]["labels"],
                             s["corr"]["matrix"][s["corr"]["labels"].index("f_AGN")])),
            "chain": chain,
            "logz": d["sampler_meta"]["logz"],
            "sha": sha[:10],
            "c_mode": d.get("c_mode", "per_pixel (legacy default)"),
        }
    return out


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED); ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=9)
    ax.grid(axis="y", color="#ececec", lw=0.7)
    ax.set_axisbelow(True)


def fmt_c(c):
    """99.7% and 100% must not both print as '100%' -- the complete rung and
    m<21 are different rungs and the tick labels are how a reader tells them
    apart."""
    return "100%" if c >= 0.9995 else f"{c:.1%}"


def rung_label(r):
    return f"m<{r[1:]}\nC={fmt_c(COMPLETENESS[r])}"


def fig_ladder(data):
    """fig1: the four free parameters down the ladder, both estimators."""
    x = np.arange(len(RUNGS))
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.4))

    for ax, (key, ylab) in zip(axes.ravel(), PANELS):
        ax.axhline(TRUTH[key], color="black", ls="--", lw=1.2, zorder=1)
        for k, (lbl, colour, d) in enumerate(data):
            if not d:
                continue
            off = (k - 0.5) * 0.13
            xs, ys, lo, hi = [], [], [], []
            for i, r in enumerate(RUNGS):
                if r not in d:
                    continue
                s = d[r]["summary"][key]
                xs.append(i + off); ys.append(s["median"])
                lo.append(s["median"] - s["ci90"][0])
                hi.append(s["ci90"][1] - s["median"])
            ax.errorbar(xs, ys, yerr=[lo, hi], fmt="o-", color=colour, ms=7,
                        lw=2.0, capsize=0, label=lbl, zorder=3)
            ax.plot(xs, ys, "o", color="white", ms=2.5, zorder=4)
        if key == "H0":
            ax.axhline(SEED100_H0_DRAW, color=MUTED, ls=":", lw=1.2, zorder=1)
            ax.text(len(RUNGS) - 1.0, SEED100_H0_DRAW, "  seed 100's own draw",
                    fontsize=7.5, color=MUTED, va="bottom")
        ax.set_xticks(x)
        ax.set_xticklabels([rung_label(r) for r in RUNGS], fontsize=8.5)
        ax.set_ylabel(ylab, fontsize=10.5, color=INK)
        style(ax)

    axes[0, 0].legend(frameon=False, fontsize=9, loc="lower left",
                      labelcolor=INK)
    axes[1, 0].text(0.02, 0.03, "bars = 90% CI   ·   dashed rule = truth",
                    transform=axes[1, 0].transAxes, fontsize=8, color=MUTED)
    fig.suptitle("Both completion densities free — the ladder, by estimator\n"
                 "seed 100, K=2 field mixture, flat priors on all four "
                 "parameters",
                 fontsize=11.5, color=INK, y=1.02)
    fig.tight_layout()
    save(fig, "fig1_free_anchors")


def contour90(ax, xs, ys, colour, label, bins=32, smooth=1.5):
    """90% credible contour of an equal-weight chain. No RNG, no fitting."""
    H, xe, ye = np.histogram2d(xs, ys, bins=bins)
    H = gaussian_filter(H, smooth)
    flat = np.sort(H.ravel())[::-1]
    csum = np.cumsum(flat)
    level = flat[np.searchsorted(csum, 0.90 * csum[-1])]
    xc = 0.5 * (xe[1:] + xe[:-1])
    yc = 0.5 * (ye[1:] + ye[:-1])
    ax.contour(xc, yc, H.T, levels=[level], colors=colour, linewidths=1.8)
    ax.contourf(xc, yc, H.T, levels=[level, H.max()], colors=colour, alpha=0.13)
    ax.plot([], [], color=colour, lw=1.8, label=label)


def fig_degeneracy(data):
    """fig2: where f_AGN's information actually comes from."""
    fig, axes = plt.subplots(1, len(RUNGS), figsize=(13.5, 3.9), sharey=True,
                             sharex=True)
    if len(RUNGS) == 1:
        axes = [axes]

    for ax, r in zip(axes, RUNGS):
        for lbl, colour, d in data:
            if r not in d:
                continue
            ch = d[r]["chain"]
            contour90(ax, ch[:, CHAIN_LABELS["log10n0"]],
                      ch[:, CHAIN_LABELS["f_AGN"]], colour, lbl)
        ax.axhline(TRUTH["f_AGN"], color="black", ls="--", lw=1.0, zorder=1)
        ax.axvline(TRUTH["log10n0"], color="black", ls="--", lw=1.0, zorder=1)
        # each correlation in its own series colour -- two unlabelled numbers
        # in one ink would be unreadable
        for j, (lbl, colour, d) in enumerate(data):
            if r not in d:
                continue
            ax.text(0.045, 0.965 - 0.085 * j, f"ρ = {d[r]['corr']['log10n0']:+.2f}",
                    transform=ax.transAxes, fontsize=9, va="top", color=colour)
        ax.set_title(rung_label(r).replace("\n", "   "), fontsize=9.5,
                     color=INK)
        ax.set_xlabel(r"$\log_{10} n_0$  (GAL anchor)", fontsize=10,
                      color=INK)
        style(ax)
        ax.grid(False)
    # the full flat prior on both axes, shared: the point is WHERE in the prior
    # each posterior sits, and a per-panel autoscale hides the railing
    axes[0].set_xlim(-4.0, -1.0)
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_ylabel(r"$f_{\rm AGN}$", fontsize=11, color=INK)
    axes[-1].legend(frameon=False, fontsize=8.5, loc="lower right",
                    labelcolor=INK)
    fig.suptitle("f_AGN against the galaxy completion density — 90% credible "
                 "contours over the full flat prior\n"
                 "at m<18 the archived posterior rails against the prior wall; "
                 "selection holds the anchor on truth",
                 fontsize=11.5, color=INK, y=1.08)
    fig.tight_layout()
    save(fig, "fig2_degeneracy")


def save(fig, stem):
    FIGS.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"{stem}.{ext}", dpi=200, bbox_inches="tight",
                    facecolor="white")
    print(f"  wrote figs/{stem}.pdf and .png")
    plt.close(fig)


def write_summary(data):
    """The numeric companion, so no figure ever has to be re-measured."""
    rows = []
    for r in RUNGS:
        row = {"rung": r, "C_in_horizon": COMPLETENESS[r]}
        for lbl, _, d in data:
            if r not in d:
                continue
            row[lbl] = {
                "medians": {k: v["median"] for k, v in d[r]["summary"].items()},
                "ci90": {k: v["ci90"] for k, v in d[r]["summary"].items()},
                "offsets": {k: v["offset"] for k, v in d[r]["summary"].items()},
                "pulls": {k: v["pull"] for k, v in d[r]["summary"].items()},
                "halfwidth90_f_AGN": 0.5 * (d[r]["summary"]["f_AGN"]["ci90"][1]
                                            - d[r]["summary"]["f_AGN"]["ci90"][0]),
                "corr_f_AGN_with": {k: v for k, v in d[r]["corr"].items()
                                    if k != "f_AGN"},
                "logz": d[r]["logz"],
                "darksirens_sha": d[r]["sha"],
                "c_mode": d[r]["c_mode"],
            }
        rows.append(row)
    out = {
        "_what": "Both completion densities free under flat priors, down the "
                 "completeness ladder, by estimator. The medians are not the "
                 "result -- with anchors free the f_AGN interval spans most of "
                 "the prior. The result is (a) which anchor the data recover "
                 "and (b) how strongly f_AGN is tied to it.",
        "_caveat": "The archived series ran on a different darksirens SHA "
                   "(2b86a2d) than this one (0c5b3db), so this ladder is not a "
                   "one-variable comparison. The SHA-controlled version of the "
                   "same contrast is experiments/experiment_dsmaster_4d_recheck.",
        "truth": TRUTH,
        "seed100_complete_H0_draw": SEED100_H0_DRAW,
        "rows": rows,
    }
    (A5 / "results" / "free_anchor_summary.json").write_text(
        json.dumps(out, indent=2))
    print("  wrote results/free_anchor_summary.json")


def main():
    data = [(lbl, c, load(d)) for lbl, c, d in SERIES]
    fig_ladder(data)
    fig_degeneracy(data)
    write_summary(data)


if __name__ == "__main__":
    main()
