"""Flux-limited catalogs: the depth ladder, and what the galaxy density recovers.

    analyses/analysis_3_incomplete_catalog_H0_fagn/results/joint_{m21,m20,m19,m18}_s100.json
    analyses/experiments/experiment_dsmaster_4d_recheck/results/fit_m18_{selection,per_pixel}_s100.json
    analyses/selection_redo/fu_seed101/results/campaign_m18_dynesty{,_pp}_s101.json
    analyses/selection_redo/fu_seed102/results/campaign_m18_dynesty{,_pp}_s102.json
    data/seed100/surveys/surveys_meta.json

Left: one realisation of the simulated universe, its two catalogs cut at four
successive flux limits, with both comoving densities held at the simulation's
own values.  Each rung is a marginal median and its 90 % equal-tailed interval
read straight from that rung's result json (`H0.median`, `H0.ci90`,
`f.median`, `f.ci90`); nothing is re-derived from the grids.  The horizontal
axis is the AGN catalog's completeness inside the detection horizon, taken from
the mock's own survey metadata (`completeness/<cut>/agn/C_within_horizon`); the
upper scale names the flux limit that produced it.

Right: the same shallowest cut with both comoving densities free, for three
realisations, under the two ways of turning a flux limit into a completeness.
Each bar is the galaxy density's median and 90 % interval
(`summary.log10n0.median`, `.ci90`) and the number over each pair is the
difference of the two ln evidences (`sampler_meta.logz`), which is positive
throughout.

Two drawing decisions worth recording:

  * the ladder carries two quantities against one horizontal axis, so they are
    stacked as two sub-panels sharing that axis rather than folded onto twin
    vertical scales, where a reader has to work out which curve owns which
    scale before reading either;
  * the right panel's vertical range is opened well past the data on both ends
    so the key and the ln-evidence row sit on empty page instead of over a
    credible interval.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.text import Text

import figstyle as fs

# Nothing internal to the analysis is allowed to reach the page: no process
# words, no identifiers out of the result files, no em dashes.  The check runs
# on the drawn text objects, which is every string the reader can see.
FORBIDDEN = ("gate", "pipeline", "campaign", "verdict", "audit", "rerun",
             "workstream", "per_pixel", "per pixel", "c_mode", "out_tag",
             "selection", "estimator", "seed", "dynesty", "json", "—")

# ---------------------------------------------------------------------------
# sources
# ---------------------------------------------------------------------------
A3 = fs.ANALYSES / "analysis_3_incomplete_catalog_H0_fagn" / "results"
RECHECK = fs.EXP / "experiment_dsmaster_4d_recheck" / "results"
FU101 = fs.ANALYSES / "selection_redo" / "fu_seed101" / "results"
FU102 = fs.ANALYSES / "selection_redo" / "fu_seed102" / "results"
META = fs.DATA / "seed100" / "surveys" / "surveys_meta.json"

RUNGS = ["m21", "m20", "m19", "m18"]
MAGS = {"m21": 21, "m20": 20, "m19": 19, "m18": 18}

# (label, luminosity-function file, number-count file)
PAIRS = [
    ("realisation 1", RECHECK / "fit_m18_selection_s100.json",
     RECHECK / "fit_m18_per_pixel_s100.json"),
    ("realisation 2", FU101 / "campaign_m18_dynesty_s101.json",
     FU101 / "campaign_m18_dynesty_pp_s101.json"),
    ("realisation 3", FU102 / "campaign_m18_dynesty_s102.json",
     FU102 / "campaign_m18_dynesty_pp_s102.json"),
]

SHA = "0c5b3db"

LF = fs.C["blue"]            # completeness from the flux limit + luminosity function
NC = fs.C["orange"]          # completeness from the observed number counts

CHECK: list[str] = []


def note(line: str):
    CHECK.append(line)


def rel(p: Path) -> str:
    """Path as written in the docstring, rooted at the working tree.

    The mock's own directory is reached through a link that leaves the tree, so
    anything outside it is reported in full rather than silently shortened.
    """
    p = Path(p)
    for cand in (p, p.resolve()):
        try:
            return str(cand.relative_to(fs.PAPER.parent))
        except ValueError:
            continue
    return str(p)


# ---------------------------------------------------------------------------
# loading, with the provenance assertions the result files are meant to carry
# ---------------------------------------------------------------------------
def load_ladder():
    meta = json.loads(META.read_text())
    out = {"_zmax": float(meta["horizon_z"])}
    note(f"  completeness measured to z = {meta['horizon_z']:.6f}"
         f"  -> axis label z < {meta['horizon_z']:.2f}"
         f"   <- {rel(META)}  horizon_z")
    for r in RUNGS:
        d = json.loads((A3 / f"joint_{r}_s100.json").read_text())
        if d.get("c_mode") != "selection":
            raise SystemExit(f"{rel(A3 / f'joint_{r}_s100.json')}: unexpected mode")
        c = meta["completeness"][r]["agn"]["C_within_horizon"]
        out[r] = {
            "C": 100.0 * c,
            "H0": (d["H0"]["median"], d["H0"]["ci90"]),
            "f": (d["f"]["median"], d["f"]["ci90"]),
        }
        # the reference values the dashed lines mark, read from the same file
        # the intervals come from rather than restated here
        out.setdefault("_ref", {})
        out["_ref"].setdefault("H0", d["H0"]["truth"])
        out["_ref"].setdefault("f", d["f"]["truth"])
        note(f"  m<{MAGS[r]}  completeness {100 * c:6.2f} %"
             f"   <- {rel(META)}  completeness/{r}/agn/C_within_horizon")
        note(f"          H0     {d['H0']['median']:7.3f}"
             f"  [{d['H0']['ci90'][0]:.3f}, {d['H0']['ci90'][1]:.3f}]"
             f"   <- {rel(A3 / f'joint_{r}_s100.json')}  H0.median, H0.ci90")
        note(f"          f_AGN  {d['f']['median']:7.4f}"
             f"  [{d['f']['ci90'][0]:.4f}, {d['f']['ci90'][1]:.4f}]"
             f"   <- same file, f.median, f.ci90")
    return out


def load_pairs():
    rows = []
    for label, p_lf, p_nc in PAIRS:
        lf = json.loads(p_lf.read_text())
        nc = json.loads(p_nc.read_text())
        if lf.get("c_mode") != "selection":
            raise SystemExit(f"{rel(p_lf)}: unexpected mode")
        if not str(lf.get("darksirens_git_sha", "")).startswith(SHA):
            raise SystemExit(f"{rel(p_lf)}: unexpected code version")
        a, b = lf["summary"]["log10n0"], nc["summary"]["log10n0"]
        za, zb = lf["sampler_meta"]["logz"], nc["sampler_meta"]["logz"]
        rows.append({"label": label,
                     "lf": (a["median"], a["ci90"]),
                     "nc": (b["median"], b["ci90"]),
                     "ref": a["truth"],
                     "dlnz": za - zb})
        note(f"  {label}")
        note(f"    luminosity function  log10 n0 {a['median']:7.4f}"
             f"  [{a['ci90'][0]:.4f}, {a['ci90'][1]:.4f}]"
             f"   ln evidence {za:.4f}")
        note(f"        <- {rel(p_lf)}"
             f"  summary.log10n0.median/.ci90, sampler_meta.logz")
        note(f"    number counts        log10 n0 {b['median']:7.4f}"
             f"  [{b['ci90'][0]:.4f}, {b['ci90'][1]:.4f}]"
             f"   ln evidence {zb:.4f}")
        note(f"        <- {rel(p_nc)}"
             f"  summary.log10n0.median/.ci90, sampler_meta.logz")
        note(f"    ln evidence difference  {za - zb:+.4f}  -> drawn as "
             f"{za - zb:+.1f}")
    return rows


# ---------------------------------------------------------------------------
# panels
# ---------------------------------------------------------------------------
def panel_ladder(ax_top, ax_bot, lad):
    x = np.array([lad[r]["C"] for r in RUNGS])

    def series(ax, key, fmt_in, input_value, ylabel, pad_top, va, word):
        med = np.array([lad[r][key][0] for r in RUNGS])
        lo = med - np.array([lad[r][key][1][0] for r in RUNGS])
        hi = np.array([lad[r][key][1][1] for r in RUNGS]) - med
        fs.truth_line(ax, input_value, axis="y")
        ax.plot(x, med, color=LF, lw=0.9, zorder=4)
        ax.errorbar(x, med, yerr=[lo, hi], fmt="o", ms=4.0, color=LF,
                    ecolor=LF, elinewidth=1.3, capsize=0, zorder=5,
                    markeredgecolor="white", markeredgewidth=0.7)
        ax.set_ylabel(ylabel)
        span = (med + hi).max() - (med - lo).min()
        ax.set_ylim(min((med - lo).min(), input_value) - 0.12 * span,
                    max((med + hi).max(), input_value) + pad_top * span)
        # the label rides the widest gap between rungs, where no interval and
        # no connecting segment can run through it
        ax.annotate(f"{word} {input_value:{fmt_in}}", (0.50, input_value),
                    xycoords=("axes fraction", "data"),
                    textcoords="offset points",
                    xytext=(0, -3 if va == "top" else 3), ha="center",
                    va=va, fontsize=6.8, color=fs.INK2)

    # H0's dashed line is the input expansion rate; the fraction's is the one
    # the drawn events actually realised, which is the reference the mixture
    # weight is judged against (see the caption and Figure 3).
    series(ax_top, "H0", ".2f", lad["_ref"]["H0"],
           r"$H_0$  [km s$^{-1}$ Mpc$^{-1}$]", 0.46, "top", "input")
    series(ax_bot, "f", ".3f", lad["_ref"]["f"], r"$f_{\rm AGN}$", 0.16,
           "bottom", "realised")

    for ax in (ax_top, ax_bot):
        ax.set_xlim(107, 1)                      # deepest catalog on the left
        ax.set_xticks([100, 80, 60, 40, 20, 0])
    ax_top.tick_params(labelbottom=False)
    ax_bot.set_xlabel("AGN catalog completeness to "
                      rf"$z < {lad['_zmax']:.2f}$  [%]")

    # the flux limit that produced each completeness, on the far scale
    axm = ax_top.twiny()
    axm.set_xlim(ax_top.get_xlim())
    axm.grid(visible=False)
    axm.spines["top"].set_visible(False)
    axm.set_xticks(list(x))
    axm.set_xticklabels([f"{MAGS[r]}" for r in RUNGS])
    axm.tick_params(axis="x", length=2.5, width=0.7, pad=1.5)
    axm.set_xlabel("limiting magnitude", labelpad=2.5)

    ax_top.legend(handles=[
        Line2D([], [], color=LF, lw=1.3, marker="o", ms=4.0,
               markeredgecolor="white", markeredgewidth=0.7,
               label="median and 90 % interval"),
        Line2D([], [], color=fs.TRUTH, lw=0.9, ls=(0, (3, 2)), alpha=0.75,
               label="input value"),
    ], loc="upper right", fontsize=7.0, handlelength=1.5, labelspacing=0.32,
        borderaxespad=0.25)


def panel_density(ax, rows):
    xs = np.arange(len(rows), dtype=float)
    dodge = 0.17
    for sign, key, colour in ((-1.0, "lf", LF), (+1.0, "nc", NC)):
        med = np.array([r[key][0] for r in rows])
        lo = med - np.array([r[key][1][0] for r in rows])
        hi = np.array([r[key][1][1] for r in rows]) - med
        ax.errorbar(xs + sign * dodge, med, yerr=[lo, hi], fmt="o", ms=4.2,
                    color=colour, ecolor=colour, elinewidth=1.5, capsize=0,
                    zorder=5, markeredgecolor="white", markeredgewidth=0.7)

    # the input density is already a labelled tick on this axis, so the line
    # carries no second copy of the number; the key names it
    fs.truth_line(ax, rows[0]["ref"], axis="y")

    ax.set_xticks(xs)
    ax.set_xticklabels([r["label"] for r in rows])
    ax.set_xlim(-0.55, len(rows) - 0.45)
    ax.grid(axis="x", visible=False)
    ax.set_ylim(-5.05, -0.55)
    ax.set_yticks([-5, -4, -3, -2, -1])
    ax.set_ylabel(r"$\log_{10}\, n_{0,\rm gal}$  [Mpc$^{-3}$]")

    for xi, r in zip(xs, rows):
        ax.annotate(f"{r['dlnz']:+.1f}", (xi, 0.965),
                    xycoords=("data", "axes fraction"), ha="center",
                    va="top", fontsize=7.4, color=fs.INK2)
    ax.set_title("ln evidence favouring the luminosity function",
                 fontsize=7.2, color=fs.INK2, pad=3.0)

    ax.legend(handles=[
        Line2D([], [], color=LF, lw=1.5, marker="o", ms=4.2,
               markeredgecolor="white", markeredgewidth=0.7,
               label="luminosity-function completeness"),
        Line2D([], [], color=NC, lw=1.5, marker="o", ms=4.2,
               markeredgecolor="white", markeredgewidth=0.7,
               label="number-count completeness"),
        Line2D([], [], color=fs.TRUTH, lw=0.9, ls=(0, (3, 2)), alpha=0.75,
               label="input value"),
    ], loc="lower left", fontsize=7.0, handlelength=1.5, labelspacing=0.32,
        borderaxespad=0.25, title="median and 90 % interval",
        title_fontsize=7.0, alignment="left")


# ---------------------------------------------------------------------------
def build():
    fs.use()
    note("fig_incomplete self-check: every numeral drawn, with its source")
    note("")
    note("left panel -- depth ladder, one realisation, densities anchored")
    lad = load_ladder()
    note("")
    note("right panel -- galaxy density with both densities free, "
         "shallowest cut")
    rows = load_pairs()

    fig = plt.figure(figsize=(fs.TWOCOL, 3.45))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.06],
                          height_ratios=[1.0, 1.0], hspace=0.10, wspace=0.30,
                          left=0.072, right=0.995, bottom=0.115, top=0.885)
    ax_top = fig.add_subplot(gs[0, 0])
    ax_bot = fig.add_subplot(gs[1, 0])
    ax_den = fig.add_subplot(gs[:, 1])

    panel_ladder(ax_top, ax_bot, lad)
    panel_density(ax_den, rows)

    fig.canvas.draw()
    shown = sorted({t.get_text() for t in fig.findobj(Text) if t.get_text()})
    bad = [(s, w) for s in shown for w in FORBIDDEN if w in s.lower()]
    if bad:
        raise SystemExit(f"text on the page that must not be there: {bad}")
    note("")
    note(f"every string drawn on the page ({len(shown)}), none of which may "
         "name anything internal:")
    note("  " + " | ".join(shown))
    return fig


def main():
    fig = build()
    fs.save(fig, "fig_incomplete")
    plt.close(fig)
    print("\n".join(CHECK))


if __name__ == "__main__":
    main()
