#!/usr/bin/env python3
"""Figures for analysis_0_pure_tracer_H0 -- one tracer at a time, matched N.

  fig_posteriors.{pdf,png}   the ten record-lane (targeted) H0 posteriors, each
                             scaled to its own peak, truth marked, the x window
                             trimmed to the union of the curves' support
  fig_recovery.{pdf,png}     per-realisation medians +- 68 % for both tracers
                             against truth, tracers dodged, each tracer's
                             five-realisation mean offset +- s.e. as a band
  fig_lanes.{pdf,png}        targeted vs popuni: the median shift per scan pair,
                             in units of that scan's 68 % half-width (upper) and
                             in km/s/Mpc (lower)
  fig_diagnostics.{pdf,png}  the guard picture: per-scan minimum selection
                             N_eff against the 5 N_obs floor (log), and the
                             per-scan PE variance sum, all 20 scans
  fig_bimodal.{pdf,png}      the one genuinely bimodal case on its own: the
                             s105 galaxy scan, targeted vs popuni, modes and
                             68 % intervals marked

Everything is read from results/; nothing is hard-coded.  Deterministic: no RNG,
no wall-clock, no network -- rerunning overwrites byte-comparable figures.

    python scripts/make_figures.py                # all five
    python scripts/make_figures.py posteriors     # one by name

--------------------------------------------------------------------------- #
Style.  Self-contained copy of the project figure system, adapted from
working/paper/scripts/figstyle.py (rcParams, the posterior/CI helpers) and
working/analyses/analysis_2_complete_catalog_H0_fagn/scripts/make_figures.py
(save/tidy conventions, light surface, PDF + PNG).  Kept local rather than
imported so this directory renders without the paper tree on the path.

Colour.  Identity is the tracer and only the tracer: categorical slot 1
(#2a78d6) = galaxies, slot 2 (#eb6834) = AGN -- the same assignment as
analysis_1, analysis_2 and the paper.  The five realisations inside one tracer
are NOT five hues; the reference realisation is drawn at full strength and the
other four at a lighter step of the same hue, which is an ordinal step within
one identity, not a new identity.  Checked with
analysis_2/scripts/validate_palette.py (the skill's node validator ported to
python; this cluster has no node):

  * the categorical pair #2a78d6 / #eb6834, all-pairs, light surface:
    CVD dE 24.7, normal-vision dE 33.6, contrast PASS.
  * the lighter steps are the same hues composited at FADE = 0.62 on the
    surface -> #7aaae4 and #f1a080.  Every CROSS-tracer pair among the four
    marks still clears both floors (worst normal-vision dE 20.5, worst
    min(protan, deutan, tritan) dE 15.0), so no reader can confuse a galaxy
    curve with an AGN curve.
  * the two SAME-hue pairs are ordinal, so the ordinal gate applies rather than
    the categorical dE floor: dL = 0.151 (blue) and 0.108 (orange), both above
    the 0.093 step, and the lighter step sits at 2.35:1 and 2.03:1 against the
    surface, above the 2:1 ordinal floor but below the 3:1 relief line -- so the
    relief rule applies and the light steps never carry identity alone: every
    panel legends them, the full-strength curve of the same hue sits beside
    them, and the per-seed numbers are tabulated in README.md and in
    results/h0_pure_tracer.json.
    (The paper's fig_pure_tracer.py uses FADE = 0.55, which puts the light
    orange at 1.86:1, under the 2:1 ordinal floor; 0.62 is the smallest step
    that clears it for both hues.  Same construction, one notch stronger.)
  * the guard floor uses the reserved status colour #d03b3b, never a series.

Print medium: light surface only, PDF + PNG at the drawn size.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
FIGS = ROOT / "figs"

H0_TRUTH = 67.74
SEEDS = [100, 101, 102, 103, 105]
REF_SEED = 100
TRACERS = ("gal", "agn")
TRACER_NAME = {"gal": "galaxies", "agn": "AGN"}
LANES = ("targeted", "popuni")

# ---- palette (dataviz reference instance, light surface) --------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BAD = "#d03b3b"            # reserved status colour: the guard floor
BLUE = "#2a78d6"           # slot 1 -> galaxies
ORANGE = "#eb6834"         # slot 2 -> AGN
COLOR = {"gal": BLUE, "agn": ORANGE}
FADE = 0.62                # ordinal step for the four non-reference seeds

RC = {
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 8.5,
    "axes.titlesize": 9.0,
    "axes.labelsize": 8.5,
    "axes.labelcolor": INK,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.7,
    "axes.titlelocation": "left",
    "axes.titlepad": 4.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.5,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK2,
    "ytick.labelcolor": INK2,
    "xtick.labelsize": 8.0,
    "ytick.labelsize": 8.0,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "legend.frameon": False,
    "legend.fontsize": 7.5,
    "legend.labelcolor": INK2,
    "legend.handlelength": 1.6,
    "legend.handletextpad": 0.6,
    "legend.borderaxespad": 0.3,
    "lines.linewidth": 1.6,
    "lines.solid_capstyle": "round",
    "errorbar.capsize": 0.0,
    "text.color": INK,
    "mathtext.fontset": "dejavusans",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
}

XLABEL = r"$H_0$  [km s$^{-1}$ Mpc$^{-1}$]"


def use():
    mpl.rcParams.update(RC)


def fade(hex_colour: str, alpha: float = FADE, bg: str = SURFACE) -> str:
    """The lighter same-hue step: `hex_colour` composited on the chart surface.

    Returned as an opaque hex so the step is a real colour that can be handed to
    the palette validator, not an alpha that depends on what it overlaps.
    """
    c = [int(hex_colour.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4)]
    b = [int(bg.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4)]
    return "#%02x%02x%02x" % tuple(
        round(alpha * ci + (1.0 - alpha) * bi) for ci, bi in zip(c, b))


FADED = {t: fade(c) for t, c in COLOR.items()}


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def tag_of(tracer: str, lane: str, seed: int) -> str:
    return f"h0_pure{tracer}_{lane}_s{seed}"


def scan_h5(tag: str):
    """(H0 grid, log-likelihood, guard arrays) from one scan file."""
    with h5py.File(RES / f"{tag}.h5", "r") as f:
        guard = {k: np.asarray(f["guard"][k][:]) for k in f["guard"]}
        return (np.asarray(f["H0_grid"][:], float),
                np.asarray(f["log_likelihood"][:], float), guard)


def scan_json(tag: str) -> dict:
    return json.loads((RES / f"{tag}.json").read_text())


def aggregate() -> dict:
    return json.loads((RES / "h0_pure_tracer.json").read_text())


def posterior(grid, logl):
    """Flat-prior posterior on the scan grid, normalised to unit integral."""
    ok = np.isfinite(logl)
    p = np.zeros_like(logl)
    p[ok] = np.exp(logl[ok] - logl[ok].max())
    norm = np.trapz(p, grid)
    return p / norm if norm > 0 else p


def cdf_of(grid, p):
    c = np.concatenate([[0.0], np.cumsum(0.5 * (p[1:] + p[:-1]) * np.diff(grid))])
    return c / c[-1]


def ci_band(x, y, lo, hi):
    """Curve restricted to [lo, hi] with both ends interpolated.

    Masking on the grid instead would shade to the nearest grid point, drawing an
    interval up to one cell wider or narrower than the quoted one.
    """
    x = np.asarray(x, float)
    inner = x[(x > lo) & (x < hi)]
    xs = np.concatenate([[lo], inner, [hi]])
    return xs, np.interp(xs, x, y)


def curves(lane="targeted"):
    """{(tracer, seed): (grid, peak-normalised posterior, quoted H0 block)}."""
    out = {}
    for t in TRACERS:
        for s in SEEDS:
            tag = tag_of(t, lane, s)
            grid, logl, _ = scan_h5(tag)
            p = posterior(grid, logl)
            out[t, s] = (grid, p / p.max(), scan_json(tag)["H0"])
    return out


def support_window(cs, frac=0.9999, round_to=1.0):
    """x window holding at least `frac` of every curve's mass, rounded outward.

    The scan runs on [50, 100] but the posteriors occupy a small part of it, so
    the drawn window is the union of the per-curve central `frac` intervals,
    rounded out to whole km/s/Mpc.  Reported in the figure so the crop is stated
    rather than assumed.
    """
    tail = 0.5 * (1.0 - frac)
    lo, hi = [], []
    for grid, y, _ in cs.values():
        c = cdf_of(grid, y)
        lo.append(float(np.interp(tail, c, grid)))
        hi.append(float(np.interp(1.0 - tail, c, grid)))
    return (float(np.floor(min(lo) / round_to) * round_to),
            float(np.ceil(max(hi) / round_to) * round_to))


def save(fig, name):
    FIGS.mkdir(parents=True, exist_ok=True)
    out = FIGS / f"{name}.pdf"
    # CreationDate=None drops the wall-clock stamp the PDF backend writes by
    # default, so a rerun on unchanged results reproduces both files byte for byte
    fig.savefig(out, metadata={"CreationDate": None})
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)
    print(f"wrote {out} + {out.with_suffix('.png').name}")


def truth_line(ax, axis="x", lw=0.9, ymax=1.0):
    if axis == "x":
        ax.axvline(H0_TRUTH, ymax=ymax, color=INK, lw=lw, ls=(0, (3, 2)),
                   alpha=0.75, zorder=1.5)
    else:
        ax.axhline(H0_TRUTH, color=INK, lw=lw, ls=(0, (3, 2)), alpha=0.75,
                   zorder=1.5)


def header(ax, title, subtitle=None, base=5.0, sub_size=7.5):
    """Title above subtitle above the axes, stacked so nothing overlaps.

    `base` is the space already spoken for above the axes (group labels), in
    points; the subtitle is laid down first and the title pad is grown to clear
    it, so neither line has to be positioned by hand per figure.
    """
    y = base
    if subtitle:
        ax.annotate(subtitle, (0.0, 1.0), xycoords="axes fraction",
                    textcoords="offset points", xytext=(0, y), ha="left",
                    va="bottom", fontsize=sub_size, color=INK2, linespacing=1.4,
                    annotation_clip=False)
        y += (subtitle.count("\n") + 1) * sub_size * 1.55 + 3.0
    ax.set_title(title, color=INK, pad=y)


# --------------------------------------------------------------------------- #
# 1. the ten posteriors
# --------------------------------------------------------------------------- #
def fig_posteriors():
    agg = aggregate()
    cs = curves("targeted")
    xlo, xhi = support_window(cs)

    fig, ax = plt.subplots(figsize=(7.0, 3.9))
    for t in TRACERS:
        for s in SEEDS:                       # others first, reference on top
            if s == REF_SEED:
                continue
            g, y, _ = cs[t, s]
            ax.plot(g, y, color=FADED[t], lw=1.0, zorder=3)
        g, y, _ = cs[t, REF_SEED]
        ax.plot(g, y, color=COLOR[t], lw=1.9, zorder=5)

    truth_line(ax, ymax=0.87)
    ax.annotate(f"truth {H0_TRUTH}", (H0_TRUTH, 1.20), textcoords="offset points",
                xytext=(4, 0), ha="left", va="center", fontsize=7.5, color=INK2)

    # the second mode, named on the curve that has it -- taken from the
    # aggregate's shape record, not from the eye
    bim = [d for d in agg["diagnostics"]["per_scan"]
           if d["n_interior_modes"] > 1 and min(d["mode_relative_heights"]) > 0.01]
    for d in bim:
        g, y, _ = cs[d["tracer"], d["seed"]]
        k = int(np.argmin(d["mode_relative_heights"]))
        xm, hm = d["mode_positions"][k], d["mode_relative_heights"][k]
        ax.plot([xm], [hm], marker="o", ms=4.5, mfc=SURFACE, mec=COLOR[d["tracer"]],
                mew=1.4, zorder=6)
        ax.annotate(f"seed {d['seed']} ({TRACER_NAME[d['tracer']]})\n"
                    f"is bimodal: second mode\n"
                    f"at {xm:g}, {hm:.2f} of the peak",
                    (xm, hm), textcoords="offset points", xytext=(-8, 22),
                    ha="right", va="bottom", fontsize=7.5, color=INK2,
                    linespacing=1.35, zorder=7,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7,
                                    shrinkA=2, shrinkB=4))

    ax.set_xlim(xlo, xhi)
    ax.set_ylim(0, 1.38)
    ax.set_yticks([0, 0.5, 1.0])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(XLABEL)
    ax.set_ylabel("posterior, scaled to peak")

    wg = agg["closure_gal"]["mean_quoted_half68"]
    wa = agg["closure_agn"]["mean_quoted_half68"]
    header(ax, "Ten independent 1000-event sets, one tracer each",
           f"mean 68 % half-width  {wg:.2f} (galaxies)  vs  {wa:.2f} (AGN) "
           f"km s$^{{-1}}$ Mpc$^{{-1}}$ at equal $N$\n"
           f"scanned on [50, 100]; drawn on [{xlo:g}, {xhi:g}], which holds "
           r"$\geq$ 99.99 % of every curve's mass")

    # one legend row, so the block under it is free for the bimodal callout
    ax.legend(handles=[
        Line2D([], [], color=BLUE, lw=1.9, label=f"galaxies, seed {REF_SEED}"),
        Line2D([], [], color=FADED["gal"], lw=1.0, label="galaxies, other four"),
        Line2D([], [], color=ORANGE, lw=1.9, label=f"AGN, seed {REF_SEED}"),
        Line2D([], [], color=FADED["agn"], lw=1.0, label="AGN, other four"),
    ], loc="upper left", ncol=4, columnspacing=1.2, fontsize=7.5,
        handlelength=1.5, borderaxespad=0.4)
    fig.tight_layout(pad=0.4)
    save(fig, "fig_posteriors")


# --------------------------------------------------------------------------- #
# 2. recovery
# --------------------------------------------------------------------------- #
def fig_recovery():
    agg = aggregate()
    cs = curves("targeted")
    x = np.arange(len(SEEDS))

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for t, dx in (("gal", -0.14), ("agn", +0.14)):
        c = COLOR[t]
        blk = agg[f"closure_{t}"]
        med = np.array([cs[t, s][2]["median"] for s in SEEDS])
        lo = med - np.array([cs[t, s][2]["ci68"][0] for s in SEEDS])
        hi = np.array([cs[t, s][2]["ci68"][1] for s in SEEDS]) - med
        m, e = blk["mean_offset"], blk["sem_offset"]
        ax.axhspan(H0_TRUTH + m - e, H0_TRUTH + m + e, color=c, alpha=0.15,
                   lw=0, zorder=2)
        ax.axhline(H0_TRUTH + m, color=c, lw=0.9, zorder=3)
        ax.errorbar(x + dx, med, yerr=[lo, hi], fmt="o", ms=4.5, color=c,
                    ecolor=c, elinewidth=1.5, capsize=0, zorder=5,
                    markeredgecolor=SURFACE, markeredgewidth=0.8)

    # truth sits above the two mean lines: the AGN mean offset is -0.001, so its
    # line coincides with truth and must not be able to hide it
    truth_line(ax, axis="y", lw=1.0)
    ax.lines[-1].set_zorder(4)

    ax.grid(axis="x", visible=False)
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in SEEDS])
    ax.set_xlim(-0.5, len(SEEDS) - 0.5)
    ax.set_xlabel("catalog realisation (seed)")
    ax.set_ylabel(XLABEL)
    ax.margins(y=0.10)
    ylo, yhi = ax.get_ylim()
    ax.set_ylim(ylo - 0.2, yhi + 2.2)         # headroom for the legend block

    g, a = agg["closure_gal"], agg["closure_agn"]
    header(ax, "Both tracers recover truth on independent draws",
           "mean offset over 5 realisations, "
           f"km s$^{{-1}}$ Mpc$^{{-1}}$:   galaxies {g['mean_offset']:+.3f} "
           f"$\\pm$ {g['sem_offset']:.3f}  (t = {g['t_statistic']:+.2f})"
           f"    |    AGN {a['mean_offset']:+.3f} $\\pm$ {a['sem_offset']:.3f}  "
           f"(t = {a['t_statistic']:+.2f})")

    # the truth rule is named in the legend, not annotated on the line: at this
    # zoom every seed's marker sits close enough to it to be collided with
    # ncol = 3 so matplotlib's column-major fill puts each tracer's marker over
    # its own band swatch, with the truth rule alone in the last column
    ax.legend(handles=[
        Line2D([], [], color=BLUE, lw=1.5, marker="o", ms=4.5,
               markeredgecolor=SURFACE, markeredgewidth=0.8,
               label=r"galaxies, median $\pm$ 68 %"),
        Patch(facecolor=BLUE, alpha=0.15, edgecolor="none",
              label=r"mean offset $\pm$ s.e."),
        Line2D([], [], color=ORANGE, lw=1.5, marker="o", ms=4.5,
               markeredgecolor=SURFACE, markeredgewidth=0.8,
               label=r"AGN, median $\pm$ 68 %"),
        Patch(facecolor=ORANGE, alpha=0.15, edgecolor="none",
              label=r"mean offset $\pm$ s.e."),
        Line2D([], [], color=INK, lw=0.9, ls=(0, (3, 2)), alpha=0.75,
               label=f"truth {H0_TRUTH}"),
    ], loc="upper left", ncol=3, columnspacing=1.2, fontsize=7.5,
        handlelength=1.5, labelspacing=0.3, borderaxespad=0.4)
    fig.tight_layout(pad=0.4)
    save(fig, "fig_recovery")


# --------------------------------------------------------------------------- #
# 3. the two injection lanes
# --------------------------------------------------------------------------- #
def fig_lanes():
    agg = aggregate()
    rows = []
    for t in TRACERS:
        for r in agg["lanes"][t]["per_seed"]:
            rows.append((t, r))
    # x positions: five galaxy pairs, a gap, five AGN pairs
    xs, gap = [], 0.9
    for i, (t, _) in enumerate(rows):
        xs.append(i + (gap if t == "agn" else 0.0))
    xs = np.asarray(xs, float)

    ratio = np.array([abs(r["difference_over_targeted_half68"]) for _, r in rows])
    absdiff = np.array([abs(r["difference"]) for _, r in rows])
    cols = [COLOR[t] for t, _ in rows]

    # two measures on different scales -> two panels, never two y-axes
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 4.9), sharex=True,
                             gridspec_kw={"height_ratios": [1.25, 1.0],
                                          "hspace": 0.13})
    for ax, vals, ylab, head in (
            (axes[0], ratio, "median shift\n[68 % half-widths]", 1.32),
            (axes[1], absdiff, "median shift\n[km s$^{-1}$ Mpc$^{-1}$]", 1.62)):
        ax.bar(xs, vals, width=0.72, color=cols, edgecolor=SURFACE, linewidth=1.0,
               zorder=3)
        ax.set_ylabel(ylab, fontsize=8.5)
        ax.grid(axis="x", visible=False)
        ax.margins(y=0.0)
        ax.set_ylim(0, vals.max() * head)     # head: room for the annotations

    # the reference the upper panel is measured against
    axes[0].axhline(1.0, color=MUTED, lw=0.9, ls=(0, (4, 3)), zorder=4)
    axes[0].annotate("one 68 % half-width", (0.995, 1.0),
                     xycoords=("axes fraction", "data"),
                     textcoords="offset points", xytext=(0, 3), ha="right",
                     va="bottom", fontsize=7.5, color=INK2)

    # name each panel's own worst case per tracer -- the two panels rank the
    # scans differently, which is the point of showing both
    for t in TRACERS:
        idx = [i for i, (tt, _) in enumerate(rows) if tt == t]
        for ax, vals in ((axes[0], ratio), (axes[1], absdiff)):
            k = max(idx, key=lambda i: vals[i])
            ax.annotate(f"{vals[k]:.2f}", (xs[k], vals[k]),
                        textcoords="offset points", xytext=(0, 3), ha="center",
                        va="bottom", fontsize=7.5, color=COLOR[t])

    axes[1].set_xticks(xs)
    axes[1].set_xticklabels([str(r["seed"]) for _, r in rows])
    axes[1].set_xlabel("catalog realisation (seed)")
    axes[1].set_xlim(xs.min() - 0.7, xs.max() + 0.7)
    for t in TRACERS:
        idx = [i for i, (tt, _) in enumerate(rows) if tt == t]
        axes[0].annotate(TRACER_NAME[t], (xs[idx].mean(), 1.0),
                         xycoords=("data", "axes fraction"),
                         textcoords="offset points", xytext=(0, 12), ha="center",
                         va="bottom", fontsize=8.5, color=COLOR[t],
                         annotation_clip=False)

    gmax = agg["lanes"]["gal"]["max_abs_difference_over_half68"]
    amax = agg["lanes"]["agn"]["max_abs_difference_over_half68"]
    a_abs = max(absdiff[i] for i, (t, _) in enumerate(rows) if t == "agn")
    header(axes[0], "The injection lane does not set the answer",
           "|targeted $-$ popuni| median, all 20 scans as 10 same-events pairs; "
           f"largest shift {gmax:.2f} half-widths (galaxies), {amax:.2f} (AGN)",
           base=20.0)
    axes[1].annotate(
        f"the same shifts in absolute units: {amax:.2f} half-widths is only "
        f"{a_abs:.2f} km s$^{{-1}}$ Mpc$^{{-1}}$,\n"
        "large in the upper panel only because the AGN posterior is narrow",
        (0.99, 0.97), xycoords="axes fraction", ha="right", va="top",
        fontsize=7.5, color=INK2, linespacing=1.4)
    fig.tight_layout(pad=0.4)
    save(fig, "fig_lanes")


# --------------------------------------------------------------------------- #
# 4. the selection guard
# --------------------------------------------------------------------------- #
def fig_diagnostics():
    agg = aggregate()
    per_scan = {d["tag"]: d for d in agg["diagnostics"]["per_scan"]}

    groups = [(t, ln) for t in TRACERS for ln in LANES]
    xs, meta = [], []
    for gi, (t, ln) in enumerate(groups):
        for si, s in enumerate(SEEDS):
            xs.append(gi * (len(SEEDS) + 1.1) + si)
            g = scan_json(tag_of(t, ln, s))["guard"]["summary"]
            meta.append((t, ln, s, g, per_scan[tag_of(t, ln, s)]))
    xs = np.asarray(xs, float)

    fig, axes = plt.subplots(2, 1, figsize=(7.4, 5.2), sharex=True,
                             gridspec_kw={"height_ratios": [1.35, 1.0],
                                          "hspace": 0.13})
    for ax in axes:
        ax.grid(axis="x", visible=False)

    thr = float(np.max([m[3]["threshold_max"] for m in meta]))
    for xi, (t, ln, s, g, d) in zip(xs, meta):
        c = COLOR[t]
        filled = (ln == "targeted")
        kw = dict(color=c, mec=c, mew=1.3, ms=6.0, ls="none",
                  mfc=(c if filled else SURFACE), zorder=5)
        axes[0].plot([xi], [g["Neff_min"]], marker="o", **kw)
        axes[0].vlines(xi, thr, g["Neff_min"], color=c, lw=0.9, alpha=0.45,
                       zorder=3)
        # the PE variance sum is a range over the 201 cells, not one number
        axes[1].vlines(xi, g["pe_variance_sum_min"], g["pe_variance_sum_max"],
                       color=c, lw=1.6, alpha=0.45, zorder=3)
        axes[1].plot([xi], [g["pe_variance_sum_median"]], marker="o", **kw)

    axes[0].axhline(thr, color=BAD, lw=1.1, zorder=6)
    axes[0].annotate(rf"guard floor  $5N_{{\rm obs}}$ = {thr:,.0f}", (0.995, thr),
                     xycoords=("axes fraction", "data"),
                     textcoords="offset points", xytext=(0, -4), ha="right",
                     va="top", fontsize=7.5, color=BAD)
    axes[0].set_yscale("log")
    axes[0].set_ylim(thr / 2.2, max(m[3]["Neff_min"] for m in meta) * 4.0)
    axes[0].set_ylabel(r"minimum selection $N_{\rm eff}$" "\n" r"over the 201 cells")

    worst = min(meta, key=lambda m: m[3]["Neff_min"] / thr)
    wx = xs[meta.index(worst)]
    axes[0].annotate(
        f"worst cell of all 20 scans:\n{worst[3]['Neff_min']:,.0f} = "
        f"{worst[3]['Neff_min'] / thr:.1f}$\\times$ the floor",
        (wx, worst[3]["Neff_min"]), textcoords="offset points", xytext=(-6, 24),
        ha="right", va="bottom", fontsize=7.5, color=INK2, linespacing=1.35,
        arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7, shrinkA=2,
                        shrinkB=5), zorder=7)

    # log on both panels: the AGN scans' PE variance sum runs an order of
    # magnitude above the galaxy scans', so a linear axis would flatten the
    # galaxy ranges into the baseline
    axes[1].set_yscale("log")
    axes[1].set_ylabel(r"PE variance sum $\sum_i \sigma^2_{{\rm PE},i}$" "\n"
                       "(range over cells, median marked)")
    lo_v = min(m[3]["pe_variance_sum_min"] for m in meta)
    hi_v = max(m[3]["pe_variance_sum_max"] for m in meta)
    axes[1].set_ylim(lo_v / 1.6, hi_v * 2.6)
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels([str(m[2]) for m in meta], fontsize=7.5)
    axes[1].set_xlabel("catalog realisation (seed)")
    axes[1].set_xlim(xs.min() - 1.0, xs.max() + 1.0)

    for gi, (t, ln) in enumerate(groups):
        sel = xs[gi * len(SEEDS):(gi + 1) * len(SEEDS)]
        axes[0].annotate(f"{TRACER_NAME[t]} / {ln}", (sel.mean(), 1.0),
                         xycoords=("data", "axes fraction"),
                         textcoords="offset points", xytext=(0, 8), ha="center",
                         va="bottom", fontsize=8.0, color=COLOR[t],
                         annotation_clip=False)

    header(axes[0], "Every cell of every scan cleared the selection guard",
           "hard $N_{\\rm eff}$ wall, variance criterion inert "
           "(max_likelihood_variance = 1e6); "
           f"{agg['diagnostics']['n_scans']} scans "
           f"$\\times$ {per_scan[tag_of('gal', 'targeted', SEEDS[0])]['n_cells']} "
           "cells, 0 rejected", base=18.0)

    axes[0].legend(handles=[
        Line2D([], [], color=INK2, marker="o", ms=6.0, ls="none", mfc=INK2,
               mec=INK2, mew=1.3, label="targeted lane (record)"),
        Line2D([], [], color=INK2, marker="o", ms=6.0, ls="none", mfc=SURFACE,
               mec=INK2, mew=1.3, label="popuni lane (cross-check)"),
    ], loc="upper left", ncol=2, fontsize=7.5, handlelength=1.0,
        borderaxespad=0.4)
    fig.tight_layout(pad=0.4)
    save(fig, "fig_diagnostics")


# --------------------------------------------------------------------------- #
# 5. the bimodal galaxy realisation
# --------------------------------------------------------------------------- #
def fig_bimodal():
    agg = aggregate()
    bim = [d for d in agg["diagnostics"]["per_scan"]
           if d["n_interior_modes"] > 1 and min(d["mode_relative_heights"]) > 0.01]
    if not bim:
        print("[skip] fig_bimodal: no genuinely multimodal scan on record")
        return
    d = bim[0]
    tracer, seed = d["tracer"], d["seed"]

    styles = {"targeted": (COLOR[tracer], "-", 1.9),
              "popuni": (FADED[tracer], (0, (5, 2.5)), 1.7)}

    fig, ax = plt.subplots(figsize=(6.8, 4.1))
    xlo_all, xhi_all, drawn = [], [], {}
    for lane in LANES:
        tag = tag_of(tracer, lane, seed)
        grid, logl, _ = scan_h5(tag)
        p = posterior(grid, logl)
        y = p / p.max()
        blk = scan_json(tag)["H0"]
        c, ls, lw = styles[lane]
        bx, by = ci_band(grid, y, *blk["ci68"])
        ax.fill_between(bx, 0, by, color=c, alpha=0.16, lw=0, zorder=2)
        ax.plot(grid, y, color=c, ls=ls, lw=lw, zorder=5)
        drawn[lane] = (grid, y, blk, c)
        cd = cdf_of(grid, y)
        xlo_all.append(float(np.interp(5e-5, cd, grid)))
        xhi_all.append(float(np.interp(1 - 5e-5, cd, grid)))

    # the two 68 % intervals overlap heavily as shaded areas, so each is also
    # drawn as its own bar under the curves where the two can be told apart
    for k, lane in enumerate(LANES):
        _, _, blk, _ = drawn[lane]
        c = styles[lane][0]
        ybar = -0.075 - 0.075 * k
        ax.plot(blk["ci68"], [ybar, ybar], color=c, lw=3.0,
                solid_capstyle="butt", zorder=6, clip_on=False)
        ax.plot([blk["median"]], [ybar], marker="|", ms=8, mew=1.4,
                color=SURFACE, zorder=7, clip_on=False)
        ax.annotate(f"{lane}  68 %", (blk["ci68"][1], ybar),
                    textcoords="offset points", xytext=(5, 0), ha="left",
                    va="center", fontsize=7.0, color=c, annotation_clip=False)

    # stopped below the legend block; the rule is named in the legend, not
    # annotated on the line, which at this zoom has no free space beside it
    truth_line(ax, ymax=0.80)

    # the two modes of the targeted scan, from the recorded shape diagnostics
    grid, y, blk, c = drawn["targeted"]
    for xm, hm in zip(d["mode_positions"], d["mode_relative_heights"]):
        ax.plot([xm], [hm], marker="o", ms=5.0, mfc=SURFACE, mec=c, mew=1.5,
                zorder=7)
        ax.annotate(f"{xm:g}\n({hm:.2f} of peak)", (xm, hm),
                    textcoords="offset points", xytext=(0, 9), ha="center",
                    va="bottom", fontsize=7.5, color=c, linespacing=1.3,
                    zorder=7)

    ax.set_xlim(np.floor(min(xlo_all)), np.ceil(max(xhi_all)))
    ax.set_ylim(-0.20, 1.38)        # room under the curves for the interval bars
    ax.set_yticks([0, 0.5, 1.0])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(XLABEL)
    ax.set_ylabel("posterior, scaled to peak")

    lines = []
    for lane in LANES:
        _, _, blk, _ = drawn[lane]
        half = 0.5 * (blk["ci68"][1] - blk["ci68"][0])
        lines.append(f"{lane:<9s} median {blk['median']:.2f}, "
                     f"68 % [{blk['ci68'][0]:.2f}, {blk['ci68'][1]:.2f}], "
                     f"half-width {half:.2f}")
    header(ax, f"The one bimodal case: {TRACER_NAME[tracer]}, seed {seed}",
           lines[0] + "\n" + lines[1] + "   km s$^{-1}$ Mpc$^{-1}$\n"
           "the wide targeted interval is the honest reading of a posterior "
           "whose 68 % has to span the gap between two modes")

    ax.legend(handles=[
        Line2D([], [], color=styles["targeted"][0], ls=styles["targeted"][1],
               lw=1.9, label="targeted injections (lane of record)"),
        Line2D([], [], color=styles["popuni"][0], ls=styles["popuni"][1],
               lw=1.7, label="popuni injections (cross-check)"),
        Line2D([], [], color=INK, lw=0.9, ls=(0, (3, 2)), alpha=0.75,
               label=f"truth {H0_TRUTH}"),
    ], loc="upper left", fontsize=7.5, handlelength=1.8, labelspacing=0.3,
        borderaxespad=0.4)
    fig.tight_layout(pad=0.4)
    save(fig, "fig_bimodal")


# --------------------------------------------------------------------------- #
FIGURES = {
    "posteriors": fig_posteriors,
    "recovery": fig_recovery,
    "lanes": fig_lanes,
    "diagnostics": fig_diagnostics,
    "bimodal": fig_bimodal,
}


def main(argv):
    use()
    which = argv or list(FIGURES)
    unknown = [w for w in which if w not in FIGURES]
    if unknown:
        sys.exit(f"unknown figure(s): {unknown}; choose from {list(FIGURES)}")
    for w in which:
        FIGURES[w]()


if __name__ == "__main__":
    main(sys.argv[1:])
