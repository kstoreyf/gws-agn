#!/usr/bin/env python3
"""Figures for analysis_1_complete_catalog_H0.

  fig_h0_recovery.{pdf,png}  the four single-tracer H0 posteriors (upper panel)
                             and the two matched-host controls (lower panel)
  fig_guard.{pdf,png}        selection-integral N_eff against its threshold, and
                             the per-cell admit/reject mask
  fig_closure_seeds.{pdf,png}  each matched-host control across the five mock
                             realisations, against truth and the mean over mocks
                             (needs results/closure_seeds.json)

Colours are categorical slots 1 (blue = GAL) and 2 (orange = AGN) of the dataviz
reference palette — the first three slots are its all-pairs-validated subset.
Identity is never colour-alone: the tracer is direct-labelled and the injection
lane is carried by line style (solid = targeted, dashed = popuni).  Every
annotation below is computed from the result files; nothing is hard-coded.
"""
import json
from pathlib import Path

import h5py
import numpy as np
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
FIGS = ROOT / "figs"
TRUTH = 67.74

# --- dataviz reference palette (light surface) -------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#8a8a85"
GRID = "#e6e5e1"
BLUE = "#2a78d6"          # categorical slot 1 -> GAL
ORANGE = "#eb6834"        # categorical slot 2 -> AGN
REJECT = "#d03b3b"        # status: critical -> guard-rejected

CONFIGS = [
    # tag,              tracer, colour, lane,       linestyle, alpha, zorder
    ("h0_gal_targeted", "GAL",  BLUE,   "targeted", "-",  1.00, 5),
    ("h0_gal_popuni",   "GAL",  BLUE,   "popuni",   "--", 0.50, 4),
    ("h0_agn_targeted", "AGN",  ORANGE, "targeted", "-",  1.00, 5),
    ("h0_agn_popuni",   "AGN",  ORANGE, "popuni",   "--", 0.50, 4),
]

CONTROLS = [
    ("ctrl_gal_matched", "GAL", BLUE,   "GAL catalog, its own hosted events"),
    ("ctrl_agn_matched", "AGN", ORANGE, "AGN catalog, its own hosted events"),
]

mpl.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.size": 9,
    "axes.labelsize": 9.5,
    "axes.titlesize": 10.5,
    "axes.edgecolor": INK_MUTED,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "axes.linewidth": 0.8,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "legend.frameon": False,
    "pdf.fonttype": 42,
})


def load(tag):
    """Return (H0 grid, logL, guard arrays, JSON summary)."""
    with h5py.File(RES / f"{tag}.h5", "r") as f:
        H0 = f["H0_grid"][:]
        logL = f["log_likelihood"][:]
        guard = {k: f[f"guard/{k}"][:] for k in f["guard"].keys()}
    summary = json.loads((RES / f"{tag}.json").read_text())
    return H0, logL, guard, summary


def _n_events(tag):
    """Number of events a scan was run on, read from the event file it names."""
    with h5py.File(RES / f"{tag}.h5", "r") as f:
        gw = str(f.attrs["arg_gw_path"])
    p = Path(gw)
    if not p.is_absolute():
        p = ROOT / p
    with h5py.File(p, "r") as f:
        return int(f.attrs["nobs"])


def peak_normalised(logL):
    out = np.full(logL.shape, np.nan)
    fin = np.isfinite(logL)
    if fin.any():
        out[fin] = np.exp(logL[fin] - logL[fin].max())
    return out


def _spines(ax, keep=("left", "bottom")):
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side in keep)
        if side in keep:
            ax.spines[side].set_color(INK_MUTED)


# ---------------------------------------------------------------------------
def fig_h0_recovery():
    fig, (axA, axB) = plt.subplots(
        2, 1, figsize=(7.6, 6.2), sharex=True,
        gridspec_kw={"height_ratios": [1.6, 1.0], "hspace": 0.24})

    def _label_peak(ax, x, colour, title, sub, side=None, dx=0.0):
        """Direct label above a curve's peak: bold tracer name on top, the
        numbers under it.  Both sit ABOVE y = 1 so neither crosses a curve;
        `side`/`dx` separate labels whose peaks are close together."""
        lo, hi = ax.get_xlim()
        ha = "center"
        if x > hi - 6:
            ha, dx = "right", dx - 0.4
        elif x < lo + 6:
            ha, dx = "left", dx + 0.4
        elif side == "left":
            ha = "right"
        elif side == "right":
            ha = "left"
        ax.annotate(title, xy=(x + dx, 1.0), xytext=(0, 22),
                    textcoords="offset points", ha=ha, va="bottom",
                    fontsize=9.5, color=colour, weight="bold")
        if sub:
            ax.annotate(sub, xy=(x + dx, 1.0), xytext=(0, 9),
                        textcoords="offset points", ha=ha, va="bottom",
                        fontsize=8.0, color=colour)

    # ---------------- upper: the four production scans ----------------------
    axA.set_xlim(50, 100)
    axA.set_ylim(0, 1.42)
    axA.axvline(TRUTH, color=INK, lw=1.1, ls=(0, (5, 3)), zorder=2)
    axA.annotate(f"truth  {TRUTH}", xy=(TRUTH - 0.7, 0.60),
                 xycoords=("data", "axes fraction"), ha="right", va="bottom",
                 rotation=90, fontsize=8.2, color=INK)

    med_by_tracer = {}
    rail_nats = {}
    for tag, tracer, colour, lane, ls, alpha, z in CONFIGS:
        H0, logL, guard, summary = load(tag)
        p = peak_normalised(logL)
        lw = 2.0 if lane == "targeted" else 1.3
        axA.plot(H0, p, color=colour, lw=lw, ls=ls, alpha=alpha, zorder=z,
                 solid_capstyle="round")
        med_by_tracer.setdefault(tracer, {})[lane] = summary["H0"]["median"]
        if lane == "targeted":
            med_by_tracer[tracer]["peak"] = float(H0[int(np.nanargmax(p))])
            med_by_tracer[tracer]["colour"] = colour
            med_by_tracer[tracer]["railed"] = bool(
                np.nanargmax(p) in (0, len(H0) - 1))
            j = int(np.argmin(np.abs(H0 - TRUTH)))
            rail_nats[tracer] = float(np.nanmax(logL) - logL[j])

    for tracer, d in med_by_tracer.items():
        sub = f"{d['targeted']:.2f} / {d['popuni']:.2f}"
        if d["railed"]:
            sub = "> 100 (railed)"
        _label_peak(axA, d["peak"], d["colour"], tracer, sub)

    axA.set_ylabel("posterior density  (peak-normalised)")
    axA.set_yticks([0, 0.5, 1.0])
    axA.grid(axis="x", color=GRID, lw=0.6, zorder=0)
    axA.set_axisbelow(True)
    _spines(axA)

    handles = [
        Line2D([], [], color=BLUE, lw=2.0, label="GAL catalog"),
        Line2D([], [], color=ORANGE, lw=2.0, label="AGN catalog"),
        Line2D([], [], color=INK_2, lw=2.0, ls="-", label="targeted lane"),
        Line2D([], [], color=INK_2, lw=1.3, ls="--", alpha=0.6,
               label="popuni lane (cross-check)"),
    ]
    axA.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.40, 0.99),
               fontsize=8.2, labelcolor=INK_2, handlelength=1.9)
    axA.set_title("Single-tracer $H_0$ on the complete catalogs",
                  loc="left", color=INK, pad=20)
    axA.annotate("dark_sirens at the complete-catalog limit, K=1, field weighting; "
                 "labels are the targeted / popuni medians",
                 xy=(0.0, 1.015), xycoords="axes fraction", ha="left", va="bottom",
                 fontsize=8.4, color=INK_2)
    if med_by_tracer.get("AGN", {}).get("railed"):
        axA.annotate("the AGN posterior rises monotonically to the top of\n"
                     "the scanned range — "
                     f"{rail_nats['AGN']:.0f} nats above its value at truth.\n"
                     "Its peak lies outside the range; the median is where\n"
                     "the prior was cut, not a measurement of $H_0$.",
                     xy=(97.5, 0.20), xycoords=("data", "axes fraction"),
                     ha="right", va="bottom", fontsize=7.8, color=ORANGE,
                     linespacing=1.4)

    # ---------------- lower: matched-host controls --------------------------
    axB.set_xlim(50, 100)
    axB.set_ylim(0, 1.42)
    axB.axvline(TRUTH, color=INK, lw=1.1, ls=(0, (5, 3)), zorder=2)
    ctrl_notes = []
    ctrl_counts = {}
    _sides = {"ctrl_gal_matched": "left", "ctrl_agn_matched": "right"}
    for tag, tracer, colour, desc in CONTROLS:
        if not (RES / f"{tag}.h5").exists():
            continue
        H0, logL, guard, summary = load(tag)
        ctrl_counts[tracer] = _n_events(tag)
        p = peak_normalised(logL)
        axB.plot(H0, p, color=colour, lw=2.0, zorder=5, solid_capstyle="round")
        med = summary["H0"]["median"]
        lo, hi = summary["H0"]["ci90"]
        _label_peak(axB, float(H0[int(np.nanargmax(p))]), colour, tracer,
                    f"{med:.2f}  ({med - TRUTH:+.2f})", side=_sides.get(tag),
                    dx=(-1.0 if _sides.get(tag) == "left" else 1.0))
        ctrl_notes.append(f"{tracer}: {med:.2f} "
                          f"(+{hi - med:.2f} / −{med - lo:.2f}), "
                          f"offset {med - TRUTH:+.2f}")
    axB.set_yticks([0, 0.5, 1.0])
    axB.set_xlabel(r"$H_0$  [km s$^{-1}$ Mpc$^{-1}$]")
    axB.set_ylabel("posterior density")
    axB.grid(axis="x", color=GRID, lw=0.6, zorder=0)
    axB.set_axisbelow(True)
    _spines(axB)
    counts = " / ".join(f"{ctrl_counts[t]} {t}" for t in ("GAL", "AGN")
                        if t in ctrl_counts)
    axB.set_title("Matched-host control — each catalog on only the events it hosts "
                  f"({counts}, targeted lane)",
                  loc="left", color=INK_2, fontsize=9.2, pad=16)

    fig.subplots_adjust(left=0.095, right=0.985, top=0.885, bottom=0.085)
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"fig_h0_recovery.{ext}", dpi=220)
    plt.close(fig)
    print("wrote fig_h0_recovery.{pdf,png}  " + " | ".join(ctrl_notes))


# ---------------------------------------------------------------------------
def fig_guard():
    fig, (ax0, ax1) = plt.subplots(
        2, 1, figsize=(7.4, 5.0), sharex=True,
        gridspec_kw={"height_ratios": [1.55, 1.0], "hspace": 0.16})

    # label anchors chosen to keep the four curves' labels apart
    LABEL_AT = {"h0_gal_targeted": (58.0, "bottom"), "h0_gal_popuni": (90.0, "top"),
                "h0_agn_targeted": (78.0, "bottom"), "h0_agn_popuni": (56.5, "top")}

    ratios, mins = [], {}
    for tag, tracer, colour, lane, ls, alpha, z in CONFIGS:
        H0, logL, guard, summary = load(tag)
        ratio = guard["Neff"] / guard["threshold"]
        ratios.append(ratio)
        mins[tag] = (float(np.min(guard["Neff"])), float(np.min(ratio)))
        lw = 2.0 if lane == "targeted" else 1.3
        ax0.plot(H0, ratio, color=colour, lw=lw, ls=ls, alpha=alpha, zorder=z)
        xa, va = LABEL_AT[tag]
        j = int(np.argmin(np.abs(H0 - xa)))
        ax0.annotate(f"{tracer} {lane}", xy=(H0[j], ratio[j]),
                     xytext=(0, 6 if va == "bottom" else -7),
                     textcoords="offset points", ha="center", va=va,
                     fontsize=7.8, color=colour, alpha=max(alpha, 0.85))

    ax0.axhline(1.0, color=REJECT, lw=1.1, zorder=6)
    ax0.annotate(r"guard threshold  $N_{\rm eff} = 5N_{\rm obs}$",
                 xy=(99.4, 1.0), xytext=(0, -6), textcoords="offset points",
                 ha="right", va="top", fontsize=8.0, color=REJECT)
    ax0.axhline(2.0, color=INK_MUTED, lw=0.9, ls=(0, (4, 3)), zorder=6)
    ax0.annotate(r"sizing target  $2\times$", xy=(99.4, 2.0), xytext=(0, 4),
                 textcoords="offset points", ha="right", va="bottom",
                 fontsize=8.0, color=INK_2)
    ax0.axvline(TRUTH, color=INK, lw=1.1, ls=(0, (5, 3)), zorder=2)
    ax0.set_yscale("log")
    top = float(np.max([r.max() for r in ratios])) * 3.0
    ax0.set_ylim(0.5, top)
    ax0.set_ylabel(r"$N_{\rm eff}\,/\,$threshold")
    ax0.grid(color=GRID, lw=0.6, zorder=0)
    ax0.set_axisbelow(True)
    _spines(ax0)
    ax0.set_title("Selection-integral convergence over the scanned range",
                  loc="left", color=INK, pad=8)
    ax0.annotate(f"truth  {TRUTH}", xy=(TRUTH - 0.7, 0.02),
                 xycoords=("data", "axes fraction"), ha="right", va="bottom",
                 rotation=90, fontsize=8.0, color=INK)

    rows = list(CONFIGS)[::-1]
    for i, (tag, tracer, colour, lane, ls, alpha, z) in enumerate(rows):
        H0, logL, guard, summary = load(tag)
        step = H0[1] - H0[0]
        rej = guard["rejected"].astype(bool)
        for k, r in enumerate(rej):
            ax1.add_patch(Rectangle(
                (H0[k] - step / 2, i - 0.34), step, 0.68,
                facecolor=("none" if r else colour),
                alpha=(1.0 if r else alpha),
                edgecolor=(REJECT if r else "none"),
                hatch=("////" if r else None), lw=0.0, zorder=3))
        n_rej = int(rej.sum())
        nmin, rmin = mins[tag]
        ax1.annotate(f"{n_rej}/{len(H0)} rejected\nmin $N_{{\\rm eff}}$ "
                     f"{nmin:,.0f}  ({rmin:.0f}× threshold)",
                     xy=(100.8, i), ha="left", va="center", fontsize=7.8,
                     color=(REJECT if n_rej else INK_2), annotation_clip=False,
                     linespacing=1.35)

    ax1.set_yticks(range(len(rows)))
    ax1.set_yticklabels([f"{t}  {l}" for _, t, _, l, _, _, _ in rows], fontsize=8.5)
    ax1.set_ylim(-0.6, len(rows) - 0.4)
    ax1.set_xlim(50, 100)
    ax1.set_xlabel(r"$H_0$  [km s$^{-1}$ Mpc$^{-1}$]")
    ax1.axvline(TRUTH, color=INK, lw=1.1, ls=(0, (5, 3)), zorder=6)
    _spines(ax1, keep=("bottom",))
    ax1.tick_params(axis="y", length=0)

    handles = [
        Rectangle((0, 0), 1, 1, facecolor=INK_MUTED, alpha=0.7, label="admitted"),
        Rectangle((0, 0), 1, 1, facecolor="none", edgecolor=REJECT, hatch="////",
                  lw=0.0, label=r"rejected  ($\log L = -\infty$)"),
    ]
    ax1.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, -0.70),
               ncol=2, fontsize=8.2, labelcolor=INK_2, handlelength=1.6)

    fig.subplots_adjust(left=0.135, right=0.700, top=0.905, bottom=0.21)
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"fig_guard.{ext}", dpi=220)
    plt.close(fig)
    print("wrote fig_guard.{pdf,png}")
    for tag, (nmin, rmin) in mins.items():
        print(f"  {tag:20s} min Neff {nmin:12,.0f}   ({rmin:.1f}x threshold)")


# ---------------------------------------------------------------------------
def fig_closure_seeds():
    """Does each catalog return the true H0 when it is given its own hosts?

    One dot-and-interval row per mock realisation, per catalog: the posterior
    median with its 90% interval.  A single point estimate cannot answer the
    question -- the interval says how far the answer is allowed to wander on one
    realisation -- so the figure's payload is the vertical band, the mean over
    the five mocks with its standard error, read against the truth line.

    The two panels deliberately share one x-axis.  The AGN intervals are ~2.8x
    tighter than the GAL ones and giving each panel its own range would hide
    exactly that, which is half the result.
    """
    path = RES / "closure_seeds.json"
    if not path.exists():
        print("skipping fig_closure_seeds: results/closure_seeds.json not found")
        return
    doc = json.loads(path.read_text())

    panels = [("closure_gal", "Galaxy catalog, on the events it hosts", BLUE),
              ("closure_agn", "AGN catalog, on the events it hosts", ORANGE)]
    panels = [(k, t, c) for k, t, c in panels if doc.get(k)]
    if not panels:
        print("skipping fig_closure_seeds: no per-realisation results yet")
        return

    lo = min(r["ci90"][0] for k, _, _ in panels for r in doc[k]["per_seed"])
    hi = max(r["ci90"][1] for k, _, _ in panels for r in doc[k]["per_seed"])
    pad = 0.16 * (hi - lo)
    xlim = (min(lo - pad, TRUTH - 1.5), max(hi + pad, TRUTH + 1.5))

    fig, axes = plt.subplots(len(panels), 1, figsize=(7.6, 5.6), sharex=True,
                             gridspec_kw={"hspace": 0.22})
    axes = np.atleast_1d(axes)

    for ax, (key, title, colour) in zip(axes, panels):
        c = doc[key]
        rows = c["per_seed"]
        n = len(rows)
        ys = np.arange(n)[::-1]          # first realisation at the top

        mean_H0 = TRUTH + c["mean_offset"]
        sem = c["sem_offset"]
        ax.axvspan(mean_H0 - sem, mean_H0 + sem, color=colour, alpha=0.13,
                   lw=0, zorder=1)
        ax.axvline(mean_H0, color=colour, lw=1.4, zorder=2)
        ax.axvline(TRUTH, color=INK, lw=1.1, ls=(0, (5, 3)), zorder=3)

        for y, r in zip(ys, rows):
            ax.plot([r["ci90"][0], r["ci90"][1]], [y, y], color=colour, lw=2.0,
                    solid_capstyle="round", zorder=4)
            if r["railed"]:
                # The posterior piles up against the edge of the scanned range,
                # so the point is a bound: hollow marker plus an arrow off-panel.
                # The explanation goes in the panel note, not on the row, where it
                # would sit on top of the mean band.
                ax.plot([r["median"]], [y], "o", ms=7.0, mfc=SURFACE,
                        markeredgecolor=colour, markeredgewidth=1.8, zorder=5)
                ax.annotate("", xy=(xlim[0] + 0.02 * (xlim[1] - xlim[0]), y),
                            xytext=(r["ci90"][0], y),
                            arrowprops=dict(arrowstyle="-|>", color=colour, lw=1.6,
                                            shrinkA=0, shrinkB=0), zorder=4)
            else:
                ax.plot([r["median"]], [y], "o", ms=7.0, color=colour,
                        markeredgecolor=SURFACE, markeredgewidth=1.4, zorder=5)
            ax.annotate(f"{r['n_events']} events", xy=(xlim[1], y),
                        xytext=(6, 0), textcoords="offset points",
                        ha="left", va="center", fontsize=7.8, color=INK_MUTED,
                        annotation_clip=False)

        ax.set_ylim(-0.6, n + 0.55)          # headroom for the in-panel note
        ax.set_yticks(ys)
        ax.set_yticklabels([f"mock {i + 1}" for i in range(n)], fontsize=8.5)
        ax.set_xlim(*xlim)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        _spines(ax, keep=("bottom",))
        ax.set_title(title, loc="left", color=INK, fontsize=9.6, pad=6)

        sign = "+" if c["mean_offset"] >= 0 else "−"
        verdict = ("consistent with the true value"
                   if c["consistent_with_truth_2sigma"]
                   else "displaced from the true value")
        note = (f"mean over mocks  {sign}{abs(c['mean_offset']):.2f} ± {sem:.2f}"
                f"   —   {verdict}")
        if c.get("n_railed"):
            which = ", ".join(f"mock {rows.index(r) + 1}" for r in rows if r["railed"])
            note += f"   ({which} runs off the scanned range)"
        ax.annotate(note, xy=(0.005, 0.985), xycoords="axes fraction",
                    ha="left", va="top", fontsize=8.4, color=colour)

    # The label goes on the GAL panel: the AGN medians sit ON the truth line, so
    # there is no room for it there.
    axes[0].annotate(f"true value  {TRUTH}", xy=(TRUTH - 0.35, 0.04),
                     xycoords=("data", "axes fraction"), ha="right", va="bottom",
                     rotation=90, fontsize=8.0, color=INK)
    axes[-1].set_xlabel(r"$H_0$  [km s$^{-1}$ Mpc$^{-1}$]")
    axes[0].annotate(f"Recovered $H_0$ on {len(doc['closure_agn']['per_seed'])} "
                     "independent mocks",
                     xy=(0.0, 1.34), xycoords="axes fraction", ha="left",
                     va="bottom", fontsize=11.5, color=INK, weight="bold")
    axes[0].annotate("posterior median and 90% interval per mock; the shaded band "
                     "is the mean over mocks ± its standard error",
                     xy=(0.0, 1.18), xycoords="axes fraction", ha="left",
                     va="bottom", fontsize=8.4, color=INK_2)

    fig.subplots_adjust(left=0.100, right=0.855, top=0.845, bottom=0.095)
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"fig_closure_seeds.{ext}", dpi=220)
    plt.close(fig)
    for key, title, _ in panels:
        c = doc[key]
        print(f"wrote fig_closure_seeds: {key} mean offset "
              f"{c['mean_offset']:+.3f} +- {c['sem_offset']:.3f} "
              f"(t = {c['t_statistic']:+.2f}, {c['n_seeds']} mocks)")


if __name__ == "__main__":
    FIGS.mkdir(exist_ok=True)
    mpl.rcParams["hatch.linewidth"] = 0.6
    mpl.rcParams["hatch.color"] = REJECT
    fig_h0_recovery()
    fig_guard()
    fig_closure_seeds()
