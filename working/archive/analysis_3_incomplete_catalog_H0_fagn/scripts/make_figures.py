#!/usr/bin/env python3
"""Figures for analysis_3_incomplete_catalog_H0_fagn — the completeness ladder.

  fig_ladder_widths.{pdf,png}    THE HEADLINE: sigma(H0) and sigma(f_AGN) against
                                 survey depth, every realisation, against BOTH
                                 references -- the same-estimator rung 0 and
                                 analysis 2's complete-limit record
  fig_closure_ladder.{pdf,png}   per-rung medians +- 68 % for both parameters
                                 across realisations, against truth
  fig_estimator_offset.{pdf,png} the offset the completion has on COMPLETE
                                 catalogs, beside the per-pixel AGN sparsity that
                                 causes it
  fig_nside_scaling.{pdf,png}    the confirmatory test: predicted vs observed
                                 shift ratio under a 4x change in hosts per pixel
  fig_null_m18.{pdf,png}         the sky-shuffle null at the faintest rung

Colours, rcParams and the light print surface are analysis_2's, unchanged, so the
three directories' figures read as one system (see
analysis_2/scripts/make_figures.py for the palette validation record -- five seeds
on the ADJACENT categorical pairlist, two lanes on ALL-PAIRS, the "fold to Other"
pattern for the 2-D exhibit).  Identity is never colour alone: every panel that
distinguishes realisations also carries a legend, and every number is tabulated in
the README and in results/*.json.

Every figure is a pure function of results/*.json (+ the scan .h5 for posterior
curves).  Nothing is hard-coded, and a figure whose inputs are not on disk yet is
SKIPPED with a message rather than drawn from partial data -- so this is safe to
run at any point in the campaign, which is how the finalizer uses it.
"""
import json
import os
from pathlib import Path

import h5py
import numpy as np
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent
RES = Path(os.environ.get("A3_RESDIR", ROOT / "results"))
FIGS = Path(os.environ.get("A3_FIGDIR", ROOT / "figs"))
A2 = ROOT.parent / "analysis_2_complete_catalog_H0_fagn" / "results"
H0_TRUTH = 67.74
F_PLANTED = 0.30
SEEDS = [100, 101, 102, 103, 105]
LEVELS = ["complete", "m21", "m20", "m19", "m18"]
PRETTY = {"complete": "complete", "m21": "m < 21", "m20": "m < 20",
          "m19": "m < 19", "m18": "m < 18"}

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#8a8a85"
GRID = "#e6e5e1"
BLUE = "#2a78d6"      # slot 1
ORANGE = "#eb6834"    # slot 2
AQUA = "#1baf7a"      # slot 3
YELLOW = "#eda100"    # slot 4
MAGENTA = "#e87ba4"   # slot 5
SLOTS = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA]
OTHER = "#a9a8a2"
REJECT = "#d03b3b"

mpl.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.size": 9, "axes.labelsize": 9.5,
    "axes.titlesize": 10.5, "axes.edgecolor": INK_MUTED, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": INK_2, "ytick.color": INK_2,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "axes.linewidth": 0.8,
    "xtick.direction": "out", "ytick.direction": "out", "legend.frameon": False,
    "pdf.fonttype": 42,
})


def save(fig, name):
    FIGS.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"{name}.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {FIGS / name}.pdf/.png")


def tidy(ax, grid_axis="both"):
    ax.grid(True, axis=grid_axis, color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def jload(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def fscan_curve(path, key="f"):
    """(grid, normalised 1-D posterior) from a 1-D scan .h5."""
    p = Path(path)
    if not p.exists():
        return None
    with h5py.File(p, "r") as f:
        keys = list(f.keys())
        gname = "f_grid" if key == "f" else "H0_grid"
        if gname not in keys:
            return None
        x = np.asarray(f[gname][:], float)
        ll = np.asarray(f["log_likelihood"][:], float)
    fin = np.isfinite(ll)
    if not fin.any():
        return None
    P = np.where(fin, np.exp(ll - ll[fin].max()), 0.0)
    area = np.trapz(P, x)
    return x, (P / area if area > 0 else P)


def ladder():
    return jload(RES / "ladder_summary.json")


def levels_present(d):
    return [l for l in LEVELS if l in (d.get("rungs") or {})]


# --------------------------------------------------------------------------- #
def fig_ladder_widths():
    d = ladder()
    if not d:
        print("[skip] fig_ladder_widths: no ladder_summary.json")
        return
    order = levels_present(d)
    if len(order) < 2:
        print(f"[skip] fig_ladder_widths: only {len(order)} rung(s) on disk")
        return
    rungs = d["rungs"]
    a2 = d.get("analysis_2_reference") or {}

    x = np.arange(len(order))
    cw = [rungs[l]["completeness_within_horizon"]["gal"] for l in order]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))
    for ax, (par, lab, nd) in zip(
        axes,
        [("sigma_H0_per_seed", r"$\sigma(H_0)$  [km s$^{-1}$ Mpc$^{-1}$]", 2),
         ("sigma_f_per_seed", r"$\sigma(f_{\rm AGN})$", 3)],
    ):
        tidy(ax)
        for k, s in enumerate(SEEDS):
            y = []
            for l in order:
                seeds = [r["seed"] for r in rungs[l]["seeds"]]
                y.append(rungs[l]["width"][par][seeds.index(s)] if s in seeds else np.nan)
            ax.plot(x, y, "o-", color=SLOTS[k], lw=1.3, ms=4.0,
                    label=f"seed {s}", zorder=3)
        mean_key = ("sigma_H0_mean_halfwidth68" if "H0" in par
                    else "sigma_f_mean_halfwidth68")
        ax.plot(x, [rungs[l]["width"][mean_key] for l in order], "s--",
                color=INK, lw=1.6, ms=5.0, label="5-realisation mean", zorder=4)
        if a2.get("width"):
            ax.axhline(a2["width"][mean_key], color=OTHER, lw=1.4, ls=":", zorder=2)
            ax.text(len(order) - 1, a2["width"][mean_key], "  analysis 2\n  (no field term)",
                    color=INK_2, fontsize=7.5, va="bottom", ha="right")
        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"{PRETTY[l]}\nC={cw[i]['mean']:.2f}" for i, l in enumerate(order)])
        ax.set_ylabel(lab)
    axes[0].legend(loc="upper left", fontsize=7.6, ncol=2)
    fig.supxlabel("host survey depth   (C = mean completeness inside the GW horizon)",
                  fontsize=9.5, y=-0.02)
    fig.suptitle("How the two measurements degrade as the host survey empties",
                 fontsize=11, y=1.01)
    fig.subplots_adjust(wspace=0.28)
    save(fig, "fig_ladder_widths")


# --------------------------------------------------------------------------- #
def fig_closure_ladder():
    d = ladder()
    if not d:
        print("[skip] fig_closure_ladder: no ladder_summary.json")
        return
    order = levels_present(d)
    if not order:
        print("[skip] fig_closure_ladder: no rungs on disk")
        return
    rungs = d["rungs"]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2), sharey=True)
    off = np.linspace(-0.26, 0.26, len(SEEDS))

    for ax, par, truthlab in zip(
        axes, ["H0", "f_vs_realised"],
        [r"$H_0$  [km s$^{-1}$ Mpc$^{-1}$]", r"$f_{\rm AGN}$ $-$ realised"],
    ):
        tidy(ax, grid_axis="x")
        for i, l in enumerate(order):
            for k, s in enumerate(SEEDS):
                rec = next((r for r in rungs[l]["seeds"] if r["seed"] == s), None)
                if not rec:
                    continue
                b = rec[par]
                y = i + off[k]
                val = b["median"] - b["truth"] if par != "H0" else b["median"]
                lo = val - b["minus68"]
                hi = val + b["plus68"]
                ax.plot([lo, hi], [y, y], color=SLOTS[k], lw=1.5, zorder=3,
                        solid_capstyle="round")
                ax.plot([val], [y], "o", color=SLOTS[k], ms=4.0, zorder=4)
        ax.axvline(H0_TRUTH if par == "H0" else 0.0, color=INK, lw=1.2, ls="--",
                   zorder=2)
        ax.set_xlabel(truthlab)
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(5))
    axes[0].set_yticks(range(len(order)))
    axes[0].set_yticklabels([PRETTY[l] for l in order])
    axes[0].invert_yaxis()
    handles = [Line2D([], [], color=SLOTS[k], marker="o", ms=4, lw=1.5,
                      label=f"seed {s}") for k, s in enumerate(SEEDS)]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=8.2,
               bbox_to_anchor=(0.5, -0.06))
    axes[0].set_title("expansion rate", fontsize=10)
    axes[1].set_title("host fraction, against each realisation's own", fontsize=10)
    fig.suptitle("Closure at every rung: medians with 68 % intervals",
                 fontsize=11, y=1.01)
    save(fig, "fig_closure_ladder")


# --------------------------------------------------------------------------- #
def fig_estimator_offset():
    cont = jload(RES / "continuity_vs_analysis2.json")
    diag = jload(RES / "continuity_failure_diag.json")
    if not (cont and diag):
        print("[skip] fig_estimator_offset: need continuity_vs_analysis2.json "
              "and continuity_failure_diag.json")
        return
    a = fscan_curve(RES / "fscan_complete_s100.h5")
    b = fscan_curve(A2 / "fscan_s100.h5")
    occ = ((diag.get("seeds", {}).get("100") or {})
           .get("per_pixel_occupancy_in_horizon")
           or diag.get("per_pixel_occupancy_in_horizon_seed100"))
    if not (a and b and occ):
        print("[skip] fig_estimator_offset: missing f-scan curves or occupancy")
        return

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.9))

    ax = axes[0]
    tidy(ax)
    ax.plot(b[0], b[1], color=OTHER, lw=2.0,
            label="analysis 2:  no missing-host budget")
    ax.plot(a[0], a[1], color=ORANGE, lw=2.0,
            label="analysis 3:  field term at the true density")
    for cur, col in ((b, OTHER), (a, ORANGE)):
        med = cur[0][np.argmax(np.cumsum(cur[1]) / np.sum(cur[1]) >= 0.5)]
        ax.axvline(med, color=col, lw=1.0, ls=":", zorder=2)
    fs = cont["scans"].get("fscan", {})
    ax.axvline(F_PLANTED, color=INK, lw=1.2, ls="--", zorder=2)
    ax.text(F_PLANTED, ax.get_ylim()[1] * 0.02, " planted 0.30", fontsize=7.6,
            color=INK_2, va="bottom")
    ax.annotate(
        f"shift {fs.get('shift_median', float('nan')):+.3f}\n"
        f"= {fs.get('shift_median_in_a2_half_widths', float('nan')):+.2f} "
        "of the 68 %\nhalf-width",
        xy=(0.995, 0.42), xycoords="axes fraction", fontsize=8.2, color=INK,
        ha="right", va="center")
    ax.set_xlim(0.05, 0.62)
    ax.set_xlabel(r"$f_{\rm AGN}$")
    ax.set_ylabel("posterior density")
    ax.set_title("On COMPLETE catalogs, where nothing is missing", fontsize=10)
    ax.legend(loc="upper left", fontsize=7.6)

    ax = axes[1]
    tidy(ax)
    for tracer, col, lab in (("gal", BLUE, "galaxies"), ("agn", ORANGE, "AGN")):
        o = occ[tracer]
        e = np.asarray(o["hist_edges"], float)
        c = np.asarray(o["hist_counts"], float)
        c = c / c.sum()
        ax.step(e[:-1], c, where="post", color=col, lw=1.6, label=lab)
        ax.axvline(o["mean_per_pixel"], color=col, lw=1.0, ls=":")
    ax.set_xscale("symlog", linthresh=10, linscale=0.6)
    ax.set_xlim(0, None)                       # no meaningless negative decade
    ax.set_xlabel("hosts per nside-32 pixel, inside the GW horizon")
    ax.set_ylabel("fraction of pixels")
    g, n = occ["gal"], occ["agn"]
    ax.annotate(
        f"AGN\nmean {n['mean_per_pixel']:.1f}/pixel\n"
        f"Poisson error {100*n['poisson_frac_err_at_mean']:.0f} %",
        xy=(0.42, 0.97), xycoords="axes fraction", fontsize=8.2, color=ORANGE,
        ha="left", va="top")
    ax.annotate(
        f"galaxies\nmean {g['mean_per_pixel']:.0f}/pixel\n"
        f"Poisson error {100*g['poisson_frac_err_at_mean']:.0f} %",
        xy=(0.42, 0.66), xycoords="axes fraction", fontsize=8.2, color=BLUE,
        ha="left", va="top")
    ax.set_title("why: the completion is evaluated per pixel", fontsize=10)

    fig.suptitle(
        "The completion is not free on a complete catalog: it manufactures a "
        "missing-AGN budget out of per-pixel Poisson noise", fontsize=10.5, y=1.03)
    save(fig, "fig_estimator_offset")


# --------------------------------------------------------------------------- #
def fig_nside_scaling():
    ns = jload(RES / "nside_scaling.json")
    if not (ns and ns.get("verdict")):
        print("[skip] fig_nside_scaling: no verdict in nside_scaling.json")
        return
    arms = ns["arms"]
    order = [k for k in ("nside32", "nside16") if k in arms]
    hosts = ns.get("agn_hosts_per_pixel", {})
    v = ns["verdict"]

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.1))

    # ---- left: the two configurations at each pixelisation -------------------
    ax = axes[0]
    tidy(ax)
    hi = []
    for i, k in enumerate(order):
        a = arms[k]
        for lab, key, col, dx in (("no missing-host budget", "n0_minus24", OTHER, -0.13),
                                  ("true-$n_0$ field term", "true_n0", ORANGE, 0.13)):
            b = a[key]
            ax.errorbar(i + dx, b["median"],
                        yerr=[[b["median"] - b["ci68"][0]],
                              [b["ci68"][1] - b["median"]]],
                        fmt="o", color=col, ms=5.5, lw=1.6, capsize=3,
                        label=lab if i == 0 else None, zorder=3)
            hi.append(b["ci68"][1])
    lo = min(arms[k][j]["ci68"][0] for k in order for j in ("true_n0", "n0_minus24"))
    top = max(hi)
    pad = 0.10 * (top - lo)
    ax.set_ylim(lo - pad, top + 3.0 * pad)
    for i, k in enumerate(order):
        a = arms[k]
        y = max(a["true_n0"]["ci68"][1], a["n0_minus24"]["ci68"][1]) + 0.9 * pad
        ax.annotate("", xy=(i - 0.13, y), xytext=(i + 0.13, y),
                    arrowprops=dict(arrowstyle="<->", color=INK_2, lw=1.0))
        ax.text(i, y + 0.25 * pad, f"shift {a['shift_f']:+.4f}", ha="center",
                fontsize=8.4, color=INK)
    ax.axhline(F_PLANTED, color=INK, lw=1.2, ls="--", zorder=2)
    ax.text(-0.46, F_PLANTED, "planted 0.30", fontsize=7.6, color=INK_2,
            va="top", ha="left")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(
        [f"nside {k[5:]}\n{hosts.get(k, float('nan')):.0f} AGN per pixel"
         if hosts.get(k) else f"nside {k[5:]}" for k in order])
    ax.set_ylabel(r"$f_{\rm AGN}$")
    ax.set_xlim(-0.5, len(order) - 0.5)
    ax.legend(loc="lower left", fontsize=7.8)
    ax.set_title("the offset, at two pixelisations", fontsize=10)

    # ---- right: expectation vs observation -----------------------------------
    ax = axes[1]
    tidy(ax, grid_axis="y")
    bars = [("pure $1/\\sqrt{N}$", v["predicted"], OTHER)]
    if ns.get("refined_prediction_shift_ratio") is not None:
        bars.append(("$1/\\sqrt{N}$ on the\nnon-ramp part",
                     ns["refined_prediction_shift_ratio"], INK_MUTED))
    bars.append(("observed",
                 v["observed"],
                 AQUA if v["consistent_with_poisson_scaling"] else REJECT))
    for i, (lab, val, col) in enumerate(bars):
        ax.bar([i], [val], width=0.55, color=col, zorder=3)
        ax.text(i, val + 0.025, f"{val:.2f}", ha="center", fontsize=9.2, color=INK)
    ax.axhline(1.0, color=INK_2, lw=1.0, ls=":", zorder=2)
    ax.text(-0.55, 1.005, "no shrinkage", fontsize=7.6, color=INK_2,
            va="bottom", ha="left")
    ax.set_xticks(range(len(bars)))
    ax.set_xticklabels([b[0] for b in bars], fontsize=8.2)
    ax.set_ylabel("shift(nside 16) / shift(nside 32)")
    ax.set_ylim(0, 1.18)
    ax.set_xlim(-0.6, len(bars) - 0.1)
    ax.set_title("expected shrinkage vs observed", fontsize=10)

    fig.suptitle(
        "Merging pixels 4:1 shrinks the offset — the direction the per-pixel "
        "mechanism requires,\nthough by less than pure Poisson scaling predicts",
        fontsize=10.5, y=1.06)
    save(fig, "fig_nside_scaling")


# --------------------------------------------------------------------------- #
def joint_f_marginal(tag):
    """f marginal of a joint (H0, f) grid -- the recorded measurement at a rung."""
    p = RES / f"{tag}.h5"
    if not p.exists():
        return None
    with h5py.File(p, "r") as f:
        H0 = np.asarray(f["H0_grid"][:], float)
        fv = np.asarray(f["f_grid"][:], float)
        ll = np.asarray(f["log_likelihood"][:], float)
    fin = np.isfinite(ll)
    if not fin.any():
        return None
    P = np.where(fin, np.exp(ll - ll[fin].max()), 0.0)
    m = np.trapz(P, H0, axis=0)
    area = np.trapz(m, fv)
    return fv, (m / area if area > 0 else m)


def fig_null_m18():
    rec = fscan_curve(RES / "fscan_null_m18_s100.h5")
    d = ladder()
    null = (d or {}).get("sky_shuffle_null")
    # the recorded m18 measurement is the joint grid's f marginal, not an f scan
    data = joint_f_marginal("joint_m18_s100")
    if rec is None:
        print("[skip] fig_null_m18: no fscan_null_m18_s100.h5")
        return
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    tidy(ax)
    if data is not None:
        ax.plot(data[0], data[1], color=ORANGE, lw=2.0, label="recorded (m < 18)")
    elif null and null.get("record_median") is not None:
        ax.axvline(null["record_median"], color=ORANGE, lw=2.0,
                   label=f"recorded median {null['record_median']:.3f}")
    ax.plot(rec[0], rec[1], color=OTHER, lw=2.0, label="sky-shuffled")
    ax.axvline(F_PLANTED, color=INK, lw=1.2, ls="--", zorder=2)
    ax.text(F_PLANTED, ax.get_ylim()[1] * 0.97, " planted 0.30", fontsize=7.6,
            color=INK_2, va="top")
    if null and null.get("separation_in_null_widths") is not None:
        ax.annotate(
            f"recorded value sits\n{null['separation_in_null_widths']:.1f} null "
            "widths away",
            xy=(0.55, 0.55), xycoords="axes fraction", fontsize=8.4, color=INK)
    ax.set_xlabel(r"$f_{\rm AGN}$")
    ax.set_ylabel("posterior density")
    ax.set_xlim(0, 0.8)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Permuting which patch of sky each event belongs to,\n"
                 "at the faintest rung", fontsize=10.5)
    save(fig, "fig_null_m18")


def main():
    for fn in (fig_ladder_widths, fig_closure_ladder, fig_estimator_offset,
               fig_nside_scaling, fig_null_m18):
        try:
            fn()
        except Exception as exc:                      # a partial campaign must not
            print(f"[skip] {fn.__name__}: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
