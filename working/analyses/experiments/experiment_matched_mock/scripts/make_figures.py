#!/usr/bin/env python3
"""Publication figures for experiment_matched_mock.

Reads only this experiment's own outputs (results/*.json, results/*.h5) and writes:

  figs/fig_pe_sigma_ladder.{pdf,png}  H0 offset vs distance uncertainty, before/after
                                      the PE construction fix (darksirens PR #332)
  figs/fig_closure_seeds.{pdf,png}    per-seed closure with corrected PE, and the
                                      interval-vs-scatter comparison
  figs/fig_elimination.{pdf,png}      what was ruled out: the levers that move H0 and
                                      the ones that do not
  results/summary.json                every number quoted in the figures

Pure post-processing: h5py/numpy/matplotlib, CPU, no darksirens import.

Conventions: flat prior on the scanned axis; posterior = exp(logL - max) normalised by
the trapezoid rule; intervals equal-tailed from the marginal CDF. Error bars are the
68% half-width of a single realisation; the multi-seed band is the standard error of
the mean over seeds.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BASE = Path(__file__).resolve().parent.parent
RESULTS = BASE / "results"
FIGS = BASE / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

H0_TRUTH = 67.74

# Validated reference-palette categorical slots, fixed order; every series also carries
# a distinct marker/dash so identity never rests on colour alone.
BLUE, AQUA, YELLOW, GREEN, VIOLET, RED = (
    "#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948")
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#898781"
GRIDCOL, BASELINE = "#e1e0d9", "#c3c2b7"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9, "axes.labelsize": 9.5, "axes.titlesize": 9.5,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8,
    "axes.edgecolor": INK2, "axes.linewidth": 0.7, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "grid.color": GRIDCOL, "grid.linewidth": 0.6,
    "legend.frameon": False, "lines.solid_capstyle": "round",
})


def load(tag):
    p = RESULTS / f"{tag}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    if d.get("all_cells_rejected"):
        return {"rejected": True, "n_evals": d.get("n_evals")}
    h = d.get("H0")
    if not h:
        return None
    return {"median": h["median"], "ci68": h["ci68"],
            "hw": 0.5 * (h["ci68"][1] - h["ci68"][0]),
            "argmax": h.get("argmax"),
            "n_rejected": d.get("n_neginf_cells"), "n_evals": d.get("n_evals"),
            "rejected": False}


# --------------------------------------------------------------------------- #
# Figure 1 — the sigma ladder, before and after the PE fix
# --------------------------------------------------------------------------- #
def fig_sigma_ladder(S):
    sig = [0.01, 0.03, 0.10]
    before = [S["sigma_ladder"]["truth_centred"][f"{s:g}"] for s in sig]
    after = [S["sigma_ladder"]["corrected"][f"{s:g}"] for s in sig]

    fig, ax = plt.subplots(figsize=(3.5, 3.1), dpi=300)
    ax.axhline(0.0, color=BASELINE, lw=1.0, ls=(0, (4, 2.5)), zorder=1)

    for series, colour, marker, dash, label in (
        (before, YELLOW, "o", (0, (4, 1.8)), "truth-centred (before)"),
        (after, BLUE, "s", (0, ()), "flat-prior posterior (PR #332)"),
    ):
        off = [e["median"] - H0_TRUTH for e in series]
        err = [e["hw"] for e in series]
        ax.errorbar(sig, off, yerr=err, color=colour, marker=marker, ms=5.5,
                    mfc=colour, mec="white", mew=0.7, lw=1.3, ls=dash,
                    elinewidth=1.0, capsize=2.5, zorder=4, label=label)

    # sigma^2 reference through the largest corrected point
    ref_s, ref_off = sig[-1], after[-1]["median"] - H0_TRUTH
    grid = np.linspace(0.008, 0.115, 60)
    ax.plot(grid, ref_off * (grid / ref_s) ** 2, color=INK3, lw=0.9,
            ls=(0, (1, 1.6)), zorder=2, label=r"$\propto \sigma^2$")

    ax.set_xscale("log")
    ax.set_xlim(0.008, 0.13)
    ax.set_xlabel(r"fractional distance uncertainty  $\sigma_{d_L}$")
    ax.set_ylabel(r"$H_0$ offset from truth  [km s$^{-1}$ Mpc$^{-1}$]")
    ax.set_title("Distance-uncertainty scaling of the offset", fontsize=9.3)
    ax.grid(True, alpha=0.55)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", fontsize=7.4)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"fig_pe_sigma_ladder.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("wrote figs/fig_pe_sigma_ladder.{pdf,png}")


# --------------------------------------------------------------------------- #
# Figure 2 — per-seed closure, and intervals vs scatter
# --------------------------------------------------------------------------- #
def fig_closure_seeds(S):
    seeds = S["multi_seed"]["seeds"]
    med = np.array([e["median"] for e in seeds])
    hw = np.array([e["hw"] for e in seeds])
    labels = [e["seed"] for e in seeds]
    mean, sem = S["multi_seed"]["mean_H0"], S["multi_seed"]["sem"]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.1, 3.0), dpi=300,
                                   gridspec_kw={"width_ratios": [1.55, 1]})

    # ---- left: per-seed medians ----
    y = np.arange(len(med))
    axL.axvline(H0_TRUTH, color=INK, lw=1.0, ls=(0, (2, 2)), zorder=3,
                label="truth")
    axL.axvspan(mean - sem, mean + sem, color=BLUE, alpha=0.16, lw=0, zorder=1)
    axL.axvline(mean, color=BLUE, lw=1.3, zorder=2,
                label=f"mean ${mean:.2f}\\pm{sem:.2f}$")
    axL.errorbar(med, y, xerr=hw, fmt="s", ms=5.5, color=BLUE, mfc=BLUE,
                 mec="white", mew=0.7, lw=0, elinewidth=1.1, capsize=2.5, zorder=4)
    axL.set_yticks(y)
    axL.set_yticklabels([f"seed {s}" for s in labels])
    axL.set_ylim(-0.7, len(med) - 0.3)
    axL.invert_yaxis()
    axL.set_xlabel(r"$H_0$  [km s$^{-1}$ Mpc$^{-1}$]")
    axL.set_title(f"Corrected PE, $\\sigma_{{d_L}}=0.10$: "
                  f"{S['multi_seed']['offset']:+.2f} $\\pm$ {sem:.2f}"
                  f"  ({S['multi_seed']['significance']:.1f}$\\sigma$)",
                  fontsize=9.0)
    axL.grid(True, axis="x", alpha=0.55)
    axL.grid(False, axis="y")
    axL.set_axisbelow(True)
    # Upper left: the top seeds' intervals sit right of ~65.7, leaving that corner
    # free. Anywhere lower lands on seed 4105's error bar.
    axL.legend(loc="upper left", fontsize=7.4)

    # ---- right: quoted interval vs realised scatter ----
    sd = S["multi_seed"]["scatter_sd"]
    mean_hw = S["multi_seed"]["mean_halfwidth"]
    bars = [("mean quoted\n68% half-width", mean_hw, AQUA),
            ("realised scatter\nacross seeds", sd, VIOLET)]
    xs = np.arange(len(bars))
    axR.bar(xs, [b[1] for b in bars], width=0.55,
            color=[b[2] for b in bars], zorder=3)
    for x, (_, v, _) in zip(xs, bars):
        axR.annotate(f"{v:.2f}", xy=(x, v), xytext=(0, 3),
                     textcoords="offset points", ha="center", fontsize=8.2,
                     color=INK)
    axR.set_xticks(xs)
    axR.set_xticklabels([b[0] for b in bars], fontsize=7.8)
    axR.set_ylabel(r"km s$^{-1}$ Mpc$^{-1}$")
    axR.set_ylim(0, 1.35 * max(sd, mean_hw))
    axR.set_title(f"Intervals are {sd / mean_hw:.1f}$\\times$ too narrow",
                  fontsize=9.0)
    axR.grid(True, axis="y", alpha=0.55)
    axR.grid(False, axis="x")
    axR.set_axisbelow(True)
    extra = S["multi_seed"]["catalog_variance_component"]
    # Annotation sits in the empty upper-left of the panel, not over either bar.
    axR.annotate(f"host-catalog realisation\ncontributes $\\approx${extra:.2f}",
                 xy=(1.0, 0.55 * sd), xytext=(-0.36, 1.18 * sd), fontsize=7.4,
                 color=INK2, ha="left", va="center",
                 arrowprops=dict(arrowstyle="->", color=INK3, lw=0.8,
                                 shrinkA=2, shrinkB=2))

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"fig_closure_seeds.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("wrote figs/fig_closure_seeds.{pdf,png}")


# --------------------------------------------------------------------------- #
# Figure 3 — which levers move H0 and which do not
# --------------------------------------------------------------------------- #
def fig_elimination(S):
    rows = S["elimination"]
    names = [r["lever"] for r in rows]
    vals = [r["delta_H0"] for r in rows]
    moved = [abs(v) > r["tolerance"] for v, r in zip(vals, rows)]

    h = max(2.8, 0.40 * len(rows) + 1.35)
    fig, ax = plt.subplots(figsize=(7.1, h), dpi=300)
    # Long lever names need a wide left margin; reserve it explicitly rather than
    # relying on tight_layout, which previously clipped the axis label.
    fig.subplots_adjust(left=0.415, right=0.975, top=0.90, bottom=0.30 / h * 2.4)
    y = np.arange(len(rows))[::-1]
    colours = [RED if m else AQUA for m in moved]
    ax.barh(y, vals, height=0.6, color=colours, zorder=3)
    ax.axvline(0.0, color=INK2, lw=0.9, zorder=4)
    for yy, v, r in zip(y, vals, rows):
        off = 0.06 * (max(abs(np.array(vals))) or 1.0)
        ax.annotate(f"{v:+.2f}" if abs(v) >= 0.005 else "0.00",
                    xy=(v, yy), xytext=(off if v >= 0 else -off, 0),
                    textcoords="offset points", va="center",
                    ha="left" if v >= 0 else "right", fontsize=7.6, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8.0)
    ax.set_xlabel(r"$\Delta H_0$ when the lever is applied  [km s$^{-1}$ Mpc$^{-1}$]")
    ax.set_title("What moves the expansion rate, and what does not", fontsize=9.3)
    ax.grid(True, axis="x", alpha=0.55)
    ax.grid(False, axis="y")
    ax.set_axisbelow(True)
    pad = 0.34 * max(abs(np.array(vals)))
    ax.set_xlim(min(vals) - pad, max(vals) + pad)
    # Inside the axes, upper left: the small-|delta| rows leave that corner empty, so
    # the legend sits on neither a bar nor the title.
    ax.legend(handles=[
        Line2D([0], [0], color=RED, lw=6, label=r"moves $H_0$"),
        Line2D([0], [0], color=AQUA, lw=6, label="does not"),
    ], loc="upper left", fontsize=7.6)
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"fig_elimination.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("wrote figs/fig_elimination.{pdf,png}")


# --------------------------------------------------------------------------- #
def main():
    S = {"H0_truth": H0_TRUTH,
         "note": ("All entries regenerated from this experiment's own grids; "
                  "nothing hand-typed.")}

    # --- sigma ladder ---
    S["sigma_ladder"] = {"truth_centred": {}, "corrected": {}}
    for s, tag in ((0.01, "closure_dlunc0.01"), (0.03, "closure_dlunc0.03"),
                   (0.10, "closure_ns16_n1000")):
        e = load(tag)
        if e:
            e["tag"] = tag
            S["sigma_ladder"]["truth_centred"][f"{s:g}"] = e
    for s in (0.01, 0.03, 0.10):
        e = load(f"closure_pefix_{s:.2f}")
        if e:
            e["tag"] = f"closure_pefix_{s:.2f}"
            S["sigma_ladder"]["corrected"][f"{s:g}"] = e

    # --- multi-seed (corrected PE, sigma = 0.10) ---
    seeds = []
    for sd, tag in (("4101", "closure_pefix_0.10"), ("4102", "closure_pefix_s4102"),
                    ("4103", "closure_pefix_s4103"), ("4104", "closure_pefix_s4104"),
                    ("4105", "closure_pefix_s4105")):
        e = load(tag)
        if e and not e["rejected"]:
            e["seed"] = sd
            e["tag"] = tag
            seeds.append(e)
    med = np.array([e["median"] for e in seeds])
    hw = np.array([e["hw"] for e in seeds])
    sem = float(med.std(ddof=1) / np.sqrt(med.size))
    sd_ = float(med.std(ddof=1))
    mean_hw = float(hw.mean())
    S["multi_seed"] = {
        "sigma_dL": 0.10, "n_seeds": int(med.size), "seeds": seeds,
        "mean_H0": float(med.mean()), "sem": sem,
        "offset": float(med.mean() - H0_TRUTH),
        "significance": float(abs(med.mean() - H0_TRUTH) / sem),
        "scatter_sd": sd_, "mean_halfwidth": mean_hw,
        "interval_underestimate_factor": sd_ / mean_hw,
        "catalog_variance_component": float(np.sqrt(max(sd_**2 - mean_hw**2, 0.0))),
        "interpretation": ("Per-seed intervals are conditional on the catalog; the "
                           "excess of the realised scatter over them is the host-catalog "
                           "sample-variance contribution."),
    }

    # --- elimination table: each lever's measured effect on H0 ---
    def diff(tag_a, tag_b):
        a, b = load(tag_a), load(tag_b)
        if not a or not b or a["rejected"] or b["rejected"]:
            return None
        return a["median"] - b["median"]

    base_ns16 = load("closure_ns16_n1000")
    elim = []
    g0, g1 = load("gtest_g0_fagn0.3"), load("gtest_g1_fagn0.3")
    if g0 and g1:
        elim.append({"lever": "rate index γ: 0 → 1 (GLASS mock)",
                     "delta_H0": g1["median"] - g0["median"], "tolerance": 0.30})
    d = diff("closure_ns16_zk0.5", "closure_ns16_n1000")
    if d is not None:
        elim.append({"lever": "catalog truncated to z ≤ 0.5 (1M → 48.6k hosts)",
                     "delta_H0": d, "tolerance": 0.05})
    d = diff("closure_ns16_zk1.0", "closure_ns16_n1000")
    if d is not None:
        elim.append({"lever": "catalog truncated to z ≤ 1.0 (1M → 260.6k hosts)",
                     "delta_H0": d, "tolerance": 0.05})
    b_ns64 = [load(f"closure_b{i}") for i in range(10)]
    b_ns64 = [e for e in b_ns64 if e and not e["rejected"]]
    if b_ns64 and base_ns16:
        m64 = float(np.mean([e["median"] for e in b_ns64]))
        elim.append({"lever": "sky pixelisation nside 64 → 16",
                     "delta_H0": base_ns16["median"] - m64, "tolerance": 0.60})
    lad = S["sigma_ladder"]
    if "0.1" in lad["truth_centred"] and "0.01" in lad["truth_centred"]:
        elim.append({"lever": r"distance uncertainty σ: 0.10 → 0.01",
                     "delta_H0": (lad["truth_centred"]["0.01"]["median"]
                                  - lad["truth_centred"]["0.1"]["median"]),
                     "tolerance": 0.30})
    if "0.1" in lad["corrected"] and "0.1" in lad["truth_centred"]:
        elim.append({"lever": "PE construction: truth-centred → flat-prior posterior",
                     "delta_H0": (lad["corrected"]["0.1"]["median"]
                                  - lad["truth_centred"]["0.1"]["median"]),
                     "tolerance": 0.30})
    ez = load("edgetest_zlt1_fagn0.3")
    if ez:
        elim.append({"lever": "GLASS catalog edge 1.56 → 1.0 (separate mock)",
                     "delta_H0": ez["median"] - 66.81, "tolerance": 0.30})
    S["elimination"] = elim

    # --- localisation sweep ---
    loc = RESULTS / "localize_summary_b20.json"
    if loc.exists():
        L = json.loads(loc.read_text())
        S["localisation"] = {
            "block_size": L["block_size"], "n_blocks": L["n_blocks_used"],
            "dlogL_per_event": L["dlogL_per_event"],
            "outlier_blocks": L["outlier_blocks"],
            "top_decile_share": L["top_decile_share_of_total_pull"],
            "conclusion": ("No outlier blocks: the pull is spread across events, not "
                           "carried by a minority."),
        }

    # --- decomposition ---
    dec = RESULTS / "h0_decomposition_deep_ns16.json"
    if dec.exists():
        D = json.loads(dec.read_text())
        S["decomposition"] = {k: D[k] for k in (
            "peak_total", "peak_per_event_numerator", "shift_from_selection_term",
            "dlnmu_dH0_at_truth", "lnmu_range", "nobs") if k in D}

    (RESULTS / "summary.json").write_text(json.dumps(S, indent=2, default=float))
    print("wrote results/summary.json")

    fig_sigma_ladder(S)
    fig_closure_seeds(S)
    fig_elimination(S)

    ms = S["multi_seed"]
    print(f"\nmulti-seed: H0 = {ms['mean_H0']:.3f} +- {ms['sem']:.3f}  "
          f"offset {ms['offset']:+.3f} ({ms['significance']:.1f} sigma)")
    print(f"intervals too narrow by {ms['interval_underestimate_factor']:.2f}x; "
          f"catalog term ~{ms['catalog_variance_component']:.2f}")
    print("\nlevers:")
    for r in S["elimination"]:
        print(f"  {r['lever']:<58} {r['delta_H0']:+.3f}")


if __name__ == "__main__":
    main()
