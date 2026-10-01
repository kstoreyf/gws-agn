#!/usr/bin/env python3
"""Publication figures for experiment_completeness_anchored.

  figs/fig_completeness_ladder.{pdf,png}  the survey completeness C(z) actually
      imposed, and the recovered H0 across the ladder against the complete-catalog
      control
  results/summary.json                    every number quoted

Pure post-processing: h5py/numpy/matplotlib, CPU, no darksirens import.

The comparison is DIFFERENTIAL against the complete-catalog control at the same
distance uncertainty, because the absolute offset carries the unresolved baseline bias
measured in ../experiment_matched_mock (-1.61 +- 0.49). The question here is whether
incompleteness adds bias beyond that baseline, not whether the total is zero.
"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent.parent
RESULTS, FIGS = BASE / "results", BASE / "figs"
FIGS.mkdir(parents=True, exist_ok=True)
H0_TRUTH = 67.74

BLUE, AQUA, YELLOW, GREEN, VIOLET = ("#2a78d6", "#1baf7a", "#eda100",
                                     "#008300", "#4a3aa7")
INK, INK2, INK3, GRIDCOL, BASELINE = ("#0b0b0b", "#52514e", "#898781",
                                      "#e1e0d9", "#c3c2b7")
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "font.family": "serif",
    "font.serif": ["DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 9, "axes.labelsize": 9.5, "axes.titlesize": 9.3,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 7.8,
    "axes.edgecolor": INK2, "axes.linewidth": 0.7, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "grid.color": GRIDCOL, "grid.linewidth": 0.6,
    "legend.frameon": False, "lines.solid_capstyle": "round",
})

LADDER = [("c100", "complete", GREEN, "o"), ("m20", r"$m<20$", BLUE, "s"),
          ("m19", r"$m<19$", AQUA, "D"), ("m18", r"$m<18$", YELLOW, "^")]


def main():
    S = {"H0_truth": H0_TRUTH,
         "framing": ("Differential against the complete-catalog control; the absolute "
                     "offset carries the unresolved baseline bias from "
                     "../experiment_matched_mock."),
         "anchor": json.loads((RESULTS / "density_model_anchor.json").read_text()),
         "levels": {}}

    for tag, lab, _, _ in LADDER:
        j = json.loads((RESULTS / f"anch_{tag}.json").read_text())
        with h5py.File(BASE / "data_derived" / f"survey_{tag}_ns16.h5", "r") as f:
            comp = json.loads(f.attrs["completeness_json"])
            empty = float(f.attrs["empty_pixel_fraction"])
            nused = int(f.attrs["n_galaxies_used"])
        h = j["H0"]
        S["levels"][tag] = {
            "label": lab, "completeness_within_z_ref": comp.get("within_z_ref", 1.0),
            "completeness_all_z": comp.get("all_z", 1.0),
            "C_of_z_bins": comp.get("C_of_z_bins"),
            "mag_limit": comp.get("mag_limit"),
            "n_hosts": nused, "empty_pixel_fraction": empty,
            "H0_median": h["median"], "ci68": h["ci68"],
            "hw": 0.5 * (h["ci68"][1] - h["ci68"][0]),
            "offset": h["median"] - H0_TRUTH,
            "n_rejected_cells": j["n_neginf_cells"], "n_evals": j["n_evals"],
        }

    ctrl = S["levels"]["c100"]
    for tag, e in S["levels"].items():
        e["offset_vs_control"] = e["offset"] - ctrl["offset"]
        e["sigma_vs_control"] = abs(e["offset_vs_control"]) / np.hypot(
            e["hw"], ctrl["hw"]) if tag != "c100" else 0.0
    S["verdict"] = {
        "max_abs_offset_vs_control": max(
            abs(e["offset_vs_control"]) for e in S["levels"].values()),
        "max_sigma_vs_control": max(e["sigma_vs_control"] for e in S["levels"].values()),
        "any_cells_rejected": any(e["n_rejected_cells"] > 0
                                  for e in S["levels"].values()),
        "interval_growth": {t: S["levels"][t]["hw"] / ctrl["hw"] for t in S["levels"]},
    }
    (RESULTS / "summary.json").write_text(json.dumps(S, indent=2, default=float))
    print("wrote results/summary.json")

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.1, 3.0), dpi=300)

    # ---- left: the imposed C(z) ----
    for tag, lab, col, mk in LADDER:
        e = S["levels"][tag]
        cb = e["C_of_z_bins"]
        if cb is None:
            axL.plot([0, 0.27], [1, 1], color=col, lw=1.6, marker=mk, ms=4.5,
                     mec="white", mew=0.6, label=f"{lab} (100%)")
            continue
        edges = np.asarray(cb["edges"], dtype=float)
        zc = 0.5 * (edges[1:] + edges[:-1])
        C = np.asarray([np.nan if c is None else c for c in cb["C"]], dtype=float)
        axL.plot(zc, C, color=col, lw=1.6, marker=mk, ms=4.5, mec="white", mew=0.6,
                 label=f"{lab} ({100*e['completeness_within_z_ref']:.0f}%)")
    axL.set_xlim(0, 0.27)
    axL.set_ylim(0, 1.06)
    axL.set_xlabel("redshift  $z$")
    axL.set_ylabel("survey completeness  $C(z)$")
    axL.set_title("Imposed isotropic completeness")
    axL.grid(True, alpha=0.55)
    axL.set_axisbelow(True)
    axL.legend(loc="lower left", title="flux limit (within horizon)",
               title_fontsize=7.4)

    # ---- right: recovered H0 vs completeness ----
    axR.axhspan(ctrl["offset"] - ctrl["hw"], ctrl["offset"] + ctrl["hw"],
                color=GREEN, alpha=0.13, lw=0, zorder=1)
    axR.axhline(ctrl["offset"], color=GREEN, lw=1.2, zorder=2,
                label="complete-catalog control")
    axR.axhline(0.0, color=INK, lw=1.0, ls=(0, (2, 2)), zorder=3, label="truth")
    for tag, lab, col, mk in LADDER:
        e = S["levels"][tag]
        x = 100 * e["completeness_within_z_ref"]
        axR.errorbar([x], [e["offset"]], yerr=[[e["hw"]], [e["hw"]]], color=col,
                     marker=mk, ms=6.5, mfc=col, mec="white", mew=0.8, lw=0,
                     elinewidth=1.2, capsize=3, zorder=5)
        # Above the upper bar: below it, the leftmost label ran off the axis.
        axR.annotate(f"{e['empty_pixel_fraction']*100:.0f}% empty",
                     xy=(x, e["offset"] + e["hw"]), xytext=(0, 6),
                     textcoords="offset points", ha="center", va="bottom",
                     fontsize=6.8, color=INK3)
    axR.set_xlim(106, 15)          # explicit, so annotations cannot clip
    axR.set_xlabel(r"completeness within the horizon  $C(z\!\leq\!0.27)$  [%]")
    axR.set_ylabel(r"$H_0$ offset from truth  [km s$^{-1}$ Mpc$^{-1}$]")
    axR.set_title("Recovery at anchored $n_0$")
    axR.grid(True, alpha=0.55)
    axR.set_axisbelow(True)
    axR.legend(loc="lower left")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"fig_completeness_ladder.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("wrote figs/fig_completeness_ladder.{pdf,png}")

    print("\nlevel   C(z<=0.27)  empty%   H0 offset        vs control")
    for tag, lab, _, _ in LADDER:
        e = S["levels"][tag]
        print(f"  {tag:5s} {100*e['completeness_within_z_ref']:7.1f}% "
              f"{100*e['empty_pixel_fraction']:7.2f}  "
              f"{e['offset']:+6.3f} +- {e['hw']:.3f}   "
              f"{e['offset_vs_control']:+6.3f} ({e['sigma_vs_control']:.1f} sigma)")
    v = S["verdict"]
    print(f"\nmax |offset vs control| = {v['max_abs_offset_vs_control']:.3f} "
          f"({v['max_sigma_vs_control']:.1f} sigma); any cells rejected: "
          f"{v['any_cells_rejected']}")


if __name__ == "__main__":
    main()
