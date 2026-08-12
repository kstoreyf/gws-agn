#!/usr/bin/env python
"""a7 verdict: does the estimator's bias scale with pixel occupancy?

Reads the six a7 grids plus the two nside-32 midpoints (which were not rerun --
per_pixel comes from the archived a3 grid, selection from this campaign's a3),
fits the f_AGN offset against log10(lambda), and states the verdict against the
prediction registered BEFORE the run.

    python analyze_a7.py            # writes ../a7/results/a7_verdict.json + figs

Prediction (desi_darksirens_selection, registered in advance): their mechanism
depends on counts per pixel and nothing else, so if it is the same disease the
per_pixel offset must move along this axis. Flat under per_pixel => our bias is
NOT their disease and the cross-code story is weaker. Both outcomes reportable.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                       # selection_redo
ANALYSES = ROOT.parent
A7 = ROOT / "a7"
TRUTH_F = 0.295

# lambda = N_AGN / npix at m18: the axis variable. Occupancy per OCCUPIED pixel
# is the wrong variable -- it saturates near 1 as pixels empty out.
LAMBDA = {16: 8274 / 3072, 32: 8274 / 12288, 64: 8274 / 49152}

SOURCES = {
    ("per_pixel", 16): A7 / "results" / "joint_m18_ns16_per_pixel_s100.json",
    ("per_pixel", 64): A7 / "results" / "joint_m18_ns64_per_pixel_s100.json",
    ("per_pixel", 32): (ANALYSES.parent / "archive" /
                        "analysis_3_incomplete_catalog_H0_fagn" / "results" /
                        "joint_m18_s100.json"),
    ("selection", 16): A7 / "results" / "joint_m18_ns16_selection_s100.json",
    ("selection", 64): A7 / "results" / "joint_m18_ns64_selection_s100.json",
    ("selection", 32): (ANALYSES / "analysis_3_incomplete_catalog_H0_fagn" /
                        "results" / "joint_m18_s100.json"),
}
COLOUR = {"per_pixel": "#0072B2", "selection": "#009E73"}
INK, MUTED = "#1a1a1a", "#6b6b6b"


def policy_check():
    """The gate: the two empty-pixel policies must be bit-identical."""
    a = A7 / "results" / "policyprobe_ns64_zero_s100.json"
    b = A7 / "results" / "policyprobe_ns64_volume_s100.json"
    if not (a.exists() and b.exists()):
        return {"status": "PENDING"}
    da, db = json.loads(a.read_text()), json.loads(b.read_text())
    fa, fb = da["f"]["median"], db["f"]["median"]
    same = abs(fa - fb) < 1e-12
    return {"status": "INERT (verified)" if same else "LIVE -- AXIS CONFOUNDED",
            "f_median_zero": fa, "f_median_volume": fb, "abs_diff": abs(fa - fb),
            "note": ("Empty-pixel policy proven inert under field weighting, so "
                     "the axis is occupancy-only." if same else
                     "The policy is NOT inert: this axis measures policy x "
                     "occupancy and the grids below cannot be read as occupancy.")}


def main():
    rows, fits = [], {}
    for mode in ("per_pixel", "selection"):
        xs, ys, es = [], [], []
        for ns in (16, 32, 64):
            p = SOURCES[(mode, ns)]
            if not p.exists():
                continue
            d = json.loads(p.read_text())
            off = d["f"]["median"] - TRUTH_F
            ci90 = d["f"]["ci90"]
            xs.append(np.log10(LAMBDA[ns])); ys.append(off)
            es.append((ci90[1] - ci90[0]) / 2)
            rows.append({"c_mode": mode, "nside": ns, "lambda": LAMBDA[ns],
                         "f_median": d["f"]["median"], "f_offset": off,
                         "ci90": ci90, "H0_median": d["H0"]["median"]})
        if len(xs) >= 2:
            s, i = np.polyfit(xs, ys, 1)
            # slope uncertainty from the cells' own 90% half-widths
            n = len(xs)
            resid = np.array(ys) - (s * np.array(xs) + i)
            sx = np.std(xs) * np.sqrt(n)
            se = (np.sqrt(np.sum(resid**2) / max(n - 2, 1)) / max(sx, 1e-9)
                  if n > 2 else float(np.mean(es)) / max(sx, 1e-9))
            fits[mode] = {"slope_per_dex": float(s), "intercept": float(i),
                          "slope_err": float(se), "n_points": n,
                          "span_dex": float(max(xs) - min(xs))}

    pol = policy_check()
    verdict = "PENDING"
    if fits.get("per_pixel") and fits.get("selection") and \
            pol["status"].startswith("INERT"):
        pp, se = fits["per_pixel"], fits["selection"]
        pp_sig = abs(pp["slope_per_dex"]) > 2 * max(pp["slope_err"], 1e-9)
        se_flat = abs(se["slope_per_dex"]) < 2 * max(se["slope_err"], 1e-9)
        if pp_sig and se_flat:
            verdict = ("CONFIRMED -- per_pixel offset scales with occupancy at "
                       "fixed completeness; selection is flat. Same mechanism as "
                       "the peer team's per-pixel catalog prior.")
        elif pp_sig:
            verdict = ("PARTIAL -- per_pixel scales with occupancy, but the "
                       "selection control is NOT flat, so something else moves "
                       "with nside too.")
        else:
            verdict = ("REFUTED -- per_pixel offset is flat in occupancy at "
                       "fixed completeness. Our bias is NOT the peer team's "
                       "low-occupancy disease; the cross-code story does not hold.")
    elif not pol["status"].startswith(("INERT", "PENDING")):
        verdict = "INVALID -- " + pol["note"]

    out = {"_what": __doc__.split("\n")[0], "lambda_definition":
           "N_AGN / npix at m18 (mean per pixel INCLUDING empties)",
           "policy_gate": pol, "cells": rows, "fits": fits, "verdict": verdict}
    (A7 / "results").mkdir(parents=True, exist_ok=True)
    (A7 / "results" / "a7_verdict.json").write_text(json.dumps(out, indent=2))

    print(f"\n  policy gate: {pol['status']}")
    for m, f in fits.items():
        print(f"  {m:<10} slope {f['slope_per_dex']:+.4f} +/- {f['slope_err']:.4f} "
              f"per dex  over {f['span_dex']:.2f} dex  ({f['n_points']} pts)")
    print(f"\n  VERDICT: {verdict}\n")

    # ---- figure ----------------------------------------------------------
    if rows:
        fig, ax = plt.subplots(figsize=(6.6, 4.3))
        ax.axhline(0, color="black", ls="--", lw=1.2, zorder=1)
        for mode in ("per_pixel", "selection"):
            pts = [r for r in rows if r["c_mode"] == mode]
            if not pts:
                continue
            x = [np.log10(r["lambda"]) for r in pts]
            y = [r["f_offset"] for r in pts]
            lo = [r["f_offset"] - (r["ci90"][0] - TRUTH_F) for r in pts]
            hi = [(r["ci90"][1] - TRUTH_F) - r["f_offset"] for r in pts]
            ax.errorbar(x, y, yerr=[lo, hi], fmt="o-", color=COLOUR[mode],
                        ms=7, lw=2.0, capsize=0, label=mode, zorder=3)
            ax.plot(x, y, "o", color="white", ms=2.5, zorder=4)
        for ns in (16, 32, 64):
            ax.annotate(f"nside {ns}", (np.log10(LAMBDA[ns]), ax.get_ylim()[1]),
                        fontsize=7.5, color=MUTED, ha="center", va="top")
        ax.set_xlabel(r"$\log_{10}\ \lambda$   (AGN per pixel, incl. empty)",
                      fontsize=10, color=INK)
        ax.set_ylabel(r"$f_{\rm AGN}$ offset from truth", fontsize=10, color=INK)
        ax.set_title("Does the bias follow OCCUPANCY at fixed completeness?\n"
                     "m18, C = 9.5% throughout; only the pixel binning changes",
                     fontsize=11, color=INK)
        ax.legend(frameon=False, fontsize=9, labelcolor=INK)
        ax.text(0.02, 0.03, "bars = 90% CI", transform=ax.transAxes,
                fontsize=8, color=MUTED)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(axis="y", color="#ececec", lw=0.7); ax.set_axisbelow(True)
        (A7 / "figs").mkdir(parents=True, exist_ok=True)
        for ext in ("pdf", "png"):
            fig.savefig(A7 / "figs" / f"fig_a7_occupancy.{ext}", dpi=200,
                        bbox_inches="tight", facecolor="white")
        print(f"  wrote a7/figs/fig_a7_occupancy.pdf and .png")


if __name__ == "__main__":
    main()
