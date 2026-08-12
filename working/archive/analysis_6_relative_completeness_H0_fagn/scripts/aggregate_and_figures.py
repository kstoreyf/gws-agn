#!/usr/bin/env python
"""Deterministic aggregation + figures for the relative-completeness surface.

Reads   results/joint_g<GLEV>_a<ALEV>_s100.h5                     (this directory)
        ../analysis_3_incomplete_catalog_H0_fagn/results/joint_<lev>_s100.h5
        ../analysis_4_density_anchoring_H0_fagn/results/joint_m18_oracle_s100.h5
        ../../data/seed100/surveys/surveys_meta.json               (completeness)
Writes  results/surface_summary.json
        figs/fig_surface_f.{pdf,png}
        figs/fig_surface_h0.{pdf,png}
        figs/fig_ratio_collapse.{pdf,png}

The four diagonal/oracle cells are REFERENCED from analyses 3 and 4, never rerun:
all copies of scan_h0f.py are byte-identical, so the surface and its diagonal
share one estimator by construction.  Marginals are recomputed here for every
cell with the project's flat-prior/trapezoid convention so all twelve are
summarised identically.
"""
import json
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RES, FIGS = ROOT / "results", ROOT / "figs"
A3 = ROOT.parent / "analysis_3_incomplete_catalog_H0_fagn" / "results"
A4 = ROOT.parent / "analysis_4_density_anchoring_H0_fagn" / "results"
META = ROOT.parent.parent / "data" / "seed100" / "surveys" / "surveys_meta.json"
FIGS.mkdir(exist_ok=True)

H0_TRUTH = 67.74
F_REALISED = 0.295
GLEVS = ["m20", "m19", "m18"]                     # galaxy depth, bright -> faint
ALEVS = ["complete", "m20", "m19", "m18"]         # AGN depth
BINOM_SD = 0.014491376746189439                   # per-realisation, sqrt(.3*.7/1000)


def marginal_ci(x, logp_1d, levels=(0.68, 0.90)):
    logp_1d = np.asarray(logp_1d, dtype=float)
    x = np.asarray(x, dtype=float)
    fin = np.isfinite(logp_1d)
    p = np.exp(np.where(fin, logp_1d, -np.inf) - np.nanmax(logp_1d[fin]))
    p = p / np.trapz(p, x)
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (p[1:] + p[:-1]) * np.diff(x))])
    cdf /= cdf[-1]
    out = {"median": float(np.interp(0.5, cdf, x))}
    for lev in levels:
        out["ci{:.0f}".format(lev * 100)] = [
            float(np.interp(0.5 - lev / 2, cdf, x)),
            float(np.interp(0.5 + lev / 2, cdf, x))]
    out["halfwidth68"] = float((out["ci68"][1] - out["ci68"][0]) / 2)
    return out


def add_truth(block, truth):
    block["truth"] = truth
    block["offset"] = block["median"] - truth
    block["pull"] = block["offset"] / block["halfwidth68"] if block["halfwidth68"] else None
    for lev in ("ci68", "ci90"):
        lo, hi = block[lev]
        block["truth_in_" + lev] = bool(lo <= truth <= hi)
    return block


def summarize_grid(path, source):
    with h5py.File(path, "r") as f:
        H0, fv = f["H0_grid"][:], f["f_grid"][:]
        logL = f["log_likelihood"][:]
        rejected = int(np.sum(f["guard/rejected"][:])) if "guard/rejected" in f else 0
        attrs = {k: f.attrs[k] for k in
                 ("arg_log10n0", "arg_log10n0_c2", "merge_sha_checked",
                  "arg_survey_path", "steady_state_median_seconds",
                  "wall_seconds_total") if k in f.attrs}
    m = np.nanmax(logL[np.isfinite(logL)])
    P = np.exp(np.where(np.isfinite(logL), logL, -np.inf) - m)
    i, j = np.unravel_index(np.nanargmax(logL), logL.shape)
    return {
        "source": source,
        "H0": add_truth(marginal_ci(H0, np.log(np.trapz(P, fv, axis=1))), H0_TRUTH),
        "f_AGN": add_truth(marginal_ci(fv, np.log(np.trapz(P, H0, axis=0))), F_REALISED),
        "map": {"H0": float(H0[i]), "f_AGN": float(fv[j])},
        "significance_f": None,          # filled below
        "cells_rejected": rejected,
        "anchors": {"log10n0": float(attrs.get("arg_log10n0", np.nan)),
                    "log10n0_c2": float(attrs.get("arg_log10n0_c2", np.nan))},
        "merge_sha_checked": str(attrs.get("merge_sha_checked", "")),
        "s_per_eval": float(attrs.get("steady_state_median_seconds", np.nan)),
        "wall_hours": float(attrs.get("wall_seconds_total", np.nan)) / 3600.0,
    }


# --------------------------------------------------------------------------- #
# completeness of every survey level, within the horizon
# --------------------------------------------------------------------------- #
meta = json.loads(META.read_text())["completeness"]
C = {lev: {"gal": meta[lev]["gal"]["C_within_horizon"],
           "agn": meta[lev]["agn"]["C_within_horizon"]} for lev in meta}

# --------------------------------------------------------------------------- #
# load the twelve cells
# --------------------------------------------------------------------------- #
cells, missing = {}, []
for glev in GLEVS:
    for alev in ALEVS:
        key = f"{glev}/{alev}"
        if glev == alev:                                   # analysis 3 diagonal
            p, src = A3 / f"joint_{glev}_s100.h5", "analysis_3 (referenced)"
        elif glev == "m18" and alev == "complete":         # analysis 4 oracle
            p, src = A4 / "joint_m18_oracle_s100.h5", "analysis_4 oracle (referenced)"
        else:
            p, src = RES / f"joint_g{glev}_a{alev}_s100.h5", "analysis_6"
        if not p.exists():
            missing.append(key)
            continue
        c = summarize_grid(p, src)
        c["C_gal"] = C[glev]["gal"]
        c["C_agn"] = C[alev]["agn"]
        c["log10_ratio_agn_over_gal"] = float(np.log10(c["C_agn"] / c["C_gal"]))
        c["significance_f"] = c["f_AGN"]["median"] / c["f_AGN"]["halfwidth68"]
        cells[key] = c

if missing:
    print(f"MISSING {len(missing)} cells (aggregates use present cells only): {missing}")


# --------------------------------------------------------------------------- #
# the collapse test: what predicts the f_AGN offset?
# --------------------------------------------------------------------------- #
def linfit(xs, ys):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    if xs.size < 3 or np.ptp(xs) == 0:
        return None
    b, a = np.polyfit(xs, ys, 1)
    pred = a + b * xs
    ss_res = float(np.sum((ys - pred) ** 2))
    ss_tot = float(np.sum((ys - ys.mean()) ** 2))
    return {"slope": float(b), "intercept": float(a),
            "r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else None,
            "rms_residual": float(np.sqrt(ss_res / xs.size)), "n": int(xs.size)}


keys = sorted(cells)
f_off = [cells[k]["f_AGN"]["offset"] for k in keys]
h0_off = [cells[k]["H0"]["offset"] for k in keys]
predictors = {
    "log10_ratio_agn_over_gal": [cells[k]["log10_ratio_agn_over_gal"] for k in keys],
    "log10_C_gal": [float(np.log10(cells[k]["C_gal"])) for k in keys],
    "log10_C_agn": [float(np.log10(cells[k]["C_agn"])) for k in keys],
}
collapse = {"f_AGN_offset": {n: linfit(v, f_off) for n, v in predictors.items()},
            "H0_offset": {n: linfit(v, h0_off) for n, v in predictors.items()}}

# Per-row structure.  A single global line in log-ratio is only an approximation:
# each row rises with relative completeness and then SATURATES once the AGN
# catalog is a few times more complete than the galaxy catalog, at a level that
# grows steeply as the galaxy survey shallows.  Recorded per row so the shape is
# not hidden behind one R^2.
SAT_DEX = 0.40                       # above this log-ratio each row is flat
rows = {}
for g in GLEVS:
    ks = sorted([k for k in keys if k.startswith(g + "/")],
                key=lambda k: cells[k]["log10_ratio_agn_over_gal"])
    if not ks:
        continue
    xr = [cells[k]["log10_ratio_agn_over_gal"] for k in ks]
    yr = [cells[k]["f_AGN"]["offset"] for k in ks]
    rise = [(x, y) for x, y in zip(xr, yr) if x <= SAT_DEX]
    flat = [y for x, y in zip(xr, yr) if x > SAT_DEX]
    rows[g] = {
        "C_gal": cells[ks[0]]["C_gal"],
        "cells": ks,
        "log10_ratio": xr,
        "f_offset": yr,
        "H0_offset": [cells[k]["H0"]["offset"] for k in ks],
        "fit_all": linfit(xr, yr),
        "fit_rising_branch": linfit([p[0] for p in rise], [p[1] for p in rise]),
        # rows with only two cells on the rising branch get a secant instead
        "secant_slope_two_lowest": (
            float((yr[1] - yr[0]) / (xr[1] - xr[0])) if len(xr) >= 2
            and xr[1] != xr[0] else None),
        "saturation": {"log10_ratio_threshold": SAT_DEX,
                       "n_cells_above": len(flat),
                       "mean_f_offset_above": float(np.mean(flat)) if flat else None,
                       "spread_above": float(np.ptp(flat)) if len(flat) > 1 else None},
    }
collapse["per_galaxy_depth"] = rows
collapse["_reading"] = (
    "A single line in log10(C_AGN/C_GAL) fits all 12 cells with R^2 ~0.89 and rms "
    "0.024 in f_AGN — above the realisation's own binomial scatter (0.0145), so "
    "the single-law description is good to about 0.02 in f_AGN and no better. The "
    "per-row fits show why: each galaxy depth rises with relative completeness "
    "and then saturates, and the saturation level grows as the galaxy survey "
    "shallows. GAL m18 has no cell below ratio 1 (m18 is the shallowest survey "
    "level available), so the sign change is demonstrated only at m20 and m19.")

summary = {
    "_what": "the joint (H0, f_AGN) measurement over the GAL-depth x AGN-depth "
             "plane at seed 100, both completion densities at truth; the "
             "diagonal and the oracle cell are referenced from analyses 3 and 4. "
             "Asks whether the f_AGN bias is a function of the relative "
             "completeness C_AGN / C_GAL rather than of either depth alone.",
    "config": {"seed": 100, "lane": "targeted", "glevs": GLEVS, "alevs": ALEVS,
               "H0_truth": H0_TRUTH, "f_realised": F_REALISED,
               "binomial_sd_per_realisation": BINOM_SD,
               "anchors": "both at truth (log10n0 = -3, log10n0_c2 = -5)",
               "grid": "H0 [50,100] x 201 * f [0,1] x 41"},
    "completeness_within_horizon": C,
    "cells": cells,
    "missing_cells": missing,
    "collapse_test": collapse,
    "totals": {"n_cells": len(cells),
               "cells_rejected_total": int(sum(c["cells_rejected"] for c in cells.values())),
               "wall_hours_new_grids": float(sum(
                   c["wall_hours"] for c in cells.values()
                   if c["source"] == "analysis_6" and np.isfinite(c["wall_hours"])))},
}
(RES / "surface_summary.json").write_text(json.dumps(summary, indent=1))
print("wrote", RES / "surface_summary.json")


# --------------------------------------------------------------------------- #
# figs 1 & 2: the surface as a matrix
# --------------------------------------------------------------------------- #
def surface_fig(param, truth, label, fname, cmap, unit=""):
    M = np.full((len(GLEVS), len(ALEVS)), np.nan)
    for gi, g in enumerate(GLEVS):
        for ai, a in enumerate(ALEVS):
            c = cells.get(f"{g}/{a}")
            if c:
                M[gi, ai] = c[param]["offset"]
    v = np.nanmax(np.abs(M)) if np.isfinite(M).any() else 1.0
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    im = ax.imshow(M, cmap=cmap, vmin=-v, vmax=v, aspect="auto")
    for gi, g in enumerate(GLEVS):
        for ai, a in enumerate(ALEVS):
            c = cells.get(f"{g}/{a}")
            if not c:
                ax.text(ai, gi, "—", ha="center", va="center", color="0.5")
                continue
            ref = c["source"] != "analysis_6"
            ax.text(ai, gi, f"{c[param]['offset']:+.3f}\n$\\pm${c[param]['halfwidth68']:.3f}",
                    ha="center", va="center", fontsize=9,
                    fontstyle="italic" if ref else "normal",
                    color="k")
            if ref:
                ax.add_patch(plt.Rectangle((ai - .5, gi - .5), 1, 1, fill=False,
                                           ec="k", lw=2, ls="--"))
    ax.set_xticks(range(len(ALEVS)))
    ax.set_xticklabels([a if a == "complete" else f"$m<{a[1:]}$" for a in ALEVS])
    ax.set_yticks(range(len(GLEVS)))
    ax.set_yticklabels([f"$m<{g[1:]}$" for g in GLEVS])
    ax.set_xlabel("AGN survey depth")
    ax.set_ylabel("galaxy survey depth")
    ax.set_title(f"{label} offset from truth{unit}\n"
                 "(dashed: referenced from analyses 3/4, italic)", fontsize=11)
    fig.colorbar(im, ax=ax, label=f"{label} offset")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"{fname}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", FIGS / f"{fname}.pdf/.png")


surface_fig("f_AGN", F_REALISED, r"$f_{\rm AGN}$", "fig_surface_f", "RdBu_r")
surface_fig("H0", H0_TRUTH, r"$H_0$", "fig_surface_h0", "PuOr_r",
            unit=r"  [km s$^{-1}$ Mpc$^{-1}$]")


# --------------------------------------------------------------------------- #
# fig 3: does the f_AGN offset collapse onto the completeness ratio?
# --------------------------------------------------------------------------- #
fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.3))
CG = {"m20": "C0", "m19": "C2", "m18": "C3"}
for ax, pname, xlabel in [
        (axes[0], "log10_ratio_agn_over_gal",
         r"$\log_{10}\,(C_{\rm AGN} / C_{\rm GAL})$"),
        (axes[1], "log10_C_gal", r"$\log_{10}\,C_{\rm GAL}$")]:
    for g in GLEVS:
        ks = [k for k in keys if k.startswith(g + "/")]
        xs = ([cells[k]["log10_ratio_agn_over_gal"] for k in ks]
              if pname == "log10_ratio_agn_over_gal"
              else [float(np.log10(cells[k]["C_gal"])) for k in ks])
        ys = [cells[k]["f_AGN"]["offset"] for k in ks]
        es = [cells[k]["f_AGN"]["halfwidth68"] for k in ks]
        if pname == "log10_ratio_agn_over_gal":
            o = np.argsort(xs)
            ax.plot(np.asarray(xs)[o], np.asarray(ys)[o], "-", color=CG[g],
                    lw=1.6, alpha=0.55, zorder=1)
        ax.errorbar(xs, ys, yerr=es, fmt="o", color=CG[g], ms=7, capsize=3,
                    label=rf"GAL $m<{g[1:]}$", zorder=3)
    fit = collapse["f_AGN_offset"][pname]
    if fit:
        xx = np.linspace(min(predictors[pname]), max(predictors[pname]), 50)
        ax.plot(xx, fit["intercept"] + fit["slope"] * xx, "k-", lw=1.4, alpha=0.7,
                label=rf"fit: $R^2={fit['r2']:.2f}$, rms {fit['rms_residual']:.3f}")
    ax.axhline(0, color="k", ls="--", lw=1.2)
    ax.axhspan(-BINOM_SD, BINOM_SD, color="0.85", alpha=0.7, zorder=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$f_{\rm AGN}$ offset from realised")
    ax.legend(frameon=False, fontsize=8.5)
axes[0].set_title("relative completeness")
axes[1].set_title("galaxy completeness alone")
fig.suptitle("What predicts the AGN-fraction bias?  "
             "(grey band: the realisation's own binomial scatter)", y=1.02)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(FIGS / f"fig_ratio_collapse.{ext}", dpi=200, bbox_inches="tight")
plt.close(fig)
print("wrote", FIGS / "fig_ratio_collapse.pdf/.png")


# --------------------------------------------------------------------------- #
print(f"\n--- surface, {len(cells)}/12 cells ---")
print("GAL  AGN        C_agn/C_gal   f_AGN            off      H0            off")
for k in keys:
    c = cells[k]
    print(f"{k:12s} {10 ** c['log10_ratio_agn_over_gal']:8.2f}   "
          f"{c['f_AGN']['median']:.3f}+-{c['f_AGN']['halfwidth68']:.3f}  "
          f"{c['f_AGN']['offset']:+.3f}   "
          f"{c['H0']['median']:.2f}+-{c['H0']['halfwidth68']:.2f}  "
          f"{c['H0']['offset']:+.2f}"
          + ("   [ref]" if c["source"] != "analysis_6" else ""))
print("\n--- collapse test (f_AGN offset) ---")
for n, fit in collapse["f_AGN_offset"].items():
    if fit:
        print(f"{n:28s} slope {fit['slope']:+.4f}  R2 {fit['r2']:+.3f}  "
              f"rms {fit['rms_residual']:.4f}")
print("guard rejections total:", summary["totals"]["cells_rejected_total"])

print("\n--- per galaxy depth: rise then saturation ---")
for g, r in collapse["per_galaxy_depth"].items():
    sat = r["saturation"]
    fr, sec = r["fit_rising_branch"], r["secant_slope_two_lowest"]
    slope = (f"rising slope {fr['slope']:+.3f}/dex (fit, n={fr['n']})" if fr
             else f"low-ratio secant {sec:+.3f}/dex" if sec is not None
             else "no slope available")
    line = f"GAL {g} (C_gal={r['C_gal']:.4f}): {slope}"
    if sat["mean_f_offset_above"] is not None:
        line += (f" | saturates at {sat['mean_f_offset_above']:+.3f}"
                 f" (n={sat['n_cells_above']}, spread {sat['spread_above'] or 0:.3f})")
    print(line)
    print("     ratios " + " ".join(f"{x:+.2f}" for x in r["log10_ratio"])
          + "  ->  f_off " + " ".join(f"{y:+.3f}" for y in r["f_offset"]))
print("\nH0 vs relative completeness: R2 =",
      round(collapse["H0_offset"]["log10_ratio_agn_over_gal"]["r2"], 4))
