#!/usr/bin/env python3
"""Estimator equivalence: `dark_sirens` (complete-catalog limit) vs `dark_sirens_complete`.

Reads the H0 log-likelihood grids written by ``run_equivalence.sh`` and answers,
per configuration: do the two estimators put the SAME number in every cell, and
if not, where and by how much?

Writes
    results/equivalence_summary.json   every number quoted anywhere
    figs/fig_equivalence.{pdf,png}     the primary arm's per-cell difference and
                                       the secondary arm's posterior shifts

Cell-level comparison is on the raw IEEE-754 float64 bit patterns, not on a
tolerance: "bitwise" means the two grids are the same 64 bits in every cell.
darksirens enables ``jax_enable_x64`` at import (``redshift/grid.py``,
``utils/cosmology.py``), so the likelihood is evaluated in double precision and
the stored grid carries every bit of it.

Verdicts, per configuration and per arm:
    bitwise                 all cells identical bit for bit
    float-level             max |Delta logL| < 1e-9 nats
    structurally different  anything larger, with the shape of the difference
                            described in the JSON

Arms (all on the float64 survey copies — see run_equivalence.sh for why):
    primary    ds_*      dark_sirens at log10n0 = -12
    deep       dsdeep_*  dark_sirens at log10n0 = -24
    secondary  dstrue_*  dark_sirens at the tracer's TRUE density
                         (CHARACTERIZATION, never pass/fail)
plus two book-keeping sections that are not the equivalence question:
    blocker    f32_ds*   the general model on the survey files as shipped
    precision  f32_dsc_* the reference model's own float32-vs-float64 shift
"""
import json
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
FIGS = ROOT / "figs"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from scan_h0f import marginal_ci  # noqa: E402  — the driver's own posterior convention

TRUTH = 67.74
FLOAT_LEVEL_TOL = 1e-9   # nats

# key -> (tracer, event-set label, true log10 n0 of that tracer)
CONFIGS = [
    ("gal_all",     "GAL", "all 1000 events", -3.0),
    ("gal_matched", "GAL", "720 GAL-hosted",  -3.0),
    ("agn_all",     "AGN", "all 1000 events", -5.0),
    ("agn_matched", "AGN", "280 AGN-hosted",  -5.0),
]
ARMS = [("primary", "ds"), ("deep", "dsdeep"), ("secondary", "dstrue")]


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def load_grid(tag):
    path = RES / f"{tag}.h5"
    if not path.exists():
        return None
    with h5py.File(path, "r") as f:
        out = {
            "tag": tag,
            "path": str(path),
            "H0": np.asarray(f["H0_grid"][:], dtype=np.float64),
            "logL": np.asarray(f["log_likelihood"][:], dtype=np.float64),
            "n_rejected": int(f["guard"].attrs["n_rejected"]),
            "labels": json.loads(f.attrs["labels"]),
            "base_coord": json.loads(f.attrs["base_coord_labeled"]),
            "universe_model": str(f.attrs["arg_universe_model"]),
            "gw_path": str(f.attrs["arg_gw_path"]),
            "survey_path": json.loads(str(f.attrs["arg_survey_path"]))[0],
            "gwselection_path": str(f.attrs["arg_gwselection_path"]),
            "kde_window": str(f.attrs["arg_kde_window"]),
            "sel_batch_size": str(f.attrs["arg_sel_batch_size"]),
            "pe_event_block": str(f.attrs["arg_pe_event_block"]),
            "catalog_sky_weighting": str(f.attrs["arg_catalog_sky_weighting"]),
            "max_likelihood_variance": float(f.attrs["max_likelihood_variance_effective"]),
            "darksirens_git_sha": str(f.attrs["darksirens_git_sha"]),
            "jax_devices": str(f.attrs["jax_devices"]),
            "logL_stored_dtype": str(f["log_likelihood"].dtype),
            "steady_state_median_seconds": float(f.attrs["steady_state_median_seconds"]),
        }
        if "Neff" in f["guard"]:
            neff = np.asarray(f["guard"]["Neff"][:], dtype=np.float64)
            fin = neff[np.isfinite(neff)]
            out["Neff_min"] = float(fin.min()) if fin.size else None
    return out


def bit_equal(a, b):
    """Elementwise IEEE-754 bit equality of two float64 arrays."""
    ua = np.ascontiguousarray(a, dtype=np.float64).view(np.uint64)
    ub = np.ascontiguousarray(b, dtype=np.float64).view(np.uint64)
    return ua == ub


def posterior(H0, logL):
    """Flat-prior posterior summary, the driver's own convention, full precision."""
    ll = np.where(np.isfinite(logL), logL, -np.inf)
    block = marginal_ci(H0, ll)
    finite = np.isfinite(ll)
    if finite.any():
        imax = int(np.nanargmax(np.where(finite, ll, np.nan)))
        block["map"] = float(H0[imax])
        block["logL_max"] = float(ll[finite].max())
    else:
        block["map"] = float("nan")
        block["logL_max"] = float("nan")
    return block


def describe_shape(H0, delta, differs):
    """Human description of WHERE and HOW the two grids differ."""
    if not differs.any():
        return {"n_differing": 0, "description": "no cell differs"}
    idx = np.flatnonzero(differs)
    d = delta[differs]
    finite = np.isfinite(d)
    contiguous = bool(idx.size == (idx[-1] - idx[0] + 1))
    if not finite.all():
        sign_txt = "at least one cell is non-finite in exactly one of the two grids"
    else:
        s = np.sign(d)
        if np.all(s > 0):
            sign_txt = "general model above the complete model everywhere"
        elif np.all(s < 0):
            sign_txt = "general model below the complete model everywhere"
        else:
            sign_txt = (f"mixed sign ({int(np.sum(s > 0))} cells above, "
                        f"{int(np.sum(s < 0))} below)")
    df = d[finite]
    rng = float(df.max() - df.min()) if df.size else float("nan")
    med = float(np.median(df)) if df.size else float("nan")
    offset_like = bool(idx.size == H0.size and df.size == d.size
                       and rng <= 1e-3 * max(abs(med), 1e-300))
    # monotone in H0?
    mono = None
    if df.size == H0.size and df.size > 2:
        dd = np.diff(df)
        mono = ("increasing with H0" if np.all(dd >= 0)
                else "decreasing with H0" if np.all(dd <= 0) else "non-monotone in H0")
    return {
        "n_differing": int(idx.size),
        "first_H0": float(H0[idx[0]]),
        "last_H0": float(H0[idx[-1]]),
        "contiguous_in_H0": contiguous,
        "delta_median": med,
        "delta_range_over_differing_cells": rng,
        "constant_offset_like": offset_like,
        "monotonicity": mono,
        "H0_of_differing_cells": [float(v) for v in H0[idx][:40]],
        "H0_list_truncated": bool(idx.size > 40),
        "description": (f"{idx.size}/{H0.size} cells differ, "
                        f"H0 in [{H0[idx[0]]:.4g}, {H0[idx[-1]]:.4g}], "
                        f"{'contiguous' if contiguous else 'scattered'}; {sign_txt}"
                        + (f"; {mono}" if mono else "")),
    }


def compare_pair(ref, other, arm):
    """Per-cell + posterior-level comparison of `other` against reference `ref`."""
    assert np.array_equal(ref["H0"], other["H0"]), "grids differ — not comparable"
    H0 = ref["H0"]
    a, b = ref["logL"], other["logL"]
    same_bits = bit_equal(a, b)
    with np.errstate(invalid="ignore"):
        delta = b - a
    delta = np.where(same_bits, 0.0, delta)   # -inf minus -inf is NaN, not a difference
    ad = np.abs(delta)
    finite_ad = ad[np.isfinite(ad)]

    n_cells = int(H0.size)
    n_bitwise = int(same_bits.sum())
    all_finite = bool(np.isfinite(ad).all())
    max_abs = float(finite_ad.max()) if finite_ad.size else float("nan")
    med_abs = float(np.median(finite_ad)) if finite_ad.size else float("nan")

    if n_bitwise == n_cells:
        verdict = "bitwise"
    elif all_finite and np.isfinite(max_abs) and max_abs < FLOAT_LEVEL_TOL:
        verdict = "float-level"
    else:
        verdict = "structurally different"

    p_ref, p_oth = posterior(H0, a), posterior(H0, b)
    scale = float(np.nanmax(np.abs(a[np.isfinite(a)]))) if np.isfinite(a).any() else 1.0
    ulp = float(np.spacing(scale))

    return {
        "arm": arm,
        "reference_tag": ref["tag"],
        "comparison_tag": other["tag"],
        "comparison_base_coord": other["base_coord"],
        "verdict": verdict,
        "cells": {
            "n_cells": n_cells,
            "n_bitwise_equal": n_bitwise,
            "n_differing": n_cells - n_bitwise,
            "fraction_bitwise_equal": n_bitwise / n_cells,
            "all_differences_finite": all_finite,
            "max_abs_delta_logL": max_abs,
            "median_abs_delta_logL": med_abs,
            "logL_scale": scale,
            "one_ulp_of_logL_scale": ulp,
            "max_abs_delta_in_ulps": (max_abs / ulp) if (ulp > 0 and np.isfinite(max_abs))
                                     else None,
            "n_neginf_reference": int(np.sum(~np.isfinite(a))),
            "n_neginf_comparison": int(np.sum(~np.isfinite(b))),
        },
        "shape": describe_shape(H0, delta, ~same_bits),
        "posterior": {
            "reference": p_ref,
            "comparison": p_oth,
            "delta_median_kmsMpc": p_oth["median"] - p_ref["median"],
            "delta_map_kmsMpc": p_oth["map"] - p_ref["map"],
            "delta_ci68_lo_kmsMpc": p_oth["ci68"][0] - p_ref["ci68"][0],
            "delta_ci68_hi_kmsMpc": p_oth["ci68"][1] - p_ref["ci68"][1],
        },
        "guard": {
            "reference_n_rejected": ref["n_rejected"],
            "comparison_n_rejected": other["n_rejected"],
            "reference_Neff_min": ref.get("Neff_min"),
            "comparison_Neff_min": other.get("Neff_min"),
        },
        "_delta": delta,
        "_H0": H0,
    }


# --------------------------------------------------------------------------- #
# figure
# --------------------------------------------------------------------------- #
def make_figure(primary, secondary, deep):
    import matplotlib as mpl
    mpl.use("Agg")
    import matplotlib.pyplot as plt

    # dataviz reference palette (light surface) — the same instance analysis_1's
    # figures use, so the two directories read as one system.
    SURFACE, INK, INK_2, INK_MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8a85"
    GRID = "#e6e5e1"
    BLUE, ORANGE = "#2a78d6", "#eb6834"          # slot 1 = GAL, slot 2 = AGN
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "font.size": 9, "axes.labelsize": 9.5,
        "axes.titlesize": 10.5, "axes.edgecolor": INK_MUTED, "axes.labelcolor": INK,
        "text.color": INK, "xtick.color": INK_2, "ytick.color": INK_2,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "axes.linewidth": 0.8,
        "legend.frameon": False, "pdf.fonttype": 42,
    })
    COL = {"GAL": BLUE, "AGN": ORANGE}
    LS = {"gal_all": "-", "agn_all": "-",
          "gal_matched": (0, (5, 2)), "agn_matched": (0, (5, 2))}

    fig = plt.figure(figsize=(8.6, 4.9))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.25, 1.0], wspace=0.24,
                          left=0.085, right=0.985, bottom=0.275, top=0.845)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])

    # ---- panel A: per-cell |Delta logL| across the H0 grid, primary arm ------
    nonzero = np.concatenate([np.abs(c["_delta"])[np.abs(c["_delta"]) > 0]
                              for c in primary.values()] or [np.array([])])
    nonzero = nonzero[np.isfinite(nonzero)]
    ulp_max = max((c["cells"]["one_ulp_of_logL_scale"] for c in primary.values()),
                  default=1e-12)
    # symlog: the linear band below `linthresh` is where the exactly-zero cells
    # live, so it has to be wide enough to see.
    linthresh = (max(float(nonzero.min()) * 0.5, 1e-18) if nonzero.size
                 else max(ulp_max * 1e-2, 1e-18))

    # Descending line widths: where two configurations coincide exactly (the
    # bitwise case puts all four on the same y = 0 line) each later series is
    # drawn thinner on top, so a coincidence reads as one line rather than
    # hiding three of them.
    for i, (key, tracer, evlabel, _) in enumerate(CONFIGS):
        c = primary.get(key)
        if c is None:
            continue
        d = np.abs(c["_delta"])
        axA.plot(c["_H0"], d, color=COL[tracer], linestyle=LS[key],
                 lw=(1.8, 1.4, 1.1, 0.9)[i], solid_capstyle="round", zorder=4 + i,
                 label=f"{tracer}, {evlabel}")

    axA.set_yscale("symlog", linthresh=linthresh, linscale=0.9)
    # Everything below one ulp of the log-likelihood is float64 rounding, not a
    # difference between the two estimators; shade it so the hash inside reads
    # as the floor it is.
    axA.axhspan(0.0, ulp_max, color=GRID, alpha=0.85, zorder=1)
    axA.axhline(ulp_max, color=INK_MUTED, lw=0.9, ls=":", zorder=3)
    axA.annotate("one float64 ulp at this log-likelihood",
                 xy=(0.30, ulp_max), xycoords=("axes fraction", "data"),
                 xytext=(0, 3), textcoords="offset points",
                 ha="left", va="bottom", fontsize=7.6, color=INK_2, zorder=9,
                 bbox=dict(facecolor=SURFACE, edgecolor="none", alpha=0.85,
                           boxstyle="square,pad=0.15"))
    # Explicit top: with every difference exactly zero matplotlib would default
    # to 1 and print seventeen empty decades.
    top = (10.0 ** (np.ceil(np.log10(float(nonzero.max()))) + 0.5) if nonzero.size
           else ulp_max * 1e3)
    top = max(top, linthresh * 1e4)
    axA.set_ylim(0.0, top)
    axA.yaxis.set_major_locator(
        mpl.ticker.SymmetricalLogLocator(linthresh=linthresh, base=10.0,
                                         subs=(1.0,)))
    axA.yaxis.set_minor_locator(mpl.ticker.NullLocator())
    first = next(iter(primary.values()))
    axA.set_xlim(float(first["_H0"][0]), float(first["_H0"][-1]))
    axA.set_xlabel(r"$H_0$  [km s$^{-1}$ Mpc$^{-1}$]")
    axA.set_ylabel(r"$|\Delta \ln \mathcal{L}|$  [nats]")
    axA.set_title("General model in its complete-catalog limit,\n"
                  "cell by cell against the complete-catalog model",
                  loc="left", pad=8)
    axA.grid(True, axis="y", color=GRID, lw=0.7, zorder=0)
    axA.set_axisbelow(True)
    for side in ("top", "right"):
        axA.spines[side].set_visible(False)
    # Legend below the panel: the curves can sit anywhere on a symlog axis, so
    # no in-panel corner is reliably free.
    axA.legend(loc="upper left", bbox_to_anchor=(0.0, -0.20), ncol=2,
               fontsize=8.2, handlelength=2.4, labelcolor=INK_2,
               columnspacing=1.8, borderaxespad=0.0)

    notes = []
    if primary and {c["verdict"] for c in primary.values()} == {"bitwise"}:
        notes.append("every cell identical to the bit in all four configurations")
    if deep and {c["verdict"] for c in deep.values()} == {"bitwise"}:
        notes.append(r"at $\log_{10} n_0 = -24$ every cell is identical to the bit")
    if notes:
        axA.annotate("\n".join(notes), xy=(0.5, 0.94), xycoords="axes fraction",
                     ha="center", va="top", fontsize=8.2, color=INK_2)

    # ---- panel B: secondary arm, posterior shift ----------------------------
    ypos, labels, vals, cols = [], [], [], []
    for i, (key, tracer, evlabel, _) in enumerate(CONFIGS):
        c = secondary.get(key)
        if c is None:
            continue
        v = c["posterior"]["delta_median_kmsMpc"]
        if not np.isfinite(v):
            continue
        ypos.append(-i); labels.append(f"{tracer}, {evlabel}")
        vals.append(float(v)); cols.append(COL[tracer])
    if vals:
        span = max(abs(min(vals)), abs(max(vals)), 1e-6)
        # The shifts can span two orders of magnitude (one configuration moves by
        # tens, the rest by ~1); a linear axis would flatten the small ones to
        # nothing, so the axis is symmetric-log about zero with a 1 km/s/Mpc
        # linear core.
        if span > 10.0:
            axB.set_xscale("symlog", linthresh=1.0, linscale=1.0)
            axB.set_xlim(-3.0 * span, 3.0 * span)
        else:
            axB.set_xlim(-1.75 * span, 1.75 * span)
        axB.set_ylim(min(ypos) - 0.75, max(ypos) + 0.85)
        axB.axvline(0.0, color=INK_MUTED, lw=0.9, zorder=2)
        for y, v, col, lab in zip(ypos, vals, cols, labels):
            axB.plot([0.0, v], [y, y], color=col, lw=2.0,
                     solid_capstyle="round", zorder=4)
            axB.plot([v], [y], marker="o", ms=6.0, mfc=col, mec=SURFACE, mew=1.2,
                     ls="none", zorder=5)
            # direct label above the mark: identity is never colour-alone, and the
            # panel keeps its full width for the data.
            axB.annotate(f"{lab}   {v:+.3f}", xy=(0.02, y + 0.28),
                         xycoords=("axes fraction", "data"),
                         ha="left", va="bottom", fontsize=8.0, color=INK_2)
    axB.set_yticks([])
    axB.set_xlabel(r"shift in median $H_0$  [km s$^{-1}$ Mpc$^{-1}$]")
    axB.set_title("Told the true host density\n(secondary arm)", loc="left", pad=8)
    axB.grid(True, axis="x", color=GRID, lw=0.7, zorder=0)
    axB.set_axisbelow(True)
    for side in ("top", "right", "left"):
        axB.spines[side].set_visible(False)
    axB.tick_params(axis="y", length=0)

    FIGS.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"fig_equivalence.{ext}", dpi=220)
    plt.close(fig)
    print(f"Wrote {FIGS / 'fig_equivalence.pdf'} and .png")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    grids, missing = {}, []
    wanted = []
    for key, *_ in CONFIGS:
        wanted += [f"dsc_{key}"] + [f"{p}_{key}" for _, p in ARMS]
        wanted += [f"f32_dsc_{key}"]
    # the float32 blocker is evidenced by full AGN scans and one GAL grid point
    # (the mechanism is dtype, not tracer, so four more 201-cell -inf grids would
    # buy nothing)
    wanted += ["f32_ds_agn_all", "f32_ds_agn_matched", "f32_ds1pt_gal_all"]
    for tag in wanted:
        g = load_grid(tag)
        if g is None:
            missing.append(tag)
        else:
            grids[tag] = g

    results = {name: {} for name, _ in ARMS}
    precision, provenance = {}, {}
    for key, tracer, evlabel, n0_true in CONFIGS:
        ref = grids.get(f"dsc_{key}")
        if ref is None:
            continue
        for name, prefix in ARMS:
            other = grids.get(f"{prefix}_{key}")
            if other is not None:
                results[name][key] = compare_pair(ref, other, name)
        f32ref = grids.get(f"f32_dsc_{key}")
        if f32ref is not None:
            precision[key] = compare_pair(ref, f32ref, "precision_control")
        provenance[key] = {
            "tracer": tracer, "events": evlabel, "true_log10n0": n0_true,
            "survey_path": ref["survey_path"], "gw_path": ref["gw_path"],
            "gwselection_path": ref["gwselection_path"],
            "kde_window": ref["kde_window"], "sel_batch_size": ref["sel_batch_size"],
            "pe_event_block": ref["pe_event_block"],
            "catalog_sky_weighting": ref["catalog_sky_weighting"],
            "reference_labels": ref["labels"],
            "reference_base_coord": ref["base_coord"],
            "general_labels": grids.get(f"ds_{key}", {}).get("labels"),
            "logL_stored_dtype": ref["logL_stored_dtype"],
            "jax_devices": ref["jax_devices"],
            "darksirens_git_sha": ref["darksirens_git_sha"],
            "seconds_per_eval": {
                t: grids[t]["steady_state_median_seconds"]
                for t in (f"dsc_{key}", f"ds_{key}", f"dsdeep_{key}", f"dstrue_{key}")
                if t in grids},
        }

    # ---- the float32 blocker -------------------------------------------------
    blocker = {}
    for tag, g in grids.items():
        if not tag.startswith("f32_ds") or tag.startswith("f32_dsc"):
            continue
        ll = g["logL"]
        blocker[tag] = {
            "universe_model": g["universe_model"],
            "survey_path": g["survey_path"],
            "base_coord": g["base_coord"],
            "n_cells": int(ll.size),
            "n_neginf_cells": int(np.sum(~np.isfinite(ll))),
            "all_cells_neginf": bool(np.all(~np.isfinite(ll))),
            "Neff_min": g.get("Neff_min"),
            "n_rejected": g["n_rejected"],
        }

    if results["primary"]:
        make_figure(results["primary"], results["secondary"], results["deep"])

    def strip(d):
        return {k: v for k, v in d.items() if not k.startswith("_")}

    def verdicts(arm):
        return {k: v["verdict"] for k, v in results[arm].items()}

    summary = {
        "question": ("does darksirens' general incomplete-catalog model `dark_sirens`, "
                     "in its complete-catalog limit (log10n0 -> -inf, delta = 0, "
                     "sigma_kde = 0, use_lss off), reproduce `dark_sirens_complete` "
                     "when inferring H0 on the same data — and where bitwise?"),
        "equality_criterion": ("IEEE-754 float64 bit equality per grid cell; "
                               "float-level = max |Delta logL| < 1e-9 nats"),
        "arms": {
            "reference": "dark_sirens_complete, K=1, catalog_sky_weighting=field",
            "primary": "dark_sirens at log10n0 = -12",
            "deep": "dark_sirens at log10n0 = -24",
            "secondary": ("dark_sirens at the tracer's TRUE log10n0 (GAL -3, AGN -5)"
                          " — characterization, not pass/fail"),
            "precision_control": ("the SAME reference model on the float32 survey files"
                                  " vs the float64 copies — the dtype's own effect"),
        },
        "common_configuration": {
            "grid": "H0 in [50, 100], 201 points",
            "population": "powerlaw+peak, fixed at the mock fiducial",
            "Om0": 0.3075,
            "guard": "selection_neff_guard=hard, max_likelihood_variance=1e6",
            "injections": "injections_targeted.h5 — the same file for every scan",
            "truth_H0": TRUTH,
            "surveys": ("float64 copies in data_derived/ (a pure precision widening of "
                        "the shipped float32 files); required because dark_sirens is "
                        "-inf everywhere on the float32 files — see blocker below"),
        },
        "missing_result_files": missing,
        "verdict_by_config": verdicts("primary"),
        "verdict_by_config_deep": verdicts("deep"),
        "overall_verdict_primary": ("bitwise" if results["primary"] and
                                    set(verdicts("primary").values()) == {"bitwise"}
                                    else "mixed / see per-config"),
        "overall_verdict_deep": ("bitwise" if results["deep"] and
                                 set(verdicts("deep").values()) == {"bitwise"}
                                 else "mixed / see per-config"),
        "blocker_float32_surveys": {
            "mechanism": (
                "darksirens 2b86a2d darksirens/redshift/completion.py::_kde_dndz_obs "
                "computes the truncated-kernel mass "
                "ndtr((ZMAX - z_i)/sigma) - ndtr(-z_i/sigma), floored with "
                "jnp.maximum(mass, 1e-300), in the CATALOG'S storage dtype, while the "
                "kernel itself is promoted to the package zgrid's float64. The survey "
                "files store galaxies in float32 and pad short rows at z = 100, so for "
                "every padded slot the float32 mass underflows to exactly 0 (1e-300 is "
                "not representable in float32), the float64 kernel is 0 there, and "
                "0/0 = NaN — which the `* real_gal` mask cannot remove (0 * NaN = NaN). "
                "Every catalog row with any padding returns all-NaN; only the single "
                "row at the maximum galaxy count survives. The NaN propagates into the "
                "survey-global field normalizer log_Z_global, so every injection and PE "
                "weight is NaN, N_eff is 0, and the selection guard rejects every cell. "
                "dark_sirens_complete never reads that KDE, which is why it is unaffected."),
            "runs": blocker,
        },
        "primary": {k: strip(v) for k, v in results["primary"].items()},
        "deep": {k: strip(v) for k, v in results["deep"].items()},
        "secondary": {k: strip(v) for k, v in results["secondary"].items()},
        "precision_control": {k: strip(v) for k, v in precision.items()},
        "provenance": provenance,
    }
    RES.mkdir(parents=True, exist_ok=True)
    out = RES / "equivalence_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {out}")

    # ---- console tables ------------------------------------------------------
    hdr = (f"{'config':<14}{'verdict':<24}{'bitwise cells':>16}"
           f"{'max|dlogL|':>14}{'d median H0':>14}")

    def table(title, block):
        if not block:
            return
        print(f"\n=== {title} ===")
        print(hdr); print("-" * len(hdr))
        for key, *_ in CONFIGS:
            c = block.get(key)
            if c is None:
                continue
            print(f"{key:<14}{c['verdict']:<24}"
                  f"{c['cells']['n_bitwise_equal']:>8}/{c['cells']['n_cells']:<7}"
                  f"{c['cells']['max_abs_delta_logL']:>14.3e}"
                  f"{c['posterior']['delta_median_kmsMpc']:>14.3e}")

    table("PRIMARY: dark_sirens(log10n0=-12) vs dark_sirens_complete", results["primary"])
    table("DEEP LIMIT: dark_sirens(log10n0=-24) vs dark_sirens_complete", results["deep"])
    table("SECONDARY: dark_sirens at the true host density (characterization)",
          results["secondary"])
    table("PRECISION CONTROL: reference model, float32 survey vs float64 copy", precision)

    if blocker:
        print("\n=== BLOCKER: general model on the survey files as shipped (float32) ===")
        for tag, b in sorted(blocker.items()):
            print(f"  {tag:<24} {b['n_neginf_cells']}/{b['n_cells']} cells -inf, "
                  f"N_eff min = {b['Neff_min']}")

    print("\nposterior medians (full precision):")
    for key, *_ in CONFIGS:
        c = results["primary"].get(key)
        if c is None:
            continue
        line = (f"  {key:<14} complete {c['posterior']['reference']['median']!r}"
                f"   general(-12) {c['posterior']['comparison']['median']!r}")
        if key in results["deep"]:
            line += f"   general(-24) {results['deep'][key]['posterior']['comparison']['median']!r}"
        if key in results["secondary"]:
            line += f"   true-density {results['secondary'][key]['posterior']['comparison']['median']!r}"
        print(line)


if __name__ == "__main__":
    main()
