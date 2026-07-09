#!/usr/bin/env python3
"""compare_results.py — comparison deliverables for the gw_agn <-> darksirens
reproduction campaign (see ../RECON.md, ../GATES.md, ../RESULTS.md).

Produces four figures (../figs/) and one machine-readable summary
(../results/comparison_summary.json) contrasting:

  1. fig_fscans.png        — this work's darksirens K=2 f_cat,2 scans vs
                              gw_agn's field-convention alpha_AGN conditional
                              profiles, on the SAME four truth sets. The two
                              panels have structurally different shapes by
                              design: GATES.md's GB incident 2 finding is that
                              these are two different estimands (isotropic-sky
                              conditional-prior mixture vs number-density /
                              sky-clustering field convention), not a
                              translation bug.
  2. fig_h0_pertracer.png  — per-tracer H0 coverage (10 realizations x 2
                              universe models x 2 tracers).
  3. fig_h0_summary.png    — per-realization H0 scatter, this work vs gw_agn.
  4. fig_joint.png         — joint (H0, f) posterior surfaces, this work vs
                              gw_agn, for fagn0.3 (and fagn0.7 if present).

Pure post-processing: plain h5py/numpy/matplotlib, no darksirens import, CPU
only. JAX_PLATFORMS is pinned to cpu defensively in case anything downstream
ever imports jax.

Every file read is defensive: missing/unreadable inputs are warned about and
skipped rather than raising, EXCEPT that a complete run requires all 40
per-tracer coverage files (h0_{ds,dsc}_{gal,agn}_r00..r09.h5) to be present —
the caller is expected to have waited for those before invoking this script.
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
SCRIPT_DIR = Path(__file__).resolve().parent
BASE = SCRIPT_DIR.parent  # .../gw_agn_darksirens
RESULTS = BASE / "results"
DATA = BASE / "data"
FIGS = BASE / "figs"
FIGS.mkdir(parents=True, exist_ok=True)
GW_AGN_REC = Path("/hildafs/projects/phy230014p/magana/gws-agn/working/gw_agn/results/recovery")

TRUTH_H0 = 67.74

# fscan truth tags -> planted alpha_AGN truth (RECON.md "Truth / fiducials")
FSCAN_TAGS = ["0.0", "0.3", "0.7", "1.0"]
FSCAN_TRUTH = {"0.0": 0.00989, "0.3": 0.307, "0.7": 0.703, "1.0": 1.0}

# --------------------------------------------------------------------------- #
# Palette — dataviz skill references/palette.md, fixed categorical order.
# Validated instance: light-mode worst adjacent CVD dE 24.2; slots used here
# are assigned in the fixed slot order (never cycled/reassigned per-filter).
# --------------------------------------------------------------------------- #
CAT = ["#2a78d6", "#1baf7a", "#eda100", "#008300",
       "#4a3aa7", "#e34948", "#e87ba4", "#eb6834"]
BLUE, AQUA, YELLOW, GREEN, VIOLET, RED, MAGENTA, ORANGE = CAT

INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDCOL = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"

# f-scan truth-set color (categorical slots 1-4, fixed order)
TRUTH_COLOR = {"0.0": BLUE, "0.3": AQUA, "0.7": YELLOW, "1.0": GREEN}

# (tracer, universe_model) -> categorical slot, reused identically across
# fig_h0_pertracer (panel accent) and fig_h0_summary (series color) so a
# reader learns the mapping once.
COMBO_COLOR = {
    ("gal", "ds"): BLUE,
    ("gal", "dsc"): AQUA,
    ("agn", "ds"): YELLOW,
    ("agn", "dsc"): GREEN,
}
COMBO_LABEL = {
    ("gal", "ds"): "GAL, dark_sirens",
    ("gal", "dsc"): "GAL, dark_sirens_complete",
    ("agn", "ds"): "AGN, dark_sirens",
    ("agn", "dsc"): "AGN, dark_sirens_complete",
}

THIS_WORK_COLOR = BLUE
GW_AGN_COLOR = RED

CLIP = -900.0  # shared display floor for DeltalogL panels (figs 1 & 2)

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.edgecolor": BASELINE,
    "axes.linewidth": 0.9,
    "axes.labelcolor": INK,
    "axes.titlecolor": INK,
    "text.color": INK,
    "xtick.color": INK_SECONDARY,
    "ytick.color": INK_SECONDARY,
    "grid.color": GRIDCOL,
    "grid.linewidth": 0.8,
    "axes.grid": True,
    "axes.axisbelow": True,
    "legend.frameon": False,
    "legend.fontsize": 8.5,
})

WARNINGS: list[str] = []


def warn(msg: str) -> None:
    print(f"[warn] {msg}", file=sys.stderr)
    WARNINGS.append(msg)


# --------------------------------------------------------------------------- #
# Defensive IO
# --------------------------------------------------------------------------- #
def read_h5(path, keys):
    """Read a set of top-level datasets from an HDF5 file. Returns None (and
    warns) on any missing file/dataset/open failure."""
    path = Path(path)
    if not path.exists():
        warn(f"missing file: {path}")
        return None
    try:
        with h5py.File(path, "r") as h:
            out = {}
            for k in keys:
                if k not in h:
                    warn(f"{path.name}: dataset {k!r} not found")
                    return None
                out[k] = h[k][()]
            return out
    except OSError as e:
        warn(f"{path.name}: failed to open ({e})")
        return None


def read_h5_attr(path, attr, fallback=None):
    path = Path(path)
    if not path.exists():
        return fallback
    try:
        with h5py.File(path, "r") as h:
            return h.attrs.get(attr, fallback)
    except OSError:
        return fallback


def read_json(path):
    path = Path(path)
    if not path.exists():
        warn(f"missing json: {path}")
        return None
    try:
        return json.loads(path.read_text())
    except Exception as e:  # noqa: BLE001 - defensive by design
        warn(f"{path.name}: failed to parse json ({e})")
        return None


# --------------------------------------------------------------------------- #
# Numerics
# --------------------------------------------------------------------------- #
def quadratic_refine_argmax(x, y):
    """Refine the grid argmax of y(x) via a 3-point parabola through the peak
    node and its immediate neighbors. Falls back to the raw grid argmax at a
    boundary peak or on a degenerate/non-concave fit."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(y)
    if not finite.any():
        return {"argmax_refined": float("nan"), "argmax_grid": float("nan"), "refined": False}
    yf = np.where(finite, y, -np.inf)
    i = int(np.argmax(yf))
    if i == 0 or i == len(x) - 1:
        return {"argmax_refined": float(x[i]), "argmax_grid": float(x[i]), "refined": False}
    x0, x1, x2 = float(x[i - 1]), float(x[i]), float(x[i + 1])
    y0, y1, y2 = yf[i - 1], yf[i], yf[i + 1]
    if not (np.isfinite(y0) and np.isfinite(y1) and np.isfinite(y2)):
        return {"argmax_refined": float(x[i]), "argmax_grid": float(x[i]), "refined": False}
    try:
        a, b, _c = np.polyfit([x0, x1, x2], [y0, y1, y2], 2)
    except Exception:  # noqa: BLE001
        return {"argmax_refined": float(x[i]), "argmax_grid": float(x[i]), "refined": False}
    if a >= 0:  # not concave -> degenerate, keep grid node
        return {"argmax_refined": float(x[i]), "argmax_grid": float(x[i]), "refined": False}
    xv = -b / (2 * a)
    if not (min(x0, x2) <= xv <= max(x0, x2)):
        return {"argmax_refined": float(x[i]), "argmax_grid": float(x[i]), "refined": False}
    return {"argmax_refined": float(xv), "argmax_grid": float(x[i]), "refined": True}


def mask_sentinel(arr, thresh=1e6):
    """Mask darksirens/gw_agn -inf & huge-negative-sentinel entries (RECON.md:
    field mixture exactly zero at some (H0,f) or (H0,alpha) cells) as NaN,
    relative to the array's own finite max."""
    arr = np.asarray(arr, dtype=float)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.full_like(arr, np.nan)
    m = arr[finite].max()
    return np.where(finite & (arr >= m - thresh), arr, np.nan)


def ci_halfwidth(block):
    if not block or "ci68" not in block or block["ci68"] is None:
        return float("nan")
    lo, hi = block["ci68"]
    return 0.5 * (hi - lo)


def hpd_dlogl_threshold(dlogl_grid, mass=0.6827):
    """Grid-cell HPD threshold (uniform-grid approximation: trapezoid edge
    correction is <~2% for these 61x41/41-pt grids and is ignored here). The
    ΔlogL value (<=0, relative to the grid's own max) such that grid cells at
    or above it contain `mass` of the total flat-prior probability."""
    finite = np.isfinite(dlogl_grid)
    if not finite.any():
        return -50.0
    with np.errstate(over="ignore", under="ignore"):
        p = np.where(finite, np.exp(np.clip(dlogl_grid, -700.0, 0.0)), 0.0)
    flat = p.ravel()
    order = np.argsort(flat)[::-1]
    csum = np.cumsum(flat[order])
    total = csum[-1]
    if total <= 0:
        return -50.0
    csum = csum / total
    k = int(np.searchsorted(csum, mass))
    k = min(k, len(order) - 1)
    p_thresh = flat[order[k]]
    if p_thresh <= 0:
        return -50.0
    lvl = float(np.log(p_thresh))
    return lvl if lvl < -1e-9 else -1e-6  # guard against a degenerate 0-width band


def joint_moments(H0v, fv, ll_grid):
    """Gaussian-approx rho/moments of a 2-D flat-prior posterior, replicating
    scan_darksirens.py's joint-scan computation (trapz marginals + 2nd
    moments) so it can also be applied to gw_agn's rec_fagn*.h5 grids, which
    do not carry a precomputed rho."""
    ll_grid = np.asarray(ll_grid, dtype=float)
    finite = np.isfinite(ll_grid)
    if not finite.any():
        return None
    lmax = float(ll_grid[finite].max())
    with np.errstate(over="ignore", under="ignore"):
        p2d = np.where(finite, np.exp(np.clip(ll_grid - lmax, -700, 0)), 0.0)
    norm = np.trapz(np.trapz(p2d, fv, axis=1), H0v, axis=0)
    if not (np.isfinite(norm) and norm > 0):
        return None
    Zn = p2d / norm
    pH0 = np.trapz(Zn, fv, axis=1)
    pf = np.trapz(Zn, H0v, axis=0)
    EH0 = np.trapz(H0v * pH0, H0v)
    Ef = np.trapz(fv * pf, fv)
    VH0 = np.trapz((H0v - EH0) ** 2 * pH0, H0v)
    Vf = np.trapz((fv - Ef) ** 2 * pf, fv)
    H0g, fg = np.meshgrid(H0v, fv, indexing="ij")
    Cov = np.trapz(np.trapz((H0g - EH0) * (fg - Ef) * Zn, fv, axis=1), H0v, axis=0)
    rho = float(Cov / np.sqrt(VH0 * Vf)) if VH0 > 0 and Vf > 0 else float("nan")
    return {"E_H0": float(EH0), "E_f": float(Ef), "sigma_H0": float(np.sqrt(VH0)),
            "sigma_f": float(np.sqrt(Vf)), "cov": float(Cov), "rho": rho}


def per_event_slope(f_grid, ll, nobs):
    if nobs in (None, 0) or not np.isfinite(nobs):
        return float("nan")
    return float((ll[-1] - ll[0]) / nobs)


def nice(v, nd=3):
    return "nan" if v is None or (isinstance(v, float) and not np.isfinite(v)) else round(float(v), nd)


def sanitize_nans(obj):
    """Recursively replace NaN/Inf floats (and numpy scalars) with JSON null so
    comparison_summary.json is strict-JSON (portable to non-Python readers)."""
    if isinstance(obj, dict):
        return {k: sanitize_nans(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_nans(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return None if not np.isfinite(v) else v
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


# --------------------------------------------------------------------------- #
# Deliverable 1: fig_fscans.png
# --------------------------------------------------------------------------- #
def build_fig_fscans(summary):
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.0, 5.3), dpi=150)

    summary["fscan"] = {}
    summary["fscan_gwagn"] = {}

    shared_handles = {}

    # ---- left: this work (darksirens K=2, isotropic-sky conditional mixture)
    for tag in FSCAN_TAGS:
        d = read_h5(RESULTS / f"fscan_fagn{tag}.h5", ["f_grid", "log_likelihood"])
        if d is None:
            continue
        f, ll = d["f_grid"], d["log_likelihood"]
        dll_disp = np.clip(ll - np.nanmax(ll), CLIP, 0)
        color = TRUTH_COLOR[tag]
        label = f"truth $f$={FSCAN_TRUTH[tag]:.3f}"
        (line,) = axL.plot(f, dll_disp, color=color, lw=2.0, solid_capstyle="round",
                            zorder=3, label=label)
        shared_handles.setdefault(label, line)
        axL.axvline(FSCAN_TRUTH[tag], color=color, lw=1.0, ls=(0, (1, 1.6)), alpha=0.65, zorder=1)
        nobs = read_h5_attr(DATA / f"gw_fagn{tag}.h5", "nobs", 1000)
        summary["fscan"][tag] = {
            "truth_f": FSCAN_TRUTH[tag],
            "argmax_grid": float(f[int(np.nanargmax(ll))]),
            "range": float(np.nanmax(ll) - np.nanmin(ll)),
            "per_event_slope": per_event_slope(f, ll, nobs),
            "nobs": int(nobs) if nobs is not None else None,
        }

    # A/B levers on fagn0.3: injB (alt injection seed), zlt1 (truncated catalogs)
    ab_variants = [("injB", (0, (4, 1.6)), "0.307, injB A/B"),
                   ("zlt1", (0, (1, 1.2, 1, 1.2)), "0.307, z<1 A/B")]
    for variant, ls, lbl in ab_variants:
        d = read_h5(RESULTS / f"fscan_fagn0.3_{variant}.h5", ["f_grid", "log_likelihood"])
        if d is None:
            continue
        f, ll = d["f_grid"], d["log_likelihood"]
        dll_disp = np.clip(ll - np.nanmax(ll), CLIP, 0)
        (line,) = axL.plot(f, dll_disp, color=AQUA, lw=1.2, ls=ls, alpha=0.9, zorder=2, label=lbl)
        shared_handles.setdefault(lbl, line)
        nobs = read_h5_attr(DATA / "gw_fagn0.3.h5", "nobs", 1000)
        summary["fscan"][f"0.3_{variant}"] = {
            "truth_f": FSCAN_TRUTH["0.3"],
            "argmax_grid": float(f[int(np.nanargmax(ll))]),
            "range": float(np.nanmax(ll) - np.nanmin(ll)),
            "per_event_slope": per_event_slope(f, ll, nobs),
            "nobs": int(nobs) if nobs is not None else None,
        }

    axL.set_xlim(0, 1)
    axL.set_ylim(CLIP - 30, 30)
    axL.set_xlabel(r"$f_{cat,2}$  (AGN-catalog mixture weight)")
    axL.set_ylabel(r"$\Delta\log\mathcal{L}$  (clipped at %d)" % CLIP)
    axL.set_title("This work — darksirens K=2 mixture\n"
                   "isotropic-sky, per-pixel-conditional completion prior", fontsize=10.5)

    # ---- right: gw_agn field convention, conditional profile at H0 node nearest truth
    node_H0_used = None
    sentinel_tags = []  # tags with a masked (zero-probability) cell, for one shared annotation
    for tag in FSCAN_TAGS:
        d = read_h5(GW_AGN_REC / f"rec_fagn{tag}.h5",
                     ["H0_grid", "alpha_agn_grid", "log_likelihood_grid"])
        if d is None:
            continue
        H0v, av, grid = d["H0_grid"], d["alpha_agn_grid"], d["log_likelihood_grid"]
        node_idx = int(np.argmin(np.abs(H0v - TRUTH_H0)))
        node_H0_used = float(H0v[node_idx])
        row = grid[node_idx, :]
        row_m = mask_sentinel(row)
        dll = row_m - np.nanmax(row_m)
        dll_disp = np.clip(dll, CLIP, 0)
        color = TRUTH_COLOR[tag]
        axR.plot(av, dll_disp, color=color, lw=2.0, solid_capstyle="round", zorder=3)
        axR.axvline(FSCAN_TRUTH[tag], color=color, lw=1.0, ls=(0, (1, 1.6)), alpha=0.65, zorder=1)

        bad_idx = np.where(np.isnan(row_m))[0]
        if len(bad_idx):
            sentinel_tags.append((tag, float(av[bad_idx[0]]), color))

        finite_dll = np.where(np.isnan(dll), -np.inf, dll)
        summary["fscan_gwagn"][tag] = {
            "truth_alpha": FSCAN_TRUTH[tag],
            "H0_node": node_H0_used,
            "argmax_grid": float(av[int(np.nanargmax(finite_dll))]) if np.isfinite(finite_dll).any() else float("nan"),
            "range": float(np.nanmax(row_m) - np.nanmin(row_m)),
            "range_raw_incl_sentinel": float(np.nanmax(row) - np.nanmin(row)),
            "n_sentinel_masked": int(len(bad_idx)),
        }

    # One shared annotation for all masked (zero-probability) f=1 cells, with per-tag
    # colored leader lines fanning out from a single label (avoids overlapping text).
    if sentinel_tags:
        label_xy = (0.62, CLIP * 0.42)
        axR.annotate(
            "zero probability at f=1:\nsome GAL-hosted event's pixel is\nabsent from the sparse AGN catalog\n"
            + ", ".join(f"f_true={FSCAN_TRUTH[t]:.2g}" for t, _, _ in sentinel_tags),
            xy=label_xy, fontsize=7.3, color=INK_SECONDARY, ha="left", va="center",
            bbox=dict(boxstyle="round,pad=0.3", fc=SURFACE, ec=BASELINE, lw=0.7),
        )
        for tag, xb, color in sentinel_tags:
            axR.annotate("", xy=(xb, CLIP * 0.97), xytext=label_xy,
                          arrowprops=dict(arrowstyle="->", color=color, lw=1.0, alpha=0.85))

    axR.set_xlim(0, 1)
    axR.set_ylim(CLIP - 30, 30)
    axR.set_xlabel(r"$\alpha_{AGN}$  (host-assignment fraction)")
    axR.set_ylabel(r"$\Delta\log\mathcal{L}$  (clipped at %d)" % CLIP)
    node_str = f"{node_H0_used:.1f}" if node_H0_used is not None else "n/a"
    axR.set_title(f"gw_agn — field convention, $H_0$ node {node_str}\n"
                   "number-density / sky-clustering estimand", fontsize=10.5)

    for ax in (axL, axR):
        ax.grid(True, which="major", axis="both")

    fig.suptitle("f-scan estimand contrast: same events, two conventions for the AGN-hosted fraction",
                 fontsize=13, y=1.03)
    fig.legend(handles=list(shared_handles.values()), loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.10), frameon=False, fontsize=8.7)
    fig.tight_layout()
    outpath = FIGS / "fig_fscans.png"
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {outpath}")
    return outpath


# --------------------------------------------------------------------------- #
# Deliverable 2: fig_h0_pertracer.png
# --------------------------------------------------------------------------- #
def build_fig_h0_pertracer(summary):
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.0), dpi=150, sharex=True, sharey=True)
    tracers = ["gal", "agn"]
    models = ["ds", "dsc"]
    model_title = {"ds": "dark_sirens (missing-floor completion)",
                   "dsc": "dark_sirens_complete (empty-pixel-zero completion)"}

    summary["coverage"] = {m: {t: {} for t in tracers} for m in models}
    summary["coverage_aggregate"] = {m: {} for m in models}

    for ir, tracer in enumerate(tracers):
        for ic, model in enumerate(models):
            ax = axes[ir, ic]
            color = COMBO_COLOR[(tracer, model)]
            refined_list, median_list, hw_list = [], [], []
            n_found = 0
            for R in range(10):
                rtag = f"r{R:02d}"
                tag = f"h0_{model}_{tracer}_{rtag}"
                d = read_h5(RESULTS / f"{tag}.h5", ["H0_grid", "log_likelihood"])
                if d is None:
                    continue
                n_found += 1
                H0v, ll = d["H0_grid"], d["log_likelihood"]
                dll = np.clip(ll - np.nanmax(ll), CLIP, 0)
                ax.plot(H0v, dll, color=color, lw=0.9, alpha=0.55, zorder=2)
                rq = quadratic_refine_argmax(H0v, ll)
                j = read_json(RESULTS / f"{tag}.json")
                med = j["H0"]["median"] if j and "H0" in j else float("nan")
                hw = ci_halfwidth(j["H0"]) if j and "H0" in j else float("nan")
                if np.isfinite(rq["argmax_refined"]):
                    refined_list.append(rq["argmax_refined"])
                if np.isfinite(med):
                    median_list.append(med)
                if np.isfinite(hw):
                    hw_list.append(hw)
                summary["coverage"][model][tracer][rtag] = {
                    "argmax_refined": rq["argmax_refined"],
                    "argmax_grid": rq["argmax_grid"],
                    "median": med,
                    "ci68_halfwidth": hw,
                }

            ax.axvline(TRUTH_H0, color=INK, lw=1.1, ls=(0, (1, 1.6)), alpha=0.7, zorder=1)

            if refined_list:
                mean_arg = float(np.mean(refined_list))
                std_arg = float(np.std(refined_list))
                ax.text(0.03, 0.04,
                        f"mean refined argmax = {mean_arg:.2f} $\\pm$ {std_arg:.2f} km/s/Mpc  (n={n_found}/10)",
                        transform=ax.transAxes, fontsize=8.2, color=INK_SECONDARY, va="bottom")
            else:
                ax.text(0.5, 0.5, "no realizations found yet", transform=ax.transAxes,
                        ha="center", va="center", color=INK_MUTED, fontsize=9)

            if median_list:
                mean_med = float(np.mean(median_list))
                sem_med = float(np.std(median_list, ddof=1) / np.sqrt(len(median_list))) if len(median_list) > 1 else float("nan")
                mean_hw = float(np.mean(hw_list)) if hw_list else float("nan")
            else:
                mean_med = sem_med = mean_hw = float("nan")
            summary["coverage_aggregate"][model][tracer] = {
                "n_realizations": n_found,
                "mean_median": mean_med,
                "sem_median": sem_med,
                "mean_ci68_halfwidth": mean_hw,
                "mean_argmax_refined": float(np.mean(refined_list)) if refined_list else float("nan"),
                "std_argmax_refined": float(np.std(refined_list)) if refined_list else float("nan"),
            }

            ax.set_xlim(50, 100)
            ax.set_ylim(CLIP - 30, 30)
            if ir == 1:
                ax.set_xlabel("$H_0$  [km/s/Mpc]")
            if ic == 0:
                ax.set_ylabel(r"$\Delta\log\mathcal{L}$")
            ax.set_title(f"{tracer.upper()}  ×  {model_title[model]}", fontsize=10)

    fig.suptitle("Per-tracer $H_0$ coverage: 10 realizations, dark_sirens vs dark_sirens_complete "
                 f"(truth $H_0$={TRUTH_H0})", fontsize=12.5, y=1.0)
    fig.tight_layout()
    outpath = FIGS / "fig_h0_pertracer.png"
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {outpath}")
    return outpath


# --------------------------------------------------------------------------- #
# Deliverable 3: fig_h0_summary.png
# --------------------------------------------------------------------------- #
def build_fig_h0_summary(summary, ref):
    fig, ax = plt.subplots(figsize=(7.0, 6.6), dpi=150)
    combos = [("gal", "ds"), ("gal", "dsc"), ("agn", "ds"), ("agn", "dsc")]

    all_vals = []
    for tracer, model in combos:
        color = COMBO_COLOR[(tracer, model)]
        xs, ys = [], []
        for R in range(10):
            rtag = f"r{R:02d}"
            xref = None
            if ref is not None:
                xref = ref.get(tracer, {}).get("per_realization", {}).get(rtag, {}).get("median")
            cov = summary.get("coverage", {}).get(model, {}).get(tracer, {}).get(rtag)
            if xref is None or cov is None:
                continue
            y = cov.get("argmax_refined")
            if y is None or not np.isfinite(y) or not np.isfinite(xref):
                continue
            xs.append(xref)
            ys.append(y)
        if not xs:
            continue
        all_vals.extend(xs)
        all_vals.extend(ys)
        ax.scatter(xs, ys, s=46, color=color, edgecolor=SURFACE, linewidth=1.2,
                   label=COMBO_LABEL[(tracer, model)], zorder=3)

    if all_vals:
        lo = min(min(all_vals), TRUTH_H0) - 1.5
        hi = max(max(all_vals), TRUTH_H0) + 1.5
    else:
        lo, hi = 60.0, 76.0
    lims = (lo, hi)

    ax.plot(lims, lims, color=BASELINE, lw=1.4, ls=(0, (5, 2.5)), zorder=1, label="y = x")
    ax.axvline(TRUTH_H0, color=INK_MUTED, lw=1.0, alpha=0.55, zorder=0)
    ax.axhline(TRUTH_H0, color=INK_MUTED, lw=1.0, alpha=0.55, zorder=0)
    ax.scatter([TRUTH_H0], [TRUTH_H0], marker="*", s=190, color=INK, edgecolor=SURFACE,
               linewidth=1.2, zorder=4, label=f"truth ($H_0$={TRUTH_H0})")

    ax.set_xlim(*lims)
    ax.set_ylim(*lims)
    ax.set_aspect("equal")
    ax.set_xlabel("gw_agn field-convention $H_0$ median  [km/s/Mpc]  (r00–r09)")
    ax.set_ylabel("this work, refined-argmax $H_0$  [km/s/Mpc]  (r00–r09)")
    ax.set_title("Per-realization $H_0$ recovery: this work vs gw_agn")
    ax.legend(loc="upper left", fontsize=8.3)
    fig.tight_layout()
    outpath = FIGS / "fig_h0_summary.png"
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {outpath}")
    return outpath


# --------------------------------------------------------------------------- #
# Deliverable 4: fig_joint.png
# --------------------------------------------------------------------------- #
def build_fig_joint(summary):
    candidate_sets = [("0.3", 0.307), ("0.7", 0.703)]
    rows = []
    for tag, ftrue in candidate_sets:
        dw = read_h5(RESULTS / f"joint_fagn{tag}.h5", ["H0_grid", "f_grid", "log_likelihood"])
        jw = read_json(RESULTS / f"joint_fagn{tag}.json")
        dg = read_h5(GW_AGN_REC / f"rec_fagn{tag}.h5",
                     ["H0_grid", "alpha_agn_grid", "log_likelihood_grid"])
        jg = read_json(GW_AGN_REC / f"rec_fagn{tag}.json")
        if dw is None or dg is None:
            warn(f"fig_joint: skipping fagn{tag} row (missing grid data)")
            continue
        rows.append((tag, ftrue, dw, jw, dg, jg))

    summary["joint"] = {}
    if not rows:
        warn("fig_joint: no joint sets available at all, figure not written")
        return None

    fig, axes = plt.subplots(len(rows), 3, figsize=(15.3, 5.15 * len(rows)), dpi=150, squeeze=False)
    levels = [-9.0, -4.0, -1.0, 0.0]
    blue_shades = [to_rgba(BLUE, a) for a in (0.25, 0.55, 0.92)]
    red_shades = [to_rgba(RED, a) for a in (0.25, 0.55, 0.92)]

    for irow, (tag, ftrue, dw, jw, dg, jg) in enumerate(rows):
        H0w, fw, llw = dw["H0_grid"], dw["f_grid"], dw["log_likelihood"]
        H0g, fg_, llg = dg["H0_grid"], dg["alpha_agn_grid"], dg["log_likelihood_grid"]

        llg_masked = mask_sentinel(llg)
        dllw = llw - np.nanmax(llw)
        dllg = llg_masked - np.nanmax(llg_masked)

        mapw = (jw["map"]["H0"], jw["map"]["f"]) if jw and "map" in jw else None
        mapg = (jg["map"]["H0"], jg["map"]["alpha_agn"]) if jg and "map" in jg else None

        axa, axb, axc = axes[irow, 0], axes[irow, 1], axes[irow, 2]

        axa.contourf(H0w, fw, dllw.T, levels=levels, colors=blue_shades)
        axa.contour(H0w, fw, dllw.T, levels=levels[:-1], colors=INK_SECONDARY, linewidths=0.6)
        if mapw:
            axa.scatter(*mapw, marker="*", s=180, color=BLUE, edgecolor=SURFACE, linewidth=1.3, zorder=5)
        axa.scatter([TRUTH_H0], [ftrue], marker="X", s=95, color=INK, edgecolor=SURFACE, linewidth=1.1, zorder=5)
        map_str = f"MAP=({mapw[0]:.1f}, {mapw[1]:.3f})" if mapw else "MAP n/a"
        axa.set_title(f"This work (darksirens K=2), $f_{{agn}}$={tag}\n{map_str}", fontsize=10)

        axb.contourf(H0g, fg_, dllg.T, levels=levels, colors=red_shades)
        axb.contour(H0g, fg_, dllg.T, levels=levels[:-1], colors=INK_SECONDARY, linewidths=0.6)
        if mapg:
            axb.scatter(*mapg, marker="*", s=180, color=RED, edgecolor=SURFACE, linewidth=1.3, zorder=5)
        axb.scatter([TRUTH_H0], [ftrue], marker="X", s=95, color=INK, edgecolor=SURFACE, linewidth=1.1, zorder=5)
        map_str_g = f"MAP=({mapg[0]:.1f}, {mapg[1]:.3f})" if mapg else "MAP n/a"
        axb.set_title(f"gw_agn (field convention), $\\alpha_{{AGN}}$={tag}\n{map_str_g}", fontsize=10)

        lvl_w = hpd_dlogl_threshold(dllw, 0.6827)
        lvl_g = hpd_dlogl_threshold(dllg, 0.6827)
        axc.contourf(H0w, fw, dllw.T, levels=[lvl_w, 0.0], colors=[to_rgba(BLUE, 0.20)], zorder=1)
        axc.contour(H0w, fw, dllw.T, levels=[lvl_w], colors=[BLUE], linewidths=2.0, zorder=3)
        axc.contourf(H0g, fg_, dllg.T, levels=[lvl_g, 0.0], colors=[to_rgba(RED, 0.20)], zorder=1)
        axc.contour(H0g, fg_, dllg.T, levels=[lvl_g], colors=[RED], linewidths=2.0, zorder=3)
        if mapw:
            axc.scatter(*mapw, marker="*", s=140, color=BLUE, edgecolor=SURFACE, linewidth=1.1, zorder=4)
        if mapg:
            axc.scatter(*mapg, marker="*", s=140, color=RED, edgecolor=SURFACE, linewidth=1.1, zorder=4)
        axc.scatter([TRUTH_H0], [ftrue], marker="X", s=95, color=INK, edgecolor=SURFACE, linewidth=1.1, zorder=5)
        axc.set_title(f"68% HPD overlay, $f_{{agn}}$={tag}\n"
                       f"(this work {lvl_w:.2f} / gw_agn {lvl_g:.2f} $\\Delta\\log L$)", fontsize=10)

        for ax_ in (axa, axb, axc):
            ax_.set_xlim(50, 100)
            ax_.set_ylim(0, 1)
            ax_.set_xlabel("$H_0$  [km/s/Mpc]")
        axa.set_ylabel(r"$f_{cat,2}$ / $\alpha_{AGN}$")

        rhow = jw.get("rho") if jw else None
        momg = joint_moments(H0g, fg_, np.where(np.isnan(dllg), -np.inf, llg_masked))
        summary["joint"][tag] = {
            "truth": {"H0": TRUTH_H0, "f": ftrue},
            "this_work": {
                "MAP": {"H0": mapw[0], "f": mapw[1]} if mapw else None,
                "H0_ci68": jw["H0"]["ci68"] if jw and "H0" in jw else None,
                "H0_ci90": jw["H0"]["ci90"] if jw and "H0" in jw else None,
                "f_ci68": jw["f"]["ci68"] if jw and "f" in jw else None,
                "f_ci90": jw["f"]["ci90"] if jw and "f" in jw else None,
                "rho": rhow,
                "hpd68_dlogl": lvl_w,
            },
            "gw_agn": {
                "MAP": {"H0": mapg[0], "alpha_agn": mapg[1]} if mapg else None,
                "H0_ci68": jg["H0"]["ci68"] if jg and "H0" in jg else None,
                "H0_ci90": jg["H0"]["ci90"] if jg and "H0" in jg else None,
                "alpha_agn_ci68": jg["alpha_agn"]["ci68"] if jg and "alpha_agn" in jg else None,
                "alpha_agn_ci90": jg["alpha_agn"]["ci90"] if jg and "alpha_agn" in jg else None,
                "rho": momg["rho"] if momg else None,
                "hpd68_dlogl": lvl_g,
            },
        }

    legend_handles = [
        Patch(facecolor=BLUE, label="this work (darksirens K=2)"),
        Patch(facecolor=RED, label="gw_agn (field convention)"),
        Line2D([0], [0], marker="X", color=INK, lw=0, markeredgecolor=SURFACE,
               markersize=9, label="truth"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.02 / len(rows)), frameon=False, fontsize=9)
    fig.suptitle("Joint $(H_0, f)$ posterior surfaces: this work vs gw_agn "
                 "(levels $\\Delta\\log L$= -1, -4, -9  ~1/2/3$\\sigma$-ish)", fontsize=13, y=1.0)
    fig.tight_layout()
    outpath = FIGS / "fig_joint.png"
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {outpath}")
    return outpath


# --------------------------------------------------------------------------- #
# comparison_summary.json extras: K=2 H0 scans, deltas vs reference
# --------------------------------------------------------------------------- #
def build_h0_k2_summary(summary):
    summary["h0_k2"] = {}
    for key, tag, ffixed in [("ftruth", "h0_k2_fagn0.3_ftruth", 0.307),
                              ("f1", "h0_k2_fagn0.3_f1", 1.0)]:
        d = read_h5(RESULTS / f"{tag}.h5", ["H0_grid", "log_likelihood"])
        j = read_json(RESULTS / f"{tag}.json")
        if d is None:
            summary["h0_k2"][key] = None
            continue
        H0v, ll = d["H0_grid"], d["log_likelihood"]
        rq = quadratic_refine_argmax(H0v, ll)
        hw = ci_halfwidth(j["H0"]) if j and "H0" in j else float("nan")
        summary["h0_k2"][key] = {
            "f_fixed": ffixed,
            "argmax_refined": rq["argmax_refined"],
            "argmax_grid": rq["argmax_grid"],
            "median": j["H0"]["median"] if j and "H0" in j else float("nan"),
            "ci68_halfwidth": hw,
            "truth_H0": TRUTH_H0,
        }


def build_deltas(summary, ref):
    summary["gw_agn_reference"] = ref if ref is not None else None
    summary["deltas_vs_reference"] = {}
    if ref is None:
        warn("gw_agn coverage reference json missing; deltas not computed")
        return
    for model in ("ds", "dsc"):
        summary["deltas_vs_reference"][model] = {}
        for tracer in ("gal", "agn"):
            agg = summary.get("coverage_aggregate", {}).get(model, {}).get(tracer)
            refagg = ref.get(tracer, {})
            if not agg or not refagg:
                continue
            summary["deltas_vs_reference"][model][tracer] = {
                "delta_mean_median": (agg["mean_median"] - refagg.get("mean_median_r00_09", float("nan")))
                if np.isfinite(agg["mean_median"]) else float("nan"),
                "delta_mean_ci68_halfwidth": (agg["mean_ci68_halfwidth"] - refagg.get("mean_ci68_halfwidth", float("nan")))
                if np.isfinite(agg["mean_ci68_halfwidth"]) else float("nan"),
                "reference_mean_median": refagg.get("mean_median_r00_09"),
                "reference_sem_median": refagg.get("sem_median"),
                "reference_mean_ci68_halfwidth": refagg.get("mean_ci68_halfwidth"),
            }


# --------------------------------------------------------------------------- #
# Verdict table
# --------------------------------------------------------------------------- #
def print_verdict_table(summary):
    print("\n" + "=" * 78)
    print("VERDICT — coverage aggregates vs gw_agn field-convention reference")
    print("=" * 78)
    header = f"{'model':<6}{'tracer':<7}{'n':>3}{'mean_median':>14}{'ref_median':>13}{'delta':>9}{'mean_hw':>10}{'ref_hw':>9}{'delta_hw':>10}"
    print(header)
    print("-" * len(header))
    for model in ("ds", "dsc"):
        for tracer in ("gal", "agn"):
            agg = summary.get("coverage_aggregate", {}).get(model, {}).get(tracer, {})
            delt = summary.get("deltas_vs_reference", {}).get(model, {}).get(tracer, {})
            print(f"{model:<6}{tracer:<7}{agg.get('n_realizations', 0):>3}"
                  f"{nice(agg.get('mean_median')):>14}"
                  f"{nice(delt.get('reference_mean_median')):>13}"
                  f"{nice(delt.get('delta_mean_median')):>9}"
                  f"{nice(agg.get('mean_ci68_halfwidth')):>10}"
                  f"{nice(delt.get('reference_mean_ci68_halfwidth')):>9}"
                  f"{nice(delt.get('delta_mean_ci68_halfwidth')):>10}")

    print("\n" + "=" * 78)
    print("VERDICT — joint MAPs vs truth (H0=%.2f)" % TRUTH_H0)
    print("=" * 78)
    header2 = f"{'set':<10}{'estimator':<14}{'MAP_H0':>9}{'MAP_f':>8}{'d_H0':>8}{'d_f':>8}{'rho':>8}"
    print(header2)
    print("-" * len(header2))
    for tag, block in summary.get("joint", {}).items():
        truth = block.get("truth", {})
        for est_key, est_label in [("this_work", "this work"), ("gw_agn", "gw_agn")]:
            e = block.get(est_key) or {}
            mp = e.get("MAP") or {}
            h0v = mp.get("H0")
            fv = mp.get("f", mp.get("alpha_agn"))
            dh0 = (h0v - truth.get("H0")) if h0v is not None else float("nan")
            df = (fv - truth.get("f")) if fv is not None else float("nan")
            print(f"fagn{tag:<6}{est_label:<14}{nice(h0v,2):>9}{nice(fv,3):>8}"
                  f"{nice(dh0,2):>8}{nice(df,3):>8}{nice(e.get('rho'),3):>8}")

    print("\n" + "=" * 78)
    print("VERDICT — K=2 H0 scans (fagn0.3), refined argmax vs truth")
    print("=" * 78)
    for key, block in summary.get("h0_k2", {}).items():
        if not block:
            print(f"  {key}: MISSING")
            continue
        d = block["argmax_refined"] - TRUTH_H0
        print(f"  f_fixed={block['f_fixed']:<6} argmax_refined={nice(block['argmax_refined'],2):<8} "
              f"median={nice(block['median'],2):<8} hw68={nice(block['ci68_halfwidth'],2):<7} "
              f"delta_vs_truth={nice(d,2)}")

    print("\n" + "=" * 78)
    print("VERDICT — f-scan argmax/range summary")
    print("=" * 78)
    header3 = f"{'set':<16}{'truth_f':>9}{'argmax':>9}{'range':>10}{'per_ev_slope':>14}"
    print(header3)
    print("-" * len(header3))
    for tag, b in summary.get("fscan", {}).items():
        print(f"fagn{tag:<12}{nice(b.get('truth_f'),3):>9}{nice(b.get('argmax_grid'),3):>9}"
              f"{nice(b.get('range'),1):>10}{nice(b.get('per_event_slope'),4):>14}")

    if WARNINGS:
        print("\n" + "=" * 78)
        print(f"WARNINGS ({len(WARNINGS)}) — see also results/comparison_summary.json['warnings']")
        print("=" * 78)
        for w in WARNINGS:
            print(f"  - {w}")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "truth": {"H0": TRUTH_H0, "fscan_truths": FSCAN_TRUTH},
        "notes": [
            "f-scan estimand contrast is a measured finding (GATES.md GB incident 2), "
            "not a translation bug: darksirens K=2 f_cat,2 (isotropic-sky, per-pixel-"
            "conditional completion) and gw_agn alpha_AGN (number-density / sky-"
            "clustering field convention) are structurally different quantities.",
            "argmax_refined = quadratic interpolation through the peak grid node and "
            "its immediate neighbors; falls back to the raw grid node at a boundary "
            "peak or degenerate fit (see per-entry 'refined' flags where present).",
            "gw_agn rec_fagn*.h5 grids contain a huge-negative sentinel (~-1.4e11) or "
            "-inf where the field mixture is exactly zero; masked as NaN for display "
            "and for HPD/moment computations (values < grid_max - 1e6).",
        ],
    }

    ref = read_json(RESULTS / "gw_agn_coverage_reference_r00_09.json")

    build_fig_fscans(summary)
    build_h0_k2_summary(summary)
    build_fig_h0_pertracer(summary)
    build_fig_h0_summary(summary, ref)
    build_fig_joint(summary)
    build_deltas(summary, ref)

    summary["warnings"] = WARNINGS

    outpath = RESULTS / "comparison_summary.json"
    outpath.write_text(json.dumps(sanitize_nans(summary), indent=2))
    print(f"\nwrote {outpath}")

    print_verdict_table(summary)


if __name__ == "__main__":
    main()
