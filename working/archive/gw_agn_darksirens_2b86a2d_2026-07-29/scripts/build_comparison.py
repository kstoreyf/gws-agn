#!/usr/bin/env python3
"""build_comparison.py — comparison deliverables for the FIXED darksirens
campaign (see ../../gw_agn_darksirens/{RECON.md,GATES.md,RESULTS.md} for the
ORIGINAL campaign that diagnosed the K=2 conditional-mixture estimand problem;
../../gw_agn_darksirens/memory notes darksirens PRs #204-#212).

STORY: darksirens' K=2 mixture under the isotropic-sky conditional convention
rails f -> 1 regardless of truth (the original campaign's central finding).
The fix (field-convention sky weighting + catalog-targeted injections, PRs
#204-#212, merged to master) is rerun here end-to-end. Headline: dark_sirens
FIELD mode at the complete-catalog limit (log10n0=-12, tag "dsf") reproduces
gw_agn almost exactly; dark_sirens_complete field ("dscf") works with a small
interior low bias; the old conditional mode is the broken control.

Produces:
  figs/fig_recovery_ladder.png   — money plot: recovered f vs truth, 4 series
  figs/fig_fscans_fixed.png      — fixed-mode f-scan curves vs broken control
  figs/fig_joint_fixed.png       — 2x3 joint (H0,f) contours
  figs/fig_h0_pertracer_fixed.png— 2x2 per-tracer H0 coverage
  results/comparison_summary.json

Pure post-processing: plain h5py/numpy/matplotlib, no darksirens import, CPU
only. JAX_PLATFORMS pinned to cpu defensively.

Every file read is defensive: missing/unreadable inputs are warned about and
skipped rather than raising, EXCEPT the three PRIMARY landing files
(c2_h0_dsf_k2_fagn0.3.h5, c2_joint_dsf_fagn0.3.h5, c2_joint_dsf_fagn0.7.h5)
which the caller is expected to have waited for before invoking the final
pass of this script.
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
BASE = SCRIPT_DIR.parent  # .../gw_agn_darksirens_fixed
RESULTS = BASE / "results"
DATA = BASE / "data"
FIGS = BASE / "figs"
FIGS.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)

ORIG = BASE.parent / "gw_agn_darksirens"          # original campaign (broken control)
ORIG_RESULTS = ORIG / "results"
PREV_RESULTS = BASE.parent / "gw_agn_darksirens_fixed" / "results"   # #212-era rerun
GW_AGN_REC = Path("/hildafs/projects/phy230014p/magana/gws-agn/working/gw_agn/results/recovery")

# Tag prefix of THIS run's scans. Arm L = the legacy-equivalent total-variance
# budget (--max_likelihood_variance 1e6), which is what is comparable with the
# #212-era numbers; Arm G (default budget) tags are prefixed "G_".
PREFIX = os.environ.get("COMPARISON_PREFIX", "L_")
PREV_PREFIX = "c2_"          # the #212-era run's catalog-targeted-injection lane

# Per-tracer K=1 coverage scans come from ARM LI, which matches the ISOTROPIC
# injection set (injections.h5) the #212-era run's published per-tracer numbers
# used. Arm L's per-tracer scans use the catalog-targeted lane instead; comparing
# those against the published numbers would conflate the injection set with the
# code change, since mu(H0) differs between the two lanes.
COVERAGE_PREFIX = os.environ.get("COVERAGE_PREFIX", "LI_")
PREV_COVERAGE_PREFIX = ""    # the #212-era run's tags are bare `h0_...`

TRUTH_H0 = 67.74

FSCAN_TAGS = ["0.0", "0.3", "0.7", "1.0"]
FSCAN_TRUTH = {"0.0": 0.00989, "0.3": 0.307, "0.7": 0.703, "1.0": 1.0}

# --------------------------------------------------------------------------- #
# Palette — dataviz skill references/palette.md, fixed categorical order.
# Reused verbatim from the original campaign's compare_results.py (validated
# instance: light-mode worst adjacent CVD dE 24.2); slots assigned in a fixed
# order and never cycled/reassigned per-filter.
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

# f-scan truth-set color (categorical slots 1-4, fixed order) — reused from
# the original campaign so the truth->color mapping reads identically across
# both campaigns' figures.
TRUTH_COLOR = {"0.0": BLUE, "0.3": AQUA, "0.7": YELLOW, "1.0": GREEN}

# Series identity colors, reused across all four figures (fixed order: the
# headline "dsf" series is BLUE throughout; RED is always gw_agn; ORANGE is
# always the broken conditional control; VIOLET/MAGENTA reserved/unused here).
SERIES_COLOR = {
    "dsf": BLUE,
    "dscf": AQUA,
    "gw_agn": RED,
    "conditional": ORANGE,
    "prev_dsf": VIOLET,
}
SERIES_LABEL = {
    "dsf": "dark_sirens field (dsf, complete-catalog limit) — master 2b86a2d",
    "dscf": "dark_sirens_complete field (dscf) — master 2b86a2d",
    "gw_agn": "gw_agn reference",
    "conditional": "darksirens conditional (broken, isotropic-sky)",
    "prev_dsf": "dsf — #212-era code (2026-07-10 run)",
}

COMBO_COLOR = {
    ("gal", "dscf"): AQUA,
    ("gal", "dsf"): BLUE,
    ("agn", "dscf"): GREEN,
    ("agn", "dsf"): YELLOW,
}
COMBO_LABEL = {
    ("gal", "dscf"): "GAL, dark_sirens_complete field",
    ("gal", "dsf"): "GAL, dark_sirens field (n0low)",
    ("agn", "dscf"): "AGN, dark_sirens_complete field",
    ("agn", "dsf"): "AGN, dark_sirens field (n0low)",
}

CLIP = -900.0  # shared display floor for DeltalogL panels

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
# Numerics (shared with the original campaign's compare_results.py)
# --------------------------------------------------------------------------- #
def quadratic_refine_argmax(x, y):
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
    if a >= 0:
        return {"argmax_refined": float(x[i]), "argmax_grid": float(x[i]), "refined": False}
    xv = -b / (2 * a)
    if not (min(x0, x2) <= xv <= max(x0, x2)):
        return {"argmax_refined": float(x[i]), "argmax_grid": float(x[i]), "refined": False}
    return {"argmax_refined": float(xv), "argmax_grid": float(x[i]), "refined": True}


def mask_sentinel(arr, thresh=1e6):
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
    return lvl if lvl < -1e-9 else -1e-6


def joint_moments(H0v, fv, ll_grid):
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


def fscan_entry_from_json(j, path_for_grid, truth_f):
    """Build a compact fscan summary dict from a scan_darksirens.py f-scan
    json summary (`f` block already has median/ci68/map/argmax), plus
    argmax_grid/range/per_event_slope recomputed from the paired .h5 grid."""
    if j is None:
        return None
    fb = j.get("f", {})
    entry = {
        "truth_f": truth_f,
        "median": fb.get("median"),
        "ci68": fb.get("ci68"),
        "ci90": fb.get("ci90"),
        "map": fb.get("map"),
        "argmax": fb.get("argmax"),
    }
    d = read_h5(path_for_grid, ["f_grid", "log_likelihood"])
    if d is not None:
        f, ll = d["f_grid"], d["log_likelihood"]
        entry["argmax_grid"] = float(f[int(np.nanargmax(ll))])
        entry["range"] = float(np.nanmax(ll) - np.nanmin(ll))
    return entry


# --------------------------------------------------------------------------- #
# Deliverable 1: fig_recovery_ladder.png — THE MONEY PLOT
# --------------------------------------------------------------------------- #
def build_fig_recovery_ladder(summary):
    fig, ax = plt.subplots(figsize=(8.2, 7.4), dpi=150)

    ax.plot([0, 1], [0, 1], color=BASELINE, lw=1.4, ls=(0, (5, 2.5)), zorder=1, label="y = x (truth)")

    # ---- gw_agn reference (rec_fagn*.json, full 2-D marginal alpha median)
    xs, ys, los, his = [], [], [], []
    for tag in FSCAN_TAGS:
        j = read_json(GW_AGN_REC / f"rec_fagn{tag}.json")
        if j is None or "alpha_agn" not in j:
            continue
        xs.append(FSCAN_TRUTH[tag])
        ys.append(j["alpha_agn"]["median"])
        lo, hi = j["alpha_agn"]["ci68"]
        los.append(ys[-1] - lo)
        his.append(hi - ys[-1])
    if xs:
        order = np.argsort(xs)
        xs, ys, los, his = (np.array(a)[order] for a in (xs, ys, los, his))
        ax.errorbar(xs, ys, yerr=[los, his], color=SERIES_COLOR["gw_agn"], fmt="o-", ms=8,
                     mec=SURFACE, mew=1.1, lw=1.4, capsize=3, zorder=4, label=SERIES_LABEL["gw_agn"])

    # ---- darksirens conditional (broken, original campaign): railed markers for truth>=0.3
    xs, ys, los, his, railed = [], [], [], [], []
    for tag in FSCAN_TAGS:
        j = read_json(ORIG_RESULTS / f"fscan_fagn{tag}.json")
        if j is None or "f" not in j:
            continue
        is_railed = j["f"].get("argmax", 0) >= 0.999
        xs.append(FSCAN_TRUTH[tag])
        if is_railed:
            ys.append(1.0)
            los.append(0.0)
            his.append(0.0)
        else:
            med = j["f"]["median"]
            lo, hi = j["f"]["ci68"]
            ys.append(med)
            los.append(med - lo)
            his.append(hi - med)
        railed.append(is_railed)
    if xs:
        order = np.argsort(xs)
        xs_a = np.array(xs)[order]
        ys_a = np.array(ys)[order]
        los_a = np.array(los)[order]
        his_a = np.array(his)[order]
        railed_a = np.array(railed)[order]
        ax.plot(xs_a, ys_a, color=SERIES_COLOR["conditional"], lw=1.2, ls=(0, (3, 1.5)), zorder=2, alpha=0.85)
        not_railed = ~railed_a
        if not_railed.any():
            ax.errorbar(xs_a[not_railed], ys_a[not_railed],
                         yerr=[los_a[not_railed], his_a[not_railed]],
                         color=SERIES_COLOR["conditional"], fmt="o", ms=8, mec=SURFACE, mew=1.1,
                         lw=1.4, capsize=3, zorder=3)
        if railed_a.any():
            ax.scatter(xs_a[railed_a], ys_a[railed_a], marker="^", s=140,
                       facecolor=SERIES_COLOR["conditional"], edgecolor=INK, linewidth=1.3, zorder=5)
        # one legend entry covering both marker styles
        ax.plot([], [], color=SERIES_COLOR["conditional"], marker="o", ls=(0, (3, 1.5)),
                 mfc=SERIES_COLOR["conditional"], mec=SURFACE, ms=8, lw=1.2,
                 label=SERIES_LABEL["conditional"])
        ax.scatter([], [], marker="^", s=140, facecolor=SERIES_COLOR["conditional"], edgecolor=INK,
                   linewidth=1.3, label="railed to $f{=}1$ (argmax at boundary)")

    # ---- #212-era dsf (the run this one supersedes): reproducibility check
    xs, ys, los, his = [], [], [], []
    for tag in FSCAN_TAGS:
        j = read_json(PREV_RESULTS / f"{PREV_PREFIX}fscan_dsf_n0low_fagn{tag}.json")
        if j is None or "f" not in j or j.get("f", {}).get("median") is None:
            continue
        xs.append(FSCAN_TRUTH[tag])
        ys.append(j["f"]["median"])
    if xs:
        order = np.argsort(xs)
        xs, ys = (np.array(a)[order] for a in (xs, ys))
        ax.plot(xs, ys, color=SERIES_COLOR["prev_dsf"], marker="x", ms=10, mew=2.0,
                ls=(0, (2, 1.4)), lw=1.3, zorder=7, label=SERIES_LABEL["prev_dsf"])

    # ---- this run's dsf (n0low, complete-catalog limit)
    xs, ys, los, his = [], [], [], []
    for tag in FSCAN_TAGS:
        j = read_json(RESULTS / f"{PREFIX}fscan_dsf_n0low_fagn{tag}.json")
        if j is None or "f" not in j or j.get("f", {}).get("median") is None:
            continue
        xs.append(FSCAN_TRUTH[tag])
        med = j["f"]["median"]
        lo, hi = j["f"]["ci68"]
        ys.append(med)
        los.append(med - lo)
        his.append(hi - med)
    if xs:
        order = np.argsort(xs)
        xs, ys, los, his = (np.array(a)[order] for a in (xs, ys, los, his))
        ax.errorbar(xs, ys, yerr=[los, his], color=SERIES_COLOR["dsf"], fmt="s-", ms=9,
                     mec=SURFACE, mew=1.2, lw=1.6, capsize=3, zorder=6, label=SERIES_LABEL["dsf"])

    # ---- fixed dscf: all four truths
    xs, ys, los, his = [], [], [], []
    for tag in FSCAN_TAGS:
        j = read_json(RESULTS / f"{PREFIX}fscan_dscf_fagn{tag}.json")
        if j is None or "f" not in j or j.get("f", {}).get("median") is None:
            continue
        xs.append(FSCAN_TRUTH[tag])
        med = j["f"]["median"]
        lo, hi = j["f"]["ci68"]
        ys.append(med)
        los.append(med - lo)
        his.append(hi - med)
    if xs:
        order = np.argsort(xs)
        xs, ys, los, his = (np.array(a)[order] for a in (xs, ys, los, his))
        ax.errorbar(xs, ys, yerr=[los, his], color=SERIES_COLOR["dscf"], fmt="D-", ms=7.5,
                     mec=SURFACE, mew=1.1, lw=1.4, capsize=3, zorder=5, label=SERIES_LABEL["dscf"])

    ax.set_xlim(-0.04, 1.04)
    ax.set_ylim(-0.04, 1.06)
    ax.set_aspect("equal")
    ax.set_xlabel(r"truth AGN-hosted fraction  $\alpha_{AGN}$")
    ax.set_ylabel(r"recovered $f$ / $\alpha_{AGN}$ (median, 68% CI)")
    ax.set_title("Recovery ladder on darksirens master @ 2b86a2d (2026-07-29)\n"
                 "K=2 mixture estimand: conditional (broken) vs field-weighted, and vs the "
                 "#212-era run it supersedes",
                 fontsize=11.5)
    ax.legend(loc="upper left", fontsize=8.3, ncol=1)
    fig.tight_layout()
    outpath = FIGS / "fig_recovery_ladder.png"
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {outpath}")
    return outpath


# --------------------------------------------------------------------------- #
# Deliverable 2: fig_fscans_fixed.png
# --------------------------------------------------------------------------- #
def build_fig_fscans_fixed(summary):
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.2, 5.4), dpi=150)

    def plot_curve(ax, path, color, lw, ls, alpha=1.0, zorder=3, label=None):
        d = read_h5(path, ["f_grid", "log_likelihood"])
        if d is None:
            return None
        f, ll = d["f_grid"], d["log_likelihood"]
        dll_disp = np.clip(ll - np.nanmax(ll), CLIP, 0)
        ax.plot(f, dll_disp, color=color, lw=lw, ls=ls, alpha=alpha, zorder=zorder,
                 solid_capstyle="round", label=label)
        return f, ll

    # ---- LEFT: fixed-mode curves ----
    # dscf 0.0 / 1.0 — thick (only mode available for these truths)
    for tag in ("0.0", "1.0"):
        plot_curve(axL, RESULTS / f"{PREFIX}fscan_dscf_fagn{tag}.h5", TRUTH_COLOR[tag], 2.2, "-", zorder=4)
    # dsf n0low 0.3 / 0.7 — thick, headline
    for tag in ("0.3", "0.7"):
        plot_curve(axL, RESULTS / f"{PREFIX}fscan_dsf_n0low_fagn{tag}.h5", TRUTH_COLOR[tag], 2.2, "-", zorder=4)
    # dscf 0.3 / 0.7 — thin, secondary
    for tag in ("0.3", "0.7"):
        plot_curve(axL, RESULTS / f"{PREFIX}fscan_dscf_fagn{tag}.h5", TRUTH_COLOR[tag], 1.0, "-", alpha=0.6, zorder=2)
    # injB — dashed
    plot_curve(axL, RESULTS / f"{PREFIX}fscan_dscf_fagn0.3_injB.h5", TRUTH_COLOR["0.3"], 1.3, (0, (4, 1.6)), zorder=3)
    # n0true demo — dotted grey, annotated
    d = plot_curve(axL, RESULTS / f"{PREFIX}fscan_dsf_n0true_fagn0.3.h5", INK_MUTED, 1.6, (0, (1, 1.3)), zorder=3)
    if d is not None:
        f, ll = d
        dll = np.clip(ll - np.nanmax(ll), CLIP, 0)
        i_lab = int(len(f) * 0.55)
        axL.annotate("misspecified $n_0$\n(complete catalog)",
                     xy=(f[i_lab], dll[i_lab]), xytext=(0.68, -260),
                     fontsize=7.6, color=INK_SECONDARY, ha="left", va="center",
                     arrowprops=dict(arrowstyle="->", color=INK_MUTED, lw=0.9, alpha=0.85),
                     bbox=dict(boxstyle="round,pad=0.25", fc=SURFACE, ec=BASELINE, lw=0.6))

    for tag in FSCAN_TAGS:
        axL.axvline(FSCAN_TRUTH[tag], color=TRUTH_COLOR[tag], lw=1.0, ls=(0, (1, 1.6)), alpha=0.6, zorder=1)

    axL.set_xlim(0, 1)
    axL.set_ylim(CLIP - 30, 30)
    axL.set_xlabel(r"$f_{cat,2}$  (AGN-catalog mixture weight)")
    axL.set_ylabel(r"$\Delta\log\mathcal{L}$  (clipped at %d)" % CLIP)
    axL.set_title("AFTER — field-convention sky weighting (PRs #204–#212)\n"
                   "dsf = complete-catalog limit; dscf = dark_sirens_complete field", fontsize=10.3)

    truth_handles = [Line2D([0], [0], color=TRUTH_COLOR[t], lw=2.2,
                             label=f"truth $f$={FSCAN_TRUTH[t]:.3f}") for t in FSCAN_TAGS]
    style_handles = [
        Line2D([0], [0], color=INK_SECONDARY, lw=2.2, ls="-", label="dsf (n0low) / dscf 0.0,1.0 — thick"),
        Line2D([0], [0], color=INK_SECONDARY, lw=1.0, ls="-", alpha=0.6, label="dscf 0.3,0.7 — thin"),
        Line2D([0], [0], color=INK_SECONDARY, lw=1.3, ls=(0, (4, 1.6)), label="injB (alt. injection seed)"),
        Line2D([0], [0], color=INK_MUTED, lw=1.6, ls=(0, (1, 1.3)), label="n0true (misspecified demo)"),
    ]

    # ---- RIGHT: code drift — this run (solid) vs the #212-era run (dashed) ----
    # Same events, same grids, same nuisance points, same estimator: any visible
    # separation is the 295 commits between 8eae3ea and 2b86a2d.
    for tag in FSCAN_TAGS:
        plot_curve(axR, RESULTS / f"{PREFIX}fscan_dsf_n0low_fagn{tag}.h5",
                   TRUTH_COLOR[tag], 2.4, "-", zorder=4)
        plot_curve(axR, PREV_RESULTS / f"{PREV_PREFIX}fscan_dsf_n0low_fagn{tag}.h5",
                   INK, 1.1, (0, (3, 1.8)), alpha=0.9, zorder=5)
        axR.axvline(FSCAN_TRUTH[tag], color=TRUTH_COLOR[tag], lw=1.0, ls=(0, (1, 1.6)), alpha=0.6, zorder=1)

    axR.set_xlim(0, 1)
    axR.set_ylim(CLIP - 30, 30)
    axR.set_xlabel(r"$f_{cat,2}$  (AGN-catalog mixture weight)")
    axR.set_ylabel(r"$\Delta\log\mathcal{L}$  (clipped at %d)" % CLIP)
    axR.set_title("CODE DRIFT — dsf on master 2b86a2d (colour) vs the\n"
                  "#212-era run (black dashed), identical inputs", fontsize=10.3)
    axR.plot([], [], color=INK, lw=1.1, ls=(0, (3, 1.8)), label="#212-era (2026-07-10)")
    axR.legend(loc="lower right", fontsize=8)

    for ax in (axL, axR):
        ax.grid(True, which="major", axis="both")

    fig.suptitle("f-scans on darksirens master @ 2b86a2d — field-mode estimand, and drift "
                 "against the #212-era run", fontsize=13, y=1.04)
    fig.legend(handles=truth_handles + style_handles, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.14), frameon=False, fontsize=8.3)
    fig.tight_layout()
    outpath = FIGS / "fig_fscans_fixed.png"
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {outpath}")
    return outpath


# --------------------------------------------------------------------------- #
# Deliverable 3: fig_joint_fixed.png
# --------------------------------------------------------------------------- #
def build_fig_joint_fixed(summary):
    levels = [-9.0, -4.0, -1.0, 0.0]

    def cell_data(h5path, jsonpath, hkey="H0_grid", fkey="f_grid", sentinel=False):
        d = read_h5(h5path, [hkey, fkey, "log_likelihood" if not sentinel else "log_likelihood_grid"] if not sentinel
                    else [hkey, fkey, "log_likelihood_grid"])
        return d

    rows_spec = [
        ("0.3", 0.307, [
            ("dsf (fixed, field)", RESULTS / f"{PREFIX}joint_dsf_fagn0.3.h5", RESULTS / f"{PREFIX}joint_dsf_fagn0.3.json",
             "H0_grid", "f_grid", "log_likelihood", False, SERIES_COLOR["dsf"]),
            ("gw_agn reference", GW_AGN_REC / "rec_fagn0.3.h5", GW_AGN_REC / "rec_fagn0.3.json",
             "H0_grid", "alpha_agn_grid", "log_likelihood_grid", True, SERIES_COLOR["gw_agn"]),
            ("conditional (broken)", ORIG_RESULTS / "joint_fagn0.3.h5", ORIG_RESULTS / "joint_fagn0.3.json",
             "H0_grid", "f_grid", "log_likelihood", False, SERIES_COLOR["conditional"]),
        ]),
        ("0.7", 0.703, [
            ("dsf (fixed, field)", RESULTS / f"{PREFIX}joint_dsf_fagn0.7.h5", RESULTS / f"{PREFIX}joint_dsf_fagn0.7.json",
             "H0_grid", "f_grid", "log_likelihood", False, SERIES_COLOR["dsf"]),
            ("gw_agn reference", GW_AGN_REC / "rec_fagn0.7.h5", GW_AGN_REC / "rec_fagn0.7.json",
             "H0_grid", "alpha_agn_grid", "log_likelihood_grid", True, SERIES_COLOR["gw_agn"]),
            ("dscf (fixed)", RESULTS / f"{PREFIX}joint_dscf_fagn0.7.h5", RESULTS / f"{PREFIX}joint_dscf_fagn0.7.json",
             "H0_grid", "f_grid", "log_likelihood", False, SERIES_COLOR["dscf"]),
        ]),
    ]

    summary["joint"] = {"dsf": {}, "dscf": {}, "conditional": {}, "gw_agn": {}}

    n_rows = len(rows_spec)
    fig, axes = plt.subplots(n_rows, 3, figsize=(15.3, 5.15 * n_rows), dpi=150, squeeze=False)

    any_row_missing_primary = False

    for irow, (tag, ftrue, cells) in enumerate(rows_spec):
        for icol, (label, h5p, jp, hkey, fkey, llkey, is_gwagn, color) in enumerate(cells):
            ax = axes[irow, icol]
            d = read_h5(h5p, [hkey, fkey, llkey])
            j = read_json(jp)
            if d is None:
                if "dsf" in label:
                    any_row_missing_primary = True
                ax.text(0.5, 0.5, f"missing:\n{h5p.name}", transform=ax.transAxes,
                        ha="center", va="center", color=INK_MUTED, fontsize=9)
                ax.set_xlim(50, 100)
                ax.set_ylim(0, 1)
                continue
            H0v, fv, ll = d[hkey], d[fkey], d[llkey]
            if is_gwagn:
                ll_use = mask_sentinel(ll)
            else:
                ll_use = ll
            dll = ll_use - np.nanmax(ll_use)
            shades = [to_rgba(color, a) for a in (0.22, 0.5, 0.88)]
            ax.contourf(H0v, fv, dll.T, levels=levels, colors=shades)
            ax.contour(H0v, fv, dll.T, levels=levels[:-1], colors=INK_SECONDARY, linewidths=0.6)

            mp = None
            if j and "map" in j:
                mp = (j["map"]["H0"], j["map"].get("f", j["map"].get("alpha_agn")))
            elif np.isfinite(dll).any():
                idx = np.unravel_index(np.nanargmax(dll), dll.shape)
                mp = (float(H0v[idx[0]]), float(fv[idx[1]]))
            if mp:
                ax.scatter(*mp, marker="*", s=190, color=color, edgecolor=SURFACE, linewidth=1.3, zorder=5)
            ax.scatter([TRUTH_H0], [ftrue], marker="X", s=100, color=INK, edgecolor=SURFACE, linewidth=1.1, zorder=6)

            map_str = f"MAP=({mp[0]:.1f}, {mp[1]:.3f})" if mp else "MAP n/a"
            ax.set_title(f"{label}, $f_{{agn}}$={tag}\n{map_str}", fontsize=10)
            if label.startswith("dsf") and mp is not None:
                # Residual convention-stack tilt: injection-based mu(H0) + informative
                # masses (spectral-siren term), distinct from the estimand fix — the f
                # axis is unbiased while H0 sits low (cf. original campaign RESULTS.md,
                # K=2 conditional H0 scan showed the same stack tilt at fixed f).
                dh0 = mp[0] - TRUTH_H0
                ax.annotate(
                    f"$\\Delta H_0$ = {dh0:+.1f} km/s/Mpc:\nresidual convention-stack tilt\n"
                    "(injection-based $\\mu(H_0)$ +\ninformative masses),\ndistinct from the estimand fix",
                    xy=mp, xytext=(0.97, 0.05), textcoords="axes fraction",
                    ha="right", va="bottom", fontsize=7.4, color=INK_SECONDARY,
                    arrowprops=dict(arrowstyle="->", color=INK_MUTED, lw=0.9, alpha=0.85,
                                    shrinkB=8),
                    bbox=dict(boxstyle="round,pad=0.3", fc=SURFACE, ec=BASELINE, lw=0.7),
                )
            ax.set_xlim(50, 100)
            ax.set_ylim(0, 1)
            if irow == n_rows - 1:
                ax.set_xlabel("$H_0$  [km/s/Mpc]")
            if icol == 0:
                ax.set_ylabel(r"$f_{cat,2}$ / $\alpha_{AGN}$")

            # ---- summary bookkeeping ----
            fkey_label = fb = j.get("f", j.get("alpha_agn")) if j else None
            hkey_label = j.get("H0") if j else None
            rho = j.get("rho") if j else None
            if is_gwagn and rho is None:
                mom = joint_moments(H0v, fv, np.where(np.isnan(dll), -np.inf, ll_use))
                rho = mom["rho"] if mom else None
            entry = {
                "truth": {"H0": TRUTH_H0, "f": ftrue},
                "MAP": {"H0": mp[0], "f": mp[1]} if mp else None,
                "H0_median": hkey_label.get("median") if hkey_label else None,
                "H0_ci68": hkey_label.get("ci68") if hkey_label else None,
                "f_median": fkey_label.get("median") if fkey_label else None,
                "f_ci68": fkey_label.get("ci68") if fkey_label else None,
                "rho": rho,
            }
            if is_gwagn:
                summary["joint"]["gw_agn"][tag] = entry
            elif "conditional" in label.lower():
                summary["joint"]["conditional"][tag] = entry
            elif "dscf" in label.lower():
                summary["joint"]["dscf"][tag] = entry
            else:
                summary["joint"]["dsf"][tag] = entry

    if any_row_missing_primary:
        warn("fig_joint_fixed: primary dsf joint grid(s) missing — figure written with placeholder panel(s)")

    # Extra summary-only bookkeeping: dscf fagn0.3 joint exists in results/ but is not
    # part of the fixed 2x3 figure layout (row 1 col 3 is the broken-conditional control
    # instead) — still record it for completeness in comparison_summary.json.
    d = read_h5(RESULTS / f"{PREFIX}joint_dscf_fagn0.3.h5", ["H0_grid", "f_grid", "log_likelihood"])
    j = read_json(RESULTS / f"{PREFIX}joint_dscf_fagn0.3.json")
    if d is not None:
        mp = (j["map"]["H0"], j["map"]["f"]) if j and "map" in j else None
        summary["joint"]["dscf"]["0.3"] = {
            "truth": {"H0": TRUTH_H0, "f": 0.307},
            "MAP": {"H0": mp[0], "f": mp[1]} if mp else None,
            "H0_median": j["H0"].get("median") if j and "H0" in j else None,
            "H0_ci68": j["H0"].get("ci68") if j and "H0" in j else None,
            "f_median": j["f"].get("median") if j and "f" in j else None,
            "f_ci68": j["f"].get("ci68") if j and "f" in j else None,
            "rho": j.get("rho") if j else None,
            "note": "not part of fig_joint_fixed layout (row1 col3 is the conditional control)",
        }

    legend_handles = [
        Patch(facecolor=SERIES_COLOR["dsf"], label="dsf (fixed, field)"),
        Patch(facecolor=SERIES_COLOR["gw_agn"], label="gw_agn reference"),
        Patch(facecolor=SERIES_COLOR["conditional"], label="conditional (broken)"),
        Patch(facecolor=SERIES_COLOR["dscf"], label="dscf (fixed)"),
        Line2D([0], [0], marker="X", color=INK, lw=0, markeredgecolor=SURFACE,
               markersize=9, label="truth"),
        Line2D([0], [0], marker="*", color=INK_SECONDARY, lw=0, markeredgecolor=SURFACE,
               markersize=12, label="MAP"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=6,
               bbox_to_anchor=(0.5, -0.02 / n_rows), frameon=False, fontsize=9)
    fig.suptitle(r"Joint $(H_0, f)$ posteriors: fixed field-mode vs gw_agn vs broken conditional control"
                 "\n" r"(levels $\Delta\log L$= -1, -4, -9)", fontsize=13, y=1.01)
    fig.tight_layout()
    outpath = FIGS / "fig_joint_fixed.png"
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {outpath}")
    return outpath


# --------------------------------------------------------------------------- #
# Deliverable 4: fig_h0_pertracer_fixed.png
# --------------------------------------------------------------------------- #
def build_fig_h0_pertracer_fixed(summary, ref):
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.0), dpi=150, sharex=True, sharey=True)
    tracers = ["gal", "agn"]
    models = ["dscf", "dsf"]
    model_title = {"dscf": "dark_sirens_complete field (dscf), 10 realizations",
                   "dsf": "dark_sirens field, n0low (dsf), 5 realizations"}
    n_real = {"dscf": 10, "dsf": 5}

    summary["coverage"] = {m: {t: {} for t in tracers} for m in models}
    summary["coverage_aggregate"] = {m: {} for m in models}

    for ir, tracer in enumerate(tracers):
        for ic, model in enumerate(models):
            ax = axes[ir, ic]
            color = COMBO_COLOR[(tracer, model)]
            refined_list, median_list, hw_list = [], [], []
            n_found = 0
            for R in range(n_real[model]):
                rtag = f"r{R:02d}"
                tag = f"{COVERAGE_PREFIX}h0_{model}_{tracer}_{rtag}"
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
                sem_arg = float(np.std(refined_list, ddof=1) / np.sqrt(len(refined_list))) if len(refined_list) > 1 else float("nan")
                mean_hw = float(np.mean(hw_list)) if hw_list else float("nan")
                ax.text(0.03, 0.10,
                        f"mean refined argmax = {mean_arg:.2f} $\\pm$ {sem_arg:.2f} km/s/Mpc (sem, n={n_found}/{n_real[model]})\n"
                        f"mean 68% half-width = {mean_hw:.2f} km/s/Mpc",
                        transform=ax.transAxes, fontsize=8.0, color=INK_SECONDARY, va="bottom")
            else:
                mean_arg = sem_arg = mean_hw = float("nan")
                ax.text(0.5, 0.5, "no realizations found yet", transform=ax.transAxes,
                        ha="center", va="center", color=INK_MUTED, fontsize=9)

            refblk = ref.get(tracer, {}) if ref else {}
            ref_mean = refblk.get("mean_median_r00_09")
            ref_sem = refblk.get("sem_median")
            ref_hw = refblk.get("mean_ci68_halfwidth")
            if ref_mean is not None:
                ax.text(0.03, 0.03,
                        f"gw_agn ref: {ref_mean:.3f} $\\pm$ {ref_sem:.3f} (sem), hw={ref_hw:.3f}",
                        transform=ax.transAxes, fontsize=7.6, color=INK_MUTED, va="bottom")

            if median_list:
                mean_med = float(np.mean(median_list))
                sem_med = float(np.std(median_list, ddof=1) / np.sqrt(len(median_list))) if len(median_list) > 1 else float("nan")
            else:
                mean_med = sem_med = float("nan")
            summary["coverage_aggregate"][model][tracer] = {
                "n_realizations": n_found,
                "mean_median": mean_med,
                "sem_median": sem_med,
                "mean_ci68_halfwidth": mean_hw,
                "mean_argmax_refined": mean_arg,
                "sem_argmax_refined": sem_arg,
                "injection_set": "injections.h5 (isotropic; matches the #212-era run)",
            }

            # Same aggregate from the #212-era run's own scans (bare h0_* tags,
            # same injection set) — the code-drift comparison for K=1.
            prev_refined, prev_med, prev_hw = [], [], []
            for R in range(n_real[model]):
                ptag = f"{PREV_COVERAGE_PREFIX}h0_{model}_{tracer}_r{R:02d}"
                pd_ = read_h5(PREV_RESULTS / f"{ptag}.h5", ["H0_grid", "log_likelihood"])
                pj = read_json(PREV_RESULTS / f"{ptag}.json")
                if pd_ is None:
                    continue
                prq = quadratic_refine_argmax(pd_["H0_grid"], pd_["log_likelihood"])
                if np.isfinite(prq["argmax_refined"]):
                    prev_refined.append(prq["argmax_refined"])
                if pj and "H0" in pj:
                    if np.isfinite(pj["H0"]["median"]):
                        prev_med.append(pj["H0"]["median"])
                    hwp = ci_halfwidth(pj["H0"])
                    if np.isfinite(hwp):
                        prev_hw.append(hwp)
            summary.setdefault("coverage_aggregate_prev", {}).setdefault(model, {})[tracer] = {
                "n_realizations": len(prev_refined),
                "mean_median": float(np.mean(prev_med)) if prev_med else float("nan"),
                "mean_ci68_halfwidth": float(np.mean(prev_hw)) if prev_hw else float("nan"),
                "mean_argmax_refined": (float(np.mean(prev_refined))
                                        if prev_refined else float("nan")),
                "delta_mean_argmax_this_minus_prev": (
                    mean_arg - float(np.mean(prev_refined)) if prev_refined
                    and np.isfinite(mean_arg) else float("nan")),
            }

            ax.set_xlim(50, 100)
            ax.set_ylim(CLIP - 30, 30)
            if ir == 1:
                ax.set_xlabel("$H_0$  [km/s/Mpc]")
            if ic == 0:
                ax.set_ylabel(r"$\Delta\log\mathcal{L}$")
            ax.set_title(f"{tracer.upper()}  ×  {model_title[model]}", fontsize=10)

    fig.suptitle(f"Per-tracer $H_0$ coverage on master @ 2b86a2d (truth $H_0$={TRUTH_H0}; "
                 "isotropic injection set, matched to the #212-era run)",
                 fontsize=12.5, y=1.0)
    fig.tight_layout()
    outpath = FIGS / "fig_h0_pertracer_fixed.png"
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {outpath}")
    return outpath


# --------------------------------------------------------------------------- #
# Deliverable 5: fig_variance_guard.png — the new-code finding
# --------------------------------------------------------------------------- #
def build_fig_variance_guard(summary):
    """sigma^2_lnL per configuration against master's default budget.

    Stacked bars split the budget into its two measured components: the summed
    per-event PE reweighting variance and the selection-integral term
    N_obs^2/Neff.  The 1.0 line is the GWTC-4.0/5.0 threshold master enforces by
    default; everything to its right is rejected outright (logL = -inf).
    """
    ga = read_json(RESULTS / "guard_audit_summary.json")
    if ga is None or not ga.get("rows"):
        warn("guard_audit_summary.json missing/empty; skipping fig_variance_guard")
        return None
    rows = [r for r in ga["rows"] if r.get("sigma2_total") is not None]
    if not rows:
        warn("no usable guard-audit rows; skipping fig_variance_guard")
        return None

    # Group: K=2 mixtures (N=1000) first, then K=1 per-tracer (N=100).
    def sort_key(r):
        return (-r["n_catalogs"], r["universe_model"], r["tag"])
    rows.sort(key=sort_key)

    labels = [r["tag"] for r in rows]
    pe = np.array([r["pe_variance_sum"] for r in rows])
    sel = np.array([r["selection_variance"] for r in rows])
    passes = np.array([bool(r["passes_default_guard"]) for r in rows])

    h = max(4.2, 0.26 * len(rows) + 2.0)
    fig, ax = plt.subplots(figsize=(9.6, h), dpi=150)
    y = np.arange(len(rows))
    ax.barh(y, pe, color=VIOLET, height=0.72, label=r"$\sum_i \sigma^2_i$ (per-event PE reweighting)")
    ax.barh(y, sel, left=pe, color=YELLOW, height=0.72, label=r"$N_{obs}^2/N_{eff}$ (selection integral)")
    ax.axvline(1.0, color=RED, lw=1.8, zorder=5)
    ax.annotate("default budget\n$\\sigma^2_{\\ln\\mathcal{L}} \\leq 1$ (GWTC-4.0/5.0)",
                xy=(1.0, len(rows) - 0.4), xytext=(1.35, len(rows) - 0.4),
                color=RED, fontsize=8.4, va="center",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.0))

    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"{'✓' if p else '✗'}  {lab}" for lab, p in zip(labels, passes)], fontsize=7.6)
    for tick, p in zip(ax.get_yticklabels(), passes):
        tick.set_color(INK if p else INK_MUTED)
    ax.set_xscale("log")
    ax.set_xlim(max(1e-3, min(0.5 * (pe + sel).min(), 0.5)), 1.6 * (pe + sel).max())
    ax.set_xlabel(r"measured $\sigma^2_{\ln\mathcal{L}} = \sum_i \sigma^2_i + N_{obs}^2/N_{eff}$")
    ax.set_title("What master @ 2b86a2d's total-variance guard says about this campaign's mock\n"
                 "(✓ = admitted at the default budget; ✗ = rejected, every grid cell $-\\infty$)",
                 fontsize=10.6)
    ax.legend(loc="lower right", fontsize=8.4)
    ax.grid(True, axis="x")
    ax.grid(False, axis="y")
    fig.tight_layout()
    outpath = FIGS / "fig_variance_guard.png"
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {outpath}")
    return outpath


# --------------------------------------------------------------------------- #
# comparison_summary.json extras
# --------------------------------------------------------------------------- #
def build_fscan_summary(summary):
    summary["fscan"] = {"dscf": {}, "dsf_n0low": {}, "conditional_broken": {},
                        "prev_dscf": {}, "prev_dsf_n0low": {}}
    summary["fscan_extras"] = {}

    for tag in FSCAN_TAGS:
        j = read_json(RESULTS / f"{PREFIX}fscan_dscf_fagn{tag}.json")
        summary["fscan"]["dscf"][tag] = fscan_entry_from_json(j, RESULTS / f"{PREFIX}fscan_dscf_fagn{tag}.h5", FSCAN_TRUTH[tag])

    # This run scans the FULL dsf ladder (the #212-era run did 0.3/0.7 only).
    for tag in FSCAN_TAGS:
        j = read_json(RESULTS / f"{PREFIX}fscan_dsf_n0low_fagn{tag}.json")
        summary["fscan"]["dsf_n0low"][tag] = fscan_entry_from_json(j, RESULTS / f"{PREFIX}fscan_dsf_n0low_fagn{tag}.h5", FSCAN_TRUTH[tag])

    for tag in FSCAN_TAGS:
        j = read_json(ORIG_RESULTS / f"fscan_fagn{tag}.json")
        summary["fscan"]["conditional_broken"][tag] = fscan_entry_from_json(j, ORIG_RESULTS / f"fscan_fagn{tag}.h5", FSCAN_TRUTH[tag])

    # The #212-era run (working/gw_agn_darksirens_fixed), same inputs and grids:
    # the reference this run is checked against for code-drift.
    for tag in FSCAN_TAGS:
        j = read_json(PREV_RESULTS / f"{PREV_PREFIX}fscan_dscf_fagn{tag}.json")
        summary["fscan"]["prev_dscf"][tag] = fscan_entry_from_json(
            j, PREV_RESULTS / f"{PREV_PREFIX}fscan_dscf_fagn{tag}.h5", FSCAN_TRUTH[tag])
        j = read_json(PREV_RESULTS / f"{PREV_PREFIX}fscan_dsf_n0low_fagn{tag}.json")
        summary["fscan"]["prev_dsf_n0low"][tag] = fscan_entry_from_json(
            j, PREV_RESULTS / f"{PREV_PREFIX}fscan_dsf_n0low_fagn{tag}.h5", FSCAN_TRUTH[tag])

    j = read_json(RESULTS / f"{PREFIX}fscan_dscf_fagn0.3_injB.json")
    summary["fscan_extras"]["injB_0.3"] = fscan_entry_from_json(j, RESULTS / f"{PREFIX}fscan_dscf_fagn0.3_injB.h5", FSCAN_TRUTH["0.3"])

    j = read_json(RESULTS / f"{PREFIX}fscan_dsf_n0true_fagn0.3.json")
    summary["fscan_extras"]["n0true_demo_0.3"] = fscan_entry_from_json(j, RESULTS / f"{PREFIX}fscan_dsf_n0true_fagn0.3.h5", FSCAN_TRUTH["0.3"])

    summary["fscan_gwagn"] = {}
    for tag in FSCAN_TAGS:
        j = read_json(GW_AGN_REC / f"rec_fagn{tag}.json")
        if j is None or "alpha_agn" not in j:
            summary["fscan_gwagn"][tag] = None
            continue
        summary["fscan_gwagn"][tag] = {
            "truth_alpha": FSCAN_TRUTH[tag],
            "median": j["alpha_agn"]["median"],
            "ci68": j["alpha_agn"]["ci68"],
            "ci90": j["alpha_agn"]["ci90"],
            "map": j["map"]["alpha_agn"] if "map" in j else None,
        }


def build_guard_compliant_summary(summary):
    """Arm N: the mixture at master's DEFAULT budget, on the event subsamples the
    budget admits (N=50 / N=25). Records the admitted-cell fraction alongside the
    posterior summary, because a partially-rejected f grid is a TRUNCATED
    posterior — the median/CI are only meaningful together with that fraction."""
    summary["guard_compliant"] = {}
    specs = []
    for k, tr50 in (("0.3", 0.30), ("0.7", 0.70)):
        specs += [
            (f"dsf_n50_fagn{k}", f"N_fscan_dsf_n50_fagn{k}", tr50, 50, "default"),
            (f"dscf_n50_fagn{k}", f"N_fscan_dscf_n50_fagn{k}", tr50, 50, "default"),
            (f"dsf_n50_fagn{k}_legacybudget", f"NL_fscan_dsf_n50_fagn{k}", tr50, 50,
             "legacy-equivalent"),
        ]
    for k, tr25 in (("0.3", 0.32), ("0.7", 0.72)):
        specs.append((f"dsf_n25_fagn{k}", f"N_fscan_dsf_n25_fagn{k}", tr25, 25, "default"))

    for key, tag, truth, nobs, budget in specs:
        j = read_json(RESULTS / f"{tag}.json")
        if j is None:
            summary["guard_compliant"][key] = None
            continue
        entry = {"tag": tag, "truth_f": truth, "n_events": nobs,
                 "variance_budget": budget,
                 "n_evals": j.get("n_evals"),
                 "n_rejected_cells": j.get("n_neginf_cells"),
                 "all_cells_rejected": j.get("all_cells_rejected"),
                 "max_likelihood_variance_effective":
                     j.get("max_likelihood_variance_effective")}
        if j.get("n_evals"):
            entry["admitted_fraction"] = 1.0 - (j.get("n_neginf_cells", 0) / j["n_evals"])
        fb = j.get("f") or {}
        entry.update({"median": fb.get("median"), "ci68": fb.get("ci68"),
                      "ci90": fb.get("ci90"), "argmax": fb.get("argmax")})
        if entry["median"] is not None:
            entry["delta_vs_truth"] = entry["median"] - truth
        # Largest admitted f: where the guard truncates the grid.
        d = read_h5(RESULTS / f"{tag}.h5", ["f_grid", "log_likelihood"])
        if d is not None:
            f, ll = d["f_grid"], d["log_likelihood"]
            ok = np.isfinite(ll)
            entry["f_admitted_range"] = ([float(f[ok].min()), float(f[ok].max())]
                                         if ok.any() else None)
        summary["guard_compliant"][key] = entry

    # A/B: default vs legacy-equivalent budget on the SAME subsample.
    summary["guard_compliant_deltas"] = {}
    for k in ("0.3", "0.7"):
        a = summary["guard_compliant"].get(f"dsf_n50_fagn{k}")
        b = summary["guard_compliant"].get(f"dsf_n50_fagn{k}_legacybudget")
        if a and b and a.get("median") is not None and b.get("median") is not None:
            summary["guard_compliant_deltas"][f"fagn{k}"] = {
                "d_median_default_minus_legacy": a["median"] - b["median"],
                "default_admitted_fraction": a.get("admitted_fraction"),
                "legacy_admitted_fraction": b.get("admitted_fraction"),
            }


def build_h0_k2_summary(summary):
    summary["h0_k2"] = {}
    specs = [
        ("dscf_fagn0.3", RESULTS / f"{PREFIX}h0_dscf_k2_fagn0.3.h5", RESULTS / f"{PREFIX}h0_dscf_k2_fagn0.3.json", 0.307),
        ("dsf_fagn0.3", RESULTS / f"{PREFIX}h0_dsf_k2_fagn0.3.h5", RESULTS / f"{PREFIX}h0_dsf_k2_fagn0.3.json", 0.307),
    ]
    for key, h5p, jp, ffixed in specs:
        d = read_h5(h5p, ["H0_grid", "log_likelihood"])
        j = read_json(jp)
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
    summary["deltas"] = {"coverage": {}, "fscan_vs_truth": {}, "fscan_vs_gwagn": {}, "joint_vs_truth": {}}

    if ref is not None:
        for model in ("dscf", "dsf"):
            summary["deltas"]["coverage"][model] = {}
            for tracer in ("gal", "agn"):
                agg = summary.get("coverage_aggregate", {}).get(model, {}).get(tracer)
                refagg = ref.get(tracer, {})
                if not agg or not refagg:
                    continue
                summary["deltas"]["coverage"][model][tracer] = {
                    "delta_mean_median": (agg["mean_median"] - refagg.get("mean_median_r00_09", float("nan")))
                    if np.isfinite(agg["mean_median"]) else float("nan"),
                    "delta_mean_ci68_halfwidth": (agg["mean_ci68_halfwidth"] - refagg.get("mean_ci68_halfwidth", float("nan")))
                    if np.isfinite(agg["mean_ci68_halfwidth"]) else float("nan"),
                    "reference_mean_median": refagg.get("mean_median_r00_09"),
                    "reference_sem_median": refagg.get("sem_median"),
                    "reference_mean_ci68_halfwidth": refagg.get("mean_ci68_halfwidth"),
                }
    else:
        warn("gw_agn coverage reference json missing; coverage deltas not computed")

    for model_key in ("dscf", "dsf_n0low"):
        summary["deltas"]["fscan_vs_truth"][model_key] = {}
        summary["deltas"]["fscan_vs_gwagn"][model_key] = {}
        block = summary.get("fscan", {}).get(model_key, {})
        for tag, entry in block.items():
            if entry is None:
                continue
            med = entry.get("median")
            truth = entry.get("truth_f")
            gref = summary.get("fscan_gwagn", {}).get(tag)
            summary["deltas"]["fscan_vs_truth"][model_key][tag] = (med - truth) if (med is not None and truth is not None) else None
            if gref and gref.get("median") is not None and med is not None:
                summary["deltas"]["fscan_vs_gwagn"][model_key][tag] = med - gref["median"]
            else:
                summary["deltas"]["fscan_vs_gwagn"][model_key][tag] = None

    # Code-drift deltas: this run (master 2b86a2d, legacy-equivalent budget) minus
    # the #212-era run on identical inputs/grids/nuisance points. Anything nonzero
    # here is attributable to the 295 commits between the two, not to the guard.
    summary["deltas"]["fscan_vs_prev"] = {}
    for model_key, prev_key in (("dscf", "prev_dscf"), ("dsf_n0low", "prev_dsf_n0low")):
        summary["deltas"]["fscan_vs_prev"][model_key] = {}
        cur = summary.get("fscan", {}).get(model_key, {})
        prev = summary.get("fscan", {}).get(prev_key, {})
        for tag in FSCAN_TAGS:
            a, b = cur.get(tag), prev.get(tag)
            if not a or not b or a.get("median") is None or b.get("median") is None:
                summary["deltas"]["fscan_vs_prev"][model_key][tag] = None
                continue
            summary["deltas"]["fscan_vs_prev"][model_key][tag] = {
                "d_median": a["median"] - b["median"],
                "d_argmax": ((a.get("argmax_grid") - b.get("argmax_grid"))
                             if a.get("argmax_grid") is not None
                             and b.get("argmax_grid") is not None else None),
                "d_logL_range": ((a.get("range") - b.get("range"))
                                 if a.get("range") is not None
                                 and b.get("range") is not None else None),
                "this_median": a["median"], "prev_median": b["median"],
            }

    for model_key in ("dsf", "dscf", "conditional"):
        summary["deltas"]["joint_vs_truth"][model_key] = {}
        block = summary.get("joint", {}).get(model_key, {})
        for tag, entry in block.items():
            if entry is None or entry.get("MAP") is None:
                continue
            truth = entry.get("truth", {})
            dH0 = entry["MAP"]["H0"] - truth.get("H0", float("nan"))
            df = entry["MAP"]["f"] - truth.get("f", float("nan"))
            summary["deltas"]["joint_vs_truth"][model_key][tag] = {"dH0": dH0, "df": df}


# --------------------------------------------------------------------------- #
# Verdict table
# --------------------------------------------------------------------------- #
def print_verdict_table(summary):
    print("\n" + "=" * 96)
    print("VERDICT — four-truth ladder: truth / gw_agn / conditional(broken) / dsf / dscf medians")
    print("=" * 96)
    header = (f"{'truth':>7}{'gw_agn':>10}{'conditional':>13}{'dsf(n0low)':>12}"
              f"{'prev_dsf':>11}{'dscf':>10}{'prev_dscf':>11}")
    print(header)
    print("-" * len(header))
    for tag in FSCAN_TAGS:
        truth = FSCAN_TRUTH[tag]
        gref = summary.get("fscan_gwagn", {}).get(tag) or {}
        cond = summary.get("fscan", {}).get("conditional_broken", {}).get(tag) or {}
        dsf = summary.get("fscan", {}).get("dsf_n0low", {}).get(tag) or {}
        pdsf = summary.get("fscan", {}).get("prev_dsf_n0low", {}).get(tag) or {}
        dscf = summary.get("fscan", {}).get("dscf", {}).get(tag) or {}
        pdscf = summary.get("fscan", {}).get("prev_dscf", {}).get(tag) or {}
        print(f"{truth:>7.3f}{nice(gref.get('median')):>10}{nice(cond.get('median')):>13}"
              f"{nice(dsf.get('median')):>12}{nice(pdsf.get('median')):>11}"
              f"{nice(dscf.get('median')):>10}{nice(pdscf.get('median')):>11}")

    print("\n" + "=" * 96)
    print("VERDICT — joint MAPs vs truth (H0=%.2f)" % TRUTH_H0)
    print("=" * 96)
    header2 = f"{'set':<8}{'estimator':<14}{'MAP_H0':>9}{'MAP_f':>8}{'d_H0':>8}{'d_f':>8}{'rho':>9}"
    print(header2)
    print("-" * len(header2))
    for model_key, model_label in [("dsf", "dsf (fixed)"), ("dscf", "dscf (fixed)"), ("gw_agn", "gw_agn"), ("conditional", "conditional")]:
        for tag, e in summary.get("joint", {}).get(model_key, {}).items():
            if e is None:
                continue
            mp = e.get("MAP") or {}
            h0v, fv = mp.get("H0"), mp.get("f")
            truth = e.get("truth", {})
            dh0 = (h0v - truth.get("H0")) if h0v is not None else float("nan")
            df = (fv - truth.get("f")) if fv is not None else float("nan")
            print(f"fagn{tag:<4}{model_label:<14}{nice(h0v,2):>9}{nice(fv,3):>8}"
                  f"{nice(dh0,2):>8}{nice(df,3):>8}{nice(e.get('rho'),3):>9}")

    print("\n" + "=" * 96)
    print("VERDICT — K=2 H0 scans (fagn0.3, f fixed at truth), refined argmax vs truth")
    print("=" * 96)
    for key, block in summary.get("h0_k2", {}).items():
        if not block:
            print(f"  {key}: MISSING")
            continue
        d = block["argmax_refined"] - TRUTH_H0
        print(f"  {key:<16} argmax_refined={nice(block['argmax_refined'],2):<8} "
              f"median={nice(block['median'],2):<8} hw68={nice(block['ci68_halfwidth'],2):<7} "
              f"delta_vs_truth={nice(d,2)}")

    gc = summary.get("guard_compliant") or {}
    if any(v for v in gc.values()):
        print("\n" + "=" * 96)
        print("VERDICT — ARM N: the mixture at master's DEFAULT variance budget "
              "(guard-admissible event counts)")
        print("=" * 96)
        h = (f"{'config':<30}{'N':>4}{'budget':>19}{'admit%':>8}{'f_med':>8}"
             f"{'truth':>7}{'delta':>8}{'f_range_admitted':>20}")
        print(h)
        print("-" * len(h))
        for key, e in gc.items():
            if not e:
                print(f"{key:<30}   MISSING")
                continue
            af = e.get("admitted_fraction")
            fr = e.get("f_admitted_range")
            frs = f"[{fr[0]:.2f}, {fr[1]:.2f}]" if fr else "none"
            print(f"{key:<30}{e['n_events']:>4}{e['variance_budget']:>19}"
                  f"{(100 * af if af is not None else float('nan')):>8.0f}"
                  f"{nice(e.get('median')):>8}{e['truth_f']:>7.2f}"
                  f"{nice(e.get('delta_vs_truth')):>8}{frs:>20}")

    print("\n" + "=" * 96)
    print("VERDICT — per-tracer H0 coverage aggregates vs gw_agn reference")
    print("=" * 96)
    header3 = (f"{'model':<7}{'tracer':<7}{'n':>3}{'mean_median':>13}{'prev_med':>10}"
               f"{'ref_median':>12}{'delta':>9}{'mean_hw':>10}{'ref_hw':>9}{'delta_hw':>10}")
    print(header3)
    print("-" * len(header3))
    for model in ("dscf", "dsf"):
        for tracer in ("gal", "agn"):
            agg = summary.get("coverage_aggregate", {}).get(model, {}).get(tracer, {})
            pagg = (summary.get("coverage_aggregate_prev", {})
                    .get(model, {}).get(tracer, {}))
            delt = summary.get("deltas", {}).get("coverage", {}).get(model, {}).get(tracer, {})
            print(f"{model:<7}{tracer:<7}{agg.get('n_realizations', 0):>3}"
                  f"{nice(agg.get('mean_median')):>13}"
                  f"{nice(pagg.get('mean_median')):>10}"
                  f"{nice(delt.get('reference_mean_median')):>12}"
                  f"{nice(delt.get('delta_mean_median')):>9}"
                  f"{nice(agg.get('mean_ci68_halfwidth')):>10}"
                  f"{nice(delt.get('reference_mean_ci68_halfwidth')):>9}"
                  f"{nice(delt.get('delta_mean_ci68_halfwidth')):>10}")

    if WARNINGS:
        print("\n" + "=" * 96)
        print(f"WARNINGS ({len(WARNINGS)}) — see also results/comparison_summary.json['warnings']")
        print("=" * 96)
        for w in WARNINGS:
            print(f"  - {w}")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "truth": {"H0": TRUTH_H0, "fscan_truths": FSCAN_TRUTH},
        "darksirens_pin": {"sha": "2b86a2d", "date": "2026-07-29",
                           "note": "295 commits after the #212 integration merge (8eae3ea) "
                                   "that the 2026-07-10 run used"},
        "tag_prefix": PREFIX,
        "notes": [
            "THIS RUN (2026-07-29) reruns the campaign on darksirens master @ 2b86a2d. Arm L "
            "tags (prefix L_) set --max_likelihood_variance 1e6, which makes master's "
            "post-#212 total-variance criterion inert and collapses the guard to the legacy "
            "Neff > 5*N_obs floor — the only setting under which these scans are comparable "
            "with the #212-era run (see ../GUARD_AUDIT.md: at the default budget of 1.0 the "
            "N=1000 mixture configurations carry sigma^2_lnL ~ 14-48 and are rejected "
            "outright). prev_* series = the 2026-07-10 run in "
            "../../gw_agn_darksirens_fixed/results (c2_ tags), same inputs and grids.",
            "FIXED campaign: darksirens PRs #204-#212 (field-convention sky weighting + "
            "catalog-targeted injections) merged to master; this compares the fixed field "
            "modes (dsf = dark_sirens field at the complete-catalog n0 limit, log10n0=-12; "
            "dscf = dark_sirens_complete field) against gw_agn and the ORIGINAL campaign's "
            "broken isotropic-sky conditional K=2 mixture (railed to f=1 for any true "
            "AGN-hosted fraction >~0.1; see ../gw_agn_darksirens/{RECON,GATES,RESULTS}.md).",
            "argmax_refined = quadratic interpolation through the peak grid node and its "
            "immediate neighbors; falls back to the raw grid node at a boundary peak or "
            "degenerate fit.",
            "gw_agn rec_fagn*.h5 grids contain a huge-negative sentinel (~-1.4e11) or -inf "
            "where the field mixture is exactly zero; masked as NaN for display and for "
            "HPD/moment computations (values < grid_max - 1e6).",
            "dsf coverage uses only 5 realizations (r00-r04); dscf uses 10 (r00-r09) — "
            "aggregate SEMs are not directly comparable in n.",
        ],
    }

    ref = read_json(ORIG_RESULTS / "gw_agn_coverage_reference_r00_09.json")

    build_fscan_summary(summary)
    build_fig_recovery_ladder(summary)
    build_fig_fscans_fixed(summary)
    build_h0_k2_summary(summary)
    build_fig_joint_fixed(summary)
    build_fig_h0_pertracer_fixed(summary, ref or {})
    build_fig_variance_guard(summary)
    build_guard_compliant_summary(summary)
    build_deltas(summary, ref)

    # Fold in the total-variance guard audit (which configurations master admits
    # at its default GWTC-4.0/5.0 budget) — the headline new-code finding.
    ga = read_json(RESULTS / "guard_audit_summary.json")
    if ga is None:
        warn("guard_audit_summary.json missing; run run_guard_audit.sh")
    else:
        summary["guard_audit"] = ga

    summary["warnings"] = WARNINGS

    outpath = RESULTS / "comparison_summary.json"
    outpath.write_text(json.dumps(sanitize_nans(summary), indent=2))
    print(f"\nwrote {outpath}")

    print_verdict_table(summary)


if __name__ == "__main__":
    main()
