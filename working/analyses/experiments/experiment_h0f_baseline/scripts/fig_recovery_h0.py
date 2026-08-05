#!/usr/bin/env python3
"""Explicit H0-recovery figure for the planted-f ladder (dsf lane).

Reads ONLY this experiment's results grids:

  results/jointzoom_fagn{0.3,0.7}.h5   (H0, f) grid + log_likelihood
  results/joint_fagn{0.3,0.7}.h5       fallback if the zoom grid is absent

and writes:

  figs/fig_h0_recovery.{pdf,png}   the H0 MARGINAL posterior (joint grid
      marginalised over f, flat prior) per planted-f rung, one colour per rung,
      truth line, medians ticked — the f-dependent tilt read directly

Only the f = 0.307 and 0.703 rungs carry joint (H0, f) scans; the marginalised
figure therefore shows those two. A SECOND figure covers all four rungs on a
uniform definition — the H0 posterior at the PLANTED f (results/h0scan_fagn*,
the 0.0/1.0 pair added by scripts/run_h0scan_extra.sh) —

  figs/fig_h0_recovery_allrungs.{pdf,png}

whose four offsets {-3.51, -1.04, -3.50, +0.33} are NON-monotonic in f: the
pure-GAL rung's numerator has no interior peak so the selection slope dominates,
the pure-AGN rung's anchor curvature suppresses the same slope, and f = 0.307
sits near the accidental cancellation (see TILT_FINDINGS.md).

Pure post-processing: h5py/numpy/matplotlib, CPU, no darksirens import.
Posterior = exp(logL - max) normalised by the trapezoid rule; the H0 marginal
integrates the normalised 2-D posterior over f. Zoom-grid edge mass is checked
(< 1% of peak) so marginalising on the refined grid loses nothing.
"""
from __future__ import annotations

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
F_TRUTH = {"0.3": 0.307, "0.7": 0.703}
# Same rung colours / dashes as make_figures.py (colour follows the rung).
SET_COLOR = {"0.3": "#1baf7a", "0.7": "#eda100"}
SET_DASH = {"0.3": (0, ()), "0.7": (0, (5, 1.8))}
JOINT_TAG = {k: (f"jointzoom_fagn{k}", f"joint_fagn{k}") for k in F_TRUTH}

INK, INK2, INK3, GRIDCOL = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
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


def h0_marginal(tags):
    tag = next((t for t in tags if (RESULTS / f"{t}.h5").exists()), None)
    if tag is None:
        raise SystemExit(f"[fatal] none of {tags} found in {RESULTS}")
    with h5py.File(RESULTS / f"{tag}.h5", "r") as f:
        H0, F, ll = f["H0_grid"][:], f["f_grid"][:], f["log_likelihood"][:]
    ok = np.isfinite(ll)
    p = np.zeros_like(ll)
    p[ok] = np.exp(ll[ok] - ll[ok].max())
    p /= np.trapz(np.trapz(p, F, axis=1), H0)
    m = np.trapz(p, F, axis=1)                      # H0 marginal
    for edge in (m[0], m[-1]):                      # zoom grid must not clip
        if edge > 0.01 * m.max():
            raise SystemExit(f"[fatal] {tag}: H0 marginal not contained "
                             f"(edge/peak = {edge / m.max():.3f})")
    cdf = np.concatenate([[0.0],
                          np.cumsum(0.5 * (m[1:] + m[:-1]) * np.diff(H0))])
    cdf /= cdf[-1]
    q = lambda u: float(np.interp(u, cdf, H0))
    return {"tag": tag, "x": H0, "pdf": m, "median": q(0.5),
            "ci68": [q(0.16), q(0.84)]}


F_TRUTH_ALL = {"0.0": 0.00989, "0.3": 0.307, "0.7": 0.703, "1.0": 1.0}
ALL_COLOR = {"0.0": "#2a78d6", "0.3": "#1baf7a", "0.7": "#eda100",
             "1.0": "#c8407e"}
ALL_DASH = {"0.0": (0, (1.2, 1.4)), "0.3": (0, ()), "0.7": (0, (5, 1.8)),
            "1.0": (0, (4, 1.2, 1, 1.2))}


def h0_slice(tag):
    """1-D H0 scan at the planted f (uniform definition across all rungs)."""
    with h5py.File(RESULTS / f"{tag}.h5", "r") as f:
        H0, ll = f["H0_grid"][:], f["log_likelihood"][:]
    ok = np.isfinite(ll)
    m = np.zeros_like(ll)
    m[ok] = np.exp(ll[ok] - ll[ok].max())
    m /= np.trapz(m, H0)
    cdf = np.concatenate([[0.0],
                          np.cumsum(0.5 * (m[1:] + m[:-1]) * np.diff(H0))])
    cdf /= cdf[-1]
    q = lambda u: float(np.interp(u, cdf, H0))
    return {"tag": tag, "x": H0, "pdf": m, "median": q(0.5),
            "ci68": [q(0.16), q(0.84)]}


def fig_allrungs():
    P = {k: h0_slice(f"h0scan_fagn{k}") for k in F_TRUTH_ALL}

    fig, ax = plt.subplots(figsize=(5.4, 3.4), dpi=300)
    ax.axvline(H0_TRUTH, color=INK, lw=1.1, ls=(0, (2, 2)), zorder=4)
    ax.annotate("truth", xy=(H0_TRUTH, 1.0), xycoords=("data", "axes fraction"),
                xytext=(3, -2), textcoords="offset points", ha="left", va="top",
                fontsize=7.6, color=INK)

    ymax = max(p["pdf"].max() for p in P.values())
    for k, p in P.items():
        col = ALL_COLOR[k]
        ax.plot(p["x"], p["pdf"], color=col, lw=1.5, ls=ALL_DASH[k], zorder=3,
                label=(rf"$f={F_TRUTH_ALL[k]:.3g}$: "
                       rf"$H_0={p['median']:.2f}"
                       rf"^{{+{p['ci68'][1] - p['median']:.2f}}}"
                       rf"_{{-{p['median'] - p['ci68'][0]:.2f}}}$"))
        ax.fill_between(p["x"], p["pdf"], color=col, alpha=0.12, lw=0, zorder=2)
        ax.plot([p["median"], p["median"]], [0, 0.05 * ymax], color=col, lw=2.0,
                solid_capstyle="butt", zorder=5)

    lo = min(p["x"][p["pdf"] > 1e-3 * p["pdf"].max()].min() for p in P.values())
    hi = max(p["x"][p["pdf"] > 1e-3 * p["pdf"].max()].max() for p in P.values())
    ax.set_xlim(min(lo, H0_TRUTH) - 0.8, max(hi, H0_TRUTH) + 0.8)
    ax.set_ylim(0, 1.22 * ymax)
    ax.set_xlabel(r"$H_0$  [km s$^{-1}$ Mpc$^{-1}$]")
    ax.set_ylabel(r"posterior at planted $f$  $p(H_0\,|\,d, f_{\rm true})$")
    ax.set_title("$H_0$ recovery at the planted AGN fraction, all four rungs:\n"
                 "the tilt is non-monotonic in $f_{\\rm AGN}$  (ticks: medians)")
    ax.legend(loc="upper left", handlelength=2.4)
    ax.grid(True, alpha=0.55)
    ax.set_axisbelow(True)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"fig_h0_recovery_allrungs.{ext}",
                    bbox_inches="tight")
    plt.close(fig)
    print("wrote figs/fig_h0_recovery_allrungs.{pdf,png}")
    for k, p in P.items():
        print(f"  planted f={F_TRUTH_ALL[k]:.3g} [{p['tag']}]: "
              f"H0 = {p['median']:.3f} ci68 [{p['ci68'][0]:.3f}, "
              f"{p['ci68'][1]:.3f}]  offset {p['median'] - H0_TRUTH:+.3f}")


def main():
    P = {k: h0_marginal(JOINT_TAG[k]) for k in F_TRUTH}

    fig, ax = plt.subplots(figsize=(5.4, 3.4), dpi=300)
    ax.axvline(H0_TRUTH, color=INK, lw=1.1, ls=(0, (2, 2)), zorder=4)
    ax.annotate("truth", xy=(H0_TRUTH, 1.0), xycoords=("data", "axes fraction"),
                xytext=(3, -2), textcoords="offset points", ha="left", va="top",
                fontsize=7.6, color=INK)

    ymax = max(p["pdf"].max() for p in P.values())
    for k, p in P.items():
        col = SET_COLOR[k]
        ax.plot(p["x"], p["pdf"], color=col, lw=1.5, ls=SET_DASH[k], zorder=3)
        ax.fill_between(p["x"], p["pdf"], color=col, alpha=0.14, lw=0, zorder=2)
        ax.plot([p["median"], p["median"]], [0, 0.05 * ymax], color=col, lw=2.0,
                solid_capstyle="butt", zorder=5)
        i = int(np.argmax(p["pdf"]))
        ax.annotate(rf"planted $f_{{\rm AGN}}={F_TRUTH[k]:.3f}$" "\n"
                    rf"$H_0={p['median']:.2f}"
                    rf"^{{+{p['ci68'][1] - p['median']:.2f}}}"
                    rf"_{{-{p['median'] - p['ci68'][0]:.2f}}}$",
                    xy=(p["x"][i], p["pdf"][i]), xytext=(0, 5),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=7.6, color=col)

    lo = min(p["x"][p["pdf"] > 1e-3 * p["pdf"].max()].min() for p in P.values())
    hi = max(p["x"][p["pdf"] > 1e-3 * p["pdf"].max()].max() for p in P.values())
    ax.set_xlim(min(lo, H0_TRUTH) - 0.5, max(hi, H0_TRUTH) + 0.7)
    ax.set_ylim(0, 1.30 * ymax)
    ax.set_xlabel(r"$H_0$  [km s$^{-1}$ Mpc$^{-1}$]")
    ax.set_ylabel(r"marginal posterior  $p(H_0\,|\,d)$")
    ax.set_title("$H_0$ recovery vs planted AGN fraction ($f$ marginalised):\n"
                 "the tilt grows with $f_{\\rm AGN}$  (ticks: medians)")
    ax.grid(True, alpha=0.55)
    ax.set_axisbelow(True)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"fig_h0_recovery.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("wrote figs/fig_h0_recovery.{pdf,png}")

    for k, p in P.items():
        print(f"  planted f={F_TRUTH[k]:.3f} [{p['tag']}]: "
              f"H0 = {p['median']:.3f} ci68 [{p['ci68'][0]:.3f}, "
              f"{p['ci68'][1]:.3f}]  offset {p['median'] - H0_TRUTH:+.3f}")


if __name__ == "__main__":
    main()
    fig_allrungs()
