#!/usr/bin/env python3
"""Publication figures for experiment_h0f_baseline.

Reads only this experiment's own grids (results/*.h5) and writes:

  figs/fig_joint_h0f.{pdf,png}   joint (H0, f) credible regions + marginals
  figs/fig_f_recovery.{pdf,png}  f posteriors and recovered-vs-planted fraction
  results/summary.json           every number quoted in the figures
  results/table_h0f.tex          LaTeX table fragment of the same numbers

Pure post-processing: h5py/numpy/matplotlib, CPU, no darksirens import.

Conventions
-----------
* Flat prior on the scanned axes; posterior = exp(logL - max), normalised by the
  trapezoid rule. 1-D intervals are equal-tailed from the marginal CDF; 2-D
  regions are highest-posterior-density (the level whose interior holds the
  stated mass).
* A planted fraction ON the boundary (f = 1) cannot be covered by an equal-tailed
  interval, so the boundary case additionally reports the one-sided interval
  [F^-1(1 - C), 1]; both are recorded and the figure marks which is shown.
"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

BASE = Path(__file__).resolve().parent.parent
RESULTS = BASE / "results"
FIGS = BASE / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

H0_TRUTH = 67.74
SETS = ["0.0", "0.3", "0.7", "1.0"]
F_TRUTH = {"0.0": 0.00989, "0.3": 0.307, "0.7": 0.703, "1.0": 1.0}
JOINT_SETS = ["0.3", "0.7"]
# Refined grids around each peak (stage `jointzoom`). The wide 81x61 grid resolves
# the 90% region with only ~5-8 cells, which renders as a polygon; prefer the
# refined grid for the contour figure and fall back to the wide one if absent.
JOINT_TAG = {k: (f"jointzoom_fagn{k}", f"joint_fagn{k}") for k in JOINT_SETS}

# Categorical slots 1/3/4/6 of the validated reference palette, in fixed order;
# each series also carries a distinct dash pattern and marker, so identity never
# rests on colour alone.
BLUE, AQUA, YELLOW, GREEN = "#2a78d6", "#1baf7a", "#eda100", "#008300"
SET_COLOR = {"0.0": BLUE, "0.3": AQUA, "0.7": YELLOW, "1.0": GREEN}
SET_DASH = {"0.0": (0, (1, 1.4)), "0.3": (0, ()), "0.7": (0, (5, 1.8)),
            "1.0": (0, (3, 1.4, 1, 1.4))}
SET_MARKER = {"0.0": "o", "0.3": "s", "0.7": "D", "1.0": "^"}

INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDCOL = "#e1e0d9"
BASELINE = "#c3c2b7"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.labelsize": 9.5,
    "axes.titlesize": 9.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8,
    "axes.edgecolor": INK_SECONDARY,
    "axes.linewidth": 0.7,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK_SECONDARY,
    "ytick.color": INK_SECONDARY,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "grid.color": GRIDCOL,
    "grid.linewidth": 0.6,
    "legend.frameon": False,
    "lines.solid_capstyle": "round",
})


# --------------------------------------------------------------------------- #
# posterior helpers
# --------------------------------------------------------------------------- #
def load_grid(tag, keys):
    path = RESULTS / f"{tag}.h5"
    if not path.exists():
        raise SystemExit(f"[fatal] missing {path} — run run_experiment.sh first")
    with h5py.File(path, "r") as f:
        out = {k: f[k][:] for k in keys}
        out["attrs"] = dict(f.attrs)
    return out


def posterior_1d(x, logp):
    """Normalised flat-prior posterior, its median and equal-tailed intervals."""
    logp = np.asarray(logp, float)
    ok = np.isfinite(logp)
    p = np.zeros_like(logp)
    p[ok] = np.exp(logp[ok] - logp[ok].max())
    norm = np.trapz(p, x)
    p = p / norm
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (p[1:] + p[:-1]) * np.diff(x))])
    cdf /= cdf[-1]
    q = lambda u: float(np.interp(u, cdf, x))
    return {
        "grid": x, "pdf": p, "cdf": cdf,
        "median": q(0.5),
        "argmax": float(x[int(np.nanargmax(np.where(ok, logp, -np.inf)))]),
        "ci68": [q(0.16), q(0.84)],
        "ci90": [q(0.05), q(0.95)],
        "onesided68_lo": q(1 - 0.68),   # [·, x_max] one-sided, for a boundary truth
        "onesided90_lo": q(1 - 0.90),
    }


def hpd_levels_2d(X, Y, p, masses=(0.68, 0.90)):
    """Posterior-density levels enclosing the requested masses (HPD)."""
    dx = np.gradient(X)
    dy = np.gradient(Y)
    cell = np.outer(dx, dy)
    w = (p * cell).ravel()
    order = np.argsort(w)[::-1]
    csum = np.cumsum(w[order]) / w.sum()
    flat = p.ravel()[order]
    return [float(flat[min(int(np.searchsorted(csum, m)), flat.size - 1)])
            for m in masses]


def joint_posterior(tag_or_tags):
    """Accept a preferred tag plus fallbacks; use the first that exists on disk."""
    tags = (tag_or_tags,) if isinstance(tag_or_tags, str) else tuple(tag_or_tags)
    tag = next((t for t in tags if (RESULTS / f"{t}.h5").exists()), None)
    if tag is None:
        raise SystemExit(f"[fatal] none of {tags} found in {RESULTS}")
    if tag != tags[0]:
        print(f"[note] {tags[0]} absent; using {tag}")
    d = load_grid(tag, ["H0_grid", "f_grid", "log_likelihood"])
    H0, F, ll = d["H0_grid"], d["f_grid"], d["log_likelihood"]
    ok = np.isfinite(ll)
    p = np.zeros_like(ll)
    p[ok] = np.exp(ll[ok] - ll[ok].max())
    p /= np.trapz(np.trapz(p, F, axis=1), H0)
    i, j = np.unravel_index(int(np.nanargmax(np.where(ok, ll, -np.inf))), ll.shape)
    mH0 = posterior_1d(H0, np.log(np.maximum(np.trapz(p, F, axis=1), 1e-300)))
    mF = posterior_1d(F, np.log(np.maximum(np.trapz(p, H0, axis=0), 1e-300)))
    # correlation of the normalised 2-D posterior
    pH0, pF = np.trapz(p, F, axis=1), np.trapz(p, H0, axis=0)
    EH, EF = np.trapz(H0 * pH0, H0), np.trapz(F * pF, F)
    VH = np.trapz((H0 - EH) ** 2 * pH0, H0)
    VF = np.trapz((F - EF) ** 2 * pF, F)
    Hg, Fg = np.meshgrid(H0, F, indexing="ij")
    cov = np.trapz(np.trapz((Hg - EH) * (Fg - EF) * p, F, axis=1), H0)
    return {
        "H0": H0, "f": F, "p": p,
        "map": {"H0": float(H0[i]), "f": float(F[j])},
        "marg_H0": mH0, "marg_f": mF,
        "rho": float(cov / np.sqrt(VH * VF)),
        "n_rejected_cells": int((~ok).sum()), "n_cells": int(ll.size),
        "attrs": d["attrs"],
    }


def fmt(v, nd=3):
    return "—" if v is None or not np.isfinite(v) else f"{v:.{nd}f}"


# --------------------------------------------------------------------------- #
# Figure 1 — joint (H0, f)
# --------------------------------------------------------------------------- #
def fig_joint(jp):
    fig = plt.figure(figsize=(7.1, 4.05), dpi=300)
    # `bottom` leaves a dedicated band for the shared legend; at the previous value
    # the legend landed on top of the panels' x-axis labels.
    outer = GridSpec(1, 2, figure=fig, wspace=0.30, left=0.075, right=0.985,
                     bottom=0.195, top=0.935)

    for col, key in enumerate(JOINT_SETS):
        J = jp[key]
        gs = outer[col].subgridspec(2, 2, width_ratios=[4, 1.15],
                                    height_ratios=[1.15, 4], wspace=0.05, hspace=0.05)
        ax = fig.add_subplot(gs[1, 0])
        axt = fig.add_subplot(gs[0, 0], sharex=ax)
        axr = fig.add_subplot(gs[1, 1], sharey=ax)

        H0, F, p = J["H0"], J["f"], J["p"]
        l68, l90 = hpd_levels_2d(H0, F, p)
        col_main = SET_COLOR[key]

        ax.contourf(H0, F, p.T, levels=[l90, l68, p.max()],
                    colors=[col_main, col_main], alpha=0.20)
        ax.contour(H0, F, p.T, levels=[l90, l68], colors=col_main,
                   linewidths=[0.8, 1.3])
        ax.plot(J["map"]["H0"], J["map"]["f"], marker=SET_MARKER[key], ms=4.5,
                mfc=col_main, mec="white", mew=0.7, zorder=5)

        ftr = F_TRUTH[key]
        ax.axvline(H0_TRUTH, color=INK, lw=0.7, ls=(0, (2, 2)), alpha=0.75, zorder=4)
        ax.axhline(ftr, color=INK, lw=0.7, ls=(0, (2, 2)), alpha=0.75, zorder=4)
        ax.plot([H0_TRUTH], [ftr], marker="+", ms=9, mew=1.2, color=INK, zorder=6)

        # marginals
        axt.plot(H0, J["marg_H0"]["pdf"], color=col_main, lw=1.2)
        axt.fill_between(H0, J["marg_H0"]["pdf"], color=col_main, alpha=0.16, lw=0)
        axt.axvline(H0_TRUTH, color=INK, lw=0.7, ls=(0, (2, 2)), alpha=0.75)
        axr.plot(J["marg_f"]["pdf"], F, color=col_main, lw=1.2)
        axr.fill_betweenx(F, J["marg_f"]["pdf"], color=col_main, alpha=0.16, lw=0)
        axr.axhline(ftr, color=INK, lw=0.7, ls=(0, (2, 2)), alpha=0.75)

        # Frame the 2-D 90% region and the planted value with a common relative
        # margin, so the contour fills the panel instead of sitting in a corner. The
        # planted value is always inside the frame — the offset is the result, not
        # something to crop. Limits come from the CONTOUR's extent, not the 1-D
        # marginal CIs: the 2-D region reaches further than either marginal, and
        # sizing off the marginals clipped it.
        inside = p >= l90
        hi_idx = np.where(inside.any(axis=1))[0]
        fi_idx = np.where(inside.any(axis=0))[0]
        hlo, hhi = float(H0[hi_idx.min()]), float(H0[hi_idx.max()])
        flo, fhi = float(F[fi_idx.min()]), float(F[fi_idx.max()])
        h0lo, h0hi = min(hlo, H0_TRUTH), max(hhi, H0_TRUTH)
        f0lo, f0hi = min(flo, ftr), max(fhi, ftr)
        hpad = max(0.18 * (h0hi - h0lo), 0.25)
        fpad = max(0.18 * (f0hi - f0lo), 0.012)
        ax.set_xlim(h0lo - hpad, h0hi + hpad)
        ax.set_ylim(max(0.0, f0lo - fpad), min(1.0, f0hi + fpad))

        ax.set_xlabel(r"$H_0$  [km s$^{-1}$ Mpc$^{-1}$]")
        ax.set_ylabel(r"AGN-hosted fraction  $f_{\rm AGN}$")
        ax.grid(True, alpha=0.55)
        ax.set_axisbelow(True)
        # Hide the marginal panels' frames WITHOUT touching the shared locators:
        # axt/axr share x/y with ax, so set_xticks([]) on axt would strip the main
        # panel's tick labels too (it did). Blank the tick LABELS instead.
        for a in (axt, axr):
            for spine in a.spines.values():
                spine.set_visible(False)
            a.tick_params(left=False, right=False, top=False, bottom=False,
                          labelleft=False, labelright=False,
                          labeltop=False, labelbottom=False)
            a.grid(False)

        mH0, mF = J["marg_H0"], J["marg_f"]
        axt.set_title(
            rf"planted $f_{{\rm AGN}}={ftr:.3f}$" "\n"
            rf"$H_0={mH0['median']:.2f}^{{+{mH0['ci68'][1]-mH0['median']:.2f}}}"
            rf"_{{-{mH0['median']-mH0['ci68'][0]:.2f}}}$,  "
            rf"$f_{{\rm AGN}}={mF['median']:.3f}^{{+{mF['ci68'][1]-mF['median']:.3f}}}"
            rf"_{{-{mF['median']-mF['ci68'][0]:.3f}}}$" "\n"
            rf"$\rho={J['rho']:+.2f}$",
            fontsize=8, pad=4)

    handles = [
        Line2D([0], [0], color=INK, lw=0.7, ls=(0, (2, 2)), label="planted value"),
        Line2D([0], [0], color=INK_SECONDARY, lw=1.3, label="68% credible region"),
        Line2D([0], [0], color=INK_SECONDARY, lw=0.8, label="90% credible region"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, 0.012), fontsize=8)
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"fig_joint_h0f.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("wrote figs/fig_joint_h0f.{pdf,png}")


# --------------------------------------------------------------------------- #
# Figure 2 — f posteriors and recovery
# --------------------------------------------------------------------------- #
def fig_recovery(fp):
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.1, 3.1), dpi=300)

    # ---- left: the four marginal posteriors on f ----
    for key in SETS:
        P = fp[key]
        axL.plot(P["grid"], P["pdf"], color=SET_COLOR[key], lw=1.4,
                 ls=SET_DASH[key], zorder=3)
        axL.axvline(F_TRUTH[key], color=SET_COLOR[key], lw=0.7, ls=(0, (1, 2)),
                    alpha=0.85, zorder=2)
        # direct label at the peak
        i = int(np.argmax(P["pdf"]))
        axL.annotate(rf"$f_{{\rm AGN}}={F_TRUTH[key]:.3f}$",
                     xy=(P["grid"][i], P["pdf"][i]),
                     xytext=(0, 4), textcoords="offset points",
                     ha="center", va="bottom", fontsize=7.2,
                     color=SET_COLOR[key])
    axL.set_xlim(0, 1)
    axL.set_ylim(bottom=0)
    axL.set_xlabel(r"AGN-hosted fraction  $f_{\rm AGN}$")
    axL.set_ylabel(r"posterior density  $p(f_{\rm AGN}\,|\,d)$")
    axL.grid(True, alpha=0.55)
    axL.set_axisbelow(True)
    axL.set_title("Posteriors at the true expansion rate", fontsize=9)
    axL.plot([], [], color=INK_SECONDARY, lw=0.7, ls=(0, (1, 2)),
             label="planted value")
    axL.legend(loc="upper center", fontsize=7.5)

    # ---- right: recovered vs planted ----
    axR.plot([0, 1], [0, 1], color=BASELINE, lw=0.9, ls=(0, (4, 2.5)), zorder=1,
             label="perfect recovery")
    for key in SETS:
        P = fp[key]
        t, med = F_TRUTH[key], P["median"]
        boundary = t >= 1.0
        if boundary:            # equal-tailed cannot cover a boundary truth
            lo, hi = P["onesided68_lo"], 1.0
        else:
            lo, hi = P["ci68"]
        axR.errorbar([t], [med], yerr=[[med - lo], [hi - med]],
                     color=SET_COLOR[key], marker=SET_MARKER[key], ms=5.5,
                     mfc=SET_COLOR[key], mec="white", mew=0.7, lw=0,
                     elinewidth=1.1, capsize=2.4, zorder=4)
    axR.annotate("one-sided interval\n(planted value on boundary)",
                 xy=(1.0, fp["1.0"]["median"]), xytext=(0.60, 0.60),
                 fontsize=7, color=INK_SECONDARY, ha="left",
                 arrowprops=dict(arrowstyle="->", color=INK_MUTED, lw=0.7))
    axR.set_xlim(-0.04, 1.06)
    axR.set_ylim(-0.04, 1.06)
    axR.set_aspect("equal")
    axR.set_xlabel(r"planted $f_{\rm AGN}$")
    axR.set_ylabel(r"recovered $f_{\rm AGN}$  (median, 68%)")
    axR.grid(True, alpha=0.55)
    axR.set_axisbelow(True)
    axR.set_title("Recovery across the planted range", fontsize=9)
    axR.legend(loc="upper left", fontsize=7.5)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"fig_f_recovery.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("wrote figs/fig_f_recovery.{pdf,png}")


# --------------------------------------------------------------------------- #
def main():
    fp = {}
    for key in SETS:
        d = load_grid(f"fscan_fagn{key}", ["f_grid", "log_likelihood"])
        fp[key] = posterior_1d(d["f_grid"], d["log_likelihood"])
        fp[key]["n_rejected_cells"] = int((~np.isfinite(d["log_likelihood"])).sum())

    h0p = {}
    for key in JOINT_SETS:
        d = load_grid(f"h0scan_fagn{key}", ["H0_grid", "log_likelihood"])
        h0p[key] = posterior_1d(d["H0_grid"], d["log_likelihood"])
        h0p[key]["n_rejected_cells"] = int((~np.isfinite(d["log_likelihood"])).sum())

    jp = {key: joint_posterior(JOINT_TAG[key]) for key in JOINT_SETS}

    fig_joint(jp)
    fig_recovery(fp)

    # ---- numbers behind the figures ----
    strip = lambda P: {k: v for k, v in P.items()
                       if k not in ("grid", "pdf", "cdf")}
    summary = {
        "experiment": "experiment_h0f_baseline",
        "darksirens_sha": str(jp[JOINT_SETS[0]]["attrs"].get("darksirens_git_sha", "")),
        "joint_grid_used": {k: str(jp[k]["attrs"].get("arg_out_tag", "")) for k in JOINT_SETS},
        "guard": "historical N_eff > 5*N_obs (total-variance criterion inert, "
                 "max_likelihood_variance = 1e6)",
        "H0_truth": H0_TRUTH,
        "f_truth": F_TRUTH,
        "f_scan_at_true_H0": {k: strip(fp[k]) for k in SETS},
        "H0_scan_at_true_f": {k: strip(h0p[k]) for k in JOINT_SETS},
        "joint": {k: {"map": jp[k]["map"], "rho": jp[k]["rho"],
                      "marg_H0": strip(jp[k]["marg_H0"]),
                      "marg_f": strip(jp[k]["marg_f"]),
                      "n_rejected_cells": jp[k]["n_rejected_cells"],
                      "n_cells": jp[k]["n_cells"]} for k in JOINT_SETS},
    }
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print("wrote results/summary.json")

    rows = []
    for k in JOINT_SETS:
        J = jp[k]
        mH, mF = J["marg_H0"], J["marg_f"]
        rows.append(
            rf"{F_TRUTH[k]:.3f} & ${mH['median']:.2f}^{{+{mH['ci68'][1]-mH['median']:.2f}}}"
            rf"_{{-{mH['median']-mH['ci68'][0]:.2f}}}$ & "
            rf"${mF['median']:.3f}^{{+{mF['ci68'][1]-mF['median']:.3f}}}"
            rf"_{{-{mF['median']-mF['ci68'][0]:.3f}}}$ & ${J['rho']:+.2f}$ \\")
    tex = "\n".join([
        r"\begin{tabular}{cccc}", r"\hline",
        r"planted $f_{\rm AGN}$ & $H_0$ [km s$^{-1}$ Mpc$^{-1}$] & "
        r"$f_{\rm AGN}$ & $\rho$ \\", r"\hline", *rows, r"\hline",
        r"\end{tabular}"])
    (RESULTS / "table_h0f.tex").write_text(tex + "\n")
    print("wrote results/table_h0f.tex")

    print("\n=== joint (H0, f) ===")
    for k in JOINT_SETS:
        J = jp[k]
        print(f"  planted f={F_TRUTH[k]:.3f}: MAP=({J['map']['H0']:.2f}, "
              f"{J['map']['f']:.3f})  H0={fmt(J['marg_H0']['median'],2)} "
              f"{[round(x,2) for x in J['marg_H0']['ci68']]}  "
              f"f={fmt(J['marg_f']['median'])} "
              f"{[round(x,3) for x in J['marg_f']['ci68']]}  rho={J['rho']:+.3f}  "
              f"rejected {J['n_rejected_cells']}/{J['n_cells']}")
    print("=== H0 at true f ===")
    for k in JOINT_SETS:
        P = h0p[k]
        print(f"  planted f={F_TRUTH[k]:.3f}: H0={fmt(P['median'],2)} "
              f"{[round(x,2) for x in P['ci68']]}  truth in 68%: "
              f"{P['ci68'][0] <= H0_TRUTH <= P['ci68'][1]}")


if __name__ == "__main__":
    main()
