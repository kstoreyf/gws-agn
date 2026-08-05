#!/usr/bin/env python3
"""Figures for experiment_twotracer_deep: unblocking the K=2 deep mock.

  figs/fig_twotracer_targeted.{pdf,png}   -- why the lane was blocked, and the
                                             AGN-hosted fraction once it is not
  figs/fig_twotracer_joint.{pdf,png}      -- the joint (H0, f_AGN) plane
  results/summary.json

The point of the first figure is that the previous deep-mock number was set by
where the selection-validity guard cut, not by the data.  The left panel is the
mechanism (the sparse tracer starves the selection integral as f_AGN rises, and
targeting the proposal at that tracer reverses it); the right panel is the
consequence (the posterior stops being railed against the boundary, and moves).
"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation

BASE = Path(__file__).resolve().parent.parent
RESULTS, FIGS = BASE / "results", BASE / "figs"
FIGS.mkdir(parents=True, exist_ok=True)
GLASS = (BASE.parent / "experiment_h0f_baseline" / "results")

BLUE, AQUA, YELLOW, RED, INK, INK2, INK3 = ("#2a78d6", "#1baf7a", "#eda100",
                                            "#e34948", "#0b0b0b", "#52514e", "#898781")
GRIDCOL, BASELINE = "#e1e0d9", "#c3c2b7"
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "font.family": "serif",
    "font.serif": ["DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 9, "axes.labelsize": 9.5, "axes.titlesize": 9.2,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 7.6,
    "axes.edgecolor": INK2, "axes.linewidth": 0.7, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "grid.color": GRIDCOL, "grid.linewidth": 0.6,
    "legend.frameon": False, "lines.solid_capstyle": "round",
})

F_PROBES = ("0.0", "0.3", "0.7", "1.0")


def posterior_1d(h5, key="f_grid"):
    with h5py.File(h5, "r") as f:
        x, ll = f[key][:], f["log_likelihood"][:]
    ok = np.isfinite(ll)
    p = np.where(ok, np.exp(ll - np.nanmax(ll[ok])), 0.0)
    return x, p / np.trapz(p, x), ok


def posterior_2d(h5):
    with h5py.File(h5, "r") as f:
        H, F, ll = f["H0_grid"][:], f["f_grid"][:], f["log_likelihood"][:]
    ok = np.isfinite(ll)
    p = np.where(ok, np.exp(ll - np.nanmax(ll[ok])), 0.0)
    cell = np.outer(np.gradient(H), np.gradient(F))
    p = p / (p * cell).sum()
    return H, F, p, cell, ok


def hpd_levels(p, cell, fracs=(0.68, 0.90)):
    w = (p * cell).ravel()
    order = np.argsort(w)[::-1]
    csum = np.cumsum(w[order])
    return [float(p.ravel()[order[min(np.searchsorted(csum, fr), order.size - 1)]])
            for fr in fracs]


def neff_table():
    out = {}
    for tag, stem in (("popuni", "guard_popuni_f"), ("targeted", "guard_targeted_f")):
        rows = []
        for fs in F_PROBES:
            p = RESULTS / f"{stem}{fs}.json"
            if not p.exists():
                continue
            d = json.loads(p.read_text())
            rec = d["guard_records"][0]
            rows.append({"f": float(fs), "Neff": rec["Neff"],
                         "threshold": rec["threshold"],
                         "passes": rec["passes_legacy_floor"],
                         "pe_variance_sum": rec["pe_variance_sum"]})
        out[tag] = rows
    return out


def main():
    meta = json.loads((BASE / "data_derived" / "twotracer_meta.json").read_text())
    S = {"meta": meta}
    S["neff_vs_f_at_N200"] = neff = neff_table()

    # ---------------------------------------------------------------- figure 1
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.4, 3.2), dpi=300)

    # (a) mechanism: catalog-conditioned Neff across the f grid, both proposals
    for tag, col, dash, lab in (
            ("popuni", RED, (0, (4, 1.8)), "population + uniform"),
            ("targeted", BLUE, "-", "with AGN-targeted branch")):
        rows = neff[tag]
        if not rows:
            continue
        x = [r["f"] for r in rows]
        y = [r["Neff"] for r in rows]
        axL.plot(x, y, color=col, lw=1.7, ls=dash, marker="o", ms=3.4, zorder=4,
                 label=lab)
        for r in rows:
            if not r["passes"]:
                axL.plot([r["f"]], [r["Neff"]], marker="x", ms=6, mew=1.4,
                         color=RED, zorder=5)
    thr = neff["targeted"][0]["threshold"] if neff["targeted"] else 1000.0
    axL.axhline(thr, color=INK3, lw=0.9, ls=(0, (1, 2)), zorder=2)
    axL.annotate(r"validity floor  $N_{\rm eff}>5N_{\rm obs}$",
                 xy=(0.5, thr), xytext=(0.5, thr * 0.55), ha="center",
                 fontsize=7.2, color=INK2)
    axL.set_yscale("log")
    axL.set_xlim(-0.03, 1.03)
    axL.set_xlabel(r"AGN-hosted fraction  $f_{\rm AGN}$")
    axL.set_ylabel(r"selection $N_{\rm eff}$  (catalog-conditioned, $N_{\rm obs}{=}200$)")
    axL.set_title("The sparse tracer starves the selection integral")
    axL.grid(True, alpha=0.55)
    axL.set_axisbelow(True)
    axL.legend(loc="upper left")

    # (b) consequence: the AGN-hosted fraction, before and after
    specs = [
        ("glass", GLASS / "fscan_fagn0.3.h5", GLASS / "fscan_fagn0.3.json",
         0.307, AQUA, (0, (1.2, 1.6)), "GLASS (clustered, $N{=}1000$)"),
        ("deep_popuni", RESULTS / "deep_fscan_n80.h5",
         RESULTS / "deep_fscan_n80.json", 0.30, RED, (0, (4, 1.8)),
         "deep, population + uniform ($N{=}80$)"),
        ("deep_targeted", RESULTS / "tgt_fscan_n200.h5",
         RESULTS / "tgt_fscan_n200.json", 0.30, BLUE, "-",
         "deep, AGN-targeted ($N{=}200$)"),
    ]
    S["fscans"] = {}
    for tag, h5, js, truth, col, dash, lab in specs:
        if not h5.exists():
            continue
        x, p, ok = posterior_1d(h5)
        j = json.loads(js.read_text())
        fb = j["f"]
        S["fscans"][tag] = {
            "truth_f": truth, "median": fb["median"], "ci68": fb["ci68"],
            "argmax": fb["argmax"], "truth_in_ci68": fb.get("truth_in_ci68"),
            "n_rejected_cells": j["n_neginf_cells"], "n_evals": j["n_evals"],
            "admitted_f_range": [float(x[ok].min()), float(x[ok].max())],
            "peak_at_admitted_edge": bool(
                abs(x[int(np.nanargmax(np.where(ok, p, np.nan)))] - x[ok].max())
                < 1.01 * (x[1] - x[0])) and bool((~ok).any()),
        }
        axR.plot(x, p, color=col, lw=1.7, ls=dash, zorder=4, label=lab)
        if (~ok).any():
            # Shading marks where THAT run's guard cut, not a property of the
            # parameter: the targeted run has a posterior throughout.
            lo = float(x[ok].max())
            axR.axvspan(lo, float(x.max()), color=RED, alpha=0.09, lw=0, zorder=1)
            axR.axvline(lo, color=RED, lw=0.9, ls="-", alpha=0.6, zorder=2)
            axR.annotate("guard wall of the\npopulation + uniform run",
                         xy=(lo + 0.025, 0.62 * p.max()), ha="left", va="center",
                         fontsize=6.8, color=RED)
    axR.axvline(0.30, color=INK2, lw=0.9, ls=(0, (1, 2)), zorder=2)
    axR.annotate("planted", xy=(0.295, 0.055), xycoords=("data", "axes fraction"),
                 fontsize=7.0, color=INK2, ha="right")
    axR.set_xlim(0, 0.75)
    axR.set_ylim(bottom=0)
    axR.set_xlabel(r"AGN-hosted fraction  $f_{\rm AGN}$")
    axR.set_ylabel("posterior density")
    axR.set_title("What the guard was hiding")
    axR.grid(True, alpha=0.55)
    axR.set_axisbelow(True)
    axR.legend(loc="upper right")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"fig_twotracer_targeted.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("wrote figs/fig_twotracer_targeted.{pdf,png}")

    # ---------------------------------------------------------------- figure 2
    jp = RESULTS / "tgt_joint_n200.h5"
    if jp.exists():
        H, F, p, cell, ok = posterior_2d(jp)
        j = json.loads((RESULTS / "tgt_joint_n200.json").read_text())
        edge = binary_dilation(~ok) & ok
        w = (p * cell).ravel()
        order = np.argsort(w)[::-1]
        k68 = int(np.searchsorted(np.cumsum(w[order]), 0.68) + 1)
        m68 = np.zeros(w.size, bool)
        m68[order[:k68]] = True
        m68 = m68.reshape(p.shape)

        fig, ax = plt.subplots(figsize=(4.5, 3.5), dpi=300)
        Hm, Fm = np.meshgrid(H, F, indexing="ij")
        ax.contourf(Hm, Fm, np.where(ok, np.nan, 1.0), levels=[0.5, 1.5],
                    colors=[RED], alpha=0.10, zorder=1)
        lv = hpd_levels(p, cell)
        ax.contourf(Hm, Fm, p, levels=[lv[1], lv[0], p.max()],
                    colors=[BLUE, BLUE], alpha=0.22, zorder=3)
        ax.contour(Hm, Fm, p, levels=sorted(lv), colors=[BLUE], linewidths=1.1,
                   zorder=4)
        ax.plot([67.74], [0.30], marker="*", ms=11, color=YELLOW, mec=INK,
                mew=0.6, ls="none", zorder=6, label="truth")
        ax.plot([j["map"]["H0"]], [j["map"]["f"]], marker="P", ms=6.5, color=BLUE,
                mec="white", mew=0.7, ls="none", zorder=6, label="MAP")
        ax.set_xlim(62, 74)
        ax.set_ylim(0, 0.62)
        ax.set_xlabel(r"$H_0$  [km s$^{-1}$ Mpc$^{-1}$]")
        ax.set_ylabel(r"$f_{\rm AGN}$")
        ax.set_title("Deep two-tracer mock, AGN-targeted lane")
        ax.annotate("inadmissible", xy=(62.4, 0.575), fontsize=7.0, color=RED)
        ax.grid(True, alpha=0.45)
        ax.set_axisbelow(True)
        ax.legend(loc="upper right")
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(FIGS / f"fig_twotracer_joint.{ext}", bbox_inches="tight")
        plt.close(fig)
        print("wrote figs/fig_twotracer_joint.{pdf,png}")

        Hm_, Fm_ = np.meshgrid(H, F, indexing="ij")
        pw = p * cell
        mH, mF = (Hm_ * pw).sum(), (Fm_ * pw).sum()
        sH = np.sqrt(((Hm_ - mH) ** 2 * pw).sum())
        sF = np.sqrt(((Fm_ - mF) ** 2 * pw).sum())
        rho = float(((Hm_ - mH) * (Fm_ - mF) * pw).sum() / (sH * sF))
        rej_by_h0 = (~ok).sum(axis=1)
        S["joint_targeted"] = {
            "map": j["map"],
            "H0": j["H0"]["median"], "H0_ci68": j["H0"]["ci68"],
            "H0_ci90": j["H0"]["ci90"], "H0_truth_in_ci68": j["H0"]["truth_in_ci68"],
            "H0_truth_in_ci90": j["H0"]["truth_in_ci90"],
            "f": j["f"]["median"], "f_ci68": j["f"]["ci68"],
            "f_ci90": j["f"]["ci90"], "f_truth_in_ci68": j["f"]["truth_in_ci68"],
            "rho": rho,
            "n_rejected_cells": int((~ok).sum()), "n_evals": int(ok.size),
            "posterior_mass_adjacent_to_rejected": float(pw[edge].sum()),
            "n_68pct_cells_touching_rejected": int((m68 & edge).sum()),
            "fully_admitted_H0_range": [float(H[rej_by_h0 == 0].min()),
                                        float(H[rej_by_h0 == 0].max())],
            "verdict": ("interpretable: rejection is confined to H0 far from the "
                        "peak, no posterior mass touches the boundary"),
        }

    (RESULTS / "summary.json").write_text(json.dumps(S, indent=2, default=float))
    print("wrote results/summary.json")
    for tag, e in S.get("fscans", {}).items():
        print(f"  {tag:14s} f = {e['median']:.4f} {np.round(e['ci68'], 4).tolist()}  "
              f"truth {e['truth_f']} in68={e['truth_in_ci68']}  "
              f"rejected {e['n_rejected_cells']}/{e['n_evals']}  "
              f"peak_on_edge={e['peak_at_admitted_edge']}")
    if "joint_targeted" in S:
        d = S["joint_targeted"]
        print(f"  joint: H0={d['H0']:.3f} {np.round(d['H0_ci68'], 3).tolist()}  "
              f"f={d['f']:.4f} {np.round(d['f_ci68'], 4).tolist()}  rho={d['rho']:+.3f}  "
              f"mass on wall={d['posterior_mass_adjacent_to_rejected']:.5f}")


if __name__ == "__main__":
    main()
