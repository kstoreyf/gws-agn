#!/usr/bin/env python
"""Analysis 14 figure from results/mc_accuracy_summary.json (deterministic; CPU, seconds).

    python make_figures.py      -> figs/mc_accuracy.{pdf,png}

Three panels against the number of posterior samples per event N_s:
  left    spread of the posterior median over sample redraws, in units of the mean 68% half-width
          (filled), with the injection-set spread at N_s = 8000 (open); 90% intervals on the sd;
          the muted guides fall as N_s^(-1/2) from each series' first point;
  middle  mean shift of the median relative to the largest N_s (paired redraws), 90% intervals,
          next to the first-order prediction from half the recorded variance (open diamonds);
  right   the recorded variance of ln L from the posterior samples at the reference cell.
Colours from working/analyses/common/figstyle.py; direct labels, no legend.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
A14 = HERE.parent
sys.path.insert(0, str(A14.parent / "common"))
import figstyle as fs  # noqa: E402

fs.use()
import matplotlib.pyplot as plt  # noqa: E402

SERIES = [("a", "H0", fs.C["blue"], r"all-AGN $H_0$ (seed 103)"),
          ("b", "H0", fs.C["orange"], r"mixed $H_0$"),
          ("b", "f", fs.C["aqua"], r"mixed $f_{\rm AGN}$")]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(A14 / "results"))
    ap.add_argument("--out", default=str(A14 / "figs" / "mc_accuracy"))
    args = ap.parse_args()
    S = json.loads((Path(args.results) / "mc_accuracy_summary.json").read_text())
    fig, axs = plt.subplots(1, 3, figsize=(fs.TWOCOL, 2.35), constrained_layout=True)
    ax1, ax2, ax3 = axs
    keys = list(S["configs"])
    for k, (cfg, p, col, lab) in enumerate(SERIES):
        # (a) runs off seed 100 since 2026-11-17 ("a_s103"); prefer that label over a seed-100 "a"
        cand = sorted((x for x in keys if x == cfg or x.startswith(cfg + "_s")), key=lambda x: x == cfg)
        blk = S["configs"].get(cand[0]) if cand else None
        if not blk or not blk["by_nsamp"]:
            continue
        ns = sorted(int(n) for n in blk["by_nsamp"])
        st = [blk["by_nsamp"][str(n)] for n in ns]
        x = np.array(ns, float) * (1.0 + 0.06 * (k - 1))
        y = np.array([s[p]["median_sd_over_hw68"] if s[p]["median_sd_over_hw68"] is not None else np.nan
                      for s in st])
        lo = np.array([s[p]["median_sd_over_hw68_ci90"][0] or np.nan for s in st])
        hi = np.array([s[p]["median_sd_over_hw68_ci90"][1] or np.nan for s in st])
        ax1.errorbar(x, y, yerr=[y - lo, hi - y], fmt="o", ms=3.5, color=col, lw=0.9, capsize=0, zorder=3)
        if np.isfinite(y[0]):
            g = np.geomspace(ns[0], ns[-1], 20)
            ax1.plot(g, y[0] * np.sqrt(ns[0] / g), color=col, lw=0.7, alpha=0.45, zorder=2)
        if np.isfinite(y[-1]):
            fs.label_line(ax1, x[-1], y[-1], lab, col, dx=6, dy=8 * (1 - k), size=6.5)
        inj = blk.get("injection_redraws")
        if inj and inj[p]["median_sd_over_hw68"] is not None:
            yi = inj[p]["median_sd_over_hw68"]
            ci = inj[p]["median_sd_over_hw68_ci90"]
            xi = 8000 * (1.18 + 0.05 * k)
            ax1.errorbar([xi], [yi], yerr=[[yi - ci[0]], [ci[1] - yi]], fmt="o", ms=3.5, mfc="white",
                         color=col, lw=0.9, capsize=0, zorder=3)
        # middle: shift relative to the largest N_s
        for n, s in zip(ns, st):
            sh = s[p].get("shift_vs_largest_nsamp")
            if not sh or sh["paired"] is None:
                continue
            hw = s[p]["hw68_mean"]
            xm = n * (1.0 + 0.08 * (k - 1))
            yv = sh["paired"] / hw
            ci = sh["paired_ci90"]
            err = None if ci[0] is None else [[yv - ci[0] / hw], [ci[1] / hw - yv]]
            ax2.errorbar([xm], [yv], yerr=err, fmt="o", ms=3.5, color=col, lw=0.9, capsize=0, zorder=3)
            ax2.plot([xm * 1.05], [sh["first_order_prediction_over_hw68"]], "D", ms=3.2, mfc="white",
                     color=col, zorder=3)
        # right: recorded variance of ln L (one series per configuration)
        if p == "H0":
            v = [s["lnL_variance_at_ref_cell"]["pe_var_mean"] for s in st]
            ax3.plot(ns, v, "o-", color=col, ms=3.5, lw=0.9)
            fs.label_line(ax3, ns[-1], v[-1], "all-AGN" if cfg == "a" else "mixed", col, dx=5, size=6.5)
    for ax in axs:
        ax.set_xscale("log")
        ax.set_xticks([2000, 8000, 32000])
        ax.set_xticklabels(["2000", "8000", "32000"])
        ax.set_xlim(1500, 75000)
        ax.set_xlabel(r"posterior samples per event $N_s$")
    ax1.set_ylabel(r"sd of median / 68% half-width")
    ax1.set_ylim(bottom=0)
    ax1.annotate("open: 6 injection sets", (0.03, 0.04), xycoords="axes fraction", fontsize=6.5,
                 color=fs.MUTED)
    ax2.axhline(0, color=fs.TRUTH, lw=0.8, ls=(0, (3, 2)), alpha=0.75)
    ax2.set_ylabel("median shift / 68% half-width")
    ax2.set_title("relative to the largest $N_s$", fontsize=7, color=fs.MUTED)
    ax2.annotate("open: first-order prediction", (0.03, 0.04), xycoords="axes fraction", fontsize=6.5,
                 color=fs.MUTED)
    ax3.set_yscale("log")
    ax3.set_ylabel(r"Var[ln $\hat L$] from samples")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"), dpi=300)
    print(f"wrote {out}.pdf + .png")


if __name__ == "__main__":
    main()
