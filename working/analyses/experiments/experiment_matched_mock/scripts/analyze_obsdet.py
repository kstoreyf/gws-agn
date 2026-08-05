#!/usr/bin/env python3
"""Does making the selection act on the observed data close the H0 bias?

Aggregates the paired closure scans written by ``run_obsdet_scans.sh``:

* ``ctrl`` -- gmd's current rule.  Detection is decided by an independent
  ``Beta(2,5)**0.5`` projection latent; the PE then conditions on a separate
  noise draw.
* ``obs``  -- the fix.  One observation per source, tested against the threshold
  and then handed to the PE unchanged, so detection is a deterministic function
  of the data the posterior conditions on.

Both arms share the five catalog realisations of the published baseline, the
event seeds, and every ancillary uncertainty model, so the arms are PAIRED BY
CATALOG -- which matters because the catalog realisation is the dominant
variance term here (seed-to-seed sd 1.09 vs a per-seed 68% half-width of 0.49).
The paired difference is therefore the statistic with the smallest error bar,
and it is reported alongside the two absolute means.

Writes ``results/obsdet_summary.json`` and ``figs/fig_obsdet_closure.{pdf,png}``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent.parent
RESULTS, FIGS = BASE / "results", BASE / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

H0_TRUE = 67.74
# The five catalog realisations of the published baseline, plus fifteen fresh
# ones -- the catalog realisation, not the events, sets the error bar here.
TAGS = (("b", "s4102", "s4103", "s4104", "s4105")
        + tuple(f"n{s}" for s in range(4201, 4216)))
ARMS = (("ctrl", "detection on true parameters (current)"),
        ("obs", "detection on the observed data (fix)"))
# Published five-seed baseline from gmd's own generator, same estimator.
BASELINE = {"offsets": [-1.349, -1.539, 0.033, -2.637, -2.556],
            "source": "gmd generate_mock_data, pe_centering=observed, sigma_dL=0.10"}

BLUE, AQUA, YELLOW, RED, INK, INK2, INK3 = ("#2a78d6", "#1baf7a", "#eda100",
                                            "#e34948", "#0b0b0b", "#52514e", "#898781")
GRIDCOL = "#e1e0d9"
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


def load(arm):
    rows = []
    for tag in TAGS:
        p = RESULTS / f"obsdet_{arm}_{tag}.json"
        if not p.exists():
            raise SystemExit(f"missing {p}")
        d = json.loads(p.read_text())
        h = d["H0"]
        rows.append({
            "tag": tag, "median": h["median"], "offset": h["median"] - H0_TRUE,
            "ci68": h["ci68"], "half_width": 0.5 * (h["ci68"][1] - h["ci68"][0]),
            "truth_in_ci68": h["truth_in_ci68"], "truth_in_ci90": h["truth_in_ci90"],
            "n_rejected": d["n_neginf_cells"], "n_evals": d["n_evals"],
        })
    return rows


def stats(x):
    x = np.asarray(x, dtype=float)
    n = x.size
    sd = float(x.std(ddof=1))
    sem = sd / np.sqrt(n)
    return {"n": int(n), "mean": float(x.mean()), "sd": sd, "sem": float(sem),
            "sigma_from_zero": float(abs(x.mean()) / sem) if sem > 0 else None}


def main():
    S = {"H0_true": H0_TRUE, "arms": {}, "baseline_published": BASELINE}
    for arm, _ in ARMS:
        rows = load(arm)
        S["arms"][arm] = {"per_seed": rows,
                          "offset_stats": stats([r["offset"] for r in rows]),
                          "mean_half_width": float(np.mean([r["half_width"] for r in rows])),
                          "n_rejected_total": int(sum(r["n_rejected"] for r in rows))}
    S["baseline_published"]["offset_stats"] = stats(BASELINE["offsets"])

    ctrl = np.array([r["offset"] for r in S["arms"]["ctrl"]["per_seed"]])
    obs = np.array([r["offset"] for r in S["arms"]["obs"]["per_seed"]])
    diff = obs - ctrl
    S["paired_difference_obs_minus_ctrl"] = stats(diff)
    S["paired_difference_obs_minus_ctrl"]["per_seed"] = dict(zip(TAGS, diff.round(4).tolist()))

    # Fraction of the control's bias removed, with the paired error propagated.
    m_ctrl = float(ctrl.mean())
    S["fraction_of_control_bias_removed"] = (
        float(-diff.mean() / m_ctrl) if m_ctrl != 0 else None)

    (RESULTS / "obsdet_summary.json").write_text(json.dumps(S, indent=2, default=float))
    print("wrote results/obsdet_summary.json\n")
    for arm, lab in ARMS:
        st = S["arms"][arm]["offset_stats"]
        offs = [f"{r['offset']:+.3f}" for r in S["arms"][arm]["per_seed"]]
        print(f"{arm:5s} {lab}")
        print(f"      per-seed {offs}")
        print(f"      mean {st['mean']:+.3f} +- {st['sem']:.3f} (sd {st['sd']:.3f})  "
              f"=> {st['sigma_from_zero']:.1f} sigma from zero")
    bs = S["baseline_published"]["offset_stats"]
    print(f"published baseline  mean {bs['mean']:+.3f} +- {bs['sem']:.3f}")
    pd_ = S["paired_difference_obs_minus_ctrl"]
    print(f"\npaired difference (obs - ctrl): {pd_['mean']:+.3f} +- {pd_['sem']:.3f}  "
          f"=> {pd_['sigma_from_zero']:.1f} sigma")

    # ------------------------------------------------------------------ figure
    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(7.6, 3.3), dpi=300,
        gridspec_kw={"width_ratios": [2.6, 1.0], "wspace": 0.05}, sharey=True)
    # sort realisations by the control offset so the pairing is legible
    order = np.argsort(ctrl)
    x = np.arange(len(TAGS))
    for arm, col, off, lab, mark in (
            ("ctrl", RED, -0.17, "detection on true parameters (current)", "o"),
            ("obs", BLUE, +0.17, "detection on the observed data (fix)", "s")):
        rows = [S["arms"][arm]["per_seed"][i] for i in order]
        y = [r["offset"] for r in rows]
        e = [r["half_width"] for r in rows]
        axL.errorbar(x + off, y, yerr=e, fmt=mark, ms=3.4, lw=0, elinewidth=0.8,
                     capsize=1.8, color=col, ecolor=col, alpha=0.85, zorder=4,
                     label=lab)
    axL.axhline(0.0, color=INK2, lw=0.9, zorder=3)
    axL.set_xticks([])
    axL.set_xlim(-0.8, len(TAGS) - 0.2)
    axL.set_xlabel(f"{len(TAGS)} catalog realisations  (sorted by the control offset)")
    axL.set_ylabel(r"$H_0$ offset  [km s$^{-1}$ Mpc$^{-1}$]")
    axL.set_title("Does selecting on the observed data close the bias?")
    axL.grid(True, alpha=0.55, axis="y")
    axL.set_axisbelow(True)
    axL.legend(loc="lower right")

    # right panel: the means, which is where the answer actually lives
    for i, (arm, col, lab) in enumerate((("ctrl", RED, "current"),
                                         ("obs", BLUE, "fix"))):
        st = S["arms"][arm]["offset_stats"]
        axR.errorbar([i], [st["mean"]], yerr=[st["sem"]], fmt="o", ms=6,
                     lw=0, elinewidth=1.8, capsize=4, color=col, ecolor=col,
                     zorder=4)
        axR.annotate(f"{st['mean']:+.2f}\n$\\pm${st['sem']:.2f}",
                     xy=(i, st["mean"]), xytext=(i + 0.16, st["mean"]),
                     fontsize=7.4, color=col, va="center", ha="left")
    axR.axhline(0.0, color=INK2, lw=0.9, zorder=3)
    axR.set_xticks([0, 1])
    axR.set_xticklabels(["current", "fix"])
    axR.set_xlim(-0.55, 1.75)
    axR.set_title("mean over realisations")
    axR.grid(True, alpha=0.55, axis="y")
    axR.set_axisbelow(True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"fig_obsdet_closure.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("\nwrote figs/fig_obsdet_closure.{pdf,png}")


if __name__ == "__main__":
    main()
