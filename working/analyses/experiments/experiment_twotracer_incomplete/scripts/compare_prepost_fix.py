#!/usr/bin/env python3
"""Pre/post-fix comparison of the completeness ladder (darksirens PR #335).

Reads results/summary.json (pre-fix events) and results/summary_fix.json
(sigma_ang-fixed events); writes

  figs/fig_ladder_prepost.{pdf,png} -- the paper's differential statistics,
      before and after the generator fix (dashed/open = pre, solid/filled = post)
  results/ladder_prepost_fix.json   -- every plotted number and the deltas

The question the figure answers: do the DIFFERENTIAL results (width degradation,
sigma(H0) non-monotonicity, peak-to-null separation) survive the fix, and by how
much do the absolute centres move?
"""
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent.parent
RESULTS, FIGS = BASE / "results", BASE / "figs"
LEVELS = ["complete", "m21.0", "m20.0", "m19.0", "m18.0"]
LABELS = {"complete": "complete", "m21.0": "$m<21$", "m20.0": "$m<20$",
          "m19.0": "$m<19$", "m18.0": "$m<18$"}

BLUE, AQUA, YELLOW, RED, INK, INK2 = ("#2a78d6", "#1baf7a", "#eda100",
                                      "#e34948", "#0b0b0b", "#52514e")
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "font.family": "serif",
    "font.serif": ["DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 9, "axes.labelsize": 9.5, "axes.titlesize": 9.2,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 7.4,
    "axes.edgecolor": INK2, "axes.linewidth": 0.7, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "grid.color": "#e1e0d9", "grid.linewidth": 0.6,
    "legend.frameon": False, "lines.solid_capstyle": "round",
})


def main():
    pre = json.loads((RESULTS / "summary.json").read_text())
    post = json.loads((RESULTS / "summary_fix.json").read_text())
    C = [pre["completeness"][l]["agn"]["completeness_within_horizon"] for l in LEVELS]

    def series(S, path):
        out = []
        for l in LEVELS:
            r = S["levels"][l]
            for k in path:
                r = r.get(k, {}) if isinstance(r, dict) else {}
            out.append(r if isinstance(r, (int, float)) else None)
        return out

    D = {"levels": LEVELS, "C_agn": C, "pre": {}, "post": {}}
    for name, S in (("pre", pre), ("post", post)):
        D[name] = {
            "f_median": series(S, ["fscan", "median"]),
            "f_hw": series(S, ["fscan", "half_width68"]),
            "H0_median": series(S, ["joint", "H0_median"]),
            "H0_hw": series(S, ["joint", "H0_half_width68"]),
            "deg_f": series(S, ["width_degradation_vs_complete", "fscan_f"]),
            "deg_H0": series(S, ["width_degradation_vs_complete", "joint_H0"]),
            "null_sep": series(S, ["sky_shuffle_null", "displacement_in_widths"]),
            "Neff": [S["levels"][l].get("Neff") for l in LEVELS],
        }

    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.15), dpi=300)
    axA, axB, axC = axes

    specs = [("pre", (0, (4, 1.8)), "none", "pre-fix"),
             ("post", "-", "full", "post-fix (PR #335)")]

    # (a) width degradation factors vs completeness
    for key, col, mk, lab in (("deg_f", BLUE, "o", r"$\sigma(f_{\rm AGN})$"),
                              ("deg_H0", RED, "s", r"$\sigma(H_0)$")):
        for arm, ls, fill, alab in specs:
            y = D[arm][key]
            pts = [(c, v) for c, v in zip(C, y) if v is not None]
            if not pts:
                continue
            axA.plot(*zip(*pts), color=col, lw=1.7, ls=ls, marker=mk, ms=4.4,
                     fillstyle=fill, mew=1.0, zorder=4,
                     label=f"{lab}, {alab}")
    axA.axhline(1.0, color=INK2, lw=0.9, ls=(0, (1, 2)), zorder=2)
    axA.set_xlim(1.03, 0.10)
    axA.set_xlabel(r"completeness within the horizon  $C(z\leq0.30)$")
    axA.set_ylabel("68% half-width / complete-rung value")
    axA.set_title("Width degradation along the ladder")
    axA.grid(True, alpha=0.55)
    axA.set_axisbelow(True)
    axA.legend(loc="upper left")

    # (b) absolute centres: f_AGN and H0 medians
    for arm, ls, fill, alab in specs:
        axB.plot(C, D[arm]["f_median"], color=BLUE, lw=1.7, ls=ls, marker="o",
                 ms=4.4, fillstyle=fill, mew=1.0, zorder=4,
                 label=f"$f_{{\\rm AGN}}$ median, {alab}")
    axB.axhline(0.30, color=INK2, lw=0.9, ls=(0, (1, 2)), zorder=2)
    axB.annotate("planted 0.30", xy=(0.16, 0.302), fontsize=7.2, color=INK2)
    axB.set_xlim(1.03, 0.10)
    axB.set_xlabel(r"completeness within the horizon  $C(z\leq0.30)$")
    axB.set_ylabel(r"$f_{\rm AGN}$ (f-scan median)")
    axB.set_title("Where the centre sits")
    axB.grid(True, alpha=0.55)
    axB.set_axisbelow(True)
    axB.legend(loc="upper left")

    # (c) peak-to-null separation, in widths
    for arm, ls, fill, alab in specs:
        pts = [(c, v) for c, v in zip(C, D[arm]["null_sep"]) if v is not None]
        if pts:
            axC.plot(*zip(*pts), color=AQUA, lw=1.7, ls=ls, marker="^", ms=5.0,
                     fillstyle=fill, mew=1.0, zorder=4, label=alab)
    axC.set_xlim(1.03, 0.10)
    axC.set_xlabel(r"completeness within the horizon  $C(z\leq0.30)$")
    axC.set_ylabel("peak $-$ null separation  [widths]")
    axC.set_title("Host-association information\n(vs sky-shuffled null)")
    axC.grid(True, alpha=0.55)
    axC.set_axisbelow(True)
    axC.legend(loc="lower left")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"fig_ladder_prepost.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("wrote figs/fig_ladder_prepost.{pdf,png}")

    # deltas
    D["deltas"] = {}
    for i, l in enumerate(LEVELS):
        d = {}
        for k in ("f_median", "f_hw", "H0_median", "H0_hw"):
            a, b = D["pre"][k][i], D["post"][k][i]
            if a is not None and b is not None:
                d[k] = {"pre": a, "post": b, "shift": b - a,
                        "ratio": (b / a if a else None)}
        D["deltas"][l] = d
    (RESULTS / "ladder_prepost_fix.json").write_text(
        json.dumps(D, indent=2, default=float))
    print("wrote results/ladder_prepost_fix.json")

    print(f"{'level':>9} {'C':>5} | f_med pre->post | hw pre->post (deg pre->post) "
          "| H0 pre->post | H0 hw pre->post | null sep pre->post")
    for i, l in enumerate(LEVELS):
        g = lambda a, k: (D[a][k][i] if D[a][k][i] is not None else float("nan"))
        print(f"{l:>9} {C[i]:5.2f} | {g('pre','f_median'):.3f}->{g('post','f_median'):.3f}"
              f" | {g('pre','f_hw'):.4f}->{g('post','f_hw'):.4f}"
              f" ({g('pre','deg_f'):.2f}->{g('post','deg_f'):.2f})"
              f" | {g('pre','H0_median'):6.2f}->{g('post','H0_median'):6.2f}"
              f" | {g('pre','H0_hw'):.3f}->{g('post','H0_hw'):.3f}"
              f" | {g('pre','null_sep'):5.2f}->{g('post','null_sep'):5.2f}")


if __name__ == "__main__":
    main()
