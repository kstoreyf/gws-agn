#!/usr/bin/env python3
"""Pre/post-fix comparison of the n0-knowledge x completeness significance table.

Reads results/n0_arms_summary.json (pre-fix events) and
results/n0_arms_summary_fix.json (sigma_ang-fixed events, darksirens PR #335);
writes

  figs/fig_n0_sig_prepost.{pdf,png} -- detection significance (median/sigma) of
      f_AGN per n0-knowledge arm along the completeness ladder, pre vs post
  results/n0_prepost_fix.json       -- both tables and their deltas
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent.parent
RESULTS, FIGS = BASE / "results", BASE / "figs"
FIGS.mkdir(exist_ok=True)
LEVELS = ["complete", "m21.0", "m20.0", "m19.0", "m18.0"]
LABELS = {"complete": "1.00", "m21.0": "0.94", "m20.0": "0.76",
          "m19.0": "0.38", "m18.0": "0.18"}
ARMS = ["fixed", "10%", "30%", "factor 2", "free"]
INK, INK2 = "#0b0b0b", "#52514e"
COLORS = {"fixed": "#0b0b0b", "10%": "#2a78d6", "30%": "#1baf7a",
          "factor 2": "#eda100", "free": "#e34948"}
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


def table(S, field="detection_sigma"):
    out = {}
    for lev in LEVELS:
        r = S["levels"].get(lev, {}).get("arms", {})
        out[lev] = {a: (r[a][field] if r.get(a) else None) for a in ARMS}
    return out


def main():
    pre = json.loads((RESULTS / "n0_arms_summary.json").read_text())
    post = json.loads((RESULTS / "n0_arms_summary_fix.json").read_text())
    tp, tq = table(pre), table(post)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.6, 3.2), dpi=300, sharey=True)
    x = range(len(LEVELS))
    for ax, T, ttl in ((axL, tp, "pre-fix"), (axR, tq, "post-fix (PR #335)")):
        for arm in ARMS:
            y = [T[l][arm] for l in LEVELS]
            pts = [(i, v) for i, v in zip(x, y) if v is not None]
            if not pts:
                continue
            ax.plot(*zip(*pts), color=COLORS[arm], lw=1.7, marker="o", ms=4.2,
                    zorder=4, label=f"$n_0$ {arm}")
            ax.annotate(arm, xy=pts[-1], xytext=(4, 0), textcoords="offset points",
                        fontsize=6.8, color=INK2, va="center")
        ax.set_xticks(list(x))
        ax.set_xticklabels([LABELS[l] for l in LEVELS])
        ax.set_xlabel(r"completeness  $C(z\leq0.30)$")
        ax.set_title(ttl)
        ax.grid(True, alpha=0.55)
        ax.set_axisbelow(True)
        ax.set_xlim(-0.3, len(LEVELS) - 0.25)
    axL.set_ylabel(r"detection significance of $f_{\rm AGN}$  [median/$\sigma$]")
    axL.legend(loc="upper right", ncols=2)
    fig.suptitle("What density knowledge buys, before and after the sky-width fix",
                 fontsize=9.4)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"fig_n0_sig_prepost.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("wrote figs/fig_n0_sig_prepost.{pdf,png}")

    D = {"arms": ARMS, "levels": LEVELS,
         "detection_sigma": {"pre": tp, "post": tq},
         "sigma_f": {"pre": table(pre, "half_width68"),
                     "post": table(post, "half_width68")},
         "f_median": {"pre": table(pre, "median"),
                      "post": table(post, "median")},
         "n0_flat_prior": {
             arm: {lev: post["levels"].get(lev, {}).get("log10n0_agn_flat_prior")
                   for lev in LEVELS} for arm in ["post"]},
    }
    (RESULTS / "n0_prepost_fix.json").write_text(json.dumps(D, indent=2,
                                                            default=float))
    print("wrote results/n0_prepost_fix.json\n")
    print("detection significance  pre -> post")
    print(f"{'level':>9} " + "".join(f"{a:>16}" for a in ARMS))
    for lev in LEVELS:
        row = ""
        for a in ARMS:
            p, q = tp[lev][a], tq[lev][a]
            row += (f"  {p:6.1f}->{q:5.1f} " if p is not None and q is not None
                    else f"{'--':>16}")
        print(f"{lev:>9} " + row)


if __name__ == "__main__":
    main()
