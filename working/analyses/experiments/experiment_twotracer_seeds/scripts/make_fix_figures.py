#!/usr/bin/env python3
"""Pre-fix vs post-fix comparison figures for the 12-seed two-tracer rerun.

Figure 1 (figs/seeds_fix_strip.png): paired per-seed offset strips, one panel per
statistic (f-scan f, joint f, joint H0).  Each seed is a light connector from its
pre-fix offset to its post-fix offset; heavy markers show the mean +- sem.

Figure 2 (figs/seeds_fix_widths.png): seed-to-seed scatter (sd) against the mean
quoted 68% half-width, pre vs post, per statistic -- the "are the intervals
underquoted" picture.  Reference line at ratio 1.

Colors: documented default categorical palette, slots 1-2 (light mode)
(#2a78d6 pre-fix, #eb6834 post-fix); passes the adjacent-pair gates per the
palette's own validation record.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parent.parent
RES = BASE / "results"
FIGS = BASE / "figs"
FIGS.mkdir(exist_ok=True)

C_PRE = "#2a78d6"
C_POST = "#eb6834"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#d9d8d4"

pre = json.loads((RES / "seeds_summary.json").read_text())
post = json.loads((RES / "seeds_summary_fix.json").read_text())
TRUTH = {"fscan_f": 0.30, "joint_f": 0.30, "joint_H0": 67.74}
LABELS = {"fscan_f": "f_AGN (f-scan, H0 fixed)",
          "joint_f": "f_AGN (joint)",
          "joint_H0": "H0 (joint) [km/s/Mpc]"}


def per_seed(summary, key):
    return {r["seed"]: r[key] - TRUTH[key] for r in summary["per_seed"] if key in r}


plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.titlecolor": INK, "figure.facecolor": "#fcfcfb",
    "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
})

# ---------------------------------------------------------------- figure 1 ----
fig, axes = plt.subplots(1, 3, figsize=(10.5, 4.2))
for ax, key in zip(axes, ("fscan_f", "joint_f", "joint_H0")):
    a, b = per_seed(pre, key), per_seed(post, key)
    seeds = sorted(set(a) & set(b))
    ya = np.array([a[s] for s in seeds])
    yb = np.array([b[s] for s in seeds])
    for va, vb in zip(ya, yb):
        ax.plot([0, 1], [va, vb], color=GRID, lw=1, zorder=1)
    ax.scatter(np.zeros(len(ya)), ya, s=28, color=C_PRE, zorder=3,
               edgecolors="#fcfcfb", linewidths=0.8, label="pre-fix")
    ax.scatter(np.ones(len(yb)), yb, s=28, color=C_POST, zorder=3,
               edgecolors="#fcfcfb", linewidths=0.8, label="post-fix")
    for x, y, c, st in ((0, ya, C_PRE, pre[key]), (1, yb, C_POST, post[key])):
        ax.errorbar([x + 0.13 if x else x - 0.13], [st["mean"]],
                    yerr=[st["sem"]], fmt="D", ms=6, color=c,
                    elinewidth=2, capsize=4, zorder=4)
    ax.axhline(0.0, color=INK2, lw=1, ls=(0, (4, 3)), zorder=2)
    ax.set_xlim(-0.55, 1.55)
    ax.set_xticks([0, 1], ["pre-fix", "post-fix"])
    ax.set_title(LABELS[key], fontsize=10)
    ax.grid(axis="y", color=GRID, lw=0.6, alpha=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ma, mb = pre[key], post[key]
    ax.annotate(f"{ma['mean']:+.3g}±{ma['sem']:.2g}", (-0.13, ma["mean"]),
                xytext=(-8, 0), textcoords="offset points", ha="right",
                va="center", fontsize=8.5, color=INK2)
    ax.annotate(f"{mb['mean']:+.3g}±{mb['sem']:.2g}", (1.13, mb["mean"]),
                xytext=(8, 0), textcoords="offset points", ha="left",
                va="center", fontsize=8.5, color=INK2)
axes[0].set_ylabel("offset from truth")
axes[0].legend(loc="lower left", frameon=False, fontsize=9)
fig.suptitle("Two-tracer 12-seed reruns: per-seed offsets, "
             "pre-fix vs sigma_ang-fixed generator", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(FIGS / "seeds_fix_strip.png", dpi=200)
plt.close(fig)

# ---------------------------------------------------------------- figure 2 ----
fig, ax = plt.subplots(figsize=(7.2, 3.6))
rows = []
for key in ("fscan_f", "joint_f", "joint_H0"):
    for tag, summ in (("pre", pre), ("post", post)):
        st = summ[key]
        rows.append((f"{LABELS[key].split(' [')[0]} — {tag}-fix",
                     st["scatter_over_quoted_half_width"],
                     C_PRE if tag == "pre" else C_POST,
                     st["sd"], st["mean_quoted_half_width"]))
ypos = np.arange(len(rows))[::-1]
for y, (lab, ratio, c, sd, hw) in zip(ypos, rows):
    ax.plot([0, ratio], [y, y], color=c, lw=2, solid_capstyle="round", zorder=2)
    ax.scatter([ratio], [y], s=46, color=c, zorder=3,
               edgecolors="#fcfcfb", linewidths=0.8)
    ax.annotate(f"{ratio:.2f}  (sd {sd:.3g} / hw {hw:.3g})", (ratio, y),
                xytext=(8, 0), textcoords="offset points", va="center",
                fontsize=8.5, color=INK2)
ax.axvline(1.0, color=INK2, lw=1, ls=(0, (4, 3)), zorder=1)
ax.annotate("scatter = quoted width", (1.0, ypos.max() + 0.55),
            ha="center", fontsize=8.5, color=INK2)
ax.set_yticks(ypos, [r[0] for r in rows], fontsize=9)
ax.set_xlabel("seed-to-seed scatter / mean quoted 68% half-width")
ax.set_xlim(0, max(r[1] for r in rows) * 1.55)
ax.set_ylim(-0.7, len(rows) - 0.1)
ax.grid(axis="x", color=GRID, lw=0.6, alpha=0.6)
ax.set_axisbelow(True)
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.tick_params(left=False)
ax.set_title("Are single-realisation intervals underquoted?", fontsize=11)
fig.tight_layout()
fig.savefig(FIGS / "seeds_fix_widths.png", dpi=200)
plt.close(fig)
print("wrote", FIGS / "seeds_fix_strip.png")
print("wrote", FIGS / "seeds_fix_widths.png")
