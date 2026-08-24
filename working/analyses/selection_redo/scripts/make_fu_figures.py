#!/usr/bin/env python
"""Follow-up campaign figures (2026-08-23): zero-density probe + seed replication.

Deterministic: reads only committed/stored result JSONs, writes PDF+PNG into
each fu_* analysis' figs/. All intervals are 90% (standing owner rule).

    python make_fu_figures.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                        # selection_redo
ANALYSES = ROOT.parent
DSX = ANALYSES / "experiments" / "experiment_dsmaster_4d_recheck" / "results"

TRUTH_F, TRUTH_GAL = 0.295, -3.0


def j(path):
    return json.load(open(path))


def save(fig, outdirs, stem):
    for d in outdirs:
        d.mkdir(parents=True, exist_ok=True)
        for ext in ("pdf", "png"):
            fig.savefig(d / f"{stem}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


# --- Fig 1: zero-density probe -----------------------------------------------
fid = j(ANALYSES / "analysis_3_incomplete_catalog_H0_fagn" / "results" /
        "joint_complete_s100.json")
m24 = j(ROOT / "fu_probes" / "results" / "joint_complete_n0m24_s100.json")

fig, ax = plt.subplots(figsize=(5.2, 3.4))
cells = [("fiducial\n$\\log_{10}n_0=(-3,-5)$", fid["f"]),
         ("zero density\n$\\log_{10}n_0=(-24,-24)$", m24["f"])]
for i, (lab, f) in enumerate(cells):
    lo, hi = f["ci90"]
    ax.errorbar(i, f["median"], yerr=[[f["median"] - lo], [hi - f["median"]]],
                fmt="o", color="C0", capsize=4, lw=1.5)
ax.axhline(TRUTH_F, color="k", lw=0.8, ls="--", label="truth 0.295")
ax.axhspan(TRUTH_F - 0.023, TRUTH_F - 0.007, color="C2", alpha=0.18,
           label="flux-limited cells (offset $-$0.007..$-$0.023)")
ax.set_xticks(range(len(cells)), [c[0] for c in cells])
ax.set_ylabel("$f_{\\rm AGN}$ (median, 90% CI)")
ax.set_title("a3 complete rung under selection: the density fiducial moves "
             "$f_{\\rm AGN}$ at $C\\equiv1$", fontsize=9)
ax.legend(fontsize=7, loc="lower right")
save(fig, [ROOT / "fu_probes" / "figs"], "fig_zero_density")

# --- Fig 2: seed replication -------------------------------------------------
SEEDS = [100, 101, 102]
src = {
    (100, "selection"): DSX / "fit_m18_selection_s100.json",
    (100, "per_pixel"): DSX / "fit_m18_per_pixel_s100.json",
    (101, "selection"): ROOT / "fu_seed101" / "results" / "campaign_m18_dynesty_s101.json",
    (101, "per_pixel"): ROOT / "fu_seed101" / "results" / "campaign_m18_dynesty_pp_s101.json",
    (102, "selection"): ROOT / "fu_seed102" / "results" / "campaign_m18_dynesty_s102.json",
    (102, "per_pixel"): ROOT / "fu_seed102" / "results" / "campaign_m18_dynesty_pp_s102.json",
}
R = {k: j(p) for k, p in src.items()}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.4))
off = {"selection": -0.08, "per_pixel": +0.08}
col = {"selection": "C0", "per_pixel": "C3"}
for arm in ("selection", "per_pixel"):
    xs, med, lo, hi = [], [], [], []
    for i, s in enumerate(SEEDS):
        v = R[(s, arm)]["summary"]["log10n0"]
        xs.append(i + off[arm]); med.append(v["median"])
        lo.append(v["median"] - v["ci90"][0]); hi.append(v["ci90"][1] - v["median"])
    ax1.errorbar(xs, med, yerr=[lo, hi], fmt="o", color=col[arm], capsize=4,
                 lw=1.5, label=arm)
ax1.axhline(TRUTH_GAL, color="k", lw=0.8, ls="--")
ax1.set_xticks(range(len(SEEDS)), [f"seed {s}" for s in SEEDS])
ax1.set_ylabel("GAL anchor $\\log_{10}n_0$ (median, 90% CI)")
ax1.text(0.02, 0.04, "truth $-3$", transform=ax1.transAxes, fontsize=7)
ax1.legend(fontsize=8, loc="upper right")

dlnz = [R[(s, "selection")]["sampler_meta"]["logz"]
        - R[(s, "per_pixel")]["sampler_meta"]["logz"] for s in SEEDS]
ax2.bar(range(len(SEEDS)), dlnz, color="C0", width=0.55)
for i, v in enumerate(dlnz):
    ax2.text(i, v + 0.3, f"+{v:.1f}", ha="center", fontsize=8)
ax2.set_xticks(range(len(SEEDS)), [f"seed {s}" for s in SEEDS])
ax2.set_ylabel("$\\Delta\\ln Z$ (selection $-$ per_pixel)")
ax2.set_ylim(0, max(dlnz) * 1.18)
fig.suptitle("m<18 free anchors, darksirens 0c5b3db: the a5 headline replicates "
             "at three seeds", fontsize=9)
fig.tight_layout()
save(fig, [ROOT / "fu_seed101" / "figs", ROOT / "fu_seed102" / "figs"],
     "fig_seed_replication")

print("figures written:",
      ROOT / "fu_probes" / "figs", "and fu_seed10{1,2}/figs")
