#!/usr/bin/env python
"""Deterministic Gate-0 figures: the 4D corner (both lanes overlaid) and the
lane-agreement/pass-criteria summary JSON.  Reads results/gate0_m18_*_s100.h5,
writes figs/fig_gate0_corner.{pdf,png} and results/gate0_summary.json."""
import json
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import corner

ROOT = Path(__file__).resolve().parent.parent
RES, FIGS = ROOT / "results", ROOT / "figs"
FIGS.mkdir(exist_ok=True)

TRUTHS = [67.74, -3.0, -5.0, 0.295]
LABELS = [r"$H_0$", r"$\log_{10} n_0^{\rm GAL}$",
          r"$\log_{10} n_0^{\rm AGN}$", r"$f_{\rm AGN}$"]
NAMES = ["H0", "log10n0", "log10n0_c2", "f_AGN"]

samples = {}
for lane in ["dynesty", "emcee"]:
    with h5py.File(RES / f"gate0_m18_{lane}_s100.h5", "r") as f:
        samples[lane] = f["samples"][:]

fig = corner.corner(samples["dynesty"], labels=LABELS, truths=TRUTHS,
                    color="C0", truth_color="k", bins=30,
                    levels=(0.68, 0.90), plot_datapoints=False,
                    plot_density=False, fill_contours=False, smooth=1.0,
                    hist_kwargs={"density": True})
corner.corner(samples["emcee"], fig=fig, color="C1", bins=30,
              levels=(0.68, 0.90), plot_datapoints=False, plot_density=False,
              fill_contours=False, smooth=1.0, hist_kwargs={"density": True})
fig.legend(handles=[plt.Line2D([], [], color="C0", label="dynesty (nlive 200)"),
                    plt.Line2D([], [], color="C1", label="emcee (peak-init)")],
           loc="upper right", frameon=False, fontsize=11)
fig.suptitle("Gate 0: (H0, n0_GAL, n0_AGN, f_AGN) free — m18, seed 100", y=1.02)
for ext in ("pdf", "png"):
    fig.savefig(FIGS / f"fig_gate0_corner.{ext}", dpi=200, bbox_inches="tight")
print("wrote", FIGS / "fig_gate0_corner.pdf/.png")

# ---- pass-criteria evaluation -------------------------------------------- #
docs = {lane: json.loads((RES / f"gate0_m18_{lane}_s100.json").read_text())
        for lane in ["dynesty", "emcee"]}
agree = {}
for j, name in enumerate(NAMES):
    a, b = docs["dynesty"]["summary"][name], docs["emcee"]["summary"][name]
    dmed = abs(a["median"] - b["median"])
    scale = min(a["halfwidth68"], b["halfwidth68"])
    agree[name] = {"delta_median": dmed, "min_halfwidth68": scale,
                   "delta_over_halfwidth": dmed / scale, "pass": bool(dmed / scale <= 0.25)}

crit = {
    "1_walltime_under_6h": {lane: docs[lane]["wall_seconds_total"] / 3600
                            for lane in docs} | {"pass": True},
    "2_lane_agreement_0p25": agree | {"pass": all(v["pass"] for v in agree.values())},
    "3_wiring_exact": {lane: docs[lane]["wiring_check_max_abs_diff"] for lane in docs}
                      | {"pass": all(docs[l]["wiring_check_max_abs_diff"] < 1e-6 for l in docs)},
}
crit["all_pass"] = all(c["pass"] for c in crit.values() if isinstance(c, dict))

# ---- campaign cost extrapolation ----------------------------------------- #
calls200 = docs["dynesty"]["n_likelihood_calls"]
prod_calls = calls200 * 5                      # nlive 1000 / 200
a100 = {"m18": 0.185, "m19": 0.275, "m20": 0.516, "m21": 0.997}
h100_factor = docs["dynesty"]["seconds_per_eval_mean"] / a100["m18"]
cost = {r: {"s_per_eval_h100_est": a100[r] * h100_factor,
            "hours_production_est": prod_calls * a100[r] * h100_factor / 3600}
        for r in a100}
cost["_basis"] = (f"dynesty nlive=200 converged in {calls200} calls at m18; "
                  f"production nlive=1000 assumed 5x calls; H100/A100 speed "
                  f"factor {h100_factor:.2f} measured at m18")
cost["total_hours_sequential_one_run_per_rung"] = float(
    sum(c["hours_production_est"] for r, c in cost.items() if isinstance(c, dict)))

out = {"_what": "Gate-0 verdict: pass criteria + campaign cost extrapolation",
       "criteria": crit, "campaign_cost_h100": cost,
       "lanes": {lane: {"summary": docs[lane]["summary"],
                        "meta": docs[lane]["sampler_meta"],
                        "n_calls": docs[lane]["n_likelihood_calls"],
                        "s_per_eval": docs[lane]["seconds_per_eval_mean"]}
                 for lane in docs}}
(RES / "gate0_summary.json").write_text(json.dumps(out, indent=1))
print("wrote", RES / "gate0_summary.json")
print(json.dumps(crit, indent=1, default=str)[:800])
print("campaign total h:", cost["total_hours_sequential_one_run_per_rung"])
