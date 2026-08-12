#!/usr/bin/env python
"""Deterministic aggregation + figures for the analysis-5 campaign.

Reads   results/campaign_<rung>_dynesty[_r2]_s100.{h5,json}   (this directory)
        ../analysis_3_incomplete_catalog_H0_fagn/results/joint_<rung>_s100.h5
                                                    (fixed-anchor seed-100 reference)
Writes  results/campaign_summary.json
        figs/fig_campaign_ladder.{pdf,png}
        figs/fig_anchor_cost.{pdf,png}
        figs/fig_campaign_corner.{pdf,png}
        figs/fig_anchor_degeneracy.{pdf,png}

The fixed-anchor reference is the SAME seed, the SAME estimator and the SAME
1000-event mixture as the free-anchor run; the only difference is that the two
completion densities are pinned at the mock's truths instead of sampled.  Its
marginals are recomputed here from the stored 2-D log-likelihood grids with the
project's flat-prior/trapezoid convention (scan_h0f.marginal_ci), so the
fixed-vs-free widths are directly comparable.
"""
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
A3 = ROOT.parent / "analysis_3_incomplete_catalog_H0_fagn" / "results"
FIGS.mkdir(exist_ok=True)

RUNGS = ["m21", "m20", "m19", "m18"]           # bright -> faint, ladder order
NAMES = ["H0", "log10n0", "log10n0_c2", "f_AGN"]
LABELS = [r"$H_0$", r"$\log_{10} n_0^{\rm GAL}$",
          r"$\log_{10} n_0^{\rm AGN}$", r"$f_{\rm AGN}$"]
TRUTHS = {"H0": 67.74, "log10n0": -3.0, "log10n0_c2": -5.0, "f_AGN": 0.295}
PRIOR_EDGE = {"log10n0": (-4.0, -1.0), "log10n0_c2": (-6.0, -4.0)}
CRUNG = {"m21": "C0", "m20": "C1", "m19": "C2", "m18": "C3"}


# --------------------------------------------------------------------------- #
# flat-prior / trapezoid marginals, identical convention to analysis 3
# --------------------------------------------------------------------------- #
def marginal_ci(x, logp_1d, levels=(0.68, 0.90)):
    logp_1d = np.asarray(logp_1d, dtype=float)
    x = np.asarray(x, dtype=float)
    finite = np.isfinite(logp_1d)
    m = np.nanmax(logp_1d[finite]) if finite.any() else 0.0
    p = np.exp(np.where(finite, logp_1d, -np.inf) - m)
    p = p / np.trapz(p, x)
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (p[1:] + p[:-1]) * np.diff(x))])
    cdf /= cdf[-1]
    out = {"median": float(np.interp(0.5, cdf, x))}
    for lev in levels:
        out["ci{:.0f}".format(lev * 100)] = [
            float(np.interp(0.5 - lev / 2, cdf, x)),
            float(np.interp(0.5 + lev / 2, cdf, x))]
    out["halfwidth68"] = float((out["ci68"][1] - out["ci68"][0]) / 2)
    return out


def fixed_anchor_reference(rung):
    """(H0, f) marginals of analysis 3's seed-100 fixed-anchor 2-D grid."""
    with h5py.File(A3 / f"joint_{rung}_s100.h5", "r") as f:
        H0, fv = f["H0_grid"][:], f["f_grid"][:]
        logL = f["log_likelihood"][:]            # (nH0, nf)
        sha = f.attrs["merge_sha_checked"]
        n0, n0c2 = float(f.attrs["arg_log10n0"]), float(f.attrs["arg_log10n0_c2"])
    m = np.nanmax(logL[np.isfinite(logL)])
    P = np.exp(np.where(np.isfinite(logL), logL, -np.inf) - m)
    logP_H0 = np.log(np.trapz(P, fv, axis=1))
    logP_f = np.log(np.trapz(P, H0, axis=0))
    out = {"H0": marginal_ci(H0, logP_H0), "f_AGN": marginal_ci(fv, logP_f),
           "anchors_pinned_at": {"log10n0": n0, "log10n0_c2": n0c2},
           "merge_sha_checked": str(sha)}
    for k, t in (("H0", TRUTHS["H0"]), ("f_AGN", TRUTHS["f_AGN"])):
        b = out[k]
        b["truth"] = t
        b["offset"] = b["median"] - t
        b["pull"] = b["offset"] / b["halfwidth68"]
        for lev in ("ci68", "ci90"):
            b["truth_in_" + lev] = bool(b[lev][0] <= t <= b[lev][1])
    return out


# --------------------------------------------------------------------------- #
# load the campaign
# --------------------------------------------------------------------------- #
def load(tag):
    with h5py.File(RES / f"{tag}.h5", "r") as f:
        s = f["samples"][:]
        labs = [x.decode() if isinstance(x, bytes) else str(x)
                for x in f["sampled_labels"][:]]
    return s, labs, json.loads((RES / f"{tag}.json").read_text())

free, samples = {}, {}
for rung in RUNGS:
    samples[rung], labs, free[rung] = load(f"campaign_{rung}_dynesty_s100")
    assert labs == ["H0", "log10n0", "log10n0_c2", "fcat_2"], labs
samples_r2, _, free_r2 = load("campaign_m18_dynesty_r2_s100")
fixed = {rung: fixed_anchor_reference(rung) for rung in RUNGS}


# --------------------------------------------------------------------------- #
# aggregation
# --------------------------------------------------------------------------- #
def rail_fraction(x, name):
    """Posterior mass within 5 % of the prior width of an edge (rail test)."""
    if name not in PRIOR_EDGE:
        return None
    lo, hi = PRIOR_EDGE[name]
    pad = 0.05 * (hi - lo)
    return {"lower": float(np.mean(x <= lo + pad)),
            "upper": float(np.mean(x >= hi - pad)),
            "pad": pad, "prior": [lo, hi]}

def constraint_shape(x, name):
    """Is an anchor MEASURED, or only bounded on one side?

    Compares the posterior CDF to the flat prior's over the whole prior range
    (max |dCDF|, a one-sample KS distance) and separately over the sub-range
    below the truth.  A parameter the data cannot see stays flat there: its
    median then reports prior volume, not a measurement, and the truth falling
    outside a 68 % interval is meaningless.
    """
    lo, hi = PRIOR_EDGE[name]
    t = TRUTHS[name]
    xs = np.sort(x)
    grid = np.linspace(lo, hi, 601)
    post_cdf = np.searchsorted(xs, grid, side="right") / xs.size
    ks_full = float(np.max(np.abs(post_cdf - (grid - lo) / (hi - lo))))
    below = grid <= t
    # prior CDF renormalised to the sub-range, posterior likewise
    m_below = float(np.mean(x <= t))
    if m_below > 1e-3:
        pc = np.searchsorted(xs, grid[below], side="right") / xs.size / m_below
        ks_below = float(np.max(np.abs(pc - (grid[below] - lo) / (t - lo))))
    else:
        ks_below = float("nan")
    return {"mass_below_truth": m_below,
            "ks_vs_prior_full_range": ks_full,
            "ks_vs_prior_below_truth": ks_below,
            "upper_limit_90": float(np.percentile(x, 90)),
            "lower_limit_10": float(np.percentile(x, 10)),
            "verdict": ("measured" if ks_below > 0.25 or m_below < 0.05
                        else "one-sided: flat below truth, only bounded above")}

rung_block = {}
for i, rung in enumerate(RUNGS):
    d, s = free[rung], samples[rung]
    j_n0c2, j_f = NAMES.index("log10n0_c2"), NAMES.index("f_AGN")
    cost = {}
    for k in ("H0", "f_AGN"):
        hw_free = d["summary"][k]["halfwidth68"]
        hw_fix = fixed[rung][k]["halfwidth68"]
        cost[k] = {"halfwidth68_free": hw_free, "halfwidth68_fixed": hw_fix,
                   "ratio_free_over_fixed": hw_free / hw_fix,
                   "median_free": d["summary"][k]["median"],
                   "median_fixed": fixed[rung][k]["median"],
                   "median_shift": d["summary"][k]["median"] - fixed[rung][k]["median"]}
    rung_block[rung] = {
        "free": d["summary"],
        "fixed_anchor_reference": fixed[rung],
        "cost_of_freeing_anchors": cost,
        "corr_n0AGN_f": d["summary"]["corr"]["matrix"][j_n0c2][j_f],
        "corr_n0GAL_f": d["summary"]["corr"]["matrix"][NAMES.index("log10n0")][j_f],
        "rails": {n: rail_fraction(s[:, k], n) for k, n in enumerate(
            ["H0", "log10n0", "log10n0_c2", "f_AGN"]) if n in PRIOR_EDGE},
        "anchor_constraint_shape": {n: constraint_shape(s[:, k], n)
                                    for k, n in enumerate(NAMES) if n in PRIOR_EDGE},
        "sampler": d["sampler_meta"] | {
            "n_likelihood_calls": d["n_likelihood_calls"],
            "n_guard_rejected_calls": d["n_guard_rejected_calls"],
            "seconds_per_eval_mean": d["seconds_per_eval_mean"],
            "wall_hours": d["wall_seconds_total"] / 3600.0},
        "wiring_check_max_abs_diff": d["wiring_check_max_abs_diff"],
        "priors": d["priors"],
    }

# m18 duplicate-rstate reproducibility
rstate = {}
for name in NAMES:
    a, b = free["m18"]["summary"][name], free_r2["summary"][name]
    dm = abs(a["median"] - b["median"])
    scale = min(a["halfwidth68"], b["halfwidth68"])
    rstate[name] = {"median_a": a["median"], "median_b": b["median"],
                    "delta_median": dm, "delta_over_halfwidth68": dm / scale}

summary = {
    "_what": "Analysis 5 campaign: 4-parameter (H0, n0_GAL, n0_AGN, f_AGN) "
             "posteriors down the magnitude ladder with both completion-density "
             "anchors free, against the fixed-anchor seed-100 reference from "
             "analysis 3.",
    "config": {
        "seed": 100, "injections": "targeted", "sampler": "dynesty static NS",
        "nlive": 1000, "dlogz_target": 0.1, "maxcall": 500000,
        "rungs": RUNGS, "truths": TRUTHS,
        "merge_sha_checked": free["m21"]["darksirens_git_sha"],
        "fixed_anchor_reference": "analysis_3/results/joint_<rung>_s100.h5, "
                                  "flat-prior trapezoid marginals recomputed here",
    },
    "rungs": rung_block,
    "m18_duplicate_rstate": rstate,
    "totals": {
        "wall_hours_all_runs": sum(
            free[r]["wall_seconds_total"] for r in RUNGS) / 3600.0
            + free_r2["wall_seconds_total"] / 3600.0,
        "n_likelihood_calls_all_runs": int(
            sum(free[r]["n_likelihood_calls"] for r in RUNGS)
            + free_r2["n_likelihood_calls"]),
        "max_wiring_abs_diff_all_runs": max(
            [free[r]["wiring_check_max_abs_diff"] for r in RUNGS]
            + [free_r2["wiring_check_max_abs_diff"]]),
        "any_stopped_by_maxcall": any(
            free[r]["sampler_meta"]["stopped_by_maxcall"] for r in RUNGS),
        "guard_rejected_calls_all_runs": int(
            sum(free[r]["n_guard_rejected_calls"] for r in RUNGS)),
    },
}
(RES / "campaign_summary.json").write_text(json.dumps(summary, indent=1))
print("wrote", RES / "campaign_summary.json")


# --------------------------------------------------------------------------- #
# fig 1: the ladder, four parameters, free vs fixed
# --------------------------------------------------------------------------- #
x = np.arange(len(RUNGS))
fig, axes = plt.subplots(1, 4, figsize=(15.5, 3.9))
for j, (name, lab) in enumerate(zip(NAMES, LABELS)):
    ax = axes[j]
    med = np.array([free[r]["summary"][name]["median"] for r in RUNGS])
    lo68 = np.array([free[r]["summary"][name]["ci68"][0] for r in RUNGS])
    hi68 = np.array([free[r]["summary"][name]["ci68"][1] for r in RUNGS])
    lo90 = np.array([free[r]["summary"][name]["ci90"][0] for r in RUNGS])
    hi90 = np.array([free[r]["summary"][name]["ci90"][1] for r in RUNGS])
    ax.errorbar(x, med, yerr=[med - lo90, hi90 - med], fmt="none",
                ecolor="C0", alpha=0.35, lw=5, capsize=0)
    ax.errorbar(x, med, yerr=[med - lo68, hi68 - med], fmt="o", color="C0",
                ms=6, lw=2.2, capsize=3,
                label="both anchors free" if j == 0 else None)
    if name in ("H0", "f_AGN"):
        fm = np.array([fixed[r][name]["median"] for r in RUNGS])
        fl = np.array([fixed[r][name]["ci68"][0] for r in RUNGS])
        fh = np.array([fixed[r][name]["ci68"][1] for r in RUNGS])
        ax.errorbar(x + 0.14, fm, yerr=[fm - fl, fh - fm], fmt="s", color="C3",
                    ms=5, lw=1.8, capsize=3, alpha=0.9,
                    label="anchors pinned at truth" if j == 0 else None)
    if name in PRIOR_EDGE:
        for e in PRIOR_EDGE[name]:
            ax.axhline(e, color="0.6", ls=":", lw=1.2)
        ax.set_ylim(PRIOR_EDGE[name][0] - 0.12, PRIOR_EDGE[name][1] + 0.12)
    ax.axhline(TRUTHS[name], color="k", ls="--", lw=1.3)
    ax.set_xticks(x)
    ax.set_xticklabels([f"$m<{r[1:]}$" for r in RUNGS])
    ax.set_ylabel(lab)
    ax.set_xlim(-0.45, len(RUNGS) - 0.35)
axes[0].legend(frameon=False, fontsize=9, loc="upper left")
axes[1].text(0.03, 0.06, "dotted: prior edges", transform=axes[1].transAxes,
             fontsize=8, color="0.35")
fig.suptitle("Freeing both completion-density anchors: seed 100, "
             "1000-event K=2 mixture", y=1.02)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(FIGS / f"fig_campaign_ladder.{ext}", dpi=200, bbox_inches="tight")
plt.close(fig)
print("wrote", FIGS / "fig_campaign_ladder.pdf/.png")


# --------------------------------------------------------------------------- #
# fig 2: what freeing the anchors costs
# --------------------------------------------------------------------------- #
fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9))
for k, (name, lab) in enumerate([("H0", r"$H_0$"), ("f_AGN", r"$f_{\rm AGN}$")]):
    ax = axes[k]
    ratio = [rung_block[r]["cost_of_freeing_anchors"][name]["ratio_free_over_fixed"]
             for r in RUNGS]
    ax.plot(x, ratio, "o-", color="C0", lw=2, ms=7)
    for xi, ri in zip(x, ratio):
        ax.annotate(f"{ri:.2f}", (xi, ri), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=9)
    ax.axhline(1.0, color="k", ls="--", lw=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels([f"$m<{r[1:]}$" for r in RUNGS])
    ax.set_ylabel(rf"$\sigma$({lab}) free / pinned")
    ax.set_title(lab)
    ax.set_ylim(0, max(ratio) * 1.28)
fig.suptitle("Cost of not knowing the completion densities", y=1.02)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(FIGS / f"fig_anchor_cost.{ext}", dpi=200, bbox_inches="tight")
plt.close(fig)
print("wrote", FIGS / "fig_anchor_cost.pdf/.png")


# --------------------------------------------------------------------------- #
# fig 3: all four rungs overlaid in the 4-D corner
# --------------------------------------------------------------------------- #
rng = [(TRUTHS["H0"] - 6, TRUTHS["H0"] + 8), PRIOR_EDGE["log10n0"],
       PRIOR_EDGE["log10n0_c2"], (0.0, 1.0)]
fig = None
for rung in RUNGS:
    kw = dict(labels=LABELS, color=CRUNG[rung], bins=28, range=rng,
              levels=(0.68, 0.90), plot_datapoints=False, plot_density=False,
              fill_contours=False, smooth=1.0, hist_kwargs={"density": True})
    fig = corner.corner(samples[rung], fig=fig,
                        truths=list(TRUTHS.values()) if fig is None else None,
                        truth_color="k", **kw)
fig.legend(handles=[plt.Line2D([], [], color=CRUNG[r], label=f"$m<{r[1:]}$")
                    for r in RUNGS],
           loc="upper right", frameon=False, fontsize=12)
fig.suptitle("Four free parameters down the ladder (dynesty, nlive 1000)", y=1.01)
for ext in ("pdf", "png"):
    fig.savefig(FIGS / f"fig_campaign_corner.{ext}", dpi=200, bbox_inches="tight")
plt.close(fig)
print("wrote", FIGS / "fig_campaign_corner.pdf/.png")


# --------------------------------------------------------------------------- #
# fig 4: the (n0_AGN, f_AGN) degeneracy and where the GAL anchor goes
# --------------------------------------------------------------------------- #
fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.3))
ax = axes[0]
for rung in RUNGS:
    s = samples[rung]
    corner.hist2d(s[:, 2], s[:, 3], ax=ax, levels=(0.68,), bins=26,
                  plot_datapoints=False, plot_density=False,
                  fill_contours=False, smooth=1.0,
                  contour_kwargs={"colors": CRUNG[rung], "linewidths": 2})
ax.axvline(TRUTHS["log10n0_c2"], color="k", ls="--", lw=1.2)
ax.axhline(TRUTHS["f_AGN"], color="k", ls="--", lw=1.2)
ax.set_xlabel(LABELS[2])
ax.set_ylabel(LABELS[3])
ax.set_xlim(*PRIOR_EDGE["log10n0_c2"])
ax.set_ylim(0, 1)
ax.legend(handles=[plt.Line2D([], [], color=CRUNG[r],
                              label=rf"$m<{r[1:]}$  $\rho={rung_block[r]['corr_n0AGN_f']:+.2f}$")
                   for r in RUNGS], frameon=False, fontsize=9, loc="upper left")
ax.set_title("AGN anchor vs. the AGN fraction")

ax = axes[1]
for rung in RUNGS:
    ax.hist(samples[rung][:, 1], bins=45, range=PRIOR_EDGE["log10n0"],
            histtype="step", lw=2, density=True, color=CRUNG[rung],
            label=f"$m<{rung[1:]}$")
ax.axvline(TRUTHS["log10n0"], color="k", ls="--", lw=1.3)
for e in PRIOR_EDGE["log10n0"]:
    ax.axvline(e, color="0.6", ls=":", lw=1.2)
ax.set_xlabel(LABELS[1])
ax.set_ylabel("posterior density")
ax.legend(frameon=False, fontsize=9)
ax.set_title("Galaxy anchor: recovered at $m<19$, railed at $m<18$")
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(FIGS / f"fig_anchor_degeneracy.{ext}", dpi=200, bbox_inches="tight")
plt.close(fig)
print("wrote", FIGS / "fig_anchor_degeneracy.pdf/.png")


# --------------------------------------------------------------------------- #
print("\n--- ladder (free anchors, seed 100) ---")
for rung in RUNGS:
    b = rung_block[rung]
    print(f"{rung}: H0 {b['free']['H0']['median']:.2f} "
          f"(+{b['free']['H0']['offset']:.2f}, x{b['cost_of_freeing_anchors']['H0']['ratio_free_over_fixed']:.2f}) "
          f"| f {b['free']['f_AGN']['median']:.3f}+-{b['free']['f_AGN']['halfwidth68']:.3f} "
          f"(x{b['cost_of_freeing_anchors']['f_AGN']['ratio_free_over_fixed']:.2f}, "
          f"in68 {b['free']['f_AGN']['truth_in_ci68']}) "
          f"| rho(n0AGN,f) {b['corr_n0AGN_f']:+.2f} "
          f"| GAL mass at high-density prior edge {b['rails']['log10n0']['upper']:.2f}")
print("\n--- anchor constraint shape ---")
for rung in RUNGS:
    for n, b in rung_block[rung]["anchor_constraint_shape"].items():
        print(f"{rung} {n:11s} mass<truth {b['mass_below_truth']:.2f} "
              f"KS(below) {b['ks_vs_prior_below_truth']:.2f} -> {b['verdict']}")
print("\ntotals:", json.dumps(summary["totals"], indent=1))
