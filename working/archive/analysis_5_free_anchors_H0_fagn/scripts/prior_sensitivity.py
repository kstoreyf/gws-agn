#!/usr/bin/env python
"""How tight an external AGN-density prior buys back sigma(f_AGN)?

Analysis 5 measures a factor 2-3.5 inflation of sigma(f_AGN) when the completion
densities are unknown, and shows it is a DEGENERACY: corr(n0_AGN, f_AGN) runs
+0.68 to +0.89 down the ladder.  A real analysis is not ignorant of the AGN
density — an AGN luminosity function constrains it.  This script prices that
information by reweighting the campaign's nested-sampling output under a Gaussian
prior on log10n0_c2, at several widths and at deliberately OFFSET centres.

No new likelihood calls.  dynesty's dead points carry (logl, logwt) under the
flat prior; multiplying the weights by pi_new/pi_old = N(x; mu, sigma) / (1/W)
is exact importance reweighting, valid because the flat prior's support contains
the Gaussian's mass wherever the Gaussian is non-negligible.  The honest limit is
the effective sample size, reported per case and refused below MIN_NEFF.

Reads   results/campaign_<rung>_dynesty_s100.h5
Writes  results/prior_sensitivity.json
        figs/fig_prior_sensitivity.{pdf,png}
"""
import json
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RES, FIGS = ROOT / "results", ROOT / "figs"
FIGS.mkdir(exist_ok=True)

RUNGS = ["m21", "m20", "m19", "m18"]
NAMES = ["H0", "log10n0", "log10n0_c2", "f_AGN"]
TRUTHS = {"H0": 67.74, "log10n0": -3.0, "log10n0_c2": -5.0, "f_AGN": 0.295}
J_N0C2, J_F, J_H0 = 2, 3, 0
FLAT_PRIOR = (-6.0, -4.0)                      # log10n0_c2 prior of the campaign
SIGMAS = [0.05, 0.10, 0.20, 0.30]              # dex, Gaussian prior widths
OFFSETS = [0.0, +0.15, -0.15]                  # dex, centre of that Gaussian
MIN_NEFF = 200.0
CRUNG = {"m21": "C0", "m20": "C1", "m19": "C2", "m18": "C3"}


def weighted_quantiles(x, w, qs):
    o = np.argsort(x)
    x, w = x[o], w[o]
    c = np.cumsum(w)
    c /= c[-1]
    # midpoint convention, standard for weighted quantiles
    c = (c - 0.5 * w / w.sum())
    return np.interp(qs, c, x)


def summarize(x, w, truth):
    q05, q16, q50, q84, q95 = weighted_quantiles(x, w, [0.05, 0.16, 0.50, 0.84, 0.95])
    hw = (q84 - q16) / 2
    return {"median": float(q50), "ci68": [float(q16), float(q84)],
            "ci90": [float(q05), float(q95)], "halfwidth68": float(hw),
            "truth": truth, "offset": float(q50 - truth),
            "pull": float((q50 - truth) / hw) if hw else None,
            "truth_in_ci68": bool(q16 <= truth <= q84),
            "truth_in_ci90": bool(q05 <= truth <= q95)}


def neff(w):
    w = w / w.sum()
    return float(1.0 / np.sum(w ** 2))


out = {"_what": "sigma(f_AGN) recovered by an external Gaussian prior on the "
                "completion's AGN density log10n0_c2, by importance reweighting "
                "the analysis-5 campaign chains. No new likelihood calls.",
       "method": {"flat_prior_log10n0_c2": list(FLAT_PRIOR),
                  "gaussian_sigmas_dex": SIGMAS,
                  "gaussian_offsets_dex": OFFSETS,
                  "truth_log10n0_c2": TRUTHS["log10n0_c2"],
                  "min_effective_sample_size": MIN_NEFF,
                  "reweighting": "w_new = w_flat * N(log10n0_c2; truth+offset, sigma)"},
       "rungs": {}}

samples, wflat = {}, {}
for rung in RUNGS:
    with h5py.File(RES / f"campaign_{rung}_dynesty_s100.h5", "r") as f:
        s = f["raw/samples"][:]
        logwt = f["raw/logwt"][:]
        logz = f["raw/logz"][-1]
    w = np.exp(logwt - logz)
    samples[rung], wflat[rung] = s, w / w.sum()

for rung in RUNGS:
    s, w0 = samples[rung], wflat[rung]
    x = s[:, J_N0C2]
    block = {"flat": {n: summarize(s[:, j], w0, TRUTHS[n])
                      for j, n in enumerate(NAMES)},
             "neff_flat": neff(w0), "cases": []}
    xg = s[:, 1]                                       # log10n0, the GAL anchor
    # (which_anchor, centre offset, sigma) — "agn" is the headline case; "gal"
    # and "both" attribute the residual width once the AGN density is known.
    plan = [("agn", off, sg) for off in OFFSETS for sg in SIGMAS]
    plan += [("gal", 0.0, sg) for sg in SIGMAS]
    plan += [("both", 0.0, sg) for sg in SIGMAS]
    for which, off, sg in plan:
            lp = np.zeros_like(w0)
            if which in ("agn", "both"):
                lp = lp - 0.5 * ((x - (TRUTHS["log10n0_c2"] + off)) / sg) ** 2
            if which in ("gal", "both"):
                lp = lp - 0.5 * ((xg - (TRUTHS["log10n0"] + off)) / sg) ** 2
            w = w0 * np.exp(lp - lp.max())
            if w.sum() <= 0:
                continue
            w = w / w.sum()
            ne = neff(w)
            case = {"anchor_constrained": which,
                    "prior_centre_offset_dex": off, "prior_sigma_dex": sg,
                    "neff": ne, "usable": bool(ne >= MIN_NEFF)}
            for j, n in enumerate(NAMES):
                case[n] = summarize(s[:, j], w, TRUTHS[n])
            case["sigma_f_ratio_vs_flat"] = (
                case["f_AGN"]["halfwidth68"] / block["flat"]["f_AGN"]["halfwidth68"])
            case["sigma_H0_ratio_vs_flat"] = (
                case["H0"]["halfwidth68"] / block["flat"]["H0"]["halfwidth68"])
            block["cases"].append(case)
    out["rungs"][rung] = block

# the pinned-anchor reference: what a perfectly known density gives (sigma -> 0)
cs = json.loads((RES / "campaign_summary.json").read_text())
for rung in RUNGS:
    out["rungs"][rung]["pinned_reference"] = (
        cs["rungs"][rung]["cost_of_freeing_anchors"])

(RES / "prior_sensitivity.json").write_text(json.dumps(out, indent=1))
print("wrote", RES / "prior_sensitivity.json")


# --------------------------------------------------------------------------- #
# figure: sigma(f_AGN) vs external prior width, per rung
# --------------------------------------------------------------------------- #
fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.2))

ax = axes[0]
for rung in RUNGS:
    b = out["rungs"][rung]
    cases = [c for c in b["cases"] if c["prior_centre_offset_dex"] == 0.0
             and c["anchor_constrained"] == "agn"]
    sg = [c["prior_sigma_dex"] for c in cases]
    hw = [c["f_AGN"]["halfwidth68"] for c in cases]
    ok = [c["usable"] for c in cases]
    ax.plot(sg, hw, "o-", color=CRUNG[rung], lw=2, ms=6, label=f"$m<{rung[1:]}$")
    for x_, y_, o_ in zip(sg, hw, ok):
        if not o_:
            ax.plot([x_], [y_], "x", color="k", ms=11, mew=2)
    # the two limits
    ax.axhline(b["flat"]["f_AGN"]["halfwidth68"], color=CRUNG[rung],
               ls=":", lw=1.2, alpha=0.8)
    ax.axhline(b["pinned_reference"]["f_AGN"]["halfwidth68_fixed"],
               color=CRUNG[rung], ls="--", lw=1.2, alpha=0.8)
ax.set_xlabel(r"external prior width on $\log_{10} n_0^{\rm AGN}$  [dex]")
ax.set_ylabel(r"$\sigma(f_{\rm AGN})$  (68 % half-width)")
ax.set_title(r"the width shrinks…")
ax.legend(frameon=False, fontsize=9, loc="center right")
ax.text(0.03, 0.97, "dotted: no prior (flat)\ndashed: density known exactly",
        transform=ax.transAxes, ha="left", va="top", fontsize=8, color="0.3")

# ---- panel 2: …but is the answer still right? -----------------------------  #
ax = axes[1]
for rung in RUNGS:
    b = out["rungs"][rung]
    cases = [c for c in b["cases"] if c["prior_centre_offset_dex"] == 0.0
             and c["anchor_constrained"] == "agn"]
    sg = [c["prior_sigma_dex"] for c in cases]
    pull = [c["f_AGN"]["pull"] for c in cases]
    ax.plot(sg, pull, "o-", color=CRUNG[rung], lw=2, ms=6, label=f"$m<{rung[1:]}$")
    ax.axhline(b["flat"]["f_AGN"]["pull"], color=CRUNG[rung], ls=":", lw=1.2,
               alpha=0.8)
ax.axhspan(-1, 1, color="0.85", alpha=0.6, zorder=0)
ax.axhline(0, color="k", ls="--", lw=1.2)
ax.set_xlabel(r"external prior width on $\log_{10} n_0^{\rm AGN}$  [dex]")
ax.set_ylabel(r"$f_{\rm AGN}$ pull  (offset / half-width)")
ax.set_title(r"…but only $m<19$ and $m<18$ stay honest")
ax.text(0.03, 0.95, "grey band: truth inside 68 %", transform=ax.transAxes,
        fontsize=8, color="0.3", va="top")
ax.legend(frameon=False, fontsize=9, loc="lower right")

ax = axes[2]
width = 0.22
xs = np.arange(len(SIGMAS))
for k, off in enumerate(OFFSETS):
    vals, cols = [], []
    for sg in SIGMAS:
        c = [c for c in out["rungs"]["m18"]["cases"]
             if c["prior_sigma_dex"] == sg and c["prior_centre_offset_dex"] == off
             and c["anchor_constrained"] == "agn"][0]
        vals.append(c["f_AGN"]["median"])
        cols.append("C0" if c["usable"] else "0.7")
    lab = "centred on truth" if off == 0 else f"centred {off:+.2f} dex off"
    ax.bar(xs + (k - 1) * width, vals, width, label=lab, alpha=0.9)
ax.axhline(TRUTHS["f_AGN"], color="k", ls="--", lw=1.3)
ax.set_xticks(xs)
ax.set_xticklabels([f"{s:g}" for s in SIGMAS])
ax.set_xlabel(r"prior width [dex]")
ax.set_ylabel(r"$f_{\rm AGN}$ median")
ax.set_title(r"$m<18$: a wrong prior is worse than none")
ax.legend(frameon=False, fontsize=9)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(FIGS / f"fig_prior_sensitivity.{ext}", dpi=200, bbox_inches="tight")
plt.close(fig)
print("wrote", FIGS / "fig_prior_sensitivity.pdf/.png")


# --------------------------------------------------------------------------- #
print("\n--- sigma(f_AGN), prior centred on truth ---")
hdr = "rung   flat  " + "  ".join(f"{s:>5.2f}" for s in SIGMAS) + "   pinned"
print(hdr)
for rung in RUNGS:
    b = out["rungs"][rung]
    row = [b["flat"]["f_AGN"]["halfwidth68"]]
    for sg in SIGMAS:
        c = [c for c in b["cases"] if c["prior_sigma_dex"] == sg
             and c["prior_centre_offset_dex"] == 0.0
             and c["anchor_constrained"] == "agn"][0]
        row.append(c["f_AGN"]["halfwidth68"])
    row.append(b["pinned_reference"]["f_AGN"]["halfwidth68_fixed"])
    print(f"{rung}  " + "  ".join(f"{v:.3f}" for v in row))
print("\n--- N_eff (centred) ---")
for rung in RUNGS:
    b = out["rungs"][rung]
    ne = [c["neff"] for sg in SIGMAS for c in b["cases"]
          if c["prior_sigma_dex"] == sg and c["prior_centre_offset_dex"] == 0.0
          and c["anchor_constrained"] == "agn"]
    print(f"{rung}  flat {b['neff_flat']:8.0f} | " +
          "  ".join(f"{s:g}:{n:7.0f}" for s, n in zip(SIGMAS, ne)))

print("\n--- which anchor owns the residual sigma(f_AGN)?  (0.05 dex prior) ---")
print("rung   flat   AGN-known  GAL-known  BOTH-known   pinned")
for rung in RUNGS:
    b = out["rungs"][rung]
    def pick(which):
        return [c for c in b["cases"] if c["anchor_constrained"] == which
                and c["prior_sigma_dex"] == 0.05
                and c["prior_centre_offset_dex"] == 0.0][0]["f_AGN"]["halfwidth68"]
    print(f"{rung}  {b['flat']['f_AGN']['halfwidth68']:.3f}   "
          f"{pick('agn'):.3f}      {pick('gal'):.3f}      {pick('both'):.3f}"
          f"       {b['pinned_reference']['f_AGN']['halfwidth68_fixed']:.3f}")
