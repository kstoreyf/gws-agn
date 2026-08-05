#!/usr/bin/env python3
"""Does fixing the sky-noise width (darksirens PR #335) close the K=1 bias?

Aggregates the third arm of the closure ladder:

* ``ctrl``   -- gmd's original rule (projection latent + independent PE noise).
* ``obs``    -- detection on the observed data (PR #334), sigma_ang still a
                deterministic function of the LATENT true parameters.
* ``fix``    -- same detected sets bit-identically (the sky never enters the
                detection statistic), but sigma_ang derived from the OBSERVED
                amplitude, drawn sequentially after dL/m1det/m2det (PR #335).

All three arms share catalogs, event seeds, surveys and the selection file, so
every comparison is PAIRED BY REALISATION.  The exact-likelihood oracle
(ORACLE_FINDINGS.md) predicts the fix arm should land at the darksirens
estimator overhead alone: -0.31 +- 0.13 measured as paired (ds - oracle) on the
old data, i.e. roughly -0.15..-0.35 (Farr 1/Neff term -0.12 systematic, mass
width latents -0.05, plus zero-mean per-realisation noise).

Writes ``results/obsdet_fix_summary.json`` and
``figs/fig_obsfix_closure.{pdf,png}``, ``figs/fig_obsfix_budget.{pdf,png}``.
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
TAGS = (("b", "s4102", "s4103", "s4104", "s4105")
        + tuple(f"n{s}" for s in range(4201, 4216)))
ARMS = (("ctrl", "obsdet_ctrl", "detection on true params (gmd original)"),
        ("obs", "obsdet_obs", "detection on observed data, latent sky width"),
        ("fix", "obsdet_fix", "observed sky width (PR #335)"))

# Oracle-derived expectations (ORACLE_FINDINGS.md sections 7, 9, 10).
ORACLE = {
    "exact_offset_old_data": (-0.489, 0.077),        # exact likelihood, latent width
    "paired_ds_minus_oracle_old": (-0.312, 0.125),   # estimator overhead, old data
    "bootstrap_fix_closure": (-0.062, 0.066),        # exact likelihood on fixed recipe
    "predicted_fix_arm": (-0.35, -0.15),             # prediction band for this rerun
    "farr_term": -0.118,                              # systematic components of the
    "pe_mass_width_latent": -0.047,                   # overhead (sec. 6/10)
}

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


def load(prefix):
    rows = []
    for tag in TAGS:
        p = RESULTS / f"{prefix}_{tag}.json"
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
    S = {"H0_true": H0_TRUE, "arms": {}, "oracle_reference": ORACLE}
    offs = {}
    for arm, prefix, lab in ARMS:
        rows = load(prefix)
        offs[arm] = np.array([r["offset"] for r in rows])
        S["arms"][arm] = {
            "label": lab, "per_seed": rows,
            "offset_stats": stats(offs[arm]),
            "mean_half_width": float(np.mean([r["half_width"] for r in rows])),
            "n_rejected_total": int(sum(r["n_rejected"] for r in rows))}

    # paired differences along the ladder
    for a, b in (("obs", "ctrl"), ("fix", "obs"), ("fix", "ctrl")):
        d = offs[a] - offs[b]
        S[f"paired_{a}_minus_{b}"] = stats(d)
        S[f"paired_{a}_minus_{b}"]["per_seed"] = dict(zip(TAGS, d.round(4).tolist()))

    # paired against the exact-likelihood oracle evaluated on the OLD
    # (latent-width) data.  NOTE the interpretation: the oracle offsets carry
    # the -0.49 sigma_ang defect (it survives the exact likelihood); the fix
    # arm's DATA no longer do.  So fix - oracle_old ~ estimator overhead
    # MINUS the oracle's own sky-defect systematic; the clean estimate of the
    # overhead is the fix arm's ABSOLUTE offset, since the exact likelihood on
    # fixed-recipe data closes at -0.06 +- 0.07 (parametric bootstrap).
    oracle = {r["tag"]: r for r in json.loads((RESULTS / "oracle_campaign.json").read_text())}
    oo = np.array([oracle[t]["offset_oracle"] for t in TAGS])
    S["oracle_offsets_old_data"] = dict(zip(TAGS, oo.round(4).tolist()))
    d = offs["fix"] - oo
    S["paired_fix_minus_oracle_old"] = stats(d)
    S["paired_fix_minus_oracle_old"]["per_seed"] = dict(zip(TAGS, d.round(4).tolist()))
    d2 = offs["obs"] - oo
    S["paired_obs_minus_oracle_old"] = stats(d2)

    lo, hi = ORACLE["predicted_fix_arm"]
    m = S["arms"]["fix"]["offset_stats"]
    S["closure_verdict"] = {
        "fix_mean_offset": m["mean"], "fix_sem": m["sem"],
        "predicted_band": [lo, hi],
        "within_predicted_band": bool(lo - 2 * m["sem"] <= m["mean"] <= hi + 2 * m["sem"]),
        "consistent_with_zero_2sigma": bool(abs(m["mean"]) <= 2 * m["sem"]),
    }

    (RESULTS / "obsdet_fix_summary.json").write_text(
        json.dumps(S, indent=2, default=float))
    print("wrote results/obsdet_fix_summary.json\n")
    for arm, _, lab in ARMS:
        st = S["arms"][arm]["offset_stats"]
        print(f"{arm:5s} {lab}")
        print(f"      mean {st['mean']:+.3f} +- {st['sem']:.3f} (sd {st['sd']:.3f})  "
              f"=> {st['sigma_from_zero']:.1f} sigma from zero")
    for k in ("paired_obs_minus_ctrl", "paired_fix_minus_obs", "paired_fix_minus_ctrl",
              "paired_fix_minus_oracle_old"):
        p = S[k]
        print(f"{k}: {p['mean']:+.3f} +- {p['sem']:.3f}")
    print("closure verdict:", json.dumps(S["closure_verdict"], indent=2))

    # -------------------------------------------------- figure 1: the ladder
    fig, ax = plt.subplots(figsize=(5.4, 3.6), dpi=300)
    cols = {"ctrl": RED, "obs": BLUE, "fix": AQUA}
    xs = {"ctrl": 0.0, "obs": 1.0, "fix": 2.0}
    rng = np.random.default_rng(7)
    jit = rng.uniform(-0.10, 0.10, len(TAGS))
    for i in range(len(TAGS)):
        ax.plot([xs[a] + jit[i] for a in ("ctrl", "obs", "fix")],
                [offs[a][i] for a in ("ctrl", "obs", "fix")],
                color=INK3, lw=0.5, alpha=0.35, zorder=2)
    for arm in ("ctrl", "obs", "fix"):
        ax.plot(np.full(len(TAGS), xs[arm]) + jit, offs[arm], "o", ms=3.0,
                color=cols[arm], alpha=0.55, zorder=3, mew=0)
        st = S["arms"][arm]["offset_stats"]
        ax.errorbar([xs[arm] + 0.28], [st["mean"]], yerr=[st["sem"]], fmt="D",
                    ms=5.5, color=cols[arm], ecolor=cols[arm], elinewidth=1.6,
                    capsize=3.5, zorder=5)
        ax.annotate(f"{st['mean']:+.2f}\n$\\pm${st['sem']:.2f}",
                    xy=(xs[arm] + 0.28, st["mean"]),
                    xytext=(xs[arm] + 0.38, st["mean"]),
                    fontsize=7.6, color=cols[arm], va="center", ha="left")
    ax.axhline(0.0, color=INK2, lw=0.9, zorder=1)
    ax.axhspan(lo, hi, color=YELLOW, alpha=0.14, zorder=0)
    ax.annotate("predicted estimator overhead\n(exact-likelihood oracle)",
                xy=(-0.42, 0.5 * (lo + hi)), fontsize=7.0, color=INK2,
                va="center", ha="left")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["gmd rule\n(ctrl)",
                        "observed-data\ndetection (obs)",
                        "+ observed sky\nwidth (fix, PR #335)"])
    ax.set_xlim(-0.5, 2.75)
    ax.set_ylabel(r"$H_0$ offset  [km s$^{-1}$ Mpc$^{-1}$]")
    ax.set_title("K=1 matched-mock closure ladder  (20 catalog realisations, paired)")
    ax.grid(True, alpha=0.55, axis="y")
    ax.set_axisbelow(True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"fig_obsfix_closure.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("wrote figs/fig_obsfix_closure.{pdf,png}")

    # ------------------------------------------------ figure 2: the waterfall
    d1 = S["paired_obs_minus_ctrl"]
    d2f = S["paired_fix_minus_obs"]
    fixm = S["arms"]["fix"]["offset_stats"]
    farr = ORACLE["farr_term"]
    pew = ORACLE["pe_mass_width_latent"]
    resid = fixm["mean"] - farr - pew
    steps = [
        ("ctrl arm\n(gmd rule)", S["arms"]["ctrl"]["offset_stats"]["mean"], "abs",
         S["arms"]["ctrl"]["offset_stats"]["sem"]),
        ("shared\nnoise draw\n(PR #334)", d1["mean"], "delta", d1["sem"]),
        ("observable\nsky width\n(PR #335)", d2f["mean"], "delta", d2f["sem"]),
        ("fix arm\n(this rerun)", fixm["mean"], "abs", fixm["sem"]),
        ("Farr\n$1/N_{\\rm eff}$", farr, "comp", None),
        ("mass-width\nlatents", pew, "comp", None),
        ("unattributed", resid, "comp", None),
    ]
    fig, ax = plt.subplots(figsize=(7.6, 3.4), dpi=300)
    x = np.arange(len(steps))
    level = 0.0
    for i, (lab, val, kind, err) in enumerate(steps):
        if kind == "abs":
            ax.bar(i, val, 0.62, color=(RED if i == 0 else AQUA), alpha=0.85, zorder=3)
            ax.errorbar([i], [val], yerr=[err], fmt="none", ecolor=INK, elinewidth=1.0,
                        capsize=2.5, zorder=5)
            ax.annotate(f"{val:+.2f}", xy=(i, val - err - 0.04), ha="center",
                        va="top", fontsize=7.8, color=INK)
            level = val
        elif kind == "delta":
            ax.bar(i, val, 0.62, bottom=level, color=BLUE, alpha=0.75, zorder=3)
            ax.errorbar([i], [level + val], yerr=[err], fmt="none", ecolor=INK,
                        elinewidth=1.0, capsize=2.5, zorder=5)
            ax.annotate(f"{val:+.2f}", xy=(i, level + val / 2), ha="center",
                        va="center", fontsize=7.8, color=INK)
            level = level + val
        else:  # component decomposition of the fix arm, from the oracle budget
            ax.bar(i, val, 0.45, color=YELLOW, alpha=0.75, zorder=3)
            ax.annotate(f"{val:+.2f}", xy=(i, val - 0.04), ha="center",
                        va="top", fontsize=7.4, color=INK2)
    ax.axhline(0.0, color=INK2, lw=0.9, zorder=1)
    ax.axvline(3.62, color=INK3, lw=0.7, ls=":")
    ax.annotate("oracle budget of the fix-arm residual",
                xy=(5.0, -1.45), fontsize=7.2, color=INK2, ha="center")
    ax.set_xticks(x)
    ax.set_xticklabels([s[0] for s in steps], fontsize=7.2)
    ax.set_ylim(-2.0, 0.06)
    ax.set_ylabel(r"$H_0$ offset  [km s$^{-1}$ Mpc$^{-1}$]")
    ax.set_title("Closure budget: where the $-1.57$ went")
    ax.grid(True, alpha=0.55, axis="y")
    ax.set_axisbelow(True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"fig_obsfix_budget.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("wrote figs/fig_obsfix_budget.{pdf,png}")


if __name__ == "__main__":
    main()
