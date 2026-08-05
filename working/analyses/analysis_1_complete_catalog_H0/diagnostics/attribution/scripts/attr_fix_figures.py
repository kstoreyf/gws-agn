#!/usr/bin/env python3
"""Figures for the ATTRIBUTION follow-up (tasks 1-3).

  attr_sampler_ratio      the sampler-vs-pdf log-ratio map + the predicted r
  fig_before_after_fix    matched-GAL / matched-AGN H0 posteriors, record vs the
                          exact-mass-PE reweighting
  attr_oracle             the quadrature oracle's three-variant attribution
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
FIGS = ROOT / "figs"
FIGS.mkdir(exist_ok=True)
COL = {"gal": "#1f77b4", "agn": "#d62728"}


def _save(fig, name):
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIGS / f"{name}.{ext}", dpi=160, bbox_inches="tight")
    print(f"wrote {FIGS/name}.png / .pdf")


# ---------------------------------------------------------------- task 1
def fig_sampler_ratio():
    d = np.load(RES / "attr_sampler_ratio.npz")
    S = json.load(open(RES / "attr_sampler_ratio.json"))
    m1e, qe = d["m1_edges"], d["q_edges"]
    H2, P, Ps, N = d["H2"], d["P_bin"], d["Ps_bin"], int(d["N"])
    lr_mc, lr_semi, sig = d["lr_mc"], d["lr_semi"], d["sig_lr"]

    fig = plt.figure(figsize=(19.0, 4.5))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.18, 1.0, 0.8, 0.85])

    # (a) the EXACT log-ratio map (no Monte-Carlo noise)
    ax = fig.add_subplot(gs[0, 0])
    Z = np.where(P > 0, lr_semi, np.nan)
    norm = matplotlib.colors.SymLogNorm(linthresh=1e-5, vmin=-0.02, vmax=0.02,
                                        base=10)
    pc = ax.pcolormesh(m1e, qe, Z.T, cmap="RdBu_r", norm=norm, shading="flat")
    cb = fig.colorbar(pc, ax=ax, ticks=[-1e-2, -1e-3, -1e-4, 0, 1e-4, 1e-3, 1e-2])
    cb.set_label(r"$\ln[\,q_{\rm sampler}/p_{\rm analytic}]$  (exact)")
    for x in (5.0, 8.0, 70.0, 80.0):
        ax.axvline(x, color="k", lw=0.7, ls=":")
    ax.set_xscale("log")
    ax.set_xticks([5, 8, 10, 20, 35, 50, 70, 80])
    ax.set_xticklabels(["5", "8", "10", "20", "35", "50", "70", "80"])
    ax.set_xlim(m1e[0], m1e[-1])
    ax.set_xlabel(r"$m_{1,\rm src}\ [M_\odot]$"); ax.set_ylabel(r"$q$")
    ax.set_title(r"(a) exact sampler density / darksirens' powerlaw+peak"
                 "\n" r"dotted: $m_{\min}$, $m_{\min}+\delta m_{\min}$, "
                 r"$m_{\max}-\delta m_{\max}$, $m_{\max}$", fontsize=9.5)

    # (b) m1 marginal: exact curve + coarse-binned draws
    ax = fig.add_subplot(gs[0, 1])
    Hm1, Pm1, Psm1 = d["Hm1"], d["P_m1"], d["Ps_m1"]
    c = 0.5 * (m1e[1:] + m1e[:-1])
    ok2 = Pm1 > 0
    ax.plot(c[ok2], np.log(Psm1[ok2] / Pm1[ok2]), "-", color="#d62728", lw=1.8,
            zorder=3, label="exact sampler density (closed form)")
    # merge adjacent bins to a target count so the Poisson error is meaningful
    tgt = 3e6
    grp, acc, start = [], 0.0, 0
    for j in range(Hm1.size):
        acc += Hm1[j]
        if acc >= tgt or j == Hm1.size - 1:
            grp.append((start, j + 1)); acc = 0.0; start = j + 1
    cc, yy, ee = [], [], []
    for a_, b_ in grp:
        n = Hm1[a_:b_].sum(); p_ = Pm1[a_:b_].sum()
        if n < 1000 or p_ <= 0:
            continue
        cc.append(np.average(c[a_:b_], weights=np.maximum(Hm1[a_:b_], 1e-9)))
        yy.append(np.log((n / N) / p_)); ee.append(1.0 / np.sqrt(n))
    ax.errorbar(cc, yy, yerr=ee, fmt="o", ms=4, lw=1.0, color="0.25", zorder=4,
                label=r"$1.2\times10^{8}$ draws, rebinned ($\pm1\sigma$)")
    ax.axhline(0, color="k", lw=0.8)
    axt = ax.twinx()
    try:
        mp = np.load(RES / "attr_mass_pe_gal_s100.npz")
        h, be = np.histogram(mp["sel_m1src"], bins=np.linspace(2, 90, 80),
                             weights=mp["sel_w"])
        axt.step(0.5 * (be[1:] + be[:-1]), h / h.max(), where="mid", color="#1f77b4",
                 lw=1.0, alpha=0.55)
        axt.fill_between(0.5 * (be[1:] + be[:-1]), 0, h / h.max(), step="mid",
                         color="#1f77b4", alpha=0.13)
        axt.set_ylim(0, 3.2); axt.set_yticks([])
        axt.text(0.03, 0.30, "detected-set\n" r"$m_{1,\rm src}$ weight",
                 transform=axt.transAxes, fontsize=7.5, color="#1f77b4")
    except FileNotFoundError:
        pass
    ax.set_xscale("log")
    ax.set_xticks([5, 8, 10, 20, 35, 50, 70, 80])
    ax.set_xticklabels(["5", "8", "10", "20", "35", "50", "70", "80"])
    ax.set_xlim(4.6, 92)
    ax.set_ylim(-0.002, 0.016)
    ax.set_zorder(axt.get_zorder() + 1); ax.patch.set_visible(False)
    ax.set_xlabel(r"$m_{1,\rm src}\ [M_\odot]$")
    ax.set_ylabel(r"$\ln[\,q_{\rm sampler}/p_{\rm analytic}]$  (marginal)")
    ax.set_title("(b) the mismatch is confined to the low-mass taper,\n"
                 "where the detected set has no weight", fontsize=10)
    ax.legend(fontsize=7.5, frameon=False, loc="upper right")

    # (c) validation: pulls of the draws against the closed form
    ax = fig.add_subplot(gs[0, 2])
    ok = (H2 >= 50) & (P > 0)
    pull = ((lr_mc - lr_semi) / sig)[ok]
    ax.hist(pull, bins=np.linspace(-5, 5, 61), density=True, color="0.7",
            edgecolor="0.4", lw=0.4)
    xg = np.linspace(-5, 5, 400)
    ax.plot(xg, np.exp(-0.5 * xg ** 2) / np.sqrt(2 * np.pi), "r-", lw=1.5)
    v = S["draws"]["validation_semi_vs_mc"]
    ax.text(0.03, 0.97, f"{v['n_bins_used']} bins\nmean {v['pull_mean']:+.3f}\n"
                        f"sd {v['pull_sd']:.3f}\nmax $|\\cdot|$ {v['pull_absmax']:.1f}",
            transform=ax.transAxes, va="top", fontsize=8)
    ax.set_xlabel(r"$(\ln$ ratio$_{\rm MC} - \ln$ ratio$_{\rm exact})/\sigma_{\rm Poisson}$")
    ax.set_ylabel("density")
    ax.set_title("(c) the closed form IS the sampler", fontsize=10)

    # (d) prediction
    ax = fig.add_subplot(gs[0, 3])
    labels, vals, errs, cols = [], [], [], []
    for t in ("gal", "agn"):
        pr = S["prediction"][t]
        labels.append(f"{t.upper()}\nmass"); vals.append(abs(pr["mass"]["delta"]))
        errs.append(pr["mass"]["delta_sem_injections"]); cols.append(COL[t])
        labels.append(f"{t.upper()}\ntotal"); vals.append(abs(pr["tot"]["delta"]))
        errs.append(pr["tot"]["delta_sem_injections"]); cols.append(COL[t])
    x = np.arange(len(labels))
    ax.bar(x, vals, yerr=errs, color=cols, alpha=0.85, capsize=3)
    ax.axhline(0.97e-3, color="k", lw=1.4, ls="--")
    ax.text(len(labels) - 0.4, 0.97e-3 * 1.25, "unexplained $|r|$ = 9.7e-4",
            ha="right", fontsize=8.5)
    ax.axhline(1.45e-3, color="0.4", lw=1.0, ls=":")
    ax.text(len(labels) - 0.4, 1.45e-3 * 1.25, "total $|r|$ = 1.45e-3", ha="right",
            fontsize=8, color="0.4")
    ax.set_yscale("log"); ax.set_ylim(1e-10, 1e-2)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel(r"$|E_q[\varsigma]-E_p[\varsigma]|$  (per event)")
    ax.set_title("(d) this channel's contribution to $r$:\n"
                 "5 orders of magnitude too small", fontsize=10)
    _save(fig, "attr_sampler_ratio")


# ---------------------------------------------------------------- task 2
def _post(tag):
    with h5py.File(RES / f"{tag}.h5", "r") as f:
        g = np.asarray(f["H0_grid"][:]); ll = np.asarray(f["log_likelihood"][:])
    p = np.exp(ll - ll.max()); p /= np.trapz(p, g)
    j = json.load(open(RES / f"{tag}.json"))
    return g, p, j


def fig_before_after():
    pairs = [("gal", "ctrl_gal_matched", "fix_named_defect_gal",
              "fix_named_defect_gal_m1"),
             ("agn", "ctrl_agn_matched", "fix_named_defect_agn",
              "fix_named_defect_agn_m1")]
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6))
    for ax, (t, rec, fix, fix1) in zip(axes, pairs):
        for tag, ls, lw, lab in ((rec, "-", 2.0, "record (stored PE)"),
                                 (fix, "-", 2.0, r"reweighted to exact $p(m_1,m_2\,|\,obs)$"),
                                 (fix1, "--", 1.2, r"reweighted, $m_1$ only")):
            if not (RES / f"{tag}.h5").exists():
                continue
            g, p, j = _post(tag)
            col = {"record (stored PE)": "0.35"}.get(lab, COL[t])
            if lab.endswith("only"):
                col = "#2ca02c"
            med = j["H0"]["median"]
            ax.plot(g, p, ls, color=col, lw=lw,
                    label=f"{lab}\n  median {med:.3f}  ({med-67.74:+.2f})")
            ax.axvline(med, color=col, lw=0.9, ls=":")
        ax.axvline(67.74, color="k", lw=1.4)
        ax.text(67.9, ax.get_ylim()[1] * 0.96, "truth 67.74", fontsize=8, va="top")
        ax.set_xlim(55, 80)
        ax.set_xlabel(r"$H_0$  [km s$^{-1}$ Mpc$^{-1}$]")
        ax.set_ylabel("posterior density")
        ax.set_title(f"matched-{t.upper()} control", fontsize=11)
        ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    fig.suptitle("Named mass-measurement defect corrected by reweighting; "
                 "remaining residual under investigation", fontsize=11)
    _save(fig, "fig_before_after_fix")


# ---------------------------------------------------------------- task 3
ARMS = ["kde_gauss", "kde_exact", "delta_gauss", "delta_exact"]
ARM_LAB = {"kde_gauss": "KDE prior\n+ stored Gaussian\n(the analysis of record)",
           "kde_exact": "KDE prior\n+ exact masses",
           "delta_gauss": "exact catalog\n+ stored Gaussian",
           "delta_exact": "exact catalog\n+ exact masses\n(fully exact)"}


def fig_oracle():
    tr = [t for t in ("gal", "agn") if (RES / f"attr_oracle_{t}.json").exists()]
    J = {t: json.load(open(RES / f"attr_oracle_{t}.json")) for t in tr}
    D = {t: np.load(RES / f"attr_oracle_{t}.npz") for t in tr}
    fig, axes = plt.subplots(1, 4, figsize=(20.0, 4.7))

    # (a) per-event validation against darksirens
    ax = axes[0]
    for t in tr:
        x = D[t]["ds_score"] * 1e3; y = D[t]["score_kde_gauss"] * 1e3
        ax.plot(x, y, ".", ms=3, color=COL[t], alpha=0.55,
                label=f"matched {t.upper()} (n={x.size})")
    lim = np.array([-0.2, 0.25]) * 1e3
    ax.plot(lim, lim, "k-", lw=0.8); ax.set_xlim(*lim); ax.set_ylim(*lim)
    txt = []
    for t in tr:
        v = J[t]["arms"]["kde_gauss"]["vs_darksirens"]
        r = D[t]["score_kde_gauss"] - D[t]["ds_score"]
        mc = float(np.sqrt((D[t]["sig_mc_full"] ** 2).mean()))
        txt.append(f"{t.upper()}: mean {v['mean']*1e3:+.3f}$\\pm${v['sem']*1e3:.3f}"
                   f"    rms {np.sqrt((r**2).mean())*1e3:.2f} vs "
                   f"darksirens' own MC {mc*1e3:.2f}")
    ax.text(0.03, 0.97, "\n".join(txt), transform=ax.transAxes, va="top", fontsize=7.5)
    ax.set_xlabel(r"darksirens $d\ln Z_i/dH_0\ \times10^{3}$")
    ax.set_ylabel(r"oracle, KDE + stored Gaussian $\times10^{3}$")
    ax.set_title("(a) the quadrature oracle reproduces darksirens\n"
                 "per event, to within darksirens' own MC error", fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc="lower right")

    # (b) r per arm
    ax = axes[1]
    xx = np.arange(len(ARMS))
    for i, t in enumerate(tr):
        y = [J[t]["arms"][a]["r"]["mean"] * 1e3 for a in ARMS]
        ax.plot(xx, y, "o-", color=COL[t], label=f"matched {t.upper()}")
        ax.axhline(J[t]["darksirens"]["r_subset"]["mean"] * 1e3, color=COL[t],
                   lw=0.9, ls=":")
        for k, v in enumerate(y):
            ax.annotate(f"{v:+.2f}", (xx[k], v), textcoords="offset points",
                        xytext=(0, 7 if i == 0 else -14), ha="center", fontsize=7.5,
                        color=COL[t])
    ax.axhline(0, color="k", lw=0.9)
    ax.set_xticks(xx); ax.set_xticklabels([ARM_LAB[a] for a in ARMS], fontsize=7)
    ax.set_ylim(-3.6, 0.6)
    ax.set_ylabel(r"$r=\langle d\ln Z_i/dH_0\rangle-d\ln\mu/dH_0\ \times10^{3}$")
    ax.set_title("(b) closed-form attribution\n"
                 "dotted: darksirens' own $r$ on the same events", fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc="lower right")

    # (c) paired substitutions
    ax = axes[2]
    subs = [("mass_model_exact_minus_stored__kde_prior",
             "exact mass\nlikelihood"),
            ("prior_delta_minus_kde__stored_masses",
             "zero-bandwidth\ncatalog prior"),
            ("fully_exact_minus_anchor", "both")]
    S = json.load(open(RES / "attr_fix_summary.json"))["task3_oracle"]
    xx = np.arange(len(subs))
    for i, t in enumerate(tr):
        y = [S[t]["substitutions"][k]["mean"] * 1e3 for k, _ in subs]
        e = [S[t]["substitutions"][k]["sem"] * 1e3 for k, _ in subs]
        ax.errorbar(xx + (i - 0.5) * 0.1, y, yerr=e, fmt="s", ms=7, color=COL[t],
                    capsize=4, label=f"matched {t.upper()}")
    ax.axhline(0, color="k", lw=0.9)
    ax.set_xticks(xx); ax.set_xticklabels([l for _, l in subs], fontsize=8)
    ax.set_ylabel(r"paired $\Delta$ score, per event $\times10^{3}$")
    ax.set_title("(c) what each substitution is worth\n(paired, no Monte-Carlo error)",
                 fontsize=10)
    ax.legend(fontsize=8, frameon=False)

    # (d) the closure ladder for matched GAL
    ax = axes[3]
    if "gal" in tr:
        r0 = J["gal"]["darksirens"]["r_subset"]["mean"] * 1e3
        d1 = S["gal"]["substitutions"][
            "mass_model_exact_minus_stored__kde_prior"]["mean"] * 1e3
        d2 = S["gal"]["substitutions"][
            "prior_delta_minus_kde__stored_masses"]["mean"] * 1e3
        lab = ["record\n$r$", "population\nsampler\n(+0.00001)",
               "exact\nmass PE", "zero-bw\ncatalog", "remaining"]
        steps = [1.3e-5, d1, d2]
        lvl = [r0]
        for st in steps:
            lvl.append(lvl[-1] + st)
        ax.bar(0, r0, color="0.55", width=0.62)
        for k, st in enumerate(steps):
            ax.bar(k + 1, st, bottom=lvl[k], color="#2ca02c", width=0.62)
        ax.bar(4, lvl[-1], color=COL["gal"], width=0.62)
        for k, st in enumerate(steps):
            ax.annotate(f"{st:+.3f}", (k + 1, lvl[k] + 0.5 * st),
                        textcoords="offset points", xytext=(26, 0), ha="center",
                        fontsize=8, color="#2ca02c")
        ax.annotate(f"{r0:.3f}", (0, r0), textcoords="offset points",
                    xytext=(0, -14), ha="center", fontsize=9)
        ax.annotate(f"{lvl[-1]:.3f}", (4, lvl[-1]), textcoords="offset points",
                    xytext=(0, -14), ha="center", fontsize=9)
        ax.set_ylim(-1.75, 0.25)
        ax.axhline(0, color="k", lw=0.9)
        ax.set_xticks(range(5)); ax.set_xticklabels(lab, fontsize=7.5)
        ax.set_ylabel(r"$r\ \times10^{3}$  (per event)")
        ax.set_title("(d) closure of $r$, matched GAL:\n"
                     "39 % named defect, 10 % KDE kernel, 52 % unexplained",
                     fontsize=10)
    _save(fig, "attr_oracle")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="all")
    a = ap.parse_args()
    w = a.which
    if w in ("all", "ratio"):
        fig_sampler_ratio()
    if w in ("all", "fix"):
        fig_before_after()
    if w in ("all", "oracle"):
        fig_oracle()


if __name__ == "__main__":
    main()
