#!/usr/bin/env python3
"""fig_selmu_oracle.{png,pdf} -- the exact selection oracle against darksirens.

  (a) d ln mu/dH0 across the scanned H0 range: the closed-form oracle (the KDE
      host measure darksirens actually conditions on, and its zero-bandwidth
      exact-host limit) against darksirens' injection Monte Carlo, per catalog.
  (b) the difference, injections - oracle, with the band each catalog's score
      residual would need on the selection side to be explained there.
  (c) F(z), the mass-integrated detection probability: the oracle against the
      GENERATOR's own population-branch detection bookkeeping (~1e8 proposals).
  (d) the closure ladder in r, extended by the selection-side term.

Usage: python scripts/fig_selmu_oracle.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
FIGS = ROOT / "figs"
TRUTH = 67.74

SURFACE = "#FFFFFF"
INK = "#1A1A1A"
INK_2 = "#4A4A4A"
INK_MUTED = "#9A9A9A"
BLUE = "#2C6E9B"
ORANGE = "#C4622D"
GREEN = "#3E7B54"
GREY = "#8A8A8A"

mpl.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.size": 9, "axes.labelsize": 9.5,
    "axes.titlesize": 10.0, "axes.edgecolor": INK_MUTED, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": INK_2, "ytick.color": INK_2,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "axes.linewidth": 0.8,
    "legend.frameon": False, "pdf.fonttype": 42,
})


def jload(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def _spines(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, color="#E6E6E6", lw=0.6)
    ax.set_axisbelow(True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default=str(FIGS))
    args = ap.parse_args(argv)
    S = jload(RES / "attr_selmu_summary.json")
    if S is None:
        raise SystemExit("run scripts/attr_selmu_summary.py first")

    fig, ax = plt.subplots(2, 2, figsize=(9.4, 7.0))
    ax = ax.ravel()

    # ---- (a) d ln mu/dH0 across H0 --------------------------------------------
    a = ax[0]
    _spines(a)
    for tr, col in (("gal", BLUE), ("agn", ORANGE)):
        rec = S["tracers"].get(tr)
        if not rec:
            continue
        H = np.asarray(rec["oracle"]["H0_grid"], float)
        a.plot(H, rec["oracle"]["dlnmu_kde"], "-", color=col, lw=1.6,
               label=f"{tr.upper()} oracle (KDE hosts)")
        a.plot(H, rec["oracle"]["dlnmu_delta"], "--", color=col, lw=1.1,
               label=f"{tr.upper()} oracle (exact hosts)")
        for lane, mk in (("targeted", "o"), ("popuni", "s")):
            g = rec["injections"].get(lane, {}).get("grid_comparison", {})
            if not g:
                continue
            hh = np.array([float(k) for k in g])
            vv = np.array([g[k]["inj"] for k in g])
            o = np.argsort(hh)
            a.plot(hh[o], vv[o], mk, ms=4.0, mfc="none", mew=1.0, color=col,
                   ls="none", label=f"{tr.upper()} injections ({lane})")
    a.axvline(TRUTH, color=INK_MUTED, lw=0.8, ls=":")
    a.set_xlabel(r"$H_0$  [km s$^{-1}$ Mpc$^{-1}$]")
    a.set_ylabel(r"$d\ln\mu/dH_0$")
    a.set_title("(a) the selection integral's slope", loc="left")
    a.legend(fontsize=7.0, ncol=1, loc="best")

    # ---- (b) injections - oracle ----------------------------------------------
    b = ax[1]
    _spines(b)
    b.axhline(0.0, color=INK, lw=0.9)
    for tr, col in (("gal", BLUE), ("agn", ORANGE)):
        rec = S["tracers"].get(tr)
        if not rec:
            continue
        # the estimator's OWN Monte-Carlo error on d ln mu/dH0 (Poisson bootstrap
        # over injections), from the targeted lane -- the analysis of record
        sd = rec["injections"].get("targeted", {}).get("mc_error_bootstrap_sd")
        if sd:
            b.axhspan(-sd, sd, color=col, alpha=0.11, lw=0)
            b.text(100.6, sd, f"  {tr.upper()}\n  $\\pm1\\sigma_{{\\rm MC}}$",
                   color=col, fontsize=6.5, va="center")
        for lane, mk in (("targeted", "o"), ("popuni", "s")):
            g = rec["injections"].get(lane, {}).get("grid_comparison", {})
            if not g:
                continue
            hh = np.array([float(k) for k in g])
            dd = np.array([g[k]["diff"] for k in g])
            o = np.argsort(hh)
            b.plot(hh[o], dd[o], mk + "-", ms=4.0, mfc="none", mew=1.0, lw=0.9,
                   color=col, label=f"{tr.upper()} ({lane})")
    tgt = S["targets"]
    b.axhline(-tgt["gal"]["r_exact_numerator"], color=BLUE, lw=1.2, ls="--")
    b.text(51, -tgt["gal"]["r_exact_numerator"],
           "  what GAL would need", color=BLUE, fontsize=7, va="bottom")
    b.axhline(-tgt["agn"]["r_record_postfix"], color=ORANGE, lw=1.2, ls="--")
    b.text(51, -tgt["agn"]["r_record_postfix"],
           "  what AGN would need", color=ORANGE, fontsize=7, va="top")
    b.axvline(TRUTH, color=INK_MUTED, lw=0.8, ls=":")
    b.set_xlim(48, 108)
    b.set_xlabel(r"$H_0$  [km s$^{-1}$ Mpc$^{-1}$]")
    b.set_ylabel(r"$\Delta\,d\ln\mu/dH_0$   (injections $-$ oracle)")
    b.set_title("(b) the estimator against the closed form", loc="left")
    b.legend(fontsize=7.5, loc="best")

    # ---- (c) F(z) against the generator ---------------------------------------
    c = ax[2]
    _spines(c)
    d = np.load(RES / "attr_selmu_agn.npz")
    bg, Gv, zk, bk = d["b_grid"], d["G"], d["zk"], d["b_zk"]
    zz = np.linspace(0.002, 0.45, 2000)
    Fz = np.interp(np.interp(zz, zk, bk), bg, Gv)
    c.plot(zz, Fz, "-", color=INK, lw=1.6, label=r"oracle $F(z)$ (closed form)")
    I = jload(RES / "attr_selmu_inj_agn_targeted.json") or \
        jload(RES / "attr_selmu_inj_gal_targeted.json")
    if I and I.get("pdet_z_empirical"):
        e = np.asarray(I["pdet_z_empirical"]["edges"], float)
        npro = np.asarray(I["pdet_z_empirical"]["n_proposed"], float)
        ndet = np.asarray(I["pdet_z_empirical"]["n_detected"], float)
        zc = 0.5 * (e[1:] + e[:-1])
        m = (npro > 2000) & (zc < 0.45) & (ndet >= 10)
        p = ndet[m] / npro[m]
        s = np.sqrt(np.maximum(p * (1 - p), 1e-12) / npro[m])
        c.errorbar(zc[m], p, yerr=s, fmt="o", ms=2.2, lw=0, elinewidth=0.6,
                   color=GREEN, alpha=0.65,
                   label="generator's own injection draws")
    c.set_yscale("log")
    c.set_ylim(3e-5, 1.6)
    c.set_xlim(0.0, 0.45)
    c.set_xlabel(r"$z$")
    c.set_ylabel(r"$F(z)=\langle P_{\rm det}\rangle_{\rm pop}$")
    c.set_title("(c) the mass-integrated detection probability", loc="left")
    c.legend(fontsize=7.5, loc="best")

    # ---- (d) the closure ladder, extended -------------------------------------
    e_ = ax[3]
    _spines(e_)
    rec = S["tracers"].get("gal", {})
    sel = None
    for lane in ("targeted",):
        v = rec.get("injections", {}).get(lane, {})
        if "delta_vs_oracle_kde_at_truth" in v:
            sel = v["delta_vs_oracle_kde_at_truth"]
    ch = jload(RES / "attr_chieff.json") or {}
    hw = jload(RES / "attr_hostw.json") or {}
    d_chi = (ch.get("tracers", {}).get("gal", {})
             .get("substitution_exact_minus_stored", {}).get("mean", 0.0))
    d_hw = (hw.get("tracers", {}).get("gal", {})
            .get("delta_r", {}).get("uniform_host_prior", {}).get("mean", 0.0))
    labels = ["record, pre-fix",
              "(c2) exact mass PE",
              "(b2) RA width",
              "oracle anchor offset",
              "photo-z kernel",
              "nside-32 pixelisation",
              "host prior (task 3)",
              r"$\chi_{\rm eff}$ clip (task 2)",
              "selection estimator (task 1)",
              "remaining"]
    vals = [-1.4491e-3, +5.6013e-4, +5.981e-5, -1.958e-5, +1.7378e-4, +9.063e-5,
            d_hw, d_chi, (sel if sel is not None else 0.0), 0.0]
    vals[-1] = sum(vals[:-1])
    cols = [GREY, BLUE, BLUE, GREY, GREEN, GREEN, ORANGE, ORANGE, ORANGE, INK]
    y = np.arange(len(labels))[::-1]
    e_.axvline(0.0, color=INK, lw=1.0)
    e_.barh(y, vals, height=0.62, color=cols)
    e_.set_yticks(y)
    e_.set_yticklabels(labels, fontsize=7.2)
    for yy, vv in zip(y, vals):
        e_.text(vv + (2.5e-5 if vv >= 0 else -2.5e-5), yy, f"{vv:+.2e}",
                va="center", ha="left" if vv >= 0 else "right", fontsize=6.4,
                color=INK_2)
    e_.set_xlabel(r"contribution to $r$ per event")
    e_.set_title("(d) the closure ladder, matched GAL", loc="left")
    e_.set_xlim(-1.9e-3, 1.1e-3)

    fig.tight_layout()
    od = Path(args.outdir)
    od.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(od / f"fig_selmu_oracle.{ext}", dpi=190,
                    bbox_inches="tight")
    print(f"Wrote {od/'fig_selmu_oracle.png'} / .pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
