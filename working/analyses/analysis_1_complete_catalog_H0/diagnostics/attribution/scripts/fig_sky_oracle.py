#!/usr/bin/env python3
"""Figures for the exact host-galaxy sky oracle and the pixelisation study.

  fig_sky_oracle.{png,pdf}
     (a) the anchor -- the oracle's `kde_pix` arm against darksirens' own per-event
         d ln Z_i/dH0, with darksirens' own PE Monte-Carlo error as the yardstick;
     (b) the per-event pixelisation substitution `delta_host - delta_pix` against the
         event's sky width, which is where the effect must live if it is the
         pixelisation;
     (c) the closure ladder in r: record -> after (b2)+(c2) -> minus the photo-z
         kernel -> minus the pixelisation -> what is left.

  fig_nside_curve.{png,pdf}   [--which nside]
     the matched-control offset and the measured per-event residual r against the
     survey resolution, with the oracle's prediction for nside 32 marked.

Usage: python scripts/fig_sky_oracle.py --which all
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


def fig_oracle(tracer="gal"):
    J = jload(RES / f"attr_sky_oracle_{tracer}.json")
    if J is None:
        print(f"[skip] no attr_sky_oracle_{tracer}.json")
        return
    D = np.load(RES / f"attr_sky_oracle_{tracer}.npz")
    ds = D["ds_score"]
    kde = D["score_kde_pix"]
    dpix = D["score_delta_pix"]
    dhost = D["score_delta_host"]
    sig = D["sig_mc"]
    sang = D["sigma_ang_deg"]
    colour = BLUE if tracer == "gal" else ORANGE

    fig, ax = plt.subplots(1, 3, figsize=(12.6, 3.9))

    # (a) anchor
    d = kde - ds
    ax[0].errorbar(ds, kde, yerr=0, fmt="o", ms=2.6, alpha=0.35, color=colour, lw=0)
    lim = [min(ds.min(), kde.min()), max(ds.max(), kde.max())]
    ax[0].plot(lim, lim, color=INK, lw=0.9)
    ax[0].set_xlabel(r"darksirens  $d\ln \hat Z_i/dH_0$")
    ax[0].set_ylabel(r"oracle (kde_pix)  $d\ln Z_i/dH_0$")
    v = J["arms"]["kde_pix"]["vs_darksirens"]
    ax[0].set_title(f"(a) anchor: {v['mean']:+.2e} $\\pm$ {v['sem']:.1e} per event\n"
                    f"rms {np.sqrt((d**2).mean()):.3e} vs darksirens' own MC "
                    f"{np.sqrt((sig**2).mean()):.3e}")
    _spines(ax[0])

    # (b) pixelisation substitution vs sky width
    s = dhost - dpix
    ax[1].axhline(0.0, color=INK, lw=0.9)
    ax[1].plot(sang, s, "o", ms=2.8, alpha=0.35, color=GREEN, lw=0)
    m = J["substitutions"]["pixelisation__host_minus_pix__delta_prior"]
    ax[1].axhline(m["mean"], color=GREEN, lw=1.2, ls="--")
    ax[1].axhspan(m["mean"] - m["sem"], m["mean"] + m["sem"], color=GREEN,
                  alpha=0.12, lw=0)
    ax[1].set_xlabel(r"$\sigma_{\rm ang}$  [deg]     (pixel side 1.83$^\circ$)")
    ax[1].set_ylabel(r"$\varsigma$(exact host sky) $-$ $\varsigma$(pixel sky)")
    q = np.percentile(np.abs(s - np.median(s)), 97.5)
    n_out = int((np.abs(s - np.median(s)) > 4 * q).sum())
    ax[1].set_ylim(np.median(s) - 4 * q, np.median(s) + 4 * q)
    ax[1].set_title(f"(b) the pixelisation term, per event\n"
                    f"mean {m['mean']:+.3e} $\\pm$ {m['sem']:.1e}"
                    + (f"   ({n_out} outside the frame)" if n_out else ""))
    _spines(ax[1])

    # (c) the ladder
    dlnmu = J["dlnmu_dH0"]
    rows = [("record (pre-fix)", None), ("after (b2)+(c2)", float(np.mean(kde)) - dlnmu),
            ("- photo-z kernel", float(np.mean(dpix)) - dlnmu),
            ("- pixelisation", float(np.mean(dhost)) - dlnmu)]
    pre = jload(RES / "closure_summary.json")
    r_pre = None
    if pre and pre.get("score_terms", {}).get(tracer, {}).get("before"):
        r_pre = pre["score_terms"][tracer]["before"].get("r_total")
    if r_pre is None:
        r_pre = -1.4499e-3 if tracer == "gal" else -1.8253e-3
    rows[0] = ("record (pre-fix)", r_pre)
    y = np.arange(len(rows))[::-1]
    vals = [v for _, v in rows]
    ax[2].axvline(0.0, color=INK, lw=1.0)
    ax[2].barh(y, vals, height=0.55, color=[GREY, colour, colour, GREEN])
    for yy, (lab, v) in zip(y, rows):
        ax[2].text(v + (2e-5 if v > 0 else -2e-5), yy, f"{v:+.3e}",
                   va="center", ha="left" if v > 0 else "right", fontsize=8)
    ax[2].set_yticks(y)
    ax[2].set_yticklabels([r for r, _ in rows], fontsize=8.5)
    ax[2].set_xlabel(r"$r = \langle d\ln Z_i/dH_0\rangle - d\ln\mu/dH_0$  per event"
                     "\n" r"[$10^{-3}$]")
    from matplotlib.ticker import FuncFormatter, MaxNLocator
    ax[2].xaxis.set_major_locator(MaxNLocator(4))
    ax[2].xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v*1e3:.1f}"))
    ax[2].set_title("(c) the closure ladder")
    ax[2].margins(x=0.28)
    _spines(ax[2])

    fig.suptitle(f"Exact host-galaxy sky oracle -- matched {tracer.upper()}, "
                 f"{J['n_events']} events, seed 100", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for ext in ("png", "pdf"):
        fig.savefig(FIGS / f"fig_sky_oracle_{tracer}.{ext}", dpi=220)
    plt.close(fig)
    print(f"wrote figs/fig_sky_oracle_{tracer}.{{png,pdf}}")


# per-event d2 logL/dH0^2 of the post-fix nside-32 matched controls, measured on
# their own grids (results/ctrl_{gal,agn}_matched.h5): the conversion from a
# per-event score residual to km/s/Mpc.
CURV = {"gal": -1.5781e-01 / 720, "agn": -6.5065e-01 / 280}


def fig_nside():
    S = jload(RES / "closure_summary.json")
    if S is None or not S.get("nside_study", {}).get("scans"):
        print("[skip] no nside scans yet")
        return
    sc = S["nside_study"]["scans"]
    surv = (S["nside_study"].get("surveys") or {}).get("surveys", {})
    fig, ax = plt.subplots(1, 2, figsize=(9.4, 4.0))
    for a, case, colour in ((ax[0], "gal", BLUE), (ax[1], "agn", ORANGE)):
        ns, off, lo, hi = [], [], [], []
        for n in (32, 64, 128):
            k = f"{case}_ns{n}"
            if k not in sc:
                continue
            r = sc[k]
            ns.append(n); off.append(r["offset"])
            lo.append(r["median"] - r["ci68"][0]); hi.append(r["ci68"][1] - r["median"])
        if not ns:
            continue
        a.axhline(0.0, color=INK, lw=1.0)
        a.errorbar(ns, off, yerr=[lo, hi], fmt="o-", ms=5, color=colour,
                   ecolor=colour, lw=1.3, capsize=0)
        a.set_xscale("log", base=2)
        a.set_xticks(ns); a.set_xticklabels([str(n) for n in ns])
        a.set_xlabel("survey HEALPix nside")
        a.set_ylabel(r"matched-control offset  $H_0 - 67.74$")
        # the oracle's prediction: the pixel-average -> exact-position substitution,
        # converted to H0 on this control's own per-event curvature.
        sk = (S.get("sky_oracle") or {}).get(case)
        d2 = CURV.get(case)
        if sk and d2:
            sub = sk["substitutions"]["pixelisation__host_minus_pix__delta_prior"]
            pred = off[0] + sub["mean"] / abs(d2)
            perr = sub["sem"] / abs(d2)
            a.axhline(pred, color=GREEN, lw=1.2, ls="--", zorder=1)
            a.axhspan(pred - perr, pred + perr, color=GREEN, alpha=0.13, lw=0,
                      zorder=0)
            a.annotate("sky oracle: the nside $\\to\\infty$ limit",
                       xy=(ns[-1], pred), xytext=(-4, 6),
                       textcoords="offset points", ha="right", va="bottom",
                       fontsize=7.8, color=GREEN)
        sizes = [f"{n}: {np.rad2deg(np.sqrt(4*np.pi/(12*n*n))):.2f}$^\\circ$"
                 for n in ns]
        a.set_title(f"{case.upper()} catalog     pixel side  " + "   ".join(sizes),
                    fontsize=8.6)
        _spines(a)
    fig.suptitle("Matched-host closure against survey resolution "
                 "(seed 100, after the (b2)+(c2) fixes)", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    for ext in ("png", "pdf"):
        fig.savefig(FIGS / f"fig_nside_curve.{ext}", dpi=220)
    plt.close(fig)
    print("wrote figs/fig_nside_curve.{png,pdf}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--which", default="all",
                    choices=["all", "oracle", "nside"])
    a = ap.parse_args(argv)
    FIGS.mkdir(exist_ok=True)
    if a.which in ("all", "oracle"):
        for t in ("gal", "agn"):
            fig_oracle(t)
    if a.which in ("all", "nside"):
        fig_nside()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
