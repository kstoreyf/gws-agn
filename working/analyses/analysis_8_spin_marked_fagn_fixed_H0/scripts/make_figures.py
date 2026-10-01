#!/usr/bin/env python3
"""Production figures for analysis 8 — the marked multitracer run on seed 100.

    python scripts/make_figures.py          # writes all four figures, pdf + png

Deterministic: it reads the recorded Gate-C scans and the per-event
decomposition and computes nothing new.  Every number it draws is printed to
stdout beside the value recorded in the corresponding JSON summary, so the
figures can be diffed against results/arm_*.json without opening them.

  1. fig_joint_f_dmu            arm J, the joint (f_AGN, Dmu_chi) posterior,
                                90 % highest-posterior-density region, with
                                BOTH truths marked: planted (0.300, +0.100)
                                and realised (0.295, +0.111924).
  2. fig_ablation_fagn          p(f_AGN) for spatial-only (S), intrinsic-only
                                (I) and joint (J) on one axis, full prior range
                                plus a zoom on the S/J peaks.
  3. fig_ablation_dmu           p(Dmu_chi) for intrinsic-only (I) and joint (J).
  4. fig_event_evidence_plane   ln BF_spatial against ln BF_intrinsic per
                                event, by true host label and by the inferred
                                P_i(AGN).

Reductions, all of them the ones the production summariser used:

  * flat priors on both axes, so the posterior is the exponentiated
    log-likelihood; guard-rejected cells carry logL = -inf and therefore zero
    density (23 such cells in arm J, all at f >= 0.750 and Dmu >= 0.2350, which
    is off the frame of figure 1 -- their bounded posterior mass is 3.9e-104);
  * marginals by trapezoid integration of that density over the other axis,
    exactly scripts/gate_c_three_arms.py::_marginals;
  * medians and equal-tailed credible intervals from the trapezoid CDF of the
    marginal, exactly analysis_2/scripts/scan_h0f.py::marginal_ci, which is
    what figstyle.quantiles_1d implements;
  * 2-D regions are highest-posterior-density, figstyle.hpd_levels_2d.

House conventions, from ../../../paper/scripts/figstyle.py:

  * that module is the visual system -- ONECOL / TWOCOL widths, rcParams,
    palette, truth ink.  Nothing here invents a colour.
  * EVERY plotted interval, band and contour is 90 %.  The 68 % numbers are
    printed to stdout and belong in text and tables, never on an axis.
  * 1-D panels show normalised posterior densities with density units on the
    y axis; nothing is scaled to its peak.
  * arms carry fixed identities: joint = blue (slot 1), intrinsic = orange
    (slot 2), spatial = aqua (slot 3).  Figure 3 uses only the validated
    slot 1 + slot 2 pair.  Host types keep the campaign's assignment from
    paper/scripts/fig_pure_tracer.py: GAL = blue, AGN = orange.
  * the P_i(AGN) ramp in figure 4 is ColorBrewer Oranges truncated to
    [0.45, 1.0]: one hue, light to dark, in the AGN identity hue so that
    "more orange" reads as "more AGN-like" next to panel (a), and with its
    lightest step at 2.12:1 against the page -- the same floor figstyle's own
    ordered ramp clears (2.11:1).
  * quoted numbers sit in a left-aligned block above the axes rather than in a
    box dropped on the posterior, so nothing the reader has to judge by eye is
    covered by something they could have read instead.

Figures are written to ../figs as both .pdf and .png.  figstyle.save() is NOT
used: it writes into the paper's figure directory, which this analysis does not
own.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
A8 = HERE.parent
RESULTS = A8 / "results"
FIGS = A8 / "figs"
WORKING = A8.parent.parent
PAPER_SCRIPTS = WORKING / "paper" / "scripts"

sys.path.insert(0, str(PAPER_SCRIPTS))
import figstyle as fs  # noqa: E402  (path has to be set first)

Z90 = float(norm.ppf(0.95))          # 1.6449: a 1-sigma error bar -> 90 %

ARM_C = {"J": fs.C["blue"], "I": fs.C["orange"], "S": fs.C["aqua"]}
ARM_NAME = {"J": "joint (arm J)", "I": "intrinsic-only (arm I)",
            "S": "spatial-only (arm S)"}
GAL_C, AGN_C = fs.C["blue"], fs.C["orange"]
P_CMAP = LinearSegmentedColormap.from_list(
    "agn_oranges", plt.get_cmap("Oranges")(np.linspace(0.45, 1.0, 256)))
LBLBOX = dict(facecolor="white", edgecolor="none", alpha=0.72, pad=1.0)

READS: list[str] = []


def _read(path: Path) -> Path:
    READS.append(str(path))
    return path


def save(fig, name: str) -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        out = FIGS / f"{name}.{ext}"
        fig.savefig(out)
        print(f"    wrote {out}")
    plt.close(fig)


def sci(v: float) -> str:
    """LaTeX 4.9e-11 -> $4.9 \\times 10^{-11}$."""
    m, e = f"{v:.1e}".split("e")
    return rf"${m}\times10^{{{int(e)}}}$"


# ---------------------------------------------------------------------------
# loading and the production reduction
# ---------------------------------------------------------------------------
def load_2d(tag):
    """(f, mu, normalised 2-D density, rejected mask, json summary)."""
    with h5py.File(_read(RESULTS / f"{tag}.h5"), "r") as h:
        f = np.asarray(h["f_grid"][:], float)
        mu = np.asarray(h["mu_chi_c2_grid"][:], float)
        ll = np.asarray(h["log_likelihood"][:], float)
        rej = np.asarray(h["guard/rejected"][:], bool)
    js = json.loads(_read(RESULTS / f"{tag}.json").read_text())
    fin = np.isfinite(ll)
    P = np.where(fin, np.exp(ll - ll[fin].max()), 0.0)
    P /= np.trapz(np.trapz(P, mu, axis=1), f)
    return f, mu, P, rej, js


def marginal_logp(P, x_other, axis):
    """log of the flat-prior marginal, the production reduction."""
    m = np.trapz(P, x_other, axis=axis)
    with np.errstate(divide="ignore"):
        return np.log(m)


def load_1d(tag):
    with h5py.File(_read(RESULTS / f"{tag}.h5"), "r") as h:
        f = np.asarray(h["f_grid"][:], float)
        ll = np.asarray(h["log_likelihood"][:], float)
    js = json.loads(_read(RESULTS / f"{tag}.json").read_text())
    return f, ll, js


def summarise(x, logp):
    """density + median + equal-tailed 68 / 90 intervals + P(x <= 0)."""
    p = fs.posterior_1d(x, logp)
    q = fs.quantiles_1d(x, logp, (0.05, 0.16, 0.5, 0.84, 0.95))
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (p[1:] + p[:-1]) * np.diff(x))])
    cdf /= cdf[-1]
    return {"x": x, "p": p, "median": float(q[2]),
            "ci68": [float(q[1]), float(q[3])],
            "ci90": [float(q[0]), float(q[4])],
            "p_le_zero": float(np.interp(0.0, x, cdf))}


def check(label, got, want, tol=1e-9):
    d = abs(got - want)
    print(f"      {label:<34s} drawn {got: .6f}   json {want: .6f}   "
          f"|d| {d:.2e} {'OK' if d <= tol else 'MISMATCH'}")


def band(ax, s, color, alpha=0.14):
    """Shade the 90 % credible interval under a density curve."""
    xs, ys = fs.ci_band(s["x"], s["p"], s["ci90"][0], s["ci90"][1])
    ax.fill_between(xs, 0.0, ys, color=color, alpha=alpha, lw=0, zorder=2)


# ---------------------------------------------------------------------------
# 1. the joint posterior
# ---------------------------------------------------------------------------
def fig_joint(J):
    f, mu, P, rej, js = J
    t = js["truth"]
    fm = summarise(f, marginal_logp(P, mu, 1))
    mm = summarise(mu, marginal_logp(P, f, 0))
    rho = js["posterior_moments"]["correlation"]
    _, l90 = fs.hpd_levels_2d(f, mu, P)
    mask = P >= l90
    xlo, xhi = f[mask.any(axis=1)].min(), f[mask.any(axis=1)].max()
    ylo, yhi = mu[mask.any(axis=0)].min(), mu[mask.any(axis=0)].max()
    px, py = 0.15 * (xhi - xlo), 0.20 * (yhi - ylo)
    xlim = (xlo - px, xhi + px)
    ylim = (ylo - py, yhi + py)
    mp = js["map"]
    i, j = np.unravel_index(np.argmax(np.where(np.isfinite(P), P, -np.inf)),
                            P.shape)

    print("\n  [1] fig_joint_f_dmu  (arm J, 90 % HPD region)")
    check("f_AGN median", fm["median"], js["f"]["median"])
    check("f_AGN 90 % low", fm["ci90"][0], js["f"]["ci90"][0])
    check("f_AGN 90 % high", fm["ci90"][1], js["f"]["ci90"][1])
    check("dmu_chi median", mm["median"], js["dmu_chi"]["median"])
    check("dmu_chi 90 % low", mm["ci90"][0], js["dmu_chi"]["ci90"][0])
    check("dmu_chi 90 % high", mm["ci90"][1], js["dmu_chi"]["ci90"][1])
    check("MAP f_AGN (grid argmax)", float(f[i]), mp["f"])
    check("MAP dmu_chi (grid argmax)", float(mu[j]), mp["dmu_chi"])
    print(f"      68 % (text only)  f {fm['ci68'][0]:.6f} {fm['ci68'][1]:.6f}"
          f"   dmu {mm['ci68'][0]:.6f} {mm['ci68'][1]:.6f}")
    print(f"      90 % HPD level {l90:.6g} of peak density {P.max():.6g}; "
          f"{int(mask.sum())} of {P.size} cells")
    print(f"      90 % region spans f [{xlo:.3f}, {xhi:.3f}], "
          f"dmu [{ylo:.4f}, {yhi:.4f}] (grid nodes)")
    print(f"      frame x {xlim[0]:.4f} {xlim[1]:.4f}   "
          f"y {ylim[0]:.4f} {ylim[1]:.4f}")
    print(f"      truths drawn: planted ({t['f_agn_planted']:.3f}, "
          f"{t['dmu_chi_planted']:+.6f})   realised "
          f"({t['f_agn_realised']:.3f}, {t['dmu_chi_realised']:+.6f}) "
          f"+/- {Z90 * t['dmu_chi_realised_err']:.6f} (90 %, "
          f"{t['dmu_chi_realised_err']:.6f} x {Z90:.4f})")
    print(f"      correlation {rho:+.4f}; {int(rej.sum())} guard-rejected "
          f"cells, all outside the frame")

    fig, ax = plt.subplots(figsize=(fs.ONECOL, 3.45))
    fig.subplots_adjust(left=0.175, right=0.985, bottom=0.125, top=0.835)
    c = ARM_C["J"]
    ax.contourf(f, mu, P.T, levels=[l90, P.max() * 1.01], colors=[c],
                alpha=0.19, zorder=2)
    ax.contour(f, mu, P.T, levels=[l90], colors=c, linewidths=1.5, zorder=3)

    ax.plot([mp["f"]], [mp["dmu_chi"]], marker="s", ms=3.6, ls="none",
            color=c, mec="white", mew=0.5, zorder=5)
    ax.plot([t["f_agn_planted"]], [t["dmu_chi_planted"]], marker="+", ms=9,
            mew=1.6, ls="none", color=fs.INK, zorder=6)
    ax.errorbar([t["f_agn_realised"]], [t["dmu_chi_realised"]],
                yerr=[Z90 * t["dmu_chi_realised_err"]], fmt="o", ms=4.0,
                mfc="white", mec=fs.INK, mew=1.2, ecolor=fs.INK,
                elinewidth=1.1, zorder=6)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel(r"$f_{\rm AGN}$")
    ax.set_ylabel(r"$\Delta\mu_\chi$")
    ax.set_title(
        f"seed 100;  medians, 90 % marginals;  $\\rho = {rho:+.3f}$\n"
        f"$f_{{\\rm AGN}} = {fm['median']:.3f}\\,"
        f"[{fm['ci90'][0]:.3f},\\,{fm['ci90'][1]:.3f}]$\n"
        f"$\\Delta\\mu_\\chi = {mm['median']:.4f}\\,"
        f"[{mm['ci90'][0]:.4f},\\,{mm['ci90'][1]:.4f}]$",
        fontsize=7.0, color=fs.INK2, linespacing=1.6)
    ax.legend(handles=[
        Patch(facecolor=c, alpha=0.19, edgecolor=c, lw=1.5,
              label="arm J (joint), 90 %"),
        Line2D([], [], color=c, marker="s", ms=3.6, ls="none",
               label="arm J MAP"),
        Line2D([], [], color=fs.INK, marker="+", ms=8, mew=1.6, ls="none",
               label="planted truth"),
        Line2D([], [], color=fs.INK, marker="o", ms=4.0, mfc="white", mew=1.2,
               ls="-", lw=1.1, label="realised truth, 90 %"),
    ], loc="upper right", fontsize=6.8, labelspacing=0.42,
        borderaxespad=0.35, handlelength=1.4)
    save(fig, "fig_joint_f_dmu")


# ---------------------------------------------------------------------------
# 2. p(f_AGN): spatial-only, intrinsic-only, joint
# ---------------------------------------------------------------------------
def fig_ablation_fagn(J, I, S):
    fJ, muJ, PJ, _, jsJ = J
    fI, muI, PI, _, jsI = I
    fS, llS, jsS = S
    s = {"J": summarise(fJ, marginal_logp(PJ, muJ, 1)),
         "I": summarise(fI, marginal_logp(PI, muI, 1)),
         "S": summarise(fS, llS)}
    js = {"J": jsJ, "I": jsI, "S": jsS}
    t = jsJ["truth"]

    print("\n  [2] fig_ablation_fagn  (p(f_AGN), three arms)")
    for k in ("S", "I", "J"):
        print(f"    {ARM_NAME[k]}")
        check("median", s[k]["median"], js[k]["f"]["median"])
        check("90 % low", s[k]["ci90"][0], js[k]["f"]["ci90"][0])
        check("90 % high", s[k]["ci90"][1], js[k]["f"]["ci90"][1])
        print(f"      68 % (text only)   {s[k]['ci68'][0]:.6f} "
              f"{s[k]['ci68'][1]:.6f}   width "
              f"{s[k]['ci68'][1] - s[k]['ci68'][0]:.4f}"
              f"   90 % width {s[k]['ci90'][1] - s[k]['ci90'][0]:.4f}")
        print(f"      peak density {s[k]['p'].max():.3f} per unit f at "
              f"f = {s[k]['x'][s[k]['p'].argmax()]:.3f};  density at f = 0 "
              f"{s[k]['p'][0]:.3g}, at f = 1 {s[k]['p'][-1]:.3g}")
    print(f"      truths drawn: planted {t['f_agn_planted']:.3f}, "
          f"realised {t['f_agn_realised']:.3f}  (0.005 apart: inside one line "
          f"width in panel a, resolved in panel b)")

    fig, axes = plt.subplots(1, 2, figsize=(fs.TWOCOL, 2.75), sharey=True)
    fig.subplots_adjust(left=0.085, right=0.995, bottom=0.155, top=0.905,
                        wspace=0.07)
    for ax, xlim, ticks, title in (
            (axes[0], (0.0, 1.0), [0.0, 0.2, 0.4, 0.6, 0.8],
             "(a) full prior range"),
            (axes[1], (0.10, 0.45), [0.10, 0.15, 0.20, 0.25, 0.30, 0.35,
                                     0.40, 0.45],
             "(b) zoom: spatial and joint")):
        for k in ("I", "S", "J"):
            band(ax, s[k], ARM_C[k])
            ax.plot(s[k]["x"], s[k]["p"], color=ARM_C[k], lw=1.6,
                    zorder=4 if k == "J" else 3)
        ax.axvline(t["f_agn_planted"], color=fs.TRUTH, lw=0.9, ls=(0, (3, 2)),
                   alpha=0.75, zorder=5)
        ax.axvline(t["f_agn_realised"], color=fs.TRUTH, lw=0.9,
                   ls=(0, (1, 1.6)), alpha=0.75, zorder=5)
        ax.set_xlim(*xlim)
        ax.set_ylim(bottom=0.0)
        ax.set_xticks(ticks)
        ax.set_xlabel(r"$f_{\rm AGN}$")
        ax.set_title(title, fontsize=8.0, color=fs.INK2)
    axes[0].set_ylabel(r"$p(f_{\rm AGN})$   [per unit $f_{\rm AGN}$]")
    axes[0].legend(handles=[
        Line2D([], [], color=ARM_C[k], lw=1.6, label=ARM_NAME[k])
        for k in ("S", "I", "J")
    ] + [
        Line2D([], [], color=fs.TRUTH, lw=0.9, ls=(0, (3, 2)), alpha=0.75,
               label="planted truth 0.300"),
        Line2D([], [], color=fs.TRUTH, lw=0.9, ls=(0, (1, 1.6)), alpha=0.75,
               label="realised truth 0.295"),
    ], loc="upper right", fontsize=6.8, labelspacing=0.42,
        borderaxespad=0.35, handlelength=1.6)
    lines = ["seed 100;  median, 90 % interval"] + [
        f"arm {k}   {s[k]['median']:.3f} "
        f"[{s[k]['ci90'][0]:.3f}, {s[k]['ci90'][1]:.3f}]"
        for k in ("S", "I", "J")]
    for n, line in enumerate(lines):
        axes[0].text(0.985, 0.50 - 0.068 * n, line, transform=axes[0].transAxes,
                     ha="right", va="top", fontsize=6.8, color=fs.INK2)
    save(fig, "fig_ablation_fagn")


# ---------------------------------------------------------------------------
# 3. p(Dmu_chi): intrinsic-only and joint
# ---------------------------------------------------------------------------
def fig_ablation_dmu(J, I):
    fJ, muJ, PJ, _, jsJ = J
    fI, muI, PI, _, jsI = I
    s = {"J": summarise(muJ, marginal_logp(PJ, fJ, 0)),
         "I": summarise(muI, marginal_logp(PI, fI, 0))}
    js = {"J": jsJ, "I": jsI}
    t = jsJ["truth"]
    err90 = Z90 * t["dmu_chi_realised_err"]

    print("\n  [3] fig_ablation_dmu  (p(Dmu_chi), two arms)")
    for k in ("I", "J"):
        print(f"    {ARM_NAME[k]}")
        check("median", s[k]["median"], js[k]["dmu_chi"]["median"])
        check("90 % low", s[k]["ci90"][0], js[k]["dmu_chi"]["ci90"][0])
        check("90 % high", s[k]["ci90"][1], js[k]["dmu_chi"]["ci90"][1])
        print(f"      68 % (text only)   {s[k]['ci68'][0]:.6f} "
              f"{s[k]['ci68'][1]:.6f}   width "
              f"{s[k]['ci68'][1] - s[k]['ci68'][0]:.4f}"
              f"   90 % width {s[k]['ci90'][1] - s[k]['ci90'][0]:.4f}")
        print(f"      peak density {s[k]['p'].max():.3f} per unit dmu at "
              f"dmu = {s[k]['x'][s[k]['p'].argmax()]:.4f};  "
              f"P(dmu <= 0) = {s[k]['p_le_zero']:.3g}")
    print(f"      truths drawn: planted {t['dmu_chi_planted']:+.6f}, "
          f"realised {t['dmu_chi_realised']:+.6f} +/- {err90:.6f} (90 %)")

    fig, ax = plt.subplots(figsize=(fs.ONECOL, 2.95))
    fig.subplots_adjust(left=0.165, right=0.985, bottom=0.148, top=0.845)
    ax.axvspan(t["dmu_chi_realised"] - err90, t["dmu_chi_realised"] + err90,
               color=fs.INK, alpha=0.08, lw=0, zorder=1)
    for k in ("I", "J"):
        band(ax, s[k], ARM_C[k])
        ax.plot(s[k]["x"], s[k]["p"], color=ARM_C[k], lw=1.6,
                zorder=4 if k == "J" else 3)
    ax.axvline(0.0, color=fs.TRUTH, lw=0.9, ls="-", alpha=0.45, zorder=5)
    ax.axvline(t["dmu_chi_planted"], color=fs.TRUTH, lw=0.9, ls=(0, (3, 2)),
               alpha=0.75, zorder=5)
    ax.axvline(t["dmu_chi_realised"], color=fs.TRUTH, lw=0.9, ls=(0, (1, 1.6)),
               alpha=0.75, zorder=5)
    ax.set_xlim(s["J"]["x"][0], s["J"]["x"][-1])
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel(r"$\Delta\mu_\chi$")
    ax.set_ylabel(r"$p(\Delta\mu_\chi)$   [per unit $\Delta\mu_\chi$]")
    ax.set_title(
        "seed 100;  medians, 90 % marginals,  "
        "$P(\\Delta\\mu_\\chi \\leq 0)$\n"
        f"arm I  ${s['I']['median']:.4f}\\,[{s['I']['ci90'][0]:.4f},\\,"
        f"{s['I']['ci90'][1]:.4f}]$,  {sci(s['I']['p_le_zero'])}\n"
        f"arm J  ${s['J']['median']:.4f}\\,[{s['J']['ci90'][0]:.4f},\\,"
        f"{s['J']['ci90'][1]:.4f}]$,  {sci(s['J']['p_le_zero'])}",
        fontsize=6.8, color=fs.INK2, linespacing=1.6)
    ax.annotate("no mark", (0.0, 0.50), xycoords=("data", "axes fraction"),
                textcoords="offset points", xytext=(-4, 0), ha="right",
                va="center", fontsize=6.8, color=fs.INK2)
    ax.legend(handles=[
        Line2D([], [], color=ARM_C[k], lw=1.6, label=ARM_NAME[k])
        for k in ("I", "J")
    ] + [
        Line2D([], [], color=fs.TRUTH, lw=0.9, ls=(0, (3, 2)), alpha=0.75,
               label="planted truth $+0.1000$"),
        Line2D([], [], color=fs.TRUTH, lw=0.9, ls=(0, (1, 1.6)), alpha=0.75,
               label="realised truth $+0.1119$"),
        Patch(facecolor=fs.INK, alpha=0.08, lw=0,
              label=f"realised truth, 90 % ($\\pm${err90:.4f})"),
    ], loc="upper left", fontsize=6.8, labelspacing=0.42, borderaxespad=0.35,
        handlelength=1.4)
    save(fig, "fig_ablation_dmu")


# ---------------------------------------------------------------------------
# 4. the per-event evidence plane
# ---------------------------------------------------------------------------
def fig_event_plane():
    with h5py.File(_read(RESULTS / "event_decomposition.h5"), "r") as h:
        x = np.asarray(h["log_BF_spatial"][:], float)
        y = np.asarray(h["log_BF_intrinsic"][:], float)
        p = np.asarray(h["P_AGN"][:], float)
        lab = np.asarray(h["true_host_type"][:], int)
        f_map = float(h.attrs["f_agn"])
        mu_map = float(h.attrs["mu_chi_c2"])
        lfg, lfa = float(h.attrs["log_f_gal"]), float(h.attrs["log_f_agn"])
    js = json.loads(_read(RESULTS / "event_decomposition.json").read_text())
    auc = js["separation"]["P_AGN"]["auc"]
    n_hi = js["classification"]["n_with_P_above_0p9"]
    n_lo = js["classification"]["n_with_P_below_0p1"]
    is_agn = lab == 1
    boundary = lfg - lfa

    print("\n  [4] fig_event_evidence_plane  (arm J at its MAP)")
    print(f"      point: f_AGN = {f_map:.3f}, dmu_chi = {mu_map:+.5f} "
          f"(json map {js['point']['f_agn']:.3f}, "
          f"{js['point']['dmu_chi']:+.5f})")
    print(f"      {int(is_agn.sum())} true AGN, {int((~is_agn).sum())} true "
          f"GAL of {x.size} events;  AUC(P_AGN) = {auc:.4f} "
          f"(json {js['separation']['log_BF_total']['auc']:.4f})")
    print(f"      ln BF_spatial   range [{x.min():.3f}, {x.max():.3f}];  "
          f"median GAL {np.median(x[~is_agn]):+.4f} "
          f"(json {js['separation']['log_BF_spatial']['median_true_GAL']:+.4f}),"
          f" median AGN {np.median(x[is_agn]):+.4f} "
          f"(json {js['separation']['log_BF_spatial']['median_true_AGN']:+.4f})")
    print(f"      ln BF_intrinsic range [{y.min():.3f}, {y.max():.3f}];  "
          f"median GAL {np.median(y[~is_agn]):+.4f} "
          f"(json {js['separation']['log_BF_intrinsic']['median_true_GAL']:+.4f}),"
          f" median AGN {np.median(y[is_agn]):+.4f} "
          f"(json {js['separation']['log_BF_intrinsic']['median_true_AGN']:+.4f})")
    print(f"      P_i = 0.5 rule: ln BF_sp + ln BF_in = ln(f_GAL/f_AGN) = "
          f"{boundary:+.4f}")
    print(f"      P_AGN in [{p.min():.3g}, {p.max():.3g}], colour scale 0 to 1;"
          f" {n_hi} events above 0.9, {n_lo} below 0.1 (json)")
    print(f"      frame: x symlog, linthresh 1, [-400, 9] (holds all "
          f"{x.size} events); y [-3.2, 3.0]")

    fig, axes = plt.subplots(1, 2, figsize=(fs.TWOCOL, 3.15), sharex=True,
                             sharey=True)
    fig.subplots_adjust(left=0.072, right=0.935, bottom=0.14, top=0.875,
                        wspace=0.06)
    xs = np.concatenate([-np.logspace(np.log10(400.0), 0.0, 240),
                         np.linspace(-1.0, 9.0, 240)])
    for ax in axes:
        ax.plot(xs, boundary - xs, color=fs.INK, lw=0.9, ls=(0, (3, 2)),
                alpha=0.8, zorder=5)
        ax.set_xscale("symlog", linthresh=1.0, linscale=1.4)
        ax.set_xlim(-400, 9)
        ax.set_ylim(-3.2, 3.0)
        ax.set_xticks([-100, -10, -1, 0, 1, 5])
        ax.set_xticklabels(["$-100$", "$-10$", "$-1$", "0", "1", "5"])
        ax.set_xlabel(r"$\ln\,\mathrm{BF}_{i,\,\rm spatial}$")

    axes[0].scatter(x[~is_agn], y[~is_agn], s=7, c=GAL_C, alpha=0.55, lw=0,
                    zorder=3, label=f"true GAL host ({int((~is_agn).sum())})")
    axes[0].scatter(x[is_agn], y[is_agn], s=7, c=AGN_C, alpha=0.8, lw=0,
                    zorder=4, label=f"true AGN host ({int(is_agn.sum())})")
    axes[0].set_ylabel(r"$\ln\,\mathrm{BF}_{i,\,\rm intrinsic}$")
    axes[0].set_title("(a) true host label (mock diagnostic)\n"
                      "the label never enters the likelihood or $P_i$",
                      fontsize=7.4, color=fs.INK2, linespacing=1.5)
    leg = axes[0].legend(handles=[
        Line2D([], [], color=GAL_C, marker="o", ms=3.0, ls="none",
               label=f"true GAL host ({int((~is_agn).sum())})"),
        Line2D([], [], color=AGN_C, marker="o", ms=3.0, ls="none",
               label=f"true AGN host ({int(is_agn.sum())})"),
        Line2D([], [], color=fs.INK, lw=0.9, ls=(0, (3, 2)), alpha=0.8,
               label=r"$P_i(\mathrm{AGN}) = 0.5$"),
    ], loc="lower left", fontsize=6.8, labelspacing=0.42, borderaxespad=0.35,
        handlelength=1.5, frameon=True, framealpha=0.85, edgecolor="none",
        facecolor="white",
        title=f"seed 100, arm J at its MAP:\n"
              f"$f_{{\\rm AGN}} = {f_map:.3f}$, "
              f"$\\Delta\\mu_\\chi = {mu_map:+.4f}$",
        title_fontsize=6.8)
    leg.set_zorder(6)

    sc = axes[1].scatter(x, y, s=7, c=p, cmap=P_CMAP, vmin=0.0, vmax=1.0,
                         lw=0, zorder=3)
    cb = fig.colorbar(sc, ax=axes[1], pad=0.025, fraction=0.055)
    cb.set_label(r"$P_i(\mathrm{AGN})$, inferred", fontsize=7.5)
    cb.ax.tick_params(labelsize=7.0)
    cb.outline.set_visible(False)
    axes[1].set_title(
        f"(b) inferred $P_i(\\mathrm{{AGN}})$, AUC = {auc:.3f}\n"
        f"{n_hi} of {x.size} events above $P_i = 0.9$, {n_lo} below 0.1",
        fontsize=7.4, color=fs.INK2, linespacing=1.5)
    save(fig, "fig_event_evidence_plane")


def main():
    fs.use()
    print(f"analysis 8 production figures -> {FIGS}")
    print(f"  visual system: {PAPER_SCRIPTS / 'figstyle.py'}")
    J = load_2d("arm_J_joint")
    I = load_2d("arm_I_intrinsic")
    S = load_1d("arm_S_spatial")
    fig_joint(J)
    fig_ablation_fagn(J, I, S)
    fig_ablation_dmu(J, I)
    fig_event_plane()
    print("\n  read:")
    for r in READS:
        print(f"    {r}")


if __name__ == "__main__":
    main()
