#!/usr/bin/env python3
"""Production figures for analysis 9 -- Analysis 8 with H0 freed, seed 100.

    python scripts/make_figures.py          # writes all three figures, pdf + png

Deterministic: it reads the recorded S9 and J9 scans and computes nothing new.
Every number it draws is printed to stdout beside the value recorded in the
corresponding JSON summary, so the figures can be diffed against
results/s9_spatial.json and results/j9_marked.json without opening them.

  1. fig_s9_spatial     arm S9, the spatial-only (H0, f_AGN) posterior at
                        dmu_chi = 0, 90 % HPD region, with the planted H0 and
                        both f truths; beside it p(H0) for S9 against Analysis
                        2's unmarked measurement on the same realisation.
  2. fig_j9_corner      arm J9, the marked joint posterior in
                        (H0, f_AGN, dmu_chi): three 90 % HPD planes and the
                        three marginals, with both truths on every axis.
  3. fig_section_11     what the mark does, measured: p(H0) spatial-only
                        against marked, p(f_AGN) the same, and p(dmu_chi)
                        at fixed H0 (Analysis 8) against free H0 (J9).
                        All three arms share the f and mu axes and every J9
                        H0 node is an S9 node, so the widths are compared on
                        matched axes.

Reductions, all of them the ones the production summariser used:

  * flat priors on every axis, so the posterior is the exponentiated
    log-likelihood; guard-rejected cells carry logL = -inf and therefore zero
    density, and the mass behind them is bounded in diagnostics/a9_guard.json;
  * marginals by trapezoid integration of that density over the other axes,
    exactly a9_scan.py::_marginals_3d / _marginals_h0_f;
  * medians and equal-tailed intervals from the trapezoid CDF of the marginal,
    exactly analysis_2/scripts/scan_h0f.py::marginal_ci;
  * 2-D regions are highest-posterior-density, figstyle.hpd_levels_2d.

House conventions, from ../../../paper/scripts/figstyle.py: that module is the
visual system and nothing here invents a colour.  EVERY plotted interval, band
and contour is 90 %; the 68 % numbers are printed to stdout and belong in text
and tables, never on an axis.  Arms keep the Analysis-8 identities: joint =
blue, spatial = aqua; the two fixed references (Analysis 8 at fixed H0,
Analysis 2 unmarked) are drawn in neutral grey because they are not arms of
this analysis.
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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
A9 = HERE.parent
RESULTS = A9 / "results"
FIGS = A9 / "figs"
WORKING = A9.parent.parent
A8_RESULTS = WORKING / "analyses" / "analysis_8_marked_multitracer_H0_fagn" / "results"
A2_RESULTS = WORKING / "analyses" / "analysis_2_complete_catalog_H0_fagn" / "results"
PAPER_SCRIPTS = WORKING / "paper" / "scripts"

sys.path.insert(0, str(PAPER_SCRIPTS))
import figstyle as fs  # noqa: E402

Z90 = float(norm.ppf(0.95))
C_J9, C_S9 = fs.C["blue"], fs.C["aqua"]
C_A8, C_A2 = fs.MUTED, fs.OTHER
READS: list[str] = []


def _read(path: Path) -> Path:
    READS.append(str(path))
    return path


def save(fig, name: str) -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    for ext, kw in (("pdf", {}), ("png", {"dpi": 300})):
        out = FIGS / f"{name}.{ext}"
        fig.savefig(out, **kw)
        print(f"    wrote {out}")
    plt.close(fig)


def check(label, got, want, tol=1e-6):
    d = abs(got - want)
    print(f"      {label:<38s} drawn {got: .6f}   json {want: .6f}   "
          f"|d| {d:.2e} {'OK' if d <= tol else 'MISMATCH'}")


def summarise(x, p):
    """median + equal-tailed 68/90 from a density, the marginal_ci convention."""
    x = np.asarray(x, float)
    p = np.asarray(p, float)
    p = p / np.trapz(p, x)
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (p[1:] + p[:-1]) * np.diff(x))])
    cdf /= cdf[-1]
    q = np.interp([0.05, 0.16, 0.5, 0.84, 0.95], cdf, x)
    return {"x": x, "p": p, "median": float(q[2]),
            "ci68": [float(q[1]), float(q[3])],
            "ci90": [float(q[0]), float(q[4])]}


def band(ax, s, color, alpha=0.16):
    xs, ys = fs.ci_band(s["x"], s["p"], s["ci90"][0], s["ci90"][1])
    ax.fill_between(xs, 0.0, ys, color=color, alpha=alpha, lw=0, zorder=2)


def curve(ax, s, color, label, lw=1.4, ls="-"):
    ax.plot(s["x"], s["p"], color=color, lw=lw, ls=ls, zorder=3, label=label)


def hpd_plane(ax, x, y, P, color, alpha=0.20):
    """The 90 % HPD region of a 2-D plane; returns the level."""
    _, l90 = fs.hpd_levels_2d(x, y, P)
    ax.contourf(x, y, P.T, levels=[l90, P.max() * 1.01], colors=[color],
                alpha=alpha, zorder=2)
    ax.contour(x, y, P.T, levels=[l90], colors=color, linewidths=1.3, zorder=3)
    return l90


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def load_s9():
    with h5py.File(_read(RESULTS / "s9_spatial.h5"), "r") as h:
        d = {"H0": np.asarray(h["H0_grid"][:], float),
             "f": np.asarray(h["f_grid"][:], float),
             "P": np.asarray(h["posterior_unnormalised"][:], float),
             "m_H0": np.asarray(h["marginal/H0"][:], float),
             "m_f": np.asarray(h["marginal/f"][:], float)}
    d["json"] = json.loads(_read(RESULTS / "s9_spatial.json").read_text())
    return d


def load_j9():
    with h5py.File(_read(RESULTS / "j9_marked.h5"), "r") as h:
        d = {"H0": np.asarray(h["H0_grid"][:], float),
             "f": np.asarray(h["f_grid"][:], float),
             "mu": np.asarray(h["mu_chi_c2_grid"][:], float),
             "m_H0": np.asarray(h["marginal/H0"][:], float),
             "m_f": np.asarray(h["marginal/f"][:], float),
             "m_mu": np.asarray(h["marginal/mu_chi_c2"][:], float),
             "H0_f": np.asarray(h["marginal/H0_f"][:], float),
             "H0_mu": np.asarray(h["marginal/H0_mu"][:], float),
             "f_mu": np.asarray(h["marginal/f_mu"][:], float)}
    d["json"] = json.loads(_read(RESULTS / "j9_marked.json").read_text())
    return d


def load_a8():
    """Analysis 8's arm J at FIXED H0: the dmu_chi reference of record."""
    with h5py.File(_read(A8_RESULTS / "arm_J_joint.h5"), "r") as h:
        f = np.asarray(h["f_grid"][:], float)
        mu = np.asarray(h["mu_chi_c2_grid"][:], float)
        ll = np.asarray(h["log_likelihood"][:], float)
    fin = np.isfinite(ll)
    P = np.where(fin, np.exp(ll - ll[fin].max()), 0.0)
    js = json.loads(_read(A8_RESULTS / "arm_J_joint.json").read_text())
    return {"f": f, "mu": mu, "m_mu": np.trapz(P, f, axis=0),
            "m_f": np.trapz(P, mu, axis=1), "json": js}


def load_a2():
    """Analysis 2, same realisation, UNMARKED events, one shared population."""
    with h5py.File(_read(A2_RESULTS / "joint_s100.h5"), "r") as h:
        H0 = np.asarray(h["H0_grid"][:], float)
        f = np.asarray(h["f_grid"][:], float)
        ll = np.asarray(h["log_likelihood"][:], float)
    fin = np.isfinite(ll)
    P = np.where(fin, np.exp(ll - ll[fin].max()), 0.0)
    js = json.loads(_read(A2_RESULTS / "joint_s100.json").read_text())
    return {"H0": H0, "f": f, "m_H0": np.trapz(P, f, axis=1), "json": js}


# --------------------------------------------------------------------------- #
# 1. the spatial-only arm
# --------------------------------------------------------------------------- #
def fig_s9(S, A2):
    js, t = S["json"], S["json"]["truth"]
    sH, sf = summarise(S["H0"], S["m_H0"]), summarise(S["f"], S["m_f"])
    a2H = summarise(A2["H0"], A2["m_H0"])
    print("\n  [1] fig_s9_spatial  (arm S9, 90 % HPD; dmu_chi = 0)")
    check("H0 median", sH["median"], js["H0"]["median"])
    check("H0 90 % low", sH["ci90"][0], js["H0"]["ci90"][0])
    check("H0 90 % high", sH["ci90"][1], js["H0"]["ci90"][1])
    check("f_AGN median", sf["median"], js["f"]["median"])
    check("f_AGN 90 % low", sf["ci90"][0], js["f"]["ci90"][0])
    check("f_AGN 90 % high", sf["ci90"][1], js["f"]["ci90"][1])
    check("A2 H0 median (unmarked)", a2H["median"], A2["json"]["H0"]["median"])
    print(f"      H0 68 % width {sH['ci68'][1] - sH['ci68'][0]:.4f}, "
          f"90 % width {sH['ci90'][1] - sH['ci90'][0]:.4f}")

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(fs.TWOCOL, 2.45))
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.165, top=0.885,
                        wspace=0.265)

    lo, hi = sH["ci90"][0] - 1.6, sH["ci90"][1] + 1.6
    m = (S["H0"] >= lo) & (S["H0"] <= hi)
    hpd_plane(ax, S["H0"][m], S["f"], S["P"][m], C_S9)
    fs.truth_line(ax, t["H0_planted"], axis="x", label="$H_0$ planted", pos=0.52)
    ax.plot([js["map"]["H0"]], [js["map"]["f_agn"]], marker="x", ms=5.0, mew=1.3,
            color=C_S9, zorder=5)
    ax.errorbar([t["f_agn_realised"]], [t["f_agn_realised"]], fmt="none")
    ax.axhline(t["f_agn_planted"], color=fs.TRUTH, lw=0.9, ls=(0, (3, 2)),
               alpha=0.75, zorder=1.5)
    ax.axhline(t["f_agn_realised"], color=fs.TRUTH, lw=0.9, ls=(0, (1, 2)),
               alpha=0.75, zorder=1.5)
    ax.set_xlabel(r"$H_0$  [km s$^{-1}$ Mpc$^{-1}$]")
    ax.set_ylabel(r"$f_{\rm AGN}$")
    ax.set_xlim(lo, hi)
    ax.set_ylim(max(0.0, sf["ci90"][0] - 0.16), min(1.0, sf["ci90"][1] + 0.16))
    ax.set_title("spatial only, $\\Delta\\mu_\\chi = 0$", fontsize=7.6, pad=3)
    ax.legend(handles=[
        Patch(fc=C_S9, alpha=0.20, ec=C_S9, lw=1.3, label="S9, 90 %"),
        Line2D([], [], color=fs.TRUTH, lw=0.9, ls=(0, (3, 2)),
               label="planted truth"),
        Line2D([], [], color=fs.TRUTH, lw=0.9, ls=(0, (1, 2)),
               label="realised truth")],
        loc="upper right", fontsize=6.2, frameon=False, handlelength=1.6)

    band(bx, sH, C_S9)
    curve(bx, sH, C_S9, "S9, marked mock, spatial only")
    curve(bx, a2H, C_A2, "Analysis 2, unmarked", lw=1.1, ls=(0, (4, 2)))
    fs.truth_line(bx, t["H0_planted"], axis="x", label="planted", pos=0.44)
    bx.set_xlabel(r"$H_0$  [km s$^{-1}$ Mpc$^{-1}$]")
    bx.set_ylabel(r"$p(H_0)$")
    bx.set_xlim(lo, hi)
    bx.set_ylim(bottom=0.0)
    bx.set_title("same realisation, same $H_0$ axis", fontsize=7.6, pad=3)
    bx.legend(loc="upper left", fontsize=6.2, frameon=False, handlelength=1.9)
    bx.annotate(
        f"S9  ${sH['median']:.2f}\\,[{sH['ci90'][0]:.2f},\\,{sH['ci90'][1]:.2f}]$\n"
        f"A2  ${a2H['median']:.2f}\\,[{a2H['ci90'][0]:.2f},\\,{a2H['ci90'][1]:.2f}]$",
        (0.985, 0.60), xycoords="axes fraction", ha="right", va="top",
        fontsize=6.2, color=fs.INK)
    save(fig, "fig_s9_spatial")


# --------------------------------------------------------------------------- #
# 2. the marked joint posterior
# --------------------------------------------------------------------------- #
def fig_j9(J):
    js, t = J["json"], J["json"]["truth"]
    sH = summarise(J["H0"], J["m_H0"])
    sf = summarise(J["f"], J["m_f"])
    sm = summarise(J["mu"], J["m_mu"])
    print("\n  [2] fig_j9_corner  (arm J9, 90 % HPD planes)")
    for name, s, key in (("H0", sH, "H0"), ("f_AGN", sf, "f"),
                         ("dmu_chi", sm, "mu_chi_c2")):
        check(f"{name} median", s["median"], js[key]["median"])
        check(f"{name} 90 % low", s["ci90"][0], js[key]["ci90"][0])
        check(f"{name} 90 % high", s["ci90"][1], js[key]["ci90"][1])
    rho = js["posterior_moments"]["correlation"]
    print(f"      rho(H0,f) {rho['H0|f_agn']:+.4f}   "
          f"rho(H0,dmu) {rho['H0|mu_chi_c2']:+.4f}   "
          f"rho(f,dmu) {rho['f_agn|mu_chi_c2']:+.4f}")

    names = [r"$H_0$", r"$f_{\rm AGN}$", r"$\Delta\mu_\chi$"]
    grids = [J["H0"], J["f"], J["mu"]]
    ones = [sH, sf, sm]
    planes = {(0, 1): J["H0_f"], (0, 2): J["H0_mu"], (1, 2): J["f_mu"]}
    truths = [[t["H0_planted"]], [t["f_agn_planted"], t["f_agn_realised"]],
              [t["dmu_chi_planted"], t["dmu_chi_realised"]]]
    lims = [(max(grids[0][0], s["ci90"][0] - w), min(grids[0][-1], s["ci90"][1] + w))
            for s, w in [(sH, 1.6)]]
    lims += [(max(0.0, sf["ci90"][0] - 0.12), min(1.0, sf["ci90"][1] + 0.12)),
             (max(grids[2][0], sm["ci90"][0] - 0.055),
              min(grids[2][-1], sm["ci90"][1] + 0.055))]

    fig, axes = plt.subplots(3, 3, figsize=(fs.TWOCOL, 4.6))
    fig.subplots_adjust(left=0.085, right=0.995, bottom=0.095, top=0.975,
                        wspace=0.09, hspace=0.09)
    for i in range(3):
        for j in range(3):
            ax = axes[i][j]
            if j > i:
                ax.axis("off")
                continue
            if i == j:
                s = ones[i]
                band(ax, s, C_J9)
                curve(ax, s, C_J9, None)
                ax.set_ylim(bottom=0.0)
                ax.set_yticks([])
            else:
                x, y = grids[j], grids[i]
                P = planes[(j, i)]
                hpd_plane(ax, x, y, P, C_J9)
                ax.plot([js["map"][["H0", "f_agn", "mu_chi_c2"][j]]],
                        [js["map"][["H0", "f_agn", "mu_chi_c2"][i]]],
                        marker="x", ms=4.5, mew=1.2, color=C_J9, zorder=5)
                for k, v in enumerate(truths[i]):
                    ax.axhline(v, color=fs.TRUTH, lw=0.85, alpha=0.75,
                               ls=(0, (3, 2)) if k == 0 else (0, (1, 2)), zorder=1.5)
                ax.set_ylim(*lims[i])
            for k, v in enumerate(truths[j]):
                ax.axvline(v, color=fs.TRUTH, lw=0.85, alpha=0.75,
                           ls=(0, (3, 2)) if k == 0 else (0, (1, 2)), zorder=1.5)
            ax.set_xlim(*lims[j])
            if i == 2:
                ax.set_xlabel(names[j])
            else:
                ax.set_xticklabels([])
            if j == 0 and i > 0:
                ax.set_ylabel(names[i])
            elif j > 0:
                ax.set_yticklabels([])
    axes[0][0].set_ylabel(r"$p(H_0)$")
    axes[0][1].axis("off")
    txt = (f"seed 100, marked joint (J9)\nmedians, 90 % intervals\n"
           f"$H_0 = {sH['median']:.2f}\\,[{sH['ci90'][0]:.2f},\\,"
           f"{sH['ci90'][1]:.2f}]$\n"
           f"$f_{{\\rm AGN}} = {sf['median']:.3f}\\,[{sf['ci90'][0]:.3f},\\,"
           f"{sf['ci90'][1]:.3f}]$\n"
           f"$\\Delta\\mu_\\chi = {sm['median']:.4f}\\,[{sm['ci90'][0]:.4f},\\,"
           f"{sm['ci90'][1]:.4f}]$\n"
           f"$\\rho(H_0,f) = {rho['H0|f_agn']:+.3f}$,  "
           f"$\\rho(H_0,\\Delta\\mu_\\chi) = {rho['H0|mu_chi_c2']:+.3f}$\n"
           f"$\\rho(f,\\Delta\\mu_\\chi) = {rho['f_agn|mu_chi_c2']:+.3f}$")
    fig.text(0.45, 0.955, txt, ha="left", va="top", fontsize=6.6, color=fs.INK)
    fig.legend(handles=[
        Patch(fc=C_J9, alpha=0.20, ec=C_J9, lw=1.3, label="J9, 90 %"),
        Line2D([], [], color=fs.TRUTH, lw=0.85, ls=(0, (3, 2)), label="planted"),
        Line2D([], [], color=fs.TRUTH, lw=0.85, ls=(0, (1, 2)), label="realised")],
        loc="upper right", bbox_to_anchor=(0.995, 0.70), fontsize=6.4,
        frameon=False, handlelength=1.7)
    save(fig, "fig_j9_corner")


# --------------------------------------------------------------------------- #
# 3. what the mark does, measured
# --------------------------------------------------------------------------- #
def fig_section_11(S, J, A8):
    sH9, jH9 = summarise(S["H0"], S["m_H0"]), summarise(J["H0"], J["m_H0"])
    sf9, jf9 = summarise(S["f"], S["m_f"]), summarise(J["f"], J["m_f"])
    a8m, jm = summarise(A8["mu"], A8["m_mu"]), summarise(J["mu"], J["m_mu"])
    t = J["json"]["truth"]
    print("\n  [3] fig_section_11  (matched axes; 90 % bands)")
    for nm, a, b in (("H0  S9 -> J9", sH9, jH9), ("f   S9 -> J9", sf9, jf9),
                     ("dmu A8 -> J9", a8m, jm)):
        print(f"      {nm}: 68 % width {a['ci68'][1]-a['ci68'][0]:.5f} -> "
              f"{b['ci68'][1]-b['ci68'][0]:.5f}   "
              f"90 % width {a['ci90'][1]-a['ci90'][0]:.5f} -> "
              f"{b['ci90'][1]-b['ci90'][0]:.5f}   "
              f"median {a['median']:.5f} -> {b['median']:.5f}")

    fig, axes = plt.subplots(1, 3, figsize=(fs.TWOCOL, 2.35))
    fig.subplots_adjust(left=0.065, right=0.995, bottom=0.175, top=0.875,
                        wspace=0.275)

    ax = axes[0]
    band(ax, jH9, C_J9); band(ax, sH9, C_S9)
    curve(ax, sH9, C_S9, "spatial only (S9)")
    curve(ax, jH9, C_J9, "marked joint (J9)")
    fs.truth_line(ax, t["H0_planted"], axis="x", label="planted", pos=0.44)
    ax.set_xlim(J["H0"][0], J["H0"][-1])
    ax.set_xlabel(r"$H_0$  [km s$^{-1}$ Mpc$^{-1}$]")
    ax.set_ylabel(r"$p(H_0)$")

    bx = axes[1]
    band(bx, jf9, C_J9); band(bx, sf9, C_S9)
    curve(bx, sf9, C_S9, "S9")
    curve(bx, jf9, C_J9, "J9")
    bx.axvline(t["f_agn_planted"], color=fs.TRUTH, lw=0.9, ls=(0, (3, 2)),
               alpha=0.75, zorder=1.5)
    bx.axvline(t["f_agn_realised"], color=fs.TRUTH, lw=0.9, ls=(0, (1, 2)),
               alpha=0.75, zorder=1.5)
    bx.set_xlim(max(0.0, min(sf9["ci90"][0], jf9["ci90"][0]) - 0.1),
                min(1.0, max(sf9["ci90"][1], jf9["ci90"][1]) + 0.1))
    bx.set_xlabel(r"$f_{\rm AGN}$")
    bx.set_ylabel(r"$p(f_{\rm AGN})$")

    cx = axes[2]
    band(cx, jm, C_J9); band(cx, a8m, C_A8)
    curve(cx, a8m, C_A8, "$H_0$ fixed (Analysis 8)", lw=1.1, ls=(0, (4, 2)))
    curve(cx, jm, C_J9, "$H_0$ free (J9)")
    cx.axvline(t["dmu_chi_planted"], color=fs.TRUTH, lw=0.9, ls=(0, (3, 2)),
               alpha=0.75, zorder=1.5)
    cx.axvline(t["dmu_chi_realised"], color=fs.TRUTH, lw=0.9, ls=(0, (1, 2)),
               alpha=0.75, zorder=1.5)
    cx.set_xlim(min(a8m["ci90"][0], jm["ci90"][0]) - 0.035,
                max(a8m["ci90"][1], jm["ci90"][1]) + 0.035)
    cx.set_xlabel(r"$\Delta\mu_\chi$")
    cx.set_ylabel(r"$p(\Delta\mu_\chi)$")

    for ax, s_a, s_b, la, lb, fmt in (
            (axes[0], sH9, jH9, "S9", "J9", "{:.2f}"),
            (axes[1], sf9, jf9, "S9", "J9", "{:.3f}"),
            (axes[2], a8m, jm, "A8", "J9", "{:.4f}")):
        ax.set_ylim(bottom=0.0)
        ax.legend(loc="upper left", fontsize=6.2, frameon=False, handlelength=1.9)
        w = lambda s, k: s[k][1] - s[k][0]
        ax.annotate(
            f"90 % width\n{la} {fmt.format(w(s_a, 'ci90'))}\n"
            f"{lb} {fmt.format(w(s_b, 'ci90'))}\n"
            f"ratio {w(s_b, 'ci90') / w(s_a, 'ci90'):.3f}",
            (0.985, 0.985), xycoords="axes fraction", ha="right", va="top",
            fontsize=6.2, color=fs.INK)
    save(fig, "fig_section_11")


def main():
    fs.use()
    print("analysis 9 figures -- every band and contour is 90 %")
    S, J, A8, A2 = load_s9(), load_j9(), load_a8(), load_a2()
    fig_s9(S, A2)
    fig_j9(J)
    fig_section_11(S, J, A8)
    print("\n  read:")
    for r in dict.fromkeys(READS):
        print(f"    {r}")


if __name__ == "__main__":
    main()
