#!/usr/bin/env python3
"""Production figures for Analysis 10's fixed-H0 stage (arms S, chi, M, J).

    python scripts/make_figures.py          # writes whatever arms exist, pdf + png

CPU only, matplotlib Agg, no likelihood, deterministic: it reads
``results/a10_arm_{S,M,chi,J}.{h5,json}`` and computes nothing beyond flat-prior
marginals and equal-tailed / HPD summaries of what is already stored there.
Every number drawn is printed to stdout beside the value the corresponding JSON
recorded, so the figures can be checked against ``results/`` without opening a
plot (Gate D2).  Arms S and M were written by ``a10_arms.py --stage assemble``;
arms chi and J are written by ``a10_scan.py --stage assemble`` and do not exist
yet as of 2026-09-22 (the 52,521-cell joint cube is still running).  This
script is safe to run at any point: any figure or panel whose input file is
missing is skipped with a printed notice, never faked, and the script still
completes and writes whatever it can.

  1. fig_a10_fagn    p(f_AGN) for S, chi, M, J overlaid on the shared 41-node
                     f axis, 90 % intervals as bars under the curves, planted
                     0.30 and the (explicitly non-target) detected-set 0.357
                     as reference lines.
  2. fig_a10_marks   two panels: p(dmu_chi) for chi + J (planted +0.10 and
                     realised +0.0995 marked); p(dmu_G) for M + J on the
                     refined 27-node axis, with M's 21-node coarse marginal
                     overlaid as a thin dotted curve so the refinement's
                     effect is visible (planted +5 marked).  A panel with no
                     arm available yet is left blank with a printed notice,
                     not drawn from anything.
  3. fig_a10_planes  three panels of A10-J's 90 % HPD planes -- (f, dmu_chi),
                     (f, dmu_G), (dmu_G, dmu_chi) -- with the matching
                     single-mark arm's own 90 % region overlaid as a contour
                     where one exists (chi on the first, M on the second; the
                     third has no single-mark analogue).  Truths marked, rho
                     printed from the JSON's own correlation block.  Skipped
                     entirely until A10-J exists.

Schema note (2026-09-22): A10-S and A10-M exist and were read directly to fix
the JSON keys and h5 dataset names used below.  A10-chi and A10-J do NOT exist
yet -- their h5 dataset names (``marginal/f_dmu_G``, ``marginal/f_dmu_chi``,
``marginal/dmu_G_dmu_chi``, ``dmu_G_grid``, ``dmu_chi_grid``, the chi arm's own
2-D ``posterior_unnormalised``) and JSON keys (``dmu_chi``, ``dmu_G``,
``correlations`` with pipe-separated keys ``"f_agn|dmu_G"`` /
``"f_agn|dmu_chi"`` / ``"dmu_G|dmu_chi"``, ``map``) were read out of
``a10_scan.py``'s ``stage_assemble`` / ``_write_chi_arm`` / ``_marginals_3d``
source, not off a live file, and should be re-verified once the cube lands --
this script will simply raise a ``KeyError``/``FileNotFoundError`` at the
matching load function if a name has changed, which is the fast way to catch
it.

Reductions:

  * flat priors on every axis, so the posterior is the exponentiated
    log-likelihood; guard-rejected cells carry logL = -inf, hence zero density;
  * marginals by trapezoid integration, exactly a10_arms.py / a10_scan.py's
    own GC._marginals / _marginals_3d (imported nowhere here -- this script
    only re-derives the SAME closed-form summary from the stored
    ``marginal_logp`` array, via ``summarise()`` below, which is a literal
    reimplementation of analysis_2/scripts/scan_h0f.py::marginal_ci);
  * 2-D regions are highest-posterior-density, figstyle.hpd_levels_2d.

House conventions, from ../../../paper/scripts/figstyle.py: that module is the
visual system and nothing here invents a colour.  EVERY plotted interval, band
and contour is 90 %; 68 % numbers are printed to stdout and never plotted.
Arm identities, fixed across every figure in this script:

  * joint (A10-J)     = blue   -- continues the Analysis 8/9 "joint" identity.
  * spatial (A10-S)   = aqua   -- continues the Analysis 8/9 "spatial" identity.
  * spin-only (A10-chi) = orange -- continues Analysis 8/9's "intrinsic" arm,
    which IS the spin-only mark; A10-chi is that same arm re-run on the
    two-mark mock, so it keeps that identity rather than inventing one.
  * mass-only (A10-M) = magenta -- a genuinely new arm in Analysis 10, given
    the next figstyle.SERIES slot not already claimed by an arm or a status
    colour (yellow was passed over: at the alpha used for HPD fills and
    interval bars it reads too light against the white page).

Reference lines (truths, axis edges) are neutral ink, never a series colour.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import json
import os
from pathlib import Path

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
A10 = HERE.parent
RESULTS = A10 / "results"
FIGS = A10 / "figs"
WORKING = A10.parent.parent
PAPER_SCRIPTS = WORKING / "paper" / "scripts"

if str(PAPER_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(PAPER_SCRIPTS))
import figstyle as fs  # noqa: E402

fs.use()

# ---- arm identities, fixed across every A10 figure (see module docstring) -- #
C_J, C_S, C_CHI, C_M = fs.C["blue"], fs.C["aqua"], fs.C["orange"], fs.C["magenta"]

READS: list[str] = []
SKIPPED: list[str] = []


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _read_json(path: Path):
    if not path.exists():
        return None
    READS.append(str(path))
    return json.loads(path.read_text())


def _read_h5(path: Path):
    if not path.exists():
        return None
    READS.append(str(path))
    return h5py.File(path, "r")


def notice(msg: str) -> None:
    print(f"  [skip] {msg}")
    SKIPPED.append(msg)


def check(label: str, got: float, want: float, tol: float = 1.0e-9) -> None:
    got, want = float(got), float(want)
    d = abs(got - want)
    print(f"        {label:<32s} drawn {got: .10f}   json {want: .10f}   "
          f"|d| {d:.2e}  {'OK' if d <= tol else 'MISMATCH'}")


def summarise(x, marginal_logp) -> dict:
    """Median + equal-tailed 68/90 from a flat-prior marginal log-density.

    A literal reimplementation of
    analysis_2/scripts/scan_h0f.py::marginal_ci (imported nowhere in this
    analysis's assemble stages either -- every JSON summary here already used
    it), so that recomputing it from the stored ``marginal_logp`` reproduces
    the JSON's own median / ci68 / ci90 to floating-point precision.
    """
    x = np.asarray(x, dtype=float)
    lp = np.asarray(marginal_logp, dtype=float)
    fin = np.isfinite(lp)
    m = float(np.nanmax(lp[fin])) if fin.any() else 0.0
    p = np.exp(lp - m)
    norm = np.trapz(p, x)
    p = p / norm
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (p[1:] + p[:-1]) * np.diff(x))])
    cdf /= cdf[-1]
    return {
        "x": x, "p": p,
        "median": float(np.interp(0.5, cdf, x)),
        "ci68": [float(np.interp(0.16, cdf, x)), float(np.interp(0.84, cdf, x))],
        "ci90": [float(np.interp(0.05, cdf, x)), float(np.interp(0.95, cdf, x))],
    }


def save(fig, name: str) -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    for ext, kw in (("pdf", {}), ("png", {"dpi": 300})):
        out = FIGS / f"{name}.{ext}"
        fig.savefig(out, **kw)
        print(f"    wrote {out}")
    plt.close(fig)


def interval_strip(ax, rows, ymax: float, unit_frac: float = 0.055) -> None:
    """Short 90 % bars + median ticks, stacked beneath the density curves."""
    if ymax <= 0 or not np.isfinite(ymax):
        ymax = 1.0
    unit = unit_frac * ymax
    n = len(rows)
    for i, (color, lo, hi, med) in enumerate(rows):
        y = -unit * (i + 1)
        ax.plot([lo, hi], [y, y], color=color, lw=2.6, solid_capstyle="butt",
                 zorder=4, clip_on=False)
        ax.plot([med], [y], marker="|", color=color, ms=7.5, mew=1.7, zorder=5,
                 clip_on=False)
    ax.set_ylim(-unit * (n + 0.8), ymax * 1.12)


def hpd_plane(ax, x, y, P, color, alpha=0.20, lw=1.3, fill=True):
    _, l90 = fs.hpd_levels_2d(x, y, P)
    if fill:
        ax.contourf(x, y, P.T, levels=[l90, P.max() * 1.01], colors=[color],
                    alpha=alpha, zorder=2)
    ax.contour(x, y, P.T, levels=[l90], colors=color, linewidths=lw, zorder=3)
    return l90


# --------------------------------------------------------------------------- #
# loaders -- each returns None (and prints a notice) if its file is missing
# --------------------------------------------------------------------------- #
def load_S():
    j = _read_json(RESULTS / "a10_arm_S.json")
    h = _read_h5(RESULTS / "a10_arm_S.h5")
    if j is None or h is None:
        notice("A10-S: results/a10_arm_S.{json,h5} not found")
        return None
    d = {"json": j, "f_grid": np.asarray(h["f_grid"][:], float)}
    h.close()
    return d


def load_M():
    j = _read_json(RESULTS / "a10_arm_M.json")
    h = _read_h5(RESULTS / "a10_arm_M.h5")
    if j is None or h is None:
        notice("A10-M: results/a10_arm_M.{json,h5} not found")
        return None
    d = {
        "json": j,
        "f_grid": np.asarray(h["f_grid"][:], float),
        "dmu_G_grid": np.asarray(h["dmu_G_grid"][:], float),
        "P": np.asarray(h["posterior_unnormalised"][:], float),
    }
    h.close()
    return d


def load_chi():
    """A10-chi: the dmu_G = 0 slab of the A10-J cube (a10_scan.py, NOT RUN yet)."""
    j = _read_json(RESULTS / "a10_arm_chi.json")
    h = _read_h5(RESULTS / "a10_arm_chi.h5")
    if j is None or h is None:
        notice("A10-chi: results/a10_arm_chi.{json,h5} not found yet "
               "(written by a10_scan.py --stage assemble once the cube lands)")
        return None
    d = {
        "json": j,
        "f_grid": np.asarray(h["f_grid"][:], float),
        "dmu_chi_grid": np.asarray(h["dmu_chi_grid"][:], float),
        "P": np.asarray(h["posterior_unnormalised"][:], float),
    }
    h.close()
    return d


def load_J():
    """A10-J: the 41 x 27 x 61 = 52,521-cell joint cube (a10_scan.py, NOT RUN yet)."""
    j = _read_json(RESULTS / "a10_arm_J.json")
    h = _read_h5(RESULTS / "a10_arm_J.h5")
    if j is None or h is None:
        notice("A10-J: results/a10_arm_J.{json,h5} not found yet "
               "(written by a10_scan.py --stage assemble once the cube lands)")
        return None
    d = {
        "json": j,
        "f_grid": np.asarray(h["f_grid"][:], float),
        "dmu_G_grid": np.asarray(h["dmu_G_grid"][:], float),
        "dmu_chi_grid": np.asarray(h["dmu_chi_grid"][:], float),
        "f_dmu_G": np.asarray(h["marginal/f_dmu_G"][:], float),
        "f_dmu_chi": np.asarray(h["marginal/f_dmu_chi"][:], float),
        "dmu_G_dmu_chi": np.asarray(h["marginal/dmu_G_dmu_chi"][:], float),
    }
    h.close()
    return d


# --------------------------------------------------------------------------- #
# 1. p(f_AGN), all arms overlaid
# --------------------------------------------------------------------------- #
def fig_a10_fagn(S, chi, M, J):
    print("\n[1] fig_a10_fagn  (p(f_AGN); S / chi / M / J overlaid, shared 41-node axis)")
    arms = []
    if S is not None:
        arms.append(("A10-S", S, C_S))
    if chi is not None:
        arms.append(("A10-chi", chi, C_CHI))
    if M is not None:
        arms.append(("A10-M", M, C_M))
    if J is not None:
        arms.append(("A10-J", J, C_J))
    if not arms:
        notice("fig_a10_fagn: no arm has an f_AGN marginal yet; nothing drawn")
        return

    fig, ax = plt.subplots(figsize=(fs.ONECOL, 2.7))
    fig.subplots_adjust(left=0.165, right=0.97, bottom=0.15, top=0.90)

    ymax, strip_rows, handles = 0.0, [], []
    for name, d, color in arms:
        js = d["json"]
        s = summarise(d["f_grid"], js["f"]["marginal_logp"])
        print(f"    {name}")
        check("f_AGN median", s["median"], js["f"]["median"])
        check("f_AGN 90% low", s["ci90"][0], js["f"]["ci90"][0])
        check("f_AGN 90% high", s["ci90"][1], js["f"]["ci90"][1])
        print(f"        f_AGN 68% [{s['ci68'][0]:.6f}, {s['ci68'][1]:.6f}]  "
              f"(json [{js['f']['ci68'][0]:.6f}, {js['f']['ci68'][1]:.6f}]; "
              f"printed only, never plotted)")
        ax.plot(s["x"], s["p"], color=color, lw=1.6, zorder=3)
        ymax = max(ymax, float(np.nanmax(s["p"])))
        strip_rows.append((color, s["ci90"][0], s["ci90"][1], s["median"]))
        handles.append(Line2D([], [], color=color, lw=1.6,
                        label=f"{name}  {s['median']:.3f} "
                              f"[{s['ci90'][0]:.3f}, {s['ci90'][1]:.3f}]"))

    interval_strip(ax, strip_rows, ymax)
    truth = arms[0][1]["json"]["truth"]
    ax.axvline(truth["f_agn_planted"], color=fs.TRUTH, lw=0.9, ls=(0, (3, 2)),
               alpha=0.75, zorder=1.5)
    ax.axvline(truth["f_agn_detected_fraction"], color=fs.TRUTH, lw=0.9,
               ls=(0, (1, 1.4)), alpha=0.75, zorder=1.5)
    handles.append(Line2D([], [], color=fs.TRUTH, lw=0.9, ls=(0, (3, 2)),
                    alpha=0.75, label=f"planted ({truth['f_agn_planted']:.2f})"))
    handles.append(Line2D([], [], color=fs.TRUTH, lw=0.9, ls=(0, (1, 1.4)),
                    alpha=0.75, label="detected-set fraction "
                                      f"({truth['f_agn_detected_fraction']:.3f}, "
                                      "descriptive)"))
    ax.set_xlabel(r"$f_{\rm AGN}$")
    ax.set_ylabel(r"$p(f_{\rm AGN})$")
    ax.set_xlim(0.0, 1.0)
    ax.legend(handles=handles, loc="upper right", fontsize=5.8, frameon=False,
              handlelength=1.6)
    for miss_name in ("A10-chi", "A10-J"):
        if miss_name not in [a[0] for a in arms]:
            ax.annotate(f"{miss_name} not yet available", (0.62, 0.60),
                        xycoords="axes fraction", ha="left", va="center",
                        fontsize=5.8, color=fs.MUTED)
            break  # one line is enough; both notices are already on stdout
    save(fig, "fig_a10_fagn")


# --------------------------------------------------------------------------- #
# 2. the two marks
# --------------------------------------------------------------------------- #
def fig_a10_marks(chi, M, J):
    print("\n[2] fig_a10_marks  (p(dmu_chi) left, p(dmu_G) right)")
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(fs.TWOCOL, 2.7))
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.155, top=0.885,
                        wspace=0.26)

    # ---- panel L: p(dmu_chi), chi + J --------------------------------------
    arms_L = []
    if chi is not None:
        arms_L.append(("A10-chi", chi, C_CHI))
    if J is not None:
        arms_L.append(("A10-J", J, C_J))
    if not arms_L:
        notice("fig_a10_marks panel L (dmu_chi): neither A10-chi nor A10-J is "
               "available yet; panel left blank")
        axL.text(0.5, 0.52, "A10-$\\chi$ / A10-J\nnot yet available",
                  ha="center", va="center", transform=axL.transAxes,
                  fontsize=8.2, color=fs.MUTED)
        axL.set_xticks([])
        axL.set_yticks([])
        axL.grid(False)
    else:
        ymax, rows, handles = 0.0, [], []
        for name, d, color in arms_L:
            js = d["json"]
            s = summarise(d["dmu_chi_grid"], js["dmu_chi"]["marginal_logp"])
            print(f"    {name}  p(dmu_chi)")
            check("dmu_chi median", s["median"], js["dmu_chi"]["median"])
            check("dmu_chi 90% low", s["ci90"][0], js["dmu_chi"]["ci90"][0])
            check("dmu_chi 90% high", s["ci90"][1], js["dmu_chi"]["ci90"][1])
            print(f"        dmu_chi 68% [{s['ci68'][0]:.6f}, {s['ci68'][1]:.6f}]  "
                  "(printed only, never plotted)")
            axL.plot(s["x"], s["p"], color=color, lw=1.6, zorder=3)
            ymax = max(ymax, float(np.nanmax(s["p"])))
            rows.append((color, s["ci90"][0], s["ci90"][1], s["median"]))
            handles.append(Line2D([], [], color=color, lw=1.6,
                            label=f"{name}  {s['median']:.4f} "
                                  f"[{s['ci90'][0]:.4f}, {s['ci90'][1]:.4f}]"))
        interval_strip(axL, rows, ymax)
        t = arms_L[0][1]["json"]["truth"]
        axL.axvline(t["dmu_chi_planted"], color=fs.TRUTH, lw=0.9,
                    ls=(0, (3, 2)), alpha=0.75, zorder=1.5)
        axL.axvline(t["dmu_chi_realised"], color=fs.TRUTH, lw=0.9,
                    ls=(0, (1, 1.4)), alpha=0.75, zorder=1.5)
        handles.append(Line2D([], [], color=fs.TRUTH, lw=0.9, ls=(0, (3, 2)),
                        alpha=0.75, label=f"planted (+{t['dmu_chi_planted']:.4f})"))
        handles.append(Line2D([], [], color=fs.TRUTH, lw=0.9, ls=(0, (1, 1.4)),
                        alpha=0.75, label=f"realised (+{t['dmu_chi_realised']:.4f})"))
        axL.set_xlabel(r"$\Delta\mu_\chi$")
        axL.set_ylabel(r"$p(\Delta\mu_\chi)$")
        axL.legend(handles=handles, loc="upper left", fontsize=5.6,
                   frameon=False, handlelength=1.5)
    axL.set_title("spin mark", fontsize=7.6, pad=3)

    # ---- panel R: p(dmu_G), M + J, with the 21-node coarse overlay --------
    arms_R = []
    if M is not None:
        arms_R.append(("A10-M", M, C_M))
    if J is not None:
        arms_R.append(("A10-J", J, C_J))
    if not arms_R:
        notice("fig_a10_marks panel R (dmu_G): neither A10-M nor A10-J is "
               "available")
        axR.text(0.5, 0.52, "A10-M / A10-J\nnot yet available",
                  ha="center", va="center", transform=axR.transAxes,
                  fontsize=8.2, color=fs.MUTED)
        axR.set_xticks([])
        axR.set_yticks([])
        axR.grid(False)
    else:
        ymax, rows, handles = 0.0, [], []
        for name, d, color in arms_R:
            js = d["json"]
            s = summarise(d["dmu_G_grid"], js["dmu_G"]["marginal_logp"])
            print(f"    {name}  p(dmu_G)  ({d['dmu_G_grid'].size}-node axis)")
            check("dmu_G median", s["median"], js["dmu_G"]["median"])
            check("dmu_G 90% low", s["ci90"][0], js["dmu_G"]["ci90"][0])
            check("dmu_G 90% high", s["ci90"][1], js["dmu_G"]["ci90"][1])
            print(f"        dmu_G 68% [{s['ci68'][0]:.6f}, {s['ci68'][1]:.6f}]  "
                  "(printed only, never plotted)")
            axR.plot(s["x"], s["p"], color=color, lw=1.6, zorder=3)
            ymax = max(ymax, float(np.nanmax(s["p"])))
            rows.append((color, s["ci90"][0], s["ci90"][1], s["median"]))
            handles.append(Line2D([], [], color=color, lw=1.6,
                            label=f"{name}  {s['median']:.3f} "
                                  f"[{s['ci90'][0]:.3f}, {s['ci90'][1]:.3f}]"))
            if name == "A10-M" and "coarse_axis_comparison" in js:
                c21 = js["coarse_axis_comparison"]["registered_21_node"]
                x21 = np.asarray(c21["dmu_G_nodes"], float)
                s21 = summarise(x21, c21["dmu_G"]["marginal_logp"])
                print("    A10-M  p(dmu_G), 21-node coarse axis")
                check("dmu_G median (21-node)", s21["median"], c21["dmu_G"]["median"])
                check("dmu_G 90% low (21-node)", s21["ci90"][0],
                      c21["dmu_G"]["ci90"][0])
                check("dmu_G 90% high (21-node)", s21["ci90"][1],
                      c21["dmu_G"]["ci90"][1])
                print(f"        A10-M correlation_f_dmu_G (own 2-D posterior) "
                      f"= {js['correlation_f_dmu_G']:+.6f}")
                axR.plot(s21["x"], s21["p"], color=color, lw=1.0,
                         ls=(0, (1, 1)), zorder=2)
                handles.append(Line2D([], [], color=color, lw=1.0, ls=(0, (1, 1)),
                                label="A10-M, 21-node axis"))
        interval_strip(axR, rows, ymax)
        t = arms_R[0][1]["json"]["truth"]
        axR.axvline(t["dmu_G_planted"], color=fs.TRUTH, lw=0.9, ls=(0, (3, 2)),
                    alpha=0.75, zorder=1.5)
        handles.append(Line2D([], [], color=fs.TRUTH, lw=0.9, ls=(0, (3, 2)),
                        alpha=0.75,
                        label=f"planted (+{t['dmu_G_planted']:.1f} $M_\\odot$)"))
        axR.set_xlabel(r"$\Delta\mu_{\rm G}$  [$M_\odot$]")
        axR.set_ylabel(r"$p(\Delta\mu_{\rm G})$")
        axR.legend(handles=handles, loc="upper left", fontsize=5.6,
                   frameon=False, handlelength=1.5)
    axR.set_title("mass mark", fontsize=7.6, pad=3)

    save(fig, "fig_a10_marks")


# --------------------------------------------------------------------------- #
# 3. J's 2-D planes, single-mark arms overlaid
# --------------------------------------------------------------------------- #
def fig_a10_planes(J, chi, M):
    print("\n[3] fig_a10_planes  (A10-J's 90% HPD planes, single-mark overlays)")
    if J is None:
        notice("fig_a10_planes: A10-J is not available yet; figure not drawn")
        return

    js = J["json"]
    truth = js["truth"]
    rho = js["correlations"]
    fgrid, mg, mu = J["f_grid"], J["dmu_G_grid"], J["dmu_chi_grid"]

    fig, axes = plt.subplots(1, 3, figsize=(fs.TWOCOL, 2.6))
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.165, top=0.90,
                        wspace=0.33)

    # panel 1: (f, dmu_chi) -- chi overlay
    ax = axes[0]
    hpd_plane(ax, fgrid, mu, J["f_dmu_chi"], C_J)
    if chi is not None:
        hpd_plane(ax, chi["f_grid"], chi["dmu_chi_grid"], chi["P"], C_CHI,
                   fill=False)
    else:
        notice("fig_a10_planes panel 1: A10-chi not available; J drawn alone")
    ax.axvline(truth["f_agn_planted"], color=fs.TRUTH, lw=0.9, ls=(0, (3, 2)),
               alpha=0.75)
    ax.axhline(truth["dmu_chi_planted"], color=fs.TRUTH, lw=0.9, ls=(0, (3, 2)),
               alpha=0.75)
    ax.axhline(truth["dmu_chi_realised"], color=fs.TRUTH, lw=0.9,
               ls=(0, (1, 1.4)), alpha=0.75)
    ax.set_xlabel(r"$f_{\rm AGN}$")
    ax.set_ylabel(r"$\Delta\mu_\chi$")
    ax.annotate(f"$\\rho$ = {rho['f_agn|dmu_chi']:+.3f}", (0.04, 0.95),
                xycoords="axes fraction", ha="left", va="top", fontsize=6.6,
                color=fs.INK)
    print(f"    panel (f, dmu_chi): rho = {rho['f_agn|dmu_chi']:+.6f} (json)")

    # panel 2: (f, dmu_G) -- M overlay
    ax = axes[1]
    hpd_plane(ax, fgrid, mg, J["f_dmu_G"], C_J)
    if M is not None:
        hpd_plane(ax, M["f_grid"], M["dmu_G_grid"], M["P"], C_M, fill=False)
    else:
        notice("fig_a10_planes panel 2: A10-M not available; J drawn alone")
    ax.axvline(truth["f_agn_planted"], color=fs.TRUTH, lw=0.9, ls=(0, (3, 2)),
               alpha=0.75)
    ax.axhline(truth["dmu_G_planted"], color=fs.TRUTH, lw=0.9, ls=(0, (3, 2)),
               alpha=0.75)
    ax.set_xlabel(r"$f_{\rm AGN}$")
    ax.set_ylabel(r"$\Delta\mu_{\rm G}$  [$M_\odot$]")
    ax.annotate(f"$\\rho$ = {rho['f_agn|dmu_G']:+.3f}", (0.04, 0.95),
                xycoords="axes fraction", ha="left", va="top", fontsize=6.6,
                color=fs.INK)
    print(f"    panel (f, dmu_G): rho = {rho['f_agn|dmu_G']:+.6f} (json)")

    # panel 3: (dmu_G, dmu_chi) -- no single-mark analogue
    ax = axes[2]
    hpd_plane(ax, mg, mu, J["dmu_G_dmu_chi"], C_J)
    ax.axvline(truth["dmu_G_planted"], color=fs.TRUTH, lw=0.9, ls=(0, (3, 2)),
               alpha=0.75)
    ax.axhline(truth["dmu_chi_planted"], color=fs.TRUTH, lw=0.9, ls=(0, (3, 2)),
               alpha=0.75)
    ax.axhline(truth["dmu_chi_realised"], color=fs.TRUTH, lw=0.9,
               ls=(0, (1, 1.4)), alpha=0.75)
    ax.set_xlabel(r"$\Delta\mu_{\rm G}$  [$M_\odot$]")
    ax.set_ylabel(r"$\Delta\mu_\chi$")
    ax.annotate(f"$\\rho$ = {rho['dmu_G|dmu_chi']:+.3f}", (0.04, 0.95),
                xycoords="axes fraction", ha="left", va="top", fontsize=6.6,
                color=fs.INK)
    print(f"    panel (dmu_G, dmu_chi): rho = {rho['dmu_G|dmu_chi']:+.6f} (json)")

    leg1 = [Patch(fc=C_J, alpha=0.20, ec=C_J, lw=1.3, label="A10-J, 90%")]
    if chi is not None:
        leg1.append(Line2D([], [], color=C_CHI, lw=1.3, label="A10-$\\chi$, 90%"))
    axes[0].legend(handles=leg1, loc="lower right", fontsize=5.6, frameon=False,
                   handlelength=1.4)
    leg2 = [Patch(fc=C_J, alpha=0.20, ec=C_J, lw=1.3, label="A10-J, 90%")]
    if M is not None:
        leg2.append(Line2D([], [], color=C_M, lw=1.3, label="A10-M, 90%"))
    axes[1].legend(handles=leg2, loc="lower right", fontsize=5.6, frameon=False,
                   handlelength=1.4)

    save(fig, "fig_a10_planes")


# --------------------------------------------------------------------------- #
def main():
    print(f"A10     = {A10}")
    print(f"RESULTS = {RESULTS}")
    print(f"FIGS    = {FIGS}")

    S, M, chi, J = load_S(), load_M(), load_chi(), load_J()

    print("\n[grid consistency checks, where more than one arm shares an axis]")
    if S is not None and M is not None:
        ok = np.array_equal(S["f_grid"], M["f_grid"])
        print(f"    S.f_grid == M.f_grid                 : {'OK' if ok else 'MISMATCH'}")
    if chi is not None and J is not None:
        okf = np.array_equal(chi["f_grid"], J["f_grid"])
        okm = np.array_equal(chi["dmu_chi_grid"], J["dmu_chi_grid"])
        print(f"    chi.f_grid == J.f_grid               : {'OK' if okf else 'MISMATCH'}")
        print(f"    chi.dmu_chi_grid == J.dmu_chi_grid   : {'OK' if okm else 'MISMATCH'}")
    if M is not None and J is not None:
        okf = np.array_equal(M["f_grid"], J["f_grid"])
        okg = np.array_equal(M["dmu_G_grid"], J["dmu_G_grid"])
        print(f"    M.f_grid == J.f_grid                 : {'OK' if okf else 'MISMATCH'}")
        print(f"    M.dmu_G_grid == J.dmu_G_grid         : {'OK' if okg else 'MISMATCH'}")

    fig_a10_fagn(S, chi, M, J)
    fig_a10_marks(chi, M, J)
    fig_a10_planes(J, chi, M)

    print("\n[files read]")
    for p in READS:
        print(f"    {p}")
    if SKIPPED:
        print("\n[skipped -- rerun this script once these land]")
        for msg in SKIPPED:
            print(f"    {msg}")
    print("\ndone.")


if __name__ == "__main__":
    main()
