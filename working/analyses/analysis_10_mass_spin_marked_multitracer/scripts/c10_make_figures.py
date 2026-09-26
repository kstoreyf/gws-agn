#!/usr/bin/env python3
"""Production figures for Analysis 10's COSMOLOGY stage (H0 released).

    python scripts/c10_make_figures.py      # writes whatever arms exist, pdf + png

CPU only, matplotlib Agg, no likelihood, deterministic: it reads
``results/c10_arm_S.{h5,json}``, ``results/c10_arm_J.{h5,json}`` and
``results/c10_mech.{h5,json}`` and computes nothing beyond flat-prior
marginals and equal-tailed / HPD summaries of what is already stored there.
Every number drawn is printed to stdout beside the value the corresponding
JSON recorded (Gate D2).  This is the sibling of ``make_figures.py`` (the
fixed-H0 stage's figure script, NOT modified here): the house conventions --
colour identities, the 90%-only rule, the drawn-vs-JSON check() printing --
are copied from it rather than imported, so this script has no import-time
dependency on it.

As of 2026-09-22: ``c10_arm_S.h5/json`` and ``c10_mech.h5/json`` exist.
``c10_arm_J.h5/json`` does NOT exist yet -- the 41 x 15 x 13 x (H0-window)
cube (``c10_scan.py --stage j``) is still running (~30 h ETA) and its
assemble stage (``stage_j_assemble``) has not been invoked.  Its exact
schema below (4-D ``log_likelihood`` on axis order ``(H0, f_agn, dmu_G,
dmu_chi)``, 1-D ``marginal/{H0,f_agn,dmu_G,dmu_chi}``, the six 2-D
``marginal_2d/*`` planes, and the JSON's ``correlations`` / ``map`` /
``edge_mass`` / ``H0_width_vs_C10_S_matched_lattice`` blocks) was read out of
``c10_scan.py``'s ``_marginals_4d`` / ``stage_j_assemble`` source, not off a
live file, and should be re-verified once the cube lands -- this script will
raise a ``KeyError``/``FileNotFoundError`` at the matching load function if a
name has changed, which is the fast way to catch it.  Every panel that needs
C10-J is skipped with a printed notice when the file is missing, so this
script completes now (figures 1 without its J curve, 3 renders in full) and
will complete again, in full, once C10-J assembles.

  1. fig_c10_h0      p(H0): C10-S (aqua, its own 202-node marginal, PLUS its
                     re-marginalisation on the C10-J H0 lattice as a thin
                     dotted aqua curve once C10-J exists) and C10-J (blue,
                     marginal over f_AGN, dmu_chi, dmu_G), planted 67.74 as a
                     dashed line, 90% bars under the curves.  x in [60, 80]
                     so C10-S's long high-H0 tail is visible.  Analysis 9's
                     S9/J9 H0 posteriors (the SPIN-ONLY mock -- a different
                     mock) are printed and annotated in grey text only, never
                     drawn as a curve.
  2. fig_c10_planes  three panels of C10-J's 90% HPD regions: (H0, f_AGN),
                     (H0, dmu_chi), (H0, dmu_G), with rho printed from the
                     JSON's own correlation block and the planted truths
                     (67.74; 0.30; +0.10; +5) as dashed lines.  Skipped
                     entirely until C10-J exists.
  3. fig_c10_mech    p(H0) for the four mechanism arms on the run's own
                     h0_window -- P0 (spatial only, aqua; the h0_window slice
                     of C10-S, NOT recomputed), P1 ([GAL,AGN], marks pinned
                     at the fixed-H0 MAP, blue), I0 ([GAL,GAL], marks 0, grey
                     solid), I1 ([GAL,GAL], marks pinned, grey dashed) -- 90%
                     bars, the planted line, and the width ratios from the
                     JSON's own ``ratios`` block in a text box.  I0's 90%
                     upper bound sits at the window's own edge (see the
                     printed note): not a contained interval.

Reductions:

  * flat priors on every axis, so the posterior is the exponentiated
    log-likelihood; guard-rejected cells carry logL = -inf, hence zero
    density;
  * marginals by trapezoid integration of the STORED marginal_logp, exactly
    ``analysis_2/scripts/scan_h0f.py::marginal_ci`` (imported nowhere here --
    ``summarise()`` below is a literal reimplementation, matching
    ``make_figures.py``'s own copy of the same routine);
  * the C10-S-on-the-C10-J-lattice curve replicates ``c10_scan.py``'s own
    ``_matched_28_node_lattice`` / ``_j_vs_s_matched_lattice`` construction:
    select C10-S's own bitwise H0 nodes matching each C10-J H0 node, exponent-
    iate that sub-cube's log-likelihood relative to ITS OWN max, trapezoid
    over f, and hand the result to the same ``summarise()``;
  * 2-D regions are highest-posterior-density, ``figstyle.hpd_levels_2d``.

House conventions, from ../../../paper/scripts/figstyle.py: colour identities
match ``make_figures.py`` exactly (same module, same slots) --

  * joint (C10-J)   = blue  -- the running Analysis 8/9/10 "joint" identity.
  * spatial (C10-S) = aqua  -- the running "spatial" identity.
  * mechanism arms P0 (aqua) and P1 (blue) borrow those two identities since
    they ARE the spatial-only and full-marked-joint arms restricted to the
    mechanism h0_window; I0/I1 are neutral grey (fs.MUTED), a DIAGNOSTIC
    construction, not a production arm, so they get no series colour.

EVERY plotted interval, band and contour is 90%; 68% numbers are printed to
stdout and never plotted.  Reference lines (truths, axis edges) are neutral
ink, never a series colour.
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
A9_RESULTS = WORKING / "analyses" / "analysis_9_marked_multitracer_H0_fagn" / "results"

if str(PAPER_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(PAPER_SCRIPTS))
import figstyle as fs  # noqa: E402

fs.use()

# ---- arm identities, copied verbatim from make_figures.py's own slots ----- #
C_J, C_S, C_CHI, C_M = fs.C["blue"], fs.C["aqua"], fs.C["orange"], fs.C["magenta"]

READS: list[str] = []
SKIPPED: list[str] = []


# --------------------------------------------------------------------------- #
# small helpers -- copies of make_figures.py's own (that script is READ-ONLY,
# not imported, so these are literal reimplementations, not references).
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

    Literal reimplementation of ``analysis_2/scripts/scan_h0f.py::marginal_ci``
    (also not imported), so recomputing it from a stored ``marginal_logp``
    reproduces the JSON's own median/ci68/ci90 to floating-point precision.
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
    """Short 90% bars + median ticks, stacked beneath the density curves."""
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


def s_on_j_lattice(s_h0, s_f, s_ll, j_h0):
    """C10-S's own cube, re-marginalised on exactly the given C10-J H0 nodes.

    Same construction as ``c10_scan.py::_matched_28_node_lattice`` /
    ``_j_vs_s_matched_lattice``: select C10-S's own bitwise H0 nodes matching
    each C10-J node (returns None, not an exception, if any node is missing --
    this is a figure script, it degrades to a printed notice, never a crash),
    exponentiate relative to the SUB-cube's own max, trapezoid over f.
    """
    idx = []
    for v in j_h0:
        hit = np.where(s_h0 == v)[0]
        if hit.size != 1:
            return None
        idx.append(int(hit[0]))
    idx = np.asarray(idx)
    sub_ll = s_ll[idx, :]
    llm = np.where(np.isfinite(sub_ll), sub_ll, -np.inf)
    if not np.isfinite(llm).any():
        return None
    mx = float(llm[np.isfinite(llm)].max())
    P = np.where(np.isfinite(llm), np.exp(llm - mx), 0.0)
    with np.errstate(divide="ignore"):
        lp = np.log(np.trapz(P, s_f, axis=1))
    return summarise(j_h0, lp)


# --------------------------------------------------------------------------- #
# loaders -- each returns None (and prints a notice) if its file is missing
# --------------------------------------------------------------------------- #
def load_S():
    j = _read_json(RESULTS / "c10_arm_S.json")
    h = _read_h5(RESULTS / "c10_arm_S.h5")
    if j is None or h is None:
        notice("C10-S: results/c10_arm_S.{json,h5} not found")
        return None
    d = {
        "json": j,
        "H0_grid": np.asarray(h["H0_grid"][:], float),
        "f_grid": np.asarray(h["f_grid"][:], float),
        "log_likelihood": np.asarray(h["log_likelihood"][:], float),
    }
    h.close()
    return d


def load_J():
    """C10-J: the cosmology-stage joint cube (c10_scan.py, j_assemble NOT run
    yet as of 2026-09-22; the j stage cube itself is still ~30h from done)."""
    j = _read_json(RESULTS / "c10_arm_J.json")
    h = _read_h5(RESULTS / "c10_arm_J.h5")
    if j is None or h is None:
        notice("C10-J: results/c10_arm_J.{json,h5} not found yet (written by "
               "c10_scan.py --stage j_assemble once the cube lands)")
        return None
    d = {
        "json": j,
        "H0_grid": np.asarray(h["H0_grid"][:], float),
        "f_grid": np.asarray(h["f_grid"][:], float),
        "dmu_G_grid": np.asarray(h["dmu_G_grid"][:], float),
        "dmu_chi_grid": np.asarray(h["dmu_chi_grid"][:], float),
        "H0_f_agn": np.asarray(h["marginal_2d/H0_f_agn"][:], float),
        "H0_dmu_G": np.asarray(h["marginal_2d/H0_dmu_G"][:], float),
        "H0_dmu_chi": np.asarray(h["marginal_2d/H0_dmu_chi"][:], float),
    }
    h.close()
    return d


def load_mech():
    j = _read_json(RESULTS / "c10_mech.json")
    h = _read_h5(RESULTS / "c10_mech.h5")
    if j is None or h is None:
        notice("C10-mech: results/c10_mech.{json,h5} not found")
        return None
    h.close()  # everything needed for this figure is already in the JSON
    return {"json": j}


# --------------------------------------------------------------------------- #
# 1. p(H0): C10-S vs C10-J
# --------------------------------------------------------------------------- #
def fig_c10_h0(S, J):
    print("\n[1] fig_c10_h0  (p(H0); C10-S vs C10-J, x in [60, 80])")
    fig, ax = plt.subplots(figsize=(fs.ONECOL, 2.7))
    fig.subplots_adjust(left=0.165, right=0.97, bottom=0.15, top=0.90)

    strip_rows, handles, ymax = [], [], 0.0

    if S is not None:
        js = S["json"]
        s = summarise(S["H0_grid"], js["H0"]["marginal_logp"])
        print("    C10-S")
        check("H0 median", s["median"], js["H0"]["median"])
        check("H0 90% low", s["ci90"][0], js["H0"]["ci90"][0])
        check("H0 90% high", s["ci90"][1], js["H0"]["ci90"][1])
        print(f"        H0 68% [{s['ci68'][0]:.6f}, {s['ci68'][1]:.6f}]  "
              f"(json [{js['H0']['ci68'][0]:.6f}, {js['H0']['ci68'][1]:.6f}]; "
              f"printed only, never plotted)")
        ax.plot(s["x"], s["p"], color=C_S, lw=1.6, zorder=3)
        ymax = max(ymax, float(np.nanmax(s["p"])))
        strip_rows.append((C_S, s["ci90"][0], s["ci90"][1], s["median"]))
        handles.append(Line2D([], [], color=C_S, lw=1.6,
                        label=f"C10-S  {s['median']:.2f} "
                              f"[{s['ci90'][0]:.2f}, {s['ci90'][1]:.2f}]"))

        if J is not None:
            sub = s_on_j_lattice(S["H0_grid"], S["f_grid"], S["log_likelihood"],
                                  J["H0_grid"])
            if sub is not None:
                print("    C10-S, re-marginalised on the C10-J H0 lattice "
                      "(\"matched lattice\")")
                matched = js.get("matched_28_node_lattice")
                if (matched is not None
                        and np.array_equal(np.asarray(matched["j_nodes"], float),
                                            J["H0_grid"])):
                    ref = matched["S_on_the_matched_J_nodes"]
                    check("S-on-J median", sub["median"], ref["median"])
                    check("S-on-J 90% low", sub["ci90"][0], ref["ci90"][0])
                    check("S-on-J 90% high", sub["ci90"][1], ref["ci90"][1])
                else:
                    print("        [note] C10-J's H0 axis differs from C10-S's "
                          "registered matched_28_node_lattice; matched-lattice "
                          "numbers computed fresh here, no JSON cross-check "
                          "available for THIS axis")
                ax.plot(sub["x"], sub["p"], color=C_S, lw=1.0, ls=(0, (1, 1)),
                        zorder=2)
                handles.append(Line2D([], [], color=C_S, lw=1.0, ls=(0, (1, 1)),
                                label="C10-S, matched lattice"))
            else:
                notice("fig_c10_h0: C10-S could not be re-marginalised on "
                       "C10-J's H0 lattice (axis mismatch); dotted curve "
                       "skipped")
    else:
        notice("fig_c10_h0: C10-S not available; panel drawn without it")

    if J is not None:
        js = J["json"]
        s = summarise(J["H0_grid"], js["H0"]["marginal_logp"])
        print("    C10-J")
        check("H0 median", s["median"], js["H0"]["median"])
        check("H0 90% low", s["ci90"][0], js["H0"]["ci90"][0])
        check("H0 90% high", s["ci90"][1], js["H0"]["ci90"][1])
        print(f"        H0 68% [{s['ci68'][0]:.6f}, {s['ci68'][1]:.6f}]  "
              f"(json [{js['H0']['ci68'][0]:.6f}, {js['H0']['ci68'][1]:.6f}]; "
              f"printed only, never plotted)")
        ax.plot(s["x"], s["p"], color=C_J, lw=1.6, zorder=3)
        ymax = max(ymax, float(np.nanmax(s["p"])))
        strip_rows.append((C_J, s["ci90"][0], s["ci90"][1], s["median"]))
        handles.append(Line2D([], [], color=C_J, lw=1.6,
                        label=f"C10-J  {s['median']:.2f} "
                              f"[{s['ci90'][0]:.2f}, {s['ci90'][1]:.2f}]"))
    else:
        notice("fig_c10_h0: C10-J not available yet; drawn without it")

    if strip_rows:
        interval_strip(ax, strip_rows, ymax)
    ax.axvline(67.74, color=fs.TRUTH, lw=0.9, ls=(0, (3, 2)), alpha=0.75,
               zorder=1.5)
    handles.append(Line2D([], [], color=fs.TRUTH, lw=0.9, ls=(0, (3, 2)),
                    alpha=0.75, label="planted (67.74)"))

    # Analysis 9's S9/J9 H0 (the SPIN-ONLY mock, a DIFFERENT mock): printed
    # and annotated in grey text ONLY, never drawn as a curve.
    ref_path = A9_RESULTS / "section_11_comparison.json"
    if ref_path.exists():
        ref = json.loads(ref_path.read_text())
        READS.append(str(ref_path))
        blk = ref["comparisons"]["H0_spatial_S9_vs_marked_J9"]
        s9, j9 = blk["S9"], blk["J9"]
        print("    [reference, DIFFERENT mock -- A9 spin-only, never drawn as "
              "a curve]")
        print(f"        A9 S9 (spatial) H0  median {s9['median']:.6f}  "
              f"90% [{s9['ci90'][0]:.6f}, {s9['ci90'][1]:.6f}]")
        print(f"        A9 J9 (marked)  H0  median {j9['median']:.6f}  "
              f"90% [{j9['ci90'][0]:.6f}, {j9['ci90'][1]:.6f}]")
        ax.annotate(
            "ref. only, spin-only mock (A9):\n"
            f"S9  {s9['median']:.2f} [{s9['ci90'][0]:.2f}, {s9['ci90'][1]:.2f}]\n"
            f"J9  {j9['median']:.2f} [{j9['ci90'][0]:.2f}, {j9['ci90'][1]:.2f}]",
            (0.985, 0.62), xycoords="axes fraction", ha="right", va="top",
            fontsize=5.2, color=fs.MUTED)
    else:
        notice("fig_c10_h0: A9 results/section_11_comparison.json not found; "
               "reference annotation omitted")

    ax.set_xlabel(r"$H_0$  [km s$^{-1}$ Mpc$^{-1}$]")
    ax.set_ylabel(r"$p(H_0)$")
    ax.set_xlim(60.0, 80.0)
    if handles:
        ax.legend(handles=handles, loc="upper right", fontsize=5.8,
                  frameon=False, handlelength=1.6)
    if J is None:
        ax.annotate("C10-J not yet available", (0.04, 0.95),
                    xycoords="axes fraction", ha="left", va="top",
                    fontsize=5.8, color=fs.MUTED)
    save(fig, "fig_c10_h0")


# --------------------------------------------------------------------------- #
# 2. C10-J's 90% HPD planes against H0
# --------------------------------------------------------------------------- #
def fig_c10_planes(J):
    print("\n[2] fig_c10_planes  (C10-J 90% HPD planes: (H0,f_AGN), "
          "(H0,dmu_chi), (H0,dmu_G))")
    if J is None:
        notice("fig_c10_planes: C10-J is not available yet; figure not drawn")
        return

    js = J["json"]
    truth = js["truth"]
    rho = js["correlations"]
    h0 = J["H0_grid"]

    fig, axes = plt.subplots(1, 3, figsize=(fs.TWOCOL, 2.6))
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.165, top=0.90,
                        wspace=0.33)

    specs = [
        ("f_agn", J["f_grid"], J["H0_f_agn"], r"$f_{\rm AGN}$",
         truth["f_agn_planted"], None, "H0|f_agn"),
        ("dmu_chi", J["dmu_chi_grid"], J["H0_dmu_chi"], r"$\Delta\mu_\chi$",
         truth["dmu_chi_planted"], truth["dmu_chi_realised"], "H0|dmu_chi"),
        ("dmu_G", J["dmu_G_grid"], J["H0_dmu_G"], r"$\Delta\mu_{\rm G}$  "
         r"[$M_\odot$]", truth["dmu_G_planted"], None, "H0|dmu_G"),
    ]
    for ax, (name, y_axis, P, ylabel, planted, realised, rho_key) in zip(axes, specs):
        hpd_plane(ax, h0, y_axis, P, C_J)
        ax.axvline(67.74, color=fs.TRUTH, lw=0.9, ls=(0, (3, 2)), alpha=0.75)
        ax.axhline(planted, color=fs.TRUTH, lw=0.9, ls=(0, (3, 2)), alpha=0.75)
        if realised is not None:
            ax.axhline(realised, color=fs.TRUTH, lw=0.9, ls=(0, (1, 1.4)),
                       alpha=0.75)
        ax.set_xlabel(r"$H_0$  [km s$^{-1}$ Mpc$^{-1}$]")
        ax.set_ylabel(ylabel)
        r = rho[rho_key]
        ax.annotate(f"$\\rho$ = {r:+.3f}", (0.04, 0.95),
                    xycoords="axes fraction", ha="left", va="top",
                    fontsize=6.6, color=fs.INK)
        print(f"    panel (H0, {name}): rho = {r:+.6f} (json)")

    axes[0].legend(handles=[
        Patch(fc=C_J, alpha=0.20, ec=C_J, lw=1.3, label="C10-J, 90%"),
        Line2D([], [], color=fs.TRUTH, lw=0.9, ls=(0, (3, 2)), label="planted"),
    ], loc="lower right", fontsize=5.6, frameon=False, handlelength=1.4)

    save(fig, "fig_c10_planes")


# --------------------------------------------------------------------------- #
# 3. p(H0), the four mechanism arms
# --------------------------------------------------------------------------- #
def fig_c10_mech(M):
    print("\n[3] fig_c10_mech  (p(H0); mechanism arms P0/P1/I0/I1)")
    if M is None:
        notice("fig_c10_mech: results/c10_mech.{json,h5} not found; figure "
               "not drawn")
        return

    js = M["json"]
    h0 = np.asarray(js["H0_grid"], float)
    window = js["h0_window"]
    specs = [
        ("P0", C_S, "-", "spatial only"),
        ("P1", C_J, "-", "[GAL,AGN], marks pinned"),
        ("I0", fs.MUTED, "-", "[GAL,GAL], marks 0"),
        ("I1", fs.MUTED, (0, (4, 2)), "[GAL,GAL], marks pinned"),
    ]

    fig, ax = plt.subplots(figsize=(fs.ONECOL, 2.9))
    fig.subplots_adjust(left=0.165, right=0.97, bottom=0.33, top=0.90)

    ymax, strip_rows, handles = 0.0, [], []
    for arm, color, ls, desc in specs:
        aj = js["arms"][arm]
        s = summarise(h0, aj["H0"]["marginal_logp"])
        print(f"    {arm}  ({desc})")
        check("H0 median", s["median"], aj["H0"]["median"])
        check("H0 90% low", s["ci90"][0], aj["H0"]["ci90"][0])
        check("H0 90% high", s["ci90"][1], aj["H0"]["ci90"][1])
        print(f"        H0 68% [{s['ci68'][0]:.6f}, {s['ci68'][1]:.6f}]  "
              f"(printed only, never plotted)")
        ax.plot(s["x"], s["p"], color=color, lw=1.6, ls=ls, zorder=3)
        ymax = max(ymax, float(np.nanmax(s["p"])))
        strip_rows.append((color, s["ci90"][0], s["ci90"][1], s["median"]))
        handles.append(Line2D([], [], color=color, lw=1.6, ls=ls,
                        label=f"{arm}  {s['median']:.2f} "
                              f"[{s['ci90'][0]:.2f}, {s['ci90'][1]:.2f}]"))

    interval_strip(ax, strip_rows, ymax)
    ax.axvline(67.74, color=fs.TRUTH, lw=0.9, ls=(0, (3, 2)), alpha=0.75,
               zorder=1.5)
    handles.append(Line2D([], [], color=fs.TRUTH, lw=0.9, ls=(0, (3, 2)),
                    alpha=0.75, label="planted (67.74)"))
    ax.set_xlabel(r"$H_0$  [km s$^{-1}$ Mpc$^{-1}$]")
    ax.set_ylabel(r"$p(H_0)$")
    ax.set_xlim(window[0], window[1])
    ax.legend(handles=handles, loc="upper right", fontsize=5.4, frameon=False,
              handlelength=1.6)

    r68, r90 = js["ratios"]["ci68"], js["ratios"]["ci90"]
    print(f"    ratios (json, ci68): {json.dumps(r68)}")
    print(f"    ratios (json, ci90): {json.dumps(r90)}")
    # 90% widths only (the plotted-interval rule).  The I1/I0 ratio is NOT
    # labelled a spectral-siren share: on this mock [GAL,GAL] also gives the
    # AGN-hosted events the wrong spatial prior, so arm I carries its own
    # H0 shift and the attribution is confounded (REPORT.md, cosmology stage).
    txt = (
        f"90% widths:  P1/P0 {r90['W_P1_over_W_P0__total_gain']:.3f}   "
        f"I1/I0 {r90['W_I1_over_W_I0__spectral_siren_only_gain']:.3f}"
    )
    fig.text(0.03, 0.13, txt, ha="left", va="top", fontsize=5.6, color=fs.INK2)

    i0 = js["arms"]["I0"]["H0"]
    note = (f"I0 peaks at the window edge (mode {i0['marginal_mode']:.1f}); "
            f"its interval is truncated by the window, not contained.")
    fig.text(0.03, 0.07, note, ha="left", va="top", fontsize=5.2,
              color=fs.MUTED, wrap=True)
    print(f"    [note] {note}")

    save(fig, "fig_c10_mech")


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# 4. C10-3: the single-tracer spectral-siren control
# --------------------------------------------------------------------------- #
def fig_c10_control():
    print("\n[4] fig_c10_control  (p(H0); single-tracer arms B5M, B0M, B5U)")
    jp, hp = RESULTS / "c10_control.json", RESULTS / "c10_control.h5"
    if not (jp.exists() and hp.exists()):
        notice("fig_c10_control: results/c10_control.{json,h5} not found")
        return
    READS.extend([str(jp), str(hp)])
    js = json.loads(jp.read_text())
    arms = (("B5M", C_M, "B5M  both marks"),
            ("B0M", C_CHI, "B0M  twin, spin only"),
            ("B5U", fs.MUTED, "B5U  no marks"))
    fig, ax = plt.subplots(figsize=(fs.ONECOL, 2.7))
    fig.subplots_adjust(left=0.165, right=0.97, bottom=0.15, top=0.90)
    rows, handles, ymax = [], [], 0.0
    with h5py.File(hp, "r") as h:
        f = h["f_grid"][:]
        for arm, col, lab in arms:
            H = h[f"{arm}/H0_grid"][:]
            L = h[f"{arm}/log_likelihood"][:]
            P = np.exp(np.where(np.isfinite(L), L, -np.inf) - np.nanmax(L))
            s = summarise(H, np.log(np.trapz(P, f, axis=1)))
            b = js["arms"][arm]
            print(f"    {arm}")
            check("H0 median", s["median"], b["H0"]["median"])
            check("H0 90% low", s["ci90"][0], b["H0"]["ci90"][0])
            check("H0 90% high", s["ci90"][1], b["H0"]["ci90"][1])
            print(f"        H0 68% [{s['ci68'][0]:.4f}, {s['ci68'][1]:.4f}] (printed only)")
            ax.plot(s["x"], s["p"], color=col, lw=1.5, zorder=3)
            ymax = max(ymax, float(np.nanmax(s["p"])))
            rows.append((col, s["ci90"][0], s["ci90"][1], s["median"]))
            trunc = "" if b["H0_contained_1e-6"] else " (truncated)"
            handles.append(Line2D([], [], color=col, lw=1.5,
                                  label=f"{lab}{trunc}"))
    interval_strip(ax, rows, ymax)
    ax.set_ylim(ax.get_ylim()[0], ymax * 1.6)
    ax.axvline(67.74, color=fs.TRUTH, lw=0.9, ls=(0, (3, 2)), alpha=0.75, zorder=1.5)
    handles.append(Line2D([], [], color=fs.TRUTH, lw=0.9, ls=(0, (3, 2)), alpha=0.75,
                          label="planted (67.74)"))
    ax.set_xlabel(r"$H_0$  [km s$^{-1}$ Mpc$^{-1}$]")
    ax.set_ylabel(r"$p(H_0)$")
    ax.set_xlim(55.0, 100.0)
    ax.legend(handles=handles, loc="upper right", fontsize=5.4, frameon=False,
              handlelength=1.6)
    ax.annotate("all hosts GAL; branch drawn\nindependently of host (p = 0.30)",
                (0.985, 0.50), xycoords="axes fraction", ha="right", va="top",
                fontsize=5.2, color=fs.MUTED)
    save(fig, "fig_c10_control")


# --------------------------------------------------------------------------- #
# 5. C10-2: routing -- fixed-f width ratio, and per-event score vs |dP|
# --------------------------------------------------------------------------- #
def fig_c10_routing():
    print("\n[5] fig_c10_routing  (fixed-f P1/P0; per-event d score vs |dP| at 67.5)")
    jp = A10 / "diagnostics" / "c10_event_routing.json"
    hp = A10 / "diagnostics" / "c10_event_routing.h5"
    if not (jp.exists() and hp.exists()):
        notice("fig_c10_routing: diagnostics/c10_event_routing.{json,h5} not found")
        return
    READS.extend([str(jp), str(hp)])
    js = json.loads(jp.read_text())
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(fs.TWOCOL, 2.5))
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.17, top=0.92, wspace=0.62)

    ff = js["fixed_f_width_ratio_P1_over_P0"]["per_f"]
    fx = np.array([r["f_agn"] for r in ff])
    r90 = np.array([r["ratio_90"] for r in ff])
    sh = np.array([r["shift_P1_minus_P0"] for r in ff])
    a1.plot(fx, r90, color=C_J, marker="o", ms=3, lw=1.3)
    a1.axhline(1.0, color=fs.TRUTH, lw=0.8, ls=(0, (3, 2)), alpha=0.7)
    a1.set_xlabel(r"$f_{\rm AGN}$ (held fixed)")
    a1.set_ylabel(r"$H_0$ 90% width, marked / spatial", color=C_J)
    b1 = a1.twinx()
    b1.plot(fx, sh, color=fs.MUTED, marker="s", ms=2.5, lw=1.0, ls=(0, (1, 1)))
    b1.set_ylabel(r"median shift, marked $-$ spatial", color=fs.MUTED)
    b1.set_ylim(-4.0, -2.0)
    print("    fixed-f ratios (90%):", np.round(r90, 4).tolist(), " shifts:",
          np.round(sh, 3).tolist())

    with h5py.File(hp, "r") as h:
        nodes = h["H0_nodes"][:]
        k = {float(x): i for i, x in enumerate(nodes)}
        a, lo, hi = k[67.5], k[67.0], k[68.0]
        lf = json.loads(h.attrs["log_w"])
        EGG = h["logE_GG"][:]

        def P(m):
            return 1.0 / (1.0 + np.exp((lf[0] + EGG[a]) - (lf[1] + h[f"logE_A_{m}"][a])))

        def score(m):
            Z = h[f"logZ_{m}"]
            return (Z[hi] - Z[lo]) / 1.0

        sG, PG = score("G"), P("G")
        for m, col, lab in (("chi", C_CHI, "spin mark"), ("M", C_M, "mass mark")):
            dP, ds = np.abs(P(m) - PG), score(m) - sG
            tot = js["centres"]["marked_peak"][m]["added_score"]["pe"]
            check(f"{m} summed added score", float(ds.sum()), tot, tol=1e-9)
            a2.scatter(dP, ds, s=3, color=col, alpha=0.55, lw=0, label=lab, zorder=3)
    a2.axhline(0.0, color=fs.TRUTH, lw=0.7, alpha=0.6)
    a2.set_xlabel(r"$|\Delta P_i({\rm AGN})|$ at $H_0 = 67.5$")
    a2.set_ylabel(r"$\Delta\, \partial \ln Z_i / \partial H_0$")
    a2.legend(loc="lower left", fontsize=5.8, frameon=False, markerscale=2.5)
    save(fig, "fig_c10_routing")


def main():
    print(f"A10     = {A10}")
    print(f"RESULTS = {RESULTS}")
    print(f"FIGS    = {FIGS}")

    S, J, M = load_S(), load_J(), load_mech()

    fig_c10_h0(S, J)
    fig_c10_planes(J)
    fig_c10_mech(M)
    fig_c10_control()
    fig_c10_routing()

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
