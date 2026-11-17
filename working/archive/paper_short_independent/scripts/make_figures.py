#!/usr/bin/env python
"""Figures of the letter (main.tex): figures/fig_concept (Fig. 1), fig_anchor (Fig. 2), fig1_budget (Fig. 3)
and fig_populations (Fig. 4); PDF and PNG.

    PY=/hildafs/home/magana/tmp_ondemand_hildafs_phy230014p_symlink/magana/.conda/envs/jax/bin/python
    $PY scripts/make_figures.py                 # all four
    $PY scripts/make_figures.py anchor          # one: concept | anchor | budget | populations

Fig. 1 (fig_concept) is final data (the simulated catalogs and mergers of realization A, no run result); its
function is the first draft's fig1_concept (archive/paper_short/scripts/make_figures.py), drawn here at the
letter's text width, and its numbers are checked against the caption macros of values_independent.tex.

Every number comes from scripts/results.py (real run summaries and grids where they exist, the expected
values of values_independent.tex otherwise). Style from ../paper_prd/scripts/figstyle.py. Marking: an
element that still waits for a run (a placeholder, or a mean or combination of which only some
realizations exist) is drawn in red tones, told apart by line style, marker and direct labels; a real
result is drawn in the final palette with the same line styles and markers. No text inside a figure says
which is which: the colour alone carries it. Every interval and region is 90%. Legend and label words are
the words of the captions in main.tex.

Widths: each figure is drawn at the width at which main.tex includes it, the REVTeX reprint text width
(510 pt) for the two full-width figures and the column width (246 pt) for Fig. 2.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import results as R  # noqa: E402  (puts ../paper_prd/scripts on the path)
import figstyle as fs  # noqa: E402
import make_figures as prd  # noqa: E402  (RED, shade90)
import build_values as bv  # noqa: E402  (distances, catalogs, localization regions of the long draft)
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import ConnectionPatch, Patch, Polygon  # noqa: E402
from matplotlib.ticker import NullLocator  # noqa: E402

FIGDIR = R.PAPER / "figures"
RED = prd.RED                              # strong, salmon, maroon, rose
INPUT = fs.INPUT
TW = 510.0 / 72.27                         # \textwidth of revtex4-2 reprint [in]
CW = 246.0 / 72.27                         # \columnwidth [in]

# final palette (Okabe-Ito): one population for all mergers = BOTH (green); a population for each tracer =
# reddish purple (lighter for the stated differences, darker for twice them); hosts known = black;
# galaxy catalog = GAL (blue); no catalog = dark grey
FINAL = {"labels_unknown": fs.BOTH, "each_tracer_s1": "#CC79A7", "each_tracer_s2": "#882E72", "hosts_known": "#000000",
         "inferred": "#CC79A7", "mass_spectrum_alone": "#4D4D4D", "galaxy_catalog_alone": fs.GAL}
PH = {"labels_unknown": RED[0], "each_tracer_s1": RED[1], "each_tracer_s2": RED[3], "hosts_known": RED[2],
      "inferred": RED[1], "mass_spectrum_alone": RED[2], "galaxy_catalog_alone": RED[0]}
# Fig. 3 caption: circles and solid lines (labels unknown); squares and diamonds with dashed lines (a
# population for each tracer, the stated differences and twice them); triangles and dotted lines (hosts known)
STYLE = {"labels_unknown": ("-", "o"), "each_tracer_s1": ("--", "s"), "each_tracer_s2": ("--", "D"),
         "hosts_known": (":", "^"), "inferred": ("none", "s")}
DODGE = {"labels_unknown": -0.021, "each_tracer_s1": -0.007, "each_tracer_s2": 0.007, "hosts_known": 0.021,
         "inferred": -0.007}
# Fig. 4c: the AGN-hosted mergers as a dark line (the caption's words), a darker shade of the AGN vermilion
AGN_DARK = "#A63F00"
LINES = {"mass_spectrum_alone": (0, (7, 2.5)), "galaxy_catalog_alone": (0, (2.2, 1.6))}

CHECKS: list[str] = []


def col(key, real):
    return FINAL[key] if real else PH[key]


def check(what, shown, ref, tol):
    CHECKS.append(f"{what}: drawn {shown:.4g} vs {ref:.4g}: {'ok' if abs(shown - ref) <= tol else 'MISMATCH'}")


def hpd_extent(x, y, P, lev):
    m = P >= lev
    xin, yin = x[m.any(axis=1)], y[m.any(axis=0)]
    return xin.min(), xin.max(), yin.min(), yin.max()


# ============================================================================================ #
# Fig. 1: two tracers of one density field and the mergers they host (final data)
# ============================================================================================ #
D100 = bv.DATA / "seed100"
HD_EVENTS = D100 / "events" / "events_marked_dmu0p10_dmuG5_corepop.h5"
SLAB_DEG = 3.0          # half-thickness in declination of the slice of panel (a), as Fig. 1 of the long draft
FAN_HALF_DEG = 20.0     # half-opening of the slice in right ascension, centred on the merger shown
FAN_ZMAX = 0.20         # depth of the slice
EVENT = 339             # the merger shown: AGN-hosted, |dec| < 0.01 deg, one of the best-localized AGN-hosted mergers in the slab
SCALE_MPC = 200.0       # length of the scale bar of panel (a)


def _rot(ra, ra0):
    """Right-ascension offset from ra0 in (-pi, pi]."""
    return (ra - ra0 + np.pi) % (2 * np.pi) - np.pi


def _caption_number(name):
    """A plain caption macro of values_independent.tex as a number ('13{,}398' -> 13398)."""
    return float(R.expected()[name][2].replace("{,}", ""))


def fig_concept():
    """The first draft's fig1_concept (archive/paper_short/scripts/make_figures.py), same design, drawn at the text
    width at which main.tex includes it (510 pt); its numbers against the caption macros of values_independent.tex."""
    name = "fig_concept"
    W, Hh = TW, 3.05
    H0 = bv.IN["H0"]
    chi = lambda z: bv.chi_mpc(np.asarray(z, float), H0)
    with h5py.File(HD_EVENTS) as h:
        t = {k: h["truth/" + k][()] for k in ("ra", "dec", "z", "host_type", "host_index", "m1src", "chieff")}
        att = {k: float(h.attrs[k]) for k in ("mu_G_gal", "mu_G_agn", "mu_chi_gal", "mu_chi_agn")}
    IN = bv.IN
    assert abs(att["mu_G_gal"] - IN["muG"]) < 1e-12 and abs(att["mu_G_agn"] - IN["muG"] - IN["dmuG"]) < 1e-12
    assert abs(att["mu_chi_gal"] - IN["muChi"]) < 1e-12 and abs(att["mu_chi_agn"] - IN["muChi"] - IN["dmuChi"]) < 1e-12
    i = EVENT
    assert t["host_type"][i] == 1 and abs(np.degrees(t["dec"][i])) < 1.0
    # the 90% localization region, as for the hosts-per-volume macros (build_values.event_regions)
    c, r, zlo, zhi = bv.event_regions(HD_EVENTS, H0)
    c, r, zlo, zhi = c[i], float(r[i]), float(zlo[i]), float(zhi[i])
    ra0 = math.atan2(c[1], c[0])
    dec0 = math.asin(c[2])
    # catalogs: the slab (|dec| < 3 deg) of the slice, and everything inside the region
    zread = max(FAN_ZMAX, zhi) + 0.01
    G = bv.read_catalog_rows(D100 / "catalogs" / "catalog_gal_complete.h5", zread, cols=("ra", "dec", "z", "z_obs"))
    A = bv.read_catalog_rows(D100 / "catalogs" / "catalog_agn_complete.h5", zread, cols=("ra", "dec", "z", "z_obs"))
    with h5py.File(D100 / "catalogs" / "catalog_agn_complete.h5") as h:
        hra, hdec, hz, hzo = (float(h[k][int(t["host_index"][i])]) for k in ("ra", "dec", "z", "z_obs"))
    assert abs(hra - t["ra"][i]) < 1e-9 and abs(hz - t["z"][i]) < 1e-9

    def in_slice(C):
        return (np.abs(C["dec"]) < np.radians(SLAB_DEG)) & (np.abs(_rot(C["ra"], ra0)) < np.radians(FAN_HALF_DEG)) & (C["z"] < FAN_ZMAX)

    def in_region(C):
        v = np.stack([np.cos(C["dec"]) * np.cos(C["ra"]), np.cos(C["dec"]) * np.sin(C["ra"]), np.sin(C["dec"])], 1)
        return (v @ c >= math.cos(r)) & (C["z_obs"] >= zlo) & (C["z_obs"] <= zhi)

    sg, sa = in_slice(G), in_slice(A)
    rg, ra_ = in_region(G), in_region(A)
    ng, na = int(rg.sum()), int(ra_.sum())
    host_in = bool(in_region({"ra": np.array([hra]), "dec": np.array([hdec]), "z_obs": np.array([hzo])})[0])

    def xy(C, m, use="z"):
        d = _rot(C["ra"][m], ra0)
        rr = chi(C[use][m])
        return rr * np.sin(d), rr * np.cos(d)

    fig = plt.figure(figsize=(W, Hh))
    # panel geometry in figure fractions: (a) the slice and the zoom on the left, (b) and (c) stacked on the right
    axa = fig.add_axes([0.000, 0.000, 0.330, 1.000])
    axi = fig.add_axes([0.345, 0.300, 0.255, 0.640])
    axk = fig.add_axes([0.345, 0.000, 0.255, 0.250])
    axb = fig.add_axes([0.700, 0.635, 0.290, 0.350])
    axc = fig.add_axes([0.700, 0.140, 0.290, 0.350])

    # ---- (a) the slice: galaxies, AGN, the merger's region ----
    Rz = float(chi(FAN_ZMAX))
    x, y = xy(G, sg)
    axa.scatter(x, y, s=0.30, c=fs.GAL, alpha=0.55, lw=0, rasterized=True, zorder=1)
    x, y = xy(A, sa)
    axa.scatter(x, y, s=5.0, marker="D", c=fs.AGN, lw=0, rasterized=True, zorder=2)
    th = np.radians(np.linspace(-FAN_HALF_DEG, FAN_HALF_DEG, 300))
    axa.plot(Rz * np.sin(th), Rz * np.cos(th), color="0.35", lw=0.6, zorder=3)
    for s_ in (-1, 1):
        axa.plot([0, Rz * np.sin(s_ * th[-1])], [0, Rz * np.cos(th[-1])], color="0.35", lw=0.6, zorder=3)
    for zz in (0.05, 0.10, 0.15, FAN_ZMAX):
        rz = float(chi(zz))
        if zz < FAN_ZMAX:
            axa.plot(rz * np.sin(th), rz * np.cos(th), color="0.45", lw=0.45, ls=(0, (2, 2)), zorder=3)
        axa.text(-rz * np.sin(th[-1]) - 0.015 * Rz, rz * np.cos(th[-1]), f"$z={zz:.2f}$", ha="right", va="center", fontsize=fs.FS_TEXT - 0.5, color="0.25")
    # the region's footprint in the slice plane: right ascension within the sky disc, distance within the redshift interval
    half = r / math.cos(dec0)
    dl, dh = float(chi(zlo)), float(chi(zhi))
    aa = np.linspace(-half, half, 60)
    fx = np.concatenate([dl * np.sin(aa), dh * np.sin(aa[::-1])])
    fy = np.concatenate([dl * np.cos(aa), dh * np.cos(aa[::-1])])
    axa.add_patch(Polygon(np.c_[fx, fy], closed=True, fill=False, ec="k", lw=1.1, zorder=6))
    axa.set_aspect("equal")
    axa.set_xlim(-0.66 * Rz, 0.37 * Rz)
    axa.set_ylim(-0.10 * Rz, 1.10 * Rz)
    axa.axis("off")
    axa.plot([-0.62 * Rz, -0.62 * Rz + SCALE_MPC], [0.06 * Rz, 0.06 * Rz], color="k", lw=1.2)
    axa.text(-0.62 * Rz + SCALE_MPC / 2, 0.075 * Rz, f"{SCALE_MPC:.0f} Mpc", ha="center", va="bottom", fontsize=fs.FS_TEXT - 0.5)
    axa.text(0.0, -0.025 * Rz, "observer", ha="center", va="top", fontsize=fs.FS_TEXT - 0.5, color="0.25")
    fs.panel_label(axa, "(a)", x=0.02, y=0.995)

    # zoom: the whole 3-D region seen from the side, with every catalog object inside it
    def side(C, m):
        d = _rot(C["ra"][m], ra0)
        rr = chi(C["z_obs"][m])
        return rr * np.sin(d) * math.cos(dec0), rr * np.cos(d)

    def disc(C, zl, zh):
        v = np.stack([np.cos(C["dec"]) * np.cos(C["ra"]), np.cos(C["dec"]) * np.sin(C["ra"]), np.sin(C["dec"])], 1)
        return (v @ c >= math.cos(r)) & (C["z_obs"] >= zl) & (C["z_obs"] <= zh)
    pad = 0.30 * (zhi - zlo)
    mgw, maw = disc(G, zlo - pad, zhi + pad) & ~rg, disc(A, zlo - pad, zhi + pad) & ~ra_
    x, y = side(G, mgw)
    axi.scatter(x, y, s=1.4, c=fs.GAL, alpha=0.25, lw=0, rasterized=True, zorder=1)
    x, y = side(A, maw)
    axi.scatter(x, y, s=14, marker="D", c=fs.AGN, alpha=0.35, lw=0, zorder=2)
    x, y = side(G, rg)
    axi.scatter(x, y, s=3.0, c=fs.GAL, alpha=0.95, lw=0, rasterized=True, zorder=3)
    x, y = side(A, ra_)
    axi.scatter(x, y, s=28, marker="D", c=fs.AGN, edgecolor="k", lw=0.5, zorder=5)
    hx = float(chi(hz)) * math.sin(_rot(hra, ra0)) * math.cos(dec0)
    hy = float(chi(hz)) * math.cos(_rot(hra, ra0))
    axi.scatter([hx], [hy], s=150, facecolor="none", edgecolor="k", lw=1.2, zorder=6)
    ix = np.concatenate([dl * np.sin(aa) * math.cos(dec0), dh * np.sin(aa[::-1]) * math.cos(dec0)])
    axi.add_patch(Polygon(np.c_[ix, fy], closed=True, fill=False, ec="k", lw=1.1, zorder=4))
    ym0, ym1 = float(chi(zlo - pad)), float(chi(zhi + pad))
    xw = 0.5 * (ym1 - ym0) * (0.255 * W) / (0.640 * Hh)
    axi.set_xlim(-xw, xw)
    axi.set_ylim(ym0 - 0.02 * (ym1 - ym0), ym1 + 0.16 * (ym1 - ym0))
    axi.set_aspect("equal", adjustable="datalim")
    axi.set_xticks([])
    axi.set_yticks([])
    for sp in axi.spines.values():
        sp.set_linewidth(0.8)
    axi.text(0.5, 0.985, "90% localization region\nof one AGN-hosted merger", transform=axi.transAxes, ha="center", va="top", fontsize=fs.FS_TEXT - 0.5,
             bbox=dict(fc="white", ec="none", pad=0.6, alpha=0.9), zorder=10)
    axi.text(0.04, 0.02, f"{ng} galaxies", transform=axi.transAxes, ha="left", va="bottom", fontsize=fs.FS_TEXT, color=fs.GAL,
             bbox=dict(fc="white", ec="none", pad=0.4, alpha=0.9), zorder=10)
    axi.text(0.96, 0.02, f"{na} AGN", transform=axi.transAxes, ha="right", va="bottom", fontsize=fs.FS_TEXT, color=fs.AGN,
             bbox=dict(fc="white", ec="none", pad=0.4, alpha=0.9), zorder=10)
    # lines from the footprint in the slice to the zoom
    j_r = int(np.argmax(fx))
    for (xa, ya), (xb, yb) in (((fx.max(), fy[j_r]), (0.0, 0.0)), ((dh * np.sin(aa[-1]), dh * np.cos(aa[-1])), (0.0, 1.0))):
        fig.add_artist(ConnectionPatch(xyA=(xa, ya), coordsA=axa.transData, xyB=(xb, yb), coordsB=axi.transAxes, color="0.45", lw=0.6, zorder=0))
    # key, under the zoom
    axk.axis("off")
    axk.set_xlim(0, 1)
    axk.set_ylim(0, 1)
    keys = [("galaxy catalog", dict(marker="o", ms=2.6, color=fs.GAL, ls="none")), ("AGN catalog", dict(marker="D", ms=4.2, color=fs.AGN, mec="k", mew=0.4, ls="none")),
            ("true host of the merger", dict(marker="o", ms=9, mfc="none", mec="k", mew=1.1, ls="none"))]
    for k_, (lab, st) in enumerate(keys):
        yk = 0.80 - k_ * 0.25
        axk.plot([0.06], [yk], **st)
        axk.text(0.13, yk, lab, ha="left", va="center", fontsize=fs.FS_TEXT)
    axk.text(0.02, 0.06, rf"slice: $|\delta|<{SLAB_DEG:.0f}^\circ$, comoving distance", ha="left", va="center", fontsize=fs.FS_TEXT - 0.5, color="0.25")
    CHECKS.append(f"fig_concept (a): realization A (seed 100); slice |dec| < {SLAB_DEG} deg, RA within {FAN_HALF_DEG} deg of the region centre "
                  f"({np.degrees(ra0) % 360:.1f} deg), true z < {FAN_ZMAX}: {int(sg.sum())} galaxies, {int(sa.sum())} AGN; merger {EVENT} of {HD_EVENTS.name}: "
                  f"AGN-hosted, true z {t['z'][i]:.3f}; 90% region: sky disc {np.degrees(r):.2f} deg x z_obs [{zlo:.3f}, {zhi:.3f}]: {ng} galaxies, {na} AGN; "
                  f"true host inside: {host_in}")

    # ---- (b) masses and (c) spins of the detected mergers, host-dependent populations ----
    hg, ha = t["host_type"] == 0, t["host_type"] == 1
    for ax, key, bins, xl, ins, xlim in ((axb, "m1src", np.arange(5.0, 75.01, 2.5), r"source-frame primary mass $m_1$ [$M_\odot$]", (att["mu_G_gal"], att["mu_G_agn"]), (5, 75)),
                                         (axc, "chieff", np.arange(-0.45, 0.5501, 0.05), r"effective spin $\chi_{\rm eff}$", (att["mu_chi_gal"], att["mu_chi_agn"]), (-0.45, 0.55))):
        top = 0.0
        for m, colr in ((hg, fs.GAL), (ha, fs.AGN)):
            v = t[key][m]
            hh, _ = np.histogram(v, bins, density=True)
            ax.stairs(hh, bins, color=colr, lw=fs.LW, zorder=3)
            ax.stairs(hh, bins, color=colr, alpha=0.13, fill=True, lw=0, zorder=2)
            top = max(top, hh.max())
            out = int(np.sum((v < bins[0]) | (v > bins[-1])))
            if out:
                CHECKS.append(f"fig_concept: {key}: {out} values outside the histogram range")
        for x0 in ins:
            fs.input_vline(ax, x0, ymax=0.80)
        ax.set_xlim(*xlim)
        ax.set_ylim(0, 1.32 * top)
        ax.set_xlabel(xl)
        ax.set_ylabel(r"$p(m_1)$ [$M_\odot^{-1}$]" if key == "m1src" else r"$p(\chi_{\rm eff})$")
        ax.tick_params(axis="y", labelsize=fs.FS_TICK - 0.5)
    yb, yc = axb.get_ylim()[1], axc.get_ylim()[1]
    fs.label(axb, 29.0, 0.55 * yb, "galaxy-\nhosted", fs.GAL, ha="right", va="center")
    fs.label(axb, 48.5, 0.55 * yb, "AGN-\nhosted", fs.AGN, ha="left", va="center")
    fs.label(axc, -0.17, 0.62 * yc, "galaxy-\nhosted", fs.GAL, ha="right", va="center")
    fs.label(axc, 0.27, 0.62 * yc, "AGN-\nhosted", fs.AGN, ha="left", va="center")
    fs.panel_label(axb, "(b)")
    fs.panel_label(axc, "(c)")
    # what is drawn against the macros of the caption and the inputs
    sh = R.shared()
    for mac, x0 in (("MockMuG", att["mu_G_gal"]), ("MockMuGAGN", att["mu_G_agn"]), ("MockMuChi", att["mu_chi_gal"]),
                    ("MockDmuChi", att["mu_chi_agn"] - att["mu_chi_gal"]), ("MockNev", float(len(t["z"])))):
        check(f"fig_concept: {mac}", x0, float(sh[mac]), 1e-9)
    for mac, x0, tol in (("IndConceptDec", SLAB_DEG, 0), ("IndConceptzMax", FAN_ZMAX, 0), ("IndConceptScale", SCALE_MPC, 0),
                         ("IndConceptNGal", int(sg.sum()), 0), ("IndConceptNAGN", int(sa.sum()), 0), ("IndConceptzTrue", float(t["z"][i]), 5e-4),
                         ("IndConceptVolNGal", ng, 0), ("IndConceptVolNAGN", na, 0), ("IndConceptNgalHosted", int(hg.sum()), 0),
                         ("IndConceptNAGNHosted", int(ha.sum()), 0)):
        check(f"fig_concept: caption \\{mac}", x0, _caption_number(mac), tol)
    fs.save(fig, FIGDIR, name)


# ============================================================================================ #
# Fig. 2: joint posterior of H0 and f_AGN, one population common to both tracers, held at its input
# ============================================================================================ #
def fig_anchor():
    name = "fig_anchor"
    fig = plt.figure(figsize=(CW, 2.75))
    ax = fig.add_axes([0.165, 0.150, 0.805, 0.825])
    sh = R.shared()
    h_in, f_in = float(sh["MockHzero"]), float(sh["MockfAGN"])
    A = R.anchor_joint("one_realization")
    C = R.anchor_joint("combined")
    cA = fs.BOTH if A["real"] else RED[2]
    cC = fs.BOTH if C["real"] else RED[0]
    ax.contourf(C["x"], C["y"], C["P"].T, levels=[C["level90"], np.inf], colors=[cC], alpha=0.30, zorder=2)
    ax.contour(C["x"], C["y"], C["P"].T, levels=[C["level90"]], colors=[cC], linewidths=fs.LW_CONTOUR, zorder=3)
    ax.contour(A["x"], A["y"], A["P"].T, levels=[A["level90"]], colors=[cA], linewidths=fs.LW_CONTOUR,
               linestyles=[(0, (4.5, 2.2))], zorder=4)
    ax.plot([h_in], [f_in], marker="+", ms=10, mew=1.6, color=INPUT, ls="none", zorder=8)
    # window: both 90% regions and the input, with room for the legend above them
    ea, ec = hpd_extent(A["x"], A["y"], A["P"], A["level90"]), hpd_extent(C["x"], C["y"], C["P"], C["level90"])
    x0, x1 = min(ea[0], ec[0], h_in), max(ea[1], ec[1], h_in)
    y0, y1 = min(ea[2], ec[2], f_in), max(ea[3], ec[3], f_in)
    wx, wy = x1 - x0, y1 - y0
    ax.set_xlim(x0 - 0.12 * wx, x1 + 0.12 * wx)
    ax.set_ylim(y0 - 0.10 * wy, y1 + 0.52 * wy)
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.05))
    ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))
    ax.set_xlabel(fs.H0_LABEL)
    ax.set_ylabel(r"$f_{\rm AGN}$")
    nk = sh["NrealKnown"]
    hs = [Line2D([], [], color=cA, lw=fs.LW_CONTOUR, ls=(0, (4.5, 2.2))),
          Patch(facecolor=cC, edgecolor=cC, alpha=0.45, lw=fs.LW_CONTOUR),
          Line2D([], [], color=INPUT, marker="+", ms=9, mew=1.5, ls="none")]
    ax.legend(hs, ["One realization", f"Combination of {nk} realizations", "Input"], loc="upper left",
              fontsize=7.6, handlelength=2.2, borderaxespad=0.45, labelspacing=0.3, ncol=1)
    # the drawn one-realization posterior against its run summary and the printed macros
    for J, tag in ((A, "one realization"), (C, "combination")):
        ph = np.trapz(J["P"], J["y"], axis=1)
        pf = np.trapz(J["P"], J["x"], axis=0)
        J["sH"], J["sF"] = R.phr.summarize_xp(J["x"], ph, h_in), R.phr.summarize_xp(J["y"], pf, f_in)
        CHECKS.append(f"fig_anchor {tag}: {J['how']}; H0 {J['sH']['median']:.3f} [{J['sH']['ci90'][0]:.3f}, "
                      f"{J['sH']['ci90'][1]:.3f}], f_AGN {J['sF']['median']:.4f} [{J['sF']['ci90'][0]:.4f}, "
                      f"{J['sF']['ci90'][1]:.4f}]")
    if A["real"]:
        for mac, s in (("IndAnchorHzero", A["sH"]), ("IndfAGNMix", A["sF"])):
            r = R.macro(mac)
            if r.real:
                check(f"fig_anchor one realization median vs \\{mac}", s["median"], r.value[0], 0.01 * (r.value[1] + r.value[2]))
                check(f"fig_anchor one realization 90% lower vs \\{mac}", s["ci90"][0], r.value[0] - r.value[2], 0.05 * (r.value[1] + r.value[2]))
                check(f"fig_anchor one realization 90% upper vs \\{mac}", s["ci90"][1], r.value[0] + r.value[1], 0.05 * (r.value[1] + r.value[2]))
    fs.save(fig, FIGDIR, name)


# ============================================================================================ #
# Fig. 3: the half-width of H0 against the fraction of mergers in AGN
# ============================================================================================ #
def fig1_panel(ax, panel):
    S = R.fig1_points(panel)
    handles = []
    for key in ("labels_unknown", "each_tracer_s1", "each_tracer_s2", "hosts_known", "inferred"):
        if key not in S:
            continue
        pts = S[key]["points"]
        ls, mk = STYLE[key]
        allreal = all(p["real"] for p in pts)
        c_line = col(key, allreal)
        x = np.array([p["x"] for p in pts]) + DODGE[key]
        y = np.array([p["y"] for p in pts])
        ms = fs.MS if mk != "D" else fs.MS - 0.6
        if key != "inferred" and len(pts) > 1:
            ax.plot(x, y, ls=ls, color=c_line, lw=1.2, zorder=3)
        for xi, p in zip(x, pts):
            c = col(key, p["real"])
            ax.plot([xi, xi], [p["lo"], p["hi"]], color=c, lw=0.9, zorder=4, solid_capstyle="butt")
            if key == "inferred":
                ax.plot(xi, p["y"], marker=mk, ms=fs.MS + 1.6, mfc="white", mec=c, mew=1.4, ls="none", zorder=6)
            else:
                ax.plot(xi, p["y"], marker=mk, ms=ms, mfc=c, mec="white", mew=0.5, ls="none", zorder=5)
        if key == "inferred":
            h = Line2D([], [], ls="none", marker=mk, ms=fs.MS + 1.6, mfc="white", mec=c_line, mew=1.4)
        else:
            h = Line2D([], [], ls=ls, lw=1.2, color=c_line, marker=mk, ms=ms, mfc=c_line, mec="white", mew=0.5)
        handles.append((key, h, S[key]["label"]))
    # horizontal lines, labelled directly at the right: the upper line's label above it, the lower's below
    hl = [(k, S[k]["points"][0]) for k in ("mass_spectrum_alone", "galaxy_catalog_alone") if k in S]
    hl.sort(key=lambda kp: -kp[1]["y"])
    for i, (k, p) in enumerate(hl):
        c = col(k, p["real"])
        ax.axhline(p["y"], color=c, lw=1.0, ls=LINES[k], zorder=2)
        fac = 1.035 if i == 0 else 1 / 1.035
        fs.label(ax, 1.035, p["y"] * fac, S[k]["label"], c, ha="right", va="bottom" if i == 0 else "top", fontsize=7.5)
    return S, handles


def fig1_budget():
    """Linear half-width axes (each panel its own range); one legend above both panels; the two horizontal
    lines labelled on the lines. Points: means over the realizations; bars: their range (minimum to maximum)."""
    name = "fig1_budget"
    fig, axs = plt.subplots(1, 2, figsize=(TW, 3.25))
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.135, top=0.845, wspace=0.17)
    sh = R.shared()
    tags = {"a": f"(a) ${sh['MockPeakFrac']}$ of the mergers in the peak",
            "b": f"(b) like the observed spectrum, ${sh['ObsPeakFrac']}$ in the peak"}
    handles = None
    for ax, panel in zip(axs, "ab"):
        S, h = fig1_panel(ax, panel)
        if panel == "a":
            handles = {k: (hh, t) for k, hh, t in h}
        top = max(max(p["hi"] for p in s["points"]) for s in S.values())
        ax.set_ylim(0, 1.17 * top)
        ax.yaxis.set_major_locator(plt.MaxNLocator(6, steps=[1, 2, 5, 10]))
        ax.set_xlim(-0.05, 1.05)
        ax.set_xticks([0, 0.1, 0.3, 0.5, 1.0])
        ax.set_xticklabels(["0", "0.1", "0.3", "0.5", "1"])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xlabel(r"Fraction of mergers hosted by AGN, $f_{\rm AGN}$")
        fs.panel_label(ax, tags[panel], x=0.03, y=0.97, fontsize=8.5)
        for key, s in S.items():
            for p in s["points"]:
                if p["real"] and p.get("macro"):
                    check(f"fig1({panel}) {key} at {p['x']}: point vs \\{p['macro']}", p["y"], R.macro(p["macro"]).value, 1e-12)
    axs[0].set_ylabel(r"90% half-width of $H_0$ [km s$^{-1}$ Mpc$^{-1}$]")
    order = ["labels_unknown", "hosts_known", "each_tracer_s1", "each_tracer_s2", "inferred"]
    hs = [handles[k] for k in order if k in handles]
    fig.legend([h for h, _ in hs], [t for _, t in hs], loc="upper center", ncol=3, fontsize=7.6,
               bbox_to_anchor=(0.53, 1.0), handlelength=2.6, columnspacing=1.4, labelspacing=0.35,
               handletextpad=0.5, borderaxespad=0.2)
    fs.save(fig, FIGDIR, name)


# ============================================================================================ #
# Fig. 4: populations that depend on the host
# ============================================================================================ #
def panel_h0(axa):
    """(a) H0 combined over realizations: one population fitted (dashed), a population for each tracer
    (solid), 90% intervals shaded; the input dashed. Key in the upper right, on an opaque ground."""
    tails, pmax = [], 0.0
    hs = []
    for key, ckey, ls, mac in (("each_tracer", "each_tracer_s1", "-", "IndTwoPopHzeroComb"),
                               ("one_population", "labels_unknown", "--", "IndOnePopHzeroComb")):
        d = R.fig3_density(key)
        c = col(ckey, d["real"])
        x, p, s = np.asarray(d["x"]), np.asarray(d["p"]), d["summary"]
        axa.plot(x, p, color=c, lw=fs.LW, ls=ls, zorder=4)
        prd.shade90(axa, x, p, s, c, alpha=0.22 if d["real"] else 0.25)
        on = x[p >= 2e-3 * p.max()]
        tails += [on.min(), on.max()]
        pmax = max(pmax, p.max())
        hs.append((Line2D([], [], color=c, lw=fs.LW, ls=ls), R.series("fig3")[key]["label"]))
        CHECKS.append(f"fig_populations(a) {key}: {d['how']}; median {s['median']:.3f}, 90% [{s['ci90'][0]:.3f}, {s['ci90'][1]:.3f}]")
        r = R.macro(mac)
        if d["real"] and r.real:
            check(f"fig_populations(a) {key} median vs \\{mac}", s["median"], r.value[0], 1e-6)
    fs.input_vline(axa, R.H0_IN, ls=(0, (5, 3)))
    hs.append((Line2D([], [], color=INPUT, lw=fs.LW_INPUT, ls=(0, (5, 3))), "Input"))
    lo, hi = min(tails + [R.H0_IN]), max(tails + [R.H0_IN])
    w = hi - lo
    axa.set_xlim(lo - 0.12 * w, hi + 0.12 * w)
    axa.set_ylim(0, 1.75 * pmax)
    wrap = {"One population fitted to all mergers": "One population\nfitted to all mergers",
            "A population for each tracer": "A population\nfor each tracer"}
    leg = axa.legend([h for h, _ in hs], [wrap.get(t, t) for _, t in hs], loc="upper right", fontsize=7.4,
                     handlelength=1.9, labelspacing=0.45, borderaxespad=0.35, handletextpad=0.5,
                     frameon=True, framealpha=1.0, facecolor="white", edgecolor="none", borderpad=0.25)
    leg.set_zorder(10)
    axa.xaxis.set_major_locator(plt.MaxNLocator(5, integer=True))
    axa.set_xlabel(fs.H0_LABEL)
    axa.set_ylabel(fs.H0_DENS)
    fs.panel_label(axa, "(a)")


def panel_joint(axb):
    """(b) 90% region of the two population differences, one realization; input (cross), equal populations
    (circle), named in a key at the upper left."""
    J = R.fig3_joint()
    c = col("each_tracer_s1", J["real"])
    x, y, P, lev = J["x"], J["y"], J["P"], J["level90"]
    axb.contourf(x, y, P.T, levels=[lev, np.inf], colors=[c], alpha=0.25 if J["real"] else 0.30, zorder=2)
    axb.contour(x, y, P.T, levels=[lev], colors=[c], linewidths=fs.LW_CONTOUR, zorder=3)
    tin = (float(R.shared()["MockDmuG"]), float(R.shared()["MockDmuChi"]))
    mk_in = dict(marker="x", ms=7.5, mew=1.6, color=INPUT, ls="none")
    mk_eq = dict(marker="o", ms=6.5, mfc="none", mec=INPUT, mew=1.3, ls="none")
    axb.plot(*tin, zorder=6, **mk_in)
    axb.plot(0, 0, zorder=6, **mk_eq)
    # the 90% region's extent sets the window, with zero and the input inside and room for the key above
    m = P >= lev
    xin, yin = x[m.any(axis=1)], y[m.any(axis=0)]
    x0, x1 = min(-1.0, xin.min()), max(tin[0], xin.max())
    y0, y1 = min(0.0, yin.min()), max(tin[1], yin.max())
    dx, dy = x1 - x0, y1 - y0
    axb.set_xlim(x0 - 0.10 * dx, x1 + 0.12 * dx)
    axb.set_ylim(y0 - 0.10 * dy, y1 + 0.42 * dy)
    jt = int(np.nonzero(m.any(axis=0))[0].max())                  # top row of the region: the label sits above it
    fs.label(axb, float(x[m[:, jt]].mean()), yin.max() + 0.03 * dy, "90%", c, ha="center", va="bottom", fontsize=7.5)
    axb.legend([Line2D([], [], **mk_in), Line2D([], [], **mk_eq)], ["Input", "Equal populations"], loc="upper left",
               bbox_to_anchor=(0.0, 0.90), fontsize=7.4, handlelength=1.0, handletextpad=0.4, labelspacing=0.35,
               borderaxespad=0.3)
    axb.xaxis.set_major_locator(plt.MaxNLocator(5, integer=True, steps=[1, 2, 5, 10]))
    axb.yaxis.set_major_locator(plt.MultipleLocator(0.05))
    axb.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}".replace("-", "−")))
    axb.axhline(0, color=fs.LIGHTGREY, lw=0.6, zorder=1)
    axb.axvline(0, color=fs.LIGHTGREY, lw=0.6, zorder=1)
    axb.set_xlabel(r"$\Delta m$ [$M_\odot$]")
    axb.set_ylabel(r"$\Delta\chi$")
    fs.panel_label(axb, "(b)")
    CHECKS.append(f"fig_populations(b): {J['how']}")


def panel_membership(axs):
    """(c) probability of an AGN host by true host: positions alone (top), with masses and spins (bottom).
    Galaxies filled, AGN a dark line; triangles at the means (values beside them); f_AGN dashed, named above
    the panel; the class key in the top sub-panel."""
    bins = np.linspace(0, 1, 41)
    data = {w: R.fig2_membership(w) for w in ("positions", "masses_spins")}
    ymax = 0.0
    for d in data.values():
        for r in (d["r_agn"], d["r_gal"]):
            ymax = max(ymax, np.histogram(r, bins, density=True)[0].max())
    top = 1.62 * ymax
    for ax, w in zip(axs, ("positions", "masses_spins")):
        d = data[w]
        real = d["real"]
        cg, ca = (fs.GAL, AGN_DARK) if real else (RED[1], RED[2])
        hg, _ = np.histogram(d["r_gal"], bins, density=True)
        ha, _ = np.histogram(d["r_agn"], bins, density=True)
        ax.stairs(hg, bins, fill=True, color=cg, alpha=0.35 if real else 0.40, lw=0, zorder=2)
        ax.stairs(ha, bins, color=ca, lw=1.6, zorder=4)
        f0 = d["f_in"]
        ax.axvline(f0, color=INPUT, lw=fs.LW_INPUT, ls=(0, (4, 2.5)), zorder=5)
        ax.set_ylim(0, top)
        for mval, c, ha_, dx in ((d["mean_gal"], cg, "right", -0.022), (d["mean_agn"], ca, "left", 0.022)):
            ax.plot(mval, 0.885 * top, marker="v", ms=5.5, color=c, mec="white", mew=0.4, zorder=6, ls="none")
            ax.text(mval + dx, 0.885 * top, f"{mval:.2f}", color=c, ha=ha_, va="center", fontsize=7.2)
        fs.panel_label(ax, R.series("fig2")[w]["label"].replace(" with ", " with\n"), x=0.97, y=0.95, ha="right",
                       va="top", fontsize=7.4, linespacing=1.0)
        ax.set_xlim(0, 1)
        ax.yaxis.set_major_locator(plt.MaxNLocator(3, integer=True))
        ax.xaxis.set_major_locator(plt.MultipleLocator(0.25))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
        ms = R.series("fig2")[w]["placeholder_macros"]
        for mval, mac in ((d["mean_agn"], ms[0]), (d["mean_gal"], ms[1])):
            r = R.macro(mac)
            if real and r.real:
                check(f"fig_populations(c) {w}: mean vs \\{mac}", round(mval, 2), r.value, 1e-9)
        CHECKS.append(f"fig_populations(c) {w}: {d['how']}")
    f0 = data["positions"]["f_in"]
    axs[0].text(f0, 1.02, r"$f_{\rm AGN}$", transform=axs[0].get_xaxis_transform(), ha="center", va="bottom",
                fontsize=7.5, color=INPUT)
    real = data["positions"]["real"]
    cg, ca = (fs.GAL, AGN_DARK) if real else (RED[1], RED[2])
    axs[0].legend([Patch(facecolor=cg, alpha=0.35 if real else 0.40, lw=0), Line2D([], [], color=ca, lw=1.6)],
                  ["Truly in galaxies", "Truly in AGN"], loc="upper right", bbox_to_anchor=(1.0, 0.70),
                  fontsize=7.2, handlelength=1.5, labelspacing=0.25, borderaxespad=0.3, handletextpad=0.5)
    axs[0].tick_params(labelbottom=False)
    axs[1].set_xlabel("Probability of an AGN host")
    fs.panel_label(axs[0], "(c)", x=0.03, y=0.95)
    return data


def fig_populations():
    name = "fig_populations"
    H, B, T = 2.85, 0.42, 0.17               # height, bottom and top margins [in]
    fig = plt.figure(figsize=(TW, H))
    bot, hgt = B / H, (H - B - T) / H
    xa, wa = 0.47, 2.02
    xb, wb = xa + wa + 0.50, 1.62
    xc = xb + wb + 0.50
    wc = TW - xc - 0.07
    axa = fig.add_axes([xa / TW, bot, wa / TW, hgt])
    axb = fig.add_axes([xb / TW, bot, wb / TW, hgt])
    gap = 0.07 / H
    hc = (hgt - gap) / 2
    axc1 = fig.add_axes([xc / TW, bot + hc + gap, wc / TW, hc])
    axc2 = fig.add_axes([xc / TW, bot, wc / TW, hc], sharex=axc1)
    panel_h0(axa)
    panel_joint(axb)
    panel_membership((axc1, axc2))
    fig.text((xc - 0.30) / TW, bot + hgt / 2, "Probability density", rotation=90, ha="center", va="center",
             fontsize=fs.FS_LABEL)
    fs.save(fig, FIGDIR, name)


FIGS = {"concept": fig_concept, "anchor": fig_anchor, "budget": fig1_budget, "populations": fig_populations}


def main(argv):
    fs.configure()
    todo = argv or list(FIGS)
    for k in todo:
        FIGS[k]()
    if CHECKS:
        print("\n=== what is drawn, and the numbers shown against the run summaries and macros ===")
        for c in CHECKS:
            print("  " + c)


if __name__ == "__main__":
    main(sys.argv[1:])
