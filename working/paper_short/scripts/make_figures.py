#!/usr/bin/env python
"""Figures of the short paper (paper_short/notes/SKELETON.md Sec. 4), drawn with the long draft's machinery.

    PY=/hildafs/home/magana/tmp_ondemand_hildafs_phy230014p_symlink/magana/.conda/envs/jax/bin/python
    $PY working/paper_short/scripts/make_figures.py            # all seven, then the figure-against-values check
    $PY working/paper_short/scripts/make_figures.py fig1 figA2 # some of them

Writes paper_short/figures/<name>.pdf and .png at the printed size (REVTeX two-column, figure* = 7.0 in).

Nothing here is copied from the long draft: the loaders, the style (figstyle.py), the placeholder logic (red tones,
the word "placeholder", final palette once values/production_map.json names a production summary), the panel
functions and the figure-against-values check are imported from paper_prd/scripts/make_figures.py. Every number a
figure draws is compared with paper_prd/values/values.json at the end (the short draft's values/ is a link to it).

    fig1_concept       new, final data: realization A (seed 100), its density field traced by the two catalogs, one
                       AGN-hosted merger's 90% localization region, and the masses and spins of the detected mergers
                       with host-dependent populations (events_marked_dmu0p10_dmuG5_corepop.h5)
    fig2_headline      the long draft's Fig. 6 (host-dependent mergers, combined over realizations)
    fig3_assumptions   ten rows grouped by what the mergers are, combined over realizations
    fig4_rate_per_host (a) the long draft's Fig. 4(a); (b) f_AGN against the assumed AGN number density from
                       Analysis 15 (final, black) with the identity's prediction
    figA1_mocks        the long draft's Fig. 1
    figA2_flux_limits  (a) the long draft's Fig. 3(a); (b) its Fig. 4(b)
    figA3_validation   the long draft's Fig. 9
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import h5py
import numpy as np

SHORT = Path(__file__).resolve().parents[1]
PRD_SCRIPTS = SHORT.parent / "paper_prd" / "scripts"
sys.path.insert(0, str(PRD_SCRIPTS))
import figstyle as fs  # noqa: E402  (Agg backend)
import make_figures as mf  # noqa: E402  (loaders, placeholder logic, panels, checks of the long draft)
import build_values as bv  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import ConnectionPatch, Polygon  # noqa: E402

OUT = SHORT / "figures"
IN = bv.IN
D100 = bv.DATA / "seed100"
NOTES = mf.NOTES
note, chk = mf.note, mf.chk


# ============================================================================================= #
# Fig. 1: two tracers of one density field and the mergers they host (final)
# ============================================================================================= #
HD_EVENTS = D100 / "events" / "events_marked_dmu0p10_dmuG5_corepop.h5"
SLAB_DEG = 3.0          # half-thickness in declination of the slice of panel (a), as Fig. 1 of the long draft
FAN_HALF_DEG = 20.0     # half-opening of the slice in right ascension, centred on the merger shown
FAN_ZMAX = 0.20         # depth of the slice
EVENT = 339             # the merger shown: AGN-hosted, |dec| < 0.01 deg, one of the best-localized AGN-hosted mergers in the slab


def _rot(ra, ra0):
    """Right-ascension offset from ra0 in (-pi, pi]."""
    return (ra - ra0 + np.pi) % (2 * np.pi) - np.pi


def fig1_concept():
    name = "fig1_concept"
    H0 = IN["H0"]
    chi = lambda z: bv.chi_mpc(np.asarray(z, float), H0)
    with h5py.File(HD_EVENTS) as h:
        t = {k: h["truth/" + k][()] for k in ("ra", "dec", "z", "host_type", "host_index", "m1src", "chieff")}
        att = {k: float(h.attrs[k]) for k in ("mu_G_gal", "mu_G_agn", "mu_chi_gal", "mu_chi_agn")}
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

    fig = plt.figure(figsize=(fs.PAGE, 3.05))
    # panel geometry in figure fractions: (a) the slice and the zoom on the left, (b) and (c) stacked on the right
    axa = fig.add_axes([0.000, 0.000, 0.330, 1.000])
    axi = fig.add_axes([0.345, 0.300, 0.255, 0.640])
    axk = fig.add_axes([0.345, 0.000, 0.255, 0.250])
    axb = fig.add_axes([0.700, 0.635, 0.290, 0.350])
    axc = fig.add_axes([0.700, 0.140, 0.290, 0.350])

    # ---- (a) the slice: galaxies, AGN, the merger's region ----
    R = float(chi(FAN_ZMAX))
    x, y = xy(G, sg)
    axa.scatter(x, y, s=0.30, c=fs.GAL, alpha=0.55, lw=0, rasterized=True, zorder=1)
    x, y = xy(A, sa)
    axa.scatter(x, y, s=5.0, marker="D", c=fs.AGN, lw=0, rasterized=True, zorder=2)
    th = np.radians(np.linspace(-FAN_HALF_DEG, FAN_HALF_DEG, 300))
    axa.plot(R * np.sin(th), R * np.cos(th), color="0.35", lw=0.6, zorder=3)
    for s_ in (-1, 1):
        axa.plot([0, R * np.sin(s_ * th[-1])], [0, R * np.cos(th[-1])], color="0.35", lw=0.6, zorder=3)
    for zz in (0.05, 0.10, 0.15, FAN_ZMAX):
        rz = float(chi(zz))
        if zz < FAN_ZMAX:
            axa.plot(rz * np.sin(th), rz * np.cos(th), color="0.45", lw=0.45, ls=(0, (2, 2)), zorder=3)
        axa.text(-rz * np.sin(th[-1]) - 0.015 * R, rz * np.cos(th[-1]), f"$z={zz:.2f}$", ha="right", va="center", fontsize=fs.FS_TEXT - 0.5, color="0.25")
    # the region's footprint in the slice plane: right ascension within the sky disc, distance within the redshift interval
    half = r / math.cos(dec0)
    dl, dh = float(chi(zlo)), float(chi(zhi))
    aa = np.linspace(-half, half, 60)
    fx = np.concatenate([dl * np.sin(aa), dh * np.sin(aa[::-1])])
    fy = np.concatenate([dl * np.cos(aa), dh * np.cos(aa[::-1])])
    axa.add_patch(Polygon(np.c_[fx, fy], closed=True, fill=False, ec="k", lw=1.1, zorder=6))
    axa.set_aspect("equal")
    axa.set_xlim(-0.66 * R, 0.37 * R)
    axa.set_ylim(-0.10 * R, 1.10 * R)
    axa.axis("off")
    axa.plot([-0.62 * R, -0.62 * R + 200.0], [0.06 * R, 0.06 * R], color="k", lw=1.2)
    axa.text(-0.62 * R + 100.0, 0.075 * R, "200 Mpc", ha="center", va="bottom", fontsize=fs.FS_TEXT - 0.5)
    axa.text(0.0, -0.025 * R, "observer", ha="center", va="top", fontsize=fs.FS_TEXT - 0.5, color="0.25")
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
    xw = 0.5 * (ym1 - ym0) * (0.255 * fs.PAGE) / (0.640 * 3.05)
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
    note(name, f"(a) realization A (seed 100); slice |dec| < {SLAB_DEG} deg, RA within {FAN_HALF_DEG} deg of the merger's region centre "
               f"({np.degrees(ra0) % 360:.1f} deg), true z < {FAN_ZMAX}: {int(sg.sum())} galaxies, {int(sa.sum())} AGN (true redshifts, comoving distance at H0 = {H0})")
    note(name, f"(a) merger {EVENT} of {HD_EVENTS.name}: AGN-hosted, true z {t['z'][i]:.3f}, m1 {t['m1src'][i]:.1f} Msun, chi_eff {t['chieff'][i]:.2f}; "
               f"90% region: sky disc radius {np.degrees(r):.2f} deg around the mean sample direction x z_obs in [{zlo:.3f}, {zhi:.3f}] "
               f"(each holding sqrt(0.9) of the samples, as PureGalHostsPerVol); inside it {ng} galaxies and {na} AGN of the complete catalogs "
               f"(cataloged positions and z_obs); true host inside: {host_in}; black ring = true host; inset: side view of the region "
               f"(transverse x radial comoving distance), faint points = the same sky disc in a redshift range wider by {pad:.3f} on each side")
    # ---- (b) masses and (c) spins of the detected mergers, host-dependent populations ----
    hg, ha = t["host_type"] == 0, t["host_type"] == 1
    for ax, key, bins, xl, ins, xlim in ((axb, "m1src", np.arange(5.0, 75.01, 2.5), r"source-frame primary mass $m_1$ [$M_\odot$]", (att["mu_G_gal"], att["mu_G_agn"]), (5, 75)),
                                         (axc, "chieff", np.arange(-0.45, 0.5501, 0.05), r"effective spin $\chi_{\rm eff}$", (att["mu_chi_gal"], att["mu_chi_agn"]), (-0.45, 0.55))):
        top = 0.0
        for m, col in ((hg, fs.GAL), (ha, fs.AGN)):
            v = t[key][m]
            hh, _ = np.histogram(v, bins, density=True)
            ax.stairs(hh, bins, color=col, lw=fs.LW, zorder=3)
            ax.stairs(hh, bins, color=col, alpha=0.13, fill=True, lw=0, zorder=2)
            top = max(top, hh.max())
            out = int(np.sum((v < bins[0]) | (v > bins[-1])))
            if out:
                note(name, f"{key}: {out} values outside the histogram range")
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
    for mac, x0, what in (("MockMuG", att["mu_G_gal"], "input mass peak, galaxy hosts"), ("MockMuGAGN", att["mu_G_agn"], "input mass peak, AGN hosts"),
                          ("MockMuChi", att["mu_chi_gal"], "input spin mean, galaxy hosts"), ("MockDmuChi", att["mu_chi_agn"] - att["mu_chi_gal"], "input spin-mean offset")):
        chk(name, ("macros", mac, "value"), x0, what)
    chk(name, ("macros", "MockNev", "value"), float(len(t["z"])), "detected mergers")
    note(name, f"(b, c) {HD_EVENTS.name}: 1000 detected mergers, {int(hg.sum())} galaxy-hosted, {int(ha.sum())} AGN-hosted (true values, each class "
               f"normalized to unit area; bins 2.5 Msun and 0.05); dashed: input peaks {att['mu_G_gal']:g} and {att['mu_G_agn']:g} Msun, spin means "
               f"{att['mu_chi_gal']:g} and {att['mu_chi_agn']:g}; mean m1 {t['m1src'][hg].mean():.1f} / {t['m1src'][ha].mean():.1f} Msun, mean chi_eff "
               f"{t['chieff'][hg].mean():.3f} / {t['chieff'][ha].mean():.3f}")
    fs.save(fig, OUT, name)


# ============================================================================================= #
# Fig. 2: the headline (the long draft's Fig. 6)
# ============================================================================================= #
def fig2_headline():
    mf.fig_host("HD", "fig2_headline", outdir=OUT)


# ============================================================================================= #
# Fig. 3: what each assumption is worth, grouped by what the mergers are
# ============================================================================================= #
GROUPS = [
    ("Mergers with equal populations", [("KnownBoth", "population known, both catalogs"), ("KnownNone", "population known, no catalog"),
                                        ("SingleBoth", "single population, both catalogs"), ("SingleNone", "single population, no catalog")]),
    ("Mergers with host-dependent populations", [("HDSingle", "single population, both catalogs"), ("HDHost", "host-dependent populations, both catalogs"),
                                                 ("HDNone", "single population, no catalog")]),
    ("The same, observed-like mass spectrum", [("ObsSingle", "single population, both catalogs"), ("ObsHost", "host-dependent populations, both catalogs"),
                                               ("ObsNone", "single population, no catalog")]),
]


def fig3_assumptions():
    name = "fig3_assumptions"
    mf.draw_summary(name, GROUPS, outdir=OUT, fs_lab=11.0, row_h=0.30, left=0.455, h0_range=(60, 80), head_x=0.010, row_x=0.030, fan=0.56,
                    f_range=(0.1, 0.6), gap=0.05, ni_x=0.97, ni_ha="right")
    note(name, "rows: medians and 90% intervals of the posteriors combined over realizations (Comb) for every row and both panels; "
               "H0 axis [60, 80], f_AGN axis [0.1, 0.6]; text 11 pt at print size")


# ============================================================================================= #
# Fig. 4: the merger rate per host
# ============================================================================================= #
A15_POINTS = [("complete catalogs", [("b_complete_x1", 1.0), ("b_complete_x2", 2.0)], "s", "-"),
              ("$m<18$", [("d_m18_x05", 0.5), ("d_m18_x1", 1.0), ("d_m18_x2", 2.0)], "D", "--")]


def panel_a15(ax, name, label="(b)"):
    """f_AGN against the assumed AGN number density, Analysis 15 (final): realization A, population known, hosts normalized to
    z = 1.10 (the production convention), complete catalogs and m < 18; black curves: the identity at constant rate per host."""
    S = bv.jload(bv.A15_SUMMARY)["scans"]
    xx = np.geomspace(0.45, 2.2, 300)
    for k, (lab, pts, mk, ls) in enumerate(A15_POINTS):
        ok = [(S[n], x) for n, x in pts if S.get(n, {}).get("status") == "done"]
        assert len(ok) == len(pts), (lab, [n for n, _ in pts])
        off = 0.975 if k == 0 else 1.025
        for r_, x in ok:
            m = r_["measured"]["f"]
            ax.errorbar([x * off], [m["median"]], yerr=[[m["median"] - m["ci90"][0]], [m["ci90"][1] - m["median"]]], fmt=mk, ms=fs.MS + 0.5, color="k",
                        mfc="k" if k == 0 else "white", mew=1.3, elinewidth=1.4, capsize=2.4, zorder=6)
        r1 = [r_ for r_, x in ok if x == 1.0][0]
        FA = r1["normaliser"]["F_at_67.74"]["AGN"]
        f1 = r1["measured"]["f"]["median"]
        ax.plot(xx, bv.expit(bv.logit(f1) + np.log(FA + xx * (1 - FA))), color="k", lw=1.2, ls=ls, zorder=3)
        # the macros the text quotes for these points
        word = "CompleteProd" if k == 0 else "EighteenProd"
        chk(name, ("macros", f"Normf{word}", "value", "median"), f1, "Analysis 15 point at the input density")
        note(name, f"(b) {lab}: Analysis 15 scans {[n for n, _ in pts]}, hosts normalized to z = 1.10; F_AGN = {FA:.4g}; identity curve through the factor-1 median {f1:.3f}")
    fs.input_hline(ax, IN["f"])
    ax.set_xscale("log")
    ax.set_xticks([0.5, 0.7, 1.0, 1.4, 2.0])
    ax.set_xticklabels(["0.5", "0.7", "1", "1.4", "2"])
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_xlim(0.45, 2.25)
    ax.set_ylim(0.03, 0.56)
    ax.set_xlabel("assumed AGN number density / input")
    ax.set_ylabel(r"$f_{\rm AGN}$")
    fs.label(ax, 1.40, S["b_complete_x1"]["measured"]["f"]["median"] - 0.022, "complete catalogs", "k", ha="center", va="top")
    fs.label(ax, 1.42, 0.392, "$m<18$", "k", ha="right", va="bottom")
    fs.label(ax, 0.47, 0.545, "curves: constant merger rate per host", "0.25", ha="left", va="top", fontsize=fs.FS_TEXT - 0.5)
    fs.panel_label(ax, label, x=0.975, y=0.04, ha="right", va="bottom")


def fig4_rate_per_host():
    name = "fig4_rate_per_host"
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(fs.PAGE, 3.0), layout="constrained", gridspec_kw=dict(width_ratios=[1.0, 1.05]))
    mf.panel_free_joint(axa, name)
    panel_a15(axb, name)
    fs.save(fig, OUT, name)


# ============================================================================================= #
# Appendix figures
# ============================================================================================= #
def figA1_mocks():
    mf.fig1_mocks(outdir=OUT, stem="figA1_mocks")


def figA2_flux_limits():
    name = "figA2_flux_limits"
    fig = plt.figure(figsize=(fs.PAGE, 3.6), layout="constrained")
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.1])
    axH = fig.add_subplot(gs[0, 0])
    axF = fig.add_subplot(gs[1, 0], sharex=axH)
    axb = fig.add_subplot(gs[:, 1])
    mf.panel_flux_points(axH, axF, name)
    mf.panel_rate_ratio(axb, name)
    fs.save(fig, OUT, name)


def figA3_validation():
    mf.fig9_validation(outdir=OUT, stem="figA3_validation")


# ============================================================================================= #
FIGS = {"fig1": fig1_concept, "fig2": fig2_headline, "fig3": fig3_assumptions, "fig4": fig4_rate_per_host,
        "figA1": figA1_mocks, "figA2": figA2_flux_limits, "figA3": figA3_validation}
NAMES = {"fig1": "fig1_concept", "fig2": "fig2_headline", "fig3": "fig3_assumptions", "fig4": "fig4_rate_per_host",
         "figA1": "figA1_mocks", "figA2": "figA2_flux_limits", "figA3": "figA3_validation"}
INPUTS_DRAWN = {"fig1": ["muG", "dmuG", "muChi", "dmuChi"], "fig2": ["H0", "f", "dmuG", "dmuChi"], "fig3": ["H0", "f"], "fig4": ["lognAGN", "f"],
                "figA1": ["muG", "dmuG", "muChi", "dmuChi"], "figA2": ["H0", "f"], "figA3": ["H0", "f"]}


def main(argv):
    fs.configure()
    which = argv[1:] or list(FIGS)
    for w in which:
        if w not in FIGS:
            raise SystemExit(f"unknown figure {w!r}; choose from {list(FIGS)}")
    for w in which:
        print(f"--- {w}")
        FIGS[w]()
        for k in INPUTS_DRAWN[w]:
            chk(NAMES[w], ("macros", mf.INPUT_MACROS[k], "value"), IN[k], "input line")
    bad = mf.run_checks([NAMES[w] for w in which])
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
