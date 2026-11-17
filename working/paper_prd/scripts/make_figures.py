#!/usr/bin/env python
"""Figures of the PRD paper, drawn from the analysis result files.

Run (from anywhere):
    PY=/hildafs/home/magana/tmp_ondemand_hildafs_phy230014p_symlink/magana/.conda/envs/jax/bin/python
    $PY working/paper_prd/scripts/make_figures.py          # every figure, then the check
    $PY working/paper_prd/scripts/make_figures.py fig3     # one figure, then the check for it

Writes figures/<name>.pdf and figures/<name>.png at the printed size (PRD two-column: 3.375 in
column, 7.0 in page; every multi-panel figure is page width).

Every posterior is computed with the machinery of scripts/build_values.py (spline-refined grid
posteriors GridPost, equal-weight nested-sampling chains ChainPost), with the same refinement
factors and nodes, so that a curve and the number quoted in the text are one computation. At the
end the script compares every number a figure displays or encodes (medians, 90% interval ends,
combined-posterior summaries, printed labels) with values/values.json and prints any mismatch.

Statistics drawn: 90% credible intervals (equal-tailed, as in the text), 90% highest-density
regions for 2-D contours. Densities are normalised over the prior range of each analysis.
Chain densities: Gaussian kernel density with reflection at the prior edges (1-D), or a 2-D
histogram smoothed by a Gaussian kernel (2-D); the smoothing scales are listed in
figures/CAPTION_FACTS.md.
"""
from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import figstyle as fs  # noqa: E402  (selects the Agg backend)
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from scipy.integrate import cumulative_trapezoid  # noqa: E402
from scipy.ndimage import gaussian_filter  # noqa: E402

import build_values as bv  # noqa: E402

PRD = HERE.parent
FIGDIR = PRD / "figures"
VALUES = json.loads((PRD / "values" / "values.json").read_text())
IN = bv.IN
DATA = bv.DATA
SEED_REF = DATA / "seed100"
LETTERS = bv.CAT_LETTERS

# ============================================================================================= #
# figure-against-values check
# ============================================================================================= #
CHECKS: list[tuple] = []
NOTES: dict[str, list[str]] = {}


def chk(fig, path, shown, what=""):
    """Record a number (or rendered string) the figure displays or encodes, with its values.json path."""
    CHECKS.append((fig, tuple(path), shown, what))


def chk_post(fig, macro, s, what="median and 90% interval"):
    for key, v in (("median", s["median"]), ("ci90.0", s["ci90"][0]), ("ci90.1", s["ci90"][1])):
        chk(fig, ("macros", macro, "value", *key.split(".")), v, what)


def note(fig, text):
    NOTES.setdefault(fig, []).append(text)
    print(f"  [{fig}] {text}")


def lookup(path):
    v = VALUES
    for p in path:
        v = v[int(p)] if isinstance(v, list) else v[p]
    return v


def run_checks(figs):
    rows = [c for c in CHECKS if c[0] in figs]
    bad = []
    for fig, path, shown, what in rows:
        try:
            ref = lookup(path)
        except (KeyError, IndexError, TypeError) as e:
            bad.append((fig, path, shown, f"missing in values.json ({e!r})"))
            continue
        if isinstance(shown, str):
            ok = shown == ref
        else:
            ref = float(ref)
            ok = abs(float(shown) - ref) <= 1e-9 * max(1.0, abs(ref))
        if not ok:
            bad.append((fig, path, shown, ref))
    print("\n=== figure-against-values check ===")
    for f in figs:
        n = sum(1 for c in rows if c[0] == f)
        nb = sum(1 for b in bad if b[0] == f)
        print(f"  {f:22s} {n:4d} numbers compared, {nb} mismatches")
    for fig, path, shown, ref in bad:
        print(f"  MISMATCH {fig}: {'/'.join(map(str, path))}: figure {shown!r}, values.json {ref!r}")
    print(f"  total: {len(rows)} compared, {len(bad)} mismatches")
    return bad


# ============================================================================================= #
# drawing helpers
# ============================================================================================= #


def kde1(x, lo, hi, n=2001, bw=None):
    """Gaussian kernel density of chain samples on [lo, hi], reflected at both prior edges, normalised
    on [lo, hi]. bw defaults to 0.3 of the sample standard deviation (equal-weight nested-sampling
    draws contain repeats; a Silverman width would resolve them)."""
    x = np.asarray(x, float)
    h = bw if bw is not None else 0.3 * float(np.std(x))
    g = np.linspace(lo, hi, n)
    p = np.zeros(n)
    for c in np.array_split(x, max(1, len(x) // 400)):
        for r in (c, 2 * lo - c, 2 * hi - c):
            p += np.exp(-0.5 * ((g[:, None] - r[None, :]) / h) ** 2).sum(axis=1)
    p /= np.trapz(p, g)
    return g, p, h


def hist2d_density(x, y, xr, yr, nb=50, smooth=1.5):
    """2-D histogram on the prior box, smoothed by a Gaussian of `smooth` bins (reflecting at the box
    edges), with the 90% highest-density level."""
    H, xe, ye = np.histogram2d(x, y, bins=nb, range=[xr, yr])
    H = gaussian_filter(H.astype(float), smooth, mode="reflect")
    s = np.sort(H.ravel())[::-1]
    c = np.cumsum(s) / s.sum()
    lev = s[np.searchsorted(c, 0.9)]
    return 0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1]), H, lev


def grid_hpd(gp, n1, n2, frac=0.9):
    x1, x2, m = gp.marg2(n1, n2)
    W = np.outer(gp.w[gp.idx(n1)], gp.w[gp.idx(n2)])
    d = m.ravel()
    o = np.argsort(d)[::-1]
    c = np.cumsum((m * W).ravel()[o])
    c /= c[-1]
    return x1, x2, m, float(d[o][np.searchsorted(c, frac)])


def contour90(ax, x, y, D, lev, color, fill=True, lw=fs.LW_CONTOUR, alpha=0.30, ls="-", zorder=3):
    if fill:
        ax.contourf(x, y, D.T, levels=[lev, np.inf], colors=[color], alpha=alpha, zorder=zorder)
    ax.contour(x, y, D.T, levels=[lev], colors=[color], linewidths=lw, linestyles=[ls], zorder=zorder + 0.1)


def shade90(ax, x, p, s, color, alpha=0.22, zorder=2):
    lo, hi = s["ci90"]
    m = (x > lo) & (x < hi)
    xs = np.concatenate([[lo], x[m], [hi]])
    ps = np.concatenate([[np.interp(lo, x, p)], p[m], [np.interp(hi, x, p)]])
    ax.fill_between(xs, 0, ps, color=color, alpha=alpha, lw=0, zorder=zorder)


def display_curve(x, p, factor=8):
    """Cubic interpolation of a density between the nodes of its refined lattice, for drawing only
    (the summaries and shaded intervals are those of the lattice posterior itself)."""
    from scipy.interpolate import CubicSpline
    keep = np.concatenate([[True], np.diff(x) > 1e-6 * (x[-1] - x[0])])   # inserted nodes can sit on stored ones
    x, p = x[keep], p[keep]
    xf = np.linspace(x[0], x[-1], (len(x) - 1) * factor + 1)
    return xf, np.maximum(CubicSpline(x, p)(xf), 0.0)


def mass_outside(x, p, lo, hi):
    """Posterior mass of a normalised 1-D density outside the plotted range [lo, hi]."""
    c = np.concatenate([[0.0], cumulative_trapezoid(p, x)])
    c /= c[-1]
    return float(np.interp(lo, x, c) + 1 - np.interp(hi, x, c))


def chain_summary_check(fig, macro, cpost, name):
    s = cpost.summary(name)
    chk_post(fig, macro, s)
    return s


# ============================================================================================= #
# posteriors (constructed exactly as in build_values.py)
# ============================================================================================= #
EXT = {"H0": [IN["H0"]], "f": [IN["f"]]}


@lru_cache(None)
def c1():
    """Complete catalogs, two-population (H0, f_AGN) grids of the 9 catalogs, and their sum."""
    H = F = None
    Ls, posts = [], {}
    for s in bv.C1_SEEDS:
        H_, F_, L, rej = bv.load_hf(bv.A2 / f"joint_s{s}.h5")
        assert rej.sum() == 0
        if H is None:
            H, F = H_, F_
        assert np.array_equal(H, H_) and np.array_equal(F, F_)
        Ls.append(L)
        posts[s] = bv.GridPost(["H0", "f"], [H, F], L, [4, 10], extra=dict(EXT), tag=f"joint_s{s}")
    Lsum = np.sum(Ls, axis=0)
    gc = bv.GridPost(["H0", "f"], [H, F], Lsum, [4, 10], extra=dict(EXT), tag="joint_combined")
    return H, F, Ls, Lsum, posts, gc


@lru_cache(None)
def galonly():
    H, F, Ls, Lsum, posts, gc = c1()
    per = {s: bv.GridPost(["H0"], [H], Ls[i][:, 0], [8], extra={"H0": [IN["H0"]]}, tag=f"galonly_s{s}")
           for i, s in enumerate(bv.C1_SEEDS)}
    comb = bv.GridPost(["H0"], [H], Lsum[:, 0], [8], extra={"H0": [IN["H0"]]}, tag="galonly_combined")
    return per, comb


@lru_cache(None)
def pure():
    ens = bv.jload(bv.A0 / "h0_pure_tracer_ens93.json")
    seeds = list(ens["seeds_requested"])
    per, comb = {"gal": [], "agn": []}, {}
    H = None
    for tr in ("gal", "agn"):
        acc = None
        for s in seeds:
            with h5py.File(bv.A0 / f"h0_pure{tr}_targeted_s{s}.h5") as h:
                H_, L, rej = h["H0_grid"][()], h["log_likelihood"][()], h["guard/rejected"][()]
            assert rej.sum() == 0
            if H is None:
                H = H_
            assert np.array_equal(H, H_)
            g = bv.GridPost(["H0"], [H], L, [8], extra={"H0": [IN["H0"]]}, tag=f"pure{tr}_s{s}")
            x, p = g.marg1("H0")
            per[tr].append((s, x, p, g.summary("H0")))
            acc = L if acc is None else acc + L
        comb[tr] = bv.GridPost(["H0"], [H], acc, [16], extra={"H0": [IN["H0"]]}, tag=f"pure{tr}_combined")
    return seeds, per, comb


@lru_cache(None)
def hd_grids():
    """Host-dependent mock: common population (C10-S), offsets free with the reference fixed (C10-J),
    pinned-offset arms (mech) and the galaxy-hosted control mocks (control)."""
    out = {}
    with h5py.File(bv.A10 / "c10_arm_S.h5") as h:
        H, F, L, rej = h["H0_grid"][()], h["f_grid"][()], h["log_likelihood"][()], h["guard/rejected"][()]
    out["S"] = bv.GridPost(["H0", "f"], [H, F], L, [4, 8], extra={"H0": [IN["H0"], 60.0, 76.0], "f": [IN["f"]]}, rejected=rej, tag="C10-S")
    with h5py.File(bv.A10 / "c10_arm_J.h5") as h:
        assert json.loads(h.attrs["axis_order"]) == ["H0", "f_agn", "dmu_G", "dmu_chi"]
        axes = [h["H0_grid"][()], h["f_grid"][()], h["dmu_G_grid"][()], h["dmu_chi_grid"][()]]
        LJ, rejJ = h["log_likelihood"][()], h["guard/rejected"][()]
    out["J"] = bv.GridPost(["H0", "f", "dmuG", "dmuChi"], axes, LJ, [4, 8, 4, 4],
                           extra={"H0": [IN["H0"]], "f": [IN["f"]], "dmuG": [IN["dmuG"]], "dmuChi": [IN["dmuChi"], 0.2]}, rejected=rejJ, tag="C10-J")
    out["J_axes"] = axes
    with h5py.File(bv.A10 / "c10_mech.h5") as h:
        Hm, Fm = h["H0_grid"][()], h["f_grid"][()]
        out["pin"] = (float(h.attrs["pin_dmu_G"]), float(h.attrs["pin_dmu_chi"]))
        for a in ("P0", "P1", "I0", "I1"):
            out[a] = bv.GridPost(["H0", "f"], [Hm, Fm], h[a + "/log_likelihood"][()], [4, 8], extra=dict(EXT), rejected=h[a + "/rejected"][()], tag="mech_" + a)
        out["mech_H0_range"] = (float(Hm[0]), float(Hm[-1]))
    with h5py.File(bv.A10 / "c10_control.h5") as h:
        Fc = h["f_grid"][()]
        for a in ("B5M", "B5U", "B0M"):
            Hc = h[a + "/H0_grid"][()]
            out[a] = bv.GridPost(["H0", "f"], [Hc, Fc], h[a + "/log_likelihood"][()], [4, 8], extra=dict(EXT), rejected=h[a + "/rejected"][()], tag="ctrl_" + a)
            out[a + "_range"] = (float(Hc[0]), float(Hc[-1]))
    return out


@lru_cache(None)
def chains():
    c11, j11 = bv.load_chain_npz(bv.A11 / "a11_11D.npz", bv.A11 / "a11_11D.json")
    p13 = bv.A13 / "a13core_e7c3007_rslice_cap20_historical_n200_s1.npz"
    c13, j13 = bv.load_chain_npz(p13, bv.A13 / "a13core_e7c3007_rslice_cap20_historical_n200_s1.json")
    box = lambda j: {b[0]: (float(b[1]), float(b[2])) for b in j["box"]}
    return {"11D": (c11, box(j11)), "13": (c13, box(j13))}


@lru_cache(None)
def a8():
    out = {}
    for arm in ("J_joint", "I_intrinsic"):
        p = bv.A8 / f"arm_{arm}.h5"
        with h5py.File(p) as h:
            assert h.attrs["events_file"].endswith("events_marked_dmu0p10.h5")
            F, L, M = h["f_grid"][()], h["log_likelihood"][()], h["mu_chi_c2_grid"][()]
            rej = h["guard/rejected"][()] if "guard/rejected" in h else None
        out[arm] = bv.GridPost(["f", "dmuChi"], [F, M], L, [8, 6], extra={"f": [IN["f"]], "dmuChi": [0.0, IN["dmuChi"]]}, rejected=rej, tag="A8-" + arm)
        out[arm + "_range"] = (float(M[0]), float(M[-1]))
    return out


RUNG_KEYS = [k for k, _, _ in bv.RUNGS]                       # m21, m20, m19, m18
RUNG_WORD = {k: w for k, w, _ in bv.RUNGS}
ARMS = {"a05": 0.5, "a07": 0.7, "a09": 0.9, "a11": 1.1, "a13": 1.3, "a20": 2.0}


@lru_cache(None)
def flux():
    """Flux-limited catalogs (catalog A): densities at input (A3) and the assumed AGN density varied (A4)."""
    at_input, arms = {}, {}
    for key in RUNG_KEYS:
        p = bv.A3 / f"joint_{key}_s100.h5"
        with h5py.File(p) as h:
            assert h.attrs["c_mode"] == "selection"
        at_input[key] = bv.hf_post(p, f"a3_{key}")
        for tag, fac in ARMS.items():
            pa = bv.A4 / f"joint_{key}_{tag}_s100.h5"
            with h5py.File(pa) as h:
                assert h.attrs["c_mode"] == "selection"
            arms[(key, fac)] = bv.hf_post(pa, f"a4_{key}_{tag}")
        arms[(key, 1.0)] = at_input[key]
    return at_input, arms


@lru_cache(None)
def free_density():
    runs = {"A": (bv.A5 / "campaign_m18_dynesty_s100.h5", bv.A5 / "campaign_m18_dynesty_s100.json", ""),
            "B": (bv.SR / "fu_seed101" / "results" / "campaign_m18_dynesty_s101.h5", bv.SR / "fu_seed101" / "results" / "campaign_m18_dynesty_s101.json", "CatB"),
            "C": (bv.SR / "fu_seed102" / "results" / "campaign_m18_dynesty_s102.h5", bv.SR / "fu_seed102" / "results" / "campaign_m18_dynesty_s102.json", "CatC")}
    out = {}
    for L, (ph5, pj, suf) in runs.items():
        c, logz, logzerr, j = bv.load_a5(ph5, pj)
        out[L] = (c, j["priors"], suf, ph5)
    return out


# ============================================================================================= #
# catalog reading for Fig. 1 (only the rows below a redshift cut; the catalogs are written shell by
# shell in true redshift, so reading stops at the first block that lies wholly above the cut)
# ============================================================================================= #


def read_rows(path, zmax, decmax=None, cols=("z",)):
    parts = {c: [] for c in cols}
    with h5py.File(path) as h:
        n = h["z"].shape[0]
        B = h["z"].chunks[0] * 64
        i = 0
        while i < n:
            z = h["z"][i:i + B]
            if z.min() > zmax:
                break
            m = z < zmax
            if decmax is not None:
                m &= np.abs(h["dec"][i:i + B]) < decmax
            if m.any():
                for c in cols:
                    parts[c].append(z[m] if c == "z" else h[c][i:i + B][m])
            i += B
    return {c: np.concatenate(v) if v else np.zeros(0) for c, v in parts.items()}


def comoving_distance(z, H0, Om):
    zg = np.linspace(0, max(1.0, float(np.max(z)) + 0.01), 20001)
    E = np.sqrt(Om * (1 + zg) ** 3 + 1 - Om)
    Dc = 299792.458 / H0 * np.concatenate([[0.0], cumulative_trapezoid(1 / E, zg)])
    return np.interp(z, zg, Dc)


# ============================================================================================= #
# Fig. 1: the simulated catalogs
# ============================================================================================= #
SLAB_DEC_DEG = 3.0       # half-thickness of the declination slab of panel (a)
WEDGE_RA = (0.0, np.pi)  # right-ascension range of panel (a)
WEDGE_ZMAX = 0.33


def fig1_mocks():
    name = "fig1_mocks"
    meta = bv.jload(bv.META100)
    H0, Om = meta["stages"]["events"]["cosmology"]["H0"], meta["stages"]["events"]["cosmology"]["Om0"]
    sur = meta["stages"]["surveys"]
    zh = float(sur["horizon_z"])
    cat = SEED_REF / "catalogs"
    dmax = np.radians(SLAB_DEC_DEG)
    g = read_rows(cat / "catalog_gal_complete.h5", WEDGE_ZMAX, dmax, ("ra", "dec", "z"))
    a = read_rows(cat / "catalog_agn_complete.h5", WEDGE_ZMAX, dmax, ("ra", "dec", "z"))
    with h5py.File(SEED_REF / "events" / "events.h5") as h:
        ev = {k: h["truth/" + k][()] for k in ("ra", "dec", "z", "host_type")}
    # completeness C(z) of the galaxy catalog for the four flux limits (AGN carry their host galaxy's
    # apparent magnitude, so the AGN catalog has the same C(z))
    zc = read_rows(cat / "catalog_gal_complete.h5", 0.34)["z"]
    n_h = int(np.sum(zc <= zh))
    assert n_h == sur["completeness"]["complete"]["gal"]["n_within_horizon_total"], (n_h,)
    edges = np.arange(0.0, 0.3401, 0.005)
    Nc = np.histogram(zc, edges)[0].astype(float)
    Cz = {}
    for key, word, mlim in bv.RUNGS:
        zm = read_rows(cat / f"catalog_gal_{key}.h5", 0.34)["z"]
        nk = int(np.sum(zm <= zh))
        assert nk == sur["completeness"][key]["gal"]["n_within_horizon_kept"], (key, nk)
        Cz[key] = np.histogram(zm, edges)[0] / np.maximum(Nc, 1)
    zmid = 0.5 * (edges[1:] + edges[:-1])
    with h5py.File(SEED_REF / "events" / "events_marked_dmu0p10_dmuG5.h5") as h:
        hd = {k: h["truth/" + k][()] for k in ("m1src", "chieff", "host_type")}
        dmuG_attr, dmuchi_attr = float(h.attrs["dmu_G_agn"]), float(h.attrs["dmu_chi_agn"])
    assert abs(dmuG_attr - IN["dmuG"]) < 1e-12 and abs(dmuchi_attr - IN["dmuChi"]) < 1e-12

    fig = plt.figure(figsize=(fs.PAGE, 5.7), layout="constrained")
    gs = fig.add_gridspec(2, 3, height_ratios=[1.5, 1.0])
    axa = fig.add_subplot(gs[0, :])
    gsb = gs[1, 0].subgridspec(2, 1, height_ratios=[0.62, 1.0], hspace=0.0)
    axb_c = fig.add_subplot(gsb[0])
    axb = fig.add_subplot(gsb[1], sharex=axb_c)
    axc, axd = fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 2])

    # ---- (a) light-cone slice ----
    def xy(ra, z):
        r = comoving_distance(z, H0, Om)
        return r * np.cos(ra), r * np.sin(ra)

    selg = (g["ra"] >= WEDGE_RA[0]) & (g["ra"] < WEDGE_RA[1])
    sela = (a["ra"] >= WEDGE_RA[0]) & (a["ra"] < WEDGE_RA[1])
    sele = (np.abs(ev["dec"]) < dmax) & (ev["ra"] >= WEDGE_RA[0]) & (ev["ra"] < WEDGE_RA[1])
    x, y = xy(g["ra"][selg], g["z"][selg])
    axa.scatter(x, y, s=0.06, c=fs.GAL, alpha=0.35, lw=0, rasterized=True, zorder=1)
    x, y = xy(a["ra"][sela], a["z"][sela])
    axa.scatter(x, y, s=1.6, c=fs.AGN, lw=0, rasterized=True, zorder=2)
    eg = sele & (ev["host_type"] == 0)
    ea = sele & (ev["host_type"] == 1)
    x, y = xy(ev["ra"][eg], ev["z"][eg])
    axa.scatter(x, y, s=34, marker="o", facecolor=fs.GAL, edgecolor="k", lw=0.8, zorder=4)
    x, y = xy(ev["ra"][ea], ev["z"][ea])
    axa.scatter(x, y, s=40, marker="D", facecolor=fs.AGN, edgecolor="k", lw=0.8, zorder=4)
    R = comoving_distance(np.array([WEDGE_ZMAX]), H0, Om)[0]
    th = np.linspace(WEDGE_RA[0], WEDGE_RA[1], 400)
    for zz in (0.1, 0.2, 0.3):
        r = comoving_distance(np.array([zz]), H0, Om)[0]
        axa.plot(r * np.cos(th), r * np.sin(th), color="0.35", lw=0.5, ls=(0, (2, 2)), zorder=3)
        axa.text(r, -0.035 * R, f"$z={zz:.1f}$", ha="center", va="top", fontsize=fs.FS_TEXT)
        axa.text(-r, -0.035 * R, f"$z={zz:.1f}$", ha="center", va="top", fontsize=fs.FS_TEXT)
    axa.plot(R * np.cos(th), R * np.sin(th), color="0.35", lw=0.6, zorder=3)
    axa.plot([-R, R], [0, 0], color="0.35", lw=0.6, zorder=3)
    axa.set_xlim(-1.01 * R, 1.01 * R)
    axa.set_ylim(-0.17 * R, 1.01 * R)
    axa.set_aspect("equal")
    axa.axis("off")
    # key split between the two empty upper corners (normal text size)
    dy = 0.095 * R
    left = [("galaxies", dict(marker="o", ms=2.6, color=fs.GAL, alpha=0.6, ls="none")),
            ("AGN", dict(marker="o", ms=3.0, color=fs.AGN, ls="none"))]
    right = [("galaxy-hosted merger", dict(marker="o", ms=6.5, mfc=fs.GAL, mec="k", mew=0.8, ls="none")),
             ("AGN-hosted merger", dict(marker="D", ms=6.0, mfc=fs.AGN, mec="k", mew=0.8, ls="none"))]
    for i, (t, st) in enumerate(left):
        yy = 0.86 * R - i * dy
        axa.plot([-0.975 * R], [yy], **st)
        axa.text(-0.94 * R, yy, t, va="center", ha="left", fontsize=fs.FS_TEXT)
    for i, (t, st) in enumerate(right):
        yy = 0.96 * R - i * dy
        axa.plot([0.985 * R], [yy], **st)
        axa.text(0.95 * R, yy, t, va="center", ha="right", fontsize=fs.FS_TEXT)
    axa.plot([-250.0, 250.0], [-0.075 * R, -0.075 * R], color="k", lw=1.2)
    axa.text(0.0, -0.095 * R, "500 Mpc", ha="center", va="top", fontsize=fs.FS_TEXT)
    axa.text(-0.985 * R, 0.86 * R - 2.2 * dy, rf"slab $|\delta|<{SLAB_DEC_DEG:.0f}^\circ$", va="center", ha="left", fontsize=fs.FS_TEXT)
    fs.panel_label(axa, "(a)", x=0.0, y=1.0)
    note(name, f"(a) slab |dec| < {SLAB_DEC_DEG} deg, RA in [0, 180) deg, z < {WEDGE_ZMAX}: {int(selg.sum())} galaxies, {int(sela.sum())} AGN, "
               f"{int(eg.sum())} galaxy-hosted and {int(ea.sum())} AGN-hosted detected mergers (of 1000) of the reference catalog")

    # ---- (b) completeness (top) and host redshifts (bottom) ----
    for i, (key, word, mlim) in enumerate(bv.RUNGS):
        axb_c.plot(zmid, Cz[key], color=fs.COMP[i], lw=fs.LW, zorder=3 + i)
    axb_c.set_ylim(0, 1.12)
    axb_c.set_yticks([0, 0.5, 1.0])
    axb_c.set_ylabel("completeness")
    axb_c.tick_params(axis="x", which="both", labelbottom=False)
    for i, (key, word, mlim) in enumerate(bv.RUNGS):
        pct = VALUES["macros"][f"MockComp{word}"]["render"]
        c = Cz[key]
        ylev, side = {"m20": (0.62, "left"), "m19": (0.25, "right"), "m18": (0.55, "right")}.get(key, (None, None))
        k = np.where(c < 0.5)[0]
        if len(k):
            j = int(np.where(c < ylev)[0][0])
            zl = float(np.interp(ylev, c[j - 1:j + 1][::-1], zmid[j - 1:j + 1][::-1]))
            right = side == "right"
            axb_c.text(zl + (0.006 if right else -0.006), ylev, f"{pct}%", color=fs.COMP[i], ha="left" if right else "right",
                       va="center", fontsize=fs.FS_TEXT, zorder=8, bbox=dict(fc="white", ec="none", pad=0.3, alpha=0.9))
        else:
            axb_c.text(0.335, 0.88, f"{pct}%", color=fs.COMP[i], ha="right", va="center", fontsize=fs.FS_TEXT)
        chk(name, ("macros", f"MockComp{word}", "render"), pct, "completeness label")
    zb = np.arange(0.0, 0.3401, 0.01)
    hg = np.histogram(ev["z"][ev["host_type"] == 0], zb)[0]
    ha = np.histogram(ev["z"][ev["host_type"] == 1], zb)[0]
    axb.stairs(hg, zb, color=fs.GAL, lw=fs.LW, fill=False, zorder=3)
    axb.stairs(ha, zb, color=fs.AGN, lw=fs.LW, fill=False, zorder=3)
    axb.set_xlim(0, 0.34)
    axb.set_ylim(0, 1.15 * max(hg.max(), ha.max()))
    axb.set_xlabel(r"redshift $z$")
    axb.set_ylabel(r"hosts per $\Delta z=0.01$")
    fs.label(axb, 0.205, 0.50 * hg.max(), "galaxy\nhosts", fs.GAL, ha="left")
    fs.label(axb, 0.215, 0.27 * hg.max(), "AGN hosts", fs.AGN, ha="left")
    fs.panel_label(axb, "(b)")
    note(name, f"(b) 1000 hosts of the reference catalog: {int((ev['host_type'] == 0).sum())} galaxies, {int((ev['host_type'] == 1).sum())} AGN; "
               f"C(z) from the galaxy catalogs in bins of 0.005; horizon z = {zh:.4f}")

    # ---- (c) primary masses, (d) effective spins of the host-dependent mock ----
    m_bins = np.arange(5.0, 80.01, 2.5)
    c_bins = np.arange(-0.5, 0.6001, 0.04)
    for ax, key, bins, xl in ((axc, "m1src", m_bins, r"source-frame primary mass $m_1$ [$M_\odot$]"),
                              (axd, "chieff", c_bins, r"effective spin $\chi_{\rm eff}$")):
        for t, col in ((0, fs.GAL), (1, fs.AGN)):
            v = hd[key][hd["host_type"] == t]
            hh, _ = np.histogram(v, bins, density=True)
            ax.stairs(hh, bins, color=col, lw=fs.LW, zorder=3)
            if np.any(v < bins[0]) or np.any(v > bins[-1]):
                note(name, f"{key}: {int(np.sum((v < bins[0]) | (v > bins[-1])))} values outside the histogram range")
        ax.set_xlabel(xl)
    pk_g, pk_a = IN["muG"], IN["muG"] + IN["dmuG"]
    for v in (pk_g, pk_a):
        fs.input_vline(axc, v)
    for v in (IN["muChi"], IN["muChi"] + IN["dmuChi"]):
        fs.input_vline(axd, v)
    chk(name, ("macros", "MockMuG", "value"), pk_g, "input mass peak, galaxy hosts")
    chk(name, ("macros", "MockMuGAGN", "value"), pk_a, "input mass peak, AGN hosts")
    chk(name, ("macros", "MockMuChi", "value"), IN["muChi"], "input spin mean, galaxy hosts")
    chk(name, ("macros", "MockDmuChi", "value"), IN["dmuChi"], "input spin-mean offset")
    axc.set_xlim(5, 80)
    axc.set_ylabel(r"probability density [$M_\odot^{-1}$]")
    axd.set_xlim(-0.5, 0.6)
    axd.set_ylabel("probability density")
    yc = axc.get_ylim()[1]
    axc.set_ylim(0, yc * 1.05)
    yd = axd.get_ylim()[1]
    axd.set_ylim(0, yd * 1.05)
    fs.label(axc, 28.0, 0.62 * yc, "galaxy\nhosts", fs.GAL, ha="right")
    fs.label(axc, 49.0, 0.62 * yc, "AGN\nhosts", fs.AGN, ha="left")
    fs.label(axd, -0.17, 0.72 * yd, "galaxy\nhosts", fs.GAL, ha="right")
    fs.label(axd, 0.30, 0.72 * yd, "AGN\nhosts", fs.AGN, ha="left")
    fs.panel_label(axc, "(c)")
    fs.panel_label(axd, "(d)")
    n_agn = int((hd["host_type"] == 1).sum())
    chk(name, ("macros", "MockHDNagn", "value"), float(n_agn), "AGN-hosted events of the host-dependent mock")
    note(name, f"(c,d) host-dependent mock, 1000 detected events: {1000 - n_agn} galaxy-hosted, {n_agn} AGN-hosted; true values; histograms normalised per class")
    fs.save(fig, FIGDIR, name)


# ============================================================================================= #
# Fig. 2: complete catalogs, joint (H0, f_AGN), one panel per catalog plus the combined posterior
# ============================================================================================= #
F2_XR, F2_YR = (62.5, 74.5), (0.0, 0.56)


def fig2_two_catalogs():
    name = "fig2_two_catalogs"
    H, F, Ls, Lsum, posts, gc = c1()
    fig, axs = plt.subplots(2, 5, figsize=(fs.PAGE, 3.25), sharex=True, sharey=True, layout="constrained")
    outlines = []
    for i, s in enumerate(bv.C1_SEEDS):
        ax = axs.flat[i]
        g = posts[s]
        x1, x2, m, lev = grid_hpd(g, "H0", "f")
        contour90(ax, x1, x2, m, lev, fs.BOTH)
        outlines.append((x1, x2, m, lev))
        fs.panel_label(ax, LETTERS[i], x=0.06, y=0.94)
        L = LETTERS[i]
        chk_post(name, f"JointHzeroCat{L}", g.summary("H0"), "marginal of the drawn 2-D posterior")
        chk_post(name, f"JointfCat{L}", g.summary("f"), "marginal of the drawn 2-D posterior")
        # crop check: posterior mass outside the axes
        xh, ph = g.marg1("H0")
        xf, pf = g.marg1("f")
        mo = max(mass_outside(xh, ph, *F2_XR), mass_outside(xf, pf, *F2_YR))
        if mo > 1e-3:
            note(name, f"catalog {L}: marginal mass outside the axes {mo:.2e}")
    ax = axs.flat[9]
    for x1, x2, m, lev in outlines:
        ax.contour(x1, x2, m.T, levels=[lev], colors=[fs.BOTH], linewidths=0.6, alpha=0.55, zorder=2)
    x1, x2, m, lev = grid_hpd(gc, "H0", "f")
    contour90(ax, x1, x2, m, lev, fs.BOTH, alpha=0.55)
    fs.panel_label(ax, "combined", x=0.06, y=0.05, va="bottom")
    chk_post(name, "JointHzeroComb", gc.summary("H0"), "combined posterior")
    chk_post(name, "JointfComb", gc.summary("f"), "combined posterior")
    chk(name, ("macros", "JointNcat", "value"), float(len(bv.C1_SEEDS)), "number of catalog panels")
    for ax in axs.flat:
        fs.input_vline(ax, IN["H0"], lw=0.7, zorder=5)
        fs.input_hline(ax, IN["f"], lw=0.7, zorder=5)
        ax.set_xlim(*F2_XR)
        ax.set_ylim(*F2_YR)
        ax.set_xticks([64, 68, 72])
        ax.set_yticks([0.0, 0.2, 0.4])
    fig.supxlabel(fs.H0_LABEL, fontsize=fs.FS_LABEL)
    fig.supylabel(r"$f_{\rm AGN}$", fontsize=fs.FS_LABEL)
    fs.save(fig, FIGDIR, name)


# ============================================================================================= #
# Fig. 3: which hosts carry the H0 information
# ============================================================================================= #
F3A_XR, F3B_XR = (52.0, 84.0), (56.0, 84.0)


def fig3_host_information():
    name = "fig3_host_information"
    seeds, per, comb = pure()
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(fs.PAGE, 2.75), layout="constrained")
    tabs = VALUES["tables"]["C3_pure_tracer_per_catalog"]
    ymax = 0.0
    for tr, col in (("gal", fs.GAL), ("agn", fs.AGN)):
        for i, (s, x, p, sm) in enumerate(per[tr]):
            axa.plot(x, p, color=col, lw=fs.LW_THIN, alpha=0.45, zorder=2 if tr == "gal" else 3)
            assert tabs[tr][i]["seed"] == s
            chk(name, ("tables", "C3_pure_tracer_per_catalog", tr, i, "median"), sm["median"], f"per-catalog curve, seed {s}")
            chk(name, ("tables", "C3_pure_tracer_per_catalog", tr, i, "ci90", 0), sm["ci90"][0], f"per-catalog curve, seed {s}")
            chk(name, ("tables", "C3_pure_tracer_per_catalog", tr, i, "ci90", 1), sm["ci90"][1], f"per-catalog curve, seed {s}")
            mo = mass_outside(x, p, *F3A_XR)
            if mo > 1e-3:
                note(name, f"(a) {tr} seed {s}: mass outside the axes {mo:.2e}")
            ymax = max(ymax, float(p.max()))
    chk(name, ("macros", "PureNcat", "value"), float(len(seeds)), "curves per host class")
    fs.input_vline(axa, IN["H0"])
    axa.set_xlim(*F3A_XR)
    axa.set_ylim(0, 1.08 * ymax)
    axa.set_xlabel(fs.H0_LABEL)
    axa.set_ylabel(fs.H0_DENS)
    fs.label(axa, 69.6, 0.80 * ymax, "1000 AGN-hosted\nmergers", fs.AGN, ha="left")
    fs.label(axa, 53.0, 0.36 * ymax, "1000 galaxy-hosted\nmergers", fs.GAL, ha="left")
    axa.xaxis.set_major_locator(plt.MultipleLocator(5))
    fs.panel_label(axa, "(a)")
    note(name, f"(a) {len(seeds)} catalogs per host class; tallest per-catalog density {ymax:.2f}")

    # (b) all events: galaxy catalog alone against both catalogs
    H, F, Ls, Lsum, posts, gc = c1()
    gper, gcomb = galonly()
    for i, s in enumerate(bv.C1_SEEDS):
        x, p = gper[s].marg1("H0")
        axb.plot(x, p, color=fs.GAL, lw=fs.LW_THIN, alpha=0.5, zorder=2)
        L = LETTERS[i]
        if i == 0:
            chk_post(name, "GalOnlyHzeroCatA", gper[s].summary("H0"), "per-catalog curve")
        x, p = posts[s].marg1("H0")
        axb.plot(x, p, color=fs.BOTH, lw=fs.LW_THIN, alpha=0.5, zorder=2)
        mo = mass_outside(*gper[s].marg1("H0"), *F3B_XR)
        if mo > 1e-3:
            note(name, f"(b) galaxy-catalog-alone catalog {L}: mass outside the axes {mo:.2e}")
    for g, col, macro in ((gcomb, fs.GAL, "GalOnlyHzeroComb"), (gc, fs.BOTH, "JointHzeroComb")):
        x, p = g.marg1("H0")
        s = g.summary("H0")
        axb.plot(x, p, color=col, lw=fs.LW, zorder=4)
        shade90(axb, x, p, s, col)
        chk_post(name, macro, s, "combined posterior (shaded 90% interval)")
    fs.input_vline(axb, IN["H0"])
    axb.set_xlim(*F3B_XR)
    yb = float(gc.marg1("H0")[1].max())
    axb.set_ylim(0, 1.12 * yb)
    axb.set_xlabel(fs.H0_LABEL)
    axb.set_ylabel(fs.H0_DENS)
    fs.label(axb, 69.6, 0.93 * yb, "both catalogs", fs.BOTH, ha="left")
    fs.label(axb, 71.0, 0.42 * yb, "galaxy catalog alone", fs.GAL, ha="left")
    fs.panel_label(axb, "(b)")
    fs.save(fig, FIGDIR, name)


# ============================================================================================= #
# Fig. 4: host-dependent populations and H0
# ============================================================================================= #
FIG4_CONTROL_PANEL = True   # panel (c): the galaxy-hosted control mock behind the "+20" statement of K1


MODEL_LABEL = {fs.LS_FIXED: "reference fixed", fs.LS_FREE: "reference free", fs.LS_EIGHT: "eight parameters",
               fs.LS_COMMON: "common population"}


def line_key(ax, x0, y0, dy, entries, color, seg=0.07, transform=None):
    """Short key of line styles inside an empty region of the axes (axes coordinates)."""
    tr = transform or ax.transAxes
    for i, ls in enumerate(entries):
        yy = y0 - i * dy
        ax.add_line(Line2D([x0, x0 + seg], [yy, yy], color=color, ls=ls, lw=fs.LW, transform=tr))
        fs.label(ax, x0 + seg + 0.025, yy, MODEL_LABEL[ls], color, transform=tr)


def fig4_host_dependent_h0():
    name = "fig4_host_dependent_h0"
    G = hd_grids()
    ch = chains()
    ncol = 3 if FIG4_CONTROL_PANEL else 2
    fig, axs = plt.subplots(1, ncol, figsize=(fs.PAGE, 2.6), layout="constrained",
                            gridspec_kw=dict(width_ratios=[1.3, 1, 1.15][:ncol]))
    TOP = 0.68   # input lines stop here; the band above holds the key and the annotations
    # (a) both catalogs
    ax = axs[0]
    c13, box13 = ch["13"]
    ya = 0.0
    for key, ls, macro in (("S", fs.LS_COMMON, "CommonHzero"), ("J", fs.LS_FIXED, "HDFixHzero")):
        g = G[key]
        x, p = g.marg1("H0")
        ax.plot(x, p, color=fs.BOTH, ls=ls, lw=fs.LW, zorder=4)
        chk_post(name, macro, g.summary("H0"), "drawn posterior")
        ya = max(ya, float(p.max()))
    x, p, bw = kde1(c13.col("H0"), *box13["H0"])
    chain_summary_check(name, "HDEightHzero", c13, "H0")
    ax.plot(x, p, color=fs.BOTH, ls=fs.LS_EIGHT, lw=fs.LW, zorder=4)
    note(name, f"(a) eight-parameter H0 density: reflected Gaussian KDE, bandwidth {bw:.3f} km/s/Mpc, {len(c13.col('H0'))} samples")
    xS, pS = G["S"].marg1("H0")
    note(name, f"(a) common population: grid H0 range [{xS[0]:.0f}, {xS[-1]:.0f}]; mass outside [60, 78]: {mass_outside(xS, pS, 60, 78):.2e}")
    fs.input_vline(ax, IN["H0"], ymax=TOP)
    ax.set_xlim(60, 78)
    ax.set_ylim(0, ya / (TOP - 0.03))
    line_key(ax, 0.04, 0.93, 0.095, (fs.LS_FIXED, fs.LS_EIGHT, fs.LS_COMMON), fs.BOTH)
    fs.panel_label(ax, "(a)", x=0.97, ha="right")
    fs.label(ax, 0.97, 0.83, "both catalogs", "k", ha="right", va="top", transform=ax.transAxes)
    ax.set_ylabel(fs.H0_DENS)

    # (b) galaxy catalog for both host classes, H0 range [60, 76]
    ax = axs[1]
    yb = 0.0
    for key, ls, macro in (("I0", fs.LS_COMMON, "PinIZeroHzero"), ("I1", fs.LS_FIXED, "PinIOneHzero")):
        g = G[key]
        x, p = g.marg1("H0")
        ax.plot(x, p, color=fs.GAL, ls=ls, lw=fs.LW, zorder=4)
        chk_post(name, macro, g.summary("H0"), "drawn posterior")
        yb = max(yb, float(p.max()))
    lo, hi = G["mech_H0_range"]
    ax.axvline(hi, ymax=TOP + 0.02, color=fs.PRIOR, lw=0.9, ls=":", zorder=1)
    ax.text(hi + 0.3, 0.42 * TOP, "end of the prior range", rotation=90, color=fs.PRIOR, ha="left", va="center",
            fontsize=fs.FS_TEXT, transform=ax.get_xaxis_transform())
    fs.input_vline(ax, IN["H0"], ymax=TOP)
    ax.set_xlim(60, 78)
    ax.set_ylim(0, yb / (TOP - 0.03))
    xI1, pI1 = G["I1"].marg1("H0")
    j = int(np.argmax(pI1))
    fs.label(ax, 66.9, 0.75 * pI1[j], "host-\ndependent", fs.GAL, ha="right")
    fs.label(ax, 74.6, 0.55 * yb, "common\npopulation", fs.GAL, ha="right")
    fs.panel_label(ax, "(b)")
    fs.label(ax, 0.97, 0.95, "galaxy catalog\nalone", "k", ha="right", va="top", transform=ax.transAxes)
    note(name, f"(b) mech arms H0 grid [{lo}, {hi}]; host-dependent curve has offsets fixed at {G['pin']}; common population density at the upper end / peak = {VALUES['diagnostics']['c7_mech_I0_H0_edge_density_hi']:.3f}")

    if FIG4_CONTROL_PANEL:
        ax = axs[2]
        yc = 0.0
        for key, ls, macro in (("B5U", fs.LS_COMMON, "CtrlUnmodHzero"), ("B5M", fs.LS_FIXED, "CtrlModHzero")):
            g = G[key]
            x, p = g.marg1("H0")
            ax.plot(x, p, color=fs.GAL, ls=ls, lw=fs.LW, zorder=4)
            chk_post(name, macro, g.summary("H0"), "drawn posterior")
            yc = max(yc, float(p.max()))
            note(name, f"(c) {key}: stored H0 range {G[key + '_range']}")
        fs.input_vline(ax, IN["H0"], ymax=TOP)
        ax.set_xlim(55, 100)
        ax.set_ylim(0, yc / (TOP - 0.03))
        fs.label(ax, 70.5, 0.62 * yc, "two\npopulations", fs.GAL, ha="left", va="bottom")
        fs.label(ax, 88.9, 1.04 * yc, "common\npopulation", fs.GAL, ha="center", va="bottom")
        fs.panel_label(ax, "(c)")
        fs.label(ax, 0.03, 0.86, "galaxy hosts only,\ngalaxy catalog", "k", ha="left", va="top", transform=ax.transAxes)
    for ax in axs:
        ax.set_xlabel(fs.H0_LABEL)
    fs.save(fig, FIGDIR, name)


# ============================================================================================= #
# Fig. 5: the offsets
# ============================================================================================= #


def fig5_offsets():
    name = "fig5_offsets"
    G = hd_grids()
    ch = chains()
    gJ = G["J"]
    c11, box11 = ch["11D"]
    c13, box13 = ch["13"]
    fig, axs = plt.subplots(2, 2, figsize=(fs.PAGE, 4.3), layout="constrained")
    specs = [("f", "f_agn", r"$f_{\rm AGN}$", r"$p(f_{\rm AGN})$", (0.05, 0.55), IN["f"], "fAGN"),
             ("dmuG", "dmu_G", r"mass-peak offset $\Delta\mu_{\rm G}$ [$M_\odot$]", r"$p(\Delta\mu_{\rm G})$ [$M_\odot^{-1}$]", (0.0, 10.0), IN["dmuG"], "DmuG"),
             ("dmuChi", "dmu_chi", r"spin-mean offset $\Delta\mu_\chi$", r"$p(\Delta\mu_\chi)$", (-0.05, 0.3), IN["dmuChi"], "DmuChi")]
    panels = [axs[0, 0], axs[0, 1], axs[1, 0]]
    for ax, (gn, cn, xl, yl, xr, x0, mac) in zip(panels, specs):
        x, p = gJ.marg1(gn)
        ax.plot(x, p, color=fs.BOTH, ls=fs.LS_FIXED, lw=fs.LW, zorder=4)
        chk_post(name, f"HDFix{mac}", gJ.summary(gn), "drawn posterior")
        top = float(p.max())
        for c, box, ls, macro in ((c11, box11, fs.LS_FREE, f"HDFree{mac}"), (c13, box13, fs.LS_EIGHT, f"HDEight{mac}")):
            xx, pp, bw = kde1(c.col(cn), *box[cn])
            ax.plot(xx, pp, color=fs.BOTH, ls=ls, lw=fs.LW, zorder=4)
            chain_summary_check(name, macro, c, cn)
            top = max(top, float(pp.max()))
            note(name, f"{mac} {macro}: KDE bandwidth {bw:.4g}, prior {box[cn]}, mass outside axes {mass_outside(xx, pp, *xr):.2e}")
        note(name, f"{mac} reference-fixed grid covers [{x[0]:.3f}, {x[-1]:.3f}]")
        fs.input_vline(ax, x0)
        ax.set_xlim(*xr)
        ax.set_ylim(0, 1.15 * top)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
    line_key(axs[0, 1], 0.61, 0.80, 0.105, (fs.LS_FIXED, fs.LS_FREE, fs.LS_EIGHT), fs.BOTH, seg=0.08)
    for ax, t in zip(panels, ("(a)", "(b)", "(c)")):
        fs.panel_label(ax, t)
    fs.label(axs[0, 1], 0.97, 0.95, r"both catalogs, $H_0$ free", "k", ha="right", va="top", transform=axs[0, 1].transAxes)
    # (d) spin offset with and without positional host information (spin-offset mock, fixed H0)
    ax = axs[1, 1]
    A = a8()
    top = 0.0
    for arm, col, macro in (("J_joint", fs.BOTH, "SpinBothDmuChi"), ("I_intrinsic", fs.GAL, "SpinIntDmuChi")):
        g = A[arm]
        x, p = g.marg1("dmuChi")
        ax.plot(x, p, color=col, lw=fs.LW, zorder=4)
        s = g.summary("dmuChi")
        chk_post(name, macro, s, "drawn posterior")
        top = max(top, float(p.max()))
        note(name, f"(d) {arm}: grid [{x[0]}, {x[-1]}], mass outside [-0.05, 0.3]: {mass_outside(x, p, -0.05, 0.3):.2e}")
    fs.input_vline(ax, IN["dmuChi"], ymax=0.80)
    ax.set_xlim(-0.05, 0.3)
    ax.set_ylim(0, 1.15 * top)
    ax.set_xlabel(r"spin-mean offset $\Delta\mu_\chi$")
    ax.set_ylabel(r"$p(\Delta\mu_\chi)$")
    xJ, pJ = A["J_joint"].marg1("dmuChi")
    j = int(np.argmax(pJ))
    fs.label(ax, xJ[j] + 0.02, 0.80 * pJ[j], "with the catalogs", fs.BOTH, ha="left")
    fs.label(ax, 0.19, 0.30 * top, "spins alone", fs.GAL, ha="left")
    fs.panel_label(ax, "(d)")
    fs.label(ax, 0.97, 0.95, r"spin offset only, $H_0$ fixed", "k", ha="right", va="top", transform=ax.transAxes)
    fs.save(fig, FIGDIR, name)


# ============================================================================================= #
# Fig. 6: flux limits and the assumed AGN number density (catalog A)
# ============================================================================================= #


def fig6_flux_limits():
    name = "fig6_flux_limits"
    H, F, Ls, Lsum, posts, gc = c1()
    ref = posts[100]
    at_input, arms = flux()
    fig, axs = plt.subplots(2, 2, figsize=(fs.PAGE, 4.2), layout="constrained", sharey="row",
                            gridspec_kw=dict(width_ratios=[1.0, 1.35]))
    pct = {k: VALUES["macros"][f"MockComp{RUNG_WORD[k]}"]["render"] for k in RUNG_KEYS}
    for k in RUNG_KEYS:
        chk(name, ("macros", f"MockComp{RUNG_WORD[k]}", "render"), pct[k], "completeness label")
    cats = ["none\ncomplete"] + [f"$m<{k[1:]}$\n{pct[k]}%" for k in RUNG_KEYS]
    xs = np.arange(len(cats))
    for row, (pn, macro_ref, macro_fmt, yl) in enumerate((("f", "JointfCatA", "Fluxf{}", r"$f_{\rm AGN}$"),
                                                          ("H0", "JointHzeroCatA", "FluxHzero{}", fs.H0_LABEL))):
        ax = axs[row, 0]
        s = ref.summary(pn)
        chk_post(name, macro_ref, s, "complete-catalog reference point")
        ax.errorbar([xs[0]], [s["median"]], yerr=[[s["median"] - s["ci90"][0]], [s["ci90"][1] - s["median"]]],
                    fmt="o", ms=fs.MS, color=fs.COMP[0], mfc="white", mew=1.3, elinewidth=1.5, capsize=2.5, zorder=4)
        for i, k in enumerate(RUNG_KEYS):
            s = at_input[k].summary(pn)
            chk_post(name, macro_fmt.format(RUNG_WORD[k]), s, "median and 90% interval at input densities")
            ax.errorbar([xs[i + 1]], [s["median"]], yerr=[[s["median"] - s["ci90"][0]], [s["ci90"][1] - s["median"]]],
                        fmt=fs.COMP_MARKERS[i], ms=fs.MS, color=fs.COMP[i], elinewidth=1.5, capsize=2.5, zorder=4)
        fs.input_hline(ax, IN["f"] if pn == "f" else IN["H0"])
        ax.set_xticks(xs)
        ax.set_xticklabels(cats)
        ax.tick_params(axis="x", which="minor", bottom=False, top=False)
        ax.set_xlim(-0.5, len(cats) - 0.5)
        ax.set_ylabel(yl)
        if row == 1:
            ax.set_xlabel("flux limit and completeness inside the horizon")
        # (b, d): assumed AGN density
        ax = axs[row, 1]
        facs = sorted({f for (k, f) in arms})
        for i, k in enumerate(RUNG_KEYS):
            med = []
            for f in facs:
                sm = arms[(k, f)].summary(pn)
                med.append(sm["median"])
                key = "medians_f" if pn == "f" else "medians_H0"
                chk(name, ("macros", f"SlopefAGN{RUNG_WORD[k]}", "extra", key, str(f)), sm["median"], "median against assumed AGN density")
            ax.plot(facs, med, color=fs.COMP[i], lw=fs.LW, marker=fs.COMP_MARKERS[i], ms=fs.MS * 0.9,
                    mfc=fs.COMP[i], mec=fs.COMP[i], zorder=4 + i)
            if k == "m18":
                lo = [arms[(k, f)].summary(pn)["ci90"][0] for f in facs]
                hi = [arms[(k, f)].summary(pn)["ci90"][1] for f in facs]
                ax.fill_between(facs, lo, hi, color=fs.COMP[i], alpha=0.13, lw=0, zorder=1)
                chk_post(name, macro_fmt.format("Eighteen"), arms[(k, 1.0)].summary(pn), "90% band at m < 18, factor 1")
        fs.input_hline(ax, IN["f"] if pn == "f" else IN["H0"])
        ax.axvline(1.0, color=fs.PRIOR, lw=0.8, ls=":", zorder=1)
        ax.set_xscale("log")
        ax.set_xticks([0.5, 0.7, 1.0, 1.4, 2.0])
        ax.set_xticklabels(["0.5", "0.7", "1", "1.4", "2"])
        ax.xaxis.set_minor_locator(plt.NullLocator())
        ax.set_xlim(0.47, 2.12)
        if row == 1:
            ax.set_xlabel(r"assumed AGN number density / input")
    axs[0, 0].set_ylim(0.05, 0.55)
    axs[1, 0].set_ylim(65.5, 72.0)
    # direct labels on the H0 curves (panel d) at the left end, where they separate
    ax = axs[1, 1]
    yv = {k: arms[(k, 0.5)].summary("H0")["median"] for k in RUNG_KEYS}
    fs.label(ax, 0.485, yv["m21"] + 0.25, f"{pct['m21']}%, {pct['m20']}%", fs.COMP[1], ha="left", va="bottom")
    fs.label(ax, 0.485, yv["m19"] - 0.22, f"{pct['m19']}%", fs.COMP[2], ha="left", va="top")
    y2 = arms[("m18", 2.0)].summary("H0")["median"]
    fs.label(ax, 1.93, y2 + 0.25, f"{pct['m18']}%", fs.COMP[3], ha="right", va="bottom")
    ax = axs[0, 1]
    fs.label(ax, 0.56, 0.47, "all four completeness levels", "0.25", ha="left")
    for ax, yb in ((axs[0, 1], 0.115), (axs[1, 1], 66.35)):
        fs.label(ax, 1.97, yb, f"90% interval, {pct['m18']}%", fs.COMP[3], ha="right", va="center")
    for ax, t in zip(axs.flat, ("(a)", "(b)", "(c)", "(d)")):
        fs.panel_label(ax, t)
    fs.save(fig, FIGDIR, name)


# ============================================================================================= #
# Fig. 7: both number densities free at 9.5% completeness, three catalogs
# ============================================================================================= #


def fig7_free_densities():
    name = "fig7_free_densities"
    R = free_density()
    fig = plt.figure(figsize=(fs.PAGE, 4.5), layout="constrained")
    gs = fig.add_gridspec(2, 6, height_ratios=[1.0, 1.15])
    axg = fig.add_subplot(gs[0, 0:3])
    axa = fig.add_subplot(gs[0, 3:6])
    axr = [fig.add_subplot(gs[1, 2 * i:2 * i + 2]) for i in range(3)]
    pr0 = R["A"][1]
    for ax, cn, col, x0, word, xl in ((axg, "log10n0", fs.GAL, IN["lognGal"], "Gal", r"$\log_{10}[n_{\rm gal}/{\rm Mpc^{-3}}]$"),
                                      (axa, "log10n0_c2", fs.AGN, IN["lognAGN"], "AGN", r"$\log_{10}[n_{\rm AGN}/{\rm Mpc^{-3}}]$")):
        lo, hi = pr0[cn]
        top = 0.0
        for L in ("A", "B", "C"):
            c, pri, suf, _ = R[L]
            assert pri[cn] == [lo, hi]
            x, p, bw = kde1(c.col(cn), lo, hi)
            ax.plot(x, p, color=col, lw=fs.LW, zorder=4)
            chain_summary_check(name, f"FreeLogn{word}Eighteen{suf}", c, cn)
            top = max(top, float(p.max()))
            note(name, f"log n {word} catalog {L}: KDE bandwidth {bw:.3f} dex, {len(c.col(cn))} samples")
        ax.axhline(1.0 / (hi - lo), color=fs.PRIOR, lw=0.9, ls=":", zorder=1)
        fs.label(ax, hi - 0.03 * (hi - lo), 1.0 / (hi - lo), "prior", fs.PRIOR, ha="right", va="bottom")
        fs.input_vline(ax, x0)
        ax.set_xlim(lo, hi)
        ax.set_ylim(0, 1.18 * top)
        ax.set_xlabel(xl)
        ax.set_ylabel("probability density [dex$^{-1}$]")
    fs.panel_label(axg, "(a)")
    fs.panel_label(axa, "(b)")
    for i, (ax, L) in enumerate(zip(axr, ("A", "B", "C"))):
        c, pri, suf, _ = R[L]
        xr, yr = tuple(pri["fcat_2"]), tuple(pri["log10n0_c2"])
        x, y, D, lev = hist2d_density(c.col("fcat_2"), c.col("log10n0_c2"), xr, yr, nb=40, smooth=1.5)
        contour90(ax, x, y, D, lev, fs.AGN)
        chain_summary_check(name, f"FreefAGNEighteen{suf}", c, "fcat_2")
        chk(name, ("macros", f"FreeRhofAGNEighteen{suf}", "value"), c.corr("fcat_2", "log10n0_c2"), "correlation of the drawn samples")
        ax.plot([IN["f"]], [IN["lognAGN"]], marker="+", ms=10, mew=1.4, color=fs.INPUT, zorder=6)
        ax.set_xlim(*xr)
        ax.set_ylim(*yr)
        ax.set_yticks([-6.0, -5.5, -5.0, -4.5, -4.0])
        ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xlabel(r"$f_{\rm AGN}$")
        if i == 0:
            ax.set_ylabel(r"$\log_{10}[n_{\rm AGN}/{\rm Mpc^{-3}}]$")
        else:
            ax.set_yticklabels([])
        fs.panel_label(ax, L, x=0.06, y=0.94)
        note(name, f"(f, n_AGN) catalog {L}: 40x40 histogram on the prior box, Gaussian smoothing 1.5 bins ({1.5 * (xr[1] - xr[0]) / 40:.4f} in f, {1.5 * (yr[1] - yr[0]) / 40:.4f} dex)")
    fs.save(fig, FIGDIR, name)


# ============================================================================================= #
# Fig. 8: combined posteriors over simulated catalogs
# ============================================================================================= #


def fig8_validation():
    name = "fig8_validation"
    seeds, per, comb = pure()
    H, F, Ls, Lsum, posts, gc = c1()
    fig, axs = plt.subplots(1, 4, figsize=(fs.PAGE, 2.15), layout="constrained")
    items = [(comb["gal"], "H0", fs.GAL, "PureGalHzeroComb", f"{len(seeds)} catalogs,\ngalaxy hosts", (66.6, 68.9), "PureNcat"),
             (comb["agn"], "H0", fs.AGN, "PureAGNHzeroComb", f"{len(seeds)} catalogs,\nAGN hosts", (67.45, 68.2), "PureNcat"),
             (gc, "H0", fs.BOTH, "JointHzeroComb", f"{len(bv.C1_SEEDS)} catalogs,\nboth host classes", (66.6, 70.1), "JointNcat"),
             (gc, "f", fs.BOTH, "JointfComb", f"{len(bv.C1_SEEDS)} catalogs,\nboth host classes", (0.22, 0.36), "JointNcat")]
    for i, (ax, (g, pn, col, macro, txt, xr, ncm)) in enumerate(zip(axs, items)):
        x, p = g.marg1(pn)
        s = g.summary(pn)
        xd, pd = display_curve(x, p)
        ax.plot(xd, pd, color=col, lw=fs.LW, zorder=4)
        shade90(ax, xd, pd, s, col, alpha=0.25)
        chk_post(name, macro, s, "combined posterior (shaded 90% interval)")
        chk(name, ("macros", ncm, "value"), float(len(seeds) if ncm == "PureNcat" else len(bv.C1_SEEDS)), "number of catalogs in the label")
        x0 = IN["H0"] if pn == "H0" else IN["f"]
        fs.input_vline(ax, x0, ymax=0.76)
        ax.set_xlim(*xr)
        ax.set_ylim(0, 1.6 * float(p.max()))
        ax.set_xlabel(fs.H0_LABEL if pn == "H0" else r"$f_{\rm AGN}$")
        ax.set_ylabel(fs.H0_DENS if pn == "H0" else r"$p(f_{\rm AGN})$")
        fs.label(ax, 0.97, 0.96, txt, "k", ha="right", va="top", transform=ax.transAxes, linespacing=1.1)
        fs.panel_label(ax, f"({'abcd'[i]})")
        mo = mass_outside(x, p, *xr)
        if mo > 1e-4:
            note(name, f"panel {'abcd'[i]}: mass outside the axes {mo:.2e}")
    axs[1].xaxis.set_major_locator(plt.MultipleLocator(0.25))
    fs.save(fig, FIGDIR, name)


# ============================================================================================= #
# Fig. C1: corner of the eight-parameter model
# ============================================================================================= #
CORNER_TICKS = {"H0": [64, 67, 70], "f_agn": [0.2, 0.3, 0.4], "mu_G": [35, 36, 37], "dmu_G": [2, 4, 6],
                "mu_chi": [-0.03, 0.0], "dmu_chi": [0.1, 0.15, 0.2], "sigma_G": [4, 5], "sigma_chi": [0.06, 0.08, 0.1]}
CORNER = [("H0", r"$H_0$", "HDEightHzero"), ("f_agn", r"$f_{\rm AGN}$", "HDEightfAGN"), ("mu_G", r"$\mu_{\rm G}$ [$M_\odot$]", "HDEightMuG"),
          ("dmu_G", r"$\Delta\mu_{\rm G}$ [$M_\odot$]", "HDEightDmuG"), ("mu_chi", r"$\mu_\chi$", "HDEightMuChi"),
          ("dmu_chi", r"$\Delta\mu_\chi$", "HDEightDmuChi"), ("sigma_G", r"$\sigma_{\rm G}$ [$M_\odot$]", "HDEightSigmaG"),
          ("sigma_chi", r"$\sigma_\chi$", "HDEightSigmaChi")]


def figC1_corner():
    name = "figC1_corner"
    assert all(bv.CHAIN_IN[c] == IN[k] for c, k in (("H0", "H0"), ("f_agn", "f"), ("mu_G", "muG"), ("dmu_G", "dmuG"), ("mu_chi", "muChi"),
                                                  ("dmu_chi", "dmuChi"), ("sigma_G", "sigmaG"), ("sigma_chi", "sigmaChi")))
    c, box = chains()["13"]
    n = len(CORNER)
    lims = {}
    for cn, _, _ in CORNER:
        v = c.col(cn)
        q = np.percentile(v, [0.05, 99.95])
        x0 = bv.CHAIN_IN[cn]
        lo, hi = min(q[0], x0), max(q[1], x0)
        pad = 0.08 * (hi - lo)
        lims[cn] = (max(box[cn][0], lo - pad), min(box[cn][1], hi + pad))
    fig, axs = plt.subplots(n, n, figsize=(fs.PAGE, fs.PAGE), gridspec_kw=dict(wspace=0.06, hspace=0.06))
    fig.subplots_adjust(left=0.095, right=0.99, bottom=0.085, top=0.995)
    for i, (ci, li, mi) in enumerate(CORNER):
        for j, (cj, lj, mj) in enumerate(CORNER):
            ax = axs[i, j]
            if j > i:
                ax.axis("off")
                continue
            if i == j:
                x, p, bw = kde1(c.col(ci), *box[ci])
                s = chain_summary_check(name, mi, c, ci)
                ax.plot(x, p, color=fs.BOTH, lw=1.4)
                shade90(ax, x, p, s, fs.BOTH, alpha=0.25)
                fs.input_vline(ax, bv.CHAIN_IN[ci], lw=0.9)
                ax.set_ylim(0, 1.12 * float(p.max()))
                ax.set_yticks([])
                ax.tick_params(axis="y", which="both", left=False, right=False)
                mo = mass_outside(x, p, *lims[ci])
                note(name, f"{ci}: axes {tuple(round(v, 4) for v in lims[ci])}, prior {box[ci]}, KDE bandwidth {bw:.4g}, mass outside axes {mo:.1e}")
            else:
                xr, yr = lims[cj], lims[ci]
                X, Y, D, lev = hist2d_density(c.col(cj), c.col(ci), xr, yr, nb=36, smooth=1.7)
                contour90(ax, X, Y, D, lev, fs.BOTH, lw=1.1, alpha=0.30)
                ax.axvline(bv.CHAIN_IN[cj], color=fs.INPUT, lw=0.6, ls="--", zorder=1)
                ax.axhline(bv.CHAIN_IN[ci], color=fs.INPUT, lw=0.6, ls="--", zorder=1)
                ax.set_ylim(*yr)
            ax.set_xlim(*lims[cj])
            ax.tick_params(labelsize=fs.FS_TICK, length=2.5)
            ax.set_xticks(CORNER_TICKS[cj])
            if i != j:
                ax.set_yticks(CORNER_TICKS[ci])
            if i < n - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel(lj, fontsize=fs.FS_LABEL)
                ax.set_xticklabels([f"{v:g}".replace("-", "\u2212") for v in CORNER_TICKS[cj]], rotation=45, ha="right", rotation_mode="anchor")
            if j > 0 or i == 0:
                if i != j:
                    ax.set_yticklabels([])
            else:
                ax.set_ylabel(li, fontsize=fs.FS_LABEL)
                ax.set_yticklabels([f"{v:g}".replace("-", "\u2212") for v in CORNER_TICKS[ci]])
    axs[0, 0].set_ylabel("")
    fig.align_ylabels(axs[:, 0])
    fig.align_xlabels(axs[-1, :])
    fs.save(fig, FIGDIR, name)


# ============================================================================================= #
FIGS = {"fig1": fig1_mocks, "fig2": fig2_two_catalogs, "fig3": fig3_host_information, "fig4": fig4_host_dependent_h0,
        "fig5": fig5_offsets, "fig6": fig6_flux_limits, "fig7": fig7_free_densities, "fig8": fig8_validation, "figC1": figC1_corner}
NAMES = {"fig1": "fig1_mocks", "fig2": "fig2_two_catalogs", "fig3": "fig3_host_information", "fig4": "fig4_host_dependent_h0",
         "fig5": "fig5_offsets", "fig6": "fig6_flux_limits", "fig7": "fig7_free_densities", "fig8": "fig8_validation", "figC1": "figC1_corner"}


# black input lines and markers drawn in each figure, against the input macros of values.json
INPUT_MACROS = {"H0": "MockHzero", "f": "MockfAGN", "lognGal": "MockLognGal", "lognAGN": "MockLognAGN", "muG": "MockMuG",
                "dmuG": "MockDmuG", "muChi": "MockMuChi", "dmuChi": "MockDmuChi", "sigmaG": "MockSigmaG", "sigmaChi": "MockSigmaChi"}
INPUTS_DRAWN = {"fig1": ["muG", "dmuG", "muChi", "dmuChi"], "fig2": ["H0", "f"], "fig3": ["H0"], "fig4": ["H0"],
                "fig5": ["f", "dmuG", "dmuChi"], "fig6": ["H0", "f"], "fig7": ["lognGal", "lognAGN", "f"], "fig8": ["H0", "f"],
                "figC1": ["H0", "f", "muG", "dmuG", "muChi", "dmuChi", "sigmaG", "sigmaChi"]}


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
            chk(NAMES[w], ("macros", INPUT_MACROS[k], "value"), IN[k], "input line")
    bad = run_checks([NAMES[w] for w in which])
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
