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
combined-posterior summaries, printed labels) with values/values.json and prints any mismatch; a number
drawn as a point or printed in a panel is also compared with the macro the text prints, at the printed
precision (the drawn value rounded as the macro rounds it must give the macro's text). Any mismatch makes
the script exit 1.

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
from matplotlib.patches import Patch  # noqa: E402
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


def chk_macro(fig, macro, shown, what=""):
    """A number drawn in a panel (or printed in it) against the macro the text prints: the same source value
    (values.json 'value', to 1e-9) and the same text at the printed precision (the drawn value rounded to the
    macro's decimals equals the macro's rendering, \\ph{} stripped)."""
    chk(fig, ("macros", macro, "value"), shown, what)
    CHECKS.append((fig, ("macros", macro, "render"), ("PRINTED", float(shown)), what + " (at the printed precision)"))


def chk_text(fig, macro, text, what=""):
    """A number printed in a panel as text against the macro's rendering (\\ph{} stripped)."""
    CHECKS.append((fig, ("macros", macro, "render"), ("TEXT", text), what + " (text printed in the panel)"))


def unwrap_ph(r):
    import re as _re
    m = _re.fullmatch(r"\\ph\{(.*)\}", r)
    return m.group(1) if m else r


def printed_as(v, render):
    """v rounded as `render` (a plain decimal, possibly in \\ph{}) is rounded: the same number of decimals."""
    t = unwrap_ph(render)
    d = len(t.split(".")[1]) if "." in t else 0
    return bv.fnum(float(v), d), t


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
        if isinstance(shown, tuple) and shown[0] == "PRINTED":
            got, ref = printed_as(shown[1], ref)
            ok = got == ref
            shown = got
        elif isinstance(shown, tuple) and shown[0] == "TEXT":
            shown, ref = shown[1], unwrap_ph(ref)
            ok = shown == ref
        elif isinstance(shown, str):
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
HD_EVENTS_FILE = "events_marked_dmu0p10_dmuG5_corepop.h5"   # fig1 (c, d): the production host-dependent events of realization A
EQ_EVENTS_FILE = bv.EV_REF.name                              # fig1 (a, b): the production equal-population events of realization A
SLAB_DEC_DEG = 3.0       # half-thickness of the declination slab of panel (a)
WEDGE_RA = (0.0, np.pi)  # right-ascension range of panel (a)
WEDGE_ZMAX = 0.33


def fig1_mocks(outdir=None, stem=None):
    """outdir, stem: where to write (the short draft draws the same figure as its figA1_mocks)."""
    name = stem or "fig1_mocks"
    meta = bv.jload(bv.META100)
    H0, Om = meta["stages"]["events"]["cosmology"]["H0"], meta["stages"]["events"]["cosmology"]["Om0"]
    sur = meta["stages"]["surveys"]
    zh = float(sur["horizon_z"])
    cat = SEED_REF / "catalogs"
    dmax = np.radians(SLAB_DEC_DEG)
    g = read_rows(cat / "catalog_gal_complete.h5", WEDGE_ZMAX, dmax, ("ra", "dec", "z"))
    a = read_rows(cat / "catalog_agn_complete.h5", WEDGE_ZMAX, dmax, ("ra", "dec", "z"))
    # (a, b): the equal-population event set of realization A that the production runs analyse (events_corepop.h5, 298 AGN-hosted);
    # until 17 Nov these panels drew the older events.h5 (295 AGN-hosted)
    with h5py.File(SEED_REF / "events" / EQ_EVENTS_FILE) as h:
        ev = {k: h["truth/" + k][()] for k in ("ra", "dec", "z", "host_type")}
    n_ev_agn = int((ev["host_type"] == 1).sum())
    chk(name, ("macros", "MockfAGNRealised", "value"), n_ev_agn / len(ev["host_type"]), "(b) AGN-hosted fraction of the drawn mergers")
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
    # (c, d): the host-dependent event set of realization A that the production runs analyse (Analysis 16, stage 4 and the
    # experiment's s1 sets); until 17 Nov these panels drew the older events_marked_dmu0p10_dmuG5.h5 (357 AGN-hosted, not 360)
    with h5py.File(SEED_REF / "events" / HD_EVENTS_FILE) as h:
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
               f"{int(eg.sum())} galaxy-hosted and {int(ea.sum())} AGN-hosted detected mergers (of 1000) of the reference catalog ({EQ_EVENTS_FILE})")

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
    note(name, f"(b) 1000 hosts of the reference catalog ({EQ_EVENTS_FILE}): {int((ev['host_type'] == 0).sum())} galaxies, {n_ev_agn} AGN; "
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
    note(name, f"(c,d) host-dependent events of realization A ({HD_EVENTS_FILE}), 1000 detected: {1000 - n_agn} galaxy-hosted, {n_agn} AGN-hosted; true values; "
               f"histograms normalised per class")
    fs.save(fig, outdir or FIGDIR, name)


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
# THE NINE FIGURES OF THE SKELETON (notes/PAPER_SKELETON.md Secs. 0, 4). Placeholder data come from
# values/placeholder_results.json and are drawn in red tones (told apart by line style and direct labels)
# with the small red word "placeholder" in a corner of the panel; an entry that values/production_map.json
# overrides is marked final and the same code draws it in the final palette.
# ============================================================================================= #
PHJ = json.loads((PRD / "values" / "placeholder_results.json").read_text())["keys"]
RED = ["#B2182B", "#E0645A", "#6E0010", "#D9786B"]          # placeholder tones: strong, salmon, maroon, rose
NONE_C = "#4D4D4D"                                            # final colour of "no catalog"
SINGLE_C = "#CC79A7"                                          # final colour of "single population"
LS4 = ["-", "--", "-.", ":"]
IN_H0, IN_F = IN["H0"], IN["f"]


def fin(key):
    return bool(PHJ[key].get("final"))


def kcol(key, final_color, i):
    return final_color if fin(key) else RED[i % len(RED)]


def ph_word(ax, x=0.985, y=0.975, ha="right", va="top"):
    # No note to the author is printed inside a figure (owner's style guide): the red colour of the
    # elements that wait for a run carries the marking. Kept as a no-op so the call sites stay.
    return


def q1(key, q, part="A"):
    s = PHJ[key]["quantities"][q][part]
    return np.asarray(s["x"]), np.asarray(s["p"]), s


def chk_key(fig, key, q, part="A"):
    s = PHJ[key]["quantities"][q][part]
    tag = "" if part == "A" else "Comb"
    for kk, v in (("median", s["median"]), ("ci90.0", s["ci90"][0]), ("ci90.1", s["ci90"][1])):
        chk(fig, ("macros", f"{key}{q}{tag}", "value", *kk.split(".")), v, "drawn placeholder posterior")


def hpd90(J):
    x, y, P = np.asarray(J["x"]), np.asarray(J["y"]), np.asarray(J["p"])
    d = np.sort(P.ravel())[::-1]
    c = np.cumsum(d) / d.sum()
    return x, y, P, float(d[np.searchsorted(c, 0.9)])


def smooth_joint(J, sigma):
    """Gaussian smoothing (in grid cells) of a stored 2-D density before its 90% contour is drawn."""
    from scipy.ndimage import gaussian_filter
    return dict(J, p=gaussian_filter(np.asarray(J["p"], float), sigma, mode="nearest").tolist())


def draw_joint(ax, J, color, ls="-", fill=False, alpha=0.25, lw=fs.LW_CONTOUR, zorder=3):
    x, y, P, lev = hpd90(J)
    if fill:
        ax.contourf(x, y, P.T, levels=[lev, np.inf], colors=[color], alpha=alpha, zorder=zorder)
    ax.contour(x, y, P.T, levels=[lev], colors=[color], linewidths=lw, linestyles=[ls], zorder=zorder + 0.1)
    return x, y, P, lev


def draw_1d(ax, key, q, part, color, ls, lw=fs.LW, shade=False, zorder=4):
    x, p, s = q1(key, q, part)
    ax.plot(x, p, color=color, ls=ls, lw=lw, zorder=zorder)
    if shade:
        shade90(ax, x, p, s, color, alpha=0.18)
    return x, p, s


def label_at(ax, x, p, frac_x, text, color, dx=0.0, dy=0.0, ha="left", va="center", **kw):
    """Direct label placed at the curve point where the cumulative mass reaches frac_x, shifted by (dx, dy) in data units."""
    c = np.concatenate([[0.0], np.cumsum(0.5 * (p[1:] + p[:-1]) * np.diff(x))])
    c /= c[-1]
    xx = float(np.interp(frac_x, c, x))
    yy = float(np.interp(xx, x, p))
    return fs.label(ax, xx + dx, yy + dy, text, color, ha=ha, va=va, **kw)


# ---------------------------------------------------------------- Fig. 2: the anchor
def fig2_anchor():
    name = "fig2_anchor"
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(fs.PAGE, 2.9), layout="constrained", gridspec_kw=dict(width_ratios=[1.0, 1.25]))
    k = "KnownBoth"
    cA, cC = kcol(k, fs.BOTH, 0), kcol(k, fs.BOTH, 0)
    draw_joint(axa, PHJ[k]["joint"]["A"], cA, ls="--", fill=False)
    draw_joint(axa, PHJ[k]["joint"]["Comb"], cC, ls="-", fill=True, alpha=0.35)
    axa.plot([IN_H0], [IN_F], marker="+", ms=11, mew=1.6, color=fs.INPUT, zorder=8)
    for part in ("A", "Comb"):
        chk_key(name, k, "Hzero", part)
        chk_key(name, k, "fAGN", part)
    axa.set_xlim(65.5, 73.0)
    axa.set_ylim(0.13, 0.43)
    axa.set_xlabel(fs.H0_LABEL)
    axa.set_ylabel(r"$f_{\rm AGN}$")
    fs.label(axa, 70.55, 0.372, "realization A", cA, ha="left", va="bottom")
    fs.label(axa, 69.35, 0.282, "combined over\nrealizations", cC, ha="left", va="center")
    fs.panel_label(axa, "(a)")
    if not fin(k):
        ph_word(axa)
    # (b) H0 with each catalog, realization A
    specs = [("KnownBoth", fs.BOTH, "both catalogs", 0, "-"), ("KnownGal", fs.GAL, "galaxy catalog alone", 1, "--"),
             ("KnownAGN", fs.AGN, "AGN catalog alone", 2, "-."), ("KnownNone", NONE_C, "no catalog", 3, ":")]
    top = 0.0
    curves = {}
    for key, fc, lab, i, ls in specs:
        c = kcol(key, fc, i)
        x, p, s = draw_1d(axb, key, "Hzero", "A", c, ls)
        chk_key(name, key, "Hzero", "A")
        curves[key] = (x, p, c)
        if key != "KnownAGN":
            top = max(top, float(p.max()))
    fs.input_vline(axb, IN_H0)
    axb.set_xlim(56, 100)
    ymax = 1.22 * top
    axb.set_ylim(0, ymax)
    xA, pA, cAg = curves["KnownAGN"]
    axb.annotate("", xy=(99.4, 0.985 * ymax), xytext=(99.4, 0.80 * ymax), arrowprops=dict(arrowstyle="->", color=cAg, lw=1.2))
    fs.label(axb, 98.6, 0.92 * ymax, "AGN catalog alone:\nrises to the prior edge,\npeak off scale", cAg, ha="right", va="top")
    for key, frac, lab, dx, dy, ha in (("KnownBoth", 0.80, "both catalogs", 0.6, 0.004, "left"), ("KnownGal", 0.93, "galaxy catalog\nalone", 0.7, 0.0, "left"),
                                       ("KnownNone", 0.06, "no catalog", -0.8, 0.0, "right")):
        x, p, c = curves[key]
        label_at(axb, x, p, frac, lab, c, dx=dx, dy=dy, ha=ha)
    axb.set_xlabel(fs.H0_LABEL)
    axb.set_ylabel(fs.H0_DENS)
    fs.panel_label(axb, "(b)")
    if not all(fin(k_) for k_, *_ in specs):
        ph_word(axb, x=0.015, ha="left", y=0.86)
    note(name, f"(b) AGN catalog alone peak density {pA.max():.2f} at H0 = {xA[np.argmax(pA)]:.2f}; plotted y range [0, {ymax:.3f}]")
    fs.save(fig, FIGDIR, name)


# ---------------------------------------------------------------- Fig. 3: flux limits
def fig3_flux_limits():
    name = "fig3_flux_limits"
    fig = plt.figure(figsize=(fs.PAGE, 3.6), layout="constrained")
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.15])
    axH = fig.add_subplot(gs[0, 0])
    axF = fig.add_subplot(gs[1, 0], sharex=axH)
    axb = fig.add_subplot(gs[:, 1])
    panel_flux_points(axH, axF, name)
    drawn = panel_density_response(axb, name)
    fs.save(fig, FIGDIR, name)
    write_fig3_facts(drawn)


def panel_flux_points(axH, axF, name):
    """Fig. 3(a): medians and 90% intervals of H0 (axH) and f_AGN (axF), realization A, against the flux limit."""
    keys = ["KnownBoth"] + [f"Flux{w}" for _, w, _ in bv.RUNGS]
    xs = np.arange(len(keys))
    for ax, q, x0, yl in ((axH, "Hzero", IN_H0, fs.H0_LABEL), (axF, "fAGN", IN_F, r"$f_{\rm AGN}$")):
        for i, k in enumerate(keys):
            s = PHJ[k]["quantities"][q]["A"]
            c = kcol(k, fs.BOTH if i == 0 else fs.COMP[i - 1], 0)
            mk = "o" if i == 0 else fs.COMP_MARKERS[i - 1]
            ax.errorbar([xs[i]], [s["median"]], yerr=[[s["median"] - s["ci90"][0]], [s["ci90"][1] - s["median"]]], fmt=mk, ms=fs.MS, color=c,
                        mfc="white" if i == 0 else c, mew=1.3, elinewidth=1.6, capsize=2.5, zorder=4)
            chk_key(name, k, q, "A")
        fs.input_hline(ax, x0)
        ax.set_ylabel(yl)
        ax.set_xlim(-0.5, len(keys) - 0.5)
        ax.tick_params(axis="x", which="minor", bottom=False, top=False)
        if not all(fin(k) for k in keys):
            ph_word(ax, y=0.95)
    axH.set_ylim(65.5, 72.5)
    axF.set_ylim(0.12, 0.45)
    axF.set_xticks(xs)
    axF.set_xticklabels(["complete"] + [f"$m<{m}$" for _, _, m in bv.RUNGS])
    axF.set_xlabel("flux limit of both catalogs")
    plt.setp(axH.get_xticklabels(), visible=False)
    sec = axH.secondary_xaxis("top")
    sec.set_xticks(xs)
    comp = ["100"] + [VALUES["macros"][f"MockEventComp{w}"]["render"] for _, w, _ in bv.RUNGS]
    for w in [w for _, w, _ in bv.RUNGS]:
        chk(name, ("macros", f"MockEventComp{w}", "render"), VALUES["macros"][f"MockEventComp{w}"]["render"], "upper-axis label")
    sec.set_xticklabels([f"{c}%" for c in comp])
    sec.tick_params(axis="x", which="minor", top=False)
    sec.set_xlabel("completeness weighted by the detected mergers", fontsize=fs.FS_TEXT)
    fs.panel_label(axH, "(a)", y=0.95)


# Fig. 3(b): the series drawn, finished results only (in drawing order). Analysis 15: one realization (catalog A); production: realization A
# of Analysis 16's stage-2 runs (scripts/density.py). x offsets (multiplicative, log axis) keep the four sets at one factor apart.
FIG3B_A15 = [  # (tag, label, [(macro, factor)], marker, filled, x offset, curve style, curve description)
    ("five", "complete, normalized to $z=5$", [("NormfCompleteFive", 1.0), ("NormfCompleteFiveTwo", 2.0)], "o", True, 0.962, ":",
     "complete catalogs with the hosts normalized to z = 5 (Analysis 15, scans p1_complete and c_complete_x2; the input AGN density and twice it)"),
    ("prod", "complete, normalized to $z=1.10$", [("NormfCompleteProd", 1.0), ("NormfCompleteProdTwo", 2.0)], "o", False, 0.987, ":",
     "complete catalogs with the hosts normalized to z = 1.10 (Analysis 15, scans b_complete_x1 and b_complete_x2)")]
FIG3B_PROD = [  # (limit word, m_lim, colour index of fs.COMP, marker, x offset, curve style)
    ("Twenty", 20, 1, "s", 1.013, "-"),
    ("Eighteen", 18, 3, "D", 1.04, "-.")]
LS_NAME = {"-": "solid", "-.": "dash-dot", ":": "dotted", "--": "dashed"}


def panel_density_response(axb, name):
    """Fig. 3(b): f_AGN against the assumed AGN number density over its input (realization A), finished results only: Analysis 15's
    complete catalogs with the hosts normalized to z = 5 and to z = 1.10, and Analysis 16's runs at m < 20 and m < 18 with the AGN density
    assumed at 0.5, 1 and 2 times the value from the catalog counts (scripts/density.py), each with the identity's prediction (black).
    Every point is a macro of values.json; every curve passes through its factor-1 point and is checked against the predicted shift's
    macro at the factors that have one. Returns what was drawn, for the generated block of figures/CAPTION_FACTS.md."""
    import density as DN
    M = VALUES["macros"]
    real = lambda n: M[n]["class"] != "ph"
    drawn = {"a15": [], "prod": [], "skipped": []}
    xx = np.asarray(DN.CURVE_X)

    def point(x, v, mk, c, filled):
        axb.errorbar([x], [v["median"]], yerr=[[v["median"] - v["ci90"][0]], [v["ci90"][1] - v["median"]]], fmt=mk, ms=fs.MS, color=c,
                     mfc=c if filled else "white", mew=1.3, elinewidth=1.4, capsize=2.2, zorder=6)

    # Analysis 15: complete catalogs (black circles; filled: normalized to z = 5, open: to z = 1.10)
    for tag, lab, pts, mk, filled, off, ls, desc in FIG3B_A15:
        if not all(real(m) for m, _ in pts):
            drawn["skipped"].append(lab)
            continue
        for m, x in pts:
            point(x * off, M[m]["value"], mk, "k", filled)
            chk_post(name, m, M[m]["value"], f"Analysis 15, {lab}, factor {x:g}")
        f1 = M[pts[0][0]]["value"]["median"]
        if tag == "five":
            cur = DN.a15_complete_curve(xx)
            y = f1 + np.asarray(cur["median"]) - cur["median"][list(xx).index(1.0)]
            pred = "NormDeltafTwoCompleteFivePred"
        else:
            assert M["NormFAGNCompleteProd"]["value"] == 1.0          # F_AGN = 1: the identity predicts no shift
            y = np.full_like(xx, f1)
            pred = "NormDeltafTwoCompleteProdPred"
        axb.plot(xx, y, color="k", lw=1.0, ls=ls, zorder=3)
        i1, i2 = list(xx).index(1.0), list(xx).index(2.0)
        chk(name, ("macros", pts[0][0], "value", "median"), y[i1], f"identity curve of {lab} at factor 1")
        CHECKS.append((name, ("macros", pred, "render"), ("PRINTED", float(y[i2] - y[i1])), f"identity curve of {lab}: shift at factor 2 (at the printed precision)"))
        drawn["a15"].append({"tag": tag, "label": lab, "points": pts, "marker": mk, "filled": filled, "ls": ls, "desc": desc, "pred": pred,
                             "curve_shift_2": float(y[i2] - y[i1])})
    # Analysis 16, stage 2: m < 20 and m < 18, realization A (the colours and markers of panel (a))
    for word, mlim, ci, mk, off, ls in FIG3B_PROD:
        names = {0.5: f"Flux{word}HalffAGN", 1.0: f"Flux{word}fAGN", 2.0: f"Flux{word}DoublefAGN"}
        c = fs.COMP[ci]
        if not all(real(n) for n in names.values()):
            drawn["skipped"].append(f"m < {mlim}")
            continue
        for x, n in names.items():
            point(x * off, M[n]["value"], mk, c, True)
            chk_post(name, n, M[n]["value"], f"m < {mlim}, factor {x:g}")
        preds = {0.5: f"Flux{word}HalfDeltafAGNPred", 2.0: f"Flux{word}DoubleDeltafAGNPred"}
        curve = None
        if all(real(p_) for p_ in preds.values()):
            cur = M[preds[2.0]]["extra"]["curve"]
            assert cur["x"] == list(xx), "curve lattice"
            f1 = M[names[1.0]]["value"]["median"]
            y = f1 + np.asarray(cur["shift"])
            axb.plot(xx, y, color="k", lw=1.1, ls=ls, zorder=3)
            chk(name, ("macros", names[1.0], "value", "median"), y[list(xx).index(1.0)], f"identity curve of m < {mlim} at factor 1")
            for x, p_ in preds.items():
                chk(name, ("macros", p_, "value"), y[list(xx).index(x)] - f1, f"identity curve of m < {mlim}: shift at factor {x:g}")
            curve = ls
        drawn["prod"].append({"word": word, "mlim": mlim, "color": c, "marker": mk, "names": names, "preds": preds, "ls": curve, "offset": off})
    fs.input_hline(axb, IN_F)
    axb.axvline(1.0, color=fs.PRIOR, lw=0.8, ls=":", zorder=1)
    axb.set_xscale("log")
    axb.set_xticks([0.5, 0.7, 1.0, 1.4, 2.0])
    axb.set_xticklabels(["0.5", "0.7", "1", "1.4", "2"])
    axb.xaxis.set_minor_locator(plt.NullLocator())
    axb.set_xlim(0.45, 2.25)
    axb.set_ylim(0.05, 0.60)
    axb.set_xlabel("assumed AGN number density / input")
    axb.set_ylabel(r"$f_{\rm AGN}$")
    # key: each series' marker beside the black line of its identity curve
    from matplotlib.legend_handler import HandlerTuple
    hs, ls_ = [], []
    for d in drawn["a15"]:
        hs.append((Line2D([], [], color="k", marker=d["marker"], ms=fs.MS, mfc="k" if d["filled"] else "white", mew=1.3, ls="none"),
                   Line2D([], [], color="k", lw=1.0, ls=d["ls"])))
        ls_.append(d["label"])
    for d in drawn["prod"]:
        mh = Line2D([], [], color=d["color"], marker=d["marker"], ms=fs.MS, mfc=d["color"], mew=1.3, ls="none")
        hs.append((mh, Line2D([], [], color="k", lw=1.1, ls=d["ls"])) if d["ls"] else mh)
        ls_.append(f"$m<{d['mlim']}$")
    axb.legend(hs, ls_, loc="upper left", handler_map={tuple: HandlerTuple(ndivide=None, pad=0.6)}, handlelength=3.2, fontsize=fs.FS_TEXT,
               borderaxespad=0.5, labelspacing=0.35, frameon=True, facecolor="white", edgecolor="none", framealpha=1.0)
    drawn["legend"] = ls_
    fs.panel_label(axb, "(b)", x=0.975, y=0.04, ha="right", va="bottom")
    for s_ in drawn["skipped"]:
        note(name, f"(b) not drawn (a run is missing): {s_}")
    return drawn


def write_fig3_facts(drawn):
    """The generated block of Fig. 3 in figures/CAPTION_FACTS.md (both panels), from values.json and placeholder_results.json."""
    keys = ["KnownBoth"] + [f"Flux{w}" for _, w, _ in bv.RUNGS]
    mk_a = ["open circle (bluish green)"] + [f"filled {m} ({c})" for m, c in zip(("circle", "square", "triangle", "diamond"),
                                                                                    ("lightest purple", "light purple", "dark purple", "darkest purple"))]
    allfin = all(fin(k) for k in keys)
    L = ["## Fig. 3 `fig3_flux_limits` (" + ("final: every element a finished result" if allfin and not drawn["skipped"] else "partly placeholder")
         + "; written by the figure script from values.json and placeholder_results.json)", "",
         "- (a) Medians and 90% intervals of H0 (top) and f_AGN (bottom) of realization A against the flux limit of both catalogs, number "
         "densities known (flux-limited catalogs: from the catalog counts, hosts normalized to z = 1.10): "
         + "; ".join(f"{lab} {mk}: `\\{k}Hzero`, `\\{k}fAGN`, {key_state(k, 'fAGN', 'A')}"
                     for k, lab, mk in zip(keys, ["complete"] + [f"m < {m}" for _, _, m in bv.RUNGS], mk_a))
         + ". Upper axis: the completeness weighted by the detected mergers, 100% for complete catalogs (by construction) and "
         + ", ".join(mfmt(f"MockEventComp{w}", "%") for _, w, _ in bv.RUNGS) + ". Black dashed: the inputs "
         + f"{mfmt('MockHzero')}, {mfmt('MockfAGN')}.",
         "- (b) Median and 90% interval of f_AGN against the assumed AGN number density over its input (log axis; factors 0.5, 1, 2), "
         "realization A, finished results only; points of one factor offset in x by the factors given (multiplicative) for legibility."]
    for d in drawn["a15"]:
        L.append(f"  - {d['desc']}: black {'filled' if d['filled'] else 'open'} circles ("
                 + ", ".join(f"factor {x:g} {mfmt(m)}" for m, x in d["points"])
                 + f"; x offset {[a for a in FIG3B_A15 if a[0] == d['tag']][0][5]:g}); one realization (catalog A, the exploratory event set "
                 "events.h5), darksirens-core e7c3007; final. Identity curve: black " + LS_NAME[d["ls"]]
                 + (f", F_AGN = {mfmt('NormFAGNComplete')} at H0 = 67.74 (H0-dependent: densities in Mpc^-3), Analysis 15's map of its zero-density "
                    "grid row by row in H0, through the factor-1 point; its shift at factor 2 is " + mfmt(d["pred"]) + " (measured "
                    + mfmt("NormDeltafTwoCompleteFive") + ")" if d["tag"] == "five" else
                    f", flat at the factor-1 median: F_AGN = {mfmt('NormFAGNCompleteProd')}, no shift ({mfmt(d['pred'])}; measured "
                    + mfmt("NormDeltafTwoCompleteProd") + ")") + ".")
    for d in drawn["prod"]:
        n = d["names"]
        L.append(f"  - m < {d['mlim']} (Analysis 16, stage 2, realization A; the runs of `values/production_map.json` 'density_response' and of key "
                 f"`Flux{d['word']}` at factor 1): filled {'squares' if d['marker'] == 's' else 'diamonds'} in the colour of m < {d['mlim']} in (a) "
                 f"(x offset {d['offset']:g}): factor 0.5 {mfmt(n[0.5])}, 1 {mfmt(n[1.0])}, 2 {mfmt(n[2.0])}; final. "
                 + (f"Identity curve: black {LS_NAME[d['ls']]}, the unit-factor grid of realization A mapped through the identity with "
                    f"F_AGN = {mfmt('NormFAGN' + d['word'] + 'Counts')} (H0-independent: densities in h^3 Mpc^-3), through the factor-1 point; "
                    f"predicted shifts {mfmt(d['preds'][0.5])} (factor 0.5) and {mfmt(d['preds'][2.0])} (factor 2), measured "
                    f"{mfmt(n[0.5].replace('fAGN', 'DeltafAGN'))} and {mfmt(n[2.0].replace('fAGN', 'DeltafAGN'))}; the map carries the "
                    "normaliser only (the hosts added where the catalog is incomplete and mergers lie also scale with the density)."
                    if d["ls"] else "No identity curve (its macros are red)."))
    if drawn["skipped"]:
        L.append("  - Not drawn (a run is missing): " + ", ".join(drawn["skipped"]) + ".")
    L.append("  - Key (upper left, lower-case labels): each series' marker beside the black line of its identity curve: "
             + "; ".join(f"\"{lab.replace('$', '')}\"" for lab in drawn.get("legend", [])) + ". No other text in the panel. Black dashed horizontal: the input "
             f"{mfmt('MockfAGN')}; gray dotted vertical: factor 1. Not drawn any more: the exploratory grids normalized to z = 5 at m < 21 and "
             "m < 18 (Analyses 3 and 4) and Analysis 15's m < 18 normalized to z = 1.10 (its production counterpart is the m < 18 series).")
    write_facts_block("fig3_flux_limits", "## Fig. 3 `fig3_flux_limits`", L)


# ---------------------------------------------------------------- Fig. 4: number densities inferred
def fig4_densities():
    name = "fig4_densities"
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(fs.PAGE, 3.0), layout="constrained", gridspec_kw=dict(width_ratios=[1.0, 1.1]))
    ya = panel_free_joint(axa, name)
    panel_rate_ratio(axb, name)
    fs.save(fig, FIGDIR, name)
    write_fig4_facts(ya)


# ---------------------------------------------------------------- generated blocks of figures/CAPTION_FACTS.md
def mfmt(m, unit=""):
    """`\\Macro` = text[unit] (real | red), from values.json."""
    V = VALUES["macros"][m]
    return f"`\\{m}` = {unwrap_ph(V['render'])}{unit} ({'real' if V['class'] != 'ph' else 'red'})"


def key_state(key, q, part):
    """Whether the drawn curve of a key is a production result or a placeholder, and its source (placeholder_results.json)."""
    E = PHJ[key]
    if E.get("final"):
        which = "realization A (100)" if part == "A" else f"{E['N_comb']} realizations {E['realizations']}"
        import re as _re
        byR = (E.get("production_runs_by_realization") or {}) if part != "A" else {}
        more = ""
        if byR:
            pats = sorted({_re.sub(r"realization_\d{3}", "realization_{R}", n) for n in byR.values()})
            more = f"; realizations {', '.join(byR)}: `{'`, `'.join(pats)}` (stage X, the same analysis and settings)"
        return f"real (production: {which} of `{E['production_run']}`{more})"
    Q = E["quantities"][q]
    src = Q["source"] if part == "A" else Q["comb_source"]
    f = Q.get("file") if part == "A" else Q.get("comb_file")
    return f"red placeholder, {src}" + (f" (`{f}`)" if f and src.startswith("exploratory") else "")


def write_facts_block(name, heading, lines):
    """Replace the block of figure `name` in figures/CAPTION_FACTS.md (between its two markers; the first time, the hand-written
    section that starts with `heading`, up to the next section)."""
    beg, end = f"<!-- {name}: written by scripts/make_figures.py; do not edit -->", f"<!-- end {name} -->"
    block = "\n".join([beg] + lines + [end])
    cf = FIGDIR / "CAPTION_FACTS.md"
    t = cf.read_text()
    if beg in t:
        t = t.split(beg)[0] + block + t.split(end, 1)[1]
    else:
        i = t.index(heading)
        rest = t[i + len(heading):]
        nxt = [j for j in (rest.find("\n## "), rest.find("\n<!-- ")) if j >= 0]
        tail = rest[min(nxt) + 1:] if nxt else ""
        t = t[:i] + block + ("\n\n" + tail if tail else "\n")
    cf.write_text(t)


def write_fig4_facts(ya):
    ylo, yhi = ya
    keys = [f"Free{w}" for _, w, _ in bv.RUNGS]
    prod = fin("FreeEighteen") and fin("FreeTwentyOne")
    q18 = PHJ["FreeEighteen"]["quantities"]["LognAGN"]["A"]
    hnote = q18.get("h_note", "")
    L = ["## Fig. 4 `fig4_densities` (number densities inferred, realization A; written by the figure script from values.json and "
         "placeholder_results.json)", "",
         "- Keys: " + "; ".join(f"`{k}` {key_state(k, 'fAGN', 'A')}" for k in keys) + ".",
         "- Unit of the number densities: log10 of n in h^3 Mpc^-3 (h = H0/100), the unit the production analyses sample (Table I, "
         f"`\\PriorLognAGNLo`, `\\PriorLognAGNHi`) and the text quotes (Table IV, Sec. IV C); y-axis label log10[n_AGN/(h^3 Mpc^-3)]. "
         f"The drawn densities are {hnote}.",
         "- (a) Joint posterior of f_AGN and log10 n_AGN with both number densities inferred (log10 n_gal marginalized): 90% regions at "
         "m < 21 (dashed) and m < 18 (solid) (`\\FreeTwentyOnefAGN`, `\\FreeEighteenfAGN`; 90% upper limits on log10 n_AGN "
         f"{mfmt('FreeTwentyOneLognAGNULh')} and {mfmt('FreeEighteenLognAGNULh')}). "
         + ("Gaussian kernel density of the run's samples (0.3 sd per axis, reflected at the prior edges) on a lattice, smoothed by 1.5 cells. " if prod else
            f"50 x 50 histograms, on the production prior box (f [0, 1] x [{ylo:g}, {yhi:g}] in h^3 Mpc^-3; bin centres "
            f"[{PHJ['FreeEighteen']['joint']['A']['y'][0]:.2f}, {PHJ['FreeEighteen']['joint']['A']['y'][-1]:.2f}]), of the exploratory samples whose two "
            "densities, converted with each sample's H0, lie inside that box (the exploratory runs drew in a box in Mpc^-3 that reaches above it; "
            "the part below about -5.5 holds no sample), smoothed; each sample's f_AGN is moved by the re-anchoring map of the key's f_AGN "
            "(placeholder_results header). ")
         + f"Black curve: constant merger rate per host at the input value {mfmt('MockRateRatio')}, with n_gal at its input "
         f"{mfmt('MockLognGalh')}: logit f_AGN = ln R + ln Z_AGN(n_AGN) - ln Z_gal, Z_k = N_k + n_k V_miss,k with n_k in h^3 Mpc^-3 and "
         f"the missing volume at H0 = 100 (so the curve does not depend on H0), m < 18 catalog counts of realization A, hosts normalized to "
         f"z = {bv.ZMAX_PROD if prod else bv.ZMAX_GRID:g}{' (the production convention)' if prod else ' (as the exploratory runs; z = 1.10 once the production runs are drawn)'}. "
         "Gray: the 90% region allowed by the priors once the merger rate per host is measured (2e6 draws from the priors weighted by the "
         f"known-density likelihood of m < 18 mapped through the identity{'' if prod else ', those inside the Table I box kept'}; "
         f"{mfmt('FreeEighteenLognAGNPriorULh')}). Black \"+\": input "
         f"({mfmt('MockfAGN')}, {mfmt('MockLognAGNh')}); the curve passes through it. Axes: f_AGN [0, 1]; log10 n_AGN [{ylo:.2f}, {yhi:.2f}] "
         "(the production prior box, `\\PriorLognAGNLo`, `\\PriorLognAGNHi`; every drawn region and limit lies inside it)."
         + " The same limits in Mpc^-3 (each sample converted with its own H0) are the macros without the final h, not drawn.",
         "- (b) Posterior of the merger rate per AGN over that per galaxy at m < 21, 20, 19, 18 (solid, dashed, dash-dot, dotted; "
         + ", ".join(mfmt(f"{k}RateRatio") for k in keys) + "), computed per posterior sample as [f_AGN/(1 - f_AGN)] Z_gal/Z_AGN with "
         "that sample's densities. The ratio has no unit and does not depend on the unit of the densities: in production Z_k = N_k + n_k "
         "V_miss,k(H0 = 100) with the sampled n_k in h^3 Mpc^-3, in the exploratory placeholder the same Z_k from n_k in Mpc^-3 with the volume "
         f"at the sample's H0. Black dashed: input {mfmt('MockRateRatio')}. With the densities at their inputs the same ratio is "
         f"{mfmt('FluxEighteenRateRatio')}."]
    write_facts_block("fig4_densities", "## Fig. 4 `fig4_densities`", L)


def panel_free_joint(axa, name, label="(a)"):
    """Fig. 4(a): f_AGN against log10 n_AGN in h^3 Mpc^-3 (the unit the analysis samples and the text quotes) with the number
    densities inferred (realization A). The joint posteriors and the prior-allowed region are those of placeholder_results.json,
    stored in h^3 Mpc^-3: the production runs' own samples and prior draws of log10 n_AGN; while a key is a placeholder, the
    exploratory run's samples converted with each sample's own H0. The curve of constant rate per host is H0-independent in this
    unit: Z_k = N_k + n_k V_miss,k(H0 = 100) with n_k in h^3 Mpc^-3."""
    PI = PHJ["FreeEighteen"]["joint"]["prior_induced"]
    J = smooth_joint(PI, 2.0)
    draw_joint(axa, J, "0.55", ls="-", fill=True, alpha=0.30, lw=0.8, zorder=1)
    joints = {}
    for i, (k, mlim, ls) in enumerate((("FreeTwentyOne", 21, "--"), ("FreeEighteen", 18, "-"))):
        c = kcol(k, fs.COMP[0 if mlim == 21 else 3], 2 * i)
        JA = PHJ[k]["joint"]["A"]
        assert JA["params"] == ["fAGN", "LognAGNh"] and JA.get("units_y") == "log10 h^3 Mpc^-3", (k, JA.get("params"), JA.get("units_y"))
        joints[k] = draw_joint(axa, smooth_joint(JA, 1.5), c, ls=ls, fill=False, zorder=4)
        chk_key(name, k, "fAGN", "A")
        chk(name, ("macros", f"{k}LognAGNULh", "value"), PHJ[k]["quantities"]["LognAGN"]["A"]["ul90_h"], "90% upper limit of the drawn log10 n_AGN [h^3 Mpc^-3]")
    assert PI["params"] == ["fAGN", "LognAGNh"] and PI.get("units_y") == "log10 h^3 Mpc^-3", (PI.get("params"), PI.get("units_y"))
    chk(name, ("macros", "FreeEighteenLognAGNPriorULh", "value"), PHJ["FreeEighteen"]["quantities"]["LognAGN"]["A"]["prior_ul90_h"],
        "90% upper limit of the prior-allowed region [h^3 Mpc^-3]")
    # vertical range: the production prior box (Table I, PriorLognAGNLo/Hi, h^3 Mpc^-3); the placeholders are reweighted to it (only the
    # exploratory samples and prior draws inside it are kept: placeholder_results header, 'prior box'), so every drawn region lies inside
    prod = fin("FreeEighteen") and fin("FreeTwentyOne")
    ylo, yhi = VALUES["macros"]["PriorLognAGNLo"]["value"], VALUES["macros"]["PriorLognAGNHi"]["value"]
    chk(name, ("macros", "PriorLognAGNLo", "value"), ylo, "vertical axis: prior box")
    chk(name, ("macros", "PriorLognAGNHi", "value"), yhi, "vertical axis: prior box")
    for J_, who in [(PI, "prior-allowed region")] + [(PHJ[k]["joint"]["A"], k) for k in joints]:
        assert ylo <= J_["y"][0] and J_["y"][-1] <= yhi, (who, J_["y"][0], J_["y"][-1], ylo, yhi)
    # constant rate-per-host ratio through the input (n_gal at its input, m < 18; hosts normalized to z = 5 as the exploratory runs, or to
    # the end of the production redshift grid, z = 1.10, once the production runs are drawn); h^3 Mpc^-3 densities with the volume at H0 = 100
    zm = bv.ZMAX_PROD if prod else bv.ZMAX_GRID
    NC = bv.norm_cfg(100)
    la = np.linspace(ylo, yhi, 300)
    Rin = VALUES["macros"]["MockRateRatio"]["value"]
    lgal_h, lagn_h = VALUES["macros"]["MockLognGalh"]["value"], VALUES["macros"]["MockLognAGNh"]["value"]
    zA = bv.Zk(NC["m18"]["agn"], 10 ** la, 100.0, zmax=zm)
    zG = float(bv.Zk(NC["m18"]["gal"], 10 ** lgal_h, 100.0, zmax=zm))
    odds = Rin * zA / zG
    axa.plot(odds / (1 + odds), la, color="k", lw=1.3, zorder=6)
    chk(name, ("macros", "MockRateRatio", "value"), Rin, "constant rate-per-host curve")
    chk(name, ("macros", "MockLognGalh", "value"), lgal_h, "constant rate-per-host curve: input galaxy density [h^3 Mpc^-3]")
    # the curve passes through the input point (f_AGN, log10 n_AGN) = (MockfAGN, MockLognAGNh) up to the counts of realization A
    f_at_in = float(np.interp(lagn_h, la, odds / (1 + odds)))
    note(name, f"(a) constant-rate curve at the input density {lagn_h:.3f}: f_AGN = {f_at_in:.4f} (input {IN_F}); y range [{ylo:.3f}, {yhi:.3f}] h^3 Mpc^-3; "
               f"hosts normalized to z = {zm:g}")
    axa.plot([IN_F], [lagn_h], marker="+", ms=11, mew=1.6, color=fs.INPUT, zorder=8)
    axa.set_xlim(0, 1)
    axa.set_ylim(ylo, yhi)
    axa.set_xlabel(r"$f_{\rm AGN}$")
    axa.set_ylabel(r"$\log_{10}[n_{\rm AGN}/(h^3\,{\rm Mpc^{-3}})]$")
    # labels in axes coordinates (the vertical range differs between the placeholder and the production prior box)
    ta = axa.transAxes
    fs.label(axa, 0.60, 0.98, "constant merger rate per host", "k", ha="right", va="top", transform=ta, bbox=dict(fc="white", ec="none", pad=0.5, alpha=0.8))
    fs.label(axa, 0.56, 0.14, "allowed by\nthe priors", "0.30", ha="left", va="center", transform=ta)
    c18, c21 = kcol("FreeEighteen", fs.COMP[3], 2), kcol("FreeTwentyOne", fs.COMP[0], 0)
    for k, c, txt, x0 in (("FreeEighteen", c18, "$m<18$", 0.45), ("FreeTwentyOne", c21, "$m<21$", 0.70)):
        x_, y_, P_, lev_ = joints[k]
        # just below the lower edge of the 90% contour at f_AGN = x0 (above its upper edge when the lower edge is the axis)
        cs = axa.contour(x_, y_, np.asarray(P_).T, levels=[lev_], alpha=0.0)
        seg = np.concatenate([s for s in cs.allsegs[0] if len(s)]) if any(len(s) for s in cs.allsegs[0]) else np.array([[x0, 0.5 * (ylo + yhi)]])
        cs.remove()
        near = seg[np.abs(seg[:, 0] - x0) < 0.03]
        near = near if len(near) else seg
        dy = 0.04 * (yhi - ylo)
        y_lo, y_hi = float(near[:, 1].min()), float(near[:, 1].max())
        above = y_lo - 3 * dy < ylo
        fs.label(axa, x0, y_hi + dy if above else y_lo - dy, txt, c, ha="center", va="bottom" if above else "top",
                 bbox=dict(fc="white", ec="none", pad=0.3, alpha=0.8))
    fs.panel_label(axa, label, x=0.97, y=0.04, ha="right", va="bottom")
    if not prod:
        ph_word(axa, x=0.97, y=0.13, va="bottom")
    return ylo, yhi


def panel_rate_ratio(axb, name, label="(b)"):
    """Fig. 4(b): posterior of the rate-per-host ratio at each flux limit, number densities inferred (realization A)."""
    Rin = VALUES["macros"]["MockRateRatio"]["value"]
    top = 0.0
    for i, (rk, w, mlim) in enumerate(bv.RUNGS):
        k = f"Free{w}"
        c = kcol(k, fs.COMP[i], i)
        x, p, s = draw_1d(axb, k, "RateRatio", "A", c, LS4[i])
        chk_key(name, k, "RateRatio", "A")
        top = max(top, float(p.max()))
        j = int(np.argmax(p))
    fs.input_vline(axb, Rin)
    axb.set_xlim(0, 100)
    axb.set_ylim(0, 1.25 * top)
    axb.set_xlabel(r"merger rate per AGN / merger rate per galaxy")
    axb.set_ylabel("probability density")
    for i, (rk, w, mlim) in enumerate(bv.RUNGS):
        x, p, s = q1(f"Free{w}", "RateRatio")
        yy = [0.95, 0.83, 0.71, 0.59][i] * 1.25 * top
        fs.label(axb, 72, yy, f"$m<{mlim}$", kcol(f"Free{w}", fs.COMP[i], i), ha="left")
        axb.plot([64, 70.5], [yy, yy], color=kcol(f"Free{w}", fs.COMP[i], i), ls=LS4[i], lw=fs.LW)
    fs.panel_label(axb, label)
    if not all(fin(f"Free{w}") for _, w, _ in bv.RUNGS):
        ph_word(axb, x=0.03, ha="left", y=0.86)


# ---------------------------------------------------------------- Fig. 5: an unknown population
def fig5_population():
    name = "fig5_population"
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(fs.PAGE, 2.9), layout="constrained", gridspec_kw=dict(width_ratios=[1.2, 1.0]))
    specs = [("KnownBoth", fs.BOTH, 0, "-", "population known,\nboth catalogs"), ("KnownNone", NONE_C, 3, ":", "population known,\nno catalog"),
             ("SingleBoth", SINGLE_C, 1, "--", "single population,\nboth catalogs"), ("SingleNone", "#7F7F7F", 2, "-.", "single population,\nno catalog")]
    top = 0.0
    cur = {}
    for key, fc, i, ls, lab in specs:
        c = kcol(key, fc, i)
        x, p, s = draw_1d(axa, key, "Hzero", "A", c, ls)
        chk_key(name, key, "Hzero", "A")
        cur[key] = (x, p, c, lab)
        top = max(top, float(p.max()))
    fs.input_vline(axa, IN_H0)
    axa.set_xlim(50, 90)
    axa.set_ylim(0, 1.15 * top)
    for key, xy, ha in (("KnownBoth", (70.3, 0.33), "left"), ("SingleBoth", (66.2, 0.235), "right"), ("KnownNone", (71.6, 0.16), "left"),
                        ("SingleNone", (76.0, 0.06), "left")):
        x, p, c, lab = cur[key]
        fs.label(axa, *xy, lab, c, ha=ha)
    axa.set_xlabel(fs.H0_LABEL)
    axa.set_ylabel(fs.H0_DENS)
    fs.panel_label(axa, "(a)")
    if not all(fin(k) for k, *_ in specs):
        ph_word(axa)
    box = []
    for key, i, ls in (("SingleNone", 2, "-."), ("SingleBoth", 1, "--")):
        c = kcol(key, SINGLE_C if key == "SingleBoth" else "#7F7F7F", i)
        x_, y_, P_, lev_ = draw_joint(axb, PHJ[key]["joint"]["A"], c, ls=ls, fill=(key == "SingleBoth"), alpha=0.25)
        chk_key(name, key, "MuG", "A")
        ii, jj = np.where(P_ >= lev_)
        box.append((x_[ii.min()], x_[ii.max()], y_[jj.min()], y_[jj.max()]))
        # the 90% region must close inside the stored lattice (it is clipped otherwise)
        assert 0 < ii.min() and ii.max() < len(x_) - 1 and 0 < jj.min() and jj.max() < len(y_) - 1, (key, "90% region reaches the lattice edge")
        note(name, f"(b) {key}: 90% region H0 [{box[-1][0]:.1f}, {box[-1][1]:.1f}], mu_G [{box[-1][2]:.2f}, {box[-1][3]:.2f}]; lattice H0 [{x_[0]:.1f}, {x_[-1]:.1f}], mu_G [{y_[0]:.2f}, {y_[-1]:.2f}]")
    axb.plot([IN_H0], [IN["muG"]], marker="+", ms=11, mew=1.6, color=fs.INPUT, zorder=8)
    x0b, x1b = min(b[0] for b in box), max(b[1] for b in box)
    y0b, y1b = min(b[2] for b in box), max(b[3] for b in box)
    axb.set_xlim(x0b - 0.08 * (x1b - x0b), x1b + 0.08 * (x1b - x0b))
    axb.set_ylim(y0b - 0.10 * (y1b - y0b), y1b + 0.16 * (y1b - y0b))
    # direct labels: "no catalog" above the upper-left end of its region, "both catalogs" to the right of the filled region
    bN, bB = box[0], box[1]
    fs.label(axb, bN[0] + 0.12 * (bN[1] - bN[0]), bN[3] + 0.05 * (y1b - y0b), "no catalog", kcol("SingleNone", "#7F7F7F", 2), ha="left", va="bottom")
    cB = kcol("SingleBoth", SINGLE_C, 1)
    xy_tip = (bB[1] - 0.03 * (bB[1] - bB[0]), bB[2] + 0.42 * (bB[3] - bB[2]))     # at the right-hand edge of the filled region
    axb.annotate("both catalogs", xy=xy_tip, xytext=(bB[1] + 0.12 * (x1b - x0b), bB[3] + 0.06 * (y1b - y0b)), color=cB, fontsize=fs.FS_TEXT, ha="left", va="bottom",
                 arrowprops=dict(arrowstyle="-", color=cB, lw=0.8, shrinkA=1, shrinkB=2), zorder=9)
    axb.set_xlabel(fs.H0_LABEL)
    axb.set_ylabel(r"mass-peak location $\mu_{\rm G}$ [$M_\odot$]")
    fs.panel_label(axb, "(b)")
    if not (fin("SingleBoth") and fin("SingleNone")):
        ph_word(axb)
    fs.save(fig, FIGDIR, name)


# ---------------------------------------------------------------- Figs. 6 and 7: host-dependent populations
def host_labels(ax, cur, kS, kH, cS, cH, q):
    """Labels above the two curves, the one with the lower median to the left."""
    mS = PHJ[kS]["quantities"][q]["Comb"]["median"]
    mH = PHJ[kH]["quantities"][q]["Comb"]["median"]
    y = 0.74 * ax.get_ylim()[1]
    span = ax.get_xlim()[1] - ax.get_xlim()[0]
    separated = abs(mS - mH) > 0.25 * span
    for k, m, c, lab in ((kS, mS, cS, "single\npopulation"), (kH, mH, cH, "host-dependent\npopulations")):
        if separated:                                   # each label centred above its own curve
            fs.label(ax, m, y, lab, c, ha="center", va="bottom")
            continue
        left = m < (mH if k == kS else mS)
        fs.label(ax, m + (-0.02 if left else 0.02) * span, y, lab, c, ha="right" if left else "left", va="bottom")


def fig_host(pre, name, outdir=None):
    fig, axs = plt.subplots(1, 3, figsize=(fs.PAGE, 2.6), layout="constrained", gridspec_kw=dict(width_ratios=[1.1, 1.0, 1.0]))
    kS, kH = f"{pre}Single", f"{pre}Host"
    cS, cH = kcol(kS, SINGLE_C, 1), kcol(kH, fs.BOTH, 0)
    # (a) H0
    ax = axs[0]
    top = 0.0
    cur = {}
    for k, c, ls in ((kS, cS, "--"), (kH, cH, "-")):
        x, p, s = draw_1d(ax, k, "Hzero", "Comb", c, ls, shade=False)
        chk_key(name, k, "Hzero", "Comb")
        top = max(top, float(p.max()))
        cur[k] = (x, p)
    fs.input_vline(ax, IN_H0, ymax=0.68)
    lo = min(cur[kH][0][0], cur[kS][0][0])
    hi = max(cur[kH][0][-1], cur[kS][0][-1])
    pad = 0.05 * (hi - lo)
    ax.set_xlim(lo + pad, hi - pad)
    ax.set_ylim(0, 1.55 * top)
    host_labels(ax, cur, kS, kH, cS, cH, "Hzero")
    ax.set_xlabel(fs.H0_LABEL)
    ax.set_ylabel(fs.H0_DENS)
    fs.panel_label(ax, "(a)")
    if not (fin(kS) and fin(kH)):
        ph_word(ax)
    # (b) the two offsets
    ax = axs[1]
    J = PHJ[kH]["joint"]["Comb"]
    x_, y_, P, lev = draw_joint(ax, J, cH, ls="-", fill=True, alpha=0.30)
    chk_key(name, kH, "DmuG", "Comb")
    chk_key(name, kH, "DmuChi", "Comb")
    ax.plot([IN["dmuG"]], [IN["dmuChi"]], marker="+", ms=11, mew=1.6, color=fs.INPUT, zorder=8)
    ax.plot([0.0], [0.0], marker="x", ms=7, mew=1.4, color=fs.INPUT, zorder=8)
    ax.axhline(0, color=fs.PRIOR, lw=0.7, ls=":", zorder=1)
    ax.axvline(0, color=fs.PRIOR, lw=0.7, ls=":", zorder=1)
    xr = (min(-0.8, x_[0]), max(x_[-1], 9.5))
    yr = (-0.03, max(0.2, y_[-1]))
    ax.set_xlim(*xr)
    ax.set_ylim(*yr)
    fs.label(ax, 0.97, 0.05, "+ input    $\\times$ no offset", fs.INPUT, ha="right", va="bottom", transform=ax.transAxes)
    ax.set_xlabel(r"AGN offset of the mass peak $\Delta\mu_{\rm G}$ [$M_\odot$]")
    ax.set_ylabel(r"AGN offset of the spin mean $\Delta\mu_\chi$")
    fs.panel_label(ax, "(b)")
    if not fin(kH):
        ph_word(ax)
    # (c) f_AGN
    ax = axs[2]
    top = 0.0
    for k, c, ls in ((kS, cS, "--"), (kH, cH, "-")):
        x, p, s = draw_1d(ax, k, "fAGN", "Comb", c, ls)
        chk_key(name, k, "fAGN", "Comb")
        top = max(top, float(p.max()))
        cur[k] = (x, p)
    fs.input_vline(ax, IN_F, ymax=0.68)
    ax.set_xlim(0.12, 0.45)
    ax.set_ylim(0, 1.55 * top)
    host_labels(ax, cur, kS, kH, cS, cH, "fAGN")
    ax.set_xlabel(r"$f_{\rm AGN}$")
    ax.set_ylabel(r"$p(f_{\rm AGN})$")
    fs.panel_label(ax, "(c)")
    if not (fin(kS) and fin(kH)):
        ph_word(ax)
    fs.save(fig, outdir or FIGDIR, name)


def fig6_host_dependent():
    fig_host("HD", "fig6_host_dependent")


def fig7_observed_like():
    fig_host("Obs", "fig7_observed_like")


# ---------------------------------------------------------------- Fig. 8: the summary
def results_subsections(tex=PRD / "main.tex"):
    """Titles of the subsections of Sec. IV (Results) of the long draft, in order."""
    import re as _re
    t = Path(tex).read_text()
    body = t.split("\\section{Results}", 1)[1].split("\\section{", 1)[0]
    return _re.findall(r"\\subsection\{([^}]*)\}", body)


_SEC = results_subsections()
# group headings: subsection titles of Sec. IV in the paper, looked up by name (the subsection of the constraint from each host
# class, PAPER_SKELETON Sec. 9.3, has no row of its own); rows: the analyses in words
_HEADS = ["Known population and complete catalogs", "Flux-limited catalogs", "An unknown population", "Populations that depend on the host",
          "A mass spectrum like the observed one"]
assert all(h in _SEC for h in _HEADS), (_HEADS, _SEC)
_HEADS = sorted(_HEADS, key=_SEC.index)
SUMMARY_ROWS = [(h, {
    "Known population and complete catalogs": [("KnownBoth", "both catalogs"), ("KnownGal", "galaxy catalog alone"), ("KnownAGN", "AGN catalog alone"), ("KnownNone", "no catalog")],
    "Flux-limited catalogs": [("FluxTwentyOne", "$m<21$, number densities known"), ("FluxTwenty", "$m<20$, number densities known"), ("FluxNineteen", "$m<19$, number densities known"),
                              ("FluxEighteen", "$m<18$, number densities known"), ("FreeEighteen", "$m<18$, number densities inferred")],
    "An unknown population": [("SingleBoth", "single population, both catalogs"), ("SingleNone", "single population, no catalog")],
    "Populations that depend on the host": [("HDSingle", "single population, both catalogs"), ("HDHost", "host-dependent populations, both catalogs"), ("HDNone", "single population, no catalog")],
    "A mass spectrum like the observed one": [("ObsSingle", "single population, both catalogs"), ("ObsHost", "host-dependent populations, both catalogs"), ("ObsNone", "single population, no catalog")],
}[h]) for h in _HEADS]
assert all(lab.lower() != head.lower() for head, items in SUMMARY_ROWS for _, lab in items)
# rows whose combined H0 is drawn as a 90% upper limit (a cap at the limit with an arrow to the left; no median, no interval), the
# limit read from the macro named here (values.json, unrounded). HDNone (single population, no catalog, host-dependent mergers): its
# combined posterior reaches the lower prior edge of H0 (50) with 0.45 of its peak density, so the table quotes the limit only
SUMMARY_UL = {"HDNone": "HDNoneHzeroCombUL"}
UL_ARROW = 2.5          # length of the arrow to the left of the cap, km/s/Mpc


def draw_summary(name, groups, outdir=None, fs_lab=10.0, row_h=0.255, left=0.40, h0_range=(60, 84), part="Comb", width=fs.PAGE, head_x=0.012, row_x=0.035,
                 fan=0.62, f_range=(0.0, 1.0), gap=0.03, ni_x=0.80, ni_ha="center"):
    """One row per analysis, grouped under headings: median (marker) and 90% interval (bar) of H0 (left panel) and f_AGN (right
    panel) of the posterior combined over realizations (part="Comb", every row); inputs as dashed lines."""
    rows = []
    for head, items in groups:
        rows.append(("head", head))
        rows += [("row", it) for it in items]
    n = len(rows)
    fig = plt.figure(figsize=(width, row_h * n + 0.75))
    right, bot, topm = 0.985, 0.075 * 6.36 / (row_h * n + 0.75), 0.985
    wH = fan * (right - left - gap)
    axH = fig.add_axes([left, bot, wH, topm - bot])
    axF = fig.add_axes([left + wH + gap, bot, right - left - gap - wH, topm - bot])
    y = np.arange(n)[::-1].astype(float)
    for ax in (axH, axF):
        ax.set_ylim(-0.7, n - 0.3)
        ax.set_yticks([])
        ax.tick_params(axis="y", which="both", left=False, right=False)
        ax.tick_params(axis="x", labelsize=fs_lab)
    anyph = False
    for yy, (kind, it) in zip(y, rows):
        if kind == "head":
            fig.text(head_x, axH.transData.transform((0, yy))[1] / fig.bbox.height, it, fontsize=fs_lab, fontweight="bold", va="center", ha="left")
            for ax in (axH, axF):
                ax.axhline(yy + 0.5, color="0.85", lw=0.6, zorder=0)
            continue
        key, lab = it
        c = "k" if fin(key) else RED[0]
        anyph |= not fin(key)
        fig.text(row_x, axH.transData.transform((0, yy))[1] / fig.bbox.height, lab, fontsize=fs_lab, va="center", ha="left", color="k")
        for ax, q in ((axH, "Hzero"), (axF, "fAGN")):
            if q not in PHJ[key]["quantities"]:
                ax.text(ni_x, yy, "not inferred", transform=ax.get_yaxis_transform(), color="0.45", fontsize=fs_lab - 1.5, ha=ni_ha, va="center")
                continue
            s = PHJ[key]["quantities"][q][part]
            if q == "Hzero" and part == "Comb" and key in SUMMARY_UL:
                # a 90% upper limit (SUMMARY_UL): cap at the limit, arrow to the left
                ul = float(VALUES["macros"][SUMMARY_UL[key]]["value"])
                ax.plot([ul, ul], [yy - 0.24, yy + 0.24], color=c, lw=2.0, solid_capstyle="butt", zorder=5)
                ax.annotate("", xy=(ul - UL_ARROW, yy), xytext=(ul, yy), arrowprops=dict(arrowstyle="->", color=c, lw=1.6, shrinkA=0, shrinkB=0), zorder=5)
                chk_macro(name, SUMMARY_UL[key], ul, f"90% upper limit of {key}'s combined H0 drawn as a cap")
                chk(name, ("macros", SUMMARY_UL[key], "value"), float(s["ul90"]), f"90th percentile of {key}'s combined H0 (placeholder_results.json)")
                continue
            ax.plot([s["ci90"][0], s["ci90"][1]], [yy, yy], color=c, lw=3.2, solid_capstyle="butt", zorder=4)
            ax.plot([s["median"]], [yy], marker="o", ms=6.5, color=c, mec="white", mew=0.8, zorder=5)
            chk_key(name, key, q, part)
    for ax, x0, xr, xl in ((axH, IN_H0, h0_range, fs.H0_LABEL), (axF, IN_F, f_range, r"$f_{\rm AGN}$")):
        ax.axvline(x0, color=fs.INPUT, lw=1.0, ls="--", zorder=1)
        ax.set_xlim(*xr)
        ax.set_xlabel(xl, fontsize=fs_lab + 0.5)
        # (no "placeholder" word in the panel: red marks the elements that wait for a run)
    # intervals running off the H0 axis: they end at the upper edge of the H0 prior (Table I, \PriorHzeroHi), the number printed
    hi_edge = h0_range[1]
    pri_hi = VALUES["macros"]["PriorHzeroHi"]
    for yy, (kind, it) in zip(y, rows):
        if kind == "row" and "Hzero" in PHJ[it[0]]["quantities"] and not (part == "Comb" and it[0] in SUMMARY_UL):
            s = PHJ[it[0]]["quantities"]["Hzero"][part]
            if s["ci90"][1] > hi_edge:
                axH.annotate("", xy=(hi_edge, yy), xytext=(hi_edge - 1.0, yy), arrowprops=dict(arrowstyle="->", color=RED[0] if not fin(it[0]) else "k", lw=1.6), annotation_clip=False)
                txt = unwrap_ph(pri_hi["render"])
                axH.text(hi_edge - 1.4, yy, f"at the prior\nedge, {txt}", fontsize=fs_lab - 2.0, ha="right", va="center", color="0.3")
                chk_text(name, "PriorHzeroHi", txt, f"prior edge printed beside the {it[0]} row")
                # the interval does end at that edge: its upper end, rounded as the printed number, is the printed number
                CHECKS.append((name, ("macros", "PriorHzeroHi", "render"), ("PRINTED", float(s["ci90"][1])),
                               f"upper end of the {it[0]} interval at the prior edge (at the printed precision)"))
    fs.save(fig, outdir or FIGDIR, name)
    return fig


def fig8_summary():
    draw_summary("fig8_summary", SUMMARY_ROWS)


# ---------------------------------------------------------------- Fig. 9: validation
def fig9_validation(outdir=None, stem=None):
    name = stem or "fig9_validation"
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(fs.PAGE, 2.6), layout="constrained")
    specs = [("KnownBoth", fs.BOTH, 0, "-", "population known"), ("SingleBoth", SINGLE_C, 1, "--", "single population"),
             ("HDHost", "#009E73", 2, "-.", "host-dependent populations")]
    top = 0.0
    cur = {}
    for key, fc, i, ls, lab in specs:
        c = kcol(key, fc, i)
        x, p, s = draw_1d(axa, key, "Hzero", "Comb", c, ls)
        chk_key(name, key, "Hzero", "Comb")
        cur[key] = (x, p, c, lab)
        top = max(top, float(p.max()))
    fs.input_vline(axa, IN_H0)
    axa.set_xlim(65.3, 70.7)
    axa.set_ylim(0, 1.18 * top)
    for key, xy, ha in (("KnownBoth", (68.75, 0.90), "left"), ("SingleBoth", (65.45, 0.86), "left"), ("HDHost", (65.45, 0.60), "left")):
        x, p, c, lab = cur[key]
        fs.label(axa, xy[0], xy[1] * top, lab.replace(" ", "\n", 1) if key != "KnownBoth" else lab, c, ha=ha)
    axa.set_xlabel(fs.H0_LABEL)
    axa.set_ylabel(fs.H0_DENS)
    fs.panel_label(axa, "(a)")
    if not all(fin(k) for k, *_ in specs):
        ph_word(axa)
    top = 0.0
    for key, x0, i, ls, lab in (("CtrlZero", 0.0, 0, "-", "input 0"), ("CtrlTenth", 0.10, 1, "--", "input 0.10"), ("KnownBoth", IN_F, 2, "-.", "input 0.30")):
        c = kcol(key, fs.BOTH, i)
        x, p, s = draw_1d(axb, key, "fAGN", "Comb", c, ls)
        chk_key(name, key, "fAGN", "Comb")
        axb.axvline(x0, ymax=0.82, color=fs.INPUT, lw=fs.LW_INPUT, ls="--", zorder=1)
        cur[key] = (x, p, c, lab)
        top = max(top, float(p.max()))
    axb.set_xlim(-0.005, 0.40)
    axb.set_ylim(0, 1.15 * top)
    for key, xx, yy in (("CtrlZero", 0.035, 0.86), ("CtrlTenth", 0.13, 0.55), ("KnownBoth", 0.32, 0.40)):
        x, p, c, lab = cur[key]
        fs.label(axb, xx, yy * 1.15 * top, lab, c, ha="left")
    axb.set_xlabel(r"$f_{\rm AGN}$")
    axb.set_ylabel(r"$p(f_{\rm AGN})$")
    fs.panel_label(axb, "(b)", x=0.97, ha="right")
    if not all(fin(k) for k in ("CtrlZero", "CtrlTenth", "KnownBoth")):
        ph_word(axb, y=0.87)
    fs.save(fig, outdir or FIGDIR, name)
    if stem is None and outdir is None:
        write_fig9_facts()


def write_fig9_facts():
    """The Fig. 9 block of figures/CAPTION_FACTS.md: which curves are production results, from placeholder_results.json (until 17 Nov a
    hand-written block, which still called the control sets invented after their runs had finished)."""
    allk = ("KnownBoth", "SingleBoth", "HDHost", "CtrlZero", "CtrlTenth")
    nreal = sum(fin(k) for k in allk)
    st = "all real" if nreal == len(allk) else ("all placeholder" if nreal == 0 else f"{nreal} of {len(allk)} keys real")
    L = [f"## Fig. 9 `fig9_validation` ({st}; written by the figure script from values.json and placeholder_results.json)", "",
         "- Every curve is combined over realizations (log-likelihoods summed for grids, product of kernel densities for nested runs); "
         "real curves in the final palette, placeholders in red tones (same line styles).",
         f"- (a) H0 posteriors with both catalogs: population known (solid, {mfmt('KnownBothHzeroComb')}, input at the "
         f"{mfmt('KnownBothHzeroCL', '%')} level; {key_state('KnownBoth', 'Hzero', 'Comb')}), single population (dashed, "
         f"{mfmt('SingleBothHzeroComb')}; {key_state('SingleBoth', 'Hzero', 'Comb')}), host-dependent populations on host-dependent events "
         f"(dash-dot, {mfmt('HDHostHzeroComb')}; {key_state('HDHost', 'Hzero', 'Comb')}). Axis [65.3, 70.7]. Black dashed: input "
         f"{mfmt('MockHzero')}.",
         f"- (b) f_AGN posteriors for inputs 0 (solid, {mfmt('CtrlZerofAGNComb')}, 90% upper limit {mfmt('CtrlZerofAGNCombUL')}; "
         f"{key_state('CtrlZero', 'fAGN', 'Comb')}), {unwrap_ph(VALUES['macros']['MockfAGNTenth']['render'])} (dashed, "
         f"{mfmt('CtrlTenthfAGNComb')}, input at the {mfmt('CtrlTenthfAGNCL', '%')} level, input `\\MockfAGNTenth`; "
         f"{key_state('CtrlTenth', 'fAGN', 'Comb')}) and {unwrap_ph(VALUES['macros']['MockfAGN']['render'])} (dash-dot, "
         f"{mfmt('KnownBothfAGNComb')}; {key_state('KnownBoth', 'fAGN', 'Comb')}); black dashed lines at the three inputs. "
         "Axis f_AGN [-0.005, 0.40].",
         "- The two control sets. Input 0: " + (PHJ["CtrlZero"].get("production_settings") or "placeholder") + ". Input 0.10: "
         + (PHJ["CtrlTenth"].get("production_settings") or "placeholder") + "."]
    write_facts_block("fig9_validation", "## Fig. 9 `fig9_validation`", L)


# ---------------------------------------------------------------- fig_classes: the constraint from each host class
# (notes/PAPER_SKELETON.md Sec. 9.4; numbers from scripts/classes.py through values/values.json). The archived letter's
# fig1_budget (a, b) and fig_populations (c) are the model for the layout; words and symbols are this paper's.
# host-dependent populations in orange (stated offsets) and brown (twice): reddish purple means "single population" elsewhere in the
# paper, and orange stays apart from the red tones of the placeholders
CL_FINAL = {"unknown": fs.BOTH, "known": "#000000", "pops": "#E69F00", "pops2": "#8C510A", "host": "#E69F00", "none": NONE_C, "galaxy": fs.GAL}
CL_RED = {"unknown": RED[0], "known": RED[2], "pops": RED[1], "pops2": RED[3], "host": RED[1], "none": RED[2], "galaxy": RED[0]}
CL_STYLE = {"unknown": ("-", "o"), "known": (":", "^"), "pops": ("--", "s"), "pops2": ("--", "D"), "host": ("none", "s")}
CL_DODGE = {"unknown": -0.03, "pops": -0.01, "pops2": 0.01, "known": 0.03, "host": 0.05}
CL_LINES = {"none": (0, (7, 2.5)), "galaxy": (0, (2.2, 1.6))}
AGN_DARK = "#A63F00"                     # the truly AGN-hosted mergers of (c): a darker shade of the AGN vermilion


def cl_labels():
    dG, dX = VALUES["macros"]["MockDmuG"]["render"], VALUES["macros"]["MockDmuChi"]["render"]
    return {"unknown": "host class unknown", "known": "host class known",
            "pops": rf"host-dependent populations, $\Delta\mu_{{\rm G}}={dG}\,M_\odot$, $\Delta\mu_\chi={dX}$",
            "pops2": "host-dependent populations, twice those offsets", "host": r"host-dependent populations inferred with $H_0$",
            "none": "no catalog", "galaxy": r"galaxy catalog alone, $f_{\rm AGN}=0$"}


def cl_color(series, real):
    return CL_FINAL[series] if real else CL_RED[series]


def panel_classes(ax, spec, name, tag, facts):
    """(a) or (b): mean 90% half-width of H0 over the realizations against f_AGN; bars from the smallest to the largest."""
    import classes as CLS
    P = CLS.points(spec)
    lab = cl_labels()
    handles = {}
    top = 0.0
    for key in ("unknown", "pops", "pops2", "known", "host"):
        pts = P.get(key) or []
        if not pts:
            continue
        ls, mk = CL_STYLE[key]
        allreal = all(p_["real"] for p_ in pts)
        cline = cl_color(key, allreal)
        x = np.array([p_["x"] for p_ in pts]) + CL_DODGE[key]
        ms = fs.MS if mk != "D" else fs.MS - 0.6
        if key != "host" and len(pts) > 1:
            ax.plot(x, [p_["y"] for p_ in pts], ls=ls, color=cline, lw=1.2, zorder=3)
        for xi, p_ in zip(x, pts):
            c = cl_color(key, p_["real"])
            ax.plot([xi, xi], [p_["lo"], p_["hi"]], color=c, lw=0.9, zorder=4, solid_capstyle="butt")
            if key == "host":
                ax.plot(xi, p_["y"], marker=mk, ms=fs.MS + 1.6, mfc="white", mec=c, mew=1.4, ls="none", zorder=6)
            else:
                ax.plot(xi, p_["y"], marker=mk, ms=ms, mfc=c, mec="white", mew=0.5, ls="none", zorder=5)
            chk_macro(name, p_["macro"], p_["y"], f"{tag} {key} f={p_['x']:g} point")
            chk_macro(name, p_["macro"] + "Min", p_["lo"], f"{tag} {key} f={p_['x']:g} bar bottom")
            chk_macro(name, p_["macro"] + "Max", p_["hi"], f"{tag} {key} f={p_['x']:g} bar top")
            top = max(top, p_["hi"])
        handles[key] = (Line2D([], [], ls="none", marker=mk, ms=fs.MS + 1.6, mfc="white", mec=cline, mew=1.4) if key == "host" else
                        Line2D([], [], ls=ls, lw=1.2, color=cline, marker=mk, ms=ms, mfc=cline, mec="white", mew=0.5))
        facts.append((tag, lab[key], [(p_["x"], p_["macro"], p_["real"]) for p_ in pts]))
    hl = sorted(((k, P[k][0]) for k in ("none", "galaxy") if P.get(k)), key=lambda kp: -kp[1]["y"])
    for i, (k, p_) in enumerate(hl):
        c = cl_color(k, p_["real"])
        ax.axhline(p_["y"], color=c, lw=1.0, ls=CL_LINES[k], zorder=2)
        chk_macro(name, p_["macro"], p_["y"], f"{tag} {k} line")
        fs.label(ax, 1.045, p_["y"] / 1.03, lab[k], c, ha="right", va="top", fontsize=7.4)       # below its line, at the right
        top = max(top, p_["y"])
        facts.append((tag, lab[k] + " (horizontal line)", [(None, p_["macro"], p_["real"])]))
    ax.set_ylim(0, 1.17 * top)
    ax.yaxis.set_major_locator(plt.MaxNLocator(6, steps=[1, 2, 5, 10]))
    ax.set_xlim(-0.06, 1.08)
    ax.set_xticks([0, 0.1, 0.3, 0.5, 1.0])
    ax.set_xticklabels(["0", "0.1", "0.3", "0.5", "1"])
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_xlabel(r"fraction of mergers hosted by AGN, $f_{\rm AGN}$")
    return handles


def panel_membership(axs, name, facts):
    """(c): the probability of an AGN host of each merger of realization A (host-dependent populations, f_AGN = 0.3), by
    true host class: positions alone (top), positions with masses and spins (bottom); means as triangles with their
    values; the input fraction as a dashed line."""
    import classes as CLS
    bins = np.linspace(0, 1, 41)
    data = {w: CLS.membership_draw(w) for w in ("Pos", "Pops")}
    ymax = max(np.histogram(r, bins, density=True)[0].max() for d in data.values() for r in (d["r_agn"], d["r_gal"]))
    top = 1.75 * ymax
    words = {"Pos": "positions alone", "Pops": "positions with\nmasses and spins"}
    for ax, w in zip(axs, ("Pos", "Pops")):
        d = data[w]
        cg, ca = (fs.GAL, AGN_DARK) if d["real"] else (RED[1], RED[2])
        hg, _ = np.histogram(d["r_gal"], bins, density=True)
        ha, _ = np.histogram(d["r_agn"], bins, density=True)
        ax.stairs(hg, bins, fill=True, color=cg, alpha=0.35 if d["real"] else 0.40, lw=0, zorder=2)
        ax.stairs(ha, bins, color=ca, lw=1.5, zorder=4)
        ax.axvline(d["f_in"], color=fs.INPUT, lw=fs.LW_INPUT, ls=(0, (4, 2.5)), zorder=5)
        chk(name, ("macros", "MockfAGN", "value"), d["f_in"], "(c) input fraction line")
        ax.set_ylim(0, top)
        printed = []
        for mval, c, ha_, dx, mac in ((d["mean_gal"], cg, "right", -0.025, f"ProbGal{w}"), (d["mean_agn"], ca, "left", 0.025, f"ProbAGN{w}")):
            ax.plot(mval, 0.89 * top, marker="v", ms=5.0, color=c, mec="white", mew=0.4, zorder=6, ls="none")
            txt = f"{mval:.2f}"
            ax.text(mval + dx, 0.89 * top, txt, color=c, ha=ha_, va="center", fontsize=7.0)
            chk_macro(name, mac, mval, f"(c) {w} mean triangle")
            chk_text(name, mac, txt, f"(c) {w} mean")
            printed.append((txt, mac))
        fs.panel_label(ax, words[w], x=0.97, y=0.76, ha="right", va="top", fontsize=7.2, linespacing=1.0)
        ax.set_xlim(0, 1)
        ax.yaxis.set_major_locator(plt.MaxNLocator(3, integer=True))
        ax.xaxis.set_major_locator(plt.MultipleLocator(0.25))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.tick_params(axis="y", labelsize=fs.FS_TICK - 0.5)
        facts.append(("(c)", words[w].replace("\n", " "), d, printed))
    axs[0].text(data["Pos"]["f_in"], 1.02, r"$f_{\rm AGN}$", transform=axs[0].get_xaxis_transform(), ha="center", va="bottom", fontsize=7.4, color=fs.INPUT)
    axs[0].tick_params(labelbottom=False)
    axs[1].set_xlabel("probability of an AGN host")
    fs.panel_label(axs[0], "(c)", x=0.03, y=0.97)
    return data


def fig_classes(outdir=None):
    name = "fig_classes"
    W, H = fs.PAGE, 3.45
    Tm, Bm = 0.62, 0.42                                  # top (legend) and bottom margins [in]
    fig = plt.figure(figsize=(W, H))
    bot, hgt = Bm / H, (H - Tm - Bm) / H
    xa, wa, xb, wb = 0.47, 2.02, 2.74, 2.02
    xc = xb + wb + 0.52
    wc = W - xc - 0.06
    axa = fig.add_axes([xa / W, bot, wa / W, hgt])
    axb = fig.add_axes([xb / W, bot, wb / W, hgt])
    gap = 0.07 / H
    hc = (hgt - gap) / 2
    axc1 = fig.add_axes([xc / W, bot + hc + gap, wc / W, hc])
    axc2 = fig.add_axes([xc / W, bot, wc / W, hc], sharex=axc1)
    facts = []
    ha_ = panel_classes(axa, "peak", name, "(a)", facts)
    hb_ = panel_classes(axb, "obslike", name, "(b)", facts)
    axa.set_ylabel(r"90% half-width of $H_0$ [km s$^{-1}$ Mpc$^{-1}$]")
    fs.panel_label(axa, "(a) peak-dominated spectrum", x=0.03, y=0.97, fontsize=8.5)
    fs.panel_label(axb, "(b) observed-like spectrum", x=0.03, y=0.97, fontsize=8.5)
    data = panel_membership((axc1, axc2), name, facts)
    fig.text((xc - 0.34) / W, bot + hgt / 2, "probability density", rotation=90, ha="center", va="center", fontsize=fs.FS_LABEL)
    lab = cl_labels()
    hs = {**hb_, **ha_}
    order = [k for k in ("unknown", "known", "pops", "pops2", "host") if k in hs]
    fig.legend([hs[k] for k in order], [lab[k] for k in order], loc="upper center", ncol=2, fontsize=7.4,
               bbox_to_anchor=((xa + (xb + wb - xa) / 2) / W, 1.0), handlelength=2.6, columnspacing=1.6, labelspacing=0.32,
               handletextpad=0.5, borderaxespad=0.25)
    for mac in ("MockDmuG", "MockDmuChi"):
        chk_text(name, mac, VALUES["macros"][mac]["render"], "legend offset")
    # the key of (c), above it in the top margin
    real = data["Pos"]["real"]
    cg, ca = (fs.GAL, AGN_DARK) if real else (RED[1], RED[2])
    fig.legend([Patch(facecolor=cg, alpha=0.35 if real else 0.40, lw=0), Line2D([], [], color=ca, lw=1.5)],
               ["truly hosted by galaxies", "truly hosted by AGN"], loc="upper center", bbox_to_anchor=((xc + wc / 2) / W, 1.0),
               fontsize=7.4, handlelength=1.5, labelspacing=0.32, borderaxespad=0.25, handletextpad=0.5)
    fs.save(fig, outdir or FIGDIR, name)
    write_classes_facts(facts, data)


CF_BEGIN, CF_END = "<!-- fig_classes: written by scripts/make_figures.py; do not edit -->", "<!-- end fig_classes -->"


def write_classes_facts(facts, data):
    """The fig_classes section of figures/CAPTION_FACTS.md (between the two markers; inserted before Fig. 3 the first time)."""
    import classes as CLS
    V = VALUES["macros"]
    fmt = lambda m: f"`\\{m}` = {unwrap_ph(V[m]['render'])} ({'real' if V[m]['class'] != 'ph' else 'red'})"
    nreal = V["NrealClasses"]["value"]
    L = [CF_BEGIN, "## Fig. `fig_classes` (Sec. IV B, after `fig2_anchor`; written by the figure script from values.json)", "",
         f"- Width 7.0 in, height 3.45 in. (a), (b): the mean over the {nreal} realizations of the experiment (`\\NrealClasses`; realizations "
         f"{V['NrealClasses']['extra']['realizations']}) of the 90% half-width of H0 of each realization (its run summary), against the input "
         "f_AGN; bars from the smallest to the largest half-width over those realizations; population known, complete catalogs, 1000 detected "
         "mergers. Points of one f are spread by up to +-0.05 in f so that they do not overlap. Horizontal lines: the mean half-width (no bar). "
         "Final colours: host class unknown bluish green (circles, solid), host class known black (triangles, dotted), host-dependent populations "
         "orange (squares, dashed; twice the offsets: brown diamonds, dashed; inferred with H0: open orange square), no catalog dark gray "
         "(long dashes, labelled below the line), galaxy catalog alone blue (short dashes, labelled below the line). An element whose macro is red is drawn in red tones (same marker and line style); "
         "a connecting line is red when any of its points is. No word in a panel marks a placeholder.",
         "- (a) peak-dominated spectrum, equal populations except the host-dependent series (AGN-hosted mergers heavier by \\MockDmuG Msun, spin mean "
         "higher by \\MockDmuChi). (b) observed-like spectrum, the same analyses at the fractions that were run. The open square is the "
         "host-dependent populations analysis with the populations and H0 inferred (keys HDHost, ObsHost: \\NrealHD and \\NrealObs realizations).",
         "- (c) realization A, host-dependent populations, input f_AGN = 0.3: histograms (40 bins on [0, 1], each class normalized to unit area) of "
         "the probability of an AGN host of each of the 1000 mergers at the posterior median of its grid, f L_AGN / (f L_AGN + (1 - f) L_gal); "
         "galaxy-hosted filled blue, AGN-hosted dark vermilion line; top: positions alone (one population held at its input), bottom: positions "
         "with masses and spins (the two populations at their inputs); triangles at the means with their values printed; black dashed line at "
         "the input f_AGN = 0.3 (`\\MockfAGN`). The key of (c) sits above it in the top margin; the key of (a, b) above them.", "",
         "Series, points and their macros (each point: mean `\\HW...`, bar `...Min` to `...Max`):"]
    for row in facts:
        if row[0] in ("(a)", "(b)"):
            tag, lab_, pts = row
            nr = sum(1 for _, _, r in pts if r)
            L.append(f"- {tag} {lab_}: {len(pts)} point{'s' if len(pts) != 1 else ''} ({nr} real, {len(pts) - nr} red): "
                     + "; ".join(f"{'f=' + format(x, 'g') + ' ' if x is not None else ''}{fmt(m)} "
                                 + (f"[{unwrap_ph(V[m + 'Min']['render'])}, {unwrap_ph(V[m + 'Max']['render'])}]" if x is not None else
                                    f"(range over the realizations {unwrap_ph(V[m + 'Min']['render'])} to {unwrap_ph(V[m + 'Max']['render'])}, not drawn)")
                                 for x, m, _ in pts))
    for row in facts:
        if row[0] == "(c)":
            tag, lab_, d, printed = row
            info = printed[0][1][len("ProbGal"):]
            L.append(f"- (c) {lab_}: {'real' if d['real'] else 'red (placeholder shape: ' + d['how'] + ')'}; {len(d['r_agn'])} AGN-hosted and "
                     f"{len(d['r_gal'])} galaxy-hosted mergers; printed means: " + ", ".join(f"{t} = {fmt(m)}" for t, m in printed)
                     + f"; evaluated at f_AGN = {fmt('ProbEvalfAGN' + info)} (the run's posterior median, where a merger with equal class "
                       f"likelihoods lies; not drawn: the dashed line is the input); mean over all the mergers {fmt('ProbAll' + info)}")
    L += ["- Numbers printed in the panels: the four means of (c) above; the legend's offsets 5 (`\\MockDmuG`) and 0.10 (`\\MockDmuChi`). "
          "The figure check compares every drawn point, bar end and line with its macro (value and printed text) and the printed text with "
          "the macro's text.", CF_END]
    cf = FIGDIR / "CAPTION_FACTS.md"
    t = cf.read_text()
    block = "\n".join(L)
    if CF_BEGIN in t:
        t = t.split(CF_BEGIN)[0] + block + t.split(CF_END, 1)[1]
    else:
        i = t.index("## Fig. 3 ")
        t = t[:i] + block + "\n\n" + t[i:]
    cf.write_text(t)


# ============================================================================================= #
FIGS = {"fig1": fig1_mocks, "fig2": fig2_anchor, "fig3": fig3_flux_limits, "fig4": fig4_densities, "fig5": fig5_population,
        "fig6": fig6_host_dependent, "fig7": fig7_observed_like, "fig8": fig8_summary, "fig9": fig9_validation, "classes": fig_classes}
NAMES = {"fig1": "fig1_mocks", "fig2": "fig2_anchor", "fig3": "fig3_flux_limits", "fig4": "fig4_densities", "fig5": "fig5_population",
         "fig6": "fig6_host_dependent", "fig7": "fig7_observed_like", "fig8": "fig8_summary", "fig9": "fig9_validation", "classes": "fig_classes"}
# the first-draft figures (functions above, kept to reproduce figures/v1/; not drawn by default)
V1_FIGS = {"fig2_two_catalogs": fig2_two_catalogs, "fig3_host_information": fig3_host_information, "fig4_host_dependent_h0": fig4_host_dependent_h0,
           "fig5_offsets": fig5_offsets, "fig6_flux_limits": fig6_flux_limits, "fig7_free_densities": fig7_free_densities,
           "fig8_validation": fig8_validation, "figC1_corner": figC1_corner}


# black input lines and markers drawn in each figure, against the input macros of values.json
INPUT_MACROS = {"H0": "MockHzero", "f": "MockfAGN", "lognGal": "MockLognGal", "lognAGN": "MockLognAGN", "muG": "MockMuG",
                "dmuG": "MockDmuG", "muChi": "MockMuChi", "dmuChi": "MockDmuChi", "sigmaG": "MockSigmaG", "sigmaChi": "MockSigmaChi",
                "lognGalh": "MockLognGalh", "lognAGNh": "MockLognAGNh"}
# the inputs as drawn: IN, with the number densities also in h^3 Mpc^-3 (log10 n - 3 log10 h, h = H0/100), the unit of Fig. 4(a)
INPUT_VALUES = dict(IN, lognGalh=IN["lognGal"] - 3 * np.log10(IN["H0"] / 100.0), lognAGNh=IN["lognAGN"] - 3 * np.log10(IN["H0"] / 100.0))
INPUTS_DRAWN = {"fig1": ["muG", "dmuG", "muChi", "dmuChi"], "fig2": ["H0", "f"], "fig3": ["H0", "f"], "fig4": ["lognAGNh", "lognGalh", "f"], "fig5": ["H0", "muG"],
                "fig6": ["H0", "f", "dmuG", "dmuChi"], "fig7": ["H0", "f", "dmuG", "dmuChi"], "fig8": ["H0", "f"], "fig9": ["H0", "f"], "classes": ["f"]}


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
            chk(NAMES[w], ("macros", INPUT_MACROS[k], "value"), INPUT_VALUES[k], "input line")
    bad = run_checks([NAMES[w] for w in which])
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
