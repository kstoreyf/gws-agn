#!/usr/bin/env python
"""Numbers chain of the PRD paper: read the analysis result files, write every number the paper quotes.

Writes (all under working/paper_prd/):
    values/results_macros.tex   the LaTeX macros (three classes, as in the lensing drafts)
    values/values.json          per macro: value(s), class, units, meaning, source, claim, containment record
    values/MACROS.md            the writer's menu
    placeholders.txt            every red macro, what it means and what run produces it

Run with no arguments:
    /hildafs/home/magana/tmp_ondemand_hildafs_phy230014p_symlink/magana/.conda/envs/jax/bin/python \
        working/paper_prd/scripts/build_values.py

Classes
    plain   design choices and inputs of the simulated catalogs (working/data/**)
    anc     teal: a real number from the exploratory runs under working/analyses/ (production replaces it)
    ph      red: not computable from what is on disk; placeholders.txt says what run produces it

Statistics (DESIGN.md section 5)
    * Grids: the stored log-likelihood is refined by a tensor-product cubic spline (scipy CubicSpline,
      not-a-knot, one axis at a time) onto a finer lattice, exponentiated, and integrated with the
      trapezoid rule. Marginals, medians, equal-tailed 68/90% intervals, correlations and evidences are
      read from the refined lattice. Priors are flat on the stated box. Cells the selection guard
      rejected (or non-finite cells) take the value of the nearest accepted cell before the spline and
      are excluded (-inf) in the unrefined cross-check; every such cell lies far in the tails. Each
      refined point is capped at 5 nats above the largest of its stored cell corners, which removes
      spline ringing next to steep far-tail drops (the f = 1 edge, rejected corners); the cap acts only
      on points at least 90 nats below the maximum (diagnostics.grid_log in values.json records both).
      Validated: on Analysis 8's 0.01-spaced f grid subsampled to 0.05, the refined quantiles agree with
      the dense ones to 2e-4, while the plain trapezoid on the coarse grid widens the 90% interval by 15%.
    * Nested-sampling runs: the stored equal-weight posterior samples (the same samples the run's JSON
      summary was computed from); quantiles by numpy.percentile; log-evidence as recorded.
    * Two-sided credible level of an input x0 in a 1-D marginal: |2 F(x0) - 1|, the level of the
      smallest equal-tailed interval that contains x0.
    * Savage-Dickey ratios and grid evidences are computed on the refined lattice; the difference to the
      same quantity on the unrefined lattice is carried as the numerical uncertainty.

Nothing is read from working/archive/. For the flux-limited analyses only c_mode=selection results.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import h5py
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.special import logsumexp

# --------------------------------------------------------------------------------------------- #
# paths
# --------------------------------------------------------------------------------------------- #
W = Path("/hildafs/projects/phy230014p/magana/gws-agn/working")
AN = W / "analyses"
DATA = W / "data"
OUT = W / "paper_prd"
A0 = AN / "analysis_0_pure_tracer_H0" / "results"
A1 = AN / "analysis_1_complete_catalog_H0" / "results"
A2 = AN / "analysis_2_complete_catalog_H0_fagn" / "results"
A3 = AN / "analysis_3_incomplete_catalog_H0_fagn" / "results"
A4 = AN / "analysis_4_density_anchoring_H0_fagn" / "results"
A5 = AN / "analysis_5_free_anchors_H0_fagn" / "results"
A8 = AN / "analysis_8_spin_marked_fagn_fixed_H0" / "results"
A9 = AN / "analysis_9_spin_marked_H0_fagn" / "results"
A10 = AN / "analysis_10_mass_spin_marked_multitracer" / "results"
A11 = AN / "analysis_11_free_common_population" / "results"
A12 = AN / "analysis_12_shared_width_robustness" / "results"
A13 = AN / "analysis_13_joint_shared_widths" / "results"
SR = AN / "selection_redo"
CORE_EQ = AN / "experiments" / "experiment_darksirens_core_equivalence" / "results"

LN10 = math.log(10.0)


def rel(p) -> str:
    """Path relative to working/ for the record."""
    p = Path(p)
    try:
        return str(p.relative_to(W))
    except ValueError:
        return str(p)


def jload(p):
    return json.loads(Path(p).read_text())


# --------------------------------------------------------------------------------------------- #
# inputs of the simulated catalogs (read from the data files, see mock section)
# --------------------------------------------------------------------------------------------- #
META100 = DATA / "seed100" / "META.json"


def _inputs():
    m = jload(META100)
    ev = m["stages"]["events"]
    hd = jload(DATA / "seed100" / "events" / "events_meta_marked_dmu0p10_dmuG5.json")
    return {
        "H0": float(ev["cosmology"]["H0"]),
        "f": float(ev["planted_f_agn"]),
        "lognGal": math.log10(m["config"]["glass"]["n_comoving_gal"]),
        "lognAGN": math.log10(m["config"]["glass"]["n_comoving_agn"]),
        "muG": float(ev["population"]["peak_mu"]),
        "sigmaG": float(ev["population"]["peak_sigma"]),
        "muChi": float(ev["population"]["chi_mu"]),
        "sigmaChi": float(ev["population"]["chi_sigma"]),
        "dmuG": float(hd["branch_mass"]["dmu_G_agn"]),
        "dmuChi": float(hd["branch_spin"]["dmu_chi_agn"]),
    }


IN = _inputs()

# --------------------------------------------------------------------------------------------- #
# grid machinery
# --------------------------------------------------------------------------------------------- #


def refine_axis(x, factor, extra=()):
    x = np.asarray(x, float)
    if len(x) == 1 or factor <= 1:
        pts = [x]
    else:
        pts = [x] + [np.linspace(x[i], x[i + 1], factor + 1)[1:-1] for i in range(len(x) - 1)]
    e = np.array([v for v in extra if x[0] <= v <= x[-1]], float)
    return np.unique(np.concatenate(pts + [e]))


def trap_w(x, lo=None, hi=None):
    """Trapezoid weights on x, optionally restricted to [lo, hi] (lo/hi must be nodes of x)."""
    x = np.asarray(x, float)
    w = np.zeros_like(x)
    if len(x) == 1:
        return np.ones(1)
    sel = np.ones(len(x), bool)
    if lo is not None:
        sel &= x >= lo - 1e-12
    if hi is not None:
        sel &= x <= hi + 1e-12
    idx = np.where(sel)[0]
    if len(idx) < 2:
        return w
    xs = x[idx]
    d = np.diff(xs)
    ws = np.zeros_like(xs)
    ws[:-1] += d / 2
    ws[1:] += d / 2
    w[idx] = ws
    return w


def spline_nd(arr, axes, fine):
    a = arr
    for k, (x, xf) in enumerate(zip(axes, fine)):
        if len(x) < 2 or len(xf) == len(x) and np.array_equal(xf, x):
            continue
        a = CubicSpline(x, a, axis=k)(xf)
    return a


ENV_DELTA = 5.0   # nats: a refined point may exceed the largest of its stored cell corners by at most this
GRIDLOG: list[dict] = []


def corner_envelope(M, axes, fine):
    """At every refined point, the largest stored value among the 2^d corners of its stored cell."""
    idxs = []
    for k, (x, xf) in enumerate(zip(axes, fine)):
        if len(x) < 2:
            idxs.append(np.zeros(len(xf), int))
            continue
        M = np.maximum(np.take(M, np.arange(len(x) - 1), axis=k), np.take(M, np.arange(1, len(x)), axis=k))
        idxs.append(np.clip(np.searchsorted(x, xf, side="right") - 1, 0, len(x) - 2))
    return M[np.ix_(*idxs)]


class GridPost:
    """Flat-prior posterior on a stored log-likelihood lattice, refined by a cubic spline.

    Guard-rejected (or non-finite) cells take the value of the nearest accepted cell (index space), so
    they leave no cliff for the spline to ring from; they all lie in far tails (counted in GRIDLOG).
    After the spline every refined point is capped at ENV_DELTA nats above the largest of its stored
    cell corners: this removes spline overshoot next to steep far-tail drops (e.g. the f = 1 edge)
    without touching a genuine sub-grid peak, whose bulge above the nodes is below 1 nat on every grid
    here (GRIDLOG records the largest cap applied and the overshoot of the maximum)."""

    def __init__(self, names, axes, logL, refine, extra=None, rejected=None, tag=""):
        from scipy import ndimage
        self.tag = tag
        self.names = list(names)
        self.axes = [np.asarray(a, float) for a in axes]
        L = np.array(logL, float)
        assert L.shape == tuple(len(a) for a in self.axes), (tag, L.shape)
        bad = ~np.isfinite(L)
        if rejected is not None:
            bad |= np.asarray(rejected, bool)
        self.n_bad = int(bad.sum())
        self.n_cells = int(L.size)
        Lmax = float(np.max(L[~bad]))
        if bad.any():
            ind = ndimage.distance_transform_edt(bad, return_distances=False, return_indices=True)
            filled = L[tuple(ind)]
        else:
            filled = L.copy()
        extra = extra or {}
        if isinstance(refine, int):
            refine = [refine] * len(self.axes)
        self.fine = [refine_axis(a, r, extra.get(n, ())) for a, r, n in zip(self.axes, refine, self.names)]
        Lf = spline_nd(filled, self.axes, self.fine)
        env = corner_envelope(filled, self.axes, self.fine) + ENV_DELTA
        capped = Lf > env
        self.n_capped = int(np.sum(capped))
        self.max_cap = float(np.max(Lf - env)) if self.n_capped else 0.0
        # how far below the stored maximum the capped points' own (capped) values lie
        self.cap_depth = float(Lmax - np.max(env[capped])) if self.n_capped else None
        self.Lf = np.minimum(Lf, env)
        del env
        self.Lraw = np.where(bad, -np.inf, L)
        self.lmax = float(self.Lf.max())
        iflat = np.unravel_index(np.argmax(self.Lf), self.Lf.shape)
        jflat = np.unravel_index(np.argmax(np.where(bad, -np.inf, L)), L.shape)
        GRIDLOG.append({"grid": tag, "cells": self.n_cells, "rejected_or_nonfinite": self.n_bad, "refined_points_capped": self.n_capped, "largest_cap_nats": self.max_cap, "capped_points_depth_below_max_nats": self.cap_depth,
                        "cap_max_minus_stored_max": float(self.Lf.max() - Lmax),
                        "spline_max_minus_stored_max": self.lmax - Lmax,
                        "refined_argmax": [float(x[i]) for x, i in zip(self.fine, iflat)],
                        "stored_argmax": [float(x[i]) for x, i in zip(self.axes, jflat)],
                        "axes_with_constant_likelihood": [n for k, n in enumerate(self.names) if L.ndim > 0 and len(self.axes[k]) > 1
                                                          and float(np.nanmax(np.ptp(np.where(bad, np.nan, L), axis=k))) < 1e-8],
                        "argmax_within_one_stored_cell": bool(all(x[max(j - 1, 0)] - 1e-12 <= xf[i] <= x[min(j + 1, len(x) - 1)] + 1e-12
                                                                  for k, (x, xf, i, j) in enumerate(zip(self.axes, self.fine, iflat, jflat))
                                                                  if not (len(x) > 1 and float(np.nanmax(np.ptp(np.where(bad, np.nan, L), axis=k))) < 1e-8)))})
        self.P = np.exp(self.Lf - self.lmax)
        self.w = [trap_w(x) for x in self.fine]
        self._m1 = {}

    # ---- integration ---------------------------------------------------------------------- #
    def _contract(self, P, weights, keep):
        m = P
        for j in sorted([j for j in range(P.ndim) if j not in keep], reverse=True):
            m = np.tensordot(m, weights[j], axes=([j], [0]))
        return m

    def idx(self, name):
        return self.names.index(name)

    def marg1(self, name):
        if name not in self._m1:
            k = self.idx(name)
            m = self._contract(self.P, self.w, (k,))
            Z = float(np.dot(m, self.w[k]))
            self._m1[name] = (self.fine[k], m / Z)
        return self._m1[name]

    def marg2(self, n1, n2):
        k1, k2 = self.idx(n1), self.idx(n2)
        m = self._contract(self.P, self.w, (k1, k2))
        if k1 > k2:
            m = m.T
        Z = float(self.w[k1] @ m @ self.w[k2])
        return self.fine[k1], self.fine[k2], m / Z

    def summary(self, name):
        x, p = self.marg1(name)
        k = self.idx(name)
        w = self.w[k]
        cdf = np.concatenate([[0.0], np.cumsum(0.5 * (p[1:] + p[:-1]) * np.diff(x))])
        cdf /= cdf[-1]
        q = lambda u: float(np.interp(u, cdf, x))
        mean = float(np.dot(x * p, w))
        sd = float(math.sqrt(max(np.dot((x - mean) ** 2 * p, w), 0.0)))
        peak = float(p.max())
        return {
            "median": q(0.5), "ci68": [q(0.16), q(0.84)], "ci90": [q(0.05), q(0.95)],
            "mean": mean, "sd": sd, "map_1d": float(x[np.argmax(p)]),
            "edge_density_lo": float(p[0] / peak), "edge_density_hi": float(p[-1] / peak),
            "_cdf": (x, cdf),
        }

    def corr(self, n1, n2):
        x1, x2, m = self.marg2(n1, n2)
        w1, w2 = self.w[self.idx(n1)], self.w[self.idx(n2)]
        p1 = m @ w2
        p2 = w1 @ m
        e1, e2 = float(np.dot(x1 * p1, w1)), float(np.dot(x2 * p2, w2))
        v1 = float(np.dot((x1 - e1) ** 2 * p1, w1))
        v2 = float(np.dot((x2 - e2) ** 2 * p2, w2))
        c = float(w1 @ (np.outer(x1 - e1, x2 - e2) * m) @ w2)
        return c / math.sqrt(v1 * v2)

    def density_at(self, name, x0):
        """Normalised marginal density at x0 (x0 must be a node of the refined axis)."""
        x, p = self.marg1(name)
        i = np.where(np.isclose(x, x0, atol=1e-12, rtol=0))[0]
        if len(i) == 0:
            raise ValueError(f"{self.tag}: {x0} not a refined node of {name}")
        return float(p[i[0]])

    def hpd_level(self, n1, n2, x1, x2):
        """Posterior mass in the 2-D region of density above the density at (x1, x2)."""
        a1, a2, m = self.marg2(n1, n2)
        i = int(np.argmin(np.abs(a1 - x1)))
        j = int(np.argmin(np.abs(a2 - x2)))
        if abs(a1[i] - x1) > 1e-9 or abs(a2[j] - x2) > 1e-9:
            raise ValueError(f"{self.tag}: input not on the refined lattice")
        dens = m[i, j]
        Wt = np.outer(self.w[self.idx(n1)], self.w[self.idx(n2)])
        return float(np.sum((m * Wt)[m > dens]) / np.sum(m * Wt))

    # ---- evidences ------------------------------------------------------------------------ #
    def log_int(self, box=None):
        """ln of the integral of L over the lattice (optionally restricted to box: name -> (lo, hi));
        returns (refined value, unrefined trapezoid value)."""
        box = box or {}
        wf, wr = [], []
        for n, xf, x in zip(self.names, self.fine, self.axes):
            lo, hi = box.get(n, (None, None))
            wf.append(trap_w(xf, lo, hi))
            wr.append(trap_w(x, lo, hi))
        m = self.P
        for j in reversed(range(m.ndim)):
            m = np.tensordot(m, wf[j], axes=([j], [0]))
        refined = self.lmax + math.log(float(m))
        Wr = wr[0]
        for v in wr[1:]:
            Wr = np.multiply.outer(Wr, v)
        with np.errstate(divide="ignore"):
            raw = float(logsumexp(self.Lraw + np.log(Wr)))
        return refined, raw


def level_from_cdf(cdf_pair, x0):
    x, cdf = cdf_pair
    if x0 < x[0]:
        return 1.0, 0.0
    if x0 > x[-1]:
        return 1.0, 1.0
    F = float(np.interp(x0, x, cdf))
    return abs(2 * F - 1), F


# --------------------------------------------------------------------------------------------- #
# chain machinery
# --------------------------------------------------------------------------------------------- #


class ChainPost:
    def __init__(self, samples, names, tag=""):
        self.s = np.asarray(samples, float)
        self.names = list(names)
        self.tag = tag

    def col(self, n):
        return self.s[:, self.names.index(n)]

    def summary(self, n):
        x = self.col(n)
        q = np.percentile(x, [5, 16, 50, 84, 95])
        return {"median": float(q[2]), "ci68": [float(q[1]), float(q[3])], "ci90": [float(q[0]), float(q[4])],
                "mean": float(np.mean(x)), "sd": float(np.std(x)), "_samples": x}

    def corr(self, a, b):
        return float(np.corrcoef(self.col(a), self.col(b))[0, 1])


def chain_level(x, x0):
    F = float(np.mean(x < x0))
    return abs(2 * F - 1), F


# --------------------------------------------------------------------------------------------- #
# formatting
# --------------------------------------------------------------------------------------------- #


def dec_for(err, sig=2):
    if err <= 0 or not np.isfinite(err):
        return 2
    return int(max(0, min(5, sig - 1 - math.floor(math.log10(err)))))


def fnum(x, d):
    s = f"{x:.{d}f}"
    if s.startswith("-") and float(s) == 0.0:
        s = s[1:]
    return s


def fci(med, lo, hi, d=None):
    up, dn = hi - med, med - lo
    if d is None:
        d = dec_for(min(up, dn) if min(up, dn) > 0 else max(up, dn))
    return f"{fnum(med, d)}^{{+{fnum(up, d)}}}_{{-{fnum(dn, d)}}}", d


def fsci(x, d=1):
    e = int(math.floor(math.log10(abs(x))))
    m = x / 10 ** e
    if round(m, d) >= 10:
        m /= 10
        e += 1
    return f"{m:.{d}f}\\times10^{{{e}}}"


def fpct(frac):
    """Percent with 0 decimals (1 below 10%); near 100% enough decimals to stay below 100, and
    '>99.99' beyond that."""
    v = 100 * frac
    if frac >= 1:
        return "100"
    if v >= 99.5:
        gap = 100 - v
        if gap < 0.01:
            return "\\ensuremath{>}99.99"
        d = int(math.ceil(-math.log10(gap)))
        s = fnum(v, d)
        while float(s) >= 100:
            d += 1
            s = fnum(v, d)
        return s
    return fnum(v, 0) if v >= 10 else fnum(v, 1)


def flb(x, err=None):
    d = 1 if abs(x) >= 10 else 2
    s = fnum(x, d)
    if err is not None:
        s += f"\\pm{fnum(err, max(d, dec_for(err, 1)))}"
    return s


# --------------------------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------------------------- #

REG: list[dict] = []
DIAG: dict = {}
TABLES: dict = {}
BAYES: list[dict] = []
NAME_RE = re.compile(r"^[A-Za-z]+$")


def add(name, cls, render, value, units, meaning, source, claim, containment=None, extra=None):
    assert NAME_RE.match(name), name
    assert cls in ("plain", "anc", "ph"), cls
    assert all(r["name"] != name for r in REG), f"duplicate macro {name}"
    REG.append({"name": name, "class": cls, "render": render, "value": value, "units": units,
                "meaning": meaning, "source": source, "claim": claim,
                "containment": containment, "extra": extra or {}})


def containment(s, x0, kind="grid"):
    if x0 is None:
        return None
    if kind == "grid":
        lev, F = level_from_cdf(s["_cdf"], x0)
    else:
        lev, F = chain_level(s["_samples"], x0)
    return {"input": x0, "in68": bool(s["ci68"][0] <= x0 <= s["ci68"][1]),
            "in90": bool(s["ci90"][0] <= x0 <= s["ci90"][1]), "level_two_sided": lev, "cdf_at_input": F}


def clean(s):
    return {k: v for k, v in s.items() if not k.startswith("_")}


def add_post(name, s, units, meaning, source, claim, x0=None, kind="grid", parts=False, d=None, cls="anc"):
    r, d = fci(s["median"], s["ci90"][0], s["ci90"][1], d)
    cont = containment(s, x0, kind)
    add(name, cls, r, clean(s), units, meaning + " (median and 90% equal-tailed interval)", source, claim,
        containment=cont, extra={"decimals": d})
    if parts:
        add(name + "Med", cls, fnum(s["median"], d), s["median"], units, "median of " + meaning, source, claim)
        add(name + "Lo", cls, fnum(s["ci90"][0], d), s["ci90"][0], units, "lower end of the 90% interval of " + meaning, source, claim)
        add(name + "Hi", cls, fnum(s["ci90"][1], d), s["ci90"][1], units, "upper end of the 90% interval of " + meaning, source, claim)
    return s


def add_num(name, x, d, units, meaning, source, claim, cls="anc"):
    add(name, cls, fnum(x, d), x, units, meaning, source, claim)


def add_ph(name, render, units, meaning, run, claim):
    add(name, "ph", render, None, units, meaning, {"missing": run}, claim)


def hw90(s):
    return 0.5 * (s["ci90"][1] - s["ci90"][0])


def hw68(s):
    return 0.5 * (s["ci68"][1] - s["ci68"][0])


def w90(s):
    return s["ci90"][1] - s["ci90"][0]


def src_grid(path, keys, how="spline-refined flat-prior grid posterior (GridPost)"):
    return {"file": rel(path), "keys": keys, "computation": how}


CAT_LETTERS = "ABCDEFGHI"
RUNGS = [("m21", "TwentyOne", 21), ("m20", "Twenty", 20), ("m19", "Nineteen", 19), ("m18", "Eighteen", 18)]

# --------------------------------------------------------------------------------------------- #
# loaders
# --------------------------------------------------------------------------------------------- #


def load_hf(path, H0key="H0_grid", fkey="f_grid", Lkey="log_likelihood", rejkey="guard/rejected"):
    with h5py.File(path, "r") as h:
        H = h[H0key][()]
        F = h[fkey][()]
        L = h[Lkey][()]
        rej = h[rejkey][()] if rejkey in h else None
    return H, F, L, rej


def hf_post(path, tag, refH=4, reff=8, extra=None, **kw):
    H, F, L, rej = load_hf(path, **kw)
    ex = {"H0": [IN["H0"]], "f": [IN["f"]]}
    if extra:
        ex.update(extra)
    return GridPost(["H0", "f"], [H, F], L, [refH, reff], extra=ex, rejected=rej, tag=tag)


# ============================================================================================= #
# MOCK DESCRIPTION (plain)
# ============================================================================================= #


def build_mock():
    m = jload(META100)
    src = lambda key: {"file": rel(META100), "key": key}
    ev = m["stages"]["events"]
    cat = m["stages"]["catalogs"]
    sur = m["stages"]["surveys"]
    add("MockHzero", "plain", fnum(IN["H0"], 2), IN["H0"], "km/s/Mpc", "input Hubble constant", src("stages.events.cosmology.H0"), "mock")
    add("MockOmegam", "plain", fnum(ev["cosmology"]["Om0"], 4), ev["cosmology"]["Om0"], "", "input matter density Omega_m (fixed in every analysis)", src("stages.events.cosmology.Om0"), "mock")
    add("MockfAGN", "plain", fnum(IN["f"], 2), IN["f"], "", "input fraction of mergers hosted by AGN", src("stages.events.planted_f_agn"), "mock")
    rf = ev["realised"]["realised_f_agn"]
    add("MockfAGNRealised", "plain", fnum(rf, 3), rf, "", "realised AGN-hosted fraction of the 1000 detected events of the reference catalog", src("stages.events.realised.realised_f_agn"), "mock")
    ng, na = m["config"]["glass"]["n_comoving_gal"], m["config"]["glass"]["n_comoving_agn"]
    add("MocknGal", "plain", f"10^{{{int(round(math.log10(ng)))}}}", ng, "Mpc^-3", "comoving number density of the galaxy catalog", src("config.glass.n_comoving_gal"), "mock")
    add("MocknAGN", "plain", f"10^{{{int(round(math.log10(na)))}}}", na, "Mpc^-3", "comoving number density of the AGN catalog", src("config.glass.n_comoving_agn"), "mock")
    add("MockLognGal", "plain", fnum(IN["lognGal"], 1), IN["lognGal"], "log10 Mpc^-3", "log10 of the galaxy number density (input of the free-abundance analysis)", src("config.glass.n_comoving_gal"), "mock")
    add("MockLognAGN", "plain", fnum(IN["lognAGN"], 1), IN["lognAGN"], "log10 Mpc^-3", "log10 of the AGN number density", src("config.glass.n_comoving_agn"), "mock")
    add("MockbGal", "plain", fnum(m["config"]["glass"]["bias_gal"], 1), m["config"]["glass"]["bias_gal"], "", "linear bias of the galaxies", src("config.glass.bias_gal"), "mock")
    add("MockbAGN", "plain", fnum(m["config"]["glass"]["bias_agn"], 1), m["config"]["glass"]["bias_agn"], "", "linear bias of the AGN", src("config.glass.bias_agn"), "mock")
    xc = cat["magnitude_model"]["x_cut_L_over_Lstar"]
    add("MockLcut", "plain", fnum(xc, 2), xc, "L*", "luminosity cut of the galaxy sample, L > x L* (B-band Schechter function)", src("stages.catalogs.magnitude_model.x_cut_L_over_Lstar"), "mock")
    mb = cat["magnitude_model"]["M_B_faint_limit"]
    add("MockMBcut", "plain", fnum(mb, 2), mb, "mag", "absolute B magnitude of the galaxy cut", src("stages.catalogs.magnitude_model.M_B_faint_limit"), "mock")
    lx = 43.7
    add("MockLXcut", "plain", "43.7", lx, "log10 erg/s", "AGN class: log10 L_X(2-10 keV) above this value (pinned literature density)", src("config.agn_lf.definition (text: 'log10 L_X(2-10 keV) >~ 43.7 erg/s')"), "mock")
    Ng, Na = cat["tracers"]["gal"]["n"], cat["tracers"]["agn"]["n"]
    add("MockNGal", "plain", fsci(Ng, 1), Ng, "objects", "realised galaxies to z = 1 (151,179,870)", src("stages.catalogs.tracers.gal.n"), "mock")
    add("MockNAGN", "plain", fsci(Na, 1), Na, "objects", "realised AGN to z = 1 (1,514,567)", src("stages.catalogs.tracers.agn.n"), "mock")
    zc = m["config"]["glass"]["z_max"]
    add("MockzCat", "plain", fnum(zc, 0), zc, "", "redshift depth of the catalogs", src("config.glass.z_max"), "mock")
    dz = sur["dz_scale"]
    add("MockDz", "plain", fnum(dz, 3), dz, "", "catalog redshift error sigma_z = this value times (1+z)", src("stages.surveys.dz_scale"), "mock")
    add("MockNside", "plain", str(sur["nside"]), sur["nside"], "", "HEALPix nside of the catalog pixelisation", src("stages.surveys.nside"), "mock")
    add("MockNev", "plain", str(ev["n_events"]), ev["n_events"], "events", "detected mergers per simulated catalog", src("stages.events.n_events"), "mock")
    add("MockNsamp", "plain", str(ev["nsamp"]), ev["nsamp"], "samples", "posterior samples per event", src("stages.events.nsamp"), "mock")
    thr = ev["detection"]["snr_threshold"]
    add("MockSNRthr", "plain", fnum(thr, 0), thr, "", "detection threshold on the observed signal-to-noise ratio", src("stages.events.detection.snr_threshold"), "mock")
    hz = ev["realised"]["horizon_z_max_detected"]
    add("MockzHorizon", "plain", fnum(hz, 2), hz, "", "largest true redshift of a detected event, reference catalog (the horizon)", src("stages.events.realised.horizon_z_max_detected"), "mock")
    zmed = ev["realised"]["z_median_detected"]
    add("MockzMedian", "plain", fnum(zmed, 2), zmed, "", "median redshift of the detected events, reference catalog", src("stages.events.realised.z_median_detected"), "mock")
    # horizon range over the catalogs of the complete-catalog joint analysis
    hz9 = {s: jload(DATA / f"seed{s}" / "events" / "events_meta.json")["realised"]["horizon_z_max_detected"] for s in C1_SEEDS}
    add("MockzHorizonMin", "plain", fnum(min(hz9.values()), 2), min(hz9.values()), "", "smallest horizon over the 9 catalogs of the joint analysis", {"files": [rel(DATA / f"seed{s}/events/events_meta.json") for s in C1_SEEDS], "key": "realised.horizon_z_max_detected", "computation": "min"}, "mock")
    add("MockzHorizonMax", "plain", fnum(max(hz9.values()), 2), max(hz9.values()), "", "largest horizon over the 9 catalogs of the joint analysis", {"files": [rel(DATA / f"seed{s}/events/events_meta.json") for s in C1_SEEDS], "key": "realised.horizon_z_max_detected", "computation": "max"}, "mock")
    v2 = m["stages"]["validation"]["checks"]["V2_widths_from_observed_snr"]
    add("MockSkyMin", "plain", fnum(v2["sigma_ang_deg_min"], 1), v2["sigma_ang_deg_min"], "deg", "smallest one-sigma sky localisation width, reference catalog", src("stages.validation.checks.V2_widths_from_observed_snr.sigma_ang_deg_min"), "mock")
    add("MockSkyMax", "plain", fnum(v2["sigma_ang_deg_max"], 1), v2["sigma_ang_deg_max"], "deg", "largest one-sigma sky localisation width, reference catalog", src("stages.validation.checks.V2_widths_from_observed_snr.sigma_ang_deg_max"), "mock")
    mm = ev["measurement_model"]
    add("MockSigLnMc", "plain", "0.08", 0.08, "", "chirp-mass error: sigma(ln Mc) = this value times 8/rho_obs", src("stages.events.measurement_model.widths.sigma_lnMc ('A_MC = 0.08')"), "mock")
    add("MockSigLnq", "plain", "0.60", 0.60, "", "mass-ratio error: sigma(ln q) = this value times 8/rho_obs", src("stages.events.measurement_model.widths.sigma_lnq ('A_Q = 0.6')"), "mock")
    add("MockSigChi", "plain", "0.20", 0.20, "", "effective-spin error: sigma(chi_eff) = this value times 8/rho_obs", src("stages.events.measurement_model.widths.sigma_chieff ('A_CHI = 0.2')"), "mock")
    assert "A_MC = 0.08" in mm["widths"]["sigma_lnMc"] and "A_Q = 0.6" in mm["widths"]["sigma_lnq"] and "A_CHI = 0.2" in mm["widths"]["sigma_chieff"]
    # flux limits and completeness inside the horizon (galaxy catalog; AGN alongside in values.json)
    comp = sur["completeness"]
    for key, word, mlim in RUNGS:
        add(f"MockMlim{word}", "plain", str(mlim), float(mlim), "mag", f"apparent-magnitude limit m < {mlim}", src("stages.surveys.magnitude_limits"), "mock,C4")
        cg = comp[key]["gal"]["C_within_horizon"]
        ca = comp[key]["agn"]["C_within_horizon"]
        add(f"MockComp{word}", "plain", fnum(100 * cg, 1), cg, "%", f"completeness of the galaxy catalog inside the horizon at m < {mlim} (AGN catalog: {100*ca:.1f}%)",
            src(f"stages.surveys.completeness.{key}.gal.C_within_horizon"), "mock,C4", extra={"agn_C_within_horizon": ca,
            "n_within_horizon_kept_gal": comp[key]["gal"]["n_within_horizon_kept"], "n_within_horizon_total_gal": comp[key]["gal"]["n_within_horizon_total"]})
    p = ev["population"]
    add("MockAlpha", "plain", fnum(p["alpha"], 1), p["alpha"], "", "power-law index of the primary mass", src("stages.events.population.alpha"), "mock")
    add("MockMmin", "plain", fnum(p["mmin"], 0), p["mmin"], "Msun", "minimum mass", src("stages.events.population.mmin"), "mock")
    add("MockMmax", "plain", fnum(p["mmax"], 0), p["mmax"], "Msun", "maximum mass", src("stages.events.population.mmax"), "mock")
    add("MockPeakFrac", "plain", fnum(p["peak_fraction"], 2), p["peak_fraction"], "", "fraction of the primary-mass distribution in the Gaussian peak", src("stages.events.population.peak_fraction"), "mock")
    add("MockMuG", "plain", fnum(p["peak_mu"], 0), p["peak_mu"], "Msun", "location of the Gaussian mass peak (galaxy hosts; both host classes in the common-population mocks)", src("stages.events.population.peak_mu"), "mock")
    add("MockSigmaG", "plain", fnum(p["peak_sigma"], 0), p["peak_sigma"], "Msun", "width of the Gaussian mass peak", src("stages.events.population.peak_sigma"), "mock")
    add("MockBeta", "plain", fnum(p["beta"], 0), p["beta"], "", "power-law index of the mass ratio", src("stages.events.population.beta"), "mock")
    add("MockMuChi", "plain", fnum(p["chi_mu"], 0), p["chi_mu"], "", "mean effective spin (galaxy hosts)", src("stages.events.population.chi_mu"), "mock")
    add("MockSigmaChi", "plain", fnum(p["chi_sigma"], 1), p["chi_sigma"], "", "width of the effective-spin distribution", src("stages.events.population.chi_sigma"), "mock")
    add("MockGammaRate", "plain", fnum(ev["gamma"], 0), ev["gamma"], "", "redshift evolution index of the merger rate, (1+z)^gamma", src("stages.events.gamma"), "mock")
    hdp = DATA / "seed100" / "events" / "events_meta_marked_dmu0p10_dmuG5.json"
    hd = jload(hdp)
    add("MockDmuG", "plain", fnum(IN["dmuG"], 0), IN["dmuG"], "Msun", "input offset of the AGN mass-peak location (host-dependent mock)", {"file": rel(hdp), "key": "branch_mass.dmu_G_agn"}, "mock,C7,C8")
    add("MockMuGAGN", "plain", fnum(hd["branch_mass"]["mu_G_agn"], 0), hd["branch_mass"]["mu_G_agn"], "Msun", "AGN mass-peak location in the host-dependent mock", {"file": rel(hdp), "key": "branch_mass.mu_G_agn"}, "mock,C7,C8")
    add("MockDmuChi", "plain", fnum(IN["dmuChi"], 2), IN["dmuChi"], "", "input offset of the AGN mean effective spin", {"file": rel(hdp), "key": "branch_spin.dmu_chi_agn"}, "mock,C7,C8,C9")
    r = hd["realised"]
    add("MockHDNagn", "plain", str(r["n_host_agn"]), r["n_host_agn"], "events", "AGN-hosted events among the 1000 detected in the host-dependent mock", {"file": rel(hdp), "key": "realised.n_host_agn"}, "mock,C7")
    add("MockHDfDet", "plain", fnum(r["realised_f_agn"], 3), r["realised_f_agn"], "", "AGN-hosted fraction of the detected events in the host-dependent mock (heavier AGN mergers are louder; input 0.30)", {"file": rel(hdp), "key": "realised.realised_f_agn"}, "mock,C7")
    sp = DATA / "seed100" / "events" / "events_meta_marked_dmu0p10.json"
    add("MockSpinNagn", "plain", str(jload(sp)["realised"]["n_host_agn"]), jload(sp)["realised"]["n_host_agn"], "events", "AGN-hosted events in the spin-offset-only mock (Analyses 8, 9)", {"file": rel(sp), "key": "realised.n_host_agn"}, "mock,C9")
    bp = DATA / "seed100" / "events" / "events_meta_single_b0p30_dmu0p10_dmuG5.json"
    b = jload(bp)
    add("MockCtrlBranchFrac", "plain", fnum(b["branch_decoupled_from_host"]["branch_frac_independent"], 2), b["branch_decoupled_from_host"]["branch_frac_independent"], "",
        "fraction of mergers drawn from the offset population in the galaxy-only control mock (all hosts galaxies)", {"file": rel(bp), "key": "branch_decoupled_from_host.branch_frac_independent"}, "mock,C7")
    DIAG["mock_seed_realised_fagn"] = {s: jload(DATA / f"seed{s}" / "events" / "events_meta.json")["realised"]["realised_f_agn"] for s in C1_SEEDS}


# ============================================================================================= #
# C1, C2: complete catalogs, joint (H0, f_AGN)
# ============================================================================================= #
C1_SEEDS = [100, 101, 102, 103, 105, 114, 115, 116, 117]


def build_c1_c2():
    posts, Ls = {}, []
    H = F = None
    for s in C1_SEEDS:
        p = A2 / f"joint_s{s}.h5"
        H_, F_, L, rej = load_hf(p)
        assert rej.sum() == 0
        if H is None:
            H, F = H_, F_
        assert np.array_equal(H, H_) and np.array_equal(F, F_)
        Ls.append(L)
        posts[s] = GridPost(["H0", "f"], [H, F], L, [4, 10], extra={"H0": [IN["H0"]], "f": [IN["f"]]}, tag=f"joint_s{s}")
    # partial chunk sets, excluded
    chunks = sorted((A2 / "chunks").glob("joint_s*_c*.h5"))
    cnt = {}
    for c in chunks:
        t = re.match(r"joint_(s\d+(?:_popuni)?)_c\d+\.h5", c.name).group(1)
        cnt[t] = cnt.get(t, 0) + 1
    DIAG["c1_chunk_counts"] = {k: v for k, v in sorted(cnt.items())}
    partial = sorted(k for k, v in cnt.items() if v < 8)
    add("JointNcat", "anc", str(len(C1_SEEDS)), len(C1_SEEDS), "catalogs", "simulated catalogs with a complete joint (H0, f_AGN) grid on complete catalogs",
        {"files": [rel(A2 / f"joint_s{s}.h5") for s in C1_SEEDS], "computation": f"count of merged 201x41 grids; partial chunk sets {partial} (fewer than 8 of 8 H0 chunks) excluded"}, "C1",
        extra={"seeds": C1_SEEDS, "letters": dict(zip(CAT_LETTERS, C1_SEEDS)), "partial_chunk_sets": {k: cnt[k] for k in partial}})
    rows, rhos = [], []
    for i, s in enumerate(C1_SEEDS):
        g = posts[s]
        L = CAT_LETTERS[i]
        src = src_grid(A2 / f"joint_s{s}.h5", ["H0_grid", "f_grid", "log_likelihood"])
        sH, sf = g.summary("H0"), g.summary("f")
        rho = g.corr("H0", "f")
        rhos.append(rho)
        ref = (i == 0)
        add_post(f"JointHzeroCat{L}", sH, "km/s/Mpc", f"H0, complete catalogs, joint (H0, f_AGN) fit, catalog {L} (seed {s})", src, "C1", IN["H0"], parts=ref)
        add_post(f"JointfCat{L}", sf, "", f"f_AGN, complete catalogs, joint fit, catalog {L} (seed {s})", src, "C1", IN["f"], parts=ref)
        add_num(f"JointRhoCat{L}", rho, 2, "", f"correlation coefficient of H0 and f_AGN, catalog {L} (seed {s})", src, "C1")
        rows.append({"letter": L, "seed": s, "H0": clean(sH), "f": clean(sf), "rho": rho,
                     "stored_json_H0_median": jload(A2 / f"joint_s{s}.json")["H0"]["median"],
                     "stored_json_f_median": jload(A2 / f"joint_s{s}.json")["f"]["median"]})
    TABLES["C1_per_catalog"] = rows
    srcs = {"files": [rel(A2 / f"joint_s{s}.h5") for s in C1_SEEDS], "keys": ["log_likelihood"], "computation": "per-catalog correlation from the refined 2-D posterior"}
    add_num("JointRhoMean", float(np.mean(rhos)), 2, "", "mean over the 9 catalogs of the H0-f_AGN correlation coefficient", srcs, "C1")
    add_num("JointRhoMin", float(np.min(rhos)), 2, "", "smallest H0-f_AGN correlation coefficient over the 9 catalogs", srcs, "C1")
    add_num("JointRhoMax", float(np.max(rhos)), 2, "", "largest H0-f_AGN correlation coefficient over the 9 catalogs", srcs, "C1")
    # combined posterior
    Lsum = np.sum(Ls, axis=0)
    gc = GridPost(["H0", "f"], [H, F], Lsum, [4, 10], extra={"H0": [IN["H0"]], "f": [IN["f"]]}, tag="joint_combined")
    srcc = {"files": [rel(A2 / f"joint_s{s}.h5") for s in C1_SEEDS], "keys": ["log_likelihood"],
            "computation": "sum of the 9 log-likelihood grids on the common 201x41 (H0, f) lattice, H0 and f shared; flat prior H0 in [50,100], f in [0,1]; spline-refined"}
    sH, sf = gc.summary("H0"), gc.summary("f")
    add_post("JointHzeroComb", sH, "km/s/Mpc", "H0, combined posterior of the 9 complete-catalog joint analyses", srcc, "C1", IN["H0"], parts=True)
    add_post("JointfComb", sf, "", "f_AGN, combined posterior of the 9 complete-catalog joint analyses", srcc, "C1", IN["f"], parts=True)
    cH = containment(sH, IN["H0"])
    cf = containment(sf, IN["f"])
    add("JointHzeroCombLevel", "anc", fpct(cH["level_two_sided"]), cH["level_two_sided"], "%", "two-sided credible level at which the input H0 = 67.74 lies in the combined complete-catalog posterior", srcc, "C1,validation")
    add("JointfCombLevel", "anc", fpct(cf["level_two_sided"]), cf["level_two_sided"], "%", "two-sided credible level at which the input f_AGN = 0.30 lies in the combined complete-catalog posterior", srcc, "C1,validation")
    hpd = gc.hpd_level("H0", "f", IN["H0"], IN["f"])
    add("JointCombLevelTwoD", "anc", fpct(hpd), hpd, "%", "credible level of the highest-density 2-D region of the combined (H0, f_AGN) posterior whose boundary passes through the input point", srcc, "C1,validation")
    add_num("JointRhoComb", gc.corr("H0", "f"), 2, "", "H0-f_AGN correlation coefficient of the combined posterior", srcc, "C1")
    # numerical accuracy: MC variance of ln L at the per-catalog maximum
    v = []
    for s in C1_SEEDS:
        with h5py.File(A2 / f"joint_s{s}.h5") as h:
            L_ = h["log_likelihood"][()]
            i = np.unravel_index(np.argmax(L_), L_.shape)
            v.append(float(h["guard/sigma2_total"][()][i]))
    add_num("JointLnLVarMax", max(v), 1, "", "largest over the 9 catalogs of the Monte Carlo variance of ln L (selection plus event terms) at the grid maximum",
            {"files": [rel(A2 / f"joint_s{s}.h5") for s in C1_SEEDS], "keys": ["guard/sigma2_total at argmax(log_likelihood)"], "computation": "max over catalogs"}, "C1,appendix")
    add_num("JointLnLVarMin", min(v), 1, "", "smallest such variance over the 9 catalogs", {"files": [rel(A2 / f"joint_s{s}.h5") for s in C1_SEEDS], "keys": ["guard/sigma2_total at argmax"]}, "C1,appendix")
    DIAG["c1_sigma2_total_at_max"] = dict(zip(C1_SEEDS, v))

    # ---------------- C2: one catalog at a time, from the f = 0 and f = 1 columns -------------- #
    # equality check against Analysis 1's stored single-catalog scans of the reference catalog
    with h5py.File(A1 / "h0_gal_targeted.h5") as h:
        Lg = h["log_likelihood"][()]
    with h5py.File(A1 / "h0_agn_targeted.h5") as h:
        La = h["log_likelihood"][()]
    DIAG["c2_column_identity"] = {"max_abs_diff_f0_vs_h0_gal_targeted": float(np.max(np.abs(Ls[0][:, 0] - Lg))),
                                  "max_abs_diff_f1_vs_h0_agn_targeted": float(np.max(np.abs(Ls[0][:, -1] - La)))}
    gal_rows, agn_rows, lbz, lbo = [], [], [], []
    n_nointerior = 0
    for i, s in enumerate(C1_SEEDS):
        L = CAT_LETTERS[i]
        Lj = Ls[i]
        src = src_grid(A2 / f"joint_s{s}.h5", ["log_likelihood[:, f=0]"],
                       "f = 0 column of the joint grid (bitwise equal to the galaxy-catalog-only scan; checked against analysis_1 h0_gal_targeted for seed 100)")
        gG = GridPost(["H0"], [H], Lj[:, 0], [8], extra={"H0": [IN["H0"]]}, tag=f"galonly_s{s}")
        sG = gG.summary("H0")
        add_post(f"GalOnlyHzeroCat{L}", sG, "km/s/Mpc", f"H0 from all 1000 events with the galaxy catalog alone, catalog {L} (seed {s})", src, "C2", IN["H0"], parts=(i == 0))
        gal_rows.append({"letter": L, "seed": s, "H0": clean(sG)})
        gA = GridPost(["H0"], [H], Lj[:, -1], [8], extra={"H0": [IN["H0"]]}, tag=f"agnonly_s{s}")
        sA = gA.summary("H0")
        imax = int(np.argmax(Lj[:, -1]))
        interior = 0 < imax < len(H) - 1
        n_nointerior += (not interior)
        agn_rows.append({"letter": L, "seed": s, "argmax_H0": float(H[imax]), "interior_maximum": bool(interior), "H0": clean(sA)})
        # Savage-Dickey at f = 0 and f = 1 (flat prior on f in [0,1], density 1): B = 1 / p(f0 | d)
        g = posts[s]
        tot_ref, tot_raw = g.log_int()
        f0_ref = gG.log_int()
        f0_raw = gG.log_int()[1]
        f1_ref = gA.log_int()
        lb0 = (tot_ref - f0_ref[0]) / LN10
        lb0_raw = (tot_raw - f0_ref[1]) / LN10
        lb1 = (tot_ref - f1_ref[0]) / LN10
        lb1_raw = (tot_raw - f1_ref[1]) / LN10
        lbz.append(lb0)
        lbo.append(lb1)
        add(f"LBfZeroCat{L}", "anc", flb(lb0), lb0, "log10", f"log10 Bayes factor of the two-population model against galaxy hosts only, Savage-Dickey at f_AGN = 0, catalog {L} (seed {s})",
            src_grid(A2 / f"joint_s{s}.h5", ["log_likelihood"], "log10[ int L(H0,f) dH0 df / int L(H0,0) dH0 ], flat priors H0 [50,100], f [0,1]"), "C2",
            extra={"unrefined_trapezoid": lb0_raw, "numerical_diff": lb0 - lb0_raw})
        add(f"LBfOneCat{L}", "anc", flb(lb1), lb1, "log10", f"log10 Bayes factor of the two-population model against AGN hosts only, Savage-Dickey at f_AGN = 1, catalog {L} (seed {s})",
            src_grid(A2 / f"joint_s{s}.h5", ["log_likelihood"], "log10[ int L dH0 df / int L(H0,1) dH0 ]"), "C2",
            extra={"unrefined_trapezoid": lb1_raw, "numerical_diff": lb1 - lb1_raw})
    TABLES["C2_galaxy_only"] = gal_rows
    TABLES["C2_agn_only"] = agn_rows
    add("AgnOnlyNoInterior", "anc", str(n_nointerior), n_nointerior, "catalogs",
        "number of the 9 catalogs whose AGN-catalog-only H0 posterior (all events) has no interior maximum in [50, 100] (log-likelihood largest at a grid edge)",
        {"files": [rel(A2 / f"joint_s{s}.h5") for s in C1_SEEDS], "keys": ["log_likelihood[:, f=1]"], "computation": "argmax at the first or last H0 node"}, "C2",
        extra={"argmax_H0_per_catalog": {r["seed"]: r["argmax_H0"] for r in agn_rows}})
    for i, r in enumerate(agn_rows):
        if r["interior_maximum"]:
            gA = GridPost(["H0"], [H], Ls[i][:, -1], [8], extra={"H0": [IN["H0"]]}, tag=f"agnonly_interior_s{r['seed']}")
            add_post(f"AgnOnlyHzeroCat{r['letter']}", gA.summary("H0"), "km/s/Mpc",
                     f"H0 from all 1000 events with the AGN catalog alone, catalog {r['letter']} (seed {r['seed']}), the one catalog whose AGN-only posterior has an interior maximum",
                     src_grid(A2 / f"joint_s{r['seed']}.h5", ["log_likelihood[:, f=1]"], "f = 1 column of the joint grid"), "C2", IN["H0"])
    rA = agn_rows[0]
    add("AgnOnlyArgmaxCatA", "anc", fnum(rA["argmax_H0"], 1), rA["argmax_H0"], "km/s/Mpc", "H0 at which the AGN-catalog-only log-likelihood (all events, catalog A) is largest; 100 is the top of the scanned range",
        src_grid(A2 / "joint_s100.h5", ["H0_grid", "log_likelihood[:, f=1]"], "argmax"), "C2")
    srcs = {"files": [rel(A2 / f"joint_s{s}.h5") for s in C1_SEEDS], "computation": "per-catalog Savage-Dickey ratios"}
    add_num("LBfZeroMin", min(lbz), 1, "log10", "smallest per-catalog log10 B (two populations against galaxy hosts only) over the 9 catalogs", srcs, "C2")
    add_num("LBfZeroMax", max(lbz), 1, "log10", "largest per-catalog log10 B (two populations against galaxy hosts only)", srcs, "C2")
    # combined
    gGc = GridPost(["H0"], [H], Lsum[:, 0], [8], extra={"H0": [IN["H0"]]}, tag="galonly_combined")
    gAc = GridPost(["H0"], [H], Lsum[:, -1], [8], extra={"H0": [IN["H0"]]}, tag="agnonly_combined")
    sGc = gGc.summary("H0")
    add_post("GalOnlyHzeroComb", sGc, "km/s/Mpc", "H0 with the galaxy catalog alone (all events), combined over the 9 catalogs", dict(srcc, computation="sum over catalogs of the f = 0 columns"), "C2", IN["H0"], parts=True)
    cG = containment(sGc, IN["H0"])
    add("GalOnlyHzeroCombLevel", "anc", fpct(cG["level_two_sided"]), cG["level_two_sided"], "%", "two-sided credible level of the input H0 in the combined galaxy-catalog-only posterior", srcc, "C2")
    tr, trr = gc.log_int()
    z0 = gGc.log_int()
    z1 = gAc.log_int()
    lb0c, lb0c_raw = (tr - z0[0]) / LN10, (trr - z0[1]) / LN10
    lb1c, lb1c_raw = (tr - z1[0]) / LN10, (trr - z1[1]) / LN10
    add("LBfZeroComb", "anc", flb(lb0c), lb0c, "log10", "log10 Bayes factor of the two-population model against galaxy hosts only, combined over the 9 catalogs (Savage-Dickey at f_AGN = 0 on the summed grid)",
        dict(srcc, computation="Savage-Dickey on the summed log-likelihood grid"), "C2", extra={"unrefined_trapezoid": lb0c_raw})
    add("LBfOneComb", "anc", flb(lb1c), lb1c, "log10", "log10 Bayes factor of the two-population model against AGN hosts only, combined over the 9 catalogs (Savage-Dickey at f_AGN = 1)",
        dict(srcc, computation="Savage-Dickey on the summed log-likelihood grid"), "C2", extra={"unrefined_trapezoid": lb1c_raw})
    BAYES.append({"pair": "two-population (H0, f free) vs galaxy hosts only (f = 0)", "qualifies": True, "how": "Savage-Dickey at f = 0 on each joint grid and on the summed grid; flat prior on f [0,1]", "macros": ["LBfZeroCat*", "LBfZeroComb"]})
    BAYES.append({"pair": "two-population vs AGN hosts only (f = 1)", "qualifies": True, "how": "Savage-Dickey at f = 1", "macros": ["LBfOneCat*", "LBfOneComb"]})
    return posts, H, F


# ============================================================================================= #
# C3: pure-tracer event sets
# ============================================================================================= #


def build_c3():
    ens = jload(A0 / "h0_pure_tracer_ens93.json")
    seeds = list(ens["seeds_requested"])
    on_disk = sorted(int(re.search(r"_s(\d+)\.h5", p.name).group(1)) for p in A0.glob("h0_pureagn_targeted_s*.h5"))
    DIAG["c3_seeds_on_disk_not_in_ensemble"] = sorted(set(on_disk) - set(seeds))
    rows = {"gal": [], "agn": []}
    sums = {}
    H = None
    for tr in ("gal", "agn"):
        acc = None
        for s in seeds:
            p = A0 / f"h0_pure{tr}_targeted_s{s}.h5"
            with h5py.File(p) as h:
                H_ = h["H0_grid"][()]
                L = h["log_likelihood"][()]
                rej = h["guard/rejected"][()]
            assert rej.sum() == 0
            if H is None:
                H = H_
            assert np.array_equal(H, H_)
            g = GridPost(["H0"], [H], L, [8], extra={"H0": [IN["H0"]]}, tag=f"pure{tr}_s{s}")
            sm = g.summary("H0")
            rows[tr].append({"seed": s, "median": sm["median"], "ci68": sm["ci68"], "ci90": sm["ci90"],
                             "half68": hw68(sm), "half90": hw90(sm), "level_two_sided": level_from_cdf(sm["_cdf"], IN["H0"])[0]})
            acc = L if acc is None else acc + L
        sums[tr] = acc
    TABLES["C3_pure_tracer_per_catalog"] = rows
    n = len(seeds)
    src = {"files": f"{rel(A0)}/h0_pure{{gal,agn}}_targeted_s<seed>.h5 for the {n} seeds listed in {rel(A0 / 'h0_pure_tracer_ens93.json')}:seeds_requested",
           "keys": ["H0_grid", "log_likelihood"], "computation": "spline-refined 1-D H0 posteriors per catalog"}
    add("PureNcat", "anc", str(n), n, "catalogs", "simulated catalogs with both a 1000-event all-galaxy-host set and a 1000-event all-AGN-host set analysed", src, "C3",
        extra={"seeds": seeds, "excluded_on_disk": DIAG["c3_seeds_on_disk_not_in_ensemble"]})
    mg68 = float(np.mean([r["half68"] for r in rows["gal"]]))
    ma68 = float(np.mean([r["half68"] for r in rows["agn"]]))
    mg90 = float(np.mean([r["half90"] for r in rows["gal"]]))
    ma90 = float(np.mean([r["half90"] for r in rows["agn"]]))
    add_num("PureGalHalfSixtyEight", mg68, 2, "km/s/Mpc", "mean over catalogs of the 68% half-width of H0 from 1000 galaxy-hosted mergers", src, "C3")
    add_num("PureAGNHalfSixtyEight", ma68, 2, "km/s/Mpc", "mean over catalogs of the 68% half-width of H0 from 1000 AGN-hosted mergers", src, "C3")
    add_num("PureGalHalfNinety", mg90, 2, "km/s/Mpc", "mean over catalogs of the 90% half-width of H0, galaxy-hosted set", src, "C3")
    add_num("PureAGNHalfNinety", ma90, 2, "km/s/Mpc", "mean over catalogs of the 90% half-width of H0, AGN-hosted set", src, "C3")
    add_num("PureRatioSixtyEight", mg68 / ma68, 1, "", "ratio of the mean 68% half-widths, galaxy-hosted over AGN-hosted", src, "C3")
    add_num("PureRatioNinety", mg90 / ma90, 1, "", "ratio of the mean 90% half-widths, galaxy-hosted over AGN-hosted", src, "C3")
    for tr, word in (("gal", "Gal"), ("agn", "AGN")):
        g = GridPost(["H0"], [H], sums[tr], [16], extra={"H0": [IN["H0"]]}, tag=f"pure{tr}_combined")
        sm = g.summary("H0")
        srcc = dict(src, computation=f"sum of the {n} H0 log-likelihood curves on the common 201-node grid [50,100]; spline-refined x16")
        add_post(f"Pure{word}HzeroComb", sm, "km/s/Mpc", f"H0, combined posterior of the {n} {'galaxy' if tr == 'gal' else 'AGN'}-hosted event sets", srcc, "C3,validation", IN["H0"], parts=True)
        c = containment(sm, IN["H0"])
        add(f"Pure{word}HzeroCombLevel", "anc", fpct(c["level_two_sided"]), c["level_two_sided"], "%", f"two-sided credible level of the input H0 in the combined {'galaxy' if tr=='gal' else 'AGN'}-hosted posterior", srcc, "C3,validation",
            extra={"combined_sd": sm["sd"], "grid_spacing": 0.25})
    from scipy.interpolate import Akima1DInterpolator
    xf = np.linspace(H[0], H[-1], (len(H) - 1) * 64 + 1)
    rob = {}
    for nm, fn in (("cubic", lambda: CubicSpline(H, sums["agn"])(xf)), ("akima", lambda: Akima1DInterpolator(H, sums["agn"])(xf))):
        Lf = fn()
        p = np.exp(Lf - Lf.max())
        c = np.concatenate([[0.0], np.cumsum(0.5 * (p[1:] + p[:-1]) * np.diff(xf))])
        c /= c[-1]
        rob[nm] = {"q05_q50_q95": [float(np.interp(u, c, xf)) for u in (0.05, 0.5, 0.95)], "level_of_input": abs(2 * float(np.interp(IN["H0"], xf, c)) - 1)}
    i = int(np.argmax(sums["agn"]))
    a, b, _ = np.polyfit(H[i - 2:i + 3], sums["agn"][i - 2:i + 3], 2)
    rob["local_quadratic_5_nodes"] = {"peak": float(-b / (2 * a)), "sd": float(np.sqrt(-1 / (2 * a)))}
    DIAG["c3_agn_combined_interpolant_check"] = rob
    DIAG["c3_stored_ensemble_mean_half68"] = {"gal": ens["closure_gal"]["widths"]["mean_half68"], "agn": ens["closure_agn"]["widths"]["mean_half68"]}


# ============================================================================================= #
# C4, C5: flux-limited catalogs with the missing-host densities at (and away from) their inputs
# ============================================================================================= #


def build_c4_c5(c1_posts):
    ref = c1_posts[100]
    sHr, sfr = ref.summary("H0"), ref.summary("f")
    srcr = src_grid(A2 / "joint_s100.h5", ["log_likelihood"], "complete-catalog reference: the C1 joint grid of the same catalog (seed 100)")
    add_num("FluxfHalfComplete", hw90(sfr), 3, "", "90% half-width of f_AGN with complete catalogs (reference catalog, C1 grid)", srcr, "C4")
    add_num("FluxHzeroHalfComplete", hw90(sHr), 2, "km/s/Mpc", "90% half-width of H0 with complete catalogs (reference catalog, C1 grid)", srcr, "C4")
    # a3's own complete entry and the zero-density probe, for the record
    a3c = hf_post(A3 / "joint_complete_s100.h5", "a3_complete")
    fu0 = hf_post(SR / "fu_probes" / "results" / "joint_complete_n0m24_s100.h5", "fu_zero_density")
    with h5py.File(A2 / "joint_s100.h5") as h:
        Lc1 = h["log_likelihood"][()]
    with h5py.File(SR / "fu_probes" / "results" / "joint_complete_n0m24_s100.h5") as h:
        Lfu = h["log_likelihood"][()]
    DIAG["c4_complete_reference_comparison"] = {
        "C1_grid_s100 (2b86a2d, log10n0=-24)": {"f": clean(sfr), "H0": clean(sHr)},
        "a3_joint_complete_s100 (0c5b3db, selection, n0=-3/-5, C=1 by deep m_lim)": {"f": clean(a3c.summary("f")), "H0": clean(a3c.summary("H0"))},
        "fu_probes_joint_complete_n0m24_s100 (0c5b3db, selection, n0=-24)": {"f": clean(fu0.summary("f")), "H0": clean(fu0.summary("H0"))},
        "max_abs_dlogL_C1_vs_fu_n0m24": float(np.max(np.abs(Lc1 - Lfu))),
    }
    fmed = {}
    Hmed = {}
    for key, word, mlim in RUNGS:
        p = A3 / f"joint_{key}_s100.h5"
        with h5py.File(p) as h:
            assert h.attrs["c_mode"] == "selection"
            assert abs(h.attrs["arg_log10n0"] + 3) < 1e-12 and abs(h.attrs["arg_log10n0_c2"] + 5) < 1e-12
        g = hf_post(p, f"a3_{key}")
        sH, sf = g.summary("H0"), g.summary("f")
        src = src_grid(p, ["H0_grid", "f_grid", "log_likelihood"], "selection-mode completeness, missing-host densities at input (log10 n = -3, -5); spline-refined")
        add_post(f"FluxHzero{word}", sH, "km/s/Mpc", f"H0, flux limit m < {mlim}, densities of missing hosts at their input values", src, "C4", IN["H0"])
        add_post(f"Fluxf{word}", sf, "", f"f_AGN, flux limit m < {mlim}, densities at input", src, "C4", IN["f"])
        add_num(f"FluxfHalf{word}", hw90(sf), 3, "", f"90% half-width of f_AGN at m < {mlim}", src, "C4")
        add_num(f"FluxHzeroHalf{word}", hw90(sH), 2, "km/s/Mpc", f"90% half-width of H0 at m < {mlim}", src, "C4")
        add_num(f"FluxfRatio{word}", hw90(sf) / hw90(sfr), 2, "", f"90% half-width of f_AGN at m < {mlim} over that with complete catalogs (C1 grid, same catalog)", {"files": [rel(p), rel(A2 / "joint_s100.h5")], "computation": "ratio of 90% half-widths"}, "C4")
        add_num(f"FluxHzeroRatio{word}", hw90(sH) / hw90(sHr), 2, "", f"90% half-width of H0 at m < {mlim} over that with complete catalogs (C1 grid, same catalog)", {"files": [rel(p), rel(A2 / "joint_s100.h5")], "computation": "ratio of 90% half-widths"}, "C4")
        add_num(f"FluxfShift{word}", sf["median"] - sfr["median"], 3, "", f"median f_AGN at m < {mlim} minus the complete-catalog median (same catalog)", {"files": [rel(p), rel(A2 / "joint_s100.h5")]}, "C4")
        add_num(f"FluxHzeroShift{word}", sH["median"] - sHr["median"], 2, "km/s/Mpc", f"median H0 at m < {mlim} minus the complete-catalog median (same catalog)", {"files": [rel(p), rel(A2 / "joint_s100.h5")]}, "C4")
        fmed[(key, 1.0)] = sf["median"]
        Hmed[(key, 1.0)] = sH["median"]
    # ---- C5 ---- #
    arms = {"a05": 0.5, "a07": 0.7, "a09": 0.9, "a11": 1.1, "a13": 1.3, "a20": 2.0}
    stored = jload(A4 / "arms_summary.json")
    slopes = {}
    for key, word, mlim in RUNGS:
        files = [rel(A3 / f"joint_{key}_s100.h5")]
        for tag, fac in arms.items():
            p = A4 / f"joint_{key}_{tag}_s100.h5"
            with h5py.File(p) as h:
                assert h.attrs["c_mode"] == "selection"
                assert abs(h.attrs["arg_log10n0_c2"] - (-5 + math.log10(fac))) < 1e-5
            g = hf_post(p, f"a4_{key}_{tag}")
            fmed[(key, fac)] = g.summary("f")["median"]
            Hmed[(key, fac)] = g.summary("H0")["median"]
            files.append(rel(p))
        facs = sorted(f for (k, f) in fmed if k == key)
        x = np.log10(facs)
        yf = np.array([fmed[(key, f)] for f in facs])
        yH = np.array([Hmed[(key, f)] for f in facs])
        sf_, if_ = np.polyfit(x, yf, 1)
        sH_, iH_ = np.polyfit(x, yH, 1)
        r2f = 1 - np.sum((yf - (sf_ * x + if_)) ** 2) / np.sum((yf - yf.mean()) ** 2)
        r2H = 1 - np.sum((yH - (sH_ * x + iH_)) ** 2) / np.sum((yH - yH.mean()) ** 2)
        x6 = np.log10([f for f in facs if f != 1.0])
        s6 = np.polyfit(x6, [fmed[(key, f)] for f in facs if f != 1.0], 1)[0]
        src = {"files": files, "keys": ["log_likelihood"], "computation": "least-squares line through the posterior medians against log10 of the assumed/true AGN density ratio, 7 grids (factors 0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 2.0; factor 1.0 is the Analysis-3 grid)"}
        ex = {"medians_f": {str(f): fmed[(key, f)] for f in facs}, "medians_H0": {str(f): Hmed[(key, f)] for f in facs}, "R2_f": r2f, "R2_H0": r2H,
              "six_arm_slope_f_without_factor1": s6, "stored_arms_summary_slope_f": stored["rungs"][key]["selection (this work)"]["slope"],
              "direct_f_change_factor2_vs_1": fmed[(key, 2.0)] - fmed[(key, 1.0)], "direct_f_change_1_vs_half": fmed[(key, 1.0)] - fmed[(key, 0.5)],
              "direct_H0_change_factor2_vs_1": Hmed[(key, 2.0)] - Hmed[(key, 1.0)], "direct_H0_change_1_vs_half": Hmed[(key, 1.0)] - Hmed[(key, 0.5)]}
        add(f"SlopefAGN{word}", "anc", fnum(sf_, 2), sf_, "per dex", f"d f_AGN / d log10 n_AGN at m < {mlim} (assumed AGN density of missing hosts varied by factors 0.5 to 2)", src, "C5", extra=ex)
        add(f"SlopeHzero{word}", "anc", fnum(sH_, 2), sH_, "km/s/Mpc per dex", f"d H0 / d log10 n_AGN at m < {mlim}", src, "C5", extra={"R2_H0": r2H})
        add(f"DeltafTwo{word}", "anc", fnum(sf_ * math.log10(2), 2), sf_ * math.log10(2), "", f"change of f_AGN for a factor of two in the assumed AGN density at m < {mlim} (slope times log10 2)", src, "C5")
        add(f"DeltaHzeroTwo{word}", "anc", fnum(sH_ * math.log10(2), 2), sH_ * math.log10(2), "km/s/Mpc", f"change of H0 for a factor of two in the assumed AGN density at m < {mlim} (slope times log10 2)", src, "C5")
        slopes[key] = (sf_, sH_)
    allf = [v[0] for v in slopes.values()]
    srcs = {"computation": "min/max over the four flux limits of SlopefAGN*"}
    add_num("SlopefAGNMin", min(allf), 2, "per dex", "smallest d f_AGN / d log10 n_AGN over the four flux limits", srcs, "C5")
    add_num("SlopefAGNMax", max(allf), 2, "per dex", "largest d f_AGN / d log10 n_AGN over the four flux limits", srcs, "C5")
    # oracle cell (AGN catalog complete, galaxy m<18), for the record only
    o = hf_post(A4 / "joint_m18_oracle_s100.h5", "a4_oracle")
    DIAG["c4_oracle_cell_gal_m18_agn_complete"] = {"f": clean(o.summary("f")), "H0": clean(o.summary("H0"))}


# ============================================================================================= #
# C6: both densities free (nested sampling)
# ============================================================================================= #


def load_a5(path_h5, path_json):
    with h5py.File(path_h5) as h:
        s = h["samples"][()]
        names = [x.decode() for x in h["sampled_labels"][()]]
        assert h.attrs["c_mode"] == "selection"
        logz, logzerr = float(h.attrs["meta_logz"]), float(h.attrs["meta_logzerr"])
    j = jload(path_json)
    return ChainPost(s, names, tag=Path(path_h5).stem), logz, logzerr, j


def build_c6():
    runs = [(key, word, A5 / f"campaign_{key}_dynesty_s100.h5", A5 / f"campaign_{key}_dynesty_s100.json", "") for key, word, _ in RUNGS]
    runs += [("m18", "Eighteen", SR / "fu_seed101" / "results" / "campaign_m18_dynesty_s101.h5", SR / "fu_seed101" / "results" / "campaign_m18_dynesty_s101.json", "CatB"),
             ("m18", "Eighteen", SR / "fu_seed102" / "results" / "campaign_m18_dynesty_s102.h5", SR / "fu_seed102" / "results" / "campaign_m18_dynesty_s102.json", "CatC")]
    H0meds = []
    for key, word, ph5, pj, cat in runs:
        c, logz, logzerr, j = load_a5(ph5, pj)
        mlim = int(key[1:])
        tagw = word + cat
        catdesc = {"": "catalog A (seed 100)", "CatB": "catalog B (seed 101)", "CatC": "catalog C (seed 102)"}[cat]
        src = {"file": rel(ph5), "keys": ["samples (equal-weight)", "sampled_labels"], "computation": "numpy.percentile of the stored equal-weight dynesty samples (reproduces the run's JSON summary)",
               "json": rel(pj), "priors": j["priors"], "darksirens_sha": j["darksirens_git_sha"][:7]}
        sH = c.summary("H0")
        sn = c.summary("log10n0")
        sa = c.summary("log10n0_c2")
        sf = c.summary("fcat_2")
        add_post(f"FreeLognGal{tagw}", sn, "log10 Mpc^-3", f"log10 galaxy number density, both densities free, m < {mlim}, {catdesc}", src, "C6", IN["lognGal"], kind="chain")
        add_post(f"FreeLognAGN{tagw}", sa, "log10 Mpc^-3", f"log10 AGN number density, both densities free, m < {mlim}, {catdesc}", src, "C6", IN["lognAGN"], kind="chain")
        add_post(f"FreefAGN{tagw}", sf, "", f"f_AGN, both densities free, m < {mlim}, {catdesc}", src, "C6", IN["f"], kind="chain")
        add_post(f"FreeHzero{tagw}", sH, "km/s/Mpc", f"H0, both densities free, m < {mlim}, {catdesc}", src, "C6", IN["H0"], kind="chain", parts=(cat == "" and key in ("m18",)))
        add_num(f"FreeRhofGal{tagw}", c.corr("fcat_2", "log10n0"), 2, "", f"correlation of f_AGN with log10 n_gal, m < {mlim}, {catdesc}", src, "C6")
        add_num(f"FreeRhofAGN{tagw}", c.corr("fcat_2", "log10n0_c2"), 2, "", f"correlation of f_AGN with log10 n_AGN, m < {mlim}, {catdesc}", src, "C6")
        pr = j["priors"]["fcat_2"]
        add_num(f"FreefWidth{tagw}", w90(sf) / (pr[1] - pr[0]), 2, "", f"width of the f_AGN 90% interval as a fraction of its prior range [0, 1], m < {mlim}, {catdesc}", src, "C6")
        add(f"FreeLnZ{tagw}", "anc", f"{fnum(logz, 2)}\\pm{fnum(logzerr, 2)}", {"lnZ": logz, "err": logzerr}, "ln", f"natural-log evidence of the free-density model, m < {mlim}, {catdesc} (dynesty)",
            {"file": rel(ph5), "keys": ["attrs meta_logz", "attrs meta_logzerr"]}, "C6")
        if cat == "":
            H0meds.append(sH["median"])
    add_num("FreeHzeroSpread", max(H0meds) - min(H0meds), 2, "km/s/Mpc", "range (max minus min) of the H0 medians over the four flux limits with both densities free (catalog A)",
            {"files": [rel(A5 / f"campaign_{k}_dynesty_s100.h5") for k, _, _ in RUNGS], "computation": "max - min of medians"}, "C6")
    stored = jload(A5 / "free_anchor_summary.json")
    DIAG["c6_stored_summary_C_in_horizon"] = {r["rung"]: r["C_in_horizon"] for r in stored["rows"]}


# ============================================================================================= #
# C7, C8: host-dependent masses and spins
# ============================================================================================= #
BOX11 = {"H0": (60.0, 76.0), "f": (0.0, 1.0), "muG": (31.0, 39.0), "dmuG": (-4.0, 10.0), "muChi": (-0.1, 0.1), "dmuChi": (-0.05, 0.2)}
BOXA10 = {"f": (0.0, 1.0), "dmuG": (-10.0, 10.0), "dmuChi": (-0.2, 0.25)}


def vol(box, names):
    return float(np.prod([box[n][1] - box[n][0] for n in names]))


def load_chain_npz(npz, js):
    z = np.load(npz)
    names = [str(x) for x in z["names"]]
    j = jload(js)
    return ChainPost(z["samples"], names, tag=Path(npz).stem), j


CHAIN_NAME = {"H0": "Hzero", "f_agn": "fAGN", "mu_G": "MuG", "dmu_G": "DmuG", "mu_chi": "MuChi", "dmu_chi": "DmuChi", "sigma_G": "SigmaG", "sigma_chi": "SigmaChi"}
CHAIN_UNITS = {"H0": "km/s/Mpc", "f_agn": "", "mu_G": "Msun", "dmu_G": "Msun", "mu_chi": "", "dmu_chi": "", "sigma_G": "Msun", "sigma_chi": ""}
CHAIN_IN = {"H0": IN["H0"], "f_agn": IN["f"], "mu_G": IN["muG"], "dmu_G": IN["dmuG"], "mu_chi": IN["muChi"], "dmu_chi": IN["dmuChi"], "sigma_G": IN["sigmaG"], "sigma_chi": IN["sigmaChi"]}
CHAIN_DESC = {"H0": "H0", "f_agn": "f_AGN", "mu_G": "reference mass-peak location mu_G", "dmu_G": "AGN mass-peak offset Delta mu_G", "mu_chi": "reference mean effective spin mu_chi",
              "dmu_chi": "AGN spin-mean offset Delta mu_chi", "sigma_G": "shared mass-peak width sigma_G", "sigma_chi": "shared effective-spin width sigma_chi"}


def add_chain_model(prefix, cpost, j, npz, desc, claim, parts_all=False):
    out = {}
    for n in cpost.names:
        s = cpost.summary(n)
        src = {"file": rel(npz), "keys": ["samples (equal-weight)", "names"], "json": rel(Path(str(npz).replace(".npz", ".json"))),
               "computation": "numpy.percentile of the stored equal-weight dynesty samples", "box": j.get("box")}
        add_post(prefix + CHAIN_NAME[n], s, CHAIN_UNITS[n], f"{CHAIN_DESC[n]}, {desc}", src, claim, CHAIN_IN[n], kind="chain", parts=(parts_all or n == "H0"))
        out[n] = s
    return out


def build_c7_c8():
    ev2 = DATA / "seed100" / "events" / "events_marked_dmu0p10_dmuG5.h5"
    # ---------------- common population (C10-S) ---------------- #
    pS = A10 / "c10_arm_S.h5"
    with h5py.File(pS) as h:
        assert h.attrs["events_file"].endswith("events_marked_dmu0p10_dmuG5.h5")
        H, F, L, rej = h["H0_grid"][()], h["f_grid"][()], h["log_likelihood"][()], h["guard/rejected"][()]
        sha_S = h.attrs["darksirens_sha"]
    gS = GridPost(["H0", "f"], [H, F], L, [4, 8], extra={"H0": [IN["H0"], 60.0, 76.0], "f": [IN["f"]]}, rejected=rej, tag="C10-S")
    srcS = src_grid(pS, ["H0_grid", "f_grid", "log_likelihood"], "flat prior H0 [50,100], f [0,1]")
    sSH = add_post("CommonHzero", gS.summary("H0"), "km/s/Mpc", "H0, host-dependent mock analysed with one common population (both offsets zero), two catalogs", srcS, "C7", IN["H0"], parts=True)
    add_post("CommonfAGN", gS.summary("f"), "", "f_AGN, common-population model on the host-dependent mock", srcS, "C7", IN["f"])
    # same model restricted to the window [60, 76] (matched to the host-dependent lattice)
    HS_w = (H >= 60 - 1e-9) & (H <= 76 + 1e-9)
    gSw = GridPost(["H0", "f"], [H[HS_w], F], L[HS_w], [4, 8], extra={"H0": [IN["H0"]], "f": [IN["f"]]}, rejected=rej[HS_w], tag="C10-S[60,76]")
    sSHw = gSw.summary("H0")
    DIAG["c7_C10S_window_60_76"] = {"H0": clean(sSHw), "f": clean(gSw.summary("f"))}
    # ---------------- offsets, reference fixed (C10-J) ---------------- #
    pJ = A10 / "c10_arm_J.h5"
    with h5py.File(pJ) as h:
        assert json.loads(h.attrs["axis_order"]) == ["H0", "f_agn", "dmu_G", "dmu_chi"]
        assert h.attrs["events_file"].endswith("events_marked_dmu0p10_dmuG5.h5")
        axes = [h["H0_grid"][()], h["f_grid"][()], h["dmu_G_grid"][()], h["dmu_chi_grid"][()]]
        LJ, rejJ = h["log_likelihood"][()], h["guard/rejected"][()]
        sha_J = h.attrs["darksirens_sha"]
    gJ = GridPost(["H0", "f", "dmuG", "dmuChi"], axes, LJ, [4, 8, 4, 4],
                  extra={"H0": [IN["H0"]], "f": [IN["f"]], "dmuG": [IN["dmuG"]], "dmuChi": [IN["dmuChi"], 0.2]}, rejected=rejJ, tag="C10-J")
    srcJ = src_grid(pJ, ["H0_grid", "f_grid", "dmu_G_grid", "dmu_chi_grid", "log_likelihood"], "4-D lattice (34x11x15x15), spline-refined; flat priors")
    sJH = add_post("HDFixHzero", gJ.summary("H0"), "km/s/Mpc", "H0, host-dependent model (offsets free, reference population fixed at mu_G = 35, mu_chi = 0)", srcJ, "C8", IN["H0"], parts=True)
    sJf = add_post("HDFixfAGN", gJ.summary("f"), "", "f_AGN, host-dependent model with the reference population fixed", srcJ, "C8", IN["f"], parts=True)
    sJG = add_post("HDFixDmuG", gJ.summary("dmuG"), "Msun", "AGN mass-peak offset, host-dependent model with the reference population fixed", srcJ, "C8", IN["dmuG"], parts=True)
    sJC = add_post("HDFixDmuChi", gJ.summary("dmuChi"), "", "AGN spin-mean offset, host-dependent model with the reference population fixed", srcJ, "C8", IN["dmuChi"], parts=True)
    for a, b, nm in (("H0", "dmuG", "HDFixRhoHzeroDmuG"), ("H0", "f", "HDFixRhoHzerof"), ("f", "dmuG", "HDFixRhofDmuG"), ("f", "dmuChi", "HDFixRhofDmuChi")):
        add_num(nm, gJ.corr(a, b), 2, "", f"correlation of {a} and {b}, host-dependent model with the reference population fixed", srcJ, "C8")
    DIAG["c7_C10J_rejected_cells"] = gJ.n_bad
    DIAG["c7_C10J_edge_densities"] = {n: (gJ.summary(n)["edge_density_lo"], gJ.summary(n)["edge_density_hi"]) for n in gJ.names}
    # ---------------- 11D, 12M, 12chi, 13 (chains) ---------------- #
    c11, j11 = load_chain_npz(A11 / "a11_11D.npz", A11 / "a11_11D.json")
    s11 = add_chain_model("HDFree", c11, j11, A11 / "a11_11D.npz", "host-dependent model with the reference peak location and spin mean free (six parameters)", "C8", parts_all=True)
    c12m, j12m = load_chain_npz(A12 / "a12_12M.npz", A12 / "a12_12M.json")
    s12m = add_chain_model("WidthM", c12m, j12m, A12 / "a12_12M.npz", "six-parameter model plus a shared mass-peak width", "C8")
    c12c, j12c = load_chain_npz(A12 / "a12_12chi.npz", A12 / "a12_12chi.json")
    s12c = add_chain_model("WidthChi", c12c, j12c, A12 / "a12_12chi.npz", "six-parameter model plus a shared spin width", "C8")
    p13 = A13 / "a13core_e7c3007_rslice_cap20_historical_n200_s1.npz"
    c13, j13 = load_chain_npz(p13, A13 / "a13core_e7c3007_rslice_cap20_historical_n200_s1.json")
    s13 = add_chain_model("HDEight", c13, j13, p13, "eight-parameter model (reference location, spin mean, both widths and both offsets free)", "C8", parts_all=True)
    # ---------------- shifts and width ratios ---------------- #
    srcpair = lambda *f: {"files": [rel(x) for x in f], "computation": "difference of posterior medians / ratio of 90% interval widths"}
    add_num("ShiftCommonMinusFix", sSHw["median"] - sJH["median"], 2, "km/s/Mpc", "H0 median of the common-population model minus that of the host-dependent model (reference fixed); both on H0 in [60, 76]", srcpair(pS, pJ), "C7")
    add_num("ShiftCommonFullMinusFix", sSH["median"] - sJH["median"], 2, "km/s/Mpc", "same, common-population model on its full H0 range [50, 100]", srcpair(pS, pJ), "C7")
    add_num("ShiftFreeMinusFix", s11["H0"]["median"] - sJH["median"], 2, "km/s/Mpc", "H0 median of the free-reference model minus the fixed-reference model", srcpair(A11 / "a11_11D.npz", pJ), "C8")
    add_num("ShiftEightMinusFix", s13["H0"]["median"] - sJH["median"], 2, "km/s/Mpc", "H0 median of the eight-parameter model minus the fixed-reference model", srcpair(p13, pJ), "C8")
    add_num("ShiftEightMinusFree", s13["H0"]["median"] - s11["H0"]["median"], 2, "km/s/Mpc", "H0 median of the eight-parameter model minus the six-parameter model", srcpair(p13, A11 / "a11_11D.npz"), "C8")
    add_num("ShiftCommonMinusEight", sSHw["median"] - s13["H0"]["median"], 2, "km/s/Mpc", "H0 median of the common-population model (H0 in [60, 76]) minus the eight-parameter model", srcpair(pS, p13), "C7")
    add_num("WRatioFixOverCommon", w90(sJH) / w90(sSHw), 2, "", "90% width of H0, fixed-reference host-dependent model over the common-population model (both on [60, 76])", srcpair(pJ, pS), "C8")
    add_num("WRatioFreeOverFix", w90(s11["H0"]) / w90(sJH), 2, "", "90% width of H0, free-reference model over fixed-reference model", srcpair(A11 / "a11_11D.npz", pJ), "C8")
    add_num("WRatioEightOverFree", w90(s13["H0"]) / w90(s11["H0"]), 2, "", "90% width of H0, eight-parameter over six-parameter model", srcpair(p13, A11 / "a11_11D.npz"), "C8")
    add_num("WRatioEightOverFix", w90(s13["H0"]) / w90(sJH), 2, "", "90% width of H0, eight-parameter over fixed-reference model", srcpair(p13, pJ), "C8")
    add_num("WRatioEightOverCommon", w90(s13["H0"]) / w90(sSHw), 2, "", "90% width of H0, eight-parameter over common-population model (on [60, 76])", srcpair(p13, pS), "C8")
    add_num("WRatioWidthMOverFree", w90(s12m["H0"]) / w90(s11["H0"]), 2, "", "90% width of H0, shared-mass-width model over six-parameter model", srcpair(A12 / "a12_12M.npz", A11 / "a11_11D.npz"), "C8")
    add_num("WRatioWidthChiOverFree", w90(s12c["H0"]) / w90(s11["H0"]), 2, "", "90% width of H0, shared-spin-width model over six-parameter model", srcpair(A12 / "a12_12chi.npz", A11 / "a11_11D.npz"), "C8")
    add_num("WRatioFreeDmuGOverFix", w90(s11["dmu_G"]) / w90(sJG), 2, "", "90% width of the mass-peak offset, free-reference over fixed-reference model", srcpair(A11 / "a11_11D.npz", pJ), "C8")
    add_num("WRatioFreeDmuChiOverFix", w90(s11["dmu_chi"]) / w90(sJC), 2, "", "90% width of the spin-mean offset, free-reference over fixed-reference model", srcpair(A11 / "a11_11D.npz", pJ), "C8")
    for nm, c, s in (("HDFree", c11, s11), ("HDEight", c13, s13)):
        add_num(nm + "RhoHzeroMuG", c.corr("H0", "mu_G"), 2, "", f"correlation of H0 and the reference mass-peak location, {nm}", {"file": rel(A11 / 'a11_11D.npz') if nm == 'HDFree' else rel(p13)}, "C8")
        add_num(nm + "RhoHzeroDmuG", c.corr("H0", "dmu_G"), 2, "", f"correlation of H0 and the AGN mass-peak offset, {nm}", {"file": rel(A11 / 'a11_11D.npz') if nm == 'HDFree' else rel(p13)}, "C8")
    # ---------------- pinned-offset arms and galaxy-catalog-only controls ---------------- #
    pM = A10 / "c10_mech.h5"
    mech = {}
    with h5py.File(pM) as h:
        Hm, Fm = h["H0_grid"][()], h["f_grid"][()]
        pin = (float(h.attrs["pin_dmu_G"]), float(h.attrs["pin_dmu_chi"]))
        for a in ("P0", "P1", "I0", "I1"):
            mech[a] = GridPost(["H0", "f"], [Hm, Fm], h[a + "/log_likelihood"][()], [4, 8], extra={"H0": [IN["H0"]], "f": [IN["f"]]}, rejected=h[a + "/rejected"][()], tag="mech_" + a)
    desc = {"P0": "two catalogs, both offsets zero", "P1": f"two catalogs, offsets pinned at (Delta mu_G, Delta mu_chi) = ({pin[0]}, {pin[1]})",
            "I0": "galaxy catalog for both host classes, both offsets zero (the posterior runs into the top of the window: a lower bound)", "I1": f"galaxy catalog for both host classes, offsets pinned at ({pin[0]}, {pin[1]})"}
    ms = {}
    for a in ("P0", "P1", "I0", "I1"):
        src = src_grid(pM, [f"{a}/log_likelihood", "H0_grid", "f_grid"], "flat prior H0 in the window [60, 76], f [0,1]")
        s = add_post(f"Pin{a[0]}{'Zero' if a[1]=='0' else 'One'}Hzero", mech[a].summary("H0"), "km/s/Mpc", f"H0 on the host-dependent mock, {desc[a]}, H0 window [60, 76]", src, "C7,C8", IN["H0"])
        ms[a] = s
        DIAG[f"c7_mech_{a}_H0_edge_density_hi"] = s["edge_density_hi"]
    add_num("WRatioPinTwoOverGalOnly", w90(ms["P1"]) / w90(ms["I1"]), 2, "", "90% width of H0 with both catalogs over that with the galaxy catalog for both host classes, offsets pinned (P1 over I1)", srcpair(pM), "C8")
    add_num("ShiftGalOnlyPinMinusTwo", ms["I1"]["median"] - ms["P1"]["median"], 2, "km/s/Mpc", "H0 median with the galaxy catalog for both host classes minus with both catalogs, offsets pinned (I1 minus P1)", srcpair(pM), "C8")
    add_num("ShiftCommonMinusPinned", sSHw["median"] - ms["P1"]["median"], 2, "km/s/Mpc", "H0 median of the common-population model (on [60, 76]) minus the two-catalog model with offsets pinned (P1)", srcpair(pS, pM), "C7")
    pC = A10 / "c10_control.h5"
    ctrl = {}
    with h5py.File(pC) as h:
        Fc = h["f_grid"][()]
        for a in ("B5M", "B5U", "B0M"):
            ctrl[a] = GridPost(["H0", "f"], [h[a + "/H0_grid"][()], Fc], h[a + "/log_likelihood"][()], [4, 8], extra={"H0": [IN["H0"]], "f": [IN["f"]]}, rejected=h[a + "/rejected"][()], tag="ctrl_" + a)
            DIAG[f"c7_ctrl_{a}_H0_range"] = [float(h[a + "/H0_grid"][()][0]), float(h[a + "/H0_grid"][()][-1])]
    cdesc = {"B5M": "galaxy-only control mock with the AGN-like population offset by +5 Msun and +0.10, offsets modelled (pinned at input)",
             "B5U": "same mock, offsets not modelled (common population); the posterior is cut by the top of the range at 100, so it is a lower bound", "B0M": "galaxy-only control mock with spin offset only, spin offset modelled (pinned at input)"}
    cs = {}
    for a, nm in (("B5M", "CtrlModHzero"), ("B5U", "CtrlUnmodHzero"), ("B0M", "CtrlSpinOnlyHzero")):
        src = src_grid(pC, [f"{a}/log_likelihood", f"{a}/H0_grid", "f_grid"], f"flat prior on the stored H0 window {DIAG[f'c7_ctrl_{a}_H0_range']}, f [0,1]")
        cs[a] = add_post(nm, ctrl[a].summary("H0"), "km/s/Mpc", "H0, " + cdesc[a], src, "C7", IN["H0"], parts=(a != "B0M"))
        with h5py.File(pC) as h:
            Lc = h[a + "/log_likelihood"][()]
        f_flat = float(np.max(np.ptp(Lc, axis=1))) < 1e-8
        DIAG[f"c7_ctrl_{a}_likelihood_independent_of_f"] = f_flat
        if not f_flat:
            add_post(nm.replace("Hzero", "fBranch"), ctrl[a].summary("f"), "", "fraction of mergers in the offset population, " + cdesc[a], src, "C7", IN["f"])
    add_num("ShiftCtrlUnmodMinusMod", cs["B5U"]["median"] - cs["B5M"]["median"], 1, "km/s/Mpc",
            "H0 median with the offsets unmodelled minus modelled, galaxy-only control mock (a lower bound: the unmodelled posterior is cut by the top of the range at 100)", srcpair(pC), "C7",)
    add_num("CtrlUnmodHzeroEdge", cs["B5U"]["edge_density_hi"], 4, "", "posterior density at H0 = 100 relative to its peak, galaxy-only control with offsets unmodelled", srcpair(pC), "C7")
    add_num("WRatioCtrlModOverSpinOnly", w90(cs["B5M"]) / w90(cs["B0M"]), 2, "", "90% width of H0 with the mass offset present and modelled over that with only the spin offset (galaxy-only control mocks)", srcpair(pC), "C7")
    # ---------------- fixed-H0 cube (Analysis 10) ---------------- #
    fx = {}
    with h5py.File(A10 / "a10_arm_S.h5") as h:
        fx["S"] = GridPost(["f"], [h["f_grid"][()]], h["log_likelihood"][()], [10], extra={"f": [IN["f"]]}, rejected=h["guard/rejected"][()], tag="A10-S")
    with h5py.File(A10 / "a10_arm_chi.h5") as h:
        fx["chi"] = GridPost(["f", "dmuChi"], [h["f_grid"][()], h["dmu_chi_grid"][()]], h["log_likelihood"][()], [8, 4], extra={"f": [IN["f"]], "dmuChi": [0.0, IN["dmuChi"], -0.05, 0.2]}, rejected=h["guard/rejected"][()], tag="A10-chi")
    with h5py.File(A10 / "a10_arm_M.h5") as h:
        fx["M"] = GridPost(["f", "dmuG"], [h["f_grid"][()], h["dmu_G_grid"][()]], h["log_likelihood"][()], [8, 4], extra={"f": [IN["f"]], "dmuG": [IN["dmuG"]]}, rejected=h["guard/rejected"][()], tag="A10-M")
    with h5py.File(A10 / "a10_arm_J.h5") as h:
        assert json.loads(h.attrs["axis_order"]) == ["f_agn", "dmu_G", "dmu_chi"]
        fx["J"] = GridPost(["f", "dmuG", "dmuChi"], [h["f_grid"][()], h["dmu_G_grid"][()], h["dmu_chi_grid"][()]], h["log_likelihood"][()], [8, 4, 4],
                           extra={"f": [IN["f"]], "dmuG": [IN["dmuG"]], "dmuChi": [0.0, IN["dmuChi"], -0.05, 0.2]}, rejected=h["guard/rejected"][()], tag="A10-J")
        n_rej_J = int(h["guard/rejected"][()].sum())
    srcfx = lambda a: src_grid(A10 / f"a10_arm_{a}.h5", ["log_likelihood", "*_grid"], "fixed H0 = 67.74; flat priors on the stored axes; spline-refined")
    add_post("FixHSpatialfAGN", fx["S"].summary("f"), "", "f_AGN at fixed H0, host-dependent mock, common population (offsets zero)", srcfx("S"), "C8,C9", IN["f"])
    add_post("FixHSpinfAGN", fx["chi"].summary("f"), "", "f_AGN at fixed H0, host-dependent mock, spin offset only modelled", srcfx("chi"), "C8,C9", IN["f"])
    add_post("FixHSpinDmuChi", fx["chi"].summary("dmuChi"), "", "spin-mean offset at fixed H0, host-dependent mock, spin offset only modelled", srcfx("chi"), "C8,C9", IN["dmuChi"])
    add_post("FixHMassfAGN", fx["M"].summary("f"), "", "f_AGN at fixed H0, host-dependent mock, mass offset only modelled", srcfx("M"), "C8", IN["f"])
    add_post("FixHMassDmuG", fx["M"].summary("dmuG"), "Msun", "mass-peak offset at fixed H0, mass offset only modelled", srcfx("M"), "C8", IN["dmuG"])
    add_post("FixHBothfAGN", fx["J"].summary("f"), "", "f_AGN at fixed H0, both offsets modelled", srcfx("J"), "C8", IN["f"])
    add_post("FixHBothDmuG", fx["J"].summary("dmuG"), "Msun", "mass-peak offset at fixed H0, both offsets modelled", srcfx("J"), "C8", IN["dmuG"])
    add_post("FixHBothDmuChi", fx["J"].summary("dmuChi"), "", "spin-mean offset at fixed H0, both offsets modelled", srcfx("J"), "C8", IN["dmuChi"])
    c11c, j11c = load_chain_npz(A11 / "a11_11C.npz", A11 / "a11_11C.json")
    add_chain_model("FixHFree", c11c, j11c, A11 / "a11_11C.npz", "fixed H0, reference location and spin mean free (Analysis 11C)", "C8")
    DIAG["c8_A10J_rejected_cells"] = n_rej_J

    # ---------------- evidences and Bayes factors ---------------- #
    # grid evidences (ln of the evidence = ln int L dtheta - ln prior volume)
    def lnZ_grid(g, box, names_vol, boxvol):
        r, raw = g.log_int(box)
        V = vol(boxvol, names_vol)
        return r - math.log(V), raw - math.log(V)

    # fixed H0 (prior box of the A10 cube: f [0,1], dmu_G [-10,10], dmu_chi [-0.2,0.25])
    zS = lnZ_grid(fx["S"], {}, ["f"], BOXA10)
    zC = lnZ_grid(fx["chi"], {}, ["f", "dmuChi"], BOXA10)
    zM = lnZ_grid(fx["M"], {}, ["f", "dmuG"], BOXA10)
    zJ = lnZ_grid(fx["J"], {}, ["f", "dmuG", "dmuChi"], BOXA10)
    dyn = jload(A11 / "a11_ns_a10J_dynesty_n300_s1.json")
    DIAG["c8_grid_vs_dynesty_lnZ_A10J_box"] = {"grid_refined": zJ[0], "grid_unrefined": zJ[1], "dynesty": dyn["logZ"], "dynesty_err": dyn["logZerr"], "box": dyn["box"]}
    # Savage-Dickey at (0,0) by interpolation in dmu_chi, as a cross-check of the evidence ratio
    x1, x2, m = fx["J"].marg2("dmuG", "dmuChi")
    i0, j0 = int(np.argmin(np.abs(x1))), int(np.argmin(np.abs(x2)))
    assert abs(x1[i0]) < 1e-12 and abs(x2[j0]) < 1e-12
    sd00 = math.log(1.0 / (20 * 0.45)) - math.log(m[i0, j0])
    DIAG["c8_SD_both_offsets_zero_interpolated_ln"] = sd00

    def addB(name, lnB, err, meaning, src, claim, pair, how, extra=None):
        e10 = None if err is None else err / LN10
        shown = e10 if (e10 is not None and e10 >= 0.005) else None
        if e10 is not None and shown is None:
            meaning += " (grid integration: numerical uncertainty below 0.005)"
        add(name, "anc", flb(lnB / LN10, shown), {"log10B": lnB / LN10, "err": e10}, "log10", meaning, src, claim, extra=extra)
        BAYES.append({"pair": pair, "qualifies": True, "macro": name, "log10B": lnB / LN10, "err": None if err is None else err / LN10, "how": how})

    num = lambda z: abs(z[0] - z[1])
    srcA10 = {"files": [rel(A10 / f"a10_arm_{a}.h5") for a in ("S", "chi", "M", "J")], "computation": "grid evidences on the fixed-H0 lattices; prior box f [0,1], Delta mu_G [-10,10], Delta mu_chi [-0.2,0.25] (the cube's own axes)"}
    addB("LBBothOffsetsFixH", zJ[0] - zS[0], math.hypot(num(zJ), num(zS)), "log10 Bayes factor of the host-dependent model (both offsets free) against both offsets zero, fixed H0 = 67.74 (Savage-Dickey: arm S is the cube's zero-offset slice)",
         srcA10, "C8", "A10-J vs A10-S (fixed H0, both offsets zero)", "evidence ratio = Savage-Dickey at (0,0)", extra={"lnB": zJ[0] - zS[0], "SD_interpolated_lnB": sd00, "grid_vs_dynesty": DIAG["c8_grid_vs_dynesty_lnZ_A10J_box"]})
    addB("LBSpinOnlyFixH", zC[0] - zS[0], math.hypot(num(zC), num(zS)), "log10 Bayes factor, spin offset only against no offsets, fixed H0, host-dependent mock", srcA10, "C8", "A10-chi vs A10-S", "grid evidence ratio")
    addB("LBMassOnlyFixH", zM[0] - zS[0], math.hypot(num(zM), num(zS)), "log10 Bayes factor, mass offset only against no offsets, fixed H0, host-dependent mock", srcA10, "C8", "A10-M vs A10-S", "grid evidence ratio")
    addB("LBBothVsSpinFixH", zJ[0] - zC[0], math.hypot(num(zJ), num(zC)), "log10 Bayes factor, both offsets against spin offset only, fixed H0", srcA10, "C8", "A10-J vs A10-chi", "grid evidence ratio")
    addB("LBBothVsMassFixH", zJ[0] - zM[0], math.hypot(num(zJ), num(zM)), "log10 Bayes factor, both offsets against mass offset only, fixed H0", srcA10, "C8", "A10-J vs A10-M", "grid evidence ratio")
    # 11C (free references, fixed H0) vs A10-J restricted to the 11C box
    zJ11 = lnZ_grid(fx["J"], {"dmuG": BOX11["dmuG"], "dmuChi": BOX11["dmuChi"]}, ["f", "dmuG", "dmuChi"], BOX11)
    addB("LBFreeRefFixH", j11c["logZ"] - zJ11[0], math.hypot(j11c["logZerr"], num(zJ11)), "log10 Bayes factor, reference location and spin mean free against fixed at (35, 0), fixed H0 = 67.74 (11C against the A10 cube in the 11C prior box)",
         {"files": [rel(A11 / "a11_11C.json"), rel(A10 / "a10_arm_J.h5")], "computation": "lnZ(11C, dynesty) - lnZ(A10-J grid, box f [0,1], Delta mu_G [-4,10], Delta mu_chi [-0.05,0.2])"}, "C8", "11C vs A10-J (fixed H0)", "nested-sampling evidence against grid evidence, same code af896ca, same mock, priors matched")
    # H0 free: C10-S, C10-J on the 11D box (H0 [60,76], f [0,1], dmu_G [-4,10], dmu_chi [-0.05,0.2])
    zCS = lnZ_grid(gS, {"H0": (60.0, 76.0)}, ["H0", "f"], BOX11)
    zCJ = lnZ_grid(gJ, {"dmuChi": (-0.05, 0.2)}, ["H0", "f", "dmuG", "dmuChi"], BOX11)
    DIAG["c7_grid_lnZ"] = {"C10-S_box11": zCS, "C10-J_box11": zCJ}
    srcHJ = {"files": [rel(pS), rel(pJ)], "computation": "grid evidences, prior box H0 [60,76], f [0,1], Delta mu_G [-4,10], Delta mu_chi [-0.05,0.2] (the 11D box); C10-J lattice covers f [0.05,0.55], Delta mu_G [1,9], Delta mu_chi [0.025,0.2] of it, the rest carries negligible posterior (edge densities in values.json diagnostics)"}
    addB("LBOffsetsVsCommon", zCJ[0] - zCS[0], math.hypot(num(zCJ), num(zCS)), "log10 Bayes factor of the host-dependent model (offsets free, reference fixed) against the common population, H0 free", srcHJ, "C7,C8",
         "C10-J vs C10-S (H0 free)", "grid evidence ratio, same code af896ca, same mock")
    add("LnZCommon", "anc", fnum(zCS[0], 2), {"lnZ": zCS[0], "err": num(zCS)}, "ln", "ln evidence of the common-population model, H0 in [60,76] (grid integration; numerical uncertainty in values.json)", srcHJ, "C7")
    add("LnZFix", "anc", fnum(zCJ[0], 2), {"lnZ": zCJ[0], "err": num(zCJ)}, "ln", "ln evidence of the fixed-reference host-dependent model in the 11D box (grid integration)", srcHJ, "C8")
    addB("LBFreeRefVsFix", j11["logZ"] - zCJ[0], math.hypot(j11["logZerr"], num(zCJ)), "log10 Bayes factor, reference location and spin mean free against fixed, H0 free (11D against C10-J)",
         {"files": [rel(A11 / "a11_11D.json"), rel(pJ)], "computation": "lnZ(11D) - lnZ(C10-J grid, 11D box)"}, "C8", "11D vs C10-J", "nested-sampling evidence against grid evidence, same code af896ca, same mock, priors matched")
    addB("LBFreeRefVsCommon", j11["logZ"] - zCS[0], math.hypot(j11["logZerr"], num(zCS)), "log10 Bayes factor, six-parameter host-dependent model against the common population, H0 free",
         {"files": [rel(A11 / "a11_11D.json"), rel(pS)], "computation": "lnZ(11D) - lnZ(C10-S grid, H0 [60,76], f [0,1])"}, "C7,C8", "11D vs C10-S", "nested-sampling evidence against grid evidence")
    addB("LBWidthMVsFree", j12m["logZ"] - j11["logZ"], math.hypot(j12m["logZerr"], j11["logZerr"]), "log10 Bayes factor, shared mass-peak width free (flat on [1, 10] Msun) against fixed at 5 Msun",
         {"files": [rel(A12 / "a12_12M.json"), rel(A11 / "a11_11D.json")], "keys": ["logZ", "logZerr"]}, "C8", "12M vs 11D", "nested-sampling evidences, same code, same mock, nested priors")
    addB("LBWidthChiVsFree", j12c["logZ"] - j11["logZ"], math.hypot(j12c["logZerr"], j11["logZerr"]), "log10 Bayes factor, shared spin width free (flat on [0.01, 1]) against fixed at 0.1",
         {"files": [rel(A12 / "a12_12chi.json"), rel(A11 / "a11_11D.json")], "keys": ["logZ", "logZerr"]}, "C8", "12chi vs 11D", "nested-sampling evidences, same code, same mock, nested priors")
    for nm, j in (("LnZFree", j11), ("LnZWidthM", j12m), ("LnZWidthChi", j12c), ("LnZFreeFixH", j11c)):
        add(nm, "anc", f"{fnum(j['logZ'], 2)}\\pm{fnum(j['logZerr'], 2)}", {"lnZ": j["logZ"], "err": j["logZerr"]}, "ln", f"ln evidence (dynesty, merged runs) of {nm}",
            {"file": rel({'LnZFree': A11 / 'a11_11D.json', 'LnZWidthM': A12 / 'a12_12M.json', 'LnZWidthChi': A12 / 'a12_12chi.json', 'LnZFreeFixH': A11 / 'a11_11C.json'}[nm]), "keys": ["logZ", "logZerr"]}, "C8")
    add("LnZEight", "anc", f"{fnum(j13['logZ'], 2)}\\pm{fnum(j13['logZerr'], 2)}", {"lnZ": j13["logZ"], "err": j13["logZerr"]}, "ln", "ln evidence of the eight-parameter model (one dynesty run, darksirens-core e7c3007, variance cap 20, Delta mu_chi box [-0.05, 0.30])",
        {"file": rel(A13 / "a13core_e7c3007_rslice_cap20_historical_n200_s1.json"), "keys": ["logZ", "logZerr"]}, "C8")
    # galaxy-only control: offsets pinned at input vs no offsets, same mock B5, prior H0 [50,100]
    zB5M = lnZ_grid(ctrl["B5M"], {}, ["H0", "f"], {"H0": (50.0, 100.0), "f": (0.0, 1.0)})
    zB5U = lnZ_grid(ctrl["B5U"], {}, ["H0", "f"], {"H0": (50.0, 100.0), "f": (0.0, 1.0)})
    addB("LBCtrlModVsUnmod", zB5M[0] - zB5U[0], math.hypot(num(zB5M), num(zB5U)), "log10 Bayes factor, galaxy-only control mock: population offsets at their input values against no offsets, (H0, f) free, prior H0 [50,100]",
         {"file": rel(pC), "computation": "grid evidences; the stored lattices cover [54,95] and [58,100] of the prior, the uncovered part carries less than 1e-10 of the peak density (B5M) or lies 30 below the B5U peak"}, "C7",
         "B5M vs B5U (galaxy-only control mock)", "grid evidence ratio; both models have offsets fixed (input values or zero), same free parameters")
    BAYES.extend([
        {"pair": "HDEight (A13) vs HDFree (11D), WidthM, WidthChi", "qualifies": False,
         "why": "different likelihood code (darksirens-core e7c3007 against af896ca: a constant ln L offset of about -3.6 nats with 0.3 spread measured in experiments/experiment_darksirens_core_equivalence/results/a11_comparison.json), a selection-variance cap of 20 in A13 against an inert 1e6 cap in 11D/12, and a wider Delta mu_chi prior box ([-0.05,0.30] against [-0.05,0.20]) with posterior mass above 0.20 in A13"},
        {"pair": "P1 vs P0, I1 vs I0 (pinned offsets vs zero offsets)", "qualifies": False, "why": "offsets pinned at the fixed-H0 maximum of the same data; an evidence ratio would count the data twice"},
        {"pair": "B0M vs B5M/B5U", "qualifies": False, "why": "different event sets (control mocks B0 and B5)"},
        {"pair": "C10-J vs 11C/A10 (H0 free vs fixed)", "qualifies": False, "why": "a fixed H0 = 67.74 is not a model of interest; not formed"},
        {"pair": "Analysis 13 unguarded run vs guarded run", "qualifies": False, "why": "same model, different likelihood truncation (variance cap); the guarded run is the record"},
    ])
    DIAG["c7_11D_recorded_events_path_note"] = ("a11_ns_11D_*.json environment.a8_provenance.inputs.gw_path_used records events_marked_dmu0p10.h5 (the spin-only mock); "
                                                "scripts/a11_likelihood.py loads GW_PATH_A11 = events_marked_dmu0p10_dmuG5.h5 (md5 427990378e29...), and the posterior's Delta mu_G = 3.9 is only possible on the two-offset mock")


# ============================================================================================= #
# C9: spin offset with and without catalogs
# ============================================================================================= #


def build_c9():
    def a8(arm):
        p = A8 / f"arm_{arm}.h5"
        with h5py.File(p) as h:
            assert h.attrs["events_file"].endswith("events_marked_dmu0p10.h5")
            F = h["f_grid"][()]
            L = h["log_likelihood"][()]
            rej = h["guard/rejected"][()] if "guard/rejected" in h else None
            M = h["mu_chi_c2_grid"][()] if "mu_chi_c2_grid" in h else None
        if M is None:
            return GridPost(["f"], [F], L, [10], extra={"f": [IN["f"]]}, rejected=rej, tag="A8-" + arm), p
        return GridPost(["f", "dmuChi"], [F, M], L, [8, 6], extra={"f": [IN["f"]], "dmuChi": [0.0, IN["dmuChi"]]}, rejected=rej, tag="A8-" + arm), p

    gS, pS = a8("S_spatial")
    gJ, pJ = a8("J_joint")
    gI, pI = a8("I_intrinsic")
    src = lambda p: src_grid(p, ["f_grid", "mu_chi_c2_grid", "log_likelihood"], "fixed H0 = 67.74, spin-offset-only mock; flat priors f [0,1], Delta mu_chi [-0.2,0.25]; spline-refined")
    sSf = add_post("SpinPosfAGN", gS.summary("f"), "", "f_AGN at fixed H0 from positions only (spin offset not modelled), spin-offset mock", src(pS), "C9", IN["f"], parts=True)
    sJf = add_post("SpinBothfAGN", gJ.summary("f"), "", "f_AGN at fixed H0 with positions and the spin offset modelled", src(pJ), "C9", IN["f"], parts=True)
    sJc = add_post("SpinBothDmuChi", gJ.summary("dmuChi"), "", "spin-mean offset at fixed H0 with positions (two catalogs) and spins", src(pJ), "C9", IN["dmuChi"], parts=True)
    sIf = add_post("SpinIntfAGN", gI.summary("f"), "", "f_AGN at fixed H0 from spins alone (one galaxy catalog in both slots: no positional host information)", src(pI), "C9", IN["f"])
    sIc = add_post("SpinIntDmuChi", gI.summary("dmuChi"), "", "spin-mean offset at fixed H0 from spins alone", src(pI), "C9", IN["dmuChi"], parts=True)
    add_num("SpinfWidthRatio", w90(sJf) / w90(sSf), 3, "", "90% width of f_AGN with the spin offset modelled over without (positions only), fixed H0", {"files": [rel(pJ), rel(pS)]}, "C9")
    add_num("SpinfWidthRatioSixtyEight", (sJf["ci68"][1] - sJf["ci68"][0]) / (sSf["ci68"][1] - sSf["ci68"][0]), 3, "", "68% width of f_AGN with the spin offset modelled over without, fixed H0", {"files": [rel(pJ), rel(pS)]}, "C9")
    add_num("SpinfWidthChange", 100 * (w90(sJf) / w90(sSf) - 1), 1, "%", "percentage change of the f_AGN 90% width when the spin offset is added, fixed H0", {"files": [rel(pJ), rel(pS)]}, "C9")
    add_num("SpinDmuChiWidthRatio", w90(sIc) / w90(sJc), 1, "", "90% width of the spin-mean offset from spins alone over that with the catalogs, fixed H0", {"files": [rel(pI), rel(pJ)]}, "C9")
    add_num("SpinDmuChiWidthRatioSixtyEight", (sIc["ci68"][1] - sIc["ci68"][0]) / (sJc["ci68"][1] - sJc["ci68"][0]), 1, "", "same with 68% widths", {"files": [rel(pI), rel(pJ)]}, "C9")
    add_num("SpinIntfWidthRatio", w90(sIf) / w90(sSf), 1, "", "90% width of f_AGN from spins alone over that from positions alone, fixed H0", {"files": [rel(pI), rel(pS)]}, "C9")
    add_num("SpinBothRho", gJ.corr("f", "dmuChi"), 2, "", "correlation of f_AGN and the spin-mean offset with positions and spins, fixed H0", src(pJ), "C9")
    # Savage-Dickey at dmu_chi = 0 (flat prior on [-0.2, 0.25], density 1/0.45)
    for g, nm, p, what in ((gJ, "LBSpinFixH", pJ, "with the catalogs"), (gI, "LBSpinIntFixH", pI, "from spins alone")):
        d0 = g.density_at("dmuChi", 0.0)
        lnB = math.log(1 / 0.45) - math.log(d0)
        # cross-check by the evidence ratio against the zero-offset model
        add(nm, "anc", flb(lnB / LN10), lnB / LN10, "log10", f"log10 Bayes factor of a spin-mean offset against none (Savage-Dickey at Delta mu_chi = 0, prior flat on [-0.2, 0.25]), fixed H0, {what}",
            src_grid(p, ["log_likelihood"], "Savage-Dickey: marginal density at 0 from the spline-refined grid (0 is not a stored node; neighbours -0.005 and +0.0025)"), "C9")
        BAYES.append({"pair": f"spin offset vs none, fixed H0, {what} (A8 {'J' if g is gJ else 'I'})", "qualifies": True, "macro": nm, "log10B": lnB / LN10, "how": "Savage-Dickey at Delta mu_chi = 0"})
    # evidence-ratio cross-check for J vs S
    zJ = gJ.log_int()[0] - math.log(0.45)
    zS = gS.log_int()[0]
    DIAG["c9_A8_J_vs_S_evidence_ratio_log10"] = (zJ - zS) / LN10
    # ---- Analysis 9: H0 free ---- #
    p9s, p9j = A9 / "s9_spatial.h5", A9 / "j9_marked.h5"
    with h5py.File(p9s) as h:
        H, F, L, rej = h["H0_grid"][()], h["f_grid"][()], h["log_likelihood"][()], h["guard/rejected"][()]
    with h5py.File(p9j) as h:
        Hj, Fj, Mj, Lj, rejj = h["H0_grid"][()], h["f_grid"][()], h["mu_chi_c2_grid"][()], h["log_likelihood"][()], h["guard/rejected"][()]
    win = (H >= Hj[0] - 1e-9) & (H <= Hj[-1] + 1e-9)
    g9s = GridPost(["H0", "f"], [H[win], F], L[win], [4, 8], extra={"H0": [IN["H0"]], "f": [IN["f"]]}, rejected=rej[win], tag="S9[63,76]")
    g9sfull = GridPost(["H0", "f"], [H, F], L, [4, 8], extra={"H0": [IN["H0"]], "f": [IN["f"]]}, rejected=rej, tag="S9")
    g9j = GridPost(["H0", "f", "dmuChi"], [Hj, Fj, Mj], Lj, [4, 8, 4], extra={"H0": [IN["H0"]], "f": [IN["f"]], "dmuChi": [0.0, IN["dmuChi"]]}, rejected=rejj, tag="J9")
    src9s = src_grid(p9s, ["H0_grid", "f_grid", "log_likelihood"], "H0 restricted to J9's window [63, 76]; spline-refined")
    src9j = src_grid(p9j, ["H0_grid", "f_grid", "mu_chi_c2_grid", "log_likelihood"], "3-D lattice, spline-refined; flat priors H0 [63,76], f [0,1], Delta mu_chi [-0.2,0.25]")
    s9sH = add_post("SpinHPosHzero", g9s.summary("H0"), "km/s/Mpc", "H0 from positions only (spin offset not modelled), spin-offset mock, H0 free (window [63, 76])", src9s, "C9", IN["H0"], parts=True)
    s9jH = add_post("SpinHBothHzero", g9j.summary("H0"), "km/s/Mpc", "H0 with positions and the spin offset modelled, spin-offset mock", src9j, "C9", IN["H0"], parts=True)
    add_post("SpinHPosfAGN", g9s.summary("f"), "", "f_AGN from positions only, H0 free", src9s, "C9", IN["f"])
    add_post("SpinHBothfAGN", g9j.summary("f"), "", "f_AGN with positions and spins, H0 free", src9j, "C9", IN["f"])
    add_post("SpinHBothDmuChi", g9j.summary("dmuChi"), "", "spin-mean offset with positions and spins, H0 free", src9j, "C9", IN["dmuChi"])
    add_num("SpinHzeroWidthRatio", w90(s9jH) / w90(s9sH), 3, "", "90% width of H0 with the spin offset modelled over without, both on H0 in [63, 76]", {"files": [rel(p9j), rel(p9s)]}, "C9")
    add_num("SpinHzeroWidthRatioSixtyEight", (s9jH["ci68"][1] - s9jH["ci68"][0]) / (s9sH["ci68"][1] - s9sH["ci68"][0]), 3, "", "same with 68% widths", {"files": [rel(p9j), rel(p9s)]}, "C9")
    DIAG["c9_S9_full_range_H0"] = clean(g9sfull.summary("H0"))
    DIAG["c9_J9_rejected_cells"] = g9j.n_bad
    d0 = g9j.density_at("dmuChi", 0.0)
    lnB = math.log(1 / 0.45) - math.log(d0)
    add("LBSpinHfree", "anc", flb(lnB / LN10), lnB / LN10, "log10", "log10 Bayes factor of a spin-mean offset against none (Savage-Dickey at Delta mu_chi = 0), H0 free", src_grid(p9j, ["log_likelihood"], "Savage-Dickey on the refined J9 grid"), "C9")
    BAYES.append({"pair": "spin offset vs none, H0 free (J9 vs S9)", "qualifies": True, "macro": "LBSpinHfree", "log10B": lnB / LN10, "how": "Savage-Dickey at Delta mu_chi = 0"})
    zj = g9j.log_int()[0] - math.log(0.45)
    zs = g9s.log_int()[0]
    DIAG["c9_A9_J_vs_S_evidence_ratio_log10"] = (zj - zs) / LN10


# ============================================================================================= #
# placeholders (red)
# ============================================================================================= #


def build_placeholders():
    P = [
        ("NullfAGNUpper", r"\ph{0.0X}", "", "90% upper limit on f_AGN from the combined posterior of simulated catalogs with no AGN-hosted mergers (input f_AGN = 0)",
         "complete-catalog joint (H0, f_AGN) grids on catalogs generated with --f_agn 0 (none exist; the sky-shuffle null of Analysis 2 is a different construction)", "validation"),
        ("NullHzeroComb", r"\ph{XX.X^{+X.X}_{-X.X}}", "km/s/Mpc", "combined H0 posterior from those catalogs", "same runs", "validation"),
        ("NullHzeroLevel", r"\ph{XX}", "%", "credible level of the input H0 in that combined posterior", "same runs", "validation"),
        ("NullLBfZero", r"\ph{-X.X}", "log10", "log10 Bayes factor of the two-population model against galaxy hosts only on those catalogs (Savage-Dickey at f_AGN = 0)", "same runs", "validation"),
        ("NullNcat", r"\ph{XX}", "catalogs", "number of simulated catalogs with no AGN-hosted mergers", "same runs", "validation"),
        ("FluxNcat", r"\ph{XX}", "catalogs", "simulated catalogs analysed with flux limits (only catalog A exists for the anchored-density grids)",
         "Analysis-3 joint grids (c_mode=selection, densities at input) at m<18 (and the other limits) on further catalogs", "C4,validation"),
        ("FluxfCombEighteen", r"\ph{0.XX^{+0.XX}_{-0.XX}}", "", "combined f_AGN posterior at m < 18 over several catalogs, densities at input", "same runs", "C4,validation"),
        ("FluxHzeroCombEighteen", r"\ph{XX.X^{+X.X}_{-X.X}}", "km/s/Mpc", "combined H0 posterior at m < 18 over several catalogs", "same runs", "C4,validation"),
        ("FluxfCombEighteenLevel", r"\ph{XX}", "%", "credible level of the input f_AGN in that combined posterior", "same runs", "C4,validation"),
        ("FluxHzeroCombEighteenLevel", r"\ph{XX}", "%", "credible level of the input H0 in that combined posterior", "same runs", "C4,validation"),
        ("FreeLognGalEighteenComb", r"\ph{-X.XX^{+0.XX}_{-0.XX}}", "log10 Mpc^-3", "combined posterior of log10 n_gal at m < 18 over the three catalogs, both densities free",
         "one nested-sampling run on the summed log-likelihood of catalogs A, B, C (shared H0, f_AGN, n_gal, n_AGN); the three per-catalog chains cannot be multiplied exactly", "C6,validation"),
        ("FreeLognAGNEighteenComb", r"\ph{-X.XX^{+0.XX}_{-0.XX}}", "log10 Mpc^-3", "combined posterior of log10 n_AGN at m < 18 over the three catalogs", "same run", "C6,validation"),
        ("FreefAGNEighteenComb", r"\ph{0.XX^{+0.XX}_{-0.XX}}", "", "combined posterior of f_AGN at m < 18 over the three catalogs, both densities free", "same run", "C6,validation"),
        ("FreeHzeroEighteenComb", r"\ph{XX.X^{+X.X}_{-X.X}}", "km/s/Mpc", "combined posterior of H0 at m < 18 over the three catalogs, both densities free", "same run", "C6,validation"),
        ("FreeLognGalEighteenCombLevel", r"\ph{XX}", "%", "credible level of the input log10 n_gal = -3 in that combined posterior", "same run", "C6,validation"),
        ("HDNcat", r"\ph{XX}", "catalogs", "simulated catalogs with host-dependent masses and spins (one exists)", "host-dependent mocks (--dmu_G_agn 5 --dmu_chi_agn 0.10) on further catalogs, eight-parameter model on each", "C7,C8,validation"),
        ("HDEightHzeroComb", r"\ph{XX.X^{+X.X}_{-X.X}}", "km/s/Mpc", "combined H0 posterior of the eight-parameter model over several host-dependent catalogs", "same runs (combination needs one sampler run on the summed likelihood)", "C8,validation"),
        ("HDEightHzeroCombLevel", r"\ph{XX}", "%", "credible level of the input H0 in that combined posterior", "same runs", "C8,validation"),
        ("HDEightDmuGCombLevel", r"\ph{XX}", "%", "credible level of the input mass-peak offset (+5 Msun) in the combined posterior", "same runs", "C8,validation"),
        ("HDEightDmuChiCombLevel", r"\ph{XX}", "%", "credible level of the input spin-mean offset (+0.10) in the combined posterior", "same runs", "C8,validation"),
        ("CommonHzeroComb", r"\ph{XX.X^{+X.X}_{-X.X}}", "km/s/Mpc", "combined H0 posterior of the common-population model over several host-dependent catalogs (size of the bias)", "common-population (H0, f_AGN) grids on the same host-dependent catalogs", "C7,validation"),
        ("LBEightVsFree", r"\ph{X.X\pm X.X}", "log10", "log10 Bayes factor of the eight-parameter model against the six-parameter model (widths fixed)",
         "both models with nested sampling on one code (darksirens-core pinned), one selection-variance cap and one Delta mu_chi prior box; the stored runs differ in all three", "C8"),
        ("LBEightVsCommon", r"\ph{X.X\pm X.X}", "log10", "log10 Bayes factor of the eight-parameter model against the common population",
         "common-population evidence on the same code, cap and prior box as the eight-parameter run", "C7,C8"),
    ]
    for name, r, units, meaning, run, claim in P:
        add_ph(name, r, units, meaning, run, claim)


# ============================================================================================= #
# writers
# ============================================================================================= #


def render_tex(reg):
    L = [
        "%% results_macros.tex -- written by scripts/build_values.py; do not edit by hand.",
        "%% RED (\\ph):   not computable from the stored results; see placeholders.txt for the run that produces it.",
        "%% TEAL (\\anc): a real number from the exploratory runs under working/analyses/; the production runs replace it.",
        "%% Plain:       design choices and inputs of the simulated catalogs (working/data).",
        "%% Every macro is traced in values/values.json (source file and key, or the computation).",
        "\\definecolor{phred}{rgb}{0.80,0.05,0.05}",
        "\\definecolor{ancteal}{rgb}{0.00,0.45,0.45}",
        "\\newcommand{\\ph}[1]{\\textcolor{phred}{#1}}",
        "\\newcommand{\\anc}[1]{\\textcolor{ancteal}{#1}}",
    ]
    order = [("plain", "simulated catalogs: design choices and inputs (plain)"), ("anc", "exploratory results (teal)"), ("ph", "placeholders (red)")]
    for cls, head in order:
        L += ["", f"%% ---- {head} ----"]
        claim_prev = None
        for r in reg:
            if r["class"] != cls:
                continue
            c = r["claim"].split(",")[0]
            if c != claim_prev:
                L.append(f"%% {c}")
                claim_prev = c
            body = r["render"] if cls != "anc" else f"\\anc{{{r['render']}}}"
            mean = r["meaning"].replace("%", "\\%")
            L.append(f"\\newcommand{{\\{r['name']}}}{{{body}}}  % {mean}")
    return "\n".join(L) + "\n"


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return [jsonable(v) for v in o.tolist()]
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, float) and not math.isfinite(o):
        return str(o)
    return o


def compute():
    REG.clear()
    DIAG.clear()
    TABLES.clear()
    BAYES.clear()
    GRIDLOG.clear()
    build_mock()
    posts, H, F = build_c1_c2()
    build_c3()
    build_c4_c5(posts)
    build_c6()
    build_c7_c8()
    build_c9()
    build_placeholders()
    DIAG["grid_log"] = list(GRIDLOG)
    return REG, DIAG, TABLES, BAYES


def write_all():
    reg, diag, tables, bayes = compute()
    (OUT / "values").mkdir(exist_ok=True)
    (OUT / "values" / "results_macros.tex").write_text(render_tex(reg))
    doc = {"_what": "every macro of values/results_macros.tex with its value(s), class, units, meaning, source and claim; containment records for measurements with a known input",
           "_inputs": IN, "_conventions": __doc__,
           "macros": {r["name"]: jsonable({k: v for k, v in r.items() if k != "name"}) for r in reg},
           "bayes_factor_ledger": jsonable(bayes), "tables": jsonable(tables), "diagnostics": jsonable(diag)}
    (OUT / "values" / "values.json").write_text(json.dumps(doc, indent=1))
    # MACROS.md
    M = ["# Macro menu (written by scripts/build_values.py; do not edit)", "",
         "Class: plain = design choice or mock input; teal = exploratory result (production replaces it); red = not yet computable (see placeholders.txt).",
         "Intervals are medians with 90% equal-tailed intervals unless stated. Full numbers, sources and containment records: values.json.", "",
         "| macro | meaning | rendered | class | claim |", "|---|---|---|---|---|"]
    for r in reg:
        rr = r["render"].replace("|", "\\|")
        M.append(f"| `\\{r['name']}` | {r['meaning'].replace('|', '/')} | ${rr}$ | {dict(plain='plain', anc='teal', ph='red')[r['class']]} | {r['claim']} |")
    (OUT / "values" / "MACROS.md").write_text("\n".join(M) + "\n")
    # placeholders.txt
    ph = [r for r in reg if r["class"] == "ph"]
    P = ["PRD paper: ledger of numbers not yet computable (written by scripts/build_values.py).",
         "Columns: macro | physical meaning | run that produces it (what is missing on disk)", "", f"RED placeholders ({len(ph)}):"]
    for r in ph:
        P.append(f"\\{r['name']} | {r['meaning']} | {r['source']['missing']}")
    (OUT / "placeholders.txt").write_text("\n".join(P) + "\n")
    n = {c: sum(1 for r in reg if r["class"] == c) for c in ("plain", "anc", "ph")}
    print(f"{n['anc']} teal, {n['plain']} plain, {n['ph']} red; {len(reg)} macros")
    return reg


if __name__ == "__main__":
    write_all()
