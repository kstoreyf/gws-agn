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
    * Digits (revision 2, decisions item 4, REVIEW C2/C6): a median with its 90% interval is rounded by the
      Particle Data Group rule applied to the smaller of its two errors (three leading digits 100-354 keep
      two significant figures, 355-949 one, 950-999 two after rounding up); every log10 Bayes factor has at
      most two significant figures, its numerical uncertainty printed only when it reaches half a unit of
      the last printed digit.

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
R2_CACHE: dict = {}


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
# the detected mergers of the reference catalog (realization A, equal populations): the production event set of Analysis 16
EV_REF = DATA / "seed100" / "events" / "events_corepop.h5"
EV_REF_META = DATA / "seed100" / "events" / "events_meta_corepop.json"
# the other production event sets of realization A (analysis_16 prod_common.EVENT_KINDS with the suffix _corepop): host-dependent
# populations, all mergers hosted by galaxies, all hosted by AGN. Since 17 Nov every design value and realised number of the event
# stage is read from these files and their metadata, not from the older sets of META.json's "stages.events" (events.h5 and the
# files without the suffix), which the exploratory analyses read.
EV_HD_META = DATA / "seed100" / "events" / "events_meta_marked_dmu0p10_dmuG5_corepop.json"
EV_PUREGAL = DATA / "seed100" / "events" / "events_puregal_corepop.h5"
EV_PUREAGN = DATA / "seed100" / "events" / "events_pureagn_corepop.h5"


def ev_meta_path(seed):
    """Metadata of the production equal-population event set of realization `seed`."""
    return DATA / f"seed{seed}" / "events" / "events_meta_corepop.json"


def _inputs():
    m = jload(META100)
    ev = jload(EV_REF_META)
    hd = jload(EV_HD_META)
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


def pdg_dec(err):
    """Decimals of an interval by the Particle Data Group rounding rule (revision 2, REVIEW C2/C6):
    the three leading digits of the smaller error decide the significant figures it keeps, 100-354
    two, 355-949 one, 950-999 two after rounding up to the next decade. 0.080 -> 0.08, 0.98 -> 1.0,
    1.7 -> 1.7. No digit beyond what the grid and the Monte Carlo accuracy support."""
    if err <= 0 or not np.isfinite(err):
        return 2
    e = math.floor(math.log10(err))
    lead = err / 10 ** (e - 2)
    if lead >= 999.5:       # floating point at the top of the decade
        e, lead = e + 1, lead / 10
    if lead < 354.5:
        sig = 2
    elif lead < 949.5:
        sig = 1
    else:
        sig, e = 2, e + 1
    return int(max(0, min(5, sig - 1 - e)))


def fnum(x, d):
    s = f"{x:.{d}f}"
    if s.startswith("-") and float(s) == 0.0:
        s = s[1:]
    return s


def fci(med, lo, hi, d=None):
    up, dn = hi - med, med - lo
    if d is None:
        d = pdg_dec(min(up, dn) if min(up, dn) > 0 else max(up, dn))
    return f"{fnum(med, d)}^{{+{fnum(up, d)}}}_{{-{fnum(dn, d)}}}", d


def fsig(x, sig=2, up=False):
    """x to `sig` significant figures as a plain decimal (up=True rounds away from zero: a bound)."""
    if x == 0 or not np.isfinite(x):
        return fnum(x, 2)
    e = math.floor(math.log10(abs(x)))
    d = max(0, sig - 1 - e)
    q = 10.0 ** (e - sig + 1)
    if up:
        x = math.copysign(math.ceil(abs(x) / q - 1e-9) * q, x)
    elif e >= sig:                       # 1455 -> 1500: round the integer part as well
        x = round(x / q) * q
    return fnum(x, d)


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
    """log10 Bayes factor to two significant figures at most (decisions item 4): 2037.9 -> 2.0x10^3,
    27.2 -> 27, 5.97 -> 6.0, 0.54 -> 0.54. The numerical uncertainty is printed only when it reaches
    half a unit of the last printed digit (it is always in values.json)."""
    ax = abs(x)
    if ax >= 100:
        return "\\ensuremath{" + fsci(x, 1) + "}"
    d = 0 if ax >= 10 else (1 if ax >= 1 else 2)
    s = fnum(x, d)
    if err is not None and err >= 0.5 * 10.0 ** (-d):
        s += f"\\pm{fnum(err, d)}"
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
    # the event stage: the metadata of the production event set of realization A (events_meta_corepop.json; until 17 Nov META.json's
    # "stages.events", the record of the older events.h5, whose design values are the same)
    ev = jload(EV_REF_META)
    cat = m["stages"]["catalogs"]
    sur = m["stages"]["surveys"]
    srce = lambda key: {"file": rel(EV_REF_META), "key": key}
    add("MockHzero", "plain", fnum(IN["H0"], 2), IN["H0"], "km/s/Mpc", "input Hubble constant", srce("cosmology.H0"), "mock")
    add("MockOmegam", "plain", fnum(ev["cosmology"]["Om0"], 4), ev["cosmology"]["Om0"], "", "input matter density Omega_m (fixed in every analysis)", srce("cosmology.Om0"), "mock")
    add("MockfAGN", "plain", fnum(IN["f"], 2), IN["f"], "", "input fraction of mergers hosted by AGN", srce("planted_f_agn"), "mock")
    # the detected mergers of the reference catalog: the production event set of realization A (events_corepop.h5, the equal-population
    # set every production analysis of Analysis 16 reads; until 17 Nov the older events.h5 of META.json, 295 AGN-hosted instead of 298)
    evr = ev["realised"]
    rf = evr["realised_f_agn"]
    add("MockfAGNRealised", "plain", fnum(rf, 3), rf, "", "realised AGN-hosted fraction of the 1000 detected events of the reference catalog", srce("realised.realised_f_agn"), "mock")
    ng, na = m["config"]["glass"]["n_comoving_gal"], m["config"]["glass"]["n_comoving_agn"]
    add("MocknGal", "plain", f"10^{{{int(round(math.log10(ng)))}}}", ng, "Mpc^-3", "comoving number density of the galaxy catalog", src("config.glass.n_comoving_gal"), "mock")
    add("MocknAGN", "plain", f"10^{{{int(round(math.log10(na)))}}}", na, "Mpc^-3", "comoving number density of the AGN catalog", src("config.glass.n_comoving_agn"), "mock")
    add("MockLognGal", "plain", fnum(IN["lognGal"], 1), IN["lognGal"], "log10 Mpc^-3", "log10 of the galaxy number density (input of the free-abundance analysis)", src("config.glass.n_comoving_gal"), "mock")
    add("MockLognAGN", "plain", fnum(IN["lognAGN"], 1), IN["lognAGN"], "log10 Mpc^-3", "log10 of the AGN number density", src("config.glass.n_comoving_agn"), "mock")
    # the same in the units the production analyses sample, h^3 Mpc^-3 (Table I gives priors and inputs in one unit): log10 n - 3 log10 h
    lh = 3 * math.log10(IN["H0"] / 100.0)
    for nm, k, what in (("MockLognGalh", "lognGal", "galaxy"), ("MockLognAGNh", "lognAGN", "AGN")):
        v = IN[k] - lh
        add(nm, "plain", fnum(v, 2), v, "log10 h^3 Mpc^-3", f"log10 of the {what} number density in h^3 Mpc^-3, the units in which the production analyses sample it "
            f"(= Mock{k[0].upper() + k[1:]} - 3 log10 h, h = MockHzero/100; the priors PriorLogn...Lo/Hi are in these units)",
            {"file": rel(META100), "key": f"config.glass.n_comoving_{k[4:].lower()}", "computation": f"log10 n - 3 log10(H0/100), H0 = cosmology.H0 of {rel(EV_REF_META)}"}, "mock")
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
    add("MockNev", "plain", str(ev["n_events"]), ev["n_events"], "events", "detected mergers per simulated catalog", srce("n_events"), "mock")
    add("MockNsamp", "plain", str(ev["nsamp"]), ev["nsamp"], "samples", "posterior samples per event", srce("nsamp"), "mock")
    thr = ev["detection"]["snr_threshold"]
    add("MockSNRthr", "plain", fnum(thr, 0), thr, "", "detection threshold on the observed signal-to-noise ratio", srce("detection.snr_threshold"), "mock")
    hz = evr["horizon_z_max_detected"]
    add("MockzHorizon", "plain", fnum(hz, 2), hz, "", "largest true redshift of a detected event, reference catalog (the horizon)", srce("realised.horizon_z_max_detected"), "mock")
    zmed = evr["z_median_detected"]
    add("MockzMedian", "plain", fnum(zmed, 2), zmed, "", "median redshift of the detected events, reference catalog", srce("realised.z_median_detected"), "mock")
    # horizon range over the production realizations of population known with both catalogs (key KnownBoth: the equal-population
    # event sets events_corepop.h5 of RUNLIST.yaml; until 17 Nov the older events.h5 of the nine exploratory catalogs C1_SEEDS)
    import placeholder_results as PR
    hseeds = list(PR.RUNREAL["KnownBoth"])
    hzr = {s: jload(ev_meta_path(s))["realised"]["horizon_z_max_detected"] for s in hseeds}
    srch = {"files": [rel(ev_meta_path(s)) for s in hseeds], "key": "realised.horizon_z_max_detected"}
    for nm, fn, w in (("MockzHorizonMin", min, "smallest"), ("MockzHorizonMax", max, "largest")):
        add(nm, "plain", fnum(fn(hzr.values()), 2), fn(hzr.values()), "", f"{w} horizon (largest true redshift of a detected merger) over the {len(hseeds)} production "
            f"realizations {hseeds} of population known with both catalogs (key KnownBoth; their equal-population event sets)", dict(srch, computation=fn.__name__),
            "mock", extra={"per_realization": hzr})
    # one-sigma sky widths of the 1000 detected mergers of realization A (truth/obs_sigma_ang of the production event set, radians; until
    # 17 Nov META.json's validation check V2 of the older events.h5, the same two numbers)
    with h5py.File(EV_REF) as h:
        sang = np.degrees(h["truth/obs_sigma_ang"][()])
    for nm, v, w in (("MockSkyMin", float(sang.min()), "smallest"), ("MockSkyMax", float(sang.max()), "largest")):
        add(nm, "plain", fnum(v, 1), v, "deg", f"{w} one-sigma sky localisation width of the 1000 detected mergers, reference catalog",
            {"file": rel(EV_REF), "key": "truth/obs_sigma_ang", "computation": f"{'min' if w == 'smallest' else 'max'} over the events, in degrees"}, "mock")
    mm = ev["measurement_model"]
    add("MockSigLnMc", "plain", "0.08", 0.08, "", "chirp-mass error: sigma(ln Mc) = this value times 8/rho_obs", srce("measurement_model.widths.sigma_lnMc ('A_MC = 0.08')"), "mock")
    add("MockSigLnq", "plain", "0.60", 0.60, "", "mass-ratio error: sigma(ln q) = this value times 8/rho_obs", srce("measurement_model.widths.sigma_lnq ('A_Q = 0.6')"), "mock")
    add("MockSigChi", "plain", "0.20", 0.20, "", "effective-spin error: sigma(chi_eff) = this value times 8/rho_obs", srce("measurement_model.widths.sigma_chieff ('A_CHI = 0.2')"), "mock")
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
    add("MockAlpha", "plain", fnum(p["alpha"], 1), p["alpha"], "", "power-law index of the primary mass", srce("population.alpha"), "mock")
    add("MockMmin", "plain", fnum(p["mmin"], 0), p["mmin"], "Msun", "minimum mass", srce("population.mmin"), "mock")
    add("MockMmax", "plain", fnum(p["mmax"], 0), p["mmax"], "Msun", "maximum mass", srce("population.mmax"), "mock")
    add("MockPeakFrac", "plain", fnum(p["peak_fraction"], 2), p["peak_fraction"], "", "fraction of the primary-mass distribution in the Gaussian peak", srce("population.peak_fraction"), "mock")
    add("MockMuG", "plain", fnum(p["peak_mu"], 0), p["peak_mu"], "Msun", "location of the Gaussian mass peak (galaxy hosts; both host classes in the common-population mocks)", srce("population.peak_mu"), "mock")
    add("MockSigmaG", "plain", fnum(p["peak_sigma"], 0), p["peak_sigma"], "Msun", "width of the Gaussian mass peak", srce("population.peak_sigma"), "mock")
    add("MockBeta", "plain", fnum(p["beta"], 0), p["beta"], "", "power-law index of the mass ratio", srce("population.beta"), "mock")
    add("MockMuChi", "plain", fnum(p["chi_mu"], 0), p["chi_mu"], "", "mean effective spin (galaxy hosts)", srce("population.chi_mu"), "mock")
    add("MockSigmaChi", "plain", fnum(p["chi_sigma"], 1), p["chi_sigma"], "", "width of the effective-spin distribution", srce("population.chi_sigma"), "mock")
    add("MockGammaRate", "plain", fnum(ev["gamma"], 0), ev["gamma"], "", "redshift evolution index of the merger rate, (1+z)^gamma", srce("gamma"), "mock")
    # the host-dependent event set of realization A that the production runs analyse (Analysis 16, stage 4 and the
    # experiment's s1 sets: events_marked_dmu0p10_dmuG5_corepop.h5); until 17 Nov the older events_marked_dmu0p10_dmuG5 set
    hdp = EV_HD_META
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
    add("LBfZeroMin", "anc", flb(min(lbz)), min(lbz), "log10", "smallest per-catalog log10 B (two populations against galaxy hosts only) over the 9 catalogs", srcs, "C2")
    add("LBfZeroMax", "anc", flb(max(lbz)), max(lbz), "log10", "largest per-catalog log10 B (two populations against galaxy hosts only)", srcs, "C2")
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
    R2_CACHE["C10J_marg_dmuChi"] = gJ.marg1("dmuChi")
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
            sfb = ctrl[a].summary("f")
            add_post(nm.replace("Hzero", "fBranch"), sfb, "", "DEPRECATED alias kept for the v1 text, use \\CtrlHeavierFrac (REVIEW C1): fraction of mergers in the offset population, " + cdesc[a], src, "C7", IN["f"])
            if a == "B5M":
                add_post("CtrlHeavierFrac", sfb, "", "fraction of mergers in the heavier population (mass peak +5 Msun, spin mean +0.10) of the galaxy-hosts-only control mock, "
                         "offsets fixed at their inputs; NOT f_AGN: no merger of this mock is hosted by an AGN, the mixture weight is the share of the offset population "
                         "(input 0.30, drawn independently of the host); print it under its own symbol in Table III (REVIEW C1, decisions item 11)", src, "C7,C1", IN["f"])
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
    add_num("SpinfWidthRatio", w90(sJf) / w90(sSf), 2, "", "90% width of f_AGN with the spin offset modelled over without (positions only), fixed H0", {"files": [rel(pJ), rel(pS)]}, "C9")
    add_num("SpinfWidthRatioSixtyEight", (sJf["ci68"][1] - sJf["ci68"][0]) / (sSf["ci68"][1] - sSf["ci68"][0]), 2, "", "68% width of f_AGN with the spin offset modelled over without, fixed H0", {"files": [rel(pJ), rel(pS)]}, "C9")
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
    add_num("SpinHzeroWidthRatio", w90(s9jH) / w90(s9sH), 2, "", "90% width of H0 with the spin offset modelled over without, both on H0 in [63, 76]", {"files": [rel(p9j), rel(p9s)]}, "C9")
    add_num("SpinHzeroWidthRatioSixtyEight", (s9jH["ci68"][1] - s9jH["ci68"][0]) / (s9sH["ci68"][1] - s9sH["ci68"][0]), 2, "", "same with 68% widths", {"files": [rel(p9j), rel(p9s)]}, "C9")
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
# REVISION 2 (decisions of 17 Nov 2026, items 3, 4, 11): design values, the host normalisation and
# the rate per host, completeness weighted by the detected mergers, hosts per localization
# volume, and the Analysis 14 / Analysis 15 summaries (red until those files exist).
# ============================================================================================= #
A14R = AN / "analysis_14_monte_carlo_accuracy" / "results"
A15R = AN / "analysis_15_host_normalisation" / "results"
A16 = AN / "analysis_16_production_runs"
A14_SUMMARY = A14R / "mc_accuracy_summary.json"     # written by scripts/aggregate_mc_accuracy.py at the end of the job


def prod_common():
    """analysis_16's prod_common (priors, survey paths, densities from counts); numpy only, no jax."""
    import importlib
    import sys as _sys
    p = str(A16 / "scripts")
    if p not in _sys.path:
        _sys.path.insert(0, p)
    return importlib.import_module("prod_common")
A15_SUMMARY = A15R / "normalisation_summary.json"   # written by scripts/aggregate_normalisation.py at the end of the job
A15_COMMON = AN / "analysis_15_host_normalisation" / "scripts" / "a15_common.py"
FITDIR = AN / "selection_redo" / "scripts" / "selection_fits_truez"
C_KMS = 299792.458
OM0 = float(jload(EV_REF_META)["cosmology"]["Om0"])
ZMAX_GRID = 5.0     # end of darksirens' internal redshift grid in every stored run (LIKELIHOOD_MATH sec. 0)
ZMAX_PROD = float(re.search(r"^ZMAX_R = ([0-9.]+)", A15_COMMON.read_text(), flags=re.M).group(1))   # 1.10
RUNGS_ALL = [("complete", "Complete", None)] + RUNGS
KEY_DEFINITIONS = """## The key scheme (notes/PAPER_SKELETON.md Sec. 6; values/placeholder_results.json)

- `\\<Key><Quantity>`: realization A, median and 90% equal-tailed interval. `\\<Key><Quantity>Comb`: the combination over
  realizations (log-likelihoods summed, parameters shared). `\\<Key><Quantity>CL`: two-sided credible level (percent) at which
  the input lies in the combination. Pieces `Med`, `Lo`, `Hi`, `Wid` (full 90% width) after the quantity or after `Comb`.
  `UL`: 90% upper limit (90th percentile), for `CtrlZerofAGN` and the number densities; `PriorUL`: the limit the priors
  allow once the rate per host is measured (prior-induced density), beside `UL`, realization A, number densities inferred.
  Units of the number densities: `Free<limit>Logn<Gal|AGN>[Comb][...]`, `...UL`, `...PriorUL` are log10 of n in Mpc^-3, each
  posterior sample (and prior draw) converted with its own H0, log10 n + 3 log10(H0/100); `Free<limit>Logn<Gal|AGN>ULh` and
  `...PriorULh` (realization A) are the same two limits in h^3 Mpc^-3, the unit the production analyses sample and the text quotes
  (Table I, `\\MockLognGalh`, `\\MockLognAGNh`): the 90th percentile of the run's own samples of its parameter and the weighted
  90th percentile of the prior draws, no conversion. While a key is a placeholder its exploratory run (which sampled Mpc^-3) is
  converted per sample with its own H0 the other way, log10 n - 3 log10(H0/100), and reweighted to the production prior: only
  the samples and prior draws with both densities inside the Table I box in h^3 Mpc^-3 are kept, and every quantity of the key
  is computed from them, so every red limit and the region drawn in Fig. 4(a) lie inside the stated prior (since 17 Nov evening,
  REVIEW_2 1.7; rule in the header of `scripts/placeholder_results.py`).
- Re-anchored placeholders (since 17 Nov evening, REVIEW_2 1.11): while `KnownBoth` is a production result and a key
  `Flux<limit>` or `Free<limit>` is not, its H0, f_AGN and rate ratio of realization A are the exploratory ones moved onto the
  finished complete-catalog result: median = `KnownBoth` median + (exploratory median - exploratory complete median), every
  quantile moved by x -> new median + s (x - exploratory median), s = `KnownBoth` 90% half-width / exploratory complete 90%
  half-width (the 90% half-width is the `KnownBoth` one times the exploratory ratio of this key's to the complete one). The
  combination follows from the re-anchored realization A by the invented-combination rule; the map of each quantity is in
  `placeholder_results.json` ("reanchor"). They stay red.
- Keys: `KnownBoth`, `KnownGal`, `KnownAGN`, `KnownNone`, `AGNhosted`, `Galhosted`, `CtrlZero`, `CtrlTenth`,
  `Flux<limit>`, `Free<limit>` (limit `TwentyOne` ... `Eighteen`), `SingleBoth`, `SingleNone`, `HDSingle`, `HDHost`, `HDNone`,
  `ObsSingle`, `ObsHost`, `ObsNone`; each key's events, analysis and catalogs are in the meaning column below and in
  PAPER_SKELETON Sec. 6. Quantities: `Hzero`, `fAGN`, `MuG`, `SigmaG`, `MuChi`, `SigmaChi`, `DmuG`, `DmuChi`, `LognGal`,
  `LognAGN`, `RateRatio`, `LB`.
- `LB` (log10 Bayes factor) references: `KnownBothLB`, `CtrlZeroLB`, `CtrlTenthLB` against galaxy hosts only (f_AGN = 0,
  Savage-Dickey, f_AGN prior [0, 1]); `HDHostLB`, `ObsHostLB` (= `\\HDLB`, `\\ObsLB`, realization A, 1000 mergers) against
  the single-population analysis of the same events with the priors of Table I. `...LBComb`: the same for the combination
  (sum over the combined realizations).
- `HalfWid` (realization A only, `Hzero` and `fAGN`): half of `Wid`, the precision of one catalog of 1000 mergers.
- Further: `\\HDBiasHzero`, `\\ObsBiasHzero` (median of `<pre>SingleHzeroComb` minus the input); `\\WidthRatioAGNGal`
  (= `\\GalhostedHzeroWid` / `\\AGNhostedHzeroWid`, realization A); `\\KnownBothCorrHzerofAGN[Comb]` (= `\\JointRhoCatA`, `\\JointRhoComb`);
  `\\MassScaleShareKnown`, `\\MassScaleShareSingle` (H0 width, no catalog over both catalogs, realization A);
  `\\CorrHzeroMuGNone`, `\\CorrHzeroMuGBoth`.
- Colour: red (`\\ph`) while a placeholder; the source of each value (exploratory with its file, or invented) is in
  placeholders.txt. When values/production_map.json maps a key to an existing production summary, its macros print black.
- Inputs used for the credible levels: H0 67.74, f_AGN 0.30 (0 and 0.10 for the two control sets), mass peak 35 and width
  5 Msun, spin mean 0 and width 0.10, offsets +5 Msun and +0.10, log10 n = -3 and -5, rate-per-host ratio `\\MockRateRatio`;
  for the `Obs...` keys the mass peak and width of the observed-like spectrum, `\\ObsMuG` and `\\ObsSigmaG`.
  The single-population analysis of host-dependent events has no single input for its population parameters (no `CL`).
"""
R2_DEFINITIONS = """## Definitions of the revision-2 macros (17 Nov 2026)

- **Rate per host, `RateRatio*`, `NormRateRatio*`, `MockRateRatio`.** R = r_AGN / r_gal = [f_AGN/(1 - f_AGN)] Z_gal/Z_AGN (Eq. rates; LIKELIHOOD_MATH sec. 8), with Z_k the normaliser of the convention the analysis used. The f_AGN posterior is transformed, never its summary: for complete catalogs (log10 n_k = -24, Analyses 2 and 8-13) Z_k = N_k, so the quantiles of R are those of f_AGN times N_gal/N_AGN (combined posterior of the 9 catalogs: geometric mean of their 9 count ratios, which differ by 0.26%); for the flux-limited grids (Analyses 3, 4) R depends on H0 through Z_k, and its CDF is summed over the H0 nodes of the refined (H0, f) posterior; for the free-density chains (Analysis 5) R is computed per sample with that sample's H0, n_gal, n_AGN; for Analysis 15 from each scan's own Z_k per H0 row. Input: `MockRateRatio` = [0.30/0.70] N_gal/N_AGN of the reference catalogs.
- **Normaliser, `NormF<class><config>[Prod]`.** Z_k(H0) = N_k + n_k 4 pi int_0^zmax [1 - C_k(z)] dV_c/(dz dOmega) dz; N_k = the cataloged count of the survey file (sum of ngals), C_k the Schechter selection completeness the run pinned (h-scaled), n_k the input density, H0 = 67.74; F_k = N_k/Z_k is the cataloged share of the normalisation volume. No suffix: zmax = 5 (every stored run). `Prod`: zmax = 1.10 (production convention, decisions item 3). `Complete`: the nominal-limit representation of complete catalogs used by Analysis 3 (m_lim 24 GAL, 25 AGN); `NormFComplete`: Analyses 0-2 and 8-13, where Z_k = N_k.
- **Eq. (8), `SlopefAGNPred*`.** d f_AGN / d log10 n_AGN = ln10 (1 - F_AGN) f (1 - f). `SlopefAGNIdentity*`, `NormDelta*Pred`: the exact identity applied to the stored factor-1 grid of Analysis 3 (cubic spline of ln L in f per H0 row, logit f' = logit f + ln[F_AGN(x; H0)/F_AGN(1; H0)]), summaries of the mapped grid, least-squares slope of the medians against log10 x at the seven factors of Analysis 4.
- **Redshift to which a flux limit is complete, `MockzComplete*`.** The redshift at which the limit reaches the faint end of the luminosity function (L = x_cut L*): DM(z; H0 = 100) = m_lim - M*_hat - Delta M_faint, from the selection fit; C(z) = 1 below it. The galaxy fit is printed (the AGN fit agrees to 0.002; the lowest true redshift of an AGN removed by the limit agrees to 0.0012, audit G2).
- **Completeness weighted by the detected mergers, `MockEventComp*`.** Percentage of the 1000 detected mergers of the reference catalog whose host object is in the flux-limited catalog of its class (host apparent magnitude below the limit; audit G3 checks membership in the catalog files). The selection model averaged over the true event redshifts is in values.json (agrees to 0.5%). `MockEventFracComplete*`: percentage of those mergers with true redshift below `MockzComplete*`. The volume-weighted numbers inside the horizon are `MockComp*`.
- **Hosts per 90% localization volume, `Pure<class>HostsPerVol*`.** For each detected merger: a sky disc around the mean direction of its posterior samples whose radius holds sqrt(0.9) = 94.87% of the samples, times the redshift interval (distance samples converted at H0 = 67.74) holding the central 94.87%; sky and distance errors are independent in the mock, so the region holds 90% of the posterior. Count of cataloged objects (catalog position and z_obs) of the complete catalog inside it; median over the 1000 events of the production all-galaxy-hosted and all-AGN-hosted event sets of realization A (`events_puregal_corepop.h5`, `events_pureagn_corepop.h5`, the events of keys `Galhosted` and `AGNhosted`; galaxy catalog on the galaxy-hosted set, AGN catalog on the AGN-hosted set; the crossed counts are the `...Events` macros). Mean, percentiles and the density-only expectation n V in values.json.
- **Prior-induced densities, `FreeLogn*InducedUpper*`, `FreeLognAGNPeakOverInduced`.** 2e6 draws from the priors of the free-density runs, weighted by the stored Analysis-3 likelihood at the input densities mapped through the identity (what the mergers say through the normalisers alone); the free-density posterior adds the information of events that lie where a catalog is incomplete. `FreeLogn*Upper*`: 90th percentile of the posterior samples.
- **Analysis 14, `MC<set><quantity><Ns>`.** set: `Mixed` (events.h5, seed 100, (H0, f_AGN)) or `AGN` (all-AGN-hosted set, seed 103, H0 at f_AGN = 1); Ns: `TwoK`, `EightK`, `ThirtyTwoK` posterior samples per event. `N`: redraws; `HzeroSd`, `fSd`: sd (ddof 1) of the posterior median over redraws; `...SdRel`: that sd over the mean 68% half-width; `...Med`: mean median; `LnLVar`: variance of the ln L estimator at the reference cell (posterior-sample term plus N^2/N_eff), mean over redraws; `SecPerCall`: median seconds per likelihood call; `Inj...`: the same over injection sets at 8000 samples. `MCShiftMedian(AGN)` and `MCWidthChange(AGN)`: sd of the medians over the 90% half-width and sd of the 90% widths over their mean, at 2000 samples, in percent (larger of H0 and f_AGN for the mixed set). Red until results/mc_accuracy_summary.json exists.
- **Analysis 15, `Norm<Hzero|f|RateRatio><scan>`, `NormDelta<f|Hzero>Two<config>[Pred]`, `SlopefAGNDepthNorm[Pred]`.** Medians and 90% intervals as the aggregation computed them (trapezoid on the scanned cells); containment levels from the scan files; Delta = median at twice the input AGN density minus at the input; Pred = the identity's prediction from the aggregation. Red until results/normalisation_summary.json exists. `NormDelta*Two<limit>Five`: the same response normalised to z = 5, from Analyses 3 and 4 (teal now).
- **Rounding.** Medians with intervals: PDG rule on the smaller error. log10 B: at most two significant figures.
"""
R2_SRC_NORM = ("Z_k(H0) = N_k + n_k 4 pi int_0^zmax [1 - C_k(z)] dV_c/(dz dOmega) dz (LIKELIHOOD_MATH sec. 4); N_k = sum of ngals of the survey "
               "file the run read; C_k the Schechter selection completeness of the fit file the run pinned (h-scaled, independent of H0); "
               "flat LCDM, Om0 from META; trapezoid on 40001 log-spaced nodes; F_k = N_k / Z_k")


# ---------------------------------------------------------------- cosmology, completeness, Z_k
_COS: dict = {}


def _cos100():
    """z, comoving distance [Mpc] and E(z) at H0 = 100 on a fine table to z = 6."""
    if "t" not in _COS:
        from scipy.integrate import cumulative_trapezoid
        z = np.expm1(np.linspace(0.0, math.log1p(6.0), 120001))
        E = np.sqrt(OM0 * (1 + z) ** 3 + 1 - OM0)
        _COS["t"] = (z, C_KMS / 100.0 * cumulative_trapezoid(1.0 / E, z, initial=0.0), E)
    return _COS["t"]


def chi_mpc(z, H0):
    zt, chi, _ = _cos100()
    return np.interp(z, zt, chi) * 100.0 / H0


def dvdz_sr(z, H0):
    """Comoving volume per unit redshift and steradian [Mpc^3]."""
    E = np.sqrt(OM0 * (1 + np.asarray(z)) ** 3 + 1 - OM0)
    return chi_mpc(z, H0) ** 2 * (C_KMS / H0) / E


def z_of_dl(dl, H0):
    zt, chi, _ = _cos100()
    return np.interp(dl, (1 + zt) * chi * 100.0 / H0, zt)


def _upper_gamma(s, x):
    from scipy.special import gamma, gammaincc
    if s > 0:
        return gamma(s) * gammaincc(s, x)
    return (gamma(s + 1.0) * gammaincc(s + 1.0, x) - x ** s * np.exp(-x)) / s


def load_fit(path):
    s = jload(path)["strata"][0]
    assert s["family"] == "schechter", path
    return {k: float(s[k]) for k in ("m_lim", "Mstar_hat", "alpha", "M_faint_offset")} | {"file": rel(path)}


def c_sel(z, fit):
    """Selection completeness C(z) of a Schechter fit, as darksirens c_mode=selection evaluates it."""
    z = np.atleast_1d(np.asarray(z, float))
    dl = (1 + z) * chi_mpc(z, 100.0)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        dm = 5 * np.log10(dl) + 25.0
        xl = 10.0 ** (-0.4 * (fit["m_lim"] - dm - fit["Mstar_hat"]))
        xf = 10.0 ** (-0.4 * fit["M_faint_offset"])
        C = _upper_gamma(fit["alpha"] + 1.0, xl) / _upper_gamma(fit["alpha"] + 1.0, xf)
    C = np.where(dl > 0, np.nan_to_num(C, nan=1.0, posinf=1.0), 1.0)
    return np.clip(C, 0.0, 1.0)


def z_complete(fit):
    """Redshift to which the catalog is complete: C = 1 exactly while the limit reaches the faint end of
    the luminosity function, i.e. until DM(z) = m_lim - M*_hat - Delta M_faint (h-scaled, H0 = 100)."""
    from scipy.optimize import brentq
    target = fit["m_lim"] - fit["Mstar_hat"] - fit["M_faint_offset"]
    return float(brentq(lambda z: 5 * math.log10((1 + z) * float(chi_mpc(z, 100.0))) + 25.0 - target, 1e-5, 5.5, xtol=1e-12))


def vmiss100(fit, zmax):
    """4 pi int_0^zmax (1 - C) dV_c/dz dz at H0 = 100 [Mpc^3]; scales as (100/H0)^3."""
    z = np.expm1(np.linspace(0.0, math.log1p(zmax), 40001))
    y = (1.0 - c_sel(z, fit)) * dvdz_sr(z, 100.0)
    return float(4 * math.pi * np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(z)))


def survey_nobs(path):
    with h5py.File(path, "r") as h:
        return int(np.sum(h["ngals"][()].astype(np.int64)))


_NORM: dict = {}


def norm_cfg(seed=100):
    """Per rung and class: fit, cataloged count N, missing volumes to z = 5 and to the production end, z_complete."""
    if seed not in _NORM:
        out = {}
        for key, word, _ in RUNGS_ALL:
            rec = {}
            for cls in ("gal", "agn"):
                fit = load_fit(FITDIR / f"fit_{cls}_{key}_schechter.json")
                sp = DATA / f"seed{seed}" / "surveys" / f"survey_{cls}_{key}_ns32.h5"
                rec[cls] = {"fit": fit, "N": survey_nobs(sp), "survey": rel(sp), "V5": vmiss100(fit, ZMAX_GRID),
                            "Vp": vmiss100(fit, ZMAX_PROD), "zc": z_complete(fit)}
            out[key] = rec
        _NORM[seed] = out
    return _NORM[seed]


def Zk(rec, n0, H0, zmax=ZMAX_GRID):
    V = rec["V5"] if zmax == ZMAX_GRID else rec["Vp"]
    return rec["N"] + n0 * V * (100.0 / np.asarray(H0, float)) ** 3


def Fk(rec, n0, H0, zmax=ZMAX_GRID):
    return rec["N"] / Zk(rec, n0, H0, zmax)


def logit(f):
    f = np.asarray(f, float)
    with np.errstate(divide="ignore"):
        return np.log(f) - np.log1p(-f)


def expit(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, float)))


# ---------------------------------------------------------------- the rate per host
def rate_ratio_constant(s, ratio, x0=None, kind="grid"):
    """R = [f/(1-f)] Z_gal/Z_AGN with an H0-independent Z ratio: the quantiles of f map one to one."""
    odds = lambda f: f / (1.0 - f)
    out = {"median": odds(s["median"]) * ratio, "ci68": [odds(v) * ratio for v in s["ci68"]], "ci90": [odds(v) * ratio for v in s["ci90"]],
           "z_ratio": ratio}
    if kind == "grid":
        x, cdf = s["_cdf"]
        out["_cdf"] = (odds(np.clip(x, 0, 1 - 1e-15)) * ratio, cdf)
    else:
        out["_samples"] = odds(s["_samples"]) * ratio
    return out


def rate_ratio_grid(g, zratio_of_H0):
    """R = [f/(1-f)] Z_gal(H0)/Z_AGN(H0) on a refined (H0, f) GridPost: at each H0 node R is monotone in f, so
    P(R <= r) = sum_H0 w(H0) P(f <= f_r(H0) | H0), with the conditional CDF of f by the trapezoid rule on the
    refined lattice; evaluated on a fine logarithmic R lattice and inverted."""
    kH, kf = g.idx("H0"), g.idx("f")
    assert (kH, kf) == (0, 1) and len(g.names) == 2
    xH, xf = g.fine[kH], g.fine[kf]
    P = g.P
    pH = (P @ g.w[kf]) * g.w[kH]
    pH /= pH.sum()
    cdf_f = np.concatenate([np.zeros((len(xH), 1)), np.cumsum(0.5 * (P[:, 1:] + P[:, :-1]) * np.diff(xf)[None, :], axis=1)], axis=1)
    cdf_f /= cdf_f[:, -1:]
    zr = np.asarray(zratio_of_H0(xH), float)
    fm = (pH[:, None] * cdf_f).sum(0)
    flo, fhi = float(np.interp(1e-7, fm, xf)), float(np.interp(1 - 1e-7, fm, xf))
    odds = lambda f: f / (1 - f)
    Rlo = odds(min(max(flo, 1e-6), 0.5)) * zr.min() / 1.5
    Rhi = odds(max(min(fhi, 1 - 1e-6), 0.5)) * zr.max() * 1.5
    r = np.geomspace(Rlo, Rhi, 20001)
    F = np.zeros_like(r)
    lr = np.log(r)
    for i in np.where(pH > 0)[0]:
        F += pH[i] * np.interp(expit(lr - math.log(zr[i])), xf, cdf_f[i])
    q = lambda u: float(np.interp(u, F, r))
    return {"median": q(0.5), "ci68": [q(0.16), q(0.84)], "ci90": [q(0.05), q(0.95)], "_cdf": (r, F),
            "z_ratio_range_over_H0_nodes": [float(zr.min()), float(zr.max())]}


def rate_ratio_samples(Rs):
    q = np.percentile(Rs, [5, 16, 50, 84, 95])
    return {"median": float(q[2]), "ci68": [float(q[1]), float(q[3])], "ci90": [float(q[0]), float(q[4])], "_samples": np.asarray(Rs)}


RR_MEANING = ("merger rate per AGN relative to the merger rate per galaxy, r_AGN/r_gal = [f_AGN/(1 - f_AGN)] Z_gal/Z_AGN (Eq. rates; LIKELIHOOD_MATH sec. 8), "
              "the f_AGN posterior transformed with the normalisers Z_k of the convention the analysis used")


def add_rr(name, R, what, src, claim, x0):
    add_post(name, R, "", RR_MEANING + "; " + what, src, claim, x0, kind="grid" if "_cdf" in R else "chain")


# ---------------------------------------------------------------- design values (plain)
def build_design():
    srcf = lambda p, key: {"file": rel(p), "key": key}
    # prior ranges, read from the runs that used them
    with h5py.File(A2 / "joint_s100.h5") as h:
        Hg, Fg = h["H0_grid"][()], h["f_grid"][()]
    j5 = jload(A5 / "campaign_m18_dynesty_s100.json")["priors"]
    box13 = {b[0]: (b[1], b[2]) for b in jload(A13 / "a13core_e7c3007_rslice_cap20_historical_n200_s1.json")["box"]}
    box11 = {b[0]: (b[1], b[2]) for b in jload(A11 / "a11_11D.json")["box"]}
    with h5py.File(A8 / "arm_J_joint.h5") as h:
        M8 = h["mu_chi_c2_grid"][()]
    P = [("PriorHzeroLo", Hg[0], 0, "lower end of the H0 prior, two-parameter and free-density analyses", srcf(A2 / "joint_s100.h5", "H0_grid[0]")),
         ("PriorHzeroHi", Hg[-1], 0, "upper end of the H0 prior, two-parameter and free-density analyses", srcf(A2 / "joint_s100.h5", "H0_grid[-1]")),
         ("PriorfLo", Fg[0], 0, "lower end of the f_AGN prior", srcf(A2 / "joint_s100.h5", "f_grid[0]")),
         ("PriorfHi", Fg[-1], 0, "upper end of the f_AGN prior", srcf(A2 / "joint_s100.h5", "f_grid[-1]")),
         ("PriorHDHzeroLo", box13["H0"][0], 0, "lower end of the H0 prior of the exploratory host-dependent models (production: PriorHzeroLo)", srcf(A13 / "a13core_e7c3007_rslice_cap20_historical_n200_s1.json", "box H0")),
         ("PriorHDHzeroHi", box13["H0"][1], 0, "upper end of the H0 prior of the exploratory host-dependent models", srcf(A13 / "a13core_e7c3007_rslice_cap20_historical_n200_s1.json", "box H0")),
         ("PriorLognGalLo", j5["log10n0"][0], 0, "lower end of the log10 n_gal prior [Mpc^-3]", srcf(A5 / "campaign_m18_dynesty_s100.json", "priors.log10n0")),
         ("PriorLognGalHi", j5["log10n0"][1], 0, "upper end of the log10 n_gal prior", srcf(A5 / "campaign_m18_dynesty_s100.json", "priors.log10n0")),
         ("PriorLognAGNLo", j5["log10n0_c2"][0], 0, "lower end of the log10 n_AGN prior", srcf(A5 / "campaign_m18_dynesty_s100.json", "priors.log10n0_c2")),
         ("PriorLognAGNHi", j5["log10n0_c2"][1], 0, "upper end of the log10 n_AGN prior", srcf(A5 / "campaign_m18_dynesty_s100.json", "priors.log10n0_c2")),
         ("PriorMuGLo", box13["mu_G"][0], 0, "lower end of the reference mass-peak location prior [Msun]", srcf(A13 / "a13core_e7c3007_rslice_cap20_historical_n200_s1.json", "box mu_G")),
         ("PriorMuGHi", box13["mu_G"][1], 0, "upper end of the reference mass-peak location prior [Msun]", srcf(A13 / "a13core_e7c3007_rslice_cap20_historical_n200_s1.json", "box mu_G")),
         ("PriorDmuGLo", box13["dmu_G"][0], 0, "lower end of the mass-peak offset prior [Msun]", srcf(A13 / "a13core_e7c3007_rslice_cap20_historical_n200_s1.json", "box dmu_G")),
         ("PriorDmuGHi", box13["dmu_G"][1], 0, "upper end of the mass-peak offset prior [Msun]", srcf(A13 / "a13core_e7c3007_rslice_cap20_historical_n200_s1.json", "box dmu_G")),
         ("PriorSigmaGLo", box13["sigma_G"][0], 0, "lower end of the mass-peak width prior [Msun]", srcf(A13 / "a13core_e7c3007_rslice_cap20_historical_n200_s1.json", "box sigma_G")),
         ("PriorSigmaGHi", box13["sigma_G"][1], 0, "upper end of the mass-peak width prior [Msun]", srcf(A13 / "a13core_e7c3007_rslice_cap20_historical_n200_s1.json", "box sigma_G")),
         ("PriorMuChiLo", box13["mu_chi"][0], 1, "lower end of the reference spin-mean prior", srcf(A13 / "a13core_e7c3007_rslice_cap20_historical_n200_s1.json", "box mu_chi")),
         ("PriorMuChiHi", box13["mu_chi"][1], 1, "upper end of the reference spin-mean prior", srcf(A13 / "a13core_e7c3007_rslice_cap20_historical_n200_s1.json", "box mu_chi")),
         ("PriorDmuChiLo", box13["dmu_chi"][0], 2, "lower end of the spin-mean offset prior (all host-dependent models)", srcf(A13 / "a13core_e7c3007_rslice_cap20_historical_n200_s1.json", "box dmu_chi")),
         ("PriorDmuChiHi", box13["dmu_chi"][1], 2, "upper end of the spin-mean offset prior, eight-parameter model", srcf(A13 / "a13core_e7c3007_rslice_cap20_historical_n200_s1.json", "box dmu_chi")),
         ("PriorDmuChiHiSix", box11["dmu_chi"][1], 2, "upper end of the spin-mean offset prior, six-parameter model and the grid evidences of LBOffsetsVsCommon; also the common prior range of Fig. 5(b)", srcf(A11 / "a11_11D.json", "box dmu_chi")),
         ("PriorSigmaChiLo", box13["sigma_chi"][0], 2, "lower end of the spin-width prior", srcf(A13 / "a13core_e7c3007_rslice_cap20_historical_n200_s1.json", "box sigma_chi")),
         ("PriorSigmaChiHi", box13["sigma_chi"][1], 0, "upper end of the spin-width prior", srcf(A13 / "a13core_e7c3007_rslice_cap20_historical_n200_s1.json", "box sigma_chi")),
         ("PriorSpinDmuChiLo", M8[0], 1, "lower end of the spin-mean offset prior in the spin-offset mock (Analyses 8, 9)", srcf(A8 / "arm_J_joint.h5", "mu_chi_c2_grid[0]")),
         ("PriorSpinDmuChiHi", M8[-1], 2, "upper end of the spin-mean offset prior in the spin-offset mock", srcf(A8 / "arm_J_joint.h5", "mu_chi_c2_grid[-1]"))]
    # Table I prints the production ranges (MACRO_REQUESTS B, 17 Nov): one range per parameter in every production run,
    # read from analysis_16's prod_common.PRIORS; the exploratory ranges stay in values.json (extra) and in the *Six/*HD/*Spin macros
    PC = prod_common()
    prod = {"PriorHzero": ("H0", 0), "Priorf": ("f_agn", 0), "PriorLognGal": ("log10n_gal", 0), "PriorLognAGN": ("log10n_agn", 0), "PriorMuG": ("mu_G", 0),
            "PriorDmuG": ("dmu_G", 0), "PriorSigmaG": ("sigma_G", 0), "PriorMuChi": ("mu_chi", 1), "PriorDmuChi": ("dmu_chi", 2), "PriorSigmaChi": ("sigma_chi", 2)}
    for name, v, d, meaning, src in P:
        stem, end = name[:-2], name[-2:]
        if stem in prod and end in ("Lo", "Hi"):
            pn, dd = prod[stem]
            pv = float(PC.PRIORS[pn][0 if end == "Lo" else 1])
            if stem == "PriorSigmaChi" and end == "Hi":
                dd = 0
            add(name, "plain", fnum(pv, dd), pv, "log10 h^3 Mpc^-3" if "Logn" in stem else "",
                meaning.replace("two-parameter and free-density analyses", "every production run") + " (production range, every run; Analysis 16)",
                {"file": rel(A16 / "scripts" / "prod_common.py"), "key": f"PRIORS['{pn}']", "computation": "read"}, "design,TableI",
                extra={"exploratory_range_value": float(v), "exploratory_source": src})
            continue
        add(name, "plain", fnum(float(v), d), float(v), "", meaning, src, "design,TableI")
    add("NormZmax", "plain", fnum(ZMAX_GRID, 0), ZMAX_GRID, "", "upper redshift of the host normalisation in every stored run (end of darksirens' internal redshift grid; no driver sets DARKSIRENS_ZMAX)",
        {"file": "paper_prd/notes/LIKELIHOOD_MATH.md", "key": "sections 0 and 4 (z_max = 5, confirmed numerically in sec. 8.3)", "computation": "code default"}, "design,C4")
    add("NormZmaxProd", "plain", fnum(ZMAX_PROD, 2), ZMAX_PROD, "", "upper redshift of the host normalisation in the production convention (redshift grid ending just above the reach of the detected injections; decisions item 3, Analysis 15)",
        {"file": rel(A15_COMMON), "key": "ZMAX_R"}, "design,C4")
    facs = []
    for tag in ("a05", "a20"):
        with h5py.File(A4 / f"joint_m18_{tag}_s100.h5") as h:
            facs.append(10 ** (float(h.attrs["arg_log10n0_c2"]) + 5.0))
    add("DensityFactorLo", "plain", fsig(facs[0], 1), facs[0], "", "smallest factor by which the assumed AGN number density is off its input (Analysis 4)", {"file": rel(A4 / "joint_m18_a05_s100.h5"), "key": "attrs arg_log10n0_c2 (+5)"}, "design,C5")
    add("DensityFactorHi", "plain", fsig(facs[1], 1), facs[1], "", "largest such factor", {"file": rel(A4 / "joint_m18_a20_s100.h5"), "key": "attrs arg_log10n0_c2 (+5)"}, "design,C5")
    gm = jload(DATA / "seed100" / "catalogs" / "glass_field_meta.json")
    dx = float(gm["glass_config"]["dx_mpc"])
    add("MockShellWidth", "plain", fnum(dx, 0), dx, "Mpc", "thickness of the lognormal matter shells", {"file": rel(DATA / "seed100/catalogs/glass_field_meta.json"), "key": "glass_config.dx_mpc"}, "mock")
    m = jload(META100)
    evm = jload(EV_REF_META)                      # the production event set of realization A (until 17 Nov META.json "stages.events")
    det = evm["detection"]
    add("MockRhoRef", "plain", fnum(det["snr_ref_detect"], 2), det["snr_ref_detect"], "", "optimal signal-to-noise ratio of a redshifted chirp mass of MockMcRef at 1 Gpc (detection scale)", {"file": rel(EV_REF_META), "key": "detection.snr_ref_detect"}, "mock,appendixB")
    readme = (DATA / "README.md").read_text()
    assert "(Mc_det,obs / 30)^(5/6) * (1000 Mpc / d_obs)" in readme
    add("MockMcRef", "plain", "30", 30.0, "Msun", "reference redshifted chirp mass of the signal-to-noise scaling", {"file": "data/README.md", "key": "rho_obs = snr_ref * (Mc_det,obs / 30)^(5/6) * (1000 Mpc / d_obs)"}, "mock,appendixB")
    # the sky-width law: the generator's configuration (META.json config.events, a design input, not an event file), checked against the
    # production event set's own record (its attribute sky_a_deg and its measurement model's sigma_ang law)
    ve = m["config"]["events"]
    with h5py.File(EV_REF) as h:
        a_prod = float(h.attrs["sky_a_deg"])
    clip = ve["v3_measurement_family"]["sky_clip_deg"]
    assert a_prod == ve["v3_measurement_family"]["sky_a_deg"], (a_prod, ve["v3_measurement_family"])
    assert f"{clip[0]:.1f}, {clip[1]:.1f}) deg" in evm["measurement_model"]["widths"]["sigma_ang"], (clip, evm["measurement_model"]["widths"]["sigma_ang"])
    coeff = a_prod * det["snr_ref_detect"] / det["snr_ref_sigma_ang"]
    add("MockSkyCoeff", "plain", fnum(coeff, 1), coeff, "deg", "sky error in degrees times the observed signal-to-noise ratio: sigma_ang = clip(this / rho_obs, lo, hi) (35 deg x 6.278/11.5)",
        {"files": [rel(EV_REF), rel(EV_REF_META)], "keys": ["attrs sky_a_deg", "detection.snr_ref_detect", "detection.snr_ref_sigma_ang"], "computation": "sky_a_deg * snr_ref_detect / snr_ref_sigma_ang"}, "mock,appendixB")
    add("MockSkyClipLo", "plain", fnum(clip[0], 0), clip[0], "deg", "lower bound of the one-sigma sky error", {"file": rel(META100), "key": "config.events.v3_measurement_family.sky_clip_deg[0]"}, "mock,appendixB")
    add("MockSkyClipHi", "plain", fnum(clip[1], 0), clip[1], "deg", "upper bound of the one-sigma sky error", {"file": rel(META100), "key": "config.events.v3_measurement_family.sky_clip_deg[1]"}, "mock,appendixB")
    ij = DATA / "seed100" / "injections" / "injections_targeted_meta.json"
    jj = jload(ij)
    add("MockNinjDrawn", "plain", fsci(jj["ndraw"], 1), jj["ndraw"], "signals", "simulated signals drawn for the reference catalog", {"file": rel(ij), "key": "ndraw"}, "mock,appendixA")
    add("MockNinjFound", "plain", fsci(jj["n_detected"], 1), jj["n_detected"], "signals", "detected simulated signals of the reference catalog (2,205,380)", {"file": rel(ij), "key": "n_detected"}, "mock,appendixA")
    # median over the detected mergers of realization A of the sd of ln d_L over each merger's posterior samples (the production event set;
    # until 17 Nov META.json's validation check V3 of the older events.h5, 0.1137 against 0.1136 here)
    with h5py.File(EV_REF) as h:
        v3 = float(np.median(np.std(np.log(h["dL"][()].reshape(int(evm["n_events"]), -1)), axis=1)))
    add("MockSigLnDL", "plain", fnum(100 * v3, 0), v3, "%", "median fractional (ln) luminosity-distance uncertainty of the detected events, reference catalog, in percent",
        {"file": rel(EV_REF), "key": "dL", "computation": "median over the events of the sd of ln dL over each event's posterior samples"}, "mock")
    with h5py.File(A0 / "h0_pureagn_targeted_s100.h5") as h:
        floor = float(h["guard/legacy_floor_5N"][()][0])
    nev = evm["n_events"]
    add("NeffFactor", "plain", fnum(floor / nev, 0), floor / nev, "", "required effective number of detected simulated signals per event (N_eff > this times N)",
        {"file": rel(A0 / "h0_pureagn_targeted_s100.h5"), "key": "guard/legacy_floor_5N / n_events"}, "appendixA")
    j13 = jload(A13 / "a13core_e7c3007_rslice_cap20_historical_n200_s1.json")
    add("VarCapEight", "plain", fnum(j13["max_likelihood_variance"], 0), j13["max_likelihood_variance"], "", "largest admitted variance of ln L in the eight-parameter run",
        {"file": rel(A13 / "a13core_e7c3007_rslice_cap20_historical_n200_s1.json"), "key": "max_likelihood_variance"}, "appendixA")
    with h5py.File(A10 / "c10_mech.h5") as h:
        pG, pC = float(h.attrs["pin_dmu_G"]), float(h.attrs["pin_dmu_chi"])
    add("PinDmuG", "plain", fnum(pG, 1), pG, "Msun", "mass-peak offset held fixed in the pinned-offset arms (equal to the input)", {"file": rel(A10 / "c10_mech.h5"), "key": "attrs pin_dmu_G"}, "C8")
    add("PinDmuChi", "anc", fnum(pC, 3), pC, "", "spin-mean offset held fixed in the pinned-offset arms: the fixed-H0 maximum of the same data, not a design choice (REVIEW C8)", {"file": rel(A10 / "c10_mech.h5"), "key": "attrs pin_dmu_chi"}, "C8")


# ---------------------------------------------------------------- normalisation: F_k per configuration
def build_normalisation():
    NC = norm_cfg(100)
    H = IN["H0"]
    ng, na = 10 ** IN["lognGal"], 10 ** IN["lognAGN"]
    src = lambda key, zmax: {"files": [NC[key]["gal"]["survey"], NC[key]["agn"]["survey"], NC[key]["gal"]["fit"]["file"], NC[key]["agn"]["fit"]["file"]],
                             "computation": R2_SRC_NORM + f"; z_max = {zmax:g}, H0 = {H}, n_k at the inputs"}
    table = {}
    fmax = 0.0
    for key, word, mlim in RUNGS_ALL:
        rg, ra = NC[key]["gal"], NC[key]["agn"]
        lim = "complete catalogs represented by a nominal magnitude limit (24 GAL, 25 AGN; Analysis 3)" if key == "complete" else f"m < {mlim}"
        for zmax, suf, conv in ((ZMAX_GRID, "", "normalised to z = 5 (stored runs)"), (ZMAX_PROD, "Prod", f"normalised to z = {ZMAX_PROD:g} (production convention)")):
            FG, FA = float(Fk(rg, ng, H, zmax)), float(Fk(ra, na, H, zmax))
            table[f"{key}{suf}"] = {"F_gal": FG, "F_AGN": FA, "Z_gal": float(Zk(rg, ng, H, zmax)), "Z_AGN": float(Zk(ra, na, H, zmax)), "N_gal": rg["N"], "N_AGN": ra["N"]}
            for cls, F in (("Gal", FG), ("AGN", FA)):
                r = "1" if F >= 1 - 1e-12 else fsig(F, 2)
                add(f"NormF{cls}{word}{suf}", "plain", r, F, "", f"cataloged share F = N/Z of the {'galaxy' if cls == 'Gal' else 'AGN'} class, {lim}, {conv}, input densities, H0 = {H}",
                    src(key, zmax), "C4,item11", extra={"Z": table[f"{key}{suf}"][f"Z_{'gal' if cls == 'Gal' else 'AGN'}"], "N": (rg if cls == "Gal" else ra)["N"]})
            if key != "complete" and zmax == ZMAX_GRID:
                fmax = max(fmax, FA)
    add("NormFAGNMax", "plain", fsig(fmax, 2, up=True), fmax, "", "upper bound (rounded up) of the cataloged AGN share F_AGN over the four flux limits, normalised to z = 5, input densities",
        {"computation": "max over m<21..m<18 of NormFAGN<limit>"}, "C4")
    add("NormFComplete", "plain", "1", 1.0, "", "cataloged share F_k of both classes in the complete-catalog analyses (Analyses 0-2, 8-13: log10 n_k = -24, so Z_k = N_k to 1e-12)",
        {"file": "paper_prd/notes/LIKELIHOOD_MATH.md", "key": "sec. 2 (missing term <= 2e-12 hosts)", "computation": "Z_k = N_k"}, "C1,item11")
    DIAG["r2_normalisation_table_H0_67.74"] = table
    # Eq. (8) at each limit: d f / d log10 n_AGN = ln10 (1 - F_AGN) f (1 - f)
    for key, word, mlim in RUNGS:
        fmed = [r for r in REG if r["name"] == f"Fluxf{word}"][0]["value"]["median"]
        for suf, fref, why in (("", fmed, "at the median of the factor-1 posterior (Fluxf<limit>)"), ("Prod", IN["f"], "at the input f_AGN = 0.30 (no production run yet)")):
            FA = table[f"{key}{suf}"]["F_AGN"]
            s = LN10 * (1 - FA) * fref * (1 - fref)
            add(f"SlopefAGNPred{word}{suf}", "anc" if suf == "" else "plain", fnum(s, 2), s, "per dex",
                f"Eq. (8) prediction of d f_AGN / d log10 n_AGN at m < {mlim}, ln10 (1 - F_AGN) f (1 - f) {why}, {'normalised to z = 5' if suf == '' else f'normalised to z = {ZMAX_PROD:g}'}",
                {"computation": "ln(10) * (1 - NormFAGN) * f * (1 - f)", "files": [rel(A3 / f'joint_{key}_s100.h5')]}, "C5,item3")


# ---------------------------------------------------------------- the identity applied to the stored grids
def _filled(L, rej):
    from scipy import ndimage
    bad = ~np.isfinite(L)
    if rej is not None:
        bad |= np.asarray(rej, bool)
    if not bad.any():
        return L.copy()
    ind = ndimage.distance_transform_edt(bad, return_distances=False, return_indices=True)
    return L[tuple(ind)]


def identity_map(H, F, L, rej, dlogit):
    """ln L_x(H0_i, f) = ln L_1(H0_i, f'), logit f' = logit f + dlogit[i] (cubic spline of ln L_1 in f per H0 row)."""
    Lf = _filled(L, rej)
    out = np.empty_like(Lf)
    lf = logit(F)
    for i in range(len(H)):
        sp = CubicSpline(F, Lf[i])
        with np.errstate(invalid="ignore"):
            fp = expit(lf + dlogit[i])
        fp = np.where(F <= 0, 0.0, np.where(F >= 1, 1.0, fp))
        out[i] = sp(fp)
    return out


IDENT: dict = {}


def build_identity():
    """Analysis 4 predicted from the factor-1 grid of Analysis 3 by the normalisation map alone (the identity
    of LIKELIHOOD_MATH sec. 8.3 check 4, moved here): for an assumed AGN density x times its input,
    logit f' = logit f + ln[F_AGN(x; H0) / F_AGN(1; H0)] row by row in H0."""
    NC = norm_cfg(100)
    facs = [0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 2.0]
    na = 10 ** IN["lognAGN"]
    for key, word, mlim in RUNGS:
        p = A3 / f"joint_{key}_s100.h5"
        H, F, L, rej = load_hf(p)
        with h5py.File(p) as h:
            pin = json.loads(h.attrs["selection_theta_pinned_at"])
        ra = NC[key]["agn"]
        assert abs(pin["Mstar_hat_c2"] - ra["fit"]["Mstar_hat"]) < 1e-9 and abs(pin["alpha_c2"] - ra["fit"]["alpha"]) < 1e-9
        med = {"f": {}, "H0": {}}
        for x in facs:
            if x == 1.0:
                g = hf_post(p, f"a3_{key}")
            else:
                d = np.log(Fk(ra, x * na, H) / Fk(ra, na, H))
                g = GridPost(["H0", "f"], [H, F], identity_map(H, F, L, rej, d), [4, 8], extra={"H0": [IN["H0"]], "f": [IN["f"]]}, tag=f"identity_{key}_x{x}")
            med["f"][str(x)] = g.summary("f")["median"]
            med["H0"][str(x)] = g.summary("H0")["median"]
        xs = np.log10(facs)
        sf = np.polyfit(xs, [med["f"][str(x)] for x in facs], 1)[0]
        sH = np.polyfit(xs, [med["H0"][str(x)] for x in facs], 1)[0]
        IDENT[key] = {"medians": med, "slope_f": sf, "slope_H0": sH}
        meas = [r for r in REG if r["name"] == f"SlopefAGN{word}"][0]
        src = {"files": [rel(p)], "computation": "identity map of the factor-1 Analysis-3 grid to the seven Analysis-4 factors (cubic spline in f per H0 row, Z_AGN(x; H0) "
                                                 "closed form), spline-refined summaries, least-squares slope of the medians against log10 x (as SlopefAGN<limit>)"}
        add(f"SlopefAGNIdentity{word}", "anc", fnum(sf, 2), sf, "per dex", f"d f_AGN / d log10 n_AGN at m < {mlim} predicted by Eq. (identity) from the posterior at the input density "
            f"(normaliser part only; the measured slope is SlopefAGN{word} = {meas['value']:.3f})", src, "C5,item3",
            extra={"predicted_medians": med, "slope_H0_predicted": sH, "measured_minus_predicted": meas["value"] - sf})
        # factor two, measured and predicted, normalised to z = 5
        dm = meas["extra"]["direct_f_change_factor2_vs_1"]
        dp = med["f"]["2.0"] - med["f"]["1.0"]
        add(f"NormDeltafTwo{word}Five", "anc", fnum(dm, 3), dm, "", f"change of the f_AGN median when the assumed AGN density is doubled, m < {mlim}, hosts normalised to z = 5 (Analysis 4 against Analysis 3, measured)",
            {"files": [rel(p), rel(A4 / f'joint_{key}_a20_s100.h5')], "computation": "difference of spline-refined medians"}, "C5,item3")
        add(f"NormDeltafTwo{word}FivePred", "anc", fnum(dp, 3), dp, "", f"the same change predicted by the identity from the factor-1 posterior, m < {mlim}, normalised to z = 5", src, "C5,item3")
        dHm = meas["extra"]["direct_H0_change_factor2_vs_1"]
        dHp = med["H0"]["2.0"] - med["H0"]["1.0"]
        add(f"NormDeltaHzeroTwo{word}Five", "anc", fnum(dHm, 2), dHm, "km/s/Mpc", f"change of the H0 median when the assumed AGN density is doubled, m < {mlim}, normalised to z = 5 (measured)",
            {"files": [rel(p), rel(A4 / f'joint_{key}_a20_s100.h5')], "computation": "difference of spline-refined medians"}, "C5,item3")
        add(f"NormDeltaHzeroTwo{word}FivePred", "anc", fnum(dHp, 2), dHp, "km/s/Mpc", f"the same predicted by the identity (normaliser part), m < {mlim}", src, "C5,item3")


# ---------------------------------------------------------------- rate per host for every analysis with f_AGN
def survey_counts(seed, rung="complete"):
    return {cls: survey_nobs(DATA / f"seed{seed}" / "surveys" / f"survey_{cls}_{rung}_ns32.h5") for cls in ("gal", "agn")}


def build_rate_ratio():
    NC = norm_cfg(100)
    c100 = survey_counts(100)
    ratio100 = c100["gal"] / c100["agn"]
    Rin = IN["f"] / (1 - IN["f"]) * ratio100
    add("MockRateRatio", "plain", fnum(Rin, 1), Rin, "", "input merger rate per AGN relative to that per galaxy: [0.30/0.70] N_gal/N_AGN of the reference catalog pair (the mock draws 30% of the hosts "
        "from the AGN and 70% from the galaxies, each class uniformly per object); with the input densities instead of the counts, 42.86",
        {"files": [rel(DATA / 'seed100/surveys/survey_gal_complete_ns32.h5'), rel(DATA / 'seed100/surveys/survey_agn_complete_ns32.h5')], "computation": "odds(planted f_AGN) * sum(ngals) ratio"},
        "item11", extra={"N_gal": c100["gal"], "N_AGN": c100["agn"]})
    # ---- complete catalogs, Z_k = N_k (Analysis 2)
    rat = {s: survey_counts(s) for s in C1_SEEDS}
    lr = np.array([math.log(rat[s]["gal"] / rat[s]["agn"]) for s in C1_SEEDS])
    srcC = lambda files: {"files": files, "computation": "R quantiles = odds(f quantiles) * N_gal/N_AGN (Z_k = N_k exactly: log10 n0 = -24); N_k = sum(ngals) of the run's complete survey files"}
    Ls = []
    for s in C1_SEEDS:
        H, F, L, _ = load_hf(A2 / f"joint_s{s}.h5")
        Ls.append(L)
    gA = GridPost(["H0", "f"], [H, F], Ls[0], [4, 10], extra={"H0": [IN["H0"]], "f": [IN["f"]]}, tag="joint_s100_rr")
    gC = GridPost(["H0", "f"], [H, F], np.sum(Ls, axis=0), [4, 10], extra={"H0": [IN["H0"]], "f": [IN["f"]]}, tag="joint_combined_rr")
    RA = rate_ratio_constant(gA.summary("f"), ratio100)
    add_rr("RateRatioJointCatA", RA, "complete catalogs, joint (H0, f_AGN), catalog A; Z_k = N_k", srcC([rel(A2 / "joint_s100.h5")]), "C1,item11", Rin)
    rc = float(np.exp(lr.mean()))
    RC = rate_ratio_constant(gC.summary("f"), rc)
    RC["N_ratio_per_catalog"] = {str(s): float(np.exp(x)) for s, x in zip(C1_SEEDS, lr)}
    add_rr("RateRatioJointComb", RC, "complete catalogs, combined posterior of the 9 catalogs (f_AGN shared); Z_gal/Z_AGN = geometric mean of the 9 count ratios "
           f"(spread {100 * (lr.max() - lr.min()):.2f}% end to end)", srcC([rel(A2 / f'joint_s{s}.h5') for s in C1_SEEDS]), "C1,item11", Rin)
    # ---- flux-limited catalogs at the input densities and the complete-catalog representation of Analysis 3 (normalised to z = 5)
    ng, na = 10 ** IN["lognGal"], 10 ** IN["lognAGN"]
    for key, word, mlim in RUNGS_ALL:
        p = A3 / f"joint_{key}_s100.h5"
        g = hf_post(p, f"a3_{key}_rr")
        rg, ra = NC[key]["gal"], NC[key]["agn"]
        R = rate_ratio_grid(g, lambda h, rg=rg, ra=ra: Zk(rg, ng, h) / Zk(ra, na, h))
        lim = "complete catalogs with the nominal-limit representation" if key == "complete" else f"m < {mlim}"
        add_rr(f"RateRatio{word if key != 'complete' else 'CompleteDeep'}", R, f"{lim}, number densities at their inputs, hosts normalised to z = 5 (Analysis 3)",
               {"files": [rel(p)], "computation": "R = odds(f) Z_gal(H0)/Z_AGN(H0) on the spline-refined (H0, f) posterior, CDF of R summed over H0 nodes; " + R2_SRC_NORM}, "C4,item11", Rin)
        if key == "complete":
            continue
        for tag, x, nm in (("a05", 0.5, "Half"), ("a20", 2.0, "Two")):
            pa = A4 / f"joint_{key}_{tag}_s100.h5"
            ga = hf_post(pa, f"a4_{key}_{tag}_rr")
            Rx = rate_ratio_grid(ga, lambda h, rg=rg, ra=ra, x=x: Zk(rg, ng, h) / Zk(ra, x * na, h))
            add_rr(f"RateRatio{nm}{word}", Rx, f"m < {mlim}, assumed AGN density {x:g} times its input (Analysis 4), the Z_AGN of that assumption",
                   {"files": [rel(pa)], "computation": "as RateRatio<limit> with n_AGN = x times the input"}, "C5,item11", Rin)
    # ---- both densities free (Analysis 5 and the seeds-101/102 repeats): R per sample
    runs = [(key, word, A5 / f"campaign_{key}_dynesty_s100.h5", A5 / f"campaign_{key}_dynesty_s100.json", "", 100) for key, word, _ in RUNGS]
    runs += [("m18", "Eighteen", SR / "fu_seed101" / "results" / "campaign_m18_dynesty_s101.h5", SR / "fu_seed101" / "results" / "campaign_m18_dynesty_s101.json", "CatB", 101),
             ("m18", "Eighteen", SR / "fu_seed102" / "results" / "campaign_m18_dynesty_s102.h5", SR / "fu_seed102" / "results" / "campaign_m18_dynesty_s102.json", "CatC", 102)]
    for key, word, ph5, pj, cat, seed in runs:
        c, _, _, _ = load_a5(ph5, pj)
        with h5py.File(ph5) as h:
            sp = json.loads(h.attrs["arg_survey_path"])
            fits = str(h.attrs["arg_selection_fit"]).split(",")
        assert all(f"seed{seed}/surveys/survey_{cls}_{key}_ns32.h5" in s_ for cls, s_ in zip(("gal", "agn"), sp))
        assert [Path(f).name for f in fits] == [f"fit_gal_{key}_schechter.json", f"fit_agn_{key}_schechter.json"]
        cnt = survey_counts(seed, key)
        rg = dict(NC[key]["gal"], N=cnt["gal"])
        ra = dict(NC[key]["agn"], N=cnt["agn"])
        Hs = c.col("H0")
        Rs = c.col("fcat_2") / (1 - c.col("fcat_2")) * Zk(rg, 10 ** c.col("log10n0"), Hs) / Zk(ra, 10 ** c.col("log10n0_c2"), Hs)
        R = rate_ratio_samples(Rs)
        add_rr(f"RateRatioFree{word}{cat}", R, f"both number densities free, m < {key[1:]}, {'catalog A' if not cat else 'catalog ' + cat[-1]}: R per posterior sample with Z_k at that sample's H0, n_gal, n_AGN (z = 5)",
               {"file": rel(ph5), "computation": "R_i = odds(f_i) Z_gal(H0_i, n_gal,i)/Z_AGN(H0_i, n_AGN,i); numpy.percentile; " + R2_SRC_NORM}, "C6,item11", Rin)
    # ---- host-dependent mock (complete catalogs of seed 100, Z_k = N_k)
    srcH = lambda f: {"file": rel(f), "computation": "R quantiles = odds(f quantiles) * N_gal/N_AGN of the seed-100 complete surveys (Z_k = N_k)"}
    with h5py.File(A10 / "c10_arm_S.h5") as h:
        Hs_, Fs_, Ls_, rj = h["H0_grid"][()], h["f_grid"][()], h["log_likelihood"][()], h["guard/rejected"][()]
    gS = GridPost(["H0", "f"], [Hs_, Fs_], Ls_, [4, 8], extra={"H0": [IN["H0"], 60.0, 76.0], "f": [IN["f"]]}, rejected=rj, tag="C10-S_rr")
    add_rr("RateRatioCommon", rate_ratio_constant(gS.summary("f"), ratio100), "host-dependent mock, single population held at the galaxy-class inputs (C10-S)", srcH(A10 / "c10_arm_S.h5"), "C7,item11", Rin)
    for nm, npz in (("HDFree", A11 / "a11_11D.npz"), ("HDEight", A13 / "a13core_e7c3007_rslice_cap20_historical_n200_s1.npz")):
        cp, _ = load_chain_npz(npz, Path(str(npz).replace(".npz", ".json")))
        add_rr(f"RateRatio{nm}", rate_ratio_constant(cp.summary("f_agn"), ratio100, kind="chain"), f"host-dependent mock, {'six' if nm == 'HDFree' else 'eight'}-parameter host-dependent model", srcH(npz), "C8,item11", Rin)


# ---------------------------------------------------------------- completeness weighted by the detected mergers
def build_event_completeness():
    NC = norm_cfg(100)
    pev = EV_REF
    with h5py.File(pev) as h:
        zt = h["truth/z"][()]
        ht = h["truth/host_type"][()]
        hi = h["truth/host_index"][()]
    app = np.empty(len(zt))
    for typ, cls in ((0, "gal"), (1, "agn")):
        sel = np.where(ht == typ)[0]
        u, inv = np.unique(hi[sel], return_inverse=True)
        with h5py.File(DATA / "seed100" / "catalogs" / f"catalog_{cls}_complete.h5") as h:
            app[sel] = h["app_mag"][u][inv]
    src_ev = {"files": [rel(pev), rel(DATA / "seed100/catalogs/catalog_gal_complete.h5"), rel(DATA / "seed100/catalogs/catalog_agn_complete.h5")]}
    rows = {}
    for key, word, mlim in RUNGS:
        zg, za = NC[key]["gal"]["zc"], NC[key]["agn"]["zc"]
        add(f"MockzComplete{word}", "plain", fnum(zg, 2), zg, "", f"redshift to which the m < {mlim} catalog is complete: C(z) = 1 below it (galaxy fit; AGN fit {za:.4f}); the limit reaches the faint end "
            "(L = x_cut L*) of the luminosity function there", {"files": [NC[key]["gal"]["fit"]["file"], NC[key]["agn"]["fit"]["file"]],
                                                                    "computation": "solve DM(z; H0 = 100) = m_lim - Mstar_hat - M_faint_offset (h-scaled)"}, "mock,C4,item3", extra={"agn": za})
        frac = float(np.mean(zt < zg))
        add(f"MockEventFracComplete{word}", "plain", fnum(100 * frac, 0), frac, "%", f"percentage of the 1000 detected mergers of the reference catalog whose true redshift lies below MockzComplete{word}",
            dict(src_ev, computation="mean(truth/z < z_complete)"), "mock,C4")
        kept = app < mlim
        ce = float(np.mean(kept))
        cmod = float(np.mean(np.where(ht == 0, c_sel(zt, NC[key]["gal"]["fit"]), c_sel(zt, NC[key]["agn"]["fit"]))))
        rows[key] = {"z_complete_gal": zg, "z_complete_agn": za, "frac_events_below": frac, "event_weighted_completeness": ce,
                     "galaxy_hosted": float(np.mean(kept[ht == 0])), "agn_hosted": float(np.mean(kept[ht == 1])), "model_C_at_event_redshifts": cmod}
        add(f"MockEventComp{word}", "plain", fpct(ce), ce, "%", f"completeness weighted by the detected mergers at m < {mlim}: percentage of the 1000 detected mergers of the reference catalog whose host is in the "
            f"flux-limited catalog of its class (host apparent magnitude below {mlim}); galaxy-hosted {100 * rows[key]['galaxy_hosted']:.1f}%, AGN-hosted {100 * rows[key]['agn_hosted']:.1f}%; "
            f"the selection model C(z) of the analysis averaged over the true event redshifts gives {100 * cmod:.1f}%",
            dict(src_ev, computation="mean over events of [app_mag(host) < m_lim], host row = truth/host_index in the complete catalog of the host's class"), "mock,C4,item3", extra=rows[key])
    DIAG["r2_event_weighted_completeness"] = rows


# ---------------------------------------------------------------- hosts per 90% localization volume
HOSTVOL_Q = math.sqrt(0.9)   # sky disc and redshift interval each hold sqrt(0.9) of the samples: 90% jointly


def event_regions(path, H0):
    """Per event: centre (unit vector), sky radius [rad] and redshift interval holding HOSTVOL_Q of the posterior samples each."""
    with h5py.File(path) as h:
        ra = h["ra"][()].reshape(1000, -1)
        dec = h["dec"][()].reshape(1000, -1)
        dl = h["dL"][()].reshape(1000, -1)
    v = np.stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)], axis=-1)
    c = v.mean(axis=1)
    c /= np.linalg.norm(c, axis=1, keepdims=True)
    ang = np.arccos(np.clip(np.einsum("ijk,ik->ij", v, c), -1, 1))
    r = np.quantile(ang, HOSTVOL_Q, axis=1)
    z = z_of_dl(dl, H0)
    zlo = np.quantile(z, (1 - HOSTVOL_Q) / 2, axis=1)
    zhi = np.quantile(z, (1 + HOSTVOL_Q) / 2, axis=1)
    return c, r, zlo, zhi


def read_catalog_rows(path, zmax, cols=("ra", "dec", "z_obs")):
    """Rows with true z < zmax (the catalogs are written shell by shell in true redshift; reading stops at the
    first block wholly above zmax)."""
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
            if m.any():
                for c in cols:
                    parts[c].append(h[c][i:i + B][m])
            i += B
    return {c: np.concatenate(v) for c, v in parts.items()}


HOSTVOL_NSIDE, HOSTVOL_DZ = 128, 0.0025


def count_in_regions(cat, regions):
    """Number of catalog objects (z_obs) inside each event's sky disc (pixel centres within the radius, HEALPix
    nside 128) and redshift interval (linear within 0.0025 bins of a cumulative count per pixel)."""
    import healpy as hp
    c, r, zlo, zhi = regions
    zmax = float(zhi.max()) + 0.02
    pix = hp.ang2pix(HOSTVOL_NSIDE, np.pi / 2 - cat["dec"], cat["ra"])
    edges = np.arange(0.0, zmax + HOSTVOL_DZ, HOSTVOL_DZ)
    iz = np.floor(cat["z_obs"] / HOSTVOL_DZ).astype(np.int64)
    ok = (iz >= 0) & (iz < len(edges) - 1)
    nb = len(edges) - 1
    cnt = np.bincount(pix[ok].astype(np.int64) * nb + iz[ok], minlength=hp.nside2npix(HOSTVOL_NSIDE) * nb).reshape(-1, nb)
    cum = np.concatenate([np.zeros((cnt.shape[0], 1), np.int32), np.cumsum(cnt, axis=1).astype(np.int32)], axis=1)
    del cnt
    out = np.empty(len(r))
    for i in range(len(r)):
        ip = hp.query_disc(HOSTVOL_NSIDE, c[i], r[i], inclusive=False)
        sub = cum[ip]
        # linear interpolation of the cumulative counts in z, vectorised over pixels
        j0 = min(int(zlo[i] / HOSTVOL_DZ), nb - 1)
        j1 = min(int(zhi[i] / HOSTVOL_DZ), nb - 1)
        t0 = (zlo[i] - edges[j0]) / HOSTVOL_DZ
        t1 = (zhi[i] - edges[j1]) / HOSTVOL_DZ
        lo = sub[:, j0] + t0 * (sub[:, j0 + 1] - sub[:, j0])
        hi = sub[:, j1] + t1 * (sub[:, j1 + 1] - sub[:, j1])
        out[i] = float(np.sum(hi - lo))
    return out


HOSTVOL_EVENTS = {"gal": EV_PUREGAL, "agn": EV_PUREAGN}   # the production all-galaxy-hosted and all-AGN-hosted sets of realization A


def build_hosts_per_volume():
    """Hosts per 90% localization volume on the production all-galaxy-hosted and all-AGN-hosted event sets of realization A
    (events_puregal_corepop.h5, events_pureagn_corepop.h5: the events of keys Galhosted and AGNhosted; until 17 Nov the older
    events_puregal.h5 and events_pureagn.h5)."""
    H = IN["H0"]
    res = {}
    for sett, ev in HOSTVOL_EVENTS.items():
        res[sett] = event_regions(ev, H)
    zcut = max(float(res[k][3].max()) for k in res) + 0.05
    counts = {}
    for cls, n0 in (("gal", 10 ** IN["lognGal"]), ("agn", 10 ** IN["lognAGN"])):
        cat = read_catalog_rows(DATA / "seed100" / "catalogs" / f"catalog_{cls}_complete.h5", zcut)
        for sett in ("gal", "agn"):
            c, r, zlo, zhi = res[sett]
            N = count_in_regions(cat, res[sett])
            V = 2 * np.pi * (1 - np.cos(r)) * (chi_mpc(zhi, H) ** 3 - chi_mpc(zlo, H) ** 3) / 3.0
            counts[(cls, sett)] = {"median": float(np.median(N)), "mean": float(np.mean(N)), "p05_p95": [float(np.percentile(N, 5)), float(np.percentile(N, 95))],
                                   "expected_n_V_median": float(np.median(n0 * V)), "median_ratio_count_over_nV": float(np.median(N / (n0 * V))),
                                   "median_volume_Mpc3": float(np.median(V)), "median_sky_radius_deg": float(np.degrees(np.median(r))),
                                   "median_z_interval": float(np.median(zhi - zlo)), "_per_event": N}
        del cat
    DIAG["r2_hosts_per_volume"] = {f"{a}_catalog_on_{b}_hosted_events": {k: v for k, v in d.items() if not k.startswith("_")} for (a, b), d in counts.items()}
    TABLES["r2_hosts_per_event"] = {f"{a}_catalog_on_{b}_hosted_events": d["_per_event"] for (a, b), d in counts.items()}
    defin = (f"90% localization region of an event: a sky disc around the mean direction of its posterior samples whose radius holds {100 * HOSTVOL_Q:.2f}% of the samples, times the "
             f"redshift interval (z of the distance samples at H0 = {H}) holding the central {100 * HOSTVOL_Q:.2f}%; sky and distance errors are independent in the mock, so the region holds 90% "
             f"of the posterior. Objects counted by their cataloged position and z_obs (HEALPix nside {HOSTVOL_NSIDE} pixel centres, z bins {HOSTVOL_DZ}).")
    src = lambda ev, cls: {"files": [rel(ev), rel(DATA / "seed100" / "catalogs" / f"catalog_{cls}_complete.h5")], "computation": defin}
    for name, cls, sett in (("PureGalHostsPerVol", "gal", "gal"), ("PureAGNHostsPerVol", "agn", "agn"),
                            ("PureGalHostsPerVolAGNEvents", "gal", "agn"), ("PureAGNHostsPerVolGalEvents", "agn", "gal")):
        d = counts[(cls, sett)]
        add(name, "plain", fsig(d["median"], 2), d["median"], "objects",
            f"median over the 1000 {'galaxy' if sett == 'gal' else 'AGN'}-hosted mergers of the reference catalog (the production all-{'galaxy' if sett == 'gal' else 'AGN'}-hosted "
            f"set of realization A, {HOSTVOL_EVENTS[sett].name}) of the number of {'galaxies' if cls == 'gal' else 'AGN'} of the complete catalog inside "
            f"the 90% localization volume (mean {d['mean']:.0f}; expected from the mean density alone n V, median {d['expected_n_V_median']:.0f})", src(HOSTVOL_EVENTS[sett], cls), "C3,item11",
            extra={k: v for k, v in d.items() if not k.startswith("_")})


# ---------------------------------------------------------------- smaller requests of MACRO_REQUESTS
def kde1(x, lo, hi, n=2001, bw=None):
    """Gaussian kernel density of chain samples on [lo, hi], reflected at both prior edges, normalised on
    [lo, hi]; bandwidth 0.3 sample standard deviations (shared with make_figures.py)."""
    x = np.asarray(x, float)
    h = bw if bw is not None else 0.3 * float(np.std(x))
    g = np.linspace(lo, hi, n)
    p = np.zeros(n)
    for c in np.array_split(x, max(1, len(x) // 400)):
        for r in (c, 2 * lo - c, 2 * hi - c):
            p += np.exp(-0.5 * ((g[:, None] - r[None, :]) / h) ** 2).sum(axis=1)
    p /= np.trapz(p, g)
    return g, p, h


INDUCED: dict = {}


def induced_density(key, n=2_000_000, seed=20261117):
    """Prior-induced posterior of (f, log10 n_gal, log10 n_AGN) at flux limit `key` (catalog A): prior draws from
    the boxes of the free-density runs, weighted by the stored Analysis-3 likelihood at the input densities mapped
    through the identity, L(H0, f*) with logit f* = logit f + ln[F_A(n_A)/F_G(n_G)] - ln[F_A,in/F_G,in]. This is
    what the mergers say through the normalisers alone; the free-density posterior adds the in-data missing hosts."""
    if key in INDUCED:
        return INDUCED[key]
    from scipy.interpolate import RegularGridInterpolator
    NC = norm_cfg(100)
    pr = jload(A5 / f"campaign_{key}_dynesty_s100.json")["priors"]
    rng = np.random.default_rng(seed)
    H = rng.uniform(*pr["H0"], n)
    f = rng.uniform(*pr["fcat_2"], n)
    lg = rng.uniform(*pr["log10n0"], n)
    la = rng.uniform(*pr["log10n0_c2"], n)
    rg, ra = NC[key]["gal"], NC[key]["agn"]
    sh = np.log(Fk(ra, 10 ** la, H) / Fk(rg, 10 ** lg, H)) - np.log(Fk(ra, 10 ** IN["lognAGN"], H) / Fk(rg, 10 ** IN["lognGal"], H))
    fs = expit(logit(np.clip(f, 1e-12, 1 - 1e-12)) + sh)
    g = hf_post(A3 / f"joint_{key}_s100.h5", f"a3_{key}_induced")
    itp = RegularGridInterpolator((g.fine[0], g.fine[1]), g.Lf - g.lmax, bounds_error=False, fill_value=-np.inf)
    w = np.exp(itp(np.stack([H, fs], axis=1)))
    w /= w.sum()
    INDUCED[key] = {"w": w, "H0": H, "f": f, "log10n0": lg, "log10n0_c2": la, "n_eff": float(1 / np.sum(w ** 2)), "priors": pr}
    return INDUCED[key]


def wquant(x, w, q):
    o = np.argsort(x)
    c = np.cumsum(w[o])
    c -= 0.5 * w[o]
    return np.interp(q, c / c[-1], x[o])


def build_misc_requests():
    v = {r["name"]: r for r in REG}
    # shift of the combined H0 median, galaxy catalog alone minus both catalogs
    d = v["GalOnlyHzeroComb"]["value"]["median"] - v["JointHzeroComb"]["value"]["median"]
    add("ShiftGalOnlyMinusJointComb", "anc", fnum(d, 1), d, "km/s/Mpc", "median of GalOnlyHzeroComb minus median of JointHzeroComb (combined over the 9 complete catalogs)",
        {"computation": "difference of the two spline-refined combined medians", "files": [rel(A2 / f"joint_s{s}.h5") for s in C1_SEEDS]}, "C2")
    # variance of ln L at the input H0 over the 93 pure-tracer sets
    ens = jload(A0 / "h0_pure_tracer_ens93.json")
    seeds = list(ens["seeds_requested"])
    for tr, word in (("agn", "AGN"), ("gal", "Gal")):
        vv, pe = [], []
        for s in seeds:
            with h5py.File(A0 / f"h0_pure{tr}_targeted_s{s}.h5") as h:
                Hg = h["H0_grid"][()]
                i = int(np.argmin(np.abs(Hg - IN["H0"])))
                vv.append(float(h["guard/sigma2_total"][()][i]))
                pe.append(float(h["guard/pe_variance_sum"][()][i]))
        add(f"LnLVar{word}Hosted", "anc", fnum(float(np.median(vv)), 0), float(np.median(vv)), "",
            f"Monte Carlo variance of ln L (posterior-sample term plus selection term N^2/N_eff) at H0 = {Hg[i]:.2f}, median over the {len(seeds)} {'AGN' if tr == 'agn' else 'galaxy'}-hosted sets "
            f"(reference catalog {vv[seeds.index(100)]:.1f}; range {min(vv):.1f} to {max(vv):.1f}; posterior-sample term alone median {np.median(pe):.1f})",
            {"files": f"{rel(A0)}/h0_pure{tr}_targeted_s<seed>.h5 ({len(seeds)} seeds of h0_pure_tracer_ens93.json)", "keys": ["guard/sigma2_total", "guard/pe_variance_sum"], "computation": "value at the H0 node nearest the input; median over sets"},
            "appendixA,item4", extra={"reference_catalog": vv[seeds.index(100)], "range": [min(vv), max(vv)], "pe_term_median": float(np.median(pe))})
    # free densities: peak of the log10 n_AGN density over the flat prior and over the induced density; 90% upper limits
    for key, word, mlim in RUNGS:
        c, _, _, j = load_a5(A5 / f"campaign_{key}_dynesty_s100.h5", A5 / f"campaign_{key}_dynesty_s100.json")
        I = induced_density(key)
        srcI = {"files": [rel(A5 / f"campaign_{key}_dynesty_s100.h5"), rel(A3 / f"joint_{key}_s100.h5")], "computation": "induced: " + (induced_density.__doc__ or "").split(". This")[0].replace("\n", " ") + "; 2e6 prior draws, fixed seed"}
        for cn, cls in (("log10n0", "Gal"), ("log10n0_c2", "AGN")):
            up = float(np.percentile(c.col(cn), 90))
            ui = float(wquant(I[cn], I["w"], 0.90))
            add(f"FreeLogn{cls}Upper{word}", "anc", fnum(up, 2), up, "log10 Mpc^-3", f"90% upper limit on log10 n_{cls.lower() if cls == 'Gal' else 'AGN'}, both densities free, m < {mlim}, catalog A (90th percentile of the samples)",
                {"file": rel(A5 / f'campaign_{key}_dynesty_s100.h5'), "computation": "numpy.percentile 90"}, "C6,item3")
            add(f"FreeLogn{cls}InducedUpper{word}", "anc", fnum(ui, 2), ui, "log10 Mpc^-3", f"the 90% upper limit the normalisers alone induce (prior seen through the identity and the f-tilde constraint), m < {mlim}, catalog A", srcI, "C6,item3",
                extra={"n_eff_of_the_weights": I["n_eff"]})
        if key == "m18":
            lo, hi = j["priors"]["log10n0_c2"]
            x, p, bw = kde1(c.col("log10n0_c2"), lo, hi)
            pk = float(p.max())
            ip = np.histogram(I["log10n0_c2"], bins=np.linspace(lo, hi, 41), weights=I["w"], density=True)
            dens_ind = float(np.interp(x[np.argmax(p)], 0.5 * (ip[1][1:] + ip[1][:-1]), ip[0]))
            add("FreeLognAGNPeakOverPrior", "anc", fnum(pk * (hi - lo), 1), pk * (hi - lo), "", f"largest posterior density of log10 n_AGN over the flat prior density {1 / (hi - lo):.1f} per dex, m < 18, catalog A (reflected Gaussian KDE, bandwidth {bw:.3f} dex, as Fig. 7)",
                {"file": rel(A5 / "campaign_m18_dynesty_s100.h5"), "computation": "max of kde1 * prior width"}, "C6")
            add("FreeLognAGNPeakOverInduced", "anc", fnum(pk / dens_ind, 1), pk / dens_ind, "", "the same peak density over the prior-induced density at the same log10 n_AGN (40-bin weighted histogram of the induced draws)",
                srcI, "C6,item3")


def build_spin_and_offsets():
    """Spins alone measure the product f_AGN x offset (REVIEW B11); the offsets under one prior range (G5)."""
    p = A8 / "arm_I_intrinsic.h5"
    with h5py.File(p) as h:
        F, M, L = h["f_grid"][()], h["mu_chi_c2_grid"][()], h["log_likelihood"][()]
        rej = h["guard/rejected"][()] if "guard/rejected" in h else None
    g = GridPost(["f", "dmuChi"], [F, M], L, [8, 6], extra={"f": [IN["f"]], "dmuChi": [0.0, IN["dmuChi"]]}, rejected=rej, tag="A8-I_product")
    x1, x2, m = g.marg2("f", "dmuChi")
    W = np.outer(g.w[0], g.w[1]) * m
    prod = np.multiply.outer(x1, x2).ravel()
    w = W.ravel() / W.sum()
    q = [float(wquant(prod, w, u)) for u in (0.05, 0.16, 0.5, 0.84, 0.95)]
    sm = {"median": q[2], "ci68": [q[1], q[3]], "ci90": [q[0], q[4]]}
    lo_dmu = g.summary("dmuChi")["ci90"][0]
    add("SpinIntProduct", "anc", fci(q[2], q[0], q[4])[0], sm, "", "f_AGN times the spin-mean offset from spins alone, fixed H0 (the mean spin of the whole population above the galaxy-class mean); since f_AGN <= 1, "
        f"the offset cannot lie below this product: the lower edge of SpinIntDmuChi (5% point {lo_dmu:.3f}) is the f_AGN = 1 edge (REVIEW B11) (median and 90% interval)",
        src_grid(p, ["f_grid", "mu_chi_c2_grid", "log_likelihood"], "weighted quantiles of f * Delta mu_chi on the spline-refined 2-D lattice"), "C9", containment={"input": IN["f"] * IN["dmuChi"],
        "in68": bool(q[1] <= IN["f"] * IN["dmuChi"] <= q[3]), "in90": bool(q[0] <= IN["f"] * IN["dmuChi"] <= q[4]), "level_two_sided": abs(2 * float(np.sum(w[prod < IN["f"] * IN["dmuChi"]])) - 1),
        "cdf_at_input": float(np.sum(w[prod < IN["f"] * IN["dmuChi"]]))})
    # the spin offset of the three host-dependent models under one prior range [-0.05, 0.20] (the six-parameter box)
    hi = float([r for r in REG if r["name"] == "PriorDmuChiHiSix"][0]["value"])
    p13 = A13 / "a13core_e7c3007_rslice_cap20_historical_n200_s1.npz"
    c13, _ = load_chain_npz(p13, A13 / "a13core_e7c3007_rslice_cap20_historical_n200_s1.json")
    keep = c13.col("dmu_chi") <= hi
    cr = ChainPost(c13.s[keep], c13.names, tag="A13<=0.20")
    add_post("HDEightDmuChiPriorSix", cr.summary("dmu_chi"), "", f"spin-mean offset of the eight-parameter model under the prior [-0.05, {hi:.2f}] of the six-parameter model (the stored samples above {hi:.2f} removed; "
             f"exact for a flat prior: {100 * (1 - keep.mean()):.1f}% of the samples lie above)", {"file": rel(p13), "computation": f"numpy.percentile of the samples with dmu_chi <= {hi}"}, "C8,G5", IN["dmuChi"], kind="chain")
    add("HDEightDmuChiMassAboveSix", "anc", fnum(100 * (1 - keep.mean()), 0), float(1 - keep.mean()), "%", f"percentage of the eight-parameter posterior of the spin-mean offset above {hi:.2f} (the end of the six-parameter prior)",
        {"file": rel(p13), "computation": f"mean(dmu_chi > {hi})"}, "C8,G5")
    x, pj = R2_CACHE["C10J_marg_dmuChi"]
    s = trunc_summary(x, pj, hi)
    add_post("HDFixDmuChiPriorSix", s, "", f"spin-mean offset of the reference-fixed model under the prior upper end {hi:.2f} (marginal truncated and renormalised; lattice starts at {x[0]:.3f})",
             src_grid(A10 / "c10_arm_J.h5", ["log_likelihood"], f"1-D marginal of the refined 4-D lattice truncated at {hi}"), "C8,G5", IN["dmuChi"])


def trunc_summary(x, p, hi):
    m = x <= hi + 1e-12
    x, p = x[m], p[m]
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (p[1:] + p[:-1]) * np.diff(x))])
    cdf /= cdf[-1]
    q = lambda u: float(np.interp(u, cdf, x))
    return {"median": q(0.5), "ci68": [q(0.16), q(0.84)], "ci90": [q(0.05), q(0.95)], "_cdf": (x, cdf), "_x": x, "_p": p / np.trapz(p, x)}


# ---------------------------------------------------------------- Analysis 14 (run R0) and Analysis 15 (normalisation)
NS_WORD = {2000: "TwoK", 8000: "EightK", 32000: "ThirtyTwoK"}
A14_RUN = "Analysis 14 (analyses/analysis_14_monte_carlo_accuracy/results/mc_accuracy_summary.json, written by scripts/aggregate_mc_accuracy.py at the end of RITA-GPU jobs 1414628 (mixed set) and 1414642 (all-AGN set); run R0)"
A15_RUN = "Analysis 15 (analyses/analysis_15_host_normalisation/results/normalisation_summary.json, written by scripts/aggregate_normalisation.py at the end of RITA-GPU job 1414681)"


def _a14_cfg(summary, which):
    if summary is None:
        return None
    cf = summary.get("configs", {})
    if which == "mixed":
        return cf.get("b")
    for k in ("a_s103", "a"):
        if k in cf:
            return cf[k]
    return None


def build_a14():
    S = jload(A14_SUMMARY) if A14_SUMMARY.exists() else None
    srcS = {"file": rel(A14_SUMMARY), "computation": "read from the Analysis-14 summary (aggregate_mc_accuracy.py: spread = sd with ddof 1 over redraws; hw68 = mean 68% half-width)"}
    for cfgname, cw, params in (("mixed", "Mixed", ("H0", "f")), ("agn", "AGN", ("H0",))):
        B = _a14_cfg(S, cfgname)
        desc = "mixed set (input f_AGN 0.30, seed 100), joint (H0, f_AGN)" if cfgname == "mixed" else "all-AGN-hosted set (f_AGN = 1, AGN catalog alone; seed 103)"
        for ns, nw in NS_WORD.items():
            st = None if B is None else B["by_nsamp"].get(str(ns))
            ok = st is not None
            pre = f"MC{cw}"

            def put(name, val, d, units, meaning, render=None):
                if ok and val is not None and np.isfinite(val):
                    add(name, "plain", render or fnum(val, d), val, units, meaning, srcS, "appendixA,item4")
                else:
                    add_ph(name, r"\ph{XX}", units, meaning, A14_RUN, "appendixA,item4")
            put(f"{pre}N{nw}", None if not ok else float(st["n"]), 0, "redraws", f"number of redraws of the posterior samples at {ns} per event, {desc}")
            for p in params:
                pw = "Hzero" if p == "H0" else "f"
                unit = "km/s/Mpc" if p == "H0" else ""
                s = None if not ok else st.get(p)
                dsd = 3 if p == "f" else 2
                put(f"{pre}{pw}Sd{nw}", None if s is None else s["median_sd"], dsd, unit, f"spread (sd over redraws) of the posterior median of {p} at {ns} samples per event, {desc}")
                put(f"{pre}{pw}SdRel{nw}", None if s is None or s["median_sd_over_hw68"] is None else s["median_sd_over_hw68"], 2, "", f"that spread over the mean 68% half-width of {p}, {ns} samples per event, {desc}")
                put(f"{pre}{pw}Med{nw}", None if s is None else s["median_mean"], 3 if p == "f" else 2, unit, f"mean over redraws of the posterior median of {p}, {ns} samples per event, {desc}")
            lv = None if not ok else st["lnL_variance_at_ref_cell"]
            tot = None if lv is None else lv["pe_var_mean"] + lv["sigma2_sel_N2_over_Neff_mean"]
            put(f"{pre}LnLVar{nw}", tot, 1, "", f"variance of the ln L estimator at the reference cell (posterior-sample term plus N^2/N_eff), mean over redraws, {ns} samples per event, {desc}")
            put(f"{pre}SecPerCall{nw}", None if not ok else st["seconds_per_call_median"], 2, "s", f"seconds per likelihood call (median), {ns} samples per event, {desc} (one A100)")
        inj = None if B is None else B.get("injection_redraws")
        ok = inj is not None

        def puti(name, val, d, units, meaning):
            if ok and val is not None and np.isfinite(val):
                add(name, "plain", fnum(val, d), val, units, meaning, srcS, "appendixA,item4")
            else:
                add_ph(name, r"\ph{XX}", units, meaning, A14_RUN, "appendixA,item4")
        puti(f"MC{cw}InjN", None if not ok else float(inj["n"]), 0, "sets", f"injection sets compared (the record plus redraws; 8000 samples per event, redraw 0), {desc}")
        for p in params:
            pw = "Hzero" if p == "H0" else "f"
            s = None if not ok else inj.get(p)
            puti(f"MC{cw}Inj{pw}Sd", None if s is None else s["median_sd"], 3 if p == "f" else 2, "km/s/Mpc" if p == "H0" else "", f"spread (sd) of the posterior median of {p} over injection sets, {desc}")
            puti(f"MC{cw}Inj{pw}SdRel", None if s is None or s["median_sd_over_hw68"] is None else s["median_sd_over_hw68"], 2, "", f"that spread over the mean 68% half-width, {desc}")
    # the three requests of MACRO_REQUESTS (2000 samples per event: the setting of every stored analysis)
    def summ(which, field):
        B = _a14_cfg(S, which)
        st = None if B is None else B["by_nsamp"].get("2000")
        if st is None or st["n"] < 2:
            return None
        vals = []
        for p in (("H0", "f") if which == "mixed" else ("H0",)):
            s = st[p]
            hw90 = 0.5 * s["w90_mean"]
            vals.append(100 * (s["median_sd"] / hw90 if field == "shift" else (s["w90_sd"] or 0.0) / s["w90_mean"]))
        return max(vals)
    for name, which, field, meaning in (("MCShiftMedian", "mixed", "shift", "spread (sd over redraws of the posterior samples) of the posterior medians, in percent of the 90% half-width, 2000 samples per event, mixed set (larger of H0 and f_AGN)"),
                                        ("MCShiftMedianAGN", "agn", "shift", "the same for the all-AGN-hosted set (H0)"),
                                        ("MCWidthChange", "mixed", "width", "spread (sd over redraws) of the 90% widths in percent of their mean, 2000 samples per event, mixed set (larger of H0 and f_AGN)"),
                                        ("MCWidthChangeAGN", "agn", "width", "the same for the all-AGN-hosted set (H0)")):
        val = summ(which, field)
        if val is not None:
            add(name, "plain", fnum(val, 0), val, "%", meaning, srcS, "appendixA,item4")
        else:
            add_ph(name, r"\ph{XX}", "%", meaning, A14_RUN, "appendixA,item4")


A15_SCANS = [("p1_complete", "CompleteFive", "complete catalogs (nominal-limit representation), normalised to z = 5, input densities"),
             ("p1_m21", "TwentyOneFive", "m < 21, normalised to z = 5, input densities"),
             ("p1_m18", "EighteenFive", "m < 18, normalised to z = 5, input densities"),
             ("ref_complete_n0m24", "CompleteZeroDensity", "complete catalogs, zero density of uncataloged hosts (Z_k = N_k), normalised to z = 5"),
             ("c_complete_x2", "CompleteFiveTwo", "complete catalogs, normalised to z = 5, AGN density twice its input"),
             ("b_complete_x1", "CompleteProd", "complete catalogs, normalised to z = 1.10, input densities (Z_k = N_k)"),
             ("b_complete_x2", "CompleteProdTwo", "complete catalogs, normalised to z = 1.10, AGN density twice its input"),
             ("d_m18_x1", "EighteenProd", "m < 18, normalised to z = 1.10, input densities"),
             ("d_m18_x05", "EighteenProdHalf", "m < 18, normalised to z = 1.10, AGN density half its input"),
             ("d_m18_x2", "EighteenProdTwo", "m < 18, normalised to z = 1.10, AGN density twice its input")]


def a15_rate_ratio(rec):
    """R quantiles from the scan's f quantiles and its normalisers at H0 = 67.74 (the Z ratio varies by < 1% over
    the scanned H0 64-74; exact on the complete catalogs normalised to 1.10, where Z_k = N_k)."""
    Z = rec["normaliser"]["Z_at_67.74"]
    ratio = Z["GAL"] / Z["AGN"]
    s = rec["measured"]["f"]
    odds = lambda f: f / (1 - f)
    return {"median": odds(s["median"]) * ratio, "ci68": [odds(v) * ratio for v in s["ci68"]], "ci90": [odds(v) * ratio for v in s["ci90"]], "z_ratio_at_67.74": ratio}


def a15_scan_post(scan, x0s, Rin):
    """From the scan file next to the summary: two-sided levels of the inputs in the H0 and f marginals (trapezoid on
    the scanned cells, as the aggregation), and the rate-per-host ratio with the scan's own Z_k per H0 row."""
    p = A15_SUMMARY.parent / "scans" / f"{scan}.h5"
    if not p.exists():
        return None
    with h5py.File(p, "r") as h:
        Hg, Fg, L = h["H0_grid"][()], h["f_grid"][()], h["lnL"][()]
        nH, Z = h["normaliser/H0"][()], h["normaliser/Z"][()]
    assert np.allclose(nH[:-1], Hg)
    P = np.where(np.isfinite(L), np.exp(L - np.nanmax(np.where(np.isfinite(L), L, -np.inf))), 0.0)
    out = {"file": rel(p)}
    for pn, x, m in (("H0", Hg, P @ trap_w(Fg)), ("f", Fg, trap_w(Hg) @ P)):
        c = np.concatenate([[0.0], np.cumsum(0.5 * (m[1:] + m[:-1]) * np.diff(x))])
        c /= c[-1]
        out[pn] = level_from_cdf((x, c), x0s[pn])[0]
    pH = (P @ trap_w(Fg)) * trap_w(Hg)
    pH /= pH.sum()
    cdf_f = np.concatenate([np.zeros((len(Hg), 1)), np.cumsum(0.5 * (P[:, 1:] + P[:, :-1]) * np.diff(Fg)[None, :], axis=1)], axis=1)
    cdf_f /= np.where(cdf_f[:, -1:] > 0, cdf_f[:, -1:], 1.0)
    zr = Z[:-1, 0] / Z[:-1, 1]
    odds = lambda f: f / (1 - f)
    r = np.geomspace(odds(max(Fg[0], 1e-4)) * zr.min() / 1.5, odds(min(Fg[-1], 1 - 1e-4)) * zr.max() * 1.5, 20001)
    F = np.zeros_like(r)
    for i in np.where(pH > 0)[0]:
        F += pH[i] * np.interp(expit(np.log(r / zr[i])), Fg, cdf_f[i])
    F /= F[-1]
    q = lambda u: float(np.interp(u, F, r))
    out["R"] = {"median": q(0.5), "ci68": [q(0.16), q(0.84)], "ci90": [q(0.05), q(0.95)], "_cdf": (r, F), "z_ratio_range": [float(zr.min()), float(zr.max())]}
    return out


def build_a15():
    S = jload(A15_SUMMARY) if A15_SUMMARY.exists() else None
    srcS = {"file": rel(A15_SUMMARY), "computation": "read from the Analysis-15 summary (aggregate_normalisation.py: trapezoid posteriors on the scanned cells, H0 step 0.25, f step 0.025)"}
    sc = {} if S is None else S.get("scans", {})
    done = lambda n: n in sc and sc[n].get("status") == "done"
    Rin = [r for r in REG if r["name"] == "MockRateRatio"][0]["value"]

    def cont(s, x0, lev):
        return {"input": x0, "in68": bool(s["ci68"][0] <= x0 <= s["ci68"][1]), "in90": bool(s["ci90"][0] <= x0 <= s["ci90"][1]),
                "level_two_sided": lev, "cdf_at_input": None}
    for scan, word, desc in A15_SCANS:
        gp = a15_scan_post(scan, {"H0": IN["H0"], "f": IN["f"]}, Rin) if done(scan) else None
        for pn, pw, unit, x0, ph in (("H0", "Hzero", "km/s/Mpc", IN["H0"], r"\ph{XX.X^{+X.X}_{-X.X}}"), ("f", "f", "", IN["f"], r"\ph{0.XX^{+0.XX}_{-0.XX}}")):
            meaning = f"{pn if pn == 'H0' else 'f_AGN'}, {desc} (Analysis 15, darksirens-core e7c3007, catalog A)"
            if done(scan):
                s = sc[scan]["measured"][pn]
                r_, d = fci(s["median"], s["ci90"][0], s["ci90"][1])
                if gp is not None:
                    add(f"Norm{pw}{word}", "plain", r_, s, unit, meaning + " (median and 90% interval)", srcS, "C5,item3", containment=cont(s, x0, gp[pn]), extra={"decimals": d})
                else:
                    add(f"Norm{pw}{word}", "plain", r_, s, unit, meaning + " (median, 90% interval; scan file absent: no containment level)", srcS, "C5,item3", extra={"decimals": d})
            else:
                add_ph(f"Norm{pw}{word}", ph, unit, meaning + " (median and 90% interval)", A15_RUN, "C5,item3")
        meaning = RR_MEANING + f"; {desc} (Analysis 15)"
        if done(scan) and gp is not None:
            add_post(f"NormRateRatio{word}", gp["R"], "", meaning + "; Z_k of the scan per H0 row", {"files": [rel(A15_SUMMARY), gp["file"]], "computation": "R CDF summed over the scanned H0 rows (trapezoid), Z_k from the scan's normaliser group"},
                     "C5,item11", Rin, cls="plain")
        elif done(scan):
            R = a15_rate_ratio(sc[scan])
            r_, d = fci(R["median"], R["ci90"][0], R["ci90"][1])
            add(f"NormRateRatio{word}", "plain", r_, R, "", meaning + "; from the summary's f quantiles and Z_k at 67.74 (median, 90% interval)", dict(srcS, computation=a15_rate_ratio.__doc__.replace("\n", " ")), "C5,item11", extra={"decimals": d, "input": Rin})
        else:
            add_ph(f"NormRateRatio{word}", r"\ph{XX^{+X}_{-X}}", "", meaning + " (median and 90% interval)", A15_RUN, "C5,item11")
    # response to a factor two in the assumed AGN density, measured and predicted, per normalisation volume
    def fmed(n, which="measured", p="f"):
        if not done(n):
            return None
        if which == "measured":
            return sc[n]["measured"][p]["median"]
        pr = sc[n].get("prediction") or {}
        return pr.get("predicted", {}).get(p, {}).get("median")
    pairs = [("CompleteFive", "c_complete_x2", "p1_complete", "pred-pred", "complete catalogs normalised to z = 5"),
             ("CompleteProd", "b_complete_x2", "b_complete_x1", "pred-pred", "complete catalogs normalised to z = 1.10 (F_k = 1: predicted zero)"),
             ("EighteenProd", "d_m18_x2", "d_m18_x1", "pred-meas", "m < 18 normalised to z = 1.10")]
    for word, n2, n1, how, desc in pairs:
        for p, pw, unit, d in (("f", "f", "", 3), ("H0", "Hzero", "km/s/Mpc", 2)):
            m2, m1 = fmed(n2, p=p), fmed(n1, p=p)
            meaning = f"change of the {p if p == 'H0' else 'f_AGN'} median when the assumed AGN density is doubled, {desc} (Analysis 15, measured)"
            if m2 is not None and m1 is not None:
                add(f"NormDelta{pw}Two{word}", "plain", fnum(m2 - m1, d), m2 - m1, unit, meaning, srcS, "C5,item3")
            else:
                add_ph(f"NormDelta{pw}Two{word}", r"\ph{0.XX}", unit, meaning, A15_RUN, "C5,item3")
            p2 = fmed(n2, "pred", p)
            p1 = fmed(n1, "pred", p) if how == "pred-pred" else m1
            meaning = f"the same predicted by the identity (LIKELIHOOD_MATH sec. 8; Analysis 15's aggregation, {'both from the zero-density grid' if how == 'pred-pred' else 'normaliser part from the input-density scan'}), {desc}"
            if p2 is not None and p1 is not None and np.isfinite(p2) and np.isfinite(p1):
                add(f"NormDelta{pw}Two{word}Pred", "plain", fnum(p2 - p1, d), p2 - p1, unit, meaning, srcS, "C5,item3")
            else:
                add_ph(f"NormDelta{pw}Two{word}Pred", r"\ph{0.XX}", unit, meaning, A15_RUN, "C5,item3")
    ds = None if S is None else S.get("checks", {}).get("d_slope")
    for name, key, meaning in (("SlopefAGNDepthNorm", "measured_slope_per_dex", "d f_AGN / d log10 n_AGN at m < 18 with the hosts normalised to z = 1.10 (factors 0.5 and 2; run R5b's first number, Analysis 15)"),
                               ("SlopefAGNDepthNormPred", "identity_predicted_slope_per_dex", "the same slope predicted by the identity from the input-density scan (normaliser part only)")):
        val = None if ds is None else ds.get(key)
        if val is not None and np.isfinite(val):
            add(name, "plain", fnum(val, 2), val, "per dex", meaning, srcS, "C5,item3")
        else:
            add_ph(name, r"\ph{0.XX}", "per dex", meaning, A15_RUN, "C5,item3")


def build_r2_placeholders():
    P = [("CommonFreeHzero", r"\ph{XX.X^{+X.X}_{-X.X}}", "km/s/Mpc", "H0 of the single population with its own free mass-peak location and width and spin mean and width, host-dependent mock, both catalogs, H0 prior [50, 100]",
          "run R8' single population (runs.txt, revision of 17 Nov; R8e of the first list)", "C7,item1"),
         ("CommonFreefAGN", r"\ph{0.XX^{+0.XX}_{-0.XX}}", "", "f_AGN of the same model", "same run", "C7,item1"),
         ("LBHDVsCommonFree", r"\ph{X.X\pm X.X}", "log10", "log10 Bayes factor of the host-dependent populations (eight parameters) against that single population, one code and one prior box",
          "runs R8' (both models, evidences)", "C7,C8,item1"),
         ("NoCatHzeroComb", r"\ph{XX.X^{+X.X}_{-X.X}}", "km/s/Mpc", "H0 from the mass distribution alone (no catalog: hosts uniform in comoving volume), combined over the realizations of R1'", "run R11", "C2,item2"),
         ("NoCatHzeroCatA", r"\ph{XX.X^{+X.X}_{-X.X}}", "km/s/Mpc", "the same on realization A", "run R11", "C2,item2"),
         ("NoCatHalfNinety", r"\ph{X.X}", "km/s/Mpc", "mean 90% half-width of H0 from the mass distribution alone", "run R11", "C2,item2")]
    for name, r, units, meaning, run, claim in P:
        add_ph(name, r, units, meaning, run, claim)


def build_r2():
    build_design()
    build_normalisation()
    build_identity()
    build_rate_ratio()
    build_event_completeness()
    build_hosts_per_volume()
    build_misc_requests()
    build_spin_and_offsets()
    build_a14()
    build_a15()
    build_r2_placeholders()


# ============================================================================================= #
# THE KEY SCHEME (notes/PAPER_SKELETON.md Sec. 6): \<Key><Quantity>[Comb][Med|Lo|Hi|Wid], \<Key><Quantity>CL
# from values/placeholder_results.json (scripts/placeholder_results.py); red while placeholders, black when
# values/production_map.json names an existing production summary for the key.
# ============================================================================================= #
Q_DESC = {"Hzero": "H0", "fAGN": "f_AGN (fraction of mergers hosted by AGN)", "MuG": "mass-peak location", "SigmaG": "mass-peak width",
          "MuChi": "spin mean (effective spin)", "SigmaChi": "spin width", "DmuG": "AGN offset of the mass peak", "DmuChi": "AGN offset of the spin mean",
          "LognGal": "log10 n_gal", "LognAGN": "log10 n_AGN", "RateRatio": "merger rate per AGN over merger rate per galaxy"}


def build_keys():
    import placeholder_results as PR
    doc = PR.main()
    K = doc["keys"]
    pj = OUT / "values" / "placeholder_results.json"
    for key, E in K.items():
        final = bool(E.get("final"))
        cls = "plain" if final else "ph"
        wrap = (lambda r: r) if final else (lambda r: f"\\ph{{{r}}}")
        desc = f"[key {key}] {E['events']}; {E['analysis']}; {E['catalogs']}"
        for q, Q in E["quantities"].items():
            srcq = {"file": E["production_file"]} if final else {"missing": E["stage"], "key": key, "placeholder_source": f"A: {Q['source']}; Comb: {Q['comb_source']}",
                                                                  "placeholder_file": {"A": Q.get("file"), "Comb": Q.get("comb_file")}, "json": rel(pj)}
            note = f" ({Q['note']})" if Q.get("note") else ""
            for part, tag in (("A", ""), ("Comb", "Comb")):
                sm = Q[part]
                where = "realization A" if part == "A" else f"combined over {E['N_comb']} realizations"
                r_, d = fci(sm["median"], sm["ci90"][0], sm["ci90"][1])
                val = {k: sm[k] for k in ("median", "ci68", "ci90", "ul90", "level_of_input") if k in sm}
                base = f"{key}{q}{tag}"
                mean = f"{Q_DESC[q]}, {desc}, {where}{note}"
                add(base, cls, wrap(r_), val, Q["units"], mean + " (median and 90% equal-tailed interval)", srcq, "skeleton,keys", extra={"decimals": d, "input": Q["input"]})
                for pc, v, w in (("Med", sm["median"], "median"), ("Lo", sm["ci90"][0], "lower end of the 90% interval"), ("Hi", sm["ci90"][1], "upper end of the 90% interval"),
                                 ("Wid", sm["ci90"][1] - sm["ci90"][0], "full width of the 90% interval")):
                    add(base + pc, cls, wrap(fnum(v, d)), v, Q["units"], f"{w} of {mean}", srcq, "skeleton,keys")
                if part == "A" and q in ("Hzero", "fAGN"):
                    hwv = 0.5 * (sm["ci90"][1] - sm["ci90"][0])
                    add(base + "HalfWid", cls, wrap(fnum(hwv, d)), hwv, Q["units"],
                        f"half of the full 90% width ({base}Wid) of {mean}: the precision of one catalog of 1000 mergers, to quote beside a bias", srcq, "skeleton,keys")
                if (q == "fAGN" and key == "CtrlZero") or (q in ("LognGal", "LognAGN")):
                    add(base + "UL", cls, wrap(fnum(sm["ul90"], max(d, 2))), sm["ul90"], Q["units"], f"90% upper limit (90th percentile) of {mean}", srcq, "skeleton,keys")
                if part == "A" and q in ("LognGal", "LognAGN") and "prior_ul90" in sm:
                    add(base + "PriorUL", cls, wrap(fnum(sm["prior_ul90"], max(d, 2))), sm["prior_ul90"], Q["units"],
                        f"the 90% upper limit the priors allow once the rate per host is measured (prior-induced density), beside {base}UL; {desc}", srcq, "skeleton,keys")
                # the same two limits in h^3 Mpc^-3, the unit the production analyses sample (the text's unit; requested 17 Nov, MACRO_REQUESTS F1)
                hn = f" ({sm['h_note']})" if sm.get("h_note") else ""
                if part == "A" and q in ("LognGal", "LognAGN") and "ul90_h" in sm:
                    add(base + "ULh", cls, wrap(fnum(sm["ul90_h"], max(d, 2))), sm["ul90_h"], "log10 h^3 Mpc^-3",
                        f"90% upper limit (90th percentile) of {Q_DESC[q]} in h^3 Mpc^-3{hn}, {desc}, realization A; {base}UL is the same limit in Mpc^-3", srcq, "skeleton,keys")
                if part == "A" and q in ("LognGal", "LognAGN") and "prior_ul90_h" in sm:
                    add(base + "PriorULh", cls, wrap(fnum(sm["prior_ul90_h"], max(d, 2))), sm["prior_ul90_h"], "log10 h^3 Mpc^-3",
                        f"the 90% upper limit on {Q_DESC[q]} in h^3 Mpc^-3 that the priors allow once the rate per host is measured (prior-induced density){hn}, "
                        f"beside {base}ULh; {desc}; {base}PriorUL is the same limit in Mpc^-3", srcq, "skeleton,keys")
            if Q["input"] is not None:
                C = Q["Comb"]
                lev = C.get("level_of_input")
                edge = key == "CtrlZero" and q == "fAGN"
                if edge or lev is None:
                    add(f"{key}{q}CL", cls, wrap("--"), None, "%", f"credible level of the input in {key}{q}Comb: not defined, the input lies at the prior edge (use {key}{q}CombUL)",
                        srcq, "skeleton,keys")
                else:
                    add(f"{key}{q}CL", cls, wrap(fpct(lev)), lev, "%", f"two-sided credible level at which the input {Q['input']:g} lies in {key}{q}Comb ({Q_DESC[q]}, {desc})",
                        srcq, "skeleton,keys")
        if E.get("LB"):
            L = E["LB"]
            srcl = {"file": E["production_file"]} if final else {"missing": E["stage"], "key": key, "placeholder_source": L["source"], "placeholder_file": L.get("file"), "json": rel(pj)}
            ncomb = E["N_comb"]
            for part, tag in (("A", ""), ("Comb", "Comb")):
                if L.get(part) is None:
                    continue
                where = "realization A" if part == "A" else f"combined over {ncomb} realizations"
                add(f"{key}LB{tag}", cls, wrap(flb(L[part])), L[part], "log10", f"log10 Bayes factor, {desc}, against {L['reference']}, {where}", srcl, "skeleton,keys")
    # further macros of Sec. 6
    q = lambda k, Q, part="Comb": K[k]["quantities"][Q][part]
    srcf = lambda keys: {"missing": "; ".join(sorted({K[k]["stage"] for k in keys})), "key": ",".join(keys), "placeholder_source": "derived from the placeholder entries", "json": rel(pj)}
    fin = lambda keys: all(K[k].get("final") for k in keys)

    def addf(name, keys, render, value, units, meaning):
        f_ = fin(keys)
        add(name, "plain" if f_ else "ph", render if f_ else f"\\ph{{{render}}}", value, units, meaning,
            {"files": [K[k]["production_file"] for k in keys]} if f_ else srcf(keys), "skeleton,keys")
    for pre in ("HD", "Obs"):
        b = q(f"{pre}Single", "Hzero")["median"] - IN["H0"]
        addf(f"{pre}BiasHzero", [f"{pre}Single"], fnum(b, 1), b, "km/s/Mpc", f"median of {pre}SingleHzeroComb minus the input H0 (bias of the single-population analysis of host-dependent events, {'peak-dominated' if pre == 'HD' else 'observed-like'} spectrum, combined)")
        L = K[f"{pre}Host"]["LB"]["A"]
        addf(f"{pre}LB", [f"{pre}Host"], flb(L), L, "log10", f"log10 Bayes factor of the host-dependent populations against the single population, same events and priors, realization A (= {pre}HostLB; combined: {pre}HostLBComb)")
    # one definition (17 Nov): the ratio of the two printed widths, GalhostedHzeroWid / AGNhostedHzeroWid (realization A, full 90%
    # widths, unrounded). The earlier value was PureRatioNinety (ratio of the mean half-widths over 93 exploratory realizations).
    wg, wa = q("Galhosted", "Hzero", "A"), q("AGNhosted", "Hzero", "A")
    wr = (wg["ci90"][1] - wg["ci90"][0]) / (wa["ci90"][1] - wa["ci90"][0])
    addf("WidthRatioAGNGal", ["Galhosted", "AGNhosted"], fnum(wr, 1), wr, "",
         "full 90% width of H0 of the all-galaxy-hosted set with the galaxy catalog over that of the all-AGN-hosted set with the AGN catalog, realization A "
         "(= GalhostedHzeroWid / AGNhostedHzeroWid, unrounded widths)")
    for nm, part, where in (("KnownBothCorrHzerofAGN", "A", "realization A (= JointRhoCatA)"), ("KnownBothCorrHzerofAGNComb", "Comb", "combined over realizations (= JointRhoComb)")):
        v = K["KnownBoth"]["joint"][part]["correlation"]
        addf(nm, ["KnownBoth"], fnum(v, 2), v, "", f"correlation coefficient r of H0 and f_AGN, population known, both catalogs, {where}")
    for nm, kn, kb in (("MassScaleShareKnown", "KnownNone", "KnownBoth"), ("MassScaleShareSingle", "SingleNone", "SingleBoth")):
        a, b_ = q(kn, "Hzero", "A"), q(kb, "Hzero", "A")
        v = (a["ci90"][1] - a["ci90"][0]) / (b_["ci90"][1] - b_["ci90"][0])
        addf(nm, [kn, kb], fnum(v, 1), v, "", f"90% width of H0 with no catalog over that with both catalogs, {'population known' if 'Known' in nm else 'single population'}, realization A")
    for nm, k in (("CorrHzeroMuGNone", "SingleNone"), ("CorrHzeroMuGBoth", "SingleBoth")):
        v = K[k]["joint"]["A"]["correlation"]
        addf(nm, [k], fnum(v, 1), v, "", f"correlation coefficient r of H0 and the mass-peak location, single population, {'no catalog' if 'None' in nm else 'both catalogs'}, realization A")


# ============================================================================================= #
# THE EXPERIMENT (notes/PAPER_SKELETON.md Sec. 9): the constraint from each host class, the cost of an unknown host
# class, the probability of an AGN host, the AGN catalog alone (scripts/classes.py)
# ============================================================================================= #
CLASS_DEFINITIONS = """## The constraint from each host class (Sec. IV B; notes/PAPER_SKELETON.md Sec. 9; scripts/classes.py)

All with the population known and complete catalogs, 1000 detected mergers per realization; half-widths are of H0 in
km/s/Mpc. The runs are those of `values/production_map.json`, block `classes` (Analysis 16's experiment,
`RUNLIST_experiment.yaml`, stages E1 and E2, and the production keys named), and each macro's meaning below names them.
A macro prints black when every run summary it needs exists, red with the expected value (the archived letter's,
`placeholder_results.json` `classes_expected`) otherwise; a derived macro is black when all its inputs are.

- **`\\HW<spectrum><analysis><f>`**: the mean over the \\NrealClasses{} realizations of half the 90% equal-tailed width of
  the H0 posterior of each realization (its `summary.json`, `posterior.H0`); **`...Min`, `...Max`**: the smallest and the
  largest of those half-widths (the bars of fig_classes). Every `\\HW...` macro, `Min` and `Max` included, prints two
  significant figures (3.5, 2.0, 0.82; since 17 Nov evening, REVIEW_2 2.2; before, two decimals); every derived macro (costs,
  shares) is computed from the unrounded values in values.json. Spectrum: none = peak-dominated, `Obs` = observed-like.
  f: `Zero`, `Ten`, `Thirty`, `Fifty`, `Hundred` = input f_AGN 0, 0.1, 0.3, 0.5, 1. Analysis:
  `Unknown` host class unknown (both catalogs, f_AGN inferred); `Known` host class known (each merger with the catalog and
  population of its true class, the two posteriors multiplied; at f = 0 this is the galaxy catalog alone, the horizontal
  line "galaxy catalog alone, f_AGN = 0" of fig_classes; at f = 1 the AGN catalog alone on all-AGN-hosted mergers);
  `None` no catalog (the mass spectrum alone, on the f = 0 set); `Pops` host-dependent populations (AGN offsets
  `\\MockDmuG` Msun and `\\MockDmuChi`) held at their inputs, host class unknown, on host-dependent mergers; `PopsTwice`
  the same with twice the offsets; `KnownPops`, `KnownPopsTwice` the host class known on those same host-dependent mergers
  (each merger with the catalog and the population of its true host class, each population at its input, the two posteriors
  multiplied: `\\HWKnownPops<f>`, `\\HWKnownPopsTwice<f>` at f = 0.1, 0.3, 0.5 and `\\HWObsKnownPopsThirty`; not drawn in
  fig_classes; red values while their runs are missing: `\\HWKnown<f>` rescaled by 1/hw^2 = (1 - x)/hw_gal^2 + x/hw_AGN^2 from
  the detected AGN-hosted fraction x of the equal-population events to that of the host-dependent events);
  `Host` host-dependent populations inferred with H0 (keys `HDHost`, `ObsHost`; f = 0.3;
  \\NrealHD{} and \\NrealObs{} realizations); `KnownAGN` the AGN-hosted mergers alone with the host class known (the AGN
  factor of the host-class-known grid, which no summary holds: Akima-refined lnL_agn of the fine pass, the summaries'
  own method).
- **`\\CostUnknown<f>`, `\\CostPops<f>`, `\\CostPopsTwice<f>`**: (HW of the analysis / HW of the host class known on the same
  mergers)^2 from the unrounded means: the number of mergers the unknown host class costs per merger with it known.
  Denominators: `\\CostUnknown<f>` uses `\\HWKnown<f>` (the equal-population mergers); `\\CostPops<f>` uses
  `\\HWKnownPops<f>` and `\\CostPopsTwice<f>` uses `\\HWKnownPopsTwice<f>` (the host-dependent mergers themselves; until
  17 Nov evening both used `\\HWKnown<f>`, which mixed the cost of the unknown host class with the larger share of AGN-hosted
  mergers among the detected host-dependent ones). Each macro's line in the table below names its denominator. The
  central value is the ratio of the means (of the half-widths over the realizations), not the mean of the per-realization
  ratios. **`...Min`, `...Max`** (one decimal, like the cost): the smallest and the largest over the realizations of the
  same ratio computed per realization, (hw_num[r] / hw_den[r])^2, with the two half-widths of one realization from their
  summaries. While either half-width is red: the central value times `\\CostUnknown<f>Min` / `\\CostUnknown<f>` and
  `\\CostUnknown<f>Max` / `\\CostUnknown<f>` (the relative spread of the finished equal-population cost at the same f), or
  times 0.8 and 1.2 where that cost is itself red.
- **`\\ShareAGNTen`, `\\ShareAGNThirty`** (percent): 100 (`\\HWKnown<f>` / `\\HWKnownAGN<f>`)^2 from the unrounded means:
  the share of the information on H0 (inverse variance) carried by the AGN-hosted mergers with the host class known. The
  two posteriors multiply, so for Gaussian posteriors 1/hw^2 = 1/hw_AGN^2 + 1/hw_gal^2 in each realization; values.json
  records the galaxy share computed the same way, and the two add to 100% within the non-Gaussianity. No share is defined
  for the host class unknown. **`\\ShareSumThirty`** (percent): `\\ShareAGNThirty` plus that galaxy share at f = 0.3,
  100 (`\\HWKnownThirty` / HW_gal)^2 with HW_gal the mean over the realizations of half the 90% width of H0 of the galaxy factor
  of the same host-class-known grids (`run_posteriors.factor_summary`, as `\\HWKnownAGNThirty` for the AGN factor); its excess
  over 100 is the size of the caveat on `\\ShareAGNThirty`. Black when `\\ShareAGNThirty` is; red 100 otherwise.
- **`\\Prob<AGN|Gal><info>`**: realization A, input f_AGN = 0.3: the mean over the mergers truly hosted by AGN (galaxies) of
  the probability of an AGN host, f L_AGN / (f L_AGN + (1 - f) L_gal), at the posterior median of the grid (the run's
  `grid/membership.npz`; = `summary.json` `membership.mean_r_true_agn` / `_gal`). info: `Equal` equal populations,
  positions alone; `Pos` host-dependent populations, positions alone (one population for both classes, held at its
  input); `Pops` host-dependent populations, positions with masses and spins (the two populations at their inputs);
  `PosTwice`, `PopsTwice` the same with twice the offsets. **`\\ProbEvalfAGN<info>`**: the f_AGN at which the probability
  of that run is evaluated, the posterior median of f_AGN of its grid (`summary.json` `membership.f_agn_median`, = the
  run's `grid/membership.npz` `f_agn`; it agrees with `posterior.f_agn.median` to 1e-4). A merger whose two class
  likelihoods are equal has exactly this probability, not the input 0.30 (the dashed line of fig_classes (c) stays at the
  input). **`\\ProbAll<info>`**: the mean of the same probability over all the mergers of the set (both true classes;
  `membership.mean_r`). Red values: `ProbEvalfAGN` the input 0.30 (`Equal`, `Pos`, `Pops`) or the value with the smaller
  offsets (`PosTwice`, `PopsTwice`); `ProbAll` the two class means weighted by the AGN-hosted fraction of the 1000 detected
  mergers of realization A of that set. Two decimals, like the class means.
- **`\\KnownAGNNoPeak`** (a word): realizations of key `KnownAGN` (the AGN catalog alone on equal-population mergers, three in
  ten AGN-hosted) whose H0 posterior has no interior maximum in the prior range [50, 100]: the largest corrected ln L
  over the nodes of the fine and coarse passes lies at a prior edge. **`\\KnownAGNPeak<n>Hzero[Med|Lo|Hi]`** (n = `One`,
  `Two`, ... in the order of the run list): median and 90% interval (its `summary.json`) of each realization that has an
  interior maximum; **`...CL`**: the credible level at which the input H0 lies in that posterior.
- **`\\NrealClasses`** (a word): the realizations of every experiment set above (each set has the same ones).
- **`\\MockLognGalh`, `\\MockLognAGNh`**: the input number densities as log10 in h^3 Mpc^-3, the units in which the production
  analyses sample them (`\\MockLognGal` - 3 log10 h), beside the Mpc^-3 macros, for Table I.
"""


def build_classes():
    import classes as CL
    CL.build(add, fnum, fci, fpct, NUM_WORD)
    N, reals, sets = CL.nreal_classes()
    add("NrealClasses", "plain", NUM_WORD.get(N, str(N)), N, "realizations",
        f"realizations of every experiment set of Sec. IV B (the HW..., Cost..., Share... macros), realizations {reals}",
        {"file": rel(CL.RUNLIST_EXP), "computation": f"realizations of the {len(sets)} experiment sets named in values/production_map.json 'classes'"},
        "design,realizations", extra={"realizations": reals, "sets": sets})
    # the galaxy share with the host class known, for the record (the two shares add to 100% for Gaussian posteriors)
    for f in ("Ten", "Thirty"):
        r = next(x for x in REG if x["name"] == f"ShareAGN{f}")
        r["extra"]["galaxy_share_percent"] = _gal_share(CL, f)
    # the sum of the two shares at f = 0.3 (MACRO_REQUESTS F1): the size of the caveat on ShareAGNThirty (100 for Gaussian posteriors)
    r = next(x for x in REG if x["name"] == "ShareAGNThirty")
    g = r["extra"]["galaxy_share_percent"]
    meaning = ("sum, in percent, of the shares of the information on H0 (inverse variance) of the two host classes with the host class known, input "
               "f_AGN = 0.3: ShareAGNThirty plus the galaxy share computed the same way, 100 (HWKnownThirty / mean half-width of the galaxy factor)^2 "
               "(values.json ShareAGNThirty.extra.galaxy_share_percent); 100 for Gaussian posteriors, so its excess over 100 is the size of the "
               "caveat on ShareAGNThirty")
    if r["class"] == "plain" and g is not None:
        v = float(r["value"]) + float(g)
        rec = CL.mapped()["macros"]["HWKnownAGNThirty"]
        files = sorted(set(r["source"].get("files", [])) | {rel(CL.run_dir(rec["run"], s) / "summary.json") for s in rec["realizations"]})
        add("ShareSumThirty", "plain", fnum(v, 0), v, "%", meaning,
            {"files": files, "computation": "ShareAGNThirty + 100 (HWKnownThirty / HW_gal)^2, HW_gal = mean over the realizations of half the 90% width of H0 of the "
                                            "galaxy factor of the host-class-known grids (run_posteriors.factor_summary 'gal', as HWKnownAGNThirty for 'agn')"},
            "classes", extra={"agn_share_percent": float(r["value"]), "galaxy_share_percent": float(g), "decimals": 0})
    else:
        add("ShareSumThirty", "ph", r"\ph{100}", 100.0, "%", meaning,
            {"missing": r["source"].get("missing", "the runs of ShareAGNThirty") if r["class"] == "ph" else "the galaxy factor of the runs of HWKnownAGNThirty",
             "placeholder_source": "invented: the two shares add to 100% for Gaussian posteriors"}, "classes")


def build_density():
    """The response to the assumed AGN number density at m < 20 and m < 18 (Analysis 16, stage 2; scripts/density.py)."""
    import density as DN
    DN.build(add, add_ph, fnum, fci, fsig, NUM_WORD)


def density_definitions():
    import density as DN
    return DN.DEFINITIONS


def _gal_share(CL, f):
    """100 (HWKnown<f> / mean half-width of the galaxy factor)^2, the galaxy counterpart of ShareAGN<f> (values.json only)."""
    rec = CL.mapped()["macros"].get(f"HWKnownAGN{f}")
    known = CL.evaluate(f"HWKnown{f}")
    if rec is None or not known.real or not CL.evaluate(f"HWKnownAGN{f}").real:
        return None
    import run_posteriors as RP
    hw = [0.5 * (lambda s: s["ci90"][1] - s["ci90"][0])(RP.factor_summary(CL.run_dir(rec["run"], r), "gal")[0]) for r in rec["realizations"]]
    return 100.0 * (known.value / float(np.mean(hw))) ** 2


# ============================================================================================= #
# REQUESTS OF 17 NOV (paper_prd/notes/MACRO_REQUESTS.md A1-A3 and the requests of the archived first short draft) and the
# observed-like mock family (data/obslike)
# ============================================================================================= #
NUM_WORD = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten", 11: "eleven", 12: "twelve"}
OBSJ = DATA / "obslike" / "results" / "obslike_paper_values.json"
REQ_DEFINITIONS = """## Definitions of the macros added on 17 Nov (second round of requests)

- **Realization A.** Every key-scheme macro without `Comb` (and every `...HalfWid`) is realization A, i.e. one catalog pair
  with 1000 detected mergers. `\\HDLB` and `\\ObsLB` are realization A (= `\\HDHostLB`, `\\ObsHostLB`); the sums over the
  combined realizations are `\\HDHostLBComb`, `\\ObsHostLBComb`.
- **`<Key>HzeroHalfWid`, `<Key>fAGNHalfWid`.** Half of the full 90% width of realization A (`<Key>...Wid` / 2), the
  precision of 1000 mergers, to quote beside a bias (`\\HDBiasHzero`, `\\ObsBiasHzero` are combined medians minus the input).
- **`\\WidthRatioAGNGal`** (one definition): `\\GalhostedHzeroWid` / `\\AGNhostedHzeroWid`, the two full 90% widths of
  realization A, unrounded. It used to be `\\PureRatioNinety` (ratio of the mean half-widths over 93 exploratory
  realizations, 4.7), which does not divide the two printed widths.
- **Numbers of realizations, `\\Nreal...`** (plain, printed as words below 13): the realizations of the key's runs in
  `analyses/analysis_16_production_runs/runs/RUNLIST.yaml` (realization A is 100). Keys not yet in the run list take
  runs.txt: the observed-like spectrum (Stage 5, five) and the two control sets (run R4, "five for the others").
  The keys of one macro must have one count; where they differ the macro is split: `\\NrealKnown` (both catalogs) and
  `\\NrealKnownAlone` (each catalog alone, no catalog); `\\NrealHD` (host-dependent populations), `\\NrealHDSingle`,
  `\\NrealHDNone`. The `Comb` placeholders are combined over the same N.
- **Observed-like spectrum, `\\Obs<design>`** (plain): darksirens-core's power law + peak at the GWTC-3 medians
  (`data/obslike/README.md`). `\\ObsPeakFrac` is the weight of the Gaussian peak in core's normalisation (0.092): GWTC-3's
  lambda_peak = 0.038 (`\\ObsLambdaPeak`) multiplies a mixture that is tapered and renormalised as a whole, so its fitted
  density has 9.2% of the mergers in the peak. Inputs of the `Obs...` keys: mass peak `\\ObsMuG`, width `\\ObsSigmaG`.
- **Detected samples, `\\<fam>Det<quantity>`**: realization A, 1000 detected mergers; fam `Mock` (peak-dominated, equal
  populations), `MockHD` (peak-dominated, host-dependent), `Obs`, `ObsHD` (observed-like). PeakShare: share of detected
  mergers in the Gaussian peak (observed-like: the realised component flag; peak-dominated: the mean membership
  probability, the files store no flag); zMedian, zMax: true redshifts; MoneMedian, MtwoMedian: source-frame masses [Msun].
  Five-realization means and sd in values.json (extra).
- **AGN counts at m < 18, `\\MockNAGNCompleteEighteen`**: AGN of realization A's m < 18 AGN catalog (survey file, photometric
  redshifts) below the redshift to which that catalog is complete (AGN selection fit, z_c = 0.108; the galaxy fit's
  `\\MockzCompleteEighteen` agrees to 0.0005), counted by the production code (`prod_common.density_from_counts`).
  `\\MocknAGNPrecEighteen` = 100/sqrt(N) (Poisson); `\\MocknAGNScatterEighteen` = sd over the five production realizations of
  the density from counts, in percent of the mean (the field's own fluctuation inside z_c, larger than Poisson).
  `\\NormDeltafCountsEighteen` = `\\SlopefAGNDepthNorm` x log10(1 + `\\MocknAGNPrecEighteen`/100): the shift of the f_AGN
  median for an AGN density off by that precision (Analysis 15's measured slope at m < 18, hosts to z = 1.10);
  `\\NormDeltafCountsScatterEighteen` the same for the realised scatter.
"""


def build_requests():
    import placeholder_results as PR
    v = {r["name"]: r for r in REG}
    # ---- A1: numbers of realizations (RUNLIST.yaml; runs.txt where the run list has no runs yet)
    def nreal(name, keys, meaning):
        Ns = {PR.NREAL[k][0] for k in keys}
        assert len(Ns) == 1, (name, {k: PR.NREAL[k][0] for k in keys})
        N = Ns.pop()
        inlist = all(k in PR.RUNREAL for k in keys)
        inexp = all(k in PR.RUNREAL_EXP for k in keys)
        src = ({"file": rel(PR.RUNLIST), "computation": "realizations of the runs of " + "; ".join(f"{k} {PR.KEY_SELECTOR[k]}" for k in keys)} if inlist else
               {"file": rel(PR.RUNLIST_EXP), "computation": "; ".join(f"{k}: {PR.NREAL[k][1]}" for k in keys)} if inexp else
               {"file": "paper_prd/runs.txt", "computation": "; ".join(f"{k}: {PR.NREAL[k][1]}" for k in keys)})
        add(name, "plain", NUM_WORD.get(N, str(N)), N, "realizations", meaning, src, "design,realizations",
            extra={"keys": keys, "realizations": {k: PR.RUNREAL.get(k, PR.RUNREAL_EXP.get(k)) for k in keys}})
    nreal("NrealKnown", ["KnownBoth"], "realizations combined, population known, both catalogs, complete (key KnownBoth)")
    nreal("NrealKnownAlone", ["KnownGal", "KnownAGN", "KnownNone"], "realizations combined, population known, galaxy catalog alone, AGN catalog alone, no catalog (keys KnownGal, KnownAGN, KnownNone)")
    nreal("NrealPure", ["AGNhosted", "Galhosted"], "realizations combined, all-AGN-hosted and all-galaxy-hosted sets (keys AGNhosted, Galhosted)")
    nreal("NrealFlux", [f"Flux{w}" for _, w, _ in RUNGS], "realizations combined, flux-limited catalogs, number densities from the catalog counts (keys Flux...)")
    nreal("NrealFree", [f"Free{w}" for _, w, _ in RUNGS], "realizations combined, number densities inferred (keys Free...)")
    nreal("NrealSingle", ["SingleBoth", "SingleNone"], "realizations combined, single population on equal-population mergers, both catalogs and no catalog (keys SingleBoth, SingleNone)")
    nreal("NrealHD", ["HDHost"], "realizations combined, host-dependent populations analysis of host-dependent mergers, peak-dominated spectrum (key HDHost)")
    nreal("NrealHDSingle", ["HDSingle"], "realizations combined, single-population analysis of host-dependent mergers, both catalogs (key HDSingle)")
    nreal("NrealHDNone", ["HDNone"], "realizations combined, single-population analysis of host-dependent mergers, no catalog (key HDNone)")
    nreal("NrealObs", ["ObsSingle", "ObsHost", "ObsNone"], "realizations combined, observed-like spectrum, all three analyses (keys ObsSingle, ObsHost, ObsNone; Stage 5 of RUNLIST.yaml)")
    nreal("NrealCtrl", ["CtrlZero", "CtrlTenth"], "realizations combined, each of the two control sets (no AGN-hosted mergers; one in ten) (keys CtrlZero, CtrlTenth; the experiment's f = 0 and f = 0.1 "
          "sets with both catalogs, RUNLIST_experiment.yaml" + ("" if all(k in PR.RUNREAL_EXP for k in ("CtrlZero", "CtrlTenth")) else "; run R4 of runs.txt, not yet in a run list") + ")")
    # ---- A2: the observed-like spectrum (design, GWTC-3 medians mapped onto core's power law + peak)
    O = jload(OBSJ)
    th, gw = O["design"]["core_theta"], O["design"]["design_gwtc3_parameters"]
    srcO = lambda key: {"file": rel(OBSJ), "key": key}
    for name, val, d, units, meaning, key in (
            ("ObsPeakFrac", 1 - th["v1"], 3, "", "fraction of the mergers in the Gaussian peak of the observed-like spectrum (weight of the peak in core's normalisation, 1 - v1; GWTC-3's lambda_peak 0.038 of a mixture tapered and renormalised as a whole)", "design.core_theta.v1"),
            ("ObsLambdaPeak", gw["lambda_peak"], 3, "", "GWTC-3's fitted mixing fraction lambda_peak (POWER LAW+PEAK, Sec. VI A), which maps onto ObsPeakFrac", "design.design_gwtc3_parameters.lambda_peak"),
            ("ObsMuG", th["mu_G"], 0, "Msun", "location of the Gaussian peak of the observed-like spectrum (galaxy hosts)", "design.core_theta.mu_G"),
            ("ObsSigmaG", th["sigma_G"], 1, "Msun", "width of the Gaussian peak of the observed-like spectrum", "design.core_theta.sigma_G"),
            ("ObsAlpha", th["alpha"], 1, "", "power-law index of the primary mass of the observed-like spectrum, printed without its sign (p(m1) ~ m1^-alpha)", "design.core_theta.alpha"),
            ("ObsMmin", th["m_min"], 0, "Msun", "minimum mass of the observed-like spectrum", "design.core_theta.m_min"),
            ("ObsMmax", th["m_max"], 0, "Msun", "maximum mass of the observed-like spectrum (sharp cut)", "design.core_theta.m_max"),
            ("ObsDeltaM", th["dm_min"], 1, "Msun", "width of the low-mass taper of the observed-like spectrum (GWTC-3 delta_m)", "design.core_theta.dm_min"),
            ("ObsBeta", th["beta"], 1, "", "power-law index of the mass ratio of the observed-like spectrum", "design.core_theta.beta")):
        add(name, "plain", fnum(float(val), d), float(val), units, meaning, srcO(key), "mock,obslike")
    assert abs(O["design"]["unchanged_from_present_family"]["f_agn"] - IN["f"]) < 1e-12
    # detected samples of realization A, both families
    for fam, blk, desc in (("Mock", "mock_equal", "peak-dominated spectrum, equal populations"), ("MockHD", "mock_host_dependent", "peak-dominated spectrum, host-dependent populations"),
                           ("Obs", "obslike_equal", "observed-like spectrum, equal populations"), ("ObsHD", "obslike_host_dependent", "observed-like spectrum, host-dependent populations")):
        r100, five = O["detected"][blk]["realisation_100"], O["detected"][blk]["five_realisations"]
        ps_key = "peak_share_detected_realised" if r100.get("peak_share_detected_realised") is not None else "peak_share_detected_expected"
        for q, key, d, units, what in (("DetPeakShare", ps_key, 2, "", "share of the detected mergers in the Gaussian peak" + (" (realised component flag)" if ps_key.endswith("realised") else " (mean peak-membership probability; the files store no flag)")),
                                       ("DetzMedian", "z_median", 2, "", "median true redshift of the detected mergers"),
                                       ("DetzMax", "z_max", 2, "", "largest true redshift of a detected merger"),
                                       ("DetMoneMedian", "m1src_median", 1, "Msun", "median source-frame primary mass of the detected mergers"),
                                       ("DetMtwoMedian", "m2src_median", 1, "Msun", "median source-frame secondary mass of the detected mergers")):
            add(f"{fam}{q}", "plain", fnum(float(r100[key]), d), float(r100[key]), units, f"{what}, realization A (1000 detected mergers), {desc}",
                srcO(f"detected.{blk}.realisation_100.{key}"), "mock,obslike", extra={"five_realisations": five.get(key)})
    # ---- A2: the control set with one merger in ten AGN-hosted; AGN counts at m < 18
    pt = DATA / "seed100" / "events" / "events_meta_fagn0p10_corepop.json"
    ft = float(jload(pt)["planted_f_agn"])
    add("MockfAGNTenth", "plain", fnum(ft, 2), ft, "", "input f_AGN of the control set with one merger in ten hosted by an AGN", {"file": rel(pt), "key": "planted_f_agn"}, "mock,validation")
    PC = prod_common()
    seeds = PR.RUNREAL["FluxEighteen"]
    dens = {s: PC.density_from_counts(PC.survey_path(s, "agn", "18"), PC.fit_path("agn", "18")) for s in seeds}
    NA = int(dens[100]["n_below_z_complete"])
    srcN = {"files": [rel(PC.survey_path(100, "agn", "18")), rel(PC.fit_path("agn", "18")), rel(A16 / "scripts" / "prod_common.py")],
            "computation": "prod_common.density_from_counts: AGN with zgals < z_complete(AGN m<18 selection fit)"}
    add("MockNAGNCompleteEighteen", "plain", str(NA), NA, "objects", f"AGN of the m < 18 AGN catalog of realization A below the redshift to which it is complete (z_c = {dens[100]['z_complete']:.4f}, AGN fit)",
        srcN, "mock,C4", extra={"z_complete": dens[100]["z_complete"], "counts_five_realizations": {s: dens[s]["n_below_z_complete"] for s in seeds}})
    prec = 100.0 / math.sqrt(NA)
    add("MocknAGNPrecEighteen", "plain", fsig(prec, 2), prec, "%", "Poisson precision (100/sqrt N) of the AGN number density from those counts, realization A", dict(srcN, computation="100 / sqrt(MockNAGNCompleteEighteen)"), "mock,C4")
    nh = np.array([dens[s]["n_h3Mpc-3"] for s in seeds])
    scat = 100.0 * float(np.std(nh, ddof=1) / np.mean(nh))
    add("MocknAGNScatterEighteen", "plain", fsig(scat, 2), scat, "%", f"scatter (sd, ddof 1) of the AGN number density from the m < 18 counts over the {len(seeds)} production realizations {seeds}, in percent of the mean",
        dict(srcN, computation="sd/mean of density_from_counts over the realizations"), "mock,C4", extra={"densities_h3Mpc-3": {s: dens[s]["n_h3Mpc-3"] for s in seeds}})
    sl = v.get("SlopefAGNDepthNorm")
    for name, pct, what in (("NormDeltafCountsEighteen", prec, "MocknAGNPrecEighteen (Poisson)"), ("NormDeltafCountsScatterEighteen", scat, "MocknAGNScatterEighteen (realised scatter)")):
        meaning = f"shift of the f_AGN median at m < 18 (hosts to z = 1.10) for an AGN number density off by {what}: SlopefAGNDepthNorm x log10(1 + p/100)"
        if sl is not None and sl["class"] != "ph":
            dv = float(sl["value"]) * math.log10(1 + pct / 100.0)
            add(name, "plain", fsig(dv, 1), dv, "", meaning, {"files": [rel(A15_SUMMARY)] + srcN["files"], "computation": "SlopefAGNDepthNorm (Analysis 15) x log10(1 + p/100)"}, "C4,C5")
        else:
            add_ph(name, r"\ph{0.00X}", "", meaning, A15_RUN, "C4,C5")
    # ---- A3: settings
    rl = PR.RUNLIST.read_text()
    ns = int(re.search(r"--nsamp (\d+)", rl).group(1))
    add("ProdNsamp", "plain", str(ns), ns, "samples", "posterior samples per merger in the production analyses (redrawn from each merger's measurement)", {"file": rel(PR.RUNLIST), "key": "driver (--nsamp)"}, "design,appendixA")
    plan = jload(A14R / "plan_b.json")
    nsl = sorted(int(k) for k in plan["seconds_per_call"])
    assert nsl == sorted(NS_WORD), nsl
    for name, n_, w in zip(("MCNsampLo", "MCNsampMid", "MCNsampHi"), nsl, ("smallest", "middle", "largest")):
        add(name, "plain", str(n_), n_, "samples", f"{w} number of posterior samples per merger in the Monte Carlo accuracy study (Analysis 14; suffix {NS_WORD[n_]} of the MC macros)",
            {"file": rel(A14R / "plan_b.json"), "key": "seconds_per_call (the sample counts of the redraws)"}, "appendixA")
    for name, val, units, what in (("SkyResShiftHzero", 0.1, "km/s/Mpc", "change of the H0 median"), ("SkyResShiftfAGN", 0.005, "", "change of the f_AGN median")):
        add(name, "ph", f"\\ph{{{fnum(val, 1 if units else 3)}}}", val, units, f"{what} of KnownBoth, realization A, when the sky pixels are half as wide (nside 64 against 32)",
            {"missing": "run R10 of runs.txt (complete catalogs at twice the pixel resolution); not in the approved stages or the run list",
             "placeholder_source": "invented (expectation: below a tenth of the 90% half-width of realization A)"}, "validation")


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
        "%% RED (\\ph):   placeholder (exploratory or invented, see placeholders.txt and values/placeholder_results.json) for a production run.",
        "%% \\anc:        a number from the exploratory runs (old names, kept for the v1 text); prints red like \\ph (PAPER_SKELETON Sec. 0).",
        "%% Plain:       black: design values and inputs of the simulated catalogs, Analyses 14 and 15, production results.",
        "%% Every macro is traced in values/values.json (source file and key, or the computation).",
        "\\definecolor{phred}{rgb}{0.80,0.05,0.05}",
        "\\definecolor{ancteal}{rgb}{0.00,0.45,0.45}",
        "\\newcommand{\\ph}[1]{\\textcolor{phred}{#1}}",
        "\\newcommand{\\anc}[1]{\\textcolor{phred}{#1}}",  # skeleton Sec. 0: no teal any more, exploratory numbers print red
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


# values/fallback_macros.tex: main.tex inputs it after values/results_macros.tex, so that the paper builds while the text uses a
# macro the chain does not define yet. It holds a red two-dot default for exactly those macros: every capitalized control word of
# main.tex (comments stripped) that results_macros.tex does not define, that main.tex does not define itself and that is not a LaTeX
# command. Written by write_all() from main.tex on every run (until 17 Nov a hand-made list, regenerated by hand).
FALLBACK = OUT / "values" / "fallback_macros.tex"
TEXFILE = OUT / "main.tex"
LATEX_NAMES = {"Delta", "Gamma", "Lambda", "Omega", "Phi", "Pi", "Psi", "Sigma", "Theta", "Upsilon", "Xi", "IfFileExists", "Big", "Bigg", "Bigl",
               "Bigr", "Biggl", "Biggr", "Bigm", "LaTeX", "TeX", "Large", "LARGE", "Huge", "Re", "Im", "Pr", "Rightarrow", "Leftarrow",
               "Leftrightarrow", "Longrightarrow", "Longleftarrow", "Longleftrightarrow", "Uparrow", "Downarrow", "Vert", "S", "P", "AA", "O", "L",
               "Alph", "Roman", "arabic", "MakeUppercase", "MakeLowercase", "Box", "Diamond", "Join", "Bbbk", "AtBeginDocument", "IfStrEq"}


def tex_macros(tex_text):
    """(control words used, control words defined) by a LaTeX source, comments stripped."""
    t = re.sub(r"(?<!\\)%.*", "", tex_text)
    used = set(re.findall(r"\\([A-Za-z]+)", t))
    own = set(re.findall(r"\\(?:newcommand|renewcommand|providecommand|DeclareMathOperator)\*?\s*\{?\\([A-Za-z]+)", t))
    own |= set(re.findall(r"\\(?:def|gdef|let)\s*\\([A-Za-z]+)", t))
    return used, own


def fallback_names(reg_names, tex_text):
    """Macros main.tex uses that the chain does not define (capitalized, not main.tex's own, not LaTeX)."""
    used, own = tex_macros(tex_text)
    return sorted(n for n in used if n[:1].isupper() and n not in set(reg_names) and n not in own and n not in LATEX_NAMES)


def render_fallback(reg_names, tex_text):
    names = fallback_names(reg_names, tex_text)
    L = ["%% fallback_macros.tex -- written by scripts/build_values.py from main.tex on every run; do not edit by hand.",
         "%% Input by main.tex after values/results_macros.tex. It gives a red default of two dots to exactly the macros that main.tex uses and",
         "%% values/results_macros.tex does not define (capitalized control words of main.tex, comments stripped, except main.tex's own",
         "%% definitions and LaTeX commands), so that the paper builds while a requested macro is missing; notes/MACRO_REQUESTS.md says what each is.",
         f"%% {len(names)} macro{'s' if len(names) != 1 else ''} at the last run" + ("." if names else ": every macro main.tex uses is defined by the chain.")]
    L += [f"\\providecommand{{\\{n}}}{{\\ph{{..}}}}" for n in names]
    return "\n".join(L) + "\n", names


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
    R2_CACHE.clear()
    IDENT.clear()
    build_mock()
    posts, H, F = build_c1_c2()
    build_c3()
    build_c4_c5(posts)
    build_c6()
    build_c7_c8()
    build_c9()
    build_placeholders()
    build_r2()
    build_keys()
    build_classes()
    build_density()
    build_requests()
    DIAG["grid_log"] = list(GRIDLOG)
    return REG, DIAG, TABLES, BAYES


def write_all():
    reg, diag, tables, bayes = compute()
    (OUT / "values").mkdir(exist_ok=True)
    (OUT / "values" / "results_macros.tex").write_text(render_tex(reg))
    fb, fb_names = render_fallback([r["name"] for r in reg], TEXFILE.read_text())
    FALLBACK.write_text(fb)
    doc = {"_what": "every macro of values/results_macros.tex with its value(s), class, units, meaning, source and claim; containment records for measurements with a known input",
           "_inputs": IN, "_conventions": __doc__,
           "macros": {r["name"]: jsonable({k: v for k, v in r.items() if k != "name"}) for r in reg},
           "bayes_factor_ledger": jsonable(bayes), "tables": jsonable(tables), "diagnostics": jsonable(diag)}
    (OUT / "values" / "values.json").write_text(json.dumps(doc, indent=1))
    # MACROS.md
    M = ["# Macro menu (written by scripts/build_values.py; do not edit)", "",
         "Class: plain = design choice or mock input; teal = exploratory result (production replaces it); red = not yet computable (see placeholders.txt).",
         "Intervals are medians with 90% equal-tailed intervals unless stated. Full numbers, sources and containment records: values.json.", "",
         KEY_DEFINITIONS, "", CLASS_DEFINITIONS, "", density_definitions(), "", REQ_DEFINITIONS, "", R2_DEFINITIONS, "",
         "| macro | meaning | rendered | class | claim |", "|---|---|---|---|---|"]
    for r in reg:
        rr = r["render"].replace("|", "\\|")
        M.append(f"| `\\{r['name']}` | {r['meaning'].replace('|', '/')} | ${rr}$ | {dict(plain='plain', anc='teal', ph='red')[r['class']]} | {r['claim']} |")
    (OUT / "values" / "MACROS.md").write_text("\n".join(M) + "\n")
    # placeholders.txt
    ph = [r for r in reg if r["class"] == "ph"]
    P = ["PRD paper: ledger of placeholders (written by scripts/build_values.py). Every line prints red in the paper.",
         "Columns: macro | physical meaning ([key ...] for the key scheme of notes/PAPER_SKELETON.md Sec. 6) | production stage or run that fills it"
         " | source of the placeholder value (exploratory: an earlier run, file named; invented: PAPER_SKELETON Sec. 7; Analysis 14 / 15: the summary file)", "",
         f"RED placeholders ({len(ph)}):"]
    for r in ph:
        src = r["source"]
        how = src.get("placeholder_source", "Analysis 14" if "Analysis 14" in src["missing"] else ("Analysis 15" if "Analysis 15" in src["missing"] else "no value yet"))
        pf = src.get("placeholder_file")
        if isinstance(pf, dict):
            pf = "; ".join(f"{k}: {v}" for k, v in pf.items() if v)
        P.append(f"\\{r['name']} | {r['meaning']} | {src['missing']} | {how}{' (' + pf + ')' if pf else ''}")
    anc = [r["name"] for r in reg if r["class"] == "anc"]
    P += ["", f"Also red: {len(anc)} macros of the exploratory runs under their old names (class 'anc' in values/MACROS.md; \\anc prints red),",
          "kept so that the v1 text compiles; the key scheme above carries the same numbers where an exploratory run fills a key."]
    (OUT / "placeholders.txt").write_text("\n".join(P) + "\n")
    n = {c: sum(1 for r in reg if r["class"] == c) for c in ("plain", "anc", "ph")}
    print(f"{n['anc']} teal, {n['plain']} plain, {n['ph']} red; {len(reg)} macros; main.tex uses {len(fb_names)} macro{'s' if len(fb_names) != 1 else ''} "
          f"the chain does not define (values/fallback_macros.tex){': ' + ', '.join(fb_names) if fb_names else ''}")
    return reg


if __name__ == "__main__":
    write_all()
