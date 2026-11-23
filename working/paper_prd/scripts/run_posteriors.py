#!/usr/bin/env python
"""Posteriors of Analysis 16's production runs, one run or several combined: the machinery of the long paper's
production results (scripts/production.py) and of the experiment's numbers (scripts/classes.py).

Until 17 Nov this code lived in the letter's scripts (results.py, now archived with the letter), from which
production.py imported it; the letter is archived and the long paper stands alone. The functions below are
that code, unchanged in what they compute (so that every production number is the same), without the letter's
map and expected values, plus three additions: `combined` (the letter's 'combined' evaluation), `kde1` (the
kernel of make_figures.kde1, copied here so that nothing imports the figure script), and `factor_summary` (the
median and 90% interval of one host class's factor of a host-class-known grid, by the run summaries' own method).

Single-run intervals are never read from a run's grid/result.json: that file holds the driver's node-based
summary; the run's summary.json (prod_summary.refined_grid_summary: Akima interpolation of ln L onto a lattice 16
times denser than the nodes) is the source of every single-run median and interval. grid/result.json is read for
two things only: the names of the free parameters and whether the fine pass was widened (which file to load).

Grids: the corrected ln L (ln L + sum sigma_i^2/2 - N/(2 N_eff), Analysis 14's correction, when the run applied
it), fine pass first, coarse pass outside it. Combinations sum the corrected ln L of the realizations on a common
mesh (cubic splines of each fine pass, linear in the coarse pass). Nested runs: the finished parts' weighted
samples, resampled systematically to equal weight; combinations multiply kernel densities of the marginals.
"""
from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import build_values as bv  # noqa: E402

A16 = bv.AN / "analysis_16_production_runs"
RUNS = A16 / "runs"
if str(A16 / "scripts") not in sys.path:
    sys.path.insert(1, str(A16 / "scripts"))
import prod_common as P  # noqa: E402  (PRIORS, half_variance_corrected, summarize_1d: the run summaries' own statistics)

H0_IN = 67.74
REFINE = 16          # lattice points per stored grid step (prod_summary.REFINE)


@lru_cache(None)
def read_json(p):
    return json.loads(Path(p).read_text())


def input_of(summary, param):
    t = summary.get("truth", {})
    if param in t:
        return float(t[param])
    return H0_IN if param == "H0" else None


def half_width(block):
    return 0.5 * (block["ci90"][1] - block["ci90"][0])


# ---------------------------------------------------------------- grids of one run
def fine_name(run_dir):
    """The final fine pass of a grid run: fine_widened.h5 when the driver widened it, fine.h5 otherwise."""
    res = read_json(Path(run_dir) / "grid" / "result.json")
    return "fine_widened" if (res.get("outside_mass_check") or {}).get("widened") else "fine"


def _grid_parts(run_dir, factor=None):
    """Corrected log-likelihood of a grid run, fine pass first then coarse: [(names, axes, lnc)]."""
    import h5py
    run_dir = Path(run_dir)
    res = read_json(run_dir / "grid" / "result.json")
    cfg = read_json(run_dir / "config.json")
    names = list(res["free"])
    out = []
    for nm in (fine_name(run_dir), "coarse"):
        p = run_dir / "grid" / f"{nm}.h5"
        if not p.exists():
            continue
        with h5py.File(p, "r") as h:
            axes = [np.asarray(h[f"axis_{n}"][()], float) for n in names]
            d = {k: np.asarray(h[k][()], float) for k in h.keys() if not k.startswith("axis_")}
            n = int(h.attrs["n_events"])
        if factor:
            lnL, pv, ne = d[f"lnL_{factor}"], d[f"pe_var_{factor}"], d[f"n_eff_{factor}"]
            n = int((res.get("model_info") or {})["n_events_by_class"][factor])
        else:
            lnL, pv, ne = d["lnL"], d["pe_var"], d["n_eff"]
        lnc = P.half_variance_corrected(lnL, pv, ne, n) if cfg.get("half_variance_correction") else lnL
        out.append((names, axes, np.asarray(lnc, float)))
    if not out:
        raise FileNotFoundError(f"{run_dir}/grid: no fine or coarse pass on disk")
    return out


def _eval_parts(parts, pts):
    """Evaluate the corrected ln L at points (M, d): the fine pass where it covers them, the coarse elsewhere."""
    from scipy.interpolate import RegularGridInterpolator
    val = np.full(len(pts), np.nan)
    for names, axes, lnc in parts:
        ok = np.isfinite(lnc)
        fill = np.where(ok, lnc, np.min(lnc[ok]) - 200.0)
        inside = np.all([(pts[:, i] >= a[0]) & (pts[:, i] <= a[-1]) for i, a in enumerate(axes)], axis=0)
        todo = inside & np.isnan(val)
        if todo.any():
            val[todo] = RegularGridInterpolator(tuple(axes), fill)(pts[todo])
    return np.where(np.isnan(val), -np.inf, val)


def _common_axes(all_parts, n2=(401, 801), n1=4001):
    """A grid spanning the union of the fine passes, widened by 30% on each side (cut at the coarse range)."""
    names = all_parts[0][0][0]
    axes = []
    for i, nm in enumerate(names):
        lo = min(p[0][1][i][0] for p in all_parts)
        hi = max(p[0][1][i][-1] for p in all_parts)
        clo = min(p[-1][1][i][0] for p in all_parts)
        chi = max(p[-1][1][i][-1] for p in all_parts)
        w = hi - lo
        lo, hi = max(clo, lo - 0.3 * w), min(chi, hi + 0.3 * w)
        n = n1 if len(names) == 1 else (n2[0] if i == 0 else n2[1])
        axes.append(np.linspace(lo, hi, n))
    return names, axes


def _spline_parts(parts):
    """Cubic splines of the corrected ln L of a 2-D grid run's fine pass (the coarse pass stays linear)."""
    from scipy.interpolate import RectBivariateSpline
    names, axes, lnc = parts[0]
    if len(axes) != 2 or not np.all(np.isfinite(lnc)):
        return None
    return RectBivariateSpline(axes[0], axes[1], lnc, kx=3, ky=3, s=0), axes


def grid_joint(run_dirs, n=(401, 321), pad=0.0):
    """Joint posterior of the two free parameters of one or more 2-D grid runs, their log-likelihoods
    summed (uniform priors): dict(names, x, y, P (density, normalised on the mesh), level90 (the density
    above which 90% of the mass lies), edge (largest density on the mesh border over the peak), runs).
    The mesh spans the intersection of the runs' fine passes (widened by `pad` of its width, where the
    coarse passes are used); ln L is a cubic spline of each fine pass and linear in the coarse pass."""
    all_parts = [_grid_parts(Path(d)) for d in run_dirs]
    names = all_parts[0][0][0]
    lo = [max(p[0][1][i][0] for p in all_parts) for i in range(2)]
    hi = [min(p[0][1][i][-1] for p in all_parts) for i in range(2)]
    w = [hi[i] - lo[i] for i in range(2)]
    lo = [lo[i] - pad * w[i] for i in range(2)]
    hi = [hi[i] + pad * w[i] for i in range(2)]
    x, y = np.linspace(lo[0], hi[0], n[0]), np.linspace(lo[1], hi[1], n[1])
    X, Y = np.meshgrid(x, y, indexing="ij")
    pts = np.stack([X.ravel(), Y.ravel()], -1)
    tot = np.zeros(X.shape)
    for parts in all_parts:
        val = _eval_parts(parts, pts).reshape(X.shape)          # fine (linear) or coarse
        sp = _spline_parts(parts)
        if sp is not None:
            spl, (ax0, ax1) = sp
            inside = (X >= ax0[0]) & (X <= ax0[-1]) & (Y >= ax1[0]) & (Y <= ax1[-1])
            val = np.where(inside, spl.ev(X, Y), val)
        tot = tot + val
    Pm = np.exp(tot - np.max(tot))
    Pm /= np.trapz(np.trapz(Pm, y, axis=1), x)
    d = np.sort(Pm.ravel())[::-1]
    c = np.cumsum(d) / d.sum()
    lev = float(d[np.searchsorted(c, 0.9)])
    edge = float(max(Pm[0].max(), Pm[-1].max(), Pm[:, 0].max(), Pm[:, -1].max()) / Pm.max())
    return {"names": names, "x": x, "y": y, "P": Pm, "level90": lev, "edge": edge, "runs": [str(r) for r in run_dirs]}


def _eval_parts_spline(parts, pts):
    """As _eval_parts, but the fine pass evaluated with cubic splines of ln L (as grid_joint): linear
    interpolation leaves dips of up to (step^2/8) |d^2 lnL/dx^2| between nodes, which add up over realizations
    (0.3 nats at the 0.025 f_AGN step of the 2-D grids summed over ten realizations). A cubic spline rings where
    ln L jumps by tens of nats between two nodes (audit_values.py K2 fails for any grid of the paper with such a
    jump where the posterior has mass)."""
    from scipy.interpolate import CubicSpline
    val = _eval_parts(parts, pts)
    names, axes, lnc = parts[0]
    if not np.all(np.isfinite(lnc)):
        return val
    inside = np.all([(pts[:, i] >= a[0]) & (pts[:, i] <= a[-1]) for i, a in enumerate(axes)], axis=0)
    if len(axes) == 1:
        val[inside] = CubicSpline(axes[0], lnc)(pts[inside, 0])
    else:
        spl, _ = _spline_parts(parts)
        val[inside] = spl.ev(pts[inside, 0], pts[inside, 1])
    return val


def grid_marginal(run_dirs, param, factor=None):
    """Posterior of `param` from one or more grid runs with their log-likelihoods summed: (x, p). ln L is a
    cubic spline of each run's fine pass and linear in its coarse pass (as grid_joint)."""
    all_parts = [_grid_parts(d, factor) for d in run_dirs]
    names, axes = _common_axes(all_parts)
    mesh = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, len(names))
    tot = sum(_eval_parts_spline(p, mesh) for p in all_parts).reshape([len(a) for a in axes])
    w = np.exp(tot - np.max(tot[np.isfinite(tot)]))
    i = names.index(param)
    x = axes[i]
    other = tuple(j for j in range(len(names)) if j != i)
    p = w
    for j in sorted(other, reverse=True):
        p = np.trapz(p, axes[j], axis=j)
    return x, p / np.trapz(p, x)


def interior_maximum(run_dir, param="H0"):
    """The rule of \\KnownAGNNoPeak (classes.interior), for a grid run of one or two parameters: the node with the largest corrected
    ln L over the fine and the coarse pass (every parameter; for two, the joint maximum) lies at a prior edge of `param` (PRIORS) -> the
    posterior of `param` has no interior maximum in the prior range. -> {"argmax": {name: value}, "lnL", "at_edge", "edges"}."""
    lo_p, hi_p = P.PRIORS[param]
    best = (-np.inf, None)
    for names, axes, lnc in _grid_parts(run_dir):
        ok = np.isfinite(lnc)
        idx = np.unravel_index(int(np.argmax(np.where(ok, lnc, -np.inf))), lnc.shape)
        if lnc[idx] > best[0]:
            best = (float(lnc[idx]), {n: float(a[i]) for n, a, i in zip(names, axes, idx)})
    v = best[1][param]
    return {"argmax": best[1], "lnL": best[0], "at_edge": bool(abs(v - lo_p) < 1e-9 or abs(v - hi_p) < 1e-9), "edges": [float(lo_p), float(hi_p)]}


def factor_summary(run_dir, factor):
    """Median and 90% interval of H0 from one host class's factor (lnL_<factor>) of a host-class-known grid run, by
    the method of the run summaries (prod_summary.refined_grid_summary): the corrected factor (its own pe_var,
    n_eff and number of mergers) on the final fine pass, Akima-interpolated onto a lattice REFINE times denser
    than the nodes, then prod_common.summarize_1d. -> (summary dict, info)."""
    from scipy.interpolate import Akima1DInterpolator
    names, axes, lnc = _grid_parts(run_dir, factor)[0]
    assert len(names) == 1, (run_dir, names)
    a = axes[0]
    ok = np.isfinite(lnc)
    lf = np.where(ok, lnc, np.min(lnc[ok]) - 300.0)
    lat = np.linspace(a[0], a[-1], (len(a) - 1) * REFINE + 1)
    lv = Akima1DInterpolator(a, lf)(lat)
    s = P.summarize_1d(lat, lv)
    return s, {"file": f"{fine_name(run_dir)}.h5", "factor": factor, "refine": REFINE, "interpolation": "akima, ln L",
               "n_nonfinite_cells": int((~ok).sum()), "mass_at_edges": s["mass_at_edges"]}


# ---------------------------------------------------------------- nested runs
def kde1(x, lo, hi, n=2001, bw=None):
    """Gaussian kernel density of chain samples on [lo, hi], reflected at both prior edges, normalised
    on [lo, hi]; bw defaults to 0.3 of the sample standard deviation (make_figures.kde1, the same code)."""
    x = np.asarray(x, float)
    h = bw if bw is not None else 0.3 * float(np.std(x))
    g = np.linspace(lo, hi, n)
    p = np.zeros(n)
    for c in np.array_split(x, max(1, len(x) // 400)):
        for r in (c, 2 * lo - c, 2 * hi - c):
            p += np.exp(-0.5 * ((g[:, None] - r[None, :]) / h) ** 2).sum(axis=1)
    p /= np.trapz(p, g)
    return g, p, h


def chain_samples(run_dir, params, n_equal=20000, seed=0):
    """Equal-weight samples of `params` from a nested run: each finished part's weights normalised, the
    parts averaged (as prod_summary's merged posterior), resampled systematically (deterministic)."""
    S, W = [], []
    for part in sorted(Path(run_dir).glob("nested_[0-9]")):
        rj = part / "result.json"
        if not (rj.exists() and read_json(rj).get("finished")):
            continue
        z = np.load(part / "samples.npz", allow_pickle=False)
        names = [str(x) for x in z["names"]]
        S.append(np.asarray(z["samples"], float)[:, [names.index(p) for p in params]])
        w = np.asarray(z["weights"], float)
        W.append(w / w.sum())
    if not S:
        raise FileNotFoundError(f"{run_dir}: no finished nested part")
    S = np.concatenate(S)
    W = np.concatenate(W) / len(W)
    u = (np.arange(n_equal) + np.random.default_rng(seed).random()) / n_equal
    idx = np.minimum(np.searchsorted(np.cumsum(W), u), len(W) - 1)
    return S[idx]


def chain_marginal_product(run_dirs, param):
    """Combined posterior of `param` over realizations from nested runs: the product of each run's kernel
    density (reflected at the prior edges), uniform prior."""
    lo, hi = P.PRIORS[param]
    logp = None
    for d in run_dirs:
        x, p, _ = kde1(chain_samples(d, [param])[:, 0], lo, hi, n=4001)
        lp = np.log(np.maximum(p, 1e-300))
        logp = lp if logp is None else logp + lp
    p = np.exp(logp - logp.max())
    return x, p / np.trapz(p, x)


# ---------------------------------------------------------------- one run or several
def is_grid(run_dir):
    return read_json(Path(run_dir) / "config.json")["mode"] == "grid"


def marginal(run_dirs, param, factor=None):
    if is_grid(run_dirs[0]):
        return grid_marginal(run_dirs, param, factor)
    return chain_marginal_product(run_dirs, param)


def combined(run_dirs, param, x0="truth"):
    """The posterior of `param` combined over the runs (grids: ln L summed; nested: kernel densities multiplied),
    summarised by placeholder_results.summarize_xp with the credible level of the input (the summaries' truth when
    x0 == "truth"). -> (summary, x, p, input)."""
    import placeholder_results as phr
    run_dirs = [Path(d) for d in run_dirs]
    x, p = marginal(run_dirs, param)
    if x0 == "truth":
        x0 = input_of(read_json(run_dirs[0] / "summary.json"), param)
    return phr.summarize_xp(x, p, x0), np.asarray(x), np.asarray(p), x0
