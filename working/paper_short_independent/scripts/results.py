#!/usr/bin/env python
"""Every number and plotted quantity of the short paper, real or placeholder, from one place.

    PY=/hildafs/home/magana/tmp_ondemand_hildafs_phy230014p_symlink/magana/.conda/envs/jax/bin/python
    $PY scripts/results.py               # print every macro: real or placeholder, value, what it waits for
    $PY scripts/results.py --prefill     # fill empty "summary" fields of results_map.json from the run lists
    $PY scripts/results.py --prefill --force   # ... and overwrite filled ones

The map is scripts/results_map.json (its "_schema" block documents every field). Each macro and each
plotted point or curve names a run summary (analysis_16_production_runs/runs/<run>/summary.json, written by
that analysis' scripts/prod_summary.py), a field, and how realizations are combined:

    mean_hw     precision: mean over realizations of half the width of the 90% interval (the bars of
                Fig. 3 are the range, minimum to maximum, over the same realizations)
    single      one realization (realization A): median with its 90% interval, or one number
    combined    bias: the posterior combined over realizations (grids: log-likelihoods summed on a common
                grid; nested runs: the product of the kernel densities of the marginal), then its median,
                90% interval, 90% upper limit or the credible level of the input
    derived     arithmetic on other macros (real only when all of them are)
    log10_bayes (ln Z of one run - ln Z of another) / ln 10, realization A
    membership  (Fig. 4c) each merger's probability of an AGN host and its true host, realization A
    joint       (Fig. 4b) the 2-D posterior of two parameters, realization A
    joint_grid  (Fig. 2) the 2-D posterior of (H0, f_AGN) of grid runs, ln L summed over the realizations

A quantity is REAL when every summary file it names exists (for all of its realizations); otherwise it is
the PLACEHOLDER: the expected value of values_independent.tex (the red \\ph{...}), or, for a plotted
point that has no macro, the invented value stored in the map with the reason. Plotted shapes of
placeholders (Fig. 4c distributions, Fig. 4a densities) are drawn from the expected values: Fig. 4c a
calibrated logit-normal distribution per true class whose means equal the macros; Fig. 4a Gaussians (the
placeholder logic of ../paper_prd/scripts/placeholder_results.py) with the expected medians and widths.
Every interval is 90%.

PARTIAL (figures only): a plotted mean or combination over realizations of which some, but not all, have
their summary is drawn from the realizations that exist, still in the red tones of an element that waits
for a run (fig1_points, fig3_density, anchor_joint report n_have/n_want). Macros stay all-or-nothing.

Figure numbers of the letter (main.tex): Fig. 2 figures/fig_anchor (anchor_joint), Fig. 3
figures/fig1_budget (series 'fig1a', 'fig1b'), Fig. 4 figures/fig_populations (series 'fig3' for panels
a and b, 'fig2' for panel c; the series keep the names of the first draft's figure files).
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
WORK = PAPER.parent
MAP_PATH = HERE / "results_map.json"
VALUES_TEX = PAPER / "values_independent.tex"
SHARED_TEX = PAPER / "values" / "results_macros.tex"
PRD_SCRIPTS = WORK / "paper_prd" / "scripts"
A16 = WORK / "analyses" / "analysis_16_production_runs"

sys.path.insert(0, str(PRD_SCRIPTS))
import placeholder_results as phr  # noqa: E402  (gauss_xp, summarize_xp, Z90: the placeholder logic)

sys.path.insert(1, str(A16 / "scripts"))
import prod_common as P  # noqa: E402  (PRIORS, summarize_1d: the run summaries' own statistics)

Z90 = phr.Z90
H0_IN = 67.74


# ============================================================================================ #
# the map and the expected values
# ============================================================================================ #
@lru_cache(None)
def load_map():
    return json.loads(MAP_PATH.read_text())


def runs_root():
    m = load_map()
    p = Path(m["runs_root"])
    return p if p.is_absolute() else (PAPER / p).resolve()


def _braced(s, i):
    """Content of the brace group opening at s[i] == '{'; -> (content, index after it)."""
    assert s[i] == "{", s[i:i + 20]
    d = 0
    for j in range(i, len(s)):
        d += {"{": 1, "}": -1}.get(s[j], 0)
        if d == 0:
            return s[i + 1:j], j + 1
    raise ValueError("unbalanced braces")


def parse_macros(path):
    """\\providecommand / \\newcommand definitions of a .tex file -> {name: body}."""
    txt = "\n".join(re.sub(r"(?<!\\)%.*$", "", line) for line in Path(path).read_text().splitlines())
    out = {}
    for m in re.finditer(r"\\(?:provide|new)command\{\\(\w+)\}(?:\[\d\])?", txt):
        body, _ = _braced(txt, m.end())
        out[m.group(1)] = body
    return out


@lru_cache(None)
def expected():
    """values_independent.tex: {name: (body, is_red, inner)} with inner the content of \\ph{...}."""
    out = {}
    for k, body in parse_macros(VALUES_TEX).items():
        b = body.strip()
        if b.startswith("\\ph{"):
            inner, _ = _braced(b, 3)
            out[k] = (b, True, inner)
        else:
            out[k] = (b, False, b)
    return out


@lru_cache(None)
def shared():
    return parse_macros(SHARED_TEX)


def parse_value(s):
    """'3.0' -> 3.0; '0.28^{+0.08}_{-0.08}' -> (0.28, 0.08, 0.08) as (median, plus, minus)."""
    s = s.strip()
    m = re.fullmatch(r"([-+]?[\d.]+)\^\{\+([\d.]+)\}_\{-([\d.]+)\}", s)
    if m:
        return tuple(float(x) for x in m.groups())
    return float(s)


def expected_value(name):
    return parse_value(expected()[name][2])


# ============================================================================================ #
# results
# ============================================================================================ #
@dataclass
class Result:
    name: str
    real: bool
    value: object                      # float, (median, plus, minus), dict, arrays (by kind)
    tex: str | None = None             # formatted TeX of a real macro
    files: list = field(default_factory=list)      # summary files used (real) or wanted (placeholder)
    missing: list = field(default_factory=list)    # summary files that do not exist yet
    how: str = ""                      # how the value was obtained
    extra: dict = field(default_factory=dict)      # per-realization values, ranges, shapes


def realizations(entry):
    r = entry.get("realizations", "exp")
    if isinstance(r, list):
        return [int(x) for x in r]
    return list(load_map()["realizations"][r])


def summary_paths(entry, which=None):
    """The summary files an entry names: {realization: Path} (a list of such dicts for two patterns)."""
    pat = entry.get("summary") or ""
    reals = which if which is not None else realizations(entry)
    if isinstance(pat, list):
        return [{r: (runs_root() / p.format(R=f"{r:03d}")) if p else None for r in reals} for p in pat]
    return {r: (runs_root() / pat.format(R=f"{r:03d}")) if pat else None for r in reals}


def _status(paths):
    flat = []
    for d in (paths if isinstance(paths, list) else [paths]):
        flat += list(d.values())
    if any(p is None for p in flat):
        return False, ["(no summary file named in results_map.json)"]
    missing = [str(p) for p in flat if not p.exists()]
    return not missing, missing


def dig(obj, path):
    for k in path.split("."):
        obj = obj[k]
    return obj


@lru_cache(None)
def read_json(p):
    return json.loads(Path(p).read_text())


def input_of(summary, param):
    t = summary.get("truth", {})
    if param in t:
        return float(t[param])
    return H0_IN if param == "H0" else None


# ---------------------------------------------------------------- grids and chains of one run
def _grid_parts(run_dir, factor=None):
    """Corrected log-likelihood of a grid run, fine pass first then coarse: [(names, axes, lnc)]."""
    import h5py
    res = read_json(run_dir / "grid" / "result.json")
    cfg = read_json(run_dir / "config.json")
    names = list(res["free"])
    widened = (res.get("outside_mass_check") or {}).get("widened")
    out = []
    for nm in ("fine_widened" if widened else "fine", "coarse"):
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


def _common_axes(all_parts, n2=(301, 151), n1=2001):
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
    P = np.exp(tot - np.max(tot))
    P /= np.trapz(np.trapz(P, y, axis=1), x)
    d = np.sort(P.ravel())[::-1]
    c = np.cumsum(d) / d.sum()
    lev = float(d[np.searchsorted(c, 0.9)])
    edge = float(max(P[0].max(), P[-1].max(), P[:, 0].max(), P[:, -1].max()) / P.max())
    return {"names": names, "x": x, "y": y, "P": P, "level90": lev, "edge": edge, "runs": [str(r) for r in run_dirs]}


def grid_marginal(run_dirs, param, factor=None):
    """Posterior of `param` from one or more grid runs with their log-likelihoods summed: (x, p)."""
    all_parts = [_grid_parts(d, factor) for d in run_dirs]
    names, axes = _common_axes(all_parts)
    mesh = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, len(names))
    tot = sum(_eval_parts(p, mesh) for p in all_parts).reshape([len(a) for a in axes])
    w = np.exp(tot - np.max(tot[np.isfinite(tot)]))
    i = names.index(param)
    x = axes[i]
    other = tuple(j for j in range(len(names)) if j != i)
    p = w
    for j in sorted(other, reverse=True):
        p = np.trapz(p, axes[j], axis=j)
    return x, p / np.trapz(p, x)


def chain_samples(run_dir, params, n_equal=20000, seed=0):
    """Equal-weight samples of `params` from a nested run: each finished part's weights normalised, the
    parts averaged (as prod_summary's merged posterior), resampled systematically (deterministic)."""
    S, W = [], []
    for part in sorted(run_dir.glob("nested_[0-9]")):
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
    density (reflected at the prior edges; ../paper_prd/scripts/make_figures.kde1), uniform prior."""
    import make_figures as mf
    lo, hi = P.PRIORS[param]
    logp = None
    for d in run_dirs:
        x, p, _ = mf.kde1(chain_samples(d, [param])[:, 0], lo, hi, n=4001)
        lp = np.log(np.maximum(p, 1e-300))
        logp = lp if logp is None else logp + lp
    p = np.exp(logp - logp.max())
    return x, p / np.trapz(p, x)


def is_grid(run_dir):
    return read_json(run_dir / "config.json")["mode"] == "grid"


def marginal(run_dirs, param, factor=None):
    if is_grid(run_dirs[0]):
        return grid_marginal(run_dirs, param, factor)
    return chain_marginal_product(run_dirs, param)


# ---------------------------------------------------------------- formatting
def fmt_num(v, fmt):
    if fmt == "sig2":       # two significant figures: 0.83, 1.7, 10
        return f"{v:.0f}" if abs(v) >= 9.95 else f"{v:.1f}" if abs(v) >= 0.995 else f"{v:.2f}"
    if fmt == "floor1":     # 'beyond the X% level': rounded down, one decimal
        return f"{math.floor(v * 10 + 1e-9) / 10:.1f}"
    return fmt.format(v)


def fmt_pm(med, lo, hi, fmt):
    return f"{fmt_num(med, fmt)}^{{+{fmt_num(hi - med, fmt)}}}_{{-{fmt_num(med - lo, fmt)}}}"


def half_width(block):
    return 0.5 * (block["ci90"][1] - block["ci90"][0])


# ---------------------------------------------------------------- one entry
def evaluate(name, entry=None):
    """The Result of one macro (or of a series point entry)."""
    m = load_map()
    if entry is None:
        entry = m["macros"][name]
    if "macro" in entry:                       # a series point that is a macro
        r = evaluate(entry["macro"])
        return Result(name, r.real, r.value, r.tex, r.files, r.missing, r.how, r.extra)
    kind = entry["combine"]
    fmt = entry.get("fmt", "{:.2f}")
    if kind == "derived":
        deps = re.findall(r"\b(Ind\w+)\b", entry["expr"])
        rs = {d: evaluate(d) for d in deps}
        miss = sorted({x for r in rs.values() for x in r.missing})
        if all(r.real for r in rs.values()):
            v = float(eval(entry["expr"], {"__builtins__": {}}, {d: rs[d].value for d in deps}))
            return Result(name, True, v, fmt_num(v, fmt), how=f"{entry['expr']} of the real values")
        return Result(name, False, None, missing=miss, how=f"{entry['expr']}: waits for {', '.join(d for d in deps if not rs[d].real)}")
    paths = summary_paths(entry, realizations(entry)[:1] if kind in ("single", "log10_bayes", "membership", "joint") else None)
    ok, missing = _status(paths)
    files = [str(p) for d in (paths if isinstance(paths, list) else [paths]) for p in d.values() if p is not None]
    if not ok:
        return Result(name, False, None, files=files, missing=missing, how=kind)
    fld = entry.get("field", "posterior.H0")
    if kind == "mean_hw":
        per = {}
        for r, p in paths.items():
            if fld.startswith("grid."):        # one factor of a hosts_known grid (e.g. grid.agn)
                x, pp = grid_marginal([p.parent], "H0", factor=fld.split(".")[1])
                s = phr.summarize_xp(x, pp)
                per[r] = half_width(s)
            else:
                per[r] = half_width(dig(read_json(p), fld))
        v = float(np.mean(list(per.values())))
        return Result(name, True, v, fmt_num(v, fmt), files, how=f"mean over {len(per)} realizations of the 90% half-width of {fld}",
                      extra={"per_realization": per, "min": min(per.values()), "max": max(per.values())})
    if kind == "single":
        (r, p), = paths.items()
        s = read_json(p)
        b = dig(s, fld)
        if entry.get("stat", "pm") == "pm":
            v = (b["median"], b["ci90"][1] - b["median"], b["median"] - b["ci90"][0])
            return Result(name, True, v, fmt_pm(b["median"], *b["ci90"], fmt), files, how=f"realization {r}: {fld}, median and 90% interval")
        v = float(b)
        return Result(name, True, v, fmt_num(v, fmt), files, how=f"realization {r}: {fld}")
    if kind == "combined":
        param = fld.split(".")[-1]
        run_dirs = [p.parent for p in paths.values()]
        x, pp = marginal(run_dirs, param)
        x0 = input_of(read_json(next(iter(paths.values()))), param)
        s = phr.summarize_xp(x, pp, x0)
        stat = entry.get("stat", "pm")
        how = f"{fld} combined over {len(run_dirs)} realizations"
        ex = {"x": x, "p": pp, "summary": s, "input": x0}
        if stat == "pm":
            v = (s["median"], s["ci90"][1] - s["median"], s["median"] - s["ci90"][0])
            return Result(name, True, v, fmt_pm(s["median"], *s["ci90"], fmt), files, how=how, extra=ex)
        if stat == "median_minus_input":
            v = s["median"] - x0
        elif stat == "level_of_input":
            v = 100.0 * s["level_of_input"]
        elif stat == "ul90":
            v = s["ul90"]
        elif stat == "density":
            return Result(name, True, (x, pp), None, files, how=how, extra=ex)
        else:
            raise ValueError(stat)
        return Result(name, True, float(v), fmt_num(v, fmt), files, how=f"{how}: {stat}", extra=ex)
    if kind == "log10_bayes":
        (r, pa), = paths[0].items()
        (_, pb), = paths[1].items()
        za, zb = read_json(pa)["ln_evidence"], read_json(pb)["ln_evidence"]
        v = (za["value"] - zb["value"]) / math.log(10)
        err = math.hypot(za.get("error") or 0.0, zb.get("error") or 0.0) / math.log(10)
        return Result(name, True, v, fmt_num(v, fmt), files, how=f"realization {r}: (ln Z - ln Z_ref)/ln 10, error {err:.2g}")
    if kind == "membership":
        (r, p), = paths.items()
        npz = p.parent / "grid" / "membership.npz"
        z = np.load(npz)
        rr, ht = np.asarray(z["r"], float), np.asarray(z["host_type"], int)
        return Result(name, True, {"r": rr, "host_type": ht}, None, files + [str(npz)], how=f"realization {r}: membership.npz")
    if kind == "joint":
        (r, p), = paths.items()
        pars = entry["params"]
        s = chain_samples(p.parent, pars)
        return Result(name, True, {"samples": s, "params": pars, "truth": read_json(p).get("truth", {})}, None, files,
                      how=f"realization {r}: equal-weight samples of {pars}")
    raise ValueError(f"{name}: unknown combine {kind!r}")


@lru_cache(None)
def macro(name):
    """A macro: Result with .real; a placeholder carries the expected value of values_independent.tex."""
    r = evaluate(name)
    if not r.real:
        r.value = expected_value(name)
        r.how = f"placeholder: expected value of values_independent.tex; {r.how}"
    return r


def macro_number(name):
    """The number a figure plots for a macro (the median of a 'pm' value)."""
    v = macro(name).value
    return v[0] if isinstance(v, tuple) else float(v)


def have_realizations(entry):
    """(realizations whose summary exists, all realizations) of an entry or of the macro a point names."""
    if "macro" in entry:
        return have_realizations(load_map()["macros"][entry["macro"]])
    pat = entry.get("summary")
    want = realizations(entry)
    if not pat or isinstance(pat, list):
        return [], want
    return [r for r, p in summary_paths(entry).items() if p.exists()], want


def evaluate_partial(name, entry):
    """A mean or combination over the realizations of `entry` (or of the macro it names) that have their
    summary, when some but not all do: Result with real False and extra n_have, n_want, have. None when
    no realization has its summary or the entry is not a mean or combination."""
    if "macro" in entry:
        return evaluate_partial(entry["macro"], load_map()["macros"][entry["macro"]])
    if entry.get("combine") not in ("mean_hw", "combined"):
        return None
    have, want = have_realizations(entry)
    if not have:
        return None
    r = evaluate(name, dict(entry, realizations=have))
    r.real = len(have) == len(want)
    r.extra.update(n_have=len(have), n_want=len(want), have=have)
    return r


# ============================================================================================ #
# what the figures draw
# ============================================================================================ #
def series(fig):
    return load_map()["series"][fig]


def fig1_points(panel):
    """{series name: dict(label, points=[dict(x, y, lo, hi, real, how)])}, horizontal lines included
    (points with x None). lo/hi: the range over realizations (real) or the placeholder spread."""
    m = load_map()
    spread = float(m["placeholder_spread"])
    out = {}
    for key, s in series(f"fig1{panel}").items():
        if key.startswith("_"):
            continue
        pts = []
        for xs, e in s["points"].items():
            nm = f"fig1{panel}.{key}[{xs}]"
            r = evaluate(nm, e)
            n_have, n_want = len(realizations(e if "macro" not in e else m["macros"][e["macro"]])), None
            if r.real:
                y = r.value
                lo, hi = r.extra.get("min", y), r.extra.get("max", y)
                how = r.how
                n_want = n_have
            else:
                rp = evaluate_partial(nm, e)
                if rp is not None:          # some realizations exist: their mean, still marked as waiting
                    y = rp.value
                    n_have, n_want = rp.extra["n_have"], rp.extra["n_want"]
                    lo, hi = (rp.extra["min"], rp.extra["max"]) if n_have > 1 else (y * (1 - spread), y * (1 + spread))
                    how = f"partial: mean over {n_have} of {n_want} realizations ({','.join(map(str, rp.extra['have']))})"
                else:
                    y = macro_number(e["macro"]) if "macro" in e else float(e["placeholder"]["value"])
                    lo, hi = y * (1 - spread), y * (1 + spread)
                    n_have, n_want = 0, len(have_realizations(e)[1])
                    how = f"placeholder: {e['macro']}" if "macro" in e else f"placeholder: {e['placeholder']['how']}"
            pts.append({"x": None if xs == "line" else float(xs), "y": float(y), "lo": float(lo), "hi": float(hi),
                        "real": r.real, "macro": e.get("macro"), "how": how, "n_have": n_have, "n_want": n_want})
        out[key] = {"label": s["label"], "points": pts}
    return out


def _logit(p):
    return math.log(p / (1 - p))


def _calibrated_r(f, n_agn, n_gal, target_agn):
    """Placeholder membership probabilities: log-likelihood ratios Gaussian with means +-s^2/2 and variance s^2
    (a calibrated classifier), at deterministic normal quantiles; s solved so that the mean probability of the
    mergers truly in AGN equals the expected value. -> r_agn, r_gal."""
    from scipy.optimize import brentq
    from scipy.special import expit, ndtri
    qa = ndtri((np.arange(n_agn) + 0.5) / n_agn)
    qg = ndtri((np.arange(n_gal) + 0.5) / n_gal)
    lf = _logit(f)

    def r_of(sig):
        return expit(lf + sig * sig / 2 + sig * qa), expit(lf - sig * sig / 2 + sig * qg)

    sig = brentq(lambda s: r_of(s)[0].mean() - target_agn, 1e-3, 6.0)
    return (*r_of(sig), sig)


def fig2_membership(which):
    """which: 'positions' or 'masses_spins' -> dict(r_agn, r_gal, mean_agn, mean_gal, real, how)."""
    s = series("fig2")[which]
    r = evaluate(f"fig2.{which}", s)
    if r.real:
        rr, ht = r.value["r"], r.value["host_type"]
        ra, rg = rr[ht == 1], rr[ht == 0]
        return {"r_agn": ra, "r_gal": rg, "mean_agn": float(ra.mean()), "mean_gal": float(rg.mean()), "real": True,
                "how": r.how, "f_in": float(s.get("f_in", 0.30))}
    f = float(s.get("f_in", 0.30))
    n = int(s.get("n_events", 1000))
    na = int(round(f * n))
    ta, tg = macro_number(s["placeholder_macros"][0]), macro_number(s["placeholder_macros"][1])
    ra, rg, sig = _calibrated_r(f, na, n - na, ta)
    return {"r_agn": ra, "r_gal": rg, "mean_agn": float(ra.mean()), "mean_gal": float(rg.mean()), "real": False,
            "how": f"placeholder: calibrated logit-normal (sd of the log-likelihood ratio {sig:.2f}) matched to "
                   f"{s['placeholder_macros'][0]} = {ta}; its galaxy mean {rg.mean():.3f} against {s['placeholder_macros'][1]} = {tg}",
            "f_in": f}


def fig3_density(which):
    """which: 'one_population' or 'each_tracer' -> dict(x, p, summary, real, how, n_real)."""
    s = series("fig3")[which]
    r = evaluate(f"fig3.{which}", s)
    n_real = len(realizations(s))
    if r.real:
        return {"x": r.extra["x"], "p": r.extra["p"], "summary": r.extra["summary"], "real": True, "how": r.how,
                "n_real": n_real, "n_have": n_real}
    rp = evaluate_partial(f"fig3.{which}", s)
    if rp is not None:      # the realizations that exist, combined; still marked as waiting
        return {"x": rp.extra["x"], "p": rp.extra["p"], "summary": rp.extra["summary"], "real": False, "n_real": n_real,
                "n_have": rp.extra["n_have"], "how": f"partial: combined over {rp.extra['n_have']} of {n_real} realizations"}
    ph = s["placeholder"]
    hw = macro_number(ph["hw_macro"]) / math.sqrt(n_real)
    if "bias_macro" in ph:
        med = H0_IN + macro_number(ph["bias_macro"])
    else:   # the input at the expected credible level, on the side of the one-catalog median
        lev = macro_number(ph["level_macro"]) / 100.0
        from scipy.special import ndtri
        side = np.sign(macro_number(ph["side_macro"]) - H0_IN) or 1.0
        med = H0_IN + side * float(ndtri(0.5 + lev / 2)) * hw / Z90
    sm = phr.gauss_xp(med, hw, P.PRIORS["H0"], H0_IN)
    return {"x": np.asarray(sm["x"]), "p": np.asarray(sm["p"]), "summary": sm, "real": False, "n_real": n_real, "n_have": 0,
            "how": f"placeholder: Gaussian, median {med:.2f}, 90% half-width {hw:.2f} ({ph['hw_macro']}/sqrt({n_real}))"}


def anchor_joint(which):
    """Fig. 2: which 'one_realization' or 'combined' -> dict(x, y, P, level90, real, n_have, n_want, how,
    edge). The 2-D posterior of (H0, f_AGN) of the grid runs named by series 'fig_anchor' (the runs that
    exist, their ln L summed); a Gaussian from the expected values of the macros when none exists."""
    s = series("fig_anchor")[which]
    have, want = have_realizations(s)
    if have:
        J = grid_joint([summary_paths(s, have)[r].parent for r in have])
        J.update(real=len(have) == len(want), n_have=len(have), n_want=len(want),
                 how=f"{len(have)} of {len(want)} realizations ({','.join(map(str, have))}): ln L summed, 90% region"
                     f"; largest density on the mesh border {J['edge']:.1e} of the peak")
        return J
    mh, mf = s["placeholder_macros"]
    qh, qf = macro(mh).value, macro(mf).value
    sh, sf = 0.5 * (qh[1] + qh[2]) / Z90, 0.5 * (qf[1] + qf[2]) / Z90
    x = np.linspace(qh[0] - 5 * sh, qh[0] + 5 * sh, 401)
    y = np.linspace(max(0.0, qf[0] - 5 * sf), min(1.0, qf[0] + 5 * sf), 321)
    X, Y = np.meshgrid(x, y, indexing="ij")
    Pm = np.exp(-0.5 * (((X - qh[0]) / sh) ** 2 + ((Y - qf[0]) / sf) ** 2))
    Pm /= np.trapz(np.trapz(Pm, y, axis=1), x)
    d = np.sort(Pm.ravel())[::-1]
    lev = float(d[np.searchsorted(np.cumsum(d) / d.sum(), 0.9)])
    return {"names": ["H0", "f_agn"], "x": x, "y": y, "P": Pm, "level90": lev, "edge": 0.0, "real": False, "n_have": 0,
            "n_want": len(want), "how": f"placeholder: uncorrelated Gaussian from {mh}, {mf}"}


def fig3_joint():
    """-> dict(x, y, P, level90, params, real, how): the 2-D density of the two population differences."""
    import make_figures as mf
    s = series("fig3")["joint"]
    r = evaluate("fig3.joint", s)
    pars = s["params"]
    if r.real:
        smp = r.value["samples"]
        x, y, H, lev = mf.hist2d_density(smp[:, 0], smp[:, 1], P.PRIORS[pars[0]], P.PRIORS[pars[1]], nb=80, smooth=1.5)
        return {"x": x, "y": y, "P": H, "level90": lev, "params": pars, "real": True, "how": r.how}
    ph = s["placeholder"]
    qx = {"median": macro_number(ph["macros"][0]), "ci90": [0, 2 * macro(ph["macros"][0]).value[1]]}
    qy = {"median": macro_number(ph["macros"][1]), "ci90": [0, 2 * macro(ph["macros"][1]).value[1]]}
    rho = float(mf.PHJ[ph["correlation_from_prd_key"]]["joint"]["A"]["correlation"])
    J = phr.joint_gauss(qx, qy, "DmuG", "DmuChi", rho)
    x, y, Pm, lev = mf.hpd90(J)
    return {"x": x, "y": y, "P": Pm, "level90": lev, "params": pars, "real": False,
            "how": f"placeholder: Gaussian from {ph['macros']} with correlation {rho:.2f} (paper_prd placeholder "
                   f"key {ph['correlation_from_prd_key']}, the exploratory Analysis 13 chain)"}


# ============================================================================================ #
# pre-filling the map from the run lists
# ============================================================================================ #
def _runlist_index():
    """(analysis, set label) -> run-name pattern, and production prefix -> pattern, from the run lists."""
    import yaml
    idx, prod = {}, {}
    rl = runs_root() / "RUNLIST_experiment.yaml"
    if rl.exists():
        d = yaml.safe_load(rl.read_text())
        for st in d["stages"]:
            for r in st["runs"]:
                idx[(r["analysis"], r["set"]["label"])] = re.sub(r"realization_\d{3}", "realization_{R}", r["name"])
        for r in d.get("reused_from_stage_1") or []:
            idx[(r["analysis"], r["set"])] = re.sub(r"realization_\d{3}", "realization_{R}", r["run"])
    rp = runs_root() / "RUNLIST.yaml"
    if rp.exists():
        d = yaml.safe_load(rp.read_text())
        for st in d["stages"]:
            for r in st["runs"]:
                pat = re.sub(r"realization_\d{3}", "realization_{R}", r["name"])
                prod[pat.rsplit("/realization_", 1)[0]] = pat
    return idx, prod


def _resolve(sel, idx, prod):
    if sel is None:
        return None
    if "production" in sel:
        p = prod.get(sel["production"])
    else:
        p = idx.get((sel["analysis"], sel["set"]))
    return None if p is None else p + "/summary.json"


def prefill(force=False):
    raw = json.loads(MAP_PATH.read_text())
    idx, prod = _runlist_index()
    n_set = n_miss = 0

    def fill(e, where):
        nonlocal n_set, n_miss
        if "select" not in e:
            return
        sels = e["select"] if isinstance(e["select"], list) else [e["select"]]
        pats = [_resolve(s, idx, prod) for s in sels]
        if any(p is None for p in pats):
            n_miss += 1
            print(f"  not in the run lists: {where} {e['select']}")
            return
        new = pats if isinstance(e["select"], list) else pats[0]
        cur = e.get("summary")
        empty = not cur or (isinstance(cur, list) and not any(cur))
        if force or empty:
            e["summary"] = new
            n_set += 1

    for k, e in raw["macros"].items():
        fill(e, k)
    for fig, ss in raw["series"].items():
        for key, s in ss.items():
            if key.startswith("_"):
                continue
            if "points" in s:
                for xs, e in s["points"].items():
                    fill(e, f"{fig}.{key}[{xs}]")
            else:
                fill(s, f"{fig}.{key}")
    MAP_PATH.write_text(json.dumps(raw, indent=1) + "\n")
    load_map.cache_clear()
    print(f"prefill: {n_set} summary fields written, {n_miss} selections not found in the run lists")


# ============================================================================================ #
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefill", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    if a.prefill:
        prefill(a.force)
    names = list(load_map()["macros"])
    nreal = 0
    for n in names:
        r = macro(n)
        nreal += r.real
        val = r.tex if r.real else expected()[n][2]
        print(f"{n:20s} {'REAL       ' if r.real else 'placeholder'} {val:24s} {r.how[:90]}")
    print(f"{nreal} of {len(names)} macros real")


if __name__ == "__main__":
    main()
