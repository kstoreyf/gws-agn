#!/usr/bin/env python
"""Diagnostics of one dynesty run (static NestedSampler, as the driver runs it), from its stored state.

    from nested_diagnostics import load_run, insertion_test, geometry
    R = load_run("runs/<run>/nested_1")          # a part directory, a state.save or a results.pkl
    ins = insertion_test(R)                       # the corrected insertion-index test
    geo = geometry(R)                             # bound volume against the contour volume; flatness

Insertion-index test (Fowlie, Handley & Su 2020). At iteration T the point with the lowest ln L, L*_T, is
removed and a new point is drawn from the prior inside {L > L*_T}. Its rank among the points that are
alive after the removal AND lie inside {L > L*_T} is uniform on 0..n_ref when the proposal samples the
constrained prior correctly. Two things the first version (sampler_report.py of 18 Nov) got wrong:
  1. ties at L*: it ranked the new point among every alive point. While dynesty works through a plateau
     (here: the live points whose ln L is the driver's floor -1e300, i.e. core's -inf where the variance
     guard trips), the other floor points are alive but outside {L > L*}; every new point ranks above all
     of them, so the ranks pile up at the top. They are now excluded (n_ref = number of alive points
     strictly above L*; n_ref = nlive - 1 outside a plateau), and ties with the new point are broken at
     random.
  2. finished runs: a converged run's state.save carries the final live points twice (appended to the
     dead points by add_live, and still in the live arrays), so they were counted twice in every alive
     set after their birth. Only the first niter dead points are used now (results.pkl: the first niter
     samples are the dead points, the last nlive the final live points).
u = (rank + V)/(n_ref + 1), V ~ U(0,1) spread over the tied positions; KS test against U(0,1).

What it can and cannot detect: it compares each new point with the live points it joins. A proposal that
misses part of {L > L*} (a bound that cuts the contour) pushes the ranks up while the live points still
remember the missed part; once every live point has been drawn by the same faulty proposal, the run is
self-consistent and the test passes. It cannot see a region that neither the live points nor the
proposals reach (a mode lost early, or a run confined to a subspace from the start). For that:
  - geometry(): the log-volume of the bound in use at each iteration against the run's own ln X_T. A bound
    that covers {L > L*} has volume >= X_T (dynesty's multi-ellipsoid volume is the sum over ellipsoids);
    ln V_bound - ln X_T far below 0 means the bound cannot contain the contour (only uniform sampling draws
    from the bound; for rslice/rwalk it sets the step axes only). Also the flatness of the live set: the
    singular values of the live points in the unit cube; a run confined to a subspace has a smallest
    singular value orders of magnitude below the largest.
  - two independent runs, and a run with another proposal (compare_sampler_runs.py).
  dynesty's bound volume is its nominal logvol_tot: dynesty 2.1.4 caps every axis at sqrt(n)/2 when it
  inflates an ellipsoid (bootstrap expansion), so the true volume can be smaller than the nominal one; a
  nominal volume below X is conclusive, one above X is not.
The KS test assumes independent ranks; for rslice/rwalk the new point is correlated with the live point it
started from (the marginal stays uniform for a correct sampler).

    <core venv python> tests/nested_diagnostics.py PART_DIR|state.save|results.pkl [...] [--out JSON] [--block N]
"""
from __future__ import annotations

import bisect
import json
import pickle
from pathlib import Path

import numpy as np

FLOOR = -1e299          # the driver maps core's -inf to -1e300 (dynesty's _LOWL_VAL)


# ============================================================================ loading
def _from_state(s, src):
    """dynesty sampler (restored state) -> run arrays (main-loop dead points only)."""
    sr = s.saved_run
    niter = int(s.it - 1)
    g = lambda k: np.asarray(sr[k])[:niter]  # noqa: E731
    out = {"source": str(src), "kind": "state", "niter": niter, "nlive": int(s.nlive),
           "dead_logl": g("logl").astype(float), "dead_it": g("it").astype(np.int64), "dead_u": g("u").astype(float),
           "dead_nc": g("nc").astype(float), "dead_logvol": g("logvol").astype(float),
           "dead_logz": g("logz").astype(float), "dead_bounditer": g("bounditer").astype(np.int64),
           "live_logl": np.asarray(s.live_logl, float), "live_it": np.asarray(s.live_it, np.int64),
           "live_u": np.asarray(s.live_u, float), "ncall": int(s.ncall),
           "logz": float(np.asarray(sr["logz"])[niter - 1]) if niter else None,
           "bound_logvol": [float(getattr(b, "logvol_tot", getattr(b, "logvol", 0.0))) for b in s.bound],
           "added_live_in_state": bool(getattr(s, "added_live", False)),
           "settings": {"method": s.method, "bootstrap": int(s.bootstrap), "enlarge": float(s.enlarge),
                        "nlive": int(s.nlive), "first_update_min_ncall": int(s.first_bound_update_ncall),
                        "first_update_min_eff": float(s.first_bound_update_eff),
                        "bound_update_interval": int(s.bound_update_interval),
                        "walks": int(getattr(s, "walks", -1)), "slices": int(getattr(s, "slices", -1))}}
    return out


def _from_results(r, src):
    """dynesty Results (add_live: niter dead points, then the nlive final live points) -> run arrays."""
    niter, nlive = int(r["niter"]), int(r["nlive"])
    logl = np.asarray(r["logl"], float)
    it = np.asarray(r["samples_it"], np.int64)
    u = np.asarray(r["samples_u"], float)
    if len(logl) != niter + nlive:
        raise ValueError(f"{src}: {len(logl)} samples, expected niter + nlive = {niter + nlive} (static run with add_live)")
    out = {"source": str(src), "kind": "results", "niter": niter, "nlive": nlive,
           "dead_logl": logl[:niter], "dead_it": it[:niter], "dead_u": u[:niter],
           "dead_nc": np.asarray(r["ncall"], float)[:niter], "dead_logvol": np.asarray(r["logvol"], float)[:niter],
           "dead_logz": np.asarray(r["logz"], float)[:niter],
           "dead_bounditer": np.asarray(r["bound_iter"], np.int64)[:niter] if "bound_iter" in r.keys() else None,
           "live_logl": logl[niter:], "live_it": it[niter:], "live_u": u[niter:],
           "ncall": int(np.sum(r["ncall"])), "logz": float(r["logz"][-1]), "logzerr": float(r["logzerr"][-1]),
           "bound_logvol": [float(getattr(b, "logvol_tot", getattr(b, "logvol", 0.0))) for b in (r["bound"] or [])]
           if "bound" in r.keys() and r["bound"] is not None else [],
           "settings": None}
    return out


def load_run(path):
    """A part directory (results.pkl preferred, else state.save), a state file or a results.pkl -> run arrays.
    A part directory also contributes its result.json / progress.json sampler record."""
    p = Path(path)
    rec = None
    if p.is_dir():
        for f in ("result.json", "progress.json"):
            if (p / f).exists():
                rec = json.loads((p / f).read_text())
                break
        if (p / "results.pkl").exists():
            p_load = p / "results.pkl"
        elif (p / "state.save").exists():
            p_load = p / "state.save"
        else:
            raise FileNotFoundError(f"{p}: no results.pkl or state.save")
    else:
        p_load = p
    with open(p_load, "rb") as fh:
        obj = pickle.load(fh)
    if isinstance(obj, dict) and "sampler" in obj:
        R = _from_state(obj["sampler"], p_load)
    elif hasattr(obj, "saved_run"):
        R = _from_state(obj, p_load)
    else:
        R = _from_results(obj, p_load)
    if R["settings"] is None:
        # the sampler settings of a results.pkl come from the run's state file next to it, if any
        sib = [p / "state.save"] if p.is_dir() else [p.with_name(p.name.replace(".results.pkl", x)) for x in (".save", ".ckpt")]
        for f in sib:
            if f.exists() and f != p_load:
                try:
                    with open(f, "rb") as fh:
                        st = pickle.load(fh)
                    st = st["sampler"] if isinstance(st, dict) and "sampler" in st else st
                    R["settings"] = _from_state(st, f)["settings"]
                    R["settings_from"] = str(f)
                    break
                except Exception as e:  # noqa: BLE001
                    R["settings_error"] = repr(e)
    if rec is not None:
        R["record"] = {k: rec.get(k) for k in ("sampler", "sampler_switch", "finished", "status", "logZ", "logZerr",
                                              "ncall_total", "niter", "n_neginf_this_session")}
    return R


# ============================================================================ insertion ranks
def insertion_ranks(R, seed=0):
    """Per iteration T = 1..niter: (u in [0,1), n_ref, plateau, n_tied) of the point born at T (u nan where
    no point was born at T; n_tied = alive points at or below L* after the removal, i.e. the rest of a
    plateau). Points: the initial live points (born 0), every point born at 1..niter."""
    niter, nlive = R["niter"], R["nlive"]
    dl, dit = R["dead_logl"], R["dead_it"]
    born_logl = {}
    for L, b in zip(np.concatenate([dl, R["live_logl"]]), np.concatenate([dit, R["live_it"]])):
        b = int(b)
        if b == 0:
            continue
        if b in born_logl:
            raise ValueError(f"two points born at iteration {b}")
        born_logl[b] = float(L)
    alive = sorted(float(L) for L, b in zip(np.concatenate([dl, R["live_logl"]]), np.concatenate([dit, R["live_it"]]))
                   if int(b) == 0)
    if len(alive) != nlive:
        raise ValueError(f"{len(alive)} initial live points, nlive {nlive}")
    rng = np.random.default_rng(seed)
    u = np.full(niter, np.nan)
    nref = np.zeros(niter, np.int64)
    plateau = np.zeros(niter, bool)
    ntied = np.zeros(niter, np.int64)
    for T in range(1, niter + 1):
        Ls = float(dl[T - 1])
        k = bisect.bisect_left(alive, Ls)
        if k >= len(alive) or alive[k] != Ls:
            raise ValueError(f"iteration {T}: the dead point (ln L {Ls}) is not among the alive points")
        del alive[k]
        i0 = bisect.bisect_right(alive, Ls)            # alive points at or below L*: outside {L > L*}
        n_ref = len(alive) - i0
        plateau[T - 1] = i0 > 0
        ntied[T - 1] = i0
        nref[T - 1] = n_ref
        Ln = born_logl.get(T)
        if Ln is None:
            continue
        lo = bisect.bisect_left(alive, Ln) - i0
        hi = bisect.bisect_right(alive, Ln) - i0
        u[T - 1] = (lo + rng.random() * (hi - lo + 1)) / (n_ref + 1)
        bisect.insort(alive, Ln)
    return u, nref, plateau, ntied


def insertion_ranks_old(R):
    """The first version's ranks (every alive point, divided by nlive) -- for comparison only."""
    niter, nlive = R["niter"], R["nlive"]
    ll = np.concatenate([R["dead_logl"], R["live_logl"]])
    born = np.concatenate([R["dead_it"], R["live_it"]])
    died = np.concatenate([np.arange(1, niter + 1), np.full(len(R["live_logl"]), np.iinfo(np.int64).max)])
    r = np.full(niter, np.nan)
    where = {int(b): i for i, b in enumerate(born) if b > 0}
    for T in range(1, niter + 1):
        i = where.get(T)
        if i is None:
            continue
        alive = (born < T) & (died > T)
        r[T - 1] = np.sum(ll[alive] < ll[i]) / nlive
    return r


def ks_summary(u, seed=None):
    from scipy.stats import kstest
    x = np.asarray(u, float)
    x = x[np.isfinite(x)]
    if x.size < 10:
        return {"n": int(x.size), "ks_p": None, "ks_D": None, "mean_u": None}
    t = kstest(x, "uniform")
    return {"n": int(x.size), "ks_p": float(t.pvalue), "ks_D": float(t.statistic), "mean_u": float(np.mean(x))}


def insertion_test(R, block=None, seed=0, start=0):
    """Whole run (iterations after `start`) and blocks of `block` iterations (default nlive)."""
    u, nref, plat, _ = insertion_ranks(R, seed=seed)
    block = block or R["nlive"]
    first_after = int(np.nonzero(plat)[0].max()) + 1 if plat.any() else 0     # 0-based index after the plateau
    out = {"convention": "rank among the alive points strictly above L* (ties at L*, e.g. the -1e300 floor, excluded), "
                         "u = (rank + V)/(n_ref + 1)",
           "n_initial_at_floor": int(np.sum(R["dead_logl"][R["dead_it"] == 0] <= FLOOR)
                                     + np.sum(R["live_logl"][R["live_it"] == 0] <= FLOOR)),
           "n_plateau_iterations": int(plat.sum()), "start_iteration": int(start),
           "whole": ks_summary(u[start:]),
           "after_plateau": ks_summary(u[max(start, first_after):]),
           "blocks": []}
    for a in range(start, R["niter"], block):
        b = min(a + block, R["niter"])
        out["blocks"].append({"iterations": [a + 1, b], **ks_summary(u[a:b]),
                              "min_n_ref": int(nref[a:b].min()) if b > a else None,
                              "calls_per_iteration": float(np.mean(R["dead_nc"][a:b])) if b > a else None})
    ps = [x["ks_p"] for x in out["blocks"] if x["ks_p"] is not None]
    out["n_blocks"] = len(ps)
    out["n_blocks_p_lt_0.01"] = int(sum(p < 0.01 for p in ps))
    out["expected_blocks_p_lt_0.01"] = 0.01 * len(ps)
    return out


# ============================================================================ geometry
def alive_at(R, T):
    """Indices (into concatenated dead+live) of the points alive after iteration T (T = 0: the initial set)."""
    niter = R["niter"]
    born = np.concatenate([R["dead_it"], R["live_it"]])
    died = np.concatenate([np.arange(1, niter + 1), np.full(len(R["live_it"]), np.iinfo(np.int64).max)])
    return np.nonzero((born <= T) & (died > T))[0]


def contour_lnX(R):
    """ln of the prior volume of {L > L*_T} after each iteration: dynesty's ln X_T, minus, inside a plateau,
    the share dynesty still assigns to the tied points not yet removed (each removal of a plateau point
    takes X_T0/(nlive+1), X_T0 the volume where the plateau began)."""
    _, _, plat, ntied = insertion_ranks(R)
    lx = np.array(R["dead_logvol"], float)
    X = np.exp(lx)
    nl = R["nlive"]
    out = lx.copy()
    T = 0
    while T < len(lx):
        if not plat[T]:
            T += 1
            continue
        a = T                                         # first plateau iteration (0-based)
        while T < len(lx) and plat[T]:
            T += 1
        dV = X[a] / nl                                 # = X_T0 / (nlive + 1)
        out[a:T] = np.log(np.maximum(X[a:T] - ntied[a:T] * dV, 1e-300))
    return out


def geometry(R, block=None):
    """(a) ln V of the bound in use minus ln X of the contour {L > L*} at each iteration (uniform sampling
    draws from the bound: it must hold the contour, so this should not fall much below 0); (b) flatness of
    the alive set: singular values of its unit-cube coordinates at the end of every block."""
    out = {}
    niter = R["niter"]
    bi = R.get("dead_bounditer")
    bl = R.get("bound_logvol") or []
    if bi is not None and len(bl):
        bi = np.asarray(bi)
        ok = (bi >= 1) & (bi < len(bl))             # bound 0 is the unit cube
        if ok.any():
            lv = np.asarray(bl)[bi[ok]]
            d = lv - contour_lnX(R)[ok]
            j = int(np.argmin(d))
            out["bound_minus_lnX"] = {"min": float(d[j]), "at_iteration": int(np.nonzero(ok)[0][j] + 1),
                                      "median": float(np.median(d)), "first_bound_iteration": int(np.argmax(ok) + 1),
                                      "first_bound_logvol": float(lv[0]),
                                      "first_bound_lnX": float(R["dead_logvol"][np.argmax(ok)]),
                                      "n_bounds": int(len(bl) - 1)}
    U = np.concatenate([R["dead_u"], R["live_u"]])
    block = block or R["nlive"]
    rows = []
    for T in list(range(block, niter, block)) + [niter]:
        idx = alive_at(R, T)
        X = U[idx]
        ll = np.concatenate([R["dead_logl"], R["live_logl"]])[idx]
        X = X[ll > FLOOR]
        if len(X) < X.shape[1] + 1:
            rows.append({"after_iteration": int(T), "n": int(len(X)), "sv_min_over_max": None})
            continue
        sv = np.linalg.svd(X - X.mean(0), compute_uv=False)
        rows.append({"after_iteration": int(T), "n": int(len(X)), "sv_min_over_max": float(sv[-1] / sv[0]),
                     "sv": [float(x) for x in sv]})
    out["flatness"] = rows
    vals = [r["sv_min_over_max"] for r in rows if r["sv_min_over_max"] is not None]
    out["min_sv_ratio"] = float(min(vals)) if vals else None
    return out


def verdict(ins, geo, method=None):
    """Short flags (not a pass/fail of the posterior)."""
    flags = []
    if geo.get("min_sv_ratio") is not None and geo["min_sv_ratio"] < 1e-3:
        flags.append(f"live set confined to a subspace (smallest/largest singular value {geo['min_sv_ratio']:.1e})")
    b = geo.get("bound_minus_lnX")
    if b and b["min"] < -1.0 and (method in (None, "unif")):
        flags.append(f"bound volume below the contour volume (ln V - ln X = {b['min']:.1f} at iteration {b['at_iteration']})")
    w = ins.get("after_plateau") or ins["whole"]
    if w.get("ks_p") is not None and w["ks_p"] < 0.01:
        flags.append(f"insertion ranks non-uniform after the plateau (KS p {w['ks_p']:.1e}, mean u {w['mean_u']:.3f})")
    if ins["n_blocks_p_lt_0.01"] > max(1, 3 * ins["expected_blocks_p_lt_0.01"]):
        flags.append(f"{ins['n_blocks_p_lt_0.01']} of {ins['n_blocks']} blocks with KS p < 0.01")
    return flags


def diagnose(path, block=None, label=None):
    """One run -> {source, niter, ncall, settings, insertion, geometry (compact), flags}."""
    R = load_run(path)
    ins = insertion_test(R, block=block)
    geo = geometry(R, block=block)
    meth = (R.get("settings") or {}).get("method") or ((R.get("record") or {}).get("sampler") or {}).get("method")
    return {"label": label or str(path), "source": R["source"], "kind": R["kind"], "niter": R["niter"], "nlive": R["nlive"],
            "ncall": R["ncall"], "settings": R.get("settings"), "record": R.get("record"),
            "insertion": ins, "geometry": {"bound_minus_lnX": geo.get("bound_minus_lnX"), "min_sv_ratio": geo["min_sv_ratio"]},
            "flags": verdict(ins, geo, meth)}


def _f(x, fmt):
    return "n/a" if x is None else format(x, fmt)


def summary_line(d):
    ins, g = d["insertion"], d["geometry"]
    w, ap_ = ins["whole"], ins["after_plateau"]
    b = g.get("bound_minus_lnX") or {}
    s = d.get("settings") or {}
    return (f"{d['label']}: {s.get('method', '?')} boot {s.get('bootstrap', '?')} enl {s.get('enlarge', '?')}; niter {d['niter']} "
            f"ncall {d['ncall']}; floor {ins['n_initial_at_floor']}/{d['nlive']}, plateau {ins['n_plateau_iterations']} it; "
            f"KS whole p {_f(w['ks_p'], '.3g')} (n {w['n']}, mean u {_f(w['mean_u'], '.3f')}), after plateau p {_f(ap_['ks_p'], '.3g')}; "
            f"blocks p<0.01: {ins['n_blocks_p_lt_0.01']}/{ins['n_blocks']}; ln V_bound - ln X min {_f(b.get('min'), '.2f')} "
            f"(first bound it {b.get('first_bound_iteration')}, ln V {_f(b.get('first_bound_logvol'), '.2f')}); "
            f"flatness {_f(g['min_sv_ratio'], '.2g')}\n    calls per iteration by block: "
            + " ".join(_f(x.get("calls_per_iteration"), ".1f") for x in ins["blocks"])
            + (f"\n    FLAGS: {'; '.join(d['flags'])}" if d["flags"] else ""))


def main(argv=None):
    import argparse
    import sys
    import time
    ap = argparse.ArgumentParser(description="corrected insertion-index test and bound geometry of dynesty runs")
    ap.add_argument("paths", nargs="+", help="part directories, state files or results.pkl (LABEL=PATH to name one)")
    ap.add_argument("--block", type=int, default=None, help="iterations per block (default nlive)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    rows = []
    for p in a.paths:
        lab, _, path = p.rpartition("=") if "=" in p else ("", "", p)
        try:
            d = diagnose(path, block=a.block, label=lab or None)
        except Exception as e:  # noqa: BLE001
            d = {"label": lab or path, "error": f"{type(e).__name__}: {e}"}
        rows.append(d)
        if "error" in d:
            print(f"{d['label']}: ERROR {d['error']}")
            continue
        print(summary_line(d))
    if a.out:
        Path(a.out).write_text(json.dumps({"written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                           "runs": rows}, indent=1, default=float))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
