#!/usr/bin/env python
"""Is dynesty's ln Z right when most initial live points sit on the driver's floor (-1e300)?

An analytic test on CPU. The likelihood, in d = 6 or 8 dimensions on the unit cube (prior volume 1):

    ln L(x) = -|x - mu|^2 / (2 sigma^2)   inside a box R of volume q (sigma = 0.03, mu well inside R)
    ln L(x) = -1e300                      outside R (the driver's floor: core's -inf, the variance guard)

so ln Z = sum_i ln[sigma sqrt(2 pi) (Phi((b_i - mu_i)/sigma) - Phi((a_i - mu_i)/sigma))] exactly, the same
for every q (the Gaussian is ~7 sigma or more from every face). q = 0.02, 0.25, 1 put 98%, 75% and 0% of the
prior on the floor (production: 66-99% of the 200 initial points).

Sampler: exactly the driver's (prod_driver.run_nested): dynesty.NestedSampler(nlive 200, bound multi,
sample unif [the driver's choice up to six parameters; V8 uses it in eight] or rslice [the driver's choice
above six], bootstrap/enlarge/slices at dynesty's defaults [the driver passes None], rstate
default_rng(seed), first_update {min_ncall 400, min_eff 100}, blob=True), checkpointing installed, the
driver's loop (run_nested(maxcall=2000, dlogz=0.1, add_live=False, resume=True) chunks until the driver's
delta_logz < 0.1, s.save after each chunk), then run_nested(add_live=True); the likelihood returns
(-1e300, blob) on the floor as the driver does. Arm "neginf" returns -inf instead (dynesty's own handling
of -inf at initialisation; not the driver's behaviour, for comparison only).

Per run: n_f (initial points on the floor = dead points at -1e300), dynesty's ln X at the end of the
plateau against ln q, ln Z - ln Z_true and its ratio to dynesty's logzerr, the plateau-corrected ln Z
(plateau_correction.py) and the posterior mean against mu.

    <core venv python> tests/test_plateau_evidence.py [--seeds 20] [--procs 8] [--only 6:0.02:unif:1e300 ...]
        -> tests/results/plateau_evidence.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np  # noqa: E402

A16 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(A16 / "scripts"))
import plateau_correction as PC  # noqa: E402

SIGMA = 0.03
NLIVE = 200
DLOGZ = 0.1
FLOOR = -1e300
SCRATCH = os.environ.get("A16_PLATEAU_SCRATCH", tempfile.gettempdir())


def geometry(d, q):
    """Box R = [a, a + s]^d (s = q^(1/d)), all boxes of one d share a centre c; mu = c + offset."""
    g = np.random.default_rng(1000 + d)
    c = g.uniform(0.43, 0.57, d)
    mu = c + g.uniform(-0.05, 0.05, d)
    s = q ** (1.0 / d)
    a = np.clip(c - s / 2, 0.0, 1.0) if q < 1 else np.zeros(d)
    b = np.clip(c + s / 2, 0.0, 1.0) if q < 1 else np.ones(d)
    return a, b, mu


def lnz_true(a, b, mu, sigma=SIGMA):
    from scipy.special import log_ndtr, ndtr
    out = 0.0
    for ai, bi, mi in zip(a, b, mu):
        p = ndtr((bi - mi) / sigma) - ndtr((ai - mi) / sigma)
        out += math.log(sigma * math.sqrt(2 * math.pi)) + math.log(p)
    return out


def delta_logz(s):
    """prod_driver.delta_logz."""
    sr = s.saved_run
    if len(sr["logl"]) == 0:
        return float("inf")
    return float(np.logaddexp(0, np.max(s.live_logl) + sr["logvol"][-1] - sr["logz"][-1]))


def run_one(spec):
    try:
        return _run_one(spec)
    except Exception as e:  # noqa: BLE001  (dynesty failures are a result of this test)
        import traceback
        tb = traceback.extract_tb(e.__traceback__)
        return {**spec, "dynesty_error": repr(e), "where": [f"{f.filename.split('/')[-1]}:{f.lineno} {f.name}" for f in tb[-4:]]}


def _run_one(spec, return_results=False):
    """One run (the driver's sampler and loop) -> its row; with return_results, (row, dynesty Results or None)."""
    import dynesty
    from darksirens.inference.dynesty_checkpoint import install_dynesty_checkpointing
    d, q, sample, floor_mode, seed = spec["d"], spec["q"], spec["sample"], spec["floor"], spec["seed"]
    a, b, mu = geometry(d, q)
    q_exact = float(np.prod(b - a))
    inv2s2 = 1.0 / (2 * SIGMA ** 2)
    floor_val = FLOOR if floor_mode == "1e300" else -np.inf
    cnt = {"n": 0, "floor": 0}

    def loglike(x):
        cnt["n"] += 1
        blob = np.array([float(x[0]), 0.0, 1.0, 0.0, 0.0])
        if np.any(x < a) or np.any(x > b):
            cnt["floor"] += 1
            return floor_val, blob
        return float(-np.sum((x - mu) ** 2) * inv2s2), blob

    lo, hi = np.zeros(d), np.ones(d)

    def ptform(u):
        return lo + u * (hi - lo)

    t0 = time.time()
    s = dynesty.NestedSampler(loglike, ptform, d, nlive=NLIVE, bound="multi", sample=sample,
                              rstate=np.random.default_rng(seed), first_update={"min_ncall": 2 * NLIVE,
                                                                                 "min_eff": 100.0}, blob=True)
    install_dynesty_checkpointing(s)
    settings = {"method": s.method, "bootstrap": int(s.bootstrap), "enlarge": float(s.enlarge),
                "slices": int(getattr(s, "slices", -1)), "bound_update_interval": int(s.bound_update_interval),
                "first_update": [int(s.first_bound_update_ncall), float(s.first_bound_update_eff)],
                "logvol_init": float(s.logvol_init)}
    n_init_floor_live = int(np.sum(np.asarray(s.live_logl) <= -1e299))
    spec["n_init_floor_live"] = n_init_floor_live          # kept in the row if dynesty fails later
    if n_init_floor_live == NLIVE:
        # every initial point on the floor: dynesty stops at once ("plateau") with no dead point and its
        # run_nested fails in compute_integrals (IndexError); the driver would fail the same way
        row = {**spec, "all_initial_on_floor": True, "settings": settings, "seconds": time.time() - t0}
        return (row, None) if return_results else row
    fd, ckpt = tempfile.mkstemp(prefix=f"plateau_{d}_{q}_{sample}_{floor_mode}_{seed}_", suffix=".save", dir=SCRATCH)
    os.close(fd)
    try:
        while True:
            dlz = delta_logz(s) if s.it > 1 else float("inf")
            if dlz < DLOGZ:
                break
            s.run_nested(maxcall=2000, maxiter=None, dlogz=DLOGZ, add_live=False, print_progress=False,
                         checkpoint_file=ckpt, checkpoint_every=600.0, resume=True)
            s.save(ckpt)
        s.run_nested(dlogz=DLOGZ, add_live=True, print_progress=False, checkpoint_file=ckpt,
                     checkpoint_every=600.0, resume=True)
    finally:
        try:
            os.remove(ckpt)
        except OSError:
            pass
    res = s.results
    logl = np.asarray(res["logl"])
    niter = int(res["niter"])
    logz = float(res["logz"][-1])
    logzerr = float(res["logzerr"][-1])
    truth = lnz_true(a, b, mu)
    n_f = int(np.sum(logl[:niter] <= -1e299))
    lv = np.asarray(res["logvol"])
    lnX_plateau_end = float(lv[n_f - 1]) if n_f > 0 else 0.0
    if floor_mode == "1e300":
        corr = PC.plateau_corrected_logz(res)
    else:   # -inf: dynesty's initialiser keeps >= 100 finite points and sets logvol_init = -ln(attempts)
        corr = {"logz": logz, "logzerr": logzerr, "note": "neginf arm: dynesty's own -inf handling, no correction"}
    w = np.exp(np.asarray(res["logwt"]) - logz)
    w /= w.sum()
    mean = np.sum(w[:, None] * np.asarray(res["samples"]), axis=0)
    # the information H from the weights (independent of dynesty's running h)
    lw = np.asarray(res["logwt"]) - logz
    ok = logl > -1e299
    H = float(np.sum(np.exp(lw[ok]) * (logl[ok] - logz)))
    row = {**spec, "q_exact": q_exact, "lnq": math.log(q_exact), "settings": settings,
            "n_init_floor_live": n_init_floor_live, "n_floor_dead": n_f, "niter": niter,
            "ncall": int(np.sum(res["ncall"])), "calls_on_floor": cnt["floor"], "calls": cnt["n"],
            "logz": logz, "logzerr": logzerr, "logz_true": truth, "dlogz": logz - truth,
            "z": (logz - truth) / logzerr if logzerr > 0 else float("nan"),
            "lnX_plateau_end": lnX_plateau_end,
            "lnX_plateau_end_minus_lnq": lnX_plateau_end - math.log(q_exact) if n_f > 0 else 0.0,
            "corrected": corr, "dlogz_corrected": corr["logz"] - truth,
            "z_corrected": (corr["logz"] - truth) / corr["logzerr"],
            "H": H, "sqrt_H_over_N": math.sqrt(max(H, 0) / NLIVE),
            "mean_offset_over_sigma": float(np.max(np.abs(mean - mu)) / SIGMA),
            "seconds": time.time() - t0}
    return (row, res) if return_results else row


def summarize(rows_all):
    out = {}
    failed = [r for r in rows_all if r.get("all_initial_on_floor")]
    errors = [r for r in rows_all if r.get("dynesty_error")]
    rows = [r for r in rows_all if not r.get("all_initial_on_floor") and not r.get("dynesty_error")]
    keys = sorted({(r["d"], r["q"], r["sample"], r["floor"]) for r in rows})
    for k in keys:
        rr = [r for r in rows if (r["d"], r["q"], r["sample"], r["floor"]) == k]
        dz = np.array([r["dlogz"] for r in rr])
        z = np.array([r["z"] for r in rr])
        err = np.array([r["logzerr"] for r in rr])
        dzc = np.array([r["dlogz_corrected"] for r in rr])
        zc = np.array([r["z_corrected"] for r in rr])
        nf = np.array([r["n_floor_dead"] for r in rr])
        pl = np.array([r["lnX_plateau_end_minus_lnq"] for r in rr])
        n = len(rr)
        out["d%d_q%g_%s_%s" % k] = {
            "d": k[0], "q": k[1], "sample": k[2], "floor": k[3], "n_runs": n,
            "n_failed_all_initial_on_floor": sum((r["d"], r["q"], r["sample"], r["floor"]) == k for r in failed),
            "n_dynesty_errors": sum((r["d"], r["q"], r["sample"], r["floor"]) == k for r in errors),
            "dynesty_errors": sorted({r["dynesty_error"] for r in errors
                                      if (r["d"], r["q"], r["sample"], r["floor"]) == k}),
            "floor_fraction_initial_mean": float(nf.mean() / NLIVE), "n_floor_range": [int(nf.min()), int(nf.max())],
            "dlogz_mean": float(dz.mean()), "dlogz_sem": float(dz.std(ddof=1) / math.sqrt(n)) if n > 1 else None,
            "dlogz_sd": float(dz.std(ddof=1)) if n > 1 else None, "logzerr_mean": float(err.mean()),
            "z_mean": float(z.mean()), "z_sd": float(z.std(ddof=1)) if n > 1 else None,
            "z_rms": float(np.sqrt(np.mean(z ** 2))), "frac_abs_z_lt_1": float(np.mean(np.abs(z) < 1)),
            "frac_abs_z_lt_2": float(np.mean(np.abs(z) < 2)),
            "plateau_lnX_minus_lnq_mean": float(pl.mean()), "plateau_lnX_minus_lnq_sd": float(pl.std(ddof=1)) if n > 1 else None,
            "rest_mean": float((dz - pl).mean()), "rest_sd": float((dz - pl).std(ddof=1)) if n > 1 else None,
            "corrected_dlogz_mean": float(dzc.mean()), "corrected_dlogz_sd": float(dzc.std(ddof=1)) if n > 1 else None,
            "corrected_logzerr_mean": float(np.mean([r["corrected"]["logzerr"] for r in rr])),
            "corrected_z_mean": float(zc.mean()), "corrected_z_sd": float(zc.std(ddof=1)) if n > 1 else None,
            "corrected_z_rms": float(np.sqrt(np.mean(zc ** 2))),
            "ncall_mean": float(np.mean([r["ncall"] for r in rr])), "niter_mean": float(np.mean([r["niter"] for r in rr])),
            "max_mean_offset_over_sigma": float(max(r["mean_offset_over_sigma"] for r in rr)),
            "seconds_mean": float(np.mean([r["seconds"] for r in rr]))}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--seed0", type=int, default=1)
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--only", nargs="*", default=None, metavar="D:Q:SAMPLE:FLOOR[:NSEEDS]")
    ap.add_argument("--out", default=str(A16 / "tests" / "results" / "plateau_evidence.json"))
    ap.add_argument("--append", action="store_true", help="add to the rows already in --out")
    args = ap.parse_args()
    if args.only:
        cells = []
        for c in args.only:
            p = c.split(":")
            cells.append((int(p[0]), float(p[1]), p[2], p[3], int(p[4]) if len(p) > 4 else args.seeds))
    else:
        cells = [(d, q, "unif", "1e300", args.seeds) for d in (6, 8) for q in (0.02, 0.25, 1.0)]
    specs = [{"d": d, "q": q, "sample": smp, "floor": fl, "seed": args.seed0 + i}
             for d, q, smp, fl, ns in cells for i in range(ns)]
    rows = []
    if args.append and Path(args.out).exists():
        rows = json.loads(Path(args.out).read_text())["runs"]
    t0 = time.time()
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(args.procs, maxtasksperchild=1) as pool:
        for r in pool.imap_unordered(run_one, specs):
            rows.append(r)
            if r.get("dynesty_error"):
                print(f"d{r['d']} q{r['q']} {r['sample']} {r['floor']} seed {r['seed']}: dynesty failed: "
                      f"{r['dynesty_error']} at {r['where']}", flush=True)
                continue
            if r.get("all_initial_on_floor"):
                print(f"d{r['d']} q{r['q']} {r['sample']} {r['floor']} seed {r['seed']}: all 200 initial points on "
                      "the floor (dynesty cannot start)", flush=True)
                continue
            print(f"d{r['d']} q{r['q']} {r['sample']} {r['floor']} seed {r['seed']}: n_f {r['n_floor_dead']} "
                  f"dlnZ {r['dlogz']:+.3f} (err {r['logzerr']:.3f}, z {r['z']:+.2f}) plateau {r['lnX_plateau_end_minus_lnq']:+.3f} "
                  f"corr {r['dlogz_corrected']:+.3f} (z {r['z_corrected']:+.2f}) ncall {r['ncall']} {r['seconds']:.0f}s",
                  flush=True)
    rows.sort(key=lambda r: (r["d"], r["q"], r["sample"], r["floor"], r["seed"]))
    out = {"written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "sigma": SIGMA, "nlive": NLIVE,
           "dlogz": DLOGZ, "floor": FLOOR, "wall_seconds": time.time() - t0,
           "dynesty": __import__("dynesty").__version__, "summary": summarize(rows),
           "binomial_expectation": PC.binomial_plateau_table(NLIVE, [0.02, 0.05, 0.1, 0.25, 0.5]),
           "runs": rows}
    tmp = Path(args.out).with_suffix(".tmp")
    tmp.write_text(json.dumps(out, indent=1))
    tmp.replace(args.out)
    for k, v in out["summary"].items():
        print(k, {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items()})


if __name__ == "__main__":
    main()
