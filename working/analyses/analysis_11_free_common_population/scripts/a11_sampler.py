#!/usr/bin/env python
"""Analysis 11 -- nested sampling on the production likelihood (S0, S1, 11C, 11D).

The likelihood is the production darksirens one (``a11_likelihood.build_a11`` or,
for the S1 validation problem, ``a10_likelihood.build_a10``) built with the
likelihood-internal redshift-prior optimisation barrier OFF, the setting darksirens
itself selects for tinyns' JAX rwalk kernel (``factory._resolve_redshift_prior_
materialization``): ``lax.optimization_barrier`` cannot be vmapped.  S0 checks that
this changes no logL value.  The sampled vector is mapped to the full production
coordinate by a fixed affine map inside JAX (reporting coordinates -> absolute
per-catalogue labels), so the sampler's loglike is pure JAX.

Problems (flat priors on the boxes below; guard-rejected points are -inf)
    a10J   S1: Analysis 10's fixed-H0 joint arm, (f, dmu_chi, dmu_G) on its own grid
           box f [0,1], dmu_chi [-0.20, 0.25], dmu_G [-10, 10]; reference posterior
           results/a10_arm_J.json of Analysis 10.
    11C    (f, mu_G, dmu_G, mu_chi, dmu_chi) on the brief's domains, H0 = 67.74.
    11D    11C plus H0 on [60, 76] (Analysis 10's contained C10-J window).

Engines (the two darksirens supports for nested sampling; nothing reimplemented)
    tinyns   tinyns.NestedSampler, sample='rwalk', kernel='jax', one chain,
             rwalk_adaptive_step_scale=True (tinyns' own adaptive step; darksirens'
             run_sampler does not forward it, and a fixed 0.1 unit-cube step is
             several posterior widths here).  Checkpoint every 50 iterations.
    dynesty  dynesty.NestedSampler(bound='multi', sample='unif'), checkpointed
             through darksirens' install_dynesty_checkpointing /
             restore_dynesty_sampler.

Stages
    s0       GPU: barrier on/off equality, compile and steady-state times, vmap,
             device memory, and darksirens' own 32-draw nested-sampler preflight on
             the 11C box.
    run      GPU: --problem --engine --nlive --seed [--walks]; resumes from its
             checkpoint; writes results/a11_ns_{problem}_{engine}_n{nlive}_s{seed}.json
             (+ .npz of equal-weight samples).
    compare  CPU: an a10J run against the A10-J grid (median, 68/90% ends,
             correlations; criterion 0.1 posterior sd).
"""
import argparse
import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
A11 = HERE.parent
RESULTS, DIAG = A11 / "results", A11 / "diagnostics"
CKPT = A11 / "queue"
sys.path.insert(0, str(HERE))
A9_SCRIPTS = A11.parent / "analysis_9_marked_multitracer_H0_fagn" / "scripts"
A10_RESULTS = A11.parent / "analysis_10_mass_spin_marked_multitracer" / "results"

BOXES = {
    "a10J": [("f_agn", 0.0, 1.0), ("dmu_chi", -0.20, 0.25), ("dmu_G", -10.0, 10.0)],
    "11C": [("f_agn", 0.0, 1.0), ("mu_G", 31.0, 39.0), ("dmu_G", -4.0, 10.0),
            ("mu_chi", -0.10, 0.10), ("dmu_chi", -0.05, 0.20)],
    "11D": [("H0", 60.0, 76.0), ("f_agn", 0.0, 1.0), ("mu_G", 31.0, 39.0),
            ("dmu_G", -4.0, 10.0), ("mu_chi", -0.10, 0.10), ("dmu_chi", -0.05, 0.20)],
}


@contextmanager
def barrier_off():
    import a11_likelihood as L
    a8 = L.a8
    true_bo = a8.build_opts

    def bo(*a, **kw):
        o = true_bo(*a, **kw)
        o.redshift_prior_barrier = "off"
        return o

    a8.build_opts = bo
    try:
        yield
    finally:
        a8.build_opts = true_bo


def build(problem, barrier="off", data=None):
    import a11_likelihood as L
    ctx = barrier_off() if barrier == "off" else _null()
    with ctx:
        if problem == "a10J":
            cell = L.A10.build_a10(f"NS_{problem}_{barrier}", [L.SURVEY_GAL, L.SURVEY_AGN],
                                   data=data, gw_path=L.GW_PATH_A11)
        else:
            cell = L.build_a11(f"NS_{problem}_{barrier}", data=data)
    return cell


@contextmanager
def _null():
    yield


def affine_map(cell, problem):
    """(A, b, idx): coord[idx] = A @ theta + b  (reporting -> absolute labels)."""
    import a11_likelihood as L
    names = [n for n, _, _ in BOXES[problem]]
    rows = {}  # label -> (coeff dict, const)
    if problem == "a10J":
        rows["fcat_2"] = ({"f_agn": 1.0}, 0.0)
        rows[L.MU_CHI_C2_LABEL] = ({"dmu_chi": 1.0}, 0.0)
        rows[L.MU_G_C2_LABEL] = ({"dmu_G": 1.0}, 35.0)
    else:
        rows["fcat_2"] = ({"f_agn": 1.0}, 0.0)
        rows[L.MU_G_LABEL] = ({"mu_G": 1.0}, 0.0)
        rows[L.MU_G_C2_LABEL] = ({"mu_G": 1.0, "dmu_G": 1.0}, 0.0)
        rows[L.MU_CHI_LABEL] = ({"mu_chi": 1.0}, 0.0)
        rows[L.MU_CHI_C2_LABEL] = ({"mu_chi": 1.0, "dmu_chi": 1.0}, 0.0)
        if problem == "11D":
            rows["H0"] = ({"H0": 1.0}, 0.0)
    labels = list(cell.labels)
    idx = np.array([labels.index(l) for l in rows])
    A = np.zeros((len(rows), len(names)))
    b = np.zeros(len(rows))
    for r, (lbl, (co, c0)) in enumerate(rows.items()):
        for n, v in co.items():
            A[r, names.index(n)] = v
        b[r] = c0
    return A, b, idx, names


def make_fns(cell, problem):
    import jax
    import jax.numpy as jnp
    A, b, idx, names = affine_map(cell, problem)
    base = jnp.asarray(cell.base)
    A_, b_, idx_ = jnp.asarray(A), jnp.asarray(b), jnp.asarray(idx)
    lo = jnp.asarray([x for _, x, _ in BOXES[problem]])
    hi = jnp.asarray([x for _, _, x in BOXES[problem]])
    like = cell.likelihood

    def loglike(theta):
        coord = base.at[idx_].set(A_ @ theta + b_)
        return like(coord)

    def ptform(u):
        return lo + u * (hi - lo)

    return loglike, ptform, names


def stage_s0(args):
    import jax
    import jax.numpy as jnp
    sys.path.insert(0, str(A9_SCRIPTS))
    import a9_scan as A9
    env = A9._gpu_setup("a11_s0")
    out = {"environment": env, "slurm_job_id": os.environ.get("SLURM_JOB_ID")}
    t0 = time.time()
    on = build("11C", barrier="on")
    off = build("11C", barrier="off", data=on.data)
    out["build_seconds"] = time.time() - t0
    pts = [dict(fcat_2=0.3, mu_G=35.0, dmu_G=5.0, mu_chi=0.0, dmu_chi=0.10),
           dict(fcat_2=0.0, mu_G=33.0, dmu_G=-2.0, mu_chi=-0.05, dmu_chi=0.0),
           dict(fcat_2=1.0, mu_G=38.0, dmu_G=9.0, mu_chi=0.08, dmu_chi=0.15),
           dict(fcat_2=0.55, mu_G=31.5, dmu_G=2.5, mu_chi=0.02, dmu_chi=-0.04)]
    eq = []
    for p in pts:
        a = on.evaluate_at(H0=67.74, **p)
        b = off.evaluate_at(H0=67.74, **p)
        eq.append({"point": p, "on": a["logL"], "off": b["logL"],
                   "bitwise": float(a["logL"]).hex() == float(b["logL"]).hex(),
                   "abs_diff": abs(a["logL"] - b["logL"])})
    out["barrier_on_vs_off"] = eq
    loglike, ptform, names = make_fns(off, "11C")
    jl = jax.jit(loglike)
    th = jnp.asarray([0.3, 35.0, 5.0, 0.0, 0.10])
    t0 = time.time(); v0 = float(jl(th)); t_first = time.time() - t0
    ts = []
    for _ in range(4):
        t0 = time.time(); float(jl(th)); ts.append(time.time() - t0)
    ref = off.evaluate_at(H0=67.74, fcat_2=0.3, mu_G=35.0, dmu_G=5.0, mu_chi=0.0,
                          dmu_chi=0.10)["logL"]
    out["jit_loglike"] = {"first_call_seconds": t_first, "steady_seconds": float(np.median(ts)),
                          "value": v0, "equals_cell_evaluate": float(v0).hex() == float(ref).hex()}
    for width in (1, 2):
        try:
            vm = jax.jit(jax.vmap(loglike))
            ths = jnp.stack([th] + [th.at[1].set(34.5)] * (width - 1))
            t0 = time.time(); vv = np.asarray(vm(ths)); t_vf = time.time() - t0
            t0 = time.time(); np.asarray(vm(ths)); t_v2 = time.time() - t0
            out[f"vmap_width{width}"] = {
                "first_seconds": t_vf, "steady_seconds": t_v2, "values": vv.tolist(),
                "matches_scalar": bool(float(vv[0]).hex() == float(v0).hex())}
        except Exception as e:  # noqa: BLE001
            out[f"vmap_width{width}"] = {"error": f"{type(e).__name__}: {e}"}
    try:
        out["device_memory"] = {k: int(v) for k, v in jax.devices()[0].memory_stats().items()
                                if k in ("bytes_in_use", "peak_bytes_in_use", "bytes_limit")}
    except Exception as e:  # noqa: BLE001
        out["device_memory"] = {"error": str(e)}
    from types import SimpleNamespace
    from darksirens.inference.sampling import _nested_sampler_preflight
    import io
    import contextlib
    buf = io.StringIO()
    t0 = time.time()
    with contextlib.redirect_stdout(buf):
        _nested_sampler_preflight(jl, jax.jit(ptform), 5,
                                  SimpleNamespace(seed=0, nlive=400, sampler_preflight="on"))
    out["darksirens_preflight_11C"] = {"stdout": buf.getvalue().strip(),
                                       "seconds": time.time() - t0}
    print(json.dumps(out, indent=1, default=str))
    (DIAG / "a11_s0_sampler_preflight.json").write_text(json.dumps(out, indent=2, default=str))


def _paths(args):
    tag = f"a11_ns_{args.problem}_{args.engine}_n{args.nlive}_s{args.seed}"
    if args.engine == "tinyns":
        tag += f"_w{args.walks}"
    return tag, CKPT / f"{tag}.ckpt", RESULTS / f"{tag}.json", RESULTS / f"{tag}.npz"


def stage_run(args):
    import jax
    import jax.numpy as jnp
    sys.path.insert(0, str(A9_SCRIPTS))
    import a9_scan as A9
    env = A9._gpu_setup(f"a11_ns_{args.problem}")
    tag, ckpt, out_json, out_npz = _paths(args)
    CKPT.mkdir(exist_ok=True)
    t_start = time.time()
    cell = build(args.problem, barrier="off")
    loglike, ptform, names = make_fns(cell, args.problem)
    ndim = len(names)
    ncall = {"n": 0}
    info = {"tag": tag, "problem": args.problem, "engine": args.engine, "nlive": args.nlive,
            "seed": args.seed, "names": names, "box": BOXES[args.problem],
            "labels": list(cell.labels), "environment": env,
            "slurm_job_ids": [os.environ.get("SLURM_JOB_ID")]}
    if out_json.exists():
        prev = json.loads(out_json.read_text())
        if prev.get("finished"):
            print(f"{out_json} already finished")
            return
    if args.engine == "tinyns":
        from tinyns import NestedSampler
        sampler = NestedSampler(loglike, ptform, ndim, args.nlive, sample="rwalk",
                                kernel="jax", walks=args.walks, step_scale=0.1,
                                min_accepts=1, replacement_chains=1,
                                rwalk_adaptive_step_scale=True, rwalk_target_accept=0.25,
                                max_attempts=args.max_attempts, jax_block_size=1)
        kw = dict(dlogz=args.dlogz, progress=True, progress_interval=25,
                  checkpoint_interval=50)
        if ckpt.exists():
            print(f"[run] resuming {ckpt}", flush=True)
            result = sampler.resume(str(ckpt), checkpoint_path_out=str(ckpt), **kw)
        else:
            result = sampler.run(jax.random.PRNGKey(args.seed), checkpoint_path=str(ckpt), **kw)
        samples = np.asarray(result.resample_equal(jax.random.PRNGKey(args.seed + 7919)))
        try:
            diag = result.diagnostics()
        except Exception as e:  # noqa: BLE001
            diag = {"error": str(e)}
        info.update({"logZ": float(result.logz), "logZerr": float(result.logzerr),
                     "diagnostics": diag})
        logl = np.asarray(getattr(result, "logl", []))
        logwt = np.asarray(getattr(result, "logwt", []))
        dead = np.asarray(getattr(result, "samples", np.empty((0, ndim))))
    else:
        import dynesty
        from darksirens.inference.checkpointing import (install_dynesty_checkpointing,
                                                        restore_dynesty_sampler)
        jl = jax.jit(loglike)

        def ll_np(x):
            ncall["n"] += 1
            v = float(jl(jnp.asarray(x)))
            return v if np.isfinite(v) else -1e300

        def pt_np(u):
            lo = np.array([a for _, a, _ in BOXES[args.problem]])
            hi = np.array([c for _, _, c in BOXES[args.problem]])
            return lo + u * (hi - lo)

        if ckpt.exists():
            print(f"[run] resuming {ckpt}", flush=True)
            sampler = restore_dynesty_sampler(str(ckpt), ll_np, pt_np)
            resume = True
        else:
            sampler = dynesty.NestedSampler(
                ll_np, pt_np, ndim, nlive=args.nlive, bound="multi", sample="unif",
                rstate=np.random.default_rng(args.seed),
                first_update={"min_ncall": 2 * args.nlive,
                              "min_eff": float(args.first_update_min_eff)})
            resume = False
        install_dynesty_checkpointing(sampler)
        sampler.run_nested(dlogz=args.dlogz, print_progress=True, resume=resume,
                           checkpoint_file=str(ckpt), checkpoint_every=900)
        res = sampler.results
        import pickle
        with open(CKPT / f"{tag}.results.pkl", "wb") as fh:
            pickle.dump(res, fh)
        info["first_update_min_eff"] = float(args.first_update_min_eff)
        logwt = np.asarray(res["logwt"])
        w = np.exp(logwt - logwt.max())
        w /= w.sum()
        from dynesty.utils import resample_equal
        samples = resample_equal(np.asarray(res.samples), w,
                                 rstate=np.random.default_rng(args.seed + 7919))
        info.update({"logZ": float(res.logz[-1]), "logZerr": float(res.logzerr[-1]),
                     "ncall_total": int(np.sum(res.ncall)), "niter": int(res.niter),
                     "eff_percent": float(res.eff),
                     "ncall_this_session": ncall["n"]})
        logl = np.asarray(res.logl)
        dead = np.asarray(res.samples)
    info["wall_seconds_this_session"] = time.time() - t_start
    info["n_equal_weight_samples"] = int(samples.shape[0])
    info["summary"] = summarise(samples, names)
    info["finished"] = True
    np.savez(out_npz, samples=samples, names=np.array(names), logl=logl, logwt=logwt, dead=dead)
    out_json.write_text(json.dumps(info, indent=2, default=str))
    print(json.dumps(info["summary"], indent=1))
    print(f"wrote {out_json}")


def summarise(s, names):
    out = {}
    for i, n in enumerate(names):
        x = s[:, i]
        q = np.quantile(x, [0.05, 0.16, 0.5, 0.84, 0.95])
        out[n] = {"median": float(q[2]), "ci68": [float(q[1]), float(q[3])],
                  "ci90": [float(q[0]), float(q[4])], "mean": float(x.mean()),
                  "sd": float(x.std())}
    c = np.corrcoef(s.T)
    out["correlations"] = {f"{names[i]}|{names[j]}": float(c[i, j])
                           for i in range(len(names)) for j in range(i + 1, len(names))}
    return out


def stage_merge(args):
    """Merge independent dynesty runs (dynesty.utils.merge_runs) into one posterior."""
    import pickle
    from dynesty.utils import merge_runs, resample_equal
    runs, infos = [], []
    for j in args.runs:
        info = json.loads(Path(j).read_text())
        pk = CKPT / f"{info['tag']}.results.pkl"
        with open(pk, "rb") as fh:
            runs.append(pickle.load(fh))
        infos.append(info)
    names = infos[0]["names"]
    res = merge_runs(runs)
    logwt = np.asarray(res["logwt"])
    w = np.exp(logwt - logwt.max())
    w /= w.sum()
    samples = resample_equal(np.asarray(res.samples), w, rstate=np.random.default_rng(12345))
    out = {"what": f"merged dynesty runs for {infos[0]['problem']}",
           "runs": [{"tag": i["tag"], "logZ": i["logZ"], "logZerr": i["logZerr"],
                     "ncall_total": i["ncall_total"], "niter": i["niter"],
                     "summary": i["summary"]} for i in infos],
           "logZ": float(res.logz[-1]), "logZerr": float(res.logzerr[-1]),
           "ncall_total": int(sum(i["ncall_total"] for i in infos)),
           "gpu_hours_approx": float(sum(i["ncall_total"] for i in infos) * 3.0 / 3600),
           "n_equal_weight_samples": int(samples.shape[0]), "names": names,
           "box": infos[0]["box"], "summary": summarise(samples, names)}
    dst = RESULTS / f"a11_{args.out}.json"
    dst.write_text(json.dumps(out, indent=2))
    np.savez(RESULTS / f"a11_{args.out}.npz", samples=samples, names=np.array(names),
             logwt=logwt, logl=np.asarray(res.logl), dead=np.asarray(res.samples))
    print(json.dumps(out["summary"], indent=1))
    print(f"wrote {dst}")


def stage_compare(args):
    run = json.loads(Path(args.run_json).read_text())
    grid = json.loads((A10_RESULTS / "a10_arm_J.json").read_text())
    key = {"f_agn": "f", "dmu_chi": "dmu_chi", "dmu_G": "dmu_G"}
    rows, ok = {}, True
    for n, g in key.items():
        s, r = run["summary"][n], grid[g]
        sd = (r["ci68"][1] - r["ci68"][0]) / 2.0
        d = {"median": (s["median"] - r["median"]) / sd,
             "ci68_lo": (s["ci68"][0] - r["ci68"][0]) / sd,
             "ci68_hi": (s["ci68"][1] - r["ci68"][1]) / sd,
             "ci90_lo": (s["ci90"][0] - r["ci90"][0]) / sd,
             "ci90_hi": (s["ci90"][1] - r["ci90"][1]) / sd}
        rows[n] = {"sampler": {k: s[k] for k in ("median", "ci68", "ci90")},
                   "grid": {k: r[k] for k in ("median", "ci68", "ci90")},
                   "half_width_68_grid": sd, "diff_in_half_widths": d,
                   "max_abs": max(abs(v) for v in d.values())}
        ok &= rows[n]["max_abs"] <= args.tol
    gc = grid.get("posterior_moments", {}).get("correlation") or grid.get("correlations", {})
    corr = {"sampler": run["summary"]["correlations"], "grid": gc}
    out = {"run": args.run_json, "tolerance_half_widths": args.tol, "coordinates": rows,
           "correlations": corr, "pass": bool(ok)}
    dst = DIAG / f"a11_s1_compare_{Path(args.run_json).stem}.json"
    dst.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=("s0", "run", "merge", "compare"))
    ap.add_argument("--problem", choices=list(BOXES), default="a10J")
    ap.add_argument("--engine", choices=("tinyns", "dynesty"), default="dynesty")
    ap.add_argument("--nlive", type=int, default=250)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--walks", type=int, default=10)
    ap.add_argument("--max_attempts", type=int, default=2000)
    ap.add_argument("--dlogz", type=float, default=0.1)
    ap.add_argument("--run_json")
    ap.add_argument("--runs", nargs="+", help="merge: run JSONs")
    ap.add_argument("--out", help="merge: output tag, results/a11_<out>.json")
    ap.add_argument("--first_update_min_eff", type=float, default=10.0,
                    help="dynesty first_update min_eff (10 = dynesty default; 100 builds "
                         "the first bound as soon as 2*nlive calls are spent)")
    ap.add_argument("--tol", type=float, default=0.1)
    args = ap.parse_args(argv)
    {"s0": stage_s0, "run": stage_run, "merge": stage_merge,
     "compare": stage_compare}[args.stage](args)


if __name__ == "__main__":
    main()
