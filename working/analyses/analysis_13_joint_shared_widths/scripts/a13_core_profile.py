#!/usr/bin/env python
"""Analysis 13 -- GPU profile of the per-call likelihood for darksirens-work (2026-10-07, owner-approved).

    A  every #52 default (a13_core_sampler.build("defaults"))
    B  every #52 default except pairing_norm = per_sample

Writes, under diagnostics/profile_<core>/<config>/:
  trace/                 jax.profiler trace (perfetto) of 10 calls after 3 warm-up calls,
                         block_until_ready on each
  compiled_analysis.json cost_analysis() and memory_analysis() of the per-call program, lowered with
                         the operands the call uses (jax.jit(bound.__call__).lower(theta).compile())
  timing.json            per-call seconds of the 10 profiled calls and their logL
HLO text dumps go to diagnostics/profile_<core>/hlo_<config>/ when the sbatch sets XLA_FLAGS.

    A13_CORE=pr57 envs/darksirens-core-pr57/bin/python a13_core_profile.py --config A
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import a13_core_sampler as S                                 # noqa: E402


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, (int, float, str, bool)) or x is None:
        return x
    try:
        return float(x)
    except Exception:  # noqa: BLE001
        return str(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=["A", "B"], required=True)
    args = ap.parse_args()
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import jax
    import jax.numpy as jnp
    import darksirens.population.utils as pu
    if args.config == "B":
        cfg_ng = pu.configure_normalization_grids
        pu.configure_normalization_grids = lambda **kw: cfg_ng(**{**kw, "pairing_norm": "per_sample"})
    out = S.A13 / "diagnostics" / f"profile_{S.CORE}" / args.config
    out.mkdir(parents=True, exist_ok=True)
    ds, b, settings = S.build("defaults")
    labels = list(b.labels)
    jb = jax.jit(b.__call__)
    Z = np.load(S.A13 / "results" / "a13core_dynesty_n200_s1.npz")
    w = np.exp(Z["logwt"] - Z["logwt"].max()); w /= w.sum()
    idx = np.random.default_rng(7).choice(len(w), size=13, replace=False, p=w)

    def theta(i):
        vals = S.absolute(Z["dead"][i])
        return jnp.asarray([vals[l] for l in labels], dtype=jnp.float64)

    for i in idx[:3]:
        jb(theta(i)).block_until_ready()
    secs, vals = [], []
    with jax.profiler.trace(str(out / "trace"), create_perfetto_trace=True):
        for i in idx[3:]:
            t = theta(i)
            t0 = time.time()
            v = jb(t).block_until_ready()
            secs.append(time.time() - t0)
            vals.append(float(v))
    comp = jb.lower(theta(idx[3])).compile()
    try:
        cost = comp.cost_analysis()
    except Exception as e:  # noqa: BLE001
        cost = {"error": str(e)}
    try:
        m = comp.memory_analysis()
        mem = {k: getattr(m, k) for k in dir(m) if not k.startswith("_") and not callable(getattr(m, k))}
    except Exception as e:  # noqa: BLE001
        mem = {"error": str(e)}
    (out / "compiled_analysis.json").write_text(json.dumps(
        {"config": args.config, "core": S.CORE, "catalog_evaluation": settings,
         "cost_analysis": jsonable(cost), "memory_analysis": jsonable(mem)}, indent=2))
    (out / "timing.json").write_text(json.dumps(
        {"config": args.config, "catalog_evaluation": settings, "seconds": secs,
         "seconds_median": float(np.median(secs)), "logL": vals,
         "seed1_logL": [float(Z["logl"][i]) for i in idx[3:]],
         "slurm_job_id": os.environ.get("SLURM_JOB_ID")}, indent=2))
    print(f"[{args.config}] {settings}  median {np.median(secs):.4f} s/call  wrote {out}")


if __name__ == "__main__":
    main()
