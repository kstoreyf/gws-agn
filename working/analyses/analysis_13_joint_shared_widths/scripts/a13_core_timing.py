#!/usr/bin/env python
"""Analysis 13 -- per-call timing of core's #52 defaults, one at a time (for darksirens-work, 2026-10-07).

The historical evaluation (a13_core_sampler.build("historical")) with at most ONE setting switched to
its #52 default:
    none             the historical baseline
    pairing_norm     pairing_norm "auto"   (historical "per_sample")
    missing_density  missing_density "auto" (historical "grid")
    kernel_window    kernel_window "auto"   (historical explicit 1e-10)
    kernel_layout    kernel_layout "galaxy_list" (historical "padded"; ~48 GB until core chunks the build)
    nopin            the baseline with model(kernel_pin="off")            (darksirens-work, PR #57 test)
    kernel_layout_nopin   galaxy_list with model(kernel_pin="off")
    all              every #52 default at once (a13_core_sampler.build("defaults"))
    all_but_<s>      every default except <s>, which keeps its historical value (darksirens-work)
A suffix picks how the bound likelihood is called (owner 2026-10-07; the driver uses the first):
    <switch>          jax.jit(bound.__call__): the data enter the outer program as constants
    <switch>@direct   bound(theta): core's own jitted call, data as operands
    <switch>@pytree   jax.jit over bound.as_pytree_callable(): data as arguments of the outer program
Times 100 of seed 1's posterior-weighted dead points after 5 warm-up calls and compares each logL with
the value seed 1 recorded (core bf58aa6, historical). One switch per process.

    A13_CORE=e7c3007 envs/darksirens-core-e7c3007/bin/python a13_core_timing.py --switch pairing_norm
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

DEFAULT = {"pairing_norm": "auto", "missing_density": "auto", "kernel_window": "auto",
           "kernel_layout": "galaxy_list"}
HISTORICAL = {"pairing_norm": "per_sample", "missing_density": "grid", "kernel_window": 1e-10,
              "kernel_layout": "padded"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--switch", required=True)
    ap.add_argument("--n", type=int, default=100)
    args = ap.parse_args()
    args.switch, _, call = args.switch.partition("@")
    call = call or "wrapped"
    if args.switch not in (["none", "nopin", "kernel_layout_nopin", "all"] + list(DEFAULT)
                           + [f"all_but_{s}" for s in DEFAULT]) or call not in ("wrapped", "direct", "pytree"):
        sys.exit(f"[fatal] unknown switch {args.switch}@{call}")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import jax
    import jax.numpy as jnp
    from darksirens.catalog import settings as cs
    import darksirens.population.utils as pu

    # build("historical") calls both configure functions explicitly; switch one keyword back
    cfg_cat, cfg_ng = cs.configure_catalog_evaluation, pu.configure_normalization_grids

    def cat(**kw):
        if args.switch in kw:
            kw[args.switch] = DEFAULT[args.switch]
        return cfg_cat(**kw)

    def ng(**kw):
        if args.switch in kw:
            kw[args.switch] = DEFAULT[args.switch]
        return cfg_ng(**kw)

    sw = args.switch.replace("_nopin", "") if args.switch != "nopin" else "none"

    def cat(**kw):  # noqa: F811
        if sw in kw:
            kw[sw] = DEFAULT[sw]
        return cfg_cat(**kw)

    cs.configure_catalog_evaluation, pu.configure_normalization_grids = cat, ng
    if args.switch.startswith("all_but_"):
        keep = args.switch[len("all_but_"):]
        if keep == "pairing_norm":   # build("defaults") calls configure_normalization_grids()
            pu.configure_normalization_grids = lambda **kw: cfg_ng(**{**kw, "pairing_norm": "per_sample"})
        else:                        # build("defaults") leaves the catalog settings as they are
            cfg_cat(**{keep: HISTORICAL[keep]})
    if args.switch.endswith("nopin"):
        import darksirens
        model = darksirens.model
        darksirens.model = lambda *a, **kw: model(*a, **{**kw, "kernel_pin": "off"})
    t0 = time.time()
    ds, b, settings = S.build("defaults" if args.switch.startswith("all") else "historical")
    build_s = time.time() - t0
    labels = list(b.labels)
    if call == "wrapped":
        jb = jax.jit(b.__call__)
    elif call == "direct":
        jb = b
    else:
        pc = b.as_pytree_callable()
        outer = jax.jit(lambda f, th: f(th))
        jb = lambda th: outer(pc, th)                       # noqa: E731

    def ll(theta):
        vals = S.absolute(theta)
        return float(jb(jnp.asarray([vals[l] for l in labels], dtype=jnp.float64)))

    Z = np.load(S.A13 / "results" / "a13core_dynesty_n200_s1.npz")
    w = np.exp(Z["logwt"] - Z["logwt"].max()); w /= w.sum()
    idx = np.random.default_rng(7).choice(len(w), size=args.n + 5, replace=False, p=w)
    secs, d = [], []
    t1 = time.time()
    first = ll(Z["dead"][idx[0]])
    compile_s = time.time() - t1
    for k, i in enumerate(idx[1:]):
        t1 = time.time()
        v = ll(Z["dead"][i])
        if k >= 4:
            secs.append(time.time() - t1)
            d.append(v - float(Z["logl"][i]))
    d = np.array(d)
    try:
        mem = {k: int(v) for k, v in jax.devices()[0].memory_stats().items()
               if k in ("peak_bytes_in_use", "bytes_limit")}
    except Exception as e:  # noqa: BLE001
        mem = {"error": str(e)}
    out = {"core": S.CORE, "switch": args.switch, "call": call, "catalog_evaluation": settings,
           "kernel_pin": getattr(b.analysis.parameters, "kernel_pin", None),
           "kernel_pin_bound": b.kernel_pin is not None,
           "device": str(jax.devices()[0].device_kind), "build_seconds": build_s,
           "first_call_seconds": compile_s, "n_timed": len(secs),
           "seconds_per_call_median": float(np.median(secs)),
           "seconds_per_call_p10_p90": [float(np.percentile(secs, 10)), float(np.percentile(secs, 90))],
           "dlogL_vs_seed1": {"max_abs": float(np.max(np.abs(d))), "mean": float(d.mean()),
                              "sd": float(d.std()), "n_bitwise": int(np.sum(d == 0.0))},
           "device_memory": mem, "slurm_job_id": os.environ.get("SLURM_JOB_ID")}
    path = S.A13 / "diagnostics" / (f"a13_core_timing_{S.CORE}_{args.switch}"
                                    + ("" if call == "wrapped" else f"_{call}") + ".json")
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1)); print(f"wrote {path}")


if __name__ == "__main__":
    main()
