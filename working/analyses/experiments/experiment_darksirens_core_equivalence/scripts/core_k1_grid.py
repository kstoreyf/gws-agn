#!/usr/bin/env python
"""darksirens-core arm: evaluate the shared cells with core's public API.

Run with the core venv's python (core is installed there; nothing else is):

    /hildafs/projects/phy230014p/magana/envs/darksirens-core-f527b94/bin/python \\
        core_k1_grid.py --mode legacy_arith

Modes
    legacy_arith  kernel_pin="off" and DARKSIRENS_GW_PAIRING_SCALE=node_max: the two
                  deliberate core changes switched back to the frozen legacy
                  arithmetic. Expected to equal legacy c042527 to ~1e-12 relative.
    default       core defaults (kernel_pin="auto", pairing_scale "analytic").
    fast          defaults plus the opt-in speed-ups that apply to this path:
                  compute_dtype="float32", kernel_layout="galaxy_list",
                  missing_density="gather".
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cells as C  # noqa: E402

MODES = {
    "legacy_arith": {"env": {"DARKSIRENS_GW_PAIRING_SCALE": "node_max"},
                     "kernel_pin": "off", "compute_dtype": None},
    "default": {"env": {}, "kernel_pin": "auto", "compute_dtype": None},
    "fast": {"env": {"DARKSIRENS_CATALOG_KERNEL_LAYOUT": "galaxy_list",
                     "DARKSIRENS_CATALOG_MISSING_DENSITY": "gather"},
             "kernel_pin": "auto", "compute_dtype": "float32"},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=list(MODES), required=True)
    ap.add_argument("--max_cells", type=int, default=None)
    ap.add_argument("--gamma", type=float, default=None,
                    help="pin the merger-rate slope (core's legacy fiducial is 2.5; the mock has 0)")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    m = MODES[args.mode]
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.update(m["env"])                      # read at darksirens import
    import darksirens as ds
    from darksirens.runtime_binding import bind_analysis
    import jax
    import jax.numpy as jnp
    if "darksirens-core" not in ds.__file__:
        sys.exit(f"[fatal] darksirens imported from {ds.__file__}, not core")
    t0 = time.time()
    cat = ds.load_catalog(C.SURVEY)
    # the resolved legacy fiducial vector, by label (core's is c042527's: gamma = 2.5)
    probe = ds.model(cosmology=ds.Cosmology(H0=(50.0, 100.0), Om0=C.OM0),
                     population=ds.Population("powerlaw+peak", fixed=True)).parameters
    fv = dict(zip(probe.population_labels, probe.fixed_population))
    gkeys = [k for k in fv if "gamma" in k.lower()]
    if len(gkeys) != 1:
        sys.exit(f"[fatal] cannot find the single gamma entry in {list(fv)}")
    gamma_fid = float(fv[gkeys[0]])
    pop = ds.Population("powerlaw+peak", fixed=True)
    if args.gamma is not None:
        fv[gkeys[0]] = float(args.gamma)
        pop = ds.Population("powerlaw+peak", fixed=fv)
    an = ds.model(cosmology=ds.Cosmology(H0=(50.0, 100.0), Om0=C.OM0),
                  population=pop,
                  catalog=cat, completeness=None, fixed_survey=dict(C.FIXED_SURVEY),
                  kernel_pin=m["kernel_pin"])
    kw = {}
    if m["compute_dtype"]:
        kw["compute_dtype"] = m["compute_dtype"]
    b = bind_analysis(an, events=ds.load_events(C.EVENTS),
                      injections=ds.load_injections(C.INJECTIONS),
                      max_likelihood_variance=C.MAX_LIKELIHOOD_VARIANCE,
                      sel_batch_size=C.SEL_BATCH_SIZE, pe_event_block=C.PE_EVENT_BLOCK, **kw)
    t_build = time.time() - t0
    labels = list(b.labels)
    if sorted(labels) != ["H0", "log10n0"]:
        sys.exit(f"[fatal] core labels {labels}, expected H0 and log10n0")
    jb = jax.jit(b.__call__)
    rows = []
    for k, c in enumerate(C.cells()[: args.max_cells]):
        theta = jnp.asarray([c[l] for l in labels], dtype=jnp.float64)
        t0 = time.time()
        v = float(jb(theta))
        dt = time.time() - t0
        rows.append({**c, "logL": v, "logL_hex": float(v).hex(), "seconds": dt})
        print(f"[core_{args.mode}{args.tag}] {k:3d} {c['grid']} H0 {c['H0']:6.2f} n0 {c['log10n0']:5.2f} "
              f"logL {v:.10f} ({dt:.2f}s)", flush=True)
    plan = an.parameters
    out = {"arm": f"core_{args.mode}{args.tag}", "darksirens_file": ds.__file__, "labels": labels,
           "gamma_fiducial_of_commit": gamma_fid, "gamma_used": float(fv[gkeys[0]]),
           "mode": args.mode, "env": m["env"], "kernel_pin": m["kernel_pin"],
           "kernel_pin_active": getattr(plan, "kernel_pin_active", None),
           "compute_dtype": m["compute_dtype"] or "float64", "jax": jax.__version__,
           "devices": [str(d) for d in jax.devices()], "build_seconds": t_build,
           "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "rows": rows}
    dst = HERE.parent / "results" / f"core_{args.mode}{args.tag}.json"
    dst.write_text(json.dumps(out, indent=1))
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
