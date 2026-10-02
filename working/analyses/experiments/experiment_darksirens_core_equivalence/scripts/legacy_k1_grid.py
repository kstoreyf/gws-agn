#!/usr/bin/env python
"""Legacy darksirens arm: evaluate the shared cells with a legacy checkout.

Run with that checkout first on PYTHONPATH, one process per arm:

    PYTHONPATH=/hildafs/projects/phy230014p/magana/src/darksirens-2b86a2d \\
        python legacy_k1_grid.py --arm legacy_2b86a2d --expect darksirens-2b86a2d

The build is gws-agn's own recipe (selection_redo/scripts/scan_h0f.py, the
driver of Analyses 3-7, which reproduces the archived per-pixel campaign): K = 1
dark_sirens universe, conditional sky weighting, c_mode per_pixel, delta and
sigma_kde pinned, Om0 pinned, hard selection guard at max_likelihood_variance
1e6, no KDE window. Keyword arguments a given commit does not accept are dropped
(and recorded), so the same script builds 2b86a2d, 0c5b3db and c042527.
"""
import argparse
import inspect
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cells as C  # noqa: E402


def _call(fn, dropped, **kw):
    sig = inspect.signature(fn)
    if any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
        return fn(**kw)
    keep = {k: v for k, v in kw.items() if k in sig.parameters}
    dropped.extend(sorted(set(kw) - set(keep)))
    return fn(**keep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--expect", required=True, help="substring darksirens.__file__ must contain")
    ap.add_argument("--max_cells", type=int, default=None)
    args = ap.parse_args()
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import darksirens
    f = darksirens.__file__
    if args.expect not in f:
        sys.exit(f"[fatal] darksirens imported from {f}, expected {args.expect}")
    from darksirens.inference.data import load_all_data
    from darksirens.likelihood.factory import make_likelihood
    from darksirens.gw.populations import get_fixed_population_params
    from darksirens.inference.prior import build_parameter_space
    import jax
    import jax.numpy as jnp

    opts = SimpleNamespace(
        universe_model="dark_sirens", survey_path=C.SURVEY, survey_paths=[C.SURVEY],
        n_catalogs=1, gw_path=C.EVENTS, gwselection_path=C.INJECTIONS, use_LSS=False,
        lss_completion=None, lss_completions=[], lss_marginalize=False, counterpart=None,
        counterpart_nside=1, counterpart_dz=1e-4, bright_siren_sky_marginalized=False,
        drop_full_catalog=False, sky_model="isotropic", mark_model="none", marks=None,
        mark_names=(), sel_batch_size=C.SEL_BATCH_SIZE, pe_event_block=C.PE_EVENT_BLOCK,
        redshift_prior_barrier="auto", selection_neff_guard="hard",
        selection_neff_soft_guard=False, max_likelihood_variance=C.MAX_LIKELIHOOD_VARIANCE,
        sampler="dynesty", fix_population=True, fix_cosmology=False, fix_de=True,
        fix_survey=False, pop_model="powerlaw+peak", shared_beta=True, shared_spin=True,
        shared_gamma=True, complete_empty_pixel_policy="zero",
        catalog_sky_weighting="conditional", c_mode="per_pixel", selection_prior=None,
        selection_fit=None)
    fixed = {"Om0": C.OM0}
    dropped = []
    t0 = time.time()
    data = load_all_data(opts)
    t_load = time.time() - t0
    res = _call(build_parameter_space, dropped, pop_model=opts.pop_model,
                fix_population=True, fix_cosmology=False, fix_survey=False, fix_de=True,
                prior_overrides={}, fixed_parameter_values=fixed,
                universe_model="dark_sirens", shared_beta=True, shared_spin=True,
                shared_gamma=True, sky_model="isotropic", mark_model="none", mark_names=(),
                n_catalogs=1, lss_completion_active=[False], use_lss=False,
                mark_names_by_catalog=None, c_mode="per_pixel")
    labels = list(res[0])
    pop = get_fixed_population_params("powerlaw+peak", shared_beta=True, shared_spin=True,
                                      shared_gamma=True)
    t0 = time.time()
    like = make_likelihood(opts=opts, data=data, pop_params_fid=pop,
                           fixed_parameter_values=fixed)
    t_build = time.time() - t0
    point = dict(C.FIXED_SURVEY)
    unknown = [l for l in labels if l not in ("H0", "log10n0") and l not in point]
    if unknown:
        sys.exit(f"[fatal] labels without a pinned value: {unknown} (labels {labels})")
    jl = jax.jit(like)
    rows = []
    for k, c in enumerate(C.cells()[: args.max_cells]):
        coord = jnp.asarray([{"H0": c["H0"], "log10n0": c["log10n0"]}.get(l, point.get(l))
                             for l in labels], dtype=jnp.float64)
        t0 = time.time()
        v = float(jl(coord))
        dt = time.time() - t0
        rows.append({**c, "logL": v, "logL_hex": float(v).hex(), "seconds": dt})
        print(f"[{args.arm}] {k:3d} {c['grid']} H0 {c['H0']:6.2f} n0 {c['log10n0']:5.2f} "
              f"logL {v:.10f} ({dt:.2f}s)", flush=True)
    out = {"arm": args.arm, "darksirens_file": f, "labels": labels,
           "dropped_kwargs": sorted(set(dropped)), "jax": jax.__version__,
           "devices": [str(d) for d in jax.devices()], "load_seconds": t_load,
           "build_seconds": t_build, "nEvents": data.get("nEvents"),
           "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "rows": rows}
    dst = HERE.parent / "results" / f"{args.arm}.json"
    dst.write_text(json.dumps(out, indent=1))
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
