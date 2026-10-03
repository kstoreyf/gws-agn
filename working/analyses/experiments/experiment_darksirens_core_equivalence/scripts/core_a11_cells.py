#!/usr/bin/env python
"""Analysis-11 likelihood on darksirens-core (pinned b47e41c) at the window-check cells.

    /hildafs/projects/phy230014p/magana/envs/darksirens-core-b47e41c/bin/python core_a11_cells.py

Core equivalent of a11_likelihood.build_a11 at af896ca (darksirens-work, 2026-10-02):
K = 2 complete GAL + AGN catalogues, field sky weighting, completeness=None with
log10n0 = log10n0_c2 = -24 (allow_out_of_prior), delta = sigma_kde = 0 for both, Om0 0.3075,
per-catalogue population blocks for G.mu and mu_chi, the rest of powerlaw+peak pinned at the
legacy fiducial with gamma = 0, hard guard 1e6, sel_batch_size 5000, pe_event_block 5.
Core's tolerance-controlled KDE window is set EXPLICITLY (kernel_window = 1e-10); every
other catalog-evaluation setting is recorded as found, because core is about to change
three defaults. Cells are given as (base, offset); core takes absolute _c2 values.
"""
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from legacy_window_check import CELLS  # noqa: E402  (same cells as the af896ca arms)

D = Path("/hildafs/projects/phy230014p/magana/gws-agn/working/data/seed100")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", action="store_true", help="the a11_grid_cells posterior grids")
    args = ap.parse_args()
    cells = CELLS
    if args.grid:
        import a11_grid_cells
        cells = a11_grid_cells.cells()
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import darksirens as ds
    from darksirens.runtime_binding import bind_analysis
    from darksirens.catalog import settings as cs
    import jax
    import jax.numpy as jnp
    tag = os.environ.get("CORE_TAG", "b47e41c")
    if f"darksirens-core-{tag}" not in ds.__file__:
        sys.exit(f"[fatal] darksirens imported from {ds.__file__}, not core {tag}")
    cs.configure_catalog_evaluation(kernel_window=1e-10, kernel_layout="padded",
                                    missing_density="grid")
    from darksirens.population.utils import configure_normalization_grids
    configure_normalization_grids(pairing_norm="per_sample", pairing_scale="analytic")
    _s = cs.catalog_evaluation_settings()
    settings = {k: str(getattr(_s, k)) for k in getattr(_s, "__dataclass_fields__", {})} or str(_s)
    t0 = time.time()
    cos = ds.Cosmology(H0=(60.0, 76.0), Om0=0.3075)
    probe = ds.model(cosmology=cos, population=ds.Population("powerlaw+peak", fixed=True)).parameters
    fv = dict(zip(probe.population_labels, probe.fixed_population))
    g = [k for k in fv if "gamma" in k.lower()]
    assert len(g) == 1
    fv[g[0]] = 0.0
    MU_G, MU_CHI = r"$\mu_{\rm G}$", r"$\mu_\chi$"
    free_pop = {k: v for k, v in fv.items() if k not in (MU_G, MU_CHI)}
    cats = [ds.load_catalog(str(D / "surveys" / f"survey_{t}_complete_ns32.h5")) for t in ("gal", "agn")]
    an = ds.model(cosmology=cos, population=ds.Population("powerlaw+peak", fixed=free_pop),
                  catalog=cats, catalog_sky_weighting="field", completeness=None,
                  fixed_survey={"log10n0": -24.0, "log10n0_c2": -24.0, "delta": 0.0,
                                "sigma_kde": 0.0, "delta_c2": 0.0, "sigma_kde_c2": 0.0},
                  allow_out_of_prior=True, per_catalog_population={2: [MU_G, MU_CHI]})
    b = bind_analysis(an, events=ds.load_events(str(D / "events" / "events_marked_dmu0p10_dmuG5.h5")),
                      injections=ds.load_injections(str(D / "injections" / "injections_targeted.h5")),
                      max_likelihood_variance=1e6, sel_batch_size=5000, pe_event_block=5)
    t_build = time.time() - t0
    labels = list(b.labels)
    print(f"[core] labels {labels}  build {t_build:.1f}s", flush=True)
    jb = jax.jit(b.__call__)
    rows = []
    for c in cells:
        absval = {"H0": c["H0"], "fcat_2": c["fcat_2"], MU_G: c["mu_G"], MU_CHI: c["mu_chi"],
                  MU_G + "_c2": c["mu_G"] + c["dmu_G"], MU_CHI + "_c2": c["mu_chi"] + c["dmu_chi"]}
        missing = [l for l in labels if l not in absval]
        if missing:
            sys.exit(f"[fatal] no value for core label(s) {missing}")
        theta = jnp.asarray([absval[l] for l in labels], dtype=jnp.float64)
        t1 = time.time()
        v = float(jb(theta))
        dt = time.time() - t1
        rows.append({**c, "logL": v, "logL_hex": float(v).hex(), "seconds": dt})
        print(f"[core] {c['name']:11s} logL {v:.10f} ({dt:.1f}s)", flush=True)
    try:
        mem = {k: int(v) for k, v in jax.devices()[0].memory_stats().items()
               if k in ("peak_bytes_in_use", "bytes_limit")}
    except Exception as e:  # noqa: BLE001
        mem = {"error": str(e)}
    import resource
    out = {"arm": f"core_{tag}_a11", "darksirens_file": ds.__file__, "labels": labels,
           "gamma_used": 0.0, "kernel_window": 1e-10, "catalog_evaluation": settings,
           "env": {k: v for k, v in os.environ.items() if k.startswith("DARKSIRENS_")},
           "build_seconds": t_build, "device_memory": mem,
           "host_maxrss_GB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6,
           "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "rows": rows}
    dst = HERE.parent / "results" / (f"core_{tag}_a11_" + ("grid" if args.grid else "cells") + ".json")
    dst.write_text(json.dumps(out, indent=2))
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
