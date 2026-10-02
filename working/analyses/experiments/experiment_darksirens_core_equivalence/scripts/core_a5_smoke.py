#!/usr/bin/env python
"""A5 free-anchor dynesty smoke, darksirens-core arm (core main 661ef3d).

    /hildafs/projects/phy230014p/magana/envs/darksirens-core-661ef3d/bin/python core_a5_smoke.py

The same problem and sampler as the legacy arm (`a5_smoke_legacy.sbatch`, which runs
selection_redo/scripts/sample_4d.py at 0c5b3db), with only the likelihood swapped:

- K = 2, GAL + AGN at m < 18, field sky weighting, completeness="selection" with the
  same true-z Schechter fits, whose (Mstar_hat, alpha) legacy pins at the fit centres
  (core fixes selection nuisances unless survey_priors names them);
- free (H0, log10n0, log10n0_c2, fcat_2) on H0 [50, 100], log10n0 [-4, -1],
  log10n0_c2 [-6, -4], fcat_2 [0, 1] (Beta(1, 1) in core at K = 2: flat);
- delta, sigma_kde and the _c2 copies at 0; Om0 0.3075; population fixed at the legacy
  fiducial with the merger-rate slope pinned to the mock's gamma = 0 (core's is 2.5);
- hard guard at 1e6, sel_batch_size 50000, pe_event_block 25;
- dynesty.NestedSampler(loglike, ptform, ndim=4, nlive=1000, rstate=default_rng(7)),
  run_nested(dlogz=10, maxcall=500000), -inf mapped to -1e300 as sample_4d does.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "results" / "a5_smoke"
W = Path("/hildafs/projects/phy230014p/magana/gws-agn/working")
D = W / "data" / "seed100"
FITS = W / "analyses" / "selection_redo" / "scripts" / "selection_fits_truez"
BOX = {"H0": (50.0, 100.0), "log10n0": (-4.0, -1.0), "log10n0_c2": (-6.0, -4.0),
       "fcat_2": (0.0, 1.0)}
ORDER = ("H0", "log10n0", "log10n0_c2", "fcat_2")      # sample_4d's theta order


def schechter(path):
    from darksirens.selection.catalog import SchechterMagnitudeSelection
    s = json.loads(Path(path).read_text())["strata"][0]
    if s["family"] != "schechter":
        sys.exit(f"[fatal] {path}: family {s['family']}")
    return SchechterMagnitudeSelection(m_lim=float(s["m_lim"]), Mstar_hat=float(s["Mstar_hat"]),
                                       alpha=float(s["alpha"]),
                                       M_faint_offset=float(s["M_faint_offset"]))


def main():
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import darksirens as ds
    from darksirens.runtime_binding import bind_analysis
    import dynesty
    import jax
    import jax.numpy as jnp
    if "darksirens-core-661ef3d" not in ds.__file__:
        sys.exit(f"[fatal] darksirens imported from {ds.__file__}, not core 661ef3d")
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    cos = ds.Cosmology(H0=BOX["H0"], Om0=0.3075)
    probe = ds.model(cosmology=cos, population=ds.Population("powerlaw+peak", fixed=True)).parameters
    fv = dict(zip(probe.population_labels, probe.fixed_population))
    g = [k for k in fv if "gamma" in k.lower()]
    assert len(g) == 1
    fv[g[0]] = 0.0
    cats = [ds.load_catalog(str(D / "surveys" / f"survey_{t}_m18_ns32.h5")) for t in ("gal", "agn")]
    sel = [schechter(FITS / f"fit_{t}_m18_schechter.json") for t in ("gal", "agn")]
    an = ds.model(cosmology=cos, population=ds.Population("powerlaw+peak", fixed=fv),
                  catalog=cats, catalog_sky_weighting="field", completeness="selection",
                  selection=sel, field_normalizer="auto",
                  survey_priors={"log10n0": BOX["log10n0"], "log10n0_c2": BOX["log10n0_c2"]},
                  fixed_survey={"delta": 0.0, "sigma_kde": 0.0, "delta_c2": 0.0,
                                "sigma_kde_c2": 0.0})
    b = bind_analysis(an, events=ds.load_events(str(D / "events" / "events.h5")),
                      injections=ds.load_injections(str(D / "injections" / "injections_targeted.h5")),
                      max_likelihood_variance=1e6, sel_batch_size=50000, pe_event_block=25)
    t_build = time.time() - t0
    labels = list(b.labels)
    if sorted(labels) != sorted(ORDER):
        sys.exit(f"[fatal] core labels {labels}, expected {ORDER}")
    perm = np.array([ORDER.index(l) for l in labels])     # core position <- theta index
    jb = jax.jit(b.__call__)
    lo = np.array([BOX[k][0] for k in ORDER]); hi = np.array([BOX[k][1] for k in ORDER])
    cnt = {"n": 0, "t": 0.0, "neginf": 0}

    def loglike(theta):
        t1 = time.time()
        v = float(jb(jnp.asarray(np.asarray(theta)[perm], dtype=jnp.float64)))
        cnt["n"] += 1; cnt["t"] += time.time() - t1
        if not np.isfinite(v):
            cnt["neginf"] += 1
            return -1e300
        return v

    def ptform(u):
        return lo + u * (hi - lo)

    # timing at the archived A5 m<18 median before sampling
    t1 = time.time(); v0 = loglike(np.array([69.2, -3.0, -5.0, 0.3])); t_first = time.time() - t1
    rstate = np.random.default_rng(7)
    s = dynesty.NestedSampler(loglike, ptform, ndim=4, nlive=1000, rstate=rstate)
    t1 = time.time()
    s.run_nested(dlogz=10.0, maxcall=500000, print_progress=True)
    res = s.results
    eq = res.samples_equal(rstate=rstate)
    try:
        peak = {k: int(v) for k, v in jax.devices()[0].memory_stats().items()
                if k in ("peak_bytes_in_use", "bytes_limit")}
    except Exception as e:  # noqa: BLE001
        peak = {"error": str(e)}
    import resource
    out = {"arm": "core_661ef3d", "darksirens_file": ds.__file__, "labels_core_order": labels,
           "theta_order": ORDER, "box": BOX, "gamma_used": 0.0, "nlive": 1000, "dlogz_target": 10.0,
           "rstate_seed": 7, "logz": float(res.logz[-1]), "logzerr": float(res.logzerr[-1]),
           "niter": int(res.niter), "ncall": int(np.sum(res.ncall)), "n_loglike_calls": cnt["n"],
           "n_neginf": cnt["neginf"], "mean_seconds_per_call": cnt["t"] / max(cnt["n"], 1),
           "first_call_seconds_incl_compile": t_first, "logL_at_probe": v0,
           "build_seconds": t_build, "sampling_seconds": time.time() - t1,
           "device_memory": peak, "host_maxrss_GB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6,
           "slurm_job_id": os.environ.get("SLURM_JOB_ID")}
    np.savez(OUT / "a5smoke_core_661ef3d_m18.npz", samples=eq, theta_order=np.array(ORDER),
             logwt=np.asarray(res.logwt), logl=np.asarray(res.logl), dead=np.asarray(res.samples))
    (OUT / "a5smoke_core_661ef3d_m18.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
