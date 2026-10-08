#!/usr/bin/env python
"""Analysis 13 on darksirens-core (pinned bf58aa6): dynesty over the joint shared-width model.

    /hildafs/projects/phy230014p/magana/envs/darksirens-core-bf58aa6/bin/python \\
        a13_core_sampler.py --seed 1 [--nlive 200]

Owner decision 2026-10-02: restart A13 on core (validated for the A11 likelihood at the
posterior level, 12x faster; ../experiments/experiment_darksirens_core_equivalence/).

Likelihood (darksirens-work's core equivalent of a11_likelihood.build_a11, plus both widths):
K = 2 complete GAL + AGN catalogues, field weighting, completeness=None at log10n0 =
log10n0_c2 = -24 (allow_out_of_prior), delta = sigma_kde = 0, Om0 0.3075,
per_catalog_population={2: [mu_G, mu_chi]}; mu_G, mu_chi, sigma_G, sigma_chi sampled, the rest
of powerlaw+peak pinned at the legacy fiducial with gamma = 0; hard guard 1e6, sel_batch 5000,
pe_event_block 5. Catalogue evaluation set EXPLICITLY: kernel_window 1e-10, kernel_layout
padded, missing_density grid, pairing_norm per_sample, pairing_scale analytic (core is changing
these defaults in #52).

Sampler: identical to Analysis 11's dynesty runs. Coordinates and flat boxes are problem 13's
(a11_sampler.BOXES["13"]): H0 [60, 76], f_agn [0, 1], mu_G [31, 39], dmu_G [-4, 10],
mu_chi [-0.1, 0.1], dmu_chi [-0.05, 0.30], sigma_G [1, 10], sigma_chi [0.01, 1]; core takes the
AGN copies as absolute values (mu + dmu; unit Jacobian). dynesty bound multi, sample unif, first
bound after 2 nlive calls, dlogz 0.1, -inf -> -1e300, checkpoint every 900 s, resumes.

Pre-flight: at sigma_G = 5, sigma_chi = 0.1 the likelihood must reproduce the core A11 cells
already measured on this commit (results/core_bf58aa6_a11_cells.json) to 1e-8, or the run stops.

Options added 2026-10-06/07 (the defaults reproduce seed 1's run and file names exactly):
  A13_CORE (env, default bf58aa6)  the core commit; the darksirens import must come from
                                   src/darksirens-core-<sha> (envs/darksirens-core-<sha>/bin/python)
  --sample rslice                  rslice from the start (slices 3 + ndim, enlarge 1.25, no bootstrap)
  --max-var X                      max_likelihood_variance for SAMPLING (default 1e6 = only the
                                   5 N_obs floor); the pre-flight always runs at 1e6. Owner 2026-10-07:
                                   20 removes the seed-1 edge clump (diagnostics/a13_edge_check.json)
  --settings defaults              core's #52 speed defaults instead of the historical evaluation
                                   (the pre-flight then compares within 1e-6, not 1e-8)
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

A13 = Path(__file__).resolve().parent.parent
EXP = A13.parent / "experiments" / "experiment_darksirens_core_equivalence"
D = Path("/hildafs/projects/phy230014p/magana/gws-agn/working/data/seed100")
# off-hilda runs (scripts/js2/): point at the copied seed-100 inputs and pre-flight reference
D = Path(os.environ.get("A13_DATA", D))
REF_CELLS = Path(os.environ.get("A13_REF_CELLS", EXP / "results" / "core_bf58aa6_a11_cells.json"))
BOX = [("H0", 60.0, 76.0), ("f_agn", 0.0, 1.0), ("mu_G", 31.0, 39.0), ("dmu_G", -4.0, 10.0),
       ("mu_chi", -0.10, 0.10), ("dmu_chi", -0.05, 0.30), ("sigma_G", 1.0, 10.0),
       ("sigma_chi", 0.01, 1.0)]
CORE = os.environ.get("A13_CORE", "bf58aa6")
MU_G, MU_CHI = r"$\mu_{\rm G}$", r"$\mu_\chi$"
SIG_G, SIG_CHI = r"$\sigma_{\rm G}$", r"$\sigma_\chi$"


def summarise(s, names):
    out = {}
    for i, n in enumerate(names):
        x = s[:, i]
        q = np.quantile(x, [0.05, 0.16, 0.5, 0.84, 0.95])
        out[n] = {"median": float(q[2]), "ci68": [float(q[1]), float(q[3])],
                  "ci90": [float(q[0]), float(q[4])], "mean": float(x.mean()), "sd": float(x.std())}
    c = np.corrcoef(s.T)
    out["correlations"] = {f"{names[i]}|{names[j]}": float(c[i, j])
                           for i in range(len(names)) for j in range(i + 1, len(names))}
    return out


def build(settings_mode="historical"):
    import darksirens as ds
    from darksirens.runtime_binding import bind_analysis
    from darksirens.catalog import settings as cs
    if f"darksirens-core-{CORE}" not in ds.__file__:
        sys.exit(f"[fatal] darksirens imported from {ds.__file__}, not core {CORE}")
    from darksirens.population.utils import configure_normalization_grids
    if settings_mode == "historical":
        # layout-only opt-ins for small GPUs (the 20 GB js2a100 vGPU): same arithmetic per galaxy,
        # held to the rita reference cells by the pre-flight below
        cs.configure_catalog_evaluation(kernel_window=1e-10,
                                        kernel_layout=os.environ.get("A13_KERNEL_LAYOUT", "padded"),
                                        missing_density=os.environ.get("A13_MISSING_DENSITY", "grid"))
        # The population pairing normaliser, explicit too: core's default moves from
        # per_sample to per_point (darksirens-core #52, up to ~1e-7 in logL). These are
        # the settings the A11 validation ran with.
        ng = configure_normalization_grids(pairing_norm="per_sample", pairing_scale="analytic")
    elif settings_mode == "defaults":
        if CORE == "bf58aa6":
            sys.exit("[fatal] --settings defaults needs a core with #52 (e7c3007 or later)")
        ng = configure_normalization_grids()
    else:
        sys.exit(f"[fatal] unknown settings mode {settings_mode}")
    st = cs.catalog_evaluation_settings()
    settings = {k: str(getattr(st, k)) for k in st.__dataclass_fields__}
    settings.update({"pairing_norm": str(getattr(ng, "pairing_norm", "per_sample")),
                     "pairing_scale": str(getattr(ng, "pairing_scale", "analytic"))})
    cos = ds.Cosmology(H0=(60.0, 76.0), Om0=0.3075)
    probe = ds.model(cosmology=cos, population=ds.Population("powerlaw+peak", fixed=True)).parameters
    fv = dict(zip(probe.population_labels, probe.fixed_population))
    g = [k for k in fv if "gamma" in k.lower()]
    assert len(g) == 1
    fv[g[0]] = 0.0
    fixed = {k: v for k, v in fv.items() if k not in (MU_G, MU_CHI, SIG_G, SIG_CHI)}
    cats = [ds.load_catalog(str(D / "surveys" / f"survey_{t}_complete_ns32.h5")) for t in ("gal", "agn")]
    an = ds.model(cosmology=cos, population=ds.Population("powerlaw+peak", fixed=fixed),
                  catalog=cats, catalog_sky_weighting="field", completeness=None,
                  fixed_survey={"log10n0": -24.0, "log10n0_c2": -24.0, "delta": 0.0,
                                "sigma_kde": 0.0, "delta_c2": 0.0, "sigma_kde_c2": 0.0},
                  allow_out_of_prior=True, per_catalog_population={2: [MU_G, MU_CHI]})
    b = bind_analysis(an, events=ds.load_events(str(D / "events" / "events_marked_dmu0p10_dmuG5.h5")),
                      injections=ds.load_injections(str(D / "injections" / "injections_targeted.h5")),
                      max_likelihood_variance=1e6, sel_batch_size=5000, pe_event_block=5)
    return ds, b, settings


def absolute(theta):
    """Sampler coordinates (BOX order) -> core labels (absolute AGN copies)."""
    H0, f, mG, dG, mC, dC, sG, sC = theta
    return {"H0": H0, "fcat_2": f, MU_G: mG, MU_G + "_c2": mG + dG, MU_CHI: mC,
            MU_CHI + "_c2": mC + dC, SIG_G: sG, SIG_CHI: sC}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--nlive", type=int, default=200)
    ap.add_argument("--dlogz", type=float, default=0.1)
    ap.add_argument("--sample", choices=["unif", "rslice"], default="unif")
    ap.add_argument("--max-var", type=float, default=1e6)
    ap.add_argument("--settings", choices=["historical", "defaults"], default="historical")
    ap.add_argument("--resume-sample", choices=["rslice"], default=None,
                    help="on resume, switch the proposal (owner 2026-10-05: rslice after the unif stalls)")
    args = ap.parse_args()
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import jax
    import jax.numpy as jnp
    import dynesty
    t0 = time.time()
    ds, b, settings = build(args.settings)
    labels = list(b.labels)
    names = [n for n, _, _ in BOX]
    print(f"[a13core] labels {labels}  build {time.time() - t0:.1f}s", flush=True)
    jb_now = [jax.jit(b.__call__)]

    def ll_abs(vals):
        missing = [l for l in labels if l not in vals]
        if missing:
            sys.exit(f"[fatal] no value for core label(s) {missing}")
        return float(jb_now[0](jnp.asarray([vals[l] for l in labels], dtype=jnp.float64)))

    # pre-flight against the validated core A11 cells (widths at their fiducials)
    ref = json.loads(REF_CELLS.read_text())
    pre = []
    for r in ref["rows"]:
        v = ll_abs(absolute([r["H0"], r["fcat_2"], r["mu_G"], r["dmu_G"], r["mu_chi"], r["dmu_chi"],
                             5.0, 0.1]))
        pre.append({"cell": r["name"], "a13core": v, "core_a11": r["logL"], "abs_diff": abs(v - r["logL"])})
        print(f"[preflight] {r['name']:11s} {v:.10f} vs {r['logL']:.10f}  |d| {abs(v - r['logL']):.2e}", flush=True)
    if max(p["abs_diff"] for p in pre) > (1e-8 if args.settings == "historical" else 1e-6):
        sys.exit("[fatal] pre-flight: the joint-width likelihood does not reproduce the core A11 cells")

    if args.max_var != b.max_likelihood_variance:
        import dataclasses
        b = dataclasses.replace(b, max_likelihood_variance=args.max_var)
        jb_now[0] = jax.jit(b.__call__)
    print(f"[a13core] sampling with max_likelihood_variance {b.max_likelihood_variance:g}", flush=True)
    lo = np.array([a for _, a, _ in BOX]); hi = np.array([c for _, _, c in BOX])
    cnt = {"n": 0, "t": 0.0, "neginf": 0}

    def loglike(theta):
        t1 = time.time()
        v = ll_abs(absolute(theta))
        cnt["n"] += 1; cnt["t"] += time.time() - t1
        if not np.isfinite(v):
            cnt["neginf"] += 1
            return -1e300
        return v

    def ptform(u):
        return lo + u * (hi - lo)

    # checkpoints store sampler state only; loglike/ptform are closures and are rebound on restore
    from darksirens.inference.dynesty_checkpoint import (install_dynesty_checkpointing,
                                                         restore_dynesty_sampler)
    tag = f"a13core_dynesty_n{args.nlive}_s{args.seed}"
    if (CORE, args.sample, args.max_var, args.settings) != ("bf58aa6", "unif", 1e6, "historical"):
        tag = (f"a13core_{CORE}_{args.sample}_cap{args.max_var:g}_{args.settings}"
               f"_n{args.nlive}_s{args.seed}")
    ckpt = A13 / "queue" / f"{tag}.save"
    switch_log = A13 / "queue" / f"{tag}.sampler_switch.json"
    (A13 / "queue").mkdir(exist_ok=True); (A13 / "results").mkdir(exist_ok=True)
    t1 = time.time()
    if ckpt.exists():
        print(f"[a13core] resuming {ckpt}", flush=True)
        s = restore_dynesty_sampler(str(ckpt), loglike, ptform)
        install_dynesty_checkpointing(s)
        if args.resume_sample:
            from a13_switch import switch_sampler
            rec = switch_sampler(s, args.resume_sample)
            if rec is not None:
                rec["slurm_job_id"] = os.environ.get("SLURM_JOB_ID")
                switch_log.write_text(json.dumps(rec, indent=2))
            print(f"[a13core] proposal {s.method} (switch {rec})", flush=True)
        s.run_nested(dlogz=args.dlogz, print_progress=True, resume=True,
                     checkpoint_file=str(ckpt), checkpoint_every=900)
    else:
        s = dynesty.NestedSampler(loglike, ptform, len(BOX), nlive=args.nlive, bound="multi",
                                  sample=args.sample, rstate=np.random.default_rng(args.seed),
                                  first_update={"min_ncall": 2 * args.nlive, "min_eff": 100.0})
        install_dynesty_checkpointing(s)
        s.run_nested(dlogz=args.dlogz, print_progress=True, checkpoint_file=str(ckpt),
                     checkpoint_every=900)
    res = s.results
    logwt = np.asarray(res["logwt"])
    w = np.exp(logwt - logwt.max()); w /= w.sum()
    from dynesty.utils import resample_equal
    samples = resample_equal(np.asarray(res.samples), w, rstate=np.random.default_rng(args.seed + 7919))
    try:
        mem = {k: int(v) for k, v in jax.devices()[0].memory_stats().items()
               if k in ("peak_bytes_in_use", "bytes_limit")}
    except Exception as e:  # noqa: BLE001
        mem = {"error": str(e)}
    import pickle
    with open(A13 / "queue" / f"{tag}.results.pkl", "wb") as fh:
        pickle.dump(res, fh)
    out = {"tag": tag, "problem": "13", "engine": "dynesty", "code": f"darksirens-core {CORE}",
           "sample": args.sample, "max_likelihood_variance": args.max_var, "settings_mode": args.settings,
           "darksirens_file": ds.__file__, "catalog_evaluation": settings, "gamma_used": 0.0,
           "nlive": args.nlive, "seed": args.seed, "names": names, "box": BOX,
           "core_labels": labels, "preflight": pre, "logZ": float(res.logz[-1]),
           "logZerr": float(res.logzerr[-1]), "ncall_total": int(np.sum(res.ncall)),
           "niter": int(res.niter), "eff_percent": float(res.eff),
           "n_loglike_calls_this_session": cnt["n"], "n_neginf": cnt["neginf"],
           "mean_seconds_per_call": cnt["t"] / max(cnt["n"], 1),
           "wall_seconds_sampling": time.time() - t1, "device_memory": mem,
           "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
           "n_equal_weight_samples": int(samples.shape[0]), "summary": summarise(samples, names),
           "sampler_switch": json.loads(switch_log.read_text()) if switch_log.exists() else None,
           "finished": True}
    np.savez(A13 / "results" / f"{tag}.npz", samples=samples, names=np.array(names), logwt=logwt,
             logl=np.asarray(res.logl), dead=np.asarray(res.samples))
    (A13 / "results" / f"{tag}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out["summary"], indent=1))


if __name__ == "__main__":
    main()
