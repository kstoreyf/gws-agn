#!/usr/bin/env python3
"""Bisect metric for the dscf-vs-dsf convergence between 8eae3ea and 2b86a2d.

The #212-era run disclosed a "kernel-normalization micro-convention" gap between
the two complete-catalog field constructions: `dark_sirens_complete` field (dscf)
recovered f low by 0.07-0.11 relative to `dark_sirens` field at the
complete-catalog n0 limit (dsf), which matched gw_agn.  On master @ 2b86a2d the
gap is gone (both recover 0.3221 / 0.6872 on the identical inputs).

This script reduces that to ONE scalar evaluated at a fixed coordinate,

    M = logL_dscf(f=0.307) - logL_dsf(f=0.307)      [H0=67.74, n0=n0_c2=-12]

so `git bisect run` can find the commit that closed it.  Exit status is the
bisect verdict, chosen by --threshold:

    |M| >  threshold  -> exit 0  ("good" = still the OLD, gapped behaviour)
    |M| <= threshold  -> exit 1  ("bad"  = the NEW, converged behaviour)

Commits before PR #308 do not accept build_parameter_space's `use_lss` /
`lss_completion_active` / `mark_names_by_catalog` kwargs, so the call is retried
without them (TypeError) — the older signature's defaults reproduce the same
parameter space for this configuration.  An evaluation that cannot run at all
exits 125, which git bisect treats as "skip".
"""
import argparse
import os
import sys
from types import SimpleNamespace

import numpy as np

OM0_FID = 0.3075
NUISANCE = {"delta": 0.0, "b_miss": 1.0, "sigma_kde": 0.0,
            "delta_c2": 0.0, "b_miss_c2": 1.0, "sigma_kde_c2": 0.0}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--f_at", type=float, default=0.307)
    ap.add_argument("--h0_at", type=float, default=67.74)
    ap.add_argument("--threshold", type=float, default=1.0)
    ap.add_argument("--device", choices=["gpu", "cpu"], default="gpu")
    ap.add_argument("--print_only", action="store_true",
                    help="Print M and exit 0 regardless (for endpoint calibration).")
    ap.add_argument("--k1_curve", nargs=2, metavar=("SURVEY", "GW"), default=None,
                    help="Instead of the bisect metric, dump K=1 H0 logL curves "
                         "(both universe models) for this survey/gw pair to "
                         "--out_json. Used to test whether the per-tracer H0 "
                         "landscape moved between revisions.")
    ap.add_argument("--out_json", default=None)
    return ap.parse_args(argv)


def logl_k1_h0_curve(universe_model, args, n0, survey, gw):
    """K=1 logL over an H0 grid — used to check whether the per-tracer H0
    landscape (not just the K=2 mixture) moved between two darksirens revisions.
    Same import recipe and the same TypeError fallback as `logl`."""
    from darksirens.inference.data import load_all_data, validate_loaded_survey_shapes
    from darksirens.likelihood.factory import make_likelihood
    from darksirens.gw.populations import get_fixed_population_params
    from darksirens.inference.prior import build_parameter_space

    opts = SimpleNamespace(
        universe_model=universe_model,
        survey_path=survey, survey_paths=[survey], n_catalogs=1,
        gw_path=gw, gwselection_path=f"{args.data_dir}/injections_cat.h5",
        use_LSS=False, lss_completion=None, lss_completions=[], lss_marginalize=False,
        counterpart=None, counterpart_nside=1, counterpart_dz=1e-4,
        bright_siren_sky_marginalized=False, drop_full_catalog=False,
        sky_model="isotropic", mark_model="none", marks=None, mark_names=(),
        sel_batch_size=None, redshift_prior_barrier="auto",
        selection_neff_guard="auto", selection_neff_soft_guard=False,
        sampler="tinyns", fix_population=True, fix_cosmology=False, fix_de=True,
        fix_survey=False, pop_model="powerlaw+peak", shared_beta=True,
        shared_spin=True, shared_gamma=True, complete_empty_pixel_policy="zero",
        catalog_sky_weighting="field", max_likelihood_variance=1e6,
    )
    fixed = {"Om0": OM0_FID}
    data = load_all_data(opts)
    validate_loaded_survey_shapes(data)
    kw = dict(fix_de=True, prior_overrides={}, fixed_parameter_values=fixed,
              universe_model=universe_model, shared_beta=True, shared_spin=True,
              shared_gamma=True, sky_model="isotropic", mark_model="none",
              mark_names=(), n_catalogs=1)
    try:
        res = build_parameter_space("powerlaw+peak", True, False, False,
                                    lss_completion_active=[False], use_lss=False,
                                    mark_names_by_catalog=None, **kw)
    except TypeError:
        res = build_parameter_space("powerlaw+peak", True, False, False, **kw)
    labels = list(res[0])
    pop_fid = get_fixed_population_params("powerlaw+peak", shared_beta=True,
                                          shared_spin=True, shared_gamma=True)
    like = make_likelihood(opts=opts, data=data, pop_params_fid=pop_fid,
                           fixed_parameter_values=fixed)
    point = dict(NUISANCE)
    point.update({"H0": 67.74, "log10n0": n0})
    grid = np.linspace(50.0, 100.0, 61)
    out = []
    ih = labels.index("H0")
    base = np.asarray([float(point[l]) for l in labels], dtype=float)
    for h in grid:
        c = base.copy(); c[ih] = h
        out.append(float(like(c)))
    return grid, np.asarray(out), labels


def logl(universe_model, args, n0):
    from darksirens.inference.data import load_all_data, validate_loaded_survey_shapes
    from darksirens.likelihood.factory import make_likelihood
    from darksirens.gw.populations import get_fixed_population_params
    from darksirens.inference.prior import build_parameter_space

    d = args.data_dir
    survey_paths = [f"{d}/gal.h5", f"{d}/agn.h5"]
    opts = SimpleNamespace(
        universe_model=universe_model,
        survey_path=survey_paths[0], survey_paths=survey_paths, n_catalogs=2,
        gw_path=f"{d}/gw_fagn0.3.h5", gwselection_path=f"{d}/injections_cat.h5",
        use_LSS=False, lss_completion=None, lss_completions=[], lss_marginalize=False,
        counterpart=None, counterpart_nside=1, counterpart_dz=1e-4,
        bright_siren_sky_marginalized=False, drop_full_catalog=False,
        sky_model="isotropic", mark_model="none", marks=None, mark_names=(),
        sel_batch_size=None, redshift_prior_barrier="auto",
        selection_neff_guard="auto", selection_neff_soft_guard=False,
        sampler="tinyns", fix_population=True, fix_cosmology=False, fix_de=True,
        fix_survey=False, pop_model="powerlaw+peak", shared_beta=True,
        shared_spin=True, shared_gamma=True, complete_empty_pixel_policy="zero",
        catalog_sky_weighting="field",
        # Make the post-#212 total-variance criterion inert; older commits simply
        # ignore the attribute, so the comparison is the legacy Neff floor on both.
        max_likelihood_variance=1e6,
    )
    fixed = {"Om0": OM0_FID}
    data = load_all_data(opts)
    validate_loaded_survey_shapes(data)

    kw = dict(fix_de=True, prior_overrides={}, fixed_parameter_values=fixed,
              universe_model=universe_model, shared_beta=True, shared_spin=True,
              shared_gamma=True, sky_model="isotropic", mark_model="none",
              mark_names=(), n_catalogs=2)
    try:
        res = build_parameter_space(
            "powerlaw+peak", True, False, False,
            lss_completion_active=[False, False], use_lss=False,
            mark_names_by_catalog=None, **kw)
    except TypeError:
        res = build_parameter_space("powerlaw+peak", True, False, False, **kw)
    labels = list(res[0])

    pop_fid = get_fixed_population_params("powerlaw+peak", shared_beta=True,
                                          shared_spin=True, shared_gamma=True)
    like = make_likelihood(opts=opts, data=data, pop_params_fid=pop_fid,
                           fixed_parameter_values=fixed)

    point = dict(NUISANCE)
    point.update({"H0": args.h0_at, "fcat_2": args.f_at,
                  "log10n0": n0, "log10n0_c2": n0})
    missing = [l for l in labels if l not in point]
    if missing:
        raise RuntimeError(f"labels not covered: {missing}")
    coord = np.asarray([float(point[l]) for l in labels], dtype=float)
    return float(like(coord)), labels


def main(argv=None):
    args = parse_args(argv)
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["JAX_PLATFORMS"] = "cuda" if args.device == "gpu" else "cpu"

    import darksirens
    print(f"darksirens: {darksirens.__file__}", flush=True)

    if args.k1_curve:
        import json
        from pathlib import Path
        survey, gw = args.k1_curve
        res = {}
        for um, n0 in (("dark_sirens", -12.0), ("dark_sirens_complete", -12.0)):
            grid, ll, labels = logl_k1_h0_curve(um, args, n0, survey, gw)
            res[um] = {"H0_grid": grid.tolist(), "logL": ll.tolist(),
                       "labels": labels,
                       "argmax": float(grid[int(np.nanargmax(ll))])}
            print(f"{um}: argmax={res[um]['argmax']:.3f} "
                  f"logLmax={float(np.nanmax(ll)):.6f}", flush=True)
        if args.out_json:
            Path(args.out_json).write_text(json.dumps(res, indent=2))
            print(f"Wrote {args.out_json}", flush=True)
        return 0

    try:
        ll_dscf, lab_dscf = logl("dark_sirens_complete", args, -12.0)
        ll_dsf, lab_dsf = logl("dark_sirens", args, -12.0)
    except Exception as exc:                       # cannot evaluate -> bisect skip
        print(f"[skip] {type(exc).__name__}: {exc}", flush=True)
        return 125

    M = ll_dscf - ll_dsf
    print(f"labels dscf={lab_dscf}", flush=True)
    print(f"logL_dscf={ll_dscf:.6f}  logL_dsf={ll_dsf:.6f}  M={M:+.6f}  "
          f"threshold={args.threshold}", flush=True)
    if not np.isfinite(M):
        print("[skip] non-finite M", flush=True)
        return 125
    if args.print_only:
        return 0
    verdict = "GOOD (old/gapped)" if abs(M) > args.threshold else "BAD (new/converged)"
    print(f"verdict: {verdict}", flush=True)
    return 0 if abs(M) > args.threshold else 1


if __name__ == "__main__":
    sys.exit(main())
