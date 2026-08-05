#!/usr/bin/env python3
"""Is the residual H0 bias the Monte-Carlo bias of log(mean of weights)?

Each event's term is ln Zhat = -log n + logsumexp(ldw), a Monte-Carlo estimate
of ln Z over that event's n PE samples.  ln of a mean is not the mean of a ln:
to leading order (Essick & Farr 2022; Talbot & Golomb 2023)

    E[ln Zhat] = ln Z - sigma_i^2 / 2,     sigma_i^2 = sum_j w_j^2/(sum_j w_j)^2 - 1/n

so the total log-likelihood is biased LOW by sum_i sigma_i^2 / 2.  darksirens
already computes exactly this sigma_i^2 -- ``log_evidence_and_mc_variance`` in
``likelihood/selection.py`` -- but only feeds it to the total-variance GUARD.
It is never subtracted off, because a constant offset in logL is harmless.

It is not constant.  sigma_i^2 measures how unevenly the weight is spread over
an event's PE samples, and that depends on where those samples land relative to
the catalog's host redshifts -- which moves with H0.  A sum that varies with H0
is a TILT, and it biases the peak.

This predicts every known property of the residual: it vanishes as the PE width
goes to zero (narrow posteriors concentrate the weight, sigma^2 -> 0), grows with
sigma_dL, is spread evenly across events rather than driven by outliers, is
always LOW in sign, and survives both generator fixes because neither changes
n_samp.

This script scans H0 capturing logL AND sum_i sigma_i^2 at every grid point, and
reports the peak of the raw and of the corrected curve

    logL_corrected(H0) = logL(H0) + sum_i sigma_i^2(H0) / 2.

If the correction moves the peak onto the truth, the mechanism is identified.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

OM0_FID = 0.3075
NUISANCE_DEFAULTS = {"delta": 0.0, "b_miss": 1.0, "sigma_kde": 0.0,
                     "delta_c2": 0.0, "b_miss_c2": 1.0, "sigma_kde_c2": 0.0}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--universe_model", required=True,
                    choices=["dark_sirens", "dark_sirens_complete"])
    ap.add_argument("--catalog_sky_weighting", default="field",
                    choices=["conditional", "field"])
    ap.add_argument("--survey_path", nargs="+", required=True)
    ap.add_argument("--gw_path", required=True)
    ap.add_argument("--gwselection_path", required=True)
    ap.add_argument("--h0_grid", nargs=3, type=float, default=[58.0, 78.0, 81])
    ap.add_argument("--f_at", type=float, default=None)
    ap.add_argument("--log10n0", type=float, default=None)
    ap.add_argument("--log10n0_c2", type=float, default=None)
    ap.add_argument("--nuisance_json", default=None)
    ap.add_argument("--h0_true", type=float, default=67.74)
    ap.add_argument("--device", choices=["gpu", "cpu"], default="gpu")
    ap.add_argument("--out_json", required=True)
    return ap.parse_args(argv)


def peak_and_median(x, ll):
    ok = np.isfinite(ll)
    p = np.where(ok, np.exp(ll - np.nanmax(ll[ok])), 0.0)
    norm = np.trapz(p, x)
    p = p / norm
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (p[1:] + p[:-1]) * np.diff(x))])
    cdf /= cdf[-1]
    return {"argmax": float(x[int(np.nanargmax(np.where(ok, ll, -np.inf)))]),
            "median": float(np.interp(0.5, cdf, x)),
            "ci68": [float(np.interp(0.16, cdf, x)), float(np.interp(0.84, cdf, x))]}


def main(argv=None):
    args = parse_args(argv)
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["JAX_PLATFORMS"] = "cuda,cpu" if args.device == "gpu" else "cpu"

    import jax
    import darksirens
    from darksirens.inference.data import load_all_data, validate_loaded_survey_shapes
    from darksirens.likelihood import core as ds_core
    from darksirens.likelihood.factory import make_likelihood
    from darksirens.gw.populations import get_fixed_population_params
    from darksirens.inference.prior import build_parameter_space
    print(f"darksirens: {darksirens.__file__}\nJAX: {jax.devices()}")

    # Capture the per-event MC variances the library already computes.
    captured = []
    _true = ds_core.log_evidence_and_mc_variance

    def _spy(ldw, nsamp):
        lz, var = _true(ldw, nsamp)
        jax.debug.callback(lambda v: captured.append(float(v)), var)
        return lz, var

    ds_core.log_evidence_and_mc_variance = _spy

    nuisance = dict(NUISANCE_DEFAULTS)
    if args.nuisance_json:
        src = (Path(args.nuisance_json).read_text()
               if os.path.exists(args.nuisance_json) else args.nuisance_json)
        nuisance.update({k: float(v) for k, v in json.loads(src).items()})

    survey_paths = [str(p) for p in args.survey_path]
    opts = SimpleNamespace(
        universe_model=args.universe_model,
        survey_path=survey_paths[0], survey_paths=survey_paths,
        n_catalogs=len(survey_paths), gw_path=args.gw_path,
        gwselection_path=args.gwselection_path, use_LSS=False, lss_completion=None,
        lss_completions=[], lss_marginalize=False, counterpart=None,
        counterpart_nside=1, counterpart_dz=1e-4,
        bright_siren_sky_marginalized=False, drop_full_catalog=False,
        sky_model="isotropic", mark_model="none", marks=None, mark_names=(),
        sel_batch_size=None, redshift_prior_barrier="auto",
        selection_neff_guard="hard", selection_neff_soft_guard=False,
        sampler="tinyns", fix_population=True, fix_cosmology=False, fix_de=True,
        fix_survey=False, pop_model="powerlaw+peak", shared_beta=True,
        shared_spin=True, shared_gamma=True, complete_empty_pixel_policy="zero",
        catalog_sky_weighting=args.catalog_sky_weighting,
        max_likelihood_variance=1e6,
    )
    data = load_all_data(opts)
    validate_loaded_survey_shapes(data)
    nobs = int(data["nEvents"])
    print(f"nEvents={nobs} nsamp={data['nsamp']}")

    res = build_parameter_space(
        opts.pop_model, opts.fix_population, opts.fix_cosmology, opts.fix_survey,
        fix_de=opts.fix_de, prior_overrides={},
        fixed_parameter_values={"Om0": OM0_FID},
        universe_model=opts.universe_model, shared_beta=opts.shared_beta,
        shared_spin=opts.shared_spin, shared_gamma=opts.shared_gamma,
        sky_model=opts.sky_model, mark_model=opts.mark_model,
        mark_names=opts.mark_names, n_catalogs=opts.n_catalogs,
        lss_completion_active=[False] * opts.n_catalogs,
        use_lss=bool(opts.use_LSS), mark_names_by_catalog=None)
    labels = list(res[0])
    likelihood = make_likelihood(
        opts=opts, data=data,
        pop_params_fid=get_fixed_population_params(
            opts.pop_model, shared_beta=opts.shared_beta,
            shared_spin=opts.shared_spin, shared_gamma=opts.shared_gamma),
        fixed_parameter_values={"Om0": OM0_FID})

    point = dict(nuisance)
    if args.log10n0 is not None:
        point["log10n0"] = args.log10n0
    if args.log10n0_c2 is not None:
        point["log10n0_c2"] = args.log10n0_c2
    if args.f_at is not None:
        point["fcat_2"] = args.f_at
    idx_h0 = labels.index("H0")
    point["H0"] = args.h0_true
    missing = [lb for lb in labels if lb not in point]
    if missing:
        raise SystemExit(f"[fatal] missing {missing}")
    base = np.asarray([float(point[lb]) for lb in labels], dtype=float)

    H0 = np.linspace(args.h0_grid[0], args.h0_grid[1], int(round(args.h0_grid[2])))
    ll = np.empty(H0.size)
    var_sum = np.empty(H0.size)
    var_max = np.empty(H0.size)
    for k, h in enumerate(H0):
        captured.clear()
        c = base.copy()
        c[idx_h0] = h
        ll[k] = float(likelihood(c))
        v = np.asarray([x for x in captured if np.isfinite(x)], dtype=float)
        var_sum[k] = float(v.sum())
        var_max[k] = float(v.max()) if v.size else np.nan
        if k % 10 == 0 or k == H0.size - 1:
            print(f"  H0={h:6.2f}  logL={ll[k]:12.4f}  sum sigma^2={var_sum[k]:9.4f}  "
                  f"n_captured={v.size}", flush=True)

    corrected = ll + 0.5 * var_sum
    raw, cor = peak_and_median(H0, ll), peak_and_median(H0, corrected)
    out = {
        "gw_path": args.gw_path, "survey_paths": survey_paths,
        "gwselection_path": args.gwselection_path,
        "nEvents": nobs, "nsamp": int(data["nsamp"]), "h0_true": args.h0_true,
        "H0_grid": H0.tolist(), "logL": ll.tolist(),
        "pe_variance_sum": var_sum.tolist(), "pe_variance_max": var_max.tolist(),
        "logL_mc_corrected": corrected.tolist(),
        "raw": raw, "corrected": cor,
        "raw_offset": raw["median"] - args.h0_true,
        "corrected_offset": cor["median"] - args.h0_true,
        "variance_sum_at_truth": float(np.interp(args.h0_true, H0, var_sum)),
        "variance_sum_range": [float(var_sum.min()), float(var_sum.max())],
        "variance_sum_slope_per_kms": float(np.gradient(var_sum, H0)[
            int(np.argmin(np.abs(H0 - args.h0_true)))]),
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, indent=2))
    print("\n=== MC-variance bias test ===")
    print(f"  sum sigma^2 at truth      : {out['variance_sum_at_truth']:.4f}  "
          f"(range {out['variance_sum_range'][0]:.3f}-{out['variance_sum_range'][1]:.3f})")
    print(f"  d(sum sigma^2)/dH0        : {out['variance_sum_slope_per_kms']:+.5f} per km/s/Mpc")
    print(f"  raw       median H0 = {raw['median']:.3f}  offset {out['raw_offset']:+.3f}")
    print(f"  corrected median H0 = {cor['median']:.3f}  offset {out['corrected_offset']:+.3f}")
    print(f"  wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
