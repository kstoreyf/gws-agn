#!/usr/bin/env python3
"""Apportion the H0 offset between the selection term and the per-event term.

The hierarchical log-likelihood at fixed mixture weight splits exactly into

    logL(H0) = SUM_i ln Z_i(H0)            <- per-event numerator (PE x prior x masses)
               + [ -N_obs ln mu(H0) + N_obs(N_obs+3)/(2 N_eff) ]   <- selection term

`selection_log_correction` returns the bracket, so instrumenting it recovers both
pieces from the same evaluations.  This matters because the two pieces implicate
different causes:

* This mock's detection is a HARD CUT ON TRUE REDSHIFT (z <= 1.0), for which the
  correct per-tracer normalisation is H0-INDEPENDENT (beta = CDF(z_max)).
  darksirens instead estimates mu(H0) from the injection set, reweighting
  injections by p_pop(z|H0); any slope that introduces is spurious here, and it
  enters multiplied by N_obs = 1000.
* The per-event numerator carries the redshift/sky prior AND the fixed
  powerlaw+peak mass model, which couples m_det/(1+z) to H0 (spectral-siren
  information).

So: if the numerator alone peaks at the truth, the selection term owns the offset.
If it does not, the per-event term contributes and the mass channel is implicated.
`ln mu(H0)` is reported directly so its slope can be read off.

Writes <outdir>/h0_decomposition_<tag>.json.
"""
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np

OM0_FID = 0.3075
NUISANCE = {"delta": 0.0, "b_miss": 1.0, "sigma_kde": 0.0,
            "delta_c2": 0.0, "b_miss_c2": 1.0, "sigma_kde_c2": 0.0}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gw_path", required=True)
    ap.add_argument("--gwselection_path", required=True)
    ap.add_argument("--survey_path", nargs="+", required=True)
    ap.add_argument("--f_fixed", type=float, default=None,
                    help="Fixed fcat_2; omit for a K=1 survey.")
    ap.add_argument("--universe_model", default="dark_sirens",
                    choices=["dark_sirens", "dark_sirens_complete"])
    ap.add_argument("--h0_grid", nargs=3, type=float, default=[55.0, 80.0, 51.0])
    ap.add_argument("--log10n0", type=float, default=-12.0)
    ap.add_argument("--h0_true", type=float, default=67.74)
    ap.add_argument("--out_tag", required=True)
    ap.add_argument("--outdir", default=None)
    return ap.parse_args(argv)


def quad_refine(x, y):
    y = np.asarray(y, float)
    ok = np.isfinite(y)
    if not ok.any():
        return float("nan")
    i = int(np.nanargmax(np.where(ok, y, np.nan)))
    if i in (0, len(y) - 1) or not (ok[i - 1] and ok[i + 1]):
        return float(x[i])
    d = y[i - 1] - 2 * y[i] + y[i + 1]
    if d == 0 or not np.isfinite(d):
        return float(x[i])
    return float(x[i] - 0.5 * (y[i + 1] - y[i - 1]) / d * (x[1] - x[0]))


def main(argv=None):
    args = parse_args(argv)
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["JAX_PLATFORMS"] = "cuda,cpu"      # debug callbacks need a CPU device

    import jax
    import jax.numpy as jnp
    import darksirens
    from darksirens.inference.data import load_all_data, validate_loaded_survey_shapes
    from darksirens.likelihood import core as ds_core
    from darksirens.likelihood.factory import make_likelihood
    from darksirens.gw.populations import get_fixed_population_params
    from darksirens.inference.prior import build_parameter_space
    from darksirens.likelihood.selection import (
        selection_log_correction as _true_slc, DEFAULT_MAX_LIKELIHOOD_VARIANCE)
    print(f"darksirens: {darksirens.__file__}")

    rec = []

    def _spy(log_mu, Neff, nEvents, soft_guard=False,
             max_likelihood_variance=DEFAULT_MAX_LIKELIHOOD_VARIANCE,
             pe_variance_sum=0.0):
        out = _true_slc(log_mu, Neff, nEvents, soft_guard=soft_guard,
                        max_likelihood_variance=max_likelihood_variance,
                        pe_variance_sum=pe_variance_sum)
        jax.debug.callback(
            lambda lm, ne, sel: rec.append(
                {"log_mu": float(lm), "Neff": float(ne), "sel_term": float(sel)}),
            log_mu, Neff, out)
        return out

    ds_core.selection_log_correction = _spy

    survey_paths = [str(p) for p in args.survey_path]
    opts = SimpleNamespace(
        universe_model=args.universe_model, survey_path=survey_paths[0],
        survey_paths=survey_paths, n_catalogs=len(survey_paths),
        gw_path=args.gw_path, gwselection_path=args.gwselection_path,
        use_LSS=False, lss_completion=None, lss_completions=[], lss_marginalize=False,
        counterpart=None, counterpart_nside=1, counterpart_dz=1e-4,
        bright_siren_sky_marginalized=False, drop_full_catalog=False,
        sky_model="isotropic", mark_model="none", marks=None, mark_names=(),
        sel_batch_size=None, redshift_prior_barrier="auto",
        selection_neff_guard="hard", selection_neff_soft_guard=False,
        sampler="tinyns", fix_population=True, fix_cosmology=False, fix_de=True,
        fix_survey=False, pop_model="powerlaw+peak", shared_beta=True,
        shared_spin=True, shared_gamma=True, complete_empty_pixel_policy="zero",
        catalog_sky_weighting="field",
        max_likelihood_variance=1e6,      # historical 5*N_obs guard only
    )
    fixed = {"Om0": OM0_FID}
    data = load_all_data(opts)
    validate_loaded_survey_shapes(data)
    nobs = int(data["nEvents"])
    print(f"nEvents={nobs} nsamp={data['nsamp']} Ndraw={data['Ndraw']}")

    res = build_parameter_space(
        "powerlaw+peak", True, False, False, fix_de=True, prior_overrides={},
        fixed_parameter_values=fixed, universe_model=args.universe_model,
        shared_beta=True, shared_spin=True, shared_gamma=True,
        sky_model="isotropic", mark_model="none", mark_names=(),
        n_catalogs=opts.n_catalogs,
        lss_completion_active=[False] * opts.n_catalogs,
        use_lss=False, mark_names_by_catalog=None)
    labels = list(res[0])
    pop_fid = get_fixed_population_params("powerlaw+peak", shared_beta=True,
                                          shared_spin=True, shared_gamma=True)
    like = make_likelihood(opts=opts, data=data, pop_params_fid=pop_fid,
                           fixed_parameter_values=fixed)

    point = dict(NUISANCE)
    point.update({"log10n0": args.log10n0, "log10n0_c2": args.log10n0,
                  "H0": args.h0_true})
    if args.f_fixed is not None:
        point["fcat_2"] = args.f_fixed
    missing = [l for l in labels if l not in point]
    if missing:
        raise SystemExit(f"[fatal] missing {missing}")
    base = np.asarray([float(point[l]) for l in labels], float)
    ih = labels.index("H0")

    H0 = np.linspace(args.h0_grid[0], args.h0_grid[1], int(round(args.h0_grid[2])))
    total, sel, log_mu, neff = [], [], [], []
    for h in H0:
        c = base.copy(); c[ih] = h
        rec.clear()
        ll = float(like(c))
        if not rec:
            raise SystemExit("[fatal] guard spy captured nothing")
        r = rec[-1]
        total.append(ll)
        sel.append(r["sel_term"])
        log_mu.append(r["log_mu"])
        neff.append(r["Neff"])
    total = np.asarray(total); sel = np.asarray(sel)
    log_mu = np.asarray(log_mu); neff = np.asarray(neff)
    numer = total - sel                      # SUM_i ln Z_i(H0)

    out = {
        "out_tag": args.out_tag, "gw_path": args.gw_path, "f_fixed": args.f_fixed,
        "nobs": nobs, "H0_true": args.h0_true, "H0_grid": H0.tolist(),
        "logL_total": total.tolist(), "selection_term": sel.tolist(),
        "per_event_numerator": numer.tolist(),
        "log_mu": log_mu.tolist(), "Neff": neff.tolist(),
        "peak_total": quad_refine(H0, total),
        "peak_per_event_numerator": quad_refine(H0, numer),
        "peak_selection_term": quad_refine(H0, sel),
    }
    out["offset_total"] = out["peak_total"] - args.h0_true
    out["offset_numerator_only"] = out["peak_per_event_numerator"] - args.h0_true
    # How much of the tilt the selection slope accounts for, in km/s/Mpc of peak shift.
    out["shift_from_selection_term"] = out["peak_total"] - out["peak_per_event_numerator"]
    # ln mu slope across the grid (the spurious H0 dependence under a true-z cut).
    out["dlnmu_dH0_at_truth"] = float(np.gradient(log_mu, H0)[np.argmin(abs(H0 - args.h0_true))])
    out["lnmu_range"] = [float(log_mu.min()), float(log_mu.max())]

    outdir = Path(args.outdir) if args.outdir else (Path(__file__).resolve().parent.parent / "results")
    outdir.mkdir(parents=True, exist_ok=True)
    p = outdir / f"h0_decomposition_{args.out_tag}.json"
    p.write_text(json.dumps(out, indent=2))

    print(f"\n=== {args.out_tag} (planted f = {args.f_fixed}) ===")
    print(f"  peak of FULL logL          : {out['peak_total']:.3f}  "
          f"(offset {out['offset_total']:+.3f})")
    print(f"  peak of per-event numerator: {out['peak_per_event_numerator']:.3f}  "
          f"(offset {out['offset_numerator_only']:+.3f})")
    print(f"  => shift attributable to the selection term: "
          f"{out['shift_from_selection_term']:+.3f} km/s/Mpc")
    print(f"  ln mu range over the grid  : {out['lnmu_range'][0]:.4f} .. "
          f"{out['lnmu_range'][1]:.4f}   d ln mu/dH0 at truth = "
          f"{out['dlnmu_dH0_at_truth']:+.5f}")
    print(f"  (selection enters as -N_obs ln mu, N_obs = {nobs})")
    print(f"Wrote {p}")


if __name__ == "__main__":
    main()
