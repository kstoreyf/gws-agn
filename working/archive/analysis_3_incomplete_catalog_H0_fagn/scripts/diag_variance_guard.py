#!/usr/bin/env python3
"""Measure the post-#312 total-variance guard budget for this campaign's data.

Master @ 2b86a2d rejects a likelihood cell unless

    Neff > max(5*N_obs, N_obs^2 / (max_likelihood_variance - pe_variance_sum))

(darksirens/likelihood/selection.py:selection_log_correction, the GWTC-4.0/5.0
sigma^2_lnL <= 1 bound added after the #212 baseline this campaign's previous
run used).  The previous run only ever faced the legacy `Neff > 5*N_obs` floor.

This script instruments `core.selection_log_correction` with a jax.debug.print
wrapper and evaluates ONE coordinate, so we learn the three numbers that decide
whether the campaign can run at all under the new guard:

  * pe_variance_sum = sum_i sigma^2_i  (per-event PE reweighting variance)
  * Neff            (selection-integral effective sample size)
  * the implied threshold and the minimum max_likelihood_variance that admits it

Usage mirrors scan_darksirens.py's data flags; --f_at picks the fcat_2 value.
"""
import argparse
import json
import os
import sys
from types import SimpleNamespace

import numpy as np

OM0_FID = 0.3075
NUISANCE_DEFAULTS = {
    "delta": 0.0, "b_miss": 1.0, "sigma_kde": 0.0,
    "delta_c2": 0.0, "b_miss_c2": 1.0, "sigma_kde_c2": 0.0,
}


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
    ap.add_argument("--h0_at", type=float, default=67.74)
    ap.add_argument("--f_at", type=float, default=None)
    ap.add_argument("--log10n0", type=float, default=None)
    ap.add_argument("--log10n0_c2", type=float, default=None)
    ap.add_argument("--max_likelihood_variance", type=float, default=None,
                    help="Guard budget to probe (default: darksirens' 1.0).")
    ap.add_argument("--device", choices=["gpu", "cpu"], default="gpu")
    ap.add_argument("--sel_batch_size", type=int, default=None,
                    help="Injections per selection-integral chunk (memory).")
    ap.add_argument("--pe_event_block", type=int, default=None,
                    help="Events per PE-reduction chunk (memory).")
    ap.add_argument("--kde_window", type=int, default=None,
                    help="Windowed catalog-KDE size W; see scan_h0f.py --kde_window.")
    ap.add_argument("--kde_window_nsigma", type=float, default=8.0)
    ap.add_argument("--out_json", default=None)
    ap.add_argument("--capture_event_vars", action="store_true",
                    help="Also capture the PER-EVENT MC variances sigma^2_i (not just "
                         "their sum), to test whether the total-variance budget is "
                         "spent uniformly across events or by a heavy tail of a few.")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    # "cuda,cpu": jax.debug.print needs a local CPU device to place its inputs on.
    os.environ["JAX_PLATFORMS"] = "cuda,cpu" if args.device == "gpu" else "cpu"

    import jax
    import jax.numpy as jnp
    import darksirens
    from darksirens.inference.data import load_all_data, validate_loaded_survey_shapes
    from darksirens.likelihood import core as ds_core
    from darksirens.likelihood.factory import make_likelihood
    if args.kde_window is not None:
        from darksirens.redshift.catalog import configure_catalog_kde_window
        configure_catalog_kde_window(size=int(args.kde_window),
                                     n_sigma=float(args.kde_window_nsigma))
        print(f"[kde] window W={args.kde_window} n_sigma={args.kde_window_nsigma}")
    from darksirens.gw.populations import get_fixed_population_params
    from darksirens.inference.prior import build_parameter_space
    from darksirens.likelihood.selection import (
        selection_log_correction as _true_slc,
        DEFAULT_MAX_LIKELIHOOD_VARIANCE,
        _MIN_VARIANCE_BUDGET,
    )
    print(f"darksirens: {darksirens.__file__}")
    print(f"JAX devices: {jax.devices()}  DEFAULT_MAX_LIKELIHOOD_VARIANCE="
          f"{DEFAULT_MAX_LIKELIHOOD_VARIANCE}")

    # --- instrument the guard -------------------------------------------------
    captured = []

    def _spy(log_mu, Neff, nEvents, soft_guard=False,
             max_likelihood_variance=DEFAULT_MAX_LIKELIHOOD_VARIANCE,
             pe_variance_sum=0.0):
        n = float(nEvents)
        budget = jnp.maximum(max_likelihood_variance - pe_variance_sum,
                             _MIN_VARIANCE_BUDGET)
        threshold = jnp.maximum(5.0 * n, (n * n) / budget)

        def _record(log_mu, Neff, pe_var, thr):
            rec = {
                "log_mu": float(log_mu), "Neff": float(Neff),
                "pe_variance_sum": float(pe_var),
                "selection_variance_N2_over_Neff": float(n * n) / float(Neff),
                "threshold": float(thr),
                "passes": bool(float(Neff) > float(thr)),
                "nEvents": n,
            }
            rec["sigma2_total"] = (rec["pe_variance_sum"]
                                   + rec["selection_variance_N2_over_Neff"])
            # Smallest budget that would admit this cell (legacy floor aside):
            # need Neff > n^2/(V - pe_var)  <=>  V > pe_var + n^2/Neff.
            rec["min_max_likelihood_variance"] = rec["sigma2_total"]
            rec["legacy_floor_5N"] = 5.0 * n
            rec["passes_legacy_floor"] = bool(float(Neff) > 5.0 * n)
            captured.append(rec)
            print("[guard] " + " ".join(
                f"{k}={v:.6g}" if isinstance(v, float) else f"{k}={v}"
                for k, v in rec.items()))

        jax.debug.callback(_record, log_mu, Neff, pe_variance_sum, threshold)
        return _true_slc(log_mu, Neff, nEvents, soft_guard=soft_guard,
                         max_likelihood_variance=max_likelihood_variance,
                         pe_variance_sum=pe_variance_sum)

    ds_core.selection_log_correction = _spy

    # Per-event sigma^2_i capture. core.py calls this inside its per-event
    # vmap/scan, so the callback fires once per event (unordered) — fine for a
    # single evaluation, and it is the only way to see the DISTRIBUTION behind
    # the pe_variance_sum scalar the guard consumes.
    event_vars = []
    if args.capture_event_vars:
        _true_levmv = ds_core.log_evidence_and_mc_variance

        def _levmv_spy(ldw, nsamp):
            lz, var = _true_levmv(ldw, nsamp)
            jax.debug.callback(lambda v: event_vars.append(float(v)), var)
            return lz, var

        ds_core.log_evidence_and_mc_variance = _levmv_spy

    survey_paths = [str(p) for p in args.survey_path]
    opts = SimpleNamespace(
        universe_model=args.universe_model,
        survey_path=survey_paths[0], survey_paths=survey_paths,
        n_catalogs=len(survey_paths),
        gw_path=args.gw_path, gwselection_path=args.gwselection_path,
        use_LSS=False, lss_completion=None, lss_completions=[],
        lss_marginalize=False, counterpart=None, counterpart_nside=1,
        counterpart_dz=1e-4, bright_siren_sky_marginalized=False,
        drop_full_catalog=False, sky_model="isotropic", mark_model="none",
        marks=None, mark_names=(), sel_batch_size=args.sel_batch_size,
        pe_event_block=args.pe_event_block,
        redshift_prior_barrier="auto", selection_neff_guard="auto",
        selection_neff_soft_guard=False, sampler="tinyns",
        fix_population=True, fix_cosmology=False, fix_de=True, fix_survey=False,
        pop_model="powerlaw+peak", shared_beta=True, shared_spin=True,
        shared_gamma=True, complete_empty_pixel_policy="zero",
        catalog_sky_weighting=args.catalog_sky_weighting,
    )
    if args.max_likelihood_variance is not None:
        opts.max_likelihood_variance = args.max_likelihood_variance

    fixed_parameter_values = {"Om0": OM0_FID}
    data = load_all_data(opts)
    validate_loaded_survey_shapes(data)
    print(f"nEvents={data['nEvents']} nsamp={data['nsamp']} Ndraw={data['Ndraw']}")

    res = build_parameter_space(
        opts.pop_model, opts.fix_population, opts.fix_cosmology, opts.fix_survey,
        fix_de=opts.fix_de, prior_overrides={},
        fixed_parameter_values=fixed_parameter_values,
        universe_model=opts.universe_model, shared_beta=opts.shared_beta,
        shared_spin=opts.shared_spin, shared_gamma=opts.shared_gamma,
        sky_model=opts.sky_model, mark_model=opts.mark_model,
        mark_names=opts.mark_names, n_catalogs=opts.n_catalogs,
        lss_completion_active=[False] * opts.n_catalogs,
        use_lss=bool(opts.use_LSS), mark_names_by_catalog=None,
    )
    labels = list(res[0])
    print(f"labels: {labels}")

    pop_fid = get_fixed_population_params(
        opts.pop_model, shared_beta=opts.shared_beta,
        shared_spin=opts.shared_spin, shared_gamma=opts.shared_gamma)
    likelihood = make_likelihood(opts=opts, data=data, pop_params_fid=pop_fid,
                                fixed_parameter_values=fixed_parameter_values)

    point = dict(NUISANCE_DEFAULTS)
    point["H0"] = args.h0_at
    if args.log10n0 is not None:
        point["log10n0"] = args.log10n0
    if args.log10n0_c2 is not None:
        point["log10n0_c2"] = args.log10n0_c2
    if args.f_at is not None:
        point["fcat_2"] = args.f_at
    missing = [l for l in labels if l not in point]
    if missing:
        sys.exit(f"[fatal] missing values for {missing}")
    coord = np.asarray([float(point[l]) for l in labels], dtype=float)
    print(f"coord: {dict(zip(labels, coord.tolist()))}")

    ll = float(likelihood(coord))
    print(f"\nlogL = {ll}")
    print(f"guard call sites traced: {len(captured)}")

    ev_stats = None
    if args.capture_event_vars and event_vars:
        v = np.asarray(event_vars, dtype=float)
        v = v[np.isfinite(v)]
        order = np.sort(v)[::-1]
        tot = float(v.sum())
        ev_stats = {
            "n_events_captured": int(v.size),
            "sum": tot,
            "mean": float(v.mean()),
            "median": float(np.median(v)),
            "max": float(v.max()),
            "p90": float(np.percentile(v, 90)),
            "p99": float(np.percentile(v, 99)),
            "share_top1": float(order[:1].sum() / tot) if tot > 0 else None,
            "share_top10": float(order[:10].sum() / tot) if tot > 0 else None,
            "share_top1pct": float(order[: max(1, v.size // 100)].sum() / tot)
                              if tot > 0 else None,
            "n_events_to_reach_budget_1.0": (int(np.searchsorted(np.cumsum(order), 1.0) + 1)
                                             if tot > 1.0 else None),
        }
        print("\n=== per-event sigma^2_i distribution ===")
        for k, val in ev_stats.items():
            print(f"  {k}: {val}")

    if args.out_json:
        from pathlib import Path
        Path(args.out_json).write_text(json.dumps({
            "labels": labels, "coord": dict(zip(labels, coord.tolist())),
            "logL": ll if np.isfinite(ll) else None,
            "logL_is_neginf": not bool(np.isfinite(ll)),
            "nEvents": int(data["nEvents"]),
            "nsamp": int(data["nsamp"]), "Ndraw": float(data["Ndraw"]),
            "universe_model": args.universe_model,
            "catalog_sky_weighting": args.catalog_sky_weighting,
            "survey_paths": survey_paths,
            "gw_path": args.gw_path,
            "gwselection_path": args.gwselection_path,
            "max_likelihood_variance": args.max_likelihood_variance,
            "default_max_likelihood_variance": float(DEFAULT_MAX_LIKELIHOOD_VARIANCE),
            "guard_records": captured,
            "event_variance_stats": ev_stats,
        }, indent=2))
        print(f"Wrote {args.out_json}")


if __name__ == "__main__":
    main()
