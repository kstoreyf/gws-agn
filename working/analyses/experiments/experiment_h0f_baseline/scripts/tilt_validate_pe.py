#!/usr/bin/env python3
"""Capture per-event ln Z_i(H0) from the PRODUCTION likelihood (spy on
log_evidence_and_mc_variance) at a few H0 values, to validate tilt_terms.py.

Writes results/tilt_validate_pe_<tag>.json with per-event lnZ arrays.
"""
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("DARKSIRENS_ZMAX", "1.5")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

OM0_FID = 0.3075
NUISANCE = {"delta": 0.0, "b_miss": 1.0, "sigma_kde": 0.0,
            "delta_c2": 0.0, "b_miss_c2": 1.0, "sigma_kde_c2": 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gw_path", required=True)
    ap.add_argument("--gwselection_path", required=True)
    ap.add_argument("--survey_path", nargs="+", required=True)
    ap.add_argument("--f_fixed", type=float, required=True)
    ap.add_argument("--h0_values", nargs="+", type=float,
                    default=[62.0, 67.74, 74.0])
    ap.add_argument("--out_tag", required=True)
    args = ap.parse_args()

    import jax
    import darksirens
    from darksirens.inference.data import load_all_data, validate_loaded_survey_shapes
    from darksirens.likelihood import core as ds_core
    from darksirens.likelihood.factory import make_likelihood
    from darksirens.gw.populations import get_fixed_population_params
    from darksirens.inference.prior import build_parameter_space
    from darksirens.likelihood.selection import (
        selection_log_correction as _true_slc,
        log_evidence_and_mc_variance as _true_lev,
        DEFAULT_MAX_LIKELIHOOD_VARIANCE)
    print(f"darksirens: {darksirens.__file__}  dev={jax.devices()}")

    lz_rec, sel_rec = [], []

    def _lev_spy(ldw, nsamp):
        lz, var = _true_lev(ldw, nsamp)
        jax.debug.callback(lambda a, b: lz_rec.append((np.array(a), np.array(b))),
                           lz, var)
        return lz, var

    def _slc_spy(log_mu, Neff, nEvents, soft_guard=False,
                 max_likelihood_variance=DEFAULT_MAX_LIKELIHOOD_VARIANCE,
                 pe_variance_sum=0.0):
        out = _true_slc(log_mu, Neff, nEvents, soft_guard=soft_guard,
                        max_likelihood_variance=max_likelihood_variance,
                        pe_variance_sum=pe_variance_sum)
        jax.debug.callback(
            lambda lm, ne, o: sel_rec.append(
                {"log_mu": float(lm), "Neff": float(ne), "sel_term": float(o)}),
            log_mu, Neff, out)
        return out

    ds_core.log_evidence_and_mc_variance = _lev_spy
    ds_core.selection_log_correction = _slc_spy

    survey_paths = [str(p) for p in args.survey_path]
    opts = SimpleNamespace(
        universe_model="dark_sirens", survey_path=survey_paths[0],
        survey_paths=survey_paths, n_catalogs=len(survey_paths),
        gw_path=args.gw_path, gwselection_path=args.gwselection_path,
        use_LSS=False, lss_completion=None, lss_completions=[],
        lss_marginalize=False, counterpart=None, counterpart_nside=1,
        counterpart_dz=1e-4, bright_siren_sky_marginalized=False,
        drop_full_catalog=False, sky_model="isotropic", mark_model="none",
        marks=None, mark_names=(), sel_batch_size=None,
        redshift_prior_barrier="auto", selection_neff_guard="hard",
        selection_neff_soft_guard=False, sampler="tinyns",
        fix_population=True, fix_cosmology=False, fix_de=True,
        fix_survey=False, pop_model="powerlaw+peak", shared_beta=True,
        shared_spin=True, shared_gamma=True,
        complete_empty_pixel_policy="zero", catalog_sky_weighting="field",
        max_likelihood_variance=1e6,
    )
    fixed = {"Om0": OM0_FID}
    data = load_all_data(opts)
    validate_loaded_survey_shapes(data)
    nobs = int(data["nEvents"])
    res = build_parameter_space(
        "powerlaw+peak", True, False, False, fix_de=True, prior_overrides={},
        fixed_parameter_values=fixed, universe_model="dark_sirens",
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
    point.update({"log10n0": -12.0, "log10n0_c2": -12.0,
                  "fcat_2": args.f_fixed, "H0": 67.74})
    base = np.asarray([float(point[lb]) for lb in labels], float)
    ih = labels.index("H0")

    out = {"h0_values": list(args.h0_values), "nobs": nobs,
           "lnZ_ev": {}, "sigma2_ev": {}, "sel": {}, "logL": {}}
    for h in args.h0_values:
        c = base.copy(); c[ih] = h
        lz_rec.clear(); sel_rec.clear()
        ll = float(like(c))
        lz = np.concatenate([np.atleast_1d(a) for a, _ in lz_rec])
        s2 = np.concatenate([np.atleast_1d(b) for _, b in lz_rec])
        print(f"H0={h}: logL={ll:.4f} captured {lz.size} events "
              f"sum lnZ={np.sum(lz[np.isfinite(lz)]):.4f} "
              f"sel={sel_rec[-1] if sel_rec else None}")
        out["lnZ_ev"][str(h)] = lz.tolist()
        out["sigma2_ev"][str(h)] = s2.tolist()
        out["sel"][str(h)] = sel_rec[-1] if sel_rec else None
        out["logL"][str(h)] = ll

    p = (Path(__file__).resolve().parent.parent / "results"
         / f"tilt_validate_pe_{args.out_tag}.json")
    p.write_text(json.dumps(out))
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
