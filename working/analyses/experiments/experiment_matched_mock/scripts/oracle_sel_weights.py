#!/usr/bin/env python3
"""Dump per-injection selection weights w_j(H0) = target/pdraw at a few H0
values (darksirens' own weight machinery, eager, CPU), for bootstrap /
delta-method error analysis of the mu_hat(H0) slope.

Writes results/oracle_selw_<tag>.npz with ldw (n_inj, n_H0).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")
EXP = Path(__file__).resolve().parents[1]
DARKSIRENS_REPO = os.environ.get(
    "DARKSIRENS_SRC", "/hildafs/projects/phy230014p/magana/src/darksirens")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--gw_path", required=True)
    ap.add_argument("--gwselection_path", required=True)
    ap.add_argument("--survey_path", required=True)
    ap.add_argument("--h0_values", nargs="+", type=float,
                    default=[62.74, 65.24, 66.74, 67.24, 67.74, 68.24, 68.74, 70.24, 72.74])
    ap.add_argument("--out_tag", required=True)
    args = ap.parse_args(argv)
    sys.path.insert(0, DARKSIRENS_REPO)

    import jax
    import jax.numpy as jnp
    from darksirens.inference.data import load_all_data, validate_loaded_survey_shapes
    from darksirens.inference.parameters import build_parameter_decoder
    from darksirens.inference.prior import build_parameter_space
    from darksirens.gw.populations import get_fixed_population_params, pop_model_parser
    from darksirens.likelihood.catalog_views import prepare_catalog_views
    from darksirens.redshift.completion import build_pixel_kde_cache
    from darksirens.core.types import EMCatalog
    from darksirens.redshift.prior import (
        prepare_redshift_prior_state, eval_redshift_prior_with_state)
    from darksirens.inference.utils import log_sample_weight
    from darksirens.utils.cosmology import dL_grid_bounds

    opts = SimpleNamespace(
        universe_model="dark_sirens_complete",
        survey_path=args.survey_path, survey_paths=[args.survey_path], n_catalogs=1,
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
        catalog_sky_weighting="field", max_likelihood_variance=1e6,
    )
    fixed = {"Om0": 0.3075}
    data = load_all_data(opts)
    validate_loaded_survey_shapes(data)
    res = build_parameter_space(
        "powerlaw+peak", True, False, False, fix_de=True, prior_overrides={},
        fixed_parameter_values=fixed, universe_model=opts.universe_model,
        shared_beta=True, shared_spin=True, shared_gamma=True,
        sky_model="isotropic", mark_model="none", mark_names=(), n_catalogs=1,
        lss_completion_active=[False], use_lss=False, mark_names_by_catalog=None)
    labels = list(res[0])
    pop_fid = get_fixed_population_params("powerlaw+peak", shared_beta=True,
                                          shared_spin=True, shared_gamma=True)
    decoder = build_parameter_decoder(opts, pop_fid, fixed_parameter_values=fixed)
    catalogs = prepare_catalog_views(opts, data, opts.universe_model, None,
                                     cache_builder=build_pixel_kde_cache)
    em_sel = EMCatalog(
        apix=data["apix"], zgals=catalogs.zgals_sel_catalog,
        dzgals=catalogs.dzgals_sel_catalog, wgals=catalogs.wgals_sel_catalog,
        ngals=catalogs.ngals_sel_catalog, delta_g_pix_z=catalogs.delta_g_pix_z,
        dN_obs_kde=catalogs.dN_obs_kde_sel,
        pixel_to_cache_idx=catalogs.pixel_to_cache_idx_sel,
        unique_pixels=catalogs.unique_pixels_sel,
        sample_to_unique_idx=catalogs.sample_to_unique_sel,
        counterpart_pixel=None, counterpart_pixels=None, counterpart_zs=None,
        counterpart_dzs=None, active_counterpart_index=0,
        bright_siren_sky_marginalized=False, lss_completion_logq=None,
        lss_completion_logq_members=None, lss_completion_indexing=0,
        mark_logmstar=None, mark_logssfr=None, mark_metallicity=None,
        mark_color=None,
        field_dN_obs_s=getattr(catalogs, "field_dN_obs_s", None),
        field_n_empty=getattr(catalogs, "field_n_empty", None),
        field_N_obs_total=getattr(catalogs, "field_N_obs_total", None),
        field_occupied_pixels=getattr(catalogs, "field_occupied_pixels", None),
        field_lss_q=None, field_lss_q_empty_sum=None, field_lss_q_members=None,
        field_lss_q_empty_sum_members=None, field_delta_g=None,
        field_mark_z=None, field_mark_w=None, field_mark_values=None,
        field_depth_z=None, field_depth_dz=None, field_depth_c=None,
    )
    to_j = lambda k: jnp.asarray(np.asarray(data[k]))
    m1 = to_j("m1detsels"); m2 = to_j("m2detsels"); dl = to_j("dLsels")
    ch = to_j("chieffsels"); pw = to_j("p_draw")
    px = jnp.asarray(catalogs.sample_to_unique_sel)
    q = m2 / m1
    log_p_pop = pop_model_parser("powerlaw+peak", shared_beta=True,
                                 shared_spin=True, shared_gamma=True)
    idxH0 = labels.index("H0")
    base = np.zeros(len(labels)); base[labels.index("H0")] = 67.74
    for i, lbl in enumerate(labels):
        if lbl not in ("H0",):
            base[i] = {"sigma_kde": 0.0, "delta": 0.0, "b_miss": 1.0}.get(lbl, 0.0)

    @jax.jit
    def ldw_at(coord):
        cosmo, survey, pop_params, _, _ = decoder.decode(coord)
        state = prepare_redshift_prior_state(
            "dark_sirens_complete", cosmo, survey, em_sel,
            catalog_sky_weighting="field")
        def log_prior_z(z, pix, cat):
            return eval_redshift_prior_with_state(
                "dark_sirens_complete", state, z, pix, cosmo, survey, em_sel,
                catalog_sky_weighting="field")
        dL_lo, dL_hi = dL_grid_bounds(cosmo.H0, cosmo.Om0, cosmo.w0, cosmo.wa)
        supported = (dl >= dL_lo) & (dl <= dL_hi)
        dL_c = jnp.clip(dl, dL_lo, dL_hi)
        ldw = log_sample_weight(m1, q, dL_c, ch, px, pw, cosmo, survey,
                                pop_params, em_sel, log_p_pop, log_prior_z)
        return jnp.where(supported & jnp.isfinite(ldw), ldw, -jnp.inf)

    out = np.empty((len(args.h0_values), int(m1.shape[0])))
    for k, h in enumerate(args.h0_values):
        c = base.copy(); c[idxH0] = h
        out[k] = np.asarray(ldw_at(jnp.asarray(c)))
        print("done", h, flush=True)
    np.savez_compressed(EXP / "results" / f"oracle_selw_{args.out_tag}.npz",
                        H0=np.asarray(args.h0_values), ldw=out,
                        Ndraw=float(data["Ndraw"]))
    print("wrote", EXP / "results" / f"oracle_selw_{args.out_tag}.npz")


if __name__ == "__main__":
    main()
