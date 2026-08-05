#!/usr/bin/env python3
"""Eager per-event mirror of darksirens' K=1 dark_sirens_complete/field likelihood.

Reimplements the exact per-H0 evaluation of the compiled likelihood by calling
darksirens' own building blocks directly (prepare_redshift_prior_state,
eval_redshift_prior_with_state, log_sample_weight, log_evidence_and_mc_variance,
_lse_to_log_mu_neff, selection_log_correction), so that

  * the per-event ln Zhat_i(H0) curves and the selection pieces
    (ln mu_hat, Neff, Farr term) are individually captured, and
  * the sum reproduces the archived compiled scan (results/obsdet_obs_<tag>.h5)
    to float re-association — the validation anchor.

CPU only.  Writes results/oracle_dsref_<tag>.npz.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

EXP = Path(__file__).resolve().parents[1]
DARKSIRENS_REPO = os.environ.get(
    "DARKSIRENS_SRC", "/hildafs/projects/phy230014p/magana/src/darksirens")

OM0_FID = 0.3075
NUISANCE = {"delta": 0.0, "b_miss": 1.0, "sigma_kde": 0.0}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gw_path", required=True)
    ap.add_argument("--gwselection_path", required=True)
    ap.add_argument("--survey_path", required=True)
    ap.add_argument("--h0_grid", nargs=3, type=float, default=[58.0, 78.0, 161])
    ap.add_argument("--out_tag", required=True)
    ap.add_argument("--validate_h5", default=None,
                    help="Archived compiled scan (same grid) to diff against.")
    ap.add_argument("--pe_chunk", type=int, default=100,
                    help="Events per chunk in the PE reduction.")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    sys.path.insert(0, DARKSIRENS_REPO)

    import jax
    import jax.numpy as jnp
    from jax.scipy.special import logsumexp
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
    from darksirens.likelihood.selection import (
        log_evidence_and_mc_variance, _lse_to_log_mu_neff, selection_log_correction)

    print(f"jax devices: {jax.devices()}")

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
    fixed = {"Om0": OM0_FID}
    data = load_all_data(opts)
    validate_loaded_survey_shapes(data)
    nEvents, nsamp, Ndraw = int(data["nEvents"]), int(data["nsamp"]), float(data["Ndraw"])
    print(f"nEvents={nEvents} nsamp={nsamp} Ndraw={Ndraw}")

    res = build_parameter_space(
        "powerlaw+peak", True, False, False, fix_de=True, prior_overrides={},
        fixed_parameter_values=fixed, universe_model=opts.universe_model,
        shared_beta=True, shared_spin=True, shared_gamma=True,
        sky_model="isotropic", mark_model="none", mark_names=(), n_catalogs=1,
        lss_completion_active=[False], use_lss=False, mark_names_by_catalog=None)
    labels = list(res[0])
    print("labels:", labels)
    pop_fid = get_fixed_population_params("powerlaw+peak", shared_beta=True,
                                          shared_spin=True, shared_gamma=True)
    decoder = build_parameter_decoder(opts, pop_fid, fixed_parameter_values=fixed)

    catalogs = prepare_catalog_views(opts, data, opts.universe_model, None,
                                     cache_builder=build_pixel_kde_cache)
    apix = data["apix"]
    em_pe = EMCatalog(
        apix=apix, zgals=catalogs.zgals_pe_catalog, dzgals=catalogs.dzgals_pe_catalog,
        wgals=catalogs.wgals_pe_catalog, ngals=catalogs.ngals_pe_catalog,
        delta_g_pix_z=catalogs.delta_g_pix_z, dN_obs_kde=catalogs.dN_obs_kde_pe,
        pixel_to_cache_idx=catalogs.pixel_to_cache_idx_pe,
        unique_pixels=catalogs.unique_pixels_pe,
        sample_to_unique_idx=catalogs.sample_to_unique_pe,
        counterpart_pixel=None, counterpart_pixels=None, counterpart_zs=None,
        counterpart_dzs=None, active_counterpart_index=0,
        bright_siren_sky_marginalized=False,
        lss_completion_logq=None, lss_completion_logq_members=None,
        lss_completion_indexing=0,
        mark_logmstar=None, mark_logssfr=None, mark_metallicity=None, mark_color=None,
        field_dN_obs_s=getattr(catalogs, "field_dN_obs_s", None),
        field_n_empty=getattr(catalogs, "field_n_empty", None),
        field_N_obs_total=getattr(catalogs, "field_N_obs_total", None),
        field_occupied_pixels=getattr(catalogs, "field_occupied_pixels", None),
        field_lss_q=None, field_lss_q_empty_sum=None,
        field_lss_q_members=None, field_lss_q_empty_sum_members=None,
        field_delta_g=None, field_mark_z=None, field_mark_w=None,
        field_mark_values=None, field_depth_z=None, field_depth_dz=None,
        field_depth_c=None,
    )
    em_sel = em_pe._replace(
        zgals=catalogs.zgals_sel_catalog, dzgals=catalogs.dzgals_sel_catalog,
        wgals=catalogs.wgals_sel_catalog, ngals=catalogs.ngals_sel_catalog,
        dN_obs_kde=catalogs.dN_obs_kde_sel,
        pixel_to_cache_idx=catalogs.pixel_to_cache_idx_sel,
        unique_pixels=catalogs.unique_pixels_sel,
        sample_to_unique_idx=catalogs.sample_to_unique_sel,
    )

    to_j = lambda k: jnp.asarray(np.asarray(data[k]))
    pe = dict(m1det=to_j("m1det"), m2det=to_j("m2det"), dL=to_j("dL"),
              chieff=to_j("chieff"), prior_wt=to_j("p_pe"),
              pixels=jnp.asarray(catalogs.sample_to_unique_pe))
    pe["q"] = pe["m2det"] / pe["m1det"]
    sel = dict(m1det=to_j("m1detsels"), m2det=to_j("m2detsels"), dL=to_j("dLsels"),
               chieff=to_j("chieffsels"), prior_wt=to_j("p_draw"),
               pixels=jnp.asarray(catalogs.sample_to_unique_sel))
    sel["q"] = sel["m2det"] / sel["m1det"]
    nsel = int(sel["dL"].shape[0])

    log_p_pop = pop_model_parser("powerlaw+peak", shared_beta=True,
                                 shared_spin=True, shared_gamma=True)
    pop_fid_j = jnp.asarray(pop_fid)

    idxH0 = labels.index("H0")
    base_coord = np.array([NUISANCE.get(lbl, 0.0) for lbl in labels], dtype=float)
    base_coord[idxH0] = 67.74

    def make_weight_fn(cosmo, survey, pop_params, state, cat):
        def log_prior_z(z, pix, catalogs_arg):
            return eval_redshift_prior_with_state(
                "dark_sirens_complete", state, z, pix, cosmo, survey, cat,
                catalog_sky_weighting="field")
        def weight(m1det, q, dL, chieff, pix, prior_wt):
            dL_lo, dL_hi = dL_grid_bounds(cosmo.H0, cosmo.Om0, cosmo.w0, cosmo.wa)
            supported = (dL >= dL_lo) & (dL <= dL_hi)
            dL_c = jnp.clip(dL, dL_lo, dL_hi)
            ldw = log_sample_weight(m1det, q, dL_c, chieff, pix, prior_wt,
                                    cosmo, survey, pop_params, cat,
                                    log_p_pop, log_prior_z)
            return jnp.where(supported & jnp.isfinite(ldw), ldw, -jnp.inf)
        return weight

    pe_chunk = int(args.pe_chunk)
    n_chunks = (nEvents + pe_chunk - 1) // pe_chunk
    sel_chunk = 131072
    n_sel_chunks = (nsel + sel_chunk - 1) // sel_chunk

    import functools

    @functools.partial(jax.jit)
    def eval_h0(coord):
        cosmo, survey, pop_params, _, _ = decoder.decode(coord)
        state = prepare_redshift_prior_state(
            "dark_sirens_complete", cosmo, survey, em_pe,
            catalog_sky_weighting="field")
        w_pe = make_weight_fn(cosmo, survey, pop_params, state, em_pe)
        w_sel = make_weight_fn(cosmo, survey, pop_params, state, em_sel)

        def pe_chunk_fn(c, _):
            s = c * (pe_chunk * nsamp)
            sl = lambda a: jax.lax.dynamic_slice_in_dim(a, s, pe_chunk * nsamp)
            ldw = w_pe(sl(pe["m1det"]), sl(pe["q"]), sl(pe["dL"]),
                       sl(pe["chieff"]), sl(pe["pixels"]), sl(pe["prior_wt"]))
            ldw = ldw.reshape(pe_chunk, nsamp)
            lls, vars_ = jax.vmap(lambda r: log_evidence_and_mc_variance(r, nsamp))(ldw)
            return c + 1, (lls, vars_)
        _, (lls, vars_) = jax.lax.scan(pe_chunk_fn, 0, None, length=n_chunks)
        event_lls = lls.reshape(-1)[:nEvents]
        event_vars = vars_.reshape(-1)[:nEvents]

        # selection: pad to multiple of sel_chunk with -inf weights
        pad = n_sel_chunks * sel_chunk - nsel
        def padded(a, fill):
            return jnp.concatenate([a, jnp.full((pad,), fill, dtype=a.dtype)])
        sm1 = padded(sel["m1det"], 30.0); sq = padded(sel["q"], 0.5)
        sdl = padded(sel["dL"], -1.0)     # out of support -> -inf weight
        sch = padded(sel["chieff"], 0.0); spx = padded(sel["pixels"], 0)
        spw = padded(sel["prior_wt"], 1.0)

        def sel_chunk_fn(c, _):
            s = c * sel_chunk
            sl = lambda a: jax.lax.dynamic_slice_in_dim(a, s, sel_chunk)
            ldw = w_sel(sl(sm1), sl(sq), sl(sdl), sl(sch), sl(spx), sl(spw))
            finite = jnp.isfinite(ldw)
            safe = jnp.where(finite, ldw, -1e30)
            lse = jnp.where(jnp.any(finite), logsumexp(safe), -jnp.inf)
            lse2 = jnp.where(jnp.any(finite), logsumexp(2.0 * safe), -jnp.inf)
            return c + 1, (lse, lse2)
        _, (lses, lse2s) = jax.lax.scan(sel_chunk_fn, 0, None, length=n_sel_chunks)
        lse = logsumexp(lses); lse2 = logsumexp(lse2s)
        log_mu, Neff, _ = _lse_to_log_mu_neff(lse, lse2, Ndraw)
        pe_var_sum = jnp.sum(event_vars)
        sel_term = selection_log_correction(
            log_mu, Neff, nEvents, soft_guard=False,
            max_likelihood_variance=1e6, pe_variance_sum=pe_var_sum)
        total = sel_term + jnp.sum(event_lls)
        return event_lls, event_vars, log_mu, Neff, sel_term, total

    H0s = np.linspace(args.h0_grid[0], args.h0_grid[1], int(round(args.h0_grid[2])))
    ev_lls = np.empty((len(H0s), nEvents))
    ev_vars = np.empty((len(H0s), nEvents))
    log_mu = np.empty(len(H0s)); neff = np.empty(len(H0s))
    sel_t = np.empty(len(H0s)); tot = np.empty(len(H0s))
    t0 = time.time()
    for k, h in enumerate(H0s):
        c = base_coord.copy(); c[idxH0] = h
        out = eval_h0(jnp.asarray(c))
        ev_lls[k] = np.asarray(out[0]); ev_vars[k] = np.asarray(out[1])
        log_mu[k] = float(out[2]); neff[k] = float(out[3])
        sel_t[k] = float(out[4]); tot[k] = float(out[5])
        if k % 20 == 0 or k == len(H0s) - 1:
            print(f"[{k+1}/{len(H0s)}] H0={h:.3f} total={tot[k]:.4f} "
                  f"logmu={log_mu[k]:.5f} Neff={neff[k]:.0f} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    outp = EXP / "results" / f"oracle_dsref_{args.out_tag}.npz"
    save = dict(H0=H0s, event_lls=ev_lls, event_vars=ev_vars, log_mu=log_mu,
                Neff=neff, sel_term=sel_t, total=tot,
                gw_path=args.gw_path, survey_path=args.survey_path,
                gwselection_path=args.gwselection_path)
    if args.validate_h5:
        import h5py
        with h5py.File(args.validate_h5) as f:
            ref = np.asarray(f["log_likelihood"])
            refH0 = np.asarray(f["H0_grid"])
        assert np.allclose(refH0, H0s), "grid mismatch"
        diff = tot - ref
        print(f"validation vs {args.validate_h5}: max|diff|={np.abs(diff).max():.3e} "
              f"(rel to range {np.ptp(ref):.1f})")
        save["validate_ref"] = ref
        save["validate_maxdiff"] = np.abs(diff).max()
    np.savez_compressed(outp, **save)
    print("wrote", outp)


if __name__ == "__main__":
    main()
