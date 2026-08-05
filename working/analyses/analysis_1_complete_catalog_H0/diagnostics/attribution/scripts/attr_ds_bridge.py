#!/usr/bin/env python3
"""Shared darksirens bridge for the ATTRIBUTION follow-up scripts.

This is the VERBATIM loader + per-sample weight reconstruction that
``attr_score_terms.py`` and ``attr_mass_pe.py`` are anchored on
(``|log_mu diff| = 0``, ``|sum_i ln Z_i diff| = 0`` on both configurations).
It is factored out here so the sampler-ratio test and the quadrature oracle
reuse EXACTLY the same operands and the same ``ldw`` -- no re-derivation.

darksirens is READ-ONLY at 2b86a2d; the two patches below are the same
import-level pass-throughs the earlier scripts (and ``scan_h0f.py``'s guard
record) already use:

  * ``ds_core.selection_log_correction``   -> spy (records log_mu, Neff, sel term)
  * ``ds_factory._jit_likelihood_body``    -> eager wrapper, so the operands the
                                              inner call receives are CONCRETE
  * ``ds_factory.darksiren_log_likelihood``-> capture (records the operands)

Every consumer must call ``check_anchor`` and record the result.
"""
from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np

DATA = Path("/hildafs/projects/phy230014p/magana/gws-agn/working/data")
ROOT = Path(__file__).resolve().parent.parent

OM0_FID = 0.3075
H0_TRUE = 67.74
NUISANCE_DEFAULTS = {"delta": 0.0, "b_miss": 1.0, "sigma_kde": 0.0}


def set_jax_env():
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")


def build(seed=100, tracer="gal", events=None, injections="targeted",
          h0=H0_TRUE, log10n0=-24.0, kde_window=None, kde_window_nsigma=8.0,
          pe_batch_events=25, sel_batch=50000, survey_override=None,
          dataroot=None, events_dir=None):
    """Load the analysis of record and capture darksirens' concrete operands.

    Returns a SimpleNamespace with the captured operands, the standalone
    per-sample evaluator factory ``make_pieces(H0)``, and the anchors.
    """
    set_jax_env()

    import jax
    import jax.numpy as jnp
    import darksirens
    from darksirens.inference.data import load_all_data, validate_loaded_survey_shapes
    from darksirens.likelihood import core as ds_core
    from darksirens.likelihood import factory as ds_factory
    from darksirens.likelihood.factory import make_likelihood
    from darksirens.gw.populations import get_fixed_population_params, pop_model_parser
    from darksirens.inference.prior import build_parameter_space
    from darksirens.likelihood.selection import (
        selection_log_correction as _true_slc, DEFAULT_MAX_LIKELIHOOD_VARIANCE)
    from darksirens.redshift.prior import (
        prepare_redshift_prior_state, eval_redshift_prior_with_state)
    from darksirens.utils.cosmology import z_of_dL, ddL_of_z, dL_grid_bounds

    print(f"darksirens: {darksirens.__file__}   devices: {jax.devices()}")

    if kde_window is not None:
        from darksirens.redshift.catalog import configure_catalog_kde_window
        configure_catalog_kde_window(size=int(kde_window),
                                     n_sigma=float(kde_window_nsigma))
        print(f"[kde] window W={kde_window} n_sigma={kde_window_nsigma}")

    sd = (Path(dataroot) if dataroot else DATA) / f"seed{seed}"
    survey = (survey_override if survey_override
              else str(sd / "surveys" / f"survey_{tracer}_complete_ns32.h5"))
    _evd = Path(events_dir) if events_dir else (ROOT / "data_derived")
    gw = (events if events else str(_evd / f"events_{tracer}_hosted.h5"))
    inj = str(sd / "injections" / f"injections_{injections}.h5")

    rec = []

    def _spy(log_mu, Neff, nEvents, soft_guard=False,
             max_likelihood_variance=DEFAULT_MAX_LIKELIHOOD_VARIANCE,
             pe_variance_sum=0.0):
        out = _true_slc(log_mu, Neff, nEvents, soft_guard=soft_guard,
                        max_likelihood_variance=max_likelihood_variance,
                        pe_variance_sum=pe_variance_sum)
        jax.debug.callback(
            lambda lm, ne, pv, sel: rec.append(
                {"log_mu": float(lm), "Neff": float(ne),
                 "pe_variance_sum": float(pv), "sel_term": float(sel)}),
            log_mu, Neff, pe_variance_sum, out)
        return out

    ds_core.selection_log_correction = _spy

    opts = SimpleNamespace(
        universe_model="dark_sirens",
        survey_path=survey, survey_paths=[survey], n_catalogs=1,
        gw_path=gw, gwselection_path=inj,
        use_LSS=False, lss_completion=None, lss_completions=[], lss_marginalize=False,
        counterpart=None, counterpart_nside=1, counterpart_dz=1e-4,
        bright_siren_sky_marginalized=False, drop_full_catalog=False,
        sky_model="isotropic", mark_model="none", marks=None, mark_names=(),
        sel_batch_size=sel_batch, pe_event_block=pe_batch_events,
        redshift_prior_barrier="auto",
        selection_neff_guard="hard", selection_neff_soft_guard=False,
        sampler="tinyns", fix_population=True, fix_cosmology=False, fix_de=True,
        fix_survey=False, pop_model="powerlaw+peak", shared_beta=True,
        shared_spin=True, shared_gamma=True, complete_empty_pixel_policy="zero",
        catalog_sky_weighting="field", max_likelihood_variance=1e6,
    )
    fixed = {"Om0": OM0_FID}

    data = load_all_data(opts)
    validate_loaded_survey_shapes(data)
    nobs = int(data["nEvents"]); nsamp = int(data["nsamp"])

    res = build_parameter_space(
        opts.pop_model, opts.fix_population, opts.fix_cosmology, opts.fix_survey,
        fix_de=opts.fix_de, prior_overrides={}, fixed_parameter_values=fixed,
        universe_model=opts.universe_model, shared_beta=True, shared_spin=True,
        shared_gamma=True, sky_model=opts.sky_model, mark_model=opts.mark_model,
        mark_names=opts.mark_names, n_catalogs=1,
        lss_completion_active=[False], use_lss=False, mark_names_by_catalog=None)
    labels = list(res[0])
    pop_fid = get_fixed_population_params(opts.pop_model, shared_beta=True,
                                          shared_spin=True, shared_gamma=True)
    gamma_fid = float(np.asarray(pop_fid)[-1])
    print(f"labels: {labels}   gamma_fid={gamma_fid}")

    point = dict(NUISANCE_DEFAULTS)
    point["H0"] = h0
    point["log10n0"] = log10n0
    base = np.asarray([float(point[l]) for l in labels], float)

    cap = {}
    _true_dsll = ds_factory.darksiren_log_likelihood
    _true_jit_body = ds_factory._jit_likelihood_body

    def _eager_body(body, operands):
        dtab = ds_factory.cosmology.distance_table()
        smop = ds_factory.completion_smoothing_operator()

        def likelihood(coord):
            with ds_factory.cosmology.bound_distance_table(dtab), \
                    ds_factory.bound_smoothing_operator(smop):
                return body(coord, operands)
        return likelihood

    def _capture(*a, **kw):
        if not cap:
            cap["args"] = a
        return _true_dsll(*a, **kw)

    ds_factory._jit_likelihood_body = _eager_body
    ds_factory.darksiren_log_likelihood = _capture
    like = make_likelihood(opts=opts, data=data, pop_params_fid=pop_fid,
                           fixed_parameter_values=fixed)
    rec.clear()
    ll_ref = float(like(base))
    ds_factory.darksiren_log_likelihood = _true_dsll
    ds_factory._jit_likelihood_body = _true_jit_body
    spy0 = rec[-1]
    print(f"reference logL={ll_ref:.6f}  sel_term={spy0['sel_term']:.6f}  "
          f"log_mu={spy0['log_mu']:.6f}")

    a = cap["args"]
    cosmo0, survey_p, pop_params, gw_pe, cat_pe, gw_sel, cat_sel = a[:7]
    Ndraw = float(a[9])

    log_p_pop = pop_model_parser(pop_model="powerlaw+peak", shared_beta=True,
                                 shared_spin=True, shared_gamma=True)

    def make_pieces(H0):
        cosmo = cosmo0._replace(H0=jnp.float64(H0))
        state = prepare_redshift_prior_state(
            "dark_sirens", cosmo, survey_p, cat_pe, mark_model="none",
            mark_params=None, mark_names=(), materialize_state=True,
            catalog_sky_weighting="field")
        dL_lo, dL_hi = dL_grid_bounds(cosmo.H0, cosmo.Om0, cosmo.w0, cosmo.wa)

        def pieces(m1det, q, dL, chieff, pix, prior_wt, valid):
            supported = (dL >= dL_lo) & (dL <= dL_hi)
            dL_c = jnp.clip(dL, dL_lo, dL_hi)
            z = z_of_dL(dL_c, cosmo.H0, cosmo.Om0, cosmo.w0, cosmo.wa)
            m1src = m1det / (1.0 + z)
            lp_pop = log_p_pop(m1src, q, z, chieff, pop_params)
            lp_z = eval_redshift_prior_with_state(
                "dark_sirens", state, z, pix, cosmo, survey_p, cat_pe,
                catalog_sky_weighting="field")
            ljac = (jnp.log(ddL_of_z(z, dL_c, cosmo.H0, cosmo.Om0, cosmo.w0, cosmo.wa))
                    + jnp.log1p(z))
            ldw = lp_pop + lp_z - ljac - jnp.log(prior_wt)
            ok = valid & (prior_wt > 0.0) & supported & jnp.isfinite(ldw)
            return jnp.where(ok, ldw, -jnp.inf), lp_pop, lp_z, ljac, z
        return pieces

    def pz_only(H0):
        """Catalog redshift prior alone, as a function of (z, pix) at fixed H0."""
        cosmo = cosmo0._replace(H0=jnp.float64(H0))
        state = prepare_redshift_prior_state(
            "dark_sirens", cosmo, survey_p, cat_pe, mark_model="none",
            mark_params=None, mark_names=(), materialize_state=True,
            catalog_sky_weighting="field")

        def f(z, pix):
            return eval_redshift_prior_with_state(
                "dark_sirens", state, z, pix, cosmo, survey_p, cat_pe,
                catalog_sky_weighting="field")
        return f

    return SimpleNamespace(
        opts=opts, data=data, nobs=nobs, nsamp=nsamp, Ndraw=Ndraw,
        cosmo0=cosmo0, survey_p=survey_p, pop_params=pop_params,
        gw_pe=gw_pe, cat_pe=cat_pe, gw_sel=gw_sel, cat_sel=cat_sel,
        pop_fid=pop_fid, gamma_fid=gamma_fid, log_p_pop=log_p_pop,
        make_pieces=make_pieces, pz_only=pz_only,
        ll_ref=ll_ref, spy0=spy0, labels=labels, base=base,
        paths={"survey": survey, "gw": gw, "injections": inj},
        h0=h0, log10n0=log10n0, kde_window=kde_window,
    )


def sel_pass(B, dh=0.5, sel_batch=50000, want_q=True):
    """Selection-integral pass: per-injection ldw, score pieces, z, m1src, q.

    Reproduces ``attr_mass_pe.py``'s SEL block exactly and returns log_mu for
    the anchor check.
    """
    import jax.numpy as jnp
    fns = [B.make_pieces(h) for h in (B.h0 - dh, B.h0, B.h0 + dh)]
    twodh = 2.0 * dh
    gamma_fid = B.gamma_fid
    nsel = int(B.gw_sel.dL.shape[0])
    nsb = int(np.ceil(nsel / sel_batch))
    acc = {k: [] for k in ("ldw", "ldw_m", "ldw_p", "pop", "rate", "pz", "jac",
                           "z", "m1src", "q")}
    for c in range(nsb):
        j0, j1 = c * sel_batch, min((c + 1) * sel_batch, nsel)
        sl = lambda arr: jnp.asarray(arr)[j0:j1]
        arg = (sl(B.gw_sel.m1det), sl(B.gw_sel.q), sl(B.gw_sel.dL),
               sl(B.gw_sel.chieff), sl(B.gw_sel.pixels), sl(B.gw_sel.prior_wt),
               sl(B.gw_sel.valid))
        out = [f(*arg) for f in fns]
        ldw = np.asarray(out[1][0])
        d = lambda i: np.nan_to_num(
            (np.asarray(out[2][i]) - np.asarray(out[0][i])) / twodh)
        zm, zp = np.asarray(out[0][4]), np.asarray(out[2][4])
        d_rate = np.nan_to_num((gamma_fid - 1.0) * (np.log1p(zp) - np.log1p(zm)) / twodh)
        z0 = np.asarray(out[1][4])
        acc["ldw"].append(ldw)
        acc["ldw_m"].append(np.asarray(out[0][0]))
        acc["ldw_p"].append(np.asarray(out[2][0]))
        acc["pop"].append(d(1)); acc["rate"].append(d_rate)
        acc["pz"].append(d(2)); acc["jac"].append(-d(3)); acc["z"].append(z0)
        acc["m1src"].append(np.asarray(B.gw_sel.m1det, float)[j0:j1] / (1.0 + z0))
        if want_q:
            acc["q"].append(np.asarray(B.gw_sel.q, float)[j0:j1])
    S = {k: np.concatenate(v) for k, v in acc.items() if v}
    fin = np.isfinite(S["ldw"])
    mx = S["ldw"][fin].max()
    wraw = np.where(fin, np.exp(S["ldw"] - mx), 0.0)
    S["log_mu"] = mx + np.log(wraw.sum()) - np.log(B.Ndraw)
    S["w"] = wraw / wraw.sum()
    S["mass"] = S["pop"] - S["rate"]

    def _lse(x):
        f = np.isfinite(x)
        m = x[f].max()
        return m + np.log(np.exp(np.where(f, x - m, -np.inf)).sum())
    # FINITE-DIFFERENCE d ln mu/dH0 -- the convention the oracle's own
    # (ln Z_+ - ln Z_-)/2dh score must be differenced against.
    S["log_mu_m"] = _lse(S["ldw_m"]) - np.log(B.Ndraw)
    S["log_mu_p"] = _lse(S["ldw_p"]) - np.log(B.Ndraw)
    S["dlnmu_fd"] = (S["log_mu_p"] - S["log_mu_m"]) / (2.0 * dh)
    S["dlnmu_terms"] = float((S["w"] * (S["pop"] + S["pz"] + S["jac"])).sum())
    return S


def pe_pass_split(B, dh=0.5, pe_batch_events=25):
    """Per-event d ln Z_i/dH0 on the FULL sample set and on two disjoint halves.

    The half-difference gives a direct, per-event estimate of the Monte-Carlo
    error darksirens' own ln Zhat_i carries -- the yardstick the quadrature
    oracle has to be compared against.
    """
    import jax
    import jax.numpy as jnp
    from darksirens.likelihood.selection import log_evidence_and_mc_variance
    fns = [B.make_pieces(h) for h in (B.h0 - dh, B.h0, B.h0 + dh)]
    nobs, nsamp = B.nobs, B.nsamp
    red = jax.jit(jax.vmap(lambda row, n: log_evidence_and_mc_variance(row, n)[0],
                           in_axes=(0, None)), static_argnums=())
    out = {k: np.zeros(nobs) for k in ("full", "hA", "hB")}
    nblk = pe_batch_events
    for c in range(int(np.ceil(nobs / nblk))):
        i0, i1 = c * nblk, min((c + 1) * nblk, nobs)
        s0, s1 = i0 * nsamp, i1 * nsamp
        sl = lambda arr: jnp.asarray(arr)[s0:s1]
        arg = (sl(B.gw_pe.m1det), sl(B.gw_pe.q), sl(B.gw_pe.dL), sl(B.gw_pe.chieff),
               sl(B.gw_pe.pixels), sl(B.gw_pe.prior_wt), sl(B.gw_pe.valid))
        m = i1 - i0
        lz = {}
        for kk, f in ((0, fns[0]), (2, fns[2])):
            ldw = np.asarray(f(*arg)[0]).reshape(m, nsamp)
            lz[(kk, "full")] = np.asarray(red(jnp.asarray(ldw), nsamp))
            lz[(kk, "hA")] = np.asarray(red(jnp.asarray(ldw[:, 0::2]), nsamp // 2))
            lz[(kk, "hB")] = np.asarray(red(jnp.asarray(ldw[:, 1::2]), nsamp // 2))
        for key in ("full", "hA", "hB"):
            out[key][i0:i1] = (lz[(2, key)] - lz[(0, key)]) / (2.0 * dh)
    return out
