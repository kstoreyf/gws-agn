#!/usr/bin/env python3
"""ATTRIBUTION -- term-by-term decomposition of the per-event score residual.

The detected-set score identity that the residual violates is, exactly,

    r  =  <d ln Z_i/dH0>_events  -  d ln mu/dH0            (evaluated at H0 = truth)

Both sides are averages of ONE function of the sample parameters.  Writing the
model's target density in the likelihood's own canonical basis
(m1det, q, dL, chieff, pix),

    ln p_target(theta | H0) = ln p_pop(m1src, q, z, chieff)
                            + ln p_z(z | pix)
                            - ln[d dL/dz] - ln(1+z),        z = z(dL; H0),

define the per-sample score

    varsigma(theta) = d ln p_target/dH0  |_theta fixed
                    = varsigma_pop + varsigma_pz + varsigma_jac.

Then EXACTLY

    d ln Z_i/dH0 = E_{PE posterior i}[varsigma]        (softmax of the event's ldw)
    d ln mu /dH0 = E_{model detected}[varsigma]        (softmax of the injections' ldw)

so r splits term by term into (events' posterior mean) - (injections' mean) of
each of the three additive pieces.  That is the attribution: whichever piece
carries the -1.6e-3 is the mis-specified channel.

This script reproduces darksirens' own per-sample weight EXACTLY -- it captures
the CONCRETE arguments the factory hands to ``darksiren_log_likelihood`` (an
import-level pass-through patch of the factory's module-global; darksirens is
READ-ONLY) and then re-evaluates ``log_sample_weight`` on them with the same
clamping, masking and prior state.  Two anchors are checked and written to the
output: log mu must reproduce the likelihood's own value, and Sum_i ln Z_i must
reproduce (total logL - selection term) from the guard spy.

Outputs
-------
results/attr_terms_<tag>.npz    per-event and per-injection arrays
results/attr_terms_<tag>.json   anchors, the term table, provenance
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = Path("/hildafs/projects/phy230014p/magana/gws-agn/working/data")

OM0_FID = 0.3075
H0_TRUE = 67.74
NUISANCE_DEFAULTS = {"delta": 0.0, "b_miss": 1.0, "sigma_kde": 0.0}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--tracer", choices=["gal", "agn"], default="gal")
    ap.add_argument("--events", default=None,
                    help="Override the events file (default: the matched-host subset).")
    ap.add_argument("--survey_override", default=None)
    ap.add_argument("--injections", default="targeted")
    ap.add_argument("--h0", type=float, default=H0_TRUE)
    ap.add_argument("--dh", type=float, default=0.5,
                    help="Central-difference step in H0.")
    ap.add_argument("--log10n0", type=float, default=-24.0)
    ap.add_argument("--kde_window", type=int, default=None)
    ap.add_argument("--kde_window_nsigma", type=float, default=8.0)
    ap.add_argument("--pe_batch_events", type=int, default=25)
    ap.add_argument("--sel_batch", type=int, default=50000)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--outdir", default=str(ROOT / "results"))
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")

    import jax
    import jax.numpy as jnp
    from jax.scipy.special import logsumexp
    import darksirens
    from darksirens.inference.data import load_all_data, validate_loaded_survey_shapes
    from darksirens.likelihood import core as ds_core
    from darksirens.likelihood import factory as ds_factory
    from darksirens.likelihood.factory import make_likelihood
    from darksirens.gw.populations import get_fixed_population_params, pop_model_parser
    from darksirens.inference.prior import build_parameter_space
    from darksirens.likelihood.selection import (
        selection_log_correction as _true_slc, DEFAULT_MAX_LIKELIHOOD_VARIANCE,
        log_evidence_and_mc_variance)
    from darksirens.redshift.prior import (
        prepare_redshift_prior_state, eval_redshift_prior_with_state)
    from darksirens.utils.cosmology import z_of_dL, ddL_of_z, dL_grid_bounds

    print(f"darksirens: {darksirens.__file__}   devices: {jax.devices()}")

    if args.kde_window is not None:
        from darksirens.redshift.catalog import configure_catalog_kde_window
        configure_catalog_kde_window(size=int(args.kde_window),
                                     n_sigma=float(args.kde_window_nsigma))
        print(f"[kde] window W={args.kde_window} n_sigma={args.kde_window_nsigma}")

    sd = DATA / f"seed{args.seed}"
    survey = (args.survey_override if args.survey_override
              else str(sd / "surveys" / f"survey_{args.tracer}_complete_ns32.h5"))
    gw = (args.events if args.events
          else str(ROOT / "data_derived" / f"events_{args.tracer}_hosted.h5"))
    inj = str(sd / "injections" / f"injections_{args.injections}.h5")
    tag = args.tag or f"{args.tracer}_s{args.seed}"

    # ---------------- guard spy (anchors the decomposition) ------------------
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
        sel_batch_size=args.sel_batch, pe_event_block=args.pe_batch_events,
        redshift_prior_barrier="auto",
        selection_neff_guard="hard", selection_neff_soft_guard=False,
        sampler="tinyns", fix_population=True, fix_cosmology=False, fix_de=True,
        fix_survey=False, pop_model="powerlaw+peak", shared_beta=True,
        shared_spin=True, shared_gamma=True, complete_empty_pixel_policy="zero",
        catalog_sky_weighting="field", max_likelihood_variance=1e6,
    )
    fixed = {"Om0": OM0_FID}

    t0 = time.time()
    data = load_all_data(opts)
    validate_loaded_survey_shapes(data)
    nobs = int(data["nEvents"]); nsamp = int(data["nsamp"])
    print(f"load_all_data {time.time()-t0:.1f}s  nEvents={nobs} nsamp={nsamp} "
          f"Ndraw={data['Ndraw']}")

    res = build_parameter_space(
        opts.pop_model, opts.fix_population, opts.fix_cosmology, opts.fix_survey,
        fix_de=opts.fix_de, prior_overrides={}, fixed_parameter_values=fixed,
        universe_model=opts.universe_model, shared_beta=True, shared_spin=True,
        shared_gamma=True, sky_model=opts.sky_model, mark_model=opts.mark_model,
        mark_names=opts.mark_names, n_catalogs=1,
        lss_completion_active=[False], use_lss=False, mark_names_by_catalog=None)
    labels = list(res[0])
    print(f"labels: {labels}")
    pop_fid = get_fixed_population_params(opts.pop_model, shared_beta=True,
                                          shared_spin=True, shared_gamma=True)
    print(f"pop_params_fid = {np.asarray(pop_fid)}")

    point = dict(NUISANCE_DEFAULTS)
    point["H0"] = args.h0
    point["log10n0"] = args.log10n0
    missing = [l for l in labels if l not in point]
    if missing:
        raise SystemExit(f"[fatal] missing label values: {missing}")
    base = np.asarray([float(point[l]) for l in labels], float)

    # ---------------- capture the concrete likelihood operands ----------------
    # The factory wraps its own body in jax.jit, so the arrays reaching
    # ``darksiren_log_likelihood`` are TRACERS.  Replace that wrapper with an
    # eager one (same table binding, same body) so the inner call receives the
    # CONCRETE operands, then pass-through-patch the inner call to record them.
    # darksirens itself is untouched; both patches are import-level.
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
        likelihood.operands = operands
        likelihood.distance_table = dtab
        return likelihood

    def _capture(*a, **kw):
        if not cap:
            cap["args"] = a
            cap["kwargs"] = kw
        return _true_dsll(*a, **kw)

    ds_factory._jit_likelihood_body = _eager_body
    ds_factory.darksiren_log_likelihood = _capture
    like = make_likelihood(opts=opts, data=data, pop_params_fid=pop_fid,
                           fixed_parameter_values=fixed)
    rec.clear()
    t0 = time.time()
    ll_ref = float(like(base))
    ds_factory.darksiren_log_likelihood = _true_dsll
    ds_factory._jit_likelihood_body = _true_jit_body
    spy0 = rec[-1]
    print(f"reference eval {time.time()-t0:.1f}s  logL={ll_ref:.6f}  "
          f"sel_term={spy0['sel_term']:.6f}  log_mu={spy0['log_mu']:.6f}  "
          f"Neff={spy0['Neff']:.3e}")

    a = cap["args"]
    cosmo0, survey_p, pop_params, gw_pe, cat_pe, gw_sel, cat_sel = a[:7]
    Ndraw = float(a[9])
    print(f"captured operands: nPE={gw_pe.dL.shape[0]} nSEL={gw_sel.dL.shape[0]} "
          f"Ndraw={Ndraw:.4g}  catalog rows={np.asarray(cat_pe.ngals).shape}")
    print(f"survey params: {survey_p}")

    log_p_pop = pop_model_parser(pop_model="powerlaw+peak", shared_beta=True,
                                 shared_spin=True, shared_gamma=True)

    # ---------------- the standalone per-sample evaluator ---------------------
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
            ldw = jnp.where(ok, ldw, -jnp.inf)
            return ldw, lp_pop, lp_z, ljac, z
        # NOT jitted: jax.jit lowers the closed-over (12288 x 14569) kernel state
        # to an HLO constant (the trap documented in factory._jit_likelihood_body),
        # and the per-batch tensors are large enough that eager dispatch is free.
        return pieces

    H0s = [args.h0 - args.dh, args.h0, args.h0 + args.dh]
    t0 = time.time()
    fns = [make_pieces(h) for h in H0s]
    print(f"prior states built ({time.time()-t0:.1f}s)")
    _red = jax.jit(jax.vmap(lambda row: log_evidence_and_mc_variance(row, nsamp)))

    # ---------------- PE pass -------------------------------------------------
    nblk = args.pe_batch_events
    nchunk = int(np.ceil(nobs / nblk))
    ev_lnZ = np.zeros(nobs); ev_var = np.zeros(nobs)
    ev_s_tot = np.zeros(nobs); ev_s_pop = np.zeros(nobs)
    ev_s_pz = np.zeros(nobs); ev_s_jac = np.zeros(nobs)
    ev_zbar = np.zeros(nobs); ev_lnZ_pm = np.zeros((nobs, 2))
    t0 = time.time()
    for c in range(nchunk):
        i0, i1 = c * nblk, min((c + 1) * nblk, nobs)
        s0, s1 = i0 * nsamp, i1 * nsamp
        sl = lambda arr: jnp.asarray(arr)[s0:s1]
        arg = (sl(gw_pe.m1det), sl(gw_pe.q), sl(gw_pe.dL), sl(gw_pe.chieff),
               sl(gw_pe.pixels), sl(gw_pe.prior_wt), sl(gw_pe.valid))
        out = [f(*arg) for f in fns]
        m = i1 - i0
        ldw = np.asarray(out[1][0]).reshape(m, nsamp)
        lnZ, var = _red(out[1][0].reshape(m, nsamp))
        ev_lnZ[i0:i1] = np.asarray(lnZ); ev_var[i0:i1] = np.asarray(var)
        for k, kk in ((0, 0), (2, 1)):
            lnZk, _ = _red(out[k][0].reshape(m, nsamp))
            ev_lnZ_pm[i0:i1, kk] = np.asarray(lnZk)
        # posterior softmax weights at the CENTRAL H0
        w = np.exp(ldw - ldw.max(axis=1, keepdims=True))
        w = np.where(np.isfinite(ldw), w, 0.0)
        w = w / w.sum(axis=1, keepdims=True)
        d = lambda idx: ((np.asarray(out[2][idx]) - np.asarray(out[0][idx]))
                         / (2.0 * args.dh)).reshape(m, nsamp)
        dpop, dpz, djac = d(1), d(2), d(3)
        dpop = np.nan_to_num(dpop); dpz = np.nan_to_num(dpz); djac = np.nan_to_num(djac)
        ev_s_pop[i0:i1] = (w * dpop).sum(axis=1)
        ev_s_pz[i0:i1] = (w * dpz).sum(axis=1)
        ev_s_jac[i0:i1] = -(w * djac).sum(axis=1)
        ev_zbar[i0:i1] = (w * np.asarray(out[1][4]).reshape(m, nsamp)).sum(axis=1)
        if c % 5 == 0:
            print(f"  PE chunk {c+1}/{nchunk}  ({time.time()-t0:.0f}s)")
    ev_s_tot = ev_s_pop + ev_s_pz + ev_s_jac
    ev_s_fd = (ev_lnZ_pm[:, 1] - ev_lnZ_pm[:, 0]) / (2.0 * args.dh)
    print(f"PE pass {time.time()-t0:.0f}s   sum lnZ = {ev_lnZ.sum():.6f}")

    # ---------------- selection pass -----------------------------------------
    nsel = int(gw_sel.dL.shape[0])
    sb = args.sel_batch
    nsb = int(np.ceil(nsel / sb))
    acc = {k: 0.0 for k in ("lse", "lse2")}
    lse_parts = []
    sel_z_all = []
    sel_w_all = []
    sel_d = {"pop": [], "pz": [], "jac": []}
    lse_pm = [[], []]
    t0 = time.time()
    for c in range(nsb):
        j0, j1 = c * sb, min((c + 1) * sb, nsel)
        sl = lambda arr: jnp.asarray(arr)[j0:j1]
        arg = (sl(gw_sel.m1det), sl(gw_sel.q), sl(gw_sel.dL), sl(gw_sel.chieff),
               sl(gw_sel.pixels), sl(gw_sel.prior_wt), sl(gw_sel.valid))
        out = [f(*arg) for f in fns]
        ldw = np.asarray(out[1][0])
        lse_parts.append(ldw)
        for k, kk in ((0, 0), (2, 1)):
            lse_pm[kk].append(np.asarray(out[k][0]))
        sel_z_all.append(np.asarray(out[1][4]))
        d = lambda idx: (np.asarray(out[2][idx]) - np.asarray(out[0][idx])) / (2.0 * args.dh)
        sel_d["pop"].append(np.nan_to_num(d(1)))
        sel_d["pz"].append(np.nan_to_num(d(2)))
        sel_d["jac"].append(-np.nan_to_num(d(3)))
        if c % 10 == 0:
            print(f"  SEL batch {c+1}/{nsb}  ({time.time()-t0:.0f}s)")
    sel_ldw = np.concatenate(lse_parts)
    sel_z = np.concatenate(sel_z_all)
    sel_dpop = np.concatenate(sel_d["pop"])
    sel_dpz = np.concatenate(sel_d["pz"])
    sel_djac = np.concatenate(sel_d["jac"])
    sel_ldw_m = np.concatenate(lse_pm[0]); sel_ldw_p = np.concatenate(lse_pm[1])
    print(f"SEL pass {time.time()-t0:.0f}s")

    def _lse(x):
        m = np.max(x[np.isfinite(x)]) if np.isfinite(x).any() else -np.inf
        return m + np.log(np.sum(np.exp(np.where(np.isfinite(x), x - m, -np.inf))))

    lse0 = _lse(sel_ldw)
    log_mu = lse0 - np.log(Ndraw)
    log_mu_m = _lse(sel_ldw_m) - np.log(Ndraw)
    log_mu_p = _lse(sel_ldw_p) - np.log(Ndraw)
    dlnmu_fd = (log_mu_p - log_mu_m) / (2.0 * args.dh)
    wsel = np.exp(np.where(np.isfinite(sel_ldw), sel_ldw - np.max(sel_ldw[np.isfinite(sel_ldw)]), -np.inf))
    wsel = wsel / wsel.sum()
    sel_pop = float((wsel * sel_dpop).sum())
    sel_pz = float((wsel * sel_dpz).sum())
    sel_jac = float((wsel * sel_djac).sum())
    sel_tot = sel_pop + sel_pz + sel_jac

    # ---------------- anchors -------------------------------------------------
    anchor_lnZ = ll_ref - spy0["sel_term"]
    print("\n=== ANCHORS ===")
    print(f"  log_mu   mine {log_mu:.10f}   darksirens {spy0['log_mu']:.10f}   "
          f"diff {log_mu - spy0['log_mu']:.3e}")
    print(f"  sum lnZ  mine {ev_lnZ.sum():.6f}   darksirens {anchor_lnZ:.6f}   "
          f"diff {ev_lnZ.sum() - anchor_lnZ:.3e}")

    score_mean = float(ev_s_tot.mean())
    score_mean_fd = float(ev_s_fd.mean())
    r_terms = {
        "pop": float(ev_s_pop.mean() - sel_pop),
        "pz": float(ev_s_pz.mean() - sel_pz),
        "jac": float(ev_s_jac.mean() - sel_jac),
    }
    r_total = sum(r_terms.values())
    print("\n=== TERM TABLE (per event) ===")
    print(f"{'term':>6} {'events':>14} {'injections':>14} {'r_term':>14}")
    for k, ev, se in (("pop", ev_s_pop.mean(), sel_pop),
                      ("pz", ev_s_pz.mean(), sel_pz),
                      ("jac", ev_s_jac.mean(), sel_jac)):
        print(f"{k:>6} {ev:14.6e} {se:14.6e} {ev - se:14.6e}")
    print(f"{'TOTAL':>6} {score_mean:14.6e} {sel_tot:14.6e} {r_total:14.6e}")
    print(f"  cross-check r from finite-difference lnZ / lnmu: "
          f"{score_mean_fd - dlnmu_fd:.6e}")

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        outdir / f"attr_terms_{tag}.npz",
        ev_lnZ=ev_lnZ, ev_var=ev_var, ev_s_tot=ev_s_tot, ev_s_fd=ev_s_fd,
        ev_s_pop=ev_s_pop, ev_s_pz=ev_s_pz, ev_s_jac=ev_s_jac, ev_zbar=ev_zbar,
        sel_ldw=sel_ldw.astype(np.float32), sel_z=sel_z.astype(np.float32),
        sel_dpop=sel_dpop.astype(np.float32), sel_dpz=sel_dpz.astype(np.float32),
        sel_djac=sel_djac.astype(np.float32),
    )
    summary = {
        "name": "attr_score_terms", "tag": tag, "seed": args.seed,
        "tracer": args.tracer, "H0": args.h0, "dh": args.dh,
        "log10n0": args.log10n0, "nobs": nobs, "nsamp": nsamp,
        "n_injections": nsel, "Ndraw": Ndraw,
        "config": {"survey": survey, "gw": gw, "injections": inj,
                   "kde_window": args.kde_window,
                   "universe_model": "dark_sirens",
                   "catalog_sky_weighting": "field"},
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "anchors": {
            "logL_reference": ll_ref,
            "sel_term_darksirens": spy0["sel_term"],
            "log_mu_darksirens": spy0["log_mu"],
            "log_mu_mine": float(log_mu),
            "log_mu_absdiff": float(abs(log_mu - spy0["log_mu"])),
            "sum_lnZ_darksirens": float(anchor_lnZ),
            "sum_lnZ_mine": float(ev_lnZ.sum()),
            "sum_lnZ_absdiff": float(abs(ev_lnZ.sum() - anchor_lnZ)),
            "Neff": spy0["Neff"], "pe_variance_sum": spy0["pe_variance_sum"],
        },
        "score_events": {"total": score_mean, "total_fd": score_mean_fd,
                         "pop": float(ev_s_pop.mean()), "pz": float(ev_s_pz.mean()),
                         "jac": float(ev_s_jac.mean()),
                         "sem_total": float(ev_s_tot.std(ddof=1) / np.sqrt(nobs))},
        "score_selection": {"total": sel_tot, "total_fd": float(dlnmu_fd),
                            "pop": sel_pop, "pz": sel_pz, "jac": sel_jac},
        "r_terms": r_terms, "r_total": r_total,
        "r_finite_difference": float(score_mean_fd - dlnmu_fd),
    }
    (outdir / f"attr_terms_{tag}.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {outdir / f'attr_terms_{tag}.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
