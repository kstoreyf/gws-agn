#!/usr/bin/env python3
"""ATTRIBUTION stage 2 -- the mass channel of the score residual.

``attr_score_terms.py`` splits the score residual

    r = <d ln Z_i/dH0>_events - d ln mu/dH0

into the three additive pieces of ln p_target = ln p_pop + ln p_z - ln|J| and
finds it entirely in ``p_pop``.  ``p_pop`` depends on H0 ONLY through
``z = z(dL; H0)``, and only in two places:

    ln p_pop(m1src, q, z, chieff) = ln p_mass_spin(m1det/(1+z), q, chieff)
                                  + (gamma - 1) ln(1+z)

so the pop term splits exactly into a MASS piece (the source-frame mass moving
under H0) and a RATE piece (the fixed (1+z)^(gamma-1) weight).  This script
reports that split, and then tests the one convention in the generator that can
mis-specify the mass piece:

    generate_dataset.py::observe   sig_m1 = 0.08 * m1det_TRUE
                                   obs_m1 = N(m1det_true, sig_m1)
    generate_dataset.py::posterior_samples
                                   m1det ~ N(obs_m1, sig_m1)   <- LATENT width

The stored PE mass samples are a Gaussian about the observation whose width is
computed from the LATENT true mass.  The measurement model actually realised is
``obs ~ N(m, f m)`` with ``f`` constant, whose EXACT flat-prior posterior is

    p_ex(m | obs) ∝ (1/(f m)) exp[ -(obs - m)^2 / (2 f^2 m^2) ],

not a fixed-width Gaussian centred on ``obs``.  (This is the mass twin of the
sky-width defect fixed upstream in darksirens PR #335; the distance channel is
exempt because its noise is multiplicative with a CONSTANT log-width, whose
flat-prior posterior really is the stored lognormal.)

The test needs no regeneration: the stored samples are reweighted from the
fixed-width proposal to ``p_ex`` by

    rho(m) = p_ex(m | obs) / N(m; obs, sig_stored),

and ln Z_i is recomputed self-normalised with ``rho``.  ``rho`` is
H0-INDEPENDENT, so it cannot manufacture or hide an H0 slope by construction --
it can only correct the measure the score is averaged over.  The selection
integral is untouched (injections carry TRUE parameters; only their detection
decision saw noise), so ``d ln mu/dH0`` is identical in every arm.

Arms: none / m1 only / m1+m2.

Outputs: results/attr_mass_pe_<tag>.{npz,json}
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = Path("/hildafs/projects/phy230014p/magana/gws-agn/working/data")

OM0_FID = 0.3075
H0_TRUE = 67.74
SIG_M1_FRAC = 0.08          # generate_dataset.py::SIG_M1_FRAC
SIG_M2_FRAC = 0.10          # generate_dataset.py::SIG_M2_FRAC
NUISANCE_DEFAULTS = {"delta": 0.0, "b_miss": 1.0, "sigma_kde": 0.0}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--tracer", choices=["gal", "agn"], default="gal")
    ap.add_argument("--events", default=None)
    ap.add_argument("--injections", default="targeted")
    ap.add_argument("--h0", type=float, default=H0_TRUE)
    ap.add_argument("--dh", type=float, default=0.5)
    ap.add_argument("--log10n0", type=float, default=-24.0)
    ap.add_argument("--kde_window", type=int, default=None)
    ap.add_argument("--kde_window_nsigma", type=float, default=8.0)
    ap.add_argument("--pe_batch_events", type=int, default=25)
    ap.add_argument("--sel_batch", type=int, default=50000)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--outdir", default=str(ROOT / "results"))
    return ap.parse_args(argv)


def log_pex(m, obs, f):
    """log of the EXACT flat-prior posterior for obs ~ N(m, f*m) (unnormalised)."""
    return -np.log(f * m) - 0.5 * ((obs - m) / (f * m)) ** 2


def log_ptilde(m, obs, sig):
    """log of the STORED proposal N(m; obs, sig) (unnormalised in the same sense)."""
    return -np.log(sig) - 0.5 * ((m - obs) / sig) ** 2


def main(argv=None):
    args = parse_args(argv)
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")

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

    if args.kde_window is not None:
        from darksirens.redshift.catalog import configure_catalog_kde_window
        configure_catalog_kde_window(size=int(args.kde_window),
                                     n_sigma=float(args.kde_window_nsigma))
        print(f"[kde] window W={args.kde_window} n_sigma={args.kde_window_nsigma}")

    sd = DATA / f"seed{args.seed}"
    survey = str(sd / "surveys" / f"survey_{args.tracer}_complete_ns32.h5")
    gw = (args.events if args.events
          else str(ROOT / "data_derived" / f"events_{args.tracer}_hosted.h5"))
    inj = str(sd / "injections" / f"injections_{args.injections}.h5")
    tag = args.tag or f"{args.tracer}_s{args.seed}"

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
    point["H0"] = args.h0
    point["log10n0"] = args.log10n0
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

    # ---------------- the PE-mass correction weights --------------------------
    with h5py.File(gw, "r") as f:
        obs_m1 = np.asarray(f["truth/obs_m1det"][:], float)
        obs_m2 = np.asarray(f["truth/obs_m2det"][:], float)
        sig_m1 = np.asarray(f["truth/obs_sig_m1"][:], float)
        sig_m2 = np.asarray(f["truth/obs_sig_m2"][:], float)
        true_m1det = np.asarray(f["truth/m1det"][:], float)
        true_m2det = np.asarray(f["truth/m2det"][:], float)
        true_m1src = np.asarray(f["truth/m1src"][:], float)
        true_z = np.asarray(f["truth/z"][:], float)
    assert obs_m1.size == nobs, (obs_m1.size, nobs)
    print(f"stored sig_m1 / (f*m1det_true) max|dev| = "
          f"{np.max(np.abs(sig_m1 / (SIG_M1_FRAC * true_m1det) - 1)):.3e}   "
          f"sig_m1 / (f*obs_m1) spread = "
          f"{np.std(sig_m1 / (SIG_M1_FRAC * obs_m1)):.4f}")

    pe_m1 = np.asarray(gw_pe.m1det, float)
    pe_q = np.asarray(gw_pe.q, float)
    pe_m2 = pe_m1 * pe_q
    ev_idx = np.repeat(np.arange(nobs), nsamp)
    lr1 = (log_pex(pe_m1, obs_m1[ev_idx], SIG_M1_FRAC)
           - log_ptilde(pe_m1, obs_m1[ev_idx], sig_m1[ev_idx]))
    lr2 = (log_pex(pe_m2, obs_m2[ev_idx], SIG_M2_FRAC)
           - log_ptilde(pe_m2, obs_m2[ev_idx], sig_m2[ev_idx]))
    # Width-from-OBSERVED arms: the minimal "PR #335" style repair (keep the
    # fixed-width Gaussian, but set the width from the DATA rather than the
    # latent mass).  The exact-posterior arms above additionally carry the
    # O(f^2) shape/shift that a fixed-width Gaussian cannot represent.
    lo1 = (log_ptilde(pe_m1, obs_m1[ev_idx], SIG_M1_FRAC * obs_m1[ev_idx])
           - log_ptilde(pe_m1, obs_m1[ev_idx], sig_m1[ev_idx]))
    lo2 = (log_ptilde(pe_m2, obs_m2[ev_idx], SIG_M2_FRAC * obs_m2[ev_idx])
           - log_ptilde(pe_m2, obs_m2[ev_idx], sig_m2[ev_idx]))
    # Width-perturbation arms calibrate dr/d ln sigma_1 (how much a residual
    # PE-width error would be worth).
    lw = lambda s: (log_ptilde(pe_m1, obs_m1[ev_idx], s * sig_m1[ev_idx])
                    - log_ptilde(pe_m1, obs_m1[ev_idx], sig_m1[ev_idx]))
    arms = {"none": np.zeros_like(lr1), "m1": lr1, "m1m2": lr1 + lr2,
            "m1obs": lo1, "m1m2obs": lo1 + lo2,
            "s1x105": lw(1.05), "s1x095": lw(0.95)}
    for k, v in arms.items():
        v = v.reshape(nobs, nsamp).copy()
        v -= v.max(axis=1, keepdims=True)
        arms[k] = v
    # MC-convergence arm: the SAME estimator on half the PE samples.  If r moves,
    # it is a finite-nsamp artefact of ln Zhat, not a model mis-specification.
    half = np.zeros((nobs, nsamp)); half[:, nsamp // 2:] = -np.inf
    arms["half"] = half

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
            return jnp.where(ok, ldw, -jnp.inf), lp_pop, lp_z, ljac, z
        return pieces

    H0s = [args.h0 - args.dh, args.h0, args.h0 + args.dh]
    t0 = time.time()
    fns = [make_pieces(h) for h in H0s]
    print(f"prior states built ({time.time()-t0:.1f}s)")

    def _terms(out, twodh):
        """(ldw_central, d_pop, d_rate, d_pz, d_jac) as numpy arrays."""
        ldw = np.asarray(out[1][0])
        d = lambda i: np.nan_to_num(
            (np.asarray(out[2][i]) - np.asarray(out[0][i])) / twodh)
        zm, zp = np.asarray(out[0][4]), np.asarray(out[2][4])
        d_rate = np.nan_to_num((gamma_fid - 1.0) * (np.log1p(zp) - np.log1p(zm)) / twodh)
        return ldw, d(1), d_rate, d(2), d(3)

    twodh = 2.0 * args.dh

    # ---------------- PE pass -------------------------------------------------
    nblk = args.pe_batch_events
    nchunk = int(np.ceil(nobs / nblk))
    keys = ("pop", "rate", "mass", "pz", "jac", "tot")
    ev = {arm: {k: np.zeros(nobs) for k in keys} for arm in arms}
    ev_ess = {arm: np.zeros(nobs) for arm in arms}
    ev_zbar = {arm: np.zeros(nobs) for arm in arms}
    ev_m1srcbar = {arm: np.zeros(nobs) for arm in arms}
    t0 = time.time()
    for c in range(nchunk):
        i0, i1 = c * nblk, min((c + 1) * nblk, nobs)
        s0, s1 = i0 * nsamp, i1 * nsamp
        sl = lambda arr: jnp.asarray(arr)[s0:s1]
        arg = (sl(gw_pe.m1det), sl(gw_pe.q), sl(gw_pe.dL), sl(gw_pe.chieff),
               sl(gw_pe.pixels), sl(gw_pe.prior_wt), sl(gw_pe.valid))
        out = [f(*arg) for f in fns]
        m = i1 - i0
        ldw, dpop, drate, dpz, djac = _terms(out, twodh)
        ldw = ldw.reshape(m, nsamp)
        dpop = dpop.reshape(m, nsamp); drate = drate.reshape(m, nsamp)
        dpz = dpz.reshape(m, nsamp); djac = djac.reshape(m, nsamp)
        z0 = np.asarray(out[1][4]).reshape(m, nsamp)
        m1src0 = np.asarray(gw_pe.m1det, float)[s0:s1].reshape(m, nsamp) / (1.0 + z0)
        for arm, lrho in arms.items():
            lw = ldw + lrho[i0:i1]
            mx = np.max(np.where(np.isfinite(lw), lw, -np.inf), axis=1, keepdims=True)
            w = np.where(np.isfinite(lw), np.exp(lw - mx), 0.0)
            w = w / w.sum(axis=1, keepdims=True)
            ev_ess[arm][i0:i1] = 1.0 / (w ** 2).sum(axis=1)
            ev[arm]["pop"][i0:i1] = (w * dpop).sum(axis=1)
            ev[arm]["rate"][i0:i1] = (w * drate).sum(axis=1)
            ev[arm]["mass"][i0:i1] = (w * (dpop - drate)).sum(axis=1)
            ev[arm]["pz"][i0:i1] = (w * dpz).sum(axis=1)
            ev[arm]["jac"][i0:i1] = -(w * djac).sum(axis=1)
            ev_zbar[arm][i0:i1] = (w * z0).sum(axis=1)
            ev_m1srcbar[arm][i0:i1] = (w * m1src0).sum(axis=1)
        if c % 5 == 0:
            print(f"  PE chunk {c+1}/{nchunk}  ({time.time()-t0:.0f}s)")
    for arm in arms:
        ev[arm]["tot"] = ev[arm]["pop"] + ev[arm]["pz"] + ev[arm]["jac"]
    print(f"PE pass {time.time()-t0:.0f}s")

    # ---------------- events evaluated at their TRUE parameters ---------------
    # Splits r EXACTLY into
    #   r = (C - A) + (A - B)
    #     C = mean_i E_post_i[varsigma]   what the likelihood actually averages
    #     A = mean_i varsigma(theta_i^true)   the empirical detected-TRUTH mean
    #     B = E_injections[varsigma]          the model's detected-truth mean
    # (A - B) is a population / detection-rule mis-specification (Poisson-noise
    # limited); (C - A) is purely the measurement model encoded in the PE samples.
    # The pixel enters only ln p_z, so a dummy row is fine for the pop/jac terms;
    # the true row is resolved when healpy + unique_pixels allow it.
    with h5py.File(gw, "r") as f:
        t_m1det = np.asarray(f["truth/m1det"][:], float)
        t_m2det = np.asarray(f["truth/m2det"][:], float)
        t_dl = np.asarray(f["truth/dl"][:], float)
        t_chi = np.asarray(f["truth/chieff"][:], float)
        t_ra = np.asarray(f["truth/ra"][:], float)
        t_dec = np.asarray(f["truth/dec"][:], float)
    try:
        import healpy as hp
        gpix = hp.ang2pix(32, np.pi / 2.0 - t_dec, t_ra)
        up = np.asarray(cat_pe.unique_pixels)
        order = np.argsort(up)
        loc = np.searchsorted(up, gpix, sorter=order)
        loc = np.clip(loc, 0, up.size - 1)
        row = order[loc]
        pix_ok = up[row] == gpix
        t_pix = np.where(pix_ok, row, 0).astype(np.int32)
        print(f"true host pixel resolved for {pix_ok.sum()}/{nobs} events")
    except Exception as exc:                                       # pragma: no cover
        print(f"[warn] true-pixel mapping unavailable ({exc}); pz-at-truth skipped")
        t_pix = np.zeros(nobs, dtype=np.int32); pix_ok = np.zeros(nobs, bool)
    targ = (jnp.asarray(t_m1det), jnp.asarray(t_m2det / t_m1det), jnp.asarray(t_dl),
            jnp.asarray(t_chi), jnp.asarray(t_pix),
            jnp.ones(nobs), jnp.ones(nobs, dtype=bool))
    outT = [f(*targ) for f in fns]
    _, dpopT, drateT, dpzT, djacT = _terms(outT, twodh)
    truth_terms = {"pop": dpopT, "rate": drateT, "mass": dpopT - drateT,
                   "pz": dpzT, "jac": -djacT}
    A = {k: float(np.mean(v)) for k, v in truth_terms.items()}
    A_sem = {k: float(np.std(v, ddof=1) / np.sqrt(nobs)) for k, v in truth_terms.items()}
    A_pz_ok = float(np.mean(truth_terms["pz"][pix_ok])) if pix_ok.any() else float("nan")

    # ---------------- selection pass -----------------------------------------
    nsel = int(gw_sel.dL.shape[0])
    sb = args.sel_batch
    nsb = int(np.ceil(nsel / sb))
    acc = {k: [] for k in ("ldw", "pop", "rate", "pz", "jac", "z", "m1src")}
    t0 = time.time()
    for c in range(nsb):
        j0, j1 = c * sb, min((c + 1) * sb, nsel)
        sl = lambda arr: jnp.asarray(arr)[j0:j1]
        arg = (sl(gw_sel.m1det), sl(gw_sel.q), sl(gw_sel.dL), sl(gw_sel.chieff),
               sl(gw_sel.pixels), sl(gw_sel.prior_wt), sl(gw_sel.valid))
        out = [f(*arg) for f in fns]
        ldw, dpop, drate, dpz, djac = _terms(out, twodh)
        z0 = np.asarray(out[1][4])
        acc["ldw"].append(ldw); acc["pop"].append(dpop); acc["rate"].append(drate)
        acc["pz"].append(dpz); acc["jac"].append(-djac); acc["z"].append(z0)
        acc["m1src"].append(np.asarray(gw_sel.m1det, float)[j0:j1] / (1.0 + z0))
        if c % 10 == 0:
            print(f"  SEL batch {c+1}/{nsb}  ({time.time()-t0:.0f}s)")
    S = {k: np.concatenate(v) for k, v in acc.items()}
    print(f"SEL pass {time.time()-t0:.0f}s")

    fin = np.isfinite(S["ldw"])
    mx = S["ldw"][fin].max()
    wsel = np.where(fin, np.exp(S["ldw"] - mx), 0.0)
    log_mu = mx + np.log(wsel.sum()) - np.log(Ndraw)
    wsel = wsel / wsel.sum()
    sel = {"pop": float((wsel * S["pop"]).sum()),
           "rate": float((wsel * S["rate"]).sum()),
           "mass": float((wsel * (S["pop"] - S["rate"])).sum()),
           "pz": float((wsel * S["pz"]).sum()),
           "jac": float((wsel * S["jac"]).sum())}
    sel["tot"] = sel["pop"] + sel["pz"] + sel["jac"]
    print(f"\nANCHOR log_mu mine {log_mu:.10f} vs darksirens {spy0['log_mu']:.10f}  "
          f"diff {log_mu - spy0['log_mu']:.2e}")

    print("\n=== r by term and arm (per event) ===")
    hdr = (f"{'arm':>8} " + " ".join(f"{k:>12}" for k in
                                     ("pop", "-> rate", "-> mass", "pz", "jac", "TOTAL"))
           + f" {'d(C-A)mass':>12} {'sem':>10} {'ESS':>7}")
    print(hdr)
    table = {}
    for arm in arms:
        row = {k: float(ev[arm][k].mean() - sel[k]) for k in
               ("pop", "rate", "mass", "pz", "jac")}
        row["tot"] = row["pop"] + row["pz"] + row["jac"]
        row["ess_mean"] = float(ev_ess[arm].mean())
        row["sem_tot"] = float(ev[arm]["tot"].std(ddof=1) / np.sqrt(nobs))
        # PAIRED statistic: per-event (E_post[varsigma] - varsigma(truth)).  Zero
        # in expectation under a correct measurement model, and far less noisy
        # than either mean separately because the two are strongly correlated.
        dmass = ev[arm]["mass"] - truth_terms["mass"]
        row["CmA_mass"] = float(dmass.mean())
        row["CmA_mass_sem"] = float(dmass.std(ddof=1) / np.sqrt(nobs))
        dtot = (ev[arm]["pop"] + ev[arm]["pz"] + ev[arm]["jac"]
                - (truth_terms["pop"] + truth_terms["pz"] + truth_terms["jac"]))
        row["CmA_tot"] = float(dtot.mean())
        row["CmA_tot_sem"] = float(dtot.std(ddof=1) / np.sqrt(nobs))
        table[arm] = row
        print(f"{arm:>8} " + " ".join(f"{row[k]:12.4e}" for k in
                                      ("pop", "rate", "mass", "pz", "jac", "tot"))
              + f" {row['CmA_mass']:12.4e} {row['CmA_mass_sem']:10.2e}"
              f" {row['ess_mean']:7.0f}")
    print("\n=== truth-point split   r = (C - A) + (A - B) ===")
    print(f"{'term':>6} {'A truth':>13} {'B model':>13} {'A-B':>13} {'+-':>10} "
          f"{'C none':>13} {'C-A none':>13} {'C-A m1m2':>13}")
    split = {}
    for k in ("pop", "rate", "mass", "pz", "jac"):
        Cn = float(ev["none"][k].mean()); Cf = float(ev["m1m2"][k].mean())
        split[k] = {"A_truth": A[k], "A_sem": A_sem[k], "B_model": sel[k],
                    "A_minus_B": A[k] - sel[k],
                    "C_none": Cn, "C_minus_A_none": Cn - A[k],
                    "C_m1m2": Cf, "C_minus_A_m1m2": Cf - A[k]}
        print(f"{k:>6} {A[k]:13.5e} {sel[k]:13.5e} {A[k]-sel[k]:13.5e} "
              f"{A_sem[k]:10.2e} {Cn:13.5e} {Cn-A[k]:13.5e} {Cf-A[k]:13.5e}")
    print(f"  (pz at truth on resolved pixels only: {A_pz_ok:.5e})")

    print("\nevents' mean score by arm/term:")
    for arm in arms:
        print(f"  {arm:>6} " + "  ".join(
            f"{k}={ev[arm][k].mean():.6e}" for k in ("pop", "rate", "mass", "pz", "jac")))
    print("  selection " + "  ".join(f"{k}={sel[k]:.6e}" for k in
                                     ("pop", "rate", "mass", "pz", "jac")))

    # population sanity: events' TRUE m1src vs the model's predicted detected m1src
    edges = np.linspace(4.0, 82.0, 40)
    h_ev, _ = np.histogram(true_m1src, bins=edges)
    h_md, _ = np.histogram(S["m1src"], bins=edges, weights=wsel)
    h_ev = h_ev / h_ev.sum(); h_md = h_md / h_md.sum()
    chi2 = float(np.sum((h_ev - h_md) ** 2 / np.maximum(h_md, 1e-12)) * nobs)
    print(f"\nm1src (events truth) vs model-predicted detected: "
          f"chi2/dof = {chi2 / (len(edges) - 2):.3f}  "
          f"mean {true_m1src.mean():.3f} vs {(wsel * S['m1src']).sum():.3f}")

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        outdir / f"attr_mass_pe_{tag}.npz",
        true_z=true_z, true_m1src=true_m1src, true_m1det=true_m1det,
        obs_m1=obs_m1, sig_m1=sig_m1,
        **{f"ev_{arm}_{k}": ev[arm][k] for arm in ev for k in keys},
        **{f"truth_{k}": v for k, v in truth_terms.items()},
        **{f"ev_ess_{arm}": ev_ess[arm] for arm in ev},
        **{f"ev_zbar_{arm}": ev_zbar[arm] for arm in ev},
        **{f"ev_m1srcbar_{arm}": ev_m1srcbar[arm] for arm in ev},
        sel_z=S["z"].astype(np.float32), sel_m1src=S["m1src"].astype(np.float32),
        sel_w=wsel.astype(np.float32),
        m1src_edges=edges, h_ev=h_ev, h_model=h_md,
    )
    summary = {
        "name": "attr_mass_pe", "tag": tag, "seed": args.seed, "tracer": args.tracer,
        "H0": args.h0, "dh": args.dh, "nobs": nobs, "nsamp": nsamp,
        "gamma_fid": gamma_fid,
        "config": {"survey": survey, "gw": gw, "injections": inj,
                   "kde_window": args.kde_window},
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "anchor_log_mu_absdiff": float(abs(log_mu - spy0["log_mu"])),
        "selection_terms": sel,
        "event_terms": {arm: {k: float(ev[arm][k].mean()) for k in keys}
                        for arm in ev},
        "r_table": table,
        "truth_split": split,
        "truth_pz_resolved_pixels": A_pz_ok,
        "n_true_pixels_resolved": int(pix_ok.sum()),
        "m1src_chi2_per_dof": chi2 / (len(edges) - 2),
        "m1src_mean_events": float(true_m1src.mean()),
        "m1src_mean_model": float((wsel * S["m1src"]).sum()),
    }
    (outdir / f"attr_mass_pe_{tag}.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {outdir / f'attr_mass_pe_{tag}.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
