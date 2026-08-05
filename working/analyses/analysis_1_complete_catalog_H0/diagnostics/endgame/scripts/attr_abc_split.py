#!/usr/bin/env python3
"""ENDGAME -- the (A - B) / (C - A) split of the per-event score residual.

CLOSURE.md 14.4.  Under a correctly specified detected-set likelihood, for EVERY
function h of the source parameters,

    C  ==  mean_i E_post_i[h]        what the likelihood averages
    A  ==  mean_i h(theta_i^true)    the empirical detected-TRUTH mean
    B  ==  E_model-detected[h]       the model's detected-truth mean

satisfy E[C] = E[A] = B.  Taking h = varsigma = d ln p_target/dH0 makes

    r  =  <d ln Z_i/dH0>  -  d ln mu/dH0  =  C - B  =  (C - A) + (A - B)

exactly, and B = d ln mu/dH0 EXACTLY (differentiate mu = INT p_target P_det).
So the split needs no new estimator on the B side: the exact selection oracle
``attr_selmu_oracle.py`` already delivers B to 1e-10, which is what removes the
injection estimator's +-1.2e-4 common-mode Monte-Carlo error from the comparison.

  (A - B) != 0  =>  the mock's detected-TRUTH set is not a draw from the model
                    (generator event-draw bookkeeping)
  (C - A) != 0  =>  the posterior-averaging step -- the measurement model

This script computes all three on the analysis of record, using darksirens' own
operands (import-level pass-through capture; darksirens is READ-ONLY), and is
anchored on ``|Delta log mu| = 0`` in every run.

``--extra_truth`` additionally evaluates A on truth arrays that are NOT the record
events -- the untruncated replay from ``regen_events_notrunc.py`` -- grouped by
(replica, rank, batch), which is what turns (A - B) from a 3.2e-3 statistic into a
1e-4 one and makes the [:N_EVENTS] truncation directly testable.

Outputs: results/abc_<tag>.{json,npz}
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
BULK = Path("/hildafs/projects/phy220048p/magana/gws-agn-data/derived/"
            "analysis_1_complete_catalog_H0")

OM0_FID = 0.3075
H0_TRUE = 67.74
NUISANCE_DEFAULTS = {"delta": 0.0, "b_miss": 1.0, "sigma_kde": 0.0}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--dataroot", default=None,
                    help="Root holding seed<N>/ (default: working/data).  Used by "
                         "the v3 pilot, which lives in a separate tree until the "
                         "closure gate passes.")
    ap.add_argument("--tracer", choices=["gal", "agn"], default="gal")
    ap.add_argument("--events", default=None)
    ap.add_argument("--survey_override", default=None,
                    help="OPT-IN (ENDGAME): analyse a different survey block -- used "
                         "only for the declared-photo-z-kernel scan.")
    ap.add_argument("--injections", default="targeted")
    ap.add_argument("--h0", type=float, default=H0_TRUE)
    ap.add_argument("--dh", type=float, default=0.5)
    ap.add_argument("--log10n0", type=float, default=-24.0)
    ap.add_argument("--kde_window", type=int, default=None,
                    help="4096 for GAL (the analysis of record); unset for AGN.")
    ap.add_argument("--kde_window_nsigma", type=float, default=8.0)
    ap.add_argument("--pe_batch_events", type=int, default=25)
    ap.add_argument("--sel_batch", type=int, default=50000)
    ap.add_argument("--truth_batch", type=int, default=50000)
    ap.add_argument("--extra_truth", default=None,
                    help="HDF5 from regen_events_notrunc.py; A is also evaluated "
                         "on its truth arrays (host_type filtered to --tracer).")
    ap.add_argument("--extra_only", action="store_true",
                    help="Skip the PE and selection passes (A on --extra_truth only).")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--outdir", default=str(ROOT / "results"))
    return ap.parse_args(argv)


def _sem(x):
    x = np.asarray(x, float)
    return float(x.std(ddof=1) / np.sqrt(x.size)) if x.size > 1 else float("nan")


def main(argv=None):
    args = parse_args(argv)
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")

    import h5py
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
        selection_log_correction as _true_slc, DEFAULT_MAX_LIKELIHOOD_VARIANCE,
        log_evidence_and_mc_variance)
    from darksirens.redshift.prior import (
        prepare_redshift_prior_state, eval_redshift_prior_with_state)
    from darksirens.utils.cosmology import z_of_dL, ddL_of_z, dL_grid_bounds

    print(f"darksirens: {darksirens.__file__}   devices: {jax.devices()}")
    if args.kde_window is None and args.tracer == "gal":
        args.kde_window = 4096
    if args.kde_window is not None:
        from darksirens.redshift.catalog import configure_catalog_kde_window
        configure_catalog_kde_window(size=int(args.kde_window),
                                     n_sigma=float(args.kde_window_nsigma))
        print(f"[kde] window W={args.kde_window} n_sigma={args.kde_window_nsigma}")

    sd = (Path(args.dataroot) if args.dataroot else DATA) / f"seed{args.seed}"
    survey = (args.survey_override if args.survey_override
              else str(sd / "surveys" / f"survey_{args.tracer}_complete_ns32.h5"))
    if args.events:
        gw = args.events
    elif args.dataroot:
        gw = str(Path(args.dataroot) / f"seed{args.seed}"
                 / f"events_{args.tracer}_hosted.h5")
    elif args.seed == 100:
        gw = str(ROOT / "data_derived" / f"events_{args.tracer}_hosted.h5")
    else:
        gw = str(BULK / f"seed{args.seed}" / f"events_{args.tracer}_hosted.h5")
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
    pop_fid = get_fixed_population_params(opts.pop_model, shared_beta=True,
                                          shared_spin=True, shared_gamma=True)
    point = dict(NUISANCE_DEFAULTS)
    point["H0"] = args.h0
    point["log10n0"] = args.log10n0
    missing = [l for l in labels if l not in point]
    if missing:
        raise SystemExit(f"[fatal] missing label values: {missing}")
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
          f"log_mu={spy0['log_mu']:.6f}  Neff={spy0['Neff']:.3e}")

    a = cap["args"]
    cosmo0, survey_p, pop_params, gw_pe, cat_pe, gw_sel, cat_sel = a[:7]
    Ndraw = float(a[9])
    gamma_fid = float(np.asarray(pop_params)[-1])
    print(f"operands: nPE={gw_pe.dL.shape[0]} nSEL={gw_sel.dL.shape[0]} "
          f"Ndraw={Ndraw:.4g}  gamma_fid={gamma_fid}")

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
            ldw = jnp.where(ok, ldw, -jnp.inf)
            return ldw, lp_pop, lp_z, ljac, z
        return pieces

    H0s = [args.h0 - args.dh, args.h0, args.h0 + args.dh]
    t0 = time.time()
    fns = [make_pieces(h) for h in H0s]
    print(f"prior states built ({time.time()-t0:.1f}s)")
    twodh = 2.0 * args.dh

    def terms(out):
        """(ldw_central, dpop, drate, dpz, dsigma_jac, z_central) as numpy."""
        dpop = (np.asarray(out[2][1]) - np.asarray(out[0][1])) / twodh
        dpz = (np.asarray(out[2][2]) - np.asarray(out[0][2])) / twodh
        djac = (np.asarray(out[2][3]) - np.asarray(out[0][3])) / twodh
        zl = np.log1p(np.asarray(out[0][4])); zp = np.log1p(np.asarray(out[2][4]))
        drate = (gamma_fid - 1.0) * (zp - zl) / twodh
        return (np.asarray(out[1][0]), np.nan_to_num(dpop), np.nan_to_num(drate),
                np.nan_to_num(dpz), -np.nan_to_num(djac), np.asarray(out[1][4]))

    KEYS = ("pop", "rate", "mass", "pz", "jac", "tot")
    out_json = {"name": "attr_abc_split", "tag": tag, "seed": args.seed,
                "tracer": args.tracer, "H0": args.h0, "dh": args.dh,
                "gamma_fid": gamma_fid, "nobs": nobs, "nsamp": nsamp,
                "config": {"survey": survey, "gw": gw, "injections": inj,
                           "kde_window": args.kde_window},
                "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "darksirens_file": darksirens.__file__}
    npz = {}

    # ------------------------------------------------------------------ C ------
    if not args.extra_only:
        _red = jax.jit(jax.vmap(lambda row: log_evidence_and_mc_variance(row, nsamp)))
        nblk = args.pe_batch_events
        nchunk = int(np.ceil(nobs / nblk))
        C = {k: np.zeros(nobs) for k in KEYS}
        ev_lnZ = np.zeros(nobs); ev_lnZ_pm = np.zeros((nobs, 2))
        ev_ess = np.zeros(nobs); ev_zbar = np.zeros(nobs)
        t0 = time.time()
        for c in range(nchunk):
            i0, i1 = c * nblk, min((c + 1) * nblk, nobs)
            s0, s1 = i0 * nsamp, i1 * nsamp
            sl = lambda arr: jnp.asarray(arr)[s0:s1]
            arg = (sl(gw_pe.m1det), sl(gw_pe.q), sl(gw_pe.dL), sl(gw_pe.chieff),
                   sl(gw_pe.pixels), sl(gw_pe.prior_wt), sl(gw_pe.valid))
            out = [f(*arg) for f in fns]
            m = i1 - i0
            ldw, dpop, drate, dpz, djac, zc = terms(out)
            ldw = ldw.reshape(m, nsamp)
            lnZ, _ = _red(out[1][0].reshape(m, nsamp))
            ev_lnZ[i0:i1] = np.asarray(lnZ)
            for k, kk in ((0, 0), (2, 1)):
                lnZk, _ = _red(out[k][0].reshape(m, nsamp))
                ev_lnZ_pm[i0:i1, kk] = np.asarray(lnZk)
            w = np.exp(ldw - ldw.max(axis=1, keepdims=True))
            w = np.where(np.isfinite(ldw), w, 0.0)
            w = w / w.sum(axis=1, keepdims=True)
            ev_ess[i0:i1] = 1.0 / (w ** 2).sum(axis=1)
            ev_zbar[i0:i1] = (w * zc.reshape(m, nsamp)).sum(axis=1)
            C["pop"][i0:i1] = (w * dpop.reshape(m, nsamp)).sum(axis=1)
            C["rate"][i0:i1] = (w * drate.reshape(m, nsamp)).sum(axis=1)
            C["pz"][i0:i1] = (w * dpz.reshape(m, nsamp)).sum(axis=1)
            C["jac"][i0:i1] = (w * djac.reshape(m, nsamp)).sum(axis=1)
            if c % 10 == 0:
                print(f"  PE chunk {c+1}/{nchunk}  ({time.time()-t0:.0f}s)", flush=True)
        C["mass"] = C["pop"] - C["rate"]
        C["tot"] = C["pop"] + C["pz"] + C["jac"]
        C_fd = (ev_lnZ_pm[:, 1] - ev_lnZ_pm[:, 0]) / twodh
        print(f"PE pass {time.time()-t0:.0f}s  sum lnZ={ev_lnZ.sum():.6f}")
        npz.update({f"C_{k}": C[k] for k in KEYS})
        npz.update({"C_fd": C_fd, "ev_ess": ev_ess, "ev_zbar": ev_zbar,
                    "ev_lnZ": ev_lnZ})

    # ------------------------------------------------------------------ A ------
    with h5py.File(gw, "r") as f:
        T = {k: np.asarray(f["truth"][k][:], float) for k in
             ("m1det", "m2det", "dl", "chieff", "ra", "dec", "z")}
        host_type_rec = np.asarray(f["truth"]["host_type"][:], int)

    import healpy as hp
    up = np.asarray(cat_pe.unique_pixels)
    order = np.argsort(up)

    def pix_rows(ra, dec):
        gpix = hp.ang2pix(32, np.pi / 2.0 - dec, ra)
        loc = np.clip(np.searchsorted(up, gpix, sorter=order), 0, up.size - 1)
        row = order[loc]
        ok = up[row] == gpix
        return np.where(ok, row, 0).astype(np.int32), ok

    def eval_truth(m1det, m2det, dl, chieff, ra, dec, batch=None):
        n = m1det.size
        bs = batch or args.truth_batch
        prow, ok = pix_rows(ra, dec)
        acc = {k: np.zeros(n) for k in ("pop", "rate", "pz", "jac")}
        zc = np.zeros(n)
        t0 = time.time()
        for i0 in range(0, n, bs):
            i1 = min(i0 + bs, n)
            arg = (jnp.asarray(m1det[i0:i1]),
                   jnp.asarray(m2det[i0:i1] / m1det[i0:i1]),
                   jnp.asarray(dl[i0:i1]), jnp.asarray(chieff[i0:i1]),
                   jnp.asarray(prow[i0:i1]),
                   jnp.ones(i1 - i0), jnp.ones(i1 - i0, dtype=bool))
            o = [f(*arg) for f in fns]
            _, dpop, drate, dpz, djac, z0 = terms(o)
            acc["pop"][i0:i1] = dpop; acc["rate"][i0:i1] = drate
            acc["pz"][i0:i1] = dpz; acc["jac"][i0:i1] = djac
            zc[i0:i1] = z0
            if (i0 // bs) % 5 == 0:
                print(f"  truth {i1}/{n} ({time.time()-t0:.0f}s)", flush=True)
        acc["mass"] = acc["pop"] - acc["rate"]
        acc["tot"] = acc["pop"] + acc["pz"] + acc["jac"]
        return acc, zc, ok

    A, A_z, A_ok = eval_truth(T["m1det"], T["m2det"], T["dl"], T["chieff"],
                              T["ra"], T["dec"])
    print(f"record truth pass done; pixels resolved {int(A_ok.sum())}/{A_ok.size}")
    npz.update({f"A_{k}": A[k] for k in KEYS})
    npz["A_pix_ok"] = A_ok
    out_json["A"] = {k: {"mean": float(A[k].mean()), "sem": _sem(A[k]),
                         "sd": float(A[k].std(ddof=1))} for k in KEYS}
    out_json["n_pix_resolved"] = int(A_ok.sum())

    # ------------------------------------------------------------------ B ------
    if not args.extra_only:
        nsel = int(gw_sel.dL.shape[0])
        sb = args.sel_batch
        parts = {k: [] for k in ("ldw", "ldwm", "ldwp", "pop", "rate", "pz", "jac")}
        t0 = time.time()
        for c in range(int(np.ceil(nsel / sb))):
            j0, j1 = c * sb, min((c + 1) * sb, nsel)
            sl = lambda arr: jnp.asarray(arr)[j0:j1]
            arg = (sl(gw_sel.m1det), sl(gw_sel.q), sl(gw_sel.dL), sl(gw_sel.chieff),
                   sl(gw_sel.pixels), sl(gw_sel.prior_wt), sl(gw_sel.valid))
            o = [f(*arg) for f in fns]
            ldw, dpop, drate, dpz, djac, _ = terms(o)
            parts["ldw"].append(ldw)
            parts["ldwm"].append(np.asarray(o[0][0]))
            parts["ldwp"].append(np.asarray(o[2][0]))
            parts["pop"].append(dpop); parts["rate"].append(drate)
            parts["pz"].append(dpz); parts["jac"].append(djac)
            if c % 10 == 0:
                print(f"  SEL batch {c+1}  ({time.time()-t0:.0f}s)", flush=True)
        S = {k: np.concatenate(v) for k, v in parts.items()}
        print(f"SEL pass {time.time()-t0:.0f}s")

        def _lse(x):
            fin = np.isfinite(x)
            m = x[fin].max()
            return m + np.log(np.exp(np.where(fin, x - m, -np.inf)).sum())

        log_mu = _lse(S["ldw"]) - np.log(Ndraw)
        dlnmu_fd = (_lse(S["ldwp"]) - _lse(S["ldwm"])) / twodh
        fin = np.isfinite(S["ldw"])
        w = np.where(fin, np.exp(S["ldw"] - S["ldw"][fin].max()), 0.0)
        w = w / w.sum()
        B = {k: float((w * S[k]).sum()) for k in ("pop", "rate", "pz", "jac")}
        B["mass"] = B["pop"] - B["rate"]
        B["tot"] = B["pop"] + B["pz"] + B["jac"]
        Neff_inj = float(1.0 / (w ** 2).sum())
        print(f"ANCHOR log_mu mine {log_mu:.10f} vs darksirens "
              f"{spy0['log_mu']:.10f}  diff {log_mu - spy0['log_mu']:.3e}")
        out_json["anchor_log_mu_absdiff"] = float(abs(log_mu - spy0["log_mu"]))
        out_json["B_inj"] = B
        out_json["B_inj_fd"] = float(dlnmu_fd)
        out_json["Neff_inj"] = Neff_inj
        out_json["C"] = {k: {"mean": float(C[k].mean()), "sem": _sem(C[k])}
                         for k in KEYS}
        out_json["C_fd_mean"] = float(C_fd.mean())
        out_json["C_fd_sem"] = _sem(C_fd)
        out_json["r_inj"] = {k: float(C[k].mean() - B[k]) for k in KEYS}
        out_json["r_inj_fd"] = float(C_fd.mean() - dlnmu_fd)
        out_json["split_inj"] = {
            k: {"A": float(A[k].mean()), "A_sem": _sem(A[k]),
                "B": B[k], "A_minus_B": float(A[k].mean() - B[k]),
                "C": float(C[k].mean()), "C_sem": _sem(C[k]),
                "C_minus_A": float((C[k] - A[k]).mean()),
                "C_minus_A_sem": _sem(C[k] - A[k])} for k in KEYS}
        print("\n=== r = (C - A) + (A - B)   [B from the injection estimator] ===")
        print(f"{'term':>6} {'A':>13} {'B':>13} {'A-B':>13} {'sem':>10} "
              f"{'C':>13} {'C-A':>13} {'sem':>10} {'r=C-B':>13}")
        for k in KEYS:
            s = out_json["split_inj"][k]
            print(f"{k:>6} {s['A']:13.5e} {s['B']:13.5e} {s['A_minus_B']:13.5e} "
                  f"{s['A_sem']:10.2e} {s['C']:13.5e} {s['C_minus_A']:13.5e} "
                  f"{s['C_minus_A_sem']:10.2e} {s['C']-s['B']:13.5e}")

    # -------------------------------------------------------- extra truth ------
    if args.extra_truth:
        want = 1 if args.tracer == "agn" else 0
        with h5py.File(args.extra_truth, "r") as f:
            ht = np.asarray(f["host_type"][:], int)
            m = ht == want
            E = {k: np.asarray(f[k][:], float)[m] for k in
                 ("m1det", "m2det", "dl", "chieff", "ra", "dec", "z")}
            G = {k: np.asarray(f[k][:])[m] for k in ("replica", "rank", "batch", "slot")}
            info = json.loads(f.attrs["info_json"])
        print(f"\nextra truth: {m.sum():,} of {m.size:,} rows with host_type={want}")
        EA, Ez, Eok = eval_truth(E["m1det"], E["m2det"], E["dl"], E["chieff"],
                                 E["ra"], E["dec"])
        npz.update({f"X_{k}": EA[k] for k in KEYS})
        npz.update({f"Xg_{k}": G[k] for k in G})
        npz["X_z"] = E["z"]; npz["X_pix_ok"] = Eok
        out_json["extra_truth"] = {
            "path": args.extra_truth, "n_rows": int(m.sum()),
            "n_pix_resolved": int(Eok.sum()),
            "source_info": {k: v for k, v in info.items() if k != "per_replica"},
        }
        head = G["rank"] < 1000
        groups = {"head_kept": head, "tail_withheld": ~head,
                  "full": np.ones_like(head)}
        gs = {}
        for name, sel in groups.items():
            gs[name] = {k: {"mean": float(EA[k][sel].mean()),
                            "sem": _sem(EA[k][sel]),
                            "n": int(sel.sum())} for k in KEYS}
        out_json["extra_truth"]["groups"] = gs
        print(f"{'group':>15} {'n':>9} {'A_tot':>13} {'sem':>11}")
        for name in ("head_kept", "tail_withheld", "full"):
            g = gs[name]["tot"]
            print(f"{name:>15} {g['n']:9d} {g['mean']:13.6e} {g['sem']:11.3e}")

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(outdir / f"abc_{tag}.npz", **npz)
    (outdir / f"abc_{tag}.json").write_text(json.dumps(out_json, indent=2))
    print(f"\nWrote {outdir / f'abc_{tag}.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
