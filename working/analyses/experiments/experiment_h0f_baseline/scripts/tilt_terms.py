#!/usr/bin/env python3
"""Standalone term-by-term rebuild of the K=2 GLASS baseline likelihood vs H0.

Reconstructs, OUTSIDE the monolithic jit likelihood but from the SAME darksirens
primitives (z_of_dL, log_jacobian, pop_model_parser, prepare/eval redshift prior
in field mode), every per-sample component of

    ldw = log p_pop(m1src,q,z,chieff) - log J - log prior_wt
          + logsumexp_k[ ln a_k + log p_z,k(z|pix) ]        (K=2, a=[1-f, f])

for both the per-event numerator (PE samples) and the selection integral
(injections), on an H0 grid.  Because the components are held separately, the
script also evaluates counterfactual variants that the production likelihood
cannot express:

  numerator + selection variants
  - full          : the production model (validated against
                    results/h0_decomposition_*.json curves)
  - zcut_<c>      : catalog support masked at z <= c (samples mapping beyond c
                    get -inf) -- the "detection-horizon leak" family
  - frozen_mass   : p_pop and the Jacobian pinned at their H0_true values;
                    only the catalog redshift prior responds to H0
  - frozen_cat    : the catalog prior pinned at H0_true; only mass/Jacobian
                    respond (the spectral-siren channel)
  - gal_only/agn_only : single-tracer priors (f=0 / f=1) on the same samples

  per-event diagnostics (full variant)
  - lnZ_i(H0), delta-method sigma^2_i(H0)  (Essick&Farr MC-bias correction)
  - weighted PE mass fraction beyond z in {1.0, 1.2, 1.4, 1.45}
  - AGN-branch posterior weight share

The redshift-prior state is built once at H0_true: in field mode with
log10n0=-12 the state's only H0 dependence is (i) the (c/H0)^3 amplitude of
g(z), which cancels between log_g_front and the per-kernel norms Z_i, and
(ii) the negligible n0-scaled missing branch.  This is verified numerically
(state rebuilt at H0=60, max |dlogp| reported).

Outputs <out>.h5 with all curves + per-event matrices, and prints a validation
table against the measured decomposition curves.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("DARKSIRENS_ZMAX", "1.5")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gw_path", required=True)
    ap.add_argument("--sel_path", required=True)
    ap.add_argument("--survey_paths", nargs=2, required=True,
                    metavar=("GAL", "AGN"))
    ap.add_argument("--f", type=float, required=True, help="fcat_2 (AGN weight)")
    ap.add_argument("--h0_grid", nargs=3, type=float, default=[55.0, 80.0, 51])
    ap.add_argument("--h0_true", type=float, default=67.74)
    ap.add_argument("--om0", type=float, default=0.3075)
    ap.add_argument("--zcuts", nargs="+", type=float,
                    default=[1.0, 1.1, 1.2, 1.3, 1.4])
    ap.add_argument("--chunk", type=int, default=200_000)
    ap.add_argument("--skip_pe", action="store_true",
                    help="Selection side only (lnmu variants); PE arrays still "
                         "loaded for metadata but not scanned.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--decomposition_json", default=None,
                    help="Measured h0_decomposition_*.json to validate against.")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    import h5py
    import healpy as hp
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from jax.scipy.special import logsumexp as jlse
    from scipy.special import logsumexp as slse

    from darksirens.core.types import CosmoParams, EMCatalog, SurveyParams
    from darksirens.redshift.prior import (
        prepare_redshift_prior_state, eval_redshift_prior_with_state)
    from darksirens.redshift.completion import build_field_normalization_inputs
    from darksirens.redshift.grid import zgrid
    from darksirens.utils.cosmology import z_of_dL, dL_grid_bounds
    from darksirens.inference.utils import log_jacobian_m1src_q_z_to_m1det_q_dL
    from darksirens.gw.populations import (
        get_fixed_population_params, pop_model_parser)

    print(f"jax devices: {jax.devices()}  zgrid max: {float(zgrid[-1])}")
    NSIDE = 64
    apix = hp.nside2pixarea(NSIDE)

    # ---------------- catalogs -> EMCatalog (full-sky rows, field inputs) -----
    def load_catalog(path):
        with h5py.File(path, "r") as f:
            zg = f["zgals"][:]; dzg = f["dzgals"][:]
            wg = f["wgals"][:]; ng = f["ngals"][:]
            assert int(f.attrs["nside"]) == NSIDE
        fni = build_field_normalization_inputs(zg, wg, ng)
        cat = EMCatalog(
            apix=apix,
            zgals=jnp.asarray(zg), dzgals=jnp.asarray(dzg),
            wgals=jnp.asarray(wg), ngals=jnp.asarray(ng, dtype=jnp.int32),
            delta_g_pix_z=jnp.zeros((1, zgrid.shape[0])),
            dN_obs_kde=None, pixel_to_cache_idx=None,
            field_dN_obs_s=jnp.asarray(fni.dN_obs_s),
            field_n_empty=jnp.asarray(float(fni.n_empty)),
            field_N_obs_total=jnp.asarray(float(fni.N_obs_total)),
            field_occupied_pixels=jnp.asarray(fni.occupied_pixels,
                                              dtype=jnp.int32),
        )
        return cat

    t0 = time.time()
    cats = [load_catalog(p) for p in args.survey_paths]
    print(f"catalogs loaded+field inputs: {time.time()-t0:.1f}s")

    survey = SurveyParams(
        n0=10.0 ** -12.0, z50=1.0, w=0.5, delta=0.0, b_miss=1.0,
        alpha_miss=1.0, sigma_kde=0.0, complete_empty_pixel_policy=0,
    )
    cosmo0 = CosmoParams(H0=args.h0_true, Om0=args.om0, w0=-1.0, wa=0.0)

    t0 = time.time()
    states = [prepare_redshift_prior_state(
        "dark_sirens", cosmo0, survey, c, catalog_sky_weighting="field")
        for c in cats]
    print(f"prior states: {time.time()-t0:.1f}s  "
          f"logZglobal={[float(s.log_Z_global) for s in states]}")

    pop_params = jnp.asarray(get_fixed_population_params(
        "powerlaw+peak", shared_beta=True, shared_spin=True, shared_gamma=True))
    log_p_pop = pop_model_parser(pop_model="powerlaw+peak", shared_beta=True,
                                 shared_spin=True, shared_gamma=True)

    # ---------------- sample sets ---------------------------------------------
    def load_pe(path):
        with h5py.File(path, "r") as f:
            d = {k: f[k][:] for k in
                 ("m1det", "m2det", "dL", "chieff", "p_pe", "ra", "dec")}
            host_type = f["host_type"][:]
            true_z = f["true_z"][:]
            nobs = int(f.attrs["nobs"]); nsamp = int(f.attrs["nsamp"])
        pix = hp.ang2pix(NSIDE, np.pi / 2 - d["dec"], d["ra"]).astype(np.int32)
        return dict(m1det=d["m1det"], q=d["m2det"] / d["m1det"], dL=d["dL"],
                    chieff=d["chieff"], pw=d["p_pe"], pix=pix,
                    nobs=nobs, nsamp=nsamp, host_type=host_type, true_z=true_z)

    def load_sel(path):
        with h5py.File(path, "r") as f:
            d = {k: f[k][:] for k in
                 ("m1det", "m2det", "dL", "chieff", "pdraw", "ra", "dec")}
            ndraw = float(f.attrs["ndraw"])
        pix = hp.ang2pix(NSIDE, np.pi / 2 - d["dec"], d["ra"]).astype(np.int32)
        return dict(m1det=d["m1det"], q=d["m2det"] / d["m1det"], dL=d["dL"],
                    chieff=d["chieff"], pw=d["pdraw"], pix=pix, ndraw=ndraw)

    pe = load_pe(args.gw_path)
    sel = load_sel(args.sel_path)
    print(f"PE: {pe['nobs']} events x {pe['nsamp']}  "
          f"SEL: {sel['m1det'].size} detected, Ndraw={sel['ndraw']:.0f}")

    # ---------------- jitted per-chunk component evaluator --------------------
    # returns z, supported, lp_pop - ljac (pop+jac), lp_z per catalog
    def make_comp_fn():
        @jax.jit
        def comp(H0, m1det, q, dL, chieff, pix):
            cosmo = CosmoParams(H0=H0, Om0=args.om0, w0=-1.0, wa=0.0)
            dL_lo, dL_hi = dL_grid_bounds(H0, args.om0, -1.0, 0.0)
            supported = (dL >= dL_lo) & (dL <= dL_hi)
            dL_c = jnp.clip(dL, dL_lo, dL_hi)
            z = z_of_dL(dL_c, H0, args.om0, -1.0, 0.0)
            m1src = m1det / (1.0 + z)
            lpop = log_p_pop(m1src, q, z, chieff, pop_params)
            ljac = log_jacobian_m1src_q_z_to_m1det_q_dL(
                z, dL_c, H0, args.om0, -1.0, 0.0)
            lpz = [eval_redshift_prior_with_state(
                "dark_sirens", states[k], z, pix, cosmo, survey, cats[k],
                catalog_sky_weighting="field") for k in range(2)]
            base_pj = lpop - ljac
            return z, supported, base_pj, lpz[0], lpz[1]
        return comp

    comp_fn = make_comp_fn()

    def eval_all(H0, ds):
        """Return per-sample components (numpy float64) for dataset ds."""
        n = ds["m1det"].size
        out = {k: np.empty(n) for k in ("z", "base_pj", "lpz1", "lpz2")}
        sup = np.empty(n, dtype=bool)
        C = args.chunk
        for s in range(0, n, C):
            e = min(s + C, n)
            m = e - s
            if m < C:  # pad to keep one jit shape
                pad = C - m
                sl = lambda a: np.concatenate([a[s:e], a[e - 1:e].repeat(pad)])
            else:
                sl = lambda a: a[s:e]
            z, su, bp, l1, l2 = comp_fn(
                jnp.asarray(float(H0)), jnp.asarray(sl(ds["m1det"])),
                jnp.asarray(sl(ds["q"])), jnp.asarray(sl(ds["dL"])),
                jnp.asarray(sl(ds["chieff"])),
                jnp.asarray(sl(ds["pix"])))
            out["z"][s:e] = np.asarray(z)[:m]
            sup[s:e] = np.asarray(su)[:m]
            out["base_pj"][s:e] = np.asarray(bp)[:m]
            out["lpz1"][s:e] = np.asarray(l1)[:m]
            out["lpz2"][s:e] = np.asarray(l2)[:m]
        out["sup"] = sup
        return out

    # ---------------- state H0-invariance check -------------------------------
    states_check = [prepare_redshift_prior_state(
        "dark_sirens", CosmoParams(H0=60.0, Om0=args.om0, w0=-1.0, wa=0.0),
        survey, c, catalog_sky_weighting="field") for c in cats]
    ztest = jnp.linspace(0.05, 1.49, 4000)
    ptest = jnp.asarray(pe["pix"][:4000])
    for k in range(2):
        a = eval_redshift_prior_with_state(
            "dark_sirens", states[k], ztest, ptest, cosmo0, survey, cats[k],
            catalog_sky_weighting="field")
        b = eval_redshift_prior_with_state(
            "dark_sirens", states_check[k], ztest, ptest, cosmo0, survey,
            cats[k], catalog_sky_weighting="field")
        an, bn = np.asarray(a), np.asarray(b)
        fin = np.isfinite(an) & np.isfinite(bn)
        # Differences live ONLY in the n0=1e-12-suppressed missing branch
        # (absolute log-density ~ -27) and in deep inter-kernel gaps; weight
        # them by exp(logp) relative to the max to measure what could actually
        # move a logsumexp.  The decisive check is the validation against the
        # measured decomposition curves below.
        d = np.abs(an - bn)[fin]
        rel_w = np.exp(an[fin] - np.max(an[fin]))
        weighted = float(np.max(rel_w * d))
        print(f"state H0-invariance cat{k+1}: max|dlogp|={float(d.max()):.2e} "
              f"weighted-by-relative-prob max={weighted:.3e}")
    del states_check

    # ---------------- grids & variant machinery -------------------------------
    H0s = np.linspace(args.h0_grid[0], args.h0_grid[1],
                      int(round(args.h0_grid[2])))
    lnf = np.log(args.f)
    ln1mf = np.log1p(-args.f)
    ln_ppe = {"pe": np.log(pe["pw"]), "sel": np.log(sel["pw"])}

    def mix(l1, l2):
        return np.logaddexp(ln1mf + l1, lnf + l2)

    # components at H0_true (frozen references)
    if args.skip_pe:
        comps0 = {"sel": eval_all(args.h0_true, sel)}
    else:
        comps0 = {"pe": eval_all(args.h0_true, pe),
                  "sel": eval_all(args.h0_true, sel)}
    for key in comps0:
        c0 = comps0[key]
        c0["mix0"] = mix(c0["lpz1"], c0["lpz2"])
        c0["base0"] = c0["base_pj"] - ln_ppe[key]

    zcuts = list(args.zcuts)
    var_names = (["full"] + [f"zcut_{c:g}" for c in zcuts]
                 + ["frozen_mass", "frozen_cat", "gal_only", "agn_only"])

    nev, ns = pe["nobs"], pe["nsamp"]
    nH = H0s.size
    res = {
        "numerator": {v: np.empty(nH) for v in var_names},
        "lnmu": {v: np.empty(nH) for v in var_names},
        "lnZ_ev": np.empty((nev, nH)),
        "sigma2_ev": np.empty((nev, nH)),
        "agn_share_ev": np.empty((nev, nH)),
        "frac_beyond": {c: np.empty((nev, nH)) for c in (1.0, 1.2, 1.4, 1.45)},
        "clip_frac": np.empty(nH),          # PE weight-frac unsupported (z>1.5)
        "sel_neff": np.empty(nH),
    }

    def per_event_reduce(ldw):
        L = ldw.reshape(nev, ns)
        lnZ = slse(L, axis=1) - np.log(ns)
        # delta-method variance: sum w^2/(sum w)^2 - 1/n
        l2 = slse(2.0 * L, axis=1)
        s2 = np.exp(l2 - 2.0 * (lnZ + np.log(ns))) - 1.0 / ns
        return lnZ, np.maximum(s2, 0.0)

    t_all = time.time()
    pairs = ((("sel", "lnmu"),) if args.skip_pe
             else (("pe", "numerator"), ("sel", "lnmu")))
    for i, H0 in enumerate(H0s):
        c = ({"sel": eval_all(H0, sel)} if args.skip_pe
             else {"pe": eval_all(H0, pe), "sel": eval_all(H0, sel)})
        for key, tgt in pairs:
            cc = c[key]
            base = cc["base_pj"] - ln_ppe[key]
            mixed = mix(cc["lpz1"], cc["lpz2"])
            neg = ~cc["sup"]

            def total(ldw):
                ldw = np.where(neg, -np.inf, ldw)
                ldw = np.where(np.isfinite(ldw), ldw, -np.inf)
                if key == "pe":
                    lnZ, s2 = per_event_reduce(ldw)
                    return lnZ.sum(), (lnZ, s2, ldw)
                lw = ldw - np.log(sel["ndraw"])
                lm = slse(lw)
                ne = np.exp(2 * lm - slse(2 * lw))
                return lm, ne

            variants = {"full": base + mixed}
            for zc in zcuts:
                variants[f"zcut_{zc:g}"] = np.where(
                    cc["z"] > zc, -np.inf, base + mixed)
            variants["frozen_mass"] = comps0[key]["base0"] + mixed
            variants["frozen_cat"] = base + comps0[key]["mix0"]
            variants["gal_only"] = base + cc["lpz1"]
            variants["agn_only"] = base + cc["lpz2"]

            for v, ldw in variants.items():
                val, extra = total(ldw)
                res[tgt][v][i] = val
                if key == "pe" and v == "full":
                    lnZ, s2, ldw_full = extra
                    res["lnZ_ev"][:, i] = lnZ
                    res["sigma2_ev"][:, i] = s2
                    # weighted diagnostics from the full ldw
                    L = ldw_full.reshape(nev, ns)
                    ln_norm = slse(L, axis=1)
                    W = np.exp(L - ln_norm[:, None])
                    zz = cc["z"].reshape(nev, ns)
                    for zc in res["frac_beyond"]:
                        res["frac_beyond"][zc][:, i] = np.sum(
                            W * (zz > zc), axis=1)
                    # AGN branch share
                    share = np.exp(
                        (lnf + cc["lpz2"]) - mixed)
                    share = np.where(np.isfinite(share), share, 0.0)
                    res["agn_share_ev"][:, i] = np.sum(
                        W * share.reshape(nev, ns), axis=1)
                    res["clip_frac"][i] = float(np.mean(neg))
                if key == "sel" and v == "full":
                    res["sel_neff"][i] = extra
        if i % 5 == 0 or i == nH - 1:
            numtxt = ("" if args.skip_pe
                      else f"numer={res['numerator']['full'][i]:.3f} ")
            print(f"[{i+1}/{nH}] H0={H0:6.2f} {numtxt}"
                  f"lnmu={res['lnmu']['full'][i]:+.5f} "
                  f"({(time.time()-t_all)/(i+1):.1f}s/pt)", flush=True)

    # ---------------- validation against measured decomposition ---------------
    validation = {}
    if args.decomposition_json:
        meas = json.loads(Path(args.decomposition_json).read_text())
        Hm = np.array(meas["H0_grid"])
        num_m = np.array(meas["per_event_numerator"])
        mu_m = np.array(meas["log_mu"])
        keep = (Hm >= H0s.min() - 1e-9) & (Hm <= H0s.max() + 1e-9)
        Hm, num_m, mu_m = Hm[keep], num_m[keep], mu_m[keep]
        num_r = (np.zeros_like(Hm) if args.skip_pe
                 else np.interp(Hm, H0s, res["numerator"]["full"]))
        mu_r = np.interp(Hm, H0s, res["lnmu"]["full"])
        # compare SHAPES (curves relative to value at truth)
        def shape_err(model, measured):
            i0 = int(np.argmin(np.abs(Hm - args.h0_true)))
            d = (model - model[i0]) - (measured - measured[i0])
            return float(np.max(np.abs(d))), float(np.max(np.abs(measured - measured[i0])))
        e_mu, r_mu = shape_err(mu_r, mu_m)
        validation = {
            "lnmu_shape_maxerr_nats": e_mu,
            "lnmu_shape_range_nats": r_mu,
        }
        if not args.skip_pe:
            e_num, r_num = shape_err(num_r, num_m)
            validation.update({
                "numerator_shape_maxerr_nats": e_num,
                "numerator_shape_range_nats": r_num,
                "numerator_offset_at_truth": float(
                    num_r[int(np.argmin(np.abs(Hm - args.h0_true)))]
                    - num_m[int(np.argmin(np.abs(Hm - args.h0_true)))]),
            })
        print("\n=== validation vs", args.decomposition_json)
        for k, v in validation.items():
            print(f"  {k}: {v:.4f}")

    # ---------------- write ---------------------------------------------------
    import h5py
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(outp, "w") as f:
        f.create_dataset("H0_grid", data=H0s)
        for v in var_names:
            if not args.skip_pe:
                f.create_dataset(f"numerator/{v}", data=res["numerator"][v])
            f.create_dataset(f"lnmu/{v}", data=res["lnmu"][v])
        if not args.skip_pe:
            f.create_dataset("lnZ_ev", data=res["lnZ_ev"])
            f.create_dataset("sigma2_ev", data=res["sigma2_ev"])
            f.create_dataset("agn_share_ev", data=res["agn_share_ev"])
            for zc, arr in res["frac_beyond"].items():
                f.create_dataset(f"frac_beyond/{zc:g}", data=arr)
            f.create_dataset("clip_frac", data=res["clip_frac"])
        f.create_dataset("sel_neff", data=res["sel_neff"])
        f.create_dataset("host_type", data=pe["host_type"])
        f.create_dataset("true_z", data=pe["true_z"])
        f.attrs["f"] = args.f
        f.attrs["h0_true"] = args.h0_true
        f.attrs["gw_path"] = args.gw_path
        f.attrs["sel_path"] = args.sel_path
        f.attrs["survey_paths"] = json.dumps(args.survey_paths)
        f.attrs["nobs"] = nev
        f.attrs["nsamp"] = ns
        f.attrs["ndraw"] = sel["ndraw"]
        f.attrs["validation"] = json.dumps(validation)
    print(f"\nWrote {outp}  ({time.time()-t_all:.0f}s eval)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
