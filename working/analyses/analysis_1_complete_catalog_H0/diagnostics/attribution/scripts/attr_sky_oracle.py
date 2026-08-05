#!/usr/bin/env python3
"""The EXACT HOST-GALAXY sky oracle -- does the nside-32 pixelisation carry the rest?

ATTRIBUTION.md A4.5 left exactly two approximations standing between the fully
exact per-event measurement model and the generative truth.  One of them, the RA
measurement width, is now FIXED in the generator (convention (b2), 2026-08-01).
The other is the one this script measures:

    the likelihood models the catalog as ``p_z(z | pix)`` -- sky and redshift
    INDEPENDENT inside an nside-32 pixel (1.83 deg) -- while the truth puts the
    host at one galaxy with a definite ``(ra, dec, z)``.  With
    ``sigma_ang in [1.0, 2.39] deg`` the sky likelihood varies substantially
    across a pixel, so the within-pixel sky-redshift correlation the model
    discards is not a small quantity.

The test is a PAIRED substitution on the same galaxies.  Every arm below writes
darksirens' own per-event evidence in the canonical basis, where every Jacobian
cancels (ATTRIBUTION.md A3):

    Z_i(H0)  =  SUM_g  w^sky_g  kw_g (N_obs(p_g)/Z_global) g(z_g)
                       (1+z_g)^(gamma-1) L_D(obs_dL | dL(z_g; H0)) M_i(z_g)

    M_i(z)   =  INT dm1src dm2src [p_mq(m1src, m2src/m1src)/m1src]
                                   L_1(obs_m1 | m1src(1+z)) L_2(obs_m2 | m2src(1+z))

with ``L_1``, ``L_2`` the EXACT generative mass likelihood ``N(obs; m, f m)`` --
which, since 2026-08-01, is also exactly the model the stored PE samples encode
(convention (c2)) -- and the ONLY difference between the two decisive arms being
the sky weight:

    delta_pix    w^sky_g = <u(Omega)>_{Omega in p_g}     the pixel AVERAGE
    delta_host   w^sky_g = u(Omega_g)                    the galaxy's OWN value

    u(Omega) = N(ra; ra_obs, sig_ra) N(dec; dec_obs, sigma_ang) / cos(dec)

``u`` is the sky posterior density PER STERADIAN (the ``1/cos(dec)`` converts from
the PE's own ``(ra, dec)`` product measure, in which both Gaussians are written, to
solid angle).  Both arms use the SAME estimator of it -- an equal-solid-angle
average over the ``4^k`` HEALPix children of each pixel -- so ``delta_host -
delta_pix`` is exactly and only "the within-pixel sky structure the pixelisation
throws away", with no aperture, normalisation or quadrature difference between them.

Arms:

    kde_pix     darksirens' catalog KDE prior, pixel-average sky   [ANCHOR: must
                reproduce darksirens' own per-event score]
    delta_pix   zero-bandwidth catalog prior, pixel-average sky    [= attr_oracle's
                `delta_exact`, rebuilt from the raw catalog]
    delta_host  zero-bandwidth catalog prior, EXACT galaxy sky     [the new arm]
    kde_host    catalog KDE prior, EXACT galaxy sky                [--with_kde_host;
                completes the 2x2 so the photo-z kernel and the pixelisation can be
                read off independently]

The galaxies' sky positions come from ``build_catalog_skyindex.py``, which stores
the complete catalog in the survey block's OWN row order, so galaxy ``(row, column)``
of darksirens' state arrays is paired with its own ``(ra, dec)``; the identity is
re-verified BITWISE against the survey file on every run.

Outputs: results/attr_sky_oracle_<tag>.{json,npz}
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

SIG_M1_FRAC = 0.08
SIG_M2_FRAC = 0.10
NSIG_M = 7.0
NSIG_D = 7.0
DZ_SCALE = 3.0e-3      # the survey block's declared photo-z kernel, dz = DZ_SCALE (1+z)
KPAD = 10.0            # kernel widths of padding on the KDE arms' galaxy window
KCHUNK = 200_000       # galaxies per chunk in the kde_host kernel integral
SKYINDEX = Path("/hildafs/projects/phy220048p/magana/gws-agn-data/derived/"
                "analysis_1_complete_catalog_H0/skyindex")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tracer", choices=["gal", "agn"], default="gal")
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--n_events", type=int, default=0, help="0 = all")
    ap.add_argument("--h0", type=float, default=67.74)
    ap.add_argument("--dh", type=float, default=0.5)
    ap.add_argument("--n_z", type=int, default=512)
    ap.add_argument("--n_m", type=int, default=192)
    ap.add_argument("--n_sub", type=int, default=5,
                    help="HEALPix refinement levels for the sub-pixel sky average: "
                         "4^n_sub children per pixel (5 -> 1024, nside 1024)")
    ap.add_argument("--sky_frac", type=float, default=1e-6,
                    help="keep pixels above this fraction of the candidate sky mass")
    ap.add_argument("--n_ap", type=float, default=6.0,
                    help="candidate aperture radius in units of sigma_ang (x1.6 pad)")
    ap.add_argument("--grid_shift", type=float, default=0.0)
    ap.add_argument("--with_kde_host", action="store_true")
    ap.add_argument("--host_prior_arms", action="store_true",
                    help="TASK 3: add the delta_host_unif / delta_host_norate arms "
                         "(opt-in; without it every product is unchanged)")
    ap.add_argument("--pz_batch", type=int, default=2_000_000)
    ap.add_argument("--sel_batch", type=int, default=50000)
    ap.add_argument("--pe_batch_events", type=int, default=25)
    ap.add_argument("--skyindex", default=str(SKYINDEX))
    ap.add_argument("--n_verify_rows", type=int, default=200,
                    help="survey-vs-index bitwise probes per run")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--outdir", default=str(ROOT / "results"))
    return ap.parse_args(argv)


def trap_w(x):
    w = np.empty_like(x)
    w[1:-1] = 0.5 * (x[2:] - x[:-2])
    w[0] = 0.5 * (x[1] - x[0])
    w[-1] = 0.5 * (x[-1] - x[-2])
    return w


def wrap(x):
    return (np.asarray(x) + np.pi) % (2.0 * np.pi) - np.pi


def main(argv=None):
    args = parse_args(argv)
    tag = args.tag or args.tracer
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")

    import jax.numpy as jnp
    import healpy as hp
    import attr_ds_bridge as bridge
    from darksirens.redshift.prior import (eval_redshift_prior_with_state,
                                           prepare_redshift_prior_state)
    from darksirens.utils.cosmology import z_of_dL, dL_of_z
    from darksirens.gw.populations.registry import get_model

    kw = dict(kde_window=4096) if args.tracer == "gal" else {}
    B = bridge.build(tracer=args.tracer, seed=args.seed, h0=args.h0,
                     sel_batch=args.sel_batch, pe_batch_events=args.pe_batch_events,
                     **kw)
    nobs = B.nobs
    model = get_model("powerlaw+peak", shared_beta=True, shared_spin=True,
                      shared_gamma=True)
    th_mix = jnp.asarray(np.asarray(B.pop_fid)[:model.mixture.n_params])
    gamma_fid = B.gamma_fid
    H0s = [args.h0 - args.dh, args.h0, args.h0 + args.dh]
    NSIDE = 32
    A_PIX = 4.0 * np.pi / hp.nside2npix(NSIDE)

    # ---------------- event data -------------------------------------------
    with h5py.File(B.paths["gw"], "r") as f:
        o1 = np.asarray(f["truth/obs_m1det"][:], float)
        o2 = np.asarray(f["truth/obs_m2det"][:], float)
        s1 = np.asarray(f["truth/obs_sig_m1"][:], float)
        s2 = np.asarray(f["truth/obs_sig_m2"][:], float)
        oD = np.asarray(f["truth/obs_dL"][:], float)
        sD = np.asarray(f["truth/obs_sigma_dl"][:], float)
        ora = np.asarray(f["truth/obs_ra"][:], float)
        odec = np.asarray(f["truth/obs_dec"][:], float)
        sang = np.asarray(f["truth/obs_sigma_ang"][:], float)
        sra = (np.asarray(f["truth/obs_sig_ra"][:], float)
               if "obs_sig_ra" in f["truth"] else
               sang / np.maximum(np.cos(odec), 0.1))
        t_z = np.asarray(f["truth/z"][:], float)
        t_ra = np.asarray(f["truth/ra"][:], float)
        t_dec = np.asarray(f["truth/dec"][:], float)
        t_m1d = np.asarray(f["truth/m1det"][:], float)
        t_q = np.asarray(f["truth/q"][:], float)
        t_dL = np.asarray(f["truth/dl"][:], float)
        t_chi = np.asarray(f["truth/chieff"][:], float)
    n_use = nobs if args.n_events <= 0 else min(args.n_events, nobs)
    idx_use = np.arange(n_use)
    print(f"[sky-oracle] {args.tracer}: {n_use}/{nobs} events  n_z={args.n_z} "
          f"n_m={args.n_m} n_sub=4^{args.n_sub} sky_frac={args.sky_frac} "
          f"n_ap={args.n_ap}", flush=True)

    # ---------------- prior states at the three H0 -------------------------
    states, pzfun = {}, {}
    for h in H0s:
        cosmo = B.cosmo0._replace(H0=jnp.float64(h))
        st = prepare_redshift_prior_state(
            "dark_sirens", cosmo, B.survey_p, B.cat_pe, mark_model="none",
            mark_params=None, mark_names=(), materialize_state=True,
            catalog_sky_weighting="field")
        states[h] = (st, cosmo)

        def _mk(st=st, cosmo=cosmo):
            def f(z, pix):
                return eval_redshift_prior_with_state(
                    "dark_sirens", st, z, pix, cosmo, B.survey_p, B.cat_pe,
                    catalog_sky_weighting="field")
            return f
        pzfun[h] = _mk()
    st_c, cosmo_c = states[args.h0]
    zgals = np.asarray(B.cat_pe.zgals)
    ngals = np.asarray(B.cat_pe.ngals)
    upix = np.asarray(B.cat_pe.unique_pixels)
    log_Nobs = {h: np.asarray(states[h][0].log_Nobs) for h in H0s}
    log_Zg = {h: float(np.asarray(states[h][0].log_Z_global)) for h in H0s}
    log_kw_h = {h: np.asarray(states[h][0].kernels.log_kw) for h in H0s}
    log_g_grid = {h: np.asarray(states[h][0].kernels.log_g_grid) for h in H0s}
    sig_eff_h = {h: np.asarray(states[h][0].kernels.sig_eff) for h in H0s}
    from darksirens.redshift.grid import zgrid as ZGRID
    ZGRID = np.asarray(ZGRID)
    dNmiss_max = float(np.asarray(st_c.dN_miss).max())
    order = np.argsort(upix)
    upix_sorted = upix[order]
    print(f"[sky-oracle] catalog rows {zgals.shape}  log_Z_global="
          f"{log_Zg[args.h0]:.6f}  max dN_miss={dNmiss_max:.3e}", flush=True)

    # ---------------- the sky index ----------------------------------------
    sk = Path(args.skyindex) / f"seed{args.seed}_{args.tracer}_ns{NSIDE}.h5"
    with h5py.File(sk, "r") as f:
        ra_s = f["ra_s"][:]
        dec_s = f["dec_s"][:]
        z_s = f["z_s"][:]
        starts = f["starts"][:]
        counts = f["counts"][:]
        sk_attrs = {k: (v.item() if hasattr(v, "item") and np.ndim(v) == 0 else v)
                    for k, v in f.attrs.items()}
    print(f"[sky-oracle] index {sk.name}: {ra_s.size:,} galaxies", flush=True)

    # bitwise re-verification against the state arrays the likelihood uses
    rng_v = np.random.default_rng(11)
    rows_v = rng_v.choice(np.flatnonzero(ngals > 0),
                          size=min(args.n_verify_rows, int((ngals > 0).sum())),
                          replace=False)
    n_bad = 0
    for r in rows_v:
        P = int(upix[r]); m = int(ngals[r])
        if not np.array_equal(zgals[r, :m], z_s[starts[P]:starts[P] + m]):
            n_bad += 1
    if n_bad:
        raise SystemExit(f"[fatal] sky index disagrees with cat_pe on {n_bad} rows")
    print(f"[anchor] index vs cat_pe zgals: bitwise identical on {rows_v.size} rows",
          flush=True)

    # ---------------- sub-pixel sky machinery -------------------------------
    k = int(args.n_sub)
    nside_sub = NSIDE * (2 ** k)
    nchild = 4 ** k
    child_off = np.arange(nchild, dtype=np.int64)

    def sky_u(ra, dec, i):
        """Sky posterior density per steradian at (ra, dec) for event i."""
        d_ra = wrap(ra - ora[i]) / sra[i]
        d_de = (dec - odec[i]) / sang[i]
        lg = -0.5 * (d_ra ** 2 + d_de ** 2)
        # the 1/(2 pi s_ra sigma_ang) prefactor is an H0-independent per-event
        # constant and cancels in d ln Z/dH0; it is kept so W_p is a genuine
        # probability and the two arms are directly comparable in absolute terms.
        norm = 1.0 / (2.0 * np.pi * sra[i] * sang[i])
        return norm * np.exp(lg) / np.maximum(np.cos(dec), 1e-12)

    def pixel_sky_average_at(pixels, i, kk_):
        """<u>_p over the 4^kk_ equal-solid-angle HEALPix children of each pixel."""
        nsub = NSIDE * (2 ** kk_)
        nch = 4 ** kk_
        nest = hp.ring2nest(NSIDE, pixels).astype(np.int64)
        ch = (nest[:, None] * nch + np.arange(nch, dtype=np.int64)[None, :]).ravel()
        th, ph = hp.pix2ang(nsub, ch, nest=True)
        return sky_u(ph, 0.5 * np.pi - th, i).reshape(pixels.size, nch).mean(axis=1)

    def pixel_sky_average(pixels, i):
        return pixel_sky_average_at(pixels, i, k)

    # --- anchor: the sub-pixel sky rule, twice -------------------------------
    # (i) self-convergence: refine the rule by one HEALPix level (4x the children).
    # (ii) against the PE's OWN measure: draw the sky exactly as
    #      generate_dataset.posterior_samples does and histogram into pixels.  This
    #      is the decisive check on the 1/cos(dec) Jacobian and the pixel area, since
    #      W_p = INT_p L(ra,dec) dra ddec is precisely the fraction of PE samples in
    #      pixel p.  Its residual is the MC error plus the dec CLIP at |dec| = pi/2,
    #      which the smooth rule does not model (relevant only within a few sigma_ang
    #      of a pole -- flagged per event as `pole_sigma`).
    def _mc_pixel_weights(i, n_mc, rng):
        ra = (ora[i] + sra[i] * rng.normal(size=n_mc)) % (2.0 * np.pi)
        dec = np.clip(odec[i] + sang[i] * rng.normal(size=n_mc),
                      -0.5 * np.pi, 0.5 * np.pi)
        px = hp.ang2pix(NSIDE, 0.5 * np.pi - dec, ra)
        up, inv = np.unique(px, return_inverse=True)
        return up, np.bincount(inv, minlength=up.size) / n_mc

    rng_mc = np.random.default_rng(2026)
    n_mc = 4_000_000
    conv_ref, mc_rel, mc_pull = [], [], []
    for i in idx_use[:6]:
        cand_a = hp.query_disc(NSIDE, hp.ang2vec(0.5 * np.pi - odec[i], ora[i]),
                               args.n_ap * sang[i] * 1.6, inclusive=True)
        w_k = A_PIX * pixel_sky_average(cand_a, i)
        w_k1 = A_PIX * pixel_sky_average_at(cand_a, i, k + 1)
        m = w_k1 >= args.sky_frac * w_k1.sum()
        conv_ref.append(float(np.max(np.abs(w_k[m] / w_k1[m] - 1.0))))
        up_m, w_m = _mc_pixel_weights(i, n_mc, rng_mc)
        big = w_m >= 1e-3
        w_s = A_PIX * pixel_sky_average(up_m[big], i)
        mc_rel.append(float(np.max(np.abs(w_s / w_m[big] - 1.0))))
        sd = np.sqrt(w_m[big] * (1.0 - w_m[big]) / n_mc)
        mc_pull.append(float(np.max(np.abs(w_s - w_m[big]) / sd)))
    sub_conv = float(np.max(conv_ref))
    sub_mc = float(np.max(mc_rel))
    sub_mc_pull = float(np.max(mc_pull))
    print(f"[anchor] sub-pixel sky rule: max |dW/W| under one more refinement level "
          f"{sub_conv:.2e};  vs {n_mc:,} PE-measure draws (pixels above 1e-3) "
          f"max rel {sub_mc:.2e}, max pull {sub_mc_pull:.1f} sigma_MC", flush=True)
    pole_sigma = (0.5 * np.pi - np.abs(odec)) / sang

    # ---------------- per-event quadrature ---------------------------------
    arms = ["kde_pix", "delta_pix", "delta_host"]
    if args.with_kde_host:
        arms.append("kde_host")
    if args.host_prior_arms:
        # TASK 3 (2026-08-01): two OPT-IN one-term substitutions on the host prior,
        # both exact-sky (delta_host) and both H0-INDEPENDENT reweightings of the
        # per-galaxy weight, so they change only the measure the score is averaged
        # over.  Default off; with the flag absent every product above is
        # bit-identical to the CLOSURE.md run.
        #   delta_host_unif   : kw_g g(z_g) N_obs/Z_global  ->  1/Z_global
        #                       i.e. the mock's OWN generative host prior, UNIFORM
        #                       over catalog rows (generate_dataset.stage_events
        #                       draws  i ~ U{0..N-1}  within the tracer).
        #   delta_host_norate : drop (1+z)^(gamma-1), the host-acceptance factor.
        arms += ["delta_host_unif", "delta_host_norate"]
    score = {a: np.full(n_use, np.nan) for a in arms}
    lnZ = {a: np.full((n_use, 3), np.nan) for a in arms}
    diag = {k_: np.zeros(n_use) for k_ in
            ("sky_mass_cand", "sky_mass_kept", "sky_mass_mapped", "n_pix_cand",
             "n_pix", "n_gal", "n_gal_kde", "ap_radius_deg", "M_tail",
             "host_in_aperture", "cos_clamped", "n_grow")}

    def mass_grid(i, z_lo, z_hi):
        lo1 = max(1.0, (o1[i] - NSIG_M * s1[i]) / (1.0 + z_hi))
        hi1 = (o1[i] + NSIG_M * s1[i]) / (1.0 + z_lo)
        lo2 = max(0.5, (o2[i] - NSIG_M * s2[i]) / (1.0 + z_hi))
        hi2 = (o2[i] + NSIG_M * s2[i]) / (1.0 + z_lo)
        m1 = np.linspace(lo1, hi1, args.n_m)
        m2 = np.linspace(lo2, hi2, args.n_m)
        if args.grid_shift:
            m1 = m1 + args.grid_shift * (m1[1] - m1[0])
            m2 = m2 + args.grid_shift * (m2[1] - m2[0])
        return m1, m2

    t0 = time.time()
    T = {"mass": 0.0, "sky": 0.0, "pz": 0.0, "gal": 0.0, "asm": 0.0}
    for kk, i in enumerate(idx_use):
        # --- z grid ------------------------------------------------------
        d_lo = oD[i] * np.exp(-NSIG_D * sD[i])
        d_hi = oD[i] * np.exp(+NSIG_D * sD[i])
        z_lo = float(z_of_dL(jnp.float64(d_lo), jnp.float64(H0s[0]), cosmo_c.Om0,
                             cosmo_c.w0, cosmo_c.wa))
        z_hi = float(z_of_dL(jnp.float64(d_hi), jnp.float64(H0s[2]), cosmo_c.Om0,
                             cosmo_c.w0, cosmo_c.wa))
        z_lo = max(z_lo, 1e-5)
        zg = np.linspace(z_lo, z_hi, args.n_z)
        if args.grid_shift:
            zg = zg + args.grid_shift * (zg[1] - zg[0])
        wz = trap_w(zg)

        # --- mass factor M(z), exact mass likelihood ----------------------
        _t = time.time()
        m1g, m2g = mass_grid(i, z_lo, z_hi)
        w1, w2 = trap_w(m1g), trap_w(m2g)
        qg = m2g[None, :] / m1g[:, None]
        P = np.asarray(model.mixture.mass_q_density(
            jnp.asarray(np.repeat(m1g, args.n_m)), jnp.asarray(qg.ravel()), th_mix)
        ).reshape(args.n_m, args.n_m) / m1g[:, None]
        A = P * w1[:, None] * w2[None, :]
        opz = 1.0 + zg
        m1det = m1g[None, :] * opz[:, None]
        m2det = m2g[None, :] * opz[:, None]
        u_ex = np.exp(-0.5 * ((o1[i] - m1det) / (SIG_M1_FRAC * m1det)) ** 2) / m1det
        v_ex = np.exp(-0.5 * ((o2[i] - m2det) / (SIG_M2_FRAC * m2det)) ** 2) / m2det
        Aj = jnp.asarray(A)
        M_z = np.asarray(((jnp.asarray(u_ex) @ Aj) * jnp.asarray(v_ex)).sum(axis=1))
        T["mass"] += time.time() - _t

        # --- candidate pixels and their sky weights -----------------------
        _t = time.time()
        # The candidate disc is GROWN until it holds the whole sky posterior.  A
        # fixed n_ap * sigma_ang radius is not enough near a pole: there
        # sig_ra is clamped at sigma_ang/0.1, so the posterior runs 10 sigma_ang
        # wide in RA, and samples that scatter a few sigma_ang AWAY from the pole
        # convert that into a large ANGULAR offset.  W_p is a probability, so
        # sum_cand W_p is its own coverage test -- grow until it is 1.
        vec = hp.ang2vec(0.5 * np.pi - odec[i], ora[i])
        p_obs = np.array([hp.ang2pix(NSIDE, 0.5 * np.pi - odec[i], ora[i])])
        r_ap = args.n_ap * sang[i] * 1.6
        for _grow in range(8):
            cand = np.union1d(hp.query_disc(NSIDE, vec, min(r_ap, np.pi),
                                            inclusive=True), p_obs)
            u_bar = pixel_sky_average(cand, i)
            W_cand = A_PIX * u_bar
            tot = float(W_cand.sum())
            if tot >= 1.0 - 1e-7 or r_ap >= np.pi:
                break
            r_ap *= 1.6
        diag["n_grow"][kk] = _grow
        keep = W_cand >= args.sky_frac * tot
        pk, Wk, ubk = cand[keep], W_cand[keep], u_bar[keep]
        kept = float(Wk.sum())
        # map to the compact catalog rows the state arrays are indexed by
        loc = np.clip(np.searchsorted(upix_sorted, pk), 0, upix_sorted.size - 1)
        okm = upix_sorted[loc] == pk
        rows = order[loc[okm]]
        pk, Wk, ubk = pk[okm], Wk[okm], ubk[okm]
        mapped = float(Wk.sum())
        wp = Wk / Wk.sum()                       # H0-independent normalisation
        diag["sky_mass_cand"][kk] = tot
        diag["sky_mass_kept"][kk] = kept / tot if tot else 0.0
        diag["sky_mass_mapped"][kk] = mapped / tot if tot else 0.0
        diag["n_pix_cand"][kk] = cand.size
        diag["n_pix"][kk] = rows.size
        diag["ap_radius_deg"][kk] = np.rad2deg(r_ap)
        T["sky"] += time.time() - _t

        # --- p_z on the z grid (KDE arms) ---------------------------------
        _t = time.time()
        zf = np.tile(zg, rows.size)
        pf = np.repeat(rows, args.n_z)
        pz_eff = {}
        for h in H0s:
            out = np.empty(zf.size)
            for j0 in range(0, zf.size, args.pz_batch):
                sl = slice(j0, j0 + args.pz_batch)
                out[sl] = np.asarray(pzfun[h](jnp.asarray(zf[sl]),
                                              jnp.asarray(pf[sl], dtype=jnp.int32)))
            pz_eff[h] = (wp[:, None] * np.exp(out).reshape(rows.size, args.n_z)
                         ).sum(axis=0)
        T["pz"] += time.time() - _t

        # --- the galaxies themselves --------------------------------------
        _t = time.time()
        # The KDE arms need the galaxies whose kernels REACH the window, so the cut
        # is padded by KPAD kernel widths; the delta arms then use the strict window
        # (mask `strict`), which is attr_oracle's own convention.
        zpad = KPAD * DZ_SCALE * (1.0 + z_hi)
        g_z, g_u, g_wp, g_row, g_col = [], [], [], [], []
        n_clamp = 0
        for r, Pp, wr in zip(rows, pk, wp):
            n = int(ngals[r])
            if n == 0:
                continue
            sl = slice(int(starts[Pp]), int(starts[Pp]) + n)
            zz = z_s[sl]
            m = (zz >= z_lo - zpad) & (zz <= z_hi + zpad)
            if not m.any():
                continue
            dd = dec_s[sl][m]
            n_clamp += int((np.cos(dd) < 1e-12).sum())
            g_z.append(zz[m])
            g_u.append(sky_u(ra_s[sl][m], dd, i))
            g_wp.append(np.full(int(m.sum()), wr))
            g_row.append(np.full(int(m.sum()), r))
            g_col.append(np.flatnonzero(m))
        if g_z:
            g_z = np.concatenate(g_z); g_u = np.concatenate(g_u)
            g_wp = np.concatenate(g_wp); g_row = np.concatenate(g_row)
            g_col = np.concatenate(g_col)
        else:
            g_z = g_u = g_wp = np.zeros(0)
            g_row = g_col = np.zeros(0, int)
        strict = (g_z >= z_lo) & (g_z <= z_hi) if g_z.size else np.zeros(0, bool)
        diag["cos_clamped"][kk] = n_clamp
        diag["n_gal"][kk] = int(strict.sum()) if g_z.size else 0
        diag["n_gal_kde"][kk] = g_z.size
        # the host arm's sky weight, on the SAME normalisation as wp
        g_wh = (A_PIX * g_u / Wk.sum()) if g_z.size else np.zeros(0)
        # is the event's own host inside the aperture?
        P_host = hp.ang2pix(NSIDE, 0.5 * np.pi - t_dec[i], t_ra[i])
        diag["host_in_aperture"][kk] = float(P_host in set(pk.tolist()))
        T["gal"] += time.time() - _t

        # --- assemble -----------------------------------------------------
        _t = time.time()
        rate_z = np.power(1.0 + zg, gamma_fid - 1.0)
        rate_g = np.power(1.0 + g_z, gamma_fid - 1.0) if g_z.size else np.zeros(0)
        for c, h in enumerate(H0s):
            dl = np.asarray(dL_of_z(jnp.asarray(zg), jnp.float64(h), cosmo_c.Om0,
                                    cosmo_c.w0, cosmo_c.wa))
            lD_z = -0.5 * ((np.log(oD[i]) - np.log(dl)) / sD[i]) ** 2
            # Phi(z): everything in the integrand except the catalog prior
            Phi = (np.exp(np.interp(zg, ZGRID, log_g_grid[h])) * rate_z
                   * np.exp(lD_z) * M_z)
            if g_z.size:
                # dL(z; H0) is smooth; interpolate the SAME grid both arms use
                cat_w = np.exp(log_kw_h[h][g_row, g_col] + log_Nobs[h][g_row]
                               - log_Zg[h])
                dlg = np.interp(g_z, zg, dl)
                lD_g = -0.5 * ((np.log(oD[i]) - np.log(dlg)) / sD[i]) ** 2
                gz_g = np.exp(np.interp(g_z, ZGRID, log_g_grid[h]))
                base = (cat_w * gz_g * rate_g * np.exp(lD_g)
                        * np.interp(g_z, zg, M_z)) * strict
                if args.host_prior_arms:
                    core = np.exp(lD_g) * np.interp(g_z, zg, M_z) * strict
                    base_u = (np.exp(-log_Zg[h]) * rate_g) * core
                    base_nr = (cat_w * gz_g) * core
                if args.with_kde_host:
                    # F_g = INT dz N(z; z_g, sig_eff_g) Phi(z) -- the delta arm is
                    # its sig_eff -> 0 limit, so kde_host and delta_host differ by
                    # exactly the photo-z kernel and nothing else.
                    sg = sig_eff_h[h][g_row, g_col]
                    Fg = np.empty(g_z.size)
                    for j0 in range(0, g_z.size, KCHUNK):
                        sl2 = slice(j0, min(j0 + KCHUNK, g_z.size))
                        d = (zg[None, :] - g_z[sl2, None]) / sg[sl2, None]
                        K = (np.exp(-0.5 * d ** 2)
                             / (np.sqrt(2.0 * np.pi) * sg[sl2, None]))
                        Fg[sl2] = (K * (wz * Phi)[None, :]).sum(axis=1)
                    base_kde = cat_w * Fg
            for a in arms:
                if a == "kde_pix":
                    Z = float((wz * pz_eff[h] * rate_z * np.exp(lD_z) * M_z).sum())
                elif a == "delta_pix":
                    Z = float((g_wp * base).sum()) if g_z.size else 0.0
                elif a == "delta_host":
                    Z = float((g_wh * base).sum()) if g_z.size else 0.0
                elif a == "delta_host_unif":
                    Z = float((g_wh * base_u).sum()) if g_z.size else 0.0
                elif a == "delta_host_norate":
                    Z = float((g_wh * base_nr).sum()) if g_z.size else 0.0
                else:                                   # kde_host
                    Z = float((g_wh * base_kde).sum()) if g_z.size else 0.0
                lnZ[a][kk, c] = np.log(Z) if Z > 0 else -np.inf
            if c == 1:
                f0 = pz_eff[h] * rate_z * np.exp(lD_z) * M_z
                diag["M_tail"][kk] = float((f0[0] + f0[-1]) / max(f0.max(), 1e-300))
        for a in arms:
            score[a][kk] = (lnZ[a][kk, 2] - lnZ[a][kk, 0]) / (2.0 * args.dh)
        T["asm"] += time.time() - _t
        if kk % 25 == 0:
            print(f"  event {kk+1}/{n_use} ({time.time()-t0:.0f}s) npix={rows.size} "
                  f"ngal={int(diag['n_gal'][kk])} "
                  f"kept={diag['sky_mass_kept'][kk]:.6f} "
                  f"s_kde={score['kde_pix'][kk]:+.4e} "
                  f"s_dpix={score['delta_pix'][kk]:+.4e} "
                  f"s_dhost={score['delta_host'][kk]:+.4e}  "
                  + " ".join(f"{a}={b:.0f}s" for a, b in T.items()), flush=True)
    print(f"[sky-oracle] quadrature done in {time.time()-t0:.0f}s  "
          + " ".join(f"{a}={b:.0f}s" for a, b in T.items()), flush=True)

    # ---------------- darksirens, selection, truth --------------------------
    pe_split = bridge.pe_pass_split(B, dh=args.dh,
                                    pe_batch_events=args.pe_batch_events)
    ds_score = pe_split["full"][idx_use]
    sig_mc = np.abs(pe_split["hA"] - pe_split["hB"])[idx_use] / 2.0
    S = bridge.sel_pass(B, dh=args.dh, sel_batch=args.sel_batch)
    anchor = abs(float(S["log_mu"]) - B.spy0["log_mu"])
    dlnmu = float(S["dlnmu_fd"])

    # A = the score evaluated at the events' OWN true parameters (the empirical
    # detected-truth mean).  ldw = lp_pop + lp_z - ljac - log(prior_wt), so with
    # prior_wt = 1 the finite difference of ldw at the truth point IS varsigma.
    fns = [B.make_pieces(h) for h in H0s]
    P_true = hp.ang2pix(NSIDE, 0.5 * np.pi - t_dec, t_ra)
    loc = np.clip(np.searchsorted(upix_sorted, P_true), 0, upix_sorted.size - 1)
    ok_true = upix_sorted[loc] == P_true
    row_true = np.where(ok_true, order[loc], 0)
    arg = (jnp.asarray(t_m1d), jnp.asarray(t_q), jnp.asarray(t_dL),
           jnp.asarray(t_chi), jnp.asarray(row_true, dtype=jnp.int32),
           jnp.ones(nobs), jnp.ones(nobs, dtype=bool))
    ldw_t = [np.asarray(f(*arg)[0]) for f in fns]
    A_truth = np.where(ok_true, (ldw_t[2] - ldw_t[0]) / (2.0 * args.dh), np.nan)[idx_use]
    print(f"[anchor] |log_mu diff| = {anchor:.3e}   d ln mu/dH0 (fd) = {dlnmu:.6e}   "
          f"truth pixels mapped: {int(ok_true.sum())}/{nobs}", flush=True)

    def stat(x, y=None):
        v = x if y is None else x - y
        v = v[np.isfinite(v)]
        return {"mean": float(v.mean()),
                "sem": float(v.std(ddof=1) / np.sqrt(v.size)), "n": int(v.size)}

    out = {"name": "attr_sky_oracle", "tracer": args.tracer, "seed": args.seed,
           "tag": tag, "n_events": int(n_use), "n_events_total": int(nobs),
           "H0": args.h0, "dh": args.dh,
           "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "config": B.paths, "skyindex": str(sk), "skyindex_attrs": sk_attrs,
           "grid": {"n_z": args.n_z, "n_m": args.n_m, "n_sub": args.n_sub,
                    "nside_sub": nside_sub, "n_children": nchild,
                    "sky_frac": args.sky_frac, "n_ap": args.n_ap,
                    "grid_shift": args.grid_shift, "nsig_mass": NSIG_M,
                    "nsig_dL": NSIG_D},
           "anchors": {"log_mu_absdiff": anchor,
                       "index_vs_cat_pe_rows_checked": int(rows_v.size),
                       "index_vs_cat_pe_rows_bad": int(n_bad),
                       "max_dN_miss": dNmiss_max,
                       "truth_pixels_mapped": int(ok_true.sum()),
                       "subpixel_rule_refinement_maxrel": sub_conv,
                       "subpixel_vs_PE_measure_MC_maxrel": sub_mc,
                       "subpixel_vs_PE_measure_MC_max_pull_sigma": sub_mc_pull,
                       "n_events_within_8_sigma_of_pole":
                           int((pole_sigma[idx_use] < 8.0).sum())},
           "dlnmu_dH0": dlnmu,
           "pole_cut_note": ("the smooth sky rule does not model the |dec| <= pi/2 "
                             "clip the PE applies, so the anchor is quoted both on "
                             "all events and on those more than 8 sigma_ang from a "
                             "pole"),
           "darksirens": {"score_mean": stat(ds_score),
                          "A_truth_mean": stat(A_truth),
                          "r": stat(ds_score - dlnmu),
                          "CmA": stat(ds_score, A_truth)},
           "darksirens_pe_mc_sigma": {
               "median": float(np.median(sig_mc)), "mean": float(sig_mc.mean()),
               "rms": float(np.sqrt((sig_mc ** 2).mean())),
               "expected_sem_of_mean": float(np.sqrt((sig_mc ** 2).sum()) / n_use)},
           "diagnostics": {a: {"mean": float(np.mean(b)), "min": float(np.min(b)),
                               "max": float(np.max(b))} for a, b in diag.items()},
           "arms": {}}
    for a in arms:
        out["arms"][a] = {"score": stat(score[a]), "r": stat(score[a] - dlnmu),
                          "CmA": stat(score[a], A_truth),
                          "vs_darksirens": stat(score[a], ds_score),
                          "vs_kde_pix": stat(score[a], score["kde_pix"])}
        s = out["arms"][a]
        print(f"  {a:>11}  <s>={s['score']['mean']:+.5e}  r={s['r']['mean']:+.5e}"
              f"  (C-A)={s['CmA']['mean']:+.4e}+-{s['CmA']['sem']:.2e}"
              f"  vs ds {s['vs_darksirens']['mean']:+.4e}"
              f"+-{s['vs_darksirens']['sem']:.2e}")
    out["substitutions"] = {
        "pixelisation__host_minus_pix__delta_prior":
            stat(score["delta_host"], score["delta_pix"]),
        "photoz_kernel__delta_minus_kde__pixel_sky":
            stat(score["delta_pix"], score["kde_pix"]),
        "both__delta_host_minus_kde_pix":
            stat(score["delta_host"], score["kde_pix"])}
    if args.with_kde_host:
        out["substitutions"]["pixelisation__host_minus_pix__kde_prior"] = stat(
            score["kde_host"], score["kde_pix"])
    if args.host_prior_arms:
        out["substitutions"]["hostprior__uniform_minus_darksirens__delta_host"] = \
            stat(score["delta_host_unif"], score["delta_host"])
        out["substitutions"]["hostprior__norate_minus_darksirens__delta_host"] = \
            stat(score["delta_host_norate"], score["delta_host"])
    for a, b in out["substitutions"].items():
        print(f"  SUBST {a:>52}  {b['mean']:+.5e} +- {b['sem']:.2e}")

    res_dir = Path(args.outdir)
    np.savez_compressed(res_dir / f"attr_sky_oracle_{tag}.npz",
                        idx=idx_use, ds_score=ds_score, A_truth=A_truth,
                        dlnmu=dlnmu, sig_mc=sig_mc,
                        **{f"score_{a}": score[a] for a in arms},
                        **{f"lnZ_{a}": lnZ[a] for a in arms},
                        pole_sigma=pole_sigma[idx_use],
                        sigma_ang_deg=np.rad2deg(sang[idx_use]),
                        sig_ra_deg=np.rad2deg(sra[idx_use]),
                        obs_dec_deg=np.rad2deg(odec[idx_use]),
                        **{f"diag_{a}": b for a, b in diag.items()})
    (res_dir / f"attr_sky_oracle_{tag}.json").write_text(json.dumps(out, indent=2))
    print(f"\nWrote {res_dir / f'attr_sky_oracle_{tag}.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
