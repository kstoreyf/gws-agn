#!/usr/bin/env python3
"""ATTRIBUTION follow-up, TASK 3 -- the (m1, m2, dL) quadrature oracle.

Every measurement channel of this mock is closed-form, so the per-event
evidence can be computed by direct quadrature instead of by Monte-Carlo over
stored PE samples.  Writing the likelihood's own target density in the
canonical basis and changing variables (m1det -> m1src at fixed z, dL -> z) the
Jacobians cancel exactly and

    Z_i(H0)  =  SUM_p W_p INT dz  p_z(z | p) (1+z)^(gamma-1)
                                  L_D(obs_dL | dL(z; H0))  M_i(z)

    M_i(z)   =  INT dm1src dm2src  [p_mq(m1src, m2src/m1src)/m1src]
                                   L_1(obs_m1 | m1src(1+z))
                                   L_2(obs_m2 | m2src(1+z))

with

    W_p   pixel mass of the (exact) sky posterior, by Gauss-Hermite quadrature
          in the PE's own (ra, dec) measure;
    p_z   the catalog redshift prior, evaluated through darksirens' OWN
          ``eval_redshift_prior_with_state`` (KDE arms) or in its exact
          zero-bandwidth limit (delta arms);
    p_mq  darksirens' OWN ``mixture.mass_q_density``;
    L_D   the generator's lognormal distance likelihood (exact in every arm);
    L_1/2 either the EXACT generative mass likelihood N(obs; m, f m) -- whose
          m-dependence carries the 1/(f m) prefactor -- or the STORED-Gaussian
          model N(m; obs, f m_true) that the stored PE samples encode.

``d ln Z_i/dH0`` is then a pure quadrature object: no PE samples, no MC noise.
Arms (all with the exact distance channel and the exact sky quadrature):

    kde_gauss     darksirens' KDE prior + stored-Gaussian masses   [ANCHOR:
                  must reproduce darksirens' own per-event score]
    kde_exact     darksirens' KDE prior + exact mass likelihood    [the named
                  defect, repaired in closed form]
    delta_gauss   zero-bandwidth catalog prior + stored Gaussian   [isolates
                  the photo-z KDE kernel the mock's exact catalog does not have]
    delta_exact   zero-bandwidth catalog prior + exact masses      [fully exact]
    host_exact    delta at the TRUE host redshift + exact masses   [counterpart
                  limit; the mass lever is switched off by construction]

Outputs: results/attr_oracle_<tracer>.{json,npz}
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

SIGMA_DL = 0.10
SIG_M1_FRAC = 0.08
SIG_M2_FRAC = 0.10
NSIG_M = 7.0
NSIG_D = 7.0


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
    ap.add_argument("--n_gh", type=int, default=48)
    ap.add_argument("--sky_frac", type=float, default=1e-4,
                    help="drop pixels below this fraction of the sky posterior")
    ap.add_argument("--grid_shift", type=float, default=0.0,
                    help="fractional-cell shift of the (m1, m2, z) trapezoid grids "
                         "(convergence test); the Gauss-Hermite sky rule is NOT "
                         "shifted -- its nodes carry their own weights.")
    ap.add_argument("--pz_batch", type=int, default=2_000_000)
    ap.add_argument("--sel_batch", type=int, default=50000)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--outdir", default=str(ROOT / "results"))
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    tag = args.tag or args.tracer
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")

    import jax.numpy as jnp
    import healpy as hp
    import attr_ds_bridge as bridge
    from darksirens.redshift.prior import eval_redshift_prior_with_state
    from darksirens.utils.cosmology import z_of_dL, dL_of_z
    from darksirens.gw.populations.registry import get_model

    kw = dict(kde_window=4096) if args.tracer == "gal" else {}
    B = bridge.build(tracer=args.tracer, seed=args.seed, h0=args.h0,
                     sel_batch=args.sel_batch, **kw)
    nobs = B.nobs
    model = get_model("powerlaw+peak", shared_beta=True, shared_spin=True,
                      shared_gamma=True)
    th_mix = jnp.asarray(np.asarray(B.pop_fid)[:model.mixture.n_params])
    gamma_fid = B.gamma_fid
    H0s = [args.h0 - args.dh, args.h0, args.h0 + args.dh]

    # ---------------- event data -------------------------------------------
    gwf = B.paths["gw"]
    with h5py.File(gwf, "r") as f:
        o1 = np.asarray(f["truth/obs_m1det"][:], float)
        o2 = np.asarray(f["truth/obs_m2det"][:], float)
        s1 = np.asarray(f["truth/obs_sig_m1"][:], float)
        s2 = np.asarray(f["truth/obs_sig_m2"][:], float)
        oD = np.asarray(f["truth/obs_dL"][:], float)
        sD = np.asarray(f["truth/obs_sigma_dl"][:], float)
        ora = np.asarray(f["truth/obs_ra"][:], float)
        odec = np.asarray(f["truth/obs_dec"][:], float)
        sang = np.asarray(f["truth/obs_sigma_ang"][:], float)
        t_z = np.asarray(f["truth/z"][:], float)
        t_ra = np.asarray(f["truth/ra"][:], float)
        t_dec = np.asarray(f["truth/dec"][:], float)
    n_use = nobs if args.n_events <= 0 else min(args.n_events, nobs)
    idx_use = np.arange(n_use)
    print(f"[oracle] {args.tracer}: {n_use}/{nobs} events, n_z={args.n_z} "
          f"n_m={args.n_m} n_gh={args.n_gh} shift={args.grid_shift}")

    # ---------------- prior states at the three H0 -------------------------
    from darksirens.redshift.prior import prepare_redshift_prior_state
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
    log_kw = np.asarray(st_c.kernels.log_kw)
    upix = np.asarray(B.cat_pe.unique_pixels)
    log_Nobs = {h: np.asarray(states[h][0].log_Nobs) for h in H0s}
    log_Zg = {h: float(np.asarray(states[h][0].log_Z_global)) for h in H0s}
    log_kw_h = {h: np.asarray(states[h][0].kernels.log_kw) for h in H0s}
    log_g_grid = {h: np.asarray(states[h][0].kernels.log_g_grid) for h in H0s}
    from darksirens.redshift.grid import zgrid as ZGRID
    ZGRID = np.asarray(ZGRID)
    sig_eff_h = {h: np.asarray(states[h][0].kernels.sig_eff) for h in H0s}
    dNmiss_max = float(np.asarray(st_c.dN_miss).max())
    print(f"[oracle] catalog rows {zgals.shape}, log_Z_global={log_Zg[args.h0]:.6f}, "
          f"max dN_miss={dNmiss_max:.3e}")

    # ---------------- anchor: reconstruct darksirens' own p_z ---------------
    # The delta (zero-bandwidth) arm is built from the SAME state arrays as the
    # KDE arm -- (log_kw, sig_eff, log_g, log_Nobs, log_Z_global) -- with the
    # Gaussian kernels replaced by point masses of weight kw_j g(z_j).  This
    # check verifies that reconstruction against darksirens' own evaluator on
    # the KDE side, so the delta limit inherits its correctness.
    def _my_log_pz(zq, row, h):
        n = int(ngals[row])
        zs = zgals[row, :n]; sg = sig_eff_h[h][row, :n]; lk = log_kw_h[h][row, :n]
        d = (zq[:, None] - zs[None, :]) / sg[None, :]
        lg = lk[None, :] - 0.5 * d ** 2 - np.log(np.sqrt(2 * np.pi) * sg[None, :])
        mx = lg.max(axis=1)
        mix = mx + np.log(np.exp(lg - mx[:, None]).sum(axis=1))
        return (log_Nobs[h][row] - log_Zg[h]
                + np.interp(zq, ZGRID, log_g_grid[h]) + mix)

    rng_a = np.random.default_rng(7)
    rows_a = rng_a.choice(np.arange(zgals.shape[0])[ngals > 0], size=8, replace=False)
    zq = np.linspace(0.02, 0.55, 97)
    d_anchor = []
    for r in rows_a:
        a_ds = np.asarray(pzfun[args.h0](jnp.asarray(zq),
                                         jnp.full(zq.size, int(r), dtype=jnp.int32)))
        a_me = _my_log_pz(zq, int(r), args.h0)
        d_anchor.append(np.max(np.abs(a_ds - a_me)))
    pz_recon_maxabs = float(np.max(d_anchor))
    print(f"[anchor] |log p_z(mine) - log p_z(darksirens)| max = {pz_recon_maxabs:.3e} "
          f"over {len(rows_a)} rows x {zq.size} z")

    # ---------------- sky pixel weights (Gauss-Hermite, PE measure) ---------
    gx, gw = np.polynomial.hermite_e.hermegauss(args.n_gh)
    gw = gw / gw.sum()
    order = np.argsort(upix)
    upix_sorted = upix[order]

    def sky_weights(i):
        sa = sang[i]
        s_ra = sa / max(np.cos(odec[i]), 0.1)
        # NOTE: the Gauss-Hermite weights are tied to the node positions, so the
        # sky nodes must NOT be translated by --grid_shift (that would break the
        # quadrature rather than test it).  The sky rule is converged separately
        # by raising n_gh and lowering --sky_frac.
        ra = (ora[i] + s_ra * gx[:, None]) % (2 * np.pi)
        dec = np.clip(odec[i] + sa * gx[None, :], -0.5 * np.pi, 0.5 * np.pi)
        w = (gw[:, None] * gw[None, :]).ravel()
        nn = args.n_gh
        pix = hp.ang2pix(
            32, np.pi / 2.0 - np.broadcast_to(dec, (nn, nn)).ravel(),
            np.broadcast_to(ra, (nn, nn)).ravel())
        up, inv = np.unique(pix, return_inverse=True)
        wp = np.bincount(inv, weights=w, minlength=up.size)
        keep = wp >= args.sky_frac
        up, wp = up[keep], wp[keep]
        loc = np.searchsorted(upix_sorted, up)
        loc = np.clip(loc, 0, upix_sorted.size - 1)
        okm = upix_sorted[loc] == up
        rows = order[loc[okm]]
        return rows, wp[okm], float(wp[okm].sum()), float(wp.sum())

    # ---------------- per-event quadrature ---------------------------------
    arms = ("kde_gauss", "kde_exact", "delta_gauss", "delta_exact",
            "host_exact")
    score = {a: np.full(n_use, np.nan) for a in arms}
    lnZ = {a: np.full((n_use, 3), np.nan) for a in arms}
    diag = {"sky_mass_kept": np.zeros(n_use), "sky_mass_mapped": np.zeros(n_use),
            "n_pix": np.zeros(n_use, int), "n_gal_delta": np.zeros(n_use, int),
            "pz_tail": np.zeros(n_use), "M_tail": np.zeros(n_use),
            "anchor_pz_reldiff": np.zeros(n_use)}

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

    def trap_w(x):
        w = np.empty_like(x)
        w[1:-1] = 0.5 * (x[2:] - x[:-2]); w[0] = 0.5 * (x[1] - x[0])
        w[-1] = 0.5 * (x[-1] - x[-2])
        return w

    t0 = time.time()
    T = {"mass": 0.0, "sky": 0.0, "pz": 0.0, "delta": 0.0, "asm": 0.0}
    for k, i in enumerate(idx_use):
        # --- z grid ---------------------------------------------------------
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

        # --- mass factor M(z) ----------------------------------------------
        _t = time.time()
        m1g, m2g = mass_grid(i, z_lo, z_hi)
        w1, w2 = trap_w(m1g), trap_w(m2g)
        qg = m2g[None, :] / m1g[:, None]
        P = np.asarray(model.mixture.mass_q_density(
            jnp.asarray(np.repeat(m1g, args.n_m)), jnp.asarray(qg.ravel()), th_mix)
        ).reshape(args.n_m, args.n_m) / m1g[:, None]
        A = P * w1[:, None] * w2[None, :]
        opz = 1.0 + zg
        m1det = m1g[None, :] * opz[:, None]          # (nz, nm)
        m2det = m2g[None, :] * opz[:, None]
        # exact generative mass likelihood  N(obs; m, f m)
        u_ex = np.exp(-0.5 * ((o1[i] - m1det) / (SIG_M1_FRAC * m1det)) ** 2) / m1det
        v_ex = np.exp(-0.5 * ((o2[i] - m2det) / (SIG_M2_FRAC * m2det)) ** 2) / m2det
        # stored-Gaussian model  N(m; obs, f m_true)
        u_ga = np.exp(-0.5 * ((m1det - o1[i]) / s1[i]) ** 2)
        v_ga = np.exp(-0.5 * ((m2det - o2[i]) / s2[i]) ** 2)
        Aj = jnp.asarray(A)
        M_ex = np.asarray(((jnp.asarray(u_ex) @ Aj) * jnp.asarray(v_ex)).sum(axis=1))
        M_ga = np.asarray(((jnp.asarray(u_ga) @ Aj) * jnp.asarray(v_ga)).sum(axis=1))
        T["mass"] += time.time() - _t

        # --- sky ------------------------------------------------------------
        _t = time.time()
        rows, wp, kept, tot = sky_weights(i)
        diag["sky_mass_kept"][k] = tot
        diag["sky_mass_mapped"][k] = kept
        diag["n_pix"][k] = rows.size
        wp = wp / wp.sum()
        T["sky"] += time.time() - _t

        # --- p_z on the grid, all three H0 ----------------------------------
        _t = time.time()
        zf = np.tile(zg, rows.size)
        pf = np.repeat(rows, args.n_z)
        pz = {}
        for h in H0s:
            out = np.empty(zf.size)
            for j0 in range(0, zf.size, args.pz_batch):
                sl = slice(j0, j0 + args.pz_batch)
                out[sl] = np.asarray(pzfun[h](jnp.asarray(zf[sl]),
                                              jnp.asarray(pf[sl], dtype=jnp.int32)))
            pz[h] = np.exp(out).reshape(rows.size, args.n_z)
        pz_eff = {h: (wp[:, None] * pz[h]).sum(axis=0) for h in H0s}
        pe = pz_eff[args.h0]
        T["pz"] += time.time() - _t

        # --- zero-bandwidth (delta) catalog prior ---------------------------
        _t = time.time()
        gal_z, gal_w = [], {h: [] for h in H0s}
        for r, wr in zip(rows, wp):
            n = int(ngals[r])
            zz = zgals[r, :n]
            m = (zz >= z_lo) & (zz <= z_hi)
            if not m.any():
                continue
            gal_z.append(zz[m])
            for h in H0s:
                gz = np.exp(np.interp(zz[m], ZGRID, log_g_grid[h]))
                gal_w[h].append(wr * np.exp(log_kw_h[h][r, :n][m]
                                            + log_Nobs[h][r] - log_Zg[h]) * gz)
        if gal_z:
            gal_z = np.concatenate(gal_z)
            gal_w = {h: np.concatenate(gal_w[h]) for h in H0s}
        else:
            gal_z = np.zeros(0); gal_w = {h: np.zeros(0) for h in H0s}
        diag["n_gal_delta"][k] = gal_z.size
        T["delta"] += time.time() - _t

        # --- assemble ln Z at the three H0 ----------------------------------
        _t = time.time()
        rate_z = np.power(1.0 + zg, gamma_fid - 1.0)
        rate_g = (np.power(1.0 + gal_z, gamma_fid - 1.0) if gal_z.size
                  else np.zeros(0))
        zh = t_z[i]
        lD_z, lD_g, lD_h = {}, {}, {}
        for h in H0s:
            dl = np.asarray(dL_of_z(jnp.asarray(zg), jnp.float64(h), cosmo_c.Om0,
                                    cosmo_c.w0, cosmo_c.wa))
            lD_z[h] = -0.5 * ((np.log(oD[i]) - np.log(dl)) / sD[i]) ** 2
            if gal_z.size:
                dlg = np.asarray(dL_of_z(jnp.asarray(gal_z), jnp.float64(h),
                                         cosmo_c.Om0, cosmo_c.w0, cosmo_c.wa))
                lD_g[h] = -0.5 * ((np.log(oD[i]) - np.log(dlg)) / sD[i]) ** 2
            dlh = float(dL_of_z(jnp.float64(zh), jnp.float64(h), cosmo_c.Om0,
                                cosmo_c.w0, cosmo_c.wa))
            lD_h[h] = -0.5 * ((np.log(oD[i]) - np.log(dlh)) / sD[i]) ** 2
        # integrand tail check on the FULL integrand (L_D supplies the cutoff)
        f0 = pz_eff[args.h0] * rate_z * np.exp(lD_z[args.h0])
        for nm_, Mz in (("ex", M_ex), ("ga", M_ga)):
            g_ = f0 * Mz
            diag["M_tail"][k] = max(diag["M_tail"][k],
                                    float((g_[0] + g_[-1]) / max(g_.max(), 1e-300)))
        diag["pz_tail"][k] = diag["M_tail"][k]
        for a in arms:
            Mz = M_ex if a.endswith("exact") else M_ga
            Mg = np.interp(gal_z, zg, Mz) if gal_z.size else np.zeros(0)
            Mh = float(np.interp(zh, zg, Mz))
            for c, h in enumerate(H0s):
                if a.startswith("kde"):
                    Z = float((wz * pz_eff[h] * rate_z * np.exp(lD_z[h]) * Mz).sum())
                elif a.startswith("delta"):
                    Z = (float((gal_w[h] * rate_g * np.exp(lD_g[h]) * Mg).sum())
                         if gal_z.size else 0.0)
                else:
                    Z = float(np.exp(lD_h[h]) * Mh * (1.0 + zh) ** (gamma_fid - 1.0))
                lnZ[a][k, c] = np.log(Z) if Z > 0 else -np.inf
            score[a][k] = (lnZ[a][k, 2] - lnZ[a][k, 0]) / (2.0 * args.dh)
        T["asm"] += time.time() - _t
        if k % 25 == 0:
            print(f"  event {k+1}/{n_use}  ({time.time()-t0:.0f}s)  "
                  f"npix={rows.size} ngal={diag['n_gal_delta'][k]}  "
                  f"s_kde_gauss={score['kde_gauss'][k]:+.4e}  "
                  + " ".join(f"{kk}={vv:.1f}s" for kk, vv in T.items()))
    print(f"[oracle] quadrature done in {time.time()-t0:.0f}s  "
          + " ".join(f"{kk}={vv:.0f}s" for kk, vv in T.items()))

    # ---------------- comparison to darksirens ------------------------------
    res_dir = Path(args.outdir)
    ds = np.load(res_dir / f"attr_terms_{args.tracer}_s{args.seed}.npz")
    ds_score = np.asarray(ds["ev_s_fd"])[idx_use]
    ds_score_terms = np.asarray(ds["ev_s_tot"])[idx_use]
    mp = np.load(res_dir / f"attr_mass_pe_{args.tracer}_s{args.seed}.npz")
    A_truth = (np.asarray(mp["truth_pop"]) + np.asarray(mp["truth_pz"])
               + np.asarray(mp["truth_jac"]))[idx_use]

    pe_split = bridge.pe_pass_split(B, dh=args.dh)
    s_full = pe_split["full"][idx_use]
    # hA, hB are independent nsamp/2 estimators with sd sigma_h; the full
    # estimator has sigma_full = sigma_h/sqrt(2), and |hA-hB|/2 = sigma_full |N|,
    # so rms(|hA-hB|/2) IS sigma_full.
    sig_mc_full = np.abs(pe_split["hA"] - pe_split["hB"])[idx_use] / 2.0
    print(f"[noise] darksirens per-event MC sigma on d ln Z/dH0: "
          f"median {np.median(sig_mc_full):.3e}  mean {sig_mc_full.mean():.3e}")

    S = bridge.sel_pass(B, dh=args.dh, sel_batch=args.sel_batch)
    anchor = abs(float(S["log_mu"]) - B.spy0["log_mu"])
    # The oracle's per-event score IS a finite difference of ln Z, so r must be
    # taken against the FINITE-DIFFERENCE d ln mu/dH0 (differencing log mu), not
    # the term-summed one; the two differ by 2.1e-5 (GAL) / 1.5e-4 (AGN).
    dlnmu = float(S["dlnmu_fd"])
    dlnmu_terms = float(S["dlnmu_terms"])
    print(f"[anchor] |log_mu diff| = {anchor:.3e}   d ln mu/dH0 (fd) = {dlnmu:.6e}"
          f"   (term-sum {dlnmu_terms:.6e})")

    def stat(x, y=None):
        v = x if y is None else x - y
        v = v[np.isfinite(v)]
        return {"mean": float(v.mean()),
                "sem": float(v.std(ddof=1) / np.sqrt(v.size)), "n": int(v.size)}

    out = {"name": "attr_oracle", "tracer": args.tracer, "seed": args.seed,
           "tag": tag, "n_events": int(n_use), "n_events_total": int(nobs),
           "H0": args.h0, "dh": args.dh,
           "grid": {"n_z": args.n_z, "n_m": args.n_m, "n_gh": args.n_gh,
                    "nsig_mass": NSIG_M, "nsig_dL": NSIG_D,
                    "grid_shift": args.grid_shift, "sky_frac": args.sky_frac},
           "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "config": B.paths, "anchor_log_mu_absdiff": anchor,
           "anchor_pz_reconstruction_maxabs": pz_recon_maxabs,
           "max_dN_miss": dNmiss_max,
           "darksirens_pe_mc_sigma": {
               "median": float(np.median(sig_mc_full)),
               "mean": float(sig_mc_full.mean()),
               "rms": float(np.sqrt((sig_mc_full ** 2).mean())),
               "expected_sem_of_mean": float(
                   np.sqrt((sig_mc_full ** 2).sum()) / n_use)},
           "ds_score_fd_vs_split_full_maxabs": float(
               np.max(np.abs(ds_score - s_full))),
           "dlnmu_dH0": dlnmu, "dlnmu_dH0_term_sum": dlnmu_terms,
           "dlnmu_convention": "finite difference of log mu at H0 +- dh",
           "darksirens": {"score_mean": stat(ds_score),
                          "score_mean_terms": stat(ds_score_terms),
                          "A_truth_mean": stat(A_truth),
                          "r_subset": stat(ds_score - dlnmu),
                          "CmA_subset": stat(ds_score, A_truth)},
           "diagnostics": {k: {"mean": float(np.mean(v)), "min": float(np.min(v)),
                               "max": float(np.max(v))}
                           for k, v in diag.items()},
           "arms": {}}
    for a in arms:
        out["arms"][a] = {
            "score": stat(score[a]),
            "r": stat(score[a] - dlnmu),
            "CmA": stat(score[a], A_truth),
            "vs_darksirens": stat(score[a], ds_score),
            "vs_kde_gauss": stat(score[a], score["kde_gauss"]),
        }
        s = out["arms"][a]
        print(f"  {a:>12}  <s>={s['score']['mean']:+.5e}  r={s['r']['mean']:+.5e}"
              f"  (C-A)={s['CmA']['mean']:+.5e}+-{s['CmA']['sem']:.2e}"
              f"  vs ds {s['vs_darksirens']['mean']:+.4e}"
              f"+-{s['vs_darksirens']['sem']:.2e}")

    np.savez_compressed(res_dir / f"attr_oracle_{tag}.npz",
                        idx=idx_use, ds_score=ds_score, A_truth=A_truth,
                        dlnmu=dlnmu, sig_mc_full=sig_mc_full,
                        ds_score_hA=pe_split["hA"][idx_use],
                        ds_score_hB=pe_split["hB"][idx_use],
                        **{f"score_{a}": score[a] for a in arms},
                        **{f"lnZ_{a}": lnZ[a] for a in arms},
                        **{f"diag_{k}": v for k, v in diag.items()})
    (res_dir / f"attr_oracle_{tag}.json").write_text(json.dumps(out, indent=2))
    print(f"\nWrote {res_dir/f'attr_oracle_{tag}.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
