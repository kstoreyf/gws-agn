#!/usr/bin/env python3
"""Exact-likelihood oracle for the matched dark-siren mock (numpy/scipy only).

Computes, per event, the EXACT marginal likelihood curve L_i(H0) of the
generative process of scripts/build_obsdet_mock.py (observed-data arm), with no
PE samples and no Monte Carlo, plus an ablation ladder that walks from the
exact likelihood to the expectation of darksirens' PE-sample estimator:

  O1  exact: discrete hosts, exact per-host sky Gaussian, heteroscedastic
      generative mass-noise likelihood (width = a * true mass).
  O2  as O1 but with the PE-implied fixed mass-noise widths (w = a * m_true,
      constant per event) -- the likelihood the mock's PE cloud encodes.
  O3  as O2 but sky pixelated: per-host Gaussian -> analytic PE-cloud pixel
      mass Q_i(pix) at nside 16 (darksirens' sky treatment).
  O3b as O3 plus the (1+z) * m1src basis-Jacobian weight darksirens' estimator
      carries because gmd stores p_pe = 1 in the (m1det, m2det, dL) basis.
  O4  as O3b with the host atoms replaced by the survey KDE kernels
      g(z) N(z; z_k, dz)/Z_k -- the full expectation of darksirens' numerator.

All variants share ln dL(z; H0) = ln dL(z; H0_ref) - ln(H0/H0_ref) (Om0 pinned,
flat w0waCDM), so H0 enters as a pure shift delta = ln(H0/H0_ref).

Also computes the exact selection function mu(H0) by deterministic quadrature.

The population density is gmd's (verified identical to darksirens' log_p_pop
to 1.4e-14); mass-pairing normalisations use fine exact quadrature.

Writes results/oracle_num_<tag>.npz and results/oracle_mu_<tag>.npz.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import h5py
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.special import ndtr, roots_hermitenorm

EXP = Path(__file__).resolve().parents[1]
GMD_DIR = "/hildafs/projects/phy230014p/magana/src/darksirens/scripts/mock_dark_sirens"
sys.path.insert(0, GMD_DIR)
import generate_mock_data as gmd  # noqa: E402  (pure numpy; the generative truth)

import healpy as hp  # noqa: E402

H0_REF = 67.74
OM0 = 0.3075
S_DL = 0.10
A1, A2 = 0.08, 0.10
SNR_REF = 6.278363879917771
SNR_THRESHOLD = 8.0
NSIDE = 16
KDE_SIG = 0.003
GH_N = 48          # mass-integral Gauss-Hermite nodes per dimension
ZNODES = 48        # per-event z-grid nodes for the mass integrals
POP = gmd.PopulationConfig(gamma=0.0)


# --------------------------------------------------------------------------- #
# population density in (m1, q), with exact (fine-quadrature) pairing norms
# --------------------------------------------------------------------------- #
class PopDensity:
    """rho_pop(m1, q): density in (m1src, q); rho(m1,m2) = rho_pop / m1."""

    def __init__(self, pop=POP, n_m1=4096, n_q=4096):
        self.pop = pop
        m1g = np.geomspace(1.0, 200.0, n_m1)
        qg = np.linspace(1e-6, 1.0, n_q)
        # exact pairing norms n(m1) = int q^beta S_low(q m1; mmin, dm) dq
        def pair_norm(m_min, dm):
            m2 = qg[None, :] * m1g[:, None]
            f = qg[None, :] ** pop.beta * gmd._sfilter_low(m2, m_min, dm)
            f = np.where(m2 < m_min, 0.0, f)
            return np.trapz(f, qg, axis=1)
        self.m1g = m1g
        self.log_npl = np.log(np.maximum(pair_norm(pop.mmin, pop.dm_min), 1e-300))
        self.log_npk = np.log(np.maximum(pair_norm(gmd._PAIR_M_LO, gmd._PAIR_DM), 1e-300))
        # exact primary-mass norms
        mg = np.geomspace(1.0, 200.0, 200000)
        pl_un = gmd._powerlaw_unnorm(mg, pop.alpha, pop.mmin, pop.mmax,
                                     pop.dm_min, pop.dm_max)
        self.pl_norm = np.trapz(pl_un, mg)
        self.pk_norm = np.sqrt(2 * np.pi) * pop.peak_sigma  # untruncated Gaussian

    def __call__(self, m1, q):
        pop = self.pop
        m1 = np.asarray(m1); q = np.asarray(q)
        ok = (q > 0.0) & (q <= 1.0) & (m1 > 0.0)
        m1s = np.where(ok, m1, 10.0); qs = np.where(ok, q, 0.5)
        m2 = qs * m1s
        p_pl = gmd._powerlaw_unnorm(m1s, pop.alpha, pop.mmin, pop.mmax,
                                    pop.dm_min, pop.dm_max) / self.pl_norm
        p_pk = np.exp(-0.5 * ((m1s - pop.peak_mu) / pop.peak_sigma) ** 2) / self.pk_norm
        npl = np.exp(np.interp(np.log(m1s), np.log(self.m1g), self.log_npl))
        npk = np.exp(np.interp(np.log(m1s), np.log(self.m1g), self.log_npk))
        pair_pl = qs ** self.pop.beta * gmd._sfilter_low(m2, pop.mmin, pop.dm_min)
        pair_pl = np.where(m2 < pop.mmin, 0.0, pair_pl) / npl
        pair_pk = qs ** self.pop.beta * gmd._sfilter_low(m2, gmd._PAIR_M_LO, gmd._PAIR_DM)
        pair_pk = np.where(m2 < gmd._PAIR_M_LO, 0.0, pair_pk) / npk
        out = ((1.0 - pop.peak_fraction) * p_pl * pair_pl
               + pop.peak_fraction * p_pk * pair_pk)
        return np.where(ok, out, 0.0)


# --------------------------------------------------------------------------- #
# cosmology (the generative one: astropy via gmd grids)
# --------------------------------------------------------------------------- #
def make_grids(zmax=2.0):
    cosmo = gmd._build_cosmology(H0_REF, OM0, -1.0, 0.0)
    return gmd._cosmology_grids(cosmo, zmax)


# --------------------------------------------------------------------------- #
# per-event mass integrals M(z) on a z-grid (three weightings, shared nodes)
# --------------------------------------------------------------------------- #
_gh_x, _gh_w = roots_hermitenorm(GH_N)   # weight e^{-x^2/2}, sum w = sqrt(2 pi)
_gh_w = _gh_w / np.sqrt(2.0 * np.pi)     # now E[f(N(0,1))] = sum w f(x)


def mass_integrals(rho_pop, M1, M2, w1, w2, zg):
    """Return (lnM_O1, lnM_O2, lnM_O2m) on z-grid zg.

    M_O2(z)  = iint rho(m1,m2) N(M1; m1 t, w1) N(M2; m2 t, w2) dm1 dm2
    M_O1(z)  = same with widths (a1 m1 t, a2 m2 t) and their normalisations
    M_O2m(z) = M_O2 with an extra m1src factor in the integrand.
    rho(m1,m2) = rho_pop(m1, m2/m1)/m1.
    Nodes: m1det_a = M1 + w1 x_a, m2det_b = M2 + w2 x_b (Gauss-Hermite).
    """
    t = 1.0 + zg[:, None, None]                      # (Z,1,1)
    m1d = (M1 + w1 * _gh_x)[None, :, None]           # (1,A,1)
    m2d = (M2 + w2 * _gh_x)[None, None, :]           # (1,1,B)
    ww = (_gh_w[:, None] * _gh_w[None, :])[None, :, :]
    m1 = m1d / t
    m2 = m2d / t
    q = np.where(m1d > 0, m2d / np.where(m1d > 0, m1d, 1.0), 0.0)
    rho = np.where((m1 > 0) & (q > 0),
                   rho_pop(m1, np.clip(q, 1e-9, 2.0)) / np.where(m1 > 0, m1, 1.0),
                   0.0)                                # (Z,A,B) density in (m1,m2)
    base = ww * rho / t ** 2                          # (Z,A,B)
    # int N(M; x, w) f(x) dx = sum_a ghw_a f(M + w x_a): all Gaussian norms
    # absorbed; overall event-constant factors are irrelevant for the shapes.
    MO2 = np.sum(base, axis=(1, 2))
    MO2m = np.sum(base * m1, axis=(1, 2))
    # O1: reweight nodes by the heteroscedastic/fixed-width likelihood ratio
    #   r = [N(M1; m1d, a1*m1d) N(M2; m2d, a2*m2d)] / [N(M1; m1d, w1) N(M2; m2d, w2)]
    # z-independent per (a,b) node (all quantities live in the detector basis).
    m1d2 = np.broadcast_to(m1d, (1, GH_N, GH_N))
    m2d2 = np.broadcast_to(m2d, (1, GH_N, GH_N))
    node_ok = (m1d2 > 0.1) & (m2d2 > 0.1)
    s1 = A1 * np.where(node_ok, m1d2, 1.0)
    s2 = A2 * np.where(node_ok, m2d2, 1.0)
    lr = (np.log(w1 / s1) - 0.5 * ((M1 - m1d2) / s1) ** 2
          + 0.5 * ((M1 - m1d2) / w1) ** 2
          + np.log(w2 / s2) - 0.5 * ((M2 - m2d2) / s2) ** 2
          + 0.5 * ((M2 - m2d2) / w2) ** 2)
    r = np.where(node_ok, np.exp(np.clip(lr, -300.0, 60.0)), 0.0)
    MO1 = np.sum(base * r, axis=(1, 2))
    f = lambda x: np.log(np.maximum(x, 1e-300))
    return f(MO1), f(MO2), f(MO2m)


# --------------------------------------------------------------------------- #
# sky: analytic PE-cloud pixel masses at nside 16
# --------------------------------------------------------------------------- #
def pe_pixel_masses(A_obs, B_obs, sa, nside_sub=512):
    """Q(pix16): mass of the PE sky cloud (independent Gaussians in ra, dec,
    widths sa/max(cos B_obs, 0.1) and sa) per nside-16 pixel."""
    vec = hp.ang2vec(np.pi / 2 - B_obs, A_obs)
    rad = min(6.5 * sa / max(np.cos(B_obs), 0.1) + 0.05, np.pi - 1e-3)
    sub = hp.query_disc(nside_sub, vec, rad, inclusive=True)
    th, ph = hp.pix2ang(nside_sub, sub)
    dec = np.pi / 2 - th
    sra = sa / max(np.cos(B_obs), 0.1)
    dra = np.angle(np.exp(1j * (ph - A_obs)))
    dens = (np.exp(-0.5 * (dra / sra) ** 2) / (sra * np.sqrt(2 * np.pi))
            * np.exp(-0.5 * ((dec - B_obs) / sa) ** 2) / (sa * np.sqrt(2 * np.pi)))
    a_sub = hp.nside2pixarea(nside_sub)
    mass = dens * a_sub / np.maximum(np.cos(dec), 1e-3)
    par = hp.ang2pix(NSIDE, th, ph)
    out = {}
    for p in np.unique(par):
        out[int(p)] = float(mass[par == p].sum())
    return out


# --------------------------------------------------------------------------- #
# numerator oracle per realisation
# --------------------------------------------------------------------------- #
def run_numerators(events_path, catalog_path, out_path, h0_grid, nevents=None):
    grids = make_grids()
    zg_glob = grids["z"][1:]
    lndl_glob = np.log(grids["dl"][1:])
    z_of_lndl = lambda ld: np.interp(ld, lndl_glob, zg_glob)
    lndl_of_z = lambda z: np.interp(z, zg_glob, lndl_glob)
    lggam = np.log(np.maximum(grids["dvc_dz"], 1e-300))  # gamma(z) shape (H0-free)
    gam_of_z = lambda z: np.exp(np.interp(z, grids["z"], lggam))

    rho_pop = PopDensity()

    with h5py.File(catalog_path) as f:
        cra = f["ra"][:]; cdec = f["dec"][:]; cz = f["z"][:]
    order = np.argsort(cdec)
    cra, cdec, cz = cra[order], cdec[order], cz[order]
    cpix = hp.ang2pix(NSIDE, np.pi / 2 - cdec, cra)
    clndl = lndl_of_z(cz)

    if isinstance(events_path, dict):
        E = events_path
    else:
        with h5py.File(events_path) as f:
            t = f["truth"]
            E = {k: t[k][:] for k in ("obs_dL", "obs_m1det", "obs_m2det", "obs_ra",
                                      "obs_dec", "obs_sigma_ang", "obs_sig_m1",
                                      "obs_sig_m2", "z", "ra", "dec")}
    N = len(E["obs_dL"]) if nevents is None else int(nevents)

    H0s = h0_grid
    delta = np.log(H0s / H0_REF)                    # ln dL shifts by -delta
    variants = ("O1", "O2", "O3", "O3b", "O4")
    curves = {v: np.zeros((N, len(H0s))) for v in variants}
    diags = {"n_hosts": np.zeros(N, int), "n_hosts_kde": np.zeros(N, int),
             "top_host_frac": np.zeros(N)}

    # O4 kernel quadrature nodes (per host): GL over +-4.5 sigma
    from numpy.polynomial.legendre import leggauss
    glx, glw = leggauss(12)

    t0 = time.time()
    for i in range(N):
        D = E["obs_dL"][i]; lnD = np.log(D)
        M1, M2 = E["obs_m1det"][i], E["obs_m2det"][i]
        w1, w2 = E["obs_sig_m1"][i], E["obs_sig_m2"][i]
        sa = E["obs_sigma_ang"][i]
        A_obs, B_obs = E["obs_ra"][i], E["obs_dec"][i]

        # --- host sky window (dec band then elliptical cut) ---------------- #
        lo = np.searchsorted(cdec, B_obs - 6.5 * sa)
        hi = np.searchsorted(cdec, B_obs + 6.5 * sa)
        j = slice(lo, hi)
        dra = np.angle(np.exp(1j * (cra[j] - A_obs)))
        cosd = np.maximum(np.cos(cdec[j]), 0.1)
        sig_ra = sa / cosd
        r2 = (dra / sig_ra) ** 2 + ((cdec[j] - B_obs) / sa) ** 2
        sky_ok = r2 < 6.5 ** 2
        # distance window
        u_all = lnD - clndl[j]
        dist_ok = np.abs(u_all) < 0.80
        keep = sky_ok & dist_ok
        idx = np.nonzero(keep)[0]
        zs = cz[j][idx]
        us = u_all[idx]
        pix_h = cpix[j][idx]
        lnF = (-0.5 * (dra[idx] / sig_ra[idx]) ** 2
               - np.log(sig_ra[idx])
               - 0.5 * ((cdec[j][idx] - B_obs) / sa) ** 2 - np.log(sa))

        # --- pixel masses of the PE sky cloud ------------------------------ #
        Q = pe_pixel_masses(A_obs, B_obs, sa)
        # hosts for the pixelated variants: all hosts in Q-pixels within the
        # distance window (sky window may miss corner hosts of included pixels)
        qpix = np.array(sorted(Q.keys()))
        in_q = np.isin(cpix[j], qpix) & dist_ok
        idxq = np.nonzero(in_q)[0]
        zq = cz[j][idxq]; uq = u_all[idxq]; pixq = cpix[j][idxq]
        Qmap = np.zeros(hp.nside2npix(NSIDE)); Qmap[qpix] = [Q[int(p)] for p in qpix]

        # --- mass-integral splines over the union z-range ------------------ #
        allz = np.concatenate([zs, zq])
        if allz.size == 0:
            for v in variants:
                curves[v][i] = -np.inf
            continue
        zlo = max(allz.min() - 0.02, 1e-4)
        zhi = min(allz.max() + 0.02, 1.99)
        zgrid_i = np.linspace(zlo, zhi, ZNODES)
        lnM1v, lnM2v, lnM2mv = mass_integrals(rho_pop, M1, M2, w1, w2, zgrid_i)
        spl1 = CubicSpline(zgrid_i, lnM1v)
        spl2 = CubicSpline(zgrid_i, lnM2v)
        spl2m = CubicSpline(zgrid_i, lnM2mv)

        inv_2s2 = 1.0 / (2.0 * S_DL ** 2)
        # per-variant (u, logw) node lists
        def curve_from(u_nodes, logw_nodes):
            # logsumexp over nodes of logw - (u+delta)^2/(2 s^2)
            arg = logw_nodes[:, None] - inv_2s2 * (u_nodes[:, None] + delta[None, :]) ** 2
            m = arg.max(axis=0)
            return m + np.log(np.sum(np.exp(arg - m), axis=0))

        lw_O1 = lnF - np.log1p(zs) + spl1(zs)
        lw_O2 = lnF - np.log1p(zs) + spl2(zs)
        curves["O1"][i] = curve_from(us, lw_O1)
        curves["O2"][i] = curve_from(us, lw_O2)

        lnQq = np.log(np.maximum(Qmap[pixq], 1e-300))
        lw_O3 = lnQq - np.log1p(zq) + spl2(zq)
        lw_O3b = lnQq + spl2m(zq)          # (1+z)^{gamma-1} * (1+z) = 1 at gamma=0
        curves["O3"][i] = curve_from(uq, lw_O3)
        curves["O3b"][i] = curve_from(uq, lw_O3b)

        # O4: replace each atom by its KDE kernel (nodes over +-4.5 sigma)
        znodes = zq[:, None] + KDE_SIG * 4.5 * glx[None, :]
        wnodes = 4.5 * KDE_SIG * glw[None, :] * (
            np.exp(-0.5 * ((znodes - zq[:, None]) / KDE_SIG) ** 2)
            / (KDE_SIG * np.sqrt(2 * np.pi)))
        okz = znodes > 1e-5
        znodes = np.where(okz, znodes, 1e-5)
        gamn = gam_of_z(znodes)
        Zk = np.sum(wnodes * np.where(okz, gamn, 0.0), axis=1)   # kernel norms
        lw_k = (lnQq[:, None] + np.log(np.maximum(wnodes * gamn, 1e-300))
                - np.log(np.maximum(Zk, 1e-300))[:, None]
                + spl2m(np.clip(znodes, zlo, zhi)))
        u_k = lnD - lndl_of_z(znodes)
        curves["O4"][i] = curve_from(
            np.where(okz, u_k, 1e3).ravel(), np.where(okz, lw_k, -np.inf).ravel())

        diags["n_hosts"][i] = idx.size
        diags["n_hosts_kde"][i] = idxq.size
        pk = np.exp(lw_O1 - inv_2s2 * us ** 2)
        diags["top_host_frac"][i] = pk.max() / pk.sum() if pk.sum() > 0 else np.nan

        if i % 100 == 0:
            print(f"[{i}/{N}] {time.time()-t0:.0f}s", flush=True)

    np.savez_compressed(out_path, H0=H0s, **{f"ln_{v}": curves[v] for v in variants},
                        **diags, events_path=str(events_path),
                        catalog_path=str(catalog_path))
    print("wrote", out_path)
    return curves


# --------------------------------------------------------------------------- #
# exact selection function
# --------------------------------------------------------------------------- #
def build_G(rho_pop, n_q=240, n_m1=400, gh_n=24, n_v=4000):
    """G(zeta) = E_pop E_eps Phi((V + zeta)/s); returns (zeta_grid, G)."""
    ghx, ghw = roots_hermitenorm(gh_n)
    ghw = ghw / np.sqrt(2 * np.pi)
    qg = np.geomspace(0.012, 1.0, n_q)
    m1g = np.geomspace(1.5, 130.0, n_m1)
    W = rho_pop(m1g[:, None] * np.ones_like(qg)[None, :],
                np.ones_like(m1g)[:, None] * qg[None, :])
    # trapz weights
    wm = np.gradient(m1g); wq = np.gradient(qg)
    W = W * wm[:, None] * wq[None, :]
    mc = lambda m1, m2: (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2
    Vvals = []
    Vw = []
    e1 = 1.0 + A1 * ghx
    e2 = 1.0 + A2 * ghx
    for b, q in enumerate(qg):
        kap = mc(np.maximum(e1[:, None], 1e-3), q * np.maximum(e2[None, :], 1e-3)) \
              / mc(1.0, q)
        lk = (5.0 / 6.0) * np.log(np.maximum(kap, 1e-9))     # (gh, gh)
        wk = ghw[:, None] * ghw[None, :]
        lmc = (5.0 / 6.0) * np.log(mc(m1g, q * m1g))          # (m1,)
        V = lmc[:, None] + lk.ravel()[None, :]                # (m1, gh^2)
        Vvals.append(V.ravel())
        Vw.append((W[:, b][:, None] * wk.ravel()[None, :]).ravel())
    Vvals = np.concatenate(Vvals); Vw = np.concatenate(Vw)
    vlo, vhi = Vvals.min(), Vvals.max()
    vg = np.linspace(vlo - 0.01, vhi + 0.01, n_v)
    dv = vg[1] - vg[0]
    pos = (Vvals - vg[0]) / dv
    i0 = np.clip(pos.astype(int), 0, n_v - 2)
    frac = pos - i0
    P = np.bincount(i0, Vw * (1 - frac), minlength=n_v) \
        + np.bincount(i0 + 1, Vw * frac, minlength=n_v)
    P = P / P.sum()
    zg = np.linspace(-vhi - 8.0, -vlo + 8.0, 3000)
    G = np.array([np.sum(P * ndtr((vg + z) / S_DL)) for z in zg])
    return zg, G


def run_mu(catalog_path, out_path, h0_grid):
    grids = make_grids()
    rho_pop = PopDensity()
    zg, G = build_G(rho_pop)
    with h5py.File(catalog_path) as f:
        cz = f["z"][:]
    lndl = np.log(np.interp(cz, grids["z"], np.maximum(grids["dl"], 1e-9)))
    c0 = np.log(SNR_REF * 1000.0 / SNR_THRESHOLD) - (5.0 / 6.0) * np.log(30.0)
    zeta_ref = c0 + (5.0 / 6.0) * np.log1p(cz) - lndl
    wj = 1.0 / (1.0 + cz)
    delta = np.log(h0_grid / H0_REF)
    mu = np.array([np.sum(wj * np.interp(zeta_ref + d, zg, G)) for d in delta])
    mu_uniform = np.array([np.mean(np.interp(zeta_ref + d, zg, G)) for d in delta])
    # realized detected fraction estimate at truth (uniform-host draw x rate acc)
    frac_pred = np.mean(wj * np.interp(zeta_ref, zg, G))
    np.savez_compressed(out_path, H0=h0_grid, ln_mu=np.log(mu / wj.sum()),
                        ln_mu_unnorm=np.log(mu), frac_pred=frac_pred,
                        mu_uniform=mu_uniform, catalog_path=str(catalog_path))
    print(f"wrote {out_path}  predicted detected fraction {frac_pred:.4e}")
    return mu


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--events", required=True)
    ap.add_argument("--catalog", required=True)
    ap.add_argument("--out_tag", required=True)
    ap.add_argument("--h0_grid", nargs=3, type=float, default=[58.0, 78.0, 161])
    ap.add_argument("--nevents", type=int, default=None)
    ap.add_argument("--skip_mu", action="store_true")
    args = ap.parse_args(argv)
    H0s = np.linspace(args.h0_grid[0], args.h0_grid[1], int(round(args.h0_grid[2])))
    run_numerators(args.events, args.catalog,
                   EXP / "results" / f"oracle_num_{args.out_tag}.npz", H0s,
                   nevents=args.nevents)
    if not args.skip_mu:
        run_mu(args.catalog, EXP / "results" / f"oracle_mu_{args.out_tag}.npz", H0s)


if __name__ == "__main__":
    main()
