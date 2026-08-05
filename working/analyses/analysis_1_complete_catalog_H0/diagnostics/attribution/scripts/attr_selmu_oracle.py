#!/usr/bin/env python3
"""TASK 1 -- the EXACT selection-function oracle:  mu(H0) and d ln mu/dH0 in closed
form, to be compared against darksirens' injection-based Monte Carlo.

The selection side has only ever been ANCHORED to darksirens' injection estimate
(``|Delta log mu| = 0`` in every attribution run), never VERIFIED.  Every channel
of the mock's detection rule is closed-form (``attr_selmu_pdet.py``), so mu is a
quadrature object and the anchor can be replaced by a measurement.

------------------------------------------------------------------------------
THE CONSTRUCTION
------------------------------------------------------------------------------
darksirens' selection integral, in the canonical basis (m1det, q, dL, chieff, Omega)
where every Jacobian of the (m1src, m2src, z) change of variables cancels
(ATTRIBUTION.md A3), is

    mu(H0) = SUM_p INT dz p_z(z|p) (1+z)^(gamma-1) F(z; H0),
    F(z;H0) = INT dm1src dq p_mq(m1src,q) P_det(m1src(1+z), q m1src(1+z), dL(z;H0))

The chieff channel integrates to 1 EXACTLY: ``log_p_pop`` is a product of a mass-q
factor, a spin factor and (1+z)^(gamma-1) (``parametric.py::log_p_pop``), and
``snr_amplitude`` never reads chieff.

TWO EXACT REDUCTIONS make this cheap and free of any H0 interpolation.

(1) ``P_det`` depends on (m1det, m2det, dL) only through ``t = ln(rho_true/8)/s``
    and the mass ratio q (measured exact to 1.8e-15 in ``attr_selmu_pdet``), and
    ``t = a(m1src,q) + b(z,H0)`` with

        a = [ln(1000 SNR_REF/8) + (5/6)(ln Mc_src(m1src,q) - ln 30)] / s
        b = [(5/6) ln(1+z) - ln dL(z;H0)] / s.

    Therefore  F(z;H0) = G(b(z,H0)),  G the CDF of the ONE-dimensional
    W = eps - V,  eps ~ N(0,1) the distance-noise latent,
    V = a(m1src,q) + (5/6) ln R(x1,x2;q)/s  the population + mass-noise latent.
    G is built ONCE by quadrature; G' = dG/db comes from the SAME construction
    with the normal PDF in place of the CDF, so d mu/dH0 is ANALYTIC.

(2) For flat wCDM at fixed (Om0,w0,wa), dL(z;H0) = (H0_fid/H0) dL(z;H0_fid)
    EXACTLY, so b(z,H0) = b(z,H0_fid) + ln(H0/H0_fid)/s and

        d ln mu/dH0  =  (1/(s H0)) * <G'(b)> / <G(b)>

    over the SAME host measure at every H0.  Both the scaling and the derivative
    are cross-checked against darksirens' own ``dL_of_z`` and a step-halving
    finite difference.

The volume factor cancels exactly.  With ``volume_weighted=False`` (the complete
catalog's convention here) ``kw_g = (1/n_pix)/Z(z_g)``,
``Z(z_g) = INT N(z;z_g,sig_g) g(z) dz``, and the evaluator reapplies ``g(z)`` in
front; ``g(z;H0) = H0^-3 g~(z)``, so ``g/Z`` carries no H0.  Verified.

------------------------------------------------------------------------------
THE HOST MEASURES
------------------------------------------------------------------------------
``N_obs[p] == ngals[p]`` and ``Z_global == SUM_p ngals[p]`` (both verified here),
so the per-pixel amplitude cancels the per-row kernel normalisation and

  mu_kde(H0)  = (1/Ntot) INT dz g~(z)(1+z)^(gamma-1) D~(z) G(b(z,H0))
                D~(z) = SUM_g N(z; z_g, sig_g)/Z~(z_g)   [the survey-summed KDE]
  mu_delta    = (1/Ntot) SUM_g [g~(z_g)/Z~(z_g)] (1+z_g)^(gamma-1) G(b(z_g,H0))
  mu_unif     = (1/Ntot) SUM_g (1+z_g)^(gamma-1) G(b(z_g,H0))
  mu_norate   = (1/Ntot) SUM_g G(b(z_g,H0))

``kde`` is exactly what darksirens conditions on -- the object the injections
estimate.  ``delta`` is the zero-bandwidth (exact-host) limit of the SAME prior.
``unif`` is the mock's OWN generative host prior (``stage_events`` draws the host
uniformly over catalog rows and accepts with (1+z)^(gamma-1)) -- the TASK-3 arm.
``norate`` drops the rate factor: the rate-convention lever.

Every catalog galaxy enters; the sums are carried on a 1e-6 redshift lattice with
per-bin first moments, which is exact to O(dz^2 f'') ~ 1e-10 and is convergence
checked by halving.

Outputs: results/attr_selmu_<tracer>.{json,npz}
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import ndtr
from scipy.signal import fftconvolve

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import attr_selmu_pdet as PD                                       # noqa: E402

H0_FID = 67.74
S_DL = PD.S_DL
LOG_D0 = PD.LOG_D0
SQ2PI = np.sqrt(2.0 * np.pi)
DZ_SCALE = 3.0e-3

# --- v3 (2026-08-01): the mock's detection rule is now rho_obs = rho_opt + N(0,1) --
# so P_det(theta) = Phi((rho_opt(theta) - 8)/sigma_rho) EXACTLY -- one Gaussian CDF
# instead of the v2 two-dimensional Gauss-Hermite average over the mass-noise
# latents.  The reduction that makes mu(H0) one-dimensional is unchanged in form:
#
#   rho_opt = A(m1src, q) * exp(b(z, H0)),
#     A = 1000 * SNR_REF_DETECT * (Mc_src/30)^(5/6)      [population only]
#     b = (5/6) ln(1+z) - ln dL(z;H0)                    [host only]
#   and dL(z;H0) = (H0_fid/H0) dL(z;fid) EXACTLY, so b(z,H0) = b(z,fid)
#   + ln(H0/H0_fid): the SAME additive shift the v2 code already applies, with the
#   coordinate scale S_COORD = 1 instead of S_DL.
#
#   G(b)  = E_A[ Phi( (A e^b - 8)/sigma_rho ) ]
#   G'(b) = E_A[ (A e^b/sigma_rho) phi( (A e^b - 8)/sigma_rho ) ]
#   d ln mu/dH0 = <G'(b)>/<G(b)> / H0
#
# Everything else in this file -- the host measures, every anchor, the lattice, the
# KDE convolution, the arms and the convergence battery -- is untouched, so with
# --pe_model v2 every product of CLOSURE.md 11 is reproduced bit for bit.
SIGMA_RHO = float(getattr(PD.G, "SIGMA_RHO", 1.0))
SNR_REF_DETECT = float(PD.G.SNR_REF_DETECT)
SNR_THR = float(PD.G.SNR_THRESHOLD)


def trap_w(x):
    w = np.empty_like(x)
    w[1:-1] = 0.5 * (x[2:] - x[:-2])
    w[0] = 0.5 * (x[1] - x[0])
    w[-1] = 0.5 * (x[-1] - x[-2])
    return w


# ================================================================================
# G(b): the one-dimensional selection kernel
# ================================================================================
def build_G(model, th_mix, n_m1, n_q, n_gh, dv,
            v_lo=0.0, v_hi=130.0, m1_lo=1.0, m1_hi=140.0,
            b_lo=-150.0, b_hi=10.0):
    import jax.numpy as jnp
    m1 = np.linspace(m1_lo, m1_hi, n_m1)
    q = np.linspace(1.0 / n_q, 1.0, n_q)
    w1 = trap_w(m1)
    wq = trap_w(q)
    m1f = np.repeat(m1, n_q)
    qf = np.tile(q, n_m1)
    P = np.empty(m1f.size)
    CHM = 400_000                      # mass_q_density allocates (n, 200) grids
    for i0 in range(0, m1f.size, CHM):
        sl = slice(i0, min(i0 + CHM, m1f.size))
        P[sl] = np.asarray(model.mixture.mass_q_density(
            jnp.asarray(m1f[sl]), jnp.asarray(qf[sl]), th_mix))
    P = P.reshape(n_m1, n_q)
    Wmq = P * w1[:, None] * wq[None, :]
    mass_norm = float(Wmq.sum())
    mc = m1[:, None] * q[None, :] ** 0.6 / (1.0 + q[None, :]) ** 0.2
    a = (LOG_D0 + (5.0 / 6.0) * (np.log(mc) - np.log(30.0))) / S_DL

    x, wgh = PD.gh_nodes(n_gh)
    nb = int(round((v_hi - v_lo) / dv)) + 1
    h = np.zeros(nb)
    mom = np.zeros(nb)
    lost = 0.0
    wf = Wmq.ravel()
    for i1 in range(n_gh):
        lr1 = np.log(max(1.0 + PD.F1 * x[i1], PD.A_FLOOR))
        for i2 in range(n_gh):
            a2 = max(1.0 + PD.F2 * x[i2], PD.A_FLOOR)
            a1 = max(1.0 + PD.F1 * x[i1], PD.A_FLOOR)
            lr = (0.6 * (lr1 + np.log(a2))
                  - 0.2 * np.log((a1 + q * a2) / (1.0 + q)))
            V = (a + (5.0 / 6.0) * lr[None, :] / S_DL).ravel()
            w = wf * (wgh[i1] * wgh[i2])
            k = np.rint((V - v_lo) / dv).astype(np.int64)
            ok = (k >= 0) & (k < nb)
            if not ok.all():
                lost += float(w[~ok].sum())
            h += np.bincount(k[ok], weights=w[ok], minlength=nb)
            mom += np.bincount(k[ok], weights=w[ok] * (V[ok] - (v_lo + k[ok] * dv)),
                               minlength=nb)
    Vgrid = v_lo + dv * np.arange(nb)
    nbk = int(round((b_hi - b_lo) / dv)) + 1
    bg = b_lo + dv * np.arange(nbk)
    n_lag = nbk + nb - 1
    xs = (b_lo + v_lo) + dv * np.arange(n_lag)
    Phi = ndtr(xs)
    phi = np.exp(-0.5 * xs ** 2) / SQ2PI
    G = (fftconvolve(Phi, h[::-1], mode="valid")[:nbk]
         + fftconvolve(phi, mom[::-1], mode="valid")[:nbk])
    Gp = (fftconvolve(phi, h[::-1], mode="valid")[:nbk]
          + fftconvolve(-xs * phi, mom[::-1], mode="valid")[:nbk])
    occ = h > 0
    diag = {"mass_norm": mass_norm, "weight_outside_V_range": lost,
            "V_mean": float((h * Vgrid + mom).sum() / h.sum()),
            "V_min_occupied": float(Vgrid[occ][0]),
            "V_max_occupied": float(Vgrid[occ][-1]),
            "G_at_b_max": float(G[-1]), "G_at_b_min": float(G[0])}
    return bg, G, Gp, diag


def build_G_v3(model, th_mix, n_m1, n_q, n_gh, dv,
               v_lo=-2.0, v_hi=12.0, m1_lo=1.0, m1_hi=140.0,
               b_lo=-18.0, b_hi=4.0, chunk=400):
    """The v3 selection kernel.

    ``n_gh`` is accepted and IGNORED (there is no mass-noise latent in v3); it is
    kept in the signature so the convergence battery and the CLI are unchanged.

    ``ln A = ln(1000 SNR_REF_DETECT) + (5/6)(ln Mc_src - ln 30)`` is binned on a
    uniform lattice of spacing ``dv`` with per-bin FIRST MOMENTS (the same device
    the v2 kernel uses for its ``V`` lattice), and

        G (b) = SUM_j [ h_j Phi(u_j) + m_j Phi'(u_j) du_j/dlnA ]
        G'(b) = dG/db, obtained from the same sum with the chain rule,
        u_j = (exp(lnA_j + b) - 8)/sigma_rho,   du/dlnA = du/db = exp(lnA_j+b)/sigma_rho
    """
    import jax.numpy as jnp
    m1 = np.linspace(m1_lo, m1_hi, n_m1)
    q = np.linspace(1.0 / n_q, 1.0, n_q)
    w1 = trap_w(m1)
    wq = trap_w(q)
    m1f = np.repeat(m1, n_q)
    qf = np.tile(q, n_m1)
    P = np.empty(m1f.size)
    CHM = 400_000
    for i0 in range(0, m1f.size, CHM):
        sl = slice(i0, min(i0 + CHM, m1f.size))
        P[sl] = np.asarray(model.mixture.mass_q_density(
            jnp.asarray(m1f[sl]), jnp.asarray(qf[sl]), th_mix))
    P = P.reshape(n_m1, n_q)
    Wmq = P * w1[:, None] * wq[None, :]
    mass_norm = float(Wmq.sum())
    mc = m1[:, None] * q[None, :] ** 0.6 / (1.0 + q[None, :]) ** 0.2
    lnA = (np.log(1000.0 * SNR_REF_DETECT)
           + (5.0 / 6.0) * (np.log(mc) - np.log(30.0))).ravel()
    wf = Wmq.ravel()

    nb = int(round((v_hi - v_lo) / dv)) + 1
    k = np.rint((lnA - v_lo) / dv).astype(np.int64)
    ok = (k >= 0) & (k < nb)
    lost = float(wf[~ok].sum())
    h = np.bincount(k[ok], weights=wf[ok], minlength=nb)
    mom = np.bincount(k[ok], weights=wf[ok] * (lnA[ok] - (v_lo + k[ok] * dv)),
                      minlength=nb)
    Agrid = v_lo + dv * np.arange(nb)
    occ = h > 0
    Ag = Agrid[occ]
    hg = h[occ]
    mg = mom[occ]

    nbk = int(round((b_hi - b_lo) / dv)) + 1
    bg = b_lo + dv * np.arange(nbk)
    G = np.empty(nbk)
    Gp = np.empty(nbk)
    for i0 in range(0, nbk, chunk):
        bb = bg[i0:i0 + chunk][:, None]
        r = np.exp(Ag[None, :] + bb)                  # rho_opt at the node
        u = (r - SNR_THR) / SIGMA_RHO
        Phi = ndtr(u)
        phi = np.exp(-0.5 * u ** 2) / SQ2PI
        d1 = r / SIGMA_RHO                            # du/dlnA = du/db
        # value: bin value + first-moment correction in lnA
        G[i0:i0 + chunk] = (hg[None, :] * Phi + mg[None, :] * phi * d1).sum(1)
        # derivative in b: d/db [Phi] = phi * d1 ; d/db [phi*d1] = d1*(phi*d1
        #   - u*phi*d1) = phi*d1*(1 - u*d1) ... note d(d1)/db = d1
        Gp[i0:i0 + chunk] = (hg[None, :] * phi * d1
                             + mg[None, :] * phi * d1 * (1.0 - u * d1)).sum(1)
    diag = {"mass_norm": mass_norm, "weight_outside_V_range": lost,
            "lnA_mean": float((h * Agrid + mom).sum() / h.sum()),
            "lnA_min_occupied": float(Agrid[occ][0]),
            "lnA_max_occupied": float(Agrid[occ][-1]),
            "G_at_b_max": float(G[-1]), "G_at_b_min": float(G[0]),
            "sigma_rho": SIGMA_RHO, "snr_threshold": SNR_THR,
            "kernel": "v3: P_det = Phi((rho_opt - 8)/sigma_rho)"}
    return bg, G, Gp, diag


class Gfun:
    """G and G' on a uniform b lattice, Catmull-Rom (cubic) interpolation."""

    def __init__(self, b, G, Gp):
        self.b0 = float(b[0])
        self.db = float(b[1] - b[0])
        self.G = np.ascontiguousarray(G)
        self.Gp = np.ascontiguousarray(Gp)
        self.n = G.size

    def _interp(self, y, b):
        t = (np.asarray(b) - self.b0) / self.db
        i = np.clip(np.floor(t).astype(np.int64), 1, self.n - 3)
        f = t - i
        y0, y1, y2, y3 = y[i - 1], y[i], y[i + 1], y[i + 2]
        return (y1 + 0.5 * f * (y2 - y0 + f * (2.0 * y0 - 5.0 * y1 + 4.0 * y2 - y3
                                               + f * (3.0 * (y1 - y2) + y3 - y0))))

    def __call__(self, b):
        return self._interp(self.G, b)

    def deriv(self, b):
        return self._interp(self.Gp, b)


S_COORD = S_DL          # v2; set to 1.0 by main() when --pe_model v3


def b_of_z(z, dL):
    """The host coordinate.  v2: b = [(5/6)ln(1+z) - ln dL]/S_DL.  v3: the same
    quantity with S_COORD = 1, because the v3 kernel's argument is ln(rho_opt)
    - ln A rather than a distance-noise z-score."""
    return ((5.0 / 6.0) * np.log1p(z) - np.log(dL)) / S_COORD


# ================================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tracer", choices=["gal", "agn"], default="gal")
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--survey_override", default=None,
                    help="OPT-IN (ENDGAME): analyse a different survey block -- used "
                         "only for the declared-photo-z-kernel scan.  Absent, every "
                         "product of the final sweep is reproduced unchanged.")
    ap.add_argument("--dz_scale", type=float, default=None,
                    help="OPT-IN (ENDGAME): the survey block's declared "
                         "photo-z scale, dz = dz_scale * (1+z).  Default: read "
                         "from --survey_override's own dz_scale attribute, else "
                         "the campaign constant DZ_SCALE.  The oracle's own "
                         "log_kw anchor catches any mismatch with the block.")
    ap.add_argument("--pe_model", choices=("v2", "v3"), default=None,
                    help="Measurement family of the DATASET being analysed.  "
                         "Default: read from the seed's injections file "
                         "(attr 'pe_model'), else v2.  v3 uses the closed-form "
                         "P_det = Phi((rho_opt - 8)/sigma_rho); with v2 every "
                         "product of CLOSURE.md 11 is reproduced bit for bit.")
    ap.add_argument("--dataroot", default=None,
                    help="Root holding seed<N>/ (default: working/data).")
    ap.add_argument("--events", default=None,
                    help="Events file handed to the bridge (only used for the "
                         "one reference likelihood evaluation and its anchor).")
    ap.add_argument("--n_m1", type=int, default=2800)
    ap.add_argument("--n_q", type=int, default=1400)
    ap.add_argument("--n_gh", type=int, default=32)
    ap.add_argument("--dv", type=float, default=1.0e-3)
    ap.add_argument("--dz_lat", type=float, default=1.0e-6)
    ap.add_argument("--dz_kde", type=float, default=1.0e-5)
    ap.add_argument("--n_sig_kde", type=float, default=9.0)
    ap.add_argument("--conv", action="store_true",
                    help="G(b) battery + the catalog lattice halving")
    ap.add_argument("--conv_lat", action="store_true",
                    help="catalog lattice halving only (the G(b) battery is "
                         "catalog-independent, so it is run once)")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--outdir", default=str(ROOT / "results"))
    args = ap.parse_args(argv)
    tag = args.tag or args.tracer
    od = Path(args.outdir)
    t00 = time.time()

    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")
    import jax.numpy as jnp
    import attr_ds_bridge as bridge
    from darksirens.gw.populations.registry import get_model
    from darksirens.redshift.grid import zgrid as ZGRID_J
    from darksirens.redshift.catalog import _row_log_kernel_norms
    from darksirens.redshift.prior import prepare_redshift_prior_state
    from darksirens.utils.cosmology import dL_of_z

    kw = dict(kde_window=4096) if args.tracer == "gal" else {}
    if args.survey_override:
        kw["survey_override"] = args.survey_override
    if args.dataroot:
        kw["dataroot"] = args.dataroot
    if args.events:
        kw["events"] = args.events
    B = bridge.build(tracer=args.tracer, seed=args.seed, h0=H0_FID, **kw)

    # --- which measurement family does this dataset carry? ----------------------
    global S_COORD
    pe_model = args.pe_model
    if pe_model is None:
        import h5py as _h5
        try:
            with _h5.File(B.paths["injections"], "r") as _f:
                pe_model = str(_f.attrs.get("pe_model", "v2"))
        except Exception:
            pe_model = "v2"
    S_COORD = 1.0 if pe_model == "v3" else S_DL
    _build_G = build_G_v3 if pe_model == "v3" else build_G
    print(f"[pe_model] {pe_model}   S_COORD={S_COORD}", flush=True)
    # The declared photo-z scale of the block actually being analysed.  With no
    # override this is the campaign constant, so every previous product is
    # reproduced bit for bit; the log_kw anchor below verifies it either way.
    DZS = args.dz_scale
    if DZS is None and args.survey_override:
        import h5py as _h5
        with _h5.File(args.survey_override, "r") as _f:
            DZS = float(_f.attrs.get("dz_scale", DZ_SCALE))
    if DZS is None:
        DZS = DZ_SCALE
    print(f"[dz] declared photo-z scale in use: dz = {DZS:g} * (1+z)")
    gamma = float(B.gamma_fid)
    model = get_model("powerlaw+peak", shared_beta=True, shared_spin=True,
                      shared_gamma=True)
    th_mix = jnp.asarray(np.asarray(B.pop_fid)[:model.mixture.n_params])
    ZGRID = np.asarray(ZGRID_J)

    # ---------------- G(b) ------------------------------------------------------
    print("[G] building the selection kernel ...", flush=True)
    tG = time.time()
    bg, Gv, Gpv, gdiag = _build_G(model, th_mix, args.n_m1, args.n_q, args.n_gh,
                                  args.dv)
    Gf = Gfun(bg, Gv, Gpv)
    print(f"[G] {gdiag}  ({time.time()-tG:.0f}s)", flush=True)

    # the alternative kernels are BUILT here and EVALUATED after the host measures
    # exist, because the quantity that has to be converged is d ln mu/dH0 itself,
    # not G in a tail the host measure never reaches.
    Galt = {}
    if args.conv:
        for name, kwx in (("n_m1x2", dict(n_m1=2 * args.n_m1)),
                          ("n_qx2", dict(n_q=2 * args.n_q)),
                          ("n_ghx2", dict(n_gh=2 * args.n_gh)),
                          ("dv_half", dict(dv=0.5 * args.dv)),
                          ("range_wide", dict(m1_lo=0.5, m1_hi=190.0))):
            kk = dict(n_m1=args.n_m1, n_q=args.n_q, n_gh=args.n_gh, dv=args.dv)
            kk.update(kwx)
            b2, G2, Gp2, _ = _build_G(model, th_mix, **kk)
            Galt[name] = Gfun(b2, G2, Gp2)
            print(f"[G-conv] built {name}", flush=True)

    # ---------------- the H0 -> b shift, verified against darksirens ------------
    zt = np.linspace(1e-4, 1.05, 5001)
    dl_fid_t = np.asarray(dL_of_z(jnp.asarray(zt), jnp.float64(H0_FID),
                                  B.cosmo0.Om0, B.cosmo0.w0, B.cosmo0.wa))
    shift_err = {}
    for h in (50.0, 60.0, H0_FID, 80.0, 100.0):
        dl = np.asarray(dL_of_z(jnp.asarray(zt), jnp.float64(h), B.cosmo0.Om0,
                                B.cosmo0.w0, B.cosmo0.wa))
        shift_err[str(h)] = float(np.max(np.abs(np.log(dl) - np.log(dl_fid_t)
                                                + np.log(h / H0_FID))))
    print(f"[check] max |ln dL(z;H0) - ln dL(z;fid) + ln(H0/fid)| = "
          f"{max(shift_err.values()):.3e}", flush=True)

    # ---------------- catalog, kernel norms, anchors ----------------------------
    zgals = np.asarray(B.cat_pe.zgals)
    ngals = np.asarray(B.cat_pe.ngals)
    npix, nmax = zgals.shape
    Ntot = int(ngals.sum())
    st = prepare_redshift_prior_state(
        "dark_sirens", B.cosmo0, B.survey_p, B.cat_pe, mark_model="none",
        mark_params=None, mark_names=(), materialize_state=True,
        catalog_sky_weighting="field")
    log_g_fid = np.asarray(st.kernels.log_g_grid)
    log_Nobs = np.asarray(st.log_Nobs)
    log_Zg = float(np.asarray(st.log_Z_global))
    dNmiss_max = float(np.asarray(st.dN_miss).max())
    nobs_err = float(np.nanmax(np.abs(np.exp(log_Nobs[ngals > 0]) - ngals[ngals > 0])))
    zg_err = float(abs(np.exp(log_Zg) - Ntot))
    print(f"[anchor] |N_obs-ngals|max={nobs_err:.3e}  |Z_global-Ntot|={zg_err:.3e}  "
          f"max dN_miss={dNmiss_max:.3e}", flush=True)

    # D3 (2026-08-01): the survey block now carries the catalog's PHOTO-Z redshift,
    # which for a galaxy at z ~ 0 can be very slightly NEGATIVE (1 row in 151 M on
    # seed 100 -- the generator does not clip, because clipping the recorded value
    # would censor).  The kernel-normalisation table must therefore extend below
    # zero, or the log_kw anchor picks up that one row as a 0.24 discrepancy.
    ztab = np.arange(-0.06, 1.35 + 1e-9, 2.0e-6)
    sig_tab = DZS * (1.0 + ztab)
    logZ_tab = np.empty(ztab.size)
    CH = 200_000
    for i0 in range(0, ztab.size, CH):
        sl = slice(i0, min(i0 + CH, ztab.size))
        logZ_tab[sl] = np.asarray(_row_log_kernel_norms(
            jnp.asarray(ztab[sl]), jnp.asarray(sig_tab[sl]),
            jnp.ones(ztab[sl].size, bool), jnp.asarray(log_g_fid)))
    log_kw_state = np.asarray(st.kernels.log_kw)
    rng = np.random.default_rng(5)
    occ = np.arange(npix)[ngals > 0]
    rows = rng.choice(occ, size=min(4000, occ.size), replace=False)
    aerr = 0.0
    for r in rows:
        n = int(ngals[r])
        zz = zgals[r, :n]
        pred = -np.log(n) - np.interp(zz, ztab, logZ_tab)
        aerr = max(aerr, float(np.max(np.abs(pred - log_kw_state[r, :n]))))
    print(f"[anchor] max |log_kw(table) - log_kw(darksirens)| = {aerr:.3e} over "
          f"{rows.size} rows", flush=True)
    del log_kw_state

    # g/Z carries no H0 (the exact H0^-3 cancellation)
    zs_c = ztab[::500]
    gz_iso = {}
    lZf = np.asarray(_row_log_kernel_norms(
        jnp.asarray(zs_c), jnp.asarray(DZS * (1 + zs_c)),
        jnp.ones(zs_c.size, bool), jnp.asarray(log_g_fid)))
    base = np.interp(zs_c, ZGRID, log_g_fid) - lZf
    for h in (50.0, 100.0):
        sth = prepare_redshift_prior_state(
            "dark_sirens", B.cosmo0._replace(H0=jnp.float64(h)), B.survey_p, B.cat_pe,
            mark_model="none", mark_params=None, mark_names=(),
            materialize_state=True, catalog_sky_weighting="field")
        lgh = np.asarray(sth.kernels.log_g_grid)
        lZh = np.asarray(_row_log_kernel_norms(
            jnp.asarray(zs_c), jnp.asarray(DZS * (1 + zs_c)),
            jnp.ones(zs_c.size, bool), jnp.asarray(lgh)))
        d = (np.interp(zs_c, ZGRID, lgh) - lZh) - base
        m = zs_c > 0.002
        gz_iso[str(h)] = float(np.max(np.abs(d[m])))
        del sth, lgh, lZh
    print(f"[check] max |d ln(g/Z)| between H0 50/100 and fid: "
          f"{max(gz_iso.values()):.3e}", flush=True)
    del st

    # ---------------- the host lattice ------------------------------------------
    n_clamped = {"n": 0, "z_min": 0.0}

    def build_lattice(dz):
        """The point-measure host lattice.

        A photo-z redshift can be marginally negative (see the ztab comment); the
        MODEL's own support starts at zero -- darksirens integrates its kernels on a
        zgrid that begins at 0 -- so such a row is clamped here, exactly as the
        likelihood effectively treats it.  The number clamped and their minimum are
        recorded; on seed 100 it is ONE galaxy of 151,179,870 at z_obs = -6e-4,
        whose weight in mu is ~1e-8 of the total."""
        nlat = int(np.ceil(1.35 / dz)) + 2
        cnt = np.zeros(nlat)
        mom = np.zeros(nlat)
        n_clamped["n"] = 0          # per build; the routine is called twice
        for i0 in range(0, npix, 512):
            blk = zgals[i0:i0 + 512]
            nb = ngals[i0:i0 + 512]
            msk = np.arange(nmax)[None, :] < nb[:, None]
            zz = blk[msk]
            if zz.size == 0:
                continue
            neg = zz < 0.0
            if neg.any():
                n_clamped["n"] += int(neg.sum())
                n_clamped["z_min"] = min(n_clamped["z_min"], float(zz.min()))
                zz = np.maximum(zz, 0.0)
            k = np.rint(zz / dz).astype(np.int64)
            cnt += np.bincount(k, minlength=nlat)[:nlat]
            mom += np.bincount(k, weights=zz - k * dz, minlength=nlat)[:nlat]
        return np.arange(nlat) * dz, cnt, mom

    tL = time.time()
    zlat, cnt, mom1 = build_lattice(args.dz_lat)
    print(f"[lattice] dz={args.dz_lat:g}  {zlat.size:,} nodes  "
          f"sum={cnt.sum():.1f} (Ntot={Ntot})  ({time.time()-tL:.0f}s)", flush=True)
    keep = cnt > 0
    zL = zlat[keep]
    cL = cnt[keep]
    mL = mom1[keep]

    gL = np.exp(np.interp(zL, ZGRID, log_g_fid))
    ZL = np.exp(np.interp(zL, ztab, logZ_tab))
    rateL = (1.0 + zL) ** (gamma - 1.0)
    dlL = np.asarray(dL_of_z(jnp.asarray(np.maximum(zL, 1e-9)), jnp.float64(H0_FID),
                             B.cosmo0.Om0, B.cosmo0.w0, B.cosmo0.wa))
    bL = b_of_z(zL, dlL)

    W = {"delta": cL * gL / ZL * rateL,
         "unif": cL * rateL,
         "norate": cL.copy()}

    # ---------------- the survey-summed KDE density D~(z) ------------------------
    tK = time.time()
    dzk = args.dz_kde
    nk = int(np.ceil(1.35 / dzk)) + 2
    kk = np.rint(zL / dzk).astype(np.int64)
    Hs = np.bincount(kk, weights=cL / ZL, minlength=nk)[:nk]
    Ms = np.bincount(kk, weights=(cL * (zL - kk * dzk) + mL) / ZL,
                     minlength=nk)[:nk]
    zk = np.arange(nk) * dzk
    sk = DZS * (1.0 + zk)
    half = int(np.ceil(args.n_sig_kde * sk.max() / dzk))
    print(f"[kde] lattice {nk:,} nodes, half-window {half}", flush=True)
    import jax
    Hj = jnp.asarray(Hs)
    Mj = jnp.asarray(Ms)
    Sj = jnp.asarray(sk)
    offs = jnp.arange(-half, half + 1)

    CB = 2048

    @jax.jit
    def _chunk(k0):
        ks = k0 + jnp.arange(CB)
        j = ks[:, None] - offs[None, :]
        jc = jnp.clip(j, 0, nk - 1)
        ok = (j >= 0) & (j < nk)
        s = Sj[jc]
        d = (offs[None, :] * dzk)
        e = jnp.exp(-0.5 * (d / s) ** 2) / (SQ2PI * s)
        dk = (d / s ** 2) * e
        return jnp.where(ok, Hj[jc] * e + Mj[jc] * dk, 0.0).sum(axis=1)

    Dk = np.zeros(nk)
    for k0 in range(0, nk, CB):
        n = min(CB, nk - k0)
        Dk[k0:k0 + n] = np.asarray(_chunk(k0))[:n]
    print(f"[kde] D~ built ({time.time()-tK:.0f}s); "
          f"int D~ dz = {Dk.sum()*dzk:.6f} vs sum 1/Z~ = {(cL/ZL).sum():.6f}",
          flush=True)
    gk = np.exp(np.interp(zk, ZGRID, log_g_fid))
    dlk = np.asarray(dL_of_z(jnp.asarray(np.maximum(zk, 1e-9)), jnp.float64(H0_FID),
                             B.cosmo0.Om0, B.cosmo0.w0, B.cosmo0.wa))
    bk = b_of_z(zk, dlk)
    wt = np.full(nk, dzk)
    wt[0] = wt[-1] = 0.5 * dzk
    nu_kde = gk * (1.0 + zk) ** (gamma - 1.0) * Dk * wt

    # ---------------- d ln mu/dH0 over the H0 grid ------------------------------
    H0_GRID = np.array([50.0, 55.0, 60.0, 62.5, 65.0, 67.74, 70.0, 72.5, 75.0,
                        80.0, 85.0, 90.0, 95.0, 100.0])
    arms = {"kde": (bk, nu_kde)}
    arms.update({a: (bL, W[a]) for a in ("delta", "unif", "norate")})
    res = {a: {"H0": H0_GRID.tolist(), "log_mu": [], "dlnmu": []} for a in arms}
    for h in H0_GRID:
        sh = np.log(h / H0_FID) / S_COORD
        for a, (bb, ww) in arms.items():
            den = float(np.dot(ww, Gf(bb + sh)))
            num = float(np.dot(ww, Gf.deriv(bb + sh)))
            res[a]["log_mu"].append(float(np.log(den)))
            res[a]["dlnmu"].append(float(num / den / (S_COORD * h)))
    i_fid = int(np.argmin(np.abs(H0_GRID - H0_FID)))
    for a in arms:
        print(f"[mu] {a:>7}  d ln mu/dH0 (67.74) = {res[a]['dlnmu'][i_fid]:+.8e}",
              flush=True)
    # per-galaxy detection probability at truth -- directly comparable to the
    # generator's OWN realised event-draw bookkeeping (events_meta 'realised'):
    #   detected_fraction          = <acc * P_det>  = mu_unif / Ntot
    #   detected_fraction_snr_only = <P_det>        = mu_norate / Ntot
    per_gal = {a: float(np.dot(arms[a][1], Gf(arms[a][0])) / Ntot)
               for a in ("unif", "norate", "delta")}
    print(f"[gen-check] <acc*Pdet> = {per_gal['unif']:.6e}   "
          f"<Pdet> = {per_gal['norate']:.6e}   (per catalog galaxy)", flush=True)

    # finite difference with step halving, using darksirens' own dL_of_z at each H0
    fd = {a: {} for a in arms}
    for a, (bb, ww) in arms.items():
        zz = zk if a == "kde" else zL
        for dh in (1.0, 0.5, 0.25, 0.125, 0.0625):
            lm = []
            for h in (H0_FID - dh, H0_FID + dh):
                dl = np.asarray(dL_of_z(jnp.asarray(np.maximum(zz, 1e-9)),
                                        jnp.float64(h), B.cosmo0.Om0, B.cosmo0.w0,
                                        B.cosmo0.wa))
                lm.append(np.log(float(np.dot(ww, Gf(b_of_z(zz, dl))))))
            fd[a][str(dh)] = float((lm[1] - lm[0]) / (2.0 * dh))
        print(f"[fd] {a:>7}  " + "  ".join(f"{k}:{v:+.8e}" for k, v in fd[a].items()),
              flush=True)

    # ---- G(b) convergence, measured on d ln mu/dH0 itself ----------------------
    gconv = {}
    for name, f2 in Galt.items():
        gconv[name] = {}
        for a, (bb, ww) in arms.items():
            d2 = float(np.dot(ww, f2.deriv(bb)) / np.dot(ww, f2(bb))
                       / (S_COORD * H0_FID))
            gconv[name][a] = {"dlnmu": d2,
                              "abs_change": d2 - res[a]["dlnmu"][i_fid]}
        print(f"[G-conv] {name}: " + "  ".join(
            f"{a}:{gconv[name][a]['abs_change']:+.2e}" for a in arms), flush=True)
    # and the relative accuracy of G where the host measure actually lives
    if Galt:
        bb, ww = arms["delta"]
        wgt = ww * Gf(bb)
        o = np.argsort(bb)
        cw = np.cumsum(wgt[o]) / wgt.sum()
        b_lo_eff = float(bb[o][np.searchsorted(cw, 1e-4)])
        b_hi_eff = float(bb[o][np.searchsorted(cw, 1.0 - 1e-6)])
        bt = np.linspace(b_lo_eff, b_hi_eff, 20001)
        for name, f2 in Galt.items():
            gconv[name]["max_rel_G_in_support"] = float(np.max(np.abs(
                f2(bt) / np.maximum(Gf(bt), 1e-300) - 1.0)))
            gconv[name]["max_rel_Gp_in_support"] = float(np.max(np.abs(
                f2.deriv(bt) / np.maximum(Gf.deriv(bt), 1e-300) - 1.0)))
        gconv["_b_support"] = [b_lo_eff, b_hi_eff]
        print(f"[G-conv] b support (1e-4 .. 1-1e-6 of the mu integrand): "
              f"[{b_lo_eff:.2f}, {b_hi_eff:.2f}]", flush=True)

    # lattice convergence: halve dz_lat on the point-measure arms
    lat_conv = {}
    if args.conv or args.conv_lat:
        z2, c2, m2 = build_lattice(0.5 * args.dz_lat)
        k2 = c2 > 0
        zz2 = z2[k2]
        c2 = c2[k2]
        g2 = np.exp(np.interp(zz2, ZGRID, log_g_fid))
        Z2 = np.exp(np.interp(zz2, ztab, logZ_tab))
        r2 = (1.0 + zz2) ** (gamma - 1.0)
        dl2 = np.asarray(dL_of_z(jnp.asarray(np.maximum(zz2, 1e-9)),
                                 jnp.float64(H0_FID), B.cosmo0.Om0, B.cosmo0.w0,
                                 B.cosmo0.wa))
        b2 = b_of_z(zz2, dl2)
        for a, wnew in (("delta", c2 * g2 / Z2 * r2), ("unif", c2 * r2),
                        ("norate", c2)):
            d = float(np.dot(wnew, Gf.deriv(b2)) / np.dot(wnew, Gf(b2))
                      / (S_COORD * H0_FID))
            lat_conv[a] = {"dlnmu_dz_half": d,
                           "abs_change": abs(d - res[a]["dlnmu"][i_fid])}
            print(f"[lat-conv] {a}: {d:+.8e} (change "
                  f"{d - res[a]['dlnmu'][i_fid]:+.2e})", flush=True)

    out = {
        "name": "attr_selmu_oracle", "tracer": args.tracer, "seed": args.seed,
        "pe_model": pe_model, "S_COORD": S_COORD,
        "tag": tag, "H0_fid": H0_FID, "gamma_fid": gamma,
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": B.paths, "n_galaxies": Ntot,
        "grid": {"n_m1": args.n_m1, "n_q": args.n_q, "n_gh": args.n_gh,
                 "dz_scale_declared": DZS,
                 "dv": args.dv, "dz_lat": args.dz_lat, "dz_kde": args.dz_kde,
                 "n_sig_kde": args.n_sig_kde},
        "G_diagnostics": gdiag, "G_convergence": gconv,
        "hosts_clamped_at_z_zero": dict(n_clamped),
        "lattice_convergence": lat_conv,
        "anchors": {"N_obs_minus_ngals_maxabs": nobs_err,
                    "Z_global_minus_Ntot_abs": zg_err,
                    "max_dN_miss": dNmiss_max,
                    "log_kw_table_vs_state_maxabs": aerr,
                    "dL_H0_scaling_maxabs": shift_err,
                    "g_over_Z_H0_independence_maxabs": gz_iso},
        "arms": {a: res[a] for a in arms},
        "per_galaxy_detection_probability_at_truth": per_gal,
        "dlnmu_at_truth": {a: res[a]["dlnmu"][i_fid] for a in arms},
        "fd_step_halving": fd,
    }
    (od / f"attr_selmu_{tag}.json").write_text(json.dumps(out, indent=2))
    np.savez_compressed(od / f"attr_selmu_{tag}.npz",
                        b_grid=bg, G=Gv, Gp=Gpv, H0_grid=H0_GRID,
                        zk=zk, nu_kde=nu_kde, D_kde=Dk, b_zk=bk,
                        zL=zL, bL=bL, **{f"w_{a}": W[a] for a in W},
                        **{f"dlnmu_{a}": np.array(res[a]["dlnmu"]) for a in arms},
                        **{f"logmu_{a}": np.array(res[a]["log_mu"]) for a in arms})
    print(f"Wrote {od/f'attr_selmu_{tag}.json'}   ({time.time()-t00:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
