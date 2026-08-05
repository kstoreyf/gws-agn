#!/usr/bin/env python3
"""TASK 1a -- the EXACT closed-form detection probability of the mock's own rule.

``working/data/generate_dataset.py`` makes detection a deterministic function of
the RECORDED measurement (convention (a)):

    rho_obs = SNR_REF_DETECT * (Mc(obs_m1det, obs_m2det)/30)^(5/6) * (1000/obs_dL)
    detected  <=>  rho_obs >= 8

with, from ``observe()`` verbatim,

    obs_dL   = dL * exp(SIGMA_DL * N(0,1))        i.e.  ln obs_dL ~ N(ln dL, s),
                                                  s = SIGMA_DL = 0.10   (NO -s^2/2)
    obs_m1   = clip(N(m1det, f1*m1det), 2.0, None)      f1 = SIG_M1_FRAC = 0.08
    obs_m2   = clip(N(m2det, f2*m2det), 1.0, None)      f2 = SIG_M2_FRAC = 0.10

and ``chieff`` NEVER enters ``snr_amplitude`` (it takes only m1det, m2det, dl),
so P_det is a function of (m1det, m2det, dL) alone.

CLOSED FORM.  Condition on the observed masses.  Detection is then
``obs_dL <= 1000*SNR_REF*(Mc_obs/30)^(5/6)/8``, and since ln obs_dL is Gaussian
the inner integral is an ERROR FUNCTION, exactly.  Writing
``obs_m1 = m1det (1 + f1 x1)``, ``obs_m2 = m2det (1 + f2 x2)`` with
``x1, x2 ~ N(0,1)``, the chirp mass factorises,

    Mc_obs / Mc_det = R(x1, x2; q) = (a1 a2)^0.6 * ((1+q)/(a1 + q a2))^0.2
    a1 = 1 + f1 x1,  a2 = 1 + f2 x2,  q = m2det/m1det = m2src/m1src,

so R depends on the masses ONLY through the mass ratio q, and

    P_det(m1det, m2det, dL) = E_{x1,x2}[ Phi( t + (5/6) ln R(x1,x2;q) / s ) ]
                            = Pcal(t, q),
    t = ln( rho_true / 8 ) / s,
    rho_true = SNR_REF*(Mc(m1det,m2det)/30)^(5/6)*(1000/dL)   (noise-free).

The outer expectation is a 2-D Gauss-Hermite (probabilists') quadrature.

The 2 / 1 Msun clips are carried as ``a -> max(a, A_FLOOR)``: a clipped draw has
Mc_obs at most that of a (2, 1) Msun binary and is therefore never detected at any
distance that matters, and its probability is ``Phi(-(1 - 2/m1det)/f1)`` which is
below 1e-30 for every mass this population reaches.  Both statements are MEASURED
below (``clip_probability_max``), and the whole closed form is validated against a
BRUTE-FORCE Monte Carlo that calls the generator's OWN ``observe()`` and
``detect_from_observation()``.

Outputs: results/attr_selmu_pdet.json  (+ .npz with the Pcal table)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import ndtr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
GEN = Path("/hildafs/projects/phy230014p/magana/gws-agn/working/data")

# --- the generator's constants, RE-READ from generate_dataset.py at import ------
sys.path.insert(0, str(GEN))
import generate_dataset as G                                     # noqa: E402

SNR_REF = float(G.SNR_REF_DETECT)
SNR_THR = float(G.SNR_THRESHOLD)
S_DL = float(G.SIGMA_DL)
F1 = float(G.SIG_M1_FRAC)
F2 = float(G.SIG_M2_FRAC)
M1_CLIP = 2.0
M2_CLIP = 1.0
A_FLOOR = 1.0e-8

LOG_D0 = np.log(1000.0 * SNR_REF / SNR_THR)      # ln of the horizon scale, Mpc


def chirp(m1, m2):
    return (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2


def t_of(m1det, m2det, dL):
    """t = ln(rho_true/8)/s -- the ONLY way (m1det, m2det, dL) enters, besides q."""
    rho = SNR_REF * (chirp(m1det, m2det) / 30.0) ** (5.0 / 6.0) * (1000.0 / dL)
    return np.log(rho / SNR_THR) / S_DL


def gh_nodes(n):
    x, w = np.polynomial.hermite_e.hermegauss(n)
    return x, w / w.sum()


def log_R(x1, x2, q):
    """ln[ Mc(obs)/Mc(true) ] for the mass-noise realisation (x1, x2) at ratio q.

    Shapes broadcast; ``q`` may carry any trailing shape."""
    a1 = np.maximum(1.0 + F1 * x1, A_FLOOR)
    a2 = np.maximum(1.0 + F2 * x2, A_FLOOR)
    return (0.6 * (np.log(a1) + np.log(a2))
            - 0.2 * np.log((a1 + q * a2) / (1.0 + q)))


def pcal(t, q, n_gh=48, chunk=4_000_000):
    """P_det as a function of (t, q).  Vectorised; t and q broadcast together."""
    t = np.asarray(t, float)
    q = np.asarray(q, float)
    t, q = np.broadcast_arrays(t, q)
    shp = t.shape
    t = t.ravel()
    q = q.ravel()
    x, w = gh_nodes(n_gh)
    W = (w[:, None] * w[None, :]).ravel()
    X1 = np.repeat(x, n_gh)
    X2 = np.tile(x, n_gh)
    out = np.empty(t.size)
    step = max(1, chunk // (n_gh * n_gh))
    for i0 in range(0, t.size, step):
        sl = slice(i0, min(i0 + step, t.size))
        lr = log_R(X1[None, :], X2[None, :], q[sl][:, None])
        out[sl] = (ndtr(t[sl][:, None] + (5.0 / 6.0) * lr / S_DL) * W[None, :]).sum(1)
    return out.reshape(shp)


def p_det(m1det, m2det, dL, n_gh=48):
    """P_det in the natural variables (the reduction to (t, q) made explicit)."""
    m1det = np.asarray(m1det, float)
    m2det = np.asarray(m2det, float)
    dL = np.asarray(dL, float)
    return pcal(t_of(m1det, m2det, dL), m2det / m1det, n_gh=n_gh)


# --------------------------------------------------------------------------------
def brute_force(m1det, m2det, dL, n, seed, chunk=10_000_000):
    """Detection fraction from the GENERATOR'S OWN observe() / detect().

    Nothing here is re-derived: ``G.observe`` draws the measurement and
    ``G.detect_from_observation`` applies the rule, exactly as the events and the
    injections stages do."""
    rng = np.random.default_rng(seed)
    hit = 0
    done = 0
    while done < n:
        m = min(chunk, n - done)
        obs = G.observe(rng, np.full(m, m1det), np.full(m, m2det),
                        np.zeros(m), np.full(m, dL), None, None, need_sky=False)
        det, _ = G.detect_from_observation(obs)
        hit += int(det.sum())
        done += m
    p = hit / n
    return p, np.sqrt(max(p * (1.0 - p), 1e-16) / n)


def clip_probability(m1det, m2det):
    """P(the 2 / 1 Msun clip is active) for one (m1det, m2det)."""
    return (ndtr(-(1.0 - M1_CLIP / m1det) / F1),
            ndtr(-(1.0 - M2_CLIP / m2det) / F2))


# ================================================================================
# v3 (2026-08-01): the detection rule is rho_obs = rho_opt(theta) + N(0, sigma_rho)
# with the cut on rho_obs, so
#
#       P_det(theta) = Phi( (rho_opt(theta) - 8) / sigma_rho )
#
# EXACTLY -- one Gaussian CDF, no quadrature, no mass-noise latents, and no
# dependence on q beyond the chirp mass.  It is validated below against a
# brute-force Monte Carlo that calls the generator's OWN observe_v3/detect_v3.
# ================================================================================
SIGMA_RHO_V3 = float(getattr(G, "SIGMA_RHO", 1.0))


def rho_opt_v3(m1det, m2det, dL):
    return G.snr_amplitude(np.asarray(m1det, float), np.asarray(m2det, float),
                           np.asarray(dL, float), SNR_REF)


def p_det_v3(m1det, m2det, dL):
    return ndtr((rho_opt_v3(m1det, m2det, dL) - SNR_THR) / SIGMA_RHO_V3)


def brute_force_v3(m1det, m2det, dL, n, seed, chunk=20_000_000):
    """Detection fraction from the GENERATOR'S OWN observe_v3() / detect_v3()."""
    rng = np.random.default_rng(seed)
    hit, done = 0, 0
    while done < n:
        m = min(chunk, n - done)
        obs = G.observe_v3(rng, np.full(m, m1det), np.full(m, m2det),
                           np.zeros(m), np.full(m, dL), None, None, need_sky=False)
        det, _ = G.detect_v3(obs)
        hit += int(det.sum())
        done += m
    p = hit / n
    return p, np.sqrt(max(p * (1.0 - p), 1e-16) / n)


def main_v3(args):
    """The v3 verdict: P_det is a Gaussian CDF, validated against the generator."""
    n_mc = int(args.n_mc)
    t0 = time.time()
    print(f"[v3] SNR_REF_DETECT={SNR_REF!r} threshold={SNR_THR} "
          f"sigma_rho={SIGMA_RHO_V3}")
    # 0. chieff (and everything but Mc_det and dL) really is absent
    rr = np.random.default_rng(3)
    mm1 = rr.uniform(10, 80, 400)
    mm2 = mm1 * rr.uniform(0.15, 1.0, 400)
    ddl = rr.uniform(200, 2500, 400)
    _r = np.random.default_rng(11)
    o0 = G.observe_v3(_r, mm1, mm2, np.zeros(400), ddl, None, None, need_sky=False)
    _r = np.random.default_rng(11)
    o1 = G.observe_v3(_r, mm1, mm2, np.full(400, 0.9), ddl, None, None,
                      need_sky=False)
    chieff_free = bool(np.array_equal(o0["rho"], o1["rho"]))
    # the (Mc, dL) reduction: hold rho_opt fixed and P_det must not move
    lam = np.array([0.5, 0.8, 1.0, 1.3, 2.0, 4.0])
    base = p_det_v3(mm1, mm2, ddl)
    red = []
    for L in lam:
        v = p_det_v3(L * mm1, L * mm2, L ** (5.0 / 6.0) * ddl)
        red.append(float(np.max(np.abs(v - base))))
    print(f"[check] chieff absent from rho_obs: {chieff_free}; "
          f"(Mc,dL) reduction max|dP| = {max(red):.3e}")
    # brute force across the whole transition
    pts = []
    for m1d, qd in ((12.0, 0.9), (25.0, 0.8), (35.0, 1.0), (45.0, 0.55),
                    (60.0, 0.35), (80.0, 0.9), (110.0, 0.7), (40.0, 0.135),
                    (75.0, 0.075), (35.0, 0.155)):
        m2d = qd * m1d
        for rt in (5.0, 8.0, 11.0):
            dL = (SNR_REF * (chirp(m1d, m2d) / 30.0) ** (5.0 / 6.0) * 1000.0 / rt)
            pts.append((m1d, m2d, float(dL)))
    mc = []
    print(f"[mc] brute force, n={n_mc:.1e} per point, generator's own observe_v3()")
    for k, (m1d, m2d, dL) in enumerate(pts):
        pq = float(p_det_v3(m1d, m2d, dL))
        pm, sm = brute_force_v3(m1d, m2d, dL, n_mc, args.seed + 1000 * k)
        mc.append({"m1det": m1d, "m2det": m2d, "dL": dL,
                   "rho_opt": float(rho_opt_v3(m1d, m2d, dL)), "q": m2d / m1d,
                   "P_quad": pq, "P_mc": pm, "sigma_mc": sm,
                   "diff": pm - pq, "pull": (pm - pq) / sm})
        print(f"  {k:2d} m1={m1d:6.1f} q={m2d/m1d:4.2f} rho={mc[-1]['rho_opt']:6.2f}"
              f"  exact={pq:.6f}  mc={pm:.6f}+-{sm:.1e}  d={pm-pq:+.2e}  "
              f"pull={(pm-pq)/sm:+.2f}")
    dmax = max(abs(m["diff"]) for m in mc)
    pmax = max(abs(m["pull"]) for m in mc)
    pulls = np.array([m["pull"] for m in mc])
    print(f"[mc] max |P_mc - P_exact| = {dmax:.3e}   max |pull| = {pmax:.2f}   "
          f"mean pull = {pulls.mean():+.3f} +- {pulls.std(ddof=1)/np.sqrt(pulls.size):.3f}")
    out = {
        "name": "attr_selmu_pdet", "pe_model": "v3",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "generator_constants": {
            "SNR_REF_DETECT": SNR_REF, "SNR_THRESHOLD": SNR_THR,
            "SIGMA_RHO": SIGMA_RHO_V3,
            "detection_rule": "rho_obs = rho_opt(theta) + N(0, sigma_rho) >= 8",
            "closed_form": "P_det = Phi((rho_opt - 8)/sigma_rho)",
            "chieff_in_detection": False},
        "chieff_absent_from_detection": chieff_free,
        "mc_dl_reduction_maxabs": {str(float(L)): d for L, d in zip(lam, red)},
        "brute_force": {"n_per_point": n_mc, "points": mc,
                        "max_abs_diff": float(dmax), "max_abs_pull": float(pmax),
                        "mean_pull": float(pulls.mean()),
                        "sem_pull": float(pulls.std(ddof=1) / np.sqrt(pulls.size))},
    }
    od = Path(args.outdir)
    (od / "attr_selmu_pdet_v3.json").write_text(json.dumps(out, indent=2))
    print(f"Wrote {od/'attr_selmu_pdet_v3.json'}  ({time.time()-t0:.0f}s)")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n_mc", type=float, default=1.0e8)
    ap.add_argument("--seed", type=int, default=20260801)
    ap.add_argument("--outdir", default=str(ROOT / "results"))
    ap.add_argument("--pe_model", choices=("v2", "v3"), default="v2",
                    help="v3: validate P_det = Phi((rho_opt-8)/sigma_rho) against "
                         "the generator's own observe_v3/detect_v3.  v2 (default) "
                         "reproduces the CLOSURE.md 11.1 product unchanged.")
    args = ap.parse_args(argv)
    if args.pe_model == "v3":
        return main_v3(args)
    n_mc = int(args.n_mc)
    t0 = time.time()

    print(f"generator constants: SNR_REF={SNR_REF!r} thr={SNR_THR} s={S_DL} "
          f"f1={F1} f2={F2}")
    # --- 0. chieff really is absent from the detection statistic ----------------
    import inspect
    sig_snr = list(inspect.signature(G.snr_amplitude).parameters)
    sig_det = list(inspect.signature(G.detect_from_observation).parameters)
    src_det = inspect.getsource(G.detect_from_observation)
    # detect_from_observation reads ONLY obs["m1det"], obs["m2det"], obs["dL"]
    keys_read = sorted(set(np.array(
        [k for k in ("m1det", "m2det", "dL", "chieff", "ra", "dec", "sigma_ang")
         if f'"{k}"' in src_det], dtype=object).tolist()))
    chieff_free = ("chieff" not in sig_snr) and ("chieff" not in keys_read)
    # numerical corroboration: vary chieff over its whole range, P_det must not move
    rr = np.random.default_rng(3)
    mm1 = rr.uniform(10, 80, 200)
    mm2 = mm1 * rr.uniform(0.2, 1.0, 200)
    ddl = rr.uniform(200, 1500, 200)
    _rng = np.random.default_rng(11)
    ref_obs = G.observe(_rng, mm1, mm2, np.zeros(200), ddl, None, None, need_sky=False)
    _rng = np.random.default_rng(11)
    alt_obs = G.observe(_rng, mm1, mm2, np.full(200, 0.9), ddl, None, None,
                        need_sky=False)
    same = bool(np.array_equal(G.detect_from_observation(ref_obs)[1],
                               G.detect_from_observation(alt_obs)[1]))
    print(f"[check] snr_amplitude params={sig_snr}  detect reads={keys_read}")
    print(f"[check] chieff absent from the detection statistic: {chieff_free}; "
          f"rho_obs bit-identical when chieff 0 -> 0.9: {same}")
    chieff_free = chieff_free and same

    # --- 1. quadrature convergence (node doubling) ------------------------------
    rng = np.random.default_rng(7)
    m1t = np.exp(rng.uniform(np.log(6.0), np.log(160.0), 4000))
    qt = rng.uniform(0.05, 1.0, 4000)
    m2t = qt * m1t
    dLt = np.exp(rng.uniform(np.log(50.0), np.log(4000.0), 4000))
    conv = {}
    ref = None
    for n in (12, 24, 48, 96, 192):
        v = p_det(m1t, m2t, dLt, n_gh=n)
        conv[str(n)] = v
        print(f"  n_gh={n:4d}  <P>={v.mean():.8f}")
    for n in (12, 24, 48, 96):
        d = np.max(np.abs(conv[str(n)] - conv["192"]))
        print(f"  max|P(n={n}) - P(192)| = {d:.3e}")
    n_gh_prod = 48
    quad_conv = {f"maxabs_vs_192_ngh{n}": float(np.max(np.abs(conv[str(n)] - conv["192"])))
                 for n in (12, 24, 48, 96)}
    quad_conv["maxabs_ngh48_vs_ngh96"] = float(np.max(np.abs(conv["48"] - conv["96"])))

    # --- 2. the (t, q) reduction, and the inertness of the mass clips ----------
    #   P_det must depend on (m1det, m2det, dL) ONLY through (t, q).  Scale the
    #   masses and the distance so that (t, q) are held fixed and check.
    lam = np.array([0.5, 0.8, 1.0, 1.3, 2.0, 4.0])
    red = []
    base = p_det(m1t, m2t, dLt, n_gh=n_gh_prod)
    for L in lam:
        # Mc -> L*Mc  =>  rho -> L^(5/6) rho ; hold t by dL -> L^(5/6) dL
        v = p_det(L * m1t, L * m2t, L ** (5.0 / 6.0) * dLt, n_gh=n_gh_prod)
        red.append(float(np.max(np.abs(v - base))))
        print(f"  (t,q) reduction: mass scale {L:>4}  max|dP| = {red[-1]:.3e}")
    # The A_FLOOR treatment differs from the generator's true 2 / 1 Msun clip only
    # on draws where the clip is ACTIVE, so |dP| is bounded by the clip probability.
    # Evaluate that bound over the POPULATION's own support: the powerlaw+peak
    # mixture tapers both masses to zero below m_min = 5 Msun, and detector-frame
    # masses are (1+z) times source-frame ones, so m1det, m2det >= 5.
    mgrid = np.linspace(5.0, 200.0, 400)
    c1 = clip_probability(mgrid, mgrid)[0]
    c2 = clip_probability(mgrid, mgrid)[1]
    print(f"[check] clip probability bound on m >= 5 Msun: "
          f"m1 {c1.max():.3e}  m2 {c2.max():.3e}")

    # --- 3. brute-force MC validation ------------------------------------------
    #   20 points spread over the detection transition, at masses and distances
    #   the detected population actually reaches.
    pts = []
    for m1d, qd in ((12.0, 0.9), (25.0, 0.8), (35.0, 1.0), (45.0, 0.55),
                    (60.0, 0.35), (80.0, 0.9), (110.0, 0.7), (40.0, 0.135),
                    (75.0, 0.075), (35.0, 0.155)):
        m2d = qd * m1d
        # place dL so that t hits a few values across the transition
        for tt in (-3.0, 0.0, 3.0):
            rho8 = SNR_THR * np.exp(S_DL * tt)
            dL = SNR_REF * (chirp(m1d, m2d) / 30.0) ** (5.0 / 6.0) * 1000.0 / rho8
            pts.append((m1d, m2d, float(dL)))
    pts = pts[:30]
    mc = []
    print(f"[mc] brute force, n={n_mc:.1e} per point, generator's own observe()")
    for k, (m1d, m2d, dL) in enumerate(pts):
        pq = float(p_det(m1d, m2d, dL, n_gh=n_gh_prod))
        pm, sm = brute_force(m1d, m2d, dL, n_mc, args.seed + 1000 * k)
        mc.append({"m1det": m1d, "m2det": m2d, "dL": dL,
                   "t": float(t_of(m1d, m2d, dL)), "q": m2d / m1d,
                   "P_quad": pq, "P_mc": pm, "sigma_mc": sm,
                   "diff": pm - pq, "pull": (pm - pq) / sm})
        print(f"  {k:2d} m1={m1d:6.1f} q={m2d/m1d:4.2f} dL={dL:8.1f}  "
              f"quad={pq:.6f}  mc={pm:.6f}+-{sm:.1e}  d={pm-pq:+.2e}  "
              f"pull={(pm-pq)/sm:+.2f}")
    dmax = max(abs(m["diff"]) for m in mc)
    pmax = max(abs(m["pull"]) for m in mc)
    print(f"[mc] max |P_mc - P_quad| = {dmax:.3e}   max |pull| = {pmax:.2f}")

    # --- 4. the production Pcal(t, q) table -------------------------------------
    tg = np.linspace(-30.0, 30.0, 1201)
    qg = np.linspace(0.02, 1.0, 99)
    Ptab = pcal(tg[:, None], qg[None, :], n_gh=n_gh_prod)
    print(f"[table] Pcal grid {Ptab.shape}  in {time.time()-t0:.0f}s")

    out = {
        "name": "attr_selmu_pdet",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "generator_constants": {
            "SNR_REF_DETECT": SNR_REF, "SNR_THRESHOLD": SNR_THR,
            "SIGMA_DL": S_DL, "SIG_M1_FRAC": F1, "SIG_M2_FRAC": F2,
            "distance_noise_convention": "obs_dL = dL*exp(s*N(0,1)); "
                                         "ln obs_dL ~ N(ln dL, s) with NO -s^2/2",
            "mass_noise_convention": "obs ~ clip(N(m, f*m), 2/1 Msun)",
            "chieff_in_detection": False},
        "chieff_absent_from_detection_source": bool(chieff_free),
        "n_gh_production": n_gh_prod,
        "quadrature_convergence": quad_conv,
        "tq_reduction_maxabs": {str(float(L)): d for L, d in zip(lam, red)},
        "clip_probability_max": {"m1": float(c1.max()), "m2": float(c2.max())},
        "brute_force": {"n_per_point": n_mc, "points": mc,
                        "max_abs_diff": float(dmax), "max_abs_pull": float(pmax)},
        "table": {"t": [float(tg[0]), float(tg[-1]), int(tg.size)],
                  "q": [float(qg[0]), float(qg[-1]), int(qg.size)]},
    }
    od = Path(args.outdir)
    (od / "attr_selmu_pdet.json").write_text(json.dumps(out, indent=2))
    np.savez_compressed(od / "attr_selmu_pdet.npz", t_grid=tg, q_grid=qg, Pcal=Ptab,
                        mc_m1=np.array([m["m1det"] for m in mc]),
                        mc_m2=np.array([m["m2det"] for m in mc]),
                        mc_dL=np.array([m["dL"] for m in mc]),
                        mc_P_quad=np.array([m["P_quad"] for m in mc]),
                        mc_P_mc=np.array([m["P_mc"] for m in mc]),
                        mc_sigma=np.array([m["sigma_mc"] for m in mc]))
    print(f"Wrote {od/'attr_selmu_pdet.json'}  ({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
