#!/usr/bin/env python3
"""TASK 2 -- the chi_eff CLIPPING substitution, and whether chi_eff carries H0.

``generate_dataset.observe()`` records

    obs_chieff = clip( N(chi, SIGMA_CHIEFF), -1, +1 )        SIGMA_CHIEFF = 0.08

so the realised measurement model is CENSORED, not Gaussian:

    p(obs | chi) = phi((obs-chi)/s)/s          for  -1 < obs < 1
    P(obs = +1 | chi) = 1 - Phi((1-chi)/s)     an ATOM
    P(obs = -1 | chi) =     Phi((-1-chi)/s)    an ATOM

and ``posterior_samples()`` draws ``clip(N(obs_chieff, s), -1, 1)``, i.e. a
Gaussian about the observation, itself clipped.  The EXACT flat-prior posterior
on the population's support chi in [-1, 1] is instead

    interior obs :  p(chi|obs) = TRUNCATED N(obs, s) on [-1, 1]   (not clipped)
    censored obs :  p(chi|+1)  propto 1 - Phi((1-chi)/s)          (a smooth ramp)

This script (1) measures how far the realised data are from the clip, (2) builds
the exact censored posterior and substitutes it for the stored one in
darksirens' OWN per-event score -- paired, per event, on both matched sets --
and (3) tests whether chi_eff carries any H0 dependence at all.

The ANALYTIC expectation, stated first so the measurement can confirm it:
``parametric.py::log_p_pop`` is

    log p_pop = log[ p_mass(m1src) p_pair(q|m1src) p_spin(chieff) ]
                + (gamma - 1) log(1+z)

-- a PRODUCT with the spin factor depending on chieff ALONE -- and the PE prior
``p_pe`` is proportional to m1det with no chieff, and ``snr_amplitude`` never
reads chieff.  So in BOTH the per-event evidence and mu the chieff channel
factorises into a multiplicative constant that carries no H0:

    Z_i(H0) = [INT dchi p_spin(chi) L_chi(obs_i|chi)] x [everything else](H0)

hence d ln Z_i/dH0 is EXACTLY independent of the chi_eff measurement model, and
so is d ln mu/dH0.  Any measured shift is finite-nsamp Monte-Carlo coupling, not
bias.

Outputs: results/attr_chieff.json (+ .npz)
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
from scipy.special import ndtr, ndtri

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

GEN = Path("/hildafs/projects/phy230014p/magana/gws-agn/working/data")
sys.path.insert(0, str(GEN))
import generate_dataset as G                                        # noqa: E402

SIG_CHI = float(G.SIGMA_CHIEFF)
CHI_LO, CHI_HI = -1.0, 1.0
H0_FID = 67.74


def exact_posterior_cdf(chi, obs, s=SIG_CHI):
    """CDF of the EXACT flat-prior posterior of the censored measurement.

    ``obs`` strictly inside (-1, 1): truncated N(obs, s) on [-1, 1].
    ``obs`` at a clip edge: the censored likelihood, normalised on [-1, 1].
    """
    chi = np.asarray(chi, float)
    if obs <= CHI_LO or obs >= CHI_HI:
        # p(chi) propto 1 - Phi((1-chi)/s)  (upper clip) -- integrate numerically
        gg = np.linspace(CHI_LO, CHI_HI, 200001)
        if obs >= CHI_HI:
            d = 1.0 - ndtr((CHI_HI - gg) / s)
        else:
            d = ndtr((CHI_LO - gg) / s)
        c = np.concatenate([[0.0], np.cumsum(0.5 * (d[1:] + d[:-1]) * np.diff(gg))])
        c /= c[-1]
        return np.interp(chi, gg, c)
    a = ndtr((CHI_LO - obs) / s)
    b = ndtr((CHI_HI - obs) / s)
    return (ndtr((chi - obs) / s) - a) / (b - a)


def sample_exact_posterior(rng, obs, n, s=SIG_CHI):
    if obs <= CHI_LO or obs >= CHI_HI:
        gg = np.linspace(CHI_LO, CHI_HI, 200001)
        d = (1.0 - ndtr((CHI_HI - gg) / s)) if obs >= CHI_HI else ndtr((CHI_LO - gg) / s)
        c = np.concatenate([[0.0], np.cumsum(0.5 * (d[1:] + d[:-1]) * np.diff(gg))])
        c /= c[-1]
        return np.interp(rng.random(n), c, gg)
    a = ndtr((CHI_LO - obs) / s)
    b = ndtr((CHI_HI - obs) / s)
    return obs + s * ndtri(a + (b - a) * rng.random(n))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--tracers", nargs="+", default=["gal", "agn"])
    ap.add_argument("--dh", type=float, default=0.5)
    ap.add_argument("--rng", type=int, default=20260801)
    ap.add_argument("--outdir", default=str(ROOT / "results"))
    args = ap.parse_args(argv)
    od = Path(args.outdir)
    t0 = time.time()
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")
    import jax
    import jax.numpy as jnp
    import attr_ds_bridge as bridge
    from darksirens.likelihood.selection import log_evidence_and_mc_variance

    out = {"name": "attr_chieff_clip",
           "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "sigma_chieff": SIG_CHI, "clip": [CHI_LO, CHI_HI], "tracers": {}}

    # ---- 1. how far the realised data are from the clip ------------------------
    ev = GEN / f"seed{args.seed}" / "events" / "events.h5"
    with h5py.File(ev, "r") as f:
        obs_all = np.asarray(f["truth/obs_chieff"][:], float)
        chi_true = np.asarray(f["truth/chieff"][:], float)
        pe_all = np.asarray(f["chieff"][:], float)
    out["realised"] = {
        "n_events": int(obs_all.size),
        "obs_chieff_min": float(obs_all.min()), "obs_chieff_max": float(obs_all.max()),
        "n_obs_at_clip": int((np.abs(obs_all) >= 1.0).sum()),
        "min_distance_to_clip_in_sigma":
            float((1.0 - np.abs(obs_all).max()) / SIG_CHI),
        "true_chieff_absmax": float(np.abs(chi_true).max()),
        "n_pe_samples": int(pe_all.size),
        "n_pe_samples_at_clip": int((np.abs(pe_all) >= 1.0).sum()),
        "pe_absmax": float(np.abs(pe_all).max()),
    }
    print(f"[data] |obs_chieff|max = {np.abs(obs_all).max():.4f} "
          f"({out['realised']['min_distance_to_clip_in_sigma']:.2f} sigma from the "
          f"clip); {out['realised']['n_obs_at_clip']} censored observations, "
          f"{out['realised']['n_pe_samples_at_clip']} clipped PE samples", flush=True)

    # exact vs stored posterior, per event, as a max CDF distance
    kd = []
    for o in obs_all:
        gg = np.linspace(max(CHI_LO, o - 8 * SIG_CHI), min(CHI_HI, o + 8 * SIG_CHI),
                         4001)
        c_ex = exact_posterior_cdf(gg, float(o))
        c_st = np.clip((ndtr((gg - o) / SIG_CHI)
                        - ndtr((CHI_LO - o) / SIG_CHI)), 0.0, 1.0)
        # the STORED model is clip(N(obs,s)) -- CDF is that of N(obs,s) with the
        # tails piled at the edges; inside the open interval it is just N(obs,s).
        c_st = ndtr((gg - o) / SIG_CHI)
        kd.append(float(np.max(np.abs(c_ex - c_st))))
    out["posterior_mismatch"] = {
        "max_cdf_distance_exact_vs_stored": float(np.max(kd)),
        "mean_cdf_distance": float(np.mean(kd)),
        "note": "with no observation within 6.9 sigma of the clip the exact "
                "truncated posterior and the stored clipped Gaussian coincide"}
    print(f"[posterior] max |CDF_exact - CDF_stored| over events = "
          f"{np.max(kd):.3e}", flush=True)
    # what the substitution WOULD be worth if an observation sat at the clip
    hypo = {}
    for o in (0.90, 0.95, 0.99, 1.0):
        gg = np.linspace(CHI_LO, CHI_HI, 200001)
        c_ex = exact_posterior_cdf(gg, o)
        c_st = np.clip(ndtr((gg - o) / SIG_CHI), 0.0, 1.0)
        hypo[str(o)] = float(np.max(np.abs(c_ex - c_st)))
    out["posterior_mismatch"]["hypothetical_obs_near_clip"] = hypo
    print(f"[posterior] hypothetical obs at 0.90/0.95/0.99/1.0: "
          + " ".join(f"{v:.3f}" for v in hypo.values()), flush=True)

    # ---- 2/3. the factorisation, and the paired substitution -------------------
    for tr in args.tracers:
        kw = dict(kde_window=4096) if tr == "gal" else {}
        B = bridge.build(tracer=tr, seed=args.seed, h0=H0_FID, **kw)
        nobs, nsamp = B.nobs, B.nsamp
        chi0 = np.asarray(B.gw_pe.chieff, float)

        # --- factorisation of log p_pop in chieff (exact product structure) -----
        rng = np.random.default_rng(args.rng)
        nT = 4000
        m1d = np.exp(rng.uniform(np.log(6.0), np.log(150.0), nT))
        qq = rng.uniform(0.05, 1.0, nT)
        dl = np.exp(rng.uniform(np.log(50.0), np.log(3000.0), nT))
        pix = np.asarray(B.gw_pe.pixels, int)[rng.integers(0, chi0.size, nT)]
        pw = np.ones(nT)
        vv = np.ones(nT, bool)
        ca, cb = np.full(nT, -0.4), np.full(nT, 0.35)
        spread = {}
        for h in (H0_FID - args.dh, H0_FID, H0_FID + args.dh):
            f = B.make_pieces(h)
            A = np.asarray(f(jnp.asarray(m1d), jnp.asarray(qq), jnp.asarray(dl),
                             jnp.asarray(ca), jnp.asarray(pix), jnp.asarray(pw),
                             jnp.asarray(vv))[1])
            Bp = np.asarray(f(jnp.asarray(m1d), jnp.asarray(qq), jnp.asarray(dl),
                              jnp.asarray(cb), jnp.asarray(pix), jnp.asarray(pw),
                              jnp.asarray(vv))[1])
            d = A - Bp
            fin = np.isfinite(d)          # -inf where the mass grid is off support
            spread[str(h)] = {"mean": float(np.mean(d[fin])),
                              "ptp": float(np.ptp(d[fin])),
                              "n_finite": int(fin.sum())}
            del f
        vals = [spread[k]["mean"] for k in spread]
        print(f"[{tr}] log p_pop(chi=-0.4) - log p_pop(chi=0.35): spread over "
              f"(m1,q,dL,pix) = {max(s['ptp'] for s in spread.values()):.3e}; "
              f"spread over H0 = {max(vals)-min(vals):.3e}", flush=True)

        # --- paired per-event score under three chieff realisations -------------
        def score_with(chi):
            fns = [B.make_pieces(h) for h in (H0_FID - args.dh, H0_FID + args.dh)]
            red = jax.jit(jax.vmap(
                lambda row, n: log_evidence_and_mc_variance(row, n)[0],
                in_axes=(0, None)))
            s = np.zeros(nobs)
            nblk = 25 if tr == "gal" else 70
            for c in range(int(np.ceil(nobs / nblk))):
                i0, i1 = c * nblk, min((c + 1) * nblk, nobs)
                s0, s1 = i0 * nsamp, i1 * nsamp
                sl = lambda a: jnp.asarray(a)[s0:s1]
                m = i1 - i0
                lz = []
                for f in fns:
                    ldw = np.asarray(f(sl(B.gw_pe.m1det), sl(B.gw_pe.q),
                                       sl(B.gw_pe.dL), jnp.asarray(chi[s0:s1]),
                                       sl(B.gw_pe.pixels), sl(B.gw_pe.prior_wt),
                                       sl(B.gw_pe.valid))[0]).reshape(m, nsamp)
                    lz.append(np.asarray(red(jnp.asarray(ldw), nsamp)))
                s[i0:i1] = (lz[1] - lz[0]) / (2.0 * args.dh)
            del fns
            return s

        s_stored = score_with(chi0)
        # (a) within-event permutation: preserves the marginal EXACTLY, so under
        #     the product structure the score is unchanged in expectation.
        rp = np.random.default_rng(args.rng + 1)
        chi_perm = chi0.copy()
        for i in range(nobs):
            sl = slice(i * nsamp, (i + 1) * nsamp)
            chi_perm[sl] = rp.permutation(chi0[sl])
        s_perm = score_with(chi_perm)
        # (b) redraw from the EXACT censored posterior
        rq = np.random.default_rng(args.rng + 2)
        with h5py.File(B.paths["gw"], "r") as fh:
            obs_tr = np.asarray(fh["truth/obs_chieff"][:], float)
        chi_ex = np.concatenate([sample_exact_posterior(rq, float(o), nsamp)
                                 for o in obs_tr])
        s_exact = score_with(chi_ex)

        S = bridge.sel_pass(B, dh=args.dh, sel_batch=50000 if tr == "gal" else 100000)
        anchor = abs(float(S["log_mu"]) - B.spy0["log_mu"])
        dlnmu = float(S["dlnmu_fd"])

        def stat(x, y=None):
            v = x if y is None else x - y
            v = v[np.isfinite(v)]
            return {"mean": float(v.mean()),
                    "sem": float(v.std(ddof=1) / np.sqrt(v.size)),
                    "rms": float(np.sqrt((v ** 2).mean())), "n": int(v.size)}

        rec = {"n_events": int(nobs), "nsamp": int(nsamp),
               "anchor_log_mu_absdiff": anchor, "dlnmu_dH0": dlnmu,
               "log_p_pop_chieff_factorisation": {
                   "max_spread_over_m1_q_dL_pix": float(
                       max(s["ptp"] for s in spread.values())),
                   "max_spread_over_H0": float(max(vals) - min(vals))},
               "r_stored": stat(s_stored - dlnmu),
               "substitution_permuted_minus_stored": stat(s_perm, s_stored),
               "substitution_exact_minus_stored": stat(s_exact, s_stored),
               "delta_r_exact_chieff_posterior": stat(s_exact, s_stored)}
        print(f"[{tr}] r(stored) = {rec['r_stored']['mean']:+.5e}; "
              f"Delta score (exact chieff posterior) = "
              f"{rec['substitution_exact_minus_stored']['mean']:+.3e} "
              f"+- {rec['substitution_exact_minus_stored']['sem']:.2e} "
              f"(rms {rec['substitution_exact_minus_stored']['rms']:.2e}); "
              f"permutation control "
              f"{rec['substitution_permuted_minus_stored']['mean']:+.3e} "
              f"+- {rec['substitution_permuted_minus_stored']['sem']:.2e}", flush=True)
        out["tracers"][tr] = rec
        np.savez_compressed(od / f"attr_chieff_{tr}.npz", s_stored=s_stored,
                            s_perm=s_perm, s_exact=s_exact, dlnmu=dlnmu)
        del B

    (od / "attr_chieff.json").write_text(json.dumps(out, indent=2))
    print(f"Wrote {od/'attr_chieff.json'}  ({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
