#!/usr/bin/env python3
"""ATTRIBUTION -- controlled toy for the mass-measurement convention (CPU).

Isolates ONE channel of the mock in closed form: the primary mass.

    truth      m ~ p_pop(m)                      (darksirens powerlaw+peak, fiducial)
    measure    obs ~ N(m, f m),   f = 0.08       (generate_dataset.py::observe)
    detect     1[obs >= m_cut]                   (a mass-dependent, DATA-based cut)
    PE         m_s ~ N(obs, f * m_TRUE)          (generate_dataset.py::posterior_samples)

and computes the score residual in the mass channel,

    r_kappa = <E_post[kappa]>_detected events  -  E_{p_pop P_det}[kappa],
    kappa(m) = d ln p_pop / d ln m,

which is the whole H0-relevant content of the population term (the likelihood's
mass score is -(dz/dH0)/(1+z) * kappa).  A correctly specified measurement model
sets r_kappa = 0 EXACTLY.  Three PE conventions are compared on the SAME truths
and the SAME detected set:

    stored   posterior ∝ N(m; obs, f m_true) p_pop(m)   -- the mock's convention
    obswidth posterior ∝ N(m; obs, f obs)    p_pop(m)   -- width from the DATA
    exact    posterior ∝ N(obs; m, f m)      p_pop(m)   -- the true likelihood

Everything is quadrature on a fixed m grid, so there is no Monte-Carlo noise in
the posteriors; the only noise is the finite number of drawn events.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--f", type=float, default=0.08)
    ap.add_argument("--n_events", type=int, default=400000)
    ap.add_argument("--m_cut", type=float, default=30.0,
                    help="Detection threshold on the OBSERVED mass.")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n_grid", type=int, default=4001)
    ap.add_argument("--outdir", default=str(ROOT / "results"))
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from darksirens.gw.populations import get_fixed_population_params, pop_model_parser

    pop = np.asarray(get_fixed_population_params("powerlaw+peak", shared_beta=True,
                                                 shared_spin=True, shared_gamma=True))
    log_p_pop = pop_model_parser(pop_model="powerlaw+peak", shared_beta=True,
                                 shared_spin=True, shared_gamma=True)
    # p_pop(m1src) at the fiducial q / chieff / z; only the m1src dependence is
    # used, so the (z, chieff, q) slice is an irrelevant constant.
    mgrid = np.linspace(2.0, 100.0, args.n_grid)
    lp = np.asarray(log_p_pop(jnp.asarray(mgrid), jnp.full(mgrid.shape, 0.8),
                              jnp.zeros_like(mgrid), jnp.zeros_like(mgrid),
                              jnp.asarray(pop)))
    p = np.where(np.isfinite(lp), np.exp(lp - np.nanmax(lp[np.isfinite(lp)])), 0.0)
    p /= np.trapz(p, mgrid)
    # kappa = d ln p / d ln m, on the same grid (central differences in ln m).
    lnm = np.log(mgrid)
    with np.errstate(divide="ignore", invalid="ignore"):
        lnp = np.where(p > 0, np.log(np.maximum(p, 1e-300)), np.nan)
    kappa = np.gradient(lnp, lnm)
    ok = np.isfinite(kappa) & (p > 0)
    kappa = np.where(ok, kappa, 0.0)

    rng = np.random.default_rng(args.seed)
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (p[1:] + p[:-1]) * np.diff(mgrid))])
    cdf /= cdf[-1]
    m_true = np.interp(rng.uniform(size=args.n_events), cdf, mgrid)
    sig_true = args.f * m_true
    obs = rng.normal(m_true, sig_true)
    det = obs >= args.m_cut
    mt, ob, st = m_true[det], obs[det], sig_true[det]
    nd = mt.size
    print(f"toy: {nd}/{args.n_events} detected ({nd/args.n_events:.3f}) "
          f"at m_cut={args.m_cut}")

    # ---- B: model's detected-truth mean of kappa --------------------------
    # P_det(m) = P(obs >= m_cut | m) = Phi((m - m_cut)/(f m))
    from scipy.stats import norm as _norm
    Pdet = _norm.sf((args.m_cut - mgrid) / (args.f * np.maximum(mgrid, 1e-6)))
    wB = p * Pdet
    B = float(np.trapz(wB * kappa, mgrid) / np.trapz(wB, mgrid))
    A = float(np.mean(np.interp(mt, mgrid, kappa)))
    A_sem = float(np.std(np.interp(mt, mgrid, kappa), ddof=1) / np.sqrt(nd))
    print(f"B (model detected-truth mean kappa) = {B:.6f}")
    print(f"A (empirical detected-truth mean)   = {A:.6f} +- {A_sem:.6f}   "
          f"A-B = {A-B:+.6f}")

    # ---- C: posterior means under the three PE conventions ----------------
    # Chunked to bound the (n_det x n_grid) working set.
    res = {}
    chunk = 20000
    for name in ("stored", "obswidth", "exact"):
        num = np.zeros(nd)
        for a0 in range(0, nd, chunk):
            a1 = min(a0 + chunk, nd)
            O = ob[a0:a1][:, None]
            S = st[a0:a1][:, None]
            M = mgrid[None, :]
            if name == "stored":
                ll = -0.5 * ((M - O) / S) ** 2
            elif name == "obswidth":
                s = args.f * O
                ll = -0.5 * ((M - O) / s) ** 2
            else:
                s = args.f * M
                ll = -np.log(s) - 0.5 * ((O - M) / s) ** 2
            w = np.exp(ll - ll.max(axis=1, keepdims=True)) * p[None, :]
            num[a0:a1] = ((w * kappa[None, :]).sum(axis=1) / w.sum(axis=1))
        C = float(num.mean()); sem = float(num.std(ddof=1) / np.sqrt(nd))
        res[name] = {"C": C, "sem": sem, "r_kappa": C - B,
                     "CmA": float(np.mean(num - np.interp(mt, mgrid, kappa))),
                     "CmA_sem": float(np.std(num - np.interp(mt, mgrid, kappa),
                                             ddof=1) / np.sqrt(nd))}
        print(f"  {name:>9}: C = {C:+.6f}   r_kappa = C-B = {C-B:+.6f} "
              f"(+- {sem:.6f})   C-A = {res[name]['CmA']:+.6f} "
              f"+- {res[name]['CmA_sem']:.6f}")

    # In the likelihood the mass score is -(dz/dH0)/(1+z) * kappa; at the mock's
    # median event redshift that prefactor is J = z / (H0 (1+z)).
    z_med, H0 = 0.1321, 67.74
    J = z_med / (H0 * (1.0 + z_med))
    print(f"\nJ = z/(H0(1+z)) = {J:.5e} at z_med={z_med}")
    for name, v in res.items():
        print(f"  {name:>9}: r_mass = -J * r_kappa = {-J * v['r_kappa']:+.4e}")

    out = {"name": "attr_toy_masswidth", "f": args.f, "m_cut": args.m_cut,
           "n_events": args.n_events, "n_detected": int(nd), "seed": args.seed,
           "B": B, "A": A, "A_sem": A_sem, "A_minus_B": A - B,
           "arms": res, "J_at_zmed": J,
           "r_mass_equivalent": {k: -J * v["r_kappa"] for k, v in res.items()},
           "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "attr_toy_masswidth.json").write_text(json.dumps(out, indent=2))
    print(f"\nWrote {outdir / 'attr_toy_masswidth.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
