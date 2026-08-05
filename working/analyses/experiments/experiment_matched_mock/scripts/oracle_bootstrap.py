#!/usr/bin/env python3
"""Parametric bootstrap of the exact-likelihood estimator.

Draw FRESH detected events + observations with the mock generator's own recipe
(gmd samplers, observed-data detection, identical clips and sigma_ang model) on
an EXISTING catalog, then evaluate the exact oracle posterior on them.  The
mean offset over fresh realisations measures the INTRINSIC bias of the
posterior-median estimator under the exact likelihood at this information level
— separating "estimator statistics" from "mock deviates from its model".

Writes results/oracle_boot_<tag>_<rep>.npz (numerator curves) and prints the
median/peak offset using the catalog's existing oracle_mu_<tag>.npz.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "scripts"))
import oracle_exact as oe  # noqa: E402
import generate_mock_data as gmd  # noqa: E402

SNR_REF = oe.SNR_REF
POP = oe.POP


def draw_fresh_events(catalog, grids, rng, nobs=1000, s_dl=0.10, a1=0.08,
                      a2=0.10, snr_ref_control=11.5, sigma_from_obs=False):
    """build_obsdet_mock's observed-data arm, re-implemented with gmd samplers."""
    cz, cra, cdec = catalog["z"], catalog["ra"], catalog["dec"]
    out = {k: [] for k in ("obs_dL", "obs_m1det", "obs_m2det", "obs_ra",
                           "obs_dec", "obs_sigma_ang", "obs_sig_m1",
                           "obs_sig_m2", "z", "ra", "dec")}
    n_have = 0
    n_tried = 0
    while n_have < nobs:
        n = 8192
        j = rng.integers(0, cz.size, n)
        z = cz[j]; ra = cra[j]; dec = cdec[j]
        dl = np.interp(z, grids["z"], grids["dl"])
        m1, up = gmd._sample_powerlaw_peak_m1(rng, n, POP, return_component=True)
        q = gmd._sample_q(rng, m1, POP, use_peak=up)
        m2 = q * m1
        t = 1.0 + z
        m1d, m2d = m1 * t, m2 * t
        # sigma_ang from rho_opt with the CONTROL snr_ref (as in the mock)
        # ONE observation (masses/distance first; sky width may derive from them)
        D = dl * np.exp(s_dl * rng.normal(size=n))
        M1 = np.clip(rng.normal(m1d, a1 * m1d), 2.0, None)
        M2 = np.clip(rng.normal(m2d, a2 * m2d), 1.0, None)
        if sigma_from_obs:
            # THE FIX under test: sigma_ang is a function of OBSERVED data, so
            # the fixed-width sky likelihood (and the mock's PE cloud) is exact.
            mch = (M1 * M2) ** 0.6 / (M1 + M2) ** 0.2
            rho_opt = snr_ref_control * (mch / 30.0) ** (5.0 / 6.0) * (1000.0 / D)
        else:
            mch = (m1d * m2d) ** 0.6 / (m1d + m2d) ** 0.2
            rho_opt = snr_ref_control * (mch / 30.0) ** (5.0 / 6.0) * (1000.0 / dl)
        sa = np.deg2rad(np.clip(35.0 / rho_opt, 1.0, 12.0))
        A = (ra + rng.normal(0.0, sa / np.maximum(np.cos(dec), 0.1))) % (2 * np.pi)
        B = np.clip(dec + rng.normal(0.0, sa), -np.pi / 2, np.pi / 2)
        mco = (M1 * M2) ** 0.6 / (M1 + M2) ** 0.2
        rho_obs = SNR_REF * (mco / 30.0) ** (5.0 / 6.0) * (1000.0 / D)
        det = (rho_obs >= 8.0) & (rng.uniform(size=n) < 1.0 / t)  # gamma=0 rate acc
        n_tried += n
        for k, v in [("obs_dL", D), ("obs_m1det", M1), ("obs_m2det", M2),
                     ("obs_ra", A), ("obs_dec", B), ("obs_sigma_ang", sa),
                     ("obs_sig_m1", a1 * m1d), ("obs_sig_m2", a2 * m2d),
                     ("z", z), ("ra", ra), ("dec", dec)]:
            out[k].append(v[det])
        n_have += int(det.sum())
    E = {k: np.concatenate(v)[:nobs] for k, v in out.items()}
    return E, n_tried


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="existing catalog tag (mu reused)")
    ap.add_argument("--catalog", required=True)
    ap.add_argument("--rep", type=int, required=True, help="fresh-events seed index")
    ap.add_argument("--nobs", type=int, default=1000)
    ap.add_argument("--sigma_from_obs", action="store_true")
    args = ap.parse_args(argv)

    grids = oe.make_grids()
    with h5py.File(args.catalog) as f:
        catalog = {k: f[k][:] for k in ("ra", "dec", "z")}
    rng = np.random.default_rng(910_000 + 1000 * args.rep + hash(args.tag) % 997)
    E, n_tried = draw_fresh_events(catalog, grids, rng, nobs=args.nobs,
                                   sigma_from_obs=args.sigma_from_obs)
    print(f"fresh events: {args.nobs} detected / {n_tried} proposed "
          f"(frac {args.nobs/n_tried:.3e}); z med {np.median(E['z']):.4f}")

    H0s = np.linspace(58.0, 78.0, 161)
    sfx = "fix" if args.sigma_from_obs else ""
    out = EXP / "results" / f"oracle_boot{sfx}_{args.tag}_{args.rep}.npz"
    oe.run_numerators(E, args.catalog, out, H0s)

    num = np.load(out); mu = np.load(EXP / "results" / f"oracle_mu_{args.tag}.npz")
    tot = num["ln_O1"].sum(0) - args.nobs * mu["ln_mu"]
    p = np.exp(tot - tot.max())
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (p[1:] + p[:-1]) * np.diff(H0s))])
    cdf /= cdf[-1]
    med = float(np.interp(0.5, cdf, H0s))
    print(f"BOOT{sfx} {args.tag} rep {args.rep}: oracle median offset {med - 67.74:+.3f}")


if __name__ == "__main__":
    main()
