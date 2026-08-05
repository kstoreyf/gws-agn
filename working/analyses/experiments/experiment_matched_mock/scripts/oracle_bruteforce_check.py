#!/usr/bin/env python3
"""Brute-force numerator scores vs the oracle, per event.

For each event, L(d;H0) is estimated by direct Monte Carlo over the generative
prior: hosts from the catalog (weight 1/(1+z)), masses from gmd's own SAMPLERS
(so any sampler-vs-pdf discrepancy is included), the theta-dependent sigma_ang
sky width (the channel the oracle freezes), and the generative heteroscedastic
mass-noise widths.  Scores d lnL/dH0 at truth (finite difference +-0.5 km/s/Mpc)
for three variants:

  BF_full   : sampler masses, sigma_ang(theta) live (varies with host, masses, H0)
  BF_fixsig : sampler masses, sigma_ang frozen at the event's stored value
  oracle O1 : pdf masses, sigma frozen (from oracle_num_<tag>.npz)

Mean differences over events isolate: (BF_fixsig - O1) = sampler-vs-pdf mass
density (+ quadrature); (BF_full - BF_fixsig) = the sigma_ang(theta) channel.
"""
import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "scripts"))
import oracle_exact as oe  # noqa: E402
import generate_mock_data as gmd  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--events_file", required=True)
    ap.add_argument("--catalog", required=True)
    ap.add_argument("--nevents", type=int, default=120)
    ap.add_argument("--nmass", type=int, default=400_000)
    ap.add_argument("--seed", type=int, default=5150)
    args = ap.parse_args(argv)

    grids = oe.make_grids()
    zg = grids["z"]; lndl_g = np.log(np.maximum(grids["dl"], 1e-9))
    with h5py.File(args.catalog) as f:
        cra = f["ra"][:]; cdec = f["dec"][:]; cz = f["z"][:]
    order = np.argsort(cdec)
    cra, cdec, cz = cra[order], cdec[order], cz[order]
    clndl = np.interp(cz, zg, lndl_g)

    with h5py.File(args.events_file) as f:
        t = f["truth"]
        E = {k: t[k][:] for k in ("obs_dL", "obs_m1det", "obs_m2det", "obs_ra",
                                  "obs_dec", "obs_sigma_ang")}

    rng = np.random.default_rng(args.seed)
    K = args.nmass
    m1, up = gmd._sample_powerlaw_peak_m1(rng, K, oe.POP, return_component=True)
    q = gmd._sample_q(rng, m1, oe.POP, use_peak=up)
    m2 = q * m1

    num = np.load(EXP / "results" / f"oracle_num_{args.tag}.npz")
    H0g = num["H0"]; i0 = int(np.argmin(np.abs(H0g - 67.74))); hg = H0g[1] - H0g[0]

    dH = 0.5
    H0pts = np.array([67.74 - dH, 67.74 + dH])
    deltas = np.log(H0pts / 67.74)

    res = {"bf_full": [], "bf_fix": [], "oracle": []}
    N = min(args.nevents, len(E["obs_dL"]))
    s = 0.10
    for i in range(N):
        D = E["obs_dL"][i]; lnD = np.log(D)
        M1, M2 = E["obs_m1det"][i], E["obs_m2det"][i]
        sa0 = E["obs_sigma_ang"][i]
        A, B = E["obs_ra"][i], E["obs_dec"][i]
        lo = np.searchsorted(cdec, B - 6.5 * sa0)
        hi = np.searchsorted(cdec, B + 6.5 * sa0)
        j = slice(lo, hi)
        dra = np.angle(np.exp(1j * (cra[j] - A)))
        cosd = np.maximum(np.cos(cdec[j]), 0.1)
        r2 = (dra * cosd / sa0) ** 2 + ((cdec[j] - B) / sa0) ** 2
        keep = (r2 < 6.5 ** 2) & (np.abs(lnD - clndl[j]) < 0.7)
        zs = cz[j][keep]; lndls = clndl[j][keep]
        dra_k = dra[keep]; ddec_k = (cdec[j] - B)[keep]; cosd_k = cosd[keep]
        nh = zs.size
        if nh == 0:
            continue
        t1 = 1.0 + zs                                   # (nh,)
        w_host = 1.0 / t1
        # chunk hosts to bound (chunk, K) memory
        CH = 96
        acc = {("bf_full", 0): 0.0, ("bf_full", 1): 0.0,
               ("bf_fix", 0): 0.0, ("bf_fix", 1): 0.0}
        mx_glob = None
        # first pass: global max of lmass over a coarse probe for stability
        for c0 in range(0, nh, CH):
            sl = slice(c0, min(c0 + CH, nh))
            m1t = m1[None, :] * t1[sl, None]
            m2t = m2[None, :] * t1[sl, None]
            s1 = 0.08 * m1t; s2 = 0.10 * m2t
            lmass = (-0.5 * ((M1 - m1t) / s1) ** 2 - np.log(s1)
                     - 0.5 * ((M2 - m2t) / s2) ** 2 - np.log(s2))
            for var in ("bf_full", "bf_fix"):
                for kd, d in enumerate(deltas):
                    u = lnD - lndls[sl] + d
                    ldist = -0.5 * (u / s) ** 2
                    if var == "bf_full":
                        dl_h = np.exp(lndls[sl] - d)
                        mch = (m1t * m2t) ** 0.6 / (m1t + m2t) ** 0.2
                        rho = 11.5 * (mch / 30.0) ** (5.0 / 6.0) \
                            * (1000.0 / dl_h[:, None])
                        sa = np.deg2rad(np.clip(35.0 / rho, 1.0, 12.0))
                        lsky = (-0.5 * (dra_k[sl, None] * cosd_k[sl, None] / sa) ** 2
                                - 0.5 * (ddec_k[sl, None] / sa) ** 2
                                - 2.0 * np.log(sa))
                    else:
                        sa0v = sa0
                        lsky = (-0.5 * (dra_k[sl] * cosd_k[sl] / sa0v) ** 2
                                - 0.5 * (ddec_k[sl] / sa0v) ** 2
                                - 2.0 * np.log(sa0v))[:, None]
                    m = lmass + lsky + ldist[:, None]
                    acc[(var, kd)] += float(np.sum(w_host[sl, None] * np.exp(m - 0.0)))
        out = {}
        for var in ("bf_full", "bf_fix"):
            l0 = np.log(max(acc[(var, 0)] / K, 1e-300))
            l1 = np.log(max(acc[(var, 1)] / K, 1e-300))
            out[var] = (l1 - l0) / (2 * dH)
        res["bf_full"].append(out["bf_full"])
        res["bf_fix"].append(out["bf_fix"])
        c = num["ln_O1"][i]
        res["oracle"].append(float((c[i0 + 4] - c[i0 - 4]) / (8 * hg)))
        if i % 20 == 0:
            print(i, flush=True)
    for k in res:
        res[k] = np.asarray(res[k])
    d_pdf = res["bf_fix"] - res["oracle"]
    d_sig = res["bf_full"] - res["bf_fix"]
    print(f"n = {len(d_pdf)}")
    print(f"BF_fixsig - oracle (sampler-vs-pdf): mean {d_pdf.mean():+.5f} "
          f"sem {d_pdf.std(ddof=1)/np.sqrt(len(d_pdf)):.5f}")
    print(f"BF_full - BF_fixsig (sigma(theta) channel): mean {d_sig.mean():+.5f} "
          f"sem {d_sig.std(ddof=1)/np.sqrt(len(d_sig)):.5f}")
    print(f"oracle mean score {res['oracle'].mean():+.5f}; "
          f"BF_full mean {res['bf_full'].mean():+.5f}")
    np.savez(EXP / "results" / f"oracle_bfcheck_{args.tag}.npz", **res)


if __name__ == "__main__":
    main()
