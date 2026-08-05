#!/usr/bin/env python3
"""Fast brute-force numerator-score check (see oracle_bruteforce_check.py).

Mass draws are sorted by m1; per host only draws with m1det within +-6 sigma of
M1 are evaluated (the rest carry exp(<-18) mass weight).  Prints running means
per event so partial output is informative.
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
    ap.add_argument("--nevents", type=int, default=400)
    ap.add_argument("--nmass", type=int, default=2_000_000)
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
    o = np.argsort(m1)
    m1 = m1[o]; m2 = m2[o]

    num = np.load(EXP / "results" / f"oracle_num_{args.tag}.npz")
    H0g = num["H0"]; i0 = int(np.argmin(np.abs(H0g - 67.74))); hg = H0g[1] - H0g[0]

    dH = 0.5
    deltas = np.log(np.array([67.74 - dH, 67.74 + dH]) / 67.74)
    s = 0.10
    res = {"bf_full": [], "bf_fix": [], "oracle": []}
    N = min(args.nevents, len(E["obs_dL"]))
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
        acc = np.zeros((2, 2))  # (variant, delta)
        for hh in range(nh):
            t1 = 1.0 + zs[hh]
            mlo = (M1 / t1) * (1 - 6 * 0.08)
            mhi = (M1 / t1) * (1 + 6 * 0.08)
            a0, a1i = np.searchsorted(m1, [mlo, mhi])
            if a1i <= a0:
                continue
            mm1 = m1[a0:a1i]; mm2 = m2[a0:a1i]
            m1t = mm1 * t1; m2t = mm2 * t1
            s1 = 0.08 * m1t; s2 = 0.10 * m2t
            lmass = (-0.5 * ((M1 - m1t) / s1) ** 2 - np.log(s1)
                     - 0.5 * ((M2 - m2t) / s2) ** 2 - np.log(s2))
            wmass = np.exp(lmass)
            for kd, d in enumerate(deltas):
                u = lnD - lndls[hh] + d
                wd = np.exp(-0.5 * (u / s) ** 2) / t1
                # fixed-sigma sky
                lsky0 = (-0.5 * (dra_k[hh] * cosd_k[hh] / sa0) ** 2
                         - 0.5 * (ddec_k[hh] / sa0) ** 2 - 2 * np.log(sa0))
                acc[1, kd] += wd * np.exp(lsky0) * wmass.sum()
                # live sigma(theta)
                dl_h = np.exp(lndls[hh] - d)
                mch = (m1t * m2t) ** 0.6 / (m1t + m2t) ** 0.2
                rho = 11.5 * (mch / 30.0) ** (5.0 / 6.0) * (1000.0 / dl_h)
                sa = np.deg2rad(np.clip(35.0 / rho, 1.0, 12.0))
                lsky = (-0.5 * (dra_k[hh] * cosd_k[hh] / sa) ** 2
                        - 0.5 * (ddec_k[hh] / sa) ** 2 - 2 * np.log(sa))
                acc[0, kd] += wd * np.sum(wmass * np.exp(lsky))
        if acc.min() <= 0:
            continue
        sc_full = (np.log(acc[0, 1]) - np.log(acc[0, 0])) / (2 * dH)
        sc_fix = (np.log(acc[1, 1]) - np.log(acc[1, 0])) / (2 * dH)
        c = num["ln_O1"][i]
        sc_or = float((c[i0 + 4] - c[i0 - 4]) / (8 * hg))
        res["bf_full"].append(sc_full)
        res["bf_fix"].append(sc_fix)
        res["oracle"].append(sc_or)
        if (i + 1) % 10 == 0:
            a = {k: np.asarray(v) for k, v in res.items()}
            dp = a["bf_fix"] - a["oracle"]; dsg = a["bf_full"] - a["bf_fix"]
            n = len(dp)
            print(f"[{i+1}] n={n} pdfgap {dp.mean():+.5f}±{dp.std(ddof=1)/np.sqrt(n):.5f} "
                  f"siggap {dsg.mean():+.5f}±{dsg.std(ddof=1)/np.sqrt(n):.5f} "
                  f"ormean {a['oracle'].mean():+.5f} fullmean {a['bf_full'].mean():+.5f}",
                  flush=True)
    a = {k: np.asarray(v) for k, v in res.items()}
    np.savez(EXP / "results" / f"oracle_bfcheck_{args.tag}.npz", **a)
    print("done")


if __name__ == "__main__":
    main()
