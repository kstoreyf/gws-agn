#!/usr/bin/env python
"""Step 3 (c) of the obslike_f0p30_s1 H0-offset diagnosis: does each branch of the injection mixture, used alone, give the
same selection integral and the same detected-population prediction? (core venv)

    JAX_PLATFORMS=cpu OMP_NUM_THREADS=8 PYTHONPATH= <core venv python> tests/obslike_offset_step3_branches.py [lane] [realizations]

The lane's rows come from three branches (population, uniform, AGN-targeted), each with its own exactly known proposal
density p_b (stored pdraw_population, pdraw_uniform, pdraw_targeted_agn) and number of proposals n_b. Each branch alone is an
unbiased importance sampler of mu wherever p_b covers the target:
    mu_b(H0) = (1 / n_b) sum_{rows of b} pi(theta_k; H0) / p_b,k,
with pi the selection term's target (inj_stress.py convention: fixed dL re-read at z'(H0); core population of the class with
its rate factor removed; galaxy class: uniform in comoving volume, AGN class: the realization's complete AGN survey kernels;
rate factor (1+z')^-1). The mixture estimator is (1 / N) sum_k pi / pdraw. Disagreement between branches beyond Monte Carlo
error means a proposal density does not describe its sampler. Reported per (class population, branch): ln mu at H0 = 65.74,
67.74, 69.74, the slope d ln mu / dH0 (central difference), the predicted mean z, m1src, snr_true of detected mergers at
67.74, and N_eff. Classes: GAL (theta_G, vol), AGN-shifted (theta_G + 5 / +0.10, agncat; the s1 AGN class), AGN-unshifted
(theta_G, agncat; the s0 AGN class).
Output: tests/results/obslike_offset_step3_branches_<lane>.json
"""
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import h5py
import numpy as np

A16 = Path(__file__).resolve().parents[1]
DATA = A16.parents[1] / "data"
sys.path.insert(0, str(DATA / "obslike" / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import core_family as CF  # noqa: E402
from obslike_offset_step3_detected import agn_kernel, zgrids  # noqa: E402

H0_FID = 67.74
ZMAX_T = 1.10
SNR_REF = 6.278363879917771
LANES = {"obslike": ("injections_targeted_obslike.h5", "obslike"), "peak": ("injections_targeted_peak.h5", "mock")}
H0S = [65.74, 67.74, 69.74]


def main():
    lane = sys.argv[1] if len(sys.argv) > 1 else "obslike"
    reals = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [100, 101]
    fn, fam = LANES[lane]
    t0 = time.time()
    lp1, _ = CF.setup("historical", fam, 1.0)   # 'gal' = theta_G, 'agn' = shifted
    lp0, _ = CF.setup("historical", fam, 0.0)   # 'agn' = theta_G
    zg, dlg, dvcg = zgrids()
    ddl = np.gradient(dlg, zg)
    sel = zg <= ZMAX_T
    vol_norm = np.trapz((dvcg / (1 + zg))[sel], zg[sel]) * 4 * np.pi
    import healpy as hp
    out = {"what": __doc__.split("\n\n")[1].strip(), "lane": fn, "realizations": {}}
    for r in reals:
        with h5py.File(DATA / f"seed{r}" / "injections" / fn, "r") as f:
            d = {k: np.asarray(f[k][:]) for k in ("m1det", "m2det", "dL", "chieff", "ra", "dec", "pdraw", "branch",
                                                  "pdraw_population", "pdraw_uniform", "pdraw_targeted_agn")}
            meta = json.loads(f.attrs["metadata_json"])
        nprop = meta["n_proposed_branch"]
        ntot = int(meta["ndraw"])
        mix = meta["mixture_weights"]
        q = d["m2det"] / d["m1det"]
        mc = d["m1det"] * q ** 0.6 / (1 + q) ** 0.2
        snr = SNR_REF * (mc / 30.0) ** (5 / 6) * (1000.0 / d["dL"])
        pix = hp.ang2pix(32, 0.5 * np.pi - d["dec"], d["ra"])
        res = {}
        for h0 in H0S:
            rr = h0 / H0_FID
            zp = np.interp(d["dL"] * rr, dlg, zg)
            jac = (1.0 + zp) * np.interp(zp, zg, ddl) / rr
            p_vol = np.interp(zp, zg, dvcg) / (1 + zp) / vol_norm * (zp <= ZMAX_T)
            p_agn = agn_kernel(r, zp, pix) / (1 + zp) * (zp <= ZMAX_T)
            m1s = d["m1det"] / (1 + zp)
            targets = {}
            for cname, lp, cls, pz in (("GAL", lp1, "gal", p_vol), ("AGN_shifted", lp1, "agn", p_agn),
                                       ("AGN_unshifted", lp0, "agn", p_agn)):
                lm = lp(m1s, q, d["chieff"], zp, cls) + np.log1p(zp)
                targets[cname] = np.where(np.isfinite(lm), np.exp(lm), 0.0) * pz / jac
            for cname, tgt in targets.items():
                rows = {"mixture": (np.ones(tgt.size, bool), d["pdraw"], ntot)}
                for b, (bname, col) in enumerate((("population", "pdraw_population"), ("uniform", "pdraw_uniform"),
                                                  ("targeted_agn", "pdraw_targeted_agn"))):
                    rows[bname] = (d["branch"] == b, d[col], int(nprop[bname]))
                for bname, (m, pb, nb) in rows.items():
                    w = np.where(m, tgt / np.maximum(pb, 1e-300), 0.0)
                    s1, s2 = w.sum(), (w ** 2).sum()
                    rec = res.setdefault(cname, {}).setdefault(bname, {})
                    rec.setdefault("ln_mu", {})[str(h0)] = float(np.log(s1 / nb))
                    rec.setdefault("rel_err_mu", {})[str(h0)] = float(np.sqrt(s2) / s1)
                    if h0 == H0_FID:
                        rec["n_eff"] = float(s1 ** 2 / s2)
                        for nm, x in (("z", zp), ("m1src", m1s), ("snr_true", snr), ("chieff", d["chieff"])):
                            rec[f"mean_{nm}"] = float((w * x).sum() / s1)
                            rec[f"err_mean_{nm}"] = float(np.sqrt((w ** 2 * (x - (w * x).sum() / s1) ** 2).sum()) / s1)
            print(f"[{time.time() - t0:.0f}s] r{r} H0 {h0} done", flush=True)
        for cname, br in res.items():
            for bname, rec in br.items():
                lm = rec["ln_mu"]
                rec["dlnmu_dH0"] = (lm[str(H0S[2])] - lm[str(H0S[0])]) / (H0S[2] - H0S[0])
        out["realizations"][str(r)] = res
        print(f"=== r{r} ({fn})")
        for cname, br in res.items():
            for bname, rec in br.items():
                print(f"  {cname:13s} {bname:12s} ln mu(67.74) {rec['ln_mu'][str(H0_FID)]:+.5f} +- {rec['rel_err_mu'][str(H0_FID)]:.5f}"
                      f"  dlnmu/dH0 {rec['dlnmu_dH0']:+.5f}  Neff {rec['n_eff']:9.0f}  <z> {rec['mean_z']:.5f}+-{rec['err_mean_z']:.5f}"
                      f"  <m1> {rec['mean_m1src']:.3f}+-{rec['err_mean_m1src']:.3f}  <snr> {rec['mean_snr_true']:.3f}+-{rec['err_mean_snr_true']:.3f}",
                      flush=True)
    p = A16 / "tests" / "results" / f"obslike_offset_step3_branches_{lane}.json"
    p.write_text(json.dumps(out, indent=1))
    print("wrote", p)


if __name__ == "__main__":
    main()
