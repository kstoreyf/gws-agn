#!/usr/bin/env python
"""Selection-term effective number of an injection file at a set of population/H0 points.
(core venv: core's population density; healpy for the AGN catalog's pixels)

    sbatch --job-name=stress run_cpu.sbatch core inj_stress.py FAMILY SEED INJ_FILE OUT.json

The selection term is mu = (1/N_draw) sum_det w_k, w_k = pi(theta_k | Lambda) / pdraw_k, in the
canonical basis (m1det, q, dL, chi_eff, sky) the injections store.  At a trial H0 a stored
injection is re-read at z' with dL(z'; H0) = dL (darksirens' convention), so

    pi = p_pop(m1det/(1+z'), q, chi | Lambda) * p_zsky(z', Omega) / [(1+z') dL'(z'; H0)]

with the rate factor (1+z')^(gamma-1) = (1+z')^-1 inside p_zsky, and p_zsky one of
  * vol    -- uniform in comoving volume (the GAL catalog at 1e-3 Mpc^-3 is ~8 galaxies per
              kernel width, i.e. smooth; its clustering is neglected), normalised on z' < 1.1;
  * agncat -- the seed's complete AGN survey, sum_j N(z'; z_j, dz_j) / Omega_pix per pixel
              (unit weights), normalised by sum_j (1+z_j)^-1.
p_pop is core's powerlaw+peak (FAMILY obslike or mock, core_family.py) with its own rate
factor removed.  Per point: N_eff = (sum w)^2 / sum w^2 and N^2/N_eff for N = 1000, for
  GAL class  (theta_G, vol), AGN class (theta_G + dmu_G 5, dmu_chi 0.10, agncat),
  host-dependent mixture  0.7 GAL class + 0.3 AGN class,
  equal-population mixture 0.7 (theta_G, vol) + 0.3 (theta_G, agncat),
  all-AGN-hosted         (theta_G, agncat).
Points: design; mu_G 30, 38; sigma_G 2.5, 6; H0 55, 90 (others at design).
"""
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import core_family as CF  # noqa: E402

DATA = Path("/hildafs/projects/phy230054p/magana/gws-agn-data-v3")
H0_FID, OM0 = 67.74, 0.3075
ZMAX_T = 1.10
N_EV = 1000
F = 0.30
# peak-dominated (mock) family: its own stress points around mu_G 35, sigma_G 5
POINTS_MOCK = {"design": {}, "mu_G_31": {"mu_G": 31.0}, "mu_G_39": {"mu_G": 39.0},
               "sigma_G_3.5": {"sigma_G": 3.5}, "sigma_G_7": {"sigma_G": 7.0},
               "H0_55": {"H0": 55.0}, "H0_90": {"H0": 90.0},
               "mu_chi_0.05": {"mu_chi": 0.05}, "sigma_chi_0.15": {"sigma_chi": 0.15}}
POINTS = {"design": {}, "mu_G_30": {"mu_G": 30.0}, "mu_G_38": {"mu_G": 38.0},
          "sigma_G_2.5": {"sigma_G": 2.5}, "sigma_G_6": {"sigma_G": 6.0},
          "H0_55": {"H0": 55.0}, "H0_90": {"H0": 90.0}}


def zgrids():
    from astropy.cosmology import FlatLambdaCDM
    c = FlatLambdaCDM(H0=H0_FID, Om0=OM0)
    z = np.linspace(0.0, 3.0, 60001)
    dl = c.luminosity_distance(z).value
    dvc = c.differential_comoving_volume(z).value
    return z, dl, dvc


def agn_kernel(seed, z, pix, chunk=100_000):
    with h5py.File(DATA / f"seed{seed}" / "surveys" / "survey_agn_complete_ns32.h5", "r") as f:
        zg, dz, ng = (np.asarray(f[k][:]) for k in ("zgals", "dzgals", "ngals"))
        nside = int(f.attrs["nside"])
    import healpy as hp
    valid = np.arange(zg.shape[1])[None, :] < ng[:, None]
    norm = float(((1.0 + zg) ** -1.0)[valid].sum())
    apix = hp.nside2pixarea(nside)
    out = np.empty(z.size)
    for i in range(0, z.size, chunk):
        s = slice(i, min(i + chunk, z.size))
        r = pix[s]
        zz = z[s][:, None]
        k = np.exp(-0.5 * ((zz - zg[r]) / dz[r]) ** 2) / (np.sqrt(2 * np.pi) * dz[r])
        out[s] = np.where(valid[r], k, 0.0).sum(axis=1)
    return out / apix / norm, nside


def main():
    fam, seed, inj, outp = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
    smult = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0     # offsets x s (s = 1: +5 Msun, +0.10)
    points = POINTS_MOCK if fam == "mock" else POINTS
    t0 = time.time()
    with h5py.File(inj, "r") as f:
        d = {k: np.asarray(f[k][:]) for k in ("m1det", "m2det", "dL", "chieff", "ra", "dec", "pdraw")}
        ndraw = int(f.attrs["Ndraw"])
    n = d["pdraw"].size
    q = d["m2det"] / d["m1det"]
    import healpy as hp
    zg, dlg, dvcg = zgrids()
    ddl = np.gradient(dlg, zg)
    sel = zg <= ZMAX_T
    rate = (1.0 + zg) ** -1.0
    vol_norm = np.trapz((dvcg * rate)[sel], zg[sel]) * 4 * np.pi   # per steradian below
    lp = {}
    out = {"family": fam, "seed": seed, "injection_file": str(inj), "ndraw": ndraw, "n_detected": n,
           "what": __doc__.split("\n\n")[1].strip(), "points": {}}
    base = CF.FAMILIES[fam]
    lps = {}
    out["offset_multiplier_s"] = smult
    for name, pt in points.items():
        h0 = pt.get("H0", H0_FID)
        r = h0 / H0_FID
        zp = np.interp(d["dL"] * r, dlg, zg)                 # dL(z'; H0) = dL_fid(z') / r
        jac = (1.0 + zp) * np.interp(zp, zg, ddl) / r
        p_vol = np.interp(zp, zg, dvcg) * (1.0 + zp) ** -1.0 / vol_norm * (zp <= ZMAX_T)
        key = round(h0, 6)
        if key not in lp:
            pix = hp.ang2pix(32, 0.5 * np.pi - d["dec"], d["ra"])
            ka, _ = agn_kernel(seed, zp, pix)
            lp[key] = ka * (1.0 + zp) ** -1.0 * (zp <= ZMAX_T)
        p_agn = lp[key]
        m1s = d["m1det"] / (1.0 + zp)
        thG = dict(base)
        thG.update({k: v for k, v in pt.items() if k != "H0"})
        thA = dict(thG)
        thA["mu_G"] += smult * CF.AGN_SHIFT["mu_G"]
        thA["mu_chi"] += smult * CF.AGN_SHIFT["mu_chi"]
        res = {"H0": h0, "theta_G": thG, "theta_A": thA}
        ws = {}
        for cls, th in (("G", thG), ("A", thA)):
            CF.FAMILIES["_tmp"] = th                      # setup() reads FAMILIES via thetas()
            if ("lp", cls, tuple(th.values())) not in lps:
                fn, _ = CF.setup("historical", "_tmp")
                lps[("lp", cls, tuple(th.values()))] = fn
            fn = lps[("lp", cls, tuple(th.values()))]
            lm = fn(m1s, q, d["chieff"], zp, "gal") + np.log1p(zp)   # drop core's rate factor
            ws[cls] = np.exp(lm) / d["pdraw"] / jac
        combos = {"GAL_class_vol": ws["G"] * p_vol,
                  "AGN_class_agncat": ws["A"] * p_agn,
                  "host_dependent_mix": (1 - F) * ws["G"] * p_vol + F * ws["A"] * p_agn,
                  "equal_population_mix": (1 - F) * ws["G"] * p_vol + F * ws["G"] * p_agn,
                  "all_agn_hosted": ws["G"] * p_agn,
                  **{f"host_dependent_mix_f{f:.1f}": (1 - f) * ws["G"] * p_vol + f * ws["A"] * p_agn
                     for f in (0.1, 0.5)},
                  **{f"equal_population_mix_f{f:.1f}": (1 - f) * ws["G"] * p_vol + f * ws["G"] * p_agn
                     for f in (0.1, 0.5)}}
        for k, w in combos.items():
            s1, s2 = w.sum(), (w ** 2).sum()
            neff = float(s1 ** 2 / s2) if s2 > 0 else 0.0
            res[k] = {"n_eff": neff, "N2_over_neff": N_EV ** 2 / neff if neff > 0 else None,
                      "mu": float(s1 / ndraw), "max_w_over_mean": float(w.max() / (s1 / n)),
                      "frac_rows_nonzero": float((w > 0).mean())}
        out["points"][name] = res
        print(name, {k: (round(v["n_eff"]), round(v["N2_over_neff"], 3)) for k, v in res.items()
                     if isinstance(v, dict) and "n_eff" in v}, f"[{time.time() - t0:.0f}s]", flush=True)
    worst = {k: max(out["points"][p][k]["N2_over_neff"] for p in points)
             for k in out["points"]["design"] if isinstance(out["points"]["design"][k], dict)
             and "n_eff" in out["points"]["design"][k]}
    out["worst_N2_over_neff"] = worst
    out["seconds"] = time.time() - t0
    outp.write_text(json.dumps(out, indent=1))
    print("worst", worst, "wrote", outp)


if __name__ == "__main__":
    main()
