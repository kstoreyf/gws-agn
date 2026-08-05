#!/usr/bin/env python3
"""Analytic model of the selection term's H0 slope (mechanism c).

The mock detects on TRUE redshift z <= 1, i.e. (injections carry exact dL(z))
on a hard luminosity-distance cut dL <= dLmax = dL(1; H0_true).  Under the
analysis model at trial H0 that cut is z <= z*(H0) with dL(z*; H0) = dLmax, so

    mu_model(H0) = (1-f) F_gal(z*(H0)) + f F_agn(z*(H0)),

with F_k the catalog redshift CDF (unit weights).  The mass-model factor
integrates to ~1 (detection is mass-blind) and only adds MC noise to the
injection estimate.  If mu_model reproduces the measured ln mu(H0) slope, the
slope is NOT an injection-coverage artifact: it is the exact model prediction
for a dL-threshold selection -- which is inconsistent with the data-generating
z-threshold (whose correct beta is H0-independent), and that inconsistency is
the selection mechanism.

Writes results/tilt_selection_model.json.
"""
import json
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("DARKSIRENS_ZMAX", "1.5")

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
H0_TRUE = 67.74
OM0 = 0.3075


def main():
    import h5py
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from darksirens.utils.cosmology import dL_of_z, z_of_dL

    # catalog redshifts
    zcat = {}
    for name, path in (("gal", ROOT / "data" / "gal.h5"),
                       ("agn", ROOT / "data" / "agn.h5")):
        with h5py.File(path, "r") as f:
            zg = f["zgals"][:]; ng = f["ngals"][:]
            mask = np.arange(zg.shape[1])[None, :] < ng[:, None]
            zcat[name] = np.sort(zg[mask])

    dLmax = float(dL_of_z(jnp.asarray(1.0), H0_TRUE, OM0, -1.0, 0.0))
    H0s = np.linspace(55.0, 80.0, 251)
    zstar = np.array([float(z_of_dL(jnp.asarray(dLmax), h, OM0, -1.0, 0.0))
                      for h in H0s])

    def F(name, z):
        return np.searchsorted(zcat[name], z) / zcat[name].size

    # population weight per catalog object: (1+z)^(gamma-1) with gamma=0
    # (the analysis' fixed rate parameter), i.e. w = 1/(1+z).
    wsum = {k: np.sum(1.0 / (1.0 + zcat[k])) for k in zcat}
    cw = {k: np.cumsum(1.0 / (1.0 + zcat[k])) / wsum[k] for k in zcat}

    def Fw(name, z):
        i = np.searchsorted(zcat[name], z)
        return cw[name][i - 1] if i > 0 else 0.0

    out = {"dLmax_Mpc": dLmax, "H0_grid": H0s.tolist(),
           "zstar": zstar.tolist()}
    i0 = int(np.argmin(np.abs(H0s - H0_TRUE)))
    out["zstar_at_60_truth_75"] = [float(np.interp(60, H0s, zstar)),
                                   float(zstar[i0]),
                                   float(np.interp(75, H0s, zstar))]

    for f_agn, tag in ((0.307, "fagn0.3"), (0.703, "fagn0.7")):
        Fg = np.array([F("gal", z) for z in zstar])
        Fa = np.array([F("agn", z) for z in zstar])
        mu = (1 - f_agn) * Fg + f_agn * Fa
        lnmu = np.log(mu)
        slope = float(np.gradient(lnmu, H0s)[i0])
        Fgw = np.array([Fw("gal", z) for z in zstar])
        Faw = np.array([Fw("agn", z) for z in zstar])
        lnmu_w = np.log((1 - f_agn) * Fgw + f_agn * Faw)
        slope_w = float(np.gradient(lnmu_w, H0s)[i0])
        # measured
        dec = json.loads((ROOT / "results" / f"h0_decomposition_{tag}.json"
                          ).read_text())
        Hm = np.array(dec["H0_grid"]); lm = np.array(dec["log_mu"])
        slope_meas = float(np.gradient(lm, Hm)[int(np.argmin(np.abs(Hm - H0_TRUE)))])
        # shape residual over the common grid
        lnmu_i = np.interp(Hm, H0s, lnmu)
        resid = (lnmu_i - lnmu_i[np.argmin(np.abs(Hm - H0_TRUE))]) \
            - (lm - lm[np.argmin(np.abs(Hm - H0_TRUE))])
        out[tag] = {
            "f_agn": f_agn,
            "lnmu_model": lnmu.tolist(),
            "lnmu_model_zweighted": lnmu_w.tolist(),
            "dlnmu_dH0_model_at_truth": slope,
            "dlnmu_dH0_model_zweighted_at_truth": slope_w,
            "dlnmu_dH0_measured_at_truth": slope_meas,
            "dlnmu_dH0_gal_component": float(np.gradient(np.log(Fg), H0s)[i0]),
            "dlnmu_dH0_agn_component": float(np.gradient(np.log(Fa), H0s)[i0]),
            "shape_residual_max_nats": float(np.max(np.abs(resid))),
            "shape_range_nats": float(lm.max() - lm.min()),
        }
        print(f"{tag}: model slope {slope:+.5f}  z-weighted {slope_w:+.5f}  "
              f"measured {slope_meas:+.5f}  "
              f"shape residual max {out[tag]['shape_residual_max_nats']:.4f} "
              f"of range {out[tag]['shape_range_nats']:.3f}")

    p = ROOT / "results" / "tilt_selection_model.json"
    p.write_text(json.dumps(out, indent=2))
    print("wrote", p)


if __name__ == "__main__":
    main()
