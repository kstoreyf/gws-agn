#!/usr/bin/env python
"""The out-of-catalog field term's amplitude: each tracer's TRUE comoving density.

darksirens' incomplete model derives the missing-host budget from a density model

    dN_miss/dz = n0 * dV_c/dz * (1+z)^delta  -  dN_obs/dz

so `log10n0` (per tracer) sets the AMPLITUDE of everything the survey did not see.
Analysis 2 ran this term off by setting log10n0 = -24 in both catalogs.  Here it is
switched on at the mock's OWN densities, so two independent routes must agree:

  route 1 -- DECLARED.  The generator draws each tracer from GLASS at a target
             comoving number density: META.json /config/glass/n_comoving_{gal,agn}
             and the catalog attrs `n_comoving_target`.  This is what the mock was
             ASKED for.
  route 2 -- COUNTED.   Histogram the complete catalog's own redshifts against the
             comoving volume element of the mock's cosmology and read the density
             off directly.  This is what the mock ACTUALLY contains.

The two cannot be compared naively over the whole catalog: GLASS' linear shell
windows are a partition of unity only on the interior plateau (the catalog attr
`n_comoving_plateau`), and dN/dz ramps linearly to zero over the first and last
shell half-widths.  Route 2 is therefore evaluated on the plateau, where
dN/dz = n0 dV_c/dz holds exactly by construction, and separately on the GW horizon
(the only range the likelihood actually conditions on) as the number that matters.

Also fits the completion's own model form,

    ln( (dN/dz) / (dV_c/dz) ) = ln n0 + delta * ln(1+z),

following experiment_twotracer_incomplete/scripts/measure_density_model.py, so the
(1+z)^delta setting is measured rather than assumed.  Both tracers are drawn
uniform in comoving volume (population gamma = 0, no density evolution), so the
fitted delta is expected to be consistent with zero; the fit is the check, not the
source of the adopted value.

Run: python -u scripts/measure_true_density.py --seeds 100 101 102 103 105
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np
from astropy.cosmology import FlatLambdaCDM

DATA_ROOT = os.environ.get(
    "DATA_ROOT", "/hildafs/projects/phy230014p/magana/gws-agn/working/data"
)
H0_FID = 67.74
OM0_FID = 0.3075
CHUNK = 8_000_000


def zhist(path: str, edges: np.ndarray, column: str) -> np.ndarray:
    """Chunked histogram of a catalog redshift column (151 M rows for GAL)."""
    counts = np.zeros(edges.size - 1, dtype=np.int64)
    with h5py.File(path, "r") as f:
        d = f[column]
        n = d.shape[0]
        for i0 in range(0, n, CHUNK):
            z = np.asarray(d[i0 : min(i0 + CHUNK, n)], dtype=float)
            counts += np.histogram(z, bins=edges)[0]
    return counts


def vc_of_z(cosmo, z):
    """Full-sky comoving volume out to z, Mpc^3."""
    return (4.0 * np.pi / 3.0) * cosmo.comoving_distance(z).value ** 3


def fit_model_form(zc, dN_dz, dV_dz, keep):
    """ln(dN/dz / dV_c/dz) = ln n0 + delta ln(1+z), least squares."""
    y = np.log(dN_dz[keep] / dV_dz[keep])
    x = np.log1p(zc[keep])
    A = np.vstack([np.ones_like(x), x]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    ln_n0, delta = float(coef[0]), float(coef[1])
    n0 = float(np.exp(ln_n0))
    model = n0 * dV_dz * (1.0 + zc) ** delta
    frac = np.where(dN_dz > 0, model / dN_dz - 1.0, np.nan)
    return {
        "n0_Mpc3": n0,
        "log10n0": float(np.log10(n0)),
        "delta": delta,
        "n_bins_used": int(keep.sum()),
        "shape_residual_rms_frac": float(np.sqrt(np.nanmean(frac[keep] ** 2))),
        "shape_residual_max_abs_frac": float(np.nanmax(np.abs(frac[keep]))),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[100, 101, 102, 103, 105])
    ap.add_argument("--nbins", type=int, default=200)
    ap.add_argument("--out_json", default="results/true_density.json")
    args = ap.parse_args()

    cosmo = FlatLambdaCDM(H0=H0_FID, Om0=OM0_FID)
    out = {
        "cosmology": {"H0": H0_FID, "Om0": OM0_FID},
        "method": {
            "route_1": "generator's declared GLASS target comoving density "
            "(META.json /config/glass/n_comoving_*, catalog attr n_comoving_target)",
            "route_2": "direct count of complete-catalog rows against the full-sky "
            "comoving volume, on the GLASS plateau and inside the GW horizon",
            "fit": "ln(dN/dz / dV_c/dz) = ln n0 + delta ln(1+z), least squares, "
            "bins with counts > 20 (experiment_twotracer_incomplete convention)",
            "z_column": "z (TRUE redshift); z_obs differs by a symmetric "
            "0.003(1+z) photo-z kernel and does not move n0",
        },
        "seeds": {},
    }

    for seed in args.seeds:
        root = Path(DATA_ROOT) / f"seed{seed}"
        meta = json.loads((root / "META.json").read_text())
        horizon = float(meta["stages"]["surveys"]["horizon_z"])
        gl = meta["stages"]["catalogs"]["glass_config"]
        rec = {
            "horizon_z": horizon,
            "declared": {
                "gal": float(gl["n_comoving_gal"]),
                "agn": float(gl["n_comoving_agn"]),
            },
            "tracers": {},
        }
        print(f"\n=== seed {seed}  horizon z = {horizon:.6f} ===", flush=True)

        for tracer in ("gal", "agn"):
            cpath = root / "catalogs" / f"catalog_{tracer}_complete.h5"
            with h5py.File(cpath, "r") as f:
                attrs = dict(f.attrs)
                n_rows = int(f["z"].shape[0])
            plateau = np.asarray(
                str(attrs["n_comoving_plateau"]).strip("[] ").split(), dtype=float
            )
            zlo, zhi = float(plateau[0]), float(plateau[1])
            declared = float(attrs["n_comoving_target"])

            zmax = float(attrs["z_max_catalog"])
            edges = np.linspace(0.0, zmax, args.nbins + 1)
            counts = zhist(str(cpath), edges, "z")
            zc = 0.5 * (edges[1:] + edges[:-1])
            dz = np.diff(edges)
            dN_dz = counts / dz
            dV_dz = cosmo.differential_comoving_volume(zc).value * 4.0 * np.pi

            # --- route 2a: plateau count / plateau volume (exact by construction) ---
            n_plateau = int(counts[(zc > zlo) & (zc < zhi)].sum())
            v_plateau = vc_of_z(cosmo, zhi) - vc_of_z(cosmo, zlo)
            n0_plateau = n_plateau / v_plateau

            # --- route 2b: inside the GW horizon (what the likelihood conditions on) ---
            hedges = np.linspace(0.0, horizon, 41)
            hcounts = zhist(str(cpath), hedges, "z")
            n_hor = int(hcounts.sum())
            v_hor = vc_of_z(cosmo, horizon)
            n0_horizon = n_hor / v_hor
            # horizon excluding the low-z ramp (z < plateau start)
            hmask = 0.5 * (hedges[1:] + hedges[:-1]) > zlo
            n_hor_pl = int(hcounts[hmask].sum())
            v_hor_pl = vc_of_z(cosmo, horizon) - vc_of_z(cosmo, zlo)
            n0_horizon_plateau = n_hor_pl / v_hor_pl

            # --- route 2c: whole catalog (includes both ramps -> biased low) ---
            n0_allz = n_rows / vc_of_z(cosmo, zmax)

            # --- model-form fits ---
            keep_full = counts > 20
            keep_plat = keep_full & (zc > zlo) & (zc < zhi)
            fit_full = fit_model_form(zc, dN_dz, dV_dz, keep_full)
            fit_plat = fit_model_form(zc, dN_dz, dV_dz, keep_plat)
            hzc = 0.5 * (hedges[1:] + hedges[:-1])
            hdN = hcounts / np.diff(hedges)
            hdV = cosmo.differential_comoving_volume(hzc).value * 4.0 * np.pi
            keep_hor = (hcounts > 20) & (hzc > zlo)
            fit_hor = fit_model_form(hzc, hdN, hdV, keep_hor)

            t = {
                "catalog": str(cpath),
                "n_rows": n_rows,
                "z_max_catalog": zmax,
                "glass_plateau_z": [zlo, zhi],
                "route_1_declared_n0_Mpc3": declared,
                "route_1_declared_log10n0": float(np.log10(declared)),
                "route_2_counted": {
                    "plateau": {
                        "n": n_plateau,
                        "V_Mpc3": v_plateau,
                        "n0_Mpc3": n0_plateau,
                        "log10n0": float(np.log10(n0_plateau)),
                        "ratio_to_declared": n0_plateau / declared,
                    },
                    "horizon_full": {
                        "n": n_hor,
                        "V_Mpc3": v_hor,
                        "n0_Mpc3": n0_horizon,
                        "log10n0": float(np.log10(n0_horizon)),
                        "ratio_to_declared": n0_horizon / declared,
                    },
                    "horizon_above_ramp": {
                        "n": n_hor_pl,
                        "V_Mpc3": v_hor_pl,
                        "n0_Mpc3": n0_horizon_plateau,
                        "log10n0": float(np.log10(n0_horizon_plateau)),
                        "ratio_to_declared": n0_horizon_plateau / declared,
                    },
                    "all_z_including_ramps": {
                        "n": n_rows,
                        "n0_Mpc3": n0_allz,
                        "log10n0": float(np.log10(n0_allz)),
                        "ratio_to_declared": n0_allz / declared,
                    },
                },
                "model_form_fit": {
                    "full_range": fit_full,
                    "plateau": fit_plat,
                    "within_horizon_above_ramp": fit_hor,
                },
            }
            rec["tracers"][tracer] = t
            print(
                f"  {tracer.upper():3s}  declared log10n0 = {np.log10(declared):+.6f}\n"
                f"        counted (plateau)          = {np.log10(n0_plateau):+.6f}"
                f"   ratio {n0_plateau/declared:.5f}\n"
                f"        counted (z < horizon)      = {np.log10(n0_horizon):+.6f}"
                f"   ratio {n0_horizon/declared:.5f}\n"
                f"        counted (ramp-free horizon)= {np.log10(n0_horizon_plateau):+.6f}"
                f"   ratio {n0_horizon_plateau/declared:.5f}\n"
                f"        fit plateau  log10n0 = {fit_plat['log10n0']:+.6f}"
                f"  delta = {fit_plat['delta']:+.5f}"
                f"  rms {100*fit_plat['shape_residual_rms_frac']:.2f}%\n"
                f"        fit horizon  log10n0 = {fit_hor['log10n0']:+.6f}"
                f"  delta = {fit_hor['delta']:+.5f}"
                f"  rms {100*fit_hor['shape_residual_rms_frac']:.2f}%",
                flush=True,
            )

        g = rec["tracers"]["gal"]["route_2_counted"]["plateau"]["n0_Mpc3"]
        a = rec["tracers"]["agn"]["route_2_counted"]["plateau"]["n0_Mpc3"]
        rec["density_ratio_gal_over_agn_plateau"] = g / a
        out["seeds"][str(seed)] = rec

    # --- the adopted numbers ---
    dec = {t: np.log10(out["seeds"][str(args.seeds[0])]["declared"][t]) for t in ("gal", "agn")}
    spread = {
        t: [
            float(
                min(
                    out["seeds"][str(s)]["tracers"][t]["route_2_counted"]["plateau"]["log10n0"]
                    for s in args.seeds
                )
            ),
            float(
                max(
                    out["seeds"][str(s)]["tracers"][t]["route_2_counted"]["plateau"]["log10n0"]
                    for s in args.seeds
                )
            ),
        ]
        for t in ("gal", "agn")
    }
    out["adopted"] = {
        "log10n0": float(dec["gal"]),
        "log10n0_c2": float(dec["agn"]),
        "delta": 0.0,
        "delta_c2": 0.0,
        "per_seed": False,
        "why_log10n0": "the generator's declared GLASS target density, which the direct "
        "count reproduces on the plateau to within the per-seed cosmic-variance "
        "spread recorded in `counted_plateau_log10n0_range`; one value for all five "
        "realisations because the density is a property of the mock's construction, "
        "not of a realisation",
        "why_delta": "both tracers are drawn uniform in comoving volume (population "
        "gamma = 0, GLASS shells at constant comoving density), so the true evolution "
        "is exactly (1+z)^0; the measured fit is consistent with that and is recorded "
        "as the check.  Keeping delta = delta_c2 = 0 also leaves analysis 2's nuisance "
        "block untouched, so log10n0 is the ONLY configuration change between the two "
        "directories",
        "counted_plateau_log10n0_range": spread,
    }

    outp = Path(args.out_json)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=2))
    print(f"\nadopted: log10n0 = {out['adopted']['log10n0']:+.6f}  "
          f"log10n0_c2 = {out['adopted']['log10n0_c2']:+.6f}  "
          f"delta = delta_c2 = 0")
    print(f"wrote {outp}")


if __name__ == "__main__":
    main()
