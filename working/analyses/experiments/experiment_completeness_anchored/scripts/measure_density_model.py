#!/usr/bin/env python3
"""Fit the completion's density model to the true host density, and report the anchor.

The completion derives the missing-host budget as

    dN_miss/dz = n0 * dV_c/dz * (1+z)^delta  -  dN_obs/dz

so completeness is never a free function; it is
`C(z) = (dN_obs/dz) / (n0 * dV_c/dz * (1+z)^delta)`.  Anchoring n0 therefore fixes
the budget's AMPLITUDE, but its SHAPE is still whatever `(1+z)^delta * dV_c/dz`
gives.  If the true host density does not follow that form, the mismatch presents as
a completion bias no amount of anchoring can remove.

So the anchor must be the BEST FIT OF THE MODEL FORM to the true density, not the raw
mean density, and the residual between fit and truth is the experiment's noise floor:
any apparent completeness bias smaller than it is not attributable.

Fits ln(dN/dz / (dV_c/dz)) = ln n0 + delta * ln(1+z) by least squares over a stated
redshift range, and reports the fractional residual of the fitted form.
"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from astropy.cosmology import FlatLambdaCDM


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--complete_catalog", required=True)
    ap.add_argument("--H0", type=float, default=67.74)
    ap.add_argument("--Om0", type=float, default=0.3075)
    ap.add_argument("--z_fit_max", type=float, default=None,
                    help="Upper edge of the fit range (default: the catalog's max z).")
    ap.add_argument("--z_ref", type=float, default=0.27,
                    help="GW horizon: the residual is also reported over z <= z_ref, "
                         "which is the range the data actually occupy.")
    ap.add_argument("--nbins", type=int, default=60)
    ap.add_argument("--out_json", required=True)
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    with h5py.File(args.complete_catalog, "r") as f:
        z = np.asarray(f["z"][:], dtype=float)

    cosmo = FlatLambdaCDM(H0=args.H0, Om0=args.Om0)
    zmax = args.z_fit_max if args.z_fit_max is not None else float(z.max())
    edges = np.linspace(0.0, zmax, args.nbins + 1)
    counts, _ = np.histogram(z, bins=edges)
    zc = 0.5 * (edges[1:] + edges[:-1])
    dz = np.diff(edges)

    # Full-sky comoving volume element [Mpc^3 per unit z].
    dV_dz = (cosmo.differential_comoving_volume(zc).value * 4.0 * np.pi)

    dN_dz = counts / dz
    ok = counts > 20                      # keep bins with usable Poisson precision
    y = np.log(dN_dz[ok] / dV_dz[ok])
    x = np.log1p(zc[ok])
    A = np.vstack([np.ones_like(x), x]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    ln_n0, delta = float(coef[0]), float(coef[1])
    n0 = float(np.exp(ln_n0))

    model = n0 * dV_dz * (1.0 + zc) ** delta
    frac = np.where(dN_dz > 0, model / dN_dz - 1.0, np.nan)
    inh = ok & (zc <= args.z_ref)

    out = {
        "complete_catalog": args.complete_catalog,
        "n_hosts": int(z.size), "z_max": float(z.max()),
        "fit_range": [0.0, zmax], "n_bins_used": int(ok.sum()),
        "cosmology": {"H0": args.H0, "Om0": args.Om0},
        "anchor": {"n0_Mpc3": n0, "log10n0": float(np.log10(n0)), "delta": delta},
        "naive_mean_density_log10n0": float(np.log10(
            z.size / (cosmo.comoving_volume(zmax).value * 1.0))),
        "shape_residual_fit_range": {
            "rms_frac": float(np.sqrt(np.nanmean(frac[ok] ** 2))),
            "max_abs_frac": float(np.nanmax(np.abs(frac[ok]))),
        },
        "shape_residual_within_z_ref": {
            "z_ref": args.z_ref, "n_bins": int(inh.sum()),
            "rms_frac": (float(np.sqrt(np.nanmean(frac[inh] ** 2)))
                         if inh.any() else None),
            "max_abs_frac": (float(np.nanmax(np.abs(frac[inh])))
                             if inh.any() else None),
        },
        "note": ("Anchor log10n0/delta at these values. The residual is the noise "
                 "floor: an apparent completeness bias below it is not attributable."),
    }
    Path(args.out_json).write_text(json.dumps(out, indent=2))

    print(f"hosts {z.size:,}  z_max {z.max():.3f}  bins used {ok.sum()}/{args.nbins}")
    print(f"ANCHOR: log10n0 = {out['anchor']['log10n0']:.6f}   delta = {delta:+.4f}")
    print(f"  (naive mean-density log10n0 would be "
          f"{out['naive_mean_density_log10n0']:.6f})")
    print(f"shape residual over fit range : rms {100*out['shape_residual_fit_range']['rms_frac']:.2f}%"
          f"  max {100*out['shape_residual_fit_range']['max_abs_frac']:.2f}%")
    r = out["shape_residual_within_z_ref"]
    if r["rms_frac"] is not None:
        print(f"shape residual within z<={args.z_ref} : rms {100*r['rms_frac']:.2f}%"
              f"  max {100*r['max_abs_frac']:.2f}%  ({r['n_bins']} bins)")
    print(f"Wrote {args.out_json}")


if __name__ == "__main__":
    main()
