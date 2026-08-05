#!/usr/bin/env python3
"""Pixelate gmd's COMPLETE galaxy catalog into a darksirens survey file.

Why not use gmd's own `mock_survey_raw.h5` / `catalog_pixelated_nside_*.h5`?
Because gmd's observed survey applies `SurveyConfig` cuts that are not exposed on
its CLI: a declination footprint (-40..+80 deg) and a HARD redshift truncation at
`z_hard_max = 1.2`.  With events reaching z ~ 1 that edge sits only ~0.2 beyond the
data — closer than the z ~ 1.56 edge of the GLASS catalog whose relocation already
moved H0 by 4 km/s/Mpc (../experiment_h0f_baseline, edge test).  A deep-catalog
control has to keep the catalog's edge far from the events, so the survey here is the
COMPLETE catalog (drawn to `--zmax`, e.g. 3.0), pixelated with gmd's own
`_pixelate_catalog` so the on-disk format is byte-for-byte what darksirens expects.

The footprint cut is also anisotropic, which for the multitracer estimand is not a
neutral detail: an anisotropic completeness modelled as isotropic imprints a
sky-density contrast, the same channel that identifies f_AGN. Using the complete
catalog keeps this control isotropic and full-sky by construction.

Outputs `<out_path>` with the survey datasets darksirens' loader expects
(`zgals`, `dzgals`, `wgals`, `ngals`) and the `nside` attr, plus provenance.
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--complete_catalog", required=True,
                    help="gmd's mock_galaxy_catalog_complete.h5")
    ap.add_argument("--out_path", required=True)
    ap.add_argument("--nside", type=int, default=64)
    ap.add_argument("--gmd_dir", required=True,
                    help="Directory containing generate_mock_data.py (for _pixelate_catalog).")
    ap.add_argument("--z_error_floor", type=float, default=0.0005,
                    help="Matches SurveyConfig.redshift_error_floor.")
    ap.add_argument("--z_error_slope", type=float, default=0.0015,
                    help="Matches SurveyConfig.redshift_error_slope.")
    ap.add_argument("--subsample", type=float, default=None,
                    help="Optional ISOTROPIC random thinning fraction in (0,1]; a "
                         "uniform thinning is a z-independent completeness, which "
                         "changes the density normalisation but NOT the shape.")
    ap.add_argument("--z_keep_max", type=float, default=None,
                    help="Keep only hosts with z <= this. Used to test whether hosts "
                         "BEYOND the GW detection horizon affect the result: in field "
                         "mode the survey-global normaliser Z sums over every observed "
                         "host, while mu integrates only over detectable parameter "
                         "space, so unreachable hosts enter one side and not the other.")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    sys.path.insert(0, str(Path(args.gmd_dir).resolve()))
    import generate_mock_data as gmd   # noqa: E402

    with h5py.File(args.complete_catalog, "r") as f:
        keys = list(f.keys())
        ra = np.asarray(f["ra"][:], dtype=float)
        dec = np.asarray(f["dec"][:], dtype=float)
        z = np.asarray(f["z"][:], dtype=float)
        src_attrs = {k: (v.item() if hasattr(v, "item") else v)
                     for k, v in f.attrs.items()}
    print(f"complete catalog: {args.complete_catalog}")
    print(f"  datasets: {keys}")
    print(f"  n = {z.size:,}  z range [{z.min():.4f}, {z.max():.4f}]")

    rng = np.random.default_rng(args.seed)
    n_before = z.size
    if args.z_keep_max is not None:
        keep = z <= args.z_keep_max
        ra, dec, z = ra[keep], dec[keep], z[keep]
        print(f"  z <= {args.z_keep_max}: {z.size:,} of {n_before:,} kept")
    if args.subsample is not None:
        if not (0.0 < args.subsample <= 1.0):
            raise SystemExit("--subsample must be in (0, 1]")
        keep = rng.uniform(size=z.size) < args.subsample
        ra, dec, z = ra[keep], dec[keep], z[keep]
        print(f"  isotropic thinning {args.subsample}: {z.size:,} of {n_before:,} kept")

    # Photometric-style redshift errors on the same footing as SurveyConfig, so the
    # catalog KDE widths match what gmd's own observed survey would have carried.
    dz = args.z_error_floor + args.z_error_slope * z
    w = np.ones_like(z)

    pix = gmd._pixelate_catalog(ra, dec, z, dz, w, args.nside, None)
    print(f"  pixelated: keys={list(pix.keys())}")
    for k, v in pix.items():
        print(f"    {k:10s} {np.shape(v)}")

    ngals = np.asarray(pix["ngals"])
    occupied = int((ngals > 0).sum())
    npix = int(ngals.size)
    print(f"  occupied pixels: {occupied:,}/{npix:,} "
          f"({100.0 * occupied / npix:.2f}%)  empty: {100.0 * (1 - occupied / npix):.2f}%")

    out = Path(args.out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out, "w") as f:
        for k, v in pix.items():
            f.create_dataset(k, data=np.asarray(v), compression="gzip", shuffle=True)
        f.attrs["nside"] = int(args.nside)
        f.attrs["built_by"] = str(Path(__file__).resolve())
        f.attrs["built_at_utc"] = datetime.now(timezone.utc).isoformat()
        f.attrs["source_complete_catalog"] = str(args.complete_catalog)
        f.attrs["source_attrs_json"] = json.dumps(src_attrs, default=str)
        f.attrs["n_galaxies_in"] = int(n_before)
        f.attrs["n_galaxies_used"] = int(z.size)
        f.attrs["subsample_fraction"] = (float(args.subsample)
                                         if args.subsample is not None else 1.0)
        f.attrs["z_keep_max"] = (float(args.z_keep_max)
                                 if args.z_keep_max is not None else -1.0)
        f.attrs["z_min"] = float(z.min())
        f.attrs["z_max"] = float(z.max())
        f.attrs["occupied_pixels"] = occupied
        f.attrs["empty_pixel_fraction"] = float(1.0 - occupied / npix)
        f.attrs["completeness"] = ("complete (gmd SurveyConfig footprint / "
                                   "magnitude / z_hard_max cuts NOT applied)")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
