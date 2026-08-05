#!/usr/bin/env python3
"""Rebuild seed 100's COMPLETE survey blocks at other HEALPix resolutions.

Step 4 of the closure campaign: if the sky oracle confirms that what survives the
(b2)+(c2) generator fixes is the nside-32 PIXELISATION -- sky and redshift modelled
as independent inside a 1.83 deg pixel -- then the residual must fall as the pixel
shrinks.  This script builds the same catalog at nside 64 and 128 so that curve can
be measured against the oracle's prediction.

It is deliberately a SEPARATE script, not a change to ``generate_dataset.py``: the
record surveys stay exactly as they are (``NSIDE_SURVEY = 32``), these files are
written under a different name, and nothing in ``working/data/seed100/surveys``
is touched.  The pixelation itself is ``generate_dataset.pixelate_catalog_vec``
verbatim, and the ``dz = 3e-3 (1+z)`` convention, the padding sentinels, the float64
storage dtype and every attribute are the generator's own.

``recommended_kde_window`` is re-measured per resolution and written to the output
attributes, because the window requirement scales with the galaxies per pixel and
the record's W = 4096 is sized for nside 32.

Usage:
    python scripts/build_nside_surveys.py --nside 64 128 --tracer gal agn
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np

DATA = Path("/hildafs/projects/phy230014p/magana/gws-agn/working/data")
OUT = Path("/hildafs/projects/phy220048p/magana/gws-agn-data/derived/"
           "analysis_1_complete_catalog_H0/surveys_nside")
DARKSIRENS = Path(os.environ.get(
    "DARKSIRENS_SRC", "/hildafs/projects/phy230014p/magana/src/darksirens"))
sys.path.insert(0, str(DATA))


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--nside", type=int, nargs="+", default=[64, 128])
    ap.add_argument("--tracer", nargs="+", default=["gal", "agn"])
    ap.add_argument("--outdir", default=str(OUT))
    ap.add_argument("--n_sigma", type=float, default=8.0)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--out_json", default=None)
    return ap.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    import generate_dataset as G
    sys.path.insert(0, str(DARKSIRENS))
    from darksirens.redshift.catalog import recommended_kde_window

    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rec = {"generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "seed": a.seed, "built_by": str(Path(__file__).resolve()),
           "pixelation": "generate_dataset.pixelate_catalog_vec (verbatim)",
           "dz_convention": f"dz = {G.DZ_SCALE} * (1 + z)",
           "dtype": G.CAT_DTYPE, "record_nside": G.NSIDE_SURVEY,
           "note": "the record surveys at nside 32 are NOT replaced; these are the "
                   "design measurement for the pixelisation study",
           "surveys": {}}

    for t in a.tracer:
        cat = DATA / f"seed{a.seed}" / "catalogs" / f"catalog_{t}_complete.h5"
        print(f"[nside] loading {cat.name}", flush=True)
        c = G.load_catalog(cat, keys=("ra", "dec", "z"))
        dz = (G.DZ_SCALE * (1.0 + c["z"])).astype(c["z"].dtype)
        w = np.ones_like(c["z"])
        for ns in a.nside:
            sp = outdir / f"survey_{t}_complete_ns{ns}.h5"
            key = f"{t}_ns{ns}"
            if sp.exists() and not a.overwrite:
                print(f"[nside] {sp.name} exists; reusing")
            else:
                t0 = time.time()
                pix = G.pixelate_catalog_vec(c["ra"], c["dec"], c["z"], dz, w, ns,
                                             dtype=np.dtype(G.CAT_DTYPE))
                ng = np.asarray(pix["ngals"])
                occ = int((ng > 0).sum())
                npix = int(ng.size)
                with h5py.File(sp, "w") as f:
                    for kk, vv in pix.items():
                        f.create_dataset(kk, data=np.asarray(vv),
                                         compression="gzip", shuffle=True)
                    f.attrs["nside"] = int(ns)
                    f.attrs["tracer"] = t
                    f.attrs["level"] = "complete"
                    f.attrs["mag_limit"] = -1.0
                    f.attrs["n_hosts"] = int(c["z"].size)
                    f.attrs["z_min"] = float(c["z"].min())
                    f.attrs["z_max"] = float(c["z"].max())
                    f.attrs["dz_scale"] = float(G.DZ_SCALE)
                    f.attrs["dz_convention"] = f"dz = {G.DZ_SCALE} * (1 + z)"
                    f.attrs["occupied_pixels"] = occ
                    f.attrs["empty_pixel_fraction"] = float(1.0 - occ / npix)
                    f.attrs["source_catalog"] = str(cat)
                    f.attrs["built_by"] = str(Path(__file__).resolve())
                    f.attrs["built_at_utc"] = time.strftime(
                        "%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                    f.attrs["purpose"] = ("pixelisation study; the record surveys "
                                          "remain at nside 32")
                print(f"[nside] wrote {sp} ({time.time()-t0:.0f}s, "
                      f"{sp.stat().st_size/1e9:.2f} GB, max {int(ng.max())}/pixel, "
                      f"{100*(1-occ/npix):.2f}% empty)", flush=True)
            with h5py.File(sp, "r") as f:
                Z = np.asarray(f["zgals"][:]); DZ = np.asarray(f["dzgals"][:])
                NG = np.asarray(f["ngals"][:]).astype(np.int64)
            t0 = time.time()
            need = int(recommended_kde_window(Z, NG, DZ, 0.0, n_sigma=a.n_sigma))
            W = 1
            while W < need:
                W *= 2
            print(f"[nside] {sp.name}: recommended_kde_window(n_sigma={a.n_sigma:g}, "
                  f"sigma_kde_max=0) = {need} -> W = {W}  ({time.time()-t0:.0f}s)",
                  flush=True)
            rec["surveys"][key] = {
                "path": str(sp), "nside": ns, "tracer": t,
                "block_shape": [int(Z.shape[0]), int(Z.shape[1])],
                "max_gals_per_pixel": int(NG.max()),
                "occupied_pixels": int((NG > 0).sum()),
                "empty_pixel_fraction": float(1.0 - (NG > 0).sum() / NG.size),
                "pixel_area_deg2": float(4 * np.pi / NG.size * (180 / np.pi) ** 2),
                "pixel_size_deg": float(np.sqrt(4 * np.pi / NG.size)
                                        * 180 / np.pi),
                "recommended_kde_window": need,
                "kde_window_power_of_two": W,
                "size_bytes": sp.stat().st_size}
        del c, dz, w

    out_json = Path(a.out_json) if a.out_json else (
        Path(__file__).resolve().parent.parent / "results" / "surveys_nside.json")
    out_json.write_text(json.dumps(rec, indent=2))
    print(f"wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
