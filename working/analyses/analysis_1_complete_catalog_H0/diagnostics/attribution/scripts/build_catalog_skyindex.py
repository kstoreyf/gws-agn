#!/usr/bin/env python3
"""Spatial index over the complete catalog, in the SURVEY BLOCK's own row order.

The survey files the likelihood conditions on carry only ``(zgals, dzgals, wgals,
ngals)`` per HEALPix row -- the galaxies' sky positions are thrown away by the
pixelation.  The exact host-galaxy oracle (``attr_sky_oracle.py``) needs them back,
attached to the SAME rows and columns darksirens' state arrays are indexed by, so
that each galaxy's ``kw`` weight can be paired with its own ``(ra, dec)``.

``generate_dataset.pixelate_catalog_vec`` fills the block with
``order = np.lexsort((z, pix))``, i.e. by pixel then by redshift, and writes column
``c`` of row ``p`` from the ``c``-th such galaxy.  This script reproduces exactly that
order and stores the catalog's ``(ra, dec, z)`` in it, flattened, with per-pixel
offsets.  Galaxy ``(row p, column c)`` of the survey block is therefore

    ra_s[starts[p] + c],  dec_s[starts[p] + c],  z_s[starts[p] + c]

and the stored ``z_s`` lets that identity be verified BITWISE against the survey file
itself, which the oracle does on every run.

Output (one file per tracer, on the bulk allocation):
    ra_s, dec_s, z_s  float64 (n_gal,)      catalog order = lexsort((z, pix))
    starts            int64   (npix + 1,)   cumulative counts
    counts            int32   (npix,)       == the survey's ``ngals``

Cost: ~4 GB and a few minutes for the 1.5e8-row GAL catalog; seconds for AGN.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import h5py
import numpy as np

DATA = Path("/hildafs/projects/phy230014p/magana/gws-agn/working/data")
BULK = Path("/hildafs/projects/phy220048p/magana/gws-agn-data/derived/"
            "analysis_1_complete_catalog_H0/skyindex")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--tracer", choices=["gal", "agn", "both"], default="both")
    ap.add_argument("--nside", type=int, default=32)
    ap.add_argument("--outdir", default=str(BULK))
    ap.add_argument("--chunk", type=int, default=20_000_000)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dataroot", default=str(DATA),
                    help="Root holding seed<N>/ (default: working/data).")
    ap.add_argument("--z_column", default="auto", choices=("auto", "z", "z_obs"),
                    help="Which catalog redshift the SURVEY block was pixelated on. "
                         "'auto' reads the block's own z_column attr (v3/D3 blocks "
                         "carry z_obs; pre-2026-08-01 blocks carry z).")
    return ap.parse_args(argv)


def build(seed, tracer, nside, outdir, chunk, overwrite, dataroot=DATA,
          z_column="auto"):
    import healpy as hp
    out = Path(outdir) / f"seed{seed}_{tracer}_ns{nside}.h5"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and not overwrite:
        print(f"[skyindex] {out} exists; use --overwrite")
        return out
    root = Path(dataroot)
    cat = root / f"seed{seed}" / "catalogs" / f"catalog_{tracer}_complete.h5"
    srv = root / f"seed{seed}" / "surveys" / f"survey_{tracer}_complete_ns{nside}.h5"
    t0 = time.time()
    # The block's OWN sort key: pixelate_catalog_vec lexsorts on the redshift it was
    # handed, which since the D3 fix is the catalog's photo-z z_obs, not z_true.
    if z_column == "auto":
        with h5py.File(srv, "r") as f:
            z_column = str(f.attrs.get("z_column", "z"))
    print(f"[skyindex] the survey block's row order is lexsort(({z_column}, pix))")
    with h5py.File(cat, "r") as f:
        if z_column not in f:
            raise SystemExit(f"[fatal] catalog has no column {z_column!r}")
        n = int(f["z"].shape[0])
        ra = np.empty(n); dec = np.empty(n); z = np.empty(n); z_true = np.empty(n)
        for j0 in range(0, n, chunk):
            j1 = min(j0 + chunk, n)
            ra[j0:j1] = f["ra"][j0:j1]
            dec[j0:j1] = f["dec"][j0:j1]
            z[j0:j1] = f[z_column][j0:j1]
            z_true[j0:j1] = f["z"][j0:j1]
            print(f"  read {j1:,}/{n:,}  ({time.time()-t0:.0f}s)", flush=True)
    npix = hp.nside2npix(nside)
    pix = hp.ang2pix(nside, np.pi / 2.0 - dec, ra)
    counts = np.bincount(pix, minlength=npix).astype(np.int32)
    print(f"[skyindex] {tracer}: {n:,} rows, {int((counts>0).sum())}/{npix} occupied, "
          f"max {counts.max()}  ({time.time()-t0:.0f}s)", flush=True)
    order = np.lexsort((z, pix))                  # EXACTLY pixelate_catalog_vec's key
    starts = np.concatenate([[0], np.cumsum(counts.astype(np.int64))])
    ra_s, dec_s, z_s = ra[order], dec[order], z[order]
    zt_s = z_true[order]
    del ra, dec, z, z_true, order, pix

    # --- verify against the survey block the likelihood actually reads ---------
    with h5py.File(srv, "r") as f:
        ngals = f["ngals"][:]
        bad, checked = 0, 0
        rows = np.flatnonzero(ngals > 0)
        rng = np.random.default_rng(0)
        probe = rng.choice(rows, size=min(400, rows.size), replace=False)
        for p in probe:
            m = int(ngals[p])
            zz = f["zgals"][p, :m]
            if not np.array_equal(zz, z_s[starts[p]:starts[p] + m]):
                bad += 1
            checked += 1
    ok_counts = bool(np.array_equal(counts, np.asarray(ngals)))
    print(f"[skyindex] survey cross-check: counts identical={ok_counts}  "
          f"z rows bitwise identical on {checked-bad}/{checked} probed rows")
    if bad or not ok_counts:
        raise SystemExit("[fatal] skyindex does not reproduce the survey block")
    # positions must land back in their own pixel
    p_re = hp.ang2pix(nside, np.pi / 2.0 - dec_s, ra_s)
    p_ex = np.repeat(np.arange(npix), counts.astype(np.int64))
    if not np.array_equal(p_re, p_ex):
        raise SystemExit("[fatal] stored positions do not reproduce their pixel")
    print("[skyindex] ang2pix(stored positions) reproduces the pixel assignment")

    with h5py.File(out, "w") as f:
        f.attrs["tracer"] = tracer
        f.attrs["seed"] = seed
        f.attrs["nside"] = nside
        f.attrs["n_gal"] = n
        f.attrs["source_catalog"] = str(cat)
        f.attrs["survey_cross_checked"] = str(srv)
        f.attrs["order"] = (f"np.lexsort(({z_column}, pix)) -- the survey block's "
                            "own row order")
        f.attrs["z_column"] = z_column
        f.create_dataset("ra_s", data=ra_s)
        f.create_dataset("dec_s", data=dec_s)
        f.create_dataset("z_s", data=z_s)
        f.create_dataset("z_true_s", data=zt_s)
        f.create_dataset("starts", data=starts)
        f.create_dataset("counts", data=counts)
    print(f"[skyindex] wrote {out} ({out.stat().st_size/1e9:.2f} GB) "
          f"in {time.time()-t0:.0f}s")
    return out


def main(argv=None):
    a = parse_args(argv)
    tr = ("gal", "agn") if a.tracer == "both" else (a.tracer,)
    for t in tr:
        build(a.seed, t, a.nside, a.outdir, a.chunk, a.overwrite,
              dataroot=a.dataroot, z_column=a.z_column)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
