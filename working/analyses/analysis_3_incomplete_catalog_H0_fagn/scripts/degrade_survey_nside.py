#!/usr/bin/env python
"""Degrade a survey block from nside 32 to a coarser nside, EXACTLY.

The confirmatory test for the continuity finding needs the same hosts binned into
coarser pixels, not a different catalog.  A HEALPix pixel at nside 32 lies wholly
inside exactly one pixel at any coarser nside, so the degrade is a regrouping of
rows with no recomputation and no approximation: every host keeps its own z, dz and
weight, and only which pixel it is filed under changes.

This is why it must be done from the survey blocks rather than by re-pixelating the
catalogs: the blocks carry the REALISED photo-z (`zgals` ARE `z_obs`), and
re-pixelating from `catalog_*.h5` would redraw nothing but would re-read 5.7 GB and
risk a convention drift against the signed-off dataset.

Conventions reproduced from working/data/generate_dataset.py `pixelate_catalog_vec`:
  * pixel index from `hp.ang2pix(nside, theta, phi)` -- RING ordering (healpy's
    default, and what the generator used)
  * padding sentinels beyond `ngals`: zgals 100.0, dzgals 1.0, wgals 0.0
  * rows SORTED IN z within each pixel's real prefix -- darksirens' windowed
    catalog-KDE evaluator requires this
    (`darksirens.redshift.catalog._rows_sorted_for_windowing`)

Verifies, before writing: total host count preserved, per-parent counts equal the
sum of their children's, and the multiset of (z, dz, w) rows is unchanged.

Run: python -u scripts/degrade_survey_nside.py --seed 100 --nside_out 16 \
         --tracers gal agn --level complete
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np

DATA_ROOT = os.environ.get(
    "DATA_ROOT", "/hildafs/projects/phy230014p/magana/gws-agn/working/data"
)
Z_PAD, DZ_PAD, W_PAD = 100.0, 1.0, 0.0


def degrade(src: Path, nside_out: int, out: Path) -> dict:
    import healpy as hp

    with h5py.File(src, "r") as f:
        zg = np.asarray(f["zgals"][:], dtype=np.float64)
        dg = np.asarray(f["dzgals"][:], dtype=np.float64)
        wg = np.asarray(f["wgals"][:], dtype=np.float64)
        ng = np.asarray(f["ngals"][:], dtype=np.int64)
        src_attrs = {k: v for k, v in f.attrs.items()}
    npix_in, maxcol = zg.shape
    nside_in = hp.npix2nside(npix_in)
    if nside_out >= nside_in:
        raise SystemExit(f"nside_out {nside_out} must be coarser than {nside_in}")

    # flatten to the real rows only, tagged by their parent pixel
    valid = np.arange(maxcol)[None, :] < ng[:, None]
    pix_in = np.repeat(np.arange(npix_in), ng)
    z = zg[valid]
    dz = dg[valid]
    w = wg[valid]
    assert z.size == ng.sum()

    # RING parent: exact, since a fine pixel is wholly inside one coarse pixel
    parent_of = hp.ang2pix(nside_out, *hp.pix2ang(nside_in, np.arange(npix_in)))
    pix_out = parent_of[pix_in]

    npix_out = hp.nside2npix(nside_out)
    counts = np.bincount(pix_out, minlength=npix_out).astype(np.int32)
    max_gals = max(1, int(counts.max()))

    # by pixel, then by z -- the sort the windowed KDE requires
    order = np.lexsort((z, pix_out))
    pix_s = pix_out[order]
    starts = np.concatenate([[0], np.cumsum(counts)])[:-1]
    col = np.arange(pix_s.size, dtype=np.int64) - starts[pix_s]
    flat = pix_s.astype(np.int64) * max_gals + col

    Z = np.full((npix_out, max_gals), Z_PAD)
    D = np.full((npix_out, max_gals), DZ_PAD)
    W = np.full((npix_out, max_gals), W_PAD)
    Z.reshape(-1)[flat] = z[order]
    D.reshape(-1)[flat] = dz[order]
    W.reshape(-1)[flat] = w[order]

    # --- verification ---
    v_out = np.arange(max_gals)[None, :] < counts[:, None]
    assert counts.sum() == ng.sum(), "host count changed"
    assert np.allclose(np.sort(Z[v_out]), np.sort(z)), "z multiset changed"
    assert np.allclose(np.sort(D[v_out]), np.sort(dz)), "dz multiset changed"
    assert np.allclose(np.sort(W[v_out]), np.sort(w)), "w multiset changed"
    # per-parent counts must equal the sum of their children's
    expect = np.bincount(parent_of, weights=ng, minlength=npix_out)
    assert np.array_equal(counts.astype(np.int64), expect.astype(np.int64)), \
        "per-pixel regrouping is wrong"
    # rows sorted in z within each real prefix
    bad = 0
    for p in np.flatnonzero(counts > 1)[:2000]:
        if np.any(np.diff(Z[p, : counts[p]]) < 0):
            bad += 1
    assert bad == 0, "rows not sorted in z"

    out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out, "w") as f:
        f.create_dataset("zgals", data=Z)
        f.create_dataset("dzgals", data=D)
        f.create_dataset("wgals", data=W)
        f.create_dataset("ngals", data=counts)
        for k, v in src_attrs.items():
            f.attrs[k] = v
        # dtype matters: darksirens reads this attr straight into
        # hp.nside2npix(), which needs an integer, not a string.  The source
        # files store it as int64, so the copy above must be overwritten with an
        # int of the same kind rather than with str(nside_out).
        f.attrs["nside"] = np.int64(nside_out)
        # these describe the PIXELISATION, so the copied nside-32 values are stale
        f.attrs["occupied_pixels"] = np.int64((counts > 0).sum())
        f.attrs["empty_pixel_fraction"] = float(1.0 - (counts > 0).mean())
        f.attrs["degraded_from"] = str(src)
        f.attrs["degraded_from_nside"] = np.int64(nside_in)
        f.attrs["degraded_by"] = (
            "analysis_3/scripts/degrade_survey_nside.py -- exact regrouping of the "
            "SAME hosts into coarser HEALPix pixels (RING); no host added, removed "
            "or altered"
        )
    occ = int((counts > 0).sum())
    stats = {
        "src": str(src),
        "out": str(out),
        "nside_in": int(nside_in),
        "nside_out": int(nside_out),
        "n_hosts": int(counts.sum()),
        "npix": int(npix_out),
        "max_gals": int(max_gals),
        "occupied_pixels": occ,
        "empty_pixel_fraction": float(1.0 - occ / npix_out),
        "mean_hosts_per_pixel": float(counts.mean()),
    }
    print(
        f"  {src.name} -> {out.name}: nside {nside_in} -> {nside_out}, "
        f"{stats['n_hosts']:,} hosts, block ({npix_out}, {max_gals}), "
        f"empty {100*stats['empty_pixel_fraction']:.2f} %"
    )
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--nside_out", type=int, default=16)
    ap.add_argument("--tracers", nargs="+", default=["gal", "agn"])
    ap.add_argument("--level", default="complete")
    ap.add_argument("--outdir", default="data_derived")
    ap.add_argument("--out_json", default="results/degrade_nside.json")
    a = ap.parse_args()

    src_dir = Path(DATA_ROOT) / f"seed{a.seed}" / "surveys"
    out = {"seed": a.seed, "level": a.level, "nside_out": a.nside_out, "tracers": {}}
    print(f"degrading seed {a.seed} {a.level} to nside {a.nside_out}")
    for t in a.tracers:
        src = src_dir / f"survey_{t}_{a.level}_ns32.h5"
        dst = Path(a.outdir) / f"survey_{t}_{a.level}_ns{a.nside_out}.h5"
        out["tracers"][t] = degrade(src, a.nside_out, dst)
    Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out_json).write_text(json.dumps(out, indent=2))
    print(f"wrote {a.out_json}")


if __name__ == "__main__":
    main()
