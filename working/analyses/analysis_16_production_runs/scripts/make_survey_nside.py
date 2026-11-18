#!/usr/bin/env python
"""Complete-catalog survey files at another HEALPix nside, made as data/generate_dataset.py
(stage_surveys) made the nside-32 ones: the same pixelation function (pixelate_catalog_vec: rows sorted by
pixel, then by redshift; padding 100 / 1 / 0), the survey's observed redshift z_obs with dz = 0.003 (1 + z_obs)
and unit weights, float64, every dataset gzip + shuffle with h5py's own chunking, and the nside-32 file's
attributes with nside, occupancy, source and build stamp replaced.

    PY=<jax env python>
    $PY scripts/make_survey_nside.py --seed 100 --nside 64            # -> data/seed100/surveys_ns64/survey_{gal,agn}_complete_ns64.h5
    $PY scripts/make_survey_nside.py --seed 100 --nside 32 --tracers agn --out-dir <scratch> --reproduce
        # the nside-32 AGN file rebuilt by this tool and compared with data/seed100/surveys/ dataset by dataset

Checks (written next to each file as <file>.check.json; the build fails if one fails):
  1. the object count equals the nside-32 file's n_hosts and the catalog's length;
  2. ngals of the four nside-64 children of every nside-32 pixel (NESTED parent/child) sum to the nside-32 ngals;
  3. the redshifts (and widths) of the children, merged and sorted, equal the nside-32 pixel's row bit for bit
     (every object sits in the parent of its pixel: only the binning changes);
  4. the occupied fraction against the Poisson expectation 1 - exp(-N/npix) (reported; clustering lowers it).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import prod_common as P  # noqa: E402

sys.path.insert(0, str(P.DATA))
from generate_dataset import CAT_DTYPE, DZ_SCALE, load_catalog, pixelate_catalog_vec  # noqa: E402

DATASETS = ("zgals", "dzgals", "wgals", "ngals")


def build(seed, tracer, nside, out_path):
    import h5py
    import healpy as hp
    ref = P.survey_path(seed, tracer, "complete")          # the nside-32 file (default nside)
    cat_path = P.DATA / f"seed{seed}" / "catalogs" / f"catalog_{tracer}_complete.h5"
    with h5py.File(ref, "r") as f:
        attrs = dict(f.attrs)
    if attrs.get("level") != "complete" or attrs.get("z_column") != "z_obs":
        sys.exit(f"[fatal] {ref}: level {attrs.get('level')}, z_column {attrs.get('z_column')}")
    t0 = time.time()
    c = load_catalog(cat_path, keys=("ra", "dec", "z", "z_obs"))
    zsrv = c["z_obs"]
    dz = (DZ_SCALE * (1.0 + zsrv)).astype(zsrv.dtype)
    w = np.ones_like(zsrv)
    pix = pixelate_catalog_vec(c["ra"], c["dec"], zsrv, dz, w, nside, dtype=np.dtype(CAT_DTYPE))
    ngals = np.asarray(pix["ngals"])
    occupied = int((ngals > 0).sum())
    npix = int(ngals.size)
    n = int(zsrv.size)
    if n != int(attrs["n_hosts"]) or int(ngals.sum()) != n:
        sys.exit(f"[fatal] {tracer}: {n} catalog rows, {int(ngals.sum())} pixelated, nside-32 file {attrs['n_hosts']}")
    new_attrs = dict(attrs)
    new_attrs.update({
        "nside": int(nside), "occupied_pixels": occupied, "empty_pixel_fraction": float(1.0 - occupied / npix),
        "z_min": float(zsrv.min()), "z_max": float(zsrv.max()),
        "z_min_true": float(c["z"].min()), "z_max_true": float(c["z"].max()),
        "source_catalog": str(cat_path), "built_by": str(Path(__file__).resolve()),
        "built_at_utc": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
        "made_like": "generate_dataset.stage_surveys (pixelate_catalog_vec, z_obs, dz = 0.003 (1 + z_obs), w = 1, "
                     "gzip + shuffle)", "nside32_reference": str(ref)})
    del c, zsrv, dz, w
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + ".tmp")
    with h5py.File(tmp, "w") as f:
        for k, v in pix.items():
            f.create_dataset(k, data=np.asarray(v), compression="gzip", shuffle=True)
        for k, v in new_attrs.items():
            f.attrs[k] = v
    tmp.replace(out_path)
    print(f"[{tracer} nside {nside}] {n:,} objects, {occupied}/{npix} pixels occupied, max {int(ngals.max())} per "
          f"pixel; written {out_path} ({out_path.stat().st_size / 1e6:.0f} MB) in {time.time() - t0:.0f}s", flush=True)
    return out_path


def _valid_rows(f, keys=("zgals", "dzgals")):
    """(pixel index per object, {key: values}) of the real entries, rows read pixel by pixel in file order."""
    ng = np.asarray(f["ngals"])
    m = ng.max()
    idx = np.repeat(np.arange(ng.size), ng)
    col = np.arange(idx.size) - np.repeat(np.cumsum(ng) - ng, ng)
    out = {}
    for k in keys:
        a = f[k]
        vals = np.empty(idx.size, dtype=a.dtype)
        step = max(1, int(2e8 // max(m, 1)))          # read in row blocks
        pos = 0
        for r0 in range(0, ng.size, step):
            blk = a[r0:r0 + step, :]
            nb = ng[r0:r0 + step]
            sel = np.arange(blk.shape[1])[None, :] < nb[:, None]
            v = blk[sel]
            vals[pos:pos + v.size] = v
            pos += v.size
        out[k] = vals
    return idx, col, out


def check(new_path, seed, tracer):
    """Checks 1-4 against the nside-32 file."""
    import h5py
    import healpy as hp
    ref = P.survey_path(seed, tracer, "complete")
    with h5py.File(ref, "r") as f32, h5py.File(new_path, "r") as fn:
        ns = int(fn.attrs["nside"])
        ng32, ngn = np.asarray(f32["ngals"]), np.asarray(fn["ngals"])
        res = {"file": str(new_path), "reference": str(ref), "nside": ns, "n_objects": int(ngn.sum()),
               "n_objects_reference": int(ng32.sum()), "count_identical": bool(ngn.sum() == ng32.sum())}
        if ns % 32 or ns < 32:
            raise SystemExit(f"[fatal] check needs nside a multiple of 32, got {ns}")
        ratio = (ns // 32) ** 2
        parent_ring = hp.nest2ring(32, hp.ring2nest(ns, np.arange(ngn.size)) // ratio)
        res["children_sum_equals_parent"] = bool(np.array_equal(np.bincount(parent_ring, weights=ngn,
                                                                            minlength=ng32.size).astype(np.int64),
                                                                ng32.astype(np.int64)))
        pix_n, _, vn = _valid_rows(fn)
        pix_32, _, v32 = _valid_rows(f32)
        par = parent_ring[pix_n]
        order = np.lexsort((vn["zgals"], par))
        res["parent_of_each_object_sorted_equal"] = bool(np.array_equal(par[order], pix_32))
        res["z_per_parent_bitwise_equal"] = bool(np.array_equal(vn["zgals"][order], v32["zgals"]))
        res["dz_per_parent_bitwise_equal"] = bool(np.array_equal(np.sort(vn["dzgals"]), np.sort(v32["dzgals"])))
        n = int(ngn.sum())
        occ_frac = float((ngn > 0).mean())
        pois = float(1 - np.exp(-n / ngn.size))
        res.update(occupied_fraction=occ_frac, poisson_expectation=pois,
                   mean_per_pixel=n / ngn.size, max_per_pixel=int(ngn.max()), max_per_pixel_nside32=int(ng32.max()))
        res["passed"] = all(res[k] for k in ("count_identical", "children_sum_equals_parent",
                                             "parent_of_each_object_sorted_equal", "z_per_parent_bitwise_equal",
                                             "dz_per_parent_bitwise_equal"))
    out = Path(str(new_path) + ".check.json")
    out.write_text(json.dumps(res, indent=1))
    print(f"[{tracer} nside {ns}] checks: {json.dumps(res)}", flush=True)
    return res


def reproduce(new_path, seed, tracer):
    """The rebuilt nside-32 file against the original, dataset by dataset (values and dtypes) and attributes."""
    import h5py
    ref = P.survey_path(seed, tracer, "complete")
    out = {"rebuilt": str(new_path), "original": str(ref), "datasets": {}}
    with h5py.File(ref, "r") as a, h5py.File(new_path, "r") as b:
        for k in DATASETS:
            out["datasets"][k] = bool(a[k].dtype == b[k].dtype and a[k].shape == b[k].shape
                                      and np.array_equal(a[k][()], b[k][()]))
        skip = {"built_at_utc", "built_by", "source_catalog", "made_like", "nside32_reference"}
        out["attributes_equal_except_build_stamp_and_source"] = all(
            str(a.attrs[k]) == str(b.attrs.get(k)) for k in a.attrs if k not in skip)
    out["passed"] = all(out["datasets"].values()) and out["attributes_equal_except_build_stamp_and_source"]
    Path(str(new_path) + ".reproduce.json").write_text(json.dumps(out, indent=1))
    print(f"[{tracer}] nside-32 rebuild vs original: {json.dumps(out)}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--nside", type=int, required=True)
    ap.add_argument("--tracers", default="gal,agn")
    ap.add_argument("--out-dir", default=None, help="default data/seedNNN/surveys_ns<nside>/")
    ap.add_argument("--reproduce", action="store_true", help="nside 32: compare the rebuilt file with the original")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    ok = True
    for t in a.tracers.split(","):
        out_dir = Path(a.out_dir) if a.out_dir else P.DATA / f"seed{a.seed}" / f"surveys_ns{a.nside}"
        out = out_dir / f"survey_{t}_complete_ns{a.nside}.h5"
        if a.nside == 32 and not a.reproduce:
            sys.exit("[fatal] nside 32 is the production file; use --reproduce with a scratch --out-dir")
        if out.exists() and not a.overwrite:
            print(f"{out} exists (--overwrite to rebuild); checking it")
        else:
            build(a.seed, t, a.nside, out)
        if a.reproduce:
            ok &= reproduce(out, a.seed, t)["passed"]
        else:
            ok &= check(out, a.seed, t)["passed"]
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
