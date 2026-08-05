#!/usr/bin/env python3
"""PROBE 1 -- pixelation audit (CPU).

Question: does the NEW vectorised survey builder
(``working/data/generate_dataset.py::pixelate_catalog_vec``) produce the same
survey block as darksirens' own reference implementation
(``scripts/mock_dark_sirens/generate_mock_data.py::_pixelate_catalog``), and
does the block it writes satisfy every invariant darksirens' WINDOWED catalog-KDE
evaluator assumes?

Three parts, all read-only:

A. HEAD-TO-HEAD.  A stride subsample of the seed-100 GAL catalog spanning the
   full z range and hitting every nside-32 pixel is pixelated by BOTH routines
   with identical inputs (nside=32, dz = 3e-3 (1+z) computed and stored exactly
   as the generator does, w = 1).  Compared three ways:
     * ngals, shapes, dtypes;
     * RAW arrays, bitwise (the layout question: gmd emits catalog order inside
       a row, the vec builder emits z-sorted order);
     * arrays after ``darksirens.catalogs.io.sort_survey_rows_by_z`` -- the
       permutation ``load_survey`` applies on EVERY load -- bitwise.  This is
       the comparison that decides whether the likelihood sees the same numbers.

B. THE PRODUCTION SURVEY FILE.  The real seed-100 GAL block is audited against
   what the windowed evaluator assumes: the z-sort invariant as WRITTEN
   (``_rows_sorted_for_windowing``), real galaxies a contiguous prefix of
   length ngals, the padding sentinels (z=100, dz=1, w=0) exactly on the tail,
   dz == 3e-3(1+z) bitwise on the real prefix, and the per-row window
   half-width driver ``sig_eff_row_max = max over ALL columns of sig_eff``
   -- which includes the padded slots.

C. WINDOW SIZING.  ``recommended_kde_window`` at the scan's actual kernel
   (sigma_kde = 0) and at the prior's widest kernel (sigma_kde = 0.05), plus
   the measured number of galaxies inside +/- n_sigma*sigma_eff at a ladder of
   redshifts -- the quantity W=4096 has to dominate.

Writes results/probe1_pixelation_audit.json.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = Path("/hildafs/projects/phy230014p/magana/gws-agn/working/data")
DARKSIRENS = Path("/hildafs/projects/phy230014p/magana/src/darksirens")

DZ_SCALE = 3.0e-3
NSIDE = 32
CAT_DTYPE = "float32"


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--stride", type=int, default=40,
                    help="Catalog stride; 151.2e6/40 ~ 3.8e6 objects, ~307 per "
                         "nside-32 pixel, spanning the whole file (the catalog is "
                         "NOT z-ordered, so a stride spans the full z range).")
    ap.add_argument("--out", default=str(ROOT / "results" / "probe1_pixelation_audit.json"))
    return ap.parse_args(argv)


def load_vec_builder():
    sys.path.insert(0, str(DATA))
    import importlib.util
    spec = importlib.util.spec_from_file_location("gen_ds", DATA / "generate_dataset.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_gmd():
    gmd_dir = DARKSIRENS / "scripts/mock_dark_sirens"
    sys.path.insert(0, str(gmd_dir))
    import generate_mock_data as gmd
    return gmd


def main(argv=None):
    args = parse_args(argv)
    import h5py
    out = {"probe": 1, "name": "pixelation_audit", "seed": args.seed,
           "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    sd = DATA / f"seed{args.seed}"
    cat_path = sd / "catalogs" / "catalog_gal_complete.h5"
    srv_path = sd / "surveys" / "survey_gal_complete_ns32.h5"

    # ------------------------------------------------------------------ #
    # A.  head-to-head on a stride subsample
    # ------------------------------------------------------------------ #
    t0 = time.time()
    with h5py.File(cat_path, "r") as f:
        n_total = int(f["z"].shape[0])
        ra = np.asarray(f["ra"][::args.stride])
        dec = np.asarray(f["dec"][::args.stride])
        z = np.asarray(f["z"][::args.stride])
    print(f"[A] read {z.size:,} of {n_total:,} objects in {time.time()-t0:.1f}s; "
          f"z in [{z.min():.5f}, {z.max():.5f}]")

    # EXACTLY the generator's convention (generate_dataset.py, surveys stage)
    dz = (DZ_SCALE * (1.0 + z)).astype(z.dtype)
    w = np.ones_like(z)

    gen = load_vec_builder()
    gmd = load_gmd()

    t0 = time.time()
    vec = gen.pixelate_catalog_vec(ra, dec, z, dz, w, NSIDE,
                                   dtype=np.dtype(CAT_DTYPE))
    t_vec = time.time() - t0
    t0 = time.time()
    ref = gmd._pixelate_catalog(ra, dec, z, dz, w, NSIDE)
    t_ref = time.time() - t0
    print(f"[A] vec {t_vec:.2f}s   gmd {t_ref:.2f}s   "
          f"({t_ref/max(t_vec,1e-9):.0f}x)")

    A = {"n_objects": int(z.size), "n_catalog_total": n_total,
         "stride": args.stride,
         "z_range": [float(z.min()), float(z.max())],
         "nside": NSIDE, "dz_convention": f"dz = {DZ_SCALE} * (1 + z)",
         "seconds": {"pixelate_catalog_vec": t_vec, "gmd__pixelate_catalog": t_ref},
         "gmd_signature": "(ra, dec, z, dz, w, nside, marks=None) -- dz is the "
                          "SAME per-object array the vec builder takes",
         "shapes": {k: list(np.shape(vec[k])) for k in ("zgals", "dzgals", "wgals", "ngals")},
         "dtypes_vec": {k: str(np.asarray(vec[k]).dtype) for k in vec},
         "dtypes_gmd": {k: str(np.asarray(ref[k]).dtype) for k in ref},
         }
    A["shapes_match"] = all(np.shape(vec[k]) == np.shape(ref[k])
                            for k in ("zgals", "dzgals", "wgals", "ngals"))
    A["ngals_identical"] = bool(np.array_equal(vec["ngals"], ref["ngals"]))
    A["n_occupied_pixels"] = int((np.asarray(vec["ngals"]) > 0).sum())
    A["max_gals"] = int(np.asarray(vec["zgals"]).shape[1])

    # gmd builds float64; the generator asks for float32 and the inputs ARE
    # float32, so the float64->float32 cast is exact.  Cast, then compare.
    def as32(a):
        return np.asarray(a, dtype=np.float32)

    raw = {}
    for k in ("zgals", "dzgals", "wgals"):
        a, b = as32(vec[k]), as32(ref[k])
        d = a != b
        raw[k] = {"bitwise_identical": bool(not d.any()),
                  "n_differing_slots": int(d.sum()),
                  "frac_differing": float(d.mean())}
    A["raw_bitwise"] = raw

    # multiset test: sort every row's REAL prefix and compare
    sys.path.insert(0, str(DARKSIRENS))
    from darksirens.catalogs.io import sort_survey_rows_by_z, _row_z_sort_order
    from darksirens.redshift.catalog import (
        _rows_sorted_for_windowing, recommended_kde_window)

    vz, vdz, vw, vng, _ = sort_survey_rows_by_z(
        as32(vec["zgals"]), as32(vec["dzgals"]), as32(vec["wgals"]), vec["ngals"])
    rz, rdz, rw, rng, _ = sort_survey_rows_by_z(
        as32(ref["zgals"]), as32(ref["dzgals"]), as32(ref["wgals"]), ref["ngals"])
    post = {}
    for k, (a, b) in {"zgals": (vz, rz), "dzgals": (vdz, rdz),
                      "wgals": (vw, rw)}.items():
        d = a != b
        post[k] = {"bitwise_identical": bool(not d.any()),
                   "n_differing_slots": int(d.sum()),
                   "max_abs_diff": float(np.abs(a.astype(np.float64)
                                                - b.astype(np.float64)).max())}
    A["after_load_survey_sort_bitwise"] = post
    A["load_survey_applies_this_sort"] = (
        "darksirens.catalogs.io.load_survey(sort_rows_by_z=True) is the DEFAULT "
        "and the only path darksirens.inference.data uses, so the arrays the "
        "likelihood sees are the post-sort ones.")

    # is the vec output ALREADY in the sorted order (i.e. is the load-time sort a
    # no-op for it)?  and is gmd's not?
    A["vec_rows_sorted_as_written"] = bool(
        _rows_sorted_for_windowing(as32(vec["zgals"]), np.asarray(vec["ngals"])))
    A["gmd_rows_sorted_as_written"] = bool(
        _rows_sorted_for_windowing(as32(ref["zgals"]), np.asarray(ref["ngals"])))
    ov = _row_z_sort_order(as32(vec["zgals"]), np.asarray(vec["ngals"]))
    A["vec_load_sort_is_identity"] = bool(
        np.array_equal(ov, np.broadcast_to(np.arange(ov.shape[1]), ov.shape)))

    out["A_head_to_head"] = A

    # ------------------------------------------------------------------ #
    # B.  the production survey block
    # ------------------------------------------------------------------ #
    t0 = time.time()
    with h5py.File(srv_path, "r") as f:
        Z = np.asarray(f["zgals"])
        DZ = np.asarray(f["dzgals"])
        W = np.asarray(f["wgals"])
        NG = np.asarray(f["ngals"])
        attrs = {k: (v.tolist() if isinstance(v, np.ndarray) else
                     (float(v) if isinstance(v, (np.floating,)) else
                      (int(v) if isinstance(v, (np.integer,)) else v)))
                 for k, v in f.attrs.items() if k != "completeness_json"}
    print(f"[B] loaded production GAL survey {Z.shape} in {time.time()-t0:.1f}s")

    npix, nmax = Z.shape
    cols = np.arange(nmax)[None, :]
    real = cols < NG[:, None]
    pad = ~real

    B = {"path": str(srv_path), "shape": [int(npix), int(nmax)], "attrs": attrs,
         "ngals": {"min": int(NG.min()), "max": int(NG.max()),
                   "mean": float(NG.mean()), "sum": int(NG.sum()),
                   "n_empty_rows": int((NG == 0).sum())}}
    B["rows_sorted_for_windowing_as_written"] = bool(_rows_sorted_for_windowing(Z, NG))
    B["padding_sentinels_exact"] = {
        "zgals_all_100": bool(np.all(Z[pad] == np.float32(100.0))),
        "dzgals_all_1": bool(np.all(DZ[pad] == np.float32(1.0))),
        "wgals_all_0": bool(np.all(W[pad] == np.float32(0.0))),
    }
    B["real_prefix_contiguous"] = {
        "wgals_positive_on_prefix": bool(np.all(W[real] > 0)),
        "n_w_positive_equals_ngals": bool(
            np.array_equal((W > 0).sum(axis=1).astype(np.int64), NG.astype(np.int64))),
    }
    # The generator computes dz in float32 throughout:
    #   dz = (DZ_SCALE * (1.0 + sub["z"])).astype(sub["z"].dtype)
    # with sub["z"] float32, so both operations are float32.  Reproduce that
    # EXACTLY, and separately the float64-then-round route, to show the only
    # difference is one float32 ULP of rounding order.
    z_real = Z[real]
    dz_f32 = (DZ_SCALE * (1.0 + z_real)).astype(z_real.dtype)
    dz_f64 = (DZ_SCALE * (1.0 + z_real.astype(np.float64))).astype(np.float32)
    B["dz_convention_bitwise"] = {
        "identical_to_generator_float32_expression": bool(
            np.array_equal(DZ[real], dz_f32)),
        "identical_to_float64_then_round": bool(np.array_equal(DZ[real], dz_f64)),
        "max_abs_diff_vs_float64_route": float(np.abs(
            DZ[real].astype(np.float64) - dz_f64.astype(np.float64)).max()),
        "float32_ulp_at_typical_dz": float(np.spacing(np.float32(0.006))),
    }
    # the windowed evaluator's half-width driver
    SIGMA_EFF_FLOOR = 1e-6
    sig_all = np.maximum(DZ.astype(np.float64), SIGMA_EFF_FLOOR)     # sigma_kde = 0
    sig_row_max_all = sig_all.max(axis=1)                            # what darksirens uses
    sig_real = np.where(real, sig_all, -np.inf)
    sig_row_max_real = sig_real.max(axis=1)
    B["sig_eff_row_max_at_sigma_kde_0"] = {
        "note": ("darksirens computes sig_eff_row_max = max over ALL columns "
                 "(catalog.py:catalog_kernel_state), and the PADDED slots carry "
                 "dzgals = 1.0, so any row with padding gets half-width "
                 "n_sigma * 1.0 instead of n_sigma * max(real dz)."),
        "including_padding": {"min": float(sig_row_max_all.min()),
                              "max": float(sig_row_max_all.max()),
                              "n_rows_equal_1.0": int((sig_row_max_all == 1.0).sum())},
        "real_slots_only": {"min": float(sig_row_max_real.min()),
                            "max": float(sig_row_max_real.max())},
        "n_rows_without_padding": int((NG == nmax).sum()),
    }
    out["B_production_block"] = B

    # ------------------------------------------------------------------ #
    # C.  window sizing
    # ------------------------------------------------------------------ #
    C = {}
    for nsig in (6.0, 8.0):
        for skmax in (0.0, 0.05):
            t0 = time.time()
            wrec = int(recommended_kde_window(Z, NG, DZ, skmax, n_sigma=nsig))
            C[f"recommended_W_nsigma{nsig:g}_sigma_kde_max{skmax:g}"] = wrec
            print(f"[C] recommended_kde_window(n_sigma={nsig}, "
                  f"sigma_kde_max={skmax}) = {wrec}  ({time.time()-t0:.1f}s)")
    C["W_used_by_the_analysis"] = 4096
    C["W_full_row"] = int(nmax)

    # how many galaxies actually live inside +/- n_sigma sigma_eff at a z ladder,
    # on the densest and a median row -- the quantity the window must dominate.
    dense = int(np.argmax(NG))
    med = int(np.argsort(NG)[npix // 2])
    ladder = [0.05, 0.13, 0.20, 0.30, 0.3565, 0.50, 0.70, 1.00]
    C["galaxies_within_8sigma_of_z"] = {}
    for name, r in (("densest_row", dense), ("median_row", med)):
        n = int(NG[r])
        zr = np.sort(Z[r, :n].astype(np.float64))
        rec = {}
        for zq in ladder:
            half = 8.0 * DZ_SCALE * (1.0 + zq)
            lo, hi = np.searchsorted(zr, [zq - half, zq + half])
            rec[f"z={zq:g}"] = {"half_width": half, "n_in_window": int(hi - lo)}
        C["galaxies_within_8sigma_of_z"][name] = {"row": r, "ngals": n, "counts": rec}
    # and the index span W=4096 actually covers, in z, at those redshifts
    C["z_span_covered_by_W4096"] = {}
    for name, r in (("densest_row", dense), ("median_row", med)):
        n = int(NG[r])
        zr = np.sort(Z[r, :n].astype(np.float64))
        rec = {}
        for zq in ladder:
            i = int(np.searchsorted(zr, zq))
            lo = max(0, min(i - 2048, n - 4096)) if n > 4096 else 0
            hi = min(n, lo + 4096)
            rec[f"z={zq:g}"] = {"z_lo": float(zr[lo]),
                                "z_hi": float(zr[min(hi, n) - 1]),
                                "n_real_in_window": int(hi - lo)}
        C["z_span_covered_by_W4096"][name] = rec
    out["C_window_sizing"] = C

    # ------------------------------------------------------------------ #
    verdict_identical = (
        A["ngals_identical"]
        and all(v["bitwise_identical"] for v in post.values())
    )
    verdict_raw_identical = all(v["bitwise_identical"] for v in raw.values())
    out["verdict"] = {
        "identical_after_the_load_time_sort": bool(verdict_identical),
        "identical_as_written": bool(verdict_raw_identical),
        "summary": (
            "IDENTICAL" if verdict_identical and verdict_raw_identical else
            ("BENIGN LAYOUT DIFFERENCE (within-row ordering only; "
             "load_survey's z-sort removes it bitwise)" if verdict_identical else
             "REAL DISCREPANCY")),
    }
    print("\nVERDICT:", out["verdict"]["summary"])

    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    print(f"Wrote {p}")


if __name__ == "__main__":
    main()
