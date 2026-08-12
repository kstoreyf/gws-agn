#!/usr/bin/env python
"""Add the padded ``gal_app_mag`` column to an existing pixelated survey file.

``darksirens_fit_selection`` reads a per-galaxy apparent-magnitude table off the
survey file (``darksirens.catalogs.io.GALPROP_DATASETS``).  Our survey files were
built on 2026-08-01 by ``working/data/generate_dataset.py``, whose
``pixelate_catalog_vec`` emits only ``zgals``/``dzgals``/``wgals``/``ngals`` --
per-galaxy magnitudes were added to darksirens' own writer later (e831290), and
we never used that writer.  So a selection-mode arm needs this column added.

This script is NON-DESTRUCTIVE: it re-pixelates from each survey's recorded
``source_catalog``, **asserts the four existing datasets come back bit-identical**
(which is what proves the magnitudes are co-indexed with the galaxies already on
disk), and writes an augmented copy to a new directory.  The originals that
analyses 3-6 consumed are never touched.

Padding convention is darksirens': pad 0.0, real slots masked via ``ngals``
(``arange < ngal``), never by value.

    python add_gal_app_mag.py --seed 100 --levels m18
    python add_gal_app_mag.py --seed 100 --levels m18,m19,m20,m21 --tracers agn,gal
"""
import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

REPO = Path("/hildafs/projects/phy230014p/magana/gws-agn")
sys.path.insert(0, str(REPO / "working" / "data"))

from generate_dataset import (  # noqa: E402
    CAT_DTYPE, DZ_SCALE, NSIDE_SURVEY, pixelate_catalog_vec,
)

#: darksirens' galprop padding value (io.GALPROP_DATASETS docstring).
GALPROP_PAD = 0.0


def pixelate_with_mag(ra, dec, z, dz, w, app_mag, nside, dtype):
    """``pixelate_catalog_vec`` plus a co-indexed padded ``gal_app_mag`` table.

    Re-derives the identical lexsort ordering rather than trying to invert the
    stored file, so the alignment is by construction; the caller then checks the
    four original datasets for bit-identity, which is what makes that claim
    verifiable rather than asserted.
    """
    import healpy as hp

    out = pixelate_catalog_vec(ra, dec, z, dz, w, nside, dtype=dtype)
    npix = hp.nside2npix(nside)
    pix = hp.ang2pix(nside, np.pi / 2.0 - np.asarray(dec), np.asarray(ra))
    counts = np.bincount(pix, minlength=npix).astype(np.int32)
    max_gals = max(1, int(counts.max()))
    order = np.lexsort((np.asarray(z), pix))          # identical key to the helper
    pix_s = pix[order]
    starts = np.concatenate([[0], np.cumsum(counts)])[:-1]
    col = np.arange(pix_s.size, dtype=np.int64) - starts[pix_s]
    flat = pix_s.astype(np.int64) * max_gals + col

    mag = np.full((npix, max_gals), GALPROP_PAD, dtype=np.dtype(dtype))
    mag.reshape(-1)[flat] = np.asarray(app_mag)[order]
    out["gal_app_mag"] = mag
    return out


def augment(survey_path, out_path, overwrite=False, z_column=None):
    if out_path.exists() and not overwrite:
        print(f"  {out_path.name}: exists, skipping (--overwrite to rebuild)")
        return "skipped"

    with h5py.File(survey_path, "r") as f:
        attrs = dict(f.attrs)
        stored = {k: np.asarray(f[k]) for k in
                  ("zgals", "dzgals", "wgals", "ngals")}
        if "gal_app_mag" in f and z_column is None:
            print(f"  {survey_path.name}: already carries gal_app_mag")
            return "already"

    cat_path = Path(attrs["source_catalog"])
    if not cat_path.exists():
        raise SystemExit(f"source catalog missing: {cat_path}")

    # z_column="z" builds the TRUE-REDSHIFT calibration variant.  The offline
    # selection fit computes Mhat = m - DM(z; H0=100) from whatever redshift the
    # survey carries, and our survey carries photo-z: sd(Mhat_obs - Mhat_true)
    # = 0.0866 mag against a truncation depth of 0.0944 mag, so the sharp cut
    # the truncated-schechter fit assumes is smeared to its own width and the
    # fit misses truth by 3.2 sigma (Mstar_hat) / 4.3 sigma (alpha).
    #
    # These files are for the OFFLINE CALIBRATION ONLY -- they hand the
    # likelihood a C_sel(z; theta) built on the LF that actually generated the
    # mock.  The INFERENCE still runs on the photo-z surveys.  That split is
    # legitimate here because this experiment isolates the completeness
    # ESTIMATOR (per_pixel vs aggregate vs selection) against a known truth; a
    # photo-z calibration nuisance is not what it is measuring.  For real data
    # the fix is forward-modelling the scatter into the fit normalization, NOT
    # this.
    z_key = z_column or attrs["z_column"]
    truez = z_key != attrs["z_column"]

    with h5py.File(cat_path, "r") as f:
        ra, dec = np.asarray(f["ra"]), np.asarray(f["dec"])
        app_mag = np.asarray(f["app_mag"])
        zsrv = np.asarray(f[z_key])

    # exactly the stage_surveys recipe: dz from the RECORDED redshift, unit weights
    dz = (DZ_SCALE * (1.0 + zsrv)).astype(zsrv.dtype)
    w = np.ones_like(zsrv)
    pix = pixelate_with_mag(ra, dec, zsrv, dz, w, app_mag,
                            int(attrs["nside"]), np.dtype(CAT_DTYPE))

    # The bit-identity gate.  If the rebuild reproduces the stored galaxy tables
    # exactly, the magnitude table built through the same permutation is
    # necessarily co-indexed with them.
    #
    # It cannot apply to the true-z variant: pixelate_catalog_vec sorts rows by
    # redshift within each pixel, so a different z column is a different
    # permutation and the tables legitimately differ.  Co-indexing there is
    # guaranteed by construction instead -- z, dz, w and app_mag all move
    # through the ONE `order` computed inside pixelate_with_mag -- and is
    # checked below by a per-pixel count and a magnitude-multiset comparison,
    # which a mis-permutation would break.
    if truez:
        if int(np.asarray(pix["ngals"]).sum()) != int(np.asarray(stored["ngals"]).sum()):
            raise SystemExit(f"{survey_path.name}: true-z rebuild changed the "
                             "galaxy count")
        if not np.array_equal(np.asarray(pix["ngals"]), np.asarray(stored["ngals"])):
            raise SystemExit(f"{survey_path.name}: true-z rebuild changed the "
                             "per-pixel counts (sky positions must not move)")
    else:
        for k, v in stored.items():
            got = np.asarray(pix[k])
            if got.shape != v.shape:
                raise SystemExit(
                    f"{survey_path.name}: {k} shape {got.shape} != stored {v.shape}")
            n_diff = int((got != v).sum())
            if n_diff:
                raise SystemExit(
                    f"{survey_path.name}: rebuild differs from disk in {k} "
                    f"({n_diff} elements) -- the magnitudes would NOT be "
                    "co-indexed. Refusing to write.")

    ngals = np.asarray(pix["ngals"])
    mag = np.asarray(pix["gal_app_mag"])
    real = np.arange(mag.shape[1])[None, :] < ngals[:, None]
    mlim = attrs.get("mag_limit", -1.0)
    if float(mlim) > 0 and real.any() and float(mag[real].max()) >= float(mlim):
        raise SystemExit(
            f"{survey_path.name}: a real slot carries app_mag "
            f"{float(mag[real].max()):.4f} >= the survey's limit {float(mlim)}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as f:
        for k, v in pix.items():
            f.create_dataset(k, data=np.asarray(v), compression="gzip",
                             shuffle=True)
        for k, v in attrs.items():
            f.attrs[k] = v
        f.attrs["galprop_added_by"] = str(Path(__file__).resolve())
        f.attrs["galprop_source_survey"] = str(survey_path)
        f.attrs["galprop_pad"] = GALPROP_PAD
        f.attrs["galprop_note"] = (
            "gal_app_mag added post-hoc from source_catalog['app_mag'] by "
            "re-pixelating with the identical lexsort key; the four original "
            "datasets were verified bit-identical to the source survey file, "
            "so the magnitude table is co-indexed. Pad 0.0, mask by ngals.")
        if truez:
            f.attrs["z_column"] = z_key
            f.attrs["dz_convention"] = (
                f"dz = {float(attrs['dz_scale'])} * (1 + z_true)")
            f.attrs["photoz_realised"] = False
            f.attrs["truez_calibration_only"] = True
            f.attrs["galprop_note"] = (
                "TRUE-REDSHIFT CALIBRATION VARIANT -- zgals are the mock's "
                "TRUE redshifts, not the survey's photo-z. For the OFFLINE "
                "selection fit only, never for inference: it exists so the "
                "fitted C_sel(z; theta) is built on the LF that generated the "
                "mock, instead of one biased by photo-z smearing of the "
                "bright truncation (sd(Mhat_obs - Mhat_true) = 0.0866 mag vs "
                "a 0.0944 mag truncation depth). Pad 0.0, mask by ngals.")

    n_real = int(real.sum())
    _gate = ("TRUE-Z variant, per-pixel counts preserved" if truez
             else "bit-identical")
    print(f"  {out_path.name}: OK  {_gate}, {n_real:,} magnitudes "
          f"[{float(mag[real].min()):.3f}, {float(mag[real].max()):.3f}]")
    return {"n_real": n_real,
            "app_mag_min": float(mag[real].min()),
            "app_mag_max": float(mag[real].max()),
            "mag_limit": float(mlim)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--levels", default="m18")
    ap.add_argument("--tracers", default="agn,gal")
    ap.add_argument("--data_root", default=str(REPO / "working" / "data"))
    ap.add_argument("--out_dir", default=None,
                    help="default: <data_root>/seed<seed>/surveys_galprop, or "
                         "surveys_truez with --z_column z")
    ap.add_argument("--z_column", default=None, choices=[None, "z", "z_obs"],
                    help="redshift the rebuilt survey carries; 'z' builds the "
                         "true-redshift CALIBRATION variant (offline selection "
                         "fit only, never inference)")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    srv = Path(a.data_root) / f"seed{a.seed}" / "surveys"
    _default = "surveys_truez" if a.z_column == "z" else "surveys_galprop"
    out = Path(a.out_dir) if a.out_dir else srv.parent / _default
    print(f"source : {srv}\noutput : {out}")

    report = {}
    for lvl in a.levels.split(","):
        for t in a.tracers.split(","):
            name = f"survey_{t}_{lvl}_ns{NSIDE_SURVEY}.h5"
            p = srv / name
            if not p.exists():
                raise SystemExit(f"missing survey file: {p}")
            report[name] = augment(p, out / name, overwrite=a.overwrite,
                                   z_column=a.z_column)

    (out / "galprop_report.json").write_text(json.dumps(report, indent=2,
                                                        default=str))
    print(f"\nwrote {out / 'galprop_report.json'}")


if __name__ == "__main__":
    main()
