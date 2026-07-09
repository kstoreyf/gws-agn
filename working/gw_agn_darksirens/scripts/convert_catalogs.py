#!/usr/bin/env python
"""Convert gw_agn's pixelated galaxy/AGN catalogs into darksirens survey files.

Reads ``cat_gal_pixelated_nside64.h5`` / ``cat_agn_pixelated_nside64.h5``
(gw_agn's glass_prod format: attr ``nside``, datasets ``n_in_pixel`` (npix,)
and ``z`` (npix, maxgals) NaN-padded, RING order) and writes the darksirens
survey format consumed by ``darksirens.catalogs.io.load_survey``:
attr ``nside``; datasets ``zgals``/``dzgals``/``wgals`` (npix, maxgals) and
``ngals`` (npix,). Also writes z<=1.0-truncated ``*_zlt1.h5`` variants and a
``catalog_meta.json`` provenance/summary file.

See working/gw_agn_darksirens/RECON.md ("Source data", "Survey files",
"Repos / environment") for the interface contract this script implements.
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

# darksirens KDE bandwidth convention validated in gw_agn scans (sigma_kde=0
# scans hold this fixed per-galaxy smoothing): dz = 3e-3 * (1 + z).
DZ_SCALE = 3e-3
Z_PAD = 100.0
DZ_PAD = 1.0
W_PAD = 0.0
Z_TRUNC = 1.0

DARKSIRENS_REPO = "/hildafs/projects/phy230014p/magana/src/darksirens"
DARKSIRENS_MERGE_BASE = "d387b4f"

REPO_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_DATA_DIR = REPO_ROOT / "working/gw_agn/data/glass_prod"
DEFAULT_OUT_DIR = REPO_ROOT / "working/gw_agn_darksirens/data"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--gal-in",
        type=Path,
        default=DEFAULT_DATA_DIR / "cat_gal_pixelated_nside64.h5",
    )
    p.add_argument(
        "--agn-in",
        type=Path,
        default=DEFAULT_DATA_DIR / "cat_agn_pixelated_nside64.h5",
    )
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--z-trunc", type=float, default=Z_TRUNC)
    p.add_argument("--z-full-cosmo", type=float, default=1.5,
                    help="V_c upper bound (z) used for log10n0_true of the full catalogs.")
    return p.parse_args()


def read_pixelated(path):
    with h5py.File(path, "r") as f:
        nside = int(f.attrs["nside"])
        n_in_pixel = np.asarray(f["n_in_pixel"], dtype=np.int64)
        z = np.asarray(f["z"], dtype=np.float64)
    return nside, n_in_pixel, z


def to_survey_arrays(n_in_pixel, z):
    """Pack (npix, maxgals) NaN-padded z into darksirens zgals/dzgals/wgals."""
    npix, maxgals = z.shape
    zgals = np.full((npix, maxgals), Z_PAD, dtype=np.float64)
    dzgals = np.full((npix, maxgals), DZ_PAD, dtype=np.float64)
    wgals = np.full((npix, maxgals), W_PAD, dtype=np.float64)
    real = np.isfinite(z)
    zgals[real] = z[real]
    dzgals[real] = DZ_SCALE * (1.0 + z[real])
    wgals[real] = 1.0
    return zgals, dzgals, wgals


def truncate_and_repack(n_in_pixel, z, z_max):
    """Drop galaxies with z > z_max, recompute per-pixel packing/ngals/maxgals."""
    npix = z.shape[0]
    keep_mask = np.isfinite(z) & (z <= z_max)
    new_n = keep_mask.sum(axis=1).astype(np.int64)
    new_maxgals = int(new_n.max()) if npix > 0 else 0
    new_maxgals = max(new_maxgals, 1)  # keep datasets non-degenerate
    new_z = np.full((npix, new_maxgals), np.nan, dtype=np.float64)
    for i in range(npix):
        vals = z[i][keep_mask[i]]
        # front-pack, preserve ascending pixel-local order as stored on disk
        new_z[i, : vals.shape[0]] = vals
    assert np.array_equal(new_n, np.isfinite(new_z).sum(axis=1))
    return new_n, new_z


def write_survey_h5(path, nside, n_in_pixel, zgals, dzgals, wgals):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs["nside"] = nside
        f.create_dataset("zgals", data=zgals, dtype=np.float64)
        f.create_dataset("dzgals", data=dzgals, dtype=np.float64)
        f.create_dataset("wgals", data=wgals, dtype=np.float64)
        f.create_dataset("ngals", data=n_in_pixel.astype(np.int64))


def comoving_volume_mpc3(zmax, H0=67.74, Om0=0.3075):
    from astropy.cosmology import FlatLambdaCDM
    import astropy.units as u

    cosmo = FlatLambdaCDM(H0=H0, Om0=Om0)
    return float(cosmo.comoving_volume(zmax).to(u.Mpc ** 3).value)


def git_head(repo):
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception as exc:  # pragma: no cover - provenance best-effort
        return f"<unavailable: {exc}>"


def git_branch(repo):
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "--abbrev-ref", "HEAD"], text=True
        ).strip()
    except Exception as exc:  # pragma: no cover
        return f"<unavailable: {exc}>"


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "gws_agn_repo_head": git_head(REPO_ROOT),
        "gws_agn_repo_branch": git_branch(REPO_ROOT),
        "darksirens_repo": DARKSIRENS_REPO,
        "darksirens_repo_head": git_head(DARKSIRENS_REPO),
        "darksirens_repo_branch": git_branch(DARKSIRENS_REPO),
        "darksirens_merge_base_required": DARKSIRENS_MERGE_BASE,
        "cosmology": {"H0": 67.74, "Om0": 0.3075, "w0": -1.0, "wa": 0.0,
                      "source": "astropy.cosmology.FlatLambdaCDM (Planck15 params)"},
        "z_trunc": args.z_trunc,
        "z_full_cosmo": args.z_full_cosmo,
        "dz_scale": DZ_SCALE,
        "z_pad": Z_PAD,
        "dz_pad": DZ_PAD,
        "w_pad": W_PAD,
        "files": {},
    }

    v_full = comoving_volume_mpc3(args.z_full_cosmo)
    v_trunc = comoving_volume_mpc3(args.z_trunc)
    meta["comoving_volume_mpc3"] = {
        f"z_le_{args.z_full_cosmo}": v_full,
        f"z_le_{args.z_trunc}": v_trunc,
    }

    specs = [
        ("gal", args.gal_in, "gal.h5", "gal_zlt1.h5"),
        ("agn", args.agn_in, "agn.h5", "agn_zlt1.h5"),
    ]

    for label, in_path, out_full_name, out_trunc_name in specs:
        print(f"[{label}] reading {in_path}")
        nside, n_in_pixel, z = read_pixelated(in_path)
        npix, maxgals = z.shape
        n_total = int(n_in_pixel.sum())
        print(f"[{label}] nside={nside} npix={npix} maxgals={maxgals} "
              f"n_total={n_total} empty_pixels={(n_in_pixel == 0).sum()}")

        # --- full catalog ---
        zgals, dzgals, wgals = to_survey_arrays(n_in_pixel, z)
        out_full = args.out_dir / out_full_name
        write_survey_h5(out_full, nside, n_in_pixel, zgals, dzgals, wgals)
        log10n0_full = np.log10(n_total / v_full)
        meta["files"][out_full_name] = {
            "input": str(in_path),
            "n_total": n_total,
            "maxgals": maxgals,
            "nside": nside,
            "empty_pixels": int((n_in_pixel == 0).sum()),
            "z_cut_for_volume": args.z_full_cosmo,
            "comoving_volume_mpc3": v_full,
            "log10n0_true": log10n0_full,
            "log10n0_count_anchored": log10n0_full,
        }
        print(f"[{label}] wrote {out_full} "
              f"({out_full.stat().st_size / 1e6:.2f} MB) "
              f"log10n0_true={log10n0_full:.6f}")

        # --- z<=z_trunc truncated catalog ---
        new_n, new_z = truncate_and_repack(n_in_pixel, z, args.z_trunc)
        new_zgals, new_dzgals, new_wgals = to_survey_arrays(new_n, new_z)
        out_trunc = args.out_dir / out_trunc_name
        write_survey_h5(out_trunc, nside, new_n, new_zgals, new_dzgals, new_wgals)
        n_total_trunc = int(new_n.sum())
        log10n0_trunc = np.log10(n_total_trunc / v_trunc)
        meta["files"][out_trunc_name] = {
            "input": str(in_path),
            "n_total": n_total_trunc,
            "maxgals": int(new_z.shape[1]),
            "nside": nside,
            "empty_pixels": int((new_n == 0).sum()),
            "z_cut_for_volume": args.z_trunc,
            "comoving_volume_mpc3": v_trunc,
            "log10n0_true": log10n0_trunc,
            "log10n0_count_anchored": log10n0_trunc,
            "n_dropped_z_gt_trunc": n_total - n_total_trunc,
        }
        print(f"[{label}] wrote {out_trunc} "
              f"({out_trunc.stat().st_size / 1e6:.2f} MB) "
              f"n_total={n_total_trunc} maxgals={new_z.shape[1]} "
              f"log10n0_true={log10n0_trunc:.6f}")

    meta_path = args.out_dir / "catalog_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, sort_keys=False)
    print(f"wrote {meta_path}")


if __name__ == "__main__":
    sys.exit(main())
