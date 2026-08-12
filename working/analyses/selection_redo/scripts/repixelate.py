#!/usr/bin/env python
"""Re-pixelate a survey at a different HEALPix nside, changing ONLY occupancy.

The point of the a7 axis is to move pixel occupancy while holding the catalog,
the completeness, the population and the events fixed -- the one manipulation a
flux-limited ladder cannot make, because there occupancy is completeness times a
per-tracer constant (measured: log10 occ = 0.999 log10 C + 1.08 for AGN).

So the axis is only meaningful if the re-pixelation is provably occupancy-only.
Two gates, adopted from the desi_darksirens_selection team who ran the same
experiment (nside 128 -> 64) on their own chain:

  1. IDENTICAL object totals across nside -- not "close", identical, since the
     same catalog rows are simply being binned differently.
  2. The occupied fraction must match the Poisson expectation 1 - exp(-N/npix).
     (Their gate was "footprint within 2%", which does NOT transfer to a sparse
     tracer: at ~1 AGN per pixel the occupied footprint IS an occupancy
     statistic, so it cannot be an invariant of an occupancy axis. Agreement
     with Poisson proves the change is binning; a deficit would be the real
     coverage loss their gate protects against.)

    python repixelate.py --seed 100 --levels m18 --tracers gal,agn --nside 16 64
"""
import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

REPO = Path("/hildafs/projects/phy230014p/magana/gws-agn")
sys.path.insert(0, str(REPO / "working" / "data"))
from generate_dataset import CAT_DTYPE, DZ_SCALE, pixelate_catalog_vec  # noqa: E402


def build(cat_path, z_column, nside, dz_scale):
    with h5py.File(cat_path, "r") as f:
        ra, dec = np.asarray(f["ra"]), np.asarray(f["dec"])
        app_mag = np.asarray(f["app_mag"])
        zsrv = np.asarray(f[z_column])
    dz = (dz_scale * (1.0 + zsrv)).astype(zsrv.dtype)
    w = np.ones_like(zsrv)
    pix = pixelate_catalog_vec(ra, dec, zsrv, dz, w, nside,
                               dtype=np.dtype(CAT_DTYPE))

    # gal_app_mag through the SAME permutation, so it stays co-indexed
    import healpy as hp
    npix = hp.nside2npix(nside)
    p = hp.ang2pix(nside, np.pi / 2.0 - dec, ra)
    counts = np.bincount(p, minlength=npix).astype(np.int32)
    max_gals = max(1, int(counts.max()))
    order = np.lexsort((zsrv, p))
    starts = np.concatenate([[0], np.cumsum(counts)])[:-1]
    col = np.arange(p.size, dtype=np.int64) - starts[p[order]]
    flat = p[order].astype(np.int64) * max_gals + col
    mag = np.full((npix, max_gals), 0.0, dtype=np.dtype(CAT_DTYPE))
    mag.reshape(-1)[flat] = app_mag[order]
    pix["gal_app_mag"] = mag
    return pix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--levels", default="m18")
    ap.add_argument("--tracers", default="gal,agn")
    ap.add_argument("--nside", nargs="+", type=int, required=True)
    ap.add_argument("--data_root",
                    default=str(REPO / "working" / "data"))
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    import healpy as hp
    root = Path(a.data_root) / f"seed{a.seed}"
    ref_dir = root / "surveys"
    report = {}

    for lvl in a.levels.split(","):
        for t in a.tracers.split(","):
            ref = ref_dir / f"survey_{t}_{lvl}_ns32.h5"
            with h5py.File(ref, "r") as f:
                attrs = dict(f.attrs)
                n_ref = int(attrs["n_hosts"])
                occ_ref = int(attrs["occupied_pixels"])
            area_ref = occ_ref * hp.nside2pixarea(32, degrees=True)

            for ns in a.nside:
                out = root / f"surveys_ns{ns}" / f"survey_{t}_{lvl}_ns{ns}.h5"
                out.parent.mkdir(parents=True, exist_ok=True)
                if out.exists() and not a.overwrite:
                    print(f"  {out.name}: exists, skipping")
                    continue
                pix = build(Path(attrs["source_catalog"]), attrs["z_column"],
                            ns, float(attrs["dz_scale"]))
                ngals = np.asarray(pix["ngals"])
                n_new = int(ngals.sum())
                occ = int((ngals > 0).sum())
                area = occ * hp.nside2pixarea(ns, degrees=True)

                # ---- GATE 1: identical totals -------------------------------
                if n_new != n_ref:
                    raise SystemExit(
                        f"{out.name}: object count {n_new:,} != reference "
                        f"{n_ref:,}. The axis would not be occupancy-only.")
                # ---- GATE 2: the occupied fraction is PURE BINNING ----------
                # The desi team's gate was "footprint within 2% across nside".
                # That is right for a dense tracer but does NOT transfer to a
                # sparse one: at 1.4 AGN per occupied pixel the occupied
                # footprint is not a footprint at all, it IS an occupancy
                # statistic (measured 37788 deg^2 at ns16 vs 19485 at ns32 --
                # a 94% "change" from re-binning the identical 8,274 objects).
                # It therefore cannot be an invariant of an axis whose whole
                # purpose is to change occupancy.
                #
                # The correct invariant for an isotropic full-sky mock: the
                # occupied fraction must match the Poisson expectation
                # 1 - exp(-lambda) for lambda = N/npix. Agreement proves the
                # change is binning; a DEFICIT would mean real coverage loss,
                # which is the thing their gate was protecting against.
                npix_new = hp.nside2npix(ns)
                lam = n_new / npix_new
                occ_frac_pred = 1.0 - np.exp(-lam)
                occ_frac = occ / npix_new
                # clustered fields sit slightly BELOW Poisson (galaxies share
                # pixels); a large deficit is the failure we care about.
                dev = (occ_frac - occ_frac_pred) / occ_frac_pred
                if dev < -0.25:
                    raise SystemExit(
                        f"{out.name}: occupied fraction {occ_frac:.3f} is "
                        f"{100*abs(dev):.0f}% BELOW the Poisson expectation "
                        f"{occ_frac_pred:.3f} -- that is real sky-coverage "
                        "loss, not re-binning. Axis would be confounded.")
                dfrac = abs(area - area_ref) / area_ref

                with h5py.File(out, "w") as f:
                    for k, v in pix.items():
                        f.create_dataset(k, data=np.asarray(v),
                                         compression="gzip", shuffle=True)
                    for k, v in attrs.items():
                        f.attrs[k] = v
                    f.attrs["nside"] = int(ns)
                    f.attrs["n_hosts"] = n_new
                    f.attrs["occupied_pixels"] = occ
                    f.attrs["empty_pixel_fraction"] = float(
                        1.0 - occ / hp.nside2npix(ns))
                    f.attrs["repixelated_from"] = str(ref)
                    f.attrs["repixelation_note"] = (
                        "Occupancy-only variant for the a7 mechanism axis. "
                        "Same catalog rows, same completeness, same events; "
                        "only the pixel area changes. Gated on identical "
                        "object totals and a Poisson-consistent occupied fraction.")
                key = f"{t}_{lvl}_ns{ns}"
                report[key] = {"n_hosts": n_new, "occupied": occ,
                               "occupied_fraction": occ_frac,
                               "poisson_expectation": occ_frac_pred,
                               "per_occupied_pixel": n_new / max(occ, 1),
                               "empty_fraction": 1.0 - occ / hp.nside2npix(ns),
                               "footprint_deg2": area,
                               "footprint_delta_vs_ns32": dfrac}
                print(f"  {out.name}: OK  n={n_new:,} (identical)  "
                      f"occ/pix={n_new/max(occ,1):.2f}  "
                      f"empty={100*(1-occ/hp.nside2npix(ns)):.1f}%  "
                      f"occ_frac {occ_frac:.3f} vs Poisson "
                      f"{occ_frac_pred:.3f}")

    if report:
        p = root / "surveys_repixelation_report.json"
        p.write_text(json.dumps(report, indent=2))
        print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
