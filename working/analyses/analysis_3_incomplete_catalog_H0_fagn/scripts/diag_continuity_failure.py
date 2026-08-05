#!/usr/bin/env python
"""Why the complete-catalog continuity check moved: the completion's C(z) on a
catalog that is, by construction, complete.

The continuity check found that switching the out-of-catalog field term on at the
mock's true densities moves the COMPLETE-catalog result -- `f_AGN` by +0.080
(1.74 of analysis 2's own 68 % half-widths) and `H0` by +0.40 -- even though a
complete catalog has no missing hosts.  The completion's implied completeness is

    C_k(z) = (dN_obs/dz) / (n0_k dV_c/dz (1+z)^delta_k)

and any redshift where the mock's true dN/dz falls below `n0 dV_c/dz` is a
redshift where the model manufactures hosts that do not exist.  This measures
C(z) directly on the complete catalogs, at the adopted anchor, over the range the
events actually occupy, and reports how much spurious missing-host budget it
implies and where it sits.

Two candidate causes are separated:

  RAMP   GLASS' linear shell windows are a partition of unity only on the interior
         plateau (catalog attr `n_comoving_plateau`).  Below the first shell's
         half-width the true dN/dz ramps linearly to zero, so C(z) < 1 there BY
         CONSTRUCTION -- a systematic, one-sided, and present in both tracers.
  SHOT   Poisson noise in dN_obs/dz makes C(z) scatter about 1.  It is two-sided
         and shrinks as the tracer gets denser, so it matters for AGN long before
         GAL.

Run: python -u scripts/diag_continuity_failure.py --seeds 100
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np
from astropy.cosmology import FlatLambdaCDM

DATA_ROOT = os.environ.get(
    "DATA_ROOT", "/hildafs/projects/phy230014p/magana/gws-agn/working/data"
)
H0_FID, OM0_FID = 67.74, 0.3075
ADOPTED = {"gal": -3.0, "agn": -5.0}
CHUNK = 8_000_000


def zhist(path, edges, column="z"):
    counts = np.zeros(edges.size - 1, dtype=np.int64)
    with h5py.File(path, "r") as f:
        d = f[column]
        for i0 in range(0, d.shape[0], CHUNK):
            counts += np.histogram(
                np.asarray(d[i0 : i0 + CHUNK], dtype=float), bins=edges
            )[0]
    return counts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[100])
    ap.add_argument("--nbins", type=int, default=30)
    ap.add_argument("--out_json", default="results/continuity_failure_diag.json")
    args = ap.parse_args()

    cosmo = FlatLambdaCDM(H0=H0_FID, Om0=OM0_FID)
    out = {"_what": __doc__.split("\n")[0], "adopted_log10n0": ADOPTED, "seeds": {}}

    for seed in args.seeds:
        root = Path(DATA_ROOT) / f"seed{seed}"
        meta = json.loads((root / "META.json").read_text())
        zhor = float(meta["stages"]["surveys"]["horizon_z"])
        edges = np.linspace(0.0, zhor, args.nbins + 1)
        zc = 0.5 * (edges[1:] + edges[:-1])
        dz = np.diff(edges)
        dV_dz = cosmo.differential_comoving_volume(zc).value * 4.0 * np.pi
        rec = {"horizon_z": zhor, "tracers": {}}
        print(f"\n=== seed {seed}   horizon z = {zhor:.4f} ===")

        for tracer in ("gal", "agn"):
            cpath = root / "catalogs" / f"catalog_{tracer}_complete.h5"
            with h5py.File(cpath, "r") as f:
                plateau = np.asarray(
                    str(dict(f.attrs)["n_comoving_plateau"]).strip("[] ").split(),
                    dtype=float,
                )
            zlo = float(plateau[0])
            n0 = 10.0 ** ADOPTED[tracer]
            counts = zhist(str(cpath), edges)
            dN_dz = counts / dz
            model = n0 * dV_dz                       # delta = 0, the adopted setting
            C = np.where(model > 0, dN_dz / model, np.nan)

            # spurious missing-host budget: only C < 1 creates hosts
            deficit = np.clip(1.0 - C, 0.0, None) * model * dz   # counts
            n_obs = counts.sum()
            in_ramp = zc < zlo
            v_ramp = (4 * np.pi / 3) * cosmo.comoving_distance(zlo).value ** 3
            v_hor = (4 * np.pi / 3) * cosmo.comoving_distance(zhor).value ** 3

            t = {
                "n0_Mpc3": n0,
                "glass_plateau_starts_at_z": zlo,
                "ramp_volume_fraction_of_horizon": float(v_ramp / v_hor),
                "n_in_horizon": int(n_obs),
                "C_of_z": {
                    "z": zc.tolist(),
                    "C": C.tolist(),
                    "in_ramp": in_ramp.tolist(),
                },
                "C_min": float(np.nanmin(C)),
                "C_mean_in_ramp": float(np.nanmean(C[in_ramp])) if in_ramp.any() else None,
                "C_mean_above_ramp": float(np.nanmean(C[~in_ramp])),
                "C_rms_scatter_above_ramp": float(np.nanstd(C[~in_ramp])),
                "spurious_missing_hosts_total": float(deficit.sum()),
                "spurious_missing_hosts_in_ramp": float(deficit[in_ramp].sum()),
                "spurious_fraction_of_observed": float(deficit.sum() / max(n_obs, 1)),
                "spurious_fraction_from_ramp": float(
                    deficit[in_ramp].sum() / deficit.sum()
                )
                if deficit.sum() > 0
                else None,
                "poisson_frac_error_per_bin_median": float(
                    np.median(1.0 / np.sqrt(np.maximum(counts, 1)))
                ),
            }
            rec["tracers"][tracer] = t
            print(
                f"  {tracer.upper():3s}  N(z<z_hor) = {n_obs:,}   "
                f"plateau starts z = {zlo:.4f} "
                f"({100*t['ramp_volume_fraction_of_horizon']:.2f} % of horizon volume)\n"
                f"        C(z) min = {t['C_min']:.4f}   "
                f"mean in ramp = {t['C_mean_in_ramp']:.4f}   "
                f"mean above ramp = {t['C_mean_above_ramp']:.5f} "
                f"+- {t['C_rms_scatter_above_ramp']:.5f}\n"
                f"        spurious missing hosts = {t['spurious_missing_hosts_total']:,.0f}"
                f"  ({100*t['spurious_fraction_of_observed']:.3f} % of observed),"
                f"  {100*t['spurious_fraction_from_ramp']:.1f} % of it from the ramp\n"
                f"        per-bin Poisson error (median) = "
                f"{100*t['poisson_frac_error_per_bin_median']:.3f} %"
            )
        # --- per-pixel occupancy: the completion is evaluated PER PIXEL, so this
        # --- is the shot noise that actually matters, not the all-sky one above.
        occ = {}
        for tracer in ("gal", "agn"):
            sp = root / "surveys" / f"survey_{tracer}_complete_ns32.h5"
            with h5py.File(sp, "r") as f:
                ng = f["ngals"][:].astype(np.int64)
                zg = f["zgals"]
                per_pix = np.zeros(ng.size, dtype=np.int64)
                step = max(1, 2_000_000 // max(zg.shape[1], 1))
                for i0 in range(0, ng.size, step):
                    i1 = min(i0 + step, ng.size)
                    blk = np.asarray(zg[i0:i1], dtype=float)
                    valid = np.arange(blk.shape[1])[None, :] < ng[i0:i1, None]
                    per_pix[i0:i1] = (valid & (blk < zhor)).sum(axis=1)
            edges = np.arange(0, int(np.percentile(per_pix, 99.5)) + 2)
            if edges.size > 60:
                edges = np.linspace(0, per_pix.max() + 1, 61)
            hist, _ = np.histogram(per_pix, bins=edges)
            occ[tracer] = {
                "npix": int(per_pix.size),
                "total_in_horizon": int(per_pix.sum()),
                "mean_per_pixel": float(per_pix.mean()),
                "median_per_pixel": float(np.median(per_pix)),
                "p10": float(np.percentile(per_pix, 10)),
                "p90": float(np.percentile(per_pix, 90)),
                "empty_pixels_in_horizon": int((per_pix == 0).sum()),
                "frac_empty_in_horizon": float((per_pix == 0).mean()),
                "poisson_frac_err_at_mean": float(1.0 / np.sqrt(max(per_pix.mean(), 1e-9))),
                "hist_edges": edges.tolist(),
                "hist_counts": hist.tolist(),
            }
            print(
                f"  {tracer.upper():3s} in-horizon hosts per nside-32 pixel: "
                f"mean {occ[tracer]['mean_per_pixel']:.1f}  "
                f"median {occ[tracer]['median_per_pixel']:.0f}  "
                f"[p10 {occ[tracer]['p10']:.0f}, p90 {occ[tracer]['p90']:.0f}]  "
                f"empty {100*occ[tracer]['frac_empty_in_horizon']:.2f} %  "
                f"Poisson err at mean {100*occ[tracer]['poisson_frac_err_at_mean']:.1f} %"
            )
        rec["per_pixel_occupancy_in_horizon"] = occ
        out["seeds"][str(seed)] = rec

    Path(args.out_json).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out_json}")


if __name__ == "__main__":
    main()
