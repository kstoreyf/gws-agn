#!/usr/bin/env python3
"""Build a TWO-TRACER deep mock from darksirens' own generator components.

`generate_mock_data` produces one host catalog, so a K=2 multitracer mock has to be
assembled from its parts. This does that without reimplementing any of the physics:
the host draw, the detection statistic, the population samplers and the posterior
construction are all gmd's own functions, so the mock remains the inference's model by
construction (and inherits PR #332's corrected posterior samples).

Construction
------------
* GAL tracer = the complete deep catalog (dense; every pixel occupied).
* AGN tracer = a random subset of it (sparse). AGN living inside galaxies is the
  physically sensible nesting, and sparsity is what supplies the number-density
  contrast that identifies the AGN-hosted fraction.
* Events are drawn PER TRACER with `gmd._draw_events_until_detected` — same grids,
  same population, same `snr_threshold` — then concatenated in `gal_then_agn` order
  with a `host_type` label, matching the layout the campaign's K=2 scans expect.
* Posterior samples come from `gmd._posterior_samples(..., pe_centering="observed")`.

SCOPE LIMIT worth stating plainly: gmd's catalog is UNCLUSTERED (uniform sky, uniform
in comoving volume). So in this mock the AGN-hosted fraction is identified purely by
the sparsity / number-density contrast, with no clustering-bias contrast of the kind
the GLASS mock carries (b = 1.2 vs 2.0). The two mocks therefore probe different parts
of the same channel and are complementary, not redundant.

The injection set is NOT rebuilt: it is a Monte-Carlo campaign for the same detection
rule and population, independent of which hosts the events landed on, so an existing
set generated with the same `--snr-threshold` and population is reused.
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
                    help="Deep complete catalog from gmd (ra, dec, z, app_mag).")
    ap.add_argument("--gmd_dir", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--n_agn", type=int, default=12000,
                    help="Size of the sparse AGN tracer (subset of the galaxies).")
    ap.add_argument("--nobs", type=int, default=200)
    ap.add_argument("--f_agn", type=float, default=0.30,
                    help="Planted AGN-hosted fraction of the DETECTED events.")
    ap.add_argument("--nsamp", type=int, default=2000)
    ap.add_argument("--snr_threshold", type=float, default=8.0)
    ap.add_argument("--dL_fractional_uncertainty", type=float, default=0.10)
    ap.add_argument("--gamma", type=float, default=0.0)
    ap.add_argument("--H0", type=float, default=67.74)
    ap.add_argument("--Om0", type=float, default=0.3075)
    ap.add_argument("--zmax", type=float, default=2.0)
    ap.add_argument("--nside", type=int, default=32)
    ap.add_argument("--z_error", type=float, default=0.003,
                    help="Catalog redshift uncertainty; must be matched to the PE "
                         "resolution (n*dz/sigma_z >~ 100), not set to a "
                         "spectroscopic value.")
    ap.add_argument("--seed", type=int, default=7301)
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    sys.path.insert(0, str(Path(args.gmd_dir).resolve()))
    import generate_mock_data as gmd  # noqa: E402

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    cosmo = gmd._build_cosmology(args.H0, args.Om0, -1.0, 0.0)
    grids = gmd._cosmology_grids(cosmo, args.zmax)
    pop = gmd.PopulationConfig(gamma=args.gamma)

    with h5py.File(args.complete_catalog, "r") as f:
        gal = {k: np.asarray(f[k][:], dtype=float) for k in ("ra", "dec", "z")}
    n_gal_hosts = gal["z"].size
    if args.n_agn > n_gal_hosts:
        raise SystemExit("[fatal] --n_agn exceeds the catalog size")
    agn_idx = rng.choice(n_gal_hosts, size=args.n_agn, replace=False)
    agn = {k: v[agn_idx] for k, v in gal.items()}
    print(f"tracers: GAL {n_gal_hosts:,} hosts | AGN {args.n_agn:,} hosts "
          f"(nested subset)")

    n_agn_ev = int(round(args.f_agn * args.nobs))
    n_gal_ev = args.nobs - n_agn_ev
    truth_f = n_agn_ev / args.nobs
    print(f"events: {n_gal_ev} GAL-hosted + {n_agn_ev} AGN-hosted = {args.nobs} "
          f"(planted f_AGN = {truth_f:.4f})")

    # Per-tracer host draws: gmd's own rejection loop, so detection, rate weighting
    # and the mass/spin population are exactly the inference's model.
    parts = []
    for name, cat, n_ev in (("gal", gal, n_gal_ev), ("agn", agn, n_agn_ev)):
        t = gmd._draw_events_until_detected(rng, n_ev, cat, grids, pop,
                                           args.snr_threshold)
        print(f"  {name}: {len(t['z'])} events, z in "
              f"[{t['z'].min():.4f}, {t['z'].max():.4f}], "
              f"snr in [{t['snr'].min():.1f}, {t['snr'].max():.1f}]")
        parts.append(t)
    truth = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    host_type = np.concatenate([np.zeros(n_gal_ev, dtype=np.int8),
                                np.ones(n_agn_ev, dtype=np.int8)])

    post, obs = gmd._posterior_samples(
        rng, truth, args.nsamp,
        dL_fractional_uncertainty=args.dL_fractional_uncertainty,
        pe_centering="observed")
    z_pe = np.interp(post["dL"], grids["dl"], grids["z"])
    post["m1src"] = post["m1det"] / (1.0 + z_pe)
    post["m2src"] = post["m2det"] / (1.0 + z_pe)

    gw_path = out / "twotracer_gw_events.h5"
    with h5py.File(gw_path, "w") as f:
        f.attrs["format_version"] = "gwcat-1.0"
        f.attrs["mock_data"] = True
        f.attrs["nobs"] = int(args.nobs)
        f.attrs["nsamp"] = int(args.nsamp)
        f.attrs["pe_cosmology_H0"] = float(args.H0)
        f.attrs["pe_cosmology_Om0"] = float(args.Om0)
        f.attrs["chi_eff_in_p_pe"] = True
        f.attrs["chi_eff_amax"] = 0.99
        f.attrs["pe_centering"] = "observed"
        f.attrs["host_order"] = "gal_then_agn"
        f.attrs["n_host_gal"] = int(n_gal_ev)
        f.attrs["n_host_agn"] = int(n_agn_ev)
        f.attrs["truth_f_agn"] = float(truth_f)
        f.attrs["snr_threshold"] = float(args.snr_threshold)
        f.attrs["gamma"] = float(args.gamma)
        f.attrs["dL_fractional_uncertainty"] = float(args.dL_fractional_uncertainty)
        f.attrs["source_complete_catalog"] = str(args.complete_catalog)
        f.attrs["n_agn_tracer_hosts"] = int(args.n_agn)
        f.attrs["built_by"] = str(Path(__file__).resolve())
        f.attrs["built_at_utc"] = datetime.now(timezone.utc).isoformat()
        f.attrs["clustering"] = ("none: gmd catalog is uniform on the sky and in "
                                 "comoving volume, so f_AGN is identified by the "
                                 "number-density contrast only")
        for k, v in post.items():
            f.create_dataset(k, data=v, compression="gzip", shuffle=True)
        f.create_dataset("host_type", data=host_type)
        g = f.create_group("truth")
        for k, v in truth.items():
            g.create_dataset(k, data=v)
        for k, v in obs.items():
            g.create_dataset(k, data=v)
    print(f"Wrote {gw_path}")

    # Survey files: one per tracer, pixelated by gmd's own routine.
    meta = {}
    for name, cat in (("gal", gal), ("agn", agn)):
        z = cat["z"]
        dz = np.full_like(z, args.z_error)
        w = np.ones_like(z)
        pix = gmd._pixelate_catalog(cat["ra"], cat["dec"], z, dz, w, args.nside, None)
        ngals = np.asarray(pix["ngals"])
        empty = float(1.0 - (ngals > 0).mean())
        p = out / f"survey_{name}_ns{args.nside}.h5"
        with h5py.File(p, "w") as f:
            for k, v in pix.items():
                f.create_dataset(k, data=np.asarray(v), compression="gzip",
                                 shuffle=True)
            f.attrs["nside"] = int(args.nside)
            f.attrs["tracer"] = name
            f.attrs["n_hosts"] = int(z.size)
            f.attrs["empty_pixel_fraction"] = empty
            f.attrs["z_error"] = float(args.z_error)
            f.attrs["source_complete_catalog"] = str(args.complete_catalog)
        meta[name] = {"n_hosts": int(z.size), "empty_pixel_fraction": empty,
                      "path": str(p),
                      "log10n0_count_anchored": float(np.log10(
                          z.size / (cosmo.comoving_volume(args.zmax).value)))}
        print(f"  survey_{name}: {z.size:,} hosts, {100*empty:.2f}% empty pixels, "
              f"log10n0 = {meta[name]['log10n0_count_anchored']:.4f}")

    (out / "twotracer_meta.json").write_text(json.dumps({
        "nobs": args.nobs, "n_gal_events": n_gal_ev, "n_agn_events": n_agn_ev,
        "truth_f_agn": truth_f, "n_agn_tracer_hosts": args.n_agn,
        "nside": args.nside, "snr_threshold": args.snr_threshold,
        "dL_fractional_uncertainty": args.dL_fractional_uncertainty,
        "gamma": args.gamma, "zmax": args.zmax, "seed": args.seed,
        "gw_path": str(gw_path), "surveys": meta,
        "clustering": "none (unclustered gmd catalog)",
    }, indent=2))
    print(f"Wrote {out / 'twotracer_meta.json'}")


if __name__ == "__main__":
    main()
