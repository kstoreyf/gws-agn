#!/usr/bin/env python3
"""Write the deep two-tracer mock's GAL and AGN host catalogs as standalone files.

``build_twotracer_mock.py`` defined the AGN tracer implicitly, as a seeded random
subset of the complete galaxy catalog, and only ever wrote out the PIXELATED
survey.  An incompleteness ladder needs the raw per-host catalog for both
tracers, because the flux limit acts on apparent magnitude -- a per-host
quantity the pixelated file does not carry.

The AGN subset is reproduced exactly: in ``build_twotracer_mock.py`` the
generator is ``np.random.default_rng(seed)`` and ``rng.choice(n_hosts,
size=n_agn, replace=False)`` is its FIRST consumption, so the same seed returns
the same indices.  That is an assumption about another script's draw order, so
it is CHECKED, not trusted: the reconstructed AGN redshifts are compared against
the ones stored in the mock's own pixelated AGN survey, and the script refuses
to write anything if they disagree.

AGN inherit their host galaxy's ``app_mag`` -- they ARE those galaxies -- so a
shared flux limit thins both tracers with the same C(z), which is what makes the
ladder a clean single-axis experiment.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

EXP_ROOT = Path(__file__).resolve().parents[1]
DEEP = EXP_ROOT.parent / "experiment_twotracer_deep"

CATALOG_KEYS = ("ra", "dec", "z", "abs_mag", "app_mag")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--complete_catalog", default=None,
                    help="Default: the catalog recorded in the deep mock's GAL survey.")
    ap.add_argument("--reference_agn_survey",
                    default=str(DEEP / "data_derived/survey_agn_ns32.h5"),
                    help="Pixelated AGN survey of the deep mock, used to VERIFY the "
                         "reconstructed subset.")
    ap.add_argument("--meta_json", default=str(DEEP / "data_derived/twotracer_meta.json"))
    ap.add_argument("--outdir", default=str(EXP_ROOT / "data_derived"))
    ap.add_argument("--seed", type=int, default=None,
                    help="Default: the seed recorded in the deep mock's metadata.")
    ap.add_argument("--n_agn", type=int, default=None)
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    meta = json.loads(Path(args.meta_json).read_text())
    seed = args.seed if args.seed is not None else int(meta["seed"])
    n_agn = args.n_agn if args.n_agn is not None else int(meta["n_agn_tracer_hosts"])

    cat_path = args.complete_catalog
    if cat_path is None:
        with h5py.File(meta["surveys"]["gal"]["path"], "r") as f:
            cat_path = str(f.attrs["source_complete_catalog"])
    with h5py.File(cat_path, "r") as f:
        cat = {k: np.asarray(f[k][:], dtype=float) for k in CATALOG_KEYS if k in f}
    missing = [k for k in ("ra", "dec", "z", "app_mag") if k not in cat]
    if missing:
        raise SystemExit(f"{cat_path}: missing {missing}; the flux limit needs app_mag")
    n_hosts = cat["z"].size
    print(f"complete catalog: {cat_path}\n  {n_hosts:,} hosts, "
          f"z in [{cat['z'].min():.4f}, {cat['z'].max():.4f}], "
          f"app_mag in [{cat['app_mag'].min():.2f}, {cat['app_mag'].max():.2f}]")

    # --- reproduce the AGN subset, then PROVE it is the right one --------------
    rng = np.random.default_rng(seed)
    agn_idx = rng.choice(n_hosts, size=n_agn, replace=False)
    z_agn = np.sort(cat["z"][agn_idx])
    with h5py.File(args.reference_agn_survey, "r") as f:
        zg = np.asarray(f["zgals"][:])
        ng = np.asarray(f["ngals"][:])
    valid = np.arange(zg.shape[1])[None, :] < ng[:, None]
    z_ref = np.sort(zg[valid])
    if z_ref.size != z_agn.size:
        raise SystemExit(f"AGN reconstruction: {z_agn.size} vs reference {z_ref.size}")
    if not np.allclose(z_agn, z_ref, rtol=0, atol=1e-12):
        raise SystemExit("AGN reconstruction does NOT match the deep mock's AGN survey; "
                         "the assumed draw order is wrong -- refusing to continue")
    print(f"AGN subset reproduced and VERIFIED against {args.reference_agn_survey} "
          f"({n_agn:,} hosts, max |dz| = {np.abs(z_agn - z_ref).max():.2e})")

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    written = {}
    for name, idx in (("gal", np.arange(n_hosts)), ("agn", agn_idx)):
        p = out / f"catalog_{name}_complete.h5"
        with h5py.File(p, "w") as f:
            f.attrs["mock_data"] = True
            f.attrs["tracer"] = name
            f.attrs["description"] = (
                "Complete host catalog of the deep two-tracer mock's "
                f"{name.upper()} tracer, before any flux limit.")
            f.attrs["source_complete_catalog"] = str(cat_path)
            f.attrs["n_hosts"] = int(idx.size)
            f.attrs["agn_subset_seed"] = int(seed)
            f.attrs["built_by"] = str(Path(__file__).resolve())
            f.attrs["built_at_utc"] = datetime.now(timezone.utc).isoformat()
            for k, v in cat.items():
                f.create_dataset(k, data=v[idx], compression="gzip", shuffle=True)
        written[name] = {"path": str(p), "n_hosts": int(idx.size)}
        print(f"  wrote {p}  ({idx.size:,} hosts)")

    (out / "tracer_catalogs.json").write_text(json.dumps(
        {"complete_catalog": str(cat_path), "seed": seed, "n_agn": n_agn,
         "verified_against": args.reference_agn_survey, "tracers": written},
        indent=2))
    print(f"  wrote {out / 'tracer_catalogs.json'}")


if __name__ == "__main__":
    main()
