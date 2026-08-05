#!/usr/bin/env python3
"""analysis_0 -- sanity-check the ten pure-tracer event sets and record an inventory.

Each set must be a clean single-tracer 1000-event draw on the signed-off v3
catalogs: nobs = 1000, nsamp = 2000, every host of the declared type, every
recorded SNR above the detection threshold, and an events sub-seed that is not one
of the record's streams.  Writes results/event_sets.json.
"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/hildafs/projects/phy220048p/magana/gws-agn-data-v3")
SEEDS = [100, 101, 102, 103, 105]
# (suffix, tracer, planted f_agn, sub-seed offset)
SETS = [("puregal", "gal", 0.0, 8), ("pureagn", "agn", 1.0, 9)]
SNR_THRESHOLD = 8.0
RECORD_OFFSETS = {1: "glass_field", 2: "magnitudes", 3: "events", 4: "injections_targeted",
                  5: "injections_popuni", 6: "validation", 7: "photoz"}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(ROOT / "results/event_sets.json"))
    a = ap.parse_args(argv)

    out = {
        "what": "ten independent single-tracer event draws for analysis 0: for each "
                "of the five v3 catalog realisations, one 1000-event set with every "
                "host a galaxy (f_agn = 0) and one with every host an AGN "
                "(f_agn = 1), drawn on the SAME signed-off catalogs and surveys with "
                "event-noise streams independent of each other and of the record's "
                "mixture events.",
        "generator": "working/data/generate_dataset.py --stage events "
                     "--f_agn {0,1} --seed_events S*1000+{8,9} --n_events 1000 "
                     "--nsamp 2000 --events_suffix _pure{gal,agn}",
        "sub_seed_offsets_taken_by_the_record": RECORD_OFFSETS,
        "sub_seed_offsets_used_here": {8: "pure-GAL events", 9: "pure-AGN events"},
        "snr_threshold": SNR_THRESHOLD,
        "sets": [],
    }
    ok = True
    for s in SEEDS:
        for sfx, tracer, fagn, off in SETS:
            p = DATA / f"seed{s}" / "events" / f"events_{sfx}.h5"
            row = {"seed": s, "tracer": tracer, "suffix": sfx, "path": str(p),
                   "planted_f_agn": fagn, "seed_events": s * 1000 + off}
            with h5py.File(p, "r") as f:
                meta = json.loads(f.attrs["metadata_json"])
                r = meta["realised"]
                ht = f["host_type"][:]
                row.update(
                    nobs=int(f.attrs["nobs"]), nsamp=int(f.attrs["nsamp"]),
                    seed_events_recorded=int(f.attrs["seed_events"]),
                    planted_f_agn_recorded=float(f.attrs["planted_f_agn"]),
                    n_host_gal=int(f.attrs["n_host_gal"]),
                    n_host_agn=int(f.attrs["n_host_agn"]),
                    unique_gal_hosts=r["unique_gal_hosts"],
                    unique_agn_hosts=r["unique_agn_hosts"],
                    max_events_per_gal_host=r["max_events_per_gal_host"],
                    max_events_per_agn_host=r["max_events_per_agn_host"],
                    host_multiplicity_hist=(r["gal_host_multiplicity_hist"] if tracer == "gal"
                                            else r["agn_host_multiplicity_hist"]),
                    n_proposed=r["n_proposed"], detected_fraction=r["detected_fraction"],
                    snr_obs_min=r["snr_obs_min"], snr_obs_max=r["snr_obs_max"],
                    z_median_detected=r["z_median_detected"],
                    horizon_z_max_detected=r["horizon_z_max_detected"],
                    dL_max_detected_Mpc=r["dL_max_detected_Mpc"],
                    generated_at_utc=meta["generated_at_utc"],
                    file_size_MB=round(p.stat().st_size / 1e6, 1),
                )
                checks = {
                    "nobs_is_1000": row["nobs"] == 1000,
                    "nsamp_is_2000": row["nsamp"] == 2000,
                    "all_hosts_declared_tracer":
                        bool(np.all(ht == (1 if tracer == "agn" else 0))),
                    "host_counts_pure":
                        (row["n_host_agn"] == 1000 and row["n_host_gal"] == 0)
                        if tracer == "agn" else
                        (row["n_host_gal"] == 1000 and row["n_host_agn"] == 0),
                    "snr_obs_min_ge_threshold": row["snr_obs_min"] >= SNR_THRESHOLD,
                    "seed_events_as_requested":
                        row["seed_events_recorded"] == row["seed_events"],
                    "seed_events_offset_unused_by_record":
                        (row["seed_events"] - 1000 * s) not in RECORD_OFFSETS,
                    "planted_f_agn_as_requested":
                        row["planted_f_agn_recorded"] == fagn,
                }
            row["checks"] = checks
            row["PASS"] = all(checks.values())
            ok &= row["PASS"]
            out["sets"].append(row)

    # every draw must have its own stream
    seeds_used = [r["seed_events"] for r in out["sets"]]
    out["all_sub_seeds_distinct"] = len(set(seeds_used)) == len(seeds_used)
    out["PASS"] = bool(ok and out["all_sub_seeds_distinct"])
    Path(a.out).write_text(json.dumps(out, indent=2))

    hdr = (f"{'seed':>5} {'tracer':>6} {'seed_ev':>8} {'nobs':>5} {'nGAL':>5} "
           f"{'nAGN':>5} {'uniqH':>6} {'maxmult':>7} {'snrmin':>7} {'zmed':>6} "
           f"{'zmax':>6}  ok")
    print(hdr); print("-" * len(hdr))
    for r in out["sets"]:
        uh = r["unique_agn_hosts"] if r["tracer"] == "agn" else r["unique_gal_hosts"]
        mm = (r["max_events_per_agn_host"] if r["tracer"] == "agn"
              else r["max_events_per_gal_host"])
        print(f"{r['seed']:>5} {r['tracer']:>6} {r['seed_events']:>8} {r['nobs']:>5} "
              f"{r['n_host_gal']:>5} {r['n_host_agn']:>5} {uh:>6} {mm:>7} "
              f"{r['snr_obs_min']:>7.3f} {r['z_median_detected']:>6.3f} "
              f"{r['horizon_z_max_detected']:>6.3f}  {'PASS' if r['PASS'] else 'FAIL'}")
    print(f"\nall sub-seeds distinct: {out['all_sub_seeds_distinct']}")
    print(f"OVERALL: {'PASS' if out['PASS'] else 'FAIL'}   -> {a.out}")
    return 0 if out["PASS"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
