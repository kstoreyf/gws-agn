#!/usr/bin/env python3
"""ENDGAME -- replay ``generate_dataset.stage_events``' proposal loop WITHOUT the
``[:N_EVENTS]`` truncation, and record every detection's position in the stream.

CLOSURE.md 14.4 item 1 is the last unaudited piece of the mock: ``stage_events``
draws fixed batches of ``ntry = 100_000`` i.i.d. proposals until ``N_EVENTS = 1000``
detections have accumulated, then keeps ``concatenate(...)[:1000]`` of however many
it found (1521 on seed 100, from 200_000 proposals -- i.e. TWO batches).  The kept
set is claimed exchangeable; this replays the identical stream and lets that claim
be measured.

WHAT IS REPLAYED.  The loop body is copied verbatim from ``stage_events`` and calls
the SAME functions off the SAME module (``generate_dataset`` is imported, not
duplicated): ``gmd._sample_powerlaw_peak_m1``, ``gmd._sample_q``,
``gmd._sample_chieff``, ``gd.observe``, ``gd.detect_from_observation``,
``gd.snr_amplitude``, and the identical rng consumption order.  Nothing under
``working/data`` is written or modified; the only outputs are in the scratch tree.

Because the RNG consumption order is preserved, running with ``--seed 100`` must
reproduce the record's first 1000 events BIT-IDENTICALLY -- that is the audit, and
``--verify`` performs it against ``seed100/events/events.h5``.

MODES
  single    one replay of one master seed (default; use --verify for seed 100)
  replicas  N independent replays with fresh EVENT sub-seeds on the SAME catalog,
            used to measure  E[A] - B  to arbitrary precision.  Everything else in
            the mock (catalog, survey, injections) is held fixed, so the exact
            oracle value of B is a single number and the only Monte Carlo left is
            the event draw itself -- which is exactly the thing under audit.

OUTPUT (scratch): one HDF5 with per-detection truth arrays plus
  replica     which replay it came from
  rank        0-based position of this detection within its replay's detection
              sequence (rank < 1000 == KEPT by the record's truncation)
  batch       which 100_000-proposal batch it came from
  slot        its index WITHIN that batch's 100_000 proposals
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

DATA = Path("/hildafs/projects/phy230014p/magana/gws-agn/working/data")
SCRATCH = Path("/hildafs/projects/phy220048p/magana/gws-agn-data/scratch_truncation_test")

TRUTH_KEYS = ("z", "ra", "dec", "dl", "m1src", "m2src", "q", "chieff",
              "m1det", "m2det", "host_type", "host_index", "snr_obs", "snr_true")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=100,
                    help="Master seed whose CATALOG is used (and, in single mode, "
                         "whose event sub-seed is replayed).")
    ap.add_argument("--replicas", type=int, default=0,
                    help="If > 0, run this many independent replays with event "
                         "sub-seeds --rep_seed0 + k instead of the record's.")
    ap.add_argument("--rep_seed0", type=int, default=9_000_000)
    ap.add_argument("--verify", action="store_true",
                    help="Single mode: check the first 1000 events bitwise against "
                         "the stored record file.")
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--dataroot", default=None,
                    help="Root holding seed<N>/ (default: working/data).")
    ap.add_argument("--pe_model", choices=("v2", "v3"), default=None,
                    help="Measurement family to replay.  Default: read from the "
                         "seed's own events.h5 attr 'pe_model', else v2.")
    return ap.parse_args(argv)


def replay(gd, gmd, rng, cats, grids, pop, rate_gmax, n_events, ntry,
           pe_model="v2"):
    """One verbatim replay of stage_events' proposal loop; keeps EVERY detection."""
    n_gal = cats["gal"]["z"].size
    n_agn = cats["agn"]["z"].size
    keep, n_tried, n_pass_snr, n_have, batch = [], 0, 0, 0, 0
    while n_have < n_events:
        u_type = rng.uniform(size=ntry)
        i_gal = rng.integers(0, n_gal, ntry)
        i_agn = rng.integers(0, n_agn, ntry)
        is_agn = u_type < gd.F_AGN
        host_idx = np.where(is_agn, i_agn, i_gal)
        z = np.where(is_agn, cats["agn"]["z"][i_agn], cats["gal"]["z"][i_gal])
        ra = np.where(is_agn, cats["agn"]["ra"][i_agn], cats["gal"]["ra"][i_gal])
        dec = np.where(is_agn, cats["agn"]["dec"][i_agn], cats["gal"]["dec"][i_gal])
        dl = gmd._interp_dl(z, grids)

        m1, use_peak = gmd._sample_powerlaw_peak_m1(rng, ntry, pop, return_component=True)
        q = gmd._sample_q(rng, m1, pop, use_peak=use_peak)
        m2 = q * m1
        chi = gmd._sample_chieff(rng, ntry, pop)

        m1det, m2det = m1 * (1.0 + z), m2 * (1.0 + z)
        if pe_model == "v3":
            obs = gd.observe_v3(rng, m1det, m2det, chi, dl, ra, dec, need_sky=True)
            det_snr, rho_obs = gd.detect_v3(obs)
        else:
            obs = gd.observe(rng, m1det, m2det, chi, dl, ra, dec, need_sky=True)
            det_snr, rho_obs = gd.detect_from_observation(obs)
        rho_true = gd.snr_amplitude(m1det, m2det, dl, gd.SNR_REF_DETECT)
        acc = rng.uniform(size=ntry) < (1.0 + z) ** (pop.gamma - 1.0) / rate_gmax
        det = det_snr & acc

        n_tried += ntry
        n_pass_snr += int(det_snr.sum())
        # NOTE: the record's rejected-proposal bookkeeping consumes no RNG, so it is
        # omitted here without changing the stream.
        if np.any(det):
            slots = np.flatnonzero(det)
            rec = {"z": z[det], "ra": ra[det], "dec": dec[det], "dl": dl[det],
                   "m1src": m1[det], "m2src": m2[det], "q": q[det], "chieff": chi[det],
                   "m1det": m1det[det], "m2det": m2det[det],
                   "host_type": is_agn[det].astype(np.int64),
                   "host_index": host_idx[det].astype(np.int64),
                   "snr_obs": rho_obs[det], "snr_true": rho_true[det],
                   "batch": np.full(slots.size, batch, dtype=np.int32),
                   "slot": slots.astype(np.int32)}
            for k, v in obs.items():
                rec[f"obs_{k}"] = v[det]
            keep.append(rec)
            n_have += int(det.sum())
        batch += 1
    out = {k: np.concatenate([x[k] for x in keep]) for k in keep[0]}
    out["rank"] = np.arange(out["z"].size, dtype=np.int32)
    return out, n_tried, n_pass_snr


def main(argv=None):
    args = parse_args(argv)
    sys.path.insert(0, str(DATA))
    import generate_dataset as gd
    gmd = gd.import_gmd(gd.DARKSIRENS_REPO)

    _root = Path(args.dataroot) if args.dataroot else DATA
    sd = gd.seed_dir(args.seed, _root)
    pe_model = args.pe_model
    if pe_model is None:
        import h5py as _h5
        try:
            with _h5.File(sd / "events" / "events.h5", "r") as _f:
                pe_model = str(_f.attrs.get("pe_model", "v2"))
        except Exception:
            pe_model = "v2"
    print(f"[pe_model] replaying the {pe_model} measurement family")
    cosmo = gmd._build_cosmology(gd.H0_FID, gd.OM0_FID, gd.W0_FID, gd.WA_FID)
    grids = gmd._cosmology_grids(cosmo, gd.ZMAX_GRID)
    pop = gmd.PopulationConfig(gamma=gd.GAMMA)
    rate_gmax = max(1.0, (1.0 + float(grids["z"][-1])) ** (pop.gamma - 1.0))
    print(f"rate_gmax = {rate_gmax}   GAMMA = {gd.GAMMA}   F_AGN = {gd.F_AGN}")

    t0 = time.time()
    cats = {t: gd.load_catalog(sd / "catalogs" / f"catalog_{t}_complete.h5",
                               keys=("ra", "dec", "z"))
            for t in gd.TRACERS}
    print(f"catalogs loaded ({time.time()-t0:.0f}s): "
          f"GAL {cats['gal']['z'].size:,}  AGN {cats['agn']['z'].size:,}")

    ntry = max(4 * gd.N_EVENTS, 100_000)
    n_events = gd.N_EVENTS
    info = {"seed": args.seed, "ntry": ntry, "n_events_target": n_events,
            "rate_gmax": rate_gmax, "pe_model": pe_model, "darksirens_sha": gd._git(gd.DARKSIRENS_REPO,
                                                              "rev-parse", "HEAD"),
            "generated_at_utc": gd._now()}

    if args.replicas > 0:
        tag = args.tag or f"replicas_s{args.seed}_n{args.replicas}"
        parts, meta = [], []
        t0 = time.time()
        for k in range(args.replicas):
            es = args.rep_seed0 + k
            rng = np.random.default_rng(es)
            out, n_tried, n_pass = replay(gd, gmd, rng, cats, grids, pop,
                                          rate_gmax, n_events, ntry, pe_model=pe_model)
            out["replica"] = np.full(out["z"].size, k, dtype=np.int32)
            parts.append(out)
            meta.append({"k": k, "event_seed": es, "n_det": int(out["z"].size),
                         "n_tried": n_tried, "n_pass_snr": n_pass})
            if (k + 1) % 10 == 0:
                el = time.time() - t0
                print(f"  replica {k+1}/{args.replicas}  {el:.0f}s  "
                      f"({el/(k+1):.2f}s each, ETA {el/(k+1)*(args.replicas-k-1):.0f}s)",
                      flush=True)
        allk = sorted(set().union(*[set(p) for p in parts]))
        data = {k: np.concatenate([p[k] for p in parts]) for k in allk}
        info.update({"mode": "replicas", "replicas": args.replicas,
                     "rep_seed0": args.rep_seed0, "per_replica": meta,
                     "n_detected_total": int(data["z"].size)})
    else:
        tag = args.tag or f"full_s{args.seed}"
        es = gd.sub_seeds(args.seed)["events"]
        rng = np.random.default_rng(es)
        t0 = time.time()
        data, n_tried, n_pass = replay(gd, gmd, rng, cats, grids, pop,
                                       rate_gmax, n_events, ntry, pe_model=pe_model)
        data["replica"] = np.zeros(data["z"].size, dtype=np.int32)
        n_det = int(data["z"].size)
        print(f"replay {time.time()-t0:.0f}s: {n_det} detections from {n_tried:,} "
              f"proposals ({n_pass} passed SNR)")
        info.update({"mode": "single", "event_seed": es, "n_detected_total": n_det,
                     "n_tried": n_tried, "n_pass_snr": n_pass,
                     "n_batches": int(data['batch'].max()) + 1})

        if args.verify:
            import h5py
            rp = sd / "events" / "events.h5"
            bad, checked = {}, 0
            with h5py.File(rp, "r") as f:
                rec_meta = json.loads(f.attrs["metadata_json"])
                for k in f["truth"]:
                    a = np.asarray(f["truth"][k][:])
                    if k in data:
                        b = data[k][:a.size]
                    elif f"obs_{k[4:]}" in data and k.startswith("obs_"):
                        b = data[k][:a.size]
                    else:
                        continue
                    checked += 1
                    nb = int(np.count_nonzero(a != b))
                    if nb:
                        bad[k] = {"n_diff": nb,
                                  "maxabs": float(np.max(np.abs(a - b)))}
            info["verify"] = {
                "record_path": str(rp), "n_fields_checked": checked,
                "n_fields_with_any_bit_difference": len(bad), "detail": bad,
                "record_n_detected_total": rec_meta["realised"]["n_detected_total"],
                "record_n_proposed": rec_meta["realised"]["n_proposed"],
                "replay_n_detected_total": n_det, "replay_n_proposed": n_tried,
                "counts_match": bool(
                    rec_meta["realised"]["n_detected_total"] == n_det
                    and rec_meta["realised"]["n_proposed"] == n_tried),
            }
            print(f"VERIFY: {checked} truth fields checked, "
                  f"{len(bad)} with any bit difference; counts_match="
                  f"{info['verify']['counts_match']}")
            if bad:
                print(json.dumps(bad, indent=1))

    SCRATCH.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if args.out else SCRATCH / f"events_notrunc_{tag}.h5"
    import h5py
    with h5py.File(out_path, "w") as f:
        for k, v in data.items():
            f.create_dataset(k, data=v, compression="gzip", compression_opts=1)
        f.attrs["info_json"] = json.dumps(info)
    (out_path.with_suffix(".json")).write_text(json.dumps(info, indent=2))
    print(f"wrote {out_path}  ({data['z'].size:,} detections)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
