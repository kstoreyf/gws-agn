#!/usr/bin/env python3
"""Regenerate the deep two-tracer mock's EVENTS on the sigma_ang-FIXED generator.

darksirens PR #335 (worktree darksirens-oraclefix, commit 853ded3) fixed the mock's
sky-localisation width: the old generator set sigma_ang = clip(35/rho, 1, 12) deg from
a LATENT SNR (here the noisy detection SNR of the TRUE parameters), which makes the
sky-noise width an H0-sensitive observable (~ dL/Mc_det^(5/6)) that the fixed-width
sky posterior cannot represent; measured bias -0.49 +- 0.08 on H0 even under the exact
likelihood.  The fix draws the distance and masses FIRST and derives sigma_ang from
the OBSERVED amplitude (gmd._measure with widths["sigma_ang"] = None), so the recorded
sigma_ang is a deterministic function of the recorded data and the fixed-width sky
posterior is exact (exact-likelihood bootstrap closure -0.06 +- 0.07).

The fix lives in gmd's observed-detection path (_measure/_detect_on_observation); the
original build (build_twotracer_mock.py) used detection_data="true" + the internal
non-recorded path of _posterior_samples, which PR #335 does not touch.  So this script
keeps the TRUTH DRAW bit-identical (same rng seed and consumption through
_draw_events_until_detected) and replaces only the PE stage:

  1. widths from gmd._pe_widths(m1det, m2det, dl, rho_opt, 0.10, 0.08, 0.10, None)
     with rho_opt the projection-free true amplitude (gmd's observed-mode convention);
  2. ONE observation per event from the FIXED gmd._measure with sigma_ang = None
     (sequential observable sky width);
  3. gmd._posterior_samples(..., pe_centering="observed",
     use_recorded_observation=True) -- the flat-prior posterior of that measurement.

Detection is sky-independent (noisy SNR of true params), so the detected set is
unchanged; this script VERIFIES bit-identity of every truth array against the original
events file and refuses to write otherwise.  Catalogs, survey files and the targeted
injection set are therefore reused unchanged.

Convention change worth recording: the old sigma_ang used the noisy detection SNR
(which includes the Beta(2,5)^0.5 projection latent), the fixed one uses the
projection-free observed amplitude on the SNR_REF_DEFAULT = 11.5 scale, so post-fix
sky widths are systematically SMALLER (better localisation), exactly as in the
validated closure test (oracle_bootstrap.py --sigma_from_obs).
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

GMD_FIX_DIR = "/hildafs/projects/phy230014p/magana/src/darksirens-oraclefix/scripts/mock_dark_sirens"


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--complete_catalog", required=True)
    ap.add_argument("--gmd_dir", default=GMD_FIX_DIR)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--reference_events", required=True,
                    help="Original (pre-fix) events file; truth arrays must match "
                         "bit-identically or this script aborts.")
    ap.add_argument("--check_json", required=True)
    ap.add_argument("--n_agn", type=int, default=12000)
    ap.add_argument("--nobs", type=int, default=200)
    ap.add_argument("--f_agn", type=float, default=0.30)
    ap.add_argument("--nsamp", type=int, default=2000)
    ap.add_argument("--snr_threshold", type=float, default=8.0)
    ap.add_argument("--dL_fractional_uncertainty", type=float, default=0.10)
    ap.add_argument("--gamma", type=float, default=0.0)
    ap.add_argument("--H0", type=float, default=67.74)
    ap.add_argument("--Om0", type=float, default=0.3075)
    ap.add_argument("--zmax", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=7301)
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    sys.path.insert(0, str(Path(args.gmd_dir).resolve()))
    import generate_mock_data as gmd  # noqa: E402
    import subprocess
    gmd_commit = subprocess.run(
        ["git", "-C", str(Path(args.gmd_dir).resolve()), "rev-parse", "HEAD"],
        capture_output=True, text=True).stdout.strip()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    cosmo = gmd._build_cosmology(args.H0, args.Om0, -1.0, 0.0)
    grids = gmd._cosmology_grids(cosmo, args.zmax)
    pop = gmd.PopulationConfig(gamma=args.gamma)

    with h5py.File(args.complete_catalog, "r") as f:
        gal = {k: np.asarray(f[k][:], dtype=float) for k in ("ra", "dec", "z")}
    n_gal_hosts = gal["z"].size
    agn_idx = rng.choice(n_gal_hosts, size=args.n_agn, replace=False)
    agn = {k: v[agn_idx] for k, v in gal.items()}
    print(f"tracers: GAL {n_gal_hosts:,} hosts | AGN {args.n_agn:,} hosts")

    n_agn_ev = int(round(args.f_agn * args.nobs))
    n_gal_ev = args.nobs - n_agn_ev
    truth_f = n_agn_ev / args.nobs

    # Truth draw: BIT-IDENTICAL code path and rng consumption to the original build.
    parts = []
    for name, cat, n_ev in (("gal", gal, n_gal_ev), ("agn", agn, n_agn_ev)):
        t = gmd._draw_events_until_detected(rng, n_ev, cat, grids, pop,
                                            args.snr_threshold)
        print(f"  {name}: {len(t['z'])} events, z in "
              f"[{t['z'].min():.4f}, {t['z'].max():.4f}]")
        parts.append(t)
    truth = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    host_type = np.concatenate([np.zeros(n_gal_ev, dtype=np.int8),
                                np.ones(n_agn_ev, dtype=np.int8)])

    # --- VERIFY: detected set unchanged vs the original mock ---
    check = {"reference_events": str(args.reference_events),
             "gmd_dir": str(Path(args.gmd_dir).resolve()),
             "gmd_commit": gmd_commit, "truth_keys": {}}
    ok = True
    with h5py.File(args.reference_events, "r") as f:
        for k in ("z", "ra", "dec", "dl", "m1", "m2", "q", "chi", "snr"):
            ref = np.asarray(f["truth"][k][:])
            same = (ref.shape == truth[k].shape) and bool(np.array_equal(ref, truth[k]))
            check["truth_keys"][k] = "bit-identical" if same else "MISMATCH"
            ok = ok and same
        ht_ref = np.asarray(f["host_type"][:])
    same_ht = bool(np.array_equal(ht_ref, host_type))
    check["host_type"] = "bit-identical" if same_ht else "MISMATCH"
    ok = ok and same_ht
    check["detected_set_unchanged"] = bool(ok)
    check["consequence"] = ("targeted injections, survey files and catalogs reused "
                            "unchanged" if ok else "ABORTED")
    Path(args.check_json).write_text(json.dumps(check, indent=2))
    if not ok:
        raise SystemExit("[fatal] truth arrays differ from the reference mock; "
                         "the detected set is NOT unchanged -- see " + args.check_json)
    print("truth check: all arrays bit-identical to the reference mock")

    # --- PE stage on the FIXED generator: sequential observable sky width ---
    m1det = truth["m1"] * (1.0 + truth["z"])
    m2det = truth["m2"] * (1.0 + truth["z"])
    rho_opt = gmd._snr_from_detector_frame(m1det, m2det, truth["dl"])
    widths = gmd._pe_widths(m1det, m2det, truth["dl"], rho_opt,
                            args.dL_fractional_uncertainty, 0.08, 0.10, None)
    widths["sigma_ang"] = None          # THE FIX: derive from the observed amplitude
    widths["sigma_chi"] = 0.08
    obs_rec = gmd._measure(rng, m1det, m2det, truth["chi"], truth["dl"],
                           truth["ra"], truth["dec"], widths)
    truth_with_obs = dict(truth)
    truth_with_obs.update(obs_rec)
    post, obs = gmd._posterior_samples(
        rng, truth_with_obs, args.nsamp,
        dL_fractional_uncertainty=args.dL_fractional_uncertainty,
        pe_centering="observed", use_recorded_observation=True)
    z_pe = np.interp(post["dL"], grids["dl"], grids["z"])
    post["m1src"] = post["m1det"] / (1.0 + z_pe)
    post["m2src"] = post["m2det"] / (1.0 + z_pe)

    sa_deg = np.rad2deg(obs_rec["obs_sigma_ang"])
    print(f"sigma_ang (deg): min {sa_deg.min():.2f}, med {np.median(sa_deg):.2f}, "
          f"max {sa_deg.max():.2f}")
    check["sigma_ang_deg"] = {"min": float(sa_deg.min()),
                              "median": float(np.median(sa_deg)),
                              "max": float(sa_deg.max())}

    gw_path = out / "twotracer_gw_events_fix.h5"
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
        f.attrs["sigma_ang_convention"] = ("sequential observable sky width "
                                           "(darksirens PR #335, commit "
                                           f"{gmd_commit[:9]})")
        f.attrs["host_order"] = "gal_then_agn"
        f.attrs["n_host_gal"] = int(n_gal_ev)
        f.attrs["n_host_agn"] = int(n_agn_ev)
        f.attrs["truth_f_agn"] = float(truth_f)
        f.attrs["snr_threshold"] = float(args.snr_threshold)
        f.attrs["gamma"] = float(args.gamma)
        f.attrs["dL_fractional_uncertainty"] = float(args.dL_fractional_uncertainty)
        f.attrs["source_complete_catalog"] = str(args.complete_catalog)
        f.attrs["n_agn_tracer_hosts"] = int(args.n_agn)
        f.attrs["reference_events"] = str(args.reference_events)
        f.attrs["built_by"] = str(Path(__file__).resolve())
        f.attrs["built_at_utc"] = datetime.now(timezone.utc).isoformat()
        for k, v in post.items():
            f.create_dataset(k, data=v, compression="gzip", shuffle=True)
        f.create_dataset("host_type", data=host_type)
        g = f.create_group("truth")
        for k, v in truth.items():
            g.create_dataset(k, data=v)
        for k, v in obs_rec.items():
            g.create_dataset(k, data=v)
    print(f"Wrote {gw_path}")
    check["gw_path"] = str(gw_path)
    Path(args.check_json).write_text(json.dumps(check, indent=2))


if __name__ == "__main__":
    main()
