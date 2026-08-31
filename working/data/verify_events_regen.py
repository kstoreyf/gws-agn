#!/usr/bin/env python3
"""Bitwise audit of the 2026-08-01 events regeneration (conventions (b2) and (c2)).

The two generator fixes are claimed to leave the DETECTED SET untouched:

  (c2) the mass PE is drawn by inverse CDF from the exact flat-prior posterior of
       ``obs ~ N(m, f m)``.  ``posterior_samples()`` runs AFTER the event loop, so it
       cannot move a single detection.
  (b2) ``observe()`` draws ``dec`` before ``ra`` and takes the RA width from the
       recorded ``dec``.  Both draws are ``n`` standard normals either way, so the
       generator's RNG stream advances identically and every quantity drawn BEFORE
       the sky block -- ``obs_dL``, ``obs_m1det``, ``obs_m2det``, ``obs_chieff``,
       ``sigma_ang`` -- and the rate-acceptance uniform drawn AFTER it are all
       untouched.  Detection reads only ``(obs_m1det, obs_m2det, obs_dL)``.

This script proves that on the files rather than asserting it: every column that must
be bit-identical is compared with ``np.array_equal``, the two columns that MUST move
are quantified, and the detection rule is re-run on both files.

    python verify_events_regen.py --seed 100 \
        [--old seed100/events_prefix2/events.h5] [--new seed100/events/events.h5]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

import generate_dataset as G

# Columns of truth/ that MUST be bit-identical: everything the detected set is made
# of, plus every observable drawn before or independently of the sky block.
MUST_MATCH = (
    "z", "ra", "dec", "dl", "m1src", "m2src", "q", "chieff", "m1det", "m2det",
    "host_type", "host_index", "snr_obs", "snr_true",
    "obs_dL", "obs_m1det", "obs_m2det", "obs_sigma_dl", "obs_sig_m1", "obs_sig_m2",
    "obs_sigma_ang", "obs_chieff",
)
# Columns that MUST move (the (b2) fix reassigns the two sky normal blocks).
MUST_MOVE = ("obs_ra", "obs_dec")
# New in the fixed generator.
NEW_COLS = ("obs_sig_ra",)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--root", default=str(Path(__file__).resolve().parent))
    ap.add_argument("--old", default=None)
    ap.add_argument("--new", default=None)
    ap.add_argument("--out", default=None)
    return ap.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    sd = Path(a.root) / f"seed{a.seed}"
    old = Path(a.old) if a.old else sd / "events_prefix2" / "events.h5"
    new = Path(a.new) if a.new else sd / "events" / "events.h5"
    out = Path(a.out) if a.out else sd / "validation" / "events_regen_bitcheck.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    def load(p):
        with h5py.File(p, "r") as f:
            tr = {k: f["truth"][k][:] for k in f["truth"].keys()}
            at = {k: (v.item() if hasattr(v, "item") and np.ndim(v) == 0 else v)
                  for k, v in f.attrs.items() if k != "metadata_json"}
            nobs, nsamp = int(f.attrs["nobs"]), int(f.attrs["nsamp"])
            pe = {k: f[k][:].reshape(nobs, nsamp)
                  for k in ("dL", "m1det", "m2det", "ra", "dec", "chieff")}
            ppe = f["p_pe"][:].reshape(nobs, nsamp)
            meta = json.loads(f.attrs["metadata_json"])
        return tr, at, pe, ppe, nobs, nsamp, meta

    tro, ato, peo, ppeo, nobs_o, nsamp_o, mo = load(old)
    trn, atn, pen, ppen, nobs_n, nsamp_n, mn = load(new)

    res = {"name": "events_regen_bitcheck", "seed": a.seed,
           "old": str(old), "new": str(new),
           "nobs": [nobs_o, nobs_n], "nsamp": [nsamp_o, nsamp_n]}

    # --- 1. the detected set --------------------------------------------------
    ident, bad = {}, []
    for k in MUST_MATCH:
        same = bool(np.array_equal(tro[k], trn[k]))
        ident[k] = same
        if not same:
            bad.append(k)
    res["bit_identical"] = ident
    res["bit_identical_all"] = (len(bad) == 0 and nobs_o == nobs_n)
    res["bit_identical_failures"] = bad

    # --- 2. what must move ----------------------------------------------------
    moved = {}
    for k in MUST_MOVE:
        d = np.abs(trn[k] - tro[k])
        if k == "obs_ra":
            d = np.abs((trn[k] - tro[k] + np.pi) % (2 * np.pi) - np.pi)
        moved[k] = {"changed_fraction": float((d > 0).mean()),
                    "median_abs_change_deg": float(np.rad2deg(np.median(d))),
                    "max_abs_change_deg": float(np.rad2deg(d.max()))}
    res["moved"] = moved
    res["new_columns_present"] = {k: bool(k in trn) for k in NEW_COLS}

    # --- 3. the detection rule, re-run on both --------------------------------
    def det(tr):
        rho = G.snr_amplitude(tr["obs_m1det"], tr["obs_m2det"], tr["obs_dL"],
                              G.SNR_REF_DETECT)
        return rho, bool(np.array_equal(rho, tr["snr_obs"])), bool(
            np.all(rho >= G.SNR_THRESHOLD))

    rho_o, ex_o, ge_o = det(tro)
    rho_n, ex_n, ge_n = det(trn)
    res["detection_replay"] = {
        "old": {"recomputed_equals_stored_bitwise": ex_o, "all_ge_threshold": ge_o},
        "new": {"recomputed_equals_stored_bitwise": ex_n, "all_ge_threshold": ge_n},
        "rho_obs_old_equals_new_bitwise": bool(np.array_equal(rho_o, rho_n)),
        "snr_ref_detect": G.SNR_REF_DETECT, "snr_threshold": G.SNR_THRESHOLD}

    # --- 4. the (b2) defect that was removed ----------------------------------
    ct_o = np.maximum(np.cos(tro["dec"]), 0.1)
    co_o = np.maximum(np.cos(tro["obs_dec"]), 0.1)
    r_old = np.abs(co_o / ct_o - 1.0)          # old PE width / old measurement width
    sig_ra_n = trn["obs_sigma_ang"] / np.maximum(np.cos(trn["obs_dec"]), 0.1)
    res["b2_fix"] = {
        "old_pe_ra_width_error_mean": float(r_old.mean()),
        "old_pe_ra_width_error_rms": float(np.sqrt((r_old ** 2).mean())),
        "old_pe_ra_width_error_max": float(r_old.max()),
        "new_sig_ra_recomputable_bitwise":
            bool(np.array_equal(sig_ra_n, trn["obs_sig_ra"])),
        "new_sig_ra_deg_range": [float(np.rad2deg(trn["obs_sig_ra"].min())),
                                 float(np.rad2deg(trn["obs_sig_ra"].max()))]}

    # --- 5. the (c2) fix, measured on the stored PE ---------------------------
    def shift(pe_m, obs):
        return float((pe_m / obs[:, None]).mean() - 1.0)

    res["c2_fix"] = {
        "old_mean_m1_over_obs_minus_1": shift(peo["m1det"], tro["obs_m1det"]),
        "new_mean_m1_over_obs_minus_1": shift(pen["m1det"], trn["obs_m1det"]),
        "predicted_2f2_m1": 2.0 * G.SIG_M1_FRAC ** 2,
        "old_mean_m2_over_obs_minus_1": shift(peo["m2det"], tro["obs_m2det"]),
        "new_mean_m2_over_obs_minus_1": shift(pen["m2det"], trn["obs_m2det"]),
        "predicted_2f2_m2": 2.0 * G.SIG_M2_FRAC ** 2,
        "old_pe_m1_sd_over_obs": float((peo["m1det"].std(axis=1, ddof=1)
                                        / tro["obs_m1det"]).mean()),
        "new_pe_m1_sd_over_obs": float((pen["m1det"].std(axis=1, ddof=1)
                                        / trn["obs_m1det"]).mean()),
        "new_pe_m1_min_over_obs": float((pen["m1det"] / trn["obs_m1det"][:, None]).min()),
        "new_pe_m1_max_over_obs": float((pen["m1det"] / trn["obs_m1det"][:, None]).max()),
        "new_pe_m2_min_over_obs": float((pen["m2det"] / trn["obs_m2det"][:, None]).min()),
        "new_pe_m2_max_over_obs": float((pen["m2det"] / trn["obs_m2det"][:, None]).max()),
        "new_clip_count_m1_at_2Msun": int((pen["m1det"] <= 2.0).sum()),
        "new_clip_count_m2_at_1Msun": int((pen["m2det"] <= 1.0).sum())}

    # --- 6. p_pe bookkeeping ---------------------------------------------------
    res["p_pe"] = {
        "old_proportional_to_m1det_maxdev": float(np.abs(
            ppeo / (peo["m1det"] / peo["m1det"].mean(axis=1, keepdims=True)) - 1).max()),
        "new_proportional_to_m1det_maxdev": float(np.abs(
            ppen / (pen["m1det"] / pen["m1det"].mean(axis=1, keepdims=True)) - 1).max()),
        "new_per_event_mean": float(ppen.mean(axis=1).min()),
        "note": "p_pe is the PE PRIOR in the canonical (m1det, q, dL, chieff) basis, "
                "proportional to m1det, stored mean-1 per event; unchanged by (b2)/(c2)"}

    # --- 7. bookkeeping from the metadata -------------------------------------
    res["realised"] = {
        k: [mo["realised"].get(k), mn["realised"].get(k)]
        for k in ("n_proposed", "n_detected_total", "detected_fraction",
                  "horizon_z_max_detected", "realised_f_agn", "n_host_gal",
                  "n_host_agn")}
    res["realised_identical"] = all(
        v[0] == v[1] for v in res["realised"].values())

    ok = (res["bit_identical_all"] and res["detection_replay"]["new"][
        "recomputed_equals_stored_bitwise"] and res["detection_replay"][
        "rho_obs_old_equals_new_bitwise"] and res["realised_identical"]
        and res["b2_fix"]["new_sig_ra_recomputable_bitwise"])
    res["PASS"] = bool(ok)
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    print(f"\n{'PASS' if ok else 'FAIL'}  -> {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
