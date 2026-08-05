#!/usr/bin/env python3
"""Collect the whole 2026-08-01 closure campaign into one file.

Reads everything the campaign produced and writes ``results/closure_summary.json``
-- the single object CLOSURE.md quotes from:

  generator      the two fixes, their validation outcomes on all five seeds, and
                 the bitwise proof that the detected sets did not move
  scans          the four production configurations and both matched controls on
                 seed 100, before and after
  closure        the five-realisation matched-host table, before and after
  score          the per-event score residual r, term by term, before and after
  sky_oracle     the exact host-galaxy oracle: the pixelisation term, the photo-z
                 kernel term, and what is left
  nside          the residual-vs-nside curve, if it was run

Nothing here recomputes physics; it only gathers.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
PRE = ROOT / "results_prefix2"
ATTIC = ROOT / "results_dsc_attic"
DATA = Path("/hildafs/projects/phy230014p/magana/gws-agn/working/data")
TRUTH = 67.74
SEEDS = [100, 101, 102, 103, 105]


def jload(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def scan(tag, where=RES):
    d = jload(Path(where) / f"{tag}.json")
    if d is None:
        return None
    h = d["H0"]
    grid = np.asarray(h["grid"], float)
    edge = 0.5 * (grid[1] - grid[0])   # grid IS the full 201-point array
    cells = d.get("guard", {}).get("cells") or []
    return {"tag": tag, "median": h["median"], "offset": h["median"] - TRUTH,
            "ci68": h["ci68"], "ci90": h["ci90"], "map": h["map"],
            "half68": 0.5 * (h["ci68"][1] - h["ci68"][0]),
            "truth_in_ci68": h["truth_in_ci68"], "truth_in_ci90": h["truth_in_ci90"],
            "railed": bool(h["map"] <= grid[0] + 1e-9 or h["map"] >= grid[-1] - 1e-9
                               or h["ci90"][0] <= grid[0] + edge
                               or h["ci90"][1] >= grid[-1] - edge),
            "n_rejected": d.get("n_rejected"),
            "min_Neff": (min(c["Neff"] for c in cells) if cells else None),
            "n_events": int(round(cells[0]["threshold"] / 5)) if cells else None}


def tstat(off):
    off = np.asarray(off, float)
    n = off.size
    if n < 2:
        return {"n": int(n), "mean": float(off.mean()) if n else None}
    sd = float(off.std(ddof=1))
    sem = sd / math.sqrt(n)
    t = float(off.mean() / sem)
    from scipy import stats
    return {"n": int(n), "mean": float(off.mean()), "sd": sd, "sem": sem,
            "t": t, "p_two_sided": float(2 * stats.t.sf(abs(t), n - 1))}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(RES / "closure_summary.json"))
    a = ap.parse_args(argv)

    out = {"name": "closure_summary", "truth_H0": TRUTH, "seeds": SEEDS,
           "scope": ("dark_sirens at log10n0 = -24, field sky weighting, K = 1, "
                     "targeted injections, H0 in [50, 100] x 201, W = 4096 (GAL), "
                     "campaign guard convention.  darksirens READ-ONLY at 2b86a2d.")}

    # ---------------- generator -------------------------------------------
    gen = {"fixes": {
        "b2_ra_width": "observe() draws dec first and takes the RA width from the "
                       "RECORDED dec: sig_ra = sigma_ang/max(cos dec_obs, 0.1), "
                       "stored as obs_sig_ra and reused verbatim by the PE",
        "c2_mass_pe": "the mass PE is the exact flat-prior posterior of "
                      "obs ~ N(m, f m), p(m|obs) ~ (1/(f m)) exp[-(obs-m)^2/"
                      "(2 f^2 m^2)], drawn by inverse CDF"},
        "per_seed": {}}
    for s in SEEDS:
        v = jload(DATA / f"seed{s}" / "validation" / "validation.json")
        b = jload(DATA / f"seed{s}" / "validation" / "events_regen_bitcheck.json")
        if v is None:
            continue
        c2b = v["checks"]["V2b_ra_width_from_observed_dec"]
        c3 = v["checks"]["V3_pe_calibration"]
        gen["per_seed"][str(s)] = {
            "n_checks": v["n_checks"], "n_failed": v["n_failed"],
            "failed": v["failed"],
            "V2b": {k: c2b[k] for k in (
                "pe_width_recomputed_from_stored_equals_stored_bitwise",
                "mean_rel_width_error_if_latent_dec_were_used",
                "rms_rel_width_error_if_latent_dec_were_used",
                "max_rel_width_error_if_latent_dec_were_used",
                "ks_pe_ra_pooled_pvalue", "ks_pe_dec_pooled_pvalue",
                "ks_measurement_pull_ra_pvalue",
                "pe_ra_width_ratio_pull_sigma", "pe_dec_width_ratio_pull_sigma")},
            "V3": {k: c3[k] for k in (
                "ks_m1det_pooled_pvalue", "ks_m2det_pooled_pvalue",
                "ks_dL_pooled_pvalue", "ks_chieff_pooled_pvalue",
                "per_event_ks_m1det_uniformity_pvalue",
                "per_event_ks_m2det_uniformity_pvalue",
                "exact_mass_posterior_numeric_cdf_max_abs_err",
                "exact_mass_posterior_table_convergence",
                "exact_mass_posterior_mean_shift",
                "clip_fraction_m1det", "clip_fraction_m2det")},
            "bitcheck": None if b is None else {
                "PASS": b["PASS"], "bit_identical_all": b["bit_identical_all"],
                "detection_replay": b["detection_replay"],
                "realised_identical": b["realised_identical"],
                "moved": b["moved"], "b2_fix": b["b2_fix"], "c2_fix": b["c2_fix"],
                "p_pe": b["p_pe"]},
        }
    out["generator"] = gen

    # ---------------- scans -------------------------------------------------
    prod = {}
    for t in ("h0_gal_targeted", "h0_gal_popuni", "h0_agn_targeted",
              "h0_agn_popuni", "ctrl_gal_matched", "ctrl_agn_matched"):
        prod[t] = {"after": scan(t), "before": scan(t, PRE)}
    out["scans_seed100"] = prod
    out["h0_single_tracer"] = {"after": jload(RES / "h0_single_tracer.json"),
                               "before": jload(PRE / "h0_single_tracer.json")}

    # ---------------- five-realisation closure ------------------------------
    cl = {}
    for case in ("gal", "agn"):
        rows = []
        for s in SEEDS:
            tag = f"ctrl_{case}_matched" if s == 100 else f"ctrl_{case}_matched_s{s}"
            aft = scan(tag)
            bef = scan(tag, PRE) or scan(tag, ATTIC)
            if aft is None:
                continue
            rows.append({"seed": s, "before": bef, "after": aft,
                         "delta": (aft["offset"] - bef["offset"]) if bef else None})
        cl[case] = {
            "per_seed": rows,
            "before": tstat([r["before"]["offset"] for r in rows if r["before"]]),
            "after": tstat([r["after"]["offset"] for r in rows]),
            "mean_shift": tstat([r["delta"] for r in rows
                                 if r["delta"] is not None])}
    out["closure_five_realisations"] = cl
    out["closure_seeds_json"] = jload(RES / "closure_seeds.json") is not None

    # ---------------- the per-event score residual --------------------------
    sc = {}
    for tr in ("gal", "agn"):
        aft = jload(RES / f"attr_terms_{tr}_s100_postfix.json")
        bef = jload(RES / f"attr_terms_{tr}_s100.json")
        sc[tr] = {"after": aft, "before": bef}
    out["score_terms"] = sc

    # ---------------- the sky oracle ----------------------------------------
    sky = {}
    for tr in ("gal", "agn"):
        d = jload(RES / f"attr_sky_oracle_{tr}.json")
        if d is None:
            continue
        conv = {}
        for lab in ("ap4", "ap8", "sf5", "sf7", "sub4", "sub6", "nz", "nm",
                    "shift", "base"):
            c = jload(RES / f"attr_sky_oracle_{tr}_conv_{lab}.json")
            if c:
                conv[lab] = {
                    "grid": c["grid"], "n_events": c["n_events"],
                    "substitutions": c["substitutions"],
                    "arms_r": {k: v["r"]["mean"] for k, v in c["arms"].items()},
                    "diag": {k: c["diagnostics"][k] for k in
                             ("n_pix", "n_gal", "sky_mass_kept", "sky_mass_mapped")}}
        # The smooth sky rule does not model the |dec| <= pi/2 CLIP the generator
        # applies to the PE dec samples, so for events within ~1 sigma_ang of a pole
        # the rule loses exactly the clipped mass.  That is verified below
        # (sum_p W_p == P(|dec| <= pi/2) to 4e-3 on every event), and every arm is
        # re-quoted on the events more than 3 sigma_ang from a pole.
        np_ = np.load(RES / f"attr_sky_oracle_{tr}.npz")
        ps = np_["pole_sigma"]
        cand = np_["diag_sky_mass_cand"]
        from scipy.stats import norm
        dec, sa = np_["obs_dec_deg"], np_["sigma_ang_deg"]
        pred = norm.cdf((90.0 - dec) / sa) - norm.cdf((-90.0 - dec) / sa)

        def _st(x, y=None):
            v = x if y is None else x - y
            v = v[np.isfinite(v)]
            return {"mean": float(v.mean()),
                    "sem": float(v.std(ddof=1) / np.sqrt(v.size)), "n": int(v.size)}

        ds_, kp, dp, dh = (np_["ds_score"], np_["score_kde_pix"],
                           np_["score_delta_pix"], np_["score_delta_host"])
        cut = ps > 3.0
        nopole = {
            "n_events": int(cut.sum()),
            "anchor_kde_pix_minus_darksirens": _st(kp[cut], ds_[cut]),
            "pixelisation__host_minus_pix__delta_prior": _st(dh[cut], dp[cut]),
            "photoz_kernel__delta_minus_kde__pixel_sky": _st(dp[cut], kp[cut]),
            "r_kde_pix": _st(kp[cut] - d["dlnmu_dH0"]),
            "r_delta_pix": _st(dp[cut] - d["dlnmu_dH0"]),
            "r_delta_host": _st(dh[cut] - d["dlnmu_dH0"])}
        if "score_kde_host" in np_:
            nopole["pixelisation__host_minus_pix__kde_prior"] = _st(
                np_["score_kde_host"][cut], kp[cut])
        sky[tr] = {"anchors": d["anchors"], "dlnmu_dH0": d["dlnmu_dH0"],
                   "pole_clip": {
                       "n_events_within_3_sigma_ang_of_a_pole": int((~cut).sum()),
                       "sky_mass_cand_min": float(cand.min()),
                       "max_abs_diff_from_P_dec_inside_sphere":
                           float(np.abs(cand - pred).max()),
                       "note": "sum_p W_p over the grown aperture equals "
                               "P(|dec| <= pi/2 | data) exactly, i.e. the ONLY sky "
                               "mass the rule misses is the generator's dec clip; "
                               "the aperture itself is converged"},
                   "excluding_polar_events": nopole,
                   "darksirens": d["darksirens"],
                   "darksirens_pe_mc_sigma": d["darksirens_pe_mc_sigma"],
                   "arms": d["arms"], "substitutions": d["substitutions"],
                   "diagnostics": d["diagnostics"], "grid": d["grid"],
                   "n_events": d["n_events"], "convergence": conv}
    out["sky_oracle"] = sky

    # ---------------- the nside curve ---------------------------------------
    ns = {"surveys": jload(RES / "surveys_nside.json"), "scans": {}}
    for case in ("gal", "agn"):
        for n in (32, 64, 128):
            tag = (f"ctrl_{case}_matched" if n == 32
                   else f"ctrl_{case}_matched_ns{n}")
            s = scan(tag)
            if s:
                ns["scans"][f"{case}_ns{n}"] = s
    out["nside_study"] = ns

    Path(a.out).write_text(json.dumps(out, indent=2, default=str))
    print(f"wrote {a.out}")
    for case in ("gal", "agn"):
        c = out["closure_five_realisations"].get(case)
        if c and c["after"].get("sem") is not None:
            print(f"  {case.upper()} matched control, {c['after']['n']} mocks: "
                  f"before {c['before']['mean']:+.3f} +- {c['before']['sem']:.3f}"
                  f"  ->  after {c['after']['mean']:+.3f} +- {c['after']['sem']:.3f}"
                  f"  (t = {c['after']['t']:+.2f}, p = {c['after']['p_two_sided']:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
