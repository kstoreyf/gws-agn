#!/usr/bin/env python3
"""Combined verdict for the ATTRIBUTION follow-up (tasks 1-3).

Collects
  * the sampler-vs-pdf prediction               (attr_sampler_ratio.json)
  * the before/after H0 scans                   (ctrl_*.json, fix_named_defect_*.json)
  * the quadrature oracle + convergence battery (attr_oracle_*.json)
and writes results/attr_fix_summary.json -- every number quoted in the
ATTRIBUTION.md appendix.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
ARMS = ["kde_gauss", "kde_exact", "delta_gauss", "delta_exact", "host_exact"]


def _j(name):
    p = RES / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def main():
    out = {"name": "attr_fix_summary",
           "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "scope": ("Seed 100, dark_sirens at log10n0=-24, field weighting, K=1, "
                     "targeted injections, W=4096 (GAL), campaign guard convention. "
                     "darksirens READ-ONLY at 2b86a2d; no generator edits, no "
                     "dataset regeneration.")}

    # ---------------- task 1 ------------------------------------------------
    S = _j("attr_sampler_ratio")
    if S:
        out["task1_sampler_vs_pdf"] = {
            "n_draws": S["draws"]["n_draws"],
            "validation": S["draws"]["validation_semi_vs_mc"],
            "peak_fraction_realised": S["draws"]["peak_fraction_realised"],
            "analytic_mass_on_grid": S["draws"]["analytic_mass_on_grid"],
            "semianalytic_mass_on_grid": S["draws"]["semianalytic_mass_on_grid"],
            "max_abs_log_ratio": S["draws"]["max_abs_log_ratio_semi"],
            "weighted_rms_log_ratio": S["draws"]["weighted_rms_log_ratio_semi"],
            "chieff_pull": [S["draws"]["chieff_pull_mean"], S["draws"]["chieff_pull_sd"]],
            "prediction": {t: {k: S["prediction"][t][k]
                               for k in ("mass", "rate", "pz", "jac", "tot")}
                           for t in S["prediction"]},
            "anchor_log_mu_absdiff": {t: S["prediction"][t]["anchor_log_mu_absdiff"]
                                      for t in S["prediction"]},
        }

    # ---------------- task 2 ------------------------------------------------
    scans = {}
    for tag in ("ctrl_gal_matched", "fix_named_defect_gal", "fix_named_defect_gal_m1",
                "ctrl_agn_matched", "fix_named_defect_agn", "fix_named_defect_agn_m1"):
        d = _j(tag)
        if d:
            h = d["H0"]
            scans[tag] = {"median": h["median"], "offset": h["median"] - 67.74,
                          "map": h["map"], "ci68": h["ci68"], "ci90": h["ci90"],
                          "truth_in_ci68": h["truth_in_ci68"],
                          "logL_max": d["logL_max"], "n_rejected": d["n_rejected"],
                          "guard": d["guard"]}
    ppe = _j("pe_corrected_events")
    arms_r = {t: _j(f"attr_mass_pe_{t}_s100") for t in ("gal", "agn")}
    pred = {}
    for t in ("gal", "agn"):
        if arms_r[t] is None or f"ctrl_{t}_matched" not in scans:
            continue
        r0 = arms_r[t]["r_table"]["none"]["tot"]
        off0 = scans[f"ctrl_{t}_matched"]["offset"]
        for arm, tag in (("m1m2", f"fix_named_defect_{t}"),
                         ("m1", f"fix_named_defect_{t}_m1")):
            if tag not in scans:
                continue
            r1 = arms_r[t]["r_table"][arm]["tot"]
            pred[tag] = {"r_before": r0, "r_after": r1,
                         "offset_before": off0,
                         "offset_predicted_linear": off0 * r1 / r0,
                         "offset_measured": scans[tag]["offset"]}
    out["task2_before_after"] = {"scans": scans, "score_arithmetic": pred,
                                 "reweighting_diagnostics": ppe["files"] if ppe else None}

    # ---------------- task 3 ------------------------------------------------
    orc = {}
    for t in ("gal", "agn"):
        d = _j(f"attr_oracle_{t}")
        if d is None:
            continue
        npz = np.load(RES / f"attr_oracle_{t}.npz")
        n = d["n_events"]
        rec = {"n_events": n, "dlnmu_dH0": d["dlnmu_dH0"],
               "anchors": {"log_mu_absdiff": d["anchor_log_mu_absdiff"],
                           "pz_reconstruction_maxabs":
                               d.get("anchor_pz_reconstruction_maxabs"),
                           "max_dN_miss": d.get("max_dN_miss"),
                           "ds_score_fd_vs_split_maxabs":
                               d.get("ds_score_fd_vs_split_full_maxabs")},
               "darksirens": d["darksirens"],
               "darksirens_pe_mc_sigma": d.get("darksirens_pe_mc_sigma"),
               "arms": d["arms"], "diagnostics": d["diagnostics"],
               "grid": d["grid"]}
        # validation: is (oracle - darksirens) consistent with darksirens' own MC?
        v = d["arms"]["kde_gauss"]["vs_darksirens"]
        sig = d.get("darksirens_pe_mc_sigma", {}).get("rms")
        if sig:
            resid = npz["score_kde_gauss"] - npz["ds_score"]
            rec["validation"] = {
                "mean_diff": v["mean"], "sem": v["sem"],
                "rms_diff": float(np.sqrt((resid ** 2).mean())),
                "rms_expected_from_ds_mc": sig,
                "ratio_rms": float(np.sqrt((resid ** 2).mean()) / sig),
                "pearson_r": float(np.corrcoef(npz["score_kde_gauss"],
                                               npz["ds_score"])[0, 1]),
                "mean_diff_in_sem": float(v["mean"] / v["sem"]) if v["sem"] else None,
            }
        # paired substitutions (zero Monte-Carlo error: same quadrature)
        def pair(a, b):
            x = npz[f"score_{a}"] - npz[f"score_{b}"]
            return {"mean": float(x.mean()),
                    "sem": float(x.std(ddof=1) / np.sqrt(x.size))}
        rec["substitutions"] = {
            "mass_model_exact_minus_stored__kde_prior": pair("kde_exact", "kde_gauss"),
            "mass_model_exact_minus_stored__delta_prior": pair("delta_exact", "delta_gauss"),
            "prior_delta_minus_kde__stored_masses": pair("delta_gauss", "kde_gauss"),
            "prior_delta_minus_kde__exact_masses": pair("delta_exact", "kde_exact"),
            "fully_exact_minus_anchor": pair("delta_exact", "kde_gauss"),
        }
        # what darksirens' own reweighted arms said about the same substitution
        if arms_r[t]:
            rec["darksirens_reweight_arm_delta_tot"] = (
                arms_r[t]["r_table"]["m1m2"]["tot"] - arms_r[t]["r_table"]["none"]["tot"])
            rec["darksirens_reweight_arm_delta_mass"] = (
                arms_r[t]["r_table"]["m1m2"]["mass"] - arms_r[t]["r_table"]["none"]["mass"])
            mp = np.load(RES / f"attr_mass_pe_{t}_s100.npz")
            idx = npz["idx"]
            ds_pair = (mp["ev_m1m2_tot"] - mp["ev_none_tot"])[idx]
            or_pair = npz["score_kde_exact"] - npz["score_kde_gauss"]
            rec["substitutions"]["mass_model_paired_cross_check"] = {
                "oracle_mean": float(or_pair.mean()),
                "oracle_sem": float(or_pair.std(ddof=1) / np.sqrt(or_pair.size)),
                "darksirens_reweight_mean_same_events": float(ds_pair.mean()),
                "darksirens_reweight_sem_same_events":
                    float(ds_pair.std(ddof=1) / np.sqrt(ds_pair.size)),
                "difference": float((or_pair - ds_pair).mean()),
                "difference_sem": float((or_pair - ds_pair).std(ddof=1)
                                        / np.sqrt(or_pair.size)),
                "darksirens_reweight_mean_all_events":
                    float((mp["ev_m1m2_tot"] - mp["ev_none_tot"]).mean()),
            }
        # convergence battery
        conv = {}
        for suf, lab in (("nz", "n_z doubled"), ("nm", "n_m doubled"),
                         ("sh", "grids shifted 0.37 cell"),
                         ("sky", "sky threshold 1e-7, n_gh 64")):
            c = _j(f"attr_oracle_{t}_conv_{suf}")
            if c is None:
                continue
            m = c["n_events"]
            base = {a: npz[f"score_{a}"][:m] for a in ARMS}
            cn = np.load(RES / f"attr_oracle_{t}_conv_{suf}.npz")
            conv[lab] = {"n_events": m, "grid": c["grid"],
                         "max_abs_score_shift": {
                             a: float(np.max(np.abs(cn[f"score_{a}"] - base[a])))
                             for a in ARMS},
                         "mean_score_shift": {
                             a: float(np.mean(cn[f"score_{a}"] - base[a]))
                             for a in ARMS},
                         "substitution_shift_exact_minus_stored_kde": float(
                             np.mean((cn["score_kde_exact"] - cn["score_kde_gauss"])
                                     - (base["kde_exact"] - base["kde_gauss"])))}
        rec["convergence"] = conv
        orc[t] = rec
    out["task3_oracle"] = orc

    (RES / "attr_fix_summary.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: (list(v) if isinstance(v, dict) else v)
                      for k, v in out.items()}, indent=1)[:2000])
    print(f"\nWrote {RES/'attr_fix_summary.json'}")


if __name__ == "__main__":
    main()
