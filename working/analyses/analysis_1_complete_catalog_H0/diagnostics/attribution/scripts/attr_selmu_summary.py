#!/usr/bin/env python3
"""TASK 1 -- the verdict: the exact selection oracle against darksirens' injections.

Collects ``attr_selmu_pdet``, ``attr_selmu_<tracer>`` and
``attr_selmu_inj_<tracer>_<lane>`` and forms the decisive comparisons:

  (i)  oracle-exact (delta, zero-bandwidth hosts) vs oracle-KDE  -> the KDE in mu
  (ii) both vs darksirens' injection-based d ln mu/dH0, on BOTH injection sets,
       across the H0 grid
  (iii) the discrepancy against the score residual it would have to explain:
       matched GAL  r = -8.2916e-4 of which -5.8433e-4 survives an exact numerator
       matched AGN  r = +1.9645e-3, offset +0.71 +- 0.20 km/s/Mpc
  (iv) the oracle's F(z) against the generator's OWN population-branch detection
       bookkeeping (~1e8 proposals), and the per-galaxy detection probability
       against the realised event-draw fractions in events_meta.json.

Outputs: results/attr_selmu_summary.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RES = ROOT / "results"
GEN = Path("/hildafs/projects/phy230014p/magana/gws-agn/working/data")
H0_FID = 67.74

# the numbers the selection side would have to explain (CLOSURE.md sections 6-7)
TARGET = {
    "gal": {"r_record_postfix": -8.2916e-4, "r_exact_numerator": -5.8433e-4,
            "curvature_per_event": -2.19e-4, "offset_km_s_Mpc": -2.996},
    "agn": {"r_record_postfix": +1.9645e-3, "r_exact_numerator": -2.7341e-3,
            "curvature_per_event": -2.148e-3, "offset_km_s_Mpc": +0.711},
}


def curvature_from_scan(tracer, half=4.0, at=H0_FID):
    """Per-event d2 logL/dH0^2 of the post-fix matched control, on its own grid."""
    import h5py
    p = RES / f"ctrl_{tracer}_matched.h5"
    n = {"gal": 720, "agn": 280}[tracer]
    with h5py.File(p, "r") as f:
        h = f["H0_grid"][:]
        L = f["log_likelihood"][:]
    out = {}
    for lab, x0 in (("truth", at), ("map", float(h[np.argmax(L)]))):
        m = np.abs(h - x0) < half
        c = np.polyfit(h[m] - x0, L[m], 2)
        out[lab] = {"d2_total": float(2 * c[0]), "d2_per_event": float(2 * c[0] / n),
                    "d1_per_event": float(c[1] / n), "H0": x0}
    return out


def load(p):
    return json.loads(Path(p).read_text()) if Path(p).exists() else None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lanes", nargs="+", default=["targeted", "popuni"])
    ap.add_argument("--outdir", default=str(RES))
    args = ap.parse_args(argv)
    od = Path(args.outdir)

    out = {"name": "attr_selmu_summary",
           "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "pdet": None, "tracers": {}}

    pd = load(RES / "attr_selmu_pdet.json")
    if pd:
        out["pdet"] = {
            "chieff_absent_from_detection": pd["chieff_absent_from_detection_source"],
            "quadrature_convergence": pd["quadrature_convergence"],
            "tq_reduction_maxabs": pd["tq_reduction_maxabs"],
            "clip_probability_max": pd["clip_probability_max"],
            "brute_force_max_abs_diff": pd["brute_force"]["max_abs_diff"],
            "brute_force_max_abs_pull": pd["brute_force"]["max_abs_pull"],
            "brute_force_n_per_point": pd["brute_force"]["n_per_point"],
            "brute_force_n_points": len(pd["brute_force"]["points"])}

    ev_meta = json.loads((GEN / "seed100" / "events" / "events_meta.json").read_text())
    real = ev_meta["realised"]
    per_gal = {}
    for tr in ("gal", "agn"):
        O = load(RES / f"attr_selmu_{tr}.json")
        if O is None:
            continue
        rec = {"oracle": {"dlnmu_at_truth": O["dlnmu_at_truth"],
                          "H0_grid": O["arms"]["kde"]["H0"],
                          "dlnmu_kde": O["arms"]["kde"]["dlnmu"],
                          "dlnmu_delta": O["arms"]["delta"]["dlnmu"],
                          "dlnmu_unif": O["arms"]["unif"]["dlnmu"],
                          "dlnmu_norate": O["arms"]["norate"]["dlnmu"],
                          "anchors": O["anchors"],
                          "G_convergence": O.get("G_convergence", {}),
                          "lattice_convergence": O.get("lattice_convergence", {}),
                          "fd_step_halving": O["fd_step_halving"]},
               "kde_in_mu__kde_minus_delta":
                   O["dlnmu_at_truth"]["kde"] - O["dlnmu_at_truth"]["delta"],
               "hostprior_in_mu__unif_minus_delta":
                   O["dlnmu_at_truth"]["unif"] - O["dlnmu_at_truth"]["delta"],
               "rate_in_mu__norate_minus_delta":
                   O["dlnmu_at_truth"]["norate"] - O["dlnmu_at_truth"]["delta"],
               "injections": {}}
        per_gal[tr] = O["per_galaxy_detection_probability_at_truth"]
        for lane in args.lanes:
            I = load(RES / f"attr_selmu_inj_{tr}_{lane}.json")
            if I is None:
                continue
            fd = I["fd_at_truth"]
            key = min(fd, key=lambda k: float(k)) if fd else None
            best = float(fd[key]) if key else None
            # the Richardson limit of the step-halving sequence
            ks = sorted(fd, key=lambda k: -float(k))
            rich = None
            if len(ks) >= 2:
                h1, h2 = float(ks[-2]), float(ks[-1])
                rich = (4.0 * fd[ks[-1]] - fd[ks[-2]]) / 3.0
            g = I["grid_fd_dh0p5"]
            rec["injections"][lane] = {
                "anchor_log_mu_absdiff": I["anchor_log_mu_absdiff"],
                "fd_at_truth": fd, "fd_richardson": rich,
                "term_sum_at_truth": I.get("term_sum_at_truth", {}),
                "grid_fd": g,
                "Neff_at_truth": I["per_H0"][str(H0_FID)]["Neff"],
                "mc_error_of_dlnmu": I.get("mc_error_of_dlnmu", {}),
                "prior_wt_minus_file_pdraw_maxabs":
                    I.get("prior_wt_minus_file_pdraw_maxabs"),
                "branch_only_fd": {k: v for k, v in
                                   I.get("branch_only_fd_at_truth", {}).items()
                                   if not k.startswith("_")},
                "delta_vs_oracle_kde_at_truth":
                    (best - O["dlnmu_at_truth"]["kde"]) if best is not None else None,
                "delta_vs_oracle_delta_at_truth":
                    (best - O["dlnmu_at_truth"]["delta"]) if best is not None else None,
            }
            # across the grid
            gg = {}
            H = O["arms"]["kde"]["H0"]
            for h, v in g.items():
                if float(h) in H:
                    i = H.index(float(h))
                    gg[h] = {"inj": v, "oracle_kde": O["arms"]["kde"]["dlnmu"][i],
                             "diff": v - O["arms"]["kde"]["dlnmu"][i]}
            rec["injections"][lane]["grid_comparison"] = gg
            # The generator's OWN population-branch detection bookkeeping is a
            # direct, likelihood-free measurement of F(z) on ~1e8 proposals:
            # ndet | npro ~ Binomial(npro, <F>_bin).
            pz = I.get("pdet_z_empirical", {})
            if pz and (RES / f"attr_selmu_{tr}.npz").exists():
                d = np.load(RES / f"attr_selmu_{tr}.npz")
                e = np.asarray(pz["edges"], float)
                npro = np.asarray(pz["n_proposed"], float)
                ndet = np.asarray(pz["n_detected"], float)
                bg, Gv = d["b_grid"], d["G"]
                zk, bk = d["zk"], d["b_zk"]
                nq = 9
                gx, gw = np.polynomial.legendre.leggauss(nq)
                lo, hi = e[:-1], e[1:]
                zz = (0.5 * (hi - lo)[:, None] * gx[None, :]
                      + 0.5 * (hi + lo)[:, None]).ravel()
                Fq = np.interp(np.interp(zz, zk, bk), bg, Gv).reshape(-1, nq)
                Fbin = (Fq * gw[None, :]).sum(1) / 2.0
                pred = npro * Fbin
                ok = ndet >= 25
                var = np.maximum(npro * Fbin * (1.0 - Fbin), 1e-12)
                pull = (ndet - pred) / np.sqrt(var)
                rec["injections"][lane]["Fz_vs_generator_population_branch"] = {
                    "n_proposed_total": float(npro.sum()),
                    "n_detected_total": float(ndet.sum()),
                    "n_predicted_total": float(pred.sum()),
                    "ratio_detected_over_predicted": float(ndet.sum() / pred.sum()),
                    "sigma_of_ratio": float(np.sqrt(ndet.sum()) / pred.sum()),
                    "n_sigma": float((ndet.sum() - pred.sum()) / np.sqrt(ndet.sum())),
                    "n_bins_ge25": int(ok.sum()),
                    "binomial_chi2": float((pull[ok] ** 2).sum()),
                    "mean_pull": float(np.mean(pull[ok])),
                    "sd_pull": float(np.std(pull[ok])),
                    "max_abs_pull": float(np.max(np.abs(pull[ok])))}
        out["tracers"][tr] = rec

    # the generative cross-check: the event draw's own detected fractions
    if set(per_gal) == {"gal", "agn"}:
        fa = float(ev_meta["planted_f_agn"])
        pred_acc = (1 - fa) * per_gal["gal"]["unif"] + fa * per_gal["agn"]["unif"]
        pred_snr = (1 - fa) * per_gal["gal"]["norate"] + fa * per_gal["agn"]["norate"]
        n = float(real["n_proposed"])
        out["generative_cross_check"] = {
            "n_proposed": n,
            "detected_fraction_measured": real["detected_fraction"],
            "detected_fraction_oracle": pred_acc,
            "detected_fraction_sigma": float(
                np.sqrt(real["detected_fraction"] / n)),
            "detected_fraction_pull": float(
                (real["detected_fraction"] - pred_acc)
                / np.sqrt(real["detected_fraction"] / n)),
            "snr_only_measured": real["detected_fraction_snr_only"],
            "snr_only_oracle": pred_snr,
            "snr_only_pull": float(
                (real["detected_fraction_snr_only"] - pred_snr)
                / np.sqrt(real["detected_fraction_snr_only"] / n)),
            "per_galaxy": per_gal, "planted_f_agn": fa}

    out["targets"] = TARGET
    out["curvature"] = {tr: curvature_from_scan(tr) for tr in out["tracers"]}
    for tr, rec in out["tracers"].items():
        d2 = out["curvature"][tr]["truth"]["d2_per_event"]
        for lane, ir in rec["injections"].items():
            d = ir.get("delta_vs_oracle_kde_at_truth")
            if d is None:
                continue
            t = TARGET[tr]
            # r_darksirens - r_true = (exact d ln mu/dH0) - (estimated) = -d.
            # A COMMON-MODE shift of r over every event, hence of the H0 offset.
            ir["r_bias_from_selection_estimator"] = -d
            ir["explains_fraction_of_r_record"] = -d / t["r_record_postfix"]
            if t["r_exact_numerator"]:
                ir["explains_fraction_of_r_exact_numerator"] = \
                    -d / t["r_exact_numerator"]
            ir["km_s_Mpc_from_selection_estimator"] = -d / abs(d2)
            mce = ir.get("mc_error_of_dlnmu", {}).get("0.5", {})
            if mce:
                ir["mc_error_bootstrap_sd"] = mce.get("bootstrap_sd")
                ir["mc_error_delta_method_sd"] = mce.get("delta_method_sd")
                ir["offset_in_sigma_of_its_own_MC_error"] = (
                    d / mce["bootstrap_sd"] if mce.get("bootstrap_sd") else None)
                ir["km_s_Mpc_uncertainty_from_selection_MC"] = \
                    mce["bootstrap_sd"] / abs(d2)
    (od / "attr_selmu_summary.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "tracers"}, indent=1)[:4000])
    for tr, rec in out["tracers"].items():
        print(f"\n=== {tr.upper()} ===")
        print(f"  oracle  d ln mu/dH0   kde={rec['oracle']['dlnmu_at_truth']['kde']:+.8e}"
              f"  delta={rec['oracle']['dlnmu_at_truth']['delta']:+.8e}"
              f"  unif={rec['oracle']['dlnmu_at_truth']['unif']:+.8e}")
        print(f"  KDE-in-mu (kde - delta)          = "
              f"{rec['kde_in_mu__kde_minus_delta']:+.5e}")
        print(f"  host prior (unif - delta)        = "
              f"{rec['hostprior_in_mu__unif_minus_delta']:+.5e}")
        print(f"  rate factor (norate - delta)     = "
              f"{rec['rate_in_mu__norate_minus_delta']:+.5e}")
        d2 = out["curvature"][tr]["truth"]["d2_per_event"]
        print(f"  per-event curvature at truth d2 logL/dH0^2 = {d2:+.4e}")
        for lane, ir in rec["injections"].items():
            sd = ir.get("mc_error_bootstrap_sd")
            print(f"  injections[{lane}]: fd={ir['fd_richardson']:+.8e}"
                  + (f" +- {sd:.3e}" if sd else "")
                  + f"  vs oracle_kde {ir['delta_vs_oracle_kde_at_truth']:+.3e}"
                  + (f" ({ir['offset_in_sigma_of_its_own_MC_error']:+.2f} sigma of "
                     f"its own MC error)" if sd else "")
                  + f"\n      -> r bias {ir['r_bias_from_selection_estimator']:+.3e} "
                    f"= {ir.get('explains_fraction_of_r_record', float('nan'))*100:.1f}% "
                    f"of r_record = "
                    f"{ir['km_s_Mpc_from_selection_estimator']:+.3f} km/s/Mpc"
                  + (f";  the estimator's own MC error is "
                     f"+-{ir['km_s_Mpc_uncertainty_from_selection_MC']:.3f} km/s/Mpc"
                     if sd else ""))
            fz = ir.get("Fz_vs_generator_population_branch")
            if fz:
                print(f"      F(z) vs generator ({fz['n_proposed_total']:.3e} "
                      f"proposals): detected {fz['n_detected_total']:.0f} vs "
                      f"predicted {fz['n_predicted_total']:.1f}, ratio "
                      f"{fz['ratio_detected_over_predicted']:.5f} +- "
                      f"{fz['sigma_of_ratio']:.5f} ({fz['n_sigma']:+.2f} sigma); "
                      f"binomial chi2 {fz['binomial_chi2']:.1f}/"
                      f"{fz['n_bins_ge25']}")
            bf = ir.get("branch_only_fd")
            if bf:
                print("      per-branch d ln mu/dH0: "
                      + "  ".join(f"{k}:{v:+.6e}" for k, v in bf.items()))
    print(f"\nWrote {od/'attr_selmu_summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
