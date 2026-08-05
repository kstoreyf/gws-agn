#!/usr/bin/env python3
"""Assemble the H0-tilt mechanism budget from tilt_terms_*.h5.

For each planted f, forms the model curves
    N(H0)  per-event numerator (variant family)
    S(H0)  selection term  -N_obs ln mu + N(N+3)/(2 Neff)   (from lnmu, Neff)
and reports quad-refined peak shifts of counterfactual totals:

    total offset            peak(N + S) - H0_true
    selection mechanism     peak(N) - peak(N + S)          [flat-beta repair]
    z>1 catalog leak        peak(N_zcut1 + S) - peak(N + S)
    PE MC (delta) bias      peak(N + S + 0.5*sum sigma^2) - peak(N + S)
    spectral vs catalog     slopes of frozen_mass / frozen_cat at truth
    host-type split         numerator peaks of GAL-hosted vs AGN-hosted events

Also validates against the measured decomposition and the measured z<=1
catalog-truncation fact, and writes results/tilt_budget.json.
"""
import json
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
H0_TRUE = 67.74
NOBS = 1000


def quad_peak(x, y):
    y = np.asarray(y, float)
    ok = np.isfinite(y)
    i = int(np.nanargmax(np.where(ok, y, np.nan)))
    if i in (0, len(y) - 1) or not (ok[i - 1] and ok[i + 1]):
        return float(x[i])
    d = y[i - 1] - 2 * y[i] + y[i + 1]
    if d == 0 or not np.isfinite(d):
        return float(x[i])
    return float(x[i] - 0.5 * (y[i + 1] - y[i - 1]) / d * (x[1] - x[0]))


def slope_at(x, y, x0):
    g = np.gradient(y, x)
    return float(np.interp(x0, x, g))


def curvature_at_peak(x, y):
    xp = quad_peak(x, y)
    g2 = np.gradient(np.gradient(y, x), x)
    return float(np.interp(xp, x, g2))


def sel_term(lnmu, neff):
    return -NOBS * lnmu + NOBS * (NOBS + 3.0) / (2.0 * neff)


def analyze(tag, f_planted, dec_json):
    with h5py.File(ROOT / "results" / f"tilt_terms_{tag}.h5", "r") as fh:
        H0 = fh["H0_grid"][:]
        num = {k: fh[f"numerator/{k}"][:] for k in fh["numerator"]}
        lnmu = {k: fh[f"lnmu/{k}"][:] for k in fh["lnmu"]}
        lnZ_ev = fh["lnZ_ev"][:]
        s2_ev = fh["sigma2_ev"][:]
        agn_share = fh["agn_share_ev"][:]
        frac_beyond = {k: fh[f"frac_beyond/{k}"][:] for k in fh["frac_beyond"]}
        clip = fh["clip_frac"][:]
        neff = fh["sel_neff"][:]
        host = fh["host_type"][:]
        true_z = fh["true_z"][:]
        validation = json.loads(fh.attrs["validation"])

    dec = json.loads((ROOT / "results" / dec_json).read_text())

    N = num["full"]
    S = sel_term(lnmu["full"], neff)
    total = N + S
    p_tot = quad_peak(H0, total)
    p_num = quad_peak(H0, N)
    curv = curvature_at_peak(H0, total)

    out = {
        "f_planted": f_planted,
        "H0_grid": [float(H0[0]), float(H0[-1]), int(H0.size)],
        "validation_vs_decomposition": validation,
        "measured_total_offset_dec": dec["offset_total"],
        "peaks": {},
        "shifts_km_s_Mpc": {},
        "slopes_nats_per_km": {},
        "diagnostics": {},
    }

    # ---- headline peaks -----------------------------------------------------
    out["peaks"]["total_full"] = p_tot
    out["peaks"]["numerator_full"] = p_num
    out["peaks"]["total_offset"] = p_tot - H0_TRUE
    out["peaks"]["numerator_offset"] = p_num - H0_TRUE
    out["peaks"]["selection_shift"] = p_tot - p_num

    # ---- counterfactual totals ---------------------------------------------
    sh = out["shifts_km_s_Mpc"]
    for zc in ("1", "1.1", "1.2", "1.3", "1.4"):
        key = f"zcut_{zc}"
        if key in num:
            sh[f"numerator_{key}"] = quad_peak(H0, num[key]) - p_num
            sh[f"total_{key}"] = quad_peak(H0, num[key] + S) - p_tot
    # z<=1 cut on BOTH numerator and selection (the measured truncation analog)
    if "zcut_1" in num and "zcut_1" in lnmu:
        S_z1 = sel_term(lnmu["zcut_1"], neff)
        sh["total_zcut1_both"] = quad_peak(H0, num["zcut_1"] + S_z1) - p_tot
    # MC delta-method bias correction
    corr = total + 0.5 * s2_ev.sum(axis=0)
    sh["mc_bias_correction"] = quad_peak(H0, corr) - p_tot
    out["diagnostics"]["sigma2_sum_at_truth"] = float(
        np.interp(H0_TRUE, H0, s2_ev.sum(axis=0)))
    out["diagnostics"]["sigma2_sum_slope"] = slope_at(
        H0, s2_ev.sum(axis=0), H0_TRUE)

    # ---- slope decomposition at truth --------------------------------------
    sl = out["slopes_nats_per_km"]
    sl["numerator_full"] = slope_at(H0, N, H0_TRUE)
    sl["numerator_frozen_mass"] = slope_at(H0, num["frozen_mass"], H0_TRUE)
    sl["numerator_frozen_cat"] = slope_at(H0, num["frozen_cat"], H0_TRUE)
    sl["selection_term"] = slope_at(H0, S, H0_TRUE)
    sl["lnmu"] = slope_at(H0, lnmu["full"], H0_TRUE)
    sl["lnmu_frozen_mass"] = slope_at(H0, lnmu["frozen_mass"], H0_TRUE)
    sl["lnmu_frozen_cat"] = slope_at(H0, lnmu["frozen_cat"], H0_TRUE)
    sl["total"] = slope_at(H0, total, H0_TRUE)
    out["diagnostics"]["total_curvature_at_peak"] = curv
    # implied peak-shift per unit slope
    for k in ("numerator_frozen_mass", "numerator_frozen_cat"):
        sh[f"implied_{k}"] = -sl[k] / curv

    # ---- single-tracer numerators and per-host splits -----------------------
    for nm in ("gal_only", "agn_only"):
        y = num[nm]
        out["peaks"][f"numerator_{nm}_prior"] = (
            quad_peak(H0, y) if np.isfinite(y).any() else None)
    for name, mask in (("galhost", host == 0), ("agnhost", host == 1)):
        yy = lnZ_ev[mask].sum(axis=0)
        out["peaks"][f"numerator_{name}"] = quad_peak(H0, yy)
        out["diagnostics"][f"n_{name}"] = int(mask.sum())
        out["diagnostics"][f"true_z_median_{name}"] = float(
            np.median(true_z[mask]))
    # host-split totals: event-group numerator + group-share of S
    for name, mask in (("galhost", host == 0), ("agnhost", host == 1)):
        n_g = int(mask.sum())
        S_g = (-n_g * lnmu["full"]) + n_g * (n_g + 3.0) / (2.0 * neff)
        out["peaks"][f"total_{name}"] = quad_peak(
            H0, lnZ_ev[mask].sum(axis=0) + S_g)

    # ---- weighted PE mass beyond z cuts ------------------------------------
    i_tru = int(np.argmin(np.abs(H0 - H0_TRUE)))
    for zc, arr in frac_beyond.items():
        mean_frac = arr.mean(axis=0)
        out["diagnostics"][f"mean_pe_massfrac_z_gt_{zc}"] = {
            "at_truth": float(mean_frac[i_tru]),
            "at_60": float(np.interp(60.0, H0, mean_frac)),
            "at_75": float(np.interp(75.0, H0, mean_frac)),
        }
    out["diagnostics"]["clip_frac_at_truth"] = float(clip[i_tru])
    out["diagnostics"]["clip_frac_at_75"] = float(np.interp(75.0, H0, clip))
    out["diagnostics"]["agn_share_mean_at_truth"] = float(
        agn_share[:, i_tru].mean())
    out["diagnostics"]["agn_share_mean_agnhost"] = float(
        agn_share[host == 1, i_tru].mean())
    out["diagnostics"]["agn_share_mean_galhost"] = float(
        agn_share[host == 0, i_tru].mean())

    # ---- budget assembly ----------------------------------------------------
    # decomposition of the total offset into: numerator offset + selection shift
    # then numerator offset into: z>1 leak + MC bias + remainder
    leak = sh.get("total_zcut_1", np.nan)
    out["budget"] = {
        "total_offset": p_tot - H0_TRUE,
        "selection_shift_full": p_tot - p_num,
        "numerator_offset": p_num - H0_TRUE,
        "numerator_leak_z_gt_1": sh.get("numerator_zcut_1"),
        "mc_bias": sh["mc_bias_correction"],
        "numerator_remainder": (p_num - H0_TRUE)
        - (sh.get("numerator_zcut_1") or 0.0) - sh["mc_bias_correction"],
    }
    return out, dict(H0=H0, N=N, S=S, total=total, num=num, lnmu=lnmu,
                     neff=neff, s2sum=s2_ev.sum(axis=0),
                     lnZ_ev=lnZ_ev, host=host,
                     frac_beyond=frac_beyond)


def main():
    results = {}
    curves = {}
    for tag, f, dec in (("fagn0.3", 0.307, "h0_decomposition_fagn0.3.json"),
                        ("fagn0.7", 0.703, "h0_decomposition_fagn0.7.json")):
        results[tag], curves[tag] = analyze(tag, f, dec)

    outp = ROOT / "results" / "tilt_budget.json"
    outp.write_text(json.dumps(results, indent=2, default=float))
    print(f"wrote {outp}\n")

    for tag, r in results.items():
        print(f"===== {tag} (planted f={r['f_planted']}) =====")
        print(f" measured total offset (dec)   : {r['measured_total_offset_dec']:+.3f}")
        print(f" model  total offset           : {r['peaks']['total_offset']:+.3f}")
        print(f" numerator offset              : {r['peaks']['numerator_offset']:+.3f}")
        print(f" selection shift               : {r['peaks']['selection_shift']:+.3f}")
        for k, v in r["shifts_km_s_Mpc"].items():
            print(f"   {k:28s}: {v:+.3f}")
        print(" slopes at truth (nats/km/s/Mpc):")
        for k, v in r["slopes_nats_per_km"].items():
            print(f"   {k:28s}: {v:+.4f}")
        print(" budget:", json.dumps(r["budget"], indent=2, default=float))
        print()


if __name__ == "__main__":
    main()
