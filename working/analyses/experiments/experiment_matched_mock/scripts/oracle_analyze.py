#!/usr/bin/env python3
"""Combine the exact oracle and the eager darksirens mirror; attribute the bias.

Inputs (per tag): results/oracle_num_<tag>.npz  (oracle numerator ladder)
                  results/oracle_mu_<tag>.npz   (exact selection function)
                  results/oracle_dsref_<tag>.npz (darksirens per-event + selection)
Writes results/oracle_report_<tag>.json and prints the attribution table.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

EXP = Path(__file__).resolve().parents[1]
H0_TRUE = 67.74


def quad_peak(x, y):
    i = int(np.argmax(y))
    if i in (0, len(y) - 1):
        return float(x[i])
    d = y[i - 1] - 2 * y[i] + y[i + 1]
    return float(x[i] - 0.5 * (y[i + 1] - y[i - 1]) / d * (x[1] - x[0]))


def curvature_at_peak(x, y):
    i = int(np.argmax(y))
    i = min(max(i, 8), len(x) - 9)
    h = x[1] - x[0]
    return float(-(y[i + 8] - 2 * y[i] + y[i - 8]) / (8 * h) ** 2)


def slope_at(x, y, x0, half=2):
    i = int(np.argmin(np.abs(x - x0)))
    h = x[1] - x[0]
    return float((y[i + half] - y[i - half]) / (2 * half * h))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    args = ap.parse_args(argv)
    tag = args.tag

    num = np.load(EXP / "results" / f"oracle_num_{tag}.npz")
    mu = np.load(EXP / "results" / f"oracle_mu_{tag}.npz")
    ds = np.load(EXP / "results" / f"oracle_dsref_{tag}.npz")
    H0 = num["H0"]
    assert np.allclose(H0, ds["H0"]) and np.allclose(H0, mu["H0"])
    N = num["ln_O1"].shape[0]

    ds_num = ds["event_lls"].sum(axis=1)              # (161,)
    ds_num_corr = (ds["event_lls"] + 0.5 * ds["event_vars"]).sum(axis=1)
    farr = N * (N + 3) / (2.0 * ds["Neff"])
    ds_sel = -N * ds["log_mu"] + farr
    ds_tot = ds["total"]

    or_mu = mu["ln_mu"]
    or_tot = {v: num[f"ln_{v}"].sum(axis=0) - N * or_mu
              for v in ("O1", "O2", "O3", "O3b", "O4")}

    rep = {"tag": tag, "nEvents": int(N)}
    curv = curvature_at_peak(H0, ds_tot)
    rep["curvature_ds_total"] = curv
    rep["peak_ds_total"] = quad_peak(H0, ds_tot)
    rep["peak_oracle_total_O1"] = quad_peak(H0, or_tot["O1"])
    rep["curvature_oracle_total_O1"] = curvature_at_peak(H0, or_tot["O1"])
    rep["offset_ds_total"] = rep["peak_ds_total"] - H0_TRUE
    rep["offset_oracle_total_O1"] = rep["peak_oracle_total_O1"] - H0_TRUE
    rep["paired_ds_minus_oracle"] = rep["peak_ds_total"] - rep["peak_oracle_total_O1"]

    # ---- slope attribution at truth (nats per km/s/Mpc); dH0 = slope/curv ----
    def att(name, y):
        s = slope_at(H0, y, H0_TRUE)
        rep[f"slope_{name}"] = s
        rep[f"dH0_{name}"] = s / curv
        print(f"  {name:34s} slope {s:+9.3f} nats/km  -> dH0 {s/curv:+7.3f}")

    print(f"=== {tag}: attribution of (darksirens - exact oracle) at truth ===")
    print(f"  curvature (ds total) = {curv:.3f} nats/(km/s/Mpc)^2")
    att("num_ladder_O2_minus_O1", (num["ln_O2"] - num["ln_O1"]).sum(0))
    att("num_ladder_O3_minus_O2", (num["ln_O3"] - num["ln_O2"]).sum(0))
    att("num_ladder_O3b_minus_O3", (num["ln_O3b"] - num["ln_O3"]).sum(0))
    att("num_ladder_O4_minus_O3b", (num["ln_O4"] - num["ln_O3b"]).sum(0))
    att("num_ds_minus_O4_residual", ds_num - num["ln_O4"].sum(0))
    att("num_ds_minus_O4_mcbias_corr", ds_num_corr - num["ln_O4"].sum(0))
    att("num_total_ds_minus_O1", ds_num - num["ln_O1"].sum(0))
    att("sel_mu_ds_minus_exact", -N * (ds["log_mu"] - or_mu))
    att("sel_farr_term", farr)
    att("sel_total_ds_minus_exact", ds_sel + N * or_mu)
    att("grand_total_ds_minus_oracle", ds_tot - or_tot["O1"])

    print()
    print(f"  peak ds total          : {rep['peak_ds_total']:.3f} "
          f"(offset {rep['offset_ds_total']:+.3f})")
    print(f"  peak oracle exact total: {rep['peak_oracle_total_O1']:.3f} "
          f"(offset {rep['offset_oracle_total_O1']:+.3f})")
    print(f"  paired ds - oracle     : {rep['paired_ds_minus_oracle']:+.3f}")
    for v in ("O2", "O3", "O3b", "O4"):
        rep[f"peak_oracle_total_{v}"] = quad_peak(H0, or_tot[v])

    # numerator-only peaks
    rep["peak_num_ds"] = quad_peak(H0, ds_num)
    rep["peak_num_O1"] = quad_peak(H0, num["ln_O1"].sum(0))
    rep["peak_num_O4"] = quad_peak(H0, num["ln_O4"].sum(0))
    # selection slopes
    rep["dlnmu_dH0_ds"] = slope_at(H0, ds["log_mu"], H0_TRUE)
    rep["dlnmu_dH0_exact"] = slope_at(H0, np.asarray(or_mu), H0_TRUE)
    rep["Neff_at_truth"] = float(ds["Neff"][np.argmin(np.abs(H0 - H0_TRUE))])
    print(f"  d ln mu/dH0: ds {rep['dlnmu_dH0_ds']:+.5f}  "
          f"exact {rep['dlnmu_dH0_exact']:+.5f}  "
          f"diff x(-N) -> {-N*(rep['dlnmu_dH0_ds']-rep['dlnmu_dH0_exact']):+.3f} nats/km")

    out = EXP / "results" / f"oracle_report_{tag}.json"
    out.write_text(json.dumps(rep, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
