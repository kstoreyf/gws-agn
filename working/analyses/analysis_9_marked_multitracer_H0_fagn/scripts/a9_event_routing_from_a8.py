#!/usr/bin/env python3
"""Analysis 9 mechanism follow-up, part A -- anchor-point event routing.

    python scripts/a9_event_routing_from_a8.py

Deterministic, CPU-only, free: it evaluates no likelihood and reads one recorded
file, Analysis 8's per-event evidence decomposition at the joint MAP,

    analysis_8_marked_multitracer_H0_fagn/results/event_decomposition.h5

which holds, for each of the 1000 events at (H0 = 67.74, f_AGN = 0.275,
dmu_chi = +0.1075), the four hybrid evidences log E[GG], log E[AG], log E[GA],
log E[AA] and the two Bayes factors they factor into.  Analysis 8 is READ-ONLY
here: sys.dont_write_bytecode is set before its directory is touched and every
write goes through _write(), which refuses any path outside Analysis 9.

What the numbers mean.  The per-event likelihood is the two-branch mixture
Z_i = f_GAL E_i[GG] + f_AGN E_i[AA], so the posterior probability that event i
came from the AGN branch is

    P_i(AGN) = sigmoid( log(f_AGN/f_GAL) + log BF_total ),
    log BF_total = log E_i[AA] - log E_i[GG].

In the SPATIAL-ONLY model the two branches share one intrinsic population, so
the AGN branch's evidence is the hybrid E_i[AG] (AGN sky/redshift factor, GAL
intrinsic factor) and the same expression runs with

    log BF_spatial = log E_i[AG] - log E_i[GG]

instead.  The difference of the two routings, delta_P_i, is what the spin mark
does to the host assignment of event i at the anchor point, and

    log BF_total = log BF_spatial + log BF_intrinsic

exactly, so delta_P_i is carried entirely by the intrinsic (mark) factor.

true_host_type is DIAGNOSTIC ONLY.  It enters no inference quantity in this
file; it is used once, at the end, to split the delta_P distribution for the
reader, and the split is labelled as such.

The output pair

    diagnostics/a9_event_routing_h0_67p74_from_a8.h5
    diagnostics/a9_event_routing_h0_67p74_from_a8.json

uses exactly the interface of diagnostics/a9_event_routing.h5 (the same
quantities computed at H0 = 69.0), so figure code can read either file.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np

sys.dont_write_bytecode = True

A9 = Path(__file__).resolve().parent.parent
DIAG = A9 / "diagnostics"
A8_DIR = A9.parent / "analysis_8_marked_multitracer_H0_fagn"
A8_DECOMP = A8_DIR / "results" / "event_decomposition.h5"

H0_REP = 67.74      # Analysis 8 ran at this fixed cosmology (arm_J_joint H0_fixed)
F_AGN = 0.275
MU_CHI = 0.1075
PCTS = (1, 5, 16, 50, 84, 95, 99)


def _write_guard(path: Path) -> Path:
    path = Path(path).resolve()
    if A9 not in path.parents:
        raise RuntimeError(f"[fatal] refusing to write outside {A9}: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic."""
    x = np.asarray(x, float)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    e = np.exp(x[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def _stats(dp: np.ndarray, ps: np.ndarray, pm: np.ndarray) -> dict:
    return {
        "n_events": int(dp.size),
        "percentiles_delta_P": {str(q): float(np.percentile(dp, q)) for q in PCTS},
        "median_delta_P": float(np.median(dp)),
        "median_abs_delta_P": float(np.median(np.abs(dp))),
        "rms_abs_delta_P": float(np.sqrt(np.mean(dp ** 2))),
        "mean_delta_P": float(np.mean(dp)),
        "max_abs_delta_P": float(np.max(np.abs(dp))),
        "n_abs_delta_P_gt_0p1": int((np.abs(dp) > 0.1).sum()),
        "n_abs_delta_P_gt_0p2": int((np.abs(dp) > 0.2).sum()),
        "n_crossing_0p5_upward": int(((ps < 0.5) & (pm >= 0.5)).sum()),
        "n_crossing_0p5_downward": int(((ps >= 0.5) & (pm < 0.5)).sum()),
        "n_crossing_0p5_either_way": int(((ps < 0.5) != (pm < 0.5)).sum()),
        "sum_P_spatial": float(ps.sum()),
        "sum_P_marked": float(pm.sum()),
        "sum_P_marked_minus_spatial": float(pm.sum() - ps.sum()),
        "n_P_spatial_ge_0p5": int((ps >= 0.5).sum()),
        "n_P_marked_ge_0p5": int((pm >= 0.5).sum()),
    }


def main() -> dict:
    with h5py.File(A8_DECOMP, "r") as h:
        ev = np.asarray(h["event_index"][:], np.int64)
        gg = np.asarray(h["logE_GG"][:], float)
        ag = np.asarray(h["logE_AG"][:], float)
        ga = np.asarray(h["logE_GA"][:], float)
        aa = np.asarray(h["logE_AA"][:], float)
        bf_sp = np.asarray(h["log_BF_spatial"][:], float)
        bf_in = np.asarray(h["log_BF_intrinsic"][:], float)
        bf_to = np.asarray(h["log_BF_total"][:], float)
        p_rec = np.asarray(h["P_AGN"][:], float)
        host = np.asarray(h["true_host_type"][:], np.int64)
        log_f_gal = float(h.attrs["log_f_gal"])
        log_f_agn = float(h.attrs["log_f_agn"])
        f_attr = float(h.attrs["f_agn"])
        mu_attr = float(h.attrs["mu_chi_c2"])

    if abs(f_attr - F_AGN) >= 1e-12 or abs(mu_attr - MU_CHI) >= 1e-12:
        raise RuntimeError(f"[fatal] the recorded anchor is f = {f_attr}, "
                           f"mu = {mu_attr}, not ({F_AGN}, {MU_CHI})")

    lam = log_f_agn - log_f_gal
    P_spatial = _sigmoid(lam + bf_sp)
    P_marked = _sigmoid(lam + bf_to)
    dP = P_marked - P_spatial

    # closures on the recorded file, so a corrupted read cannot pass silently
    closure = {
        "max_abs_P_marked_minus_recorded_P_AGN":
            float(np.max(np.abs(P_marked - p_rec))),
        "max_abs_sum_rule_residual_logBF":
            float(np.max(np.abs(bf_sp + bf_in - bf_to))),
        "max_abs_logBF_spatial_minus_AG_minus_GG":
            float(np.max(np.abs(bf_sp - (ag - gg)))),
        "max_abs_logBF_total_minus_AA_minus_GG":
            float(np.max(np.abs(bf_to - (aa - gg)))),
        "log_f_gal_vs_log1m_f": float(log_f_gal - np.log1p(-F_AGN)),
        "log_f_agn_vs_log_f": float(log_f_agn - np.log(F_AGN)),
    }
    for k, v in closure.items():
        if abs(v) > 1e-9:
            raise RuntimeError(f"[fatal] closure {k} = {v:.3e} exceeds 1e-9")

    n = ev.size
    h5_path = _write_guard(DIAG / "a9_event_routing_h0_67p74_from_a8.h5")
    with h5py.File(h5_path, "w") as h:
        h.create_dataset("event_index", data=ev)
        h.create_dataset("true_host_type", data=host)
        h.create_dataset("H0_nodes", data=np.array([H0_REP], float))
        for name, arr in (("logE_GG", gg), ("logE_AG", ag),
                          ("logE_GA", ga), ("logE_AA", aa)):
            h.create_dataset(name, data=arr.reshape(1, n))
        h.create_dataset("P_AGN_spatial", data=P_spatial)
        h.create_dataset("P_AGN_marked", data=P_marked)
        h.create_dataset("delta_P", data=dP)
        h.create_dataset("log_BF_spatial", data=bf_sp)
        h.create_dataset("log_BF_intrinsic", data=bf_in)
        h.create_dataset("log_BF_total", data=bf_to)
        h.attrs["H0_rep"] = H0_REP
        h.attrs["f_agn"] = F_AGN
        h.attrs["mu_chi_c2"] = MU_CHI
        h.attrs["log_f_gal"] = log_f_gal
        h.attrs["log_f_agn"] = log_f_agn
        h.attrs["source"] = str(A8_DECOMP)
        h.attrs["true_host_type_use"] = ("DIAGNOSTIC ONLY -- it enters no inference "
                                         "quantity in this file")
        h.attrs["definition_P_spatial"] = ("sigmoid(log(f_AGN/f_GAL) + log_BF_spatial); "
                                           "in the spatial-only model both branches "
                                           "share the GAL population, so the AGN "
                                           "branch evidence is E[AG]")
        h.attrs["definition_P_marked"] = "sigmoid(log(f_AGN/f_GAL) + log_BF_total)"
    print(f"wrote {h5_path}")

    out = {
        "analysis": "analysis_9_marked_multitracer_H0_fagn",
        "diagnostic": ("mechanism part A -- per-event host routing with and without "
                       "the spin mark, at Analysis 8's anchor point"),
        "seed": 100,
        "recomputed_through_the_likelihood": False,
        "source": str(A8_DECOMP),
        "point": {"H0": H0_REP, "f_agn": F_AGN, "mu_chi_c2": MU_CHI,
                  "log_f_gal": log_f_gal, "log_f_agn": log_f_agn,
                  "log_odds_prior": lam},
        "definitions": {
            "P_AGN_spatial": "sigmoid(log(f_AGN/f_GAL) + log_BF_spatial)",
            "P_AGN_marked": "sigmoid(log(f_AGN/f_GAL) + log_BF_total) = recorded P_AGN",
            "log_BF_spatial": "log E[AG] - log E[GG]",
            "log_BF_total": "log E[AA] - log E[GG] = log_BF_spatial + log_BF_intrinsic",
            "delta_P": "P_AGN_marked - P_AGN_spatial",
        },
        "closure_checks": closure,
        "all_events": _stats(dP, P_spatial, P_marked),
        "interpretation_only_split_by_true_label": {
            "note": ("true_host_type is not used anywhere in the inference; this "
                     "split exists so the reader can see which events move"),
            "GAL": _stats(dP[host == 0], P_spatial[host == 0], P_marked[host == 0]),
            "AGN": _stats(dP[host == 1], P_spatial[host == 1], P_marked[host == 1]),
            "n_true_GAL": int((host == 0).sum()),
            "n_true_AGN": int((host == 1).sum()),
        },
        "log_bayes_factor_summary": {
            "median_log_BF_spatial": float(np.median(bf_sp)),
            "median_log_BF_intrinsic": float(np.median(bf_in)),
            "median_log_BF_total": float(np.median(bf_to)),
            "std_log_BF_spatial": float(np.std(bf_sp)),
            "std_log_BF_intrinsic": float(np.std(bf_in)),
            "std_log_BF_total": float(np.std(bf_to)),
        },
        "outputs": [str(h5_path)],
    }
    json_path = _write_guard(DIAG / "a9_event_routing_h0_67p74_from_a8.json")
    json_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {json_path}")

    a = out["all_events"]
    print(f"\n  anchor (H0 {H0_REP}, f {F_AGN}, dmu {MU_CHI}), {n} events")
    print(f"    delta_P percentiles " + "  ".join(
        f"{q}%: {a['percentiles_delta_P'][str(q)]:+.4f}" for q in PCTS))
    print(f"    median delta_P {a['median_delta_P']:+.5f}   "
          f"median |delta_P| {a['median_abs_delta_P']:.5f}   "
          f"RMS {a['rms_abs_delta_P']:.5f}   max |delta_P| {a['max_abs_delta_P']:.5f}")
    print(f"    N(|dP|>0.1) {a['n_abs_delta_P_gt_0p1']}   "
          f"N(|dP|>0.2) {a['n_abs_delta_P_gt_0p2']}   "
          f"crossings of 0.5: up {a['n_crossing_0p5_upward']}, "
          f"down {a['n_crossing_0p5_downward']}")
    print(f"    sum P_spatial {a['sum_P_spatial']:.3f}   "
          f"sum P_marked {a['sum_P_marked']:.3f}   "
          f"delta {a['sum_P_marked_minus_spatial']:+.3f}   "
          f"(N f_AGN = {n * F_AGN:.1f})")
    return out


if __name__ == "__main__":
    main()
