#!/usr/bin/env python
"""Analysis 14 aggregation: results/scans/*.json -> results/mc_accuracy_summary.json (CPU, seconds).

    python aggregate_mc_accuracy.py [--results DIR]

For each configuration ((a) all-AGN, H0 at f_AGN = 1; (b) mixed, H0 and f_AGN) and each N_s:
  * the spread (sd, ddof 1) of the posterior median over sample redraws, in absolute units and
    as a fraction of the mean 68% half-width, with a 90% interval on the sd (chi-square);
  * the mean and sd of the 68% and 90% widths;
  * the mean shift of the median relative to the largest N_s, unpaired and paired (common random
    numbers: redraw r at a smaller N_s is nested in redraw r at the larger one), with 90% intervals;
  * the recorded variance of ln L (posterior-sample variance sum and N^2/N_eff at the reference
    cell and at the posterior median);
  * the first-order prediction of the bias: the posterior recomputed with
    ln L + sigma^2_PE/2 - N/(2 N_eff) (the log of a Monte Carlo mean is low by half its relative
    variance), and the median it moves, next to the measured shift;
and, at fixed samples (N_s = 8000, redraw 0), the same spread over injection sets (record + 1..5).
Missing pieces are reported as null; the script runs on whatever has finished.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import chi2

A14 = Path(__file__).resolve().parent.parent
PARAMS = {"a": ["H0"], "b": ["H0", "f"]}
LABEL = {"a": "all-AGN-hosted events, f_AGN = 1 (AGN catalog only)",
         "b": "mixed events (input f_AGN 0.30), joint (H0, f_AGN)"}


def label_of(s):
    """Configuration label of a scan: 'b' (seed 100), 'a_s103' (seed 103), 'a' (seed 100)."""
    return s.get("label") or s.get("config")
TRUTH = {"H0": 67.74, "f": 0.30}


def sd_ci90(x):
    x = np.asarray(x, float)
    n = x.size
    if n < 2:
        return None, [None, None]
    s = float(np.std(x, ddof=1))
    lo = s * np.sqrt((n - 1) / chi2.ppf(0.95, n - 1))
    hi = s * np.sqrt((n - 1) / chi2.ppf(0.05, n - 1))
    return s, [float(lo), float(hi)]


def mean_ci90(x):
    x = np.asarray(x, float)
    if x.size == 0:
        return None, [None, None]
    m = float(np.mean(x))
    if x.size < 2:
        return m, [None, None]
    se = float(np.std(x, ddof=1) / np.sqrt(x.size))
    return m, [m - 1.645 * se, m + 1.645 * se]


def load(results):
    scans = {}
    for p in sorted((results / "scans").glob("*.json")):
        scans[p.stem] = json.loads(p.read_text())
    return scans


def group(scans, cfg):
    pe, inj = {}, {}
    for tid, s in scans.items():
        if label_of(s) != cfg:
            continue
        if s.get("kind") == "pe":
            pe.setdefault(int(s["nsamp"]), {})[int(s["redraw"])] = s
        elif s.get("kind") == "inj":
            inj[s["injections"]] = s
    return pe, inj


def cell_value(res_h5, key, cell):
    import h5py
    with h5py.File(res_h5, "r") as f:
        return float(np.asarray(f[key])[cell])


def stats_block(runs, params):
    """Spread of medians and widths over a list of scan summaries."""
    out = {"n": len(runs)}
    for p in params:
        med = np.array([r["summary"][p]["median"] for r in runs])
        hw = np.array([r["summary"][p]["hw68"] for r in runs])
        w68 = np.array([r["summary"][p]["w68"] for r in runs])
        w90 = np.array([r["summary"][p]["w90"] for r in runs])
        medc = np.array([r["summary_bias_corrected"][p]["median"] for r in runs])
        sd, sdci = sd_ci90(med)
        hwm = float(hw.mean())
        out[p] = {
            "median_mean": float(med.mean()),
            "median_sd": sd, "median_sd_ci90": sdci,
            "median_sd_over_hw68": None if sd is None else sd / hwm,
            "median_sd_over_hw68_ci90": [None if v is None else v / hwm for v in sdci],
            "implied_scatter_inflation": None if sd is None else float(np.sqrt(1.0 + (sd / hwm) ** 2)),
            "hw68_mean": hwm,
            "w68_mean": float(w68.mean()), "w68_sd": sd_ci90(w68)[0],
            "w90_mean": float(w90.mean()), "w90_sd": sd_ci90(w90)[0],
            "first_order_bias_of_median": float(np.mean(med - medc)),
            "first_order_bias_of_median_over_hw68": float(np.mean(med - medc) / hwm),
            "medians": med.tolist(),
        }
    pv = np.array([r["at_ref_cell"]["pe_var"] for r in runs])
    ne = np.array([r["at_ref_cell"]["n_eff"] for r in runs])
    n_ev = runs[0]["n_events"]
    out["lnL_variance_at_ref_cell"] = {
        "ref_cell": runs[0].get("ref_cell"),
        "pe_var_mean": float(pv.mean()), "pe_var_sd": sd_ci90(pv)[0],
        "n_eff_mean": float(ne.mean()),
        "sigma2_sel_N2_over_Neff_mean": float(np.mean(n_ev ** 2 / ne)),
        "pe_var_range_over_grid_mean": [float(np.mean([r["pe_var_range"][0] for r in runs])),
                                        float(np.mean([r["pe_var_range"][1] for r in runs]))]}
    out["seconds_per_call_median"] = float(np.median([r["seconds_per_call_median"] for r in runs]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(A14 / "results"))
    args = ap.parse_args()
    res = Path(args.results)
    scans = load(res)
    summary = {"description": "Analysis 14 (run R0): Monte Carlo accuracy of ln L on seed 100; "
                              "spread of posterior summaries over redraws of the per-event samples "
                              "and of the injection set. Intervals are 90%.",
               "truth": TRUTH, "configs": {}}
    for pat in ("run_config*.json", "timing*.json", "plan*.json", "repro_*.json"):
        for p in sorted(res.glob(pat)):
            summary[p.stem] = json.loads(p.read_text())
    if (res / "diag_low_m2.json").exists():
        d = json.loads((res / "diag_low_m2.json").read_text())
        summary["seed100_all_agn_retired"] = {
            "why": "event 495 of seed 100's all-AGN set has true m2src 3.37 < m_min = 5; core e7c3007 "
                   "gives it zero population density, legacy 2b86a2d (the generator) does not",
            "log_p_pop_at_truth": {"core": d["density_core"].get("log_p_pop_at_truth"),
                                   "legacy": d["density_legacy"].get("log_p_pop_at_truth")},
            "event_sets_summary": d.get("event_sets_summary")}
    v1 = res / "validation_redraw.json"
    if v1.exists():
        v = json.loads(v1.read_text())
        summary["validation_redraw"] = {"pass": v["pass"], **{k: {kk: s[kk] for kk in (
            "per_event_ks_min_p", "per_event_ks_n_tests", "pooled_standardised_ks_p", "checks")}
            for k, s in v["sets"].items()}}
    labels = sorted({label_of(s) for s in scans.values() if s.get("kind") in ("pe", "inj")})
    for cfg in labels:
        pe, inj = group(scans, cfg)
        params = PARAMS[cfg[0]]
        seed = next(iter(s.get("seed", 100) for s in scans.values() if label_of(s) == cfg), 100)
        block = {"label": f"{LABEL[cfg[0]]}; catalog seed {seed}", "seed": seed, "params": params,
                 "by_nsamp": {}, "injection_redraws": None}
        ns_all = sorted(pe)
        nmax = ns_all[-1] if ns_all else None
        for ns in ns_all:
            runs = [pe[ns][r] for r in sorted(pe[ns])]
            st = stats_block(runs, params)
            st["redraws"] = sorted(pe[ns])
            # shift of the median relative to the largest N_s, and the first-order prediction of it
            if nmax is not None and ns != nmax:
                big = pe[nmax]
                for p in params:
                    m_big = np.array([big[r]["summary"][p]["median"] for r in sorted(big)])
                    hw = st[p]["hw68_mean"]
                    unp = st[p]["median_mean"] - float(m_big.mean())
                    se = np.sqrt(np.var([pe[ns][r]["summary"][p]["median"] for r in pe[ns]], ddof=1) / len(pe[ns])
                                 + np.var(m_big, ddof=1) / max(len(m_big), 1)) if len(m_big) > 1 and len(pe[ns]) > 1 else None
                    common = sorted(set(pe[ns]) & set(big))
                    d = np.array([pe[ns][r]["summary"][p]["median"] - big[r]["summary"][p]["median"] for r in common])
                    pm, pci = mean_ci90(d)
                    pred = st[p]["first_order_bias_of_median"] - stats_block(
                        [big[r] for r in sorted(big)], [p])[p]["first_order_bias_of_median"]
                    st[p]["shift_vs_largest_nsamp"] = {
                        "largest_nsamp": nmax,
                        "unpaired": unp, "unpaired_ci90": None if se is None else [unp - 1.645 * se, unp + 1.645 * se],
                        "paired_common_redraws": common, "paired": pm, "paired_ci90": pci,
                        "paired_over_hw68": None if pm is None else pm / hw,
                        "first_order_prediction": pred, "first_order_prediction_over_hw68": pred / hw}
            block["by_nsamp"][str(ns)] = st
        if inj:
            runs = []
            names = []
            rec = pe.get(8000, {}).get(0)
            if rec is not None:
                runs.append(rec)
                names.append("record")
            for k in sorted(inj):
                runs.append(inj[k])
                names.append(k)
            st = stats_block(runs, params)
            st["sets"] = names
            st["nsamp"] = 8000
            st["pe_redraw"] = 0
            st["n_detected_injections"] = [r.get("n_detected_injections") for r in runs]
            block["injection_redraws"] = st
        summary["configs"][cfg] = block
    out = res / "mc_accuracy_summary.json"
    out.write_text(json.dumps(summary, indent=1))
    print(f"wrote {out} ({len(scans)} scans)")
    for cfg, b in summary["configs"].items():
        for ns, st in b["by_nsamp"].items():
            txt = "  ".join(f"{p}: sd(med)/hw68 {st[p]['median_sd_over_hw68'] if st[p]['median_sd_over_hw68'] is None else round(st[p]['median_sd_over_hw68'], 3)}"
                            for p in b["params"])
            print(f"  ({cfg}) N_s {ns:>5s}  R={st['n']:2d}  {txt}  pe_var@ref {st['lnL_variance_at_ref_cell']['pe_var_mean']:.2f}")
        if b["injection_redraws"]:
            st = b["injection_redraws"]
            print(f"  ({cfg}) injections x{st['n']}: " + "  ".join(
                f"{p}: sd(med)/hw68 {st[p]['median_sd_over_hw68']}" for p in b["params"]))


if __name__ == "__main__":
    main()
