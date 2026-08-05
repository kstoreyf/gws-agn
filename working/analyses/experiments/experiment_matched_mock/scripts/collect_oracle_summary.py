#!/usr/bin/env python3
"""Assemble the exact-likelihood oracle campaign into results/oracle_summary.json.

Sources (all under results/):
  oracle_campaign.json      per-realisation darksirens / exact-oracle peaks and
                            the numerator-ladder slope terms (20 realisations)
  oracle_num_<tag>.npz      per-event exact numerator curves ln_O1..ln_O4 (161-pt
                            H0 grid) for the real events of realisation <tag>
  oracle_mu_<tag>.npz       the exact selection function ln mu(H0) per catalog
  oracle_selw_<tag>.npz     per-injection log selection weights at 9 H0 nodes
                            (the darksirens mu_hat estimator's raw material)
  oracle_boot_<tag>_<k>.npz / oracle_bootfix_<tag>_<k>.npz
                            parametric-bootstrap ensembles: 1000 fresh events per
                            file drawn with the generator recipe as-is / with the
                            sky width derived from the observables (the fix)
  oracle_report_{b,s4102}.json  the two deep single-realisation reports (Farr
                            term, dlnmu slopes)

Everything quoted in the paper about the oracle comes out of the summary this
script writes; the heavy lifting was done by oracle_exact.py / oracle_campaign.py.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
H0_TRUTH = 67.74
N_EVENTS = 1000


def _median_offset(H0, lnL):
    p = np.exp(lnL - lnL.max())
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (p[1:] + p[:-1]) * np.diff(H0))])
    cdf /= cdf[-1]
    return float(np.interp(0.5, cdf, H0) - H0_TRUTH)


def _curvature(H0, lnL):
    """-d2 lnL / dH0^2 from a local quadratic fit around the peak."""
    i = int(np.argmax(lnL))
    sl = slice(max(0, i - 8), min(len(H0), i + 9))
    c = np.polyfit(H0[sl] - H0[sl].mean(), lnL[sl], 2)
    return float(-2.0 * c[0])


def _slope_at_truth(H0, y):
    i = int(np.argmin(np.abs(H0 - H0_TRUTH)))
    return float((y[i + 1] - y[i - 1]) / (H0[i + 1] - H0[i - 1]))


def _stats(x):
    x = np.asarray(x, float)
    sd = float(x.std(ddof=1))
    return {"n": int(x.size), "mean": float(x.mean()), "sd": sd,
            "sem": sd / np.sqrt(x.size),
            "sigma_from_zero": abs(float(x.mean())) / (sd / np.sqrt(x.size))}


def main():
    camp = json.loads((RES / "oracle_campaign.json").read_text())
    tags = [r["tag"] for r in camp]

    # -- campaign-level offsets straight from the archived peaks ------------
    out = {
        "n_realisations": len(camp),
        "offset_darksirens": _stats([r["offset_ds"] for r in camp]),
        "offset_exact_oracle": _stats([r["offset_oracle"] for r in camp]),
        "paired_ds_minus_oracle": _stats([r["paired"] for r in camp]),
    }

    # -- per-realisation exact curves: curvature, score identity, mu_hat ----
    curv, score_def, muhat_dH0, ladder = {}, [], [], {
        "pe_width_O2_minus_O1": [], "pixelation_O3_minus_O2": [],
        "jacobian_O3b_minus_O3": [], "kde_O4_minus_O3b": []}
    med_check = []
    for r in camp:
        t = r["tag"]
        num = np.load(RES / f"oracle_num_{t}.npz")
        mu = np.load(RES / f"oracle_mu_{t}.npz")
        H0 = num["H0"]
        lnL = num["ln_O1"].sum(axis=0) - N_EVENTS * mu["ln_mu"]
        curv[t] = _curvature(H0, lnL)
        med_check.append(_median_offset(H0, lnL) - r["offset_oracle"])
        dlnmu = _slope_at_truth(H0, mu["ln_mu"])
        score = _slope_at_truth(H0, num["ln_O1"].mean(axis=0))
        score_def.append(score - dlnmu)
        for key, sl in (("pe_width_O2_minus_O1", r["slope_O2_minus_O1"]),
                        ("pixelation_O3_minus_O2", r["slope_O3_minus_O2"]),
                        ("jacobian_O3b_minus_O3", r["slope_O3b_minus_O3"]),
                        ("kde_O4_minus_O3b", r["slope_O4_minus_O3b"])):
            ladder[key].append(sl / curv[t])
        # darksirens mu_hat vs exact mu: excess slope at truth -> dH0
        sw = np.load(RES / f"oracle_selw_{t}.npz")
        Hn = sw["H0"]
        ln_muhat = np.array([np.logaddexp.reduce(sw["ldw"][k]) for k in
                             range(len(Hn))]) - np.log(float(sw["Ndraw"]))
        i = int(np.argmin(np.abs(Hn - H0_TRUTH)))
        hat_slope = float((ln_muhat[i + 1] - ln_muhat[i - 1]) / (Hn[i + 1] - Hn[i - 1]))
        exact_at = np.interp(Hn, H0, mu["ln_mu"])
        ex_slope = float((exact_at[i + 1] - exact_at[i - 1]) / (Hn[i + 1] - Hn[i - 1]))
        muhat_dH0.append(-N_EVENTS * (hat_slope - ex_slope) / curv[t])

    out["validation_median_recompute_max_abs_err"] = float(np.max(np.abs(med_check)))
    out["score_identity"] = {
        "per_event_score_deficit_at_truth": _stats(score_def),
        "note": ("mean_i d lnZ_i/dH0 at truth minus d ln mu/dH0; the detected-set "
                 "score identity demands zero"),
    }
    out["numerator_ladder_dH0"] = {k: _stats(v) for k, v in ladder.items()}
    out["muhat_minus_exact_dH0"] = _stats(muhat_dH0)
    out["muhat_note"] = ("zero-mean per-realisation scatter of the injection x "
                        "catalog-KDE interaction in mu_hat, expressed on H0")

    # -- Farr 1/Neff term (deep reports) ------------------------------------
    farr = {}
    for t in ("b", "s4102"):
        rep = json.loads((RES / f"oracle_report_{t}.json").read_text())
        farr[t] = {"dH0_sel_farr_term": rep["dH0_sel_farr_term"],
                   "dlnmu_dH0_exact": rep["dlnmu_dH0_exact"],
                   "dlnmu_dH0_ds": rep["dlnmu_dH0_ds"],
                   "Neff_at_truth": rep["Neff_at_truth"]}
    out["farr_term"] = farr

    # -- parametric bootstrap: recipe as-is vs observable-derived sky width -
    for label, pat in (("bootstrap_asis", "oracle_boot_{t}_{k}.npz"),
                       ("bootstrap_fix", "oracle_bootfix_{t}_{k}.npz")):
        offs, defs = [], []
        ks = (1, 2) if label == "bootstrap_asis" else (11, 12)
        for t in tags:
            for k in ks:
                p = RES / pat.format(t=t, k=k)
                if not p.exists():
                    continue
                b = np.load(p)
                mu = np.load(RES / f"oracle_mu_{t}.npz")
                H0 = b["H0"]
                lnL = b["ln_O1"].sum(axis=0) - N_EVENTS * mu["ln_mu"]
                offs.append(_median_offset(H0, lnL))
                defs.append(_slope_at_truth(H0, b["ln_O1"].mean(axis=0))
                            - _slope_at_truth(H0, mu["ln_mu"]))
        out[label] = {"offset": _stats(offs),
                      "per_event_score_deficit": _stats(defs)}

    p = RES / "oracle_summary.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"Wrote {p}")
    print(f"  validation (median recompute): {out['validation_median_recompute_max_abs_err']:.4f}")
    for k in ("offset_darksirens", "offset_exact_oracle", "paired_ds_minus_oracle"):
        s = out[k]
        print(f"  {k}: {s['mean']:+.3f} +- {s['sem']:.3f} (sd {s['sd']:.3f})")
    print(f"  score deficit: {out['score_identity']['per_event_score_deficit_at_truth']['mean']:+.5f}"
          f" +- {out['score_identity']['per_event_score_deficit_at_truth']['sem']:.5f}")
    print(f"  muhat dH0 noise: sd {out['muhat_minus_exact_dH0']['sd']:.3f}"
          f" (mean {out['muhat_minus_exact_dH0']['mean']:+.3f})")
    for k in ("bootstrap_asis", "bootstrap_fix"):
        s = out[k]["offset"]
        print(f"  {k}: {s['mean']:+.3f} +- {s['sem']:.3f} (n={s['n']})")


if __name__ == "__main__":
    main()
