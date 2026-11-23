#!/usr/bin/env python
"""Step 2 of the obslike_f0p30_s1 H0-offset diagnosis: which mergers carry it.

hosts_known's grids store each host class's factor separately (prod_model.HostsKnownModel):
    lnL_c(H0) = sum_{i in c} ln L_i,c(H0) - N_c ln mu_c(H0),   c = agn (AGN catalog) / gal (galaxy catalog),
with pe_var_c, n_eff_c, log_mu_c. Per set (hosts_known, complete catalogs, realizations 100-103, 105) this script gives
  * per class and realization: the corrected factor's H0 posterior (run_posteriors.factor_summary, the run summaries' method),
  * per class, combined over the five: the corrected factors summed (run_posteriors.grid_marginal(..., factor=c)),
    median / 90% CI / mean / sd and the credible level of the input,
  * an additive decomposition of the combined offset: near the peak every factor is close to quadratic, so the total's
    mean is the information-weighted mean of the classes' means; class c contributes I_c (m_c - H0_in) / I_tot,
  * the score at the input split into the event sum and the selection term:
        d/dH0 [sum_i ln L_i,c]  and  -N_c d ln mu_c/dH0  at H0 = 67.74 (cubic spline of the fine pass, raw ln L),
    and z_c = score_c(H0_in) / sqrt(I_c) with I_c the curvature of the corrected factor (expected ~ N(0, 1) per class
    when the model is right).
Output: tests/results/obslike_offset_step2_class_split.json
"""
import json
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.interpolate import CubicSpline

A16 = Path(__file__).resolve().parents[1]
RUNS = A16 / "runs"
sys.path.insert(0, str(A16.parents[1] / "paper_prd" / "scripts"))
import run_posteriors as rp  # noqa: E402

H0_IN = 67.74
REAL = [100, 101, 102, 103, 105]
SETS = ["obslike_f0p30_s1", "obslike_f0p30_s0", "obslike_f1p00_s0", "peak_f0p30_s1"]
if len(sys.argv) > 1 and sys.argv[1] == "all":   # every hosts_known set on disk -> ..._all.json
    SETS = sorted(p.name for p in (RUNS / "hosts_known" / "both_catalogs" / "complete").iterdir() if p.is_dir())


def dirs_of(label):
    base = RUNS / "hosts_known" / "both_catalogs" / "complete" / label
    return [base / f"realization_{r}_agn_s32000_inj1" for r in REAL]


def fine_arrays(d):
    with h5py.File(d / "grid" / f"{rp.fine_name(d)}.h5", "r") as h:
        x = np.asarray(h["axis_H0"][()], float)
        out = {k: np.asarray(h[k][()], float) for k in h.keys() if not k.startswith("axis_")}
    return x, out


def score_parts(d, n_by):
    """Score (d lnL/dH0) at the input of each class's event sum and selection term, and the corrected factor's
    curvature from a quadratic fit within +-1.5 of the factor's own maximum (fine pass)."""
    x, a = fine_arrays(d)
    out = {}
    for c in ("gal", "agn"):
        n = n_by[c]
        if n == 0:
            continue
        lnL, lmu = a[f"lnL_{c}"], a[f"log_mu_{c}"]
        lnc = lnL + 0.5 * a[f"pe_var_{c}"] - 0.5 * n / a[f"n_eff_{c}"]
        ev = lnL + n * lmu
        sel = -n * lmu
        s_ev = float(CubicSpline(x, ev)(H0_IN, 1))
        s_sel = float(CubicSpline(x, sel)(H0_IN, 1))
        s_corr = float(CubicSpline(x, lnc)(H0_IN, 1))
        # curvature: quadratic fit of the corrected factor around the input (+-2.5), the information
        m = np.abs(x - H0_IN) <= 2.5
        cq = np.polyfit(x[m] - H0_IN, lnc[m], 2)
        info = float(-2 * cq[0])
        out[c] = {"n": n, "score_event_sum": s_ev, "score_selection": s_sel, "score_corrected": s_corr,
                  "score_event_per_merger": s_ev / n, "dlogmu_dH0": -s_sel / n,
                  "info_quadfit": info, "z_score": s_corr / np.sqrt(info) if info > 0 else None,
                  "mle_shift_quad": float(cq[1] / info) if info > 0 else None}
    return out


def main():
    res = {}
    for label in SETS:
        dirs = dirs_of(label)
        R = {"per_realization": {}, "combined": {}}
        n_by_r = []
        for r, d in zip(REAL, dirs):
            s = rp.read_json(d / "summary.json")
            n_by = {"gal": 0, "agn": 0}
            n_by.update(s["n_events_by_class"])
            n_by_r.append(n_by)
            row = {"total": s["posterior"]["H0"], "n": n_by}
            for c in ("gal", "agn"):
                if n_by[c] == 0:
                    continue
                fs, _ = rp.factor_summary(d, c)
                row[c] = {k: fs[k] for k in ("median", "ci90", "mean", "sd")}
            row["score"] = score_parts(d, n_by)
            R["per_realization"][str(r)] = row
        for c in ("gal", "agn", None):
            if c is not None and all(nb[c] == 0 for nb in n_by_r):
                continue
            x, p = rp.grid_marginal(dirs, "H0", c)
            cdf = np.concatenate([[0], np.cumsum(0.5 * (p[1:] + p[:-1]) * np.diff(x))])
            cdf /= cdf[-1]
            q = lambda u: float(np.interp(u, cdf, x))
            mean = float(np.trapz(x * p, x))
            sd = float(np.sqrt(np.trapz((x - mean) ** 2 * p, x)))
            F = float(np.interp(H0_IN, x, cdf))
            R["combined"][c or "total"] = {"median": q(0.5), "ci90": [q(0.05), q(0.95)], "mean": mean, "sd": sd,
                                           "cdf_at_input": F, "level_of_input": abs(2 * F - 1),
                                           "z_mean": (mean - H0_IN) / sd}
        # additive decomposition of the total's offset (Gaussian approximation)
        comb = R["combined"]
        I = {c: 1 / comb[c]["sd"] ** 2 for c in ("gal", "agn") if c in comb}
        It = sum(I.values())
        R["offset_decomposition"] = {
            c: {"info_fraction": I[c] / It, "contribution_to_mean_offset": I[c] * (comb[c]["mean"] - H0_IN) / It}
            for c in I}
        R["offset_decomposition"]["sum_of_contributions"] = sum(v["contribution_to_mean_offset"]
                                                                for v in R["offset_decomposition"].values())
        R["offset_decomposition"]["total_mean_offset"] = comb["total"]["mean"] - H0_IN
        # summed scores over realizations, per class: event, selection, and z
        agg = {}
        for c in ("gal", "agn"):
            rows = [R["per_realization"][str(r)]["score"].get(c) for r in REAL]
            rows = [v for v in rows if v]
            if not rows:
                continue
            se = sum(v["score_event_sum"] for v in rows)
            ss = sum(v["score_selection"] for v in rows)
            sc = sum(v["score_corrected"] for v in rows)
            info = sum(v["info_quadfit"] for v in rows)
            agg[c] = {"score_event_sum": se, "score_selection": ss, "score_corrected": sc, "info": info,
                      "z": sc / np.sqrt(info) if info > 0 else None, "per_realization_z": [v["z_score"] for v in rows],
                      "per_merger_event_score": se / sum(v["n"] for v in rows),
                      "per_merger_dlogmu_dH0": -ss / sum(v["n"] for v in rows)}
        R["scores_summed"] = agg
        res[label] = R
        print(f"=== {label}")
        for r in REAL:
            row = R["per_realization"][str(r)]
            txt = f"  r{r}: total {row['total']['median']:.2f}"
            for c in ("gal", "agn"):
                if c in row:
                    txt += f" | {c} (N={row['n'][c]}) {row[c]['median']:.2f} [{row[c]['ci90'][0]:.2f},{row[c]['ci90'][1]:.2f}] z={(row['score'][c]['z_score'] or float('nan')):+.2f}"
            print(txt)
        for c, v in comb.items():
            print(f"  combined {c:5s}: median {v['median']:.3f} ci90 [{v['ci90'][0]:.3f},{v['ci90'][1]:.3f}] mean {v['mean']:.3f} "
                  f"sd {v['sd']:.3f} z {v['z_mean']:+.2f} level {v['level_of_input']:.4f}")
        for c, v in R["offset_decomposition"].items():
            print(f"  offset part {c}: {v}")
        for c, v in agg.items():
            print(f"  scores {c}: event {v['score_event_sum']:+.2f} selection {v['score_selection']:+.2f} "
                  f"corrected {v['score_corrected']:+.2f} info {v['info']:.2f} z {(v['z'] or float('nan')):+.2f} "
                  f"per-merger event {v['per_merger_event_score']:.5f} dlogmu/dH0 {v['per_merger_dlogmu_dH0']:.5f}")
    out = A16 / "tests" / "results" / ("obslike_offset_step2_class_split" + ("_all" if len(sys.argv) > 1 and sys.argv[1] == "all" else "") + ".json")
    out.write_text(json.dumps(res, indent=1, default=float))
    print("wrote", out)


if __name__ == "__main__":
    main()
