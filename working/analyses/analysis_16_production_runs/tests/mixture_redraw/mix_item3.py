#!/usr/bin/env python
"""Item 3 of the validation (numpy only): on - off per merger for the refilled mergers that are not edge mergers.

    python tests/mixture_redraw/mix_item3.py LINE_STEM SCAN_JSON REALIZATION [--pa-cut 0.5]

From results/<LINE_STEM>.npz (mix_eval.py line: T refilled mergers, ev/var/ess off then on at every node) and the scan's
per-merger P_A: for mergers with P_A >= cut (posterior mostly inside A), d_i = ln L_i(on) - ln L_i(off) and the half-variance
corrected d_i^c = (ln L_i + sigma_i^2/2)(on) - (...)(off), at every node: mean over mergers, its standard error, rms; the
mean over nodes of those. ESS census (< 5, 20, 100) over the bound mergers, on and off, at the line's peak node.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parent / "results"


def main(stem, scan, real, cut):
    z = np.load(R / f"{stem}.npz")
    H, T, ev, var, ess = z["H0"], z["T"], z["ev"], z["var"], z["ess"]
    nT = T.size
    pa = json.loads((R / scan).read_text())["realizations"][str(real)]["per_merger"]
    P_A = np.array([pa[str(int(i))]["P_A"] for i in T])
    ne = P_A >= cut
    off, on = ev[:, :nT], ev[:, nT:2 * nT]
    voff, von = var[:, :nT], var[:, nT:2 * nT]
    d = (on - off)[:, ne]
    dc = ((on + 0.5 * von) - (off + 0.5 * voff))[:, ne]
    lj = json.loads((R / f"{stem}.json").read_text())
    jpk = int(np.argmin(np.abs(H - lj["per_merger_on_minus_off_at_peak"]["H0"])))
    n = int(ne.sum())

    def st(x):
        m = x.mean(1)
        se = x.std(1, ddof=1) / np.sqrt(x.shape[1])
        return {"at_peak": {"mean": float(m[jpk]), "sem": float(se[jpk]), "rms": float(np.sqrt((x[jpk] ** 2).mean())),
                            "z": float(m[jpk] / se[jpk])},
                "over_nodes": {"max_abs_z": float(np.max(np.abs(m / se))), "mean_z": float(np.mean(m / se)),
                               "mean_of_means": float(m.mean())}}

    # core normalises p_pe per merger over the merger's own samples, so ln L_i carries a Lambda-independent constant
    # (ln sum_j p_pe r) that differs between the two sample sets: the test is on the H0 dependence, d_i(H) - mean_H d_i
    ci = lj["lines"]["on"]["ci90"]
    inn = (H >= ci[0]) & (H <= ci[1])
    shape = {}
    for lab, x in (("d", d), ("d_corrected", dc)):
        xs = x[inn] - x[inn].mean(axis=0, keepdims=True)
        m, se = xs.mean(1), xs.std(1, ddof=1) / np.sqrt(xs.shape[1])
        zz = m / se
        D = x[inn][-1] - x[inn][0]
        shape[lab] = {"nodes": int(inn.sum()), "H0_range": [float(H[inn][0]), float(H[inn][-1])],
                      "max_abs_mean_node": float(np.abs(m).max()), "max_abs_z": float(np.abs(zz).max()),
                      "chi2_over_nodes": float((zz ** 2).sum()),
                      "D_end_minus_start": {"mean": float(D.mean()), "sem": float(D.std(ddof=1) / np.sqrt(D.size)),
                                            "rms": float(np.sqrt((D ** 2).mean())), "z": float(D.mean() / (D.std(ddof=1) / np.sqrt(D.size)))}}
    out = {"line": stem, "pa_cut": cut, "shape_within_ci90": shape, "n_refilled": int(nT), "n_non_edge": n, "H0_peak": float(H[jpk]),
           "d": st(d), "d_corrected": st(dc),
           "ess_census_at_peak": {lab: {f"lt_{c}": int((ess[jpk, sl] < c).sum()) for c in (5, 20, 100)}
                                  for lab, sl in (("off", slice(0, nT)), ("on", slice(nT, 2 * nT)))},
           "ess_census_non_edge": {lab: {f"lt_{c}": int((ess[jpk, sl][ne] < c).sum()) for c in (5, 20, 100)}
                                   for lab, sl in (("off", slice(0, nT)), ("on", slice(nT, 2 * nT)))},
           "ess_ratio_on_over_off_median_non_edge": float(np.median(ess[jpk, nT:2 * nT][ne] / ess[jpk, :nT][ne])),
           "ess_ratio_on_over_off_median_edge": float(np.median(ess[jpk, nT:2 * nT][~ne] / ess[jpk, :nT][~ne])) if (~ne).any() else None,
           "edge_mergers": [{"i": int(T[k]), "P_A": float(P_A[k]), "ess_off": float(ess[jpk, k]), "ess_on": float(ess[jpk, nT + k]),
                             "d_at_peak": float(on[jpk, k] - off[jpk, k])} for k in np.where(~ne)[0]]}
    (R / f"item3_{stem}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "edge_mergers"}, indent=1))
    print("edge mergers (P_A < cut):", [(e["i"], round(e["P_A"], 3), round(e["ess_off"], 1), round(e["ess_on"], 1)) for e in out["edge_mergers"]])


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("stem")
    ap.add_argument("scan")
    ap.add_argument("real", type=int)
    ap.add_argument("--pa-cut", type=float, default=0.5)
    a = ap.parse_args()
    main(a.stem, a.scan, a.real, a.pa_cut)
