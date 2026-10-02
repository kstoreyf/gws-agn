#!/usr/bin/env python
"""A11 likelihood: core b47e41c against af896ca on the a11_grid_cells grids.

    python compare_a11.py      # writes ../results/a11_comparison.json

Per grid: the per-cell dlogL split into its mean (a constant offset, invisible to the
posterior) and its spread; then, with flat priors on the grid, each 1-D marginal's median,
90% interval, width ratio, median shift in half-widths and KS distance (compare.post_compare).
These are conditional slices through the 11D posterior, not full marginals.
"""
import json
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

import compare as C

RES = Path(__file__).resolve().parent.parent / "results"


def main():
    A = json.loads((RES / "legacy_af896ca_window_4096_grid.json").read_text())
    B = json.loads((RES / "core_b47e41c_a11_grid.json").read_text())
    key = lambda r: (r["name"], r["H0"], r["fcat_2"], r["mu_G"], r["dmu_G"], r["mu_chi"], r["dmu_chi"])
    ra = {key(r): r for r in A["rows"]}
    rb = {key(r): r for r in B["rows"]}
    out = {"af896ca_seconds_median": float(np.median([r["seconds"] for r in A["rows"][1:]])),
           "core_seconds_median": float(np.median([r["seconds"] for r in B["rows"][1:]])),
           "core_device_memory": B.get("device_memory"), "core_host_maxrss_GB": B.get("host_maxrss_GB"),
           "grids": {}}
    axes = {"gH": ["H0"], "gF": ["fcat_2"], "gM": ["mu_G", "dmu_G"], "gC": ["mu_chi", "dmu_chi"]}
    for g, ax in axes.items():
        ks = [k for k in ra if k[0] == g and k in rb]
        d = np.array([rb[k]["logL"] - ra[k]["logL"] for k in ks])
        res = {"n_cells": len(ks), "mean_dlogL": float(d.mean()), "spread_dlogL": float(d.std()),
               "range_dlogL": [float(d.min()), float(d.max())]}
        vals = [sorted({ra[k][a] for k in ks}) for a in ax]
        La = np.full([len(v) for v in vals], np.nan); Lb = La.copy()
        for k in ks:
            ix = tuple(vals[i].index(ra[k][a]) for i, a in enumerate(ax))
            La[ix] = ra[k]["logL"]; Lb[ix] = rb[k]["logL"]
        for i, a in enumerate(ax):
            other = tuple(j for j in range(len(ax)) if j != i)
            la = logsumexp(La, axis=other) if other else La
            lb = logsumexp(Lb, axis=other) if other else Lb
            res[a] = C.post_compare(np.array(vals[i]), la, np.array(vals[i]), lb)
        out["grids"][g] = res
        print(f"{g}: mean dlogL {res['mean_dlogL']:+.4f} spread {res['spread_dlogL']:.4f}  " +
              "  ".join(f"{a}: KS {res[a]['ks']:.4f} W90 {res[a]['W90_ratio_b_over_a']:.4f} "
                        f"shift {res[a]['median_shift_over_halfwidth']:+.4f}hw" for a in ax))
    print(f"seconds/call: af896ca {out['af896ca_seconds_median']:.2f}, core {out['core_seconds_median']:.2f}")
    (RES / "a11_comparison.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
