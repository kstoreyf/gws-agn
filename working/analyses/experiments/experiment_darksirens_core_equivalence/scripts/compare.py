#!/usr/bin/env python
"""Compare the arms cell by cell and at the posterior level.

    python compare.py        # reads ../results/<arm>.json, writes ../results/comparison.json

Pairs, in the order a change enters:
    legacy_2b86a2d -> legacy_0c5b3db    gws-agn's two legacy commits
    legacy_0c5b3db -> legacy_c042527    legacy history up to core's frozen reference
    legacy_c042527 -> core_legacy_arith core vs its reference, deliberate changes off
    legacy_c042527 -> core_default      core defaults (kernel pin, analytic pairing)
    core_default   -> core_fast         the opt-in speed-ups

Per pair: bitwise count, max |dlogL|, max relative difference, and the difference split
into its mean (a constant offset, invisible to the posterior) and its spread (what can
move the posterior). Posterior level (flat priors on the grid, the owner's criterion):
g1 p(H0); g2 marginals p(H0), p(log10n0) and the joint. For each marginal: median, 90%
interval, width ratio, and the KS distance max|CDF_a - CDF_b| on a cubic-spline
refinement of the log density.
"""
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.special import logsumexp

RES = Path(__file__).resolve().parent.parent / "results"
PAIRS = [("legacy_2b86a2d", "legacy_0c5b3db"), ("legacy_0c5b3db", "legacy_c042527"),
         ("legacy_c042527", "core_legacy_arith"), ("legacy_c042527", "core_default"),
         ("core_default", "core_fast"),
         # merger-rate slope pinned to the mock's gamma = 0 (c042527/core fiducial is 2.5)
         ("legacy_0c5b3db", "legacy_c042527_gamma0"),
         ("legacy_c042527_gamma0", "core_default_gamma0"),
         ("core_default_gamma0", "core_fast_gamma0"),
         ("legacy_0c5b3db", "core_default_gamma0"),
         ("legacy_0c5b3db", "core_fast_gamma0")]


def load(arm):
    p = RES / f"{arm}.json"
    return json.loads(p.read_text()) if p.exists() else None


def grid(rows, g):
    rs = [r for r in rows if r["grid"] == g]
    H = np.unique([r["H0"] for r in rs]); N = np.unique([r["log10n0"] for r in rs])
    L = np.full((H.size, N.size), np.nan)
    for r in rs:
        L[np.searchsorted(H, r["H0"]), np.searchsorted(N, r["log10n0"])] = r["logL"]
    return H, N, L


def marginal(x, logp):
    """Spline-refined normalised density and CDF of a 1-D log marginal."""
    xf = np.linspace(x[0], x[-1], 4001)
    lf = CubicSpline(x, logp)(xf) if x.size > 3 else np.interp(xf, x, logp)
    p = np.exp(lf - lf.max())
    c = np.concatenate([[0.0], np.cumsum(0.5 * (p[1:] + p[:-1]) * np.diff(xf))])
    c /= c[-1]
    q = lambda t: float(np.interp(t, c, xf))
    return xf, c, {"median": q(0.5), "ci90": [q(0.05), q(0.95)]}


def post_compare(xa, la, xb, lb):
    xfa, ca, sa = marginal(xa, la)
    xfb, cb, sb = marginal(xb, lb)
    ks = float(np.max(np.abs(ca - np.interp(xfa, xfb, cb))))
    wa, wb = sa["ci90"][1] - sa["ci90"][0], sb["ci90"][1] - sb["ci90"][0]
    return {"a": sa, "b": sb, "ks": ks, "W90_ratio_b_over_a": wb / wa,
            "median_shift_over_halfwidth": (sb["median"] - sa["median"]) / (0.5 * wa)}


def main():
    arms = {a: load(a) for pr in PAIRS for a in pr}
    out = {"arms": {a: {k: v for k, v in d.items() if k != "rows"} for a, d in arms.items() if d},
           "timing_median_seconds": {a: float(np.median([r["seconds"] for r in d["rows"][1:]]))
                                     for a, d in arms.items() if d},
           "pairs": {}}
    for a, b in PAIRS:
        A, B = arms.get(a), arms.get(b)
        if not (A and B):
            out["pairs"][f"{a}->{b}"] = {"missing": [x for x, d in ((a, A), (b, B)) if not d]}
            continue
        ka = {(r["grid"], r["H0"], r["log10n0"]): r for r in A["rows"]}
        common = [k for k in ((r["grid"], r["H0"], r["log10n0"]) for r in B["rows"]) if k in ka]
        rb = {(r["grid"], r["H0"], r["log10n0"]): r for r in B["rows"]}
        d = np.array([rb[k]["logL"] - ka[k]["logL"] for k in common])
        rel = np.abs(d) / np.abs([ka[k]["logL"] for k in common])
        pr = {"n_cells": len(common), "n_bitwise": int(sum(rb[k]["logL_hex"] == ka[k]["logL_hex"] for k in common)),
              "max_abs_dlogL": float(np.max(np.abs(d))), "max_rel": float(np.max(rel)),
              "mean_dlogL": float(np.mean(d)), "spread_dlogL": float(np.std(d)),
              "range_dlogL": [float(d.min()), float(d.max())]}
        Ha, _, La = grid(A["rows"], "g1"); Hb, _, Lb = grid(B["rows"], "g1")
        pr["g1_H0"] = post_compare(Ha, La[:, 0], Hb, Lb[:, 0])
        Ha, Na, La = grid(A["rows"], "g2"); Hb, Nb, Lb = grid(B["rows"], "g2")
        if np.isfinite(La).all() and np.isfinite(Lb).all():
            pr["g2_H0"] = post_compare(Ha, logsumexp(La, axis=1), Hb, logsumexp(Lb, axis=1))
            pr["g2_log10n0"] = post_compare(Na, logsumexp(La, axis=0), Nb, logsumexp(Lb, axis=0))
            pa = np.exp(La - logsumexp(La)); pb = np.exp(Lb - logsumexp(Lb))
            pr["g2_joint_total_variation"] = float(0.5 * np.abs(pa - pb).sum())
        out["pairs"][f"{a}->{b}"] = pr
    (RES / "comparison.json").write_text(json.dumps(out, indent=2))
    for k, v in out["pairs"].items():
        if "missing" in v:
            print(f"{k:42s} missing {v['missing']}"); continue
        print(f"{k:42s} bitwise {v['n_bitwise']}/{v['n_cells']}  max|d| {v['max_abs_dlogL']:.3e}  "
              f"mean {v['mean_dlogL']:+.4f}  spread {v['spread_dlogL']:.2e}  "
              f"KS(H0 g1) {v['g1_H0']['ks']:.4f}  W90 {v['g1_H0']['W90_ratio_b_over_a']:.4f}")
    print("timing (median s/call):", {k: round(v, 3) for k, v in out["timing_median_seconds"].items()})


if __name__ == "__main__":
    main()
