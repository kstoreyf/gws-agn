#!/usr/bin/env python
"""Analysis 11D against Analysis 10's C10-J and against 11C (brief sections 19 and 20).

    python scripts/a11_11D_compare.py

Reads results/a11_11D.{json,npz}, results/a11_11C.json and Analysis 10's
results/c10_arm_J.{h5,json}; writes diagnostics/a11_11D_comparisons.json.

C10-J is a grid posterior. Its quantiles are taken from a cubic spline of the log
marginal on the grid nodes (the method S1 and the 11C comparisons used; the linear
CDF between coarse nodes widens 90% ends by up to 0.22 half-widths). The linear-CDF
numbers recorded in c10_arm_J.json are carried alongside for reference.
"""
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.interpolate import CubicSpline

HERE = Path(__file__).resolve().parent
A11 = HERE.parent
RESULTS, DIAG = A11 / "results", A11 / "diagnostics"
A10_RESULTS = A11.parent / "analysis_10_mass_spin_marked_multitracer" / "results"
H0_PLANTED = 67.74
POP = ("f_agn", "mu_G", "dmu_G", "mu_chi", "dmu_chi")
C10_KEY = {"H0": ("H0_grid", "marginal/H0", "H0"), "f_agn": ("f_grid", "marginal/f_agn", "f_agn"),
           "dmu_G": ("dmu_G_grid", "marginal/dmu_G", "dmu_G"),
           "dmu_chi": ("dmu_chi_grid", "marginal/dmu_chi", "dmu_chi")}


def spline_quantiles(x, p, n=20001):
    xf = np.linspace(x[0], x[-1], n)
    pf = np.exp(CubicSpline(x, np.log(np.maximum(p, 1e-300)))(xf))
    c = np.concatenate([[0.0], np.cumsum(0.5 * (pf[1:] + pf[:-1]) * np.diff(xf))])
    c /= c[-1]
    q = lambda t: float(np.interp(t, c, xf))
    return {"median": q(0.5), "ci68": [q(0.16), q(0.84)], "ci90": [q(0.05), q(0.95)]}


def width(s, level):
    lo, hi = s[f"ci{level}"]
    return hi - lo


def ratio(a, b):
    return {"W68": width(a, 68) / width(b, 68), "W90": width(a, 90) / width(b, 90),
            "median_shift": a["median"] - b["median"]}


def main():
    d = json.loads((RESULTS / "a11_11D.json").read_text())
    c = json.loads((RESULTS / "a11_11C.json").read_text())
    npz = np.load(RESULTS / "a11_11D.npz")
    S, names = npz["samples"], [str(n) for n in npz["names"]]
    col = {n: S[:, names.index(n)] for n in names}
    s11d = d["summary"]
    cj = json.loads((A10_RESULTS / "c10_arm_J.json").read_text())

    c10 = {}
    with h5py.File(A10_RESULTS / "c10_arm_J.h5", "r") as h:
        for n, (ax, mg, jk) in C10_KEY.items():
            c10[n] = {"spline": spline_quantiles(h[ax][:], h[mg][:]),
                      "linear_json": {k: cj[jk][k] for k in ("median", "ci68", "ci90")}}

    out = {"what": "11D (H0 and all five population coordinates free) against C10-J and 11C",
           "C10_J_quantiles": c10,
           "vs_C10_J": {n: {"spline": ratio(s11d[n], c10[n]["spline"]),
                            "linear_json": ratio(s11d[n], c10[n]["linear_json"])}
                        for n in C10_KEY},
           "vs_11C_H0_release_cost": {n: ratio(s11d[n], c["summary"][n]) for n in POP}}

    sd = s11d["H0"]["sd"]
    out["H0"] = {"summary": s11d["H0"], "planted": H0_PLANTED,
                 "offset_from_planted": s11d["H0"]["median"] - H0_PLANTED,
                 "offset_over_sd": (s11d["H0"]["median"] - H0_PLANTED) / sd,
                 "minus_C10_J_median": s11d["H0"]["median"] - c10["H0"]["spline"]["median"]}

    # brief section 20: which population coordinate H0 trades against
    h = col["H0"]
    abs_G = col["mu_G"] + col["dmu_G"]
    abs_chi = col["mu_chi"] + col["dmu_chi"]
    extra = {"mu_G+dmu_G": abs_G, "mu_chi+dmu_chi": abs_chi}
    rho, slope, shift = {}, {}, {}
    for n, x in [(k, col[k]) for k in POP] + list(extra.items()):
        rho[n] = float(np.corrcoef(h, x)[0, 1])
        slope[n] = float(np.cov(h, x)[0, 1] / np.var(x, ddof=1))
        # H0 displacement produced by a one-sd move of x along the posterior regression
        shift[n] = slope[n] * float(x.std(ddof=1))
    out["H0_correlations"] = {"rho": rho, "dH0_dx_regression": slope, "dH0_per_1sd_of_x": shift,
                              "C10_J": {k: v for k, v in cj["correlations"].items() if k.startswith("H0|")}}

    # H0 inside slices of the reference mass scale (sample-based; not the section-21 profile)
    mG = col["mu_G"]
    slices = []
    for lo, hi in ((34.6, 35.2), (35.2, 35.6), (35.6, 36.0), (36.0, 36.6)):
        m = (mG >= lo) & (mG < hi)
        slices.append({"mu_G": [lo, hi], "n": int(m.sum()), "H0_mean": float(h[m].mean()),
                       "H0_sd": float(h[m].std())})
    X = np.column_stack([col["mu_G"], col["mu_chi"]])
    Xc = X - X.mean(0)
    b, *_ = np.linalg.lstsq(Xc, h - h.mean(), rcond=None)
    res = h - h.mean() - Xc @ b
    out["H0_given_reference"] = {
        "mu_G_slices": slices,
        "C10_J_H0_mean_sd": [cj["posterior_moments"]["mean"]["H0"], cj["posterior_moments"]["sd"]["H0"]],
        "D11_H0_mean_sd": [float(h.mean()), float(h.std())],
        "linear_regression_on_mu_G_mu_chi": {
            "coef": b.tolist(), "R2": float(1 - res.var() / h.var()),
            "E_H0_at_35_0": float(h.mean() + (np.array([35.0, 0.0]) - X.mean(0)) @ b)}}

    # the two seeds
    r1, r2 = (r["summary"] for r in d["runs"])
    seed = {}
    for n in ("H0",) + POP:
        s_ = s11d[n]["sd"]
        seed[n] = {"median": (r1[n]["median"] - r2[n]["median"]) / s_,
                   "ci90_lo": (r1[n]["ci90"][0] - r2[n]["ci90"][0]) / s_,
                   "ci90_hi": (r1[n]["ci90"][1] - r2[n]["ci90"][1]) / s_}
    out["seed1_minus_seed2_in_sd"] = seed
    out["seed_logZ"] = [(r["logZ"], r["logZerr"]) for r in d["runs"]]

    out["offset_zero"] = {n: {"median_over_sd": s11d[n]["median"] / s11d[n]["sd"],
                              "n_samples_below_0": int((col[n] < 0).sum()),
                              "min_sample": float(col[n].min())}
                          for n in ("dmu_G", "dmu_chi")}
    box = {b[0]: (b[1], b[2]) for b in d["box"]}
    out["edge_fraction_within_2pct_of_box"] = {
        n: float(((col[n] - lo) < 0.02 * (hi - lo)).mean() + ((hi - col[n]) < 0.02 * (hi - lo)).mean())
        for n, (lo, hi) in box.items()}
    out["cost"] = {"ncall_total": d["ncall_total"], "gpu_hours_approx": d["gpu_hours_approx"],
                   "logZ": d["logZ"], "logZerr": d["logZerr"],
                   "n_equal_weight_samples": d["n_equal_weight_samples"]}

    DIAG.mkdir(exist_ok=True)
    dst = DIAG / "a11_11D_comparisons.json"
    dst.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
