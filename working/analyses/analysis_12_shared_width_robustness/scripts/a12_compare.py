#!/usr/bin/env python
"""Analysis 12 against 11D (brief section 25): does one shared width absorb its offset?

    python scripts/a12_compare.py

Reads results/a12_{12M,12chi}.json/.npz (merged), their per-seed run JSONs, and
../analysis_11_free_common_population/results/a11_11D.json; writes
diagnostics/a12_comparisons.json.

Per arm: R = W(offset with the width free) / W(offset in 11D) at 68% and 90%, the median
shift, rho(width, offset), the response of every other coordinate (H0 included), seed
agreement, containment, the offset's distance from zero, and the realised spin and mass
statistics of the mock's events by host branch (what one draw actually contains).
"""
import json
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
A12 = HERE.parent
RESULTS, DIAG = A12 / "results", A12 / "diagnostics"
A11_RESULTS = A12.parent / "analysis_11_free_common_population" / "results"
EVENTS = A12.parent.parent / "data" / "seed100" / "events" / "events_marked_dmu0p10_dmuG5.h5"
ARMS = {"12M": ("sigma_G", "dmu_G", 5.0), "12chi": ("sigma_chi", "dmu_chi", 0.10)}
COMMON = ("H0", "f_agn", "mu_G", "dmu_G", "mu_chi", "dmu_chi")


def width(s, level):
    lo, hi = s[f"ci{level}"]
    return hi - lo


def ratio(a, b):
    return {"W68": width(a, 68) / width(b, 68), "W90": width(a, 90) / width(b, 90),
            "median_shift": a["median"] - b["median"],
            "median_shift_over_11D_sd": (a["median"] - b["median"]) / b["sd"]}


def edge_check(x, lo, hi, bin_frac=0.02):
    """How much a coordinate leans on the TOP of its prior box.

    The density in the last bin_frac of the box against the peak bin, and a normal
    truncated to the box fitted to the samples: its untruncated mass above the edge and
    the 90% interval it implies without the cut (an extrapolation, recorded as such).
    """
    from scipy import stats
    from scipy.optimize import minimize
    w = bin_frac * (hi - lo)
    h, _ = np.histogram(x, bins=np.arange(lo, hi + 0.5 * w, w))
    mu0, s0 = float(x.mean()), float(x.std())
    nll = lambda p: -np.sum(stats.truncnorm.logpdf(x, (lo - p[0]) / np.exp(p[1]),
                                                  (hi - p[0]) / np.exp(p[1]),
                                                  loc=p[0], scale=np.exp(p[1])))
    r = minimize(nll, [mu0, np.log(s0)], method="Nelder-Mead")
    mu, sd = float(r.x[0]), float(np.exp(r.x[1]))
    return {"top_bin_over_peak": float(h[-1] / h.max()),
            "frac_within_2pct_of_top": float((x > hi - w).mean()),
            "truncnorm_mu": mu, "truncnorm_sd": sd,
            "untruncated_mass_above_edge": float(1 - stats.norm.cdf(hi, mu, sd)),
            "untruncated_ci90": [mu - 1.6449 * sd, mu + 1.6449 * sd]}


def realised():
    with h5py.File(EVENTS, "r") as f:
        ht, chi, m1 = f["truth/host_type"][:], f["truth/chieff"][:], f["truth/m1src"][:]
        obs, sig = f["truth/obs_chieff"][:], f["truth/obs_sig_chieff"][:]
    out = {}
    noise_sd = float((obs - chi).std(ddof=1))
    for t, name in ((0, "GAL"), (1, "AGN")):
        m = ht == t
        c, o = chi[m], obs[m]
        o_sd = float(o.std(ddof=1))
        o_sd_se = o_sd / np.sqrt(2 * o.size)
        dec = float(np.sqrt(max(o_sd ** 2 - noise_sd ** 2, 0.0)))
        out[name] = {"n": int(c.size), "chi_eff_mean": float(c.mean()),
                     "chi_eff_sd": float(c.std(ddof=1)),
                     "chi_eff_mean_se": float(c.std(ddof=1) / np.sqrt(c.size)),
                     "chi_eff_sd_se": float(c.std(ddof=1) / np.sqrt(2 * c.size)),
                     "obs_chi_eff_mean": float(o.mean()),
                     "obs_chi_eff_sd": o_sd, "obs_chi_eff_sd_se": o_sd_se,
                     "obs_chi_eff_sd_expected": float(np.sqrt(c.std(ddof=1) ** 2 + noise_sd ** 2)),
                     # moment deconvolution: what the observed spread implies for the
                     # intrinsic width once the (calibrated) measurement noise is removed
                     "deconvolved_sd": dec,
                     "deconvolved_sd_se": float(o_sd / dec * o_sd_se) if dec > 0 else None,
                     "m1src_median": float(np.median(m1[m]))}
    out["dmu_chi_realised"] = out["AGN"]["chi_eff_mean"] - out["GAL"]["chi_eff_mean"]
    out["dmu_chi_observed"] = out["AGN"]["obs_chi_eff_mean"] - out["GAL"]["obs_chi_eff_mean"]
    out["chi_eff_noise"] = {"obs_minus_true_sd": noise_sd,
                            "mean_quoted_sigma": float(sig.mean()),
                            "pull_sd": float(((obs - chi) / sig).std(ddof=1))}
    out["note"] = ("true and observed chi_eff of the 1000 detected events by true host; the "
                   "population widths are intrinsic, detection selects weakly on chi_eff, and "
                   "the deconvolution is a moment estimate for orientation, not the likelihood")
    return out


def main():
    d11 = json.loads((A11_RESULTS / "a11_11D.json").read_text())
    s11 = d11["summary"]
    out = {"what": __doc__.split("    python")[0].strip(), "realised": realised(), "arms": {}}
    for arm, (w, mark, truth_w) in ARMS.items():
        jp = RESULTS / f"a12_{arm}.json"
        if not jp.exists():
            print(f"[skip] {arm}: not merged")
            continue
        d = json.loads(jp.read_text())
        s = d["summary"]
        Z = np.load(RESULTS / f"a12_{arm}.npz")
        S, names = Z["samples"], [str(n) for n in Z["names"]]
        col = {n: S[:, names.index(n)] for n in names}
        wts = np.exp(Z["logwt"] - Z["logwt"].max())
        a = {"width": w, "mark": mark, "width_truth": truth_w,
             "width_summary": s[w], "mark_summary": s[mark],
             "R": ratio(s[mark], s11[mark]),
             "rho_width_mark": s["correlations"].get(f"{mark}|{w}", s["correlations"].get(f"{w}|{mark}")),
             "rho_width_all": {k: s["correlations"].get(f"{k}|{w}", s["correlations"].get(f"{w}|{k}"))
                               for k in COMMON},
             "vs_11D": {k: ratio(s[k], s11[k]) for k in COMMON},
             "width_truth_in_ci90": bool(s[w]["ci90"][0] <= truth_w <= s[w]["ci90"][1]),
             "mark_zero": {"median_over_sd": s[mark]["median"] / s[mark]["sd"],
                           "n_samples_below_0": int((col[mark] < 0).sum())},
             "logZ": d["logZ"], "logZerr": d["logZerr"],
             "logZ_minus_11D": d["logZ"] - d11["logZ"],
             "ncall_total": d["ncall_total"], "gpu_hours_approx": d["gpu_hours_approx"],
             "n_equal_weight_samples": d["n_equal_weight_samples"],
             "kish_neff": float(wts.sum() ** 2 / (wts ** 2).sum())}
        box = {b[0]: (b[1], b[2]) for b in d["box"]}
        a["edge_fraction_within_2pct_of_box"] = {
            n: float(((col[n] - lo) < 0.02 * (hi - lo)).mean() + ((hi - col[n]) < 0.02 * (hi - lo)).mean())
            for n, (lo, hi) in box.items()}
        a["mark_top_edge"] = edge_check(col[mark], *box[mark])
        if a["mark_top_edge"]["frac_within_2pct_of_top"] > 1e-3:
            u = a["mark_top_edge"]["untruncated_ci90"]
            a["R_untruncated_W90"] = (u[1] - u[0]) / width(s11[mark], 90)
        runs = [r["summary"] for r in d["runs"]]
        if len(runs) == 2:
            a["seed1_minus_seed2_in_sd"] = {
                n: {"median": (runs[0][n]["median"] - runs[1][n]["median"]) / s[n]["sd"],
                    "ci90_lo": (runs[0][n]["ci90"][0] - runs[1][n]["ci90"][0]) / s[n]["sd"],
                    "ci90_hi": (runs[0][n]["ci90"][1] - runs[1][n]["ci90"][1]) / s[n]["sd"]}
                for n in COMMON + (w,)}
            a["max_seed_disagreement_sd"] = max(abs(v) for x in a["seed1_minus_seed2_in_sd"].values()
                                                for v in x.values())
        a["seed_logZ"] = [(r["logZ"], r["logZerr"]) for r in d["runs"]]
        out["arms"][arm] = a
    DIAG.mkdir(exist_ok=True)
    dst = DIAG / "a12_comparisons.json"
    dst.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
