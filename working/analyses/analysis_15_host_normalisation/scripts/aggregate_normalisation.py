#!/usr/bin/env python
"""Analysis 15 aggregation -> results/normalisation_summary.json (CPU, seconds).

    python aggregate_normalisation.py [--results DIR]

Per scan: the configuration, each catalog's N_obs, Z_k and F_k = N_obs/Z_k (core's own normaliser,
at H0 = 67.74 and over the H0 rows), the measured posterior (medians, 68%/90% intervals,
correlation), and the normalisation identity's prediction from the scan's reference grid
(LIKELIHOOD_MATH.md section 8):

    ln L_s(H0, f) = ln L_r(H0, f'),   logit f' = logit f + ln(F_A,s/F_G,s) - ln(F_A,r/F_G,r),

evaluated row by row in H0 (F_k depends on H0 through n0 V_c ~ H0^-3) by a cubic spline of the
reference ln L in logit f; cells whose f' falls outside the reference grid are left undefined
(-inf in the predicted posterior; their measured posterior mass is reported). The identity is
exact when the two configurations differ only in Z_k (the numerator is the same wherever a PE
sample or an injection lies): (a), (b), (c) against the zero-density grid, and (d) at the input
density against the grid-to-5 m<18 scan. For (d) at 0.5x and 2x the AGN density the numerator
also changes (events and injections lie where the m<18 catalog is incomplete), so the
prediction from the input-density (d) grid carries only the normaliser part; the difference is
the part from the in-data missing hosts. Also: the Part 1 gate table, the (d) slope, and
cross-checks (b x1 against b x2, b against the zero-density grid).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import a15_common as C  # noqa: E402


def load(res, name):
    import h5py
    p = res / "scans" / f"{name}.h5"
    if not p.exists() or not (res / "scans" / f"{name}.json").exists():
        return None
    with h5py.File(p, "r") as f:
        d = {k: np.asarray(f[k]) for k in ("H0_grid", "f_grid", "lnL", "pe_var", "n_eff")}
        g = f["normaliser"]
        d["norm_H0"] = np.asarray(g["H0"])
        d["obs"] = np.asarray(g["observed_total"])
        d["Z"] = np.asarray(g["Z"])
        d["Zc"] = np.asarray(g["Z_closed_form"])
        d["meta"] = json.loads(f.attrs["meta_json"])
    d["json"] = json.loads((res / "scans" / f"{name}.json").read_text())
    d["lnF"] = np.log(d["obs"] / d["Z"])            # rows: H0 grid then 67.74; cols GAL, AGN
    return d


def shift_rows(s, r):
    """ln(F_A,s/F_G,s) - ln(F_A,r/F_G,r) on the H0 grid rows (scan s and reference r share H0)."""
    hs = s["H0_grid"]
    if not np.allclose(s["norm_H0"][:-1], hs) or not np.allclose(r["norm_H0"][:-1], hs):
        raise ValueError("normaliser rows do not match the H0 grid")
    ds_ = s["lnF"][:-1, 1] - s["lnF"][:-1, 0]
    dr_ = r["lnF"][:-1, 1] - r["lnF"][:-1, 0]
    d67 = (s["lnF"][-1, 1] - s["lnF"][-1, 0]) - (r["lnF"][-1, 1] - r["lnF"][-1, 0])
    return ds_ - dr_, float(d67)


def predict(s, r):
    from scipy.interpolate import CubicSpline
    if not np.allclose(s["H0_grid"], r["H0_grid"]):
        raise ValueError("H0 grids differ")
    d, d67 = shift_rows(s, r)
    xs = C.logit(r["f_grid"])
    fs = s["f_grid"]
    pred = np.full(s["lnL"].shape, np.nan)
    for i in range(len(s["H0_grid"])):
        sp = CubicSpline(xs, r["lnL"][i])
        x = C.logit(fs) + d[i]
        ok = (x >= xs[0] - 1e-12) & (x <= xs[-1] + 1e-12)
        pred[i, ok] = sp(x[ok])
    return pred, d, d67


def compare_grids(meas, pred, fs, hs):
    w = np.exp(meas - np.nanmax(meas))
    defined = np.isfinite(pred)
    bulk = (w > 1e-3) & defined
    diff = (meas - pred)[bulk]
    pfill = np.where(defined, pred, -np.inf)
    nan = float("nan")
    stats = {"mean": float(np.mean(diff)), "rms": float(np.sqrt(np.mean(diff ** 2))),
             "sd": float(np.std(diff)), "max_abs": float(np.max(np.abs(diff)))} if diff.size else \
        {"mean": nan, "rms": nan, "sd": nan, "max_abs": nan}
    return {"n_bulk_cells": int(bulk.sum()),
            "dlnL_measured_minus_predicted": stats,
            "measured_mass_where_prediction_undefined": float(w[~defined].sum() / w.sum()),
            "predicted_summary": C.summarize_2d(hs, fs, pfill)}


def short(sm):
    return {p: {k: sm[p][k] for k in ("median", "ci68", "ci90", "hw68", "w90", "mass_at_edges")} for p in ("H0", "f")} \
        | {"rho": sm["rho"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(C.A15 / "results"))
    args = ap.parse_args()
    res = Path(args.results)
    S = {n: load(res, n) for n in C.BY_NAME}
    out = {"created_by": "scripts/aggregate_normalisation.py", "f_input": C.F_INPUT, "H0_input": C.H0_REF,
           "restricted_grid_end": C.ZMAX_R, "scans": {}, "gate": {}, "checks": {}, "legacy_context": {}}

    for spec in C.SCANS:
        n = spec["name"]
        s = S[n]
        rec = {"label": spec["label"], "part": spec["part"], "rung": spec["rung"],
               "zmax": C.zmax_of(spec["zmax"]), "log10n0": spec["n0"][0], "log10n0_c2": spec["n0"][1],
               "agn_density_factor": float(10 ** (spec["n0"][1] + 5.0)) if spec["n0"][1] > -10 else 0.0,
               "reference_scan": spec["ref"], "depends_on_gate": spec["needs"], "status": "missing"}
        if s is None:
            out["scans"][n] = rec
            continue
        j = s["json"]
        rec["status"] = "done"
        rec["normaliser"] = {
            "N_obs": j["N_obs"], "Z_at_67.74": j["Z_at_67.74"], "F_at_67.74": j["F_at_67.74"],
            "ln_FA_over_FG_at_67.74": j["ln_FA_over_FG_at_67.74"],
            "F_over_H0_rows": {"H0": [float(s["norm_H0"][0]), float(s["norm_H0"][-2])],
                               "F_GAL": [float(np.exp(s["lnF"][0, 0])), float(np.exp(s["lnF"][-2, 0]))],
                               "F_AGN": [float(np.exp(s["lnF"][0, 1])), float(np.exp(s["lnF"][-2, 1]))]},
            "Z_core_over_closed_form_max_abs_rel_dev": j["Z_core_over_closed_max_abs_rel_dev"],
            "what_Z_is": ("Z_k = N_obs,k exactly (C_k = 1 on the whole grid 0 < z < 1.10)"
                          if spec["rung"] == "complete" and spec["zmax"] == "R" else
                          f"Z_k = N_obs,k + n0_k 4 pi int_0^{C.zmax_of(spec['zmax']):g} (1 - C_k) dV_c")}
        rec["measured"] = short(j["summary"])
        rec["grid"] = {"H0": [float(s["H0_grid"][0]), float(s["H0_grid"][-1]), 0.25],
                       "f": [float(s["f_grid"][0]), float(s["f_grid"][-1]), 0.025]}
        rec["seconds_per_call"] = j["seconds_per_call_median"]
        rec["n_nonfinite"] = j["n_nonfinite"]
        rec["n_eff_range"] = j["n_eff_range"]
        rec["pe_var_range"] = j["pe_var_range"]
        r = S.get(spec["ref"]) if spec["ref"] else None
        if spec["ref"] and r is not None:
            pred, d, d67 = predict(s, r)
            cmp_ = compare_grids(s["lnL"], pred, s["f_grid"], s["H0_grid"])
            ps = cmp_.pop("predicted_summary")
            m = j["summary"]
            ref_med = r["json"]["summary"]["f"]["median"]
            rec["prediction"] = {
                "reference_scan": spec["ref"],
                "exact": spec["name"] not in ("d_m18_x05", "d_m18_x2"),
                "logit_shift_at_67.74": d67, "logit_shift_over_H0": [float(d.min()), float(d.max())],
                "f_median_simple": float(C.expit(C.logit(ref_med) - d67)),
                "predicted": short(ps),
                "f_median_measured_minus_predicted": m["f"]["median"] - ps["f"]["median"],
                "f_median_diff_over_hw68": (m["f"]["median"] - ps["f"]["median"]) / ps["f"]["hw68"],
                "H0_median_measured_minus_predicted": m["H0"]["median"] - ps["H0"]["median"],
                "H0_median_diff_over_hw68": (m["H0"]["median"] - ps["H0"]["median"]) / ps["H0"]["hw68"],
                "f_w90_ratio_measured_over_predicted": m["f"]["w90"] / ps["f"]["w90"],
                **cmp_}
        elif spec["ref"]:
            rec["prediction"] = {"reference_scan": spec["ref"], "status": "reference missing"}
        else:
            rec["prediction"] = None
        # the legacy stored result on the same cells
        lp = C.LEGACY_CONTEXT.get(n)
        if lp is not None:
            st = C.stored_on_cells(lp, s["H0_grid"], s["f_grid"])
            if st is not None:
                rec["legacy_same_cells"] = {"file": str(lp), **short(C.summarize_2d(s["H0_grid"], s["f_grid"], st["lnL"]))}
        out["scans"][n] = rec

    # gate table
    for spec in C.SCANS:
        if spec["gate"]:
            p = res / f"gate_{spec['name']}.json"
            if p.exists():
                g = json.loads(p.read_text())
                out["gate"][spec["name"]] = {
                    "pass": g["pass"], "reference": g["reference"],
                    **{p_: {k: g["params"][p_][k] for k in ("core_median", "stored_median", "median_shift_over_hw68",
                                                            "w68_ratio", "w90_ratio", "pass")} for p_ in ("H0", "f")},
                    "dlnL_core_minus_stored_bulk": g["dlnL_core_minus_stored"]}
            else:
                out["gate"][spec["name"]] = {"pass": None, "status": "not run"}

    # cross-checks
    b1, b2, z0, p18, d1 = (S.get(k) for k in ("b_complete_x1", "b_complete_x2", "ref_complete_n0m24", "p1_m18", "d_m18_x1"))
    if b1 is not None and b2 is not None:
        out["checks"]["b_x2_minus_b_x1_max_abs_dlnL"] = float(np.max(np.abs(b2["lnL"] - b1["lnL"])))
    for a, b, key in ((b1, z0, "b_x1_vs_zero_density_grid5"), (d1, p18, "d_x1_vs_m18_grid5")):
        if a is not None and b is not None and np.allclose(a["H0_grid"], b["H0_grid"]):
            # the f cells the two grids share
            ia = [i for i, x in enumerate(a["f_grid"]) if np.min(np.abs(b["f_grid"] - x)) < 1e-9]
            ib = [int(np.argmin(np.abs(b["f_grid"] - a["f_grid"][i]))) for i in ia]
            if ia:
                La, Lb = a["lnL"][:, ia], b["lnL"][:, ib]
                diff = La - Lb
                w = np.exp(La - La.max())
                out["checks"][key] = {"raw_dlnL_bulk_mean": float(np.mean(diff[w > 1e-3])),
                                      "raw_dlnL_bulk_sd": float(np.std(diff[w > 1e-3])),
                                      "raw_dlnL_bulk_max_abs": float(np.max(np.abs(diff[w > 1e-3]))),
                                      "n_common_f_cells": len(ia)}
    # (d) slope in f per dex of the AGN density
    dd = {x: S.get(n) for x, n in ((0.5, "d_m18_x05"), (1.0, "d_m18_x1"), (2.0, "d_m18_x2"))}
    if all(v is not None for v in dd.values()):
        fm = {x: v["json"]["summary"]["f"]["median"] for x, v in dd.items()}
        FA = dd[1.0]["json"]["F_at_67.74"]["AGN"]
        f1 = fm[1.0]
        pm = {x: out["scans"][n]["prediction"]["predicted"]["f"]["median"]
              for x, n in ((0.5, "d_m18_x05"), (2.0, "d_m18_x2"))}
        out["checks"]["d_slope"] = {
            "f_median": {str(k): v for k, v in fm.items()},
            "measured_slope_per_dex": (fm[2.0] - fm[0.5]) / (2 * C.LOG2),
            "normaliser_slope_per_dex_ln10_(1-FA)_f(1-f)": float(np.log(10) * (1 - FA) * f1 * (1 - f1)),
            "F_AGN_at_67.74": FA,
            "identity_predicted_f_median": {"0.5": pm[0.5], "2.0": pm[2.0]},
            "identity_predicted_slope_per_dex": (pm[2.0] - pm[0.5]) / (2 * C.LOG2),
            "residual_from_in_data_missing_hosts_per_dex": ((fm[2.0] - fm[0.5]) - (pm[2.0] - pm[0.5])) / (2 * C.LOG2)}
    # legacy context (stored grids, full 201 x 41 domain)
    import h5py
    for k, p in C.LEGACY_CONTEXT.items():
        with h5py.File(p, "r") as f:
            sm = C.summarize_2d(np.asarray(f["H0_grid"]), np.asarray(f["f_grid"]), np.asarray(f["log_likelihood"]))
            out["legacy_context"][k] = {"file": str(p), "log10n0": float(f.attrs["arg_log10n0"]),
                                        "log10n0_c2": float(f.attrs["arg_log10n0_c2"]), **short(sm)}
    vp = res / "validation_cpu.json"
    if vp.exists():
        v = json.loads(vp.read_text())
        out["data_extent"] = v.get("data_extent")
    (res / "normalisation_summary.json").write_text(json.dumps(out, indent=1))
    rows = []
    for n, r in out["scans"].items():
        if r["status"] != "done":
            rows.append(f"{n:20s} {r['status']}")
            continue
        m = r["measured"]
        pr = r.get("prediction") or {}
        pp = pr.get("predicted", {}).get("f", {}).get("median", float("nan")) if pr else float("nan")
        rows.append(f"{n:20s} F=({r['normaliser']['F_at_67.74']['GAL']:.4g},{r['normaliser']['F_at_67.74']['AGN']:.4g}) "
                    f"f {m['f']['median']:.4f} [{m['f']['ci90'][0]:.3f},{m['f']['ci90'][1]:.3f}] pred {pp:.4f} "
                    f"H0 {m['H0']['median']:.3f} rms {pr.get('dlnL_measured_minus_predicted', {}).get('rms', float('nan')):.3g}")
    print("\n".join(rows))
    print(json.dumps(out["gate"], indent=0)[:2000])
    print(json.dumps(out["checks"], indent=0)[:2000])


if __name__ == "__main__":
    main()
