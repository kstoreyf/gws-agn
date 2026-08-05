#!/usr/bin/env python3
"""Post-fix summary of the deep two-tracer scans (sigma_ang-fixed events, PR #335).

Writes results/summary_fix.json: the _fix scan numbers, the guard/N_eff table on
the fixed events, and the pre/post-fix deltas for every quoted number.
"""
import json
from pathlib import Path

import h5py
import numpy as np

BASE = Path(__file__).resolve().parent.parent
RESULTS = BASE / "results"
TRUTH_F, TRUTH_H0 = 0.30, 67.74


def fblock(tag):
    p = RESULTS / f"{tag}.json"
    if not p.exists():
        return None
    j = json.loads(p.read_text())
    b = j["f"]
    return {"median": b["median"], "ci68": b["ci68"], "ci90": b["ci90"],
            "argmax": b["argmax"],
            "half_width68": 0.5 * (b["ci68"][1] - b["ci68"][0]),
            "offset": b["median"] - TRUTH_F,
            "offset_sigma": (b["median"] - TRUTH_F) / (0.5 * (b["ci68"][1] - b["ci68"][0])),
            "truth_in_ci68": bool(b["ci68"][0] <= TRUTH_F <= b["ci68"][1]),
            "truth_in_ci90": bool(b["ci90"][0] <= TRUTH_F <= b["ci90"][1]),
            "n_rejected": j["n_neginf_cells"], "n_evals": j["n_evals"]}


def jblock(tag):
    p = RESULTS / f"{tag}.json"
    if not p.exists():
        return None
    j = json.loads(p.read_text())
    with h5py.File(RESULTS / f"{tag}.h5", "r") as f:
        H, F, ll = f["H0_grid"][:], f["f_grid"][:], f["log_likelihood"][:]
    ok = np.isfinite(ll)
    pw = np.where(ok, np.exp(ll - np.nanmax(ll[ok])), 0.0)
    pw *= np.outer(np.gradient(H), np.gradient(F))
    pw /= pw.sum()
    Hm, Fm = np.meshgrid(H, F, indexing="ij")
    mH, mF = (Hm * pw).sum(), (Fm * pw).sum()
    sH = float(np.sqrt(((Hm - mH) ** 2 * pw).sum()))
    sF = float(np.sqrt(((Fm - mF) ** 2 * pw).sum()))
    rho = float(((Hm - mH) * (Fm - mF) * pw).sum() / (sH * sF)) if sH * sF else 0.0
    from scipy.ndimage import binary_dilation
    edge = binary_dilation(~ok) & ok
    hb, fb = j["H0"], j["f"]
    hw = 0.5 * (hb["ci68"][1] - hb["ci68"][0])
    return {"map": j["map"], "H0_median": hb["median"], "H0_ci68": hb["ci68"],
            "H0_ci90": hb["ci90"], "H0_half_width68": hw,
            "H0_offset": hb["median"] - TRUTH_H0,
            "H0_offset_sigma": (hb["median"] - TRUTH_H0) / hw,
            "H0_truth_in_ci90": bool(hb["ci90"][0] <= TRUTH_H0 <= hb["ci90"][1]),
            "f_median": fb["median"], "f_ci68": fb["ci68"],
            "f_half_width68": 0.5 * (fb["ci68"][1] - fb["ci68"][0]),
            "rho": rho, "n_rejected": int((~ok).sum()), "n_evals": int(ok.size),
            "posterior_mass_adjacent_to_rejected": float(pw[edge].sum())}


def guard_table(sfx):
    rows = []
    for fs in ("0.0", "0.3", "0.7", "1.0"):
        p = RESULTS / f"guard_targeted_f{fs}{sfx}.json"
        if not p.exists():
            continue
        r = json.loads(p.read_text())["guard_records"][0]
        rows.append({"f": float(fs), "Neff": r["Neff"], "threshold": r["threshold"],
                     "passes": r["passes_legacy_floor"],
                     "pe_variance_sum": r["pe_variance_sum"]})
    return rows


def main():
    S = {"events": "data_derived/twotracer_gw_events_fix.h5",
         "events_check": json.loads((RESULTS / "events_fix_check.json").read_text()),
         "truth": {"f_AGN": TRUTH_F, "H0": TRUTH_H0},
         "guard_targeted_fix": guard_table("_fix"),
         "guard_targeted_prefix": guard_table(""),
         "fix": {"tgt_fscan_n80": fblock("tgt_fscan_n80_fix"),
                 "tgt_fscan_n200": fblock("tgt_fscan_n200_fix"),
                 "tgt_joint_n200": jblock("tgt_joint_n200_fix")},
         "prefix": {"tgt_fscan_n80": fblock("tgt_fscan_n80"),
                    "tgt_fscan_n200": fblock("tgt_fscan_n200"),
                    "tgt_joint_n200": jblock("tgt_joint_n200")}}
    d = {}
    for k in ("tgt_fscan_n80", "tgt_fscan_n200"):
        a, b = S["prefix"][k], S["fix"][k]
        if a and b:
            d[k] = {"f_median_shift": b["median"] - a["median"],
                    "width_ratio_fix_over_pre": b["half_width68"] / a["half_width68"]}
    a, b = S["prefix"]["tgt_joint_n200"], S["fix"]["tgt_joint_n200"]
    if a and b:
        d["tgt_joint_n200"] = {
            "H0_median_shift": b["H0_median"] - a["H0_median"],
            "f_median_shift": b["f_median"] - a["f_median"],
            "H0_width_ratio": b["H0_half_width68"] / a["H0_half_width68"],
            "f_width_ratio": b["f_half_width68"] / a["f_half_width68"],
            "rho_pre_post": [a["rho"], b["rho"]]}
    S["pre_to_post_deltas"] = d
    (RESULTS / "summary_fix.json").write_text(json.dumps(S, indent=2, default=float))
    print("wrote results/summary_fix.json")
    for k in ("tgt_fscan_n80", "tgt_fscan_n200"):
        a, b = S["prefix"][k], S["fix"][k]
        if a and b:
            print(f"{k}: f {a['median']:.4f} -> {b['median']:.4f} "
                  f"(hw {a['half_width68']:.4f} -> {b['half_width68']:.4f})")
    if a and b and "tgt_joint_n200" in d:
        a, b = S["prefix"]["tgt_joint_n200"], S["fix"]["tgt_joint_n200"]
        print(f"joint: H0 {a['H0_median']:.3f} -> {b['H0_median']:.3f} "
              f"({b['H0_offset']:+.3f} = {b['H0_offset_sigma']:+.2f} sigma), "
              f"f {a['f_median']:.4f} -> {b['f_median']:.4f}, "
              f"rho {a['rho']:+.3f} -> {b['rho']:+.3f}")


if __name__ == "__main__":
    main()
