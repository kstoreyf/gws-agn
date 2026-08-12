#!/usr/bin/env python
"""Rebuild a joint-scan summary JSON from its grid h5.

Why this exists: J2 ships numpy 2, which removed `np.trapz`. Every marginal in
`scan_h0f.py` used it, and all eleven call sites run AFTER the grid is computed
-- so the first four tasks burned their full GPU time, wrote a complete 8241-cell
h5, and then died in post-processing. The grids are intact; only the summary was
lost. Recomputing them would cost ~2.4 GPU-h for nothing.

The marginalisation is imported from `scan_h0f` (`marginal_ci`, `add_truth_flags`)
and the joint branch is reproduced here verbatim, so the numbers come from the
same code that would have written them.

VALIDATE BEFORE TRUSTING. `--validate <h5> <json>` reruns this against an
archived pair where both already exist and diffs every scalar; that is what
licenses using it on the new grids.

    python recover_summary_from_h5.py --validate <archived.h5> <archived.json>
    python recover_summary_from_h5.py results/joint_m21_s100.h5 [...]
"""
import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scan_h0f import marginal_ci, add_truth_flags, _trapz  # noqa: E402


def summarize_joint(h5_path):
    """The `--scan joint` branch of scan_h0f.main(), fed from the stored grid."""
    with h5py.File(h5_path, "r") as f:
        a = dict(f.attrs)
        H0_vals = np.asarray(f["H0_grid"])
        f_vals = np.asarray(f["f_grid"])
        ll = np.asarray(f["log_likelihood"])
        n_rej = int(np.asarray(f["guard/rejected"]).sum()) if "guard" in f else 0

    finite = np.isfinite(ll)
    if not finite.any():
        raise SystemExit(f"{h5_path}: every cell is non-finite")
    lmax = float(np.nanmax(np.where(finite, ll, np.nan)))
    ll_safe = np.where(finite, ll, -np.inf)

    i, j = np.unravel_index(
        int(np.nanargmax(np.where(finite, ll_safe, np.nan))), ll_safe.shape)
    out = {"map": {"H0": float(H0_vals[i]), "f": float(f_vals[j]), "logL": lmax}}

    p2d = np.where(finite, np.exp(ll_safe - lmax), 0.0)
    logp_H0 = np.log(np.maximum(_trapz(p2d, f_vals, axis=1), 1e-300))
    logp_f = np.log(np.maximum(_trapz(p2d, H0_vals, axis=0), 1e-300))
    bH0 = marginal_ci(H0_vals, logp_H0)
    bH0["map"] = float(H0_vals[i]); bH0["argmax"] = float(H0_vals[i])
    bf = marginal_ci(f_vals, logp_f)
    bf["map"] = float(f_vals[j]); bf["argmax"] = float(f_vals[j])
    h0_true = a.get("arg_h0_true"); f_true = a.get("arg_f_true")
    add_truth_flags(bH0, None if h0_true is None else float(h0_true))
    add_truth_flags(bf, None if f_true is None else float(f_true))
    out["H0"] = bH0
    out["f"] = bf

    norm = _trapz(_trapz(p2d, f_vals, axis=1), H0_vals, axis=0)
    if np.isfinite(norm) and norm > 0:
        Zn = p2d / norm
        pH0 = _trapz(Zn, f_vals, axis=1)
        pf = _trapz(Zn, H0_vals, axis=0)
        EH0 = _trapz(H0_vals * pH0, H0_vals)
        Ef = _trapz(f_vals * pf, f_vals)
        VH0 = _trapz((H0_vals - EH0) ** 2 * pH0, H0_vals)
        Vf = _trapz((f_vals - Ef) ** 2 * pf, f_vals)
        H0g, fg = np.meshgrid(H0_vals, f_vals, indexing="ij")
        Cov = _trapz(_trapz((H0g - EH0) * (fg - Ef) * Zn, f_vals, axis=1),
                     H0_vals, axis=0)
        out["rho"] = (float(Cov / np.sqrt(VH0 * Vf))
                      if VH0 > 0 and Vf > 0 else float("nan"))

    out["n_evals"] = int(ll.size)
    out["n_neginf_cells"] = int((~finite).sum())
    out["n_guard_rejected"] = n_rej
    out["c_mode"] = str(a.get("c_mode", ""))
    out["file"] = str(h5_path)
    out["_recovered"] = (
        "Summary rebuilt from the stored grid after scan_h0f's post-processing "
        "hit numpy 2's removal of np.trapz. The GRID itself completed normally "
        "and is unmodified; only the marginals are recomputed, by the same "
        "code path (scan_h0f.marginal_ci / add_truth_flags).")
    return out


def flatten(d, pre=""):
    o = {}
    for k, v in d.items():
        if isinstance(v, dict):
            o.update(flatten(v, f"{pre}{k}."))
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            o[f"{pre}{k}"] = float(v)
    return o


def validate(h5_path, json_path):
    got = flatten(summarize_joint(h5_path))
    ref = flatten(json.loads(Path(json_path).read_text()))
    keys = sorted(set(got) & set(ref))
    if not keys:
        raise SystemExit("no comparable scalars -- wrong pair?")
    worst, worst_k = 0.0, None
    for k in keys:
        a, b = got[k], ref[k]
        if not (np.isfinite(a) and np.isfinite(b)):
            continue
        d = abs(a - b) / max(1.0, abs(b))
        if d > worst:
            worst, worst_k = d, k
    print(f"  compared {len(keys)} scalars against {Path(json_path).name}")
    print(f"  worst relative difference: {worst:.3e}  ({worst_k})")
    ok = worst < 1e-12
    print("  VALIDATED — recovery reproduces the archived summary" if ok
          else "  FAILED — do NOT use this to recover grids")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("h5", nargs="*")
    ap.add_argument("--validate", nargs=2, metavar=("H5", "JSON"))
    a = ap.parse_args()

    if a.validate:
        raise SystemExit(0 if validate(*a.validate) else 1)

    for p in a.h5:
        p = Path(p)
        out = p.with_suffix(".json")
        out.write_text(json.dumps(summarize_joint(p), indent=2))
        s = json.loads(out.read_text())
        print(f"  {out.name}: H0 {s['H0']['median']:.3f}  "
              f"f {s['f']['median']:.4f}  rho {s.get('rho', float('nan')):+.3f}  "
              f"rejected {s['n_guard_rejected']}")


if __name__ == "__main__":
    main()
