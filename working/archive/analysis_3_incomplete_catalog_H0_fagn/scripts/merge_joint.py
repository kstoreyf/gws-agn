#!/usr/bin/env python3
"""Stitch H0-chunked joint (H0, fcat_2) grids into one grid + summary.

The joint grid this analysis needs is 201 x 41 = 8241 likelihood evaluations, and
a K = 2 evaluation on the COMPLETE GAL block costs seconds, not milliseconds, so
each seed's grid is split into contiguous H0 chunks that run on separate GPUs and
are stitched here.  The chunk boundaries are exact subsets of
`numpy.linspace(50, 100, 201)` (step 0.25, a power of two, so no float drift), and
the merge asserts that the reassembled axis reproduces that linspace bit for bit.

Output mirrors `scan_h0f.py --scan joint` exactly: `<tag>.h5` (H0_grid, f_grid,
log_likelihood, guard/*, provenance attrs) and `<tag>.json` (MAP, marginal
medians and equal-tailed 68/90% CIs for BOTH parameters, the 2-D correlation rho,
the guard block).  The summary is computed by the same code path -- `marginal_ci`
is imported from `scan_h0f`, not reimplemented.
"""
import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scan_h0f import marginal_ci, add_truth_flags  # noqa: E402

GUARD_KEYS = ("rejected", "Neff", "pe_variance_sum",
              "selection_variance_N2_over_Neff", "sigma2_total", "threshold",
              "legacy_floor_5N", "passes", "passes_legacy_floor")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chunks", nargs="+", required=True,
                    help="chunk .h5 files, any order (sorted here by H0_grid[0])")
    ap.add_argument("--out_tag", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--h0_grid", nargs=3, type=float, default=[50.0, 100.0, 201.0],
                    metavar=("MIN", "MAX", "N"),
                    help="the full H0 axis the chunks must reassemble into")
    ap.add_argument("--h0_true", type=float, default=67.74)
    ap.add_argument("--f_true", type=float, default=0.30)
    return ap.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    files = sorted(a.chunks, key=lambda p: h5py.File(p, "r")["H0_grid"][0])

    H0_parts, ll_parts = [], []
    guard_parts = {k: [] for k in GUARD_KEYS}
    f_vals = None
    attrs0 = None
    wall = 0.0
    eval_s = 0.0
    steady = []
    for p in files:
        with h5py.File(p, "r") as f:
            h0 = np.asarray(f["H0_grid"][:], dtype=float)
            fg = np.asarray(f["f_grid"][:], dtype=float)
            ll = np.asarray(f["log_likelihood"][:], dtype=float)
            if f_vals is None:
                f_vals = fg
                attrs0 = dict(f.attrs)
            elif not np.array_equal(f_vals, fg):
                sys.exit(f"[fatal] f_grid mismatch in {p}")
            if ll.shape != (h0.size, fg.size):
                sys.exit(f"[fatal] log_likelihood shape {ll.shape} != {(h0.size, fg.size)} in {p}")
            H0_parts.append(h0)
            ll_parts.append(ll)
            for k in GUARD_KEYS:
                guard_parts[k].append(np.asarray(f["guard"][k][:]) if k in f["guard"]
                                      else np.full(ll.shape, np.nan))
            wall += float(f.attrs.get("wall_seconds_total", 0.0))
            eval_s += float(f.attrs.get("total_eval_seconds", 0.0))
            steady.append(float(f.attrs.get("steady_state_median_seconds", np.nan)))

    H0_vals = np.concatenate(H0_parts)
    ll_grid = np.concatenate(ll_parts, axis=0)
    guard = {k: np.concatenate(v, axis=0) for k, v in guard_parts.items()}

    want = np.linspace(a.h0_grid[0], a.h0_grid[1], int(round(a.h0_grid[2])))
    if H0_vals.size != want.size or not np.array_equal(H0_vals, want):
        # no float drift is expected (step 0.25); report exactly what differs
        bad = int(np.sum(H0_vals != want)) if H0_vals.size == want.size else -1
        sys.exit(f"[fatal] reassembled H0 axis does not reproduce the target linspace "
                 f"(size {H0_vals.size} vs {want.size}, {bad} differing values)")

    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    h5_path = outdir / f"{a.out_tag}.h5"
    json_path = outdir / f"{a.out_tag}.json"

    n = ll_grid.size
    rejected = ~np.isfinite(ll_grid)
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("H0_grid", data=H0_vals)
        f.create_dataset("f_grid", data=f_vals)
        f.create_dataset("log_likelihood", data=ll_grid)
        g = f.create_group("guard")
        for k, v in guard.items():
            g.create_dataset(k, data=v)
        g.attrs["guard_record_enabled"] = True
        g.attrs["n_rejected"] = int(rejected.sum())
        for k, v in (attrs0 or {}).items():
            if k in ("arg_h0_grid", "arg_out_tag", "arg_outdir", "wall_seconds_total",
                     "total_eval_seconds", "steady_state_median_seconds",
                     "first_eval_seconds", "n_evals", "n_neginf_cells"):
                continue
            f.attrs[k] = v
        f.attrs["arg_h0_grid"] = json.dumps(list(a.h0_grid))
        f.attrs["merged_from"] = json.dumps([str(p) for p in files])
        f.attrs["n_chunks"] = len(files)
        f.attrs["wall_seconds_total"] = wall
        f.attrs["total_eval_seconds"] = eval_s
        f.attrs["steady_state_median_seconds"] = float(np.nanmedian(steady))
        f.attrs["n_evals"] = int(n)
        f.attrs["n_neginf_cells"] = int(rejected.sum())
    print(f"Wrote {h5_path}  shape={ll_grid.shape}  rejected={int(rejected.sum())}/{n}")

    # ---- joint summary: the scan_h0f.py joint branch, verbatim in structure ----
    ll_safe = np.where(np.isfinite(ll_grid), ll_grid, -np.inf)
    finite = np.isfinite(ll_safe)
    lmax = float(ll_safe[finite].max()) if finite.any() else 0.0
    summary = {
        "file": str(h5_path),
        "scan": "joint",
        "merged_from": [str(p) for p in files],
        "labels": json.loads(attrs0.get("labels", "[]")) if attrs0 else [],
        "n_catalogs": int(attrs0.get("n_catalogs", 2)) if attrs0 else 2,
        "n_evals": int(n),
        "n_neginf_cells": int(rejected.sum()),
        "n_rejected": int(rejected.sum()),
        "logL_max": lmax,
        "base_coord": json.loads(attrs0.get("base_coord_labeled", "{}")) if attrs0 else {},
        "selection_neff_guard": str(attrs0.get("selection_neff_guard", "hard")),
        "max_likelihood_variance_effective":
            float(attrs0.get("max_likelihood_variance_effective", 1e6)),
        "all_cells_rejected": bool(not finite.any()),
        "timing": {"total_eval_seconds": eval_s,
                   "steady_state_median_seconds": float(np.nanmedian(steady)),
                   "wall_seconds_total": wall},
    }

    i, j = np.unravel_index(int(np.nanargmax(np.where(finite, ll_safe, np.nan))),
                            ll_safe.shape)
    summary["map"] = {"H0": float(H0_vals[i]), "f": float(f_vals[j]), "logL": lmax}
    p2d = np.where(finite, np.exp(ll_safe - lmax), 0.0)
    logp_H0 = np.log(np.maximum(np.trapz(p2d, f_vals, axis=1), 1e-300))
    logp_f = np.log(np.maximum(np.trapz(p2d, H0_vals, axis=0), 1e-300))
    bH0 = marginal_ci(H0_vals, logp_H0)
    bH0["map"] = float(H0_vals[i]); bH0["argmax"] = float(H0_vals[i])
    bf = marginal_ci(f_vals, logp_f)
    bf["map"] = float(f_vals[j]); bf["argmax"] = float(f_vals[j])
    add_truth_flags(bH0, a.h0_true)
    add_truth_flags(bf, a.f_true)
    bH0["grid"] = [float(v) for v in H0_vals]
    bf["grid"] = [float(v) for v in f_vals]
    summary["H0"] = bH0
    summary["f"] = bf

    norm = np.trapz(np.trapz(p2d, f_vals, axis=1), H0_vals, axis=0)
    if np.isfinite(norm) and norm > 0:
        Zn = p2d / norm
        pH0 = np.trapz(Zn, f_vals, axis=1)
        pf = np.trapz(Zn, H0_vals, axis=0)
        EH0 = np.trapz(H0_vals * pH0, H0_vals)
        Ef = np.trapz(f_vals * pf, f_vals)
        VH0 = np.trapz((H0_vals - EH0) ** 2 * pH0, H0_vals)
        Vf = np.trapz((f_vals - Ef) ** 2 * pf, f_vals)
        H0g, fg2 = np.meshgrid(H0_vals, f_vals, indexing="ij")
        Cov = np.trapz(np.trapz((H0g - EH0) * (fg2 - Ef) * Zn, f_vals, axis=1),
                       H0_vals, axis=0)
        summary["rho"] = (float(Cov / np.sqrt(VH0 * Vf))
                          if VH0 > 0 and Vf > 0 else float("nan"))
        summary["moments"] = {"E_H0": float(EH0), "E_f": float(Ef),
                              "sigma_H0": float(np.sqrt(VH0)),
                              "sigma_f": float(np.sqrt(Vf)), "cov": float(Cov)}
    else:
        summary["rho"] = float("nan")

    # ---- guard block, with the f-dimension behaviour made explicit -------------
    neff = guard["Neff"]
    fin = np.isfinite(neff)
    guard_block = {
        "enabled": True,
        "n_cells": int(n),
        "n_rejected": int(rejected.sum()),
        "rejected_fraction": float(rejected.mean()),
        "selection_neff_guard": summary["selection_neff_guard"],
        "max_likelihood_variance_effective": summary["max_likelihood_variance_effective"],
        "summary": {
            "Neff_min": float(neff[fin].min()) if fin.any() else None,
            "Neff_median": float(np.median(neff[fin])) if fin.any() else None,
            "Neff_max": float(neff[fin].max()) if fin.any() else None,
            "pe_variance_sum_min": float(np.nanmin(guard["pe_variance_sum"])),
            "pe_variance_sum_max": float(np.nanmax(guard["pe_variance_sum"])),
            "sigma2_total_min": float(np.nanmin(guard["sigma2_total"])),
            "sigma2_total_max": float(np.nanmax(guard["sigma2_total"])),
            "threshold_min": float(np.nanmin(guard["threshold"])),
            "threshold_max": float(np.nanmax(guard["threshold"])),
            "n_pass_legacy_floor": int(np.sum(guard["passes_legacy_floor"] == 1)),
            "n_fail_legacy_floor": int(np.sum(guard["passes_legacy_floor"] == 0)),
            "n_guard_would_reject": int(np.sum(guard["passes"] == 0)),
        },
    }
    # N_eff along f at the H0 cell nearest truth, and the global f-profile
    ih0 = int(np.argmin(np.abs(H0_vals - a.h0_true)))
    guard_block["neff_vs_f_at_truth_H0"] = {
        "H0": float(H0_vals[ih0]),
        "f": [float(v) for v in f_vals],
        "Neff": [float(v) for v in neff[ih0]],
        "pe_variance_sum": [float(v) for v in guard["pe_variance_sum"][ih0]],
        "sigma2_total": [float(v) for v in guard["sigma2_total"][ih0]],
        "threshold": [float(v) for v in guard["threshold"][ih0]],
    }
    guard_block["neff_vs_f_min_over_H0"] = {
        "f": [float(v) for v in f_vals],
        "Neff_min": [float(v) for v in np.nanmin(neff, axis=0)],
        "Neff_max": [float(v) for v in np.nanmax(neff, axis=0)],
    }
    summary["guard"] = guard_block

    json_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {json_path}")
    terse = {k: v for k, v in summary.items() if k not in ("guard", "H0", "f")}
    terse["H0"] = {k: v for k, v in summary["H0"].items() if k != "grid"}
    terse["f"] = {k: v for k, v in summary["f"].items() if k != "grid"}
    print(json.dumps(terse, indent=2))


if __name__ == "__main__":
    main()
