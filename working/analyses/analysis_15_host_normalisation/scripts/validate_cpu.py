#!/usr/bin/env python
"""Analysis 15 CPU validation (before the GPU job) -> results/validation_cpu.json.

    python validate_cpu.py [--dryrun logs/dryrun]

1. Where the data lie: z(d_L; H0) of the 2,205,380 detected injections and the 2e6 PE samples
   at the smallest and largest H0 of the grid (64, 74; 67.74 and 100 for reference), against the
   catalog edge (1.014), the restricted grid end (1.10) and the redshift where the complete
   catalogs' C first drops below 1.
2. The completeness curves C_k(z) of every rung (Schechter fits as stored) and where C first < 1.
3. Z_k and F_k by the closed form for every scan configuration at H0 = 67.74, against the table
   in paper_prd/notes/LIKELIHOOD_MATH.md section 4 (grid to 5, input densities).
4. Z_k evaluated by darksirens-core's own normaliser code in the dry run (each scan file's
   `normaliser` group) against the closed form on core's grid.
5. The dry-run scans: finite ln L, seconds per call, the diag == bound check (in the log).
"""
import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
import sys  # noqa: E402
sys.path.insert(0, str(HERE))
import a15_common as C  # noqa: E402


def z_of_dl(dl, H0, zmax=4.0):
    z = np.expm1(np.linspace(0.0, np.log1p(zmax), 400001))
    chi, _ = C.comoving(z, H0)
    return np.interp(dl, (1 + z) * chi, z)


def main():
    import h5py
    ap = argparse.ArgumentParser()
    ap.add_argument("--dryrun", default=str(C.A15 / "logs" / "dryrun"))
    ap.add_argument("--out", default=str(C.A15 / "results" / "validation_cpu.json"))
    args = ap.parse_args()
    out = {"created_by": "scripts/validate_cpu.py"}

    # 1. data extents
    with h5py.File(C.INJECTIONS, "r") as f:
        dli = np.asarray(f["dL"])
    with h5py.File(C.EVENTS, "r") as f:
        dle = np.asarray(f["dL"])
    ext = {}
    for H0 in (64.0, C.H0_REF, 74.0, 100.0):
        zi, ze = z_of_dl(dli, H0), z_of_dl(dle, H0)
        ext[f"{H0:g}"] = {"injections_z_max": float(zi.max()), "pe_samples_z_max": float(ze.max()),
                          "injections_above": {str(t): int((zi > t).sum()) for t in (1.014, 1.05, 1.10, 1.16)},
                          "pe_samples_above": {str(t): int((ze > t).sum()) for t in (1.014, 1.10)}}
    out["data_extent"] = {"n_injections": int(dli.size), "n_pe_samples": int(dle.size), "by_H0": ext,
                          "restricted_grid_end": C.ZMAX_R,
                          "verdict": ("no detected injection and no PE sample lies above the restricted grid end "
                                      f"{C.ZMAX_R} anywhere on H0 64-74")
                          if ext["74"]["injections_above"]["1.1"] == 0 and ext["74"]["pe_samples_above"]["1.1"] == 0
                          else "TRUNCATION: data above the restricted grid end"}

    # 2. completeness curves
    zf = np.linspace(0.0, 5.0, 50001)
    curves = {}
    for r, R in C.RUNGS.items():
        for t, p in zip(("GAL", "AGN"), R["fits"]):
            fit = C.schechter_fit(p)
            c = C.c_sel(zf, fit)
            below = np.nonzero(c < 1.0)[0]
            curves[f"{r}_{t}"] = {"fit": str(p), **{k: fit[k] for k in ("m_lim", "Mstar_hat", "alpha", "M_faint_offset")},
                                  "z_first_C_below_1": float(zf[below[0]]) if below.size else None,
                                  "C_at": {str(z): float(C.c_sel(np.array([z]), fit)[0])
                                           for z in (0.1, 0.2, 0.37, 0.5, 1.0, 1.014, 1.10, 1.16, 1.2, 1.69, 2.0, 3.0)}}
    out["completeness_curves"] = curves
    out["completeness_notes_z_first_below_1"] = {"complete_GAL": 1.16, "complete_AGN": 1.69, "m21": 0.37, "m18": 0.11}

    # 3. closed-form Z_k, F_k for every scan configuration
    nobs = {r: [C.survey_nobs(p) for p in R["surveys"]] for r, R in C.RUNGS.items()}
    fits = {r: [C.schechter_fit(p) for p in R["fits"]] for r, R in C.RUNGS.items()}
    zk = {}
    for s in C.SCANS:
        r = s["rung"]
        zmax = C.zmax_of(s["zmax"])
        rows = {}
        for H0 in (64.0, C.H0_REF, 74.0):
            Z = [C.z_closed(nobs[r][k], s["n0"][k], fits[r][k], H0, zmax, "fine") for k in (0, 1)]
            F = [nobs[r][k] / Z[k] for k in (0, 1)]
            rows[f"{H0:g}"] = {"Z_GAL": Z[0], "Z_AGN": Z[1], "F_GAL": F[0], "F_AGN": F[1],
                               "ln_FA_over_FG": float(np.log(F[1] / F[0]))}
        zk[s["name"]] = {"rung": r, "zmax": zmax, "log10n0": s["n0"], "N_obs": nobs[r], **rows}
        if s["zmax"] == "5" and s["n0"] == (-3.0, -5.0):
            nt = C.NOTES_TABLE[r]
            v = rows[f"{C.H0_REF:g}"]
            zk[s["name"]]["vs_notes_table"] = {
                "Z_rel_dev": [v["Z_GAL"] / nt["Z"][0] - 1, v["Z_AGN"] / nt["Z"][1] - 1],
                "F_rel_dev": [v["F_GAL"] / nt["F"][0] - 1, v["F_AGN"] / nt["F"][1] - 1],
                "notes": nt}
    out["Z_closed_form"] = zk

    # 4./5. the dry run
    dd = Path(args.dryrun)
    dry = {}
    for s in C.SCANS:
        p = dd / "scans" / f"{s['name']}.h5"
        if not p.exists():
            dry[s["name"]] = None
            continue
        with h5py.File(p, "r") as f:
            Z = np.asarray(f["normaliser/Z"])
            Zc = np.asarray(f["normaliser/Z_closed_form"])
            obs = np.asarray(f["normaliser/observed_total"])
            H = np.asarray(f["normaliser/H0"])
            L = np.asarray(f["lnL"])
            sec = np.asarray(f["seconds"])
            meta = json.loads(f.attrs["meta_json"])
        fine = [[C.z_closed(nobs[s["rung"]][k], s["n0"][k], fits[s["rung"]][k], float(h), meta["zmax_grid"], "fine")
                 for k in (0, 1)] for h in H]
        dry[s["name"]] = {"zmax_grid": meta["zmax_grid"], "grid_nodes": meta["grid_nodes"],
                          "H0_rows": H.tolist(), "Z_core": Z.tolist(), "F_core": (obs / Z).tolist(),
                          "Z_core_over_closed_core_grid_max_abs_rel_dev": float(np.max(np.abs(Z / Zc - 1))),
                          "Z_core_over_closed_fine_grid_max_abs_rel_dev": float(np.max(np.abs(Z / np.array(fine) - 1))),
                          "lnL_cells": L.tolist(), "all_finite": bool(np.all(np.isfinite(L))),
                          "seconds_per_call": sec.tolist(), "n_events": meta["n_events"],
                          "n_injections": meta["n_injections"], "labels": meta["labels"]}
    out["dry_run"] = dry
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({"data_extent_74": ext["74"], "verdict": out["data_extent"]["verdict"],
                      "z_first_C_below_1": {k: v["z_first_C_below_1"] for k, v in curves.items()},
                      "vs_notes": {k: v.get("vs_notes_table", {}).get("F_rel_dev") for k, v in zk.items()},
                      "dry_Z_dev": {k: (v or {}).get("Z_core_over_closed_core_grid_max_abs_rel_dev") for k, v in dry.items()},
                      "dry_finite": {k: (v or {}).get("all_finite") for k, v in dry.items()}}, indent=1))


if __name__ == "__main__":
    main()
