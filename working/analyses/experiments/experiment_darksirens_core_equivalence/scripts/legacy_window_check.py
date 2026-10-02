#!/usr/bin/env python
"""Isolate legacy's catalog-KDE window inside af896ca (the A8-A13 code), before core.

    source analysis_9_spin_marked_H0_fagn/scripts/env_a9.sh   # darksirens-a8 @ af896ca
    python legacy_window_check.py --window 4096
    python legacy_window_check.py --window full

The Analysis-11 likelihood (a11_likelihood.build_a11: K = 2 complete GAL + AGN catalogues,
field weighting, log10n0 = -24, KDE window W = 4096 at n_sigma 8) at four cells, with the
window as A8-A13 ran it and with full rows (configure_catalog_kde_window(size=None)).
recommended_kde_window gave 3410 on the GAL complete catalogue, so the two should agree
to rounding unless af896ca's pre-fix window sizing (legacy 62eb7d5 came later) drops mass.
Both modes use the same smaller reduction blocks (sel_batch_size 5000, pe_event_block 5)
so full rows fit; reduction blocking only reorders sums.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
A11 = HERE.parents[2] / "analysis_11_free_common_population" / "scripts"
sys.path.insert(0, str(A11))

CELLS = [
    {"name": "fiducial", "H0": 67.74, "fcat_2": 0.30, "mu_G": 35.0, "dmu_G": 5.0, "mu_chi": 0.0, "dmu_chi": 0.10},
    {"name": "11D_H0_62", "H0": 62.0, "fcat_2": 0.266, "mu_G": 35.61, "dmu_G": 3.94, "mu_chi": -0.0035, "dmu_chi": 0.131},
    {"name": "11D_median", "H0": 67.16, "fcat_2": 0.266, "mu_G": 35.61, "dmu_G": 3.94, "mu_chi": -0.0035, "dmu_chi": 0.131},
    {"name": "11D_H0_72", "H0": 72.0, "fcat_2": 0.266, "mu_G": 35.61, "dmu_G": 3.94, "mu_chi": -0.0035, "dmu_chi": 0.131},
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", choices=("4096", "full"), required=True)
    ap.add_argument("--grid", action="store_true", help="the a11_grid_cells posterior grids")
    args = ap.parse_args()
    cells = CELLS
    if args.grid:
        import a11_grid_cells
        cells = a11_grid_cells.cells()
    import a11_likelihood as L
    a8 = L.a8
    if "darksirens-a8" not in os.environ.get("PYTHONPATH", ""):
        sys.exit("[fatal] source env_a9.sh first (darksirens-a8 @ af896ca)")
    a8.SETTINGS["sel_batch_size"] = 5000
    a8.SETTINGS["pe_event_block"] = 5
    from darksirens.redshift.catalog import configure_catalog_kde_window
    if args.window == "full":
        configure_catalog_kde_window(size=None, n_sigma=float(a8.SETTINGS["kde_window_nsigma"]))
    else:
        configure_catalog_kde_window(size=4096, n_sigma=float(a8.SETTINGS["kde_window_nsigma"]))
    a8._KDE_CONFIGURED = True            # build() must not reconfigure it
    import darksirens
    t0 = time.time()
    cell = L.build_a11(f"WINCHK_{args.window}")
    t_build = time.time() - t0
    rows = []
    for c in cells:
        kw = {k: v for k, v in c.items() if k != "name"}
        t1 = time.time()
        r = cell.evaluate_at(**kw)
        rows.append({**c, "logL": r["logL"], "logL_hex": float(r["logL"]).hex(),
                     "logL_pe": r.get("logL_pe"), "logL_selection": r.get("logL_selection"),
                     "seconds": time.time() - t1})
        print(f"[{args.window}] {c['name']:11s} logL {r['logL']:.10f} ({time.time() - t1:.1f}s)", flush=True)
    import jax
    try:
        mem = {k: int(v) for k, v in jax.devices()[0].memory_stats().items()
               if k in ("peak_bytes_in_use", "bytes_limit")}
    except Exception as e:  # noqa: BLE001
        mem = {"error": str(e)}
    out = {"window": args.window, "darksirens_file": darksirens.__file__, "build_seconds": t_build,
           "sel_batch_size": 5000, "pe_event_block": 5, "device_memory": mem,
           "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "rows": rows}
    dst = HERE.parent / "results" / (f"legacy_af896ca_window_{args.window}"
                                     + ("_grid" if args.grid else "") + ".json")
    dst.write_text(json.dumps(out, indent=2))
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
