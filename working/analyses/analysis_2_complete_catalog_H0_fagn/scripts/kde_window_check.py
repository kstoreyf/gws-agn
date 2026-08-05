#!/usr/bin/env python3
"""Size the windowed catalog-KDE evaluator for this analysis's survey files.

darksirens' windowed catalog-KDE evaluator keeps only the W nearest galaxies in
redshift around each evaluation point.  If W is smaller than the number of
galaxies actually inside the kernel support, the kernel is truncated and the
catalog prior is wrong.  ``darksirens.redshift.catalog.recommended_kde_window``
returns, over every pixel row and every interval position, the largest number of
galaxies any interval of width ``2 n_sigma max_row(sigma_eff)`` contains — i.e.
the smallest W that cannot truncate.

This is a host-side numpy diagnostic, run once per dataset when sizing W.  It is
re-run whenever the survey files are rebuilt; the float64 regeneration of
2026-07-31 changed the stored redshifts at the 1e-7 level and could in principle
move a galaxy across a pixel boundary, so the number is measured, not carried
over.

The scans use ``sigma_kde = 0`` (fixed), so the relevant kernel width is the
survey's own ``dzgals``; ``sigma_kde_max = 0.05`` is reported alongside as the
value that would be needed if ``sigma_kde`` were ever sampled under darksirens'
default prior.

Usage:
    python scripts/kde_window_check.py                       # both surveys, seed100
    python scripts/kde_window_check.py --survey_path A.h5 B.h5 --out_json out.json
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = Path("/hildafs/projects/phy230014p/magana/gws-agn/working/data/seed100")
DARKSIRENS = Path(os.environ.get(
    "DARKSIRENS_SRC", "/hildafs/projects/phy230014p/magana/src/darksirens"))


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--survey_path", nargs="+", default=[
        str(DATA / "surveys" / "survey_gal_complete_ns32.h5"),
        str(DATA / "surveys" / "survey_agn_complete_ns32.h5")])
    ap.add_argument("--n_sigma", type=float, nargs="+", default=[8.0])
    ap.add_argument("--sigma_kde_max", type=float, nargs="+", default=[0.0, 0.05])
    ap.add_argument("--out_json", default=str(ROOT / "results" / "kde_window.json"))
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    sys.path.insert(0, str(DARKSIRENS))
    from darksirens.redshift.catalog import recommended_kde_window

    out = {"darksirens": str(DARKSIRENS), "surveys": {}}
    for path in args.survey_path:
        p = Path(path)
        with h5py.File(p, "r") as f:
            Z = np.asarray(f["zgals"][:])
            DZ = np.asarray(f["dzgals"][:])
            NG = np.asarray(f["ngals"][:]).astype(np.int64)
            attrs = {k: (v.item() if hasattr(v, "item") else v)
                     for k, v in f.attrs.items() if k in
                     ("nside", "tracer", "level", "n_hosts", "z_min", "z_max")}
        rec = {"path": str(p), "block_shape": [int(Z.shape[0]), int(Z.shape[1])],
               "storage_dtype": str(Z.dtype), "max_gals_per_pixel": int(NG.max()),
               "occupied_pixels": int((NG > 0).sum()), "attrs": attrs,
               "recommended": {}}
        for nsig in args.n_sigma:
            for skm in args.sigma_kde_max:
                t0 = time.time()
                w = int(recommended_kde_window(Z, NG, DZ, skm, n_sigma=nsig))
                rec["recommended"][f"n_sigma={nsig:g},sigma_kde_max={skm:g}"] = w
                print(f"{p.name}: recommended_kde_window(n_sigma={nsig:g}, "
                      f"sigma_kde_max={skm:g}) = {w}   ({time.time() - t0:.1f} s)")
        out["surveys"][p.name] = rec

    # the window the scans should use: the smallest power of two that clears the
    # requirement at the scans' own kernel (sigma_kde = 0) and n_sigma = 8
    key = "n_sigma=8,sigma_kde_max=0"
    need = max(r["recommended"][key] for r in out["surveys"].values()
               if key in r["recommended"])
    w = 1
    while w < need:
        w *= 2
    out["window_required"] = int(need)
    out["window_recommended_power_of_two"] = int(w)
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, indent=1))
    print(f"\nrequirement over both surveys: W >= {need};  use W = {w}")
    print(f"wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
