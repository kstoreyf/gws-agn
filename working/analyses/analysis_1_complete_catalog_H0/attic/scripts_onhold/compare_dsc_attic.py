#!/usr/bin/env python3
"""How far did the analysis of record move when it changed estimator and dtype?

The scans in `results/` use `dark_sirens` at log10n0 = -24 on the float64 dataset.
The scans in `results_dsc_attic/` used `dark_sirens_complete` on the float32
dataset.  Two things changed at once, and they are not the same size:

  * THE ESTIMATOR is not a change at all.  experiment_model_equivalence measured
    the two likelihoods against each other on identical inputs and found them
    bit-for-bit identical at log10n0 = -24 — 201/201 cells, max |delta ln L| = 0,
    in every configuration.  Any difference this script finds is therefore NOT
    the model.
  * THE DATA changed at the 1e-7 level.  Catalog columns are stored float64 now,
    so the event host draw sees unrounded redshifts and positions.  The host
    INDICES are unchanged (bit-identical `host_type`/`host_index`, 720/280 as
    before), but every stored redshift, distance and PE sample moved by ~2e-8
    relative.

So this comparison measures the propagation of a 1e-7 perturbation of the data
through the likelihood, and nothing else.  It also carries the device/blocking
difference between the two jobs, which is why bit equality is not expected.

Writes results/vs_dsc_attic.json.
"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np

# 2026-08-01 reorg: this script now lives in attic/scripts_onhold/, so ROOT is
# still the analysis directory, but the dsc-era archive moved under attic/ and
# the comparison output was archived alongside it.
ROOT = Path(__file__).resolve().parent.parent.parent
NEW = ROOT / "results"
OLD = ROOT / "attic" / "results_dsc_attic"
TAGS = ["h0_gal_targeted", "h0_gal_popuni", "h0_agn_targeted", "h0_agn_popuni",
        "ctrl_gal_matched", "ctrl_agn_matched"]


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tags", nargs="+", default=TAGS)
    ap.add_argument("--out_json", default=str(NEW / "vs_dsc_attic.json"))
    return ap.parse_args(argv)


def grid(path):
    with h5py.File(path, "r") as f:
        return (np.asarray(f["H0_grid"][:], dtype=np.float64),
                np.asarray(f["log_likelihood"][:], dtype=np.float64),
                {k: np.asarray(f[f"guard/{k}"][:]) for k in f["guard"].keys()},
                {k: f.attrs[k] for k in ("arg_universe_model", "arg_log10n0",
                                         "arg_sel_batch_size", "arg_pe_event_block")
                 if k in f.attrs})


def main(argv=None):
    args = parse_args(argv)
    out = {"_what": __doc__.strip().splitlines()[0],
           "new_dir": str(NEW), "old_dir": str(OLD), "configs": {}}
    for tag in args.tags:
        pn, po = NEW / f"{tag}.h5", OLD / f"{tag}.h5"
        if not (pn.exists() and po.exists()):
            print(f"skip {tag}: missing {'new' if not pn.exists() else 'old'} grid")
            continue
        Hn, Ln, Gn, An = grid(pn)
        Ho, Lo, Go, Ao = grid(po)
        if not np.array_equal(Hn, Ho):
            print(f"skip {tag}: grids differ")
            continue
        jn = json.loads((NEW / f"{tag}.json").read_text())
        jo = json.loads((OLD / f"{tag}.json").read_text())
        fin = np.isfinite(Ln) & np.isfinite(Lo)
        # log-likelihoods carry an arbitrary additive constant only if the model
        # changes; here it does not, so the raw difference is meaningful.
        d = np.abs(Ln - Lo)
        # shape-only difference: what the POSTERIOR sees, i.e. after removing the
        # common peak offset
        s = np.abs((Ln - np.nanmax(Ln)) - (Lo - np.nanmax(Lo)))
        rec = {
            "n_cells": int(Hn.size),
            "n_cells_bitwise_identical": int(np.sum(Ln == Lo)),
            "max_abs_dlogL": float(np.nanmax(d[fin])),
            "median_abs_dlogL": float(np.nanmedian(d[fin])),
            "max_abs_dlogL_peaknorm": float(np.nanmax(s[fin])),
            "median_H0_new": jn["H0"]["median"], "median_H0_old": jo["H0"]["median"],
            "delta_median_H0": jn["H0"]["median"] - jo["H0"]["median"],
            "ci68_new": jn["H0"]["ci68"], "ci68_old": jo["H0"]["ci68"],
            "delta_ci68_halfwidth": (0.5 * (jn["H0"]["ci68"][1] - jn["H0"]["ci68"][0])
                                     - 0.5 * (jo["H0"]["ci68"][1]
                                              - jo["H0"]["ci68"][0])),
            "map_new": jn["H0"]["map"], "map_old": jo["H0"]["map"],
            "n_rejected_new": jn["n_rejected"], "n_rejected_old": jo["n_rejected"],
            "Neff_min_new": float(np.min(Gn["Neff"])),
            "Neff_min_old": float(np.min(Go["Neff"])),
            "attrs_new": {k: (v.item() if hasattr(v, "item") else str(v))
                          for k, v in An.items()},
            "attrs_old": {k: (v.item() if hasattr(v, "item") else str(v))
                          for k, v in Ao.items()},
        }
        hw = 0.5 * (jn["H0"]["ci68"][1] - jn["H0"]["ci68"][0])
        rec["delta_median_in_68pct_halfwidths"] = (
            rec["delta_median_H0"] / hw if hw > 0 else None)
        out["configs"][tag] = rec
        print(f"{tag:18s} dH0 = {rec['delta_median_H0']:+.4g}  "
              f"({rec['delta_median_in_68pct_halfwidths']:+.3g} half-widths)   "
              f"max|dlnL| = {rec['max_abs_dlogL']:.3g}   "
              f"bitwise {rec['n_cells_bitwise_identical']}/{rec['n_cells']}")

    Path(args.out_json).write_text(json.dumps(out, indent=1) + "\n")
    print(f"\nwrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
