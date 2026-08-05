#!/usr/bin/env python3
"""Collect the two single-tracer production measurements into one small file.

`working/paper/scripts/build_values.py` reads `results/h0_single_tracer.json` and
nothing else from this analysis, so this script is the only place where a scan
result becomes a paper number.  The schema is fixed by that consumer:

    gal_h0_ci, gal_h0_width, agn_h0_ci, agn_h0_width   (the four macros)
    gal_h0_median, gal_h0_crosscheck_median, agn_grid_top_median,
    agn_railed_at_grid_top, truth, _provenance          (context, not consumed)

The measurement is the TARGETED lane; the popuni lane is carried alongside as a
cross-check, never as the quoted value.

A posterior that piles up against the edge of the scanned H0 range is not a
measurement of H0, so its `*_ci` and `*_width` are written as null rather than as
the number the prior cut produces.  "Railed" is decided from the result file, not
asserted: the MAP sits in the first or last grid cell.
"""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
TRUTH = 67.74


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_json", default=str(RES / "h0_single_tracer.json"))
    return ap.parse_args(argv)


def load(tag):
    return json.loads((RES / f"{tag}.json").read_text())


def railed(doc):
    g = np.asarray(doc["H0"]["grid"], dtype=float)
    return bool(np.isclose(doc["H0"]["map"], g[0]) or np.isclose(doc["H0"]["map"], g[-1]))


def fmt_ci(doc):
    med = doc["H0"]["median"]
    lo, hi = doc["H0"]["ci68"]
    return f"{med:.1f}^{{+{hi - med:.1f}}}_{{-{med - lo:.1f}}}"


def main(argv=None):
    args = parse_args(argv)
    gal, gal_x = load("h0_gal_targeted"), load("h0_gal_popuni")
    agn, agn_x = load("h0_agn_targeted"), load("h0_agn_popuni")

    out = {}
    prov = ["built from h0_{gal,agn}_targeted.json (measurement) + h0_*_popuni.json "
            "(cross-check); estimator dark_sirens at log10n0 = -24, the "
            "complete-catalog limit (bitwise equal to dark_sirens_complete, "
            "experiment_model_equivalence)"]

    for name, doc, xdoc in (("gal", gal, gal_x), ("agn", agn, agn_x)):
        med = doc["H0"]["median"]
        lo, hi = doc["H0"]["ci68"]
        is_railed = railed(doc)
        out[f"{name}_h0_ci"] = None if is_railed else fmt_ci(doc)
        out[f"{name}_h0_width"] = None if is_railed else round(hi - lo, 2)
        if is_railed:
            out[f"{name}_railed_at_grid_top"] = bool(
                np.isclose(doc["H0"]["map"], doc["H0"]["grid"][-1]))
            out[f"{name}_grid_top_median"] = med
            prov.append(f"{name.upper()} entries null: posterior railed at the edge "
                        f"of the analysed range (MAP = {doc['H0']['map']}) -> pending, "
                        "per the todo discipline")
        else:
            out[f"{name}_h0_median"] = med
        out[f"{name}_h0_crosscheck_median"] = xdoc["H0"]["median"]

    out["truth"] = TRUTH
    ordered = {"_provenance": "; ".join(prov)}
    for k in ("gal_h0_ci", "gal_h0_median", "gal_h0_width",
              "gal_h0_crosscheck_median", "gal_railed_at_grid_top",
              "gal_grid_top_median", "agn_h0_ci", "agn_h0_median", "agn_h0_width",
              "agn_h0_crosscheck_median", "agn_railed_at_grid_top",
              "agn_grid_top_median", "truth"):
        if k in out:
            ordered[k] = out[k]

    Path(args.out_json).write_text(json.dumps(ordered, indent=1) + "\n")
    print(json.dumps(ordered, indent=1))
    print(f"\nwrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
