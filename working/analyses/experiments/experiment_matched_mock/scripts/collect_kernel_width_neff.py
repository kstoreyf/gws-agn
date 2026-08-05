#!/usr/bin/env python3
"""Assemble the kernel-width N_eff / PE-variance trade into one results file.

Reads the two guard-instrumented single-point evaluations at truth
(results/guard_kw_dzsurveyconf.json, results/guard_kw_dz3e3.json), both on the
same events (ev_obs_b) and injection campaign (sel_obs), differing only in the
catalog pixelation's redshift kernel widths:

  * dzsurveyconf : gmd SurveyConfig widths, dz = 0.0005 + 0.0015 z
                   (spectroscopic-style; ~0.002 at the median host)
  * dz3e3        : the campaign's adopted constant dz = 3e-3

Writes results/kernel_width_neff.json with, per arm, the selection-integral
N_eff at truth, the summed per-event MC variance, the 5 N_obs floor, and
whether the truth cell is admissible under the hard guard.
"""
import json
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"

ARMS = {
    "surveyconf": {
        "guard_json": "guard_kw_dzsurveyconf.json",
        "survey": "data_derived/deep_survey_z2_ns16_dzsurveyconf.h5",
        "kernel": "dz = 0.0005 + 0.0015 z (SurveyConfig, spectroscopic-style)",
    },
    "dz3e3": {
        "guard_json": "guard_kw_dz3e3.json",
        "survey": "data_derived/deep_survey_z2_ns16.h5",
        "kernel": "dz = 3e-3 (constant, adopted)",
    },
}


def median_kernel_width(survey_path: Path) -> float:
    with h5py.File(survey_path, "r") as f:
        z = f["zgals"][:]
        dz = f["dzgals"][:]
    ok = (z < 90) & (dz < 0.9)          # mask padding sentinels
    return float(np.median(dz[ok]))


def main():
    out = {"description": __doc__.strip().splitlines()[0],
           "coordinate": {"H0": 67.74, "sigma_kde": 0.0},
           "gw_path": "data_derived/obsdet/ev_obs_b.h5",
           "gwselection_path": "data_derived/obsdet/sel_obs.h5",
           "arms": {}}
    for name, meta in ARMS.items():
        g = json.loads((RES / meta["guard_json"]).read_text())
        rec = g["guard_records"][0]
        out["arms"][name] = {
            "survey": meta["survey"],
            "kernel": meta["kernel"],
            "median_kernel_width": median_kernel_width(ROOT / meta["survey"]),
            "Neff_at_truth": rec["Neff"],
            "pe_variance_sum": rec["pe_variance_sum"],
            "legacy_floor_5N": rec["legacy_floor_5N"],
            "passes_5N_floor": rec["passes_legacy_floor"],
            "nEvents": rec["nEvents"],
        }
    a = out["arms"]["surveyconf"]
    b = out["arms"]["dz3e3"]
    out["neff_ratio_dz3e3_over_surveyconf"] = a and b["Neff_at_truth"] / a["Neff_at_truth"]
    out["pe_variance_ratio_surveyconf_over_dz3e3"] = (
        a["pe_variance_sum"] / b["pe_variance_sum"])
    p = RES / "kernel_width_neff.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"Wrote {p}")
    for name, arm in out["arms"].items():
        print(f"  {name}: Neff={arm['Neff_at_truth']:.0f} "
              f"pe_var={arm['pe_variance_sum']:.1f} "
              f"median dz={arm['median_kernel_width']:.4f} "
              f"passes 5N floor: {arm['passes_5N_floor']}")


if __name__ == "__main__":
    main()
