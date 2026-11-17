#!/usr/bin/env python
"""working/data/experiment_sets.json: every event set of the short paper's experiment
(paper_short_independent/EXPERIMENT.md), with the file each realization holds and whether it
exists on disk now.  Re-run after generation to refresh the existence fields.

    python write_experiment_sets.py
"""
import json
from datetime import datetime, timezone
from pathlib import Path

DATA = Path("/hildafs/projects/phy230014p/magana/gws-agn/working/data")
REALS = [100, 101, 102, 103, 105]
# (family, f_agn, s, suffix, sub-seed offset, note)
SETS = [
    ("peak", 0.0, 0, "_puregal_corepop", 8, "all mergers in galaxies"),
    ("peak", 0.1, 0, "_fagn0p10_corepop", 10, ""),
    ("peak", 0.3, 0, "_corepop", 3, "record sub-seed"),
    ("peak", 0.5, 0, "_fagn0p50_corepop", 13, ""),
    ("peak", 1.0, 0, "_pureagn_corepop", 9, "all mergers in AGN"),
    ("peak", 0.1, 1, "_marked_dmu0p10_dmuG5_fagn0p10_corepop", 10, ""),
    ("peak", 0.3, 1, "_marked_dmu0p10_dmuG5_corepop", 3, "record sub-seed"),
    ("peak", 0.5, 1, "_marked_dmu0p10_dmuG5_fagn0p50_corepop", 13, ""),
    ("peak", 0.1, 2, "_marked_dmu0p20_dmuG10_fagn0p10_corepop", 10, ""),
    ("peak", 0.3, 2, "_marked_dmu0p20_dmuG10_corepop", 3, "record sub-seed"),
    ("peak", 0.5, 2, "_marked_dmu0p20_dmuG10_fagn0p50_corepop", 13, ""),
    ("obslike", 0.0, 0, "_puregal_obslike", 14, "all mergers in galaxies"),
    ("obslike", 0.3, 0, "_obslike", 11, ""),
    ("obslike", 1.0, 0, "_pureagn_obslike", 15, "all mergers in AGN"),
    ("obslike", 0.3, 1, "_marked_dmu0p10_dmuG5_obslike", 11, ""),
]
POP = {
    "peak": {"v1": 0.1, "alpha": 2.3, "m_min": 5.0, "m_max": 80.0, "dm_min": 3.0, "dm_max": 10.0,
             "mu_G": 35.0, "sigma_G": 5.0, "beta": 1.0, "mu_chi": 0.0, "sigma_chi": 0.1, "gamma": 0.0},
    "obslike": {"v1": 0.908, "alpha": 3.5, "m_min": 5.0, "m_max": 87.0, "dm_min": 4.9, "dm_max": 0.01,
                "mu_G": 34.0, "sigma_G": 3.6, "beta": 1.1, "mu_chi": 0.0, "sigma_chi": 0.1, "gamma": 0.0},
}
# sets deliberately not generated, and why (none at present)
NOT_GENERATED = {}
# injection support, for the record (results/inj_stress_peaklane_s<R>_x{1,2}.json,
# results/inj_stress_dedicated_s<R>.json); filled by fill_support() from the result files
SUPPORT = {}
INJ = {"peak": "injections/injections_targeted_peak.h5", "obslike": "injections/injections_targeted_obslike.h5"}
INJ_OVERRIDE = {("peak", 1.0, 0): "injections/injections_targeted.h5"}
INJ_NOTE = ("Injections: peak-family analyses use injections/injections_targeted_peak.h5 (dedicated lane covering "
            "AGN offsets 0 to 2x), except the all-AGN set (f 1, s 0), which keeps the record's "
            "injections/injections_targeted.h5 (N^2/N_eff 2.6 vs 3.4); obslike analyses use "
            "injections/injections_targeted_obslike.h5. Each entry's 'injections' field is the file to use, "
            "relative to working/data/seed<R>/.")


def fill_support():
    """N^2/N_eff (N = 1000) of each set's own analysis on its lane, realization 100 (design / worst)."""
    R = Path(__file__).resolve().parent.parent / "results"
    out = {}
    try:
        x1 = json.loads((R / "inj_stress_peaklane_s100_x1.json").read_text())
        x2 = json.loads((R / "inj_stress_peaklane_s100_x2.json").read_text())
        for f in (0.1, 0.3, 0.5):
            ke = "equal_population_mix" if f == 0.3 else f"equal_population_mix_f{f:.1f}"
            kh = "host_dependent_mix" if f == 0.3 else f"host_dependent_mix_f{f:.1f}"
            for s_, src, k in ((0, x1, ke), (1, x1, kh), (2, x2, kh)):
                out[("peak", f, s_)] = (src["points"]["design"][k]["N2_over_neff"], src["worst_N2_over_neff"][k])
        out[("peak", 0.0, 0)] = (x1["points"]["design"]["GAL_class_vol"]["N2_over_neff"], x1["worst_N2_over_neff"]["GAL_class_vol"])
        xo = json.loads((R / "inj_stress_peak_existing_s100_x1.json").read_text())   # the record's lane
        out[("peak", 1.0, 0)] = (xo["points"]["design"]["all_agn_hosted"]["N2_over_neff"], xo["worst_N2_over_neff"]["all_agn_hosted"])
    except FileNotFoundError:
        pass
    try:
        o = json.loads((R / "inj_stress_dedicated_s100.json").read_text())
        for (f, s_, k) in ((0.0, 0, "GAL_class_vol"), (0.3, 0, "equal_population_mix"), (1.0, 0, "all_agn_hosted"),
                           (0.3, 1, "host_dependent_mix")):
            out[("obslike", f, s_)] = (o["points"]["design"][k]["N2_over_neff"], o["worst_N2_over_neff"][k])
    except FileNotFoundError:
        pass
    return out


def main():
    entries = []
    sup = fill_support()
    for fam, f, s, sfx, off, note in SETS:
        fname = f"events{sfx}.h5"
        ex = {str(r): (DATA / f"seed{r}" / "events" / fname).exists() for r in REALS}
        vj = {str(r): (DATA / f"seed{r}" / "validation" / f"validation_events{sfx}.json").exists() for r in REALS}
        entries.append({
            "family": fam, "f_agn": f, "s": s, "dmu_mass": 5.0 * s, "dmu_spin": 0.10 * s,
            "suffix": sfx, "file": f"seed<R>/events/{fname}", "realizations": REALS,
            "sub_seed": f"R*1000+{off}", "population_code": "core",
            "exists": ex, "all_exist": all(ex.values()), "validated": vj,
            "injections": INJ_OVERRIDE.get((fam, f, s), INJ[fam]), "note": note,
            "injections_exist": {str(r): (DATA / f"seed{r}" / INJ_OVERRIDE.get((fam, f, s), INJ[fam])).exists()
                                 for r in REALS},
            "injection_N2_over_Neff_R100_design_worst": (list(sup[(fam, f, s)]) if (fam, f, s) in sup else None),
            "status": ("complete" if all(ex.values()) else NOT_GENERATED.get((fam, s), "in progress"))})
    extra = {}
    for r in REALS:
        sets = [f"mc_accuracy/inj_set{j}/seed{r}/injections/injections_targeted.h5" for j in range(1, 6)]
        extra[str(r)] = {"files": sets, "exist": [(DATA / f"seed{r}" / x).exists() for x in sets],
                         "sub_seeds": [r * 1000 + 900 + j for j in range(1, 6)]}
    out = {
        "injections_note": INJ_NOTE,
        "extra_record_lane_sets_note": (
            "For sets in which every merger is AGN-hosted, pool six injection sets: the record's "
            "injections/injections_targeted.h5 and the five independent redraws of the same lane listed in "
            "'extra_record_lane_sets' (Analysis 14's recipe, scripts/make_injection_redraws.py: 1.5e8 "
            "proposals each, targeted sub-seed R*1000+900+j). Paths are relative to working/data/seed<R>/."),
        "extra_record_lane_sets": extra,
        "what": "event sets of the short paper's experiment (working/paper_short_independent/EXPERIMENT.md): "
                "family, planted f_AGN, size s of the AGN-class offset (mass peak +5 s Msun, mean chi_eff "
                "+0.10 s), and the file. 'suffix' is everything between 'events' and '.h5'; files live under "
                "working/data/seed<R>/events/; 'injections' is relative to working/data/seed<R>/. 'exists' is "
                "per realization at 'updated_utc'. 'injection_N2_over_Neff_R100_design_worst' is the selection-"
                "term variance of the set's own analysis on its lane (realization 100; design point, worst "
                "stressed point; obslike/README.md).",
        "updated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "populations_galaxy_class": POP,
        "agn_class": "the galaxy-class vector with mu_G + dmu_mass and mu_chi + dmu_spin",
        "docs": {"peak": "working/data/corepop/README.md", "obslike": "working/data/obslike/README.md"},
        "sets": entries}
    p = DATA / "experiment_sets.json"
    p.write_text(json.dumps(out, indent=1))
    print("wrote", p, sum(e["all_exist"] for e in entries), "of", len(entries), "sets complete")


if __name__ == "__main__":
    main()
