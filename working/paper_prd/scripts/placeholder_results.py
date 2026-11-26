#!/usr/bin/env python
"""Placeholder results of the PRD paper (notes/PAPER_SKELETON.md, Secs. 0, 6, 7): one entry per key and
quantity, with the median, the 68% and 90% equal-tailed intervals, a posterior shape for plotting, and its
source. Writes values/placeholder_results.json. Deterministic (no random draws; everything is a stored
posterior or an analytic shape).

    /hildafs/home/magana/tmp_ondemand_hildafs_phy230014p_symlink/magana/.conda/envs/jax/bin/python \
        working/paper_prd/scripts/placeholder_results.py

Sources
    exploratory  an earlier run of the same analysis under the old settings (the file is named); its
                 posterior is used as it is (grid or equal-weight samples)
    invented     no run exists: the expectation of PAPER_SKELETON.md Sec. 7, a Gaussian (truncated to the
                 prior range; half-normal for an upper limit) with the stated median and 90% half-width
    production   a production summary named in values/production_map.json exists and overrides the entry

Rules of 17 Nov evening (REVIEW_2 1.7 and 1.11)
    prior box    number densities inferred (Free<limit>): the exploratory runs sampled log10 n in Mpc^-3 on a box
                 whose image in h^3 Mpc^-3 reaches above the production prior (Table I). Each posterior sample and
                 each prior draw is converted with its own H0 and the set is reweighted to the production prior:
                 only what has both densities inside the Table I box in h^3 Mpc^-3 is kept (both priors are flat
                 in (H0, log10 n_h) where nonzero, so this is the exact reweighting), and every quantity of the key
                 (its limits in Mpc^-3 and h^3 Mpc^-3, the region allowed by the priors, the joint posterior of
                 Fig. 4a) is computed from what is kept. The bottom of the box (log10 n_AGN below about -5.5,
                 log10 n_gal below about -3.5) holds no sample: the exploratory box did not reach it.
    re-anchoring flux-limited keys (Flux<limit>, densities known) and number densities inferred (Free<limit>),
                 for H0, f_AGN and the rate ratio of realization A, once the complete-catalog key KnownBoth is a
                 production result: median = KnownBoth median + (exploratory median - exploratory complete
                 median), every quantile moved by x -> new median + s (x - exploratory median) with s = KnownBoth
                 90% half-width / exploratory complete 90% half-width, so the 90% half-width is the KnownBoth
                 half-width times the exploratory ratio of this key's half-width to the complete one (exploratory
                 complete: Analysis 2, joint_s100.h5; for Free keys the exploratory values are those after the prior
                 box above). The combination over realizations follows from the re-anchored realization A by the
                 rule below (f_AGN of the Free keys: equal to realization A). The joint (f_AGN, log10 n_AGN) of a
                 Free key moves each sample's f_AGN by the same map as its fAGN. The number densities themselves
                 have no complete counterpart and are not re-anchored. Each quantity records the map ("reanchor").

Realization A ("A") and the combination over realizations ("Comb"). N, the number of realizations the
production combines for each key, is read from analyses/analysis_16_production_runs/runs/RUNLIST.yaml (the
runs of the key's analysis, catalogs, flux limit and event set); the keys the run list does not hold yet take
it from runs.txt (observed-like spectrum: Stage 5, five; the two control sets: run R4, "five for the others").
Where no combination exists on disk the combination is invented from realization A as realization A combined
with N - 1 realizations whose posteriors have A's width and sit at their expectation (input + bias): narrower
by sqrt(N) and centred at
    med_Comb = input + bias + (med_A - input - bias) / N,
where `bias` is the systematic offset the skeleton expects to survive the combination (the single-population
analysis of host-dependent events) and zero otherwise. (Until 17 Nov the shift was divided by sqrt(N), which
kept the input at realization A's credible level and left it outside the combined region wherever it lay
outside A's.) Exploratory combinations use the exploratory realizations that are also production
realizations of the key where those exist (all-AGN and all-galaxy sets, each catalog alone: 100, 101, 102,
103, 105), and all nine exploratory realizations for both catalogs.

Production summaries (contract): values/production_map.json (written by scripts/production.py --write-map from
Analysis 16's run lists) names, for each key whose runs are in analyses/analysis_16_production_runs/runs/
RUNLIST.yaml or RUNLIST_experiment.yaml, its runs: realization A (100), the realizations combined, and per
quantity the summary.json field, the combination and the input of the credible level. When EVERY summary.json
the key needs exists, scripts/production.py builds the entry from them (realization A: the summary's median and
90% interval, the shape from the run's grid or samples; Comb and CL: scripts/run_posteriors.py, ln L of the grids
summed or the product of the kernel densities of nested runs) and marks it final (black). A key
with only part of its runs finished keeps its placeholder entry unchanged (red) and is listed as partial in
"_production_status"; keys without a run in the lists are listed there as well. Keys the map does not name are
built exactly as before. (The old form of the map, key -> a JSON file with the schema of one entry, still works.)
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.modules.setdefault("placeholder_results", sys.modules[__name__])   # one module when run as a script

import h5py
import numpy as np
from scipy.special import ndtr, ndtri

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_values as bv  # noqa: E402

PRD = HERE.parent
OUTJ = PRD / "values" / "placeholder_results.json"
PMAP = PRD / "values" / "production_map.json"
IN = bv.IN
Z90 = 1.6448536269514722

# production runs: analysis_16's run list (one selector per key) and realizations combined
RUNLIST = bv.AN / "analysis_16_production_runs" / "runs" / "RUNLIST.yaml"
KEY_SELECTOR = {   # key -> the run-list arguments of its runs (realization excluded)
    "KnownBoth": dict(analysis="known_population", catalogs="both", flux="complete", events="equal_populations"),
    "KnownGal": dict(analysis="known_population", catalogs="galaxy", flux="complete", events="equal_populations"),
    "KnownAGN": dict(analysis="known_population", catalogs="agn", flux="complete", events="equal_populations"),
    "KnownNone": dict(analysis="known_population", catalogs="none", flux="complete", events="equal_populations"),
    "AGNhosted": dict(analysis="known_population", catalogs="agn", flux="complete", events="agn_hosted"),
    "Galhosted": dict(analysis="known_population", catalogs="galaxy", flux="complete", events="galaxy_hosted"),
    "SingleBoth": dict(analysis="single_population", catalogs="both", flux="complete", events="equal_populations"),
    "SingleNone": dict(analysis="single_population", catalogs="none", flux="complete", events="equal_populations"),
    "HDSingle": dict(analysis="single_population", catalogs="both", flux="complete", events="host_dependent"),
    "HDHost": dict(analysis="host_dependent_populations", catalogs="both", flux="complete", events="host_dependent"),
    "HDNone": dict(analysis="single_population", catalogs="none", flux="complete", events="host_dependent"),
    "ObsSingle": dict(analysis="single_population", catalogs="both", flux="complete", events="host_dependent", suffix="_obslike"),
    "ObsHost": dict(analysis="host_dependent_populations", catalogs="both", flux="complete", events="host_dependent", suffix="_obslike"),
    "ObsNone": dict(analysis="single_population", catalogs="none", flux="complete", events="host_dependent", suffix="_obslike"),
}
for _k, _w, _m in bv.RUNGS:
    KEY_SELECTOR[f"Flux{_w}"] = dict(analysis="known_population", catalogs="both", flux=str(_m), events="equal_populations", density="counts")
    KEY_SELECTOR[f"Free{_w}"] = dict(analysis="free_densities", catalogs="both", flux=str(_m), events="equal_populations", density="free")
# the two control keys are runs of the experiment (RUNLIST_experiment.yaml, Sec. IV B): its f = 0 and f = 0.1 sets of the
# peak-dominated family, both catalogs, f_AGN free (two_tracers_one_population = population known, both catalogs)
RUNLIST_EXP = RUNLIST.parent / "RUNLIST_experiment.yaml"
EXP_SELECTOR = {"CtrlZero": ("two_tracers_one_population", "peak_f0p00_s0"), "CtrlTenth": ("two_tracers_one_population", "peak_f0p10_s0")}
# 23 Nov (Sec. IV E on three steps, notes/MACRO_REQUESTS.md G1): the experiment's runs on the host-dependent sets as keys of their own
# (the same runs that scripts/classes.py reads for Pos, Pops, PosTwice): the population assumed (the galaxy class's mass and spin
# distribution for both classes: two_tracers_one_population), the two populations at their inputs (two_tracers_two_populations) and the
# single population inferred on the set with twice the offsets (stage E3)
EXP_SELECTOR.update({
    "HDAssumed": ("two_tracers_one_population", "peak_f0p30_s1"), "HDAssumedTwice": ("two_tracers_one_population", "peak_f0p30_s2"),
    "HDPops": ("two_tracers_two_populations", "peak_f0p30_s1"), "HDSingleTwice": ("single_population", "peak_f0p30_s2"),
    "ObsAssumed": ("two_tracers_one_population", "obslike_f0p30_s1"), "ObsPops": ("two_tracers_two_populations", "obslike_f0p30_s1")})


# 23 Nov, 23:30 UTC (bv.OBS_MIX): the observed-like keys' runs are those of stage E2M (E2's grids with the mixture redraw)
EXP_STAGE = {k: bv.OBS_MIX["stage"] for k in bv.OBS_MIX["keys"]}


def _experiment_realizations():
    import yaml
    out = {}
    if not RUNLIST_EXP.exists():
        return out
    for st in yaml.safe_load(RUNLIST_EXP.read_text())["stages"]:
        for r in st["runs"]:
            for key, (an, lab) in EXP_SELECTOR.items():
                if key in EXP_STAGE and str(st["stage"]) != EXP_STAGE[key]:
                    continue
                if r["analysis"] == an and r["set"]["label"] == lab:
                    a = r["args"]
                    out.setdefault(key, set()).add(int(a[a.index("--realization") + 1]))
    # 25 Nov (bv.KEY_REALIZATIONS_HELD): HDPops and HDAssumed keep their five while the run list's further realizations wait for their runs
    return {k: bv.held_realizations(k, sorted(v)) for k, v in out.items()}


RUNREAL_EXP = _experiment_realizations()
# keys whose runs are not in the run lists yet: runs.txt, block "REVISION OF 17 NOV" and "APPROVED BY THE OWNER"
NOT_IN_RUNLIST = {
    "CtrlZero": (5, "run R4 of runs.txt (no AGN-hosted mergers), not yet in the run list or the approved stages; 'five for the others' (revision of 17 Nov)"),
    "CtrlTenth": (5, "run R4 of runs.txt (one merger in ten AGN-hosted), not yet in the run list or the approved stages; 'five for the others' (revision of 17 Nov)"),
}


def _runlist_realizations():
    """key -> sorted list of the realizations of its runs in RUNLIST.yaml (only runs without an AGN-density factor; the
    validation (V...) and check (X...) stages skipped, bv.production_stage)."""
    import yaml
    doc = yaml.safe_load(RUNLIST.read_text())
    out = {}
    for st in doc["stages"]:
        if not bv.production_stage(st["stage"]):
            continue
        for r in st["runs"]:
            a = r["args"]
            kv = {a[i].lstrip("-"): a[i + 1] for i in range(len(a) - 1) if a[i].startswith("--") and not a[i + 1].startswith("--")}
            if "agn-density-factor" in kv:
                continue
            sel = dict(analysis=kv["analysis"], catalogs=kv["catalogs"], flux=kv["flux"], events=kv["events-kind"])
            if "density" in kv:
                sel["density"] = kv["density"]
            if kv.get("events-suffix"):                       # Stage 5: the observed-like family (--events-suffix _obslike)
                sel["suffix"] = kv["events-suffix"]
            for key, want in KEY_SELECTOR.items():
                if want == sel:
                    out.setdefault(key, set()).add(int(kv["realization"]))
    # the one deliberate exception (bv.STAGE_X_KEY_RUNS): KnownBoth's realizations 111-120 of stage X (realization_NNN_nside32), the same
    # analysis at the same settings as stage 1; no other stage-X or V run reaches a key
    idx = {r["name"]: (str(st["stage"]), list(r["args"])) for st in doc["stages"] for r in st["runs"]}
    for key in bv.STAGE_X_KEY_RUNS:
        extra = bv.stage_x_key_runs(idx, key)
        if extra:
            out.setdefault(key, set()).update(extra)
    # the second deliberate exception (23 Nov, bv.STAGE_V_KEY_RUNS): HDHost's realizations are those of stage V8's runs
    # realization_NNN_unif_then_rslice (the mixed sampler), which replace the key's production runs; stage 4's pure-rslice run of
    # realization 100, which the selector above also finds, feeds no key
    for key in bv.STAGE_V_KEY_RUNS:
        vexc = bv.stage_v_key_runs(idx, key)
        if vexc:
            out[key] = set(vexc)
    return {k: sorted(v) for k, v in out.items()}


RUNREAL = _runlist_realizations()
STAGE_X_REAL = {k: sorted(bv.stage_x_key_runs(bv.runlist_index(RUNLIST), k)) for k in bv.STAGE_X_KEY_RUNS}
STAGE_V_REAL = {k: sorted(bv.stage_v_key_runs(bv.runlist_index(RUNLIST), k)) for k in bv.STAGE_V_KEY_RUNS}
NREAL = {k: (len(v), f"RUNLIST.yaml: {len(v)} runs {KEY_SELECTOR[k]}, realizations {v}"
             + (f" (of which {STAGE_X_REAL[k]} the stage-X runs realization_NNN_nside32, the same analysis at the same settings)" if STAGE_X_REAL.get(k) else "")
             + (f" (all of them stage {bv.STAGE_V_KEY_RUNS[k]['stage']}'s runs realization_NNN_{bv.STAGE_V_KEY_RUNS[k]['variant']}, the mixed sampler; "
                "build_values.STAGE_V_KEY_RUNS)" if STAGE_V_REAL.get(k) else ""))
         for k, v in RUNREAL.items()}
NREAL.update({k: v for k, v in NOT_IN_RUNLIST.items() if k not in RUNREAL_EXP})
NREAL.update({k: (len(v), f"RUNLIST_experiment.yaml: {len(v)} runs of {EXP_SELECTOR[k][0]} on the experiment's set {EXP_SELECTOR[k][1]}"
                         + (f" in stage {EXP_STAGE[k]} (the mixture redraw)" if k in EXP_STAGE else "") + f", realizations {v}")
              for k, v in RUNREAL_EXP.items()})
assert set(NREAL) == set(KEY_SELECTOR) | set(NOT_IN_RUNLIST) | set(EXP_SELECTOR), sorted((set(KEY_SELECTOR) | set(EXP_SELECTOR)) - set(NREAL))
STAGE_TITLE = {"known": "Stage 1: population known, complete catalogs", "pure": "Stage 1: all-AGN and all-galaxy event sets",
               "ctrl": ("experiment stage E1 (RUNLIST_experiment.yaml): the f = 0 and f = 0.1 sets, both catalogs" if RUNREAL_EXP else
                        "run R4 of runs.txt (not yet in the run list)"), "flux": "Stage 2: flux-limited catalogs, densities from catalog counts",
               "free": "Stage 2: number densities inferred", "single": "Stage 3: single population, equal populations",
               "hdsingle": "Stage 4: host-dependent events, single population",
               "hdhost": ("Stage 4: host-dependent events, host-dependent populations, run as stage V8's realization_NNN_unif_then_rslice (the "
                          "mixed sampler; build_values.STAGE_V_KEY_RUNS)"),
               "hdnone": "Stage 4: host-dependent events, no catalog", "obs": "Stage 5: observed-like spectrum, host-dependent events",
               # 23 Nov: the experiment's runs on the host-dependent sets (RUNLIST_experiment.yaml)
               "hdassumed": "experiment stage E1 (RUNLIST_experiment.yaml), set peak_f0p30_s1: population assumed (the galaxy class's), both catalogs",
               "hdassumedtwice": "experiment stage E1 (RUNLIST_experiment.yaml), set peak_f0p30_s2 (twice the offsets): population assumed, both catalogs",
               "hdpops": "experiment stage E1 (RUNLIST_experiment.yaml), set peak_f0p30_s1: the two populations at their inputs, both catalogs",
               "hdsingletwice": "experiment stage E3 (RUNLIST_experiment.yaml), set peak_f0p30_s2 (twice the offsets): single population, both catalogs",
               # 23 Nov, 23:30 UTC: stage E2M, E2's grids with the mixture redraw (bv.OBS_MIX)
               "obsassumed": "experiment stage E2M (RUNLIST_experiment.yaml; E2's grids with the mixture redraw), set obslike_f0p30_s1: population assumed (the galaxy class's), both catalogs",
               "obspops": "experiment stage E2M (RUNLIST_experiment.yaml; E2's grids with the mixture redraw), set obslike_f0p30_s1: the two populations at their inputs, both catalogs"}
PRIOR = {"Hzero": (50.0, 100.0), "fAGN": (0.0, 1.0), "MuG": (20.0, 50.0), "SigmaG": (1.0, 10.0), "MuChi": (-0.2, 0.3), "SigmaChi": (0.01, 1.0),
         "DmuG": (-10.0, 10.0), "DmuChi": (-0.05, 0.30), "LognGal": (-4.0, -1.0), "LognAGN": (-6.0, -4.0), "RateRatio": (0.0, 400.0)}
UNIT = {"Hzero": "km/s/Mpc", "fAGN": "", "MuG": "Msun", "SigmaG": "Msun", "MuChi": "", "SigmaChi": "", "DmuG": "Msun", "DmuChi": "",
        "LognGal": "log10 Mpc^-3", "LognAGN": "log10 Mpc^-3", "RateRatio": ""}
# number densities in h^3 Mpc^-3 (the unit the production runs sample and the text quotes): how each source gets there
H_NOTE_EXPLORATORY = ("in h^3 Mpc^-3: the exploratory run sampled log10 n in Mpc^-3, so each posterior sample and each prior draw is converted "
                      "with its own H0, log10 n - 3 log10(H0/100); reweighted to the production prior: only the samples and prior draws whose two "
                      "densities lie inside the Table I box in h^3 Mpc^-3 are kept")
H_NOTE_PRODUCTION = "in h^3 Mpc^-3: the run's own samples and prior draws of its parameter, no conversion"
CHAIN_Q = {"H0": "Hzero", "f_agn": "fAGN", "mu_G": "MuG", "sigma_G": "SigmaG", "mu_chi": "MuChi", "sigma_chi": "SigmaChi", "dmu_G": "DmuG", "dmu_chi": "DmuChi"}


def _ratio100():
    c = bv.survey_counts(100)
    return c["gal"] / c["agn"]


RIN = IN["f"] / (1 - IN["f"]) * _ratio100()
INPUT = {"Hzero": IN["H0"], "fAGN": IN["f"], "MuG": IN["muG"], "SigmaG": IN["sigmaG"], "MuChi": IN["muChi"], "SigmaChi": IN["sigmaChi"],
         "DmuG": IN["dmuG"], "DmuChi": IN["dmuChi"], "LognGal": IN["lognGal"], "LognAGN": IN["lognAGN"], "RateRatio": RIN}
# the observed-like family (data/obslike/README.md): only the mass spectrum differs; offsets, spin, f_AGN as above
OBSJ = bv.DATA / "obslike" / "results" / "obslike_paper_values.json"
_obs = bv.jload(OBSJ)["design"]["core_theta"]
_obs_un = bv.jload(OBSJ)["design"]["unchanged_from_present_family"]
assert abs(_obs_un["host_dependent_offsets"]["dmu_G"] - IN["dmuG"]) < 1e-12 and abs(_obs_un["host_dependent_offsets"]["dmu_chi"] - IN["dmuChi"]) < 1e-12
assert abs(_obs["mu_chi"] - IN["muChi"]) < 1e-12 and abs(_obs["sigma_chi"] - IN["sigmaChi"]) < 1e-12 and abs(_obs_un["f_agn"] - IN["f"]) < 1e-12
INPUT_OBS = dict(INPUT, MuG=float(_obs["mu_G"]), SigmaG=float(_obs["sigma_G"]))


# ---------------------------------------------------------------- the experiment's numbers (Sec. IV B; scripts/classes.py)
# Expected values of the macros of scripts/classes.py, used (red) while a run is missing. Copied on 17 Nov from the
# archived letter: its values_independent.tex (the \ph{...} expected values; 'letter macro <name>') and its
# scripts/results_map.json (the invented points of its Fig. 3; 'letter Fig. 3 point'). Where the letter had none:
# 'invented', with the rule. The smallest and largest value over realizations of a placeholder half-width are the
# value times 0.9 and 1.1 (the letter's placeholder bars).
LETTER_FIG3_RULE = ("information model of the letter's notes (1/hw^2 adds over mergers; host class known = (1-f)/hw_gal^2 + f/hw_agn^2; "
                    "host class unknown keeps a fraction of it; host-dependent populations recover 0.40 (5 Msun, 0.10) or 0.67 (twice those) "
                    "of the gap at f = 0.3)")
CLASS_SPREAD = 0.10
CLASS_EXPECTED = {
    "HWUnknownZero": (3.59, "letter Fig. 3 point, " + LETTER_FIG3_RULE), "HWUnknownTen": (2.4, "letter macro IndHWMixTenth"),
    "HWUnknownThirty": (2.0, "letter macro IndHWMix"), "HWUnknownFifty": (1.33, "letter Fig. 3 point, " + LETTER_FIG3_RULE),
    "HWUnknownHundred": (0.84, "letter Fig. 3 point, " + LETTER_FIG3_RULE),
    "HWKnownZero": (3.4, "letter macro IndHWGal"), "HWKnownTen": (2.0, "letter macro IndHWKnownTenth"), "HWKnownThirty": (1.3, "letter macro IndHWKnown"),
    "HWKnownFifty": (1.05, "letter Fig. 3 point, " + LETTER_FIG3_RULE), "HWKnownHundred": (0.8, "letter macro IndHWAGN"),
    "HWNone": (6.0, "letter macro IndHWSpec"),
    "HWKnownAGNThirty": (1.5, "letter macro IndHWKnownAGNPart"),
    "HWKnownAGNTen": (2.6, "invented: the AGN-hosted mergers at f = 0.1 are a third of those at 0.3, so their half-width is sqrt(3) times IndHWKnownAGNPart"),
    "HWPopsTen": (2.21, "letter Fig. 3 point, " + LETTER_FIG3_RULE), "HWPopsThirty": (1.5, "letter macro IndHWMark"),
    "HWPopsFifty": (1.19, "letter Fig. 3 point, " + LETTER_FIG3_RULE),
    "HWPopsTwiceTen": (2.11, "letter Fig. 3 point, " + LETTER_FIG3_RULE), "HWPopsTwiceThirty": (1.4, "letter macro IndHWMarkTwice"),
    "HWPopsTwiceFifty": (1.12, "letter Fig. 3 point, " + LETTER_FIG3_RULE),
    **{f"HW{pre}Known{tag}{w}": ("same_mergers", "invented: HW" + pre + "Known" + w + " (host class known, equal-population mergers, same f) "
                                 "rescaled by the letter's information model, 1/hw^2 = (1 - x)/HW" + pre + "KnownZero^2 + x/HW" + pre +
                                 "KnownHundred^2, from the detected AGN-hosted fraction x of the equal-population events to that of the "
                                 "host-dependent events (events_meta realised_f_agn, mean over the realizations; classes.same_mergers_expected)")
       for pre, tag, w in (("", "Pops", "Ten"), ("", "Pops", "Thirty"), ("", "Pops", "Fifty"), ("", "PopsTwice", "Ten"),
                           ("", "PopsTwice", "Thirty"), ("", "PopsTwice", "Fifty"), ("Obs", "Pops", "Thirty"))},
    "HWHost": (1.6, "mean of the four finished realizations on 23 Nov 2026 (1.45, 1.52, 1.84, 1.62); was 2.1, the letter macro IndTwoPopHW"),
    "HWObsNone": (10.0, "letter macro IndObsHWSpec"),
    "HWObsUnknownZero": (8.21, "letter Fig. 3 point, " + LETTER_FIG3_RULE), "HWObsUnknownThirty": (2.1, "letter macro IndObsHWMix"),
    "HWObsUnknownHundred": (0.84, "letter Fig. 3 point, " + LETTER_FIG3_RULE),
    "HWObsKnownZero": (8.0, "letter Fig. 3 point (invented: the catalogs matter more than for the sharp mass scale)"),
    "HWObsKnownThirty": (1.37, "letter Fig. 3 point, " + LETTER_FIG3_RULE), "HWObsKnownHundred": (0.8, "letter macro IndObsHWAGN"),
    "HWObsPopsThirty": (1.62, "letter Fig. 3 point, " + LETTER_FIG3_RULE),
    "HWObsHost": (None, "the paper's placeholder of key ObsHost: half the 90% width of H0 of realization A (placeholder_results.json keys.ObsHost)"),
    "HWHDSingle": (None, "the paper's placeholder of key HDSingle: half the 90% width of H0 of realization A (placeholder_results.json keys.HDSingle)"),
    "HWObsSingle": (None, "the paper's placeholder of key ObsSingle: half the 90% width of H0 of realization A (placeholder_results.json keys.ObsSingle)"),
    "ProbAGNEqual": (0.33, "invented: positions alone, equal populations, as the letter's IndProbAGNPos less the effect of the heavier AGN-hosted mergers"),
    "ProbGalEqual": (0.25, "invented: as ProbAGNEqual"),
    "ProbAGNPos": (0.36, "letter macro IndProbAGNPos"), "ProbGalPos": (0.27, "letter macro IndProbGalPos"),
    "ProbAGNPops": (0.41, "letter macro IndProbAGNMark"), "ProbGalPops": (0.25, "letter macro IndProbGalMark"),
    "ProbAGNPosTwice": ("ProbAGNPos", "invented: positions alone carry the same information as with the smaller offsets (= ProbAGNPos)"),
    "ProbGalPosTwice": ("ProbGalPos", "invented: = ProbGalPos"),
    "ProbAGNPopsTwice": ("double", "invented: a calibrated classifier (log-likelihood ratios Gaussian, mean +-s^2/2, variance s^2) matched to "
                                   "ProbAGNPops, its separation s doubled with the offsets; the mean over the truly AGN-hosted mergers"),
    "ProbGalPopsTwice": ("double", "invented: as ProbAGNPopsTwice, the mean over the truly galaxy-hosted mergers"),
    # where the probability is evaluated (the posterior median of f_AGN) and its mean over every merger (REVIEW_2 1.1)
    **{f"ProbEvalfAGN{i}": (0.3, "invented: the input f_AGN, at which a posterior median is expected") for i in ("Equal", "Pos", "Pops")},
    "ProbEvalfAGNPosTwice": ("ProbEvalfAGNPos", "invented: as with the smaller offsets (= ProbEvalfAGNPos)"),
    "ProbEvalfAGNPopsTwice": ("ProbEvalfAGNPops", "invented: as with the smaller offsets (= ProbEvalfAGNPops)"),
    **{f"ProbAll{i}": ("all_weighted", f"invented: x ProbAGN{i} + (1 - x) ProbGal{i}, x the AGN-hosted fraction of the 1000 detected mergers of "
                                       "realization A of that event set (events_meta realised_f_agn)")
       for i in ("Equal", "Pos", "Pops", "PosTwice", "PopsTwice")},
    "KnownAGNNoPeak": (4, "the skeleton's Sec. 9.1 (four of five realizations)"),
    "KnownAGNPeakOneHzero": ((76.8, 75.5, 78.1), "the skeleton's Sec. 9.1 (76.8 [75.5, 78.1])"),
    "KnownAGNPeakOneHzeroCL": (0.9999, "the skeleton's Sec. 9.1 (the interval excludes the input)"),
}


# ---------------------------------------------------------------- shapes and summaries
def _cdf(x, p):
    c = np.concatenate([[0.0], np.cumsum(0.5 * (p[1:] + p[:-1]) * np.diff(x))])
    return c / c[-1]


def summarize_xp(x, p, x0=None, n=400):
    """Median, 68%/90% equal-tailed intervals, the level of the input and a compact plotting grid."""
    x = np.asarray(x, float)
    p = np.maximum(np.asarray(p, float), 0.0)
    c = _cdf(x, p)
    q = lambda u: float(np.interp(u, c, x))
    lo, hi = q(1e-6), q(1 - 1e-6)
    pad = 0.15 * (hi - lo)
    lo, hi = max(x[0], lo - pad), min(x[-1], hi + pad)
    xs = np.linspace(lo, hi, n)
    ps = np.interp(xs, x, p)
    ps /= np.trapz(ps, xs)
    out = {"median": q(0.5), "ci68": [q(0.16), q(0.84)], "ci90": [q(0.05), q(0.95)], "ul90": q(0.90),
           "x": [float(f"{v:.7g}") for v in xs], "p": [float(f"{v:.6g}") for v in ps]}
    if x0 is not None:
        F = float(np.interp(x0, x, c)) if x[0] <= x0 <= x[-1] else (0.0 if x0 < x[0] else 1.0)
        out["level_of_input"] = abs(2 * F - 1)
        out["input"] = x0
    return out


def summarize_samples(s, lo, hi, x0=None):
    q = np.percentile(s, [5, 16, 50, 84, 90, 95])
    x, p, bw = bv.kde1(s, lo, hi)
    out = summarize_xp(x, p, x0)
    out.update({"median": float(q[2]), "ci68": [float(q[1]), float(q[3])], "ci90": [float(q[0]), float(q[5])], "ul90": float(q[4]),
                "kde_bandwidth": bw, "n_samples": int(len(s))})
    if x0 is not None:
        F = float(np.mean(np.asarray(s) < x0))
        out["level_of_input"] = abs(2 * F - 1)
    return out


def gauss_xp(med, hw, rng, x0=None, kind="normal"):
    """A Gaussian with median `med` and 90% half-width `hw`, truncated to the prior range `rng`; kind
    'halfnormal': density exp(-x^2/2s^2) on [0, ...) with its 90% upper limit at `hw`."""
    lo, hi = rng
    if kind == "halfnormal":
        s = hw / Z90
        x = np.linspace(lo, min(hi, lo + 8 * s), 4001)
        p = np.exp(-0.5 * ((x - lo) / s) ** 2)
    else:
        s = hw / Z90
        a, b = max(lo, med - 8 * s), min(hi, med + 8 * s)
        x = np.linspace(a, b, 4001)
        p = np.exp(-0.5 * ((x - med) / s) ** 2)
    return summarize_xp(x, p, x0)


def joint_grid(x, y, P, n=110):
    """Downsample a 2-D density to at most n x n points (linear interpolation), normalised."""
    from scipy.interpolate import RegularGridInterpolator
    xs = np.linspace(x[0], x[-1], min(n, len(x)))
    ys = np.linspace(y[0], y[-1], min(n, len(y)))
    Q = RegularGridInterpolator((x, y), P)(np.stack(np.meshgrid(xs, ys, indexing="ij"), -1))
    Q = np.maximum(Q, 0)
    Q /= np.trapz(np.trapz(Q, ys, axis=1), xs)
    return {"x": [float(f"{v:.7g}") for v in xs], "y": [float(f"{v:.7g}") for v in ys], "p": [[float(f"{v:.5g}") for v in row] for row in Q]}


def joint_gauss(qx, qy, sx, sy, rho, n=261):
    """2-D Gaussian density from two 1-D summaries (median, 90% half-width) and a correlation, on a lattice
    reaching 5 sd along each axis (cut at the prior edges): the 90% region (2.15 sd) lies well inside it, and the
    thin tips of strongly correlated regions are resolved."""
    mx, my = qx["median"], qy["median"]
    hx = 0.5 * (qx["ci90"][1] - qx["ci90"][0])
    hy = 0.5 * (qy["ci90"][1] - qy["ci90"][0])
    s1, s2 = hx / Z90, hy / Z90
    xs = np.linspace(max(PRIOR[sx][0], mx - 5 * s1), min(PRIOR[sx][1], mx + 5 * s1), n)
    ys = np.linspace(max(PRIOR[sy][0], my - 5 * s2), min(PRIOR[sy][1], my + 5 * s2), n)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    u, v = (X - mx) / s1, (Y - my) / s2
    P = np.exp(-0.5 * (u * u - 2 * rho * u * v + v * v) / (1 - rho * rho))
    return joint_grid(xs, ys, P, n) | {"params": [sx, sy], "correlation": rho, "shape": "Gaussian"}


def joint_samples(a, b, ra, rb, sx, sy, nb=60, smooth=1.4):
    from scipy.ndimage import gaussian_filter
    H, xe, ye = np.histogram2d(a, b, bins=nb, range=[ra, rb])
    H = gaussian_filter(H.astype(float), smooth, mode="reflect")
    xs, ys = 0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1])
    return joint_grid(xs, ys, H, nb) | {"params": [sx, sy], "correlation": float(np.corrcoef(a, b)[0, 1]),
                                        "shape": f"{nb}x{nb} histogram on the prior box, Gaussian smoothing {smooth} bins"}


FREE_JOINT: dict = {}     # Free<limit> -> (f_AGN samples, log10 n_AGN samples in h^3 Mpc^-3, f box, n_AGN box): realization A, inside the Table I box


def free_joint(key, fmap=None):
    """Joint posterior of f_AGN and log10 n_AGN [h^3 Mpc^-3] of a placeholder Free key (Fig. 4a) from the kept exploratory samples, with
    f_AGN moved by `fmap` (the re-anchoring map of the key's fAGN; None: as sampled)."""
    f, la, fb, ab = FREE_JOINT[key]
    ff = f if fmap is None else fmap(f)
    return joint_samples(ff, la, fb, ab, "fAGN", "LognAGNh", nb=50) | {
        "units_y": "log10 h^3 Mpc^-3", "note": H_NOTE_EXPLORATORY + ("" if fmap is None else "; f_AGN of each sample moved by the re-anchoring map of fAGN")}


# ---------------------------------------------------------------- re-anchoring of the flux-limited placeholders (REVIEW_2 1.11)
REANCHOR_Q = ("Hzero", "fAGN", "RateRatio")
REANCHOR_RULE = ("re-anchored to the finished complete-catalog result of the same realization (key KnownBoth, realization A): median = "
                 "KnownBoth median + (exploratory median - exploratory complete median), every quantile moved by the same map, "
                 "x -> new median + s (x - exploratory median) with s = KnownBoth 90% half-width / exploratory complete 90% half-width, "
                 "so that the 90% half-width is the KnownBoth half-width times the exploratory ratio of this key's half-width to the "
                 "complete one (exploratory complete: Analysis 2, joint_s100.h5)")


def hw90(s):
    return 0.5 * (s["ci90"][1] - s["ci90"][0])


def affine_summary(ex, m_new, s, rng, x0):
    """The summary `ex` (median, intervals, plotting shape) under x -> m_new + s (x - median_ex); the shape is cut at the prior range
    `rng` (the quantiles are mapped exactly; the mass cut is recorded)."""
    T = lambda v: m_new + s * (v - ex["median"])
    x, p = T(np.asarray(ex["x"], float)), np.asarray(ex["p"], float) / s
    ok = (x >= rng[0]) & (x <= rng[1])
    tot = np.trapz(p, x)
    out = summarize_xp(x[ok], p[ok], x0)
    out.update({"median": T(ex["median"]), "ci68": [T(ex["ci68"][0]), T(ex["ci68"][1])], "ci90": [T(ex["ci90"][0]), T(ex["ci90"][1])],
                "ul90": T(ex["ul90"]), "mass_cut_at_prior_edge": float(1 - np.trapz(p[ok], x[ok]) / tot) if ok.sum() > 1 else 1.0})
    for k, v in ex.items():
        if k not in out and k not in ("x", "p", "level_of_input", "input"):
            out[k] = v
    return out


def reanchor(K, expl_complete):
    """Flux<limit> and Free<limit> placeholders (H0, f_AGN, rate ratio; realization A) re-anchored to the finished complete-catalog
    result (REANCHOR_RULE); their combination follows from the new realization A by the rule of the header (or equals it where the
    prior dominates). The identity while KnownBoth is itself a placeholder. Free keys: the joint (f_AGN, log10 n_AGN) of Fig. 4(a) is
    drawn again with f_AGN moved by the map of fAGN."""
    anchor = K["KnownBoth"]
    if not anchor.get("final"):
        return
    for _, word, _ in bv.RUNGS:
        for key in (f"Flux{word}", f"Free{word}"):
            E = K[key]
            if E.get("final"):
                continue
            for q in REANCHOR_Q:
                Q = E["quantities"][q]
                ex, ec, fc = Q["A"], expl_complete[q], anchor["quantities"][q]["A"]
                s = hw90(fc) / hw90(ec)
                m_new = fc["median"] + (ex["median"] - ec["median"])
                A = affine_summary(ex, m_new, s, PRIOR[q], INPUT[q])
                Q["reanchor"] = {"rule": REANCHOR_RULE, "exploratory": {k: ex[k] for k in ("median", "ci68", "ci90", "ul90")},
                                 "exploratory_complete": {k: ec[k] for k in ("median", "ci90")}, "complete": {k: fc[k] for k in ("median", "ci90")},
                                 "complete_key": "KnownBoth", "scale": s, "new_median": m_new}
                Q["A"] = A
                Q["source"] = "exploratory, " + REANCHOR_RULE.split(":")[0]
                if Q["comb_source"].startswith("invented (the prior dominates"):
                    Q["Comb"] = A
                else:
                    Q["Comb"] = invented_comb(A, E["N_comb"], INPUT[q], rng=PRIOR[q])
                    Q["comb_source"] = "invented (from the re-anchored realization A)"
                if q == "fAGN" and key in FREE_JOINT:
                    m0 = ex["median"]
                    E["joint"]["A"] = free_joint(key, lambda f, m0=m0, s=s, m_new=m_new: m_new + s * (f - m0))


def invented_comb(sA, N, x0, bias=0.0, rng=None, kind="normal"):
    """Combination over N realizations from realization A (header rule): A combined with N - 1 realizations of
    A's width centred at their expectation x0 + bias."""
    if x0 is None:
        x0 = sA["median"]
    med = x0 + bias + (sA["median"] - x0 - bias) / N
    hw = 0.5 * (sA["ci90"][1] - sA["ci90"][0]) / math.sqrt(N)
    if kind == "halfnormal":
        return gauss_xp(0.0, sA["ul90"] / math.sqrt(N), rng, x0, "halfnormal")
    return gauss_xp(med, hw, rng, x0)


# ---------------------------------------------------------------- the entries
# 25 Nov: the number of realizations a key's invented combination was made for, where the key's runs have grown since: ObsSingle and ObsNone
# combine ten realizations since 25 Nov (analysis_16 make_runlist.STAGE5_LATER), their red placeholders were invented for five (23 Nov). The
# header rule is applied with this N (field N_placeholder; audit H3 reads it), so the placeholders keep their values; N_comb is the run list's.
PLACEHOLDER_N = {"ObsSingle": 5, "ObsNone": 5}


def entry(events, analysis, catalogs, free, stage, key):
    N, why = NREAL[key]
    E = {"events": events, "analysis": analysis, "catalogs": catalogs, "free": free, "stage": f"{STAGE_TITLE[stage]}, {N} realizations",
         "N_comb": N, "N_comb_source": why, "realizations": RUNREAL.get(key, RUNREAL_EXP.get(key)),
         "final": False, "quantities": {}, "joint": {}, "LB": None}
    if key in PLACEHOLDER_N and PLACEHOLDER_N[key] != N:
        E["N_placeholder"] = PLACEHOLDER_N[key]
        E["N_placeholder_why"] = (f"the invented combination of this red placeholder is the one made for {PLACEHOLDER_N[key]} realizations on 23 Nov; "
                                  f"the key combines {N} since 25 Nov (placeholder_results.PLACEHOLDER_N)")
    return E


def n_invent(E):
    """The N of the header rule for E's invented combinations: N_placeholder where the key's runs grew after its placeholder was made."""
    return E.get("N_placeholder", E["N_comb"])


def q_put(E, q, A, comb, srcA, srcC, fileA=None, fileC=None, x0="default", note=None):
    x0 = INPUT.get(q) if x0 == "default" else x0
    E["quantities"][q] = {"A": A, "Comb": comb, "source": srcA, "comb_source": srcC, "file": fileA, "comb_file": fileC,
                          "input": x0, "units": UNIT[q], "note": note}


def rel(p):
    return bv.rel(p)


def build():
    K = {}
    EXT = {"H0": [IN["H0"]], "f": [IN["f"]]}
    # ======== population known, complete catalogs (Analysis 2; f = 0 and f = 1 columns) ========
    Ls = []
    for s in bv.C1_SEEDS:
        H, F, L, _ = bv.load_hf(bv.A2 / f"joint_s{s}.h5")
        Ls.append(L)
    gA = bv.GridPost(["H0", "f"], [H, F], Ls[0], [4, 10], extra=dict(EXT), tag="ph_joint_s100")
    gC = bv.GridPost(["H0", "f"], [H, F], np.sum(Ls, axis=0), [4, 10], extra=dict(EXT), tag="ph_joint_comb")
    fa = rel(bv.A2 / "joint_s100.h5")
    fc = f"{rel(bv.A2)}/joint_s<seed>.h5, the {len(bv.C1_SEEDS)} realizations {bv.C1_SEEDS} (log-likelihoods summed)"
    E = entry("equal populations", "population known", "both catalogs, complete", "H0, f_AGN", "known", "KnownBoth")
    E["comb_realizations_used"] = list(bv.C1_SEEDS)
    for q, n in (("Hzero", "H0"), ("fAGN", "f")):
        q_put(E, q, summarize_xp(*gA.marg1(n), INPUT[q]), summarize_xp(*gC.marg1(n), INPUT[q]), "exploratory", "exploratory", fa, fc)
    lr = [math.log(bv.survey_counts(s_)["gal"] / bv.survey_counts(s_)["agn"]) for s_ in bv.C1_SEEDS]
    for g, key in ((gA, "A"), (gC, "Comb")):
        ratio = _ratio100() if key == "A" else math.exp(float(np.mean(lr)))   # as RateRatioJointCatA / RateRatioJointComb
        x, p = g.marg1("f")
        m = (x > 1e-9) & (x < 1 - 1e-9)
        r = x[m] / (1 - x[m]) * ratio
        pr = p[m] * (1 - x[m]) ** 2 / ratio          # density in R
        o = np.argsort(r)
        rr = np.linspace(r.min(), min(r.max(), 400), 20001)
        sm = summarize_xp(rr, np.interp(rr, r[o], pr[o]), RIN)
        Rx = bv.rate_ratio_constant(g.summary("f"), ratio)          # exact: quantiles of f mapped one to one
        sm.update({k: Rx[k] for k in ("median", "ci68", "ci90")})
        sm["ul90"] = float(np.interp(0.9, Rx["_cdf"][1], Rx["_cdf"][0]))
        sm["level_of_input"] = bv.level_from_cdf(Rx["_cdf"], RIN)[0]
        E.setdefault("_rr", {})[key] = sm
    q_put(E, "RateRatio", E["_rr"]["A"], E["_rr"]["Comb"], "exploratory", "exploratory", fa, fc, note="R = [f/(1-f)] N_gal/N_AGN (Z_k = N_k)")
    del E["_rr"]
    for key, g in (("A", gA), ("Comb", gC)):
        x1, x2, m = g.marg2("H0", "f")
        sel1 = (x1 > 62) & (x1 < 76)
        sel2 = x2 < 0.65
        E["joint"][key] = joint_grid(x1[sel1], x2[sel2], m[np.ix_(sel1, sel2)], 120) | {"params": ["Hzero", "fAGN"], "correlation": g.corr("H0", "f"), "shape": "stored grid"}
    E["LB"] = {"reference": "galaxy hosts only (f_AGN = 0), Savage-Dickey on the same grid, f_AGN prior [0, 1]", "source": "exploratory", "file": fa,
               "A": None, "Comb": None}
    K["KnownBoth"] = E
    for key, col, cat, desc in (("KnownGal", 0, "galaxy catalog alone", "f_AGN = 0 column"), ("KnownAGN", -1, "AGN catalog alone", "f_AGN = 1 column")):
        E = entry("equal populations", "population known", cat, "H0", "known", key)
        use = [s_ for s_ in bv.C1_SEEDS if s_ in RUNREAL[key]]          # the production realizations among the exploratory ones
        E["comb_realizations_used"] = use
        g1 = bv.GridPost(["H0"], [H], Ls[0][:, col], [8], extra={"H0": [IN["H0"]]}, tag=f"ph_{key}_A")
        g2 = bv.GridPost(["H0"], [H], np.sum([Ls[bv.C1_SEEDS.index(s_)] for s_ in use], axis=0)[:, col], [8], extra={"H0": [IN["H0"]]}, tag=f"ph_{key}_C")
        q_put(E, "Hzero", summarize_xp(*g1.marg1("H0"), IN["H0"]), summarize_xp(*g2.marg1("H0"), IN["H0"]), "exploratory", "exploratory",
              fa + f" ({desc})", f"{rel(bv.A2)}/joint_s<seed>.h5, the {len(use)} realizations {use} (log-likelihoods summed; {desc})",
              note="the posterior rises to the upper end of the prior" if col == -1 else None)
        K[key] = E
    E = entry("equal populations", "population known", "no catalog", "H0", "known", "KnownNone")
    A = gauss_xp(68.6, 3.2, PRIOR["Hzero"], IN["H0"])
    q_put(E, "Hzero", A, invented_comb(A, E["N_comb"], IN["H0"], rng=PRIOR["Hzero"]), "invented", "invented")
    K["KnownNone"] = E
    # ======== all-AGN-hosted and all-galaxy-hosted sets (Analysis 0) ========
    ens = bv.jload(bv.A0 / "h0_pure_tracer_ens93.json")
    seeds = list(ens["seeds_requested"])
    for key, tr, cat in (("AGNhosted", "agn", "AGN catalog"), ("Galhosted", "gal", "galaxy catalog")):
        E = entry("all AGN-hosted" if tr == "agn" else "all galaxy-hosted", "population known", cat, "H0", "pure", key)
        use = [s_ for s_ in seeds if s_ in RUNREAL[key]]               # the production realizations among the 93 exploratory ones
        assert len(use) == E["N_comb"], (key, use)
        E["comb_realizations_used"] = use
        acc = None
        for s in use:
            with h5py.File(bv.A0 / f"h0_pure{tr}_targeted_s{s}.h5") as h:
                Hp, L = h["H0_grid"][()], h["log_likelihood"][()]
            if s == 100:
                g1 = bv.GridPost(["H0"], [Hp], L, [8], extra={"H0": [IN["H0"]]}, tag=f"ph_pure{tr}_A")
            acc = L if acc is None else acc + L
        g2 = bv.GridPost(["H0"], [Hp], acc, [16], extra={"H0": [IN["H0"]]}, tag=f"ph_pure{tr}_C")
        q_put(E, "Hzero", summarize_xp(*g1.marg1("H0"), IN["H0"]), summarize_xp(*g2.marg1("H0"), IN["H0"]), "exploratory", "exploratory",
              rel(bv.A0 / f"h0_pure{tr}_targeted_s100.h5"), f"{rel(bv.A0)}/h0_pure{tr}_targeted_s<seed>.h5, the {len(use)} realizations {use} summed")
        K[key] = E
    # ======== no AGN-hosted mergers; one in ten (invented) ========
    for key, f0, fmed, fhw, lb in (("CtrlZero", 0.0, None, 0.05, -1.0), ("CtrlTenth", 0.10, 0.11, 0.05, 2.5)):
        E = entry(f"mergers with {'no AGN hosts' if f0 == 0 else 'one in ten hosted by AGN'}", "population known", "both catalogs, complete", "H0, f_AGN", "ctrl", key)
        A = gauss_xp(67.9, 1.7, PRIOR["Hzero"], IN["H0"])
        q_put(E, "Hzero", A, invented_comb(A, E["N_comb"], IN["H0"], rng=PRIOR["Hzero"]), "invented", "invented",
              note="H0 width as for population known with both catalogs")
        if f0 == 0:
            A = gauss_xp(0.0, fhw, PRIOR["fAGN"], 0.0, "halfnormal")
            C = invented_comb(A, E["N_comb"], 0.0, rng=PRIOR["fAGN"], kind="halfnormal")
            note = "upper limit: 90% upper limit 0.05 (half-normal)"
        else:
            A = gauss_xp(fmed, fhw, PRIOR["fAGN"], f0)
            C = invented_comb(A, E["N_comb"], f0, rng=PRIOR["fAGN"])
            note = None
        q_put(E, "fAGN", A, C, "invented", "invented", x0=f0, note=note)
        E["LB"] = {"reference": "galaxy hosts only (f_AGN = 0), Savage-Dickey, f_AGN prior [0, 1]", "source": "invented", "file": None,
                   "A": lb, "Comb": -1.5 if lb < 0 else lb * E["N_comb"]}
        K[key] = E
    # ======== flux-limited catalogs, densities known (Analysis 3) ========
    NC = bv.norm_cfg(100)
    ng, na = 10 ** IN["lognGal"], 10 ** IN["lognAGN"]
    for rk, word, mlim in bv.RUNGS:
        E = entry("equal populations", "population known, number densities from the catalog counts", f"both catalogs, m < {mlim}", "H0, f_AGN", "flux", f"Flux{word}")
        p = bv.A3 / f"joint_{rk}_s100.h5"
        g = bv.hf_post(p, f"ph_a3_{rk}")
        for q, n in (("Hzero", "H0"), ("fAGN", "f")):
            A = summarize_xp(*g.marg1(n), INPUT[q])
            q_put(E, q, A, invented_comb(A, E["N_comb"], INPUT[q], rng=PRIOR[q]), "exploratory", "invented", rel(p))
        rg, ra = NC[rk]["gal"], NC[rk]["agn"]
        R = bv.rate_ratio_grid(g, lambda h, rg=rg, ra=ra: bv.Zk(rg, ng, h) / bv.Zk(ra, na, h))
        rr, F = R["_cdf"]
        pr = np.gradient(F, rr)
        A = summarize_xp(rr, pr, RIN)
        A.update({k: R[k] for k in ("median", "ci68", "ci90")})
        q_put(E, "RateRatio", A, invented_comb(A, E["N_comb"], RIN, rng=PRIOR["RateRatio"]), "exploratory", "invented", rel(p),
              note="hosts normalised to z = 5 (stored run); the production convention normalises to z = 1.10")
        K[f"Flux{word}"] = E
    # ======== number densities inferred (Analysis 5) ========
    # the production prior box of the two densities (Table I, h^3 Mpc^-3): the exploratory samples and prior draws are reweighted to it
    PB = bv.prod_common().PRIORS
    box_g, box_a = tuple(map(float, PB["log10n_gal"])), tuple(map(float, PB["log10n_agn"]))
    for rk, word, mlim in bv.RUNGS:
        E = entry("equal populations", "number densities inferred", f"both catalogs, m < {mlim}", "H0, f_AGN, log10 n_gal, log10 n_AGN", "free", f"Free{word}")
        ph5, pj = bv.A5 / f"campaign_{rk}_dynesty_s100.h5", bv.A5 / f"campaign_{rk}_dynesty_s100.json"
        c0, _, _, j = bv.load_a5(ph5, pj)
        pri = j["priors"]
        # the exploratory runs sampled log10 n in Mpc^-3 (their inputs -3 and -5; box [-4, -1] and [-6, -4] in Mpc^-3); the production runs
        # sample it in h^3 Mpc^-3, the unit of the text, on the same numbers as a box in h^3 Mpc^-3 (Table I). Each sample (and each prior
        # draw) is converted with its own H0, log10 n_h = log10 n - 3 log10(H0/100), and reweighted to the production prior: in (H0, log10 n_h)
        # the exploratory prior is flat wherever it is nonzero, as is the production prior, so the reweighting keeps the samples whose two
        # densities lie inside the Table I box and drops the rest (REVIEW_2 1.7). The part of the box below about -5.5 (AGN) and -3.5
        # (galaxies), which the exploratory box did not reach at H0 near its posterior, holds no sample.
        lh0 = 3 * np.log10(c0.col("H0") / 100.0)
        keep = ((c0.col("log10n0") - lh0 >= box_g[0]) & (c0.col("log10n0") - lh0 <= box_g[1])
                & (c0.col("log10n0_c2") - lh0 >= box_a[0]) & (c0.col("log10n0_c2") - lh0 <= box_a[1]))
        c = bv.ChainPost(c0.s[keep], c0.names, tag=c0.tag + "_prodbox")
        cnt = bv.survey_counts(100, rk)
        rg, ra = dict(NC[rk]["gal"], N=cnt["gal"]), dict(NC[rk]["agn"], N=cnt["agn"])
        Rs = c.col("fcat_2") / (1 - c.col("fcat_2")) * bv.Zk(rg, 10 ** c.col("log10n0"), c.col("H0")) / bv.Zk(ra, 10 ** c.col("log10n0_c2"), c.col("H0"))
        cols = {"Hzero": (c.col("H0"), pri["H0"]), "fAGN": (c.col("fcat_2"), pri["fcat_2"]), "LognGal": (c.col("log10n0"), pri["log10n0"]),
                "LognAGN": (c.col("log10n0_c2"), pri["log10n0_c2"]), "RateRatio": (Rs, (0.0, float(np.percentile(Rs, 99.9)) * 1.3))}
        I0 = bv.induced_density(rk)
        lh0I = 3 * np.log10(I0["H0"] / 100.0)
        keepI = ((I0["log10n0"] - lh0I >= box_g[0]) & (I0["log10n0"] - lh0I <= box_g[1])
                 & (I0["log10n0_c2"] - lh0I >= box_a[0]) & (I0["log10n0_c2"] - lh0I <= box_a[1]))
        I = {k: (v[keepI] if isinstance(v, np.ndarray) and v.shape == keepI.shape else v) for k, v in I0.items()}
        E["prior_reweighting"] = {
            "rule": "exploratory samples and prior draws reweighted to the production prior: kept where log10 n_gal and log10 n_AGN, each converted "
                    "with its own H0 to h^3 Mpc^-3, lie inside the Table I box (prod_common.PRIORS log10n_gal, log10n_agn)",
            "box_gal_h": list(box_g), "box_agn_h": list(box_a), "posterior_samples_kept": int(keep.sum()), "posterior_samples": int(len(keep)),
            "prior_weight_kept": float(I0["w"][keepI].sum() / I0["w"].sum()), "prior_draws_kept": int(keepI.sum())}
        lh_c = 3 * np.log10(c.col("H0") / 100.0)
        lh_I = 3 * np.log10(I["H0"] / 100.0)
        for q, (s, rng) in cols.items():
            A = summarize_samples(s, *rng, x0=INPUT[q])
            if q in ("Hzero", "RateRatio"):
                C, cs = invented_comb(A, E["N_comb"], INPUT[q], rng=PRIOR[q]), "invented"
                note = None
            else:
                C, cs = A, "invented (the prior dominates: the combination is taken equal to realization A)"
                note = "follows its prior along the curve of constant rate per host"
            q_put(E, q, A, C, "exploratory", cs, rel(ph5), note=note)
            if q in ("LognGal", "LognAGN"):
                cn = "log10n0" if q == "LognGal" else "log10n0_c2"
                E["quantities"][q]["A"]["prior_ul90"] = float(bv.wquant(I[cn], I["w"], 0.90))
                E["quantities"][q]["A"]["ul90_h"] = float(np.percentile(c.col(cn) - lh_c, 90))
                E["quantities"][q]["A"]["prior_ul90_h"] = float(bv.wquant(I[cn] - lh_I, I["w"], 0.90))
                E["quantities"][q]["A"]["h_note"] = H_NOTE_EXPLORATORY
        # joint posterior of f_AGN and log10 n_AGN in h^3 Mpc^-3 (Fig. 4a), and the region the priors allow once the rate per host is
        # measured (prior-induced density), both on the production prior box (Table I) in h^3 Mpc^-3. The joint posterior is drawn again
        # by reanchor() with f_AGN moved by the re-anchoring map of the key's fAGN (FREE_JOINT keeps the samples)
        la_h, laI_h = c.col("log10n0_c2") - lh_c, I["log10n0_c2"] - lh_I
        FREE_JOINT[f"Free{word}"] = (c.col("fcat_2").copy(), la_h.copy(), tuple(pri["fcat_2"]), box_a)
        E["joint"]["A"] = free_joint(f"Free{word}")
        Hh, xe, ye = np.histogram2d(I["f"], laI_h, bins=50, range=[tuple(pri["fcat_2"]), box_a], weights=I["w"])
        from scipy.ndimage import gaussian_filter
        Hh = gaussian_filter(Hh, 1.0, mode="reflect")
        E["joint"]["prior_induced"] = joint_grid(0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1]), Hh, 50) | {
            "params": ["fAGN", "LognAGNh"], "shape": "2e6 prior draws weighted by the stored known-density likelihood mapped through the identity (build_values.induced_density), "
                                                     "those inside the Table I box in h^3 Mpc^-3 kept",
            "file": rel(bv.A3 / f"joint_{rk}_s100.h5"), "units_y": "log10 h^3 Mpc^-3", "note": H_NOTE_EXPLORATORY}
        K[f"Free{word}"] = E
    # ======== single population, equal populations (invented) ========
    E = entry("equal populations", "single population", "both catalogs, complete", "H0, f_AGN, mass-peak location and width, spin mean and width", "single", "SingleBoth")
    spec = {"Hzero": (68.0, 2.6), "fAGN": (0.29, 0.09), "MuG": (34.9, 0.6), "SigmaG": (5.0, 0.4), "MuChi": (0.000, 0.010), "SigmaChi": (0.100, 0.008)}
    for q, (m, hw) in spec.items():
        A = gauss_xp(m, hw, PRIOR[q], INPUT[q])
        q_put(E, q, A, invented_comb(A, E["N_comb"], INPUT[q], rng=PRIOR[q]), "invented", "invented")
    E["joint"]["A"] = joint_gauss(E["quantities"]["Hzero"]["A"], E["quantities"]["MuG"]["A"], "Hzero", "MuG", -0.5)
    E["joint"]["Comb"] = joint_gauss(E["quantities"]["Hzero"]["Comb"], E["quantities"]["MuG"]["Comb"], "Hzero", "MuG", -0.5)
    K["SingleBoth"] = E
    E = entry("equal populations", "single population", "no catalog", "H0, mass-peak location and width, spin mean and width", "single", "SingleNone")
    spec = {"Hzero": (69.0, 9.0), "MuG": (34.6, 1.6), "SigmaG": (5.0, 0.5), "MuChi": (0.000, 0.012), "SigmaChi": (0.100, 0.010)}
    for q, (m, hw) in spec.items():
        A = gauss_xp(m, hw, PRIOR[q], INPUT[q])
        q_put(E, q, A, invented_comb(A, E["N_comb"], INPUT[q], rng=PRIOR[q]), "invented", "invented",
              note="mass-peak width, spin mean and width not stated in the skeleton: invented at the inputs" if q in ("SigmaG", "MuChi", "SigmaChi") else None)
    E["joint"]["A"] = joint_gauss(E["quantities"]["Hzero"]["A"], E["quantities"]["MuG"]["A"], "Hzero", "MuG", -0.9)
    E["joint"]["Comb"] = joint_gauss(E["quantities"]["Hzero"]["Comb"], E["quantities"]["MuG"]["Comb"], "Hzero", "MuG", -0.9)
    K["SingleNone"] = E
    # ======== host-dependent events, peak-dominated and observed-like spectra ========
    p13 = bv.A13 / "a13core_e7c3007_rslice_cap20_historical_n200_s1.npz"
    c13, j13 = bv.load_chain_npz(p13, bv.A13 / "a13core_e7c3007_rslice_cap20_historical_n200_s1.json")
    box13 = {b[0]: (float(b[1]), float(b[2])) for b in j13["box"]}
    rho13 = c13.corr("dmu_G", "dmu_chi")
    for pre, stage_s, stage_n, label in (("HD", "hdsingle", "hdnone", "peak-dominated"), ("Obs", "obs", "obs", "observed-like")):
        ev = "host-dependent populations" + ("" if pre == "HD" else ", observed-like spectrum")
        INP = INPUT if pre == "HD" else INPUT_OBS
        inv_note = None if pre == "HD" else (f"observed-like inputs (data/obslike): mass peak {INP['MuG']:g} Msun, width {INP['SigmaG']:g} Msun; "
                                             "the invented medians keep their offsets from the inputs of the earlier placeholder")
        # H0, f_AGN and the offsets of the observed-like keys: PAPER_SKELETON Sec. 7 as revised on 17 Nov evening (H0 of ObsSingle 68.4 +- 2.4,
        # bias about 0.7; ObsHost 67.9 +- 2.3; ObsNone 70 +- 13); what Sec. 7 does not state (population parameters, the f_AGN shift and
        # the H0 shift of ObsNone kept in the combination, the correlation of the offsets) is kept from the earlier placeholder
        sec7_note = "PAPER_SKELETON Sec. 7, revised 17 Nov evening"
        qnote = lambda q: (sec7_note if q in ("Hzero", "fAGN", "DmuG", "DmuChi") else inv_note) if pre == "Obs" else None
        # single population with its parameters inferred. Expectation of 23 Nov 2026, from the five finished runs of the peak-dominated
        # family (realizations 100 to 105: H0 67.6, 66.7, 68.9, 70.5, 67.6 with half-widths 1.75 to 1.98; f_AGN 0.34, 0.28, 0.42, 0.33,
        # 0.41, the AGN share of the DETECTED mergers and not the input; mass peak 36.2 to 36.8, spin mean 0.02 to 0.05): no bias in H0
        # beyond the scatter of the realizations, f_AGN above the input. The earlier placeholder (H0 70.9, f_AGN 0.24) was the bias of
        # the exploratory analysis with the population held, which is now the key HDAssumed. Observed-like: the same expectation.
        E = entry(ev, "single population", "both catalogs, complete", "six", stage_s, f"{pre}Single")
        if pre == "HD":
            spec = {"Hzero": (67.6, 1.9, 0.4), "fAGN": (0.34, 0.085, 0.055), "MuG": (36.8, 0.5, 1.6), "SigmaG": (5.0, 0.4, 0.5), "MuChi": (0.036, 0.010, 0.04), "SigmaChi": (0.109, 0.010, 0.015)}
        else:
            spec = {"Hzero": (68.0, 2.4, 0.2), "fAGN": (0.34, 0.09, 0.04), "MuG": (INP["MuG"] + 0.4, 1.2, 0.4), "SigmaG": (INP["SigmaG"] + 0.3, 0.8, 0.3),
                    "MuChi": (0.030, 0.014, 0.03), "SigmaChi": (0.110, 0.012, 0.01)}
        for q, (m, hw, b) in spec.items():
            x0 = INP[q] if q in ("Hzero", "fAGN") else None
            A = gauss_xp(m, hw, PRIOR[q], x0)
            ref = INP[q]
            q_put(E, q, A, invented_comb(A, n_invent(E), ref, bias=b, rng=PRIOR[q]), "invented", "invented (the bias survives the combination)", x0=x0,
                  note=("one population for a mixture: no single input" if x0 is None else None) if inv_note is None else qnote(q))
        K[f"{pre}Single"] = E
        # host-dependent populations
        E = entry(ev, "host-dependent populations", "both catalogs, complete", "eight", "hdhost" if pre == "HD" else "obs", f"{pre}Host")
        if pre == "HD":
            for cn, q in CHAIN_Q.items():
                A = summarize_samples(c13.col(cn), *box13[cn], x0=INPUT[q])
                q_put(E, q, A, invented_comb(A, E["N_comb"], INPUT[q], rng=PRIOR[q]), "exploratory", "invented", rel(p13),
                      note="H0 prior [60, 76] in the exploratory run" if q == "Hzero" else None)
            E["joint"]["A"] = joint_samples(c13.col("dmu_G"), c13.col("dmu_chi"), box13["dmu_G"], box13["dmu_chi"], "DmuG", "DmuChi", nb=50) | {"file": rel(p13)}
        else:
            spec = {"Hzero": (67.9, 2.3), "fAGN": (0.30, 0.08), "MuG": (INP["MuG"], 1.2), "SigmaG": (INP["SigmaG"], 0.8), "MuChi": (0.000, 0.014), "SigmaChi": (0.100, 0.012),
                    "DmuG": (4.5, 3.5), "DmuChi": (0.10, 0.04)}
            for q, (m, hw) in spec.items():
                A = gauss_xp(m, hw, PRIOR[q], INP[q])
                q_put(E, q, A, invented_comb(A, E["N_comb"], INP[q], rng=PRIOR[q]), "invented", "invented", x0=INP[q],
                      note=inv_note if q in ("MuG", "SigmaG") else (sec7_note if q in ("Hzero", "fAGN", "DmuG", "DmuChi") else None))
            E["joint"]["A"] = joint_gauss(E["quantities"]["DmuG"]["A"], E["quantities"]["DmuChi"]["A"], "DmuG", "DmuChi", rho13)
        E["joint"]["Comb"] = joint_gauss(E["quantities"]["DmuG"]["Comb"], E["quantities"]["DmuChi"]["Comb"], "DmuG", "DmuChi", rho13) | {
            "correlation_from": rel(p13)}
        # log10 Bayes factors. Peak-dominated, 23 Nov 2026: realization A 8.4 and 6.8, 3.9, 5.7 on realizations 101 to 103, so about 31
        # for the five combined (the sum); observed-like: invented.
        lb_a, lb_comb = (8.4, 31.0) if pre == "HD" else (3.0, 3.0 * E["N_comb"])
        E["LB"] = {"reference": "the single-population analysis of the same events, same priors (Table I)", "source": "invented", "file": None,
                   "A": lb_a, "Comb": lb_comb}
        K[f"{pre}Host"] = E
        # no catalog
        E = entry(ev, "single population", "no catalog", "five", stage_n, f"{pre}None")
        if pre == "HD":
            spec = {"Hzero": (73.0, 9.0, 5.0), "MuG": (36.4, 1.6, 1.4), "SigmaG": (5.6, 0.5, 0.6), "MuChi": (0.030, 0.012, 0.03), "SigmaChi": (0.110, 0.010, 0.01)}
        else:
            spec = {"Hzero": (70.0, 13.0, 2.0), "MuG": (INP["MuG"] + 0.4, 2.4, 0.4), "SigmaG": (INP["SigmaG"] + 0.3, 1.0, 0.3), "MuChi": (0.030, 0.016, 0.03),
                    "SigmaChi": (0.110, 0.014, 0.01)}
        for q, (m, hw, b) in spec.items():
            x0 = INP[q] if q == "Hzero" else None
            A = gauss_xp(m, hw, PRIOR[q], x0)
            q_put(E, q, A, invented_comb(A, n_invent(E), INP[q], bias=b, rng=PRIOR[q]), "invented", "invented (the bias survives the combination)", x0=x0,
                  note=qnote(q))
        K[f"{pre}None"] = E
    # ======== 23 Nov: the experiment's runs on the host-dependent sets as keys (notes/MACRO_REQUESTS.md G1; invented placeholders) ========
    # (median, 90% half-width) of realization A and the shift of the combination from the input that survives it (header rule)
    ev_hd, ev_hd2, ev_obs = ("host-dependent populations", "host-dependent populations, twice the offsets (10 Msun, 0.20)",
                             "host-dependent populations, observed-like spectrum")
    for key, ev, an, stage, spec in (
            ("HDAssumed", ev_hd, "population assumed (the galaxy class's mass and spin distribution for both classes)", "hdassumed",
             {"Hzero": (70.9, 2.2, 3.2), "fAGN": (0.32, 0.09, 0.05)}),
            ("HDAssumedTwice", ev_hd2, "population assumed (the galaxy class's mass and spin distribution for both classes)", "hdassumedtwice",
             {"Hzero": (99.7, 0.7, 32.0), "fAGN": (0.05, 0.08, -0.25)}),
            ("HDPops", ev_hd, "the two populations at their inputs", "hdpops", {"Hzero": (67.2, 1.45, 0.0), "fAGN": (0.30, 0.05, 0.0)}),
            ("ObsAssumed", ev_obs, "population assumed (the galaxy class's mass and spin distribution for both classes)", "obsassumed",
             {"Hzero": (68.1, 1.7, 0.9), "fAGN": (0.35, 0.07, 0.05)}),
            ("ObsPops", ev_obs, "the two populations at their inputs", "obspops", {"Hzero": (67.0, 1.6, 0.0), "fAGN": (0.33, 0.05, 0.0)})):
        E = entry(ev, an, "both catalogs, complete", "H0, f_AGN", stage, key)
        for q, (m, hw, b) in spec.items():
            A = gauss_xp(m, hw, PRIOR[q], INPUT[q])
            q_put(E, q, A, invented_comb(A, E["N_comb"], INPUT[q], bias=b, rng=PRIOR[q]), "invented", "invented (the shift survives the combination)")
        K[key] = frozen_entry(key, E) if key in bv.OBS_MIX["keys"] else E
    # the single population on the set with twice the offsets (stage E3): MACRO_REQUESTS G1's placeholder values (H0 combined 68.6 +- 0.9
    # at the 80% level, realization A 68.4 +- 1.9, mu_G 38.3 +- 0.6, mu_chi 0.08 +- 0.01, f_AGN combined 0.41 +- 0.04); the widths of the
    # population (not requested) invented
    E = entry(ev_hd2, "single population", "both catalogs, complete", "six", "hdsingletwice", "HDSingleTwice")
    g = "MACRO_REQUESTS.md G1 placeholder"
    for q, (m, hw, mc, hwc) in {"Hzero": (68.4, 1.9, 68.6, 0.9), "fAGN": (0.41, 0.09, 0.41, 0.04), "MuG": (38.3, 0.6, 38.3, 0.6 / math.sqrt(5)),
                                "SigmaG": (6.5, 0.5, 6.5, 0.5 / math.sqrt(5)), "MuChi": (0.08, 0.01, 0.08, 0.01 / math.sqrt(5)),
                                "SigmaChi": (0.12, 0.01, 0.12, 0.01 / math.sqrt(5))}.items():
        x0 = INPUT[q] if q in ("Hzero", "fAGN") else None
        A, C = gauss_xp(m, hw, PRIOR[q], x0), gauss_xp(mc, hwc, PRIOR[q], x0)
        if q == "Hzero":
            C["level_of_input"] = 0.80                     # the requested placeholder level (G1)
        # the requested combinations (H0, f_AGN) are given, not invented from realization A by the header rule (audit H3 checks that rule)
        q_put(E, q, A, C, "invented", g + " (requested values)" if q in ("Hzero", "fAGN") else "invented", x0=x0,
              note=None if x0 is not None else "one population for a mixture: no single input")
        E["quantities"][q]["placeholder_rule"] = g if q in ("Hzero", "fAGN", "MuG", "MuChi") else "invented (not requested)"
    K["HDSingleTwice"] = E
    # Bayes factors of the anchor from the stored grids (exploratory)
    vals = {}
    for g, key in ((gA, "A"), (gC, "Comb")):
        tot = g.log_int()[0]
        col = bv.GridPost(["H0"], [H], (Ls[0] if key == "A" else np.sum(Ls, axis=0))[:, 0], [8], extra={"H0": [IN["H0"]]}, tag=f"ph_lb_{key}").log_int()[0]
        vals[key] = (tot - col) / bv.LN10
    K["KnownBoth"]["LB"].update(vals)
    return K


def frozen_entry(key, E):
    """23 Nov, 23:30 UTC (bv.OBS_MIX): the placeholder of an observed-like key (ObsAssumed, ObsPops), whose runs are stage E2M's (the mixture
    redraw): the entry it had from stage E2's grids on 23 Nov, whole (realization A, the combination, its shapes), from
    values/obslike_e2_frozen.json, marked not final; E (the invented entry) only where the frozen file has none."""
    fz = (bv.obs_frozen().get("keys") or {}).get(key)
    if not fz:
        return E
    import copy
    F = copy.deepcopy(fz)
    prov = {k: F.pop(k) for k in ("production_file", "production_files", "production_run", "production_settings", "comb_realizations_used") if k in F}
    F.update({"final": False, "stage": E["stage"], "N_comb_source": E["N_comb_source"] + f" (the frozen values: {F['N_comb']} realizations)",
              "frozen_from": dict(prov, note=bv.frozen_note(), commit=(bv.obs_frozen().get("_from") or {}).get("commit"))})
    for Q in F["quantities"].values():
        Q["source"] = Q["comb_source"] = bv.frozen_note()
    return F


def production_override(K):
    """Entries from Analysis 16's run summaries wherever every run of a key exists (scripts/production.py)."""
    import production
    pm, status = production.apply(K, INPUT)
    return pm, status


def main():
    import copy
    K = build()
    expl_complete = copy.deepcopy({q: K["KnownBoth"]["quantities"][q]["A"] for q in REANCHOR_Q})   # before the production override
    if not PMAP.exists():
        PMAP.write_text(json.dumps({}, indent=1) + "\n")
    pm, status = production_override(K)
    reanchor(K, expl_complete)
    doc = {"_what": "placeholder results of the PRD paper (notes/PAPER_SKELETON.md Secs. 6 and 7), one entry per key; written by scripts/placeholder_results.py",
           "_conventions": __doc__, "_inputs": {**INPUT, "RateRatio_note": "[0.30/0.70] N_gal/N_AGN of the reference catalog pair"},
           "_inputs_observed_like": {**INPUT_OBS, "file": bv.rel(OBSJ)},
           "_realizations_combined": {k: {"N": v[0], "source": v[1], "realizations": RUNREAL.get(k, RUNREAL_EXP.get(k))} for k, v in NREAL.items()},
           "_production_map": bv.rel(PMAP), "_production_status": status, "keys": bv.jsonable(K),
           "classes_expected": {"_what": "expected values of the macros of scripts/classes.py (red while a run is missing); "
                                         f"placeholder bars of a half-width: the value times {1 - CLASS_SPREAD:g} and {1 + CLASS_SPREAD:g}",
                                "values": {k: {"value": v, "source": src} for k, (v, src) in CLASS_EXPECTED.items()}}}
    OUTJ.write_text(json.dumps(doc, indent=0))
    n = sum(len(E["quantities"]) for E in K.values())
    print(f"wrote {OUTJ.relative_to(PRD)}: {len(K)} keys, {n} key-quantities")
    if status:
        import production
        print("production keys (values/production_map.json):")
        print("\n".join(production.status_lines(status)))
    return doc


if __name__ == "__main__":
    main()
