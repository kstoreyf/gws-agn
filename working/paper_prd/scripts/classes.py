#!/usr/bin/env python
"""The experiment's numbers in the long paper (notes/PAPER_SKELETON.md Sec. 9): which mergers carry the H0
constraint and what the unknown host class costs (Sec. IV B, `sec:classes`), with the observed-like spectrum and
the probability of an AGN host. Analysis 16's experiment (RUNLIST_experiment.yaml, stages E1-E3) and two of its
production keys (HDHost, ObsHost for the open symbols of fig_classes; KnownAGN for the AGN catalog alone).

    PY=/hildafs/home/magana/tmp_ondemand_hildafs_phy230014p_symlink/magana/.conda/envs/jax/bin/python
    $PY working/paper_prd/scripts/classes.py          # every macro: real or red, value, what it waits for

Used by build_values.py (the macros), production.py (--write-map writes the runs into values/production_map.json
under "classes"; --status/--report count them), make_figures.py (fig_classes) and audit_values.py (K checks).

Macros (all half-widths are of H0 in km/s/Mpc; "realizations" = the experiment's five, \\NrealClasses):
  HW<spectrum><analysis><f>        mean over the realizations of half the 90% equal-tailed width of the H0 posterior,
  HW...Min, HW...Max               and its smallest and largest value over the same realizations (all printed to two
                                   significant figures, f2sig; derived macros use the unrounded values)
      spectrum   (none) peak-dominated, Obs observed-like
      analysis   Unknown  host class unknown: both catalogs, f_AGN inferred, population known (two_tracers_one_population)
                 Known    host class known: each merger with the catalog and population of its true host class, the two
                          H0 posteriors multiplied (hosts_known); at f = 0 that is the galaxy catalog alone
                          (galaxy_catalog_alone), at f = 1 the AGN catalog alone on all-AGN-hosted mergers
                 KnownAGN the AGN-hosted mergers alone of the host-class-known analysis (its AGN factor)
                 Pops     host-dependent populations held at their inputs (5 Msun, 0.10), host class unknown
                          (two_tracers_two_populations); PopsTwice the same with twice those offsets
                 KnownPops, KnownPopsTwice   host class known on those host-dependent mergers themselves (hosts_known on the
                          s1 and s2 sets; each class with its own population at its input); not drawn in fig_classes
                 None     no catalog: the mass spectrum alone (mass_spectrum_alone, f = 0 set)
                 Host     host-dependent populations inferred with H0 (key HDHost; Obs: key ObsHost), no f
      f          Zero, Ten, Thirty, Fifty, Hundred = input f_AGN 0, 0.1, 0.3, 0.5, 1
  CostUnknown<f>, CostPops<f>, CostPopsTwice<f>   (HW<analysis><f> / denominator)^2 from the unrounded means: the number
                 of mergers needed with the host class unknown per merger with it known, on the same mergers: the denominator
                 of CostUnknown<f> is HWKnown<f> (equal-population mergers); of CostPops<f> HWKnownPops<f> and of
                 CostPopsTwice<f> HWKnownPopsTwice<f> (the host-dependent mergers themselves), HWKnown<f> only where the run
                 lists hold no such runs (COST_DENOMINATOR); the central value is the ratio of the means.
  Cost...Min, Cost...Max   the smallest and largest over the realizations of the same ratio computed per realization
                 (cost_range; while red: the central value times the relative spread of CostUnknown<f>)
  ShareAGN<f>    100 (HWKnown<f> / HWKnownAGN<f>)^2 from the unrounded means: percent of the inverse variance of H0 that
                 the AGN-hosted mergers carry with the host class known (the two posteriors multiply; exact for
                 Gaussian posteriors, per realization 1/hw^2 = 1/hw_AGN^2 + 1/hw_gal^2)
  Prob<AGN|Gal><info>   realization A, f = 0.3: mean over the mergers truly hosted by AGN (galaxies) of the probability
                 of an AGN host f L_AGN / (f L_AGN + (1 - f) L_gal) at the posterior median (membership.npz of the grid);
                 info: Equal (equal populations, positions alone), Pos (host-dependent populations, positions alone),
                 Pops (host-dependent populations, positions with masses and spins), PosTwice, PopsTwice (twice the offsets)
  ProbEvalfAGN<info>   the f_AGN at which that probability is evaluated: the run's posterior median (membership.f_agn_median)
  ProbAll<info>  the mean of that probability over all the mergers (membership.mean_r)
  KnownAGNNoPeak, KnownAGNPeak<n>Hzero[Med|Lo|Hi|CL]  the AGN catalog alone on equal-population mergers with f = 0.3
                 (key KnownAGN): the number of realizations whose H0 posterior has no interior maximum in the prior
                 range, and for each one that has (n = One, Two, ...) its median and 90% interval (summary.json) and
                 the credible level of the input
  HWHDSingle, HWObsSingle   (23 Nov) as HW...: the single population inferred (keys HDSingle, ObsSingle) on the realizations of
                 HWHost / HWObsHost (map record field realizations_of); not drawn in fig_classes
  CostSingleHost, CostObsSingleHost   (23 Nov) (HW<pre>Single / HW[Obs]Host)^2 from the unrounded means, Min and Max per realization,
                 as CostPopsThirty: the single population's factor in mergers against the host-dependent populations, both inferred
  NrealClasses   the number of realizations of every experiment set above (printed as a word)
  ClsDiff<f>HzeroMean, ClsDiff<f>HzeroSd   (21 Nov; f = Thirty, Fifty) mean and standard deviation (ddof 1) over the realizations of
                 (H0 median of the host-class-unknown run minus that of the host-class-known run of the same realization; summary.json;
                 the runs of HWUnknown<f> and HWKnown<f>, the same mergers), signed, two decimals; red: invented 0 and 0.3. (23 Nov: the
                 combinations Cls<Unknown|Known><f>Hzero<Comb|CL> of the same runs are removed; main.tex used none.)
  ClsDiff<f>Npos the number of realizations in which that difference is positive (a word); red: half the realizations

A macro is real (black) when every summary.json it names exists; otherwise it is red and prints the expected value
of placeholder_results.CLASS_EXPECTED (the archived letter's expected values; 'invented' where it had none).
Single-run numbers come from summary.json (never from grid/result.json); the AGN factor of a host-class-known grid,
which no summary holds, from run_posteriors.factor_summary (the summaries' own method on lnL_agn).
"""
from __future__ import annotations

import json
import math
import re
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import build_values as bv  # noqa: E402

PRD = HERE.parent
PMAP = PRD / "values" / "production_map.json"
A16 = bv.AN / "analysis_16_production_runs"
RUNS = A16 / "runs"
RUNLIST = RUNS / "RUNLIST.yaml"
RUNLIST_EXP = RUNS / "RUNLIST_experiment.yaml"
REAL_A = 100
F_WORD = {0.0: "Zero", 0.1: "Ten", 0.3: "Thirty", 0.5: "Fifty", 1.0: "Hundred"}
F_LAB = {0.0: "f0p00", 0.1: "f0p10", 0.3: "f0p30", 0.5: "f0p50", 1.0: "f1p00"}
ORD = {1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five"}
SPEC_W = {"peak": "peak-dominated spectrum", "obslike": "observed-like spectrum"}
# the cells (spectrum, offset size s) -> fractions of the host-dependent series at which RUNLIST_experiment.yaml also runs the host class
# known on the same mergers (hosts_known on the s1 and s2 sets): HW[Obs]KnownPops<f>, HWKnownPopsTwice<f>
SAME_MERGERS = {("peak", 1): (0.1, 0.3, 0.5), ("peak", 2): (0.1, 0.3, 0.5), ("obslike", 1): (0.3,)}
# the denominator of each cost ratio: the host class known on the same mergers as the numerator
COST_DENOMINATOR = {(tag, f): (f"HWKnown{tag}{F_WORD[f]}" if tag != "Unknown" and f in SAME_MERGERS.get(("peak", 1 if tag == "Pops" else 2), ()) else
                               f"HWKnown{F_WORD[f]}")
                    for tag in ("Unknown", "Pops", "PopsTwice") for f in (0.1, 0.3, 0.5)}


# ============================================================================================= #
# the table: macro -> what it is computed from
# ============================================================================================= #
def _table():
    """Ordered {name: entry}. entry: kind (mean_hw | factor_hw | derived | membership | interior), select (('exp',
    analysis, set label) or ('prod', run-name prefix) or ('key', production key)), and for figures: spectrum,
    series, f."""
    T = {}

    def hw(name, sel, spec, series, f, what):
        T[name] = {"kind": "mean_hw", "select": sel, "spectrum": spec, "series": series, "f": f, "what": what}

    for spec, pre in (("peak", ""), ("obslike", "Obs")):
        fr = (0.0, 0.1, 0.3, 0.5, 1.0) if spec == "peak" else (0.0, 0.3, 1.0)
        for f in fr:
            hw(f"HW{pre}Unknown{F_WORD[f]}", ("exp", "two_tracers_one_population", f"{spec}_{F_LAB[f]}_s0"), spec, "unknown", f,
               f"host class unknown (both catalogs, f_AGN inferred, population known), equal populations, input f_AGN = {f:g}, {SPEC_W[spec]}")
        for f in fr:
            if f == 0.0:
                sel, how = ("exp", "galaxy_catalog_alone", f"{spec}_f0p00_s0"), "the galaxy catalog alone on mergers with no AGN host"
            else:
                sel, how = ("exp", "hosts_known", f"{spec}_{F_LAB[f]}_s0"), ("each merger with the catalog and population of its true host class, "
                                                                            "the two H0 posteriors multiplied" if f < 1 else
                                                                            "the AGN catalog alone on mergers all hosted by AGN")
            hw(f"HW{pre}Known{F_WORD[f]}", sel, spec, "known", f, f"host class known ({how}), equal populations, input f_AGN = {f:g}, {SPEC_W[spec]}")
        hw(f"HW{pre}None", ("exp", "mass_spectrum_alone", f"{spec}_f0p00_s0"), spec, "none", None,
           f"no catalog (the mass spectrum alone, population known; the experiment's f = 0 set), {SPEC_W[spec]}")
        for f in ((0.1, 0.3, 0.5) if spec == "peak" else (0.3,)):
            hw(f"HW{pre}Pops{F_WORD[f]}", ("exp", "two_tracers_two_populations", f"{spec}_{F_LAB[f]}_s1"), spec, "pops", f,
               f"host-dependent populations held at their inputs (AGN offsets 5 Msun and 0.10), host class unknown, both catalogs, f_AGN inferred, "
               f"host-dependent mergers, input f_AGN = {f:g}, {SPEC_W[spec]}")
        if spec == "peak":
            for f in (0.1, 0.3, 0.5):
                hw(f"HWPopsTwice{F_WORD[f]}", ("exp", "two_tracers_two_populations", f"peak_{F_LAB[f]}_s2"), spec, "pops2", f,
                   f"host-dependent populations held at their inputs, twice the offsets (10 Msun, 0.20), host class unknown, both catalogs, "
                   f"f_AGN inferred, input f_AGN = {f:g}, {SPEC_W[spec]}")
        # host class known on the host-dependent mergers themselves (the same mergers as HW...Pops...; not drawn in fig_classes):
        # the denominators of CostPops<f> and CostPopsTwice<f>
        for s, tag, off in ((1, "Pops", "AGN offsets 5 Msun and 0.10"), (2, "PopsTwice", "twice the offsets, 10 Msun and 0.20")):
            for f in SAME_MERGERS.get((spec, s), ()):
                hw(f"HW{pre}Known{tag}{F_WORD[f]}", ("exp", "hosts_known", f"{spec}_{F_LAB[f]}_s{s}"), spec, "known_" + tag.lower(), f,
                   f"host class known (each merger with the catalog and population of its true host class, each population at its input, "
                   f"the two H0 posteriors multiplied), host-dependent mergers ({off}; the same mergers as HW{pre}{tag}{F_WORD[f]}), "
                   f"input f_AGN = {f:g}, {SPEC_W[spec]}")
                T[f"HW{pre}Known{tag}{F_WORD[f]}"]["fig"] = False
        hw(f"HW{pre}Host", ("key", "HDHost" if spec == "peak" else "ObsHost"), spec, "host", 0.3,
           f"host-dependent populations inferred with H0 (eight parameters; key {'HDHost' if spec == 'peak' else 'ObsHost'}), both catalogs, "
           f"host-dependent mergers, input f_AGN = 0.3, {SPEC_W[spec]}")
    # 23 Nov (notes/MACRO_REQUESTS.md G2): the single population inferred on the same mergers and realizations as the host-dependent
    # populations inferred (HWHost, HWObsHost): its runs are those of key HDSingle / ObsSingle restricted to the realizations of that
    # macro ("realizations_of"); not drawn in fig_classes
    for spec, pre, kS, kH, host in (("peak", "HD", "HDSingle", "HDHost", "HWHost"), ("obslike", "Obs", "ObsSingle", "ObsHost", "HWObsHost")):
        T[f"HW{pre}Single"] = {"kind": "mean_hw", "select": ("key", kS), "realizations_of": host, "spectrum": spec, "series": "single", "f": 0.3,
                               "fig": False,
                               "what": f"single population inferred with H0 (six parameters; key {kS}), both catalogs, host-dependent mergers, input "
                                       f"f_AGN = 0.3, {SPEC_W[spec]}, on the realizations of {host}"}
    for f in (0.1, 0.3):
        T[f"HWKnownAGN{F_WORD[f]}"] = {"kind": "factor_hw", "select": ("exp", "hosts_known", f"peak_{F_LAB[f]}_s0"), "factor": "agn",
                                       "spectrum": "peak", "series": None, "f": f,
                                       "what": f"the AGN-hosted mergers alone (the AGN factor of the host-class-known analysis), input f_AGN = {f:g}, "
                                               "peak-dominated spectrum"}
    for f in (0.1, 0.3):
        T[f"ShareAGN{F_WORD[f]}"] = {"kind": "derived", "deps": [f"HWKnown{F_WORD[f]}", f"HWKnownAGN{F_WORD[f]}"],
                                     "fn": "share", "f": f,
                                     "what": f"percent of the information on H0 (inverse variance) carried by the AGN-hosted mergers with the host class "
                                             f"known, input f_AGN = {f:g}: 100 (HWKnown{F_WORD[f]} / HWKnownAGN{F_WORD[f]})^2 from the unrounded means"}
    for tag, nm in (("Unknown", "host class unknown"), ("Pops", "host-dependent populations held at their inputs (5 Msun, 0.10)"),
                    ("PopsTwice", "host-dependent populations held at their inputs, twice the offsets")):
        for f in (0.1, 0.3, 0.5):
            den = COST_DENOMINATOR[(tag, f)]
            which = ("the host-class-known analysis of the same host-dependent mergers" if den != f"HWKnown{F_WORD[f]}" else
                     "the host-class-known analysis of the equal-population mergers" + (", the same mergers" if tag == "Unknown" else
                                                                                         " (no host-class-known run on the host-dependent mergers)"))
            T[f"Cost{tag}{F_WORD[f]}"] = {"kind": "derived", "deps": [f"HW{tag}{F_WORD[f]}", den], "fn": "cost", "f": f,
                                         "what": f"mergers needed with the {nm} per merger with the host class known, input f_AGN = {f:g}: "
                                                 f"(HW{tag}{F_WORD[f]} / {den})^2 from the unrounded means (denominator: {which})"}
    # 23 Nov (G2): the factor in the number of mergers between the single population and the host-dependent populations, both inferred,
    # on the same mergers and realizations, defined as CostPopsThirty is (the ratio of the means squared; Min, Max per realization)
    for pre, host in (("HD", "HWHost"), ("Obs", "HWObsHost")):
        nm = "CostSingleHost" if pre == "HD" else "CostObsSingleHost"
        T[nm] = {"kind": "derived", "deps": [f"HW{pre}Single", host], "fn": "cost", "f": 0.3,
                 "what": f"mergers needed with a single population inferred per merger with host-dependent populations inferred, for the same 90% "
                         f"half-width of H0, {'peak-dominated' if pre == 'HD' else 'observed-like'} spectrum, the same mergers and realizations: "
                         f"(HW{pre}Single / {host})^2 from the unrounded means"}
    for info, sel, what in (("Equal", ("exp", "two_tracers_one_population", "peak_f0p30_s0"), "equal populations, positions alone"),
                            ("Pos", ("exp", "two_tracers_one_population", "peak_f0p30_s1"), "host-dependent populations (5 Msun, 0.10), positions alone "
                                                                                             "(one population for both classes, held at its input)"),
                            ("Pops", ("exp", "two_tracers_two_populations", "peak_f0p30_s1"), "host-dependent populations (5 Msun, 0.10), positions "
                                                                                                "with masses and spins (the two populations at their inputs)"),
                            ("PosTwice", ("exp", "two_tracers_one_population", "peak_f0p30_s2"), "host-dependent populations with twice the offsets "
                                                                                                   "(10 Msun, 0.20), positions alone"),
                            ("PopsTwice", ("exp", "two_tracers_two_populations", "peak_f0p30_s2"), "host-dependent populations with twice the "
                                                                                                     "offsets, positions with masses and spins")):
        for cls, ht in (("AGN", 1), ("Gal", 0)):
            T[f"Prob{cls}{info}"] = {"kind": "membership", "select": sel, "host_type": ht, "info": info, "stat": cls.lower(),
                                     "what": f"mean probability of an AGN host over the mergers truly hosted by {'AGN' if ht else 'galaxies'}, "
                                             f"{what}, input f_AGN = 0.3, realization A (membership of the grid at its posterior median)"}
        # where the probability is evaluated, and its mean over every merger (REVIEW_2 1.1): the same run and membership file as
        # ProbAGN<info> ('partner': its record of the map is used while the map has none of its own)
        T[f"ProbEvalfAGN{info}"] = {"kind": "membership", "select": sel, "info": info, "stat": "f", "partner": f"ProbAGN{info}",
                                    "what": f"f_AGN at which the probability of an AGN host of ProbAGN{info}, ProbGal{info} and ProbAll{info} is "
                                            f"evaluated: the posterior median of f_AGN of the run's grid (membership.json f_agn_median), {what}, "
                                            "input f_AGN = 0.3, realization A"}
        T[f"ProbAll{info}"] = {"kind": "membership", "select": sel, "info": info, "stat": "all", "partner": f"ProbAGN{info}",
                               "what": f"mean probability of an AGN host over all the mergers (both true host classes), {what}, input f_AGN = 0.3, "
                                       f"realization A (membership of the grid at its posterior median, ProbEvalfAGN{info})"}
    T["KnownAGNNoPeak"] = {"kind": "interior", "select": ("key", "KnownAGN"),
                           "what": "number of realizations in which the H0 posterior of the AGN catalog alone (equal populations, three in ten "
                                   "mergers AGN-hosted, population known) has no interior maximum in the prior range [50, 100]"}
    # 21 Nov: the paired difference of the H0 medians with the host class unknown and known on the same mergers (the runs of
    # HWUnknown<f> and HWKnown<f>). (23 Nov: the combined Cls<Unknown|Known><f>Hzero<Comb|CL> are removed, main.tex uses none; the
    # differences read the two half-width macros' records, the same runs and realizations)
    for f in (0.3, 0.5):
        w = F_WORD[f]
        deps = [f"HWUnknown{w}", f"HWKnown{w}"]
        for nm, stat in ((f"ClsDiff{w}HzeroMean", "mean"), (f"ClsDiff{w}HzeroSd", "sd"), (f"ClsDiff{w}Npos", "npos")):
            T[nm] = {"kind": "paired", "stat": stat, "deps": deps, "f": f,
                     "what": f"the H0 median (summary.json) with the host class unknown minus with it known, on the same mergers (the runs of "
                             f"HWUnknown{w} and HWKnown{w}; equal populations, input f_AGN = {f:g}, peak-dominated spectrum), per realization"}
    return T


TABLE = _table()


# ============================================================================================= #
# the run lists -> the map block (written into values/production_map.json by production.py --write-map)
# ============================================================================================= #
def _pattern(name):
    return re.sub(r"realization_\d{3}", "realization_{R}", name)


def _runlist_index():
    """(analysis, set label) -> record and production prefix -> record, records: {run, runlist, stage, realizations}."""
    import yaml
    exp, prod = {}, {}
    if RUNLIST_EXP.exists():
        d = yaml.safe_load(RUNLIST_EXP.read_text())
        # 23 Nov: the experiment's macros stay on the experiment's realizations (the run list's header 'realizations'); the further
        # realizations of a set (make_runlist.EXPERIMENT_EXTRA: obslike_f0p30_s1, 106-115) feed the paper's keys, not these means
        xr = {int(x) for x in (d.get("realizations") or [])}
        for st in d["stages"]:
            for r in st["runs"]:
                R = int(re.search(r"realization_(\d{3})", r["name"]).group(1))
                if xr and R not in xr:
                    continue
                rec = exp.setdefault((r["analysis"], r["set"]["label"]), {"run": _pattern(r["name"]), "runlist": RUNLIST_EXP.name,
                                                                          "stage": str(st["stage"]), "realizations": set()})
                rec["realizations"].add(R)
        for r in d.get("reused_from_stage_1") or []:
            rec = exp.setdefault((r["analysis"], r["set"]), {"run": _pattern(r["run"]), "runlist": RUNLIST.name, "stage": "1 (reused by the experiment)",
                                                             "realizations": set()})
            rec["realizations"].add(int(r["realization"]))
    if RUNLIST.exists():
        d = yaml.safe_load(RUNLIST.read_text())
        for st in d["stages"]:
            if not bv.production_stage(st["stage"]):     # validation (V...) and check (X...) stages: never a production run
                continue
            for r in st["runs"]:
                pat = _pattern(r["name"])
                rec = prod.setdefault(pat.rsplit("/realization_", 1)[0], {"run": pat, "runlist": RUNLIST.name, "stage": str(st["stage"]),
                                                                          "realizations": set()})
                rec["realizations"].add(int(re.search(r"realization_(\d{3})", r["name"]).group(1)))
    for idx in (exp, prod):
        for rec in idx.values():
            rec["realizations"] = sorted(rec["realizations"])
    return exp, prod


def map_block(_runs=None, keys=None):
    """{name: {run, runlist, stage, realizations, how, ...}} for every macro of TABLE whose runs are in the run lists;
    production keys (HDHost, ObsHost, KnownAGN) take the runs of values/production_map.json's key table (passed as
    `keys` by production.build_map, read from the map otherwise)."""
    exp, prod = _runlist_index()
    if keys is None:
        import production
        keys = {k: {"run": pat, "realizations": None} for k, (pat, _) in production.KEY_RUNS.items()}
    out, missing = {}, {}
    for name, e in TABLE.items():
        if e["kind"] == "derived":
            out[name] = {"how": "derived", "fn": e["fn"], "from": e["deps"]}
            continue
        if e["kind"] == "paired":
            # 21 Nov: from the runs of its two half-width macros (the same mergers, realization by realization)
            out[name] = {"how": "derived", "fn": f"paired_{e['stat']}", "from": e["deps"]}
            continue
        sel = e["select"]
        if sel[0] == "exp":
            rec = exp.get((sel[1], sel[2]))
        elif sel[1] in bv.STAGE_V_KEY_RUNS:
            # the second deliberate exception (23 Nov, bv.STAGE_V_KEY_RUNS): HWHost takes key HDHost's runs, stage V8's
            # realization_NNN_unif_then_rslice (the mixed sampler), which the production index above skips; stage 4's pure-rslice run of
            # realization 100 (the only one of its prefix in a production stage) feeds no macro
            vexc = bv.stage_v_key_runs(bv.runlist_index(RUNLIST), sel[1]) if RUNLIST.exists() else {}
            V = bv.STAGE_V_KEY_RUNS[sel[1]]
            rec = ({"run": V["pattern"], "runlist": RUNLIST.name, "stage": V["stage"], "realizations": sorted(vexc), "key": sel[1]}
                   if vexc else None)
        else:
            pat = keys[sel[1]]["run"]
            rec = prod.get(pat.rsplit("/realization_", 1)[0])
            if rec is not None and rec["run"] != pat:
                # 23 Nov: the prefix's first run must be the key's own pattern (e.g. ObsHost's ..._unif_then_rslice), never another run of it
                missing[name] = f"the production runs of {pat.rsplit('/realization_', 1)[0]} in {RUNLIST.name} are {rec['run']}, not key {sel[1]}'s {pat}"
                continue
            if rec is not None:
                rec = dict(rec, key=sel[1])
        if rec is not None and e.get("realizations_of"):
            # 23 Nov: restricted to the realizations of another macro's record (HW<pre>Single: those of HWHost / HWObsHost), all of which the
            # key's runs must have
            ref = out.get(e["realizations_of"])
            if ref is None or not set(ref["realizations"]) <= set(rec["realizations"]):
                missing[name] = (f"the runs of {sel[1:]} lack realizations of {e['realizations_of']} "
                                 f"({None if ref is None else ref['realizations']} against {rec['realizations']})")
                continue
            rec = dict(rec, realizations=list(ref["realizations"]), realizations_of=e["realizations_of"])
        if rec is None:
            missing[name] = f"no run of {sel[1:]} in {RUNLIST_EXP.name} or {RUNLIST.name}"
            continue
        out[name] = dict(rec, how=e["kind"], **({"factor": e["factor"]} if "factor" in e else {}),
                         **({"realization_A": REAL_A} if e["kind"] == "membership" else {}))
    return {"macros": out, "not_in_run_lists": missing}


@lru_cache(None)
def mapped():
    m = json.loads(PMAP.read_text()) if PMAP.exists() else {}
    blk = m.get("classes")
    if not blk:
        blk = map_block()
    elif any(n not in blk["macros"] and n not in (blk.get("not_in_run_lists") or {}) for n in TABLE):
        # a map written before a macro existed (21 Nov: Cls...): its record from the run lists, the map's own records unchanged
        fresh = map_block()
        blk = {"macros": dict(blk["macros"], **{n: r for n, r in fresh["macros"].items() if n not in blk["macros"]}),
               "not_in_run_lists": dict(blk.get("not_in_run_lists") or {},
                                        **{n: w for n, w in fresh["not_in_run_lists"].items() if n not in blk["macros"]})}
    return blk


# ============================================================================================= #
# evaluation
# ============================================================================================= #
@dataclass
class Res:
    name: str
    real: bool
    value: object = None              # float; (median, lo, hi) for an interval
    vmin: float | None = None
    vmax: float | None = None
    files: list = field(default_factory=list)
    missing: list = field(default_factory=list)
    how: str = ""
    extra: dict = field(default_factory=dict)


def run_dir(pat, r):
    return RUNS / pat.format(R=f"{r:03d}")


def _need(rec, which=None):
    reals = rec["realizations"] if which is None else which
    paths = [run_dir(rec["run"], r) / "summary.json" for r in reals]
    return paths, [p for p in paths if not p.exists()]


def _jr(p):
    return json.loads(Path(p).read_text())


@lru_cache(None)
def expected(name):
    import placeholder_results as PR
    v, src = PR.CLASS_EXPECTED[name]
    # the half-width of a key's placeholder of realization A (placeholder_results.json); HWHDSingle, HWObsSingle since 23 Nov
    from_key = {"HWObsHost": "ObsHost", "HWObsSingle": "ObsSingle", "HWHDSingle": "HDSingle"}
    if name in from_key:
        doc = json.loads((PRD / "values" / "placeholder_results.json").read_text())["keys"][from_key[name]]["quantities"]["Hzero"]["A"]
        v = 0.5 * (doc["ci90"][1] - doc["ci90"][0])
    elif isinstance(v, str) and v.startswith("Prob"):
        v = evaluate(v).value
    elif v == "double":
        agn = evaluate("ProbAGNPops").value
        ra, rg, sig = calibrated_r(0.3, 300, 700, agn)
        ra2, rg2, _ = calibrated_r(0.3, 300, 700, None, sig=2 * sig)
        v = float(ra2.mean() if name.startswith("ProbAGN") else rg2.mean())
    elif v == "same_mergers":
        v = same_mergers_expected(name)
    elif v == "all_weighted":
        v = all_weighted_expected(name)
    return v, src


def all_weighted_expected(name):
    """Expected ProbAll<info> while red: the class means ProbAGN<info> and ProbGal<info> (real or red) weighted by the AGN-hosted
    fraction x of the 1000 detected mergers of realization A of that event set (events_meta realised_f_agn)."""
    e = TABLE[name]
    info = e["info"]
    s = int(e["select"][2].rsplit("_s", 1)[1])
    x = det_fraction("peak", 0.3, s, [REAL_A])
    return float(x * evaluate(f"ProbAGN{info}").value + (1 - x) * evaluate(f"ProbGal{info}").value)


def _events_meta(spec, f, s, r):
    """events_meta JSON of the experiment's event set (spectrum, input f, offset size s = 0, 1, 2) of realization r."""
    if spec == "obslike":
        assert f == 0.3 and s in (0, 1), (spec, f, s)
        stem = ("", "marked_dmu0p10_dmuG5_")[s] + "obslike"
    else:
        stem = ("", "marked_dmu0p10_dmuG5_", "marked_dmu0p20_dmuG10_")[s] + ("" if f == 0.3 else f"fagn{F_LAB[f][1:]}_") + "corepop"
    return bv.DATA / f"seed{r}" / "events" / f"events_meta_{stem}.json"


def det_fraction(spec, f, s, reals):
    """Mean over the realizations of the AGN-hosted fraction of the 1000 detected mergers (events_meta realised.realised_f_agn)."""
    return float(np.mean([_jr(_events_meta(spec, f, s, r))["realised"]["realised_f_agn"] for r in reals]))


def same_mergers_expected(name):
    """Expected HW[Obs]KnownPops[Twice]<f> while red: the host-class-known half-width of the equal-population mergers at the same f,
    HW[Obs]Known<f>, rescaled by the information model of the letter (1/hw^2 = (1 - x)/hw_gal^2 + x/hw_AGN^2, hw_gal = HW[Obs]KnownZero,
    hw_AGN = HW[Obs]KnownHundred) from the detected AGN-hosted fraction x of the equal-population set to that of the host-dependent set
    (means over the realizations of the events files' realised_f_agn)."""
    e = TABLE[name]
    spec, _, s = e["select"][2].split("_")
    s, f = int(s[1:]), e["f"]
    pre = "" if spec == "peak" else "Obs"
    hw_eq = evaluate(f"HW{pre}Known{F_WORD[f]}").value
    hg, ha = evaluate(f"HW{pre}KnownZero").value, evaluate(f"HW{pre}KnownHundred").value
    rec = mapped()["macros"].get(name)
    reals = rec["realizations"] if rec else [100, 101, 102, 103, 105]
    x_eq, x_hd = det_fraction(spec, f, 0, reals), det_fraction(spec, f, s, reals)
    info = lambda x: (1 - x) / hg ** 2 + x / ha ** 2
    return float(hw_eq * math.sqrt(info(x_eq) / info(x_hd)))


@lru_cache(None)
def evaluate(name):
    """Res of one macro of TABLE (real or the placeholder)."""
    e = TABLE[name]
    blk = mapped()["macros"]
    kind = e["kind"]
    if kind == "paired":
        return _paired(name, e, blk)
    if kind == "derived":
        deps = [evaluate(d) for d in e["deps"]]
        a, b = deps[0].value, deps[1].value
        v = 100.0 * (a / b) ** 2 if e["fn"] == "share" else (a / b) ** 2
        real = all(d.real for d in deps)
        vmin = vmax = None
        extra = {"from": {d.name: d.value for d in deps}}
        if e["fn"] == "cost":
            vmin, vmax, extra["range"] = cost_range(name, deps, float(v), real)
        return Res(name, real, float(v), vmin, vmax, files=sorted({f for d in deps for f in d.files}), missing=sorted({m for d in deps for m in d.missing}),
                   how=f"{'100 ' if e['fn'] == 'share' else ''}({e['deps'][0]} / {e['deps'][1]})^2 of the unrounded "
                       f"{'values' if real else 'values (some red)'}",
                   extra=extra)
    rec = blk.get(name)
    if rec is None and e.get("partner"):
        rec = blk.get(e["partner"])
    if rec is None:
        v, src = expected(name)
        return _placeholder(name, kind, v, src, [], ["not in the run lists"])
    if kind in ("mean_hw", "factor_hw"):
        paths, miss = _need(rec)
        if miss:
            v, src = expected(name)
            return _placeholder(name, kind, v, src, [str(p) for p in paths], [str(p) for p in miss], rec)
        per = {}
        info = {}
        for r, p in zip(rec["realizations"], paths):
            if kind == "mean_hw":
                b = _jr(p)["posterior"]["H0"]
                per[r] = 0.5 * (b["ci90"][1] - b["ci90"][0])
            else:
                import run_posteriors as RP
                s, inf = RP.factor_summary(p.parent, rec["factor"])
                per[r] = 0.5 * (s["ci90"][1] - s["ci90"][0])
                info[r] = {"median": s["median"], "ci90": s["ci90"], **inf}
        v = float(np.mean(list(per.values())))
        how = (f"mean over {len(per)} realizations of half the 90% width of H0 (summary.json posterior.H0)" if kind == "mean_hw" else
               f"mean over {len(per)} realizations of half the 90% width of H0 of the {rec['factor']} factor (lnL_{rec['factor']} of the fine pass, "
               "Akima onto a 16x lattice, run_posteriors.factor_summary)")
        return Res(name, True, v, min(per.values()), max(per.values()), [str(p) for p in paths], [], how,
                   {"per_realization": per, "run": rec["run"], "realizations": rec["realizations"], "factor_summaries": info})
    if kind == "membership":
        paths, miss = _need(rec, [REAL_A])
        if miss:
            v, src = expected(name)
            return _placeholder(name, kind, v, src, [str(p) for p in paths], [str(p) for p in miss], dict(rec, realizations=[REAL_A]))
        d = paths[0].parent
        mem = _jr(paths[0])["membership"]
        z = np.load(d / "grid" / "membership.npz")
        r_, ht = np.asarray(z["r"], float), np.asarray(z["host_type"], int)
        at = f"r at H0 = {mem['H0_median']:.3f}, f_AGN = {mem['f_agn_median']:.4f}"
        st = e["stat"]
        if st in ("agn", "gal"):
            v = float(r_[ht == e["host_type"]].mean())
            key = "mean_r_true_agn" if e["host_type"] == 1 else "mean_r_true_gal"
            assert abs(v - mem[key]) < 1e-9, (name, v, mem[key])
            how = (f"realization {REAL_A}: mean of r over the {int((ht == e['host_type']).sum())} mergers with host_type {e['host_type']} "
                   f"(membership.npz; = summary.json membership.{key}); {at}")
        elif st == "f":
            v = float(mem["f_agn_median"])
            assert abs(v - float(z["f_agn"])) < 1e-12, (name, v, float(z["f_agn"]))
            how = (f"realization {REAL_A}: the f_AGN at which r is evaluated, summary.json membership.f_agn_median (= membership.npz f_agn; the "
                   f"posterior median of f_AGN of the grid, summary.json posterior.f_agn.median = {_jr(paths[0])['posterior']['f_agn']['median']:.5f}); {at}")
        else:
            v = float(r_.mean())
            assert abs(v - mem["mean_r"]) < 1e-9, (name, v, mem["mean_r"])
            how = f"realization {REAL_A}: mean of r over all {len(r_)} mergers (membership.npz; = summary.json membership.mean_r); {at}"
        return Res(name, True, v, files=[str(paths[0]), str(d / "grid" / "membership.npz")], how=how,
                   extra={"r": r_, "host_type": ht, "f_median": mem["f_agn_median"], "H0_median": mem["H0_median"], "run": rec["run"],
                          "realizations": [REAL_A]})
    if kind == "interior":
        paths, miss = _need(rec)
        if miss:
            return Res(name, False, None, files=[str(p) for p in paths], missing=[str(p) for p in miss], how="interior: waits")
        return interior(rec, paths)
    raise ValueError(kind)


# ---- 21 Nov: the paired difference of the medians (kind 'paired'; 23 Nov: the combinations of kind 'comb', Cls...Hzero<Comb|CL>, removed)
PAIRED_PH = {"mean": 0.0, "sd": 0.3}     # invented values of the paired difference while red (km/s/Mpc); npos: half the realizations


def _paired(name, e, blk):
    """Res of a statistic over the realizations of (H0 median with the host class unknown minus with it known), from the summaries
    of the runs of the two half-width macros e['deps'] (HWUnknown<f>, HWKnown<f>: the same mergers, the same realizations)."""
    st = e["stat"]
    recs = [blk.get(d) for d in e["deps"]]
    if all(r is not None for r in recs):
        assert recs[0]["realizations"] == recs[1]["realizations"], (name, recs)
        (pu, mu), (pk, mk) = _need(recs[0]), _need(recs[1])
        miss = mu + mk
    else:
        pu = pk = []
        miss = ["not in the run lists"]
    if miss:
        n = len(recs[0]["realizations"]) if recs[0] else 5
        v = PAIRED_PH.get(st, n // 2)
        return Res(name, False, v, files=[str(p) for p in pu + pk], missing=[str(p) for p in miss],
                   how=f"placeholder: invented ({'half the realizations, rounded down' if st == 'npos' else f'{v:g} km/s/Mpc'})",
                   extra={"run": recs[0]["run"] if recs[0] else None, "realizations": recs[0]["realizations"] if recs[0] else None})
    reals = recs[0]["realizations"]
    per = {r: float(_jr(a)["posterior"]["H0"]["median"] - _jr(b)["posterior"]["H0"]["median"]) for r, a, b in zip(reals, pu, pk)}
    d = np.array(list(per.values()))
    v = {"mean": float(d.mean()), "sd": float(d.std(ddof=1)), "npos": int((d > 0).sum())}[st]
    how = (f"posterior.H0.median of each summary of {recs[0]['run']} minus that of the same realization of {recs[1]['run']}; "
           + {"mean": "their mean", "sd": "their standard deviation (ddof 1)", "npos": "the number of them above zero"}[st])
    return Res(name, True, v, files=[str(p) for p in pu + pk], how=how,
               extra={"per_realization": per, "run": recs[0]["run"], "partner": recs[1]["run"], "realizations": reals})


def _signed(x, d):
    """x to d decimals, '-0' as '0', a negative value wrapped in \\ensuremath (as the sky-resolution shifts print)."""
    s = bv.fnum(x, d)
    return f"\\ensuremath{{{s}}}" if s.startswith("-") else s


def _placeholder(name, kind, v, src, files, missing, rec=None):
    import placeholder_results as PR
    lo = hi = None
    if kind in ("mean_hw", "factor_hw"):
        lo, hi = v * (1 - PR.CLASS_SPREAD), v * (1 + PR.CLASS_SPREAD)
    return Res(name, False, float(v), lo, hi, files, missing, f"placeholder: {src}",
               {"run": rec["run"] if rec else None, "realizations": rec["realizations"] if rec else None,
                "stage": rec.get("stage") if rec else None, "runlist": rec.get("runlist") if rec else None})


def cost_range(name, deps, v, real):
    """Cost<tag><f>Min, Max: the smallest and largest over the realizations of the same ratio computed per realization,
    (hw_num[r] / hw_den[r])^2, from the summaries' half-widths of the two macros (one realization at a time). While either
    half-width is red (no per-realization values): the central value times the relative spread of the finished equal-population
    cost at the same f, CostUnknown<f>Min / CostUnknown<f> and CostUnknown<f>Max / CostUnknown<f>; where that cost is itself red,
    the central value times 1 -+ 2 CLASS_SPREAD (two half-widths with placeholder bars of +-CLASS_SPREAD). -> (min, max, record)."""
    import placeholder_results as PR
    a, b = deps
    if real:
        pa, pb = a.extra["per_realization"], b.extra["per_realization"]
        assert sorted(pa) == sorted(pb), (name, sorted(pa), sorted(pb))
        per = {r: (pa[r] / pb[r]) ** 2 for r in sorted(pa)}
        return min(per.values()), max(per.values()), {"per_realization": per,
                                                      "how": f"({a.name}[r] / {b.name}[r])^2 per realization r, smallest and largest"}
    ref = f"CostUnknown{F_WORD[TABLE[name]['f']]}"
    rr = evaluate(ref) if name != ref else None
    if rr is not None and rr.real:
        return v * rr.vmin / rr.value, v * rr.vmax / rr.value, {
            "how": f"placeholder: the central value times {ref}Min / {ref} = {rr.vmin / rr.value:.4f} and {ref}Max / {ref} = {rr.vmax / rr.value:.4f} "
                   "(relative spread over the realizations of the finished equal-population cost at the same f)"}
    s = 2 * PR.CLASS_SPREAD
    return v * (1 - s), v * (1 + s), {"how": f"placeholder: the central value times {1 - s:g} and {1 + s:g} (two half-widths with placeholder "
                                             f"bars of +-{PR.CLASS_SPREAD:g})"}


def interior(rec, paths):
    """The AGN catalog alone (key KnownAGN): which realizations have an interior maximum of H0 in the prior range."""
    import run_posteriors as RP
    import placeholder_results as phr
    lo_p, hi_p = RP.P.PRIORS["H0"]
    rows = []
    for r, p in zip(rec["realizations"], paths):
        d = p.parent
        im = RP.interior_maximum(d, "H0")               # the rule, shared since 23 Nov (run_posteriors.interior_maximum; 1-D unchanged)
        best = (im["lnL"], im["argmax"]["H0"])
        at_edge = im["at_edge"]
        sm = _jr(p)["posterior"]["H0"]
        row = {"realization": r, "argmax_H0": best[1], "interior": not at_edge, "median": sm["median"], "ci90": sm["ci90"]}
        if not at_edge:
            x, pp = RP.marginal([d], "H0")
            s = phr.summarize_xp(x, pp, RP.H0_IN)
            row["level_of_input"] = s["level_of_input"]
        rows.append(row)
    n_no = sum(1 for w in rows if not w["interior"])
    return Res("KnownAGNNoPeak", True, n_no, files=[str(p) for p in paths],
               how=f"argmax of the corrected ln L over the nodes of the fine and coarse passes at a prior edge ({lo_p:g} or {hi_p:g}) in {n_no} of "
                   f"{len(rows)} realizations", extra={"rows": rows, "run": rec["run"], "realizations": rec["realizations"]})


# ============================================================================================= #
# the placeholder shapes of fig_classes (c)
# ============================================================================================= #
def calibrated_r(f, n_agn, n_gal, target_agn, sig=None):
    """Placeholder membership probabilities: log-likelihood ratios Gaussian with means +-s^2/2 and variance s^2
    (a calibrated classifier), at deterministic normal quantiles; s solved so that the mean probability of the
    mergers truly hosted by AGN equals target_agn (or given). -> r_agn, r_gal, s."""
    from scipy.optimize import brentq
    from scipy.special import expit, ndtri
    qa = ndtri((np.arange(n_agn) + 0.5) / n_agn)
    qg = ndtri((np.arange(n_gal) + 0.5) / n_gal)
    lf = math.log(f / (1 - f))

    def r_of(s):
        return expit(lf + s * s / 2 + s * qa), expit(lf - s * s / 2 + s * qg)

    if sig is None:
        sig = brentq(lambda s: r_of(s)[0].mean() - target_agn, 1e-3, 6.0)
    return (*r_of(sig), sig)


def membership_draw(info):
    """fig_classes (c): {r_agn, r_gal, mean_agn, mean_gal, real, how, f_in} for info 'Pos' or 'Pops'."""
    ra_, rg_ = evaluate(f"ProbAGN{info}"), evaluate(f"ProbGal{info}")
    if ra_.real and rg_.real:
        r, ht = ra_.extra["r"], ra_.extra["host_type"]
        return {"r_agn": r[ht == 1], "r_gal": r[ht == 0], "mean_agn": ra_.value, "mean_gal": rg_.value, "real": True, "how": ra_.how, "f_in": 0.3}
    ra, rg, sig = calibrated_r(0.3, 300, 700, ra_.value)
    return {"r_agn": ra, "r_gal": rg, "mean_agn": ra_.value, "mean_gal": rg_.value, "real": False, "f_in": 0.3,
            "how": f"placeholder: calibrated logit-normal (sd of the log-likelihood ratio {sig:.2f}) whose AGN mean is ProbAGN{info}"}


# ============================================================================================= #
# the macros (build_values.py)
# ============================================================================================= #
def missing_text(missing):
    """Missing summaries as 'run pattern (missing realizations ...)'."""
    pats = {}
    for m in missing:
        mm = re.match(r"^.*/runs/(.*)/realization_(\d{3})(.*)/summary\.json$", m)
        if mm:
            pats.setdefault(f"{bv.rel(RUNS)}/{mm.group(1)}/realization_{{R}}{mm.group(3)}", []).append(mm.group(2))
        else:
            pats.setdefault(m, [])
    return "; ".join(f"{p} (missing realization{'s' if len(r) > 1 else ''} {','.join(r)})" if r else p for p, r in pats.items())


def _src(res, rec_note=""):
    if res.real:
        return {"files": [bv.rel(f) for f in res.files], "computation": res.how}
    return {"missing": missing_text(res.missing) or "derived from red macros", "placeholder_source": res.how.replace("placeholder: ", "")}


def _runs_text(res):
    ex = res.extra
    if ex.get("run"):
        return f"; runs {ex['run']} (analyses/analysis_16_production_runs/runs), realizations {ex.get('realizations')}"
    return ""


def f2sig(x):
    """x to two significant figures (REVIEW_2 2.2: 3.528 -> 3.5, 0.819 -> 0.82, 0.996 -> 1.0, 9.96 -> 10) -> (text, decimals)."""
    if x == 0 or not np.isfinite(x):
        return bv.fnum(x, 2), 2
    r = round(float(x), 1 - math.floor(math.log10(abs(x))))
    d = max(0, 1 - math.floor(math.log10(abs(r))))
    return bv.fnum(r, d), d


def build(add, fnum, fci, fpct, num_word):
    """Register every macro (build_values.add). Half-widths print two significant figures (f2sig); every derived macro
    (cost, share) is computed from the unrounded half-widths."""
    n_real = 0
    for name, e in TABLE.items():
        res = evaluate(name)
        cls = "plain" if res.real else "ph"
        wrap = (lambda t: t) if res.real else (lambda t: f"\\ph{{{t}}}")
        what = e["what"]
        if e["kind"] in ("mean_hw", "factor_hw"):
            mean = f"mean over the realizations of the 90% half-width of H0, {what}{_runs_text(res)}"
            for suf, v, m in (("", res.value, mean), ("Min", res.vmin, f"smallest over the realizations of the 90% half-width of H0 of {name}"),
                              ("Max", res.vmax, f"largest over the realizations of the 90% half-width of H0 of {name}")):
                t, d = f2sig(v)
                ex = {"decimals": d, "rounding": "two significant figures"}
                if suf == "":
                    ex.update({"per_realization": res.extra.get("per_realization"), "kind": e["kind"]})
                add(name + suf, cls, wrap(t), v, "km/s/Mpc", m, _src(res), "classes", extra=ex)
            n_real += res.real
        elif e["kind"] == "derived":
            d = 0 if e["fn"] == "share" else 1
            src = _src(res) if not res.real else {"files": [bv.rel(f) for f in res.files], "computation": res.how}
            add(name, cls, wrap(fnum(res.value, d)), res.value, "%" if e["fn"] == "share" else "", what, src, "classes",
                extra={"from": res.extra["from"], "decimals": d})
            if e["fn"] == "cost":
                rg = res.extra["range"]
                srcr = dict(src, computation=rg["how"]) if res.real else dict(src, placeholder_source=rg["how"].replace("placeholder: ", ""))
                for suf, v, w in (("Min", res.vmin, "smallest"), ("Max", res.vmax, "largest")):
                    add(name + suf, cls, wrap(fnum(v, d)), v, "",
                        f"{w} over the realizations of the same ratio computed per realization, ({e['deps'][0]}[r] / {e['deps'][1]}[r])^2 "
                        f"(the central {name} is the ratio of the means)", srcr, "classes",
                        extra={"decimals": d, "per_realization": rg.get("per_realization")})
            n_real += res.real
        elif e["kind"] == "membership":
            add(name, cls, wrap(fnum(res.value, 2)), res.value, "", what + _runs_text(res), _src(res), "classes", extra={"decimals": 2})
            n_real += res.real
        elif e["kind"] == "paired":
            ex = {"per_realization": res.extra.get("per_realization"), "realizations": res.extra.get("realizations")}
            if e["stat"] == "npos":
                add(name, cls, wrap(num_word.get(res.value, str(res.value))), res.value, "realizations",
                    f"number of realizations in which the difference is positive, the difference being {what}{_runs_text(res)}", _src(res), "classes",
                    extra=ex)
            else:
                w_ = "mean" if e["stat"] == "mean" else "standard deviation (ddof 1)"
                add(name, cls, wrap(_signed(res.value, 2)), res.value, "km/s/Mpc", f"{w_} over the realizations of {what}{', signed' if e['stat'] == 'mean' else ''}"
                    f"{_runs_text(res)}", _src(res), "classes", extra=dict(ex, decimals=2))
            n_real += res.real
        elif e["kind"] == "interior":
            n_real += _build_interior(add, res, what, fnum, fci, fpct, num_word)
    return n_real


def _build_interior(add, res, what, fnum, fci, fpct, num_word):
    rec = mapped()["macros"].get("KnownAGNNoPeak")
    if res.real:
        rows = res.extra["rows"]
        add("KnownAGNNoPeak", "plain", num_word.get(res.value, str(res.value)), res.value, "realizations", what + _runs_text(res),
            {"files": [bv.rel(f) for f in res.files], "computation": res.how}, "classes", extra={"rows": rows})
        k = 0
        for w in rows:
            if not w["interior"]:
                continue
            k += 1
            base = f"KnownAGNPeak{ORD[k]}Hzero"
            src = {"file": bv.rel(run_dir(rec["run"], w["realization"]) / "summary.json"),
                   "computation": f"summary.json posterior.H0 of the {['first', 'second', 'third', 'fourth', 'fifth'][k - 1]} realization (in the order of "
                                  f"the run list) whose AGN-catalog-alone H0 posterior has an interior maximum (realization {w['realization']}, argmax "
                                  f"{w['argmax_H0']:.2f})"}
            r_, dd = fci(w["median"], w["ci90"][0], w["ci90"][1])
            mean = f"H0 of the AGN catalog alone on equal-population mergers with f_AGN = 0.3 (key KnownAGN), the {['first', 'second', 'third', 'fourth', 'fifth'][k - 1]} realization with an interior maximum"
            add(base, "plain", r_, {"median": w["median"], "ci90": w["ci90"]}, "km/s/Mpc", mean + " (median and 90% equal-tailed interval)", src, "classes",
                extra={"decimals": dd, "realization": w["realization"]})
            for pc, v, t in (("Med", w["median"], "median"), ("Lo", w["ci90"][0], "lower end of the 90% interval"), ("Hi", w["ci90"][1], "upper end of the 90% interval")):
                add(base + pc, "plain", fnum(v, dd), v, "km/s/Mpc", f"{t} of {base}", src, "classes")
            # a level that rounds to 1 prints as '>99.99' (fpct prints exactly 1 as '100'): the input lies outside, not at the edge
            add(base + "CL", "plain", fpct(min(w["level_of_input"], 1.0 - 1e-9)), w["level_of_input"], "%",
                f"two-sided credible level at which the input H0 = 67.74 lies in the H0 posterior of {base} (the run's grid marginal)",
                dict(src, computation=src["computation"] + "; level from run_posteriors.marginal of its grid"), "classes")
        return 1
    v, src = expected("KnownAGNNoPeak")
    srcm = {"missing": missing_text(res.missing), "placeholder_source": src}
    add("KnownAGNNoPeak", "ph", f"\\ph{{{num_word.get(v, str(v))}}}", v, "realizations", what, srcm, "classes")
    (m_, lo, hi), s2 = expected("KnownAGNPeakOneHzero")
    r_, dd = fci(m_, lo, hi)
    add("KnownAGNPeakOneHzero", "ph", f"\\ph{{{r_}}}", None, "km/s/Mpc", "H0 of the AGN catalog alone, the first realization with an interior maximum",
        dict(srcm, placeholder_source=s2), "classes")
    for pc, v_ in (("Med", m_), ("Lo", lo), ("Hi", hi)):
        add("KnownAGNPeakOneHzero" + pc, "ph", f"\\ph{{{fnum(v_, dd)}}}", None, "km/s/Mpc", f"{pc} of KnownAGNPeakOneHzero", dict(srcm, placeholder_source=s2), "classes")
    lv, s3 = expected("KnownAGNPeakOneHzeroCL")
    add("KnownAGNPeakOneHzeroCL", "ph", f"\\ph{{{fpct(lv)}}}", None, "%", "credible level of the input in KnownAGNPeakOneHzero", dict(srcm, placeholder_source=s3), "classes")
    return 0


def nreal_classes():
    """The realizations of every experiment set the macros use: one number, or an error naming the sets that differ."""
    blk = mapped()["macros"]
    sets = {}
    for name, e in TABLE.items():
        rec = blk.get(name)
        if rec is None or e["kind"] not in ("mean_hw", "factor_hw") or e["select"][0] != "exp":
            continue
        sets[rec["run"]] = tuple(rec["realizations"])
    counts = {len(v) for v in sets.values()}
    assert len(counts) == 1, {k: v for k, v in sets.items()}
    return counts.pop(), sorted({r for v in sets.values() for r in v}), sorted(sets)


# ============================================================================================= #
# fig_classes (a), (b): the points
# ============================================================================================= #
SERIES = {"peak": ["unknown", "known", "pops", "pops2", "host", "none", "galaxy"], "obslike": ["unknown", "known", "pops", "host", "none", "galaxy"]}


def points(spec):
    """{series: [dict(x, y, lo, hi, real, macro)]} for panel (a) (spec 'peak') or (b) ('obslike'); 'none' and 'galaxy'
    are horizontal lines (x None); 'galaxy' is the host-class-known point at f = 0 (the galaxy catalog alone)."""
    out = {s: [] for s in SERIES[spec]}
    for name, e in TABLE.items():
        if e.get("spectrum") != spec or e["kind"] != "mean_hw" or e.get("fig") is False:
            continue
        res = evaluate(name)
        pt = {"x": e["f"], "y": res.value, "lo": res.vmin, "hi": res.vmax, "real": res.real, "macro": name}
        out[e["series"]].append(pt)
        if e["series"] == "known" and e["f"] == 0.0:
            out["galaxy"].append(dict(pt, x=None))
    for s in ("none",):
        for pt in out[s]:
            pt["x"] = None
    for s in out:
        out[s].sort(key=lambda p: -1 if p["x"] is None else p["x"])
    return out


def status():
    """{name: (real, missing run dirs)} for production.py --status / --report."""
    out = {}
    for name in TABLE:
        r = evaluate(name)
        out[name] = (bool(r.real), sorted({bv.rel(Path(m).parent) for m in r.missing if m.endswith("summary.json")}))
    return out


def main():
    nreal = 0
    for name in TABLE:
        r = evaluate(name)
        nreal += r.real
        v = r.value if not isinstance(r.value, float) else f"{r.value:.4f}"
        rng = f" [{r.vmin:.3f}, {r.vmax:.3f}]" if r.vmin is not None else ""
        print(f"{name:22s} {'REAL       ' if r.real else 'placeholder'} {v}{rng}  {r.how[:110]}")
    print(f"{nreal} of {len(TABLE)} entries real; realizations: {nreal_classes()[:2]}")


if __name__ == "__main__":
    main()
