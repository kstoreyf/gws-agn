#!/usr/bin/env python
"""Write the run lists: runs/RUNLIST.yaml (the approved production stages 1-5) and
runs/RUNLIST_experiment.yaml (the short paper's experiment, archive/paper_short_independent/EXPERIMENT.md:
stages E1 grids of the present mass spectrum, E2 grids of the observed-like spectrum, E3 its
six-parameter nested runs), with every run's driver arguments, parts (schedulable tasks), estimated
likelihood calls and GPU hours, and its rank on the GPU lanes (`lane_rank`, from LANE_RANKS below).
Both files are written atomically (temporary file, then rename): the lane workers re-read them.

    python scripts/make_runlist.py [--timing tests/results/timing.json] [--realizations-10 100,...]

Stage 5 (paper_prd/runs.txt): the observed-like mass spectrum (data/obslike/README.md), host-dependent
events, on the "5" realizations: stage 4's three classes with the family in the run name
(host_dependent_obslike), the family's events suffix, population inputs and injection lane as E2 takes
them for its (obslike, f = 0.3, s = 1) set from data/experiment_sets.json.

Cost model (the brief's numbers until the tests measure them; --timing merges measured spc_* keys):
  seconds per call at 8,000 samples: 0.39 with the population fixed, 0.44 with it free (both catalogs,
  complete; test a measures spc_fixed, test b the nested classes). One catalog and no catalog are
  assumed as costly as two (conservative). hosts_known evaluates two one-catalog models per cell
  (2 x spc_one_fixed); mass_spectrum_alone with two populations evaluates both populations on every
  sample (2 x spc_none_fixed).
  grid: 2-D (H0, f_AGN) = 861 fine cells (41 x 21) + 286 coarse; 1-D H0 = 81 fine + 51 coarse.
  nested calls PER RUN, by number of sampled parameters. The measured references are for a MERGED PAIR
  of independent runs: Analysis 11D (six parameters) 29,262 calls = 14,245 + 15,017 per run, ~2,900
  iterations each (5.0 calls per iteration); Analysis 12 (seven) 47,705 and 48,959 = ~24,000 per run.
  Per run: six 18,000 = 14,600 + the wider production priors (nlive x ln(prior-volume ratio) = +676
  iterations at 5.0 calls); seven 24,000; five (no catalog) 14,500 and four (free densities) 7,500 are
  estimates (half the former per-pair figures). Eight: 272,000 for the single run = Analysis 13's guarded
  rslice run (228k calls, box H0 [60,76] etc.) + 747 iterations at its 58 calls (test c may change it).
  Two independent nested runs (merged) up to six parameters, one for eight.
  Stage V8 (the eight-parameter sampler, 18 Nov): stage 4's host_dependent_populations runs (same arguments,
  events, injection lane, redraw, one part nested_1, seed 1) sampling uniformly while that is cheaper than rslice
  and with rslice after, run names realization_NNN_unif_then_rslice (--variant):
    realization 100: realization_100_unif_boot5 (the six-parameter configuration, --sample unif: dynesty 2.1.4's
      unif defaults bootstrap 5, enlarge 1) stalled near 150 calls per iteration and was stopped at iteration 1,594
      (30,102 calls); it is held, and realization_100_unif_then_rslice continues it from its nested_1/state.save
      (--init-from, --switch-sample rslice with no further option = dynesty's rslice defaults = the production
      eight-parameter setting: slices 11, enlarge 1.25, bootstrap 0).
    realizations 101, 102, 103, 105: --sample unif --auto-switch-sample rslice --auto-switch-cpi 60: the driver
      switches to the same rslice once a 2,000-call chunk needs more than 60 calls per iteration.
  Cost (V8_COST): seconds per call = the stopped run's own (its sessions.json, likelihood time over calls); an
  eight-parameter run takes calls_by_ndim[8] / 58 iterations (Analysis 13's 58 calls per iteration: 4,690), rslice
  60 calls per iteration; the automatic runs spend realization 100's unif calls up to its switch point at 60
  (iteration 1,558, 24,769 calls; tests/results/unif_then_rslice_checks.json, "cost").
  Stage X (checks of two settings, 18 Nov; build_checks): stage 1's two-tracer grid of realization 100 with the
  catalogs at nside 64 (--nside 64, run name realization_100_nside64), and the m < 18 grid of realization 100 with both
  number densities given in Mpc^-3 (--density given --density-units physical, the counts' values converted at
  H0 = 67.74), the AGN density at that value and twice it (run names ..._units_physical). Seconds per call: the measured
  value of the production run each one varies (stage 1's, stage 2's m < 18 grid). And (18 Nov, appended) stage 1's
  two-tracer grid of realization 100 at nside 32 and at nside 64 with Analysis 14's Monte Carlo settings (--nsamp 32000,
  the record lane pooled with the realization's five further sets; run names realization_100_s32000_inj6 and
  realization_100_nside64_s32000_inj6); seconds per call measured on the experiment's two-catalog grid of realization
  100 at the same settings (X_MC_COST_REFERENCE). And (19 Nov, appended after those five) the same three sky-resolution
  grids on realizations 101, 102, 103, 105 (X_SKYRES_REALIZATIONS: the realizations with six injection sets; run names
  realization_NNN_nside64, realization_NNN_s32000_inj6, realization_NNN_nside64_s32000_inj6), seconds per call measured on
  each realization's own stage-1 grid and X_MC_COST_REFERENCE; held while a survey file or injection set is missing.
  And (19 Nov, second batch, appended after those) the sky-resolution sequence on the same five realizations: stage 1's
  one-catalog grids of the keys AGNhosted, Galhosted, KnownGal at nside 16, 64, 128, and KnownBoth at Analysis 14's settings at
  nside 128 and 16 (build_checks (e)); seconds per call measured on the partner at nside 32 (or the nside-64 run of (c) for
  nside 128), times the ratio of the longest catalog rows below the reference's nside (_x_seq_cost; upper bounds).
  And (21 Nov, appended after those) realization_NNN_nside64 at the production settings on the five other realizations of
  stage 1 (106-110, X_SKYRES_LATER; build_checks (f)); the sequence's AGNhosted grids at nside 128 are held (X_SEQ_HOLD).
  And (21 Nov, appended after those) the experiment's host-class-unknown and host-class-known grids of the sets peak_f0p30_s0
  and peak_f0p50_s0 at nside 64 (build_checks (g)): two_tracers_one_population .../realization_NNN_nside64 and hosts_known
  .../realization_NNN_nside64_agn_s32000_inj1, the E1 partner's arguments plus --nside 64; seconds per call measured on the
  partner (nside 32; an upper bound).
  And (21 Nov, third batch, appended after those) KnownBoth at the production settings of the first ten on ten NEW realizations
  (111-120, X_SKYRES_MORE; build_checks (h)), per realization the nside-32 grid (stage 1's arguments plus --variant nside32: run
  name realization_NNN_nside32, a stage-X run) and the nside-64 grid (realization_NNN_nside64, as realization 106's); seconds per
  call: no partner on record, so the mean measured on 106-110 at the same nside (X_MORE_SPC_FROM).
  Plus a fixed time per process for loading, redrawing, binding and compiling (overhead_hours_per_process;
  measured from the closure test's production grid when its result exists).
  Another family's lane (stage 5, observed-like): the record-lane seconds per call times the measured ratio
  of that family's population-fixed grids on its lane (E2, set obslike_f0p30_s0) to stage 1's on the
  record lane, per catalog class (COST["lane_factor"]; measured_lane_factors).
"""
import argparse
import json
import math
import os
import re
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import prod_common as P  # noqa: E402

# ============================================================================ the order on the GPU lanes
# Each run's `lane_rank` is the rank of the FIRST row that matches it: its stage, and where given its
# mode, analysis, catalogs and realizations. The launcher sorts a lane by (lane_rank, priority, estimated GPU hours per part, name, part); a run that
# matches no row has no rank and falls back to its stage order (launch.py STAGE_ORDER).
R_FIRST = (100, 101, 102, 103, 105)
R_LATER = (106, 107, 108, 109, 110)
LANE_RANKS = [
    # rank  stage  match
    (1.0, 1, {}),                                                      # population known, complete
    (1.5, "E1", {}),                                                   # experiment grids
    (1.5, "E2", {}),
    (2, 2, {"mode": "grid"}),                                          # known population, flux-limited
    (2.5, "V8", {"realizations": (100,)}),                             # realization 100: unif then rslice (continuation)
    (3.5, "V8", {"realizations": (101, 102, 103, 105)}),               # unif then rslice, the other four (after rank 3)
    (2.8, 2, {"mode": "nested"}),                                      # free densities (was 6 until 18 Nov)
    (2.85, "X", {}),                                                   # checks of two settings, realization 100 (18 Nov); sky resolution on 101-105 (19 Nov)
    (2.9, 4, {"analysis": "host_dependent_populations", "realizations": (100,)}),   # V8's rslice reference
    (3, 4, {"realizations": R_FIRST}),                                 # all three classes
    (4, "E3", {}),
    (5, 4, {"analysis": "single_population", "catalogs": "both", "realizations": R_LATER}),
    (7, 3, {"realizations": R_FIRST}),
    (8, 5, {}),
    (9, 3, {"realizations": R_LATER}),
]

COST = {"spc_fixed": 0.39, "spc_free": 0.44, "spc_none_fixed": 0.39, "spc_none_free": 0.44,
        "spc_one_fixed": 0.39, "spc_free_hostdep": 0.44, "spc_flux_free_densities": 0.44,
        "grid_2d_cells": 861 + 286, "grid_1d_cells": 81 + 51,
        "calls_by_ndim": {4: 7500, 5: 14500, 6: 18000, 7: 24000, 8: 272000},
        "calls_measured_reference": {"6_A11D_pair": 29262, "6_A11D_per_run": [14245, 15017],
                                     "7_A12M_pair": 47705, "7_A12M_per_run": [25007, 22698],
                                     "8_A13_box_single_run": 228000},
        "overhead_hours_per_process": 0.05,
        # Analysis 14 (A100, both catalogs): 0.25 / 0.40 / 1.03 s per call at 2000 / 8000 / 32000 samples,
        # i.e. 0.20 s fixed (selection over one injection set of 2.2e6 rows and the rest) + 2.5e-5 s per sample
        "spc_per_sample": 2.5e-5, "spc_per_injection_set": 0.20,
        # another family's lane relative to the record lane, per catalog class (both / none; one catalog
        # takes both's): defaults measured 17 Nov 15:00 UTC on realizations 100-105, replaced by
        # measured_lane_factors() whenever the reference grids are on record
        "lane_factor": {"obslike": {"both": 1.1528, "none": 1.2369}}}
TIMING_KEYS = ("spc_fixed", "spc_free", "spc_none_fixed", "spc_none_free", "spc_one_fixed", "spc_free_hostdep",
               "spc_flux_free_densities", "overhead_hours_per_process")
# the population-fixed grids that measure a family's lane factor: (this family on its lane, the production
# family on the record lane), the same analysis class and realization
LANE_FACTOR_REFERENCES = {
    "obslike": {
        "both": ("two_tracers_one_population/both_catalogs/complete/obslike_f0p30_s0/realization_{R:03d}",
                 "known_population/both_catalogs/complete/equal_populations/realization_{R:03d}"),
        "none": ("mass_spectrum_alone/no_catalog/complete/obslike_f0p30_s0/realization_{R:03d}",
                 "known_population/no_catalog/complete/equal_populations/realization_{R:03d}"),
    },
}


def run_seconds_per_call(name, part="grid"):
    """Seconds per likelihood call of a finished part on record (sum over its sessions), else None."""
    rj = P.RUNS / name / part / "result.json"
    if not rj.exists():
        return None
    try:
        r = json.loads(rj.read_text())
    except json.JSONDecodeError:
        return None
    if not r.get("finished"):
        return None
    ses = r.get("sessions") or []
    calls = sum(s.get("calls", 0) for s in ses)
    return sum(s.get("seconds_in_likelihood", 0.0) for s in ses) / calls if calls else None


def measured_lane_factors(realizations=R_FIRST):
    """{family: {catalog class: {factor, n, ratios}}}: median over realizations of the ratio of seconds per
    call, the family's population-fixed grid on its lane over stage 1's on the record lane."""
    out = {}
    for fam, refs in LANE_FACTOR_REFERENCES.items():
        for cat, (num, den) in refs.items():
            ratios = {}
            for R in realizations:
                a, b = run_seconds_per_call(num.format(R=R)), run_seconds_per_call(den.format(R=R))
                if a and b:
                    ratios[R] = round(a / b, 4)
            if ratios:
                out.setdefault(fam, {})[cat] = {"factor": round(statistics.median(ratios.values()), 4),
                                                "n_realizations": len(ratios), "ratios": ratios,
                                                "numerator": num, "denominator": den}
    return out


def lane_factor(popfam, catalogs):
    """Seconds per call of a population family on its own lane relative to the record lane's (1 for
    the production family: its spc_* were measured on the record lane)."""
    if popfam == "corepop":
        return 1.0
    f = COST["lane_factor"][popfam]
    return f["none"] if catalogs == "none" else f["both"]

TITLES = {1: "population known, complete catalogs", 2: "flux-limited catalogs",
          3: "equal-population events, single population", 4: "host-dependent events",
          5: "observed-like mass spectrum, host-dependent events",
          "E1": "experiment grids, present mass spectrum", "E2": "experiment grids, observed-like mass spectrum",
          "E3": "experiment nested runs: single population on (f = 0.3, s = 2)",
          "V8": "the eight-parameter sampler: stage 4's host-dependent populations sampled uniformly (bootstrap 5) while "
                "that is cheaper than rslice and with rslice after (realization 100 continued from its stopped uniform "
                "run, which is held; realizations 101-105 with the automatic switch at 60 calls per iteration)",
          "X": "checks of two analysis settings on realization 100 (not production results): stage 1's two-tracer grid with "
               "the catalogs pixelized at nside 64; the m < 18 grid with both number densities given in Mpc^-3 (physical "
               "units) at the values from the counts, the AGN density at that value and at twice it; and the two-tracer grid "
               "at nside 32 and at nside 64 with Analysis 14's Monte Carlo settings (32000 samples per merger, six pooled "
               "injection sets)"}
EXPERIMENT_REALIZATIONS = (100, 101, 102, 103, 105)
MEMBERSHIP_REALIZATION = 100
STAGE5_SET = ("obslike", 0.3, 1)     # stage 5's events: the observed-like host-dependent set (E2's obslike_f0p30_s1)
H0_INPUT = 67.74                     # the mocks' H0 (prod_driver.truth_coords)


def arg_of(e, flag):
    a = e.get("args") or []
    return a[a.index(flag) + 1] if flag in a else None


def lane_rank_of(stage, e):
    """The rank of the first LANE_RANKS row matching run e (None: no row matches)."""
    real = arg_of(e, "--realization")
    for rank, st, m in LANE_RANKS:
        if st != stage:
            continue
        if "mode" in m and e.get("mode") != m["mode"]:
            continue
        if "analysis" in m and e.get("analysis") != m["analysis"]:
            continue
        if "catalogs" in m and arg_of(e, "--catalogs") != m["catalogs"]:
            continue
        if "realizations" in m and (real is None or int(real) not in m["realizations"]):
            continue
        return rank
    return None


def with_lane_rank(stage, e):
    """e with `lane_rank` placed after `priority` (the other fields keep their order)."""
    rank = lane_rank_of(stage, e)
    out = {}
    for k, v in e.items():
        out[k] = v
        if k == "priority":
            out["lane_rank"] = rank
    out.setdefault("lane_rank", rank)
    return out


def write_atomic(path, text):
    """Write a file the lane workers may read at any moment: temporary file in the same directory,
    flushed, then renamed over the old one."""
    path = Path(path)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with open(tmp, "w") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def nested_calls(ndim):
    return COST["calls_by_ndim"][ndim]


def spc_of(analysis, catalogs, free, events_kind="equal_populations", flux="complete"):
    pop_free = any(p in free for p in P.POP_FREE)
    if analysis == "free_densities":
        return COST["spc_flux_free_densities"]
    if analysis == "host_dependent_populations":
        return COST["spc_free_hostdep"]
    if analysis == "hosts_known":
        return 2 * COST["spc_one_fixed"]
    if analysis == "mass_spectrum_alone":
        return COST["spc_none_fixed"] * (2 if "f_agn" in free else 1)
    if catalogs == "none":
        return COST["spc_none_free"] if pop_free else COST["spc_none_fixed"]
    if catalogs in ("galaxy", "agn"):
        return COST["spc_free"] if pop_free else COST["spc_one_fixed"]
    return COST["spc_free"] if pop_free else COST["spc_fixed"]


SAMPLER_8 = {"rslice": [], "unif_b0": ["--sample", "unif", "--enlarge", "1.25", "--bootstrap", "0"],
             "rwalk": ["--sample", "rwalk", "--walks", "28"]}
SAMPLER_8_CHOICE = ["rslice"]


def cost(mode, free, s, n_parts=None, calls_per_part=None):
    """n_parts, calls_per_part: overrides for nested runs (stage V8: two parts of eight parameters)."""
    if mode == "grid":
        cells = COST["grid_2d_cells"] if len(free) == 2 else COST["grid_1d_cells"]
        parts = ["grid"]
        calls = cells
    else:
        n_ind = n_parts or (2 if len(free) <= 6 else 1)
        parts = [f"nested_{k}" for k in range(1, n_ind + 1)]
        calls = (calls_per_part or nested_calls(len(free))) * n_ind
    gpu_h = calls * s / 3600 + COST["overhead_hours_per_process"] * len(parts)
    return parts, calls, gpu_h


def finish(e, stage, mode, free, s, ev, inj, priority=0.0, n_parts=None, calls_per_part=None):
    parts, calls, gpu_h = cost(mode, free, s, n_parts=n_parts, calls_per_part=calls_per_part)
    e.update({"stage": stage, "mode": mode, "free": free, "n_parameters": len(free), "parts": parts,
              "est_calls": int(calls), "est_seconds_per_call": round(s, 4), "est_gpu_hours": round(gpu_h, 3),
              "est_gpu_hours_per_part": round(gpu_h / len(parts), 3), "priority": priority,
              "events_file": str(ev) if ev else None, "injections_file": str(inj) if inj else None,
              "data_exists": bool(ev and Path(ev).exists() and P.injections_ready(inj))})
    return e


POOL_SETS_WANTED = [6]   # the record's lane + Analysis 14's five further sets (--pool-sets-wanted)


def agn_only_settings(real, family="corepop", lane=None):
    """Analysis 14's settings for analyses whose mergers are all AGN-hosted and matched to the AGN
    catalog: 32000 samples per merger, and the realization's further injection sets pooled with its lane
    when they were drawn from the same proposal (Analysis 14's sets, record proposal: only on the record's
    lane; one event set keeps one lane for all its analyses). Returns (nsamp, pool, variant, hold)."""
    lane = lane or P.default_injection_lane(family)
    pool, hold = [], None
    if family == "corepop" and P.present_lane(lane):
        pool = P.extra_injection_lanes(real, family)
        if 1 + len(pool) < POOL_SETS_WANTED[0]:
            hold = (f"realization {real} has {1 + len(pool)} of the {POOL_SETS_WANTED[0]} injection sets to pool "
                    "(further sets are being generated); rerun make_runlist.py when they exist")
    return P.NSAMP_AGN_ONLY, pool, P.variant_tag(P.NSAMP_AGN_ONLY, 1 + len(pool)), hold


def spc_mc(base, nsamp=8000, n_sets=1, frac=1.0):
    """Seconds per call at nsamp samples (for a share frac of the mergers) and n_sets pooled sets."""
    return base + COST["spc_per_sample"] * (nsamp - 8000) * frac + COST["spc_per_injection_set"] * (n_sets - 1)


def inputs_of(popfam, kind):
    """The values a production event set was drawn with, per parameter (H0, planted f_AGN, the family's
    population inputs, the AGN offsets of host-dependent sets, the class densities' inputs 1e-3 / 1e-5
    Mpc^-3 at H0 = 67.74 in h^3 Mpc^-3)."""
    pv = P.POPULATION_INPUTS[popfam]
    v = {"H0": H0_INPUT, "f_agn": P.EVENT_KINDS[kind][2]}
    v.update({k: pv[k] for k in P.POP_FREE})
    v.update(P.AGN_OFFSET_INPUTS[popfam] if kind == "host_dependent" else {k: 0.0 for k in P.OFFSETS})
    v["log10n_gal"] = -3.0 + 3 * math.log10(100 / H0_INPUT)
    v["log10n_agn"] = -5.0 + 3 * math.log10(100 / H0_INPUT)
    return v


def check_inputs_in_priors(name, free, popfam, kind):
    """Every sampled parameter's prior contains the value the events were drawn with."""
    inp = inputs_of(popfam, kind)
    bad = {p: (inp[p], P.PRIORS[p]) for p in free
           if inp.get(p) is not None and not (P.PRIORS[p][0] <= inp[p] <= P.PRIORS[p][1])}
    if bad:
        sys.exit(f"[fatal] {name}: inputs outside their priors {bad}")
    return {p: inp[p] for p in free}


def entry(stage, mode, analysis, catalogs, flux, kind, real, density="counts", agn_factor=1.0, extra_args=(),
          lane=None, agn_only=False, family=None, sampler_args=None, name_variant="", n_parts=None,
          calls_per_part=None, a14_mc=False, variant_prefix=""):
    """One production run. family: None for the production family (events *_corepop, POPULATION_INPUTS
    ["corepop"]); else {"name", "suffix", "population_family"} of another mock family (stage 5), passed to
    the driver as --events-suffix, --family-name and --population-family: the name then joins the events
    segment of the run name (<kind>_<name>), as the driver names the run directory.
    a14_mc: Analysis 14's Monte Carlo settings (agn_only_settings: 32000 samples, the realization's injection sets
    pooled) on a run whose mergers are not all AGN-hosted (stage X); variant_prefix goes before the settings tag in
    the run name (e.g. nside64_ -> realization_100_nside64_s32000_inj6)."""
    free = P.free_parameters(analysis, catalogs, flux)
    popfam = family["population_family"] if family else "corepop"
    lane = lane or P.default_injection_lane(popfam)
    variant, nsamp, pool = name_variant, 8000, []
    hold_pool = None
    if agn_only or a14_mc:
        nsamp, pool, variant, hold_pool = agn_only_settings(real, popfam, lane)
        variant = variant_prefix + variant
        extra_args = list(extra_args) + ["--nsamp", str(nsamp), "--variant", variant] + \
            [x for ln in pool for x in ("--pool-injections", ln)]
    args = ["--mode", mode, "--analysis", analysis, "--catalogs", catalogs, "--flux", flux,
            "--events-kind", kind]
    if family:
        args += ["--events-suffix", family["suffix"], "--family-name", family["name"], "--population-family", popfam]
    args += ["--realization", str(real)]
    if flux != "complete" and analysis != "free_densities":
        args += ["--density", density]
    if analysis == "free_densities":
        args += ["--density", "free"]
    if agn_factor != 1.0:
        args += ["--agn-density-factor", f"{agn_factor:g}"]
    args += list(extra_args)
    if not P.present_lane(lane):
        args += ["--injections-lane", lane]
    if sampler_args is not None:
        args += list(sampler_args)
    elif mode == "nested" and len(free) > 6:
        args += SAMPLER_8[SAMPLER_8_CHOICE[0]]
    if name_variant and not (agn_only or a14_mc):
        args += ["--variant", name_variant]
    name = P.run_name(analysis, catalogs, flux, kind, real, density=density, agn_density_factor=agn_factor,
                      family=family["name"] if family else "", variant=variant)
    check_inputs_in_priors(name, free, popfam, kind)
    e = {"name": name, "analysis": analysis, "args": args, "injections_lane": lane, "nsamp": nsamp,
         "injection_pool": pool}
    if kind == "host_dependent":
        hr = hold_reason(family["name"] if family else "peak", 1, lane)
        if hr:
            e["hold"] = hr
    if hold_pool:
        e["hold"] = hold_pool
    s = spc_mc(spc_of(analysis, catalogs, free, kind, flux) * lane_factor(popfam, catalogs), nsamp, 1 + len(pool))
    return finish(e, stage, mode, free, s, P.events_path(real, kind, family["suffix"] if family else P.DEFAULT_SUFFIX),
                  P.injections_path(real, popfam, lane), n_parts=n_parts, calls_per_part=calls_per_part)


def build(r5, r10, r3, hd_lane=None):
    """hd_lane: the injection lane of the host-dependent set (peak, f = 0.3, s = 1) in
    experiment_sets.json (stage 4 is held while it is the present lane)."""
    st = {1: [], 2: [], 3: [], 4: []}
    # Stage 1: population known, complete catalogs (grids). Realization 100's two-tracer grid also
    # writes each merger's AGN-host probability (the experiment's membership output at f = 0.3, s = 0).
    for r in r10:
        st[1].append(entry(1, "grid", "known_population", "both", "complete", "equal_populations", r,
                           extra_args=["--membership"] if r == MEMBERSHIP_REALIZATION else ()))
        if r == MEMBERSHIP_REALIZATION:
            st[1][-1]["priority"] = -1.0   # first on the GPU: it checks against the closure test (a)
    for r in r5:
        st[1].append(entry(1, "grid", "known_population", "galaxy", "complete", "equal_populations", r))
        st[1].append(entry(1, "grid", "known_population", "agn", "complete", "equal_populations", r))
        st[1].append(entry(1, "grid", "known_population", "none", "complete", "equal_populations", r))
        # every merger AGN-hosted, AGN catalog: Analysis 14's 32000 samples and pooled injection sets
        st[1].append(entry(1, "grid", "known_population", "agn", "complete", "agn_hosted", r, agn_only=True))
        st[1].append(entry(1, "grid", "known_population", "galaxy", "complete", "galaxy_hosted", r))
    # Stage 2: flux-limited catalogs
    for fl in ("21", "20", "19", "18"):
        for r in r5:
            st[2].append(entry(2, "grid", "known_population", "both", fl, "equal_populations", r))
    for fl in ("20", "18"):
        for x in (0.5, 2.0):
            for r in r3:
                st[2].append(entry(2, "grid", "known_population", "both", fl, "equal_populations", r, agn_factor=x))
    for fl in ("21", "20", "19", "18"):
        for r in r3:
            st[2].append(entry(2, "nested", "free_densities", "both", fl, "equal_populations", r))
    # Stage 3: equal-population events, single population
    for r in r10:
        st[3].append(entry(3, "nested", "single_population", "both", "complete", "equal_populations", r))
    for r in r10:
        st[3].append(entry(3, "nested", "single_population", "none", "complete", "equal_populations", r))
    # Stage 4: host-dependent events
    for r in r10:
        st[4].append(entry(4, "nested", "single_population", "both", "complete", "host_dependent", r, lane=hd_lane))
    for r in r5:
        st[4].append(entry(4, "nested", "host_dependent_populations", "both", "complete", "host_dependent", r,
                           lane=hd_lane))
    for r in r5:
        st[4].append(entry(4, "nested", "single_population", "none", "complete", "host_dependent", r, lane=hd_lane))
    return st


# stage V8: the eight-parameter model with the six-parameter sampler configuration (validation of the sampler)
V8_VARIANT = "unif_boot5"           # run name: .../realization_NNN_unif_boot5
V8_SAMPLER_ARGS = ["--sample", "unif"]   # bootstrap and enlarge left to dynesty, as in the six-parameter runs
V8_CALLS = {"calls_per_part": 9128,
            "source": "test c's projection for unif with bootstrap 5 (tests/results/sampler_test.json, projection."
                      "per_setting.phaseB_unif_boot5: Phase A's calls to iteration 1500 + 1.96 calls per iteration "
                      "after); a lower bound: Phase A was confined to a plane (tests/results/sampler_test_corrected.json); "
                      "Analysis 13's eight-parameter run with this configuration used 588,771 calls for 2,884 iterations"}


# 18 Nov: unif while it is cheaper than rslice, rslice after (run names realization_NNN_unif_then_rslice)
V8_UNIF_THEN_RSLICE = "unif_then_rslice"
V8_STOPPED = {"realization": 100, "part": "nested_1", "iterations": 1594, "ncall": 30102,
              "source": "its state.save (dynesty it 1595, ncall 30102), stopped 18 Nov at about 150 calls per iteration"}
# the continuation: the stopped state, switched to rslice with no further option (dynesty's rslice defaults, which are
# the production eight-parameter runs' setting: they pass no sampler option, SAMPLER_8["rslice"] == [])
V8_AUTO_ARGS = ["--sample", "unif", "--auto-switch-sample", "rslice", "--auto-switch-cpi", "60"]
V8_COST = {"rslice_calls_per_iteration": 60, "a13_calls_per_iteration": 58,
           # realization 100's own trajectory with 2,000-call chunks: the first chunk above 60 calls per iteration ended
           # at iteration 1,558 (24,769 calls; 29 iterations, 2,421 calls, 83.5 per iteration); the largest chunk before it
           # was 45.5 (tests/results/unif_then_rslice_checks.json, "cost")
           "auto_switch_at": {"iterations": 1558, "ncall": 24769}}


def v8_seconds_per_call():
    """The stopped run's own seconds per call (likelihood time over calls in its sessions.json), else the class's."""
    p = P.RUNS / P.run_name("host_dependent_populations", "both", "complete", "host_dependent", V8_STOPPED["realization"],
                            variant=V8_VARIANT) / V8_STOPPED["part"] / "sessions.json"
    if p.exists():
        ses = json.loads(p.read_text())
        calls = sum(s.get("calls", 0) for s in ses)
        if calls:
            return sum(s.get("seconds_in_likelihood", 0.0) for s in ses) / calls, f"{p.relative_to(P.A16)} ({calls} calls)"
    return COST["spc_free_hostdep"], "spc_free_hostdep (test b)"


def build_v8(realizations, hd_lane=None):
    """Stage V8 (18 Nov): stage 4's host_dependent_populations run of each realization (same arguments, events,
    injection lane, priors, redraw), one part nested_1 (seed 1), sampled uniformly while that is cheaper than rslice
    and with rslice after, named realization_NNN_unif_then_rslice:
      realization 100  --init-from the stopped realization_100_unif_boot5 run's nested_1/state.save, --switch-sample
                       rslice (dynesty's rslice defaults = the production eight-parameter setting); the stopped run
                       (--sample unif, two parts, as listed before) stays in the list, held
      the others       --sample unif --auto-switch-sample rslice --auto-switch-cpi 60."""
    assert SAMPLER_8["rslice"] == [], "the production eight-parameter setting is dynesty's rslice defaults"
    spc, spc_src = v8_seconds_per_call()
    n_it = COST["calls_by_ndim"][8] / V8_COST["a13_calls_per_iteration"]
    cpi = V8_COST["rslice_calls_per_iteration"]
    it_src = (f"{n_it:.0f} iterations in all (calls_by_ndim[8] {COST['calls_by_ndim'][8]:,} at Analysis 13's "
              f"{V8_COST['a13_calls_per_iteration']} calls per iteration; extrapolating the stopped run's ln L(X) as an "
              "eight-dimensional Gaussian gives 4,970-5,100, i.e. about 7% more; tests/results/unif_then_rslice_checks.json, \"cost\")")
    out = []
    for r in realizations:
        stage4 = P.run_name("host_dependent_populations", "both", "complete", "host_dependent", r)
        if r == V8_STOPPED["realization"]:
            e = entry("V8", "nested", "host_dependent_populations", "both", "complete", "host_dependent", r, lane=hd_lane,
                      sampler_args=V8_SAMPLER_ARGS, name_variant=V8_VARIANT, n_parts=2,
                      calls_per_part=V8_CALLS["calls_per_part"])
            e["est_calls_source"] = V8_CALLS["source"]
            e["validates"] = stage4
            cont = P.run_name("host_dependent_populations", "both", "complete", "host_dependent", r,
                              variant=V8_UNIF_THEN_RSLICE)
            e["hold"] = (f"stopped 18 Nov at iteration {V8_STOPPED['iterations']:,} ({V8_STOPPED['ncall']:,} calls; about "
                         f"150 calls per iteration at the end); continued with rslice as {cont} from its "
                         f"{V8_STOPPED['part']}/state.save: never queue it again")
            out.append(e)
            state = P.RUNS / e["name"] / V8_STOPPED["part"] / "state.save"
            left = max(n_it - V8_STOPPED["iterations"], 0)
            c = entry("V8", "nested", "host_dependent_populations", "both", "complete", "host_dependent", r, lane=hd_lane,
                      sampler_args=["--init-from", str(state), "--switch-sample", "rslice"],
                      name_variant=V8_UNIF_THEN_RSLICE, n_parts=1, calls_per_part=int(round(left * cpi)))
            c = finish(c, "V8", "nested", c["free"], spc, c["events_file"], c["injections_file"], n_parts=1,
                       calls_per_part=int(round(left * cpi)))
            c["est_calls_source"] = (f"{left:.0f} iterations left after the stopped run's {V8_STOPPED['iterations']:,} at "
                                     f"{cpi} calls per iteration (rslice); {it_src}")
            c["est_seconds_per_call_source"] = f"the stopped run's own: {spc_src}"
            c["continues"] = {"run": e["name"], "part": V8_STOPPED["part"], "state": str(state),
                              "iterations": V8_STOPPED["iterations"], "ncall": V8_STOPPED["ncall"],
                              "source": V8_STOPPED["source"]}
            c["data_exists"] = bool(c["data_exists"] and state.exists())
            c["validates"] = stage4
            out.append(c)
            continue
        sw = V8_COST["auto_switch_at"]
        calls = int(round(sw["ncall"] + max(n_it - sw["iterations"], 0) * cpi))
        e = entry("V8", "nested", "host_dependent_populations", "both", "complete", "host_dependent", r, lane=hd_lane,
                  sampler_args=V8_AUTO_ARGS, name_variant=V8_UNIF_THEN_RSLICE, n_parts=1, calls_per_part=calls)
        e = finish(e, "V8", "nested", e["free"], spc, e["events_file"], e["injections_file"], n_parts=1,
                   calls_per_part=calls)
        e["est_calls_source"] = (f"unif to the switch as on realization 100 (iteration {sw['iterations']:,}, "
                                 f"{sw['ncall']:,} calls), then {cpi} calls per iteration (rslice) to the end; {it_src}")
        e["est_seconds_per_call_source"] = f"realization 100's stopped run: {spc_src}"
        e["validates"] = stage4
        out.append(e)
    return out


# stage X (18 Nov): checks of two analysis settings on realization 100; the paper's number chain reads the nside-64 run by
# name (paper_prd/scripts/skyres.py) and its selectors skip this stage (as they skip V8)
X_REALIZATION = 100
# 19 Nov: the sky-resolution checks (a) and (c) also on the other realizations with six injection sets (the units check (b)
# stays on realization 100 alone); realization 100's five entries come first and keep their order, the others are appended
X_SKYRES_REALIZATIONS = (100, 101, 102, 103, 105)
# 21 Nov: (a) alone (the production Monte Carlo settings: 8000 samples, the realization's own injection set) on the five other
# realizations of stage 1, which have one injection set each; appended after (e), the paper does not read them
X_SKYRES_LATER = (106, 107, 108, 109, 110)
# 21 Nov, third batch (build_checks (h)): KnownBoth at the production settings of the first ten (8000 samples, the realization's own
# injection set, no pool) on ten NEW realizations (data/corepop/README.md "Third batch"), at nside 32 and 64; they have no stage-1
# run, so the nside-32 grid is a stage-X run of its own (--variant nside32); appended after (g), the paper does not read them.
# Seconds per call: the mean measured on X_MORE_SPC_FROM's runs at the same nside (no partner run is on record).
X_SKYRES_MORE = (111, 112, 113, 114, 115, 116, 117, 118, 119, 120)
X_MORE_SPC_FROM = X_SKYRES_LATER
X_NSIDE = 64
X_UNITS_LIMIT = "18"
# (c) the cost of the sky-resolution pair at Analysis 14's Monte Carlo settings: the measured seconds per call of the experiment's
# two-catalog grid of the same realization at the same settings (both catalogs complete, f_AGN and H0 free, population fixed,
# 32000 samples per merger, the record lane pooled with Analysis 14's five sets: six)
X_MC_COST_REFERENCE = "two_tracers_one_population/both_catalogs/complete/peak_f1p00_s0/realization_{R:03d}_s32000_inj6"


def _x_hold_if_missing(e, surveys, nside=X_NSIDE):
    """Hold a stage-X entry whose survey files or injection sets (the lane and every pooled set) are not on disk yet (the
    pooled sets: agn_only_settings already holds a run with fewer than six); data_exists then says the same."""
    real = arg_of(e, "--realization")
    inj = [Path(e["injections_file"])] + [P.injections_path(int(real), "corepop", ln) for ln in e["injection_pool"]]
    miss = [str(p) for p in surveys if not Path(p).exists()] + [str(p) for p in inj if not P.injections_ready(p)]
    if miss and not e.get("hold"):
        e["hold"] = (f"missing {', '.join(miss)} (survey files: scripts/make_survey_nside.py --seed {real} --nside {nside}); "
                     "rerun make_runlist.py when they exist")
    return e


def _x_skyres(real, spc_src=None):
    """(a) for one realization: stage 1's two-tracer grid with the survey files at nside 64. spc_src: (seconds per call, source)
    instead of the partner's measured value ((h): the realization has no stage-1 run on record); None keeps (a)'s rule."""
    partner = P.run_name("known_population", "both", "complete", "equal_populations", real)
    e = entry("X", "grid", "known_population", "both", "complete", "equal_populations", real,
              extra_args=["--nside", str(X_NSIDE)], name_variant=f"nside{X_NSIDE}")
    surveys = [P.survey_path(real, t, "complete", nside=X_NSIDE) for t in ("gal", "agn")]
    if spc_src is None:
        spc = run_seconds_per_call(partner)
        src = f"measured on {partner} (its sessions)"
    else:
        spc, src = spc_src
    if spc:
        e = finish(e, "X", "grid", e["free"], spc, e["events_file"], e["injections_file"])
        e["est_seconds_per_call_source"] = src
    e["data_exists"] = bool(e["data_exists"] and all(p.exists() for p in surveys))
    e["surveys"] = [str(p) for p in surveys]
    e["check"] = {"what": "sky resolution of the catalog term", "partner": partner, "differs_by": f"--nside {X_NSIDE}"}
    return _x_hold_if_missing(e, surveys)


def _x_skyres_mc(real):
    """(c) for one realization: the pair of (a) at nside 32 and 64 with Analysis 14's Monte Carlo settings."""
    out = []
    ref = X_MC_COST_REFERENCE.format(R=real)
    spc_ref = run_seconds_per_call(ref)
    names = {}
    for nside in (P.SURVEY_NSIDE, X_NSIDE):
        off = nside != P.SURVEY_NSIDE
        e = entry("X", "grid", "known_population", "both", "complete", "equal_populations", real,
                  extra_args=["--nside", str(nside)] if off else (), a14_mc=True, variant_prefix=f"nside{nside}_" if off else "")
        if spc_ref:
            e = finish(e, "X", "grid", e["free"], spc_ref, e["events_file"], e["injections_file"])
            e["est_seconds_per_call_source"] = (f"measured on {ref} (its sessions: the same two-catalog grid class, realization {real}, "
                                                f"{e['nsamp']} samples, {1 + len(e['injection_pool'])} pooled sets)")
        else:
            e["est_seconds_per_call_source"] = "cost model: spc_mc(spc_fixed, nsamp, sets)"
        surveys = [P.survey_path(real, t, "complete", nside=nside) for t in ("gal", "agn")]
        e["data_exists"] = bool(e["data_exists"] and all(p.exists() for p in surveys) and all(
            P.injections_ready(P.injections_path(real, "corepop", ln)) for ln in e["injection_pool"]))
        e["surveys"] = [str(p) for p in surveys]
        names[nside] = e["name"]
        out.append(_x_hold_if_missing(e, surveys))
    for e in out:
        other = names[X_NSIDE] if e["name"] == names[P.SURVEY_NSIDE] else names[P.SURVEY_NSIDE]
        e["check"] = {"what": "sky resolution of the catalog term at Analysis 14's Monte Carlo settings (32000 samples per merger, "
                              "six pooled injection sets)", "partner": other, "differs_by": f"--nside {X_NSIDE}",
                      "settings_of": "README, Monte Carlo settings of the all-AGN-hosted analyses; data/experiment_sets.json extra_record_lane_sets"}
    return out


# (e) 19 Nov, second batch: the sky-resolution sequence. Which part of the likelihood carries KnownBoth's nside-64 shift, and does
# it converge with finer pixels? On X_SKYRES_REALIZATIONS, appended after (d):
#   A  stage 1's three one-catalog grids (keys AGNhosted, Galhosted, KnownGal of paper_prd/values/production_map.json) at nside 16,
#      64 and 128, every other argument the stage-1 run's (the all-AGN grid keeps its 32000 samples and six pooled sets);
#   B  KnownBoth at Analysis 14's Monte Carlo settings (the pair of (c)) at nside 128 and 16.
# Lane order: rank 2.85 as the rest of stage X; launch.py sorts a lane by (rank, priority, estimated hours per part, name), and the
# priority puts A first, then B at nside 128, then B at nside 16 (the name alone would not: it comes after the estimated hours and
# interleaves the realizations).
X_SEQ_ONE_CATALOG = (   # (paper key, catalogs, events kind, mergers all AGN-hosted: Analysis 14's settings, as in stage 1)
    ("AGNhosted", "agn", "agn_hosted", True),
    ("Galhosted", "galaxy", "galaxy_hosted", False),
    ("KnownGal", "galaxy", "equal_populations", False),
)
X_SEQ_ONE_NSIDES = (16, 64, 128)
X_SEQ_BOTH_NSIDES = (128, 16)
X_SEQ_PRIORITY = {"A": 0.0, ("B", 128): 1.0, ("B", 16): 2.0}
# 21 Nov: sequence runs held after they failed (key, nside): launch.py does not queue them and the lane worker skips them
X_SEQ_HOLD = {
    ("AGNhosted", 128): ("held 21 Nov: the Monte Carlo sum is -inf at every H0 at nside 128 with 32000 samples per merger "
                         "(ln L = -inf at every H0, 'diag vs bound ... -inf vs -inf', 'outside mass nan'; two attempts, jobs "
                         "1415023 and 1416415, failed after about one minute each)"),
}


# (g) 21 Nov: the mixture's H0 offset. With both complete catalogs and f_AGN free, the host-class-unknown analysis of the
# experiment's sets (two_tracers_one_population) sits above the host-class-known analysis of the same mergers (hosts_known) in
# all five realizations at nside 32; does the difference change at nside 64? The two analyses of E1 on the sets peak_f0p30_s0 and
# peak_f0p50_s0 at nside 64 on X_SKYRES_REALIZATIONS, appended after (f): the E1 partner's arguments (its events, injection lane,
# redraw, Monte Carlo settings) plus --nside 64, the variant nside64 before the settings tag (run names realization_NNN_nside64,
# realization_NNN_nside64_agn_s32000_inj1). Rank 2.85; priority 0 for f = 0.3, 0.5 for f = 0.5 (a lane takes f = 0.3 first).
# (KnownBoth with f_AGN held at 0.30 is not a run here: the driver cannot hold f_AGN with both catalogs; --free H0 stops at
# the bind with core labels [H0, fcat_2]. tests/skyres_sequence.py reads the f = 0.30 line of the stored 2-D grids instead.)
X_MIX_SETS = (("peak", 0.3, 0), ("peak", 0.5, 0))
X_MIX_ANALYSES = ("hosts_known", "two_tracers_one_population")
X_MIX_PRIORITY = {0.3: 0.0, 0.5: 0.5}


def _x_mix_nside(analysis, fam, f, s, real, smap, nside=X_NSIDE):
    """(g) one E1 grid of the set (fam, f, s) at another nside: exp_entry's run with --nside and the variant nside<N>
    before the settings tag; checked against the partner (E1's entry, built here the way build_experiment builds it)."""
    rec = smap.get((fam, round(f, 4), s)) or {}
    lane = rec.get("lane") or P.default_injection_lane("corepop")
    inj_ok = rec.get("injections_exist")
    lane_ok = None if not isinstance(inj_ok, dict) else bool(inj_ok.get(str(real)))
    prio = X_MIX_PRIORITY[round(f, 4)]
    partner = exp_entry("E1", "grid", analysis, fam, f, s, real, rec.get("suffix"), prio, lane=lane, lane_ok=lane_ok)
    e = exp_entry("X", "grid", analysis, fam, f, s, real, rec.get("suffix"), prio, lane=lane, lane_ok=lane_ok)
    if not e.get("args"):
        sys.exit(f"[fatal] (g) {partner['name']}: the set has no events file yet")
    old = arg_of(e, "--variant") or ""
    variant = f"nside{nside}" + (f"_{old}" if old else "")
    args = list(e["args"])
    if "--variant" in args:
        args[args.index("--variant") + 1] = variant
        args += ["--nside", str(nside)]
    else:
        args += ["--nside", str(nside), "--variant", variant]
    e["args"] = args
    e["name"] = P.run_name(analysis, P.EXPERIMENT_CATALOGS[analysis], "complete", "", real, set_label=e["set"]["label"],
                           variant=variant)
    if _x_strip_nside(e["args"]) != _x_strip_nside(partner["args"]) or arg_of(e, "--nside") != str(nside):
        sys.exit(f"[fatal] {e['name']}: arguments differ from {partner['name']}'s by more than --nside {nside}")
    spc = run_seconds_per_call(partner["name"])
    if spc:
        e = finish(e, "X", "grid", e["free"], spc, e["events_file"], e["injections_file"], priority=prio)
        e["est_seconds_per_call_source"] = f"measured on {partner['name']} (its sessions, nside {P.SURVEY_NSIDE}): an upper bound"
    else:
        e["est_seconds_per_call_source"] = "cost model (partner not on record)"
    tracers = P.CATALOG_SETS["both"]
    surveys = [P.survey_path(real, t, "complete", nside=nside) for t in tracers]
    e["data_exists"] = bool(e["data_exists"] and all(p.exists() for p in surveys))
    e["surveys"] = [str(p) for p in surveys]
    e["check"] = {"what": "the mixture's H0 offset: host class unknown (two_tracers_one_population) against known (hosts_known) "
                          "on the same mergers, at nside 64", "partner": partner["name"], "differs_by": f"--nside {nside}"}
    return _x_hold_if_missing(e, surveys, nside)


def _x_strip_nside(args):
    """args without the --nside and --variant pairs (what may differ between a sequence run and its nside-32 partner)."""
    out, i = [], 0
    while i < len(args):
        if args[i] in ("--nside", "--variant"):
            i += 2
            continue
        out.append(args[i])
        i += 1
    return out


def _x_max_per_pixel(real, tracers, nside):
    """The longest catalog row (objects in the fullest pixel) over the run's complete survey files at this nside; None when one
    is missing."""
    import h5py
    m = []
    for t in tracers:
        p = P.survey_path(real, t, "complete", nside=nside)
        if not p.exists():
            return None
        with h5py.File(p, "r") as f:
            m.append(int(f["ngals"][()].max()))
    return max(m)


def _x_seq_cost(e, ref, ref_nside, nside, real, tracers):
    """Seconds per call of a sequence run: the measured value of `ref` (the same grid class and Monte Carlo settings at ref_nside,
    its sessions), as is when nside >= ref_nside (finer pixels, shorter catalog rows: an upper bound), and below it times the ratio
    of the longest catalog row at nside to that at ref_nside (an upper bound when the time per call grows at most in proportion to
    the row: on the GPU the nside-64 grids of (a), (c), (d) took 0.45-0.50 of the nside-32 time for 0.28 of the row). The cost
    model when ref is not on record. Keeps the entry's priority."""
    spc = run_seconds_per_call(ref)
    if not spc:
        e["est_seconds_per_call_source"] = f"cost model ({ref} not on record)"
        return e
    src = f"measured on {ref} (its sessions)"
    if nside < ref_nside:
        m_new, m_ref = _x_max_per_pixel(real, tracers, nside), _x_max_per_pixel(real, tracers, ref_nside)
        if m_new and m_ref:
            fac = m_new / m_ref
            src += (f" x {fac:.3f}, the longest catalog row at nside {nside} ({m_new} objects) over that at nside {ref_nside} "
                    f"({m_ref}): an upper bound")
        else:
            fac = (ref_nside / nside) ** 2
            src += f" x {fac:g}, the ratio of the pixel areas (a survey file is missing)"
        spc *= fac
    elif nside > ref_nside:
        src += f" at nside {ref_nside}: an upper bound (finer pixels)"
    e = finish(e, "X", "grid", e["free"], spc, e["events_file"], e["injections_file"], priority=e["priority"])
    e["est_seconds_per_call_source"] = src
    return e


def _x_seq_data(e, real, tracers, nside):
    """data_exists with the survey files and every pooled injection set; the surveys listed; held while one is missing."""
    surveys = [P.survey_path(real, t, "complete", nside=nside) for t in tracers]
    e["data_exists"] = bool(e["data_exists"] and all(p.exists() for p in surveys) and all(
        P.injections_ready(P.injections_path(real, "corepop", ln)) for ln in e["injection_pool"]))
    e["surveys"] = [str(p) for p in surveys]
    return surveys


def _x_seq_one(real, key, catalogs, kind, agn_only, nside):
    """A: one stage-1 one-catalog grid at another nside, its arguments otherwise the stage-1 run's."""
    partner = entry(1, "grid", "known_population", catalogs, "complete", kind, real, agn_only=agn_only)
    if agn_only:
        e = entry("X", "grid", "known_population", catalogs, "complete", kind, real, extra_args=["--nside", str(nside)],
                  agn_only=True, variant_prefix=f"nside{nside}_")
    else:
        e = entry("X", "grid", "known_population", catalogs, "complete", kind, real, extra_args=["--nside", str(nside)],
                  name_variant=f"nside{nside}")
    if _x_strip_nside(e["args"]) != _x_strip_nside(partner["args"]) or arg_of(e, "--nside") != str(nside):
        sys.exit(f"[fatal] {e['name']}: arguments differ from {partner['name']}'s by more than --nside {nside}")
    e["priority"] = X_SEQ_PRIORITY["A"]
    tracers = P.CATALOG_SETS[catalogs]
    e = _x_seq_cost(e, partner["name"], P.SURVEY_NSIDE, nside, real, tracers)
    surveys = _x_seq_data(e, real, tracers, nside)
    e["check"] = {"what": "sky resolution of the catalog term, one catalog (the sequence nside 16, 32, 64, 128)", "key": key,
                  "partner": partner["name"], "differs_by": f"--nside {nside}"}
    e = _x_hold_if_missing(e, surveys, nside)
    if (key, nside) in X_SEQ_HOLD and not e.get("hold"):
        e["hold"] = X_SEQ_HOLD[(key, nside)]
    return e


def _x_seq_both(real, nside):
    """B: KnownBoth at Analysis 14's Monte Carlo settings at another nside, its arguments otherwise the pair of (c)'s."""
    e = entry("X", "grid", "known_population", "both", "complete", "equal_populations", real,
              extra_args=["--nside", str(nside)], a14_mc=True, variant_prefix=f"nside{nside}_")
    p32 = entry("X", "grid", "known_population", "both", "complete", "equal_populations", real, a14_mc=True)
    p64 = entry("X", "grid", "known_population", "both", "complete", "equal_populations", real,
                extra_args=["--nside", str(X_NSIDE)], a14_mc=True, variant_prefix=f"nside{X_NSIDE}_")
    if _x_strip_nside(e["args"]) != _x_strip_nside(p64["args"]) or e["args"].index("--nside") != p64["args"].index("--nside"):
        sys.exit(f"[fatal] {e['name']}: arguments differ from {p64['name']}'s by more than the nside")
    e["priority"] = X_SEQ_PRIORITY[("B", nside)]
    tracers = P.CATALOG_SETS["both"]
    ref, ref_nside = (p64["name"], X_NSIDE) if nside > X_NSIDE else (p32["name"], P.SURVEY_NSIDE)
    e = _x_seq_cost(e, ref, ref_nside, nside, real, tracers)
    surveys = _x_seq_data(e, real, tracers, nside)
    e["check"] = {"what": "sky resolution of the catalog term at Analysis 14's Monte Carlo settings (32000 samples per merger, "
                          "six pooled injection sets; the sequence nside 16, 32, 64, 128)", "key": "KnownBoth",
                  "partner": p32["name"], "differs_by": f"--nside {nside}", "arguments_of": p64["name"]}
    return _x_hold_if_missing(e, surveys, nside)


def _x_mean_spc(names, what):
    """(mean seconds per call over the finished runs `names` on record (their sessions), source text), or None if none is."""
    got = {n: run_seconds_per_call(n) for n in names}
    got = {n: v for n, v in got.items() if v}
    if not got:
        return None
    m = statistics.fmean(got.values())
    reals = ", ".join(n.split("realization_")[1][:3] for n in got)
    return m, (f"no partner run on record: the mean ({m:.4f}) of the values measured (their sessions) on {what} of realizations "
               f"{reals} ({len(got)} of {len(names)}: {', '.join(f'{v:.4f}' for v in got.values())})")


def _x_more_pair(real):
    """(h) one new realization (X_SKYRES_MORE): KnownBoth at the production settings at nside 32 and 64, in that order.
    nside 32: stage 1's KnownBoth arguments (as build() makes them for realizations 106-110: 8000 samples, the realization's own
    injection set, no pool, no --membership) plus --variant nside32 (run name realization_NNN_nside32: not a stage-1 name, so the run
    stays in stage X and no paper selector reads it); the driver's default nside 32 reads data/seedNNN/surveys/.
    nside 64: _x_skyres, as realization_106_nside64 (stage 1's arguments plus --nside 64 --variant nside64).
    Checked here: each entry's arguments are stage 1's plus only those flags; stage 1's arguments are those of 106-110 with the
    realization changed; both entries have the same events file, injection set and lane, samples, pool, parameters and grid cost
    (the driver's resolved configurations differ only in run_name, variant, surveys and survey_nside: README, stage X, 21 Nov
    third batch). Seconds per call: the means measured on
    X_MORE_SPC_FROM's runs at the same nside (_x_mean_spc), as no partner run is on record. Held while a file is missing
    (_x_hold_if_missing; a missing events file sets data_exists false, as everywhere)."""
    v32, v64 = f"nside{P.SURVEY_NSIDE}", f"nside{X_NSIDE}"
    p1 = entry(1, "grid", "known_population", "both", "complete", "equal_populations", real)   # as build() makes stage 1's
    p1_args = [x for x in p1["args"] if x != "--membership"]
    for t in X_MORE_SPC_FROM:
        ta = entry(1, "grid", "known_population", "both", "complete", "equal_populations", t)["args"]
        ta = [str(real) if (i > 0 and ta[i - 1] == "--realization") else x for i, x in enumerate(ta)]
        if ta != p1_args:
            sys.exit(f"[fatal] (h) realization {real}: stage 1's arguments differ from realization {t}'s by more than the realization")
    s32 = _x_mean_spc([P.run_name("known_population", "both", "complete", "equal_populations", t) for t in X_MORE_SPC_FROM],
                      f"stage 1's KnownBoth grids (nside {P.SURVEY_NSIDE})")
    s64 = _x_mean_spc([P.run_name("known_population", "both", "complete", "equal_populations", t, variant=v64)
                       for t in X_MORE_SPC_FROM], f"stage X's KnownBoth grids at nside {X_NSIDE} (realization_NNN_{v64})")
    # nside 64: (a)'s entry with the mean seconds per call
    e64 = _x_skyres(real, spc_src=s64 or (None, None))
    if not s64:
        e64["est_seconds_per_call_source"] = "cost model (no run of X_MORE_SPC_FROM on record at nside 64)"
    # nside 32: built the way _x_skyres builds (a)'s entry, at the default nside
    e = entry("X", "grid", "known_population", "both", "complete", "equal_populations", real, name_variant=v32)
    surveys = [P.survey_path(real, t, "complete") for t in ("gal", "agn")]
    if s32:
        e = finish(e, "X", "grid", e["free"], s32[0], e["events_file"], e["injections_file"])
        e["est_seconds_per_call_source"] = s32[1]
    else:
        e["est_seconds_per_call_source"] = "cost model (no run of X_MORE_SPC_FROM on record at nside 32)"
    e["data_exists"] = bool(e["data_exists"] and all(p.exists() for p in surveys))
    e["surveys"] = [str(p) for p in surveys]
    e32 = e
    if e32["args"] != p1_args + ["--variant", v32] or e64["args"] != p1_args + ["--nside", str(X_NSIDE), "--variant", v64]:
        sys.exit(f"[fatal] (h) realization {real}: arguments are not stage 1's plus the nside flags: {e32['args']} / {e64['args']}")
    same = ("analysis", "injections_lane", "nsamp", "injection_pool", "mode", "free", "est_calls", "events_file", "injections_file")
    if any(e32[k] != e64[k] for k in same) or any(e32[k] != p1[k] for k in same):
        sys.exit(f"[fatal] (h) realization {real}: the pair differs in {[k for k in same if e32[k] != e64[k]]}")
    what = ("KnownBoth at the production settings on a new realization (is the H0 offset over the first ten a fluctuation?): "
            "both complete catalogs, f_AGN free, 8000 samples per merger, the realization's own injection set")
    e32["check"] = {"what": what, "partner": e64["name"], "differs_by": f"--nside {X_NSIDE} (the survey files)",
                    "arguments_of": "stage 1's known_population/both_catalogs/complete/equal_populations/realization_NNN of "
                                    f"{', '.join(map(str, X_MORE_SPC_FROM))} (without --membership) plus --variant {v32}"}
    e64["check"] = {"what": what, "partner": e32["name"], "differs_by": f"--nside {X_NSIDE} (the survey files)",
                    "arguments_of": f"as realization_{X_MORE_SPC_FROM[0]}_{v64} (_x_skyres)"}
    return [_x_hold_if_missing(e32, surveys, P.SURVEY_NSIDE), e64]


def build_checks(real=X_REALIZATION, skyres_realizations=X_SKYRES_REALIZATIONS, later_realizations=X_SKYRES_LATER,
                 more_realizations=X_SKYRES_MORE):
    """Stage X.
    (a) Sky resolution: stage 1's known_population grid with both catalogs (realization 100, the redraw of index 0, 8000
        samples) with the survey files at nside 64 (data/seedNNN/surveys_ns64/, scripts/make_survey_nside.py): --nside 64,
        run name realization_100_nside64 (--variant); no membership output. Seconds per call: the stage-1 run's measured
        value (its sessions; the catalog arrays have the same number of entries at nside 64).
    (b) Density units: stage 2's m < 18 grid (realization 100) with both number densities given in Mpc^-3 instead of from
        the counts in h^3 Mpc^-3, at the values from the counts converted at H0 = 67.74 (log10 n + 3 log10 0.6774, so the
        model equals the production one at H0 = 67.74 and differs only in how the densities scale with H0), the AGN
        density at that value and at twice it; run names ..._densities_given[_agn_density_double]/.../realization_100_
        units_physical. Seconds per call: the stage-2 m < 18 run's measured value.
    (c) Sky resolution at Analysis 14's Monte Carlo settings (18 Nov): the pair of (a) at nside 32 and at nside 64 with 32000
        samples per merger and the realization's six injection sets pooled (the record lane and Analysis 14's five further sets,
        data/experiment_sets.json 'extra_record_lane_sets'; agn_only_settings, the settings of the all-AGN analyses), run names
        realization_100_s32000_inj6 and realization_100_nside64_s32000_inj6. At 8000 samples and one set the nside-64 run has
        about twice the nside-32 run's Monte Carlo variance (posterior-sample sums 4.1 against 2.2, selection N^2/N_eff 2.5
        against 1.4), so the shift between (a) and its partner does not isolate the resolution. Seconds per call: the measured
        value of X_MC_COST_REFERENCE (the same grid class at the same settings, realization 100), else the cost model.
    (d) 19 Nov: (a) and (c) on every realization of skyres_realizations (100, 101, 102, 103, 105: the realizations with six
        injection sets), same arguments with the realization's own events, lane, five further sets and survey files (nside 64:
        scripts/make_survey_nside.py --seed R --nside 64), run names realization_NNN_nside64, realization_NNN_s32000_inj6,
        realization_NNN_nside64_s32000_inj6; seconds per call measured on the same realization's stage-1 grid and
        X_MC_COST_REFERENCE. Realization `real`'s five entries come first, in the order above; the others are appended (per
        realization: nside64, s32000_inj6, nside64_s32000_inj6). An entry whose survey files or injection sets are missing is
        held (_x_hold_if_missing; fewer than six sets: agn_only_settings). The paper's mean and scatter of the shift over the
        realizations: paper_prd/scripts/skyres.py.
    (e) 19 Nov, second batch, appended after (d): the sky-resolution sequence on skyres_realizations. A: stage 1's one-catalog
        grids of the keys AGNhosted (agn_hosted events, AGN catalog; 32000 samples, six pooled sets as stage 1), Galhosted
        (galaxy_hosted events, galaxy catalog) and KnownGal (equal_populations events, galaxy catalog) at nside 16, 64 and 128
        (--nside N; run names realization_NNN_nside<N>[_s32000_inj6]), per realization in that order; B: KnownBoth at
        Analysis 14's Monte Carlo settings at nside 128 and 16 (realization_NNN_nside<N>_s32000_inj6, the arguments of (c)'s
        nside-64 run with the other nside). Every other argument is the partner's (checked here). Priority 0 (A), 1 (B, nside
        128), 2 (B, nside 16) orders them on the lane within rank 2.85; seconds per call: _x_seq_cost; held while a survey file
        or injection set is missing. Survey files: scripts/make_survey_nside.py --seed R --nside 16|128. Tabulated by
        tests/skyres_sequence.py; the paper does not read them. 21 Nov: A's AGNhosted grids at nside 128 are held
        (X_SEQ_HOLD: ln L = -inf at every H0).
    (f) 21 Nov, appended after (e): (a) alone, at the production Monte Carlo settings, on the realizations of later_realizations
        (106-110, the other five of stage 1, one injection set each): run names realization_NNN_nside64, the arguments of the
        stage-1 KnownBoth run without --membership plus --nside 64 --variant nside64 (as realization 100's (a)). Does the
        nside-64 shift of the first five repeat on five more? Tabulated by tests/skyres_sequence.py (block KnownBoth at 8000
        samples); the paper's skyres.py reads only skyres_realizations.
    (g) 21 Nov, appended after (f): the experiment's host-class-known (hosts_known, D1) and host-class-unknown
        (two_tracers_one_population, D2) grids of the sets X_MIX_SETS (peak_f0p30_s0, peak_f0p50_s0) at nside 64 on
        skyres_realizations: E1's arguments plus --nside 64, run names realization_NNN_nside64_agn_s32000_inj1 and
        realization_NNN_nside64 (_x_mix_nside); per set, per realization, hosts_known then two_tracers_one_population. Seconds
        per call: the E1 partner's measured value (nside 32, an upper bound). Priority X_MIX_PRIORITY (f = 0.3 first). Held while
        a survey file is missing. Tabulated by tests/skyres_sequence.py (blocks HostsKnown_mix, HostsUnknown_mix).
    (h) 21 Nov, third batch, appended after (g): KnownBoth at the production settings of the first ten on the new realizations of
        more_realizations (111-120, data/corepop/README.md "Third batch"): per realization, in that order, realization_NNN_nside32
        (stage 1's arguments plus --variant nside32; stage X, so no paper selector reads it) and realization_NNN_nside64 (as (f)),
        a pair that differs only in the survey files (_x_more_pair). Seconds per call: the means measured on X_MORE_SPC_FROM at the
        same nside. Held while a survey file or injection set is missing; data_exists false while the events file is. Tabulated by
        tests/skyres_sequence.py (block KnownBoth_production_111_120); the paper does not read them."""
    out = [_x_skyres(real)]
    lim = X_UNITS_LIMIT
    unit_partner = P.run_name("known_population", "both", lim, "equal_populations", real)
    dens = {t: P.density_from_counts(P.survey_path(real, t, lim), P.fit_path(t, lim)) for t in ("gal", "agn")}
    phys = {t: dens[t]["log10n_h_scaled"] + 3 * math.log10(H0_INPUT / 100.0) for t in dens}
    spc = run_seconds_per_call(unit_partner)
    for x in (1.0, 2.0):
        e = entry("X", "grid", "known_population", "both", lim, "equal_populations", real, density="given", agn_factor=x,
                  extra_args=["--log10n-gal", f"{phys['gal']:.6f}", "--log10n-agn", f"{phys['agn']:.6f}",
                              "--density-units", "physical"], name_variant="units_physical")
        if spc:
            e = finish(e, "X", "grid", e["free"], spc, e["events_file"], e["injections_file"])
            e["est_seconds_per_call_source"] = f"measured on {unit_partner} (its sessions)"
        prod = P.run_name("known_population", "both", lim, "equal_populations", real, density="counts", agn_density_factor=x)
        e["check"] = {"what": "units of the assumed number densities (Mpc^-3 against h^3 Mpc^-3)", "partner": prod,
                      "log10n_Mpc-3": {t: round(phys[t], 6) for t in phys},
                      "log10n_h3Mpc-3_from_counts": {t: dens[t]["log10n_h_scaled"] for t in dens},
                      "converted_at_H0": H0_INPUT}
        out.append(e)
    # (c) the sky-resolution pair at Analysis 14's Monte Carlo settings (appended: the entries above keep their order)
    out += _x_skyres_mc(real)
    # (d) the other realizations' (a) and (c), appended after realization `real`'s five entries
    for r in skyres_realizations:
        if r != real:
            out.append(_x_skyres(r))
            out += _x_skyres_mc(r)
    # (e) the sky-resolution sequence, appended after (d): A (one catalog at nside 16, 64, 128), then B (KnownBoth, 128 and 16)
    for r in skyres_realizations:
        for key, catalogs, kind, agn_only in X_SEQ_ONE_CATALOG:
            for nside in X_SEQ_ONE_NSIDES:
                out.append(_x_seq_one(r, key, catalogs, kind, agn_only, nside))
    for r in skyres_realizations:
        for nside in X_SEQ_BOTH_NSIDES:
            out.append(_x_seq_both(r, nside))
    # (f) (a) on the later realizations, appended after (e)
    for r in later_realizations:
        e = _x_skyres(r)
        partner = P.run_name("known_population", "both", "complete", "equal_populations", r)
        p1 = entry(1, "grid", "known_population", "both", "complete", "equal_populations", r,       # as build() makes stage 1's
                   extra_args=["--membership"] if r == MEMBERSHIP_REALIZATION else ())
        p1_args = [x for x in p1["args"] if x != "--membership"]
        if _x_strip_nside(e["args"]) != p1_args or arg_of(e, "--nside") != str(X_NSIDE) or p1["name"] != partner:
            sys.exit(f"[fatal] {e['name']}: arguments differ from {partner}'s by more than --nside {X_NSIDE}")
        out.append(e)
    # (g) the mixture's offset: E1's host-class-known and -unknown grids at nside 64, appended after (f)
    smap, _ = experiment_set_map()
    for fam, f, s in X_MIX_SETS:
        for r in skyres_realizations:
            for analysis in X_MIX_ANALYSES:
                out.append(_x_mix_nside(analysis, fam, f, s, r, smap))
    # (h) the new realizations' pairs (nside 32, then 64), appended after (g)
    for r in more_realizations:
        out += _x_more_pair(r)
    return out


def build_stage5(realizations, rec):
    """Stage 5: stage 4's three classes (single population with both catalogs, host-dependent populations
    with the eight-parameter sampler setting, single population without a catalog) on the observed-like
    host-dependent set, which E2 runs as obslike_f0p30_s1: its events suffix and injection lane from
    data/experiment_sets.json, its population inputs POPULATION_INPUTS["obslike"]. The redraw stream is
    E2's on that set (same kind and suffix), so a merger has the same samples in both."""
    label = P.set_name(*STAGE5_SET)
    if not rec or not rec.get("suffix"):
        sys.exit(f"[fatal] stage 5: no {label} set in {SETS_JSON}")
    kind, sfx = P.kind_and_suffix_of_file(f"events{rec['suffix']}.h5")
    popfam = P.family_of_suffix(sfx)
    if kind != "host_dependent" or popfam != STAGE5_SET[0]:
        sys.exit(f"[fatal] stage 5: {label} is events{rec['suffix']}.h5 (kind {kind}, family {popfam}); "
                 "expected the observed-like host-dependent file")
    family = {"name": STAGE5_SET[0], "suffix": sfx, "population_family": popfam}
    lane = rec.get("lane") or P.default_injection_lane(popfam)
    out = []
    for analysis, catalogs in (("single_population", "both"), ("host_dependent_populations", "both"),
                               ("single_population", "none")):
        for r in realizations:
            out.append(entry(5, "nested", analysis, catalogs, "complete", kind, r, lane=lane, family=family))
    inj_ok = rec.get("injections_exist")
    if isinstance(inj_ok, dict):
        for e in out:
            r = arg_of(e, "--realization")
            if not inj_ok.get(str(r)):
                e["lane_not_validated"] = (f"data/experiment_sets.json reports no {lane} for realization {r} "
                                           "yet; rerun make_runlist.py when it does")
                e["data_exists"] = False
    return out


# ============================================================================ the experiment
# (family, f, s) -> events suffix (file events<suffix>.h5) of the sets that exist today; the rest of
# EXPERIMENT.md's table comes from working/data/experiment_sets.json when it appears.
KNOWN_SETS = {("peak", 0.0, 0): "_puregal_corepop", ("peak", 0.1, 0): "_fagn0p10_corepop",
              ("peak", 0.3, 0): "_corepop", ("peak", 1.0, 0): "_pureagn_corepop",
              ("peak", 0.3, 1): "_marked_dmu0p10_dmuG5_corepop",
              ("obslike", 0.3, 0): "_obslike", ("obslike", 0.3, 1): "_marked_dmu0p10_dmuG5_obslike"}
PLANNED_SETS = ([("peak", f, s) for s in (0, 1, 2) for f in (0.0, 0.1, 0.3, 0.5, 1.0)
                 if (s == 0) or f in (0.1, 0.3, 0.5)]
                + [("obslike", 0.0, 0), ("obslike", 0.3, 0), ("obslike", 1.0, 0), ("obslike", 0.3, 1)])
# family names as data/experiment_sets.json writes them: "peak" (the present mass spectrum, 90% in the
# Gaussian peak; events *_corepop) and "obslike" (the observed-like spectrum)
FAMILY_ALIASES = {"peak": "peak", "mock": "peak", "corepop": "peak", "present": "peak", "core": "peak",
                  "obslike": "obslike", "observed_like": "obslike", "observed-like": "obslike", "gwtc3_pp": "obslike"}
SETS_JSON = P.DATA / "experiment_sets.json"


def _norm_key(fam, f, s):
    fam = FAMILY_ALIASES.get(str(fam).strip().lower(), str(fam).strip().lower())
    return fam, round(float(f), 4), int(round(float(s)))


def _suffix_of(v):
    if isinstance(v, str):
        name = Path(v).name
        if name.endswith(".h5"):
            name = name[:-3]
        return name[len("events"):] if name.startswith("events") else name
    if isinstance(v, dict):
        for k in ("suffix", "events_suffix", "file_suffix", "events_file", "file", "filename"):
            if k in v and v[k]:
                return _suffix_of(v[k])
    raise ValueError(f"no events suffix in {v!r}")


def _lane_of(v):
    """The injection lane a set record names (relative to the realization's directory), or None."""
    if not isinstance(v, dict):
        return None
    for k in ("injections", "injections_lane", "injection_lane", "injections_file", "injection_file"):
        if v.get(k):
            lane = str(v[k])
            for pre in ("seed<R>/", "seed{R}/", "seed<r>/"):
                lane = lane.replace(pre, "")
            return lane.lstrip("/")
    return None


def parse_experiment_sets(obj):
    """working/data/experiment_sets.json -> {(family, f, s): {"suffix", "lane"}}. Accepts a list of records (or
    {"sets": [...]}) with family/f/s and suffix (or events_file), a flat map "family|f|s" or
    "family,f,s" -> suffix (or record), or nested {family: {f: {s: suffix}}}."""
    out = {}
    if isinstance(obj, dict) and "sets" in obj:
        obj = obj["sets"]
    if isinstance(obj, list):
        for rec in obj:
            fam = rec.get("family") or rec.get("spectrum") or rec.get("mass_spectrum")
            f = rec.get("f", rec.get("f_agn"))
            s = rec.get("s", rec.get("size", rec.get("population_difference")))
            out[_norm_key(fam, f, s)] = {"suffix": _suffix_of(rec), "lane": _lane_of(rec),
                                         "injections_exist": rec.get("injections_exist")}
        return out
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k.startswith("_") or k in ("what", "note", "notes", "written_utc", "generated_utc", "source"):
                continue
            parts = [x for x in k.replace("(", "").replace(")", "").replace("|", ",").replace(";", ",").split(",")
                     if x.strip()]
            if len(parts) == 3:
                out[_norm_key(parts[0].strip().strip("'\""), parts[1], parts[2])] = {"suffix": _suffix_of(v),
                                                                                    "lane": _lane_of(v)}
            elif isinstance(v, dict):
                for fk, fv in v.items():
                    if isinstance(fv, dict) and not any(x in fv for x in ("suffix", "events_suffix", "events_file")):
                        for sk, sv in fv.items():
                            out[_norm_key(k, fk, sk)] = {"suffix": _suffix_of(sv), "lane": _lane_of(sv)}
                    else:
                        raise ValueError(f"cannot read entry {k!r} of {SETS_JSON}")
        return out
    raise ValueError(f"cannot read {SETS_JSON}")


def experiment_set_map():
    """(family, f, s) -> {"suffix", "lane"}: the sets known today, updated from experiment_sets.json;
    a set's lane defaults to its family's (the present lane for the peak family)."""
    m = {k: {"suffix": v, "lane": None} for k, v in KNOWN_SETS.items()}
    info = {"experiment_sets_json": str(SETS_JSON), "present": SETS_JSON.exists()}
    if SETS_JSON.exists():
        new = parse_experiment_sets(json.loads(SETS_JSON.read_text()))
        clash = {k: (m[k]["suffix"], v["suffix"]) for k, v in new.items() if k in m and m[k]["suffix"] != v["suffix"]}
        if clash:
            info["differs_from_known"] = {f"{k}": v for k, v in clash.items()}
        m.update(new)
        info["n_sets_in_json"] = len(new)
    for (fam, f, s), v in m.items():
        if not v.get("lane"):
            v["lane"] = P.default_injection_lane("obslike" if fam == "obslike" else "corepop")
    info["lanes"] = {f"{k[0]}|{k[1]}|{k[2]}": v["lane"] for k, v in sorted(m.items())}
    return m, info


def hold_reason(fam, s, lane):
    """Runs on host-dependent sets (s > 0) wait while their lane is the present one: it serves equal
    populations (selection N^2/N_eff 1.3-1.6) but not the host-dependent classes (about 3 at s = 1,
    f = 0.3; 5.5 at f = 0.5; up to 18 at stressed points). They run once experiment_sets.json names a
    broadened lane for the set (rerun make_runlist.py, then launch.py)."""
    if s > 0 and P.present_lane(lane):
        return (f"host-dependent set (s = {s:g}) on the present injection lane {lane}: waiting for the "
                "broadened lane to be named in data/experiment_sets.json")
    return None


def f_priority(f):
    """EXPERIMENT order on one GPU: the f = 0.3 sets, then f = 0.1 and 0.5, then the rest."""
    return 0 if abs(f - 0.3) < 1e-9 else 1 if (abs(f - 0.1) < 1e-9 or abs(f - 0.5) < 1e-9) else 2


def experiment_analyses(f, s):
    """The grids run on a set (f, s): (analysis, note). hosts_known at f = 0 is galaxy_catalog_alone
    (every merger galaxy-hosted), so it is not run twice."""
    out = ["mass_spectrum_alone"]
    if f < 1.0:
        out.append("galaxy_catalog_alone")
    out.append("two_tracers_one_population")
    if s > 0:
        out.append("two_tracers_two_populations")
    if f > 0.0:
        out.append("hosts_known")
    return out


def stage1_equivalent(analysis, fam, f, s, r):
    """The production stage-1 run that IS this experiment run (same events file, same samples)."""
    if fam != "peak" or s != 0:
        return None
    k = (analysis, f)
    table = {("mass_spectrum_alone", 0.3): ("none", "equal_populations"),
             ("galaxy_catalog_alone", 0.3): ("galaxy", "equal_populations"),
             ("two_tracers_one_population", 0.3): ("both", "equal_populations"),
             ("galaxy_catalog_alone", 0.0): ("galaxy", "galaxy_hosted"),
             ("hosts_known", 1.0): ("agn", "agn_hosted")}
    if k not in table:
        return None
    cat, kind = table[k]
    variant = agn_only_settings(r)[2] if kind == "agn_hosted" else ""   # (the record's lane)
    return P.run_name("known_population", cat, "complete", kind, r, variant=variant)


def exp_entry(stage, mode, analysis, fam, f, s, r, suffix, priority, membership=False, lane=None,
              lane_ok=None):
    """One experiment run. Mergers all AGN-hosted and matched to the AGN catalog (two_tracers_* and
    hosts_known on f = 1 sets) and the AGN-hosted subset of hosts_known get Analysis 14's settings
    (32000 samples per merger, the realization's injection sets pooled); the run name carries them."""
    label = P.set_name(fam, f, s)
    popfam = "obslike" if fam == "obslike" else "corepop"
    catalogs = "both" if analysis == "single_population" else P.EXPERIMENT_CATALOGS[analysis]
    if analysis == "single_population":
        free = P.free_parameters(analysis, catalogs, "complete")
    else:
        free = P.free_parameters(analysis, catalogs, "complete", populations_differ=s > 0)
    mc_args, variant, nsamp, pool, frac, hold_pool = [], "", 8000, [], 1.0, None
    if analysis == "hosts_known" and f > 0:
        nsamp, pool, tag, hold_pool = agn_only_settings(r, popfam, lane)
        variant = "agn_" + tag
        mc_args = ["--agn-subset-nsamp", str(nsamp), "--variant", variant] + \
            [x for ln in pool for x in ("--agn-subset-pool", ln)]
        frac = f
    elif f >= 1.0 and analysis in ("two_tracers_one_population", "two_tracers_two_populations"):
        nsamp, pool, variant, hold_pool = agn_only_settings(r, popfam, lane)
        mc_args = ["--nsamp", str(nsamp), "--variant", variant] + [x for ln in pool for x in ("--pool-injections", ln)]
    name = P.run_name(analysis, catalogs, "complete", "", r, set_label=label, variant=variant)
    e = {"name": name, "analysis": analysis,
         "set": {"family": fam, "f": f, "s": s, "label": label, "suffix": suffix, "population_family": popfam}}
    if lane_ok is False:
        e["lane_not_validated"] = (f"data/experiment_sets.json: injections_exist[{r}] is false for {lane}; "
                                   "rerun make_runlist.py when it is true")
    if suffix is None:
        e["args"] = None
        e["awaiting_set"] = True
        ev = None
    else:
        kind, sfx = P.kind_and_suffix_of_file(f"events{suffix}.h5")
        e["args"] = ["--mode", mode, "--analysis", analysis, "--catalogs", catalogs, "--flux", "complete",
                     "--events-kind", kind, "--events-suffix", sfx, "--population-family", popfam,
                     "--set-label", label, "--realization", str(r), "--injections-lane", lane] + mc_args + \
                    (["--membership"] if membership else [])
        ev = P.events_path(r, kind, sfx)
    e["injections_lane"] = lane
    e["nsamp"] = nsamp
    e["injection_pool"] = pool
    if lane_ok is False:
        ev = None          # not ready: the set's lane is not validated for this realization yet
    hr = hold_reason(fam, s, lane) or hold_pool
    if hr:
        e["hold"] = hr
    return finish(e, stage, mode, free, spc_mc(spc_of(analysis, catalogs, free), nsamp, 1 + len(pool), frac), ev,
                  P.injections_path(r, popfam, lane), priority=priority)


def build_experiment(realizations):
    smap, info = experiment_set_map()
    st = {"E1": [], "E2": [], "E3": []}
    reused, awaiting = [], []
    for fam, f, s in PLANNED_SETS:
        rec = smap.get((fam, round(f, 4), s)) or {}
        suffix = rec.get("suffix")
        lane = rec.get("lane") or P.default_injection_lane("obslike" if fam == "obslike" else "corepop")
        stage = "E1" if fam == "peak" else "E2"
        inj_ok = rec.get("injections_exist")
        if suffix is None:
            awaiting.append({"family": fam, "f": f, "s": s, "label": P.set_name(fam, f, s)})
        for a in experiment_analyses(f, s):
            for r in realizations:
                # a stage-1 run is this run only when it uses the same injection lane
                eq = stage1_equivalent(a, fam, f, s, r) if P.present_lane(lane) else None
                if eq:
                    reused.append({"analysis": a, "set": P.set_name(fam, f, s), "realization": r, "run": eq})
                    continue
                mem = (r == MEMBERSHIP_REALIZATION and abs(f - 0.3) < 1e-9
                       and a in ("two_tracers_one_population", "two_tracers_two_populations"))
                # one GPU: the s = 0 grids first, then s = 1, then s = 2; within each, the f = 0.3 sets,
                # then f = 0.1 and 0.5, then the rest; peak family before observed-like
                prio = s * 100 + f_priority(f) * 10 + (0 if fam == "peak" else 1)
                lane_ok = None if not isinstance(inj_ok, dict) else bool(inj_ok.get(str(r)))
                st[stage].append(exp_entry(stage, "grid", a, fam, f, s, r, suffix, prio, membership=mem, lane=lane,
                                           lane_ok=lane_ok))
    # E3: the six-parameter single population on (f = 0.3, s = 2)
    rec = smap.get(("peak", 0.3, 2)) or {}
    for r in realizations:
        inj_ok = rec.get("injections_exist")
        st["E3"].append(exp_entry("E3", "nested", "single_population", "peak", 0.3, 2, r, rec.get("suffix"), 0,
                                  lane=rec.get("lane") or P.default_injection_lane("corepop"),
                                  lane_ok=None if not isinstance(inj_ok, dict) else bool(inj_ok.get(str(r)))))
    for k in ("E1", "E2"):
        st[k].sort(key=lambda e: (e["priority"], e["est_gpu_hours"], e["name"]))
    return st, reused, awaiting, info


def stage_block(k, runs):
    tot = sum(r["est_gpu_hours"] for r in runs)
    return {"stage": k, "title": TITLES[k], "n_runs": len(runs), "n_tasks": sum(len(r["parts"]) for r in runs),
            "est_calls": sum(r["est_calls"] for r in runs), "est_gpu_hours": round(tot, 2),
            "est_gpu_hours_existing_data": round(sum(r["est_gpu_hours"] for r in runs if r["data_exists"]), 2),
            "est_gpu_hours_ready": round(sum(r["est_gpu_hours"] for r in runs if r["data_exists"] and not r.get("hold")), 2),
            "n_runs_held": sum(1 for r in runs if r.get("hold")),
            "runs": runs}


def measured_overhead():
    """Load + redraw + bind + compile + checks per process, from the closure test's production grid
    (wall time of its session minus the time in likelihood calls)."""
    rj = P.A16 / "tests" / "closure_runs" / "production_a14_redraw" / "grid" / "result.json"
    if not rj.exists():
        return None
    r = json.loads(rj.read_text())
    ses = r.get("sessions") or []
    if not ses:
        return None
    return sum(s["wall_seconds"] - s["seconds_in_likelihood"] for s in ses) / 3600.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--realizations-5", default="100,101,102,103,105")
    ap.add_argument("--realizations-10", default="100,101,102,103,105,106,107,108,109,110")
    ap.add_argument("--realizations-3", default="100,101,102")
    ap.add_argument("--realizations-experiment", default=",".join(map(str, EXPERIMENT_REALIZATIONS)))
    ap.add_argument("--timing", default=None, help="JSON with measured spc_* to override COST")
    ap.add_argument("--out", default=str(P.RUNS / "RUNLIST.yaml"))
    ap.add_argument("--out-experiment", default=str(P.RUNS / "RUNLIST_experiment.yaml"))
    ap.add_argument("--sampler-8", choices=list(SAMPLER_8), default="rslice",
                    help="dynesty setting of the eight-parameter runs (test c); rslice = Analysis 13's guarded run")
    ap.add_argument("--calls-8", type=int, default=None, help="projected calls per eight-parameter run (test c)")
    ap.add_argument("--calls-v8", type=int, default=None,
                    help=f"calls per part of a stage-V8 run (default {V8_CALLS['calls_per_part']}: test c's projection)")
    ap.add_argument("--pool-sets-wanted", type=int, default=6,
                    help="all-AGN runs on the record's lane wait until the realization has this many injection sets")
    a = ap.parse_args()
    POOL_SETS_WANTED[0] = a.pool_sets_wanted
    SAMPLER_8_CHOICE[0] = a.sampler_8
    if a.calls_8:
        COST["calls_by_ndim"][8] = a.calls_8
    if a.calls_v8:
        V8_CALLS.update(calls_per_part=a.calls_v8, source=f"--calls-v8 {a.calls_v8}")
    measured = {}
    if a.timing and Path(a.timing).exists():
        t = json.loads(Path(a.timing).read_text())
        measured = {k: v for k, v in t.items() if k in TIMING_KEYS and isinstance(v, (int, float))}
        COST.update(measured)
    ov = measured_overhead()
    if ov is not None:
        COST["overhead_hours_per_process"] = round(max(ov, 0.01), 4)
        measured["overhead_hours_per_process"] = COST["overhead_hours_per_process"]
    lf = measured_lane_factors()
    for fam, d in lf.items():
        for cat, v in d.items():
            COST["lane_factor"][fam][cat] = v["factor"]
            measured[f"lane_factor.{fam}.{cat}"] = v["factor"]
    COST["lane_factor_measured"] = lf
    COST["measured"] = sorted(measured)
    r5 = [int(x) for x in a.realizations_5.split(",")]
    r10 = [int(x) for x in a.realizations_10.split(",")]
    r3 = [int(x) for x in a.realizations_3.split(",")]
    rx = [int(x) for x in a.realizations_experiment.split(",")]
    import yaml
    head = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "driver": "scripts/prod_driver.py (core e7c3007 venv); every run also gets the defaults: "
                      "--nsamp 8000 --half-variance-correction on --max-var 20 --zmax 1.10, nested --nlive 200 "
                      "--dlogz 0.1, unif up to six parameters, rslice for eight",
            "cost_model": COST, "sampler_8": a.sampler_8}
    smap, _ = experiment_set_map()
    hd = smap.get(("peak", 0.3, 1)) or {}
    st = build(r5, r10, r3, hd_lane=hd.get("lane"))
    if isinstance(hd.get("injections_exist"), dict) and not P.present_lane(hd.get("lane") or ""):
        # stage 4 runs on the broadened lane once experiment_sets.json reports it for the realization
        for e in st[4]:
            r = e["args"][e["args"].index("--realization") + 1]
            if str(r) in hd["injections_exist"] and not hd["injections_exist"][str(r)]:
                e["lane_not_validated"] = (f"data/experiment_sets.json reports no {hd['lane']} for realization {r} "
                                           "yet; rerun make_runlist.py when it does")
                e["data_exists"] = False
    st[5] = build_stage5(r5, smap.get(STAGE5_SET))
    vs = {"V8": build_v8(r5, hd_lane=hd.get("lane"))}
    if isinstance(hd.get("injections_exist"), dict) and not P.present_lane(hd.get("lane") or ""):
        for e in vs["V8"]:
            r = arg_of(e, "--realization")
            if str(r) in hd["injections_exist"] and not hd["injections_exist"][str(r)]:
                e["lane_not_validated"] = (f"data/experiment_sets.json reports no {hd['lane']} for realization {r} "
                                           "yet; rerun make_runlist.py when it does")
                e["data_exists"] = False
    vs["X"] = build_checks()
    xs, reused, awaiting, info = build_experiment(rx)
    unranked = []
    for stages in (st, xs, vs):
        for k in stages:
            stages[k] = [with_lane_rank(k, e) for e in stages[k]]
            unranked += [e["name"] for e in stages[k] if e["lane_rank"] is None]
    # stage V8 after stage 4 and 5, stage X last (the paper's classes.py groups RUNLIST.yaml runs by the part of the
    # name before /realization_ and keeps the first run's pattern: stage 1's and 4's must come first; its selectors
    # also skip the stages whose names start with V or X)
    doc = dict(head, realizations={"5": r5, "10": r10, "3": r3},
               stages=[stage_block(k, st[k]) for k in sorted(st)] + [stage_block("V8", vs["V8"]), stage_block("X", vs["X"])])
    write_atomic(a.out, yaml.safe_dump(doc, sort_keys=False, width=140))
    xdoc = dict(head, what="the short paper's experiment (archive/paper_short_independent/EXPERIMENT.md); runs that "
                           "production stage 1 already contains are listed under reused_from_stage_1, not run again",
                realizations=rx, event_sets=info, awaiting_sets=awaiting, reused_from_stage_1=reused,
                stages=[stage_block(k, xs[k]) for k in ("E1", "E2", "E3")])
    write_atomic(a.out_experiment, yaml.safe_dump(xdoc, sort_keys=False, width=140))
    for s in doc["stages"] + xdoc["stages"]:
        ranks = sorted({r["lane_rank"] for r in s["runs"] if r["lane_rank"] is not None})
        print(f"stage {s['stage']} ({s['title']}): {s['n_runs']} runs, {s['n_tasks']} tasks, "
              f"{s['est_calls']:,} calls, {s['est_gpu_hours']} GPU-h "
              f"({s['est_gpu_hours_existing_data']} on existing data, {s['est_gpu_hours_ready']} ready to run; "
              f"{s['n_runs_held']} runs held); lane ranks {ranks}")
    if unranked:
        print(f"[warning] {len(unranked)} runs match no LANE_RANKS row (the launcher uses their stage order), "
              f"e.g. {unranked[:3]}")
    print("stage 5 by class (per run: parts, calls per part, seconds per call, GPU-h; inputs checked inside the priors):")
    classes = {}
    for e in st[5]:
        classes.setdefault((e["analysis"], arg_of(e, "--catalogs")), []).append(e)
    for (an, cat), runs in classes.items():
        e = runs[0]
        print(f"  {an:28s} {cat:5s} {len(runs)} runs x {len(e['parts'])} parts, "
              f"{e['est_calls'] // len(e['parts']):,} calls per part, {e['est_seconds_per_call']} s/call, "
              f"{e['est_gpu_hours']} GPU-h per run, {sum(r['est_gpu_hours'] for r in runs):.1f} GPU-h; "
              f"inputs {check_inputs_in_priors(e['name'], e['free'], arg_of(e, '--population-family'), arg_of(e, '--events-kind'))}")
    print(f"experiment: {len(reused)} runs reused from stage 1; sets awaiting data: "
          f"{', '.join(x['label'] for x in awaiting) or 'none'}; sets file {info}")
    print(f"cost model: {json.dumps({k: v for k, v in COST.items() if k != 'calls_measured_reference'})}")
    print(f"-> {a.out}, {a.out_experiment}")


if __name__ == "__main__":
    main()
