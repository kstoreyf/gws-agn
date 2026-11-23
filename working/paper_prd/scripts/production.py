#!/usr/bin/env python
"""Production results of the PRD paper from Analysis 16's run summaries (the "production" source of
scripts/placeholder_results.py).

    PY=/hildafs/home/magana/tmp_ondemand_hildafs_phy230014p_symlink/magana/.conda/envs/jax/bin/python
    $PY working/paper_prd/scripts/production.py --write-map   # (re)write values/production_map.json from the run lists
    $PY working/paper_prd/scripts/production.py --status      # which keys and experiment macros are real, partial, waiting
    $PY working/paper_prd/scripts/production.py --report      # one line per paper-level count, from values.json

values/production_map.json (written by --write-map, data afterwards) maps each key of notes/PAPER_SKELETON.md
Sec. 6 whose runs are in analysis_16_production_runs/runs/RUNLIST.yaml or RUNLIST_experiment.yaml to its runs:
the run pattern ('{R}' = three-digit realization), the realizations, realization A (100), and per quantity the
summary field of realization A, the summaries combined, and the input whose credible level is quoted (see the
map's "_schema"). Keys of the paper without a run in the lists are listed under "not_in_run_lists".

A key's runs are the runs of its pattern in the production stages; stages V and X never feed a key, with two deliberate exceptions.
(21 Nov, build_values.STAGE_X_KEY_RUNS): KnownBoth also takes realizations 111-120, which exist only as stage-X runs
known_population/both_catalogs/complete/equal_populations/realization_NNN_nside32 because they are the same analysis at the same
settings as stage 1 (its arguments without --membership, plus --variant nside32); the map names them under runs_by_realization.
(23 Nov, build_values.STAGE_V_KEY_RUNS): HDHost reads stage V8's host_dependent_populations/both_catalogs/complete/host_dependent/
realization_NNN_unif_then_rslice for every realization (100, 101, 102, 103, 105): the eight-parameter model with the mixed sampler,
stage 4's arguments plus the sampler options and the variant only; its pattern is that one (stage_v_realizations, stage_v_why), and
stage 4's pure-rslice run of realization 100 feeds no key. ObsHost reads stage 5's own .../realization_NNN_unif_then_rslice runs.

A key turns REAL (final, black macros, final palette in the figures) only when every summary.json it names
exists (all realizations, and the runs it depends on: the reference of its Bayes factor, the known-density run
of its prior-induced region). A key with some but not all is PARTIAL: it stays exactly as its placeholder
(red), and the status lists it. A key whose real entry fails to compute stays red and is listed as ERROR.

How each part is computed (the functions of scripts/run_posteriors.py; until 17 Nov they lived in the letter's
scripts, which the paper no longer reads)
  realization A   median and 90% interval = summary.json "posterior" block of realization A (the run's own
                  summary, Akima-refined lattice); never the node-based summary of grid/result.json; the plotted
                  shape, 68% interval, 90% upper limit and level of the input from the run's marginal
                  (run_posteriors.marginal: grid ln L, or the kernel density of the nested samples)
  Comb            run_posteriors.combined: grids: the corrected ln L of the realizations summed on a common
                  mesh (run_posteriors.grid_marginal); nested runs: the product of the kernel densities of the
                  marginals (run_posteriors.chain_marginal_product); then median, 90% interval, upper limit and
                  the credible level of the input (placeholder_results.summarize_xp)
  CL              level_of_input of Comb: |2 F(input) - 1|, the input = the summaries' "truth" (or the
                  paper's input where the map says so)
  number densities (keys Free<limit>, LognGal and LognAGN, converted to Mpc^-3 per sample): Comb is the product of
                  kernel densities that respect each sample's own support (kde_product_boxed: the prior box of the
                  sampled parameter in h^3 Mpc^-3 shifted by the sample's 3 log10 h; reflected and truncated there).
                  A density the data do not bound on both sides (edge_bounds: posterior density at a prior edge
                  >= EDGE_FRACTION of its maximum in some realization) gets no combined median, interval, width or
                  credible level, only its 90% upper limit (Comb "limit_only"; the paper's rule for posteriors
                  against a prior edge or following the prior)
  RateRatio       R = [f/(1-f)] Z_gal/Z_AGN: complete catalogs Z_k = N_k (counts of the realization's survey
                  files); flux-limited catalogs with densities from counts Z_k = N_k + n_k V_miss,k (h-scaled
                  densities of the run, missing volume to the end of the redshift grid, z = 1.10, from the
                  Schechter fit the run used); densities free: per sample. Comb: the f_AGN combination mapped
                  with the geometric mean of the realizations' Z ratios (as the placeholder), or for nested runs
                  the product of the kernel densities of R
  LB              grids: Savage-Dickey at f_AGN = 0 on ln L summed over the realizations (the f = 0 column
                  spline-interpolated in H0); nested: (ln Z - ln Z_ref)/ln 10 of realization A, and Comb = the
                  sum over the realizations (each realization analysed on its own)
  joint           grids: run_posteriors.grid_joint (A: realization A; Comb: ln L summed); nested: a Gaussian kernel
                  density (0.3 sd, reflected at the prior edges) on a lattice, Comb the product over realizations
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_values as bv  # noqa: E402

PRD = HERE.parent
PMAP = PRD / "values" / "production_map.json"
A16 = bv.AN / "analysis_16_production_runs"
RUNS = A16 / "runs"
RUNLISTS = ("RUNLIST.yaml", "RUNLIST_experiment.yaml")
LN10 = math.log(10.0)
H_IN = 0.6774

# key -> (run pattern, quantities {Q: run parameter or derived}, what the key is in the paper)
PARAM = {"Hzero": "H0", "fAGN": "f_agn", "MuG": "mu_G", "SigmaG": "sigma_G", "MuChi": "mu_chi", "SigmaChi": "sigma_chi",
         "DmuG": "dmu_G", "DmuChi": "dmu_chi", "LognGal": "log10n_gal", "LognAGN": "log10n_agn"}
SIX = ["Hzero", "fAGN", "MuG", "SigmaG", "MuChi", "SigmaChi"]
FIVE = ["Hzero", "MuG", "SigmaG", "MuChi", "SigmaChi"]


def _key_table():
    T = {
        "KnownBoth": ("known_population/both_catalogs/complete/equal_populations/realization_{R}", ["Hzero", "fAGN", "RateRatio"]),
        "KnownGal": ("known_population/galaxy_catalog/complete/equal_populations/realization_{R}", ["Hzero"]),
        "KnownAGN": ("known_population/agn_catalog/complete/equal_populations/realization_{R}", ["Hzero"]),
        "KnownNone": ("known_population/no_catalog/complete/equal_populations/realization_{R}", ["Hzero"]),
        "AGNhosted": ("known_population/agn_catalog/complete/agn_hosted/realization_{R}_s32000_inj6", ["Hzero"]),
        "Galhosted": ("known_population/galaxy_catalog/complete/galaxy_hosted/realization_{R}", ["Hzero"]),
        "CtrlZero": ("two_tracers_one_population/both_catalogs/complete/peak_f0p00_s0/realization_{R}", ["Hzero", "fAGN"]),
        "CtrlTenth": ("two_tracers_one_population/both_catalogs/complete/peak_f0p10_s0/realization_{R}", ["Hzero", "fAGN"]),
        "SingleBoth": ("single_population/both_catalogs/complete/equal_populations/realization_{R}", SIX),
        "SingleNone": ("single_population/no_catalog/complete/equal_populations/realization_{R}", FIVE),
        "HDSingle": ("single_population/both_catalogs/complete/host_dependent/realization_{R}", SIX),
        # 23 Nov: the eight-parameter keys read the runs of the mixed sampler (unif, then rslice; the production sampler since 20 Nov).
        # HDHost's are stage V8's, the deliberate exception bv.STAGE_V_KEY_RUNS (see build_map); ObsHost's are stage 5's own
        "HDHost": ("host_dependent_populations/both_catalogs/complete/host_dependent/realization_{R}_unif_then_rslice", SIX + ["DmuG", "DmuChi"]),
        "HDNone": ("single_population/no_catalog/complete/host_dependent/realization_{R}", FIVE),
        "ObsSingle": ("single_population/both_catalogs/complete/host_dependent_obslike/realization_{R}", SIX),
        "ObsHost": ("host_dependent_populations/both_catalogs/complete/host_dependent_obslike/realization_{R}_unif_then_rslice", SIX + ["DmuG", "DmuChi"]),
        "ObsNone": ("single_population/no_catalog/complete/host_dependent_obslike/realization_{R}", FIVE),
        # 23 Nov (notes/MACRO_REQUESTS.md G1): the experiment's runs on the host-dependent sets (RUNLIST_experiment.yaml, stages E1-E3, which
        # build_map admits as it admits the control keys': only stages V and X are not production stages). Population assumed = the galaxy
        # class's mass and spin distribution for both classes, f_AGN inferred (two_tracers_one_population); the two populations at their
        # inputs (two_tracers_two_populations); the single population inferred on the set with twice the offsets (stage E3, nested)
        "HDAssumed": ("two_tracers_one_population/both_catalogs/complete/peak_f0p30_s1/realization_{R}", ["Hzero", "fAGN"]),
        "HDAssumedTwice": ("two_tracers_one_population/both_catalogs/complete/peak_f0p30_s2/realization_{R}", ["Hzero", "fAGN"]),
        "HDPops": ("two_tracers_two_populations/both_catalogs/complete/peak_f0p30_s1/realization_{R}", ["Hzero", "fAGN"]),
        "HDSingleTwice": ("single_population/both_catalogs/complete/peak_f0p30_s2/realization_{R}", SIX),
        "ObsAssumed": ("two_tracers_one_population/both_catalogs/complete/obslike_f0p30_s1/realization_{R}", ["Hzero", "fAGN"]),
        "ObsPops": ("two_tracers_two_populations/both_catalogs/complete/obslike_f0p30_s1/realization_{R}", ["Hzero", "fAGN"]),
    }
    for rk, w, m in bv.RUNGS:
        T[f"Flux{w}"] = (f"known_population/both_catalogs/flux_limit_{m}_densities_from_counts/equal_populations/realization_{{R}}", ["Hzero", "fAGN", "RateRatio"])
        T[f"Free{w}"] = (f"free_densities/both_catalogs/flux_limit_{m}/equal_populations/realization_{{R}}", ["Hzero", "fAGN", "LognGal", "LognAGN", "RateRatio"])
    return T


KEY_RUNS = _key_table()
# keys of the paper that have no run in either list (none since Stage 5 entered RUNLIST.yaml on 17 Nov)
PAPER_KEYS_WITHOUT_RUNS = {}
# where the run differs from the key as the paper defines it (notes/PAPER_SKELETON.md Sec. 6, runs.txt)
SET_NOTES = {
    "CtrlZero": "the experiment's f = 0 set (peak_f0p00_s0: every merger hosted by a galaxy, the events of Galhosted) analysed with "
                "both catalogs and f_AGN free (two_tracers_one_population = population known, both catalogs), on the experiment's "
                "injection lane of the peak family (injections_targeted_peak.h5, 2.5e8 draws) instead of the realization's record lane; "
                "the experiment's five realizations",
    "CtrlTenth": "the experiment's f = 0.10 set (peak_f0p10_s0, events_fagn0p10_corepop.h5) analysed with both catalogs and f_AGN "
                 "free (two_tracers_one_population), on the peak family's injection lane (injections_targeted_peak.h5, 2.5e8 draws); "
                 "the experiment's five realizations",
    # 23 Nov: the host-dependent sets of the experiment
    **{k: (f"the experiment's set {lab} ({ev}) analysed {how}, on the {fam} family's injection lane ({lane}); the experiment's five realizations")
       for k, lab, ev, how, fam, lane in (
           ("HDAssumed", "peak_f0p30_s1", "the events of stage 4, AGN offsets 5 Msun and 0.10", "with both catalogs, f_AGN free and the galaxy class's population assumed for both classes (two_tracers_one_population)", "peak", "injections_targeted_peak.h5"),
           ("HDAssumedTwice", "peak_f0p30_s2", "AGN offsets 10 Msun and 0.20", "with both catalogs, f_AGN free and the galaxy class's population assumed for both classes (two_tracers_one_population)", "peak", "injections_targeted_peak.h5"),
           ("HDPops", "peak_f0p30_s1", "the events of stage 4, AGN offsets 5 Msun and 0.10", "with both catalogs, f_AGN free and each class's population at its input (two_tracers_two_populations)", "peak", "injections_targeted_peak.h5"),
           ("HDSingleTwice", "peak_f0p30_s2", "AGN offsets 10 Msun and 0.20", "with both catalogs and one population inferred for both classes (single_population, stage E3)", "peak", "injections_targeted_peak.h5"),
           ("ObsAssumed", "obslike_f0p30_s1", "the events of stage 5", "with both catalogs, f_AGN free and the galaxy class's population assumed for both classes (two_tracers_one_population)", "observed-like", "injections_targeted_obslike.h5"),
           ("ObsPops", "obslike_f0p30_s1", "the events of stage 5", "with both catalogs, f_AGN free and each class's population at its input (two_tracers_two_populations)", "observed-like", "injections_targeted_obslike.h5"))},
}
FREE_UNITS_NOTE = ("the run samples log10 n in h^3 Mpc^-3 (prior U[-4, -1] and U[-6, -4] in those units); the macros <Key>Logn...[Comb][...] "
                   "are in Mpc^-3, converted per sample with its own H0 (log10 n + 3 log10 h); <Key>Logn...ULh, ...PriorULh and the "
                   "joint posterior of Fig. 4(a) are in h^3 Mpc^-3, the sampled parameter itself")


# ============================================================================================= #
# the run lists and the map
# ============================================================================================= #
def runlist_runs():
    """run name -> run-list record (with 'runlist' and 'stage')."""
    import yaml
    out = {}
    for f in RUNLISTS:
        p = RUNS / f
        if not p.exists():
            continue
        d = yaml.safe_load(p.read_text())
        for st in d["stages"]:
            for r in st["runs"]:
                out[r["name"]] = dict(r, runlist=f, stage=str(st["stage"]))
        for r in d.get("reused_from_stage_1") or []:
            out.setdefault(r["run"], {"name": r["run"], "runlist": f, "stage": "1 (reused)"})
    return out


def _pattern_re(pat):
    return re.compile("^" + re.escape(pat).replace(re.escape("{R}"), r"(\d{3})") + "$")


def _settings(key, recs):
    """One line: how the runs of a key differ from the production convention (8000 samples, the
    realization's own record injection set, no pooled sets) and from the key as the paper defines it."""
    ns = sorted({r.get("nsamp", 8000) for r in recs})
    lanes = sorted({r.get("injections_lane", "injections/injections_targeted.h5") for r in recs})
    pools = sorted({len(r.get("injection_pool") or []) for r in recs})
    diff = []
    if ns != [8000]:
        diff.append(f"{'/'.join(map(str, ns))} posterior samples per merger (Analysis 14's setting for mergers all hosted by AGN; the convention is 8000)")
    if pools != [0]:
        diff.append(f"the realization's injection set pooled with {max(pools)} more drawn from the same proposal (Analysis 14's sets; one set of {max(pools) + 1})")
    if lanes != ["injections/injections_targeted.h5"] and key not in SET_NOTES:
        if lanes == ["injections/injections_targeted_peak.h5"]:
            diff.append("the peak family's injection lane (injections_targeted_peak.h5, 2.5e8 draws, as the experiment of Sec. IV B) instead of "
                        "the realization's record lane")
        else:
            diff.append(f"injection lane {', '.join(lanes)}")
    if key in SET_NOTES:
        diff.insert(0, SET_NOTES[key])
    if key.startswith("Free"):
        diff.append(FREE_UNITS_NOTE)
    return "; ".join(diff) if diff else "as the key (production convention: 8000 samples per merger, the realization's own injection set)"


def build_map():
    """The map from the run lists."""
    runs = runlist_runs()
    keys, missing = {}, {}
    idx1 = bv.runlist_index(RUNS / RUNLISTS[0])
    for key, (pat, quants) in KEY_RUNS.items():
        rx = _pattern_re(pat)
        # the runs of the key's pattern in the production stages (23 Nov: explicit; stages V and X never feed a key, bv.production_stage)
        recs = {int(m.group(1)): r for n, r in runs.items() if bv.production_stage(r["stage"]) and (m := rx.match(n))}
        # The second deliberate exception (23 Nov, bv.STAGE_V_KEY_RUNS): key HDHost reads stage V8's runs realization_NNN_unif_then_rslice
        # for every realization (the mixed sampler, the production sampler of the eight-parameter model since its comparison with rslice
        # on realization 100), each with stage 4's arguments plus the sampler options and the variant only; stage 4's pure-rslice run of
        # realization 100 feeds no key. The pattern is the V8 one, so the realizations come from the exception alone.
        vexc = bv.stage_v_key_runs(idx1, key)
        for R, n in vexc.items():
            recs[R] = runs[n]
        if not recs:
            missing[key] = f"no run named {pat} in {' or '.join(RUNLISTS)}"
            continue
        # The one deliberate exception to "stages V and X never feed a key" (bv.STAGE_X_KEY_RUNS): KnownBoth also takes realizations
        # 111-120, run in stage X as realization_NNN_nside32 because they are the same analysis at the same settings as stage 1 (stage
        # 1's arguments without --membership, plus the variant tag); named per realization (runs_by_realization). Nothing else of stage X.
        extra = bv.stage_x_key_runs(idx1, key)
        for R, n in extra.items():
            recs[R] = runs[n]
        reals = sorted(recs)
        rA = 100 if 100 in reals else reals[0]
        R0 = recs[rA]
        mode = R0.get("mode") or ("grid" if "grid" in (R0.get("parts") or []) else "nested")
        sumpat = pat + "/summary.json"
        A = sumpat.format(R=f"{rA:03d}")
        Q = {}
        for q in quants:
            if q == "RateRatio":
                zr = "free_densities" if key.startswith("Free") else ("counts_and_missing_volume" if key.startswith("Flux") else "catalog_counts")
                Q[q] = {"derived": "rate_ratio", "z_ratio": zr, "A": {"summary": A, "from": "posterior.f_agn" + (" and the samples" if zr == "free_densities" else "")},
                        "Comb": {"summaries": sumpat, "realizations": reals, "combine": "f_AGN combination mapped with the geometric-mean Z ratio" if mode == "grid" else "product of the kernel densities of R"},
                        "CL": {"of": "Comb", "input": "paper: MockRateRatio = [0.30/0.70] N_gal/N_AGN of realization A"}}
                continue
            par = PARAM[q]
            no_input = key in ("HDSingle", "HDNone", "ObsSingle", "ObsNone", "HDSingleTwice") and q in ("MuG", "SigmaG", "MuChi", "SigmaChi")
            conv = key.startswith("Free") and q in ("LognGal", "LognAGN")
            Q[q] = {"param": par, "A": {"summary": A, "field": f"posterior.{par}"},
                    "Comb": {"summaries": sumpat, "realizations": reals,
                             "combine": "grid ln L summed (run_posteriors.grid_marginal)" if mode == "grid" else "product of the kernel densities (run_posteriors.chain_marginal_product)"},
                    "CL": {"of": "Comb", "input": None if no_input else f"truth.{par} of the summaries"}}
            if no_input:
                Q[q]["CL"]["why"] = "one population for a mixture: no single input (as the placeholder)"
            if conv:
                Q[q]["units"] = "converted to Mpc^-3 per sample: log10 n + 3 log10(H0/100)"
                Q[q]["Comb"]["combine"] = "product of the kernel densities of the converted samples"
        if extra:
            for qq in Q.values():
                qq["Comb"]["summaries_by_realization"] = {str(R): f"{n}/summary.json" for R, n in sorted(extra.items())}
        E = {"run": pat, "runlist": sorted({r["runlist"] for r in recs.values()})[0], "stage": "/".join(sorted({r["stage"] for r in recs.values()})),
             "mode": mode, "realizations": reals, "realization_A": rA, "settings": _settings(key, list(recs.values())), "quantities": Q}
        if extra:
            E["runs_by_realization"] = {str(R): n for R, n in sorted(extra.items())}
            E["stage_x_realizations"] = sorted(extra)
            E["stage_x_why"] = ("the one deliberate exception to 'stages V and X never feed a key' (build_values.STAGE_X_KEY_RUNS): realizations "
                                f"{sorted(extra)} exist only as stage-X runs {bv.STAGE_X_KEY_RUNS[key]['pattern']}, the same analysis at the same "
                                "settings as stage 1 (stage 1's arguments without --membership, plus --variant "
                                f"{bv.STAGE_X_KEY_RUNS[key]['variant']}), run to test whether the H0 offset of the first realizations is a fluctuation")
        if vexc:
            V = bv.STAGE_V_KEY_RUNS[key]
            E["stage_v_realizations"] = sorted(vexc)
            E["stage_v_why"] = ("the second deliberate exception to 'stages V and X never feed a key' (build_values.STAGE_V_KEY_RUNS): every "
                                f"realization is read from stage {V['stage']}'s run {V['pattern']} (the eight-parameter model sampled uniformly while "
                                "that is cheap and by rslice after, the production sampler since its comparison with rslice on realization 100, "
                                "analysis_16 tests/results/compare_r100_eight_parameter.json), with the arguments of stage 4's run "
                                f"{V['base']} plus only the sampler options and --variant {V['variant']}; stage 4's pure-rslice run of realization "
                                "100 is the sampler reference and feeds no key")
        if key in ("KnownBoth", "CtrlZero", "CtrlTenth"):
            E["joint"] = {"params": ["H0", "f_agn"], "parts": ["A", "Comb"], "how": "run_posteriors.grid_joint"}
            E["LB"] = {"kind": "savage_dickey_f0", "reference": "galaxy hosts only (f_AGN = 0), Savage-Dickey on the same grids (Comb: ln L summed), f_AGN prior [0, 1]"}
        elif key in ("SingleBoth", "SingleNone"):
            E["joint"] = {"params": ["H0", "mu_G"], "parts": ["A", "Comb"], "how": "kernel density on a lattice; Comb the product"}
        elif key in ("HDHost", "ObsHost"):
            E["joint"] = {"params": ["dmu_G", "dmu_chi"], "parts": ["A", "Comb"], "how": "kernel density on a lattice; Comb the product"}
            rk_ = key.replace("Host", "Single")
            ref = KEY_RUNS[rk_][0]
            E["LB"] = {"kind": "evidence_ratio", "reference_run": ref, "reference_key": rk_,
                       "reference": "the single-population analysis of the same events, same priors (Table I)",
                       "A": "(ln Z - ln Z_ref)/ln 10 of realization A (summary ln_evidence)",
                       "Comb": "sum over the realizations of (ln Z - ln Z_ref)/ln 10 (each realization analysed on its own)"}
        elif key.startswith("Free"):
            w = key[4:]
            E["joint"] = {"params": ["f_agn", "log10n_agn"], "parts": ["A", "prior_induced"], "how": "kernel density on a lattice (A); "
                          "prior draws weighted by the known-density likelihood of realization A at the same flux limit (prior_induced)"}
            E["prior_induced_from"] = KEY_RUNS[f"Flux{w}"][0].format(R=f"{rA:03d}")
        keys[key] = E
    for k, why in PAPER_KEYS_WITHOUT_RUNS.items():
        missing.setdefault(k, why)
    import classes
    import density
    import skyres
    sky = skyres.map_block(runs)
    # the sky-resolution pair over the twenty realizations takes exactly key KnownBoth's runs
    tw, kb = sky.get("pair_twenty") or {}, keys.get("KnownBoth") or {}
    assert not tw or (tw["realizations_wanted"] == kb.get("realizations") and tw.get("partners", {}) == kb.get("runs_by_realization", {})), (tw, kb)
    return {"_schema": SCHEMA, "runs_root": bv.rel(RUNS), "keys": keys, "not_in_run_lists": missing, "classes": classes.map_block(runs),
            "density_response": density.map_block(runs), "sky_resolution": sky, "obslike_mc": obslike_mc_block(idx1, keys)}


def obslike_mc_block(idx1, keys):
    """23 Nov (bv.STAGE_X_MC_RUNS; the macros Obs<Assumed|Pops>MC..., ObsAssumedMCShiftHzero..., ObsMCOneHzero[Prod]): realization 106's
    grids of keys ObsAssumed and ObsPops at 32000 samples per merger (stage X of RUNLIST.yaml), each with its production partner (the
    key's own run of that realization), for a combination over the key's realizations with that one grid replaced. Feeds no key."""
    idx_exp = bv.runlist_index(RUNS / RUNLISTS[1]) if (RUNS / RUNLISTS[1]).exists() else {}
    sel = bv.stage_x_mc_runs(idx1, idx_exp)
    R = bv.STAGE_X_MC["realization"]
    out = {"_what": ("the runs of the Obs<Assumed|Pops>MC... macros (build_values.build_obslike_mc): per key, realization "
                     f"{R}'s grid at {bv.STAGE_X_MC['nsamp']} samples per merger (run, stage X) in place of its production run (partner) "
                     "in a combination over the key's realizations (keys.<Key>.realizations); feeds no key"),
           "realization": R, "nsamp": bv.STAGE_X_MC["nsamp"], "variant": bv.STAGE_X_MC["variant"], "stage": bv.STAGE_X_MC["stage"],
           "runlist": RUNLISTS[0], "runs": {}}
    why = {}
    for k in bv.STAGE_X_MC_RUNS:
        if k in sel and k in keys and R in keys[k]["realizations"] and sel[k][1] == key_run(keys[k], R):
            out["runs"][k] = {"run": sel[k][0], "partner": sel[k][1], "differs_by": f"--nsamp {bv.STAGE_X_MC['nsamp']} --variant {bv.STAGE_X_MC['variant']}",
                              "realizations": keys[k]["realizations"]}
        else:
            why[k] = (f"no stage-X run {bv.STAGE_X_MC_RUNS[k]['pattern'].format(R=f'{R:03d}')} with the arguments of the key's run of realization "
                      f"{R} plus --nsamp {bv.STAGE_X_MC['nsamp']} --variant {bv.STAGE_X_MC['variant']} only, or the key lacks realization {R}")
    if why:
        out["not_in_run_lists"] = why
    return out


SCHEMA = [
    "production_map.json: which production runs (analysis_16_production_runs) fill each key of notes/PAPER_SKELETON.md Sec. 6.",
    "Written by `scripts/production.py --write-map` from runs/RUNLIST.yaml and runs/RUNLIST_experiment.yaml; read by scripts/placeholder_results.py.",
    "keys.<Key>: run (pattern under runs_root, '{R}' = three-digit realization), runlist, stage, mode (grid | nested), realizations (all of them",
    "  are combined), realization_A, settings (how the runs differ from the key as the paper defines it, or 'as the key'), quantities, joint, LB.",
    "  21 Nov: runs_by_realization {realization: run name} where a realization's run is not the pattern: KnownBoth's realizations 111-120,",
    "  stage X's realization_NNN_nside32 (stage_x_realizations, stage_x_why; build_values.STAGE_X_KEY_RUNS, the one deliberate exception to",
    "  'stages V and X never feed a key': the same analysis at the same settings as stage 1); quantities.<Q>.Comb.summaries_by_realization.",
    "  23 Nov: stage_v_realizations, stage_v_why (key HDHost; build_values.STAGE_V_KEY_RUNS, the second deliberate exception): the key's",
    "  pattern is stage V8's realization_{R}_unif_then_rslice (the mixed sampler) for every realization; the key's stage is then V8.",
    "quantities.<Q>: param (the run's parameter) or derived; A = the summary field of realization A (median and 90% interval printed as is);",
    "  Comb = the summaries of every realization, combined (grids: corrected ln L summed; nested: product of kernel densities);",
    "  CL = the credible level of the input in Comb (input: the summaries' truth, or null when the paper quotes none).",
    "A key is real when every summary named exists (all realizations and the runs it depends on); otherwise it stays the placeholder of",
    "  placeholder_results.py (partial when some exist). Keys of the paper with no run in the lists: not_in_run_lists.",
    "classes: the macros of Sec. IV B and of the experiment's numbers elsewhere (scripts/classes.py, notes/PAPER_SKELETON.md Sec. 9):",
    "  classes.<macro>: run (pattern, '{R}' = three-digit realization), runlist, stage, realizations, how (mean_hw | factor_hw | membership |",
    "  interior); derived macros (cost, share) name the macros they are computed from. A macro is real when every summary it names exists.",
    "density_response: the runs of the macros Flux<limit><Half|Double>... (scripts/density.py): arms.<limit><factor>: run (the factor-x",
    "  runs, '{R}' = three-digit realization), unit_run (their unit-factor partners, the runs of key unit_key), factor, realizations,",
    "  realization_A, stage; validation and check stages of the run list (names starting with 'V' or 'X') excluded. Real when every summary",
    "  named exists.",
    "sky_resolution: the runs of SkyResShift<Hzero|fAGN> (scripts/skyres.py): pair.run (the nside-64 run, stage X, '{R}' = three-digit",
    "  realization), pair.partner (its nside-32 partner, key KnownBoth), realization; mc_pair: the same two grids at 32000 samples per",
    "  merger and six pooled injection sets (both stage X; macros SkyResMC...). Each pair is real when both its summaries exist.",
    "  <pair>.realizations: every realization whose pair is in the run list (of realizations_wanted, the realizations with six injection",
    "  sets; stages per realization), for the means and scatters over them (SkyRes<P>Mean..., SkyRes<P>Sd..., SkyResNreal; real when every",
    "  wanted realization has both summaries); mc_pair.n_injection_sets_by_run: 1 + the --pool-injections of each run (SkyResMCNsets).",
    "  21 Nov: pair_all (the pair at the production settings on every realization of key KnownBoth in stage 1; its macros SkyResTen...,",
    "  SkyResOffsetHzero... were removed on 23 Nov, main.tex used none: the record feeds no macro)",
    "  and agn_pair (key AGNhosted's runs against stage X's ..._nside64_s32000_inj6; macros SkyResAGN...): run, partner, partner_key,",
    "  realizations (whose nside-64 run is in stage X), realizations_wanted (those of the partner's production runs), stages.",
    "  21 Nov, later: pair_twenty (the pair at the production settings on the twenty realizations of key KnownBoth; partners = the key's",
    "  runs_by_realization; macros SkyResAll...) and pair_half (host class unknown, half of the mergers in AGN: RUNLIST_experiment.yaml's",
    "  peak_f0p50_s0 runs of stage E1 against stage X's ..._nside64; partner_runlist; macros SkyResHalf...).",
    "classes (21 Nov): ClsDiff... records are derived (fn paired_mean | paired_sd | paired_npos) from the two half-width records",
    "  HWUnknown<f>, HWKnown<f> of the same f (23 Nov: the Cls...Comb/CL records of how 'comb' are removed with their macros).",
    "density_response.units_physical: the two stage-X runs of realization A with both densities given in Mpc^-3 (macros",
    "  FluxEighteenPhys..., scripts/density.py); never a production run of any key.",
    "obslike_mc (23 Nov; build_values.STAGE_X_MC_RUNS): runs.<Key> (ObsAssumed, ObsPops): run (realization 106 at 32000 samples per",
    "  merger, stage X), partner (the key's production run of that realization; the run's arguments are the partner's plus --nsamp 32000",
    "  --variant s32000 only), realizations (the key's); macros Obs<Assumed|Pops>MC..., ObsAssumedMCShiftHzero..., ObsMCOneHzero[Prod]",
    "  combine the key's realizations with that grid replaced. Feeds no key; real when every summary named exists.",
    "Old form (still read): <Key>: path of a JSON file with the schema of one placeholder entry.",
]


def load_map():
    return json.loads(PMAP.read_text()) if PMAP.exists() else {}


def write_map():
    m = build_map()
    PMAP.write_text(json.dumps(m, indent=1) + "\n")
    return m


# ============================================================================================= #
# status
# ============================================================================================= #
def run_dir(pat, r):
    return RUNS / pat.format(R=f"{r:03d}")


def key_run(spec, r):
    """The run name of realization r of a key: its pattern, or the name the map gives that realization (runs_by_realization: the
    deliberate stage-X runs of KnownBoth, realizations 111-120, bv.STAGE_X_KEY_RUNS)."""
    return (spec.get("runs_by_realization") or {}).get(str(r)) or spec["run"].format(R=f"{r:03d}")


def key_dir(spec, r):
    return RUNS / key_run(spec, r)


def needed(spec, pm=None):
    """{run dir: role} of every run a key needs."""
    out = {key_dir(spec, r): "run" for r in spec["realizations"]}
    lb = spec.get("LB") or {}
    if lb.get("reference_run"):
        for r in spec["realizations"]:
            out[run_dir(lb["reference_run"], r)] = f"reference of the Bayes factor ({lb.get('reference_key')})"
    if spec.get("prior_induced_from"):
        out[RUNS / spec["prior_induced_from"]] = "known-density run of the prior-induced region"
    return out


def key_status(key, spec):
    need = needed(spec)
    have = [d for d in need if (d / "summary.json").exists()]
    own = [r for r in spec["realizations"] if (key_dir(spec, r) / "summary.json").exists()]
    st = "real" if len(have) == len(need) else ("partial" if have else "waiting")
    return {"state": st, "n_have": len(have), "n_need": len(need), "realizations_have": own, "realizations_want": spec["realizations"],
            "missing": sorted(bv.rel(d) for d in need if d not in have), "stage": spec["stage"], "runlist": spec["runlist"]}


# ============================================================================================= #
# computation (scripts/run_posteriors.py)
# ============================================================================================= #
def rp():
    import run_posteriors as RP  # noqa: E402  (imports placeholder_results lazily and prod_common)
    return RP


def _phr():
    import placeholder_results as phr
    return phr


def jread(p):
    return json.loads(Path(p).read_text())


def _cdf(x, p):
    c = np.concatenate([[0.0], np.cumsum(0.5 * (p[1:] + p[:-1]) * np.diff(x))])
    return c / c[-1]


def _summ(x, p, x0, exact=None):
    """placeholder_results.summarize_xp, with the median and 90% interval replaced by `exact` (the run's own
    summary) when given; the recomputed ones are kept under 'recomputed'."""
    s = _phr().summarize_xp(np.asarray(x, float), np.asarray(p, float), x0)
    if exact is not None:
        s["recomputed"] = {"median": s["median"], "ci90": list(s["ci90"])}
        s["median"], s["ci90"] = float(exact["median"]), [float(v) for v in exact["ci90"]]
    return s


def _is_grid(d):
    return jread(d / "config.json")["mode"] == "grid"


def marg_A(dA, par):
    return rp().marginal([dA], par)


def comb(key, q, spec, par, x0="truth"):
    """One parameter combined over the key's realizations (run_posteriors.combined)."""
    dirs = [key_dir(spec, r) for r in spec["realizations"]]
    missing = [d for d in dirs if not (d / "summary.json").exists()]
    assert not missing, (key, q, missing)
    return rp().combined(dirs, par, x0="truth" if x0 == "truth" else x0)


def kde1(s, lo, hi, n=4001):
    return rp().kde1(np.asarray(s, float), lo, hi, n=n)          # the kernel of run_posteriors.chain_marginal_product


def kde_product(sample_sets, lo, hi, n=4001):
    logp = None
    for s in sample_sets:
        x, p, _ = kde1(s, lo, hi, n)
        lp = np.log(np.maximum(p, 1e-300))
        logp = lp if logp is None else logp + lp
    p = np.exp(logp - logp.max())
    return x, p / np.trapz(p, x)


# ---------------------------------------------------------------- number densities with both densities free
# A side of a number density's posterior is bounded by the data when, in every realization, the posterior density of the run's
# own sampled parameter (log10 n in h^3 Mpc^-3, flat prior on the box of Table I) at that prior edge is below EDGE_FRACTION of its
# maximum (a Gaussian falls to 1% at 3.0 standard deviations). A density the data do not bound on both sides gets no combined
# interval, width or credible level (the paper's rule for posteriors against a prior edge or following the prior: a limit only);
# its combination keeps the 90% upper limit and the shape.
EDGE_FRACTION = 0.01


def edge_bounds(samples_h, prior, n=2001):
    """{"lower"|"upper": {"edge_over_peak": {realization: ratio}, "bounded": bool}, "both_bounded", "rule"}: the kernel density of
    each realization's equal-weight samples of the sampled parameter (run_posteriors.kde1: 0.3 sd, reflected at the prior edges,
    the bounded-support kernel of a flat prior on a box) at each prior edge over its maximum."""
    lo, hi = prior
    rat = {"lower": {}, "upper": {}}
    for r, s in samples_h.items():
        _, p, _ = kde1(s, lo, hi, n)
        rat["lower"][int(r)] = float(p[0] / p.max())
        rat["upper"][int(r)] = float(p[-1] / p.max())
    out = {side: {"edge_over_peak": v, "bounded": bool(all(x < EDGE_FRACTION for x in v.values()))} for side, v in rat.items()}
    out["both_bounded"] = out["lower"]["bounded"] and out["upper"]["bounded"]
    out["prior_h3Mpc-3"] = [float(lo), float(hi)]
    out["rule"] = (f"a side is bounded by the data when the posterior density of the sampled parameter (h^3 Mpc^-3) at that prior edge is "
                   f"below {EDGE_FRACTION:g} of its maximum in every realization (kernel density reflected at the prior edges, 0.3 sd)")
    return out


def kde_product_boxed(sets, prior_h, bw=0.3, n=4001, chunk=400):
    """Product over realizations of kernel densities of log10 n in Mpc^-3, y = x + 3 log10(H0/100) per sample, that respect each
    sample's own support: the prior box of the sampled parameter x (h^3 Mpc^-3) carried to Mpc^-3 by the sample's own H0. Each
    sample's Gaussian kernel (bw sd of y) is reflected at, and truncated to, [lo + s, hi + s], s = 3 log10(H0/100) of the sample, so
    every kernel keeps unit mass on its own support. sets: {realization: (x [h^3 Mpc^-3], H0)}. -> (y lattice, normalised density)."""
    lo, hi = prior_h
    shifts = [3 * np.log10(np.asarray(H, float) / 100.0) for _, H in sets.values()]
    y = np.linspace(lo + min(s.min() for s in shifts), hi + max(s.max() for s in shifts), n)
    logp = np.zeros(n)
    for (xh, H), sft in zip(sets.values(), shifts):
        xh = np.asarray(xh, float)
        yy = xh + sft
        h = bw * float(np.std(yy))
        p = np.zeros(n)
        for c in np.array_split(np.arange(len(yy)), max(1, len(yy) // chunk)):
            el, eh = lo + sft[c], hi + sft[c]
            k = np.zeros((n, len(c)))
            for m in (yy[c], 2 * el - yy[c], 2 * eh - yy[c]):
                k += np.exp(-0.5 * ((y[:, None] - m[None, :]) / h) ** 2)
            k *= (y[:, None] >= el[None, :]) & (y[:, None] <= eh[None, :])
            p += k.sum(axis=1)
        logp += np.log(np.maximum(p, 1e-300))
    p = np.exp(logp - logp.max())
    return y, p / np.trapz(p, y)


def kde2(a, b, xs, ys, ra, rb, bw=0.3):
    """2-D Gaussian kernel density (bandwidth bw sd per axis, reflected at the prior edges ra, rb) on the
    lattice xs x ys."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ha, hb = bw * np.std(a), bw * np.std(b)

    def kmat(v, g, h, rng):
        K = np.zeros((len(v), len(g)))
        for vv in (v, 2 * rng[0] - v, 2 * rng[1] - v):
            K += np.exp(-0.5 * ((g[None, :] - vv[:, None]) / h) ** 2)
        return K
    P = kmat(a, xs, ha, ra).T @ kmat(b, ys, hb, rb)
    return P / np.trapz(np.trapz(P, ys, axis=1), xs)


def _lattice(samples, rng, n, pad=0.25):
    lo, hi = np.quantile(samples, [0.0005, 0.9995])
    w = hi - lo
    return np.linspace(max(rng[0], lo - pad * w), min(rng[1], hi + pad * w), n)


def corr_lattice(x, y, P):
    w = P / P.sum()
    X, Y = np.meshgrid(x, y, indexing="ij")
    mx, my = (w * X).sum(), (w * Y).sum()
    cxy = (w * (X - mx) * (Y - my)).sum()
    return float(cxy / math.sqrt((w * (X - mx) ** 2).sum() * (w * (Y - my) ** 2).sum()))


def samples_of(d, params):
    return rp().chain_samples(d, params)


def phys_logn(s_logn_h, s_H0):
    return np.asarray(s_logn_h) + 3 * np.log10(np.asarray(s_H0) / 100.0)


# ---------------------------------------------------------------- the rate per host
def z_ratio(key, spec, r, sm=None):
    """Z_gal/Z_AGN of realization r (complete catalogs: the counts; flux-limited, densities from counts:
    N_k + n_k V_miss,k with the run's h-scaled densities and the missing volume to z = 1.10)."""
    zr = spec["quantities"]["RateRatio"]["z_ratio"]
    if zr == "catalog_counts":
        c = bv.survey_counts(r)
        return c["gal"] / c["agn"]
    rk = {f"Flux{w}": k for k, w, _ in bv.RUNGS}[key]
    sm = sm or jread(key_dir(spec, r) / "summary.json")
    Z = {}
    for cls in ("gal", "agn"):
        d = sm["densities"][cls]
        assert d["units"] == "h_scaled", (key, r, d)
        rec = bv.norm_cfg(r)[rk][cls]
        Z[cls] = rec["N"] + 10 ** d["log10n_used"] * rec["Vp"]
    return Z["gal"] / Z["agn"]


def free_Z(key, r, lg_h, la_h):
    rk = {f"Free{w}": k for k, w, _ in bv.RUNGS}[key]
    NC = bv.norm_cfg(r)[rk]
    return (NC["gal"]["N"] + 10 ** np.asarray(lg_h) * NC["gal"]["Vp"]) / (NC["agn"]["N"] + 10 ** np.asarray(la_h) * NC["agn"]["Vp"])


def rr_from_f(x, p, exact_f, ratio, x0):
    """R = odds(f) * ratio from a 1-D f posterior (x, p): quantiles mapped one to one (exact_f: the f summary
    whose median and 90% interval are mapped), shape by the change of variables."""
    odds = lambda f: f / (1.0 - f)
    m = (x > 1e-9) & (x < 1 - 1e-9)
    r = odds(x[m]) * ratio
    pr = p[m] * (1 - x[m]) ** 2 / ratio
    rr = np.linspace(r.min(), min(r.max(), 400.0), 20001)
    s = _phr().summarize_xp(rr, np.interp(rr, r, pr), x0)
    c = _cdf(x, p)
    q = lambda u: float(np.interp(u, c, x))
    s["ci68"] = [odds(q(0.16)) * ratio, odds(q(0.84)) * ratio]
    s["ul90"] = odds(q(0.90)) * ratio
    s["median"] = odds(exact_f["median"]) * ratio
    s["ci90"] = [odds(v) * ratio for v in exact_f["ci90"]]
    F = float(np.interp(x0 / (x0 + ratio), x, c))       # P(R < x0) = P(f < f(x0))
    s["level_of_input"] = abs(2 * F - 1)
    s["input"], s["z_ratio"] = x0, ratio
    return s


# ---------------------------------------------------------------- Savage-Dickey at f = 0 on grids
def ln_int_grid_f0(dirs):
    """ln of int exp(sum ln L(H0, f = 0)) dH0 over the prior range: each run's f = 0 column (the fine pass where it
    contains f = 0, the coarse pass elsewhere) as a cubic spline of ln L in H0, summed on a dense H0 mesh."""
    from scipy.interpolate import CubicSpline
    LR = rp()
    H = np.linspace(50.0, 100.0, 20001)
    tot = np.zeros_like(H)
    for d in dirs:
        parts = LR._grid_parts(Path(d))                 # [(names, axes, lnc)] fine first, coarse last
        names = parts[0][0]
        iF = names.index("f_agn")
        val = np.full_like(H, np.nan)
        for nm, axes, lnc in parts:
            fa = axes[iF]
            j = np.where(np.abs(fa) < 1e-12)[0]
            if not len(j):
                continue
            col = lnc[:, j[0]] if iF == 1 else lnc[j[0], :]
            ha = axes[1 - iF]
            ok = np.isfinite(col)
            sp = CubicSpline(ha[ok], col[ok])
            inside = (H >= ha[ok][0]) & (H <= ha[ok][-1]) & np.isnan(val)
            val[inside] = sp(H[inside])
        assert not np.isnan(val).any(), (d, "f = 0 column does not cover the H0 prior")
        tot += val
    m = tot.max()
    return m + math.log(np.trapz(np.exp(tot - m), H))


def ln_int_grid_total(dirs):
    """ln of int int exp(sum ln L) dH0 df on the mesh of run_posteriors.grid_marginal (fine passes, coarse outside)."""
    LR = rp()
    all_parts = [LR._grid_parts(Path(d)) for d in dirs]
    names, axes = LR._common_axes(all_parts)
    mesh = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, len(names))
    tot = sum(LR._eval_parts_spline(p, mesh) for p in all_parts).reshape([len(a) for a in axes])
    m = np.max(tot[np.isfinite(tot)])
    I = np.exp(tot - m)
    for j in sorted(range(len(names)), reverse=True):
        I = np.trapz(I, axes[j], axis=j)
    return m + math.log(float(I))


def lb_savage_dickey(dirs):
    return (ln_int_grid_total(dirs) - ln_int_grid_f0(dirs)) / LN10


# ============================================================================================= #
# one key
# ============================================================================================= #
def build_entry(key, spec, ph, inputs):
    """The production entry of `key` (same schema as its placeholder entry `ph`)."""
    LR = rp()
    reals, rA = spec["realizations"], spec["realization_A"]
    dirs = [key_dir(spec, r) for r in reals]
    dA = key_dir(spec, rA)
    sA = jread(dA / "summary.json")
    grid = spec["mode"] == "grid"
    files_comb = f"{bv.rel(RUNS)}/{spec['run']}/summary.json, realizations {reals}"
    byR = spec.get("runs_by_realization") or {}
    if byR:
        files_comb += (f" (realizations {', '.join(byR)}: {bv.rel(RUNS)}/{bv.STAGE_X_KEY_RUNS[key]['pattern']}/summary.json, stage X, the same "
                       "analysis and settings)")
    E = {k: ph[k] for k in ("events", "analysis", "catalogs", "free", "stage", "N_comb", "N_comb_source", "realizations") if k in ph}
    E.update({"final": True, "N_comb": len(reals), "realizations": reals, "comb_realizations_used": reals, "quantities": {}, "joint": {}, "LB": None,
              "production_file": bv.rel(dA / "summary.json"), "production_files": [bv.rel(d / "summary.json") for d in dirs],
              "production_settings": spec["settings"], "production_run": spec["run"]})
    if byR:
        E["production_runs_by_realization"] = dict(byR)
    note_set = None if spec["settings"].startswith("as the key") else spec["settings"]
    free = key.startswith("Free")
    cache = {}
    for q, qs in spec["quantities"].items():
        units = ph["quantities"][q]["units"] if q in ph["quantities"] else ""
        if q == "RateRatio":
            continue
        par = qs["param"]
        x0 = None if qs["CL"]["input"] is None else "truth"
        if free and q in ("LognGal", "LognAGN"):
            # converted to Mpc^-3 per sample; A from the samples, Comb the product of kernel densities
            S = {r: samples_of(d, [par, "H0"]) for r, d in zip(reals, dirs)}
            conv = {r: phys_logn(s[:, 0], s[:, 1]) for r, s in S.items()}
            inp = inputs[q]
            lo = min(c.min() for c in conv.values()) - 0.1
            hi = max(c.max() for c in conv.values()) + 0.1
            A = _phr().summarize_samples(conv[rA], lo, hi, x0=inp)
            A["ul90_h"] = float(np.percentile(S[rA][:, 0], 90))           # the sampled parameter itself, h^3 Mpc^-3
            A["h_note"] = _phr().H_NOTE_PRODUCTION
            # the combination respects each sample's own support (the prior box in h^3 Mpc^-3 carried by the sample's H0); a density
            # the data do not bound on both sides keeps only its 90% upper limit and shape (no interval, width or credible level)
            prior_h = tuple(_priors()[par])
            bnd = edge_bounds({r: s[:, 0] for r, s in S.items()}, prior_h)
            xc, pc = kde_product_boxed({r: (s[:, 0], s[:, 1]) for r, s in S.items()}, prior_h)
            C = _phr().summarize_xp(xc, pc, inp)
            C["bounds"] = bnd
            C["kernel"] = ("product over the realizations of Gaussian kernel densities (0.3 sd) of the converted samples, each kernel reflected "
                           "at and truncated to its sample's own prior box (h^3 Mpc^-3 box shifted by 3 log10(H0/100) of the sample)")
            if not bnd["both_bounded"]:
                C = {k: v for k, v in C.items() if k not in ("median", "ci68", "ci90", "level_of_input", "input")}
                C["limit_only"] = True
                C["why"] = ("not bounded by the data on the " + " and ".join(s for s in ("lower", "upper") if not bnd[s]["bounded"])
                            + " side: a limit only (no combined interval, width or credible level)")
            cache[q] = (S, conv)
            E["quantities"][q] = {"A": A, "Comb": C, "source": "production", "comb_source": "production", "file": bv.rel(dA / "grid") if grid else bv.rel(dA),
                                  "comb_file": files_comb, "input": inp, "units": units,
                                  "note": "converted from the run's h^3 Mpc^-3 per sample with its own H0" + ("; " + note_set if note_set else "")}
            continue
        xa, pa = marg_A(dA, par)
        tA = sA["truth"].get(par) if x0 == "truth" else None
        A = _summ(xa, pa, tA, exact=sA["posterior"][par])
        C, xc, pc, inp = comb(key, q, spec, par, x0="truth" if x0 == "truth" else None)
        if x0 == "truth":
            assert abs(inp - tA) < 1e-12, (key, q, inp, tA)
        E["quantities"][q] = {"A": A, "Comb": C, "source": "production", "comb_source": "production", "file": bv.rel(dA / "summary.json"),
                              "comb_file": files_comb, "input": inp if x0 == "truth" else None, "units": units, "note": note_set}
        cache[q] = (xa, pa, xc, pc)
    if "RateRatio" in spec["quantities"]:
        Rin = inputs["RateRatio"]
        if spec["quantities"]["RateRatio"]["z_ratio"] == "free_densities":
            Rs = {}
            for r, d in zip(reals, dirs):
                s = samples_of(d, ["f_agn", "log10n_gal", "log10n_agn"])
                f = np.clip(s[:, 0], 1e-12, 1 - 1e-12)
                Rs[r] = f / (1 - f) * free_Z(key, r, s[:, 1], s[:, 2])
            hi = 1.3 * max(float(np.percentile(v, 99.9)) for v in Rs.values())
            A = _phr().summarize_samples(Rs[rA], 0.0, hi, x0=Rin)
            xc, pc = kde_product(list(Rs.values()), 0.0, hi)
            C = _phr().summarize_xp(xc, pc, Rin)
            how = "R = [f/(1-f)] Z_gal/Z_AGN per sample, Z_k = N_k + n_k V_miss,k (h-scaled densities of the sample, missing volume to z = 1.10)"
        else:
            xa, pa, xc, pc = cache["fAGN"]
            ratios = {r: z_ratio(key, spec, r) for r in reals}
            A = rr_from_f(xa, pa, sA["posterior"]["f_agn"], ratios[rA], Rin)
            gm = math.exp(float(np.mean([math.log(v) for v in ratios.values()])))
            C = rr_from_f(xc, pc, E["quantities"]["fAGN"]["Comb"], gm, Rin)
            how = ("R = [f/(1-f)] N_gal/N_AGN (complete catalogs, Z_k = N_k)" if spec["quantities"]["RateRatio"]["z_ratio"] == "catalog_counts" else
                   "R = [f/(1-f)] Z_gal/Z_AGN, Z_k = N_k + n_k V_miss,k (the run's densities from the counts, h-scaled; missing volume to z = 1.10)")
            how += f"; Comb: the combined f_AGN mapped with the geometric mean of the realizations' ratios ({gm:.4g})"
        E["quantities"]["RateRatio"] = {"A": A, "Comb": C, "source": "production", "comb_source": "production", "file": bv.rel(dA / "summary.json"),
                                        "comb_file": files_comb, "input": Rin, "units": "", "note": how}
    # joint posteriors
    J = spec.get("joint")
    if J:
        pars = J["params"]
        short = {"H0": "Hzero", "f_agn": "fAGN", "mu_G": "MuG", "dmu_G": "DmuG", "dmu_chi": "DmuChi", "log10n_agn": "LognAGN"}
        sx, sy = short[pars[0]], short[pars[1]]
        phr = _phr()
        if grid:
            for part, ds in (("A", [dA]), ("Comb", dirs)):
                G = LR.grid_joint(ds)
                E["joint"][part] = phr.joint_grid(G["x"], G["y"], G["P"], 120) | {
                    "params": [sx, sy], "correlation": corr_lattice(G["x"], G["y"], G["P"]),
                    "shape": f"run_posteriors.grid_joint: corrected ln L of {len(ds)} grid run(s) summed, cubic spline of the fine pass"}
            E["joint"]["A"]["correlation_summary"] = sA.get("rho_H0_f")
        elif free:
            # Fig. 4(a): f_AGN against log10 n_AGN in h^3 Mpc^-3, the sampled parameter itself (reflected at its prior edges)
            S, conv = cache["LognAGN"]
            fa = samples_of(dA, ["f_agn"])[:, 0]
            la = S[rA][:, 0]
            rb = tuple(_priors()["log10n_agn"])
            xs, ys = np.linspace(0.0, 1.0, 141), _lattice(la, rb, 141)
            P = kde2(fa, la, xs, ys, (0.0, 1.0), rb)
            E["joint"]["A"] = phr.joint_grid(xs, ys, P, 120) | {"params": ["fAGN", "LognAGNh"], "correlation": float(np.corrcoef(fa, la)[0, 1]),
                                                                 "shape": "Gaussian kernel density (0.3 sd), f_AGN reflected at 0 and 1, log10 n_AGN at its prior edges",
                                                                 "units_y": "log10 h^3 Mpc^-3", "note": phr.H_NOTE_PRODUCTION}
            E["joint"]["prior_induced"] = prior_induced(key, spec, E)
        else:
            pri = _priors()
            ra, rb = pri[pars[0]], pri[pars[1]]
            S = {r: samples_of(d, pars) for r, d in zip(reals, dirs)}
            for part, rs in (("A", [rA]), ("Comb", reals)):
                allx = np.concatenate([S[r][:, 0] for r in rs])
                ally = np.concatenate([S[r][:, 1] for r in rs])
                n = 141 if part == "A" else 201
                xs, ys = _lattice(allx, ra, n), _lattice(ally, rb, n)
                logP = sum(np.log(np.maximum(kde2(S[r][:, 0], S[r][:, 1], xs, ys, ra, rb), 1e-300)) for r in rs)
                P = np.exp(logP - logP.max())
                E["joint"][part] = phr.joint_grid(xs, ys, P, 120) | {
                    "params": [sx, sy], "correlation": float(np.corrcoef(S[rA][:, 0], S[rA][:, 1])[0, 1]) if part == "A" else corr_lattice(xs, ys, P),
                    "shape": "Gaussian kernel density (0.3 sd per axis, reflected at the prior edges)" + ("" if part == "A" else f", product over {len(rs)} realizations")}
    # Bayes factors
    L = spec.get("LB")
    if L:
        if L["kind"] == "savage_dickey_f0":
            E["LB"] = {"reference": ph["LB"]["reference"] if ph.get("LB") else L["reference"], "source": "production", "file": bv.rel(dA / "summary.json"),
                       "A": lb_savage_dickey([dA]), "Comb": lb_savage_dickey(dirs),
                       "how": "Savage-Dickey at f_AGN = 0: ln int int L - ln int L(H0, 0) (f_AGN prior [0, 1]; Comb: ln L summed over the realizations)"}
        else:
            per = {}
            for r in reals:
                za = jread(key_dir(spec, r) / "summary.json")["ln_evidence"]
                zb = jread(run_dir(L["reference_run"], r) / "summary.json")["ln_evidence"]
                per[r] = (za["value"] - zb["value"]) / LN10
            E["LB"] = {"reference": ph["LB"]["reference"] if ph.get("LB") else L["reference"], "source": "production", "file": bv.rel(dA / "summary.json"),
                       "A": per[rA], "Comb": float(sum(per.values())), "per_realization": per,
                       "how": "(ln Z - ln Z_ref)/ln 10 from the summaries' ln_evidence; Comb = the sum over the realizations"}
    return E


def _priors():
    return rp().P.PRIORS


def prior_induced(key, spec, E, n=2_000_000, seed=20261117):
    """Region of (f_AGN, log10 n_AGN [h^3 Mpc^-3]) the priors allow once the rate per host is measured (the production
    analogue of build_values.induced_density): prior draws of the free-density run, weighted by the likelihood of the
    known-density run of realization A at the same flux limit (densities from the counts), mapped through the
    normalisation identity logit f* = logit f + ln[F_A(n_A)/F_G(n_G)] - ln[F_A(n_A,used)/F_G(n_G,used)],
    F_k = N_k / (N_k + n_k V_miss,k) (h-scaled densities: Z_k does not depend on H0). Sets the A entries' prior_ul90 (Mpc^-3, each
    draw converted with its own H0) and prior_ul90_h (h^3 Mpc^-3, the drawn parameter)."""
    LR = rp()
    pri = _priors()
    w = key[4:]
    rk = {ww: k for k, ww, _ in bv.RUNGS}[w]
    rA = spec["realization_A"]
    fdir = RUNS / spec["prior_induced_from"]
    sF = jread(fdir / "summary.json")
    NC = bv.norm_cfg(rA)[rk]
    Fk = lambda cls, l10: NC[cls]["N"] / (NC[cls]["N"] + 10 ** np.asarray(l10, float) * NC[cls]["Vp"])
    used = {c: sF["densities"][c]["log10n_used"] for c in ("gal", "agn")}
    assert all(sF["densities"][c]["units"] == "h_scaled" for c in ("gal", "agn")), sF["densities"]
    rng = np.random.default_rng(seed)
    H = rng.uniform(*pri["H0"], n)
    f = rng.uniform(*pri["f_agn"], n)
    lg = rng.uniform(*pri["log10n_gal"], n)
    la = rng.uniform(*pri["log10n_agn"], n)
    sh = np.log(Fk("agn", la) / Fk("gal", lg)) - math.log(Fk("agn", used["agn"]) / Fk("gal", used["gal"]))
    fs_ = bv.expit(bv.logit(np.clip(f, 1e-12, 1 - 1e-12)) + sh)
    parts = LR._grid_parts(fdir)
    lv = LR._eval_parts(parts, np.stack([H, fs_], 1))
    wt = np.exp(lv - np.max(lv[np.isfinite(lv)]))
    wt /= wt.sum()
    lap, lgp = phys_logn(la, H), phys_logn(lg, H)
    E["quantities"]["LognAGN"]["A"]["prior_ul90"] = float(bv.wquant(lap, wt, 0.90))
    E["quantities"]["LognGal"]["A"]["prior_ul90"] = float(bv.wquant(lgp, wt, 0.90))
    E["quantities"]["LognAGN"]["A"]["prior_ul90_h"] = float(bv.wquant(la, wt, 0.90))     # the sampled parameter, h^3 Mpc^-3
    E["quantities"]["LognGal"]["A"]["prior_ul90_h"] = float(bv.wquant(lg, wt, 0.90))
    from scipy.ndimage import gaussian_filter
    ry = tuple(pri["log10n_agn"])
    Hh, xe, ye = np.histogram2d(f, la, bins=50, range=[(0.0, 1.0), ry], weights=wt)
    Hh = gaussian_filter(Hh, 1.0, mode="reflect")
    return _phr().joint_grid(0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1]), Hh, 50) | {
        "params": ["fAGN", "LognAGNh"], "n_eff": float(1 / np.sum(wt ** 2)), "file": bv.rel(fdir / "summary.json"), "units_y": "log10 h^3 Mpc^-3",
        "note": _phr().H_NOTE_PRODUCTION,
        "shape": f"{n:.0e} prior draws weighted by the known-density likelihood of {spec['prior_induced_from']} mapped through the identity; log10 n_AGN in h^3 Mpc^-3 (the drawn parameter)"}


# ============================================================================================= #
# all keys
# ============================================================================================= #
CACHE = PRD / "values" / ".production_cache"      # one entry per real key, reused while its signature holds


def signature(spec):
    """Hash of the key's map entry, of the code that computes it (this file, run_posteriors.py,
    placeholder_results.py, build_values.py) and of every summary.json it needs (path, mtime, size: the lane
    worker rewrites a run's summary whenever the run changes)."""
    import hashlib
    h = hashlib.sha1(json.dumps(spec, sort_keys=True).encode())
    for f in (Path(__file__), HERE / "run_posteriors.py", HERE / "placeholder_results.py", HERE / "build_values.py"):
        h.update(Path(f).read_bytes())
    for d in sorted(needed(spec), key=str):
        s = (d / "summary.json").stat()
        h.update(f"{d}|{s.st_mtime_ns}|{s.st_size}".encode())
    return h.hexdigest()
def apply(K, inputs):
    """Replace the placeholder entries of K by production entries wherever every run exists. Returns
    (map as read, status {key: {...}})."""
    pm = load_map()
    status = {}
    if not pm:
        return pm, status
    if "keys" not in pm:                                  # the old form: key -> entry file
        for key, path in pm.items():
            p = Path(path) if Path(path).is_absolute() else bv.W / path
            if key in K and p.exists():
                E = json.loads(p.read_text())
                E.update({"final": True, "production_file": str(path)})
                for qq in E.get("quantities", {}).values():
                    qq["source"] = qq["comb_source"] = "production"
                K[key] = E
                status[key] = {"state": "real", "how": "entry file"}
        return pm, status
    for key, spec in pm["keys"].items():
        if key not in K:
            continue
        st = key_status(key, spec)
        if st["state"] == "real":
            try:
                sig = signature(spec)
                cf = CACHE / f"{key}.json"
                c = json.loads(cf.read_text()) if cf.exists() else {}
                if c.get("signature") == sig:
                    K[key] = c["entry"]
                    st["cached"] = True
                else:
                    K[key] = bv.jsonable(build_entry(key, spec, K[key], inputs))
                    CACHE.mkdir(exist_ok=True)
                    cf.write_text(json.dumps({"signature": sig, "entry": K[key]}))
            except Exception as e:  # noqa: BLE001  (the key stays red and the status says why)
                import traceback
                st["state"] = "error"
                st["error"] = f"{type(e).__name__}: {e}"
                st["traceback"] = traceback.format_exc()[-2000:]
                print(f"  [production] {key}: ERROR {st['error']} (stays the placeholder)")
        if st["state"] != "real":
            K[key]["production_status"] = {k: v for k, v in st.items() if k != "traceback"}
        status[key] = st
    for key, why in (pm.get("not_in_run_lists") or {}).items():
        status[key] = {"state": "no run", "why": why}
    return pm, status


def status_lines(status):
    out = []
    order = ("real", "partial", "waiting", "error", "no run")
    for s in order:
        ks = [k for k, v in status.items() if v["state"] == s]
        if not ks:
            continue
        if s == "real":
            out.append(f"  real ({len(ks)}): {', '.join(ks)}")
        elif s in ("partial", "waiting"):
            out.append(f"  {s} ({len(ks)}):")
            for k in ks:
                v = status[k]
                out.append(f"    {k:14s} {len(v['realizations_have'])}/{len(v['realizations_want'])} realizations, "
                           f"{v['n_have']}/{v['n_need']} runs; stage {v['stage']} ({v['runlist']})")
        elif s == "error":
            out.append(f"  ERROR ({len(ks)}): " + "; ".join(f"{k}: {status[k]['error']}" for k in ks))
        else:
            out.append(f"  no run in the run lists ({len(ks)}): {', '.join(ks)}")
    return out


def classes_status_lines():
    """The experiment's macros (scripts/classes.py, map block 'classes'): real, and the runs the others wait for."""
    import classes
    st = classes.status()
    real = [k for k, (r, _) in st.items() if r]
    out = [f"  experiment macros (Sec. IV B; values/production_map.json 'classes'): {len(real)} of {len(st)} real"]
    wait = {}
    for k, (r, miss) in st.items():
        if not r:
            pats = {}
            for m in miss:
                mm = re.match(r"^.*/runs/(.*)/realization_(\d{3})(.*)$", m)
                pats.setdefault(mm.group(1) + "/realization_{R}" + mm.group(3), []).append(mm.group(2)) if mm else pats.setdefault(m, [])
            txt = "; ".join(f"{pp} (missing {','.join(rr)})" for pp, rr in pats.items())
            wait.setdefault(txt or "derived from red macros", []).append(k)
    for runs, ks in wait.items():
        out.append(f"    waiting {', '.join(ks)}: {runs}")
    return out


def density_status_lines():
    """The response to the assumed AGN density (scripts/density.py, map block 'density_response'): real, or what it waits for."""
    import density
    st = density.status()
    real = [k for k, (r, _) in st.items() if r]
    out = [f"  assumed AGN density (Sec. IV C, Fig. 3(b); values/production_map.json 'density_response'): {len(real)} of {len(st)} groups real"
           + (f" ({', '.join(real)})" if real else "")]
    for k, (r, miss) in st.items():
        if not r:
            out.append(f"    waiting {k}: {'; '.join(miss[:3])}{' ...' if len(miss) > 3 else ''}")
    return out


def skyres_status_lines():
    """The sky resolution (scripts/skyres.py, map block 'sky_resolution'): real, or what it waits for."""
    import skyres
    out = []
    for which, what in (("pair", "sky resolution (Sec. V; values/production_map.json 'sky_resolution')"),
                        ("mc_pair", "sky resolution at 32000 samples and six injection sets (stage X; 'sky_resolution.mc_pair')")):
        real, miss = skyres.status(which)
        out.append(f"  {what}: " + ("real" if real else f"waiting: {'; '.join(miss)}"))
        # 19 Nov: the pair of every realization, and the mean and scatter over them (SkyRes<P>Mean..., SkyRes<P>Sd...)
        ok, miss_all, per = skyres.status_all(which)
        short = lambda m: m.rsplit("/", 2)[-2] if m.endswith("summary.json") else m
        out.append("    per realization: " + ("; ".join(f"{R} " + ("real" if per[R][0] else f"waiting ({', '.join(short(m) for m in per[R][1])})")
                                                         for R in sorted(per)) or "none in the map"))
        n_ok = sum(1 for R in per if per[R][0])
        out.append(f"    mean and scatter over the realizations (SkyRes{skyres.PAIRS[which]}Mean..., SkyRes{skyres.PAIRS[which]}Sd...): "
                   + ("real" if ok else f"waiting: {n_ok} of {len(skyres.REALIZATIONS)} pairs real"
                                                             + "".join(f"; {m}" for m in miss_all if not m.endswith("summary.json") and "(no posterior" not in m)))
    # 21 Nov: the sets of pairs combined over their realizations (SkyResMCHzero..., SkyResAGN..., SkyResAll..., SkyResHalf...); 23 Nov:
    # a set without macros (skyres.SETS_WITHOUT_MACROS: pair_all, whose SkyResTen... and SkyResOffsetHzero... were removed) is only named
    for which in skyres.SETS:
        if which in skyres.SETS_WITHOUT_MACROS:
            out.append(f"  sky resolution combined over the realizations, set {which}: no macros (removed 23 Nov; the map keeps the set)")
            continue
        E = skyres.evaluate_set(which)
        out.append(f"  sky resolution combined over the realizations, set {which} (SkyRes{skyres.SETS[which][0]}...): "
                   + (f"real ({len(E['realizations'])} realizations)" if E["real"] else
                      f"waiting: {'; '.join(E['missing'][:3])}{' ...' if len(E['missing']) > 3 else ''}"))
    import density
    E = density.phys_evaluate()
    out.append("  densities given in Mpc^-3 (stage X; 'density_response.units_physical'): " + ("real" if E["real"] else f"waiting: {'; '.join(E['missing'])}"))
    # 23 Nov: realization 106 of keys ObsAssumed and ObsPops at 32000 samples per merger (map block 'obslike_mc'; Obs...MC... macros)
    pm = load_map()
    blk, keys = pm.get("obslike_mc") or {}, pm.get("keys") or {}
    runs = blk.get("runs") or {}
    if not runs or any(k not in runs for k in bv.STAGE_X_MC_RUNS):
        out.append(f"  realization 106 at 32000 samples ('obslike_mc'; Obs...MC... macros): not in the map ({blk.get('not_in_run_lists') or 'no block'})")
    else:
        need = [RUNS / v["run"] for v in runs.values()] + [key_dir(keys[k], r) for k in runs for r in keys[k]["realizations"]]
        miss = [bv.rel(d / "summary.json") for d in need if not (d / "summary.json").exists()]
        out.append("  realization 106 at 32000 samples (stage X; 'obslike_mc'; Obs...MC... macros): "
                   + ("real" if not miss else f"waiting: {'; '.join(miss[:3])}{' ...' if len(miss) > 3 else ''}"))
    return out


def macro_counts(values_json=PRD / "values" / "values.json", tex=PRD / "main.tex"):
    """Result macros of the long paper: the key-scheme macros (claim 'skeleton,keys') and the experiment's (claim
    'classes'); real = printed black."""
    mac = json.loads(Path(values_json).read_text())["macros"]
    used = set(re.findall(r"\\([A-Za-z]+)", Path(tex).read_text()))
    keys = {k: v for k, v in mac.items() if v["claim"].startswith(("skeleton", "classes", "density"))}
    real = [k for k, v in keys.items() if v["class"] != "ph"]
    ureal = [k for k in real if k in used]
    uall = [k for k in keys if k in used]
    other_red = sorted(k for k, v in mac.items() if v["class"] in ("ph", "anc") and k in used and k not in keys)
    cl = {k: v for k, v in keys.items() if v["claim"].startswith("classes")}
    return {"n_all": len(keys), "n_real": len(real), "n_used": len(uall), "n_used_real": len(ureal), "other_red_in_text": other_red,
            "n_classes": len(cl), "n_classes_real": sum(1 for v in cl.values() if v["class"] != "ph")}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-map", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args(argv)
    if a.write_map:
        m = write_map()
        print(f"wrote {bv.rel(PMAP)}: {len(m['keys'])} keys mapped; not in the run lists: {', '.join(m['not_in_run_lists'])}")
    if a.status or a.write_map:
        pm = load_map()
        st = {k: key_status(k, s) for k, s in pm.get("keys", {}).items()}
        st.update({k: {"state": "no run", "why": w} for k, w in (pm.get("not_in_run_lists") or {}).items()})
        print("\n".join(status_lines(st)))
        print("\n".join(classes_status_lines()))
        print("\n".join(density_status_lines()))
        print("\n".join(skyres_status_lines()))
    if a.report:
        c = macro_counts()
        doc = json.loads((PRD / "values" / "placeholder_results.json").read_text())
        st = doc.get("_production_status", {})
        n = {s: sorted(k for k, v in st.items() if v["state"] == s) for s in ("real", "partial", "waiting", "error", "no run")}
        if c["other_red_in_text"]:
            print(f"long paper, other red macros in main.tex (exploratory or not in a run list, not counted below): {', '.join(c['other_red_in_text'])}")
        print(f"long paper: {c['n_used_real']} of {c['n_used']} result macros in main.tex real, {c['n_used'] - c['n_used_real']} waiting for runs "
              f"(all result macros: {c['n_real']} of {c['n_all']} real, of which the experiment's {c['n_classes_real']} of {c['n_classes']}; "
              f"keys real {len(n['real'])}, partial {len(n['partial'])}, waiting {len(n['waiting'])}, "
              f"no run {len(n['no run'])}" + (f", ERROR {len(n['error'])}" if n["error"] else "") + ")")


if __name__ == "__main__":
    main()

