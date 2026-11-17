#!/usr/bin/env python
"""Write runs/RUNLIST.yaml: the approved production stages, every run's driver arguments, its parts
(schedulable tasks), estimated likelihood calls and GPU hours.

    python scripts/make_runlist.py [--realizations-10 100,...] [--timing tests/results/timing.json]

Cost model (the brief's numbers until the tests measure them; --timing overrides):
  seconds per call at 8,000 samples: 0.39 with the population fixed, 0.44 with it free (both
  catalogs, complete); the same is assumed for one catalog, flux-limited catalogs and no catalog
  (conservative until test b measures them);
  grid: 2-D (H0, f_AGN) = 861 fine cells (41 x 21) + 286 coarse (H0 step 2 x f step 0.1 over the
  prior); 1-D H0 = 81 fine + 51 coarse;
  nested calls per run, by number of sampled parameters: six 36k = the measured 29k (Analyses 11/12,
  unif) plus the wider production priors (+ nlive * ln(prior-volume ratio) = +676 iterations at the
  ~10 calls per iteration of those runs); five (no catalog) 29k and four (free densities) 15k are
  estimates, not measurements; eight 272k = Analysis 13's guarded rslice run (228k calls, box H0
  [60,76] etc.) + 747 iterations at its 58 calls each for the production priors (test c may change
  the setting and this number);
  two independent nested runs (merged) up to six parameters, one for eight;
  plus 3 minutes per process for loading, binding and compiling.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import prod_common as P  # noqa: E402

COST = {"spc_fixed": 0.39, "spc_free": 0.44, "spc_none_fixed": 0.39, "spc_none_free": 0.44,
        "grid_2d_cells": 861 + 286, "grid_1d_cells": 81 + 51,
        "calls_by_ndim": {4: 15000, 5: 29000, 6: 36000, 8: 272000},
        "calls_measured_reference": {"6_A11_A12_box": 29000, "7_A11": 48000, "8_A13_box": 228000},
        "overhead_hours_per_process": 0.05}


def nested_calls(ndim):
    return COST["calls_by_ndim"][ndim]


def spc(catalogs, pop_free):
    if catalogs == "none":
        return COST["spc_none_free"] if pop_free else COST["spc_none_fixed"]
    return COST["spc_free"] if pop_free else COST["spc_fixed"]


SAMPLER_8 = {"rslice": [], "unif_b0": ["--sample", "unif", "--enlarge", "1.25", "--bootstrap", "0"],
             "rwalk": ["--sample", "rwalk", "--walks", "28"]}
SAMPLER_8_CHOICE = ["rslice"]


def entry(stage, mode, analysis, catalogs, flux, kind, real, density="counts", agn_factor=1.0, extra_args=()):
    free = P.free_parameters(analysis, catalogs, flux)
    args = ["--mode", mode, "--analysis", analysis, "--catalogs", catalogs, "--flux", flux,
            "--events-kind", kind, "--realization", str(real)]
    if flux != "complete" and analysis != "free_densities":
        args += ["--density", density]
    if analysis == "free_densities":
        args += ["--density", "free"]
    if agn_factor != 1.0:
        args += ["--agn-density-factor", f"{agn_factor:g}"]
    args += list(extra_args)
    if mode == "nested" and len(free) > 6:
        args += SAMPLER_8[SAMPLER_8_CHOICE[0]]
    name = P.run_name(analysis, catalogs, flux, kind, real, density=density, agn_density_factor=agn_factor)
    pop_free = any(p in free for p in P.POP_FREE)
    s = spc(catalogs, pop_free)
    if mode == "grid":
        cells = COST["grid_2d_cells"] if len(free) == 2 else COST["grid_1d_cells"]
        parts = ["grid"]
        calls = cells
    else:
        n_ind = 2 if len(free) <= 6 else 1
        parts = [f"nested_{k}" for k in range(1, n_ind + 1)]
        calls = nested_calls(len(free)) * n_ind
    gpu_h = calls * s / 3600 + COST["overhead_hours_per_process"] * len(parts)
    ev = P.events_path(real, kind)
    return {"name": name, "stage": stage, "mode": mode, "analysis": analysis, "free": free,
            "n_parameters": len(free), "parts": parts, "args": args, "est_calls": int(calls),
            "est_seconds_per_call": s, "est_gpu_hours": round(gpu_h, 3),
            "est_gpu_hours_per_part": round(gpu_h / len(parts), 3),
            "events_file": str(ev), "data_exists": ev.exists()}


def build(r5, r10, r3):
    st = {1: [], 2: [], 3: [], 4: []}
    # Stage 1: population known, complete catalogs (grids)
    for r in r10:
        st[1].append(entry(1, "grid", "known_population", "both", "complete", "equal_populations", r))
    for r in r5:
        st[1].append(entry(1, "grid", "known_population", "galaxy", "complete", "equal_populations", r))
        st[1].append(entry(1, "grid", "known_population", "agn", "complete", "equal_populations", r))
        st[1].append(entry(1, "grid", "known_population", "none", "complete", "equal_populations", r))
        st[1].append(entry(1, "grid", "known_population", "agn", "complete", "agn_hosted", r))
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
        st[4].append(entry(4, "nested", "single_population", "both", "complete", "host_dependent", r))
    for r in r5:
        st[4].append(entry(4, "nested", "host_dependent_populations", "both", "complete", "host_dependent", r))
    for r in r5:
        st[4].append(entry(4, "nested", "single_population", "none", "complete", "host_dependent", r))
    return st


TITLES = {1: "population known, complete catalogs", 2: "flux-limited catalogs",
          3: "equal-population events, single population", 4: "host-dependent events"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--realizations-5", default="100,101,102,103,105")
    ap.add_argument("--realizations-10", default="100,101,102,103,105,106,107,108,109,110",
                    help="the five beyond 105 are placeholders until the new realizations exist")
    ap.add_argument("--realizations-3", default="100,101,102")
    ap.add_argument("--timing", default=None, help="JSON with measured spc_* / calls_* to override COST")
    ap.add_argument("--out", default=str(P.RUNS / "RUNLIST.yaml"))
    ap.add_argument("--sampler-8", choices=list(SAMPLER_8), default="rslice",
                    help="dynesty setting of the eight-parameter runs (test c); rslice = Analysis 13's guarded run")
    ap.add_argument("--calls-8", type=int, default=None, help="projected calls per eight-parameter run (test c)")
    a = ap.parse_args()
    SAMPLER_8_CHOICE[0] = a.sampler_8
    if a.calls_8:
        COST["calls_by_ndim"][8] = a.calls_8
    if a.timing and Path(a.timing).exists():
        COST.update({k: v for k, v in json.loads(Path(a.timing).read_text()).items() if k in COST})
    r5 = [int(x) for x in a.realizations_5.split(",")]
    r10 = [int(x) for x in a.realizations_10.split(",")]
    r3 = [int(x) for x in a.realizations_3.split(",")]
    st = build(r5, r10, r3)
    import yaml
    doc = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "driver": "scripts/prod_driver.py (core e7c3007 venv); every run also gets the defaults: "
                     "--nsamp 8000 --half-variance-correction on --max-var 20 --zmax 1.10, nested --nlive 200 "
                     "--dlogz 0.1, unif up to six parameters, rslice for eight",
           "cost_model": COST, "sampler_8": a.sampler_8, "realizations": {"5": r5, "10": r10, "3": r3},
           "stages": []}
    for k in sorted(st):
        runs = st[k]
        tot = sum(r["est_gpu_hours"] for r in runs)
        doc["stages"].append({"stage": k, "title": TITLES[k], "n_runs": len(runs),
                              "n_tasks": sum(len(r["parts"]) for r in runs),
                              "est_calls": sum(r["est_calls"] for r in runs), "est_gpu_hours": round(tot, 2),
                              "est_gpu_hours_existing_data": round(sum(r["est_gpu_hours"] for r in runs
                                                                       if r["data_exists"]), 2),
                              "runs": runs})
    Path(a.out).write_text(yaml.safe_dump(doc, sort_keys=False, width=140))
    for s in doc["stages"]:
        print(f"stage {s['stage']} ({s['title']}): {s['n_runs']} runs, {s['n_tasks']} tasks, "
              f"{s['est_calls']:,} calls, {s['est_gpu_hours']} GPU-h "
              f"({s['est_gpu_hours_existing_data']} on existing data)")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
