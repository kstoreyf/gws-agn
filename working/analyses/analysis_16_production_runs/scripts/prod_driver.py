#!/usr/bin/env python
"""Analysis 16 production driver: one run (grid or nested sampling) on darksirens-core e7c3007.

    <core venv python> prod_driver.py --mode grid --analysis known_population --catalogs both \\
        --flux complete --events-kind equal_populations --realization 100
    <core venv python> prod_driver.py --mode nested --analysis single_population --catalogs both \\
        --flux complete --events-kind equal_populations --realization 100 --nested-index 1

Every option is explicit and recorded in runs/<run>/config.json; nothing reaches a core default
(prod_model.py lists the fixed settings). A resubmission with the same arguments resumes: grids
from runs/<run>/grid/<pass>.partial.npz, nested runs from runs/<run>/nested_<k>/state.save.

Output (runs/<run>/):
  config.json                 resolved configuration (population values, densities and z_c, priors,
                              sampler settings, files, core commit, evaluation settings)
  grid/{coarse,fine}.h5       per cell: lnL (core's, uncorrected, guard included), pe_var = sum_i
                              sigma_i^2, n_eff, log_mu, sel_corr, seconds; grid/result.json
  nested_<k>/state.save       dynesty checkpoint (state only; resumed automatically)
  nested_<k>/results.pkl      dynesty Results (blob per dead point = [lnL, pe_var, n_eff, log_mu,
                              sel_corr]: the uncorrected ln L and the variance terms, so the
                              posterior without the correction is a reweighting)
  nested_<k>/result.json      run summary; finished flag
  <part>/sessions.json        one entry per process: Slurm job id, host, wall time, calls, exit
Exit codes: 0 finished, 75 stopped at the deadline (resubmit), 3 configuration/validation error.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import shutil
import signal
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import prod_common as P  # noqa: E402

EXIT_DEADLINE = 75
T_START = time.time()


def log(msg):
    P.log("a16", msg)


# ============================================================================ configuration
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a = ap.add_argument
    a("--mode", choices=["grid", "nested"], required=True)
    a("--analysis", choices=P.ANALYSES, required=True)
    a("--catalogs", choices=list(P.CATALOG_SETS), required=True, help="both, galaxy, agn or none")
    a("--flux", choices=P.FLUX_LIMITS, default="complete", help="complete or the apparent-magnitude limit")
    a("--events-kind", choices=list(P.EVENT_KINDS), required=True)
    a("--realization", type=int, required=True)
    a("--events-suffix", default=P.DEFAULT_SUFFIX, help="events file suffix (a mock family); default _corepop")
    a("--family-name", default="", help="reader-facing name of a non-default mock family (run directory)")
    a("--set-label", default="", help="experiment runs: the event set's name in the run path (e.g. mock_f0p30_s1)")
    a("--membership", action="store_true",
      help="two-tracer grids: after the grid, each merger's probability of an AGN host at the posterior "
           "median (grid/membership.npz, membership.json)")
    a("--population-family", default="corepop", choices=list(P.POPULATION_INPUTS),
      help="the population inputs held fixed (and the truth recorded)")
    a("--pop", action="append", default=[], metavar="NAME=VALUE",
      help="override one fixed population value (ASCII name, e.g. mu_G=34)")
    a("--free", default=None, help="comma list overriding the analysis' sampled parameters")
    a("--prior", action="append", default=[], metavar="NAME=LO,HI", help="override a prior range (tests)")
    a("--zmax", type=float, default=P.ZMAX_PRODUCTION, help="end of darksirens' redshift grid (DARKSIRENS_ZMAX)")
    a("--complete-model", choices=["complete", "count_ratio", "selection"], default="complete",
      help="completeness model of complete catalogs (production: complete, Z_k = N_obs)")
    a("--density", choices=["counts", "given", "free"], default="counts",
      help="number densities of flux-limited catalogs: from catalog counts, given, or sampled")
    a("--log10n-gal", type=float, default=None, help="given log10 galaxy density (density units)")
    a("--log10n-agn", type=float, default=None, help="given log10 AGN density (density units)")
    a("--gal-density-factor", type=float, default=1.0)
    a("--agn-density-factor", type=float, default=1.0)
    a("--density-units", choices=["h_scaled", "physical"], default="h_scaled")
    a("--nsamp", type=int, default=8000, help="posterior samples per event (redrawn); 0 = the file's own")
    a("--redraw-index", type=int, default=0)
    a("--redraw-key", default=None, help="override the redraw SeedSequence prefix (comma ints)")
    a("--half-variance-correction", choices=["on", "off"], default="on")
    a("--max-var", type=float, default=20.0, help="max_likelihood_variance (the variance guard)")
    a("--pe-event-block", type=int, default=5)
    a("--sel-batch", type=int, default=5000)
    a("--events-file", default=None, help="override the events file (closure tests)")
    a("--injections-file", default=None, help="override the injection file (a full path; tests)")
    a("--pool-injections", action="append", default=[], metavar="LANE",
      help="another injection set of the realization (file relative to data/seedNNN/) pooled with the lane "
           "(repeat for several; same proposal, so the pool is one set of the summed draws)")
    a("--agn-subset-nsamp", type=int, default=None,
      help="hosts_known: samples per merger of the AGN-hosted subset (default --nsamp)")
    a("--agn-subset-pool", action="append", default=[], metavar="LANE",
      help="hosts_known: injection sets pooled with the lane for the AGN-hosted subset only")
    a("--variant", default="", help="run-name suffix of non-default Monte Carlo settings (e.g. s32000_inj6)")
    a("--injections-lane", default=None,
      help="the injection lane, a file relative to data/seedNNN/ (default: the population family's, "
           "injections/injections_targeted[_obslike].h5); the run list takes it from data/experiment_sets.json")
    a("--limit-events", type=int, default=None, help="first N events only (CPU checks)")
    a("--limit-injections", type=int, default=None, help="first N detected injections only (CPU checks)")
    a("--run-name", default=None, help="run path under --out-root (default: from the options)")
    a("--out-root", default=str(P.RUNS))
    a("--tag", default="", help="suffix of the output part (test variants)")
    a("--deadline", type=float, default=None, help="epoch seconds; default env A16_DEADLINE")
    # grid
    a("--grid", action="append", default=[], metavar="NAME=LO,HI,STEP", help="explicit fine grid axis")
    a("--coarse", default="auto", help="auto, none, or NAME=LO,HI,STEP[;NAME=...]")
    a("--fine-cells", default="H0=41,f_agn=21", help="cells per axis of the automatic fine grid (2-D)")
    a("--fine-cells-1d", type=int, default=81)
    a("--fine-halfwidth-sd", type=float, default=5.0)
    a("--max-cells", type=int, default=None, help="evaluate at most N cells per pass (dry runs)")
    # nested
    a("--nested-index", type=int, default=1, help="independent run k (rstate seed k unless --seed)")
    a("--seed", type=int, default=None)
    a("--nlive", type=int, default=200)
    a("--dlogz", type=float, default=0.1)
    a("--sample", choices=["unif", "rwalk", "rslice"], default=None,
      help="default: unif up to 6 parameters, rslice above (Analysis 13's guarded run)")
    a("--bootstrap", type=int, default=None)
    a("--enlarge", type=float, default=None)
    a("--walks", type=int, default=None)
    a("--slices", type=int, default=None)
    a("--checkpoint-every", type=float, default=600.0)
    a("--chunk-calls", type=int, default=2000)
    a("--maxcall-total", type=int, default=None, help="stop (unfinished) at this many calls (tests)")
    a("--maxiter-total", type=int, default=None, help="stop (unfinished) at this many iterations (tests)")
    a("--maxcall-more", type=int, default=None, help="tests: stop after this many more calls than at start")
    a("--maxiter-more", type=int, default=None, help="tests: stop after this many more iterations than at start")
    a("--kill-after-calls", type=int, default=None, help="tests: hard exit after N calls in this process")
    a("--save-at-iterations", default="", help="tests: copy the state at these iterations")
    a("--init-from", default=None, help="tests: start from this dynesty state (copied), then --switch-sample")
    a("--switch-sample", choices=["unif", "rwalk", "rslice"], default=None)
    a("--dry-run", action="store_true", help="CPU: 20 events, 20000 injections, a few cells/calls")
    return ap.parse_args(argv)


def parse_axis(spec):
    name, rng = spec.split("=")
    lo, hi, step = (float(x) for x in rng.split(","))
    n = int(round((hi - lo) / step)) + 1
    return name, np.round(lo + step * np.arange(n), 10)


def resolve(args):
    """Every option -> explicit values (the run's config.json)."""
    if args.dry_run:
        args.limit_events = args.limit_events or 20
        args.limit_injections = args.limit_injections or 20000
    tracers = P.CATALOG_SETS[args.catalogs]
    events_file = args.events_file or str(P.events_path(args.realization, args.events_kind, args.events_suffix))
    # the planted values the events file records (None where the file predates them)
    ev_truth = P.read_event_truth(events_file) if Path(events_file).exists() else {}
    pv = dict(P.POPULATION_INPUTS[args.population_family])
    for s in args.pop:
        k, v = s.split("=")
        if k not in pv:
            sys.exit(f"[fatal] --pop {k}: not a population parameter ({list(pv)})")
        pv[k] = float(v)
    # the galaxy class must be the population family's (refuse a family/events mismatch)
    if ev_truth.get("mu_G_gal") is not None and not args.pop:
        fam_spec = ev_truth.get("mass_spectrum") or "mock"
        want_spec = "gwtc3_pp" if args.population_family == "obslike" else "mock"
        if (abs(ev_truth["mu_G_gal"] - pv["mu_G"]) > 1e-9 or abs(ev_truth["mu_chi_gal"] - pv["mu_chi"]) > 1e-9
                or fam_spec != want_spec):
            sys.exit(f"[fatal] {events_file}: galaxy class mu_G {ev_truth['mu_G_gal']}, mu_chi "
                     f"{ev_truth['mu_chi_gal']}, spectrum {fam_spec}; population family {args.population_family} "
                     f"has {pv['mu_G']}, {pv['mu_chi']}, {want_spec}")
    if ev_truth.get("dmu_G_agn") is not None:
        planted_offsets = {"dmu_G": float(ev_truth["dmu_G_agn"]), "dmu_chi": float(ev_truth["dmu_chi_agn"])}
    else:
        planted_offsets = dict(P.AGN_OFFSET_INPUTS.get(args.population_family, {})) \
            if args.events_kind == "host_dependent" else {k: 0.0 for k in P.OFFSETS}
    populations_differ = any(abs(v) > 0 for v in planted_offsets.values())
    planted_f = ev_truth.get("planted_f_agn")
    if planted_f is None:
        planted_f = P.EVENT_KINDS[args.events_kind][2]
    elif P.EVENT_KINDS[args.events_kind][2] is not None and args.events_file is None \
            and abs(planted_f - P.EVENT_KINDS[args.events_kind][2]) > 1e-12:
        sys.exit(f"[fatal] {events_file} planted f_AGN {planted_f}, kind {args.events_kind} says "
                 f"{P.EVENT_KINDS[args.events_kind][2]}")
    if planted_f is None:
        if Path(events_file).exists():
            sys.exit(f"[fatal] no planted f_AGN for {events_file}")
        planted_f = float("nan")         # the file does not exist yet: main() reports missing data
    free = args.free.split(",") if args.free else P.free_parameters(args.analysis, args.catalogs, args.flux,
                                                                    populations_differ=populations_differ)
    for p in free:
        if p not in P.PRIORS:
            sys.exit(f"[fatal] unknown parameter {p!r}")
    priors = {k: list(v) for k, v in P.PRIORS.items()}
    for s in args.prior:
        k, v = s.split("=")
        priors[k] = [float(x) for x in v.split(",")]
    offsets_truth = dict(planted_offsets)
    if (set(P.OFFSETS) & set(free)) and args.catalogs != "both":
        sys.exit("[fatal] AGN offsets need both catalogs")
    # each class's population: the galaxy class at the family's inputs, the AGN class shifted by the
    # planted offsets (experiment analyses that give the AGN class its own population use it)
    class_pops = {"gal": dict(pv), "agn": dict(pv)}
    class_pops["agn"]["mu_G"] = pv["mu_G"] + planted_offsets["dmu_G"]
    class_pops["agn"]["mu_chi"] = pv["mu_chi"] + planted_offsets["dmu_chi"]
    fixed_offsets = None
    if args.analysis == "two_tracers_two_populations":
        fixed_offsets = dict(planted_offsets)
    if args.analysis in ("hosts_known", "mass_spectrum_alone") and args.free:
        sys.exit(f"[fatal] {args.analysis} has its own parameters; --free is not available")
    flux = args.flux
    comp = (args.complete_model if flux == "complete" else "selection") if tracers else None
    lane = args.injections_lane or P.default_injection_lane(args.population_family)
    inj_file = args.injections_file or str(P.injections_path(args.realization, args.population_family, lane))
    if args.pool_injections and args.analysis == "hosts_known":
        sys.exit("[fatal] hosts_known pools injections for its AGN-hosted subset only: --agn-subset-pool")
    if (args.agn_subset_nsamp or args.agn_subset_pool) and args.analysis != "hosts_known":
        sys.exit("[fatal] --agn-subset-* are options of hosts_known")
    inj_files = [inj_file] + [str(P.injections_path(args.realization, args.population_family, x))
                              for x in args.pool_injections]
    agn_subset = None
    if args.analysis == "hosts_known" and (args.agn_subset_nsamp or args.agn_subset_pool):
        agn_subset = {"nsamp": int(args.agn_subset_nsamp or args.nsamp),
                      "injections_files": [inj_file] + [str(P.injections_path(args.realization,
                                                                               args.population_family, x))
                                                        for x in args.agn_subset_pool],
                      "injection_lanes": [lane] + list(args.agn_subset_pool)}
        if agn_subset["nsamp"] % P.REDRAW_BLOCK or int(args.nsamp) == 0:
            sys.exit("[fatal] --agn-subset-nsamp must be a multiple of 2000 with redrawn samples")
    surveys = {t: str(P.survey_path(args.realization, t, flux)) for t in tracers}
    fits = {}
    densities = {}
    if comp == "selection":
        for t in tracers:
            fits[t] = {"path": str(P.fit_path(t, flux)), "values": P.schechter_fit(P.fit_path(t, flux))}
    if comp in ("selection", "count_ratio"):
        for t in tracers:
            pname = "log10n_gal" if t == "gal" else "log10n_agn"
            factor = args.gal_density_factor if t == "gal" else args.agn_density_factor
            given = args.log10n_gal if t == "gal" else args.log10n_agn
            d = {"mode": "free" if pname in free else args.density, "units": args.density_units,
                 "factor": factor}
            if pname in free:
                d["prior"] = priors[pname]
            elif args.density == "given" or comp == "count_ratio":
                if given is None:
                    sys.exit(f"[fatal] --density given needs --log10{'n-gal' if t == 'gal' else 'n-agn'}")
                d["log10n_given"] = given
                d["log10n_used"] = given + math.log10(factor)
            elif args.density == "counts":
                if comp != "selection":
                    sys.exit("[fatal] densities from counts need a selection completeness")
                if args.density_units != "h_scaled":
                    sys.exit("[fatal] densities from counts are h-scaled (n0_units h_scaled)")
                c = P.density_from_counts(surveys[t], fits[t]["path"])
                d.update(c)
                d["log10n_used"] = c["log10n_h_scaled"] + math.log10(factor)
            else:
                sys.exit("[fatal] --density free needs log10n_gal/log10n_agn among the sampled parameters")
            densities[t] = d
    if args.analysis == "free_densities" and args.density != "free":
        args.density = "free"
    if args.redraw_key:
        key = [int(x) for x in args.redraw_key.split(",")]
    else:
        key = P.redraw_key(args.realization, args.events_kind, args.events_suffix, args.redraw_index)
    ndim = len(free)
    sample = args.sample or ("unif" if ndim <= 6 else "rslice")
    seed = args.seed if args.seed is not None else args.nested_index
    density_mode = args.density if (tracers and flux != "complete") else None
    name = args.run_name or P.run_name(args.analysis, args.catalogs, flux, args.events_kind, args.realization,
                                       density=density_mode or "counts",
                                       agn_density_factor=args.agn_density_factor, family=args.family_name,
                                       set_label=args.set_label, variant=args.variant)
    if args.membership and (args.catalogs != "both" or "f_agn" not in free or args.mode != "grid"):
        sys.exit("[fatal] --membership needs a two-tracer grid (both catalogs, f_AGN sampled)")
    cfg = {
        "run_name": name, "mode": args.mode, "analysis": args.analysis, "catalogs": args.catalogs,
        "tracers": list(tracers), "flux": flux, "events_kind": args.events_kind,
        "realization": args.realization, "events_suffix": args.events_suffix, "family_name": args.family_name,
        "events_file": events_file, "injections_file": inj_file,
        "injections_lane": None if args.injections_file else lane, "surveys": surveys,
        "injections_files": inj_files, "injection_lanes_pooled": list(args.pool_injections),
        "n_injection_sets": len(inj_files), "agn_subset": agn_subset, "variant": args.variant,
        "completeness_model": comp, "selection_fits": fits, "densities": densities,
        "density_mode": density_mode, "density_units": args.density_units if comp in ("selection", "count_ratio") else None,
        "zmax": float(args.zmax), "free": free, "priors": {k: priors[k] for k in free},
        "population_family": args.population_family, "population_values": pv,
        "population_fixed": {k: v for k, v in pv.items() if k not in free},
        "agn_offsets_truth": offsets_truth,
        "planted_f_agn": float(planted_f),
        "events_truth": ev_truth, "set_label": args.set_label,
        "class_populations": class_pops, "fixed_offsets": fixed_offsets,
        "membership": bool(args.membership),
        "nsamp": int(args.nsamp), "redraw_key": key if args.nsamp else None,
        "redraw_block": P.REDRAW_BLOCK, "half_variance_correction": args.half_variance_correction == "on",
        "max_var": float(args.max_var), "pe_event_block": int(args.pe_event_block), "sel_batch": int(args.sel_batch),
        "catalog_evaluation": {"kernel_window": 1e-10, "kernel_layout": "padded", "missing_density": "grid",
                               "pairing_norm": "per_sample", "pairing_scale": "analytic"},
        "Om0": P.OM0, "core": P.CORE, "limit_events": args.limit_events, "limit_injections": args.limit_injections,
        "dry_run": bool(args.dry_run),
    }
    if args.mode == "nested":
        cfg["nested"] = {"index": args.nested_index, "seed": seed, "nlive": args.nlive, "dlogz": args.dlogz,
                         "bound": "multi", "sample": sample, "bootstrap": args.bootstrap, "enlarge": args.enlarge,
                         "walks": args.walks, "slices": args.slices,
                         "first_update": {"min_ncall": 2 * args.nlive, "min_eff": 100.0},
                         "checkpoint_every": args.checkpoint_every, "neginf_to": -1e300}
    else:
        cfg["grid"] = {"explicit": args.grid, "coarse": args.coarse, "fine_cells": args.fine_cells,
                       "fine_cells_1d": args.fine_cells_1d, "fine_halfwidth_sd": args.fine_halfwidth_sd,
                       "outside_mass_limit": 1e-4}
        gp = [p for p in free]
        if len(gp) > 2:
            sys.exit(f"[fatal] grid mode takes at most two parameters, got {gp}")
    return cfg


# ============================================================================ records
class Session:
    """One process's record in <part>/sessions.json (GPU hours = sum of session walls)."""

    def __init__(self, part_dir):
        self.path = Path(part_dir) / "sessions.json"
        self.rows = json.loads(self.path.read_text()) if self.path.exists() else []
        self.row = {"slurm_job_id": os.environ.get("SLURM_JOB_ID"), "host": os.uname().nodename,
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(T_START)),
                    "calls": 0, "seconds_in_likelihood": 0.0, "exit": "running"}
        self.rows.append(self.row)
        self.save()

    def save(self):
        self.row["wall_seconds"] = time.time() - T_START
        self.row["end_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.rows, indent=1))
        tmp.replace(self.path)

    def close(self, how):
        self.row["exit"] = how
        self.save()

    @property
    def gpu_hours_total(self):
        return sum(r.get("wall_seconds", 0.0) for r in self.rows) / 3600.0


def write_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=_jsonable))
    tmp.replace(path)


def _jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def deadline_of(args):
    if args.deadline:
        return float(args.deadline)
    return float(os.environ.get("A16_DEADLINE", T_START + 1e8))


# ============================================================================ model setup
def setup_model(cfg):
    os.environ["DARKSIRENS_ZMAX"] = repr(float(cfg["zmax"]))
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import prod_model as PM
    m = PM.make_model(cfg)
    m.import_core()
    import jax
    jax.config.update("jax_enable_x64", True)
    t0 = time.time()
    m.build()
    m.bind()
    log(f"built + bound in {time.time() - t0:.0f}s; labels {m.labels}; n_events {m.n_events}; nsamp {m.nsamp}; "
        f"n_draw {m.n_draw:g}; devices {jax.devices()}")
    return m


def truth_coords(cfg):
    pv = cfg["population_values"]
    t = {"H0": 67.74, "f_agn": cfg["planted_f_agn"] if cfg["events_kind"] != "host_dependent" else 0.30}
    t.update({k: pv[k] for k in P.POP_FREE})
    t.update(cfg["agn_offsets_truth"])
    for tr, pname in (("gal", "log10n_gal"), ("agn", "log10n_agn")):
        d = cfg["densities"].get(tr)
        if d is not None and "log10n_used" in d:
            t[pname] = d["log10n_used"]
        else:
            t[pname] = {"log10n_gal": -3.0 + 3 * math.log10(100 / 67.74),
                        "log10n_agn": -5.0 + 3 * math.log10(100 / 67.74)}[pname]
    return {k: float(t[k]) for k in cfg["free"]}


def validation_points(cfg):
    tr = truth_coords(cfg)
    mid = {k: 0.5 * (lo + hi) for k, (lo, hi) in cfg["priors"].items()}
    if "f_agn" in mid:
        mid["f_agn"] = 0.6
    return [tr, mid]


# ============================================================================ grid mode
def nice_step(x, options):
    for s in options:
        if s >= x:
            return s
    return options[-1]


STEPS = {"H0": [0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5],
         "f_agn": [0.002, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05]}


def auto_coarse(free, priors):
    if len(free) == 1:
        lo, hi = priors[free[0]]
        return {free[0]: np.round(np.arange(lo, hi + 1e-9, 1.0 if free[0] == "H0" else 0.02), 10)}
    out = {}
    for p in free:
        lo, hi = priors[p]
        out[p] = np.round(np.arange(lo, hi + 1e-9, 2.0 if p == "H0" else 0.1), 10)
    return out


def axis_moments(x, logp):
    s = P.summarize_1d(x, logp)
    return s["mean"], s["sd"]


def marginal_logs(axes, lnL):
    names = list(axes)
    if len(names) == 1:
        return {names[0]: lnL}
    a, b = names
    la, lb = P.marginals_2d(axes[a], axes[b], lnL)
    return {a: la, b: lb}


def auto_fine(free, priors, coarse_axes, coarse_lnL, cells, halfwidth_sd):
    margs = marginal_logs(coarse_axes, coarse_lnL)
    out = {}
    for p in free:
        mean, sd = axis_moments(coarse_axes[p], margs[p])
        # Sheppard's correction for the coarse spacing (floor: a quarter of a coarse step)
        dx = float(np.median(np.diff(coarse_axes[p])))
        sd = float(np.sqrt(max(sd ** 2 - dx ** 2 / 12.0, (dx / 4.0) ** 2)))
        n = cells[p]
        lo_p, hi_p = priors[p]
        step = nice_step(2 * halfwidth_sd * sd / (n - 1), STEPS.get(p, [0.01, 0.1, 1.0]))
        width = step * (n - 1)
        if width >= hi_p - lo_p:
            out[p] = np.round(np.linspace(lo_p, hi_p, n), 10)   # the whole prior
            continue
        start = round((mean - 0.5 * width) / step) * step
        start = min(max(start, lo_p), hi_p - width)
        out[p] = np.round(start + step * np.arange(n), 10)
    return out


def outside_mass(fine_axes, fine_lnL, coarse_axes, coarse_lnL):
    """Posterior mass outside the fine box: coarse nodes outside it (midpoint rule) against the fine
    grid's trapezoid integral, on one absolute scale. Also the largest coarse ln L outside the box."""
    names = list(fine_axes)
    ref = max(np.nanmax(fine_lnL), np.nanmax(coarse_lnL))
    pf = np.where(np.isfinite(fine_lnL), np.exp(fine_lnL - ref), 0.0)
    pc = np.where(np.isfinite(coarse_lnL), np.exp(coarse_lnL - ref), 0.0)
    if len(names) == 1:
        x = fine_axes[names[0]]
        m_in = P._trapz(pf, x)
        xc = coarse_axes[names[0]]
        out = (xc < x[0] - 1e-9) | (xc > x[-1] + 1e-9)
        dx = np.gradient(xc)
        m_out = float(np.sum(pc[out] * dx[out]))
        lmax_out = float(np.max(coarse_lnL[out])) if out.any() else -np.inf
    else:
        a, b = names
        m_in = P._trapz(P._trapz(pf, fine_axes[b], axis=1), fine_axes[a])
        A, B = np.meshgrid(coarse_axes[a], coarse_axes[b], indexing="ij")
        out = ((A < fine_axes[a][0] - 1e-9) | (A > fine_axes[a][-1] + 1e-9)
               | (B < fine_axes[b][0] - 1e-9) | (B > fine_axes[b][-1] + 1e-9))
        da = np.gradient(coarse_axes[a])[:, None] * np.ones_like(A)
        db = np.gradient(coarse_axes[b])[None, :] * np.ones_like(B)
        m_out = float(np.sum((pc * da * db)[out]))
        lmax_out = float(np.max(coarse_lnL[out])) if out.any() else -np.inf
    frac = m_out / (m_in + m_out) if (m_in + m_out) > 0 else float("nan")
    return {"mass_outside_fraction": frac, "mass_inside": float(m_in), "mass_outside": m_out,
            "lnL_max_fine": float(np.nanmax(fine_lnL)), "lnL_max_coarse_outside": lmax_out,
            "dlnL_outside_minus_peak": lmax_out - float(np.nanmax(fine_lnL)), "ln_ref": float(ref)}


def run_pass(m, cfg, gdir, name, axes, sess, deadline, max_cells=None):
    """Evaluate every cell of one grid pass (resumable); -> dict of arrays shaped like the grid."""
    import h5py
    names = list(axes)
    shape = tuple(len(axes[n]) for n in names)
    cells = [dict(zip(names, map(float, c))) for c in np.array(np.meshgrid(*[axes[n] for n in names],
                                                                          indexing="ij")).reshape(len(names), -1).T]
    keys = ("lnL", "pe_var", "n_eff", "log_mu", "sel_corr", "seconds") + tuple(getattr(m, "extra_keys", ()))
    partial = gdir / f"{name}.partial.npz"
    res = {k: np.full(len(cells), np.nan) for k in keys}
    if partial.exists():
        z = np.load(partial)
        if all(np.array_equal(z[f"axis_{n}"], axes[n]) for n in names) and all(k in z for k in keys):
            for k in keys:
                res[k] = np.asarray(z[k])
            log(f"[{name}] resuming: {int(np.isfinite(res['seconds']).sum())}/{len(cells)} cells on record")
        else:
            log(f"[{name}] partial file has other axes; starting over")
    todo = [i for i in range(len(cells)) if not np.isfinite(res["seconds"][i])]
    if max_cells is not None:
        todo = todo[:max_cells]

    def dump():
        np.savez(partial, **{f"axis_{n}": axes[n] for n in names}, **res)

    t_last = time.time()
    for j, i in enumerate(todo):
        if time.time() > deadline - 120:
            dump()
            sess.save()
            return None
        tc = time.time()
        r = m.evaluate(cells[i])
        dt = time.time() - tc
        for k in keys:
            if k != "seconds":
                res[k][i] = r[k]
        res["seconds"][i] = dt
        sess.row["calls"] += 1
        sess.row["seconds_in_likelihood"] += dt
        if time.time() - t_last > 60 or j == len(todo) - 1:
            dump()
            sess.save()
            t_last = time.time()
    if max_cells is not None and np.isnan(res["seconds"]).any():
        # dry run: fill the rest from the evaluated cells is not meaningful; keep NaN
        pass
    out = {k: res[k].reshape(shape) for k in keys}
    with h5py.File(gdir / f"{name}.h5", "w") as f:
        for n in names:
            f.create_dataset(f"axis_{n}", data=axes[n])
        for k in keys:
            f.create_dataset(k, data=out[k])
        f.attrs["axes"] = json.dumps(names)
        f.attrs["n_events"] = m.n_events
    return out


def corrected(cfg, m, res):
    if not cfg["half_variance_correction"]:
        return res["lnL"]
    return P.half_variance_corrected(res["lnL"], res["pe_var"], res["n_eff"], m.n_events)


def grid_summary(axes, lnL):
    names = list(axes)
    if len(names) == 1:
        return {names[0]: P.summarize_1d(axes[names[0]], lnL)}
    return P.summarize_2d(axes[names[0]], axes[names[1]], lnL, names=tuple(names))


def run_grid(args, cfg, run_dir):
    gdir = run_dir / ("grid" + (f"_{args.tag}" if args.tag else ""))
    gdir.mkdir(parents=True, exist_ok=True)
    write_json(gdir / "config.json", cfg)
    if (gdir / "result.json").exists() and json.loads((gdir / "result.json").read_text()).get("finished"):
        rec = json.loads((gdir / "result.json").read_text())
        if cfg.get("membership") and not (rec.get("membership") or {}).get("ok"):
            # the grid is on record without the membership output: compute it alone
            log(f"{gdir} finished on record; computing the membership probabilities")
            m = setup_model(cfg)
            rec["membership"] = membership_safe(m, cfg, rec["summary"], gdir)
            write_json(gdir / "result.json", rec)
            return 0
        log(f"{gdir} finished on record; nothing to do")
        return 0
    sess = Session(gdir)
    deadline = deadline_of(args)
    m = setup_model(cfg)
    cfg_info = dict(m.info)
    checks = m.check_diag_equals_bound(validation_points(cfg))
    free = cfg["free"]
    priors = cfg["priors"]
    explicit = dict(parse_axis(s) for s in args.grid)
    passes = {}
    if args.coarse != "none":
        coarse_axes = auto_coarse(free, priors) if args.coarse == "auto" else \
            dict(parse_axis(s) for s in args.coarse.split(";"))
        if args.dry_run:
            coarse_axes = {k: v[[0, len(v) // 2, -1]] for k, v in coarse_axes.items()}
        log(f"coarse pass: {{{', '.join(f'{k}: {v[0]}..{v[-1]} ({len(v)})' for k, v in coarse_axes.items())}}}")
        cres = run_pass(m, cfg, gdir, "coarse", coarse_axes, sess, deadline, args.max_cells)
        if cres is None:
            sess.close("deadline")
            return EXIT_DEADLINE
        passes["coarse"] = (coarse_axes, cres)
    if explicit:
        fine_axes = {p: explicit[p] for p in free}
    else:
        if "coarse" not in passes:
            sys.exit("[fatal] no fine grid given and no coarse pass to place one")
        cells = dict(s.split("=") for s in args.fine_cells.split(","))
        cells = {k: int(v) for k, v in cells.items()}
        if len(free) == 1:
            cells = {free[0]: args.fine_cells_1d}
        ca, cr = passes["coarse"]
        fine_axes = auto_fine(free, priors, ca, corrected(cfg, m, cr), cells, args.fine_halfwidth_sd)
    if args.dry_run:
        fine_axes = {k: v[[0, len(v) // 2, -1]] for k, v in fine_axes.items()}
    log(f"fine pass: {{{', '.join(f'{k}: {v[0]}..{v[-1]} step {v[1] - v[0]:.4g} ({len(v)})' for k, v in fine_axes.items())}}}")
    fres = run_pass(m, cfg, gdir, "fine", fine_axes, sess, deadline, args.max_cells)
    if fres is None:
        sess.close("deadline")
        return EXIT_DEADLINE
    out_check = None
    if "coarse" in passes:
        ca, cr = passes["coarse"]
        out_check = outside_mass(fine_axes, corrected(cfg, m, fres), ca, corrected(cfg, m, cr))
        widen = (not explicit and not args.dry_run and
                 (not (out_check["mass_outside_fraction"] < 1e-4)))
        if widen:
            log(f"outside mass {out_check['mass_outside_fraction']:.2e} >= 1e-4: widening the fine grid once")
            cells = {k: len(v) for k, v in fine_axes.items()}
            fine_axes = auto_fine(free, priors, ca, corrected(cfg, m, cr), cells, 1.6 * args.fine_halfwidth_sd)
            fres = run_pass(m, cfg, gdir, "fine_widened", fine_axes, sess, deadline, args.max_cells)
            if fres is None:
                sess.close("deadline")
                return EXIT_DEADLINE
            out_check = outside_mass(fine_axes, corrected(cfg, m, fres), ca, corrected(cfg, m, cr))
            out_check["widened"] = True
    lnc = corrected(cfg, m, fres)
    summ = grid_summary(fine_axes, lnc)
    summ_raw = grid_summary(fine_axes, fres["lnL"])
    vol = float(np.prod([priors[p][1] - priors[p][0] for p in free]))
    ln_z = None
    if out_check is not None:
        ln_z = out_check["ln_ref"] + math.log(out_check["mass_inside"] + out_check["mass_outside"]) - math.log(vol)
    ok = np.isfinite(fres["lnL"])
    peak = np.unravel_index(int(np.nanargmax(np.where(ok, lnc, -np.inf))), lnc.shape)
    res = {
        "finished": not args.dry_run, "dry_run": bool(args.dry_run), "run_name": cfg["run_name"],
        "mode": "grid", "free": free, "summary": summ, "summary_uncorrected": summ_raw,
        "half_variance_correction": cfg["half_variance_correction"],
        "fine_axes": {k: [float(v[0]), float(v[-1]), float(v[1] - v[0]) if len(v) > 1 else 0.0, len(v)]
                      for k, v in fine_axes.items()},
        "coarse_axes": ({k: [float(v[0]), float(v[-1]), len(v)] for k, v in passes["coarse"][0].items()}
                        if "coarse" in passes else None),
        "outside_mass_check": out_check,
        "outside_mass_ok": None if out_check is None else bool(out_check["mass_outside_fraction"] < 1e-4),
        "ln_evidence_grid": ln_z,
        "variance": {"pe_var_at_peak": float(fres["pe_var"][peak]), "n_eff_at_peak": float(fres["n_eff"][peak]),
                     "sigma2_sel_at_peak": float(fres["sigma2_sel"][peak] if "sigma2_sel" in fres
                                                 else m.n_events ** 2 / fres["n_eff"][peak]),
                     "pe_var_range": [float(np.nanmin(fres["pe_var"])), float(np.nanmax(fres["pe_var"]))],
                     "n_eff_range": [float(np.nanmin(fres["n_eff"])), float(np.nanmax(fres["n_eff"]))],
                     "n_guarded_cells": int((~ok).sum()), "max_var": cfg["max_var"]},
        "lnL_max": float(np.nanmax(fres["lnL"])), "calls_total": int(sum(r.get("calls", 0) for r in sess.rows)),
        "seconds_per_call_median": float(np.nanmedian(fres["seconds"])),
        "model_info": cfg_info, "diag_vs_bound": checks,
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if extra := getattr(m, "extra_keys", ()):
        res["extra_at_peak"] = {k: float(fres[k][peak]) for k in extra}
    if cfg.get("membership"):
        res["membership"] = membership_safe(m, cfg, summ, gdir)
    sess.close("finished" if not args.dry_run else "dry_run")
    res["gpu_hours_total"] = sess.gpu_hours_total
    res["sessions"] = sess.rows
    write_json(gdir / "result.json", res)
    s = summ
    log("grid result: " + "  ".join(f"{p} {s[p]['median']:.4f} [{s[p]['ci90'][0]:.4f}, {s[p]['ci90'][1]:.4f}]"
                                    for p in free) + (f"  outside mass {out_check['mass_outside_fraction']:.2e}"
                                                       if out_check else ""))
    return 0


# ============================================================================ membership
def membership_safe(m, cfg, summ, gdir):
    try:
        return membership(m, cfg, summ, gdir)
    except Exception as e:  # noqa: BLE001  (the grid is the expensive part; keep it)
        import traceback
        log(f"[membership] failed: {e!r}")
        return {"ok": False, "error": repr(e), "traceback": traceback.format_exc()}


def membership(m, cfg, summ, gdir):
    """Each merger's probability of an AGN host at the posterior median (H0, f):

        r_i = f L_i,agn / (f L_i,agn + (1 - f) L_i,gal),

    L_i,c the merger's likelihood under class c alone: core's per-event evidence of the same mixture
    model at f = 1 (AGN branch) and f = 0 (galaxy branch), each branch normalised over its catalog (and
    carrying its class population in two_tracers_two_populations). Written with the true host class.
    Checks (exact identities of the model, recorded): (i) the per-event evidence at the median f equals
    ln[(1 - f) L_gal + f L_agn] and ln mu(f) = ln[(1 - f) mu_gal + f mu_agn]; (ii) at the f* that
    maximises ln L(f | H0_median), mean_i r_i(f*) = f* mu_agn / mu(f*) (the detected AGN fraction the
    model implies; = f* when mu_agn = mu_gal)."""
    import prod_model as PM
    h0 = float(summ["H0"]["median"])
    f = float(summ["f_agn"]["median"])
    r0 = m.evaluate_events({"H0": h0, "f_agn": 0.0})
    r1 = m.evaluate_events({"H0": h0, "f_agn": 1.0})
    rf = m.evaluate_events({"H0": h0, "f_agn": f})
    lG, lA = r0["event_log_evidence"], r1["event_log_evidence"]
    lmG, lmA = r0["log_mu"], r1["log_mu"]
    lmix = np.logaddexp(np.log1p(-f) + lG, np.log(f) + lA)
    r = np.exp(np.log(f) + lA - lmix)
    id_events = float(np.max(np.abs(rf["event_log_evidence"] - lmix)))
    id_mu = float(abs(rf["log_mu"] - np.logaddexp(np.log1p(-f) + lmG, np.log(f) + lmA)))
    n = len(lG)

    def lnl(fx):
        return float(np.sum(np.logaddexp(np.log1p(-fx) + lG, np.log(fx) + lA))
                     - n * np.logaddexp(np.log1p(-fx) + lmG, np.log(fx) + lmA))

    def dlnl(fx):
        mix = np.logaddexp(np.log1p(-fx) + lG, np.log(fx) + lA)
        mu = np.logaddexp(np.log1p(-fx) + lmG, np.log(fx) + lmA)
        return float(np.sum(np.exp(lA - mix) - np.exp(lG - mix)) - n * (np.exp(lmA - mu) - np.exp(lmG - mu)))

    lo, hi = 1e-9, 1 - 1e-9
    if dlnl(lo) <= 0:
        fstar = 0.0
    elif dlnl(hi) >= 0:
        fstar = 1.0
    else:
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if dlnl(mid) > 0:
                lo = mid
            else:
                hi = mid
        fstar = 0.5 * (lo + hi)
    ident_ml = None
    if 0.0 < fstar < 1.0:
        mix_s = np.logaddexp(np.log1p(-fstar) + lG, np.log(fstar) + lA)
        r_s = np.exp(np.log(fstar) + lA - mix_s)
        mu_s = np.logaddexp(np.log1p(-fstar) + lmG, np.log(fstar) + lmA)
        det_frac = float(np.exp(np.log(fstar) + lmA - mu_s))
        ident_ml = {"f_star": fstar, "mean_r_at_f_star": float(r_s.mean()),
                    "f_star_mu_agn_over_mu": det_frac, "difference": float(r_s.mean() - det_frac)}
    ht = PM.event_host_types(cfg["events_file"], cfg.get("limit_events"))
    mu_f = np.logaddexp(np.log1p(-f) + lmG, np.log(f) + lmA)
    out = {"ok": True, "H0_median": h0, "f_agn_median": f, "n_events": n,
           "mean_r": float(r.mean()), "f_median_mu_agn_over_mu": float(np.exp(np.log(f) + lmA - mu_f)),
           "mu_agn_over_mu_gal": float(np.exp(lmA - lmG)),
           "true_agn_fraction": float(ht.mean()), "n_true_agn": int(ht.sum()),
           "mean_r_true_agn": float(r[ht == 1].mean()) if (ht == 1).any() else None,
           "mean_r_true_gal": float(r[ht == 0].mean()) if (ht == 0).any() else None,
           "n_r_gt_0p5": int((r > 0.5).sum()), "n_true_agn_with_r_gt_0p5": int(((r > 0.5) & (ht == 1)).sum()),
           "identity_event_evidence_max_abs": id_events, "identity_log_mu_abs": id_mu,
           "identity_ml": ident_ml, "lnL_at_median_from_branches": lnl(f), "lnL_at_median_core": rf["lnL"],
           "file": str(gdir / "membership.npz")}
    np.savez(gdir / "membership.npz", event=np.arange(n), host_type=ht, r=r, ln_L_agn=lA, ln_L_gal=lG,
             mc_var_agn=r1["event_mc_variance"], mc_var_gal=r0["event_mc_variance"],
             ln_mu_agn=lmA, ln_mu_gal=lmG, H0=h0, f_agn=f)
    write_json(gdir / "membership.json", out)
    log(f"[membership] H0 {h0:.3f} f {f:.4f}: mean r {out['mean_r']:.4f} (true AGN {out['mean_r_true_agn']}, "
        f"true galaxy {out['mean_r_true_gal']}); identities {id_events:.1e}, {id_mu:.1e}; ML {ident_ml}")
    return out


# ============================================================================ nested mode
def switch_sampler(s, method, enlarge=None, bootstrap=None, walks=None, slices=None):
    """Rewire a restored dynesty 2.1.x MultiEllipsoidSampler to another proposal (generalises
    Analysis 13's a13_switch.switch_sampler, which wires rslice only); -> record."""
    from dynesty.dynamicsampler import _get_update_interval_ratio
    from dynesty.dynesty import _get_walks_slices
    from dynesty.nestedsamplers import _SAMPLING
    from dynesty.utils import get_enlarge_bootstrap
    rec = {"from": s.method, "to": method, "it": int(s.it), "ncall": int(s.ncall),
           "bootstrap_before": int(s.bootstrap), "enlarge_before": float(s.enlarge)}
    w, sl = _get_walks_slices(walks, slices, method, s.ncdim)
    s.method = s.sampling = method
    s.evolve_point = _SAMPLING[method]
    s.propose_point = s._PROPOSE[method]
    s.update_proposal = s._UPDATE[method]
    s.enlarge, s.bootstrap = get_enlarge_bootstrap(method, enlarge, bootstrap)
    s.kwargs.update(enlarge=s.enlarge, bootstrap=s.bootstrap)
    if w is not None:
        s.walks = max(2, int(w))
        s.kwargs["walks"] = s.walks
        s.facc = min(1.0, max(1.0 / s.walks, s.facc))
    if sl is not None:
        s.slices = int(sl)
        s.kwargs["slices"] = s.slices
    s.scale = 1.0
    s.rwalk_history = {"naccept": 0, "nreject": 0}
    s.slice_history = {"ncontract": 0, "nexpand": 0}
    ratio = _get_update_interval_ratio(None, method, "multi", s.ncdim, s.nlive, s.slices, s.walks)
    s.bound_update_interval = int(max(min(np.round(ratio * s.nlive), sys.maxsize), 1))
    s.queue, s.nqueue = [], 0
    s.update_bound_if_needed(-np.inf, force=True)
    rec.update(enlarge=float(s.enlarge), bootstrap=int(s.bootstrap), walks=int(s.walks), slices=int(s.slices),
               bound_update_interval=int(s.bound_update_interval), n_ellipsoids=int(len(s.mell.ells)))
    return rec


def sampler_settings(s):
    return {"method": s.method, "bootstrap": int(s.bootstrap), "enlarge": float(s.enlarge),
            "walks": int(getattr(s, "walks", -1)), "slices": int(getattr(s, "slices", -1)),
            "bound_update_interval": int(s.bound_update_interval), "nlive": int(s.nlive),
            "first_update": {"min_ncall": int(s.first_bound_update_ncall),
                             "min_eff": float(s.first_bound_update_eff)}}


def delta_logz(s):
    sr = s.saved_run
    if len(sr["logl"]) == 0:
        return float("inf")
    return float(np.logaddexp(0, np.max(s.live_logl) + sr["logvol"][-1] - sr["logz"][-1]))


def run_nested(args, cfg, run_dir):
    import dynesty
    from dynesty.utils import resample_equal
    nk = cfg["nested"]
    part = f"nested_{nk['index']}" + (f"_{args.tag}" if args.tag else "")
    ndir = run_dir / part
    ndir.mkdir(parents=True, exist_ok=True)
    write_json(ndir / "config.json", cfg)
    if (ndir / "result.json").exists() and json.loads((ndir / "result.json").read_text()).get("finished"):
        log(f"{ndir} finished on record; nothing to do")
        return 0
    sess = Session(ndir)
    deadline = deadline_of(args)
    m = setup_model(cfg)
    checks = m.check_diag_equals_bound(validation_points(cfg))
    free = cfg["free"]
    lo = np.array([cfg["priors"][p][0] for p in free], float)
    hi = np.array([cfg["priors"][p][1] for p in free], float)
    corr_on = cfg["half_variance_correction"]
    cnt = {"n": 0, "t": 0.0, "neginf": 0}
    kill_after = args.kill_after_calls

    def loglike(x):
        t1 = time.time()
        r = m.evaluate(dict(zip(free, map(float, x))))
        dt = time.time() - t1
        cnt["n"] += 1
        cnt["t"] += dt
        sess.row["calls"] += 1
        sess.row["seconds_in_likelihood"] += dt
        blob = np.array([r["lnL"], r["pe_var"], r["n_eff"], r["log_mu"], r["sel_corr"]])
        v = r["lnL_corrected"] if corr_on else r["lnL"]
        if kill_after is not None and cnt["n"] >= kill_after:
            sess.close(f"killed_by_test_after_{cnt['n']}_calls")
            log(f"[test] hard exit after {cnt['n']} calls (simulated kill; resume from the last checkpoint)")
            os._exit(137)
        if not np.isfinite(v):
            cnt["neginf"] += 1
            return -1e300, blob
        return v, blob

    def ptform(u):
        return lo + u * (hi - lo)

    from darksirens.inference.dynesty_checkpoint import install_dynesty_checkpointing, restore_dynesty_sampler
    ckpt = ndir / "state.save"
    switch_rec = None
    if args.init_from and not ckpt.exists():
        # restore the given state, switch, and only then write this part's own checkpoint
        s = restore_dynesty_sampler(str(args.init_from), loglike, ptform)
        log(f"[test] initial state restored from {args.init_from} (it {s.it - 1}, ncall {s.ncall})")
        if args.switch_sample:
            switch_rec = switch_sampler(s, args.switch_sample, enlarge=args.enlarge, bootstrap=args.bootstrap,
                                        walks=args.walks, slices=args.slices)
            log(f"[test] sampler switched: {switch_rec}")
        write_json(ndir / "init_from.json", {"init_from": args.init_from, "switch": switch_rec,
                                             "it": int(s.it), "ncall": int(s.ncall)})
        install_dynesty_checkpointing(s)
        s.save(str(ckpt))
    if ckpt.exists():
        s = restore_dynesty_sampler(str(ckpt), loglike, ptform)
        install_dynesty_checkpointing(s)
        log(f"resumed {ckpt}: it {s.it}, ncall {s.ncall}, delta_logz {delta_logz(s):.3f}")
    else:
        kw = {}
        for k in ("bootstrap", "enlarge", "walks", "slices"):
            if nk[k] is not None:
                kw[k] = nk[k]
        s = dynesty.NestedSampler(loglike, ptform, len(free), nlive=nk["nlive"], bound="multi",
                                  sample=nk["sample"], rstate=np.random.default_rng(nk["seed"]),
                                  first_update=dict(nk["first_update"]), blob=True, **kw)
        install_dynesty_checkpointing(s)
        log(f"new sampler: {sampler_settings(s)}")
    settings_now = sampler_settings(s)
    save_at = sorted(int(x) for x in args.save_at_iterations.split(",") if x)
    saved_copies = []
    maxcall_total = args.maxcall_total
    maxiter_total = args.maxiter_total
    if args.maxcall_more is not None:
        maxcall_total = int(s.ncall) + args.maxcall_more
    if args.maxiter_more is not None:
        maxiter_total = int(s.it - 1) + args.maxiter_more
    start_state = {"it": int(s.it - 1), "ncall": int(s.ncall)}
    status = "running"
    spc = 0.5
    while True:
        dlz = delta_logz(s) if s.it > 1 else float("inf")
        if dlz < nk["dlogz"]:
            status = "converged"
            break
        if maxcall_total is not None and s.ncall >= maxcall_total:
            status = "maxcall"
            break
        if maxiter_total is not None and s.it - 1 >= maxiter_total:
            status = "maxiter"
            break
        chunk = int(args.chunk_calls)
        if maxcall_total is not None:
            chunk = max(1, min(chunk, maxcall_total - s.ncall))
        if cnt["n"] > 20:
            spc = cnt["t"] / cnt["n"]
        if time.time() + 1.5 * chunk * spc + 300 > deadline:
            status = "deadline"
            break
        maxiter = None
        if save_at:
            nxt = [i for i in save_at if i > s.it - 1]
            if nxt:
                maxiter = max(1, nxt[0] - (s.it - 1))
        if maxiter_total is not None:
            rem = maxiter_total - (s.it - 1)
            maxiter = rem if maxiter is None else min(maxiter, rem)
        s.run_nested(maxcall=chunk, maxiter=maxiter if maxiter is None else maxiter - 1, dlogz=nk["dlogz"],
                     add_live=False, print_progress=False, checkpoint_file=str(ckpt),
                     checkpoint_every=nk["checkpoint_every"], resume=True)
        s.save(str(ckpt))
        sess.save()
        it_done = s.it - 1
        if save_at and it_done in save_at and not (ndir / f"state_it{it_done}.save").exists():
            shutil.copy2(ckpt, ndir / f"state_it{it_done}.save")
            saved_copies.append(it_done)
            log(f"[test] state copied at iteration {it_done} (ncall {s.ncall})")
        log(f"it {it_done}  ncall {s.ncall}  delta_logz {delta_logz(s):.4g}  logz {s.saved_run['logz'][-1]:.6g}  "
            f"logl* {s.saved_run['logl'][-1]:.6g}  {spc:.3f} s/call  eff {100 * it_done / max(s.ncall, 1):.1f}%")
    s.save(str(ckpt))
    rec_base = {"run_name": cfg["run_name"], "mode": "nested", "part": part, "free": free,
                "priors": cfg["priors"], "sampler": settings_now, "sampler_switch": switch_rec,
                "diag_vs_bound": checks, "model_info": m.info, "saved_copies": saved_copies,
                "start_state_this_session": start_state,
                "n_neginf_this_session": cnt["neginf"], "calls_this_session": cnt["n"],
                "seconds_per_call_this_session": cnt["t"] / max(cnt["n"], 1)}
    if status != "converged":
        rec = {**rec_base, "finished": False, "status": status, "it": int(s.it - 1), "ncall": int(s.ncall),
               "delta_logz": delta_logz(s), "logz_so_far": float(s.saved_run["logz"][-1]) if s.it > 1 else None,
               "dry_run": bool(args.dry_run)}
        sess.close(status)
        rec["gpu_hours_total"] = sess.gpu_hours_total
        write_json(ndir / "progress.json", rec)
        log(f"stopped ({status}) at it {s.it - 1}, ncall {s.ncall}; resubmit to continue")
        return 0 if status in ("maxcall", "maxiter") else EXIT_DEADLINE
    # converged: add the final live points the way run_nested does, then write the products
    s.run_nested(dlogz=nk["dlogz"], add_live=True, print_progress=False, checkpoint_file=str(ckpt),
                 checkpoint_every=nk["checkpoint_every"], resume=True)
    res = s.results
    with open(ndir / "results.pkl", "wb") as fh:
        pickle.dump(res, fh)
    logwt = np.asarray(res["logwt"])
    w = np.exp(logwt - logwt.max())
    w /= w.sum()
    samples = np.asarray(res["samples"])
    blobs = np.asarray(res["blob"])
    rng = np.random.default_rng(1000 + nk["seed"])
    eq = resample_equal(samples, w, rstate=rng)
    # the posterior without the half-variance correction: reweight by exp(lnL_raw - lnL_sampled)
    lnl = np.asarray(res["logl"])
    raw = blobs[:, 0]
    dw = np.where(np.isfinite(raw) & (lnl > -1e299), raw - lnl, -np.inf)
    w_raw = w * np.exp(dw - np.max(dw[np.isfinite(dw)]))
    w_raw /= w_raw.sum()
    np.savez(ndir / "samples.npz", samples=samples, names=np.array(free), logwt=logwt, logl=lnl, blob=blobs,
             blob_names=np.array(["lnL", "pe_var", "n_eff", "log_mu", "sel_corr"]), weights=w,
             weights_uncorrected=w_raw, equal_weight=eq, samples_u=np.asarray(res["samples_u"]),
             ncall=np.asarray(res["ncall"]), samples_it=np.asarray(res["samples_it"]))
    ess = float(1.0 / np.sum(w ** 2))
    pe_var_post = float(np.sum(w * blobs[:, 1]))
    sel_post = float(np.sum(w * m.n_events ** 2 / blobs[:, 2]))
    rec = {**rec_base, "finished": not args.dry_run, "status": "converged", "dry_run": bool(args.dry_run),
           "logZ": float(res["logz"][-1]), "logZerr": float(res["logzerr"][-1]), "niter": int(res["niter"]),
           "ncall_total": int(np.sum(res["ncall"])), "eff_percent": float(res["eff"]),
           "ess_kish": ess, "summary": P.summarize_samples(samples, free, weights=w),
           "summary_uncorrected": P.summarize_samples(samples, free, weights=w_raw),
           "half_variance_correction": corr_on,
           "variance_posterior_mean": {"pe_var": pe_var_post, "sigma2_sel": sel_post,
                                       "sigma2_total": pe_var_post + sel_post, "max_var": cfg["max_var"]},
           "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    sess.close("finished" if not args.dry_run else "dry_run")
    rec["gpu_hours_total"] = sess.gpu_hours_total
    rec["sessions"] = sess.rows
    write_json(ndir / "result.json", rec)
    log("nested result: logZ %.3f +- %.3f, ncall %d; " % (rec["logZ"], rec["logZerr"], rec["ncall_total"]) +
        "  ".join(f"{p} {rec['summary'][p]['median']:.4f}" for p in free))
    return 0


# ============================================================================ main
def main(argv=None):
    args = parse_args(argv)
    cfg = resolve(args)
    run_dir = Path(args.out_root) / cfg["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "config.json", cfg)
    log(f"run {cfg['run_name']}  mode {cfg['mode']}  free {cfg['free']}  events {cfg['events_file']}  "
        f"nsamp {cfg['nsamp']}  zmax {cfg['zmax']}  job {os.environ.get('SLURM_JOB_ID')}")
    if not Path(cfg["events_file"]).exists():
        log(f"[missing data] {cfg['events_file']} does not exist")
        return 4
    for f in cfg["injections_files"] + ((cfg.get("agn_subset") or {}).get("injections_files") or []):
        if not P.injections_ready(f):
            log(f"[missing data] {f} (or its _meta.json) does not exist")
            return 4
    if cfg["mode"] == "grid":
        return run_grid(args, cfg, run_dir)
    return run_nested(args, cfg, run_dir)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda *a: sys.exit(EXIT_DEADLINE))
    sys.exit(main())
