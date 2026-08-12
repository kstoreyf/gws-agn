#!/usr/bin/env python
"""EXPERIMENT COPY of analysis_5/scripts/sample_4d.py, retargeted at darksirens
origin/master (e8d5035) to cross-check the 4-parameter fit after the 2026-08
completeness work.  Deltas from the analysis-5 original, all marked `# [exp]`:

  1. A4_SCRIPTS path gains a .parent (this copy sits one directory deeper).
  2. --c_mode selects the completeness estimator and is threaded to BOTH opts
     and build_parameter_space, then recorded in every output.  It defaults to
     "aggregate" (the current sky-anchored mode) rather than the library's
     "per_pixel" (retained for legacy support only).
  3. --wiring_nonfatal: report the wiring diff instead of asserting on it.  A
     likelihood change is the QUANTITY UNDER TEST here, not an error.
  4. --probe_only: build the closure, run the wiring probe, write
     <out_tag>_probe.json and exit before the sampler.  Minutes, not hours.

Everything else — priors, guard, KDE window, nuisance defaults, the sampler
call and the output contract — is byte-identical to analysis 5.

Original docstring follows.

Gate-0 pilot for analysis 5: sample the joint 4-parameter posterior
(H0, log10n0 [GAL], log10n0_c2 [AGN], fcat_2 = f_AGN) of the K = 2 darksirens
mixture with BOTH completion-density anchors free, under flat priors.

The likelihood is the exact closure analyses 3/4 grid-scanned (their
scan_h0f.py builds it; here the grid loop is swapped for a sampler).  Before
any sampling the closure is validated pointwise against stored analysis-3/4
grid values (--wiring_ref), including f = 0 and f = 1 endpoint cells.

Two cheap lanes (Gate 0 of the analysis-5 proposal):
  --sampler dynesty   static nested sampling, low nlive, dlogz, hard maxcall cap
  --sampler emcee     short ensemble chain initialized in a tight ball at the
                      known peak (grid MAP from --init_ref_h5 x anchor truths)

Outputs <outdir>/<out_tag>.h5 (post-burn / equal-weight samples + raw sampler
products + provenance) and <outdir>/<out_tag>.json (marginal summaries, truth
flags, call counts and the measured s/eval that prices the full campaign).
"""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
# This copy lives under analyses/experiments/<exp>/scripts, one level deeper than
# analysis_5's, so the hop up to the analyses root gains a .parent.
# analysis 4 was moved to working/archive/ on 2026-08-10 when the c_mode
# re-check retired the per_pixel-era ladder; this copy still imports its
# scan_h0f constants/helpers, so it follows the move. The analyses/ location is
# tried first so an un-archived checkout keeps working.
_A4 = "analysis_4_density_anchoring_H0_fagn"
_ANALYSES = HERE.parent.parent.parent                 # working/analyses
A4_SCRIPTS = next(
    (p for p in (_ANALYSES / _A4 / "scripts",
                 _ANALYSES.parent / "archive" / _A4 / "scripts")
     if (p / "scan_h0f.py").exists()),
    _ANALYSES / _A4 / "scripts")
sys.path.insert(0, str(A4_SCRIPTS))
import scan_h0f  # noqa: E402  (constants + helpers only; its work is under main())

DARKSIRENS_REPO = scan_h0f.DARKSIRENS_REPO
MERGE_SHA = scan_h0f.MERGE_SHA
OM0_FID = scan_h0f.OM0_FID
NUISANCE_DEFAULTS = scan_h0f.NUISANCE_DEFAULTS

SAMPLED = ("H0", "log10n0", "log10n0_c2", "fcat_2")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # ---- darksirens configuration (names match scan_h0f.py) ----
    p.add_argument("--universe_model", default="dark_sirens",
                   choices=["dark_sirens", "dark_sirens_complete"])
    p.add_argument("--survey_path", nargs="+", required=True)
    p.add_argument("--gw_path", required=True)
    p.add_argument("--gwselection_path", required=True)
    p.add_argument("--catalog_sky_weighting", default="field")
    p.add_argument("--selection_neff_guard", default="hard",
                   choices=["auto", "hard", "soft"])
    p.add_argument("--max_likelihood_variance", type=float, default=None)
    p.add_argument("--sel_batch_size", type=int, default=None)
    p.add_argument("--pe_event_block", type=int, default=None)
    p.add_argument("--kde_window", type=int, default=None)
    p.add_argument("--kde_window_nsigma", type=float, default=8.0)
    p.add_argument("--gamma", type=float, default=None)
    p.add_argument("--nuisance_json", default=None)
    p.add_argument("--device", default="gpu", choices=["gpu", "cpu"])
    # ---- flat priors on the four sampled parameters ----
    p.add_argument("--h0_prior", nargs=2, type=float, default=[50.0, 100.0])
    p.add_argument("--f_prior", nargs=2, type=float, default=[0.0, 1.0])
    p.add_argument("--n0_prior", nargs=2, type=float, default=[-4.0, -2.0],
                   help="flat prior on log10n0 (GAL anchor)")
    p.add_argument("--n0c2_prior", nargs=2, type=float, default=[-6.0, -4.0],
                   help="flat prior on log10n0_c2 (AGN anchor)")
    # ---- sampler ----
    p.add_argument("--sampler", required=True, choices=["dynesty", "emcee"])
    p.add_argument("--rstate_seed", type=int, default=7)
    p.add_argument("--checkpoint_file", default=None,
                   help="dynesty checkpoint path; if it exists, the run RESUMES")
    p.add_argument("--checkpoint_every", type=float, default=900.0)
    p.add_argument("--nlive", type=int, default=200)
    p.add_argument("--dlogz", type=float, default=0.1)
    p.add_argument("--maxcall", type=int, default=80_000)
    p.add_argument("--nwalkers", type=int, default=32)
    p.add_argument("--nsteps", type=int, default=1500)
    p.add_argument("--burn_frac", type=float, default=0.5)
    p.add_argument("--init_ref_h5", default=None,
                   help="grid h5 whose (H0, f) MAP centres the emcee init ball")
    # ---- wiring validation ----
    p.add_argument("--wiring_ref", action="append", default=[],
                   help="grid h5 to reproduce pointwise before sampling (repeatable)")
    p.add_argument("--wiring_tol", type=float, default=1e-6)
    p.add_argument("--skip_wiring", action="store_true")
    # [exp] completeness estimator.  "aggregate" is the current mode;
    # "per_pixel" is the legacy one the library still defaults to.
    # "selection" is the third arm.  K>=2 support landed upstream in
    # darksirens 5aa90fa (PRs #342-#346): each catalog now anchors its OWN
    # theta labels from its OWN offline magnitude fit (M0hat/sigma_M for
    # catalog 1, M0hat_c2/sigma_M_c2 for catalog 2), anchoring is
    # all-or-nothing, and field sky weighting is required -- which we already
    # run.  See the "selection arm" block below for the two guards that still
    # stand between us and the MATCHED (schechter) version of this arm.
    p.add_argument("--c_mode", default="aggregate",
                   choices=["aggregate", "per_pixel", "selection"],
                   help="completeness estimator (default: aggregate; "
                        "'selection' additionally requires --selection_fit)")
    # [exp] one offline magnitude fit per catalog, in catalog order, matching
    # --survey_path.  Upstream refuses a partially-anchored mixture outright
    # (a catalog with no fit would sample its theta flat across the truncation
    # bounds while the other sits within ~1e-2 mag of its fit, so the missing-
    # galaxy budget would be set by an unconstrained nuisance).  Omit entirely
    # for the wide-open ablation on every catalog.
    p.add_argument("--selection_fit", default=None, metavar="JSON[,JSON...]",
                   help="c_mode=selection only: comma-separated selection-fit "
                        "JSONs, one per catalog in --survey_path order")
    # [exp] the wiring diff is the measurement, not a precondition
    p.add_argument("--wiring_nonfatal", action="store_true",
                   help="report the wiring diff and continue instead of asserting")
    p.add_argument("--probe_only", action="store_true",
                   help="run the wiring probe, write <out_tag>_probe.json, exit")
    # ---- truths / output ----
    p.add_argument("--h0_true", type=float, default=67.74)
    p.add_argument("--f_true", type=float, default=0.295,
                   help="realised host fraction of the seed (closure reference)")
    p.add_argument("--n0_true", type=float, default=-3.0)
    p.add_argument("--n0c2_true", type=float, default=-5.0)
    p.add_argument("--outdir", default=str(HERE.parent / "results"))
    p.add_argument("--out_tag", required=True)
    return p.parse_args(argv)


# --------------------------------------------------------------------------- #
# likelihood closure — replicates scan_h0f.main()'s proven build block exactly
# --------------------------------------------------------------------------- #
def build_closure(args):
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["JAX_PLATFORMS"] = "cuda" if args.device == "gpu" else "cpu"

    survey_paths = [str(p) for p in args.survey_path]
    n_catalogs = len(survey_paths)
    if n_catalogs != 2:
        sys.exit("[fatal] the 4D fit is K=2 by construction; give two --survey_path.")

    import darksirens
    print(f"darksirens module file: {darksirens.__file__}")
    from darksirens.inference.data import load_all_data, validate_loaded_survey_shapes
    from darksirens.likelihood.factory import make_likelihood
    from darksirens.gw.populations import get_fixed_population_params
    from darksirens.inference.prior import build_parameter_space
    import jax
    print(f"JAX devices: {jax.devices()}")

    if args.kde_window is not None:
        from darksirens.redshift.catalog import configure_catalog_kde_window
        configure_catalog_kde_window(size=int(args.kde_window),
                                     n_sigma=float(args.kde_window_nsigma))
        print(f"[kde] window W={args.kde_window} n_sigma={args.kde_window_nsigma}")

    opts = SimpleNamespace(
        universe_model=args.universe_model,
        survey_path=survey_paths[0],
        survey_paths=survey_paths,
        n_catalogs=n_catalogs,
        gw_path=args.gw_path,
        gwselection_path=args.gwselection_path,
        use_LSS=False,
        lss_completion=None,
        lss_completions=[],
        lss_marginalize=False,
        counterpart=None,
        counterpart_nside=1,
        counterpart_dz=1e-4,
        bright_siren_sky_marginalized=False,
        drop_full_catalog=False,
        sky_model="isotropic",
        mark_model="none",
        marks=None,
        mark_names=(),
        sel_batch_size=args.sel_batch_size,
        redshift_prior_barrier="auto",
        selection_neff_guard=args.selection_neff_guard,
        sampler="tinyns",
        fix_population=True,
        fix_cosmology=False,
        fix_de=True,
        fix_survey=False,
        pop_model="powerlaw+peak",
        shared_beta=True,
        shared_spin=True,
        shared_gamma=True,
        complete_empty_pixel_policy="zero",
        catalog_sky_weighting=args.catalog_sky_weighting,
        # [exp] The completeness estimator is now selectable, and the DEFAULT
        # ("per_pixel") is the legacy one kept for back-compatibility:
        #   per_pixel  C(z|pix) = clip(dN_obs_s(z|pix) / dN_exp_s(z), 0, 1)
        #              per-pixel numerator over an isotropic denominator, so it
        #              estimates C_sel(z) * (1 + delta_obs(pix, z)) -- angular
        #              clustering is absorbed into the completeness and
        #              overdense complete pixels clip at 1, discarding excess.
        #   aggregate  Cbar(z) = clip(Sum_p dN_obs_s(z|p)
        #                             / (N_pix_total * dN_exp_s(z)), 0, 1)
        #              ONE sky-anchored radial budget broadcast to every pixel,
        #              N_pix_total counting empty pixels too; no per-pixel
        #              clipping, one global clip.  Angular structure moves to
        #              the mean-one Q field.  Requires the field convention,
        #              which we already run.
        # This experiment drives it explicitly so the arm is never implicit.
        c_mode=args.c_mode,
        selection_prior=None,
        selection_fit=None,
    )
    # [exp] selection arm.  Gated on c_mode == "selection" so the aggregate and
    # per_pixel arms take byte-for-byte the code path they already ran -- the
    # two completed arms must stay reproducible from this same driver.
    if args.c_mode == "selection":
        if args.selection_fit is None:
            _paths = [None] * n_catalogs          # the wide-open ablation
        else:
            _paths = [e.strip() or None
                      for e in args.selection_fit.split(",")]
        if len(_paths) != n_catalogs:
            raise SystemExit(
                f"--selection_fit gives {len(_paths)} entries for "
                f"{n_catalogs} catalogs: pass one fit per catalog in "
                "--survey_path order (anchoring is all-or-nothing upstream), "
                "or omit the flag entirely for the wide-open ablation.")
        opts.selection_fit_paths = _paths
    elif args.selection_fit:
        raise SystemExit("--selection_fit is meaningful only with "
                         "--c_mode selection")
    opts.pe_event_block = args.pe_event_block
    if args.max_likelihood_variance is not None:
        opts.max_likelihood_variance = args.max_likelihood_variance
    opts.selection_neff_soft_guard = (
        args.selection_neff_guard == "soft"
        or (args.selection_neff_guard == "auto" and opts.sampler == "numpyro")
    )

    prior_overrides = {}                                  # pure likelihood
    fixed_parameter_values = {"Om0": OM0_FID}

    t0 = time.time()
    data = load_all_data(opts)
    validate_loaded_survey_shapes(data)
    print(f"load_all_data: {time.time() - t0:.2f}s  nEvents={data['nEvents']} "
          f"nsamp={data['nsamp']} Ndraw={data['Ndraw']} n_catalogs={n_catalogs}")

    # [exp] Reads each catalog's fit JSON, builds the truncated-normal theta
    # priors on that catalog's OWN labels (M0hat/sigma_M, M0hat_c2/sigma_M_c2),
    # pins each m_lim to its fit's datum in fixed_parameter_values, and stamps
    # opts.selection_family.  Must run BEFORE build_parameter_space, which is
    # what consumes selection_prior/selection_family -- same ordering as
    # darksirens' own cli/inference.py.
    _sel_kw = {}
    if args.c_mode == "selection":
        from darksirens.cli.inference import _resolve_selection_fits
        _resolve_selection_fits(opts, data, fixed_parameter_values)
        _sel_kw = {"selection_prior": opts.selection_prior,
                   "selection_family": opts.selection_family}
        print(f"[selection] family={opts.selection_family!r} "
              f"fits={opts.selection_fit_paths}")
        print(f"[selection] anchored priors: {opts.selection_prior}")
        _pinned = {k: v for k, v in fixed_parameter_values.items()
                   if "m_lim" in k}
        print(f"[selection] pinned data: {_pinned}")

    res = build_parameter_space(
        opts.pop_model, opts.fix_population, opts.fix_cosmology, opts.fix_survey,
        fix_de=opts.fix_de, prior_overrides=prior_overrides,
        fixed_parameter_values=fixed_parameter_values,
        universe_model=opts.universe_model,
        shared_beta=opts.shared_beta, shared_spin=opts.shared_spin,
        shared_gamma=opts.shared_gamma, sky_model=opts.sky_model,
        mark_model=opts.mark_model, mark_names=opts.mark_names,
        n_catalogs=opts.n_catalogs,
        lss_completion_active=[False] * opts.n_catalogs,
        use_lss=bool(opts.use_LSS),
        mark_names_by_catalog=None,
        c_mode=opts.c_mode,          # [exp] broadcast to both catalogs
        **_sel_kw,
    )
    labels = list(res[0])
    print(f"[c_mode] completeness estimator = {opts.c_mode!r}")   # [exp]
    print(f"Free parameters ({len(labels)}): {labels}")

    pop_params_fid = get_fixed_population_params(
        opts.pop_model, shared_beta=opts.shared_beta, shared_spin=opts.shared_spin,
        shared_gamma=opts.shared_gamma,
    )
    if args.gamma is not None:
        pop_params_fid = np.asarray(pop_params_fid, dtype=float).copy()
        print(f"[gamma override] {pop_params_fid[-1]} -> {args.gamma}")
        pop_params_fid[-1] = float(args.gamma)

    t0 = time.time()
    likelihood = make_likelihood(
        opts=opts, data=data, pop_params_fid=pop_params_fid,
        fixed_parameter_values=fixed_parameter_values,
    )
    print(f"make_likelihood (closure build): {time.time() - t0:.2f}s")

    nuisance = dict(NUISANCE_DEFAULTS)
    if args.nuisance_json:
        if os.path.exists(args.nuisance_json):
            override = json.loads(Path(args.nuisance_json).read_text())
        else:
            override = json.loads(args.nuisance_json)
        nuisance.update({k: float(v) for k, v in override.items()})

    point = dict(nuisance)
    # [exp] c_mode="selection" adds four LF labels to the free set that the
    # other two arms do not carry (Mstar_hat/alpha per catalog).  PIN them at
    # their anchored prior centres -- i.e. at the offline fits -- exactly as
    # delta/sigma_kde are pinned at NUISANCE_DEFAULTS in every arm.
    #
    # Pinning rather than sampling is what keeps the three arms comparable:
    # all of them then sample the SAME four parameters, so a difference in the
    # H0 or f_AGN posterior is the completeness estimator and not a wider
    # marginalisation.  It is also the honest choice given where the centres
    # come from -- the fits recover the mock's true LF to 1.75 sigma at worst,
    # so pinning there is pinning at (very nearly) truth, and the arm measures
    # the estimator rather than our LF uncertainty.
    if args.c_mode == "selection":
        for _lbl, (_loc, _sd) in (opts.selection_prior or {}).items():
            point.setdefault(_lbl, float(_loc))
        _pinned_theta = {k: point[k] for k in
                         ("Mstar_hat", "alpha", "Mstar_hat_c2", "alpha_c2")
                         if k in point}
        print(f"[selection] theta pinned at the fit centres: {_pinned_theta}")
    for lbl in SAMPLED:
        point.setdefault(lbl, 0.0)     # placeholders — overwritten every call
    missing = [lbl for lbl in labels if lbl not in point or point[lbl] is None]
    if missing:
        sys.exit(f"[fatal] no value for required label(s): {missing}")
    for lbl in SAMPLED:
        if lbl not in labels:
            sys.exit(f"[fatal] sampled label {lbl!r} not in the free-label set "
                     f"{labels} — wrong model configuration.")

    base = np.asarray([float(point[lbl]) for lbl in labels], dtype=float)
    idx = {lbl: i for i, lbl in enumerate(labels)}
    print(f"base coord: {dict(zip(labels, base.tolist()))}")

    # [exp] the selection arm's inputs are provenance, and `opts` is local to
    # this function -- so build the record HERE and hand it back, rather than
    # having the writer in main() reach for a name it cannot see.
    sel_provenance = None
    if args.c_mode == "selection":
        _prior = getattr(opts, "selection_prior", None) or {}
        sel_provenance = {
            "family": getattr(opts, "selection_family", None),
            "fit_paths": list(getattr(opts, "selection_fit_paths", []) or []),
            "anchored_prior": {k: list(v) for k, v in _prior.items()},
            "fit_records": getattr(opts, "selection_fits", None),
            "theta_treatment": (
                "PINNED at the anchored prior centres (the offline fits), not "
                "sampled -- so all three arms sample the same four parameters "
                "and the difference is the estimator, not a wider "
                "marginalisation. Same treatment delta/sigma_kde get in every "
                "arm."),
            "theta_pinned_at": {k: float(v[0]) for k, v in _prior.items()},
        }
    return likelihood, labels, base, idx, sel_provenance


# --------------------------------------------------------------------------- #
# wiring validation: reproduce stored grid logL values pointwise
# --------------------------------------------------------------------------- #
def wiring_check(likelihood, base, idx, refs, tol, nonfatal=False, record=None):
    # [exp] `nonfatal` downgrades the assert to a printed verdict, and `record`
    # (a list) collects every probe cell so the probe JSON can carry the actual
    # stored/got pair rather than just the worst absolute difference.
    import h5py
    worst = 0.0
    for ref in refs:
        with h5py.File(ref, "r") as f:
            H0g = f["H0_grid"][:]
            fg = f["f_grid"][:]
            ll = f["log_likelihood"][:]
            n0 = float(f.attrs["arg_log10n0"])
            n0c2 = float(f.attrs["arg_log10n0_c2"])
        # deterministic probe cells: (mid, mid), (mid, f=0), (mid, f=1)
        i = len(H0g) // 2
        cells = [(i, len(fg) // 2), (i, 0), (i, len(fg) - 1)]
        for (ih, jf) in cells:
            stored = float(ll[ih, jf])
            if not np.isfinite(stored):
                print(f"[wiring] {Path(ref).name} cell({ih},{jf}) stored=-inf, skipped")
                continue
            c = base.copy()
            c[idx["H0"]] = H0g[ih]
            c[idx["log10n0"]] = n0
            c[idx["log10n0_c2"]] = n0c2
            c[idx["fcat_2"]] = fg[jf]
            got = float(likelihood(c))
            diff = abs(got - stored)
            worst = max(worst, diff)
            print(f"[wiring] {Path(ref).name} H0={H0g[ih]:.2f} f={fg[jf]:.3f} "
                  f"n0c2={n0c2:.5f}: stored={stored:.10f} got={got:.10f} "
                  f"|diff|={diff:.3e}")
            if record is not None:
                record.append({"ref": Path(ref).name, "H0": float(H0g[ih]),
                               "f": float(fg[jf]), "log10n0": n0,
                               "log10n0_c2": n0c2, "stored": stored,
                               "got": got, "abs_diff": diff})
    if worst <= tol:
        print(f"[wiring] PASS  max|diff|={worst:.3e} <= {tol:.1e}")
    elif nonfatal:
        # [exp] the whole point of this experiment: say how far it moved.
        print(f"[wiring] CHANGED  max|diff|={worst:.3e} > tol={tol:.1e} "
              f"-- continuing (--wiring_nonfatal)")
    else:
        raise AssertionError(f"[fatal] wiring check FAILED: max|diff|={worst:.3e} "
                             f"> tol={tol:.1e}")
    return worst


# --------------------------------------------------------------------------- #
# samplers
#
# The dynesty log-likelihood and prior transform are MODULE-LEVEL functions on
# module globals (_G): dynesty checkpointing pickles the sampler, and a closure
# over the (unpicklable) jax likelihood would break it — a top-level function
# pickles by reference and resolves on restore once _G is repopulated.
# --------------------------------------------------------------------------- #
_G = {}


def _dynesty_loglike(theta):
    c = _G["base"].copy()
    c[_G["i_h0"]], c[_G["i_n0"]], c[_G["i_n0c2"]], c[_G["i_f"]] = theta
    t0 = time.time()
    ll = float(_G["likelihood"](c))
    counter = _G["counter"]
    counter["n"] += 1
    counter["t"] += time.time() - t0
    if not np.isfinite(ll):
        counter["neginf"] += 1
        return _G["floor"] if _G["floor"] is not None else -np.inf
    return ll


def _dynesty_ptform(u):
    return _G["lo"] + u * (_G["hi"] - _G["lo"])


def make_loglike(likelihood, base, idx, counter, floor=None, bounds=None):
    _G.update(likelihood=likelihood, base=base, counter=counter, floor=floor,
              i_h0=idx["H0"], i_n0=idx["log10n0"],
              i_n0c2=idx["log10n0_c2"], i_f=idx["fcat_2"])
    if bounds is not None:
        _G["lo"] = np.array([b[0] for b in bounds])
        _G["hi"] = np.array([b[1] for b in bounds])
    return _dynesty_loglike


def run_dynesty(args, loglike, bounds):
    import dynesty
    rstate = np.random.default_rng(args.rstate_seed)
    ckpt = args.checkpoint_file
    if ckpt and os.path.exists(ckpt):
        print(f"[dynesty] RESUMING from checkpoint {ckpt}")
        sampler = dynesty.NestedSampler.restore(ckpt)
        sampler.run_nested(dlogz=args.dlogz, maxcall=args.maxcall,
                           print_progress=True, resume=True,
                           checkpoint_file=ckpt,
                           checkpoint_every=args.checkpoint_every)
    else:
        sampler = dynesty.NestedSampler(loglike, _dynesty_ptform, ndim=4,
                                        nlive=args.nlive, rstate=rstate)
        sampler.run_nested(dlogz=args.dlogz, maxcall=args.maxcall,
                           print_progress=True,
                           checkpoint_file=ckpt if ckpt else None,
                           checkpoint_every=args.checkpoint_every)
    res = sampler.results
    samples = res.samples_equal(rstate=rstate)
    meta = {
        "sampler": "dynesty",
        "nlive": args.nlive, "dlogz_target": args.dlogz, "maxcall": args.maxcall,
        "rstate_seed": args.rstate_seed,
        "ncall_total": int(np.sum(res.ncall)),
        "niter": int(res.niter),
        "logz": float(res.logz[-1]),
        "logzerr": float(res.logzerr[-1]),
        "dlogz_reached": float(res.logz[-1] - res.logz[-2]) if len(res.logz) > 1 else None,
        "eff_percent": float(res.eff),
        "stopped_by_maxcall": bool(np.sum(res.ncall) >= args.maxcall),
    }
    raw = {"samples": res.samples, "logwt": res.logwt, "logl": res.logl,
           "logz": res.logz, "logzerr": res.logzerr}
    return samples, meta, raw


def run_emcee(args, loglike, bounds):
    import emcee
    import h5py
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])

    def logprob(theta):
        if np.any(theta < lo) or np.any(theta > hi):
            return -np.inf
        return loglike(theta)

    # init ball at the known peak: grid MAP in (H0, f) x anchor truths
    if args.init_ref_h5:
        with h5py.File(args.init_ref_h5, "r") as f:
            ll = f["log_likelihood"][:]
            H0g, fg = f["H0_grid"][:], f["f_grid"][:]
        ih, jf = np.unravel_index(np.nanargmax(np.where(np.isfinite(ll), ll, -np.inf)),
                                  ll.shape)
        center = np.array([H0g[ih], args.n0_true, args.n0c2_true, fg[jf]])
        print(f"[emcee] init at grid MAP H0={H0g[ih]:.2f} f={fg[jf]:.3f} "
              f"+ anchor truths ({args.n0_true}, {args.n0c2_true})")
    else:
        center = np.array([args.h0_true, args.n0_true, args.n0c2_true, args.f_true])
        print("[emcee] init at truth (no --init_ref_h5 given)")
    scale = np.array([0.5, 0.05, 0.05, 0.02])
    rng = np.random.default_rng(args.rstate_seed)
    p0 = center + scale * rng.standard_normal((args.nwalkers, 4))
    p0 = np.clip(p0, lo + 1e-6 * (hi - lo), hi - 1e-6 * (hi - lo))

    sampler = emcee.EnsembleSampler(args.nwalkers, 4, logprob)
    t0 = time.time()
    for i, _ in enumerate(sampler.sample(p0, iterations=args.nsteps)):
        if (i + 1) % 50 == 0 or (i + 1) == args.nsteps:
            el = time.time() - t0
            print(f"[emcee] step {i + 1}/{args.nsteps}  elapsed={el:.0f}s  "
                  f"acc={np.mean(sampler.acceptance_fraction):.3f}")
    burn = int(args.burn_frac * args.nsteps)
    samples = sampler.get_chain(discard=burn, flat=True)
    try:
        act = sampler.get_autocorr_time(discard=burn, quiet=True)
        act = [float(x) for x in np.atleast_1d(act)]
    except Exception as e:  # short pilot chains can defeat the ACT estimator
        print(f"[emcee] autocorr estimate unavailable: {e}")
        act = None
    meta = {
        "sampler": "emcee",
        "nwalkers": args.nwalkers, "nsteps": args.nsteps,
        "burn_steps": burn, "rstate_seed": args.rstate_seed,
        "init_ref_h5": args.init_ref_h5,
        "init_center": [float(x) for x in center],
        "acceptance_fraction_mean": float(np.mean(sampler.acceptance_fraction)),
        "autocorr_time": act,
    }
    raw = {"chain": sampler.get_chain(), "log_prob": sampler.get_log_prob()}
    return samples, meta, raw


# --------------------------------------------------------------------------- #
# summaries
# --------------------------------------------------------------------------- #
def summarize(samples, truths):
    out = {}
    names = ["H0", "log10n0", "log10n0_c2", "f_AGN"]
    for j, name in enumerate(names):
        x = samples[:, j]
        qs = np.percentile(x, [5, 16, 50, 84, 95])
        block = {
            "median": float(qs[2]),
            "ci68": [float(qs[1]), float(qs[3])],
            "ci90": [float(qs[0]), float(qs[4])],
            "mean": float(np.mean(x)),
            "sd": float(np.std(x)),
            "halfwidth68": float((qs[3] - qs[1]) / 2),
            "truth": truths[j],
            "offset": float(qs[2] - truths[j]),
        }
        block["pull"] = block["offset"] / block["halfwidth68"] if block["halfwidth68"] else None
        block["truth_in_ci68"] = bool(qs[1] <= truths[j] <= qs[3])
        block["truth_in_ci90"] = bool(qs[0] <= truths[j] <= qs[4])
        out[name] = block
    c = np.corrcoef(samples.T)
    out["corr"] = {"labels": names, "matrix": [[float(v) for v in row] for row in c]}
    return out


# --------------------------------------------------------------------------- #
def main(argv=None):
    args = parse_args(argv)
    started_at = datetime.now(timezone.utc).isoformat()
    t_wall0 = time.time()

    likelihood, labels, base, idx, sel_provenance = build_closure(args)

    wiring_worst = None
    wiring_cells = []                                              # [exp]
    if args.wiring_ref and not args.skip_wiring:
        wiring_worst = wiring_check(likelihood, base, idx,
                                    args.wiring_ref, args.wiring_tol,
                                    nonfatal=args.wiring_nonfatal,   # [exp]
                                    record=wiring_cells)             # [exp]
    elif not args.skip_wiring:
        print("[wiring] WARNING: no --wiring_ref given; sampling unvalidated wiring")

    # [exp] stage A: the cheap gate.  If the likelihood is bit-identical the
    # 4D fit cannot move, and hours of sampler time are unnecessary.
    if args.probe_only:
        git_sha = subprocess.run(["git", "-C", DARKSIRENS_REPO, "rev-parse", "HEAD"],
                                 capture_output=True, text=True).stdout.strip()
        import darksirens
        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        probe_path = outdir / f"{args.out_tag}_probe.json"
        probe_path.write_text(json.dumps({
            "_what": "wiring probe only: the analysis-5 K=2 closure rebuilt "
                     "against this darksirens checkout and evaluated at the "
                     "stored analysis-3 grid cells. abs_diff == 0 means the "
                     "likelihood is bit-identical and the 4D fit must reproduce.",
            "darksirens_git_sha": git_sha,
            "c_mode": args.c_mode,          # [exp]
            "darksirens_module_file": darksirens.__file__,
            "free_labels": list(labels),
            "n_free": len(labels),
            "wiring_max_abs_diff": wiring_worst,
            "wiring_cells": wiring_cells,
            "started_at": started_at,
            "probe_seconds": time.time() - t_wall0,
        }, indent=2))
        print(f"[probe] wrote {probe_path}")
        print(f"[probe] max|diff| = {wiring_worst:.6e}")
        return 0

    bounds = [tuple(args.h0_prior), tuple(args.n0_prior),
              tuple(args.n0c2_prior), tuple(args.f_prior)]
    print(f"priors (flat): H0 {bounds[0]}  log10n0 {bounds[1]}  "
          f"log10n0_c2 {bounds[2]}  fcat_2 {bounds[3]}")

    counter = {"n": 0, "t": 0.0, "neginf": 0}
    t_s0 = time.time()
    if args.sampler == "dynesty":
        loglike = make_loglike(likelihood, base, idx, counter, floor=-1e300,
                               bounds=bounds)
        samples, meta, raw = run_dynesty(args, loglike, bounds)
    else:
        loglike = make_loglike(likelihood, base, idx, counter, floor=None)
        samples, meta, raw = run_emcee(args, loglike, bounds)
    sampling_s = time.time() - t_s0

    s_per_eval = counter["t"] / counter["n"] if counter["n"] else None
    print(f"\nSampling done: {counter['n']} likelihood calls in {sampling_s:.0f}s "
          f"({s_per_eval:.3f} s/eval incl. JIT amortized), "
          f"{counter['neginf']} guard-rejected (-inf) calls, "
          f"{samples.shape[0]} posterior samples")

    truths = [args.h0_true, args.n0_true, args.n0c2_true, args.f_true]
    summary = summarize(samples, truths)

    # provenance (same contract as scan_h0f)
    git_sha = subprocess.run(["git", "-C", DARKSIRENS_REPO, "rev-parse", "HEAD"],
                             capture_output=True, text=True).stdout.strip()
    ancestor = subprocess.run(
        ["git", "-C", DARKSIRENS_REPO, "merge-base", "--is-ancestor",
         MERGE_SHA, "HEAD"]).returncode == 0
    assert ancestor, (f"[fatal] merge {MERGE_SHA} is NOT an ancestor of darksirens "
                      f"HEAD ({git_sha}); wrong darksirens checkout.")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    h5_path = outdir / f"{args.out_tag}.h5"
    json_path = outdir / f"{args.out_tag}.json"

    import h5py
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("samples", data=samples)
        f.create_dataset("sampled_labels", data=np.array(SAMPLED, dtype="S"))
        g = f.create_group("raw")
        for k, v in raw.items():
            g.create_dataset(k, data=np.asarray(v))
        f.attrs["labels"] = json.dumps(labels)
        f.attrs["base_coord"] = base
        f.attrs["darksirens_git_sha"] = git_sha
        f.attrs["c_mode"] = args.c_mode          # [exp]
        f.attrs["merge_sha_checked"] = MERGE_SHA
        for k, v in vars(args).items():
            try:
                f.attrs[f"arg_{k}"] = v if not isinstance(v, list) else json.dumps(v)
            except TypeError:
                f.attrs[f"arg_{k}"] = str(v)
        for k, v in meta.items():
            if v is not None:
                try:
                    f.attrs[f"meta_{k}"] = v if not isinstance(v, list) else json.dumps(v)
                except TypeError:
                    f.attrs[f"meta_{k}"] = str(v)

    doc = {
        "_what": ("Gate-0 pilot of analysis 5: 4-parameter posterior "
                  "(H0, log10n0, log10n0_c2, f_AGN) with both completion anchors "
                  "free under flat priors; darksirens K=2 field mixture, same "
                  "config of record as analyses 3/4"),
        "out_tag": args.out_tag,
        "sampler_meta": meta,
        "summary": summary,
        "wiring_check_max_abs_diff": wiring_worst,
        "n_likelihood_calls": counter["n"],
        "n_guard_rejected_calls": counter["neginf"],
        "seconds_per_eval_mean": s_per_eval,
        "sampling_seconds": sampling_s,
        "wall_seconds_total": time.time() - t_wall0,
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "priors": {"H0": bounds[0], "log10n0": bounds[1],
                   "log10n0_c2": bounds[2], "fcat_2": bounds[3]},
        "truths": {"H0": args.h0_true, "log10n0": args.n0_true,
                   "log10n0_c2": args.n0c2_true, "f_realised": args.f_true},
        "darksirens_git_sha": git_sha,
        "c_mode": args.c_mode,          # [exp]
    }
    if sel_provenance is not None:      # [exp] the arm's inputs are provenance
        doc["selection"] = sel_provenance
    json_path.write_text(json.dumps(doc, indent=1))
    print(f"\nwrote {h5_path}\nwrote {json_path}")
    print(json.dumps({k: {kk: vv for kk, vv in v.items()
                          if kk in ("median", "ci68", "offset", "pull")}
                      for k, v in summary.items() if k != "corr"}, indent=1))


if __name__ == "__main__":
    main()
