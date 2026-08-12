"""(H0, f) likelihood-grid driver for analysis_1_complete_catalog_H0.

Adapted verbatim from experiments/experiment_matched_mock/scripts/scan_h0f.py,
with two changes for this round:

  * MERGE_SHA is pinned to 2b86a2d — the darksirens checkout the campaign
    dataset (working/data/seed100) was generated against.
  * `--guard_record` (default ON) instruments
    `darksirens.likelihood.core.selection_log_correction` with a
    jax.debug.callback spy, so every grid cell carries its own
    (Neff, pe_variance_sum, threshold, passes) record.  The template only
    reported `n_neginf_cells`; the campaign convention is to RECORD guard
    behaviour cell by cell rather than fight it.

Evaluates the darksirens K=2 multitracer likelihood on fixed grids by module
import (no sampler): load_all_data -> validate_loaded_survey_shapes ->
build_parameter_space -> get_fixed_population_params -> make_likelihood, then
scans H0, the AGN-catalog mixture weight fcat_2, or the (H0, fcat_2) joint at a
fixed nuisance point.  The closure is the PURE likelihood (no prior); scan values
are physical coordinates in `labels` order.

Selection validity guard: this experiment uses the HISTORICAL guard,
`N_eff > 5 * N_obs` (the Vitale et al. 2022 mean floor), and nothing else.
Current darksirens master additionally enforces the GWTC-4.0/5.0 total-variance
criterion `sigma^2_lnL = sum_i sigma^2_i + N_obs^2/N_eff <= max_likelihood_variance`
with a default cap of 1.0, so `--max_likelihood_variance` DEFAULTS TO 1e6 here,
which makes that criterion inert and collapses the threshold to exactly
`max(5*N_obs, ~1) = 5*N_obs`.  `--selection_neff_guard soft` does NOT achieve
this: it only softens the wall's shape and leaves the variance threshold in place
(darksirens/likelihood/selection.py:311-312).

Outputs `<outdir>/<out_tag>.h5` (grids + logL + provenance attrs) and
`<outdir>/<out_tag>.json` (flat-prior posterior summary: MAP, median,
equal-tailed 68/90% CIs, trapezoid marginals, and for joint scans the
correlation rho from the normalized 2-D posterior).
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

DARKSIRENS_REPO = os.environ.get(
    "DARKSIRENS_SRC", "/hildafs/projects/phy230014p/magana/src/darksirens"
)
MERGE_SHA = "2b86a2d"  # the checkout the campaign dataset was built on — must be an ancestor
OM0_FID = 0.3075       # pinned via fixed_parameter_values

# Nuisance labels (base survey block + K=2 _c2 block) and their hard-coded defaults.
NUISANCE_DEFAULTS = {
    "delta": 0.0,
    "b_miss": 1.0,
    "sigma_kde": 0.0,
    "delta_c2": 0.0,
    "b_miss_c2": 1.0,
    "sigma_kde_c2": 0.0,
}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--universe_model", choices=["dark_sirens", "dark_sirens_complete"],
                    required=True)
    ap.add_argument("--catalog_sky_weighting", choices=["conditional", "field"],
                    default="conditional",
                    help="Catalog-prior normalization (PR-3 branch feature): "
                         "'field' = survey-global Z (host-fraction estimand).")
    ap.add_argument("--survey_path", nargs="+", required=True, metavar="PATH",
                    help="1 path => K=1; 2 paths => K=2 mixture, order [GAL, AGN] "
                         "so fcat_2 = alpha_AGN.")
    ap.add_argument("--gw_path", required=True)
    ap.add_argument("--gwselection_path", required=True)
    ap.add_argument("--scan", choices=["h0", "f", "joint"], required=True)
    ap.add_argument("--h0_grid", nargs=3, type=float, default=[50.0, 100.0, 61.0],
                    metavar=("MIN", "MAX", "N"))
    ap.add_argument("--f_grid", nargs=3, type=float, default=[0.0, 1.0, 41.0],
                    metavar=("MIN", "MAX", "N"))
    ap.add_argument("--h0_fixed", type=float, default=67.74,
                    help="Fixed H0 for f scans.")
    ap.add_argument("--f_fixed", type=float, default=None,
                    help="Fixed fcat_2 for h0 scans with K=2 (error if K=2 and not given).")
    ap.add_argument("--log10n0", type=float, default=None,
                    help="Nuisance scan point for catalog-1 log10n0 (required for dark_sirens).")
    ap.add_argument("--log10n0_c2", type=float, default=None,
                    help="Nuisance scan point for catalog-2 log10n0 (required for K=2).")
    ap.add_argument("--nuisance_json", default=None,
                    help="Inline JSON string or path overriding the hard-coded nuisance "
                         "defaults (delta=0, b_miss=1, sigma_kde=0, and _c2 variants).")
    ap.add_argument("--selection_neff_guard", choices=["auto", "hard", "soft"],
                    default="hard",
                    help="Sparse-selection Neff guard mode. 'hard' = the historical "
                         "-inf wall (this experiment's default).")
    ap.add_argument("--max_likelihood_variance", type=float, default=1e6,
                    help="Total-variance guard budget on sigma^2_lnL = "
                         "pe_variance_sum + N_obs^2/Neff (darksirens default 1.0, the "
                         "GWTC-4.0/5.0 threshold; post-#212 addition). Pass 1e6 to make "
                         "the criterion inert and recover the legacy Neff > 5*N_obs floor "
                         "exactly (for like-for-like comparison with the #212-era run).")
    ap.add_argument("--sel_batch_size", type=int, default=None,
                    help="Injections per selection-integral chunk.  With v2's dense "
                         "GAL survey the per-chunk working set is "
                         "sel_batch_size * kde_window * 8 B, so a single pass over "
                         "2.1e6 injections needs ~69 GB and OOMs an 80 GB A100.")
    ap.add_argument("--pe_event_block", type=int, default=None,
                    help="Events per PE-reduction chunk (working set "
                         "pe_event_block * nsamp * kde_window * 8 B).  Unset = one "
                         "pass over all events, which is ~66 GB at nsamp=2000 and "
                         "kde_window=4096.")
    ap.add_argument("--guard_record", dest="guard_record", action="store_true",
                    default=True,
                    help="Instrument selection_log_correction and record "
                         "(Neff, pe_variance_sum, threshold, passes) for EVERY grid "
                         "cell (default: on).")
    ap.add_argument("--no_guard_record", dest="guard_record", action="store_false",
                    help="Disable the per-cell guard record.")
    # NOTE: the --guard_bypass diagnostic pathway of the first pass has been DELETED.
    # Injection sets are now sized so every production cell clears the guard by a
    # factor of two or more, so there is nothing behind a wall to look at, and no
    # number in this directory comes from an unguarded selection correction.
    ap.add_argument("--kde_window", type=int, default=None,
                    help="Static window W for darksirens' windowed catalog-KDE "
                         "evaluator (darksirens.redshift.catalog."
                         "configure_catalog_kde_window; module default 1024).  v2's "
                         "complete GAL survey has ~1.3e4 galaxies per nside-32 row, "
                         "so W must exceed the number inside the kernel support or "
                         "the KDE is nearest-neighbour truncated; size it with "
                         "recommended_kde_window.")
    ap.add_argument("--kde_window_nsigma", type=float, default=8.0,
                    help="n_sigma for the same evaluator (module default 8).")
    ap.add_argument("--out_tag", required=True)
    ap.add_argument("--outdir", default=None,
                    help="Default: ../results relative to this script.")
    ap.add_argument("--device", choices=["gpu", "cpu"], default="gpu")
    # Optional truth values -> summarize_grid-style truth-coverage flags in the JSON.
    ap.add_argument("--gamma", type=float, default=None,
                    help="Override the fixed population's redshift-rate index gamma "
                         "(fiducial 0). The mock's host draw was flat over eligible "
                         "catalog entries, which corresponds to gamma = 1 (a bare "
                         "dV_c/dz draw), so this isolates the rate-weighting mismatch.")
    ap.add_argument("--h0_true", type=float, default=None)
    ap.add_argument("--f_true", type=float, default=None)
    return ap.parse_args(argv)


# --------------------------------------------------------------------------- #
# Posterior helpers (mirror summarize_grid.py conventions: flat prior, trapz)
# --------------------------------------------------------------------------- #
def marginal_ci(x, logp_1d, levels=(0.68, 0.90)):
    """Median + equal-tailed CIs of a 1-D flat-prior posterior exp(logp)."""
    logp_1d = np.asarray(logp_1d, dtype=float)
    x = np.asarray(x, dtype=float)
    m = np.nanmax(logp_1d[np.isfinite(logp_1d)]) if np.isfinite(logp_1d).any() else 0.0
    p = np.exp(np.where(np.isfinite(logp_1d), logp_1d, -np.inf) - m)
    norm = np.trapz(p, x)
    if not np.isfinite(norm) or norm <= 0:
        return {"median": float("nan"),
                "ci68": [float("nan"), float("nan")],
                "ci90": [float("nan"), float("nan")]}
    p = p / norm
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (p[1:] + p[:-1]) * np.diff(x))])
    cdf /= cdf[-1]
    out = {"median": float(np.interp(0.5, cdf, x))}
    for lev in levels:
        lo = float(np.interp(0.5 - lev / 2, cdf, x))
        hi = float(np.interp(0.5 + lev / 2, cdf, x))
        out["ci{:.0f}".format(lev * 100)] = [lo, hi]
    return out


def _terse(summary):
    """Console copy of the summary with the bulky per-cell arrays elided."""
    out = json.loads(json.dumps(summary))
    for key in ("H0", "f"):
        if isinstance(out.get(key), dict):
            out[key].pop("grid", None)
    if isinstance(out.get("guard"), dict):
        cells = out["guard"].pop("cells", None)
        if cells is not None:
            out["guard"]["cells"] = f"<{len(cells)} per-cell records in the JSON file>"
    return out


def add_truth_flags(block, truth):
    if truth is None:
        return
    block["truth"] = float(truth)
    for lev in ("ci68", "ci90"):
        if lev in block:
            lo, hi = block[lev]
            block["truth_in_" + lev] = bool(lo <= truth <= hi)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv=None):
    args = parse_args(argv)

    # ---- device / env: MUST precede darksirens (JAX backend) import ----------
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    if args.device == "gpu":
        # jax.debug.callback needs a local CPU device to place its inputs on, so
        # the guard spy requires "cuda,cpu" (cuda stays the default device).
        os.environ["JAX_PLATFORMS"] = "cuda,cpu" if args.guard_record else "cuda"
    else:
        os.environ["JAX_PLATFORMS"] = "cpu"

    survey_paths = [str(p) for p in args.survey_path]
    n_catalogs = len(survey_paths)
    if n_catalogs not in (1, 2):
        sys.exit(f"[fatal] --survey_path takes 1 (K=1) or 2 (K=2) paths; got {n_catalogs}.")

    # ---- fail-fast guards mirroring the core.py K>=2 restriction -------------
    if n_catalogs == 2 and args.universe_model != "dark_sirens":
        # Post-PR-#208: dark_sirens_complete K=2 is legal under FIELD weighting.
        if not (args.universe_model == "dark_sirens_complete"
                and args.catalog_sky_weighting == "field"):
            sys.exit("[fatal] K=2 mixture requires --universe_model dark_sirens, "
                     "or dark_sirens_complete with --catalog_sky_weighting field "
                     "(got {!r}/{!r}).".format(args.universe_model,
                                               args.catalog_sky_weighting))
    if args.scan in ("f", "joint") and n_catalogs != 2:
        sys.exit(f"[fatal] --scan {args.scan} scans fcat_2 and requires K=2 "
                 "(two --survey_path entries).")
    if args.scan == "h0" and n_catalogs == 2 and args.f_fixed is None:
        sys.exit("[fatal] --scan h0 with K=2 requires --f_fixed (fixed fcat_2).")

    started_at = datetime.now(timezone.utc).isoformat()
    t_wall0 = time.time()

    # ---- nuisance point ------------------------------------------------------
    nuisance = dict(NUISANCE_DEFAULTS)
    if args.nuisance_json:
        if os.path.exists(args.nuisance_json):
            override = json.loads(Path(args.nuisance_json).read_text())
        else:
            override = json.loads(args.nuisance_json)
        nuisance.update({k: float(v) for k, v in override.items()})

    # ---- import darksirens (backend initializes here) ------------------------
    import darksirens
    print(f"darksirens module file: {darksirens.__file__}")
    from darksirens.inference.data import load_all_data, validate_loaded_survey_shapes
    from darksirens.likelihood.factory import make_likelihood
    from darksirens.gw.populations import get_fixed_population_params
    from darksirens.inference.prior import build_parameter_space
    import jax

    print(f"JAX devices: {jax.devices()}")

    # ---- per-cell guard spy (must be installed BEFORE make_likelihood) -------
    # Master @ 2b86a2d rejects a cell unless
    #     Neff > max(5*N_obs, N_obs^2 / (max_likelihood_variance - pe_variance_sum))
    # (darksirens/likelihood/selection.py).  The spy is a pass-through: it records
    # the three deciding numbers, then calls the real function, so the returned
    # logL is bit-identical to an uninstrumented run.
    guard_hits = []
    if args.guard_record:
        import jax.numpy as jnp
        from darksirens.likelihood import core as ds_core
        from darksirens.likelihood.selection import (
            selection_log_correction as _true_slc,
            DEFAULT_MAX_LIKELIHOOD_VARIANCE,
            _MIN_VARIANCE_BUDGET,
        )

        def _guard_spy(log_mu, Neff, nEvents, soft_guard=False,
                       max_likelihood_variance=DEFAULT_MAX_LIKELIHOOD_VARIANCE,
                       pe_variance_sum=0.0):
            n = float(nEvents)
            budget = jnp.maximum(max_likelihood_variance - pe_variance_sum,
                                 _MIN_VARIANCE_BUDGET)
            threshold = jnp.maximum(5.0 * n, (n * n) / budget)

            def _record(log_mu_v, Neff_v, pe_var_v, thr_v):
                Neff_f = float(Neff_v)
                pe_f = float(pe_var_v)
                sel_var = (n * n) / Neff_f if Neff_f > 0 else float("inf")
                guard_hits.append({
                    "log_mu": float(log_mu_v),
                    "Neff": Neff_f,
                    "pe_variance_sum": pe_f,
                    "selection_variance_N2_over_Neff": sel_var,
                    "sigma2_total": pe_f + sel_var,
                    "threshold": float(thr_v),
                    "passes": bool(Neff_f > float(thr_v)),
                    "legacy_floor_5N": 5.0 * n,
                    "passes_legacy_floor": bool(Neff_f > 5.0 * n),
                    "nEvents": n,
                })

            jax.debug.callback(_record, log_mu, Neff, pe_variance_sum, threshold)
            return _true_slc(log_mu, Neff, nEvents, soft_guard=soft_guard,
                             max_likelihood_variance=max_likelihood_variance,
                             pe_variance_sum=pe_variance_sum)

        ds_core.selection_log_correction = _guard_spy
        print("[guard] per-cell guard record ENABLED "
              f"(DEFAULT_MAX_LIKELIHOOD_VARIANCE={DEFAULT_MAX_LIKELIHOOD_VARIANCE})")

    # ---- catalog-KDE window (must precede the first likelihood trace) ---------
    if args.kde_window is not None:
        from darksirens.redshift.catalog import configure_catalog_kde_window
        configure_catalog_kde_window(size=int(args.kde_window),
                                     n_sigma=float(args.kde_window_nsigma))
        print(f"[kde] window W={args.kde_window} n_sigma={args.kde_window_nsigma}")

    # ---- opts: copied VERBATIM from scan_fcat_conditional.py:40-73, then
    #      adjusted for the CLI (universe_model, survey paths, n_catalogs,
    #      gw/selection paths, sel_batch_size, guard) --------------------------
    opts = SimpleNamespace(
        universe_model="dark_sirens",
        survey_path=survey_paths[0],
        survey_paths=survey_paths,
        n_catalogs=len(survey_paths),
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
        sel_batch_size=None,
        redshift_prior_barrier="auto",
        selection_neff_guard="auto",
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
    )
    # --- adjustments for this CLI ---
    opts.universe_model = args.universe_model
    opts.survey_path = survey_paths[0]
    opts.survey_paths = survey_paths
    opts.n_catalogs = n_catalogs
    opts.gw_path = args.gw_path
    opts.gwselection_path = args.gwselection_path
    opts.sel_batch_size = args.sel_batch_size
    opts.pe_event_block = args.pe_event_block
    opts.selection_neff_guard = args.selection_neff_guard
    if args.max_likelihood_variance is not None:
        opts.max_likelihood_variance = args.max_likelihood_variance
    # Resolve the guard to the boolean the factory consumes, exactly as
    # darksirens.cli.inference does (soft => True; auto => soft only for numpyro).
    opts.selection_neff_soft_guard = (
        args.selection_neff_guard == "soft"
        or (args.selection_neff_guard == "auto" and opts.sampler == "numpyro")
    )

    prior_overrides = {}                                  # pure-likelihood scan
    fixed_parameter_values = {"Om0": OM0_FID}             # Om0 pinned

    # ---- build the closure (proven recipe) -----------------------------------
    t0 = time.time()
    data = load_all_data(opts)
    validate_loaded_survey_shapes(data)
    print(f"load_all_data: {time.time() - t0:.2f}s  nEvents={data['nEvents']} "
          f"nsamp={data['nsamp']} Ndraw={data['Ndraw']} n_catalogs={n_catalogs}")

    res = build_parameter_space(
        opts.pop_model, opts.fix_population, opts.fix_cosmology, opts.fix_survey,
        fix_de=opts.fix_de, prior_overrides=prior_overrides,
        fixed_parameter_values=fixed_parameter_values, universe_model=opts.universe_model,
        shared_beta=opts.shared_beta, shared_spin=opts.shared_spin,
        shared_gamma=opts.shared_gamma, sky_model=opts.sky_model,
        mark_model=opts.mark_model, mark_names=opts.mark_names,
        n_catalogs=opts.n_catalogs,
        # Post-#212 master (survey-block registry, PR #308): the sampled survey
        # block is derived from (universe_model, use_lss, lss_completion_active),
        # so these must be threaded exactly as darksirens.cli.inference does —
        # with use_LSS off, b_miss is inert and must not be a phantom dimension.
        lss_completion_active=[False] * opts.n_catalogs,
        use_lss=bool(opts.use_LSS),
        mark_names_by_catalog=None,
    )
    labels = list(res[0])
    print(f"Free parameters ({len(labels)}): {labels}")

    pop_params_fid = get_fixed_population_params(
        opts.pop_model, shared_beta=opts.shared_beta, shared_spin=opts.shared_spin,
        shared_gamma=opts.shared_gamma,
    )
    if args.gamma is not None:
        # gamma is the LAST entry of the powerlaw+peak fiducial vector.
        pop_params_fid = np.asarray(pop_params_fid, dtype=float).copy()
        print(f"[gamma override] {pop_params_fid[-1]} -> {args.gamma}")
        pop_params_fid[-1] = float(args.gamma)

    t0 = time.time()
    likelihood = make_likelihood(
        opts=opts, data=data, pop_params_fid=pop_params_fid,
        fixed_parameter_values=fixed_parameter_values,
    )
    print(f"make_likelihood (closure build): {time.time() - t0:.2f}s")

    # ---- base coordinate (nuisance point + fixed values) in label order ------
    point = dict(nuisance)
    point["H0"] = args.h0_fixed
    if args.log10n0 is not None:
        point["log10n0"] = args.log10n0
    if args.log10n0_c2 is not None:
        point["log10n0_c2"] = args.log10n0_c2
    if args.f_fixed is not None:
        point["fcat_2"] = args.f_fixed

    # Scanned axes are overwritten each iteration; give them a placeholder so the
    # required-value check below only fires on genuinely missing fixed values.
    scanned_labels = set()
    if args.scan in ("h0", "joint"):
        scanned_labels.add("H0")
    if args.scan in ("f", "joint"):
        scanned_labels.add("fcat_2")
    for lbl in scanned_labels:
        point.setdefault(lbl, 0.0)

    missing = [lbl for lbl in labels if lbl not in point or point[lbl] is None]
    if missing:
        hints = {"log10n0": "--log10n0", "log10n0_c2": "--log10n0_c2",
                 "fcat_2": "--f_fixed", "H0": "--h0_fixed"}
        need = ", ".join(f"{m} ({hints.get(m, '--nuisance_json')})" for m in missing)
        sys.exit(f"[fatal] no value supplied for required label(s): {need}")

    base = np.asarray([float(point[lbl]) for lbl in labels], dtype=float)
    idx = {lbl: i for i, lbl in enumerate(labels)}
    print(f"base coord: {dict(zip(labels, base.tolist()))}")

    # ---- grids ---------------------------------------------------------------
    H0_vals = np.linspace(args.h0_grid[0], args.h0_grid[1], int(round(args.h0_grid[2])))
    f_vals = np.linspace(args.f_grid[0], args.f_grid[1], int(round(args.f_grid[2])))

    # ---- evaluation ----------------------------------------------------------
    def build_coords():
        """Return (coords_list, fill_fn, shape, grids_used)."""
        if args.scan == "h0":
            coords = []
            for h in H0_vals:
                c = base.copy(); c[idx["H0"]] = h; coords.append(c)
            return coords, ("H0",), (len(H0_vals),)
        if args.scan == "f":
            coords = []
            for v in f_vals:
                c = base.copy(); c[idx["fcat_2"]] = v; coords.append(c)
            return coords, ("f",), (len(f_vals),)
        # joint
        coords = []
        for h in H0_vals:
            for v in f_vals:
                c = base.copy(); c[idx["H0"]] = h; c[idx["fcat_2"]] = v
                coords.append(c)
        return coords, ("H0", "f"), (len(H0_vals), len(f_vals))

    coords, axes, shape = build_coords()
    n = len(coords)
    print(f"\nScan '{args.scan}': {n} evaluations, axes={axes}, shape={shape}")

    lls = np.empty(n, dtype=float)
    per_eval = np.empty(n, dtype=float)
    cell_guard = [None] * n            # one guard record per grid cell
    first_eval_s = None
    t_loop0 = time.time()
    for k, c in enumerate(coords):
        if args.guard_record:
            guard_hits.clear()
        t0 = time.time()
        ll = float(likelihood(c))
        dt = time.time() - t0
        lls[k] = ll
        per_eval[k] = dt
        if args.guard_record:
            if len(guard_hits) == 1:
                cell_guard[k] = dict(guard_hits[0])
            elif len(guard_hits) > 1:
                # more than one selection call site fired: keep them all
                cell_guard[k] = {"multi": [dict(g) for g in guard_hits]}
        if k == 0:
            first_eval_s = dt
            print(f"[eval] first eval (incl. JIT compile): {dt:.3f}s  logL={ll:.4f}")
        elif (k + 1) % 25 == 0 or (k + 1) == n:
            elapsed = time.time() - t_loop0
            rate = (k + 1) / elapsed if elapsed > 0 else float("inf")
            print(f"[eval] {k + 1}/{n}  elapsed={elapsed:.1f}s  {rate:.2f} eval/s  "
                  f"logL={ll:.4f}")
    total_eval_s = time.time() - t_loop0
    steady = per_eval[1:] if n > 1 else per_eval
    steady_median_s = float(np.median(steady)) if steady.size else float(first_eval_s)
    n_neginf = int(np.sum(~np.isfinite(lls)))
    print(f"\nEval done: {n} evals in {total_eval_s:.1f}s | first(JIT)={first_eval_s:.3f}s "
          f"| steady-median={steady_median_s:.4f}s | n_neginf_cells={n_neginf}")

    ll_grid = lls.reshape(shape)
    finite_any = bool(np.isfinite(lls).any())

    # ---- per-cell guard record ----------------------------------------------
    # `rejected` is the ground truth (logL == -inf); the spy fields explain WHY.
    rejected = ~np.isfinite(lls)
    guard_arrays = {"rejected": rejected.astype(np.int8)}
    guard_cells = None
    if args.guard_record:
        keys = ("Neff", "pe_variance_sum", "selection_variance_N2_over_Neff",
                "sigma2_total", "threshold", "legacy_floor_5N")
        for key in keys:
            guard_arrays[key] = np.array(
                [(g.get(key, np.nan) if isinstance(g, dict) and "multi" not in g
                  else np.nan) for g in cell_guard], dtype=float)
        for key in ("passes", "passes_legacy_floor"):
            guard_arrays[key] = np.array(
                [(int(bool(g.get(key))) if isinstance(g, dict) and "multi" not in g
                  else -1) for g in cell_guard], dtype=np.int8)
        n_traced = int(sum(1 for g in cell_guard if g is not None))
        print(f"[guard] traced {n_traced}/{n} cells; "
              f"rejected(logL=-inf) = {int(rejected.sum())}")
        guard_cells = cell_guard
    from darksirens.likelihood.selection import DEFAULT_MAX_LIKELIHOOD_VARIANCE
    eff_max_var = (args.max_likelihood_variance
                   if args.max_likelihood_variance is not None
                   else DEFAULT_MAX_LIKELIHOOD_VARIANCE)
    if not finite_any:
        print("[warn] EVERY cell is -inf: the configuration is rejected by the "
              f"likelihood guards at max_likelihood_variance={eff_max_var}. "
              "Writing the grid + a rejected-marked summary (no argmax/CI).")

    # ---- provenance ----------------------------------------------------------
    git_sha = subprocess.run(
        ["git", "-C", DARKSIRENS_REPO, "rev-parse", "HEAD"],
        capture_output=True, text=True).stdout.strip()
    ancestor = subprocess.run(
        ["git", "-C", DARKSIRENS_REPO, "merge-base", "--is-ancestor", MERGE_SHA, "HEAD"]
    ).returncode == 0
    assert ancestor, (f"[fatal] merge {MERGE_SHA} is NOT an ancestor of darksirens "
                      f"HEAD ({git_sha}); wrong darksirens checkout.")

    finished_at = datetime.now(timezone.utc).isoformat()

    # ---- output paths --------------------------------------------------------
    outdir = Path(args.outdir) if args.outdir else (Path(__file__).resolve().parent.parent / "results")
    outdir.mkdir(parents=True, exist_ok=True)
    h5_path = outdir / f"{args.out_tag}.h5"
    json_path = outdir / f"{args.out_tag}.json"

    # ---- write HDF5 ----------------------------------------------------------
    import h5py
    with h5py.File(h5_path, "w") as f:
        if args.scan in ("h0", "joint"):
            f.create_dataset("H0_grid", data=H0_vals)
        if args.scan in ("f", "joint"):
            f.create_dataset("f_grid", data=f_vals)
        f.create_dataset("log_likelihood", data=ll_grid)
        g = f.create_group("guard")
        for key, arr in guard_arrays.items():
            g.create_dataset(key, data=np.asarray(arr).reshape(shape))
        g.attrs["guard_record_enabled"] = bool(args.guard_record)
        g.attrs["n_rejected"] = int(rejected.sum())

        # every CLI arg
        for key, val in vars(args).items():
            akey = f"arg_{key}"
            if val is None:
                f.attrs[akey] = "None"
            elif isinstance(val, (list, tuple)):
                f.attrs[akey] = json.dumps(list(val))
            else:
                f.attrs[akey] = val
        # scan metadata / labels / base coord
        f.attrs["scan"] = args.scan
        f.attrs["scan_axes"] = json.dumps(list(axes))
        f.attrs["labels"] = json.dumps(labels)
        f.attrs["base_coord"] = base
        f.attrs["base_coord_labeled"] = json.dumps(dict(zip(labels, base.tolist())))
        f.attrs["n_catalogs"] = n_catalogs
        f.attrs["Om0_fixed"] = OM0_FID
        f.attrs["selection_neff_guard"] = args.selection_neff_guard
        f.attrs["selection_neff_soft_guard"] = bool(opts.selection_neff_soft_guard)
        f.attrs["max_likelihood_variance_effective"] = float(eff_max_var)
        f.attrs["all_cells_rejected"] = bool(not finite_any)
        f.attrs["sampler"] = opts.sampler
        # provenance
        f.attrs["darksirens_git_sha"] = git_sha
        f.attrs["darksirens_file"] = darksirens.__file__
        f.attrs["merge_sha_checked"] = MERGE_SHA
        f.attrs["merge_sha_is_ancestor"] = bool(ancestor)
        f.attrs["jax_devices"] = str(jax.devices())
        f.attrs["jax_platform"] = os.environ["JAX_PLATFORMS"]
        # timing / timestamps
        f.attrs["started_at"] = started_at
        f.attrs["finished_at"] = finished_at
        f.attrs["wall_seconds_total"] = time.time() - t_wall0
        f.attrs["first_eval_seconds"] = float(first_eval_s)
        f.attrs["steady_state_median_seconds"] = steady_median_s
        f.attrs["total_eval_seconds"] = float(total_eval_s)
        f.attrs["n_evals"] = int(n)
        f.attrs["n_neginf_cells"] = n_neginf
    print(f"Wrote {h5_path}")

    # ---- JSON summary (flat-prior posterior; summarize_grid.py conventions) --
    ll_safe = np.where(np.isfinite(ll_grid), ll_grid, -np.inf)
    finite = np.isfinite(ll_safe)
    lmax = float(ll_safe[finite].max()) if finite.any() else 0.0

    summary = {
        "file": str(h5_path),
        "scan": args.scan,
        "labels": labels,
        "n_catalogs": n_catalogs,
        "n_evals": int(n),
        "n_neginf_cells": n_neginf,
        "n_rejected": int(rejected.sum()),
        "logL_max": lmax,
        "base_coord": dict(zip(labels, base.tolist())),
        "selection_neff_guard": args.selection_neff_guard,
        "selection_neff_soft_guard": bool(opts.selection_neff_soft_guard),
        "max_likelihood_variance_effective": float(eff_max_var),
        "all_cells_rejected": bool(not finite_any),
        "timing": {
            "first_eval_seconds": float(first_eval_s),
            "steady_state_median_seconds": steady_median_s,
            "total_eval_seconds": float(total_eval_s),
        },
    }

    # ---- guard block ---------------------------------------------------------
    scan_axis_vals = (H0_vals if args.scan == "h0"
                      else f_vals if args.scan == "f" else None)
    guard_block = {
        "enabled": bool(args.guard_record),
        "guard_bypass_available": False,   # pathway deleted; see parse_args
        "kde_window": (int(args.kde_window) if args.kde_window else None),
        "sel_batch_size": args.sel_batch_size,
        "pe_event_block": args.pe_event_block,
        "kde_window_nsigma": float(args.kde_window_nsigma),
        "n_cells": int(n),
        "n_rejected": int(rejected.sum()),
        "rejected_fraction": float(rejected.mean()),
        "selection_neff_guard": args.selection_neff_guard,
        "max_likelihood_variance_effective": float(eff_max_var),
    }
    if scan_axis_vals is not None:
        guard_block["rejected_values"] = [
            float(v) for v, r in zip(scan_axis_vals, rejected) if r]
    if args.guard_record:
        def _fin(a):
            a = np.asarray(a, dtype=float)
            return a[np.isfinite(a)]
        neff = _fin(guard_arrays["Neff"])
        pev = _fin(guard_arrays["pe_variance_sum"])
        s2 = _fin(guard_arrays["sigma2_total"])
        thr = _fin(guard_arrays["threshold"])
        guard_block["summary"] = {
            "Neff_min": float(neff.min()) if neff.size else None,
            "Neff_median": float(np.median(neff)) if neff.size else None,
            "Neff_max": float(neff.max()) if neff.size else None,
            "pe_variance_sum_min": float(pev.min()) if pev.size else None,
            "pe_variance_sum_median": float(np.median(pev)) if pev.size else None,
            "pe_variance_sum_max": float(pev.max()) if pev.size else None,
            "sigma2_total_min": float(s2.min()) if s2.size else None,
            "sigma2_total_max": float(s2.max()) if s2.size else None,
            "threshold_min": float(thr.min()) if thr.size else None,
            "threshold_max": float(thr.max()) if thr.size else None,
            "legacy_floor_5N": (float(guard_arrays["legacy_floor_5N"][0])
                                if np.isfinite(guard_arrays["legacy_floor_5N"][0])
                                else None),
            "n_pass_legacy_floor": int(np.sum(guard_arrays["passes_legacy_floor"] == 1)),
            "n_fail_legacy_floor": int(np.sum(guard_arrays["passes_legacy_floor"] == 0)),
            # the guard's own verdict, independent of whether it was bypassed
            "n_guard_would_reject": int(np.sum(guard_arrays["passes"] == 0)),
        }
        if scan_axis_vals is not None:
            guard_block["guard_would_reject_values"] = [
                float(v) for v, p in zip(scan_axis_vals, guard_arrays["passes"])
                if p == 0]
        cells = []
        for k in range(n):
            rec = {"index": int(k),
                   "rejected": bool(rejected[k])}
            if scan_axis_vals is not None:
                rec[axes[0]] = float(scan_axis_vals[k])
            g = cell_guard[k]
            if isinstance(g, dict) and "multi" not in g:
                rec.update({kk: (float(g[kk]) if isinstance(g[kk], float) else g[kk])
                            for kk in ("Neff", "pe_variance_sum", "sigma2_total",
                                       "threshold", "passes", "passes_legacy_floor")})
            elif g is not None:
                rec["multi"] = g["multi"]
            cells.append(rec)
        guard_block["cells"] = cells
    summary["guard"] = guard_block

    if not finite_any:
        # Fully guard-rejected configuration: no posterior exists to summarize.
        # Recorded as a first-class outcome rather than a crash, so the audit
        # can report WHICH configurations the new total-variance guard refuses.
        summary["logL_max"] = None
        if args.scan in ("h0", "f"):
            summary["h0_fixed" if args.scan == "f" else "f_fixed"] = (
                args.h0_fixed if args.scan == "f" else args.f_fixed
            )
        json_path.write_text(json.dumps(summary, indent=2))
        print(f"Wrote {json_path}")
        print("\n=== SUMMARY (ALL CELLS REJECTED) ===")
        print(json.dumps(_terse(summary), indent=2))
        return

    if args.scan in ("h0", "f"):
        x = H0_vals if args.scan == "h0" else f_vals
        name = "H0" if args.scan == "h0" else "f"
        truth = args.h0_true if args.scan == "h0" else args.f_true
        imax = int(np.nanargmax(np.where(finite, ll_safe, np.nan)))
        block = marginal_ci(x, ll_safe)
        block["map"] = float(x[imax])       # flat-prior MAP == grid argmax of logL
        block["argmax"] = float(x[imax])
        block["logL_max"] = lmax
        block["grid"] = [float(v) for v in x]
        add_truth_flags(block, truth)
        summary[name] = block
        summary["h0_fixed" if args.scan == "f" else "f_fixed"] = (
            args.h0_fixed if args.scan == "f" else args.f_fixed
        )
    else:  # joint
        i, j = np.unravel_index(int(np.nanargmax(np.where(finite, ll_safe, np.nan))),
                                ll_safe.shape)
        summary["map"] = {"H0": float(H0_vals[i]), "f": float(f_vals[j]), "logL": lmax}

        p2d = np.where(finite, np.exp(ll_safe - lmax), 0.0)  # -inf -> 0
        logp_H0 = np.log(np.maximum(np.trapz(p2d, f_vals, axis=1), 1e-300))
        logp_f = np.log(np.maximum(np.trapz(p2d, H0_vals, axis=0), 1e-300))
        bH0 = marginal_ci(H0_vals, logp_H0)
        bH0["map"] = float(H0_vals[i]); bH0["argmax"] = float(H0_vals[i])
        bf = marginal_ci(f_vals, logp_f)
        bf["map"] = float(f_vals[j]); bf["argmax"] = float(f_vals[j])
        add_truth_flags(bH0, args.h0_true)
        add_truth_flags(bf, args.f_true)
        summary["H0"] = bH0
        summary["f"] = bf

        # Gaussian-approx correlation rho from second moments of the normalized 2-D posterior.
        norm = np.trapz(np.trapz(p2d, f_vals, axis=1), H0_vals, axis=0)
        if np.isfinite(norm) and norm > 0:
            Zn = p2d / norm
            pH0 = np.trapz(Zn, f_vals, axis=1)
            pf = np.trapz(Zn, H0_vals, axis=0)
            EH0 = np.trapz(H0_vals * pH0, H0_vals)
            Ef = np.trapz(f_vals * pf, f_vals)
            VH0 = np.trapz((H0_vals - EH0) ** 2 * pH0, H0_vals)
            Vf = np.trapz((f_vals - Ef) ** 2 * pf, f_vals)
            H0g, fg = np.meshgrid(H0_vals, f_vals, indexing="ij")
            Cov = np.trapz(np.trapz((H0g - EH0) * (fg - Ef) * Zn, f_vals, axis=1),
                           H0_vals, axis=0)
            rho = float(Cov / np.sqrt(VH0 * Vf)) if VH0 > 0 and Vf > 0 else float("nan")
            summary["rho"] = rho
            summary["moments"] = {"E_H0": float(EH0), "E_f": float(Ef),
                                  "sigma_H0": float(np.sqrt(VH0)),
                                  "sigma_f": float(np.sqrt(Vf)), "cov": float(Cov)}
        else:
            summary["rho"] = float("nan")

    json_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {json_path}")
    print("\n=== SUMMARY ===")
    print(json.dumps(_terse(summary), indent=2))


if __name__ == "__main__":
    main()
