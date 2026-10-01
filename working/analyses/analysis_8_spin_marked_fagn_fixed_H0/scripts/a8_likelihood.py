"""Reusable Analysis-8 likelihood builder (shared by Gate A and the production scan).

Wraps the Analysis-2 recipe of record --
``analysis_2_complete_catalog_H0_fagn/scripts/scan_h0f.py`` -- so the two shapes
compared by Gate A are built by ONE code path that differs only in the three
arguments that define them:

  OLD (analysis-2 shape)
      ``fix_population=True``, no ``per_catalog_pop_params``.  The mixture
      evaluates ``p_pop(theta) * sum_k f_k p_k(x)``: one shared population
      OUTSIDE the branch sum.

  NEW (analysis-8 shape)
      ``fix_population=False`` with all TWELVE base population parameters pinned
      by their LaTeX labels through ``fixed_parameter_values`` at the
      powerlaw+peak fiducial, plus ``per_catalog_pop_params=('mu_chi_c2',)``.
      The mixture evaluates ``sum_k f_k p_k(x) p_pop(theta | Lambda_k)``:
      a per-tracer population INSIDE the branch sum.

Nothing about the likelihood is reimplemented here.  ``load_all_data``,
``validate_loaded_survey_shapes``, ``build_parameter_space``,
``get_fixed_population_params`` and ``make_likelihood`` are called exactly as
``scan_h0f.main`` calls them, with the same ``opts`` namespace, the same
nuisance defaults, the same guard spy and the same catalog-KDE window.

The guard spy is extended by ONE field relative to scan_h0f's: it also records
the VALUE ``selection_log_correction`` returns.  That is the selection
contribution to logL, so PE = total - selection without a second evaluation.
The spy remains a pass-through, so the returned logL is bit-identical to an
uninstrumented run.

Environment (must hold before this module is imported):

    export PYTHONPATH=/hildafs/projects/phy230014p/magana/src/darksirens-a8
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
"""
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

# --------------------------------------------------------------------------- #
# Pins of record (all three copied from the analysis-2 configuration)
# --------------------------------------------------------------------------- #
DARKSIRENS_REPO = os.environ.get(
    "DARKSIRENS_SRC", "/hildafs/projects/phy230014p/magana/src/darksirens-a8"
)
GWS_AGN_REPO = "/hildafs/projects/phy230014p/magana/gws-agn"
MERGE_SHA = "2b86a2d"           # the pin the seed-100 dataset was generated on
OM0_FID = 0.3075
H0_FID = 67.74
POP_MODEL = "powerlaw+peak"

DATA_ROOT = Path("/hildafs/projects/phy230014p/magana/gws-agn/working/data/seed100")
SURVEY_GAL = str(DATA_ROOT / "surveys" / "survey_gal_complete_ns32.h5")
SURVEY_AGN = str(DATA_ROOT / "surveys" / "survey_agn_complete_ns32.h5")
GW_PATH = str(DATA_ROOT / "events" / "events.h5")
# The Gate-B marked mock (dmu_chi planted at +0.10 on the AGN branch).  Gate C
# runs ALL THREE arms on this file; Gates A and B kept the unmarked default.
GW_PATH_MARKED = str(DATA_ROOT / "events" / "events_marked_dmu0p10.h5")
GWSEL_PATH = str(DATA_ROOT / "injections" / "injections_targeted.h5")

# scan_h0f.NUISANCE_DEFAULTS, verbatim.
NUISANCE_DEFAULTS = {
    "delta": 0.0,
    "b_miss": 1.0,
    "sigma_kde": 0.0,
    "delta_c2": 0.0,
    "b_miss_c2": 1.0,
    "sigma_kde_c2": 0.0,
}

# The analysis-2 settings of record.
SETTINGS = dict(
    universe_model="dark_sirens",
    catalog_sky_weighting="field",
    log10n0=-24.0,
    log10n0_c2=-24.0,
    selection_neff_guard="hard",
    max_likelihood_variance=1e6,
    kde_window=4096,
    kde_window_nsigma=8.0,
    sel_batch_size=50000,
    pe_event_block=25,
)

_ANALYSIS_2_SCRIPTS = (
    "/hildafs/projects/phy230014p/magana/gws-agn/working/analyses/"
    "analysis_2_complete_catalog_H0_fagn/scripts"
)


def import_scan_h0f():
    """The analysis-2 driver as a module (its ``main()`` is ``__main__``-guarded).

    Gate A imports ``marginal_ci`` from here rather than reimplementing the
    posterior convention.
    """
    if _ANALYSIS_2_SCRIPTS not in sys.path:
        sys.path.insert(0, _ANALYSIS_2_SCRIPTS)
    import scan_h0f
    return scan_h0f


# --------------------------------------------------------------------------- #
# Process-wide setup (order matters: env -> import -> spy -> kde -> likelihood)
# --------------------------------------------------------------------------- #
_GUARD_HITS = []
_SPY_INSTALLED = False
_KDE_CONFIGURED = False


def set_env(guard_record=True):
    """Set the JAX environment.  MUST run before darksirens is imported."""
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    # jax.debug.callback needs a local CPU device to place its inputs on.
    os.environ["JAX_PLATFORMS"] = "cuda,cpu" if guard_record else "cuda"


def install_guard_spy():
    """Pass-through spy on ``darksirens.likelihood.core.selection_log_correction``.

    Identical to ``scan_h0f``'s spy plus one field: ``sel_logL``, the value the
    real function returns.  Records into the module-level ``_GUARD_HITS`` list,
    which callers clear before each evaluation.
    """
    global _SPY_INSTALLED
    if _SPY_INSTALLED:
        return _GUARD_HITS
    import jax
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
        out = _true_slc(log_mu, Neff, nEvents, soft_guard=soft_guard,
                        max_likelihood_variance=max_likelihood_variance,
                        pe_variance_sum=pe_variance_sum)

        def _record(log_mu_v, Neff_v, pe_var_v, thr_v, sel_v):
            Neff_f = float(Neff_v)
            pe_f = float(pe_var_v)
            sel_var = (n * n) / Neff_f if Neff_f > 0 else float("inf")
            _GUARD_HITS.append({
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
                "sel_logL": float(sel_v),
            })

        jax.debug.callback(_record, log_mu, Neff, pe_variance_sum, threshold, out)
        return out

    ds_core.selection_log_correction = _guard_spy
    _SPY_INSTALLED = True
    return _GUARD_HITS


def configure_kde(size=None, n_sigma=None):
    """Catalog-KDE window.  MUST precede the first likelihood trace."""
    global _KDE_CONFIGURED
    if _KDE_CONFIGURED:
        return
    from darksirens.redshift.catalog import configure_catalog_kde_window
    configure_catalog_kde_window(
        size=int(SETTINGS["kde_window"] if size is None else size),
        n_sigma=float(SETTINGS["kde_window_nsigma"] if n_sigma is None else n_sigma),
    )
    _KDE_CONFIGURED = True


# --------------------------------------------------------------------------- #
# opts namespace (scan_h0f.main's, verbatim, then the same CLI adjustments)
# --------------------------------------------------------------------------- #
def build_opts(survey_paths, **overrides):
    survey_paths = [str(p) for p in survey_paths]
    opts = SimpleNamespace(
        universe_model="dark_sirens",
        survey_path=survey_paths[0],
        survey_paths=survey_paths,
        n_catalogs=len(survey_paths),
        gw_path=GW_PATH,
        gwselection_path=GWSEL_PATH,
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
        sel_batch_size=SETTINGS["sel_batch_size"],
        redshift_prior_barrier="auto",
        selection_neff_guard=SETTINGS["selection_neff_guard"],
        sampler="tinyns",
        fix_population=True,
        fix_cosmology=False,
        fix_de=True,
        fix_survey=False,
        pop_model=POP_MODEL,
        shared_beta=True,
        shared_spin=True,
        shared_gamma=True,
        complete_empty_pixel_policy="zero",
        catalog_sky_weighting=SETTINGS["catalog_sky_weighting"],
        pe_event_block=SETTINGS["pe_event_block"],
        max_likelihood_variance=SETTINGS["max_likelihood_variance"],
    )
    # scan_h0f resolves the guard to the boolean the factory consumes exactly as
    # darksirens.cli.inference does (soft => True; auto => soft only for numpyro).
    opts.selection_neff_soft_guard = (
        opts.selection_neff_guard == "soft"
        or (opts.selection_neff_guard == "auto" and opts.sampler == "numpyro")
    )
    for key, val in overrides.items():
        setattr(opts, key, val)
    return opts


# --------------------------------------------------------------------------- #
# The two shapes
# --------------------------------------------------------------------------- #
def population_labels_and_fiducial():
    """``(latex labels, plain names, fiducial vector)`` of the 12-slot pop block."""
    from darksirens.inference.prior import pop_model_prior_parser
    from darksirens.gw.populations import get_fixed_population_params, get_model

    _lo, _hi, labels, _kinds, _name = pop_model_prior_parser(
        POP_MODEL, shared_beta=True, shared_spin=True, shared_gamma=True
    )
    model = get_model(POP_MODEL, shared_beta=True, shared_spin=True, shared_gamma=True)
    plain = [spec.name for spec in model.param_specs]
    fid = np.asarray(
        get_fixed_population_params(
            POP_MODEL, shared_beta=True, shared_spin=True, shared_gamma=True
        ),
        dtype=float,
    )
    return list(labels), list(plain), fid


def fixed_parameter_values_for(mode):
    """``fixed_parameter_values`` for ``mode`` in {'old', 'new'}.

    'old' pins only ``Om0`` -- the analysis-2 configuration verbatim.

    'new' additionally pins all twelve base population parameters at the
    powerlaw+peak fiducial.  They are pinned by their LaTeX PRIOR LABELS
    (``'$\\mu_\\chi$'``), because ``build_parameter_space`` accepts only those
    for the BASE block; the plain ASCII spelling is aliased for the ``_c{k}``
    blocks only.
    """
    fpv = {"Om0": OM0_FID}
    if mode == "new":
        labels, _plain, fid = population_labels_and_fiducial()
        for label, value in zip(labels, fid):
            fpv[label] = float(value)
    elif mode != "old":
        raise ValueError(f"mode must be 'old' or 'new'; got {mode!r}")
    return fpv


MU_CHI_C2_LABEL = "$\\mu_\\chi$_c2"


class LikelihoodCell:
    """A built likelihood plus the label bookkeeping needed to evaluate a cell."""

    def __init__(self, name, mode, survey_paths, data, opts, labels, likelihood,
                 fixed_parameter_values, per_catalog_pop_params, guard_hits):
        self.name = name
        self.mode = mode
        self.survey_paths = list(survey_paths)
        self.n_catalogs = len(survey_paths)
        self.data = data
        self.opts = opts
        self.labels = list(labels)
        self.likelihood = likelihood
        self.fixed_parameter_values = fixed_parameter_values
        self.per_catalog_pop_params = tuple(per_catalog_pop_params)
        self._guard_hits = guard_hits
        self._idx = {lbl: i for i, lbl in enumerate(self.labels)}

        point = dict(NUISANCE_DEFAULTS)
        point["H0"] = H0_FID
        point["log10n0"] = SETTINGS["log10n0"]
        point["log10n0_c2"] = SETTINGS["log10n0_c2"]
        point["fcat_2"] = 0.0
        point[MU_CHI_C2_LABEL] = 0.0
        missing = [lbl for lbl in self.labels if lbl not in point]
        if missing:
            raise KeyError(f"no value supplied for required label(s): {missing}")
        self.base_point = point
        self.base = np.asarray([float(point[lbl]) for lbl in self.labels], dtype=float)

    def coord(self, **overrides):
        c = self.base.copy()
        for lbl, val in overrides.items():
            if lbl not in self._idx:
                raise KeyError(
                    f"{self.name}: label {lbl!r} is not sampled "
                    f"(labels={self.labels})"
                )
            c[self._idx[lbl]] = float(val)
        return c

    def evaluate(self, **overrides):
        """Evaluate one cell; return logL plus the guard/selection bookkeeping."""
        import time
        c = self.coord(**overrides)
        self._guard_hits.clear()
        t0 = time.time()
        ll = float(self.likelihood(c))
        dt = time.time() - t0
        rec = {
            "config": self.name,
            "coord": {lbl: float(v) for lbl, v in zip(self.labels, c)},
            "logL": ll,
            "logL_hex": float(ll).hex(),
            "seconds": dt,
            "finite": bool(np.isfinite(ll)),
        }
        hits = [dict(h) for h in self._guard_hits]
        rec["n_guard_calls"] = len(hits)
        if len(hits) == 1:
            g = hits[0]
            rec["guard"] = g
            rec["log_mu"] = g["log_mu"]
            rec["mu"] = float(np.exp(g["log_mu"]))
            rec["Neff"] = g["Neff"]
            rec["threshold"] = g["threshold"]
            rec["guard_passes"] = g["passes"]
            rec["logL_selection"] = g["sel_logL"]
            rec["logL_pe"] = ll - g["sel_logL"]
        elif hits:
            rec["guard_multi"] = hits
        return rec


def build(name, mode, survey_paths, data=None, verbose=True, gw_path=None):
    """Build one ``LikelihoodCell``.

    ``mode`` selects the shape ('old' or 'new'); ``survey_paths`` selects K.
    ``data`` may be reused across builds that share the same survey list.
    ``gw_path`` overrides the events file (``GW_PATH`` when None); ``data`` must
    have been loaded with the SAME events file.
    """
    import time
    from darksirens.inference.data import load_all_data, validate_loaded_survey_shapes
    from darksirens.likelihood.factory import make_likelihood
    from darksirens.inference.prior import build_parameter_space
    from darksirens.gw.populations import get_fixed_population_params

    guard_hits = install_guard_spy()
    configure_kde()

    opts = build_opts(survey_paths,
                      **({} if gw_path is None else {"gw_path": str(gw_path)}))
    n_catalogs = opts.n_catalogs
    per_catalog_pop_params = ()
    if mode == "new":
        opts.fix_population = False
        if n_catalogs >= 2:
            per_catalog_pop_params = ("mu_chi_c2",)
    opts.per_catalog_pop_params = per_catalog_pop_params

    fpv = fixed_parameter_values_for(mode)

    if data is None:
        t0 = time.time()
        data = load_all_data(opts)
        validate_loaded_survey_shapes(data)
        if verbose:
            print(f"[{name}] load_all_data: {time.time() - t0:.2f}s "
                  f"nEvents={data['nEvents']} nsamp={data['nsamp']} "
                  f"Ndraw={data['Ndraw']} n_catalogs={n_catalogs}")

    res = build_parameter_space(
        opts.pop_model, opts.fix_population, opts.fix_cosmology, opts.fix_survey,
        fix_de=opts.fix_de, prior_overrides={},
        fixed_parameter_values=fpv, universe_model=opts.universe_model,
        shared_beta=opts.shared_beta, shared_spin=opts.shared_spin,
        shared_gamma=opts.shared_gamma, sky_model=opts.sky_model,
        mark_model=opts.mark_model, mark_names=opts.mark_names,
        n_catalogs=n_catalogs,
        lss_completion_active=[False] * n_catalogs,
        use_lss=bool(opts.use_LSS),
        mark_names_by_catalog=None,
        per_catalog_pop_params=per_catalog_pop_params,
    )
    labels = list(res[0])
    if verbose:
        print(f"[{name}] free parameters ({len(labels)}): {labels}")

    pop_params_fid = get_fixed_population_params(
        opts.pop_model, shared_beta=opts.shared_beta, shared_spin=opts.shared_spin,
        shared_gamma=opts.shared_gamma,
    )
    t0 = time.time()
    likelihood = make_likelihood(
        opts=opts, data=data, pop_params_fid=pop_params_fid,
        fixed_parameter_values=fpv,
    )
    if verbose:
        print(f"[{name}] make_likelihood: {time.time() - t0:.2f}s")

    return LikelihoodCell(
        name=name, mode=mode, survey_paths=survey_paths, data=data, opts=opts,
        labels=labels, likelihood=likelihood, fixed_parameter_values=fpv,
        per_catalog_pop_params=per_catalog_pop_params, guard_hits=guard_hits,
    )


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #
def provenance(gw_path=None, survey_paths=None):
    def _sha(repo):
        return subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"],
                              capture_output=True, text=True).stdout.strip()

    def _dirty(repo):
        out = subprocess.run(["git", "-C", repo, "status", "--porcelain"],
                             capture_output=True, text=True).stdout
        return bool(out.strip())

    import darksirens
    ancestor = subprocess.run(
        ["git", "-C", DARKSIRENS_REPO, "merge-base", "--is-ancestor",
         MERGE_SHA, "HEAD"]
    ).returncode == 0
    prov = {
        "gws_agn_sha": _sha(GWS_AGN_REPO),
        "gws_agn_dirty": _dirty(GWS_AGN_REPO),
        "darksirens_repo": DARKSIRENS_REPO,
        "darksirens_sha": _sha(DARKSIRENS_REPO),
        "darksirens_dirty": _dirty(DARKSIRENS_REPO),
        "darksirens_module_file": darksirens.__file__,
        "darksirens_pinned_base_sha": MERGE_SHA,
        "darksirens_pinned_base_is_ancestor": bool(ancestor),
        "inputs": {
            "survey_gal": SURVEY_GAL,
            "survey_agn": SURVEY_AGN,
            "gw_path": GW_PATH,
            "gw_path_marked": GW_PATH_MARKED,
            "gw_path_used": GW_PATH if gw_path is None else str(gw_path),
            "survey_paths_used": (None if survey_paths is None
                                  else [str(x) for x in survey_paths]),
            "gwselection_path": GWSEL_PATH,
        },
        "settings": dict(SETTINGS),
        "H0_fixed": H0_FID,
        "Om0_fixed": OM0_FID,
        "pop_model": POP_MODEL,
    }
    assert ancestor, (
        f"[fatal] pinned base {MERGE_SHA} is NOT an ancestor of darksirens HEAD "
        f"({prov['darksirens_sha']}); wrong checkout."
    )
    if not prov["darksirens_module_file"].startswith(DARKSIRENS_REPO):
        raise RuntimeError(
            "[fatal] darksirens did not resolve under darksirens-a8: "
            f"{prov['darksirens_module_file']}.  Set PYTHONPATH="
            f"{DARKSIRENS_REPO}."
        )
    return prov
