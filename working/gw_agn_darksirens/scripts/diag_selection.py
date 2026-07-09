#!/usr/bin/env python3
"""Diagnose the -inf logL from GB1: print (log_mu, Neff) inside the closure.

Monkeypatches factory.selection_log_correction with a jax.debug.print wrapper,
then evaluates the K=2 mixture likelihood at a few fcat_2 values, and the two
K=1 configs, printing selection internals + the PE-vs-selection split.
"""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

DATA = Path(__file__).resolve().parent.parent / "data"

import jax
import darksirens.likelihood.core as dscore
from darksirens.inference.data import load_all_data, validate_loaded_survey_shapes
from darksirens.likelihood.factory import make_likelihood
from darksirens.gw.populations import get_fixed_population_params
from darksirens.inference.prior import build_parameter_space

_orig_corr = dscore.selection_log_correction


def _wrapped(log_mu, Neff, nEvents, soft_guard=False):
    jax.debug.print(
        "SELDIAG log_mu={a}  Neff={b}  nEvents={c}  soft={d}",
        a=log_mu, b=Neff, c=nEvents, d=soft_guard,
    )
    corr = _orig_corr(log_mu, Neff, nEvents, soft_guard=soft_guard)
    jax.debug.print("SELDIAG correction={a}", a=corr)
    return corr


dscore.selection_log_correction = _wrapped


def build(universe_model, survey_paths, gw_path):
    opts = SimpleNamespace(
        universe_model=universe_model,
        survey_path=survey_paths[0],
        survey_paths=survey_paths,
        n_catalogs=len(survey_paths),
        gw_path=gw_path,
        gwselection_path=str(DATA / "injections.h5"),
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
    )
    fixed = {"Om0": 0.3075}
    data = load_all_data(opts)
    validate_loaded_survey_shapes(data)
    labels = build_parameter_space(
        opts.pop_model, opts.fix_population, opts.fix_cosmology, opts.fix_survey,
        fix_de=opts.fix_de, prior_overrides={},
        fixed_parameter_values=fixed, universe_model=opts.universe_model,
        shared_beta=True, shared_spin=True, shared_gamma=True,
        sky_model="isotropic", mark_model="none", mark_names=(),
        n_catalogs=opts.n_catalogs,
    )[0]
    pop_fid = get_fixed_population_params("powerlaw+peak", shared_beta=True,
                                          shared_spin=True, shared_gamma=True)
    like = make_likelihood(opts=opts, data=data, pop_params_fid=pop_fid,
                           fixed_parameter_values=fixed)
    return labels, like


LOG10N0_GAL = -5.50627668499162
LOG10N0_AGN = -7.508083961432144

print("========== K=2 dark_sirens (gal+agn), gw_fagn0.3 ==========")
labels, like = build("dark_sirens", [str(DATA / "gal.h5"), str(DATA / "agn.h5")],
                     str(DATA / "gw_fagn0.3.h5"))
print("labels:", labels)
base = {"H0": 67.74, "log10n0": LOG10N0_GAL, "delta": 0.0, "b_miss": 1.0,
        "sigma_kde": 0.0, "log10n0_c2": LOG10N0_AGN, "delta_c2": 0.0,
        "b_miss_c2": 1.0, "sigma_kde_c2": 0.0, "fcat_2": 0.307}
for f in (0.0, 0.307, 1.0):
    base["fcat_2"] = f
    coord = np.asarray([base[l] for l in labels])
    ll = float(like(coord))
    print(f"  fcat_2={f}: total logL = {ll}")

print("========== K=1 dark_sirens GAL, gw_cov_gal_r00 ==========")
labels, like = build("dark_sirens", [str(DATA / "gal.h5")],
                     str(DATA / "gw_cov_gal_r00.h5"))
b = {"H0": 67.74, "log10n0": LOG10N0_GAL, "delta": 0.0, "b_miss": 1.0, "sigma_kde": 0.0}
coord = np.asarray([b[l] for l in labels])
print("  total logL =", float(like(coord)))

print("========== K=1 dark_sirens AGN, gw_cov_agn_r00 ==========")
labels, like = build("dark_sirens", [str(DATA / "agn.h5")],
                     str(DATA / "gw_cov_agn_r00.h5"))
b = {"H0": 67.74, "log10n0": LOG10N0_AGN, "delta": 0.0, "b_miss": 1.0, "sigma_kde": 0.0}
coord = np.asarray([b[l] for l in labels])
print("  total logL =", float(like(coord)))

print("========== K=1 dark_sirens_complete GAL, gw_cov_gal_r00 ==========")
labels, like = build("dark_sirens_complete", [str(DATA / "gal.h5")],
                     str(DATA / "gw_cov_gal_r00.h5"))
b = {"H0": 67.74, "sigma_kde": 0.0}
coord = np.asarray([b[l] for l in labels])
print("  labels:", labels, " total logL =", float(like(coord)))
