#!/usr/bin/env python3
"""Find the rogue max-weight injection in the K=1 GAL dark_sirens config."""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
from pathlib import Path
from types import SimpleNamespace

import numpy as np

DATA = Path(__file__).resolve().parent.parent / "data"

import jax
import jax.numpy as jnp
import darksirens.likelihood.core as dscore
from darksirens.inference.data import load_all_data, validate_loaded_survey_shapes
from darksirens.likelihood.factory import make_likelihood
from darksirens.gw.populations import get_fixed_population_params
from darksirens.inference.prior import build_parameter_space

_orig_cst = dscore.compute_selection_term


def _wrapped(gw_sel, em_catalog_sel, log_weight_fn, Ndraw, nEvents,
             sel_batch_size=None, sky_log_weight_fn=None):
    ldw = log_weight_fn(gw_sel.m1det, gw_sel.q, gw_sel.dL, gw_sel.chieff,
                        gw_sel.pixels, gw_sel.prior_wt, em_catalog_sel)
    valid = gw_sel.valid & (gw_sel.prior_wt > 0.0)
    ldw = jnp.where(valid & jnp.isfinite(ldw), ldw, -jnp.inf)
    top_v, top_i = jax.lax.top_k(ldw, 5)
    for r in range(5):
        i = top_i[r]
        jax.debug.print(
            "TOP{r}: ldw={v}  dL={dl}  m1det={m1}  q={q}  chieff={chi}  "
            "pdraw={pw}  pix={pix}",
            r=r, v=top_v[r], dl=gw_sel.dL[i], m1=gw_sel.m1det[i], q=gw_sel.q[i],
            chi=gw_sel.chieff[i], pw=gw_sel.prior_wt[i], pix=gw_sel.pixels[i],
        )
    jax.debug.print("median finite ldw={m}  n_neginf={n}",
                    m=jnp.median(jnp.where(jnp.isfinite(ldw), ldw, jnp.nan)),
                    n=jnp.sum(~jnp.isfinite(ldw)))
    return _orig_cst(gw_sel, em_catalog_sel, log_weight_fn, Ndraw, nEvents,
                     sel_batch_size=sel_batch_size,
                     sky_log_weight_fn=sky_log_weight_fn)


dscore.compute_selection_term = _wrapped

opts = SimpleNamespace(
    universe_model="dark_sirens",
    survey_path=str(DATA / "gal.h5"),
    survey_paths=[str(DATA / "gal.h5")],
    n_catalogs=1,
    gw_path=str(DATA / "gw_cov_gal_r00.h5"),
    gwselection_path=str(DATA / "injections.h5"),
    use_LSS=False, lss_completion=None, lss_completions=[], lss_marginalize=False,
    counterpart=None, counterpart_nside=1, counterpart_dz=1e-4,
    bright_siren_sky_marginalized=False, drop_full_catalog=False,
    sky_model="isotropic", mark_model="none", marks=None, mark_names=(),
    sel_batch_size=None, redshift_prior_barrier="auto",
    selection_neff_guard="auto", sampler="tinyns",
    fix_population=True, fix_cosmology=False, fix_de=True, fix_survey=False,
    pop_model="powerlaw+peak", shared_beta=True, shared_spin=True,
    shared_gamma=True, complete_empty_pixel_policy="zero",
)
fixed = {"Om0": 0.3075}
data = load_all_data(opts)
validate_loaded_survey_shapes(data)
labels = build_parameter_space(
    opts.pop_model, True, False, False, fix_de=True, prior_overrides={},
    fixed_parameter_values=fixed, universe_model="dark_sirens",
    shared_beta=True, shared_spin=True, shared_gamma=True,
    sky_model="isotropic", mark_model="none", mark_names=(), n_catalogs=1,
)[0]
pop_fid = get_fixed_population_params("powerlaw+peak", shared_beta=True,
                                      shared_spin=True, shared_gamma=True)
like = make_likelihood(opts=opts, data=data, pop_params_fid=pop_fid,
                       fixed_parameter_values=fixed)
b = {"H0": 67.74, "log10n0": -5.50627668499162, "delta": 0.0, "b_miss": 1.0,
     "sigma_kde": 0.0}
coord = np.asarray([b[l] for l in labels])
print("total logL =", float(like(coord)))
