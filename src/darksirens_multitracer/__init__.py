"""Tracer-pair-agnostic multitracer dark-siren likelihood.

Validated (gate G4) bit-for-bit against code/run_inference.py on the
two-tracer path with the matched fixed-z selection; generalizes the same
mixture algebra to K tracers. See working/gw_agn/GOAL.md for the math and
working/gw_agn/GATES.md for the validation ladder.

Named darksirens_multitracer (not darksirens) to avoid shadowing the
installed darksirens library; the API is designed to slot into it later.
"""
from .core import (
    setup_cosmology,
    load_gw_samples,
    load_tracer,
    tracer_beta_fixed,
    build_prior_functions,
    compute_log_likelihood,
    compute_likelihood_grid,
)

__all__ = [
    'setup_cosmology', 'load_gw_samples', 'load_tracer', 'tracer_beta_fixed',
    'build_prior_functions', 'compute_log_likelihood', 'compute_likelihood_grid',
]
