"""ln Z of a finished nested run whose initial live points partly sit on the driver's floor (-1e300).

The driver returns core's -inf (the variance guard) to dynesty as -1e300, a finite value, so all nlive
initial points are kept and the n_f of them on the floor form a likelihood plateau. dynesty 2.1.4 removes a
plateau linearly (sampler.py plateau_mode; utils._merge_two does the same when two parts are merged): after
the n_f floor points it sets the prior volume to

    X_dyn = (N + 1 - n_f) / (N + 1),      N = number of live points (2 nlive for a merged pair),

and every later prior volume, weight and ln Z carries that factor. The volume above the floor is q, and
k = N - n_f ~ Binomial(N, q), so ln X_dyn is biased by about +(1 - q)/(2 N q) and scatters by
sqrt((1 - q)/(N q)); dynesty's logzerr does not contain that scatter. tests/test_plateau_evidence.py
measures both on analytic likelihoods (tests/results/plateau_evidence.json).

The correction uses only what a finished run stores (results.pkl, or the merged Results of two parts):

    ln X_hat = ln(k/N) + n_f / (2 N k)                 (second-order bias-corrected; |bias| < 0.01 nats
                                                        for k >= 7, +0.05 at k ~ 4)
    ln Z_corr = ln Z_dyn - ln X_dyn + ln X_hat         (posterior weights unchanged: a constant factor)
    var_corr  = logzerr_dyn^2 + n_f / (N k)            (the binomial scatter of the plateau)

    from plateau_correction import plateau_corrected_logz
    c = plateau_corrected_logz(results)    # dynesty Results (static run with add_live, or merge_runs output)
    c["logz"], c["logzerr"], c["n_floor"], c["n_live"], c["shift"]

No jax, no dynesty import (dynesty only to read a pickle).
"""
from __future__ import annotations

import math

import numpy as np

FLOOR = -1e299          # the driver's floor is -1e300 (dynesty's _LOWL_VAL)


def _get(res, key, default=None):
    try:
        return res[key]
    except (KeyError, TypeError, AttributeError):
        return getattr(res, key, default)


def plateau_counts(res):
    """-> (n_floor, n_live_at_start, ln X dynesty assigned after the plateau). Checks that the floor points
    are the first n_floor samples (the plateau is worked through before any other point dies)."""
    logl = np.asarray(_get(res, "logl"), float)
    logvol = np.asarray(_get(res, "logvol"), float)
    on = logl <= FLOOR
    n_f = int(on.sum())
    if n_f and not on[:n_f].all():
        raise ValueError("floor points are not the first samples of the run")
    samples_n = _get(res, "samples_n")
    if samples_n is not None and len(np.atleast_1d(samples_n)) == len(logl):
        n_live = int(np.asarray(samples_n)[0])
    else:
        n_live = int(_get(res, "nlive"))
    lnx = float(logvol[n_f - 1]) if n_f else 0.0
    return n_f, n_live, lnx


def ln_volume_estimate(n_floor, n_live):
    """ln of the prior fraction above the floor from n_floor of n_live uniform draws on the floor, and its
    variance: ln(k/N) + n_f/(2 N k), n_f/(N k)."""
    k = n_live - n_floor
    if k <= 0:
        raise ValueError("no initial point above the floor")
    return math.log(k / n_live) + n_floor / (2.0 * n_live * k), n_floor / (n_live * k)


def dynesty_plateau_lnx(n_floor, n_live):
    """dynesty 2.1.4's ln X after a plateau of n_floor of n_live points (n_floor <= 1: one ordinary step)."""
    if n_floor == 0:
        return 0.0
    if n_floor == 1:
        return -math.log((n_live + 1.0) / n_live)
    return math.log((n_live + 1.0 - n_floor) / (n_live + 1.0))


def plateau_corrected_logz(res, n_live=None):
    """dynesty Results (static with add_live, or merged) -> dict with the corrected ln Z and its error."""
    n_f, n_live_res, lnx_dyn = plateau_counts(res)
    n_live = int(n_live or n_live_res)
    logz = float(np.asarray(_get(res, "logz"))[-1])
    logzerr = float(np.asarray(_get(res, "logzerr"))[-1])
    lnx_formula = dynesty_plateau_lnx(n_f, n_live)
    if n_f == 0:
        return {"logz": logz, "logzerr": logzerr, "logz_dynesty": logz, "logzerr_dynesty": logzerr,
                "n_floor": 0, "n_live": n_live, "lnX_dynesty": 0.0, "lnX_hat": 0.0, "shift": 0.0, "var_plateau": 0.0}
    if abs(lnx_formula - lnx_dyn) > 1e-9:
        raise ValueError(f"stored ln X after the plateau {lnx_dyn} is not dynesty's formula {lnx_formula} "
                         f"(n_floor {n_f}, n_live {n_live}): not a plateau run this function knows")
    lnx_hat, var_pl = ln_volume_estimate(n_f, n_live)
    shift = lnx_hat - lnx_dyn
    return {"logz": logz + shift, "logzerr": math.sqrt(logzerr ** 2 + var_pl), "logz_dynesty": logz,
            "logzerr_dynesty": logzerr, "n_floor": n_f, "n_live": n_live, "lnX_dynesty": lnx_dyn,
            "lnX_hat": lnx_hat, "shift": shift, "var_plateau": var_pl}


def binomial_plateau_table(n_live, qs):
    """Bias (E[ln X_est] - ln q) and scatter of dynesty's plateau volume and of ln X_hat, for k ~ Bin(N, q)
    conditioned on k >= 1 (exact sums)."""
    from scipy.stats import binom
    out = []
    k = np.arange(1, n_live + 1)
    for q in qs:
        p = binom.pmf(k, n_live, q)
        p = p / p.sum()
        dyn = np.log((k + 1.0) / (n_live + 1.0))
        hat = np.log(k / n_live) + (n_live - k) / (2.0 * n_live * k)
        row = {"q": q, "n_live": n_live, "p_k0": float(binom.pmf(0, n_live, q))}
        for name, e in (("dynesty", dyn), ("hat", hat)):
            m = float(np.sum(p * e))
            row[f"{name}_bias"] = m - math.log(q)
            row[f"{name}_sd"] = float(np.sqrt(np.sum(p * e ** 2) - m ** 2))
        row["sd_approx"] = math.sqrt((1 - q) / (n_live * q))
        out.append(row)
    return out
