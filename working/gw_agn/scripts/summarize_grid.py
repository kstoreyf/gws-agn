#!/usr/bin/env python3
"""Summarize a saved (H0, alpha_agn) log-likelihood grid.

Flat priors on the grid ranges. Emits JSON: MAP, marginal medians, 68/90%
equal-tailed CIs for H0 and alpha_agn, and truth-coverage booleans when
truth values are passed.

Usage: python summarize_grid.py grid.h5 [--h0-true 67.74] [--alpha-true X] [--out out.json]
"""
import argparse
import json

import numpy as np
import h5py


def marginal_ci(x, logp_1d, levels=(0.68, 0.90)):
    p = np.exp(logp_1d - logp_1d.max())
    p /= np.trapz(p, x)
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (p[1:] + p[:-1]) * np.diff(x))])
    cdf /= cdf[-1]
    out = {'median': float(np.interp(0.5, cdf, x))}
    for lev in levels:
        lo = float(np.interp(0.5 - lev / 2, cdf, x))
        hi = float(np.interp(0.5 + lev / 2, cdf, x))
        out['ci{:.0f}'.format(lev * 100)] = [lo, hi]
    return out


def summarize(fn, h0_true=None, alpha_true=None):
    with h5py.File(fn, 'r') as f:
        ll = f['log_likelihood_grid'][:]
        H0 = f['H0_grid'][:]
        al = f['alpha_agn_grid'][:]
    ll = np.where(np.isfinite(ll), ll, -np.inf)
    i, j = np.unravel_index(np.argmax(ll), ll.shape)
    lmax = ll.max()
    # marginals with flat priors (trapezoid over the other axis)
    p2d = np.exp(ll - lmax)
    logp_H0 = np.log(np.maximum(np.trapz(p2d, al, axis=1), 1e-300))
    logp_al = np.log(np.maximum(np.trapz(p2d, H0, axis=0), 1e-300))
    s = {
        'file': fn,
        'map': {'H0': float(H0[i]), 'alpha_agn': float(al[j]), 'logL': float(lmax)},
        'H0': marginal_ci(H0, logp_H0),
        'alpha_agn': marginal_ci(al, logp_al),
        'n_neginf_cells': int(np.sum(~np.isfinite(ll))),
    }
    for name, truth in (('H0', h0_true), ('alpha_agn', alpha_true)):
        if truth is not None:
            s[name]['truth'] = truth
            for lev in ('ci68', 'ci90'):
                lo, hi = s[name][lev]
                s[name]['truth_in_' + lev] = bool(lo <= truth <= hi)
    return s


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('grid')
    ap.add_argument('--h0-true', type=float, default=None)
    ap.add_argument('--alpha-true', type=float, default=None)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    s = summarize(a.grid, a.h0_true, a.alpha_true)
    txt = json.dumps(s, indent=2)
    print(txt)
    if a.out:
        with open(a.out, 'w') as f:
            f.write(txt)
