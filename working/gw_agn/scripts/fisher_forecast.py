#!/usr/bin/env python3
"""G3a: local-curvature Fisher forecast of sigma(alpha_agn), sigma(H0) vs N.

Fits a 2-D quadratic to logL around the grid MAP (window +-w steps),
inverts the Hessian, and scales sigma by sqrt(N_grid/N') for iid events.

Usage: python fisher_forecast.py grid.h5 --n-grid 1000 [--n-targets 50 100 200 1000] [--window 4]
"""
import argparse
import json

import numpy as np
import h5py


def forecast(fn, n_grid, n_targets, window=4):
    with h5py.File(fn, 'r') as f:
        ll = f['log_likelihood_grid'][:]
        H0 = f['H0_grid'][:]
        al = f['alpha_agn_grid'][:]
    ll = np.where(np.isfinite(ll), ll, -np.inf)
    i, j = np.unravel_index(np.argmax(ll), ll.shape)
    i0, i1 = max(0, i - window), min(len(H0), i + window + 1)
    j0, j1 = max(0, j - window), min(len(al), j + window + 1)
    hs, as_ = np.meshgrid(H0[i0:i1], al[j0:j1], indexing='ij')
    ls = ll[i0:i1, j0:j1]
    m = np.isfinite(ls)
    # quadratic: a + b x + c y + d x^2 + e y^2 + f xy  (x=H0-H0map, y=al-almap)
    x = (hs - H0[i])[m]
    y = (as_ - al[j])[m]
    A = np.stack([np.ones_like(x), x, y, x**2, y**2, x * y], axis=1)
    coef, *_ = np.linalg.lstsq(A, ls[m], rcond=None)
    Hmat = -np.array([[2 * coef[3], coef[5]], [coef[5], 2 * coef[4]]])
    cov = np.linalg.inv(Hmat)
    sd = np.sqrt(np.diag(cov))
    rho = cov[0, 1] / (sd[0] * sd[1])
    out = {
        'file': fn, 'map': {'H0': float(H0[i]), 'alpha_agn': float(al[j])},
        'n_grid': n_grid,
        'sigma_H0': float(sd[0]), 'sigma_alpha': float(sd[1]), 'rho_H0_alpha': float(rho),
        'window_pts': int(m.sum()),
        'scaling': {},
    }
    for n in n_targets:
        s = np.sqrt(n_grid / n)
        out['scaling'][str(n)] = {
            'sigma_H0': float(sd[0] * s), 'sigma_alpha': float(sd[1] * s),
            'alpha_prior_dominated_at_0.25': bool(sd[1] * s > 0.25),
        }
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('grid')
    ap.add_argument('--n-grid', type=int, required=True)
    ap.add_argument('--n-targets', nargs='*', type=int, default=[50, 100, 200, 1000])
    ap.add_argument('--window', type=int, default=4)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    r = forecast(a.grid, a.n_grid, a.n_targets, a.window)
    txt = json.dumps(r, indent=2)
    print(txt)
    if a.out:
        with open(a.out, 'w') as f:
            f.write(txt)
