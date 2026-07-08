#!/usr/bin/env python3
"""Gate figures: G2 coverage panel, G3 recovery panel, joint posterior contours.

Writes PNGs to working/gw_agn/figs/ and prints the multitracer H0-width
comparison (mean 68% half-widths per tracer).
"""
import glob
import json
import os

import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RES = os.path.join(BASE, 'results')
FIGS = os.path.join(BASE, 'figs')
os.makedirs(FIGS, exist_ok=True)
H0_TRUE = 67.74


def load_sums(tracer):
    out = []
    for fn in sorted(glob.glob(os.path.join(RES, 'coverage_' + tracer, '*.json'))):
        with open(fn) as f:
            out.append(json.load(f))
    return out


def fig_g2():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    widths = {}
    for ax, tracer, color in zip(axes, ('gal', 'agn'), ('C0', 'C3')):
        sums = load_sums(tracer)
        med = np.array([s['H0']['median'] for s in sums])
        lo = np.array([s['H0']['ci68'][0] for s in sums])
        hi = np.array([s['H0']['ci68'][1] for s in sums])
        n68 = sum(s['H0']['truth_in_ci68'] for s in sums)
        n90 = sum(s['H0']['truth_in_ci90'] for s in sums)
        x = np.arange(len(sums))
        ax.errorbar(x, med, yerr=[med - lo, hi - med], fmt='o', ms=3.5,
                    color=color, ecolor=color, alpha=0.85, lw=1.2)
        ax.axhline(H0_TRUE, color='k', ls='--', lw=1)
        ax.set_title('{}-only: 68% cov {}/{}  90% cov {}/{}'.format(
            tracer.upper(), n68, len(sums), n90, len(sums)))
        ax.set_xlabel('realization')
        widths[tracer] = float(np.mean(0.5 * (hi - lo)))
    axes[0].set_ylabel(r'$H_0$ [km/s/Mpc]')
    fig.suptitle(r'G2: single-tracer $H_0$ coverage (N=100 ev/real., exact estimator)')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, 'g2_coverage.png'), dpi=150)
    print('mean 68% H0 half-width: GAL {:.2f}, AGN {:.2f} (ratio {:.2f}x)'.format(
        widths['gal'], widths['agn'], widths['gal'] / widths['agn']))
    return widths


def fig_g3():
    sums = []
    for fn in sorted(glob.glob(os.path.join(RES, 'recovery', 'rec_fagn*.json'))):
        with open(fn) as f:
            sums.append(json.load(f))
    at = [s['alpha_agn']['truth'] for s in sums]
    med = np.array([s['alpha_agn']['median'] for s in sums])
    lo = np.array([s['alpha_agn']['ci68'][0] for s in sums])
    hi = np.array([s['alpha_agn']['ci68'][1] for s in sums])
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.errorbar(at, med, yerr=[med - lo, hi - med], fmt='o', color='C2', capsize=3)
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlabel(r'injected $\alpha_{\rm AGN}$ (eligible-pool truth)')
    ax.set_ylabel(r'recovered $\alpha_{\rm AGN}$ (median, 68% CI)')
    ax.set_title(r'G3: $\alpha_{\rm AGN}$ recovery, N=1000 events')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, 'g3_recovery.png'), dpi=150)


def fig_joint(tag='rec_fagn0.3', truth_alpha=0.307):
    with h5py.File(os.path.join(RES, 'recovery', tag + '.h5')) as f:
        ll = f['log_likelihood_grid'][:]
        H0 = f['H0_grid'][:]
        al = f['alpha_agn_grid'][:]
    ll = np.where(np.isfinite(ll), ll, -np.inf)
    p = np.exp(ll - ll.max())
    # 68/95 credible contours in 2-D
    ps = np.sort(p.ravel())[::-1]
    cs = np.cumsum(ps) / ps.sum()
    lev = [ps[np.searchsorted(cs, q)] for q in (0.95, 0.68)]
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.contourf(H0, al, p.T, levels=[lev[0], lev[1], p.max()], colors=['C0', 'C9'], alpha=0.7)
    ax.axvline(H0_TRUE, color='k', ls='--', lw=1)
    ax.axhline(truth_alpha, color='k', ls='--', lw=1)
    ax.set_xlabel(r'$H_0$ [km/s/Mpc]')
    ax.set_ylabel(r'$\alpha_{\rm AGN}$')
    ax.set_xlim(64, 71)
    ax.set_title(r'Joint posterior, $f_{\rm agn}=0.3$, N=1000 (68/95%)')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, 'joint_posterior_fagn0.3.png'), dpi=150)


if __name__ == '__main__':
    w = fig_g2()
    fig_g3()
    fig_joint()
    print('figures written to', FIGS)
