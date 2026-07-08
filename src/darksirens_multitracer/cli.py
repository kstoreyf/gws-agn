#!/usr/bin/env python3
"""Config-driven multitracer grid runner.

Usage: python -m darksirens_multitracer.cli config.yaml [--overwrite]

Config schema (tracer-pair-agnostic; K=2 required for the 2-D grid):

    fn_gwsamples: /abs/path/gwsamples.h5
    tracers:                       # ordered; alpha = weight of the LAST entry
      - {name: gal, fn_pixelated: /abs/path/cat_gal_pixelated.h5, Dz: 0.003}
      - {name: agn, fn_pixelated: /abs/path/cat_agn_pixelated.h5, Dz: 0.003}
    selection: {mode: fixed_z, z_max_gw: 1.0}
    grid: {N_H0: 61, H0_bounds: [50, 100], N_alpha: 21, alpha_bounds: [0, 1]}
    n_gw_inf: null
    seed_shuffle: 1234             # seeds the event-shuffle RNG (reproducibility)
    paths: {fn_inf: /abs/path/out_grid.h5}
"""
import argparse
import os

import numpy as np
import h5py
import yaml

from . import core


def main(config, overwrite=False):
    fn_inf = config['paths']['fn_inf']
    if os.path.exists(fn_inf) and not overwrite:
        raise FileExistsError(fn_inf + ' exists (use --overwrite)')

    sel = config['selection']
    if sel.get('mode', 'fixed_z') != 'fixed_z':
        raise ValueError("only selection.mode == 'fixed_z' is supported; the "
                         "legacy dl_horizon correction is mismatched to a "
                         "fixed-true-z host cut (working/gw_agn/BIAS_DIAGNOSIS.md)")
    z_max_gw = float(sel['z_max_gw'])

    tracers = [core.load_tracer(t['fn_pixelated'], float(t['Dz']), t.get('name'))
               for t in config['tracers']]
    npix = tracers[0]['npix']
    for t in tracers:
        assert t['npix'] == npix, 'tracer pixelizations disagree'
    import healpy as hp
    nside = hp.pixelfunc.npix2nside(npix)
    # betas in the float32 phase (legacy-parity; see core.tracer_beta_fixed)
    betas = [core.tracer_beta_fixed(t, z_max_gw) for t in tracers]

    # Legacy-parity precision order: catalog + cosmology tables in float32,
    # then x64 for the samples and all likelihood arithmetic (see enable_x64).
    cosmo = core.setup_cosmology()
    core.enable_x64()
    if config.get('seed_shuffle') is not None:
        np.random.seed(int(config['seed_shuffle']))
    gw = core.load_gw_samples(config['fn_gwsamples'], config.get('n_gw_inf'))

    g = config['grid']
    H0_grid = np.linspace(*g.get('H0_bounds', [50, 100]), g['N_H0'])
    alpha_grid = np.linspace(*g.get('alpha_bounds', [0, 1]), g['N_alpha'])
    ll = core.compute_likelihood_grid(gw, tracers, cosmo, nside, z_max_gw,
                                      H0_grid, alpha_grid, betas=betas)

    os.makedirs(os.path.dirname(fn_inf) or '.', exist_ok=True)
    with h5py.File(fn_inf, 'w') as f:
        f.create_dataset('log_likelihood_grid', data=ll)
        f.create_dataset('H0_grid', data=H0_grid)
        f.create_dataset('alpha_agn_grid', data=alpha_grid)
        f.attrs['selection_mode'] = 'fixed_z'
        f.attrs['z_max_gw'] = z_max_gw
        f.attrs['tracers'] = ','.join(t['name'] for t in tracers)
        for k, t in enumerate(tracers):
            f.attrs['beta_fixed_{}'.format(t['name'])] = betas[k]
    print('wrote', fn_inf)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('config')
    ap.add_argument('--overwrite', action='store_true')
    a = ap.parse_args()
    with open(a.config) as f:
        cfg = yaml.safe_load(f)
    main(cfg, overwrite=a.overwrite)
