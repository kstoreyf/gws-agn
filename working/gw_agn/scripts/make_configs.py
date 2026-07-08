#!/usr/bin/env python3
"""Config factory for the multitracer fagn program (working/gw_agn).

Emits data configs (mock -> inject -> gwsamples -> pixelize schema of
code/*.py) and inference configs (run_inference.py schema) with absolute
paths under working/gw_agn/{data,configs,results}.

Usage: python make_configs.py <case> [case ...]
Cases: s0_smoke | glass_prod | help
"""
import os
import sys
import yaml

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CONFIGS = os.path.join(BASE, 'configs')
DATA = os.path.join(BASE, 'data')
RESULTS = os.path.join(BASE, 'results')

COSMO = {'h': 0.6774, 'H0': 67.74, 'Om0': 0.3075, 'Ob0': 0.0486}


def data_config(tag, *, tag_mocktype, seed, nbar_gal, nbar_agn, bias_gal, bias_agn,
                z_min, z_max, nside, f_agn, lambda_agn, N_gw, seed_gw, z_max_gw,
                N_samples_gw, dL_uncertainty_fac, pe_centering='obs', seed_samples=None,
                mass_mean=35.0, mass_std=5.0, ra_uncertainty=0.01, dec_uncertainty=0.01,
                mass_uncertainty=1.5, dir_mock=None, gw_tag=None):
    """Build one data-config dict. dir_mock defaults to DATA/<tag>; gw_tag
    distinguishes multiple injection sets sharing one catalog."""
    if dir_mock is None:
        dir_mock = os.path.join(DATA, tag)
    if gw_tag is None:
        gw_tag = 'fagn{}_lam{}_seedgw{}'.format(f_agn, lambda_agn, seed_gw)
    if seed_samples is None:
        seed_samples = seed_gw + 500000
    cfg = {
        'mock_catalog': {
            'seed': seed, 'nbar_gal': nbar_gal, 'nbar_agn': nbar_agn,
            'bias_gal': bias_gal, 'bias_agn': bias_agn,
            'z_min': z_min, 'z_max': z_max, 'nside': nside,
            'gamma_agn': 0.0, 'gamma_gal': 0.0,
        },
        'gw_injection': {
            'f_agn': f_agn, 'lambda_agn': lambda_agn, 'N_gw': N_gw,
            'seed_gw': seed_gw, 'z_max_gw': z_max_gw,
        },
        'paths': {
            'dir_mock': dir_mock + '/',
            'name_cat': 'mock_catalog.h5',
            'name_gw': 'gws_{}.h5'.format(gw_tag),
            'name_gwsamples': 'gwsamples_{}_dLunc{}_{}.h5'.format(gw_tag, dL_uncertainty_fac, pe_centering),
            'name_cat_gal_pixelated': 'cat_gal_pixelated_nside{}.h5'.format(nside),
            'name_cat_agn_pixelated': 'cat_agn_pixelated_nside{}.h5'.format(nside),
            'tag_cat': '_' + tag, 'tag_pix': '_nside{}'.format(nside),
            'tag_gw': '_' + gw_tag, 'tag_gwsamp': '_dLunc{}'.format(dL_uncertainty_fac),
            'tag_mocktype': tag_mocktype,
        },
        'gw_samples': {
            'N_samples_gw': N_samples_gw, 'mass_mean': mass_mean, 'mass_std': mass_std,
            'ra_uncertainty': ra_uncertainty, 'dec_uncertainty': dec_uncertainty,
            'dL_uncertainty_fac': dL_uncertainty_fac, 'mass_uncertainty': mass_uncertainty,
            'pe_centering': pe_centering, 'seed_samples': seed_samples,
        },
        'cosmology': dict(COSMO),
        'pixelization': {'nside': nside},
    }
    return cfg


def inference_config(fn_config_data, fn_inf, *, selection_mode='fixed_z',
                     N_H0=61, N_alpha_agn=21, N_gw_inf=None,
                     parameters_vary=('H0', 'alpha_agn')):
    return {
        'fn_config_data': fn_config_data,
        'mode_inf': 'grid',
        'selection_mode': selection_mode,
        'mcmc': {'N_walkers': 32, 'N_steps': 100, 'burnin_frac': 0.3, 'seed_mcmc': 42},
        'grid': {'N_H0': N_H0, 'N_alpha_agn': N_alpha_agn},
        'parameters': {'parameters_vary': list(parameters_vary)},
        'N_gw_inf': N_gw_inf,
        'catalog': {'Dz_gal': 0.0001, 'Dz_agn': 0.0001},
        'paths': {'fn_inf': fn_inf},
    }


def write(cfg, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    print('wrote', path)
    return path


def case_s0_smoke():
    """Tiny uniform-catalog end-to-end smoke (machinery bring-up, gate S0)."""
    dc = data_config(
        's0_uniform', tag_mocktype='_uniform', seed=7,
        nbar_gal=1e-3, nbar_agn=1e-5, bias_gal=None, bias_agn=None,
        z_min=0.0, z_max=1.5, nside=32,
        f_agn=0.5, lambda_agn=0.5, N_gw=60, seed_gw=1007, z_max_gw=1.0,
        N_samples_gw=500, dL_uncertainty_fac=0.1,
    )
    p = write(dc, os.path.join(CONFIGS, 'data_s0_smoke.yaml'))
    ic = inference_config(p, os.path.join(RESULTS, 's0_smoke', 'grid_fixed_z.h5'),
                          selection_mode='fixed_z', N_H0=51, N_alpha_agn=21)
    write(ic, os.path.join(CONFIGS, 'inf_s0_smoke_fixed_z.yaml'))


def case_glass_prod():
    """Production GLASS catalog with density+bias contrast; first recovery injection."""
    dc = data_config(
        'glass_prod', tag_mocktype='_glass', seed=101,
        nbar_gal=1e-2, nbar_agn=1e-4, bias_gal=1.2, bias_agn=2.0,
        z_min=0.0, z_max=1.5, nside=64,
        f_agn=0.3, lambda_agn=0.5, N_gw=1000, seed_gw=5001, z_max_gw=1.0,
        N_samples_gw=1000, dL_uncertainty_fac=0.1,
    )
    write(dc, os.path.join(CONFIGS, 'data_glass_prod_fagn0.3.yaml'))


CASES = {'s0_smoke': case_s0_smoke, 'glass_prod': case_glass_prod}

if __name__ == '__main__':
    args = sys.argv[1:] or ['help']
    if args == ['help']:
        print(__doc__)
        sys.exit(0)
    for a in args:
        CASES[a]()
