#!/usr/bin/env python3
"""G3 recovery batch: joint (H0, alpha_agn) grids at f_agn in {0, 0.3, 0.7, 1.0}.

Shared glass_prod catalog; N_gw=1000 per injection set (full-N grids; G3a
scales sigma(alpha) to smaller N analytically). Resumable. alpha_true is
parsed from the injector's own 'Fraction in AGN' line (eligible-pool
convention).

Usage: python run_recovery.py [--sets 0.0 0.3 0.7 1.0]
"""
import argparse
import json
import os
import re
import subprocess
import sys

HERE = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, HERE)
from make_configs import data_config, inference_config, write, CONFIGS, RESULTS  # noqa: E402

CODE = os.path.abspath(os.path.join(HERE, '..', '..', '..', 'code'))
H0_TRUE = 67.74

SETS = {0.0: 5000, 0.3: 5001, 0.7: 5002, 1.0: 5003}  # f_agn -> seed_gw

DZ = 3e-3
N_SAMPLES = 2000
N_GW = 1000
DL_UNC = 0.1


def run(cmd, log):
    with open(log, 'a') as f:
        r = subprocess.run(cmd, cwd=CODE, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        raise RuntimeError('FAILED ({}): {} (log: {})'.format(r.returncode, ' '.join(cmd), log))


def alpha_true_from_log(log):
    with open(log) as f:
        txt = f.read()
    m = re.findall(r'Fraction in AGN: ([0-9.eE+-]+)', txt)
    if not m:
        raise RuntimeError('no "Fraction in AGN" line in ' + log)
    return float(m[-1])


def one_set(f_agn):
    seed_gw = SETS[f_agn]
    tag = 'rec_fagn{}'.format(f_agn)
    outdir = os.path.join(RESULTS, 'recovery')
    os.makedirs(outdir, exist_ok=True)
    fn_grid = os.path.join(outdir, tag + '.h5')
    fn_sum = os.path.join(outdir, tag + '.json')
    if os.path.exists(fn_sum):
        print(tag, 'already summarized, skipping')
        return
    dc = data_config(
        'glass_prod', tag_mocktype='_glass', seed=101,
        nbar_gal=1e-2, nbar_agn=1e-4, bias_gal=1.2, bias_agn=2.0,
        z_min=0.0, z_max=1.5, nside=64,
        f_agn=f_agn, lambda_agn=0.5, N_gw=N_GW, seed_gw=seed_gw, z_max_gw=1.0,
        N_samples_gw=N_SAMPLES, dL_uncertainty_fac=DL_UNC,
    )
    fn_dc = write(dc, os.path.join(CONFIGS, 'recovery', 'data_{}.yaml'.format(tag)))
    ic = inference_config(fn_dc, fn_grid, selection_mode='fixed_z', N_H0=61, N_alpha_agn=41)
    ic['catalog'] = {'Dz_gal': DZ, 'Dz_agn': DZ}
    fn_ic = write(ic, os.path.join(CONFIGS, 'recovery', 'inf_{}.yaml'.format(tag)))
    log = os.path.join(outdir, tag + '.log')
    open(log, 'w').close()
    run([sys.executable, 'make_mocks.py', fn_dc], log)
    a_true = alpha_true_from_log(log)
    run([sys.executable, 'generate_gwsamples.py', fn_dc], log)
    if not os.path.exists(fn_grid):
        run([sys.executable, 'run_inference.py', fn_ic, '--overwrite'], log)
    run([sys.executable, os.path.join(HERE, 'summarize_grid.py'), fn_grid,
         '--h0-true', str(H0_TRUE), '--alpha-true', str(a_true), '--out', fn_sum], log)
    with open(fn_sum) as f:
        s = json.load(f)
    print('{}: alpha_true={:.4f} -> alpha med={:.3f} 68%[{:.3f},{:.3f}] in68={} | '
          'H0 med={:.2f} 68%[{:.2f},{:.2f}] in68={}'.format(
              tag, a_true, s['alpha_agn']['median'], *s['alpha_agn']['ci68'],
              s['alpha_agn'].get('truth_in_ci68'), s['H0']['median'], *s['H0']['ci68'],
              s['H0']['truth_in_ci68']), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--sets', nargs='*', type=float, default=[0.0, 0.3, 0.7, 1.0])
    a = ap.parse_args()
    for f_agn in a.sets:
        one_set(f_agn)
