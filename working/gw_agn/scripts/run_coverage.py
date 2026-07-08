#!/usr/bin/env python3
"""G2 coverage batch: single-tracer H0 runs on the shared glass_prod catalog.

For each realization r: inject (seed_gw=seed0+r) -> PE clouds -> 2-D
(H0, alpha_agn) grid -> summary JSON. Resumable: existing outputs are
skipped. Run inside the jax env from the repo's code/ directory's parent
(paths are absolute).

Usage:
  python run_coverage.py --tracer gal --n-real 25
  python run_coverage.py --tracer agn --n-real 25
  python run_coverage.py --aggregate            # coverage table from summaries
"""
import argparse
import glob
import json
import os
import subprocess
import sys

HERE = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, HERE)
from make_configs import data_config, inference_config, write, CONFIGS, DATA, RESULTS  # noqa: E402

CODE = os.path.abspath(os.path.join(HERE, '..', '..', '..', 'code'))
H0_TRUE = 67.74

TRACER = {
    # f_agn, lambda_agn, alpha_true, seed0
    'gal': dict(f_agn=0.0, lambda_agn=0.0, alpha_true=0.0, seed0=6000),
    'agn': dict(f_agn=1.0, lambda_agn=0.5, alpha_true=1.0, seed0=7000),
}

# Owner-decision estimator knobs (GATES.md 2026-07-08): resolved KDE + enough samples
DZ = 3e-3
N_SAMPLES = 2000
N_GW = 100
DL_UNC = 0.1


def run(cmd, log):
    with open(log, 'a') as f:
        r = subprocess.run(cmd, cwd=CODE, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        raise RuntimeError('FAILED ({}): {} (log: {})'.format(r.returncode, ' '.join(cmd), log))


def one_realization(tracer, r):
    t = TRACER[tracer]
    seed_gw = t['seed0'] + r
    tag = 'cov_{}_r{:02d}'.format(tracer, r)
    outdir = os.path.join(RESULTS, 'coverage_' + tracer)
    fn_grid = os.path.join(outdir, '{}.h5'.format(tag))
    fn_sum = os.path.join(outdir, '{}.json'.format(tag))
    if os.path.exists(fn_sum):
        print(tag, 'already summarized, skipping')
        return
    dc = data_config(
        'glass_prod', tag_mocktype='_glass', seed=101,
        nbar_gal=1e-2, nbar_agn=1e-4, bias_gal=1.2, bias_agn=2.0,
        z_min=0.0, z_max=1.5, nside=64,
        f_agn=t['f_agn'], lambda_agn=t['lambda_agn'], N_gw=N_GW,
        seed_gw=seed_gw, z_max_gw=1.0,
        N_samples_gw=N_SAMPLES, dL_uncertainty_fac=DL_UNC,
        gw_tag=tag,
    )
    fn_dc = write(dc, os.path.join(CONFIGS, 'coverage', 'data_{}.yaml'.format(tag)))
    ic = inference_config(fn_dc, fn_grid, selection_mode='fixed_z',
                          N_H0=61, N_alpha_agn=21)
    ic['catalog'] = {'Dz_gal': DZ, 'Dz_agn': DZ}
    fn_ic = write(ic, os.path.join(CONFIGS, 'coverage', 'inf_{}.yaml'.format(fn_dc.split('data_')[-1][:-5])))
    log = os.path.join(RESULTS, 'coverage_' + tracer, tag + '.log')
    os.makedirs(outdir, exist_ok=True)
    open(log, 'w').close()
    run([sys.executable, 'make_mocks.py', fn_dc], log)
    run([sys.executable, 'generate_gwsamples.py', fn_dc], log)
    if not os.path.exists(fn_grid):
        run([sys.executable, 'run_inference.py', fn_ic, '--overwrite'], log)
    run([sys.executable, os.path.join(HERE, 'summarize_grid.py'), fn_grid,
         '--h0-true', str(H0_TRUE), '--alpha-true', str(t['alpha_true']),
         '--out', fn_sum], log)
    with open(fn_sum) as f:
        s = json.load(f)
    print('{}: H0 med={:.2f} 68%[{:.2f},{:.2f}] in68={} in90={} | alpha med={:.3f}'.format(
        tag, s['H0']['median'], *s['H0']['ci68'], s['H0']['truth_in_ci68'],
        s['H0']['truth_in_ci90'], s['alpha_agn']['median']), flush=True)


def aggregate():
    out = {}
    for tracer in TRACER:
        sums = sorted(glob.glob(os.path.join(RESULTS, 'coverage_' + tracer, '*.json')))
        if not sums:
            continue
        n = len(sums)
        rec = {'n': n, 'in68': 0, 'in90': 0, 'H0_medians': [], 'alpha_medians': []}
        for fn in sums:
            with open(fn) as f:
                s = json.load(f)
            rec['in68'] += int(s['H0']['truth_in_ci68'])
            rec['in90'] += int(s['H0']['truth_in_ci90'])
            rec['H0_medians'].append(s['H0']['median'])
            rec['alpha_medians'].append(s['alpha_agn']['median'])
        import numpy as np
        med = np.asarray(rec['H0_medians'])
        rec['H0_median_mean'] = float(med.mean())
        rec['H0_median_se'] = float(med.std(ddof=1) / max(1, np.sqrt(n)))
        rec['rate68'] = rec['in68'] / n
        rec['rate90'] = rec['in90'] / n
        out[tracer] = rec
        print('{}: n={} rate68={:.2f} rate90={:.2f} <H0med>={:.2f}+-{:.2f} (truth {})'.format(
            tracer, n, rec['rate68'], rec['rate90'],
            rec['H0_median_mean'], rec['H0_median_se'], H0_TRUE))
    fn = os.path.join(RESULTS, 'coverage_aggregate.json')
    with open(fn, 'w') as f:
        json.dump(out, f, indent=2)
    print('wrote', fn)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--tracer', choices=list(TRACER))
    ap.add_argument('--n-real', type=int, default=25)
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--aggregate', action='store_true')
    a = ap.parse_args()
    if a.aggregate:
        aggregate()
        sys.exit(0)
    if not a.tracer:
        ap.error('--tracer required unless --aggregate')
    for r in range(a.start, a.start + a.n_real):
        one_realization(a.tracer, r)
