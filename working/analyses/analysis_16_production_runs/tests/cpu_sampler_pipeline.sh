#!/bin/bash
# CPU rehearsal of test (c)'s pipeline on the cheap no-catalog model (20 events, 20,000 injections,
# nlive 40): Phase A with state copies, the four Phase B switches from the last copy, the report.
# Checks the mechanics only (switching, --init-from, --maxiter-more, insertion ranks, projection).
set -u
A16=/hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_16_production_runs
cd $A16
export JAX_PLATFORMS=cpu PYTHONPATH= XLA_PYTHON_CLIENT_PREALLOCATE=false
PY=/hildafs/projects/phy230014p/magana/envs/darksirens-core-e7c3007/bin/python
M="--mode nested --analysis single_population --catalogs none --events-kind host_dependent --realization 100
   --nsamp 2000 --dry-run --nlive 40 --out-root tests/cpu_sampler_runs --checkpoint-every 30 --chunk-calls 300"
D=tests/cpu_sampler_runs/single_population/no_catalog/complete/host_dependent/realization_100
rm -rf tests/cpu_sampler_runs
$PY scripts/prod_driver.py $M --tag phaseA_unif_b0 --sample unif --enlarge 1.25 --bootstrap 0 --maxiter-total 300 \
    --save-at-iterations 150,300 > logs/dryrun/cpu_sampler_A.log 2>&1; echo "A exit $?"
S=$D/nested_1_phaseA_unif_b0/state_it300.save
B="$M --init-from $S --maxiter-more 40"
$PY scripts/prod_driver.py $B --tag phaseB_rslice --switch-sample rslice --enlarge 1.25 --bootstrap 0 --slices 8 > logs/dryrun/cpu_sampler_B1.log 2>&1; echo "B rslice exit $?"
$PY scripts/prod_driver.py $B --tag phaseB_rwalk --switch-sample rwalk --walks 25 > logs/dryrun/cpu_sampler_B2.log 2>&1; echo "B rwalk exit $?"
$PY scripts/prod_driver.py $B --tag phaseB_unif_b0 --switch-sample unif --enlarge 1.25 --bootstrap 0 > logs/dryrun/cpu_sampler_B3.log 2>&1; echo "B unif_b0 exit $?"
$PY scripts/prod_driver.py $B --tag phaseB_unif_boot5 --switch-sample unif --bootstrap 5 > logs/dryrun/cpu_sampler_B4.log 2>&1; echo "B unif_boot5 exit $?"
$PY tests/sampler_report.py --root $D 2>&1 | tail -40
