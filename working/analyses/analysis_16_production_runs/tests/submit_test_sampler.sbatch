#!/bin/bash
#SBATCH --job-name=agn-prodtest
#SBATCH --account=phy220048p
#SBATCH --partition=RITA-GPU
#SBATCH --qos=rita
#SBATCH --nice=50
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100-80:1
#SBATCH --mem=120G
#SBATCH --time=02:15:00
#SBATCH --output=/hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_16_production_runs/logs/%x_sampler_%j.out
#SBATCH --error=/hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_16_production_runs/logs/%x_sampler_%j.err
# Test (c): dynesty settings for the eight-parameter model (host_dependent_populations, both complete
# catalogs, realization 100's host-dependent events, 2,000 samples per event, production priors,
# correction on, cap 20, nlive 200), bounded at 2.25 GPU hours by --time (with tests a and b: 4 h).
#   Phase A  multi-ellipsoid + uniform sampling, bootstrap off, enlarge 1.25, from scratch, to 12,000
#            calls or 3,000 iterations; state copies every 500 iterations. Measures calls per iteration
#            along the run and, by the insertion-index test, whether the bounds cover the contours.
#   Phase B  from Phase A's last state copy, 100 more iterations with each of
#              rslice (Analysis 13's guarded setting: slices 11, enlarge 1.25, no bootstrap)
#              unif, enlarge 1.25, bootstrap off (Phase A's setting, bounds rebuilt)
#              rwalk (walks 28 = dynesty's 20 + ndim)
#              unif with dynesty's default bootstrap 5 (Analyses 11/12's six-parameter setting)
#            each capped in calls; same starting live points and random state.
# Then tests/sampler_report.py -> tests/results/sampler_test.json (calls per iteration, insertion-index
# KS p-values, ln L* reached and ln Z increments over the same iterations, projected full-run calls).
set -uo pipefail
A16=/hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_16_production_runs
cd $A16
export PYTHONPATH= XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 JAX_PLATFORMS=cuda OMP_NUM_THREADS=4
T0=$(date +%s)
export A16_DEADLINE=$(( T0 + 135*60 - 240 ))
PY=/hildafs/projects/phy230014p/magana/envs/darksirens-core-e7c3007/bin/python
M="--mode nested --analysis host_dependent_populations --catalogs both --events-kind host_dependent --realization 100
   --nsamp 2000 --out-root tests/sampler_runs --checkpoint-every 120 --chunk-calls 500"
D=tests/sampler_runs/host_dependent_populations/both_catalogs/complete/host_dependent/realization_100
echo "[agn-prodtest sampler] host=$(hostname) job=${SLURM_JOB_ID:-none} start=$(date -u +%FT%TZ)"
# Phase A: stop by 52 min after the start at the latest (the deadline also stops it)
A16_DEADLINE=$(( T0 + 52*60 )) $PY scripts/prod_driver.py $M --tag phaseA_unif_b0 --sample unif --enlarge 1.25 --bootstrap 0 \
    --maxcall-total 12000 --maxiter-total 3000 --save-at-iterations 500,1000,1500,2000,2500,3000
echo "[sampler] phase A exit $?"
S=$(ls -1 $D/nested_1_phaseA_unif_b0/state_it*.save 2>/dev/null | sed 's/.*state_it\([0-9]*\).save/\1 &/' | sort -n | tail -1 | cut -d' ' -f2)
if [ -z "$S" ]; then
  S=$D/nested_1_phaseA_unif_b0/state.save
fi
echo "[sampler] phase B starts from $S"
B="$M --init-from $S --maxiter-more 100"
$PY scripts/prod_driver.py $B --tag phaseB_rslice --switch-sample rslice --enlarge 1.25 --bootstrap 0 --slices 11 --maxcall-more 6000
echo "[sampler] rslice exit $?"
$PY scripts/prod_driver.py $B --tag phaseB_rwalk --switch-sample rwalk --walks 28 --maxcall-more 3500
echo "[sampler] rwalk exit $?"
$PY scripts/prod_driver.py $B --tag phaseB_unif_b0 --switch-sample unif --enlarge 1.25 --bootstrap 0 --maxcall-more 3000
echo "[sampler] unif_b0 exit $?"
$PY scripts/prod_driver.py $B --tag phaseB_unif_boot5 --switch-sample unif --bootstrap 5 --maxcall-more 2000
echo "[sampler] unif_boot5 exit $?"
JAX_PLATFORMS=cpu $PY tests/sampler_report.py
echo "[agn-prodtest sampler] end=$(date -u +%FT%TZ)"
