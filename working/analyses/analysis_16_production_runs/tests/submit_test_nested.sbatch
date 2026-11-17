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
#SBATCH --time=01:00:00
#SBATCH --output=/hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_16_production_runs/logs/%x_nested_%j.out
#SBATCH --error=/hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_16_production_runs/logs/%x_nested_%j.err
# Test (b): a short nested-sampling run of each model class at production settings (8,000 samples,
# correction on, cap 20, nlive 200, first bound after 400 calls), each interrupted by a hard exit and
# resumed from its last checkpoint (written every 20 s). Two classes also run uninterrupted to the
# same call count; the resumed run must repeat them bit for bit (tests/compare_resume.py).
#   single_population, both catalogs, equal events, real. 100  (unif)   ref 700 calls + kill at 500 + resume
#   single_population, no catalog,   equal events, real. 100  (unif)   ref 700 calls + kill at 500 + resume
#   host_dependent_populations, both, host-dep. events, 100   (rslice) kill at 500 + resume to 700
#   free_densities, both, m < 18, equal events, real. 100     (unif)   kill at 500 + resume to 700
# Then tests/collect_nested_tests.py -> tests/results/nested_tests.json and tests/results/timing.json.
set -uo pipefail
A16=/hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_16_production_runs
cd $A16
export PYTHONPATH= XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 JAX_PLATFORMS=cuda OMP_NUM_THREADS=4
export A16_DEADLINE=$(( $(date +%s) + 60*60 - 120 ))
PY=/hildafs/projects/phy230014p/magana/envs/darksirens-core-e7c3007/bin/python
JPY=/hildafs/home/magana/tmp_ondemand_hildafs_phy230014p_symlink/magana/.conda/envs/jax/bin/python
T="--out-root tests/nested_runs --maxcall-total 700 --checkpoint-every 20 --chunk-calls 200"
echo "[agn-prodtest nested] host=$(hostname) job=${SLURM_JOB_ID:-none} start=$(date -u +%FT%TZ)"
run3() {  # reference + killed + resumed
  $PY scripts/prod_driver.py "$@" $T --tag ref;                           echo "[nested] $1.. ref exit $?"
  $PY scripts/prod_driver.py "$@" $T --tag resumed --kill-after-calls 500; echo "[nested] kill exit $?"
  $PY scripts/prod_driver.py "$@" $T --tag resumed;                       echo "[nested] resume exit $?"
}
run2() {  # killed + resumed
  $PY scripts/prod_driver.py "$@" $T --tag resumed --kill-after-calls 500; echo "[nested] kill exit $?"
  $PY scripts/prod_driver.py "$@" $T --tag resumed;                       echo "[nested] resume exit $?"
}
run3 --mode nested --analysis single_population --catalogs none --events-kind equal_populations --realization 100
run3 --mode nested --analysis single_population --catalogs both --events-kind equal_populations --realization 100
run2 --mode nested --analysis host_dependent_populations --catalogs both --events-kind host_dependent --realization 100
run2 --mode nested --analysis free_densities --catalogs both --flux 18 --events-kind equal_populations --realization 100
JAX_PLATFORMS=cpu $PY tests/collect_nested_tests.py
echo "[agn-prodtest nested] end=$(date -u +%FT%TZ)"
