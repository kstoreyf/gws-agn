#!/bin/bash
#SBATCH --job-name=a3_ladder
#SBATCH --account=phy220048p
#SBATCH --partition=HENON-GPU
#SBATCH --qos=henon-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm_a3_ladder_%A_%a.out
#SBATCH --error=logs/slurm_a3_ladder_%A_%a.err
#
# A GPU worker on the shared ladder queue.  Short walltime so it backfills; the
# worker prices each candidate task from the measured per-rung s/eval and claims
# only what its remaining allocation covers, so small workers drain the queue
# safely and never die mid-grid.  Submit one array element per GPU you expect to
# get, on every partition that has any.  6 h covers the most expensive rung
# (m21, a measured 2.3 GPU-h per grid) with margin plus a second cheap grid.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export DARKSIRENS_SRC=/hildafs/projects/phy230014p/magana/src/darksirens
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
PY=/hildafs/home/magana/tmp_ondemand_hildafs_phy230014p_symlink/magana/.conda/envs/jax/bin
export PATH="$PY:$PATH"
./scripts/ladder_worker.sh
