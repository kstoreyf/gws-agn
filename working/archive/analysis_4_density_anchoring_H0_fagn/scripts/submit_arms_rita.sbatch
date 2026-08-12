#!/bin/bash
#SBATCH --job-name=a4_arms
#SBATCH --account=phy220048p
#SBATCH --partition=RITA-GPU
#SBATCH --qos=rita
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm_a4_arms_%A_%a.out
#SBATCH --error=logs/slurm_a4_arms_%A_%a.err
#
# A GPU worker on the analysis-4 arm queue.  RITA-GPU has 2x A100-80, so at most
# two elements run at once; the rest wait and backfill as GPUs free up.  6 h
# covers the most expensive arm (m21, 2.28 GPU-h measured in analysis 3) with
# margin plus cheap grids.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export DARKSIRENS_SRC=/hildafs/projects/phy230014p/magana/src/darksirens
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
PY=/hildafs/home/magana/tmp_ondemand_hildafs_phy230014p_symlink/magana/.conda/envs/jax/bin
export PATH="$PY:$PATH"
./scripts/arm_worker.sh
