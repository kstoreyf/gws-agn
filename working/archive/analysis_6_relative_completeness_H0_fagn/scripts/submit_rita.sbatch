#!/bin/bash
#SBATCH --job-name=a6_surface
#SBATCH --account=phy220048p
#SBATCH --partition=RITA-GPU
#SBATCH --qos=rita
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=110G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm_a6_%A_%a.out
#SBATCH --error=logs/slurm_a6_%A_%a.err
#
# One GPU worker on the analysis-6 surface queue.  Sized against the real node:
# rita is 2x A100-80, 256 CPUs, 250 G, partition limit 7 days.  Two array
# elements therefore both start immediately (2x110 G = 220 G of 250 G,
# 2x32 = 64 of 256 CPUs) and each gets 24 h — the worst task in the queue prices
# at ~5 h under scripts/_task_cost.py, so a worker never exits leaving work
# unclaimed, and either worker alone can drain the whole 8-cell queue.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export DARKSIRENS_SRC=/hildafs/projects/phy230014p/magana/src/darksirens
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
PY=/hildafs/home/magana/tmp_ondemand_hildafs_phy230014p_symlink/magana/.conda/envs/jax/bin
export PATH="$PY:$PATH"
./scripts/worker.sh
