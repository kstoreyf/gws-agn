#!/bin/bash
#SBATCH --job-name=a2_null
#SBATCH --account=phy220048p
#SBATCH --partition=HENON-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm_a2_null_%j.out
#SBATCH --error=logs/slurm_a2_null_%j.err
# The sky-shuffle null for the f scan, seed 100 (101 evaluations, ~7 min).
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export DARKSIRENS_SRC=/hildafs/projects/phy230014p/magana/src/darksirens
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
PY=/hildafs/home/magana/tmp_ondemand_hildafs_phy230014p_symlink/magana/.conda/envs/jax/bin
export PATH="$PY:$PATH"
KIND=fnull SEED=100 LANE=targeted ./scripts/run_scan.sh
