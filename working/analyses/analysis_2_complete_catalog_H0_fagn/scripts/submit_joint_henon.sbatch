#!/bin/bash
#SBATCH --job-name=a2_joint
#SBATCH --account=phy220048p
#SBATCH --partition=HENON-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/slurm_a2_joint_%A_%a.out
#SBATCH --error=logs/slurm_a2_joint_%A_%a.err
#
# A GPU worker on the shared joint-grid queue.  Submit one per free GPU:
#   sbatch --array=0-1 scripts/submit_joint_henon.sbatch
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export DARKSIRENS_SRC=/hildafs/projects/phy230014p/magana/src/darksirens
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
PY=/hildafs/home/magana/tmp_ondemand_hildafs_phy230014p_symlink/magana/.conda/envs/jax/bin
export PATH="$PY:$PATH"
./scripts/joint_worker.sh
