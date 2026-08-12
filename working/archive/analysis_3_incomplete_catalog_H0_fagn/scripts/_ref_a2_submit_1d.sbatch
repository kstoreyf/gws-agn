#!/bin/bash
#SBATCH --job-name=a2_1d
#SBATCH --account=phy220048p
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm_a2_1d_%A_%a.out
#SBATCH --error=logs/slurm_a2_1d_%A_%a.err
#
# The 1-D scans: fscan at H0 = 67.74 (f in [0,1] x 101) and h0scan at f = 0.30
# (H0 in [50,100] x 201), five realisations on the targeted injection lane, the
# seed-100 popuni cross-check, and the seed-100 sky-shuffle null.
#   sbatch --array=0-12 --partition=HENON-GPU scripts/submit_1d.sbatch
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export DARKSIRENS_SRC=/hildafs/projects/phy230014p/magana/src/darksirens
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
PY=/hildafs/home/magana/tmp_ondemand_hildafs_phy230014p_symlink/magana/.conda/envs/jax/bin
export PATH="$PY:$PATH"

LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" scripts/tasks_1d.txt)
read -r KIND SEED LANE <<< "$LINE"
echo "[$(date -u)] task $SLURM_ARRAY_TASK_ID: KIND=$KIND SEED=$SEED LANE=$LANE on $(hostname)"
KIND=$KIND SEED=$SEED LANE=$LANE ./scripts/run_scan.sh
