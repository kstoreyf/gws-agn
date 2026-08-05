#!/bin/bash
#SBATCH --job-name=a0_pure
#SBATCH --partition=HENON-GPU
#SBATCH --qos=henon-gpu
#SBATCH --account=phy220048p
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=160G
#SBATCH --time=08:00:00
#SBATCH --array=0-4
#SBATCH --output=logs/slurm_a0_%A_%a.out
#SBATCH --error=logs/slurm_a0_%A_%a.err
#
#   sbatch scripts/submit_scans.sbatch
#
# One array task per catalog realisation; four K=1 H0 scans each (pure-GAL and
# pure-AGN 1000-event draws x targeted/popuni injection lanes).  Environment and
# resources copied from analysis_1/scripts/submit_v3_controls.sbatch.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export DARKSIRENS_SRC=/hildafs/projects/phy230014p/magana/src/darksirens
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
PY=/hildafs/home/magana/tmp_ondemand_hildafs_phy230014p_symlink/magana/.conda/envs/jax/bin
export PATH="$PY:$PATH"
export SEL_BATCH=${SEL_BATCH:-50000}
export PE_BLOCK=${PE_BLOCK:-25}
export KDE_W=${KDE_W:-4096}
export LOG10N0=${LOG10N0:--24}

SEEDS=(100 101 102 103 105)
S=${SEEDS[${SLURM_ARRAY_TASK_ID:-0}]}

echo "=== host: $(hostname)  job: ${SLURM_JOB_ID}  task: ${SLURM_ARRAY_TASK_ID}  seed: $S  $(date -u) ==="
echo "python: $(which python)"
echo "darksirens: $DARKSIRENS_SRC @ $(git -C "$DARKSIRENS_SRC" rev-parse HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv || true

./scripts/run_scans.sh "$S"
echo "=== $(date -u) TASK DONE (seed $S) ==="
