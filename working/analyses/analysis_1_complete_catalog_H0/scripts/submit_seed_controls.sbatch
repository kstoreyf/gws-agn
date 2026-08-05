#!/bin/bash
#SBATCH --job-name=a1_seedctrl
#SBATCH --partition=HENON-GPU
#SBATCH --account=phy220048p
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm_a1_seedctrl_%j.out
#SBATCH --error=logs/slurm_a1_seedctrl_%j.err

# analysis_1_complete_catalog_H0 -- the matched-host controls on the further
# realisations, under the estimator of record (`dark_sirens` at log10n0 = -24).
# Usage:  sbatch scripts/submit_seed_controls.sbatch 101 102 103 105
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export DARKSIRENS_SRC=/hildafs/projects/phy230014p/magana/src/darksirens
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4

PY=/hildafs/home/magana/tmp_ondemand_hildafs_phy230014p_symlink/magana/.conda/envs/jax/bin/python
export PATH="$(dirname "$PY"):$PATH"

export SEL_BATCH=${SEL_BATCH:-50000}
export PE_BLOCK=${PE_BLOCK:-25}
export KDE_W=${KDE_W:-4096}
export LOG10N0=${LOG10N0:--24}

echo "=== host: $(hostname)  job: ${SLURM_JOB_ID:-none}  $(date -u) ==="
echo "python: $(which python)"
echo "darksirens: $DARKSIRENS_SRC @ $(git -C "$DARKSIRENS_SRC" rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv || true

./scripts/run_seed_controls.sh "$@"
echo "=== $(date -u) JOB DONE ==="
