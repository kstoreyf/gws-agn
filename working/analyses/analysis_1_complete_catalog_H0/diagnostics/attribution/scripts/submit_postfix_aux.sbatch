#!/bin/bash
#SBATCH --job-name=a1_postfix_aux
#SBATCH --partition=HENON-GPU
#SBATCH --account=phy220048p
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=140G
#SBATCH --time=10:00:00
#SBATCH --output=logs/slurm_a1_postfix_aux_%j.out
#SBATCH --error=logs/slurm_a1_postfix_aux_%j.err

# Post-fix auxiliaries that do not fit on the one local A100 alongside the sky
# oracle: the four guard/variance diagnostics at truth, and the sky oracle's
# convergence battery.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export DARKSIRENS_SRC=/hildafs/projects/phy230014p/magana/src/darksirens
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
PY=/hildafs/home/magana/tmp_ondemand_hildafs_phy230014p_symlink/magana/.conda/envs/jax/bin/python
export PATH="$(dirname "$PY"):$PATH"
export SEL_BATCH=${SEL_BATCH:-50000} PE_BLOCK=${PE_BLOCK:-25} KDE_W=${KDE_W:-4096}
export LOG10N0=${LOG10N0:--24}

echo "=== host: $(hostname)  job ${SLURM_JOB_ID:-none}  $(date -u) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv || true

echo "=== $(date -u) guard diagnostics ==="
if [ "${SKIP_GUARD:-0}" != "1" ]; then ./scripts/run_guard_diag.sh; fi
echo "=== $(date -u) sky-oracle convergence battery ==="
TRACERS=${TRACERS:-agn} NS=${NS:-120} ./scripts/run_sky_oracle.sh convonly
echo "=== $(date -u) JOB DONE ==="
