#!/bin/bash
#SBATCH --job-name=a1_galconv
#SBATCH --partition=HENON-GPU
#SBATCH --account=phy220048p
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=140G
#SBATCH --time=10:00:00
#SBATCH --output=logs/slurm_a1_galconv_%j.out
#SBATCH --error=logs/slurm_a1_galconv_%j.err
# GAL sky-oracle convergence battery.  HENON carries A100-40GB cards, so the
# darksirens reference-likelihood call inside attr_ds_bridge.build must be blocked
# more tightly than on the 80GB production card; the blocking is identical across
# every run in the battery and cannot enter any comparison between them.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export DARKSIRENS_SRC=/hildafs/projects/phy230014p/magana/src/darksirens
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
PY=/hildafs/home/magana/tmp_ondemand_hildafs_phy230014p_symlink/magana/.conda/envs/jax/bin/python
export PATH="$(dirname "$PY"):$PATH"
export EXTRA="--sel_batch 20000 --pe_batch_events 8 --pz_batch 400000"
echo "=== $(hostname) $(date -u)  EXTRA=$EXTRA ==="
nvidia-smi --query-gpu=name,memory.total --format=csv || true
TRACERS=gal NS=${NS:-120} ./scripts/run_sky_oracle.sh convonly
echo "=== $(date -u) JOB DONE ==="
