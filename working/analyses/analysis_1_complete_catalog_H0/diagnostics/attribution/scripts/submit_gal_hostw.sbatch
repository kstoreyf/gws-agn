#!/bin/bash
#SBATCH --job-name=a1_galhostw
#SBATCH --partition=HENON-GPU
#SBATCH --account=phy220048p
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=140G
#SBATCH --time=10:00:00
#SBATCH --output=logs/slurm_a1_galhostw_%j.out
#SBATCH --error=logs/slurm_a1_galhostw_%j.err
# TASK 3, GAL: the exact host-galaxy sky oracle with the opt-in --host_prior_arms
# flag (two extra arms; nothing else changes).  HENON carries A100-40GB cards, so
# the darksirens reference-likelihood call inside attr_ds_bridge.build is blocked
# more tightly than on the 80GB production card -- identical blocking across the
# arms, which are a PAIRED per-event statistic, so it cannot enter the comparison.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export DARKSIRENS_SRC=/hildafs/projects/phy230014p/magana/src/darksirens
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
PY=/hildafs/home/magana/tmp_ondemand_hildafs_phy230014p_symlink/magana/.conda/envs/jax/bin/python
export PATH="$(dirname "$PY")):$PATH"
echo "=== $(hostname) $(date -u) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv || true
"$PY" -u scripts/attr_sky_oracle.py --tracer gal --tag gal_hostw \
    --host_prior_arms --sel_batch 20000 --pe_batch_events 8 --pz_batch 400000 \
    > logs/attr_sky_oracle_gal_hostw.log 2>&1
echo "=== $(date -u) JOB DONE ==="
