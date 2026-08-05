#!/bin/bash
#SBATCH --job-name=a1_agnaux
#SBATCH --partition=HENON-GPU
#SBATCH --account=phy220048p
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=140G
#SBATCH --time=10:00:00
#SBATCH --output=logs/slurm_a1_agnaux_%j.out
#SBATCH --error=logs/slurm_a1_agnaux_%j.err
# The AGN half of the final sweep: darksirens' injection-based log mu(H0) curves
# on both injection sets, and the TASK-3 host-prior arms of the sky oracle.  The
# AGN catalog is 1.5e6 rows, so this fits the 40GB HENON card; the GAL half needs
# the 80GB card and runs locally.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export DARKSIRENS_SRC=/hildafs/projects/phy230014p/magana/src/darksirens
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
PY=/hildafs/home/magana/tmp_ondemand_hildafs_phy230014p_symlink/magana/.conda/envs/jax/bin/python
echo "=== $(hostname) $(date -u) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv || true
for LANE in targeted popuni; do
  echo "[$(date -u +%H:%M:%S)] injections agn/$LANE"
  "$PY" -u scripts/attr_selmu_inj.py --tracer agn --injections $LANE \
      --sel_batch 50000 > logs/attr_selmu_inj_agn_${LANE}.log 2>&1
done
echo "[$(date -u +%H:%M:%S)] sky oracle agn host-prior arms"
"$PY" -u scripts/attr_sky_oracle.py --tracer agn --tag agn_hostw \
    --host_prior_arms --sel_batch 50000 --pe_batch_events 25 \
    > logs/attr_sky_oracle_agn_hostw.log 2>&1
echo "=== $(date -u) JOB DONE ==="
