#!/bin/bash
#SBATCH --job-name=a1_agninj2
#SBATCH --partition=HENON-GPU
#SBATCH --account=phy220048p
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1 --mem=140G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm_a1_agninj2_%j.out
#SBATCH --error=logs/slurm_a1_agninj2_%j.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export DARKSIRENS_SRC=/hildafs/projects/phy230014p/magana/src/darksirens
PY=/hildafs/home/magana/tmp_ondemand_hildafs_phy230014p_symlink/magana/.conda/envs/jax/bin/python
for LANE in targeted popuni; do
  echo "[$(date -u +%H:%M:%S)] injections agn/$LANE (with the MC-error bootstrap)"
  "$PY" -u scripts/attr_selmu_inj.py --tracer agn --injections $LANE \
      --sel_batch 50000 > logs/attr_selmu_inj_agn_${LANE}.log 2>&1
done
echo "=== $(date -u) JOB DONE ==="
