#!/bin/bash
#SBATCH --job-name=dscore_k1
#SBATCH --account=phy220048p
#SBATCH --partition=RITA-GPU
#SBATCH --qos=rita
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100-80:1
#SBATCH --mem=100G
#SBATCH --time=06:00:00
#SBATCH --output=/hildafs/projects/phy230014p/magana/gws-agn/working/analyses/experiments/experiment_darksirens_core_equivalence/logs/%x_%j.out
#SBATCH --error=/hildafs/projects/phy230014p/magana/gws-agn/working/analyses/experiments/experiment_darksirens_core_equivalence/logs/%x_%j.err
#
# Every arm on the SAME GPU, one process each (legacy and core are both `darksirens`).
#   sbatch [--export=ALL,MAX_CELLS=2] scripts/run_arms.sbatch
set -uo pipefail
cd /hildafs/projects/phy230014p/magana/gws-agn/working/analyses/experiments/experiment_darksirens_core_equivalence/scripts
SRC=/hildafs/projects/phy230014p/magana/src
BASE_PY=/hildafs/home/magana/tmp_ondemand_hildafs_phy230014p_symlink/magana/.conda/envs/jax/bin/python
CORE_PY=/hildafs/projects/phy230014p/magana/envs/darksirens-core-f527b94/bin/python
export XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 JAX_PLATFORMS=cuda
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
MC=${MAX_CELLS:+--max_cells $MAX_CELLS}
echo "[dscore] host=$(hostname) job=${SLURM_JOB_ID:-none} max_cells=${MAX_CELLS:-all}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
for SHA in 2b86a2d 0c5b3db c042527; do
  PYTHONPATH=$SRC/darksirens-$SHA $BASE_PY legacy_k1_grid.py --arm legacy_$SHA --expect darksirens-$SHA $MC \
    || echo "[dscore] legacy_$SHA FAILED"
done
for MODE in legacy_arith default fast; do
  PYTHONPATH= $CORE_PY core_k1_grid.py --mode $MODE $MC || echo "[dscore] core_$MODE FAILED"
done
echo "[dscore] done"
