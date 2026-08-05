#!/bin/bash
#SBATCH --job-name=a1_abc
#SBATCH --partition=HENON-GPU
#SBATCH --account=phy220048p
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --time=8:00:00
#SBATCH --output=logs/slurm_a1_abc_%j.out
#SBATCH --error=logs/slurm_a1_abc_%j.err
# ENDGAME (A-B)/(C-A) split.  HENON carries A100-40GB cards, so the batches are
# shrunk; batching cannot enter any reported statistic (all are exact sums).
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export DARKSIRENS_SRC=/hildafs/projects/phy230014p/magana/src/darksirens
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
PY=/hildafs/home/magana/tmp_ondemand_hildafs_phy230014p_symlink/magana/.conda/envs/jax/bin/python
export PATH="$(dirname "$PY"):$PATH"
echo "=== $(hostname) $(date -u)  SEEDS=${SEEDS:-} TRACERS=${TRACERS:-} ==="
nvidia-smi --query-gpu=name,memory.total --format=csv || true
EXTRA="--sel_batch 12000 --pe_batch_events 5 --truth_batch 12000" ./scripts/run_abc.sh
echo "=== $(date -u) JOB DONE ==="
