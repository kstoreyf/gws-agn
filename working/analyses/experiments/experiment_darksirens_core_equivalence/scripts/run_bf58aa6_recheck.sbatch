#!/bin/bash
#SBATCH --job-name=core_bf58aa6_recheck
#SBATCH --account=phy220048p
#SBATCH --partition=RITA-GPU
#SBATCH --qos=rita
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100-80:1
#SBATCH --mem=140G
#SBATCH --time=02:00:00
#SBATCH --output=/hildafs/projects/phy230014p/magana/gws-agn/working/analyses/experiments/experiment_darksirens_core_equivalence/logs/%x_%j.out
#SBATCH --error=/hildafs/projects/phy230014p/magana/gws-agn/working/analyses/experiments/experiment_darksirens_core_equivalence/logs/%x_%j.err
# Post-#54 recheck: A11 cells + grid and the A5 probe on core bf58aa6, explicit settings.
set -uo pipefail
cd /hildafs/projects/phy230014p/magana/gws-agn/working/analyses/experiments/experiment_darksirens_core_equivalence/scripts
export PYTHONPATH= XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 JAX_PLATFORMS=cuda OMP_NUM_THREADS=4 CORE_TAG=bf58aa6
PY=/hildafs/projects/phy230014p/magana/envs/darksirens-core-bf58aa6/bin/python
$PY core_a11_cells.py || echo "[recheck] a11 cells FAILED"
$PY core_a11_cells.py --grid || echo "[recheck] a11 grid FAILED"
PROBE_ONLY=1 $PY core_a5_smoke.py || echo "[recheck] a5 probe FAILED"
