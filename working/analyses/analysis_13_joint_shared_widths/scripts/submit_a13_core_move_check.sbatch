#!/bin/bash
#SBATCH --job-name=a13coremove
#SBATCH --account=phy220048p
#SBATCH --partition=RITA-GPU
#SBATCH --qos=rita
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100-80:1
#SBATCH --mem=120G
#SBATCH --time=03:00:00
#SBATCH --output=/hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_13_joint_shared_widths/logs/%x_%j.out
#SBATCH --error=/hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_13_joint_shared_widths/logs/%x_%j.err
# A13 core-move check (CORE_MOVE_PLAN.md): historical then defaults evaluation on a new core commit.
#   sbatch --export=ALL,A13_CORE=e7c3007 scripts/submit_a13_core_move_check.sbatch
set -euo pipefail
cd /hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_13_joint_shared_widths/scripts
export PYTHONPATH= XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 JAX_PLATFORMS=cuda OMP_NUM_THREADS=4
PY=/hildafs/projects/phy230014p/magana/envs/darksirens-core-${A13_CORE:?}/bin/python
echo "[a13coremove] host=$(hostname) job=${SLURM_JOB_ID:-none} core=${A13_CORE}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
# A13_MODES picks the modes (default both); XLA_PYTHON_CLIENT_MEM_FRACTION may be raised for
# the defaults build (it asked for a single 30.4 GB buffer and hit the 75% cap on 2026-10-07)
for m in ${A13_MODES:-historical defaults}; do $PY a13_core_move_check.py --settings $m; done
