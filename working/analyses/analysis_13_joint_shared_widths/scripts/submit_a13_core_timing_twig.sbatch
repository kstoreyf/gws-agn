#!/bin/bash
#SBATCH --job-name=a13timing
#SBATCH --account=phy220048p
#SBATCH --partition=TWIG-GPU
#SBATCH --qos=twig
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100-40:1
#SBATCH --mem=100G
#SBATCH --time=02:00:00
#SBATCH --output=/hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_13_joint_shared_widths/logs/%x_%j.out
#SBATCH --error=/hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_13_joint_shared_widths/logs/%x_%j.err
# Per-call timing of core #52 defaults, one at a time (darksirens-work request 2026-10-07).
#   sbatch --export=ALL,A13_CORE=e7c3007 scripts/submit_a13_core_timing_twig.sbatch
# kernel_layout (galaxy_list) is left out: its build needs ~48 GB, more than an A100-40.
set -euo pipefail
cd /hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_13_joint_shared_widths/scripts
export PYTHONPATH= XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 JAX_PLATFORMS=cuda OMP_NUM_THREADS=4
PY=/hildafs/projects/phy230014p/magana/envs/darksirens-core-${A13_CORE:?}/bin/python
echo "[a13timing] host=$(hostname) job=${SLURM_JOB_ID:-none} core=${A13_CORE}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
for s in ${A13_SWITCHES:-none pairing_norm missing_density kernel_window}; do $PY a13_core_timing.py --switch $s || echo "[a13timing] switch $s FAILED"; done
