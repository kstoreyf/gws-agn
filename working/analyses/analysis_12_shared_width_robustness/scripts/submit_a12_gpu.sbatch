#!/bin/bash
#SBATCH --job-name=a12_gpu
#SBATCH --account=phy220048p
#SBATCH --partition=RITA-GPU
#SBATCH --qos=rita
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100-80:1
#SBATCH --mem=100G
#SBATCH --time=02:00:00
#SBATCH --output=/hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_12_shared_width_robustness/logs/%x_%j.out
#SBATCH --error=/hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_12_shared_width_robustness/logs/%x_%j.err
#
# Generic Analysis-12 GPU job on rita (RITA-GPU, QOS rita):
#   sbatch --job-name=NAME --export=ALL,A12_SCRIPT=scripts/X.py,A12_ARGS="..." scripts/submit_a12_gpu.sbatch
# A 'TaskProlog failed' death within seconds is a rita node fault: resubmit once, identical.
set -euo pipefail
source /hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_9_spin_marked_H0_fagn/scripts/env_a9.sh
export PYTHONUNBUFFERED=1
cd /hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_12_shared_width_robustness
echo "[a12_gpu] host=$(hostname) job=${SLURM_JOB_ID:-none} script=${A12_SCRIPT:?} args=${A12_ARGS:-}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
exec "${A9_PY}" "${A12_SCRIPT}" ${A12_ARGS:-}
