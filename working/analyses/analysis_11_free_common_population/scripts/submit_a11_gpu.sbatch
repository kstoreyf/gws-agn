#!/bin/bash
#SBATCH --job-name=a11_gpu
#SBATCH --account=phy220048p
#SBATCH --partition=RITA-GPU
#SBATCH --qos=rita
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100-80:1
#SBATCH --mem=100G
#SBATCH --time=02:00:00
#SBATCH --output=/hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_11_free_common_population/logs/%x_%j.out
#SBATCH --error=/hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_11_free_common_population/logs/%x_%j.err
#
# Generic Analysis-11 GPU job on rita (RITA-GPU, QOS rita):
#   sbatch --job-name=NAME --export=ALL,A11_SCRIPT=scripts/X.py,A11_ARGS="..." scripts/submit_a11_gpu.sbatch
# A 'TaskProlog failed' death within seconds is a rita node fault: resubmit once, identical.
set -euo pipefail
source /hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_9_marked_multitracer_H0_fagn/scripts/env_a9.sh
cd /hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_11_free_common_population
echo "[a11_gpu] host=$(hostname) job=${SLURM_JOB_ID:-none} script=${A11_SCRIPT:?} args=${A11_ARGS:-}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
exec "${A9_PY}" "${A11_SCRIPT}" ${A11_ARGS:-}
