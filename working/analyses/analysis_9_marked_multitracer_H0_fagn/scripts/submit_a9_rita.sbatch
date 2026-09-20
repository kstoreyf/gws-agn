#!/bin/bash
#SBATCH --job-name=a9_scan
#SBATCH --account=phy220048p
#SBATCH --partition=RITA-GPU
#SBATCH --qos=rita
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100-80:1
#SBATCH --mem=100G
#SBATCH --time=1-00:00:00
#SBATCH --output=/hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_9_marked_multitracer_H0_fagn/logs/a9_scan_%A_%a.out
#SBATCH --error=/hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_9_marked_multitracer_H0_fagn/logs/a9_scan_%A_%a.err
#
# One GPU worker on one interleaved slice of the H0 axis.  The node has two
# A100-80s, so run at most two at a time:
#
#   sbatch --array=0-7%2 --export=ALL,N_CHUNKS=8 \
#     /hildafs/.../analysis_9_marked_multitracer_H0_fagn/scripts/submit_a9_rita.sbatch
#
# The partition/qos/account triple is load-bearing: this account holds PRIORITY
# on RITA-GPU, so a PENDING spell is normal and the scheduler will place these
# ahead of the qos=rita-s jobs already there.  Do NOT drop --qos=rita, do NOT
# shrink the request to squeeze past other users, do NOT resubmit -- poll with
#   squeue -u $USER
# Each worker appends to its OWN checkpoint and resumes after an eviction, so a
# kill costs at most one 61-cell row.
#
# If the timing stage projects the registered cube above 80 GPU-h, the
# pre-registered trim is passed straight through:
#   sbatch --array=0-7%2 --export=ALL,N_CHUNKS=8,A9_EXTRA="--h0_axis fallback" ...
set -euo pipefail
source /hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_9_marked_multitracer_H0_fagn/scripts/env_a9.sh
CHUNK=${SLURM_ARRAY_TASK_ID:-0}
NCH=${N_CHUNKS:-${SLURM_ARRAY_TASK_COUNT:-1}}
echo "[a9] host=$(hostname) job=${SLURM_JOB_ID:-none} chunk=${CHUNK}/${NCH}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
exec "${A9_PY}" "${A9_DIR}/scripts/a9_scan.py" --stage scan \
     --chunk "${CHUNK}" --n_chunks "${NCH}" ${A9_EXTRA:-}
