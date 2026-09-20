#!/bin/bash
#SBATCH --job-name=a9_s9
#SBATCH --account=phy220048p
#SBATCH --partition=RITA-GPU
#SBATCH --qos=rita
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100-80:1
#SBATCH --mem=100G
#SBATCH --time=08:00:00
#SBATCH --output=/hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_9_marked_multitracer_H0_fagn/logs/a9_s9_%A_%a.out
#SBATCH --error=/hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_9_marked_multitracer_H0_fagn/logs/a9_s9_%A_%a.err
#
# ARM S9 -- spatial-only cosmology, dmu_chi = 0, free (H0, f_AGN).
# 202 H0 nodes (Analysis 2's own [50, 100] x 201 axis plus the anchor 67.74)
# x 41 f nodes = 8,282 cells = 6.94 GPU-h at the measured 3.0188 s/eval.
#
#   sbatch --array=0-7%2 --export=ALL,N_CHUNKS=8 \
#     /hildafs/.../analysis_9_marked_multitracer_H0_fagn/scripts/submit_a9_s9_rita.sbatch
#
# One GPU worker per array task, at most two at a time (the node has two
# A100-80s and one worker peaks at 34 GB of 82 GB).  The partition/qos/account
# triple is load-bearing and --qos=rita contributes ZERO to the priority number,
# so a PENDING spell behind other users is normal and expected: WAIT.  Do not
# resubmit, do not shrink the request, do not drop the qos, do not relocate.
# Each worker appends to its OWN checkpoint and resumes after an eviction, so a
# kill costs at most one 41-cell row.
set -euo pipefail
source /hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_9_marked_multitracer_H0_fagn/scripts/env_a9.sh
CHUNK=${SLURM_ARRAY_TASK_ID:-0}
NCH=${N_CHUNKS:-${SLURM_ARRAY_TASK_COUNT:-1}}
echo "[a9] host=$(hostname) job=${SLURM_JOB_ID:-none} chunk=${CHUNK}/${NCH} arm=S9"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
exec "${A9_PY}" "${A9_DIR}/scripts/a9_scan.py" --stage s9 \
     --chunk "${CHUNK}" --n_chunks "${NCH}" ${A9_EXTRA:-}
