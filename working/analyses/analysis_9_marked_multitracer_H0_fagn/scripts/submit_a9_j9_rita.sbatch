#!/bin/bash
#SBATCH --job-name=a9_j9
#SBATCH --account=phy220048p
#SBATCH --partition=RITA-GPU
#SBATCH --qos=rita
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100-80:1
#SBATCH --mem=100G
#SBATCH --time=12:00:00
#SBATCH --output=/hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_9_marked_multitracer_H0_fagn/logs/a9_j9_%A_%a.out
#SBATCH --error=/hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_9_marked_multitracer_H0_fagn/logs/a9_j9_%A_%a.err
#
# ARM J9 -- marked joint cosmology, free (H0, f_AGN, dmu_chi).
# 28 H0 nodes (the S9-confirmed window on 0.5 spacing, plus the anchor 67.74)
# x 41 f x 61 mu_chi_c2 = 70,028 cells = 58.72 GPU-h at 3.0188 s/eval.
# Run ONCE; there is no grid-refinement sequence.
#
#   sbatch --array=0-7%2 --export=ALL,N_CHUNKS=8 \
#     /hildafs/.../analysis_9_marked_multitracer_H0_fagn/scripts/submit_a9_j9_rita.sbatch
#
# Chunk 0 is dealt the H0 = 67.74 ANCHOR slab and takes it first, so the whole
# 2501-cell Gate-A comparison against Analysis 8's arm J closes in the first
# wave of workers, while the rest of the cube is still running.
#
# Two workers at a time (the node has two A100-80s; one worker peaks at 34 GB of
# 82 GB).  --qos=rita contributes ZERO to the priority number, so PENDING behind
# other users is normal: WAIT.  Do not resubmit, do not shrink the request, do
# not drop the qos, do not relocate to another node.  Each worker appends to its
# OWN checkpoint and resumes after an eviction, so a kill costs at most one
# 61-cell row.
set -euo pipefail
source /hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_9_marked_multitracer_H0_fagn/scripts/env_a9.sh
CHUNK=${SLURM_ARRAY_TASK_ID:-0}
NCH=${N_CHUNKS:-${SLURM_ARRAY_TASK_COUNT:-1}}
echo "[a9] host=$(hostname) job=${SLURM_JOB_ID:-none} chunk=${CHUNK}/${NCH} arm=J9"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
exec "${A9_PY}" "${A9_DIR}/scripts/a9_scan.py" --stage scan --h0_axis j9 \
     --chunk "${CHUNK}" --n_chunks "${NCH}" ${A9_EXTRA:-}
