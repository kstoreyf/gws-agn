#!/bin/bash
#SBATCH --job-name=a10_j
#SBATCH --account=phy220048p
#SBATCH --partition=RITA-GPU
#SBATCH --qos=rita
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100-80:1
#SBATCH --mem=100G
#SBATCH --time=08:00:00
#SBATCH --output=/hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_10_mass_spin_marked_multitracer/logs/a10_j_%A_%a.out
#SBATCH --error=/hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_10_mass_spin_marked_multitracer/logs/a10_j_%A_%a.err
#
# ARM A10-J -- the joint fixed-H0 cube, free (f_AGN, dmu_chi, dmu_G).
# 41 f x 61 dmu_chi x 21 dmu_G = 52,521 cells.  At the MEASURED rita A100-80
# rate of 3.0143 s/eval (median of 20, sd 1.3 ms; a10_closure.json) that is
# 43.98 GPU-h.  One chunk of 8 is at most 108 rows of 61 cells = 6,588 cells =
# 5.52 h; the registered 30% margin is 7.17 h and --time=08:00:00 carries it
# with room for the 23 s build and the 11 s first (compiling) evaluation.
# Two workers at a time finish the cube in 21.99 h of wall clock.
#
#   sbatch --array=0-7%2 --export=ALL,N_CHUNKS=8 \
#     /hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_10_mass_spin_marked_multitracer/scripts/submit_a10_j_rita.sbatch
#
# The dmu_G = 0 rows are dealt FIRST, so arm A10-chi (the spin-only arm on the
# same axes, which closure 15.2 validates) closes in the first wave of workers
# while the rest of the cube is still running.
#
# --qos=rita contributes ZERO to the priority number, so PENDING behind other
# users is normal: WAIT.  Do not resubmit, do not shrink the request, do not
# drop the qos, do not relocate to another node.  Each worker appends to its OWN
# checkpoint and resumes after an eviction, so a kill costs at most one 61-cell
# row.  A 'TaskProlog failed' death is a node fault: resubmit once, identical.
set -euo pipefail
source /hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_9_marked_multitracer_H0_fagn/scripts/env_a9.sh
A10_DIR=/hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_10_mass_spin_marked_multitracer
CHUNK=${SLURM_ARRAY_TASK_ID:-0}
NCH=${N_CHUNKS:-${SLURM_ARRAY_TASK_COUNT:-8}}
echo "[a10] host=$(hostname) job=${SLURM_JOB_ID:-none} chunk=${CHUNK}/${NCH} arm=A10-J"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
exec "${A9_PY}" "${A10_DIR}/scripts/a10_scan.py" --stage scan \
     --chunk "${CHUNK}" --n_chunks "${NCH}" ${A10_EXTRA:-}
