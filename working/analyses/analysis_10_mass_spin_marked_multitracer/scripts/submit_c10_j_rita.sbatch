#!/bin/bash
#SBATCH --job-name=c10_j
#SBATCH --account=phy220048p
#SBATCH --partition=RITA-GPU
#SBATCH --qos=rita
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100-80:1
#SBATCH --mem=100G
#SBATCH --time=09:00:00
#SBATCH --output=/hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_10_mass_spin_marked_multitracer/logs/c10_j_%A_%a.out
#SBATCH --error=/hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_10_mass_spin_marked_multitracer/logs/c10_j_%A_%a.err
#
# C10-J -- the reduced asymmetric cosmology cube (design-note "G2", run as
# "H2"): H0 (--h0_window, default [63,76] on the 0.5 lattice + 67.74, 28
# nodes) x f_AGN 11 nodes {0.05,...,0.55} x dmu_chi 13 nodes {0.025,...,0.205}
# x dmu_G 15 nodes {1,2,2.5,...,8,9}.  28x11x15 = 4,620 rows x 13 dmu_chi
# cells = 60,060 cells, ~50.4 GPU-h at the measured 3.0143 s/eval.  One chunk
# of 8 is at most 578 rows x 13 cells = 7,514 cells = 6.29 h; --time=09:00:00
# is a 1.3x-plus margin over that plus the ~23 s build and ~11 s first
# (compiling) evaluation.  Two workers at a time finish the cube in ~25 h of
# wall clock.  H0 = 67.74 is dealt FIRST in every chunk's row order, so the
# free closure against results/a10_arm_J.h5 (ULP) is available from the first
# completed wave.
#
# DO NOT SUBMIT until the owner has reviewed the C10-S / C10-P results and
# confirmed the H0 window (default [63, 76]).
#
#   sbatch --array=0-7%2 --export=ALL,N_CHUNKS=8,C10_EXTRA="--h0_window 63 76" \
#     scripts/submit_c10_j_rita.sbatch
#
# --qos=rita contributes ZERO to the priority number, so PENDING behind other
# users is normal: WAIT.  Do not resubmit, shrink or relocate.  Each worker
# appends to its OWN checkpoint and resumes after an eviction, so a kill
# costs at most one 13-cell row.  A 'TaskProlog failed' death is a node
# fault: resubmit that index once, identical.
set -euo pipefail
source /hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_9_marked_multitracer_H0_fagn/scripts/env_a9.sh
A10_DIR=/hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_10_mass_spin_marked_multitracer
CHUNK=${SLURM_ARRAY_TASK_ID:-0}
NCH=${N_CHUNKS:-${SLURM_ARRAY_TASK_COUNT:-8}}
echo "[c10] host=$(hostname) job=${SLURM_JOB_ID:-none} chunk=${CHUNK}/${NCH} arm=C10-J"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
exec "${A9_PY}" "${A10_DIR}/scripts/c10_scan.py" --stage j \
     --chunk "${CHUNK}" --n_chunks "${NCH}" ${C10_EXTRA:-}
