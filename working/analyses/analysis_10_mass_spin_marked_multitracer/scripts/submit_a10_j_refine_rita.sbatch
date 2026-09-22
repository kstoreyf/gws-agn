#!/bin/bash
#SBATCH --job-name=a10_jr
#SBATCH --account=phy220048p
#SBATCH --partition=RITA-GPU
#SBATCH --qos=rita
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100-80:1
#SBATCH --mem=100G
#SBATCH --time=05:00:00
#SBATCH --output=/hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_10_mass_spin_marked_multitracer/logs/a10_jr_%A_%a.out
#SBATCH --error=/hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_10_mass_spin_marked_multitracer/logs/a10_jr_%A_%a.err
#
# ARM A10-J, THE ONE ALLOWED GRID REFINEMENT (owner decision, 2026-09-21).
#
# A10-M on the registered 1 Msun dmu_G axis gave sd(dmu_G) = 0.634 Msun and a
# 68% width of 1.87 Msun -- fewer than two nodes across the interval -- so six
# HALF-INTEGER nodes {2.5, 3.5, 4.5, 5.5, 6.5, 7.5} are added inside [2, 8],
# where the measured marginal sits above ~e^-9 of its peak.  Outside that the
# integer nodes carry the tails.
#
# These are ADDITIVE ROWS on the same physical-coordinate key, NOT a new grid:
# 246 (f, dmu_G) rows x 61 dmu_chi = 15,006 cells = 12.57 GPU-h at the measured
# 3.0143 s/eval.  Four chunks of at most 62 rows = 3,782 cells = 3.17 h each;
# --time=05:00:00 is that with a 58% margin (the registered 30% is 4.12 h).
# The main 861-row array is untouched, its rows are reused, and dmu_G = 0 is not
# a refinement node, so arm A10-chi is unaffected.  There is no second
# refinement.
#
#   sbatch --array=0-3%2 --dependency=afterany:<main array id> \
#     --export=ALL,N_CHUNKS=4 \
#     /hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_10_mass_spin_marked_multitracer/scripts/submit_a10_j_refine_rita.sbatch
#
# The dependency makes this run after the main array's last wave; rita has two
# A100-80s, so at most two of these workers are ever resident either way.
# --qos=rita contributes ZERO to the priority number, so PENDING is normal:
# WAIT.  Do not resubmit, shrink or relocate.  Each worker appends to its OWN
# checkpoint (_a10_scan_jr_c<k>of<n>.jsonl) and resumes after an eviction, so a
# kill costs at most one 61-cell row.  A 'TaskProlog failed' death is a node
# fault: resubmit that index once, identical.
set -euo pipefail
source /hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_9_marked_multitracer_H0_fagn/scripts/env_a9.sh
A10_DIR=/hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_10_mass_spin_marked_multitracer
CHUNK=${SLURM_ARRAY_TASK_ID:-0}
NCH=${N_CHUNKS:-${SLURM_ARRAY_TASK_COUNT:-4}}
echo "[a10] host=$(hostname) job=${SLURM_JOB_ID:-none} chunk=${CHUNK}/${NCH} arm=A10-J refinement"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
exec "${A9_PY}" "${A10_DIR}/scripts/a10_scan.py" --stage scan --dmuG_nodes refine \
     --chunk "${CHUNK}" --n_chunks "${NCH}" ${A10_EXTRA:-}
