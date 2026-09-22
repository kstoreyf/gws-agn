#!/bin/bash
#SBATCH --job-name=c10_s
#SBATCH --account=phy220048p
#SBATCH --partition=RITA-GPU
#SBATCH --qos=rita
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100-80:1
#SBATCH --mem=100G
#SBATCH --time=03:00:00
#SBATCH --output=/hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_10_mass_spin_marked_multitracer/logs/c10_s_%A_%a.out
#SBATCH --error=/hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_10_mass_spin_marked_multitracer/logs/c10_s_%A_%a.err
#
# C10-S -- the spatial-only cosmology arm p(H0, f_AGN), BOTH marks held at 0.0
# exactly, on A9's S9 lattice: H0 202 nodes ([50, 100] step 0.25, plus 67.74,
# bitwise-asserted against results/s9_spatial.h5) x f 41 nodes.  8,282 cells,
# ~6.9 GPU-h at the measured 3.0143 s/eval.  One chunk of 4 is at most 51
# rows x 41 cells = 2,091 cells = 1.75 h; --time=03:00:00 carries that with a
# ~1.7x margin plus the ~23 s build and ~11 s first (compiling) evaluation.
# Two workers at a time finish the cube in ~3.5 h of wall clock.
#
#   sbatch --array=0-3%2 --export=ALL,N_CHUNKS=4 \
#     scripts/submit_c10_s_rita.sbatch
#
# --qos=rita contributes ZERO to the priority number, so PENDING behind other
# users is normal: WAIT.  Do not resubmit, shrink or relocate.  Each worker
# appends to its OWN checkpoint and resumes after an eviction, so a kill
# costs at most one 41-cell row.  A 'TaskProlog failed' death is a node
# fault: resubmit that index once, identical.
set -euo pipefail
source /hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_9_marked_multitracer_H0_fagn/scripts/env_a9.sh
A10_DIR=/hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_10_mass_spin_marked_multitracer
CHUNK=${SLURM_ARRAY_TASK_ID:-0}
NCH=${N_CHUNKS:-${SLURM_ARRAY_TASK_COUNT:-4}}
echo "[c10] host=$(hostname) job=${SLURM_JOB_ID:-none} chunk=${CHUNK}/${NCH} arm=C10-S"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
exec "${A9_PY}" "${A10_DIR}/scripts/c10_scan.py" --stage s \
     --chunk "${CHUNK}" --n_chunks "${NCH}" ${C10_EXTRA:-}
