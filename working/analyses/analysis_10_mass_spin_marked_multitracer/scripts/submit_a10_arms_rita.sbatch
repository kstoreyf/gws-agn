#!/bin/bash
#SBATCH --job-name=a10_arms
#SBATCH --account=phy220048p
#SBATCH --partition=RITA-GPU
#SBATCH --qos=rita
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100-80:1
#SBATCH --mem=100G
#SBATCH --time=04:00:00
#SBATCH --output=/hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_10_mass_spin_marked_multitracer/logs/a10_arms_%j.out
#SBATCH --error=/hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_10_mass_spin_marked_multitracer/logs/a10_arms_%j.err
#
# Analysis 10 -- the two CHEAP fixed-H0 arms, run ONLY after Gate 15 passes.
#
#   A10-S   41 cells   (both marks held at zero)          ~2 min
#   A10-M   861 cells  (41 f x 21 dmu_G, dmu_chi = 0)     ~45 min
#
#   sbatch --export=ALL,STAGE="S M" scripts/submit_a10_arms_rita.sbatch
#
# THE ONE ALLOWED GRID REFINEMENT (owner decision, 2026-09-21): six 0.5 Msun
# dmu_G nodes {2.5, 3.5, 4.5, 5.5, 6.5, 7.5} added to arm M as ADDITIVE ROWS.
# 6 x 41 = 246 cells, ~13 min.  The coarse rows are untouched and reused.
#
#   sbatch --export=ALL,STAGE="M",DMUG_NODES=refine \
#       scripts/submit_a10_arms_rita.sbatch
#
# Each arm checkpoints one f row at a time into diagnostics/_a10_arm_<arm>.jsonl
# and resumes from it, so an eviction costs at most one row.  Both stages run in
# the same job but in separate processes.
set -euo pipefail
source /hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_9_marked_multitracer_H0_fagn/scripts/env_a9.sh
A10_DIR=/hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_10_mass_spin_marked_multitracer
echo "[a10] host=$(hostname) job=${SLURM_JOB_ID:-none} stages='${STAGE:-S M}'"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
NODES=${DMUG_NODES:-registered}
for s in ${STAGE:-S M}; do
  echo "[a10] === arm ${s} (dmu_G nodes: ${NODES}) ==="
  "${A9_PY}" "${A10_DIR}/scripts/a10_arms.py" --stage "${s}" --dmuG_nodes "${NODES}"
done
echo "[a10] arms done"
