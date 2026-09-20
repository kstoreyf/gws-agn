#!/bin/bash
#SBATCH --job-name=a9_stage
#SBATCH --account=phy220048p
#SBATCH --partition=RITA-GPU
#SBATCH --qos=rita
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100-80:1
#SBATCH --mem=100G
#SBATCH --time=08:00:00
#SBATCH --output=/hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_9_marked_multitracer_H0_fagn/logs/a9_stage_%j.out
#SBATCH --error=/hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_9_marked_multitracer_H0_fagn/logs/a9_stage_%j.err
#
# The single-GPU stages, in order.  STAGE is a space-separated list:
#
#   sbatch --export=ALL,STAGE="timing"        scripts/submit_a9_stage_rita.sbatch
#   sbatch --export=ALL,STAGE="timing anchor" scripts/submit_a9_stage_rita.sbatch
#
# 'timing' is a handful of cells: the rita per-eval rate, the projected cube
# cost and the H0 liveness probe.  'anchor' is the H0 = 67.74 slab -- the same
# 41 x 61 cells Analysis 8 ran as arm J, so it is the equivalence gate AND
# 1/36 of the production cube; nothing it computes is thrown away.
set -euo pipefail
source /hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_9_marked_multitracer_H0_fagn/scripts/env_a9.sh
echo "[a9] host=$(hostname) job=${SLURM_JOB_ID:-none} stages='${STAGE:-timing}'"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
for s in ${STAGE:-timing}; do
  echo "[a9] === stage ${s} ==="
  "${A9_PY}" "${A9_DIR}/scripts/a9_scan.py" --stage "${s}" ${A9_EXTRA:-}
done
