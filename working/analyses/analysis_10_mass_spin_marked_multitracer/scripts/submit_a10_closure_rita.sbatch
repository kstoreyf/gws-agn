#!/bin/bash
#SBATCH --job-name=a10_closure
#SBATCH --account=phy220048p
#SBATCH --partition=RITA-GPU
#SBATCH --qos=rita
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100-80:1
#SBATCH --mem=100G
#SBATCH --time=01:00:00
#SBATCH --output=/hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_10_mass_spin_marked_multitracer/logs/a10_closure_%j.out
#SBATCH --error=/hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_10_mass_spin_marked_multitracer/logs/a10_closure_%j.err
#
# Analysis 10, Gate 15 -- the closure identities on the TWO-MARK mock.
# STAGE is a space-separated list of stages, each run in its OWN process so no
# two builds share a GPU allocator:
#
#   sbatch --export=ALL,STAGE="closure k1refs" scripts/submit_a10_closure_rita.sbatch
#
# 'closure' builds the 10-label two-mark model on
# events_marked_dmu0p10_dmuG5.h5, runs 15.1-15.6 and the two-mark side of 15.7,
# and measures the cost model.  'k1refs' builds the four single-branch
# references 15.7 compares against.  The CPU assemble
# (python scripts/a10_closure.py --stage assemble) writes the verdict.
#
# The local H100 is in use by another process and must stay free: every
# darksirens evaluation runs here.
set -euo pipefail
source /hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_9_marked_multitracer_H0_fagn/scripts/env_a9.sh
A10_DIR=/hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_10_mass_spin_marked_multitracer
echo "[a10] host=$(hostname) job=${SLURM_JOB_ID:-none} stages='${STAGE:-closure}'"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
for s in ${STAGE:-closure}; do
  echo "[a10] === stage ${s} ==="
  "${A9_PY}" "${A10_DIR}/scripts/a10_closure.py" --stage "${s}"
done
echo "[a10] all stages done"
