#!/bin/bash
#SBATCH --job-name=a10_gateb
#SBATCH --account=phy220048p
#SBATCH --partition=RITA-GPU
#SBATCH --qos=rita
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100-80:1
#SBATCH --mem=100G
#SBATCH --time=02:00:00
#SBATCH --output=/hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_10_mass_spin_marked_multitracer/logs/a10_gate_b_%j.out
#SBATCH --error=/hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_10_mass_spin_marked_multitracer/logs/a10_gate_b_%j.err
#
# Analysis 10, Gate B -- the GPU stage.  ~120 live selection-term evaluations
# on the two-mark parameter space, at H0 = 67.74.  The local H100 is in use by
# another process and must stay free; every darksirens evaluation runs here.
#
#   sbatch --export=ALL,STAGE=likelihood scripts/submit_a10_gate_b_rita.sbatch
#
# The environment is Analysis 9's, sourced (not inlined): it pins PYTHONPATH to
# darksirens-a8 and refuses to proceed unless that checkout is at af896ca and
# clean.
set -euo pipefail
source /hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_9_marked_multitracer_H0_fagn/scripts/env_a9.sh
A10_DIR=/hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_10_mass_spin_marked_multitracer
echo "[a10] host=$(hostname) job=${SLURM_JOB_ID:-none} stage='${STAGE:-likelihood}'"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
"${A9_PY}" "${A10_DIR}/scripts/a10_selection_support.py" --stage "${STAGE:-likelihood}"
