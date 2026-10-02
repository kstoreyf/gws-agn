#!/bin/bash
#SBATCH --job-name=af896ca_window
#SBATCH --account=phy220048p
#SBATCH --partition=RITA-GPU
#SBATCH --qos=rita
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100-80:1
#SBATCH --mem=140G
#SBATCH --time=04:00:00
#SBATCH --output=/hildafs/projects/phy230014p/magana/gws-agn/working/analyses/experiments/experiment_darksirens_core_equivalence/logs/%x_%j.out
#SBATCH --error=/hildafs/projects/phy230014p/magana/gws-agn/working/analyses/experiments/experiment_darksirens_core_equivalence/logs/%x_%j.err
# af896ca KDE window W=4096 vs full rows on the A11 likelihood (legacy_window_check.py).
set -uo pipefail
source /hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_9_spin_marked_H0_fagn/scripts/env_a9.sh
cd /hildafs/projects/phy230014p/magana/gws-agn/working/analyses/experiments/experiment_darksirens_core_equivalence/scripts
export PYTHONUNBUFFERED=1 JAX_PLATFORMS=cuda,cpu   # the a8 guard spy needs a CPU device
for W in 4096 full; do "$A9_PY" legacy_window_check.py --window $W || echo "[window] $W FAILED"; done
