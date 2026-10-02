#!/bin/bash
#SBATCH --job-name=dscore_bisect
#SBATCH --account=phy220048p
#SBATCH --partition=RITA-GPU
#SBATCH --qos=rita
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100-80:1
#SBATCH --mem=100G
#SBATCH --time=05:00:00
#SBATCH --output=/hildafs/projects/phy230014p/magana/gws-agn/working/analyses/experiments/experiment_darksirens_core_equivalence/logs/%x_%j.out
#SBATCH --error=/hildafs/projects/phy230014p/magana/gws-agn/working/analyses/experiments/experiment_darksirens_core_equivalence/logs/%x_%j.err
#
# Legacy-history bisection 0c5b3db..c042527 on one GPU (bisect_legacy.py).
#   sbatch scripts/run_bisect.sbatch
set -uo pipefail
cd /hildafs/projects/phy230014p/magana/gws-agn/working/analyses/experiments/experiment_darksirens_core_equivalence/scripts
SRC=/hildafs/projects/phy230014p/magana/src
BASE_PY=/hildafs/home/magana/tmp_ondemand_hildafs_phy230014p_symlink/magana/.conda/envs/jax/bin/python
CORE_PY=/hildafs/projects/phy230014p/magana/envs/darksirens-core-f527b94/bin/python
export XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 JAX_PLATFORMS=cuda
echo "[bisect] host=$(hostname) job=${SLURM_JOB_ID:-none}"
$BASE_PY bisect_legacy.py
