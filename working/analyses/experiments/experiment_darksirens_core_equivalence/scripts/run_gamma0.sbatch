#!/bin/bash
#SBATCH --job-name=dscore_g0
#SBATCH --account=phy220048p
#SBATCH --partition=RITA-GPU
#SBATCH --qos=rita
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100-80:1
#SBATCH --mem=100G
#SBATCH --time=02:00:00
#SBATCH --output=/hildafs/projects/phy230014p/magana/gws-agn/working/analyses/experiments/experiment_darksirens_core_equivalence/logs/%x_%j.out
#SBATCH --error=/hildafs/projects/phy230014p/magana/gws-agn/working/analyses/experiments/experiment_darksirens_core_equivalence/logs/%x_%j.err
# c042527 and core with the merger-rate slope pinned to the mock's gamma = 0.
set -uo pipefail
cd /hildafs/projects/phy230014p/magana/gws-agn/working/analyses/experiments/experiment_darksirens_core_equivalence/scripts
SRC=/hildafs/projects/phy230014p/magana/src
BASE_PY=/hildafs/home/magana/tmp_ondemand_hildafs_phy230014p_symlink/magana/.conda/envs/jax/bin/python
CORE_PY=/hildafs/projects/phy230014p/magana/envs/darksirens-core-f527b94/bin/python
export XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 JAX_PLATFORMS=cuda
# PYTHONPATH=$SRC/darksirens-c042527 $BASE_PY legacy_k1_grid.py --arm legacy_c042527_gamma0 --expect darksirens-c042527 --gamma 0 || echo FAILED
for MODE in default fast; do PYTHONPATH= $CORE_PY core_k1_grid.py --mode $MODE --gamma 0 --tag _gamma0 || echo FAILED; done
