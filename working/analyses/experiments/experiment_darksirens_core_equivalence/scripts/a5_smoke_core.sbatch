#!/bin/bash
#SBATCH --job-name=a5smoke_core
#SBATCH --account=phy220048p
#SBATCH --partition=RITA-GPU
#SBATCH --qos=rita
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100-80:1
#SBATCH --mem=160G
#SBATCH --time=06:00:00
#SBATCH --output=/hildafs/projects/phy230014p/magana/gws-agn/working/analyses/experiments/experiment_darksirens_core_equivalence/logs/%x_%j.out
#SBATCH --error=/hildafs/projects/phy230014p/magana/gws-agn/working/analyses/experiments/experiment_darksirens_core_equivalence/logs/%x_%j.err
# A5 free-anchor smoke, darksirens-core 661ef3d arm (core_a5_smoke.py); memory raised for the
# two catalog copies per catalog that core's K=2 path holds.
set -euo pipefail
cd /hildafs/projects/phy230014p/magana/gws-agn/working/analyses/experiments/experiment_darksirens_core_equivalence/scripts
export PYTHONPATH= XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 JAX_PLATFORMS=cuda OMP_NUM_THREADS=4
/usr/bin/time -v /hildafs/projects/phy230014p/magana/envs/darksirens-core-661ef3d/bin/python core_a5_smoke.py
