#!/bin/bash
#SBATCH --job-name=dsm_agg
#SBATCH --account=phy220048p
#SBATCH --partition=RITA-GPU
#SBATCH --qos=rita
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=110G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm_agg_%j.out
#SBATCH --error=logs/slurm_agg_%j.err
#
# The aggregate (current) completeness mode: probe first, then the full 4D fit
# in the same allocation, so the GPU is not handed back between them.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export CMODE=aggregate
./scripts/run_probe.sh
./scripts/run_4d.sh
