#!/bin/bash
#SBATCH --job-name=dsm_probe
#SBATCH --account=phy220048p
#SBATCH --partition=RITA-GPU
#SBATCH --qos=rita
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=110G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm_probe_%j.out
#SBATCH --error=logs/slurm_probe_%j.err
#
# Stage A only: minutes of work, but it needs the GPU because the closure is
# built and evaluated exactly as the production fit builds it.  One of rita's
# two A100-80s was free at submit time.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
./scripts/run_probe.sh
