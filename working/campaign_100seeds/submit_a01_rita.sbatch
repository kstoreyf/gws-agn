#!/bin/bash
#SBATCH --job-name=a01_100
#SBATCH --account=phy220048p
#SBATCH --partition=RITA-GPU
#SBATCH --qos=rita
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/slurm_a01_%A_%a.out
#SBATCH --error=logs/slurm_a01_%A_%a.err
#
# One worker per rita GPU on the shared a0+a1 seed queue:
#   sbatch --array=0-1 submit_a01_rita.sbatch
#
# RESOURCES.  analysis_0's own sbatch measured its eight-scan sibling at 4.5 GB RSS
# against a 160 GB request and settled on 32 GB; a1's controls read the same files,
# so 64 GB covers both arms with margin.  The GAL grids are the cost: ~790 s each on
# an A100-40, less on rita's A100-80s.
#
# Walltime: 95 seeds x ~0.45 GPU-h / 2 GPUs is ~21 h of scanning, but the workers
# also idle while generation catches up, so 2 days.  Workers exit on their own once
# every queued seed has its four a0 grids.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
exec ./a01_worker.sh
