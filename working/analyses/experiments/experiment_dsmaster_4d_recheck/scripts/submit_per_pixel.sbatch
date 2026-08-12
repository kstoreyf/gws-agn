#!/bin/bash
#SBATCH --job-name=dsm_pp
#SBATCH --account=phy220048p
#SBATCH --partition=RITA-GPU
#SBATCH --qos=rita
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=110G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm_pp_%j.out
#SBATCH --error=logs/slurm_pp_%j.err
#
# The LEGACY per_pixel arm -- the control.  Stage A already showed the per_pixel
# likelihood is bit-identical to de2a8df at the truth anchors; this extends that
# over the full prior volume, so any aggregate-vs-analysis-5 difference can be
# attributed to the estimator change rather than to 14 commits of code drift.
#
# Deliberately queued BEHIND the aggregate arm: aggregate is the current mode
# and the one the owner asked for, so it gets the GPU first.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export CMODE=per_pixel
./scripts/run_4d.sh
