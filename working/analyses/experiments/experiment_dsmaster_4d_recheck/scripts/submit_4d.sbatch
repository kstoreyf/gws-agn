#!/bin/bash
#SBATCH --job-name=dsm_4d
#SBATCH --account=phy220048p
#SBATCH --partition=RITA-GPU
#SBATCH --qos=rita
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=110G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm_4d_%j.out
#SBATCH --error=logs/slurm_4d_%j.err
#
# Stage B: the full 4D dynesty fit at analysis-5 settings on darksirens master.
# m18 cost 5731 s on the js2 H100; 8 h is generous headroom for a rita A100-80.
# dynesty checkpoints every 900 s, so a walltime kill resumes on resubmission.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
./scripts/run_4d.sh
