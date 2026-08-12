#!/bin/bash
#SBATCH --job-name=a5_campaign
#SBATCH --account=phy220048p
#SBATCH --partition=HENON-GPU
#SBATCH --qos=henon-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm_a5_campaign_%j.out
#SBATCH --error=logs/slurm_a5_campaign_%j.err
#
# Sequential analysis-5 campaign worker (skips finished tags — effectively the
# m21 4D fit: ~90k calls at ~1.25 s/eval on A100-40 ≈ 31 h; dynesty checkpoints
# every 15 min, so on walltime-kill just resubmit and it resumes).
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export DARKSIRENS_SRC=/hildafs/projects/phy230014p/magana/src/darksirens
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
PY=/hildafs/home/magana/tmp_ondemand_hildafs_phy230014p_symlink/magana/.conda/envs/jax/bin
export PATH="$PY:$PATH"
./scripts/run_campaign_hilda.sh
