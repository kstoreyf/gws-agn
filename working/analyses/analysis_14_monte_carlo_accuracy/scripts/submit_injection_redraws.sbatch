#!/bin/bash
#SBATCH --job-name=agn-mcinj
#SBATCH --account=phy220048p
#SBATCH --partition=HENON
#SBATCH --qos=henon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#SBATCH --output=/hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_14_monte_carlo_accuracy/logs/%x_%j.out
#SBATCH --error=/hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_14_monte_carlo_accuracy/logs/%x_%j.err
# Analysis 14: five independent targeted injection sets for one seed (generator's own stage).
#   seed 100 (configuration b): sbatch scripts/submit_injection_redraws.sbatch
#   seed 103 (configuration a): sbatch --export=ALL,SEED=103 scripts/submit_injection_redraws.sbatch
# Skips sets that already exist, so resubmitting identically resumes:
#   sbatch scripts/submit_injection_redraws.sbatch
set -euo pipefail
cd /hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_14_monte_carlo_accuracy/scripts
export JAX_PLATFORMS=cpu PYTHONUNBUFFERED=1
echo "[agn-mcinj] host=$(hostname) job=${SLURM_JOB_ID:-none} start=$(date -u +%FT%TZ)"
/hildafs/home/magana/tmp_ondemand_hildafs_phy230014p_symlink/magana/.conda/envs/jax/bin/python \
    make_injection_redraws.py --sets ${SETS:-1 2 3 4 5} --jobs 15 --seed ${SEED:-100}
echo "[agn-mcinj] end=$(date -u +%FT%TZ)"
