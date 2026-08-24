#!/bin/bash
#SBATCH --account=phy220048p
#SBATCH --partition=RITA-GPU
#SBATCH --qos=rita
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100-80:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=3-00:00:00
#SBATCH --output=/hildafs/projects/phy230014p/magana/gws-agn/working/analyses/selection_redo/logs_slurm/%x-%j.log

# One seed's m18 replication (two dynesty arms, sequential) on one RITA
# A100-80.  Submit as:
#   sbatch -J fu-m18-s101 sbatch_hilda_seed.sh 101
#   sbatch -J fu-m18-s102 sbatch_hilda_seed.sh 102
# mem=120G x2 fits rita's 250G node so both jobs start together; dynesty
# checkpoints every 900 s, so a requeued job resumes from its .ckpt.
set -uo pipefail
SEED=${1:?usage: sbatch sbatch_hilda_seed.sh <SEED>}

SCRIPTS_DIR=/hildafs/projects/phy230014p/magana/gws-agn/working/analyses/selection_redo/scripts

export DATA_ROOT=/hildafs/projects/phy230014p/magana/gws-agn/working/data
export DARKSIRENS_SRC=/hildafs/projects/phy230014p/magana/src/darksirens-0c5b3db
export PY=/hildafs/projects/phy230014p/magana/.conda/envs/jax/bin/python
export FITS_DIR=${SCRIPTS_DIR}/selection_fits_truez

bash "${SCRIPTS_DIR}/run_campaign.sh" "fu_seed${SEED}"
