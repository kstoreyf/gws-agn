#!/bin/bash
#SBATCH --job-name=gen100
#SBATCH --account=phy220048p
#SBATCH --partition=RM
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm_gen_%A_%a.out
#SBATCH --error=logs/slurm_gen_%A_%a.err
#
# Mock generation for the 100-seed campaign.  CPU only -- generate_dataset.py pins
# JAX_PLATFORMS=cpu -- so this belongs on RM, not on a GPU partition.
#
#   sbatch --array=0-94%16 submit_gen.sbatch 106
#
# $1 is the seed the array's task 0 maps to; task i generates seed $1 + i.
#
# CONCURRENCY IS A DISK LIMIT, NOT A CPU ONE.  Each task holds ~9.4 GB until it
# prunes catalogs/, so %16 bounds the transient at ~150 GB.  /hildafs was at 98%
# (1.6 TB free) when this campaign was sized; the retained footprint is ~2.7 GB per
# seed, ~260 GB for 95 seeds.
#
# Measured on seeds 102/103/105: 12 min per seed (catalogs 3:54, events 0:15,
# surveys 3:42, injections 1:22, validation 2:49).  2 h is walltime with margin for
# contention.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
BASE=${1:?usage: sbatch --array=0-94%16 submit_gen.sbatch FIRST_SEED}
S=$(( BASE + SLURM_ARRAY_TASK_ID ))
echo "=== host $(hostname)  job ${SLURM_JOB_ID}  task ${SLURM_ARRAY_TASK_ID}  seed $S  $(date -u) ==="
exec ./gen_one_seed.sh "$S"
