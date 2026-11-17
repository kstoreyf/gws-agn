#!/bin/bash
#SBATCH --job-name=corepop-more
#SBATCH --account=phy220048p
#SBATCH --partition=RM
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/hildafs/projects/phy230014p/magana/gws-agn/working/data/corepop/logs/more_seeds/slurm_%A_%a.out
#
# CPU only.  sbatch --array=0-4 submit_more_seeds.sbatch 106  -> seeds 106..110
set -uo pipefail
BASE=${1:?usage: sbatch --array=0-4 submit_more_seeds.sbatch FIRST_SEED}
S=$(( BASE + SLURM_ARRAY_TASK_ID ))
exec /hildafs/projects/phy230014p/magana/gws-agn/working/data/corepop/scripts/more_seeds_job.sh "$S"
