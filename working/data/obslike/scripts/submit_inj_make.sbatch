#!/bin/bash
#SBATCH --job-name=inj-obslike
#SBATCH --account=phy220048p
#SBATCH --partition=RM
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=03:00:00
#SBATCH --output=/hildafs/projects/phy230014p/magana/gws-agn/working/data/obslike/logs/slurm_%x_%A_%a.out
# CPU only.  sbatch --array=0-4 submit_inj_make.sbatch NDRAW MIXPOP MIXUNIF MIXTGT
SEEDS=(100 101 102 103 105)
S=${SEEDS[$SLURM_ARRAY_TASK_ID]}
echo "host $(hostname) job ${SLURM_JOB_ID} seed $S $(date -u)"
exec /hildafs/projects/phy230014p/magana/gws-agn/working/data/obslike/scripts/inj_make.sh "$S" "$@"
