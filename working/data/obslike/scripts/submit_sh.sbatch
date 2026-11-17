#!/bin/bash
#SBATCH --job-name=obslike-sh
#SBATCH --account=phy220048p
#SBATCH --partition=RM
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=02:00:00
#SBATCH --output=/hildafs/projects/phy230014p/magana/gws-agn/working/data/obslike/logs/slurm_%x_%j.out
# CPU only: sbatch --job-name=<tag> submit_sh.sbatch <script> [args...]
echo "host $(hostname) job $SLURM_JOB_ID: $*  $(date -u)"
exec "$@"
