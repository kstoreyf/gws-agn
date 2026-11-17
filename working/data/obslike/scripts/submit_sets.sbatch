#!/bin/bash
#SBATCH --job-name=extra-sets
#SBATCH --account=phy220048p
#SBATCH --partition=RM
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/hildafs/projects/phy230014p/magana/gws-agn/working/data/obslike/logs/slurm_%x_%j.out
# CPU only.  sbatch --job-name=<kind> submit_sets.sbatch <kind> [seeds...]
exec /hildafs/projects/phy230014p/magana/gws-agn/working/data/obslike/scripts/make_sets.sh "$@"
