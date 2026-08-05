#!/bin/bash
#SBATCH --job-name=a1_v3ctrl
#SBATCH --partition=HENON-GPU
#SBATCH --account=phy220048p
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm_a1_v3ctrl_%j.out
#SBATCH --error=logs/slurm_a1_v3ctrl_%j.err
#
#   sbatch scripts/submit_v3_controls.sbatch 101 102 103 105
#
# The four remaining matched-host controls of the v3 analysis of record.  Assumes
# working/data/seed* already point at the v3 dataset.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export DARKSIRENS_SRC=/hildafs/projects/phy230014p/magana/src/darksirens
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
PY=/hildafs/home/magana/tmp_ondemand_hildafs_phy230014p_symlink/magana/.conda/envs/jax/bin
export PATH="$PY:$PATH"
export SEL_BATCH=${SEL_BATCH:-50000}
export PE_BLOCK=${PE_BLOCK:-25}
export KDE_W=${KDE_W:-4096}
export LOG10N0=${LOG10N0:--24}
BULK=/hildafs/projects/phy220048p/magana/gws-agn-data/derived/analysis_1_complete_catalog_H0
for S in "$@"; do
  rm -f "$BULK/seed$S/events_gal_hosted.h5" "$BULK/seed$S/events_agn_hosted.h5"
done
./scripts/run_seed_controls.sh "$@"
