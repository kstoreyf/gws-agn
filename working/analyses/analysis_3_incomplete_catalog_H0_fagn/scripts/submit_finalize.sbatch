#!/bin/bash
#SBATCH --job-name=a3_final
#SBATCH --account=phy220048p
#SBATCH --partition=RM
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm_a3_final_%j.out
#SBATCH --error=logs/slurm_a3_final_%j.err
#
# Collate the campaign once the GPU work is done.  CPU only -- no darksirens
# evaluation happens here, just the chunk stitch, the aggregation and the render.
# Submit with `--dependency=afterany:<ladder arrays>:<nside16 job>` so it fires when
# the GPU work has finished however it finished; it is idempotent, so re-running it
# after a partial campaign is harmless and just produces a smaller table.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
PY=/hildafs/home/magana/tmp_ondemand_hildafs_phy230014p_symlink/magana/.conda/envs/jax/bin
export PATH="$PY:$PATH"
export OMP_NUM_THREADS=4
./scripts/finalize.sh
