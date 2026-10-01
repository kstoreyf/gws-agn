#!/bin/bash
#SBATCH --job-name=fn0_fix
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=60G
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm_fn0_fix_%j.out
#SBATCH --error=logs/slurm_fn0_fix_%j.err
#SBATCH --account=phy220048p

# Post-fix (f_AGN, log10 n0_AGN) grids; LEVELS passed via --export so the ladder
# can be split across HENON-GPU/TWIG-GPU. See run_fn0_ladder_fix.sh.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=2
LEVELS="${LEVELS:?set LEVELS via sbatch --export}" ./scripts/run_fn0_ladder_fix.sh
