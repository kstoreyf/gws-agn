#!/bin/bash
#SBATCH --job-name=dsm_fit
#SBATCH --account=phy220048p
#SBATCH --partition=RITA-GPU
#SBATCH --qos=rita
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/slurm_fit_%j.out
#SBATCH --error=logs/slurm_fit_%j.err
#
# The two offline magnitude fits that anchor the c_mode=selection arm, plus the
# recovery acceptance test against the mock's own LF.
#
# On a GPU node rather than the login node: the BRIGHT-TRUNCATED fit carries a
# per-galaxy truncated normalization that the un-cut fit does not, which turns
# a seconds-long job into a >20 min single-core one at our 821k galaxies.  That
# is too much to park on a shared login node, and JAX vectorises it here.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

# Let JAX see the GPU (fit_selection.sh defaults to cpu for login-node safety).
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false

./scripts/fit_selection.sh
