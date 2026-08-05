#!/bin/bash
#SBATCH --job-name=model_equiv
#SBATCH --partition=HENON-GPU
#SBATCH --account=phy220048p
#SBATCH --qos=henon-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm_model_equiv_%j.out
#SBATCH --error=logs/slurm_model_equiv_%j.err
#SBATCH --mail-user=imhernan@andrew.cmu.edu
#SBATCH --mail-type=END,FAIL

# experiment_model_equivalence — all 12 scans on ONE GPU in ONE job.
#
# The claim under test is BITWISE equality of two log-likelihood grids, and bit
# equality does not survive a change of device or of compilation, so every arm
# must run on the same physical GPU in the same process environment.  That is
# the whole reason this is a single serial job and not a job array.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

# conda.sh dereferences PS1 unguarded, which trips `set -u` in a batch shell.
PS1=${PS1:-}
set +u
source /opt/packages/anaconda3/etc/profile.d/conda.sh
conda activate jax
set -u

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export DARKSIRENS_SRC=/hildafs/projects/phy230014p/magana/src/darksirens
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4

echo "=== host: $(hostname)  job: ${SLURM_JOB_ID}  $(date -u) ==="
echo "python: $(which python)"
echo "darksirens: $DARKSIRENS_SRC @ $(git -C "$DARKSIRENS_SRC" rev-parse --short HEAD)"
nvidia-smi || true

./scripts/run_equivalence.sh

echo "=== $(date -u) scans complete; comparing ==="
python scripts/compare_models.py > logs/compare_models.log 2>&1 || \
  echo "[warn] comparison failed; the 12 grids are on disk, rerun compare_models.py by hand"
echo "=== $(date -u) JOB DONE ==="
