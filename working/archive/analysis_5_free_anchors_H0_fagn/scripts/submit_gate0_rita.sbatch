#!/bin/bash
#SBATCH --job-name=a5_gate0
#SBATCH --account=phy220048p
#SBATCH --partition=RITA-GPU
#SBATCH --qos=rita
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm_a5_gate0_%A_%a.out
#SBATCH --error=logs/slurm_a5_gate0_%A_%a.err
#
# Gate-0 pilot, one GPU per lane: array 0 = dynesty (nlive 200, dlogz 0.1,
# maxcall 80k -> <=4.2 h at m18's 0.185 s/eval), array 1 = emcee (32x1500 =
# 48k evals -> ~2.5 h, peak-initialized).  Submit with --array=0-1.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export DARKSIRENS_SRC=/hildafs/projects/phy230014p/magana/src/darksirens
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
PY=/hildafs/home/magana/tmp_ondemand_hildafs_phy230014p_symlink/magana/.conda/envs/jax/bin
export PATH="$PY:$PATH"

LANES=(dynesty emcee)
./scripts/run_gate0.sh "${LANES[$SLURM_ARRAY_TASK_ID]}"
