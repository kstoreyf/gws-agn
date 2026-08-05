#!/bin/bash
#SBATCH --job-name=a1_gconv
#SBATCH --partition=RM
#SBATCH --account=phy220048p
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --array=0-5
#SBATCH --output=logs/slurm_a1_gconv_%A_%a.out
#SBATCH --error=logs/slurm_a1_gconv_%A_%a.err
# G(b) convergence battery for the selection oracle -- pure quadrature, no GPU
# and no darksirens likelihood: each task rebuilds the kernel with one knob moved
# and re-integrates the oracle's OWN saved host measures, so the reported number
# is the change in d ln mu/dH0 itself.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16
PY=/hildafs/home/magana/tmp_ondemand_hildafs_phy230014p_symlink/magana/.conda/envs/jax/bin/python
V=(base n_m1x2 n_qx2 n_ghx2 dv_half range_wide)
"$PY" -u scripts/attr_selmu_gconv.py --variant "${V[$SLURM_ARRAY_TASK_ID]}"
