#!/bin/bash
#SBATCH --job-name=a11_cpu
#SBATCH --account=phy230014p
#SBATCH --partition=RM
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_11_free_common_population/logs/%x_%j.out
#SBATCH --error=/hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_11_free_common_population/logs/%x_%j.err
# CPU-only Analysis-11 script:  sbatch --job-name=NAME --export=ALL,A11_SCRIPT=scripts/X.py,A11_ARGS="..." scripts/submit_a11_cpu.sbatch
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu
export PYTHONPATH=/hildafs/projects/phy230014p/magana/src/darksirens-a8
PY=/hildafs/home/magana/tmp_ondemand_hildafs_phy230014p_symlink/magana/.conda/envs/jax/bin/python
cd /hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_11_free_common_population
echo "[a11_cpu] host=$(hostname) job=${SLURM_JOB_ID} script=${A11_SCRIPT}"
exec $PY "${A11_SCRIPT:?}" ${A11_ARGS:-}
