#!/bin/bash
#SBATCH --job-name=obslike-cpu
#SBATCH --account=phy220048p
#SBATCH --partition=RM
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/hildafs/projects/phy230014p/magana/gws-agn/working/data/obslike/logs/slurm_%x_%j.out
#
# Generic CPU wrapper:  sbatch --job-name=<tag> run_cpu.sbatch <venv: core|jax> <script> [args...]
# core = darksirens-core e7c3007 venv (PYTHONPATH cleared); jax = project env with the
# legacy darksirens pin on PYTHONPATH (what the generator uses).
set -uo pipefail
ENV=${1:?core|jax}; shift
export JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
if [ "$ENV" = core ]; then
  PY=/hildafs/projects/phy230014p/magana/envs/darksirens-core-e7c3007/bin/python; export PYTHONPATH=
else
  PY=/hildafs/home/magana/tmp_ondemand_hildafs_phy230014p_symlink/magana/.conda/envs/jax/bin/python
  export PYTHONPATH=/hildafs/projects/phy230014p/magana/src/darksirens-2b86a2d
fi
echo "host $(hostname) job $SLURM_JOB_ID env $ENV: $*  $(date -u)"
/usr/bin/time -v $PY -u "$@"
