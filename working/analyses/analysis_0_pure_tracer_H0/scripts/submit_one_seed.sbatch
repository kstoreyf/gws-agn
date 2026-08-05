#!/bin/bash
#SBATCH --job-name=a0_pure
#SBATCH --account=phy220048p
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm_a0_%j.out
#SBATCH --error=logs/slurm_a0_%j.err
#
#   sbatch --partition=HENON-GPU --qos=henon-gpu scripts/submit_one_seed.sbatch 102
#
# The four K=1 H0 scans of ONE catalog realisation, sized to backfill.  Partition
# and QOS are deliberately NOT set here, so the same file can be aimed at whichever
# GPU partition is free; every one of them takes --account=phy220048p.  Pairings:
# HENON-GPU/henon-gpu, RITA-GPU/rita, MIKO/miko.  Do NOT submit to TWIG-GPU -- it
# is saturated with other users' long-running jobs (owner, 2026-08-02).
#
# RESOURCES, from measurement rather than habit.  analysis_1's eight-scan control
# job (SLURM 1059155) peaked at 4.5 GB RSS against a 160 GB request, so the request
# here is 32 GB.  Seeds 100/101 measured the cost of one realisation on an A100-40:
# two AGN scans at 45 s (0.20 s/cell) and two GAL scans at 795 s (3.93 s/cell),
# ~28 min in all, so 1 h is the walltime with margin -- short enough to backfill.
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

S=${1:?usage: sbatch [--partition=P --qos=Q] scripts/submit_one_seed.sbatch SEED}

echo "=== host: $(hostname)  job: ${SLURM_JOB_ID}  partition: ${SLURM_JOB_PARTITION}  seed: $S  $(date -u) ==="
echo "python: $(which python)"
echo "darksirens: $DARKSIRENS_SRC @ $(git -C "$DARKSIRENS_SRC" rev-parse HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv || true

./scripts/run_scans.sh "$S"
echo "=== $(date -u) JOB DONE (seed $S) ==="
