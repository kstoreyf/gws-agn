#!/bin/bash
#SBATCH --job-name=a1_ds_scans
#SBATCH --partition=TWIG-GPU
#SBATCH --account=phy220048p
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm_a1_scans_%j.out
#SBATCH --error=logs/slurm_a1_scans_%j.err

# analysis_1_complete_catalog_H0 — the six scans of record (four production
# configurations + two matched-host controls) under `dark_sirens` at
# log10n0 = -24, then the four guard diagnostics at truth, then the figures and
# the paper-facing summary.
#
# twig carries A100-40GB cards, so both reductions are blocked twice as tightly
# as the 80GB production setting.  The blocking is identical for every scan here,
# so it cannot enter any comparison between them.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export DARKSIRENS_SRC=/hildafs/projects/phy230014p/magana/src/darksirens
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4

PY=/hildafs/home/magana/tmp_ondemand_hildafs_phy230014p_symlink/magana/.conda/envs/jax/bin/python
export PATH="$(dirname "$PY"):$PATH"

export SEL_BATCH=${SEL_BATCH:-50000}
export PE_BLOCK=${PE_BLOCK:-25}
export KDE_W=${KDE_W:-4096}
export LOG10N0=${LOG10N0:--24}

echo "=== host: $(hostname)  job: ${SLURM_JOB_ID:-none}  $(date -u) ==="
echo "python: $(which python)"
echo "darksirens: $DARKSIRENS_SRC @ $(git -C "$DARKSIRENS_SRC" rev-parse --short HEAD)"
echo "SEL_BATCH=$SEL_BATCH PE_BLOCK=$PE_BLOCK KDE_W=$KDE_W LOG10N0=$LOG10N0"
nvidia-smi --query-gpu=name,memory.total --format=csv || true

./scripts/run_scans.sh
echo "=== $(date -u) scans done; guard diagnostics ==="
./scripts/run_guard_diag.sh || echo "[warn] guard diagnostics failed"
echo "=== $(date -u) building h0_single_tracer.json ==="
python scripts/build_single_tracer.py > logs/build_single_tracer.log 2>&1 || \
  echo "[warn] build_single_tracer failed"
echo "=== $(date -u) figures ==="
python scripts/make_figures.py > logs/make_figures.log 2>&1 || \
  echo "[warn] make_figures failed"
echo "=== $(date -u) JOB DONE ==="
