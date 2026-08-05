#!/bin/bash
#SBATCH --job-name=probe3_decomp
#SBATCH --partition=TWIG-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm_probe3_%j.out
#SBATCH --error=logs/slurm_probe3_%j.err
#SBATCH --account=phy220048p

# PROBE 3 -- numerator/selection decomposition of the matched-host scans,
# both tracers x all five realisations.  twig carries A100-40GB cards, so the
# reduction blocking is tighter than the 80GB production setting; all ten runs
# share it, and the GAL seed-100 arm is compared against the stored
# ctrl_gal_matched scan in the aggregate step.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2

PY=/hildafs/home/magana/tmp_ondemand_hildafs_phy230014p_symlink/magana/.conda/envs/jax/bin/python
SEEDS="100 101 102 103 105"
SEL=${SEL:-50000}
PEB=${PEB:-25}

echo "=== $(date -u) probe3 on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv || true

for s in $SEEDS; do
  for tr in agn gal; do
    OUT=results/probe3_decomp_${tr}_s${s}.json
    if [ -f "$OUT" ]; then echo "skip $OUT"; continue; fi
    EXTRA=""
    [ "$tr" = "gal" ] && EXTRA="--kde_window 4096"
    echo "--- $(date -u) $tr seed $s ---"
    $PY scripts/probe3_decomposition.py --seed "$s" --tracer "$tr" \
        --h0_grid 50.0 100.0 201 --sel_batch_size "$SEL" --pe_event_block "$PEB" \
        $EXTRA > "logs/probe3_${tr}_s${s}.log" 2>&1
    echo "    done $tr s$s"
  done
done

$PY scripts/probe3_decomposition.py --aggregate --seeds $SEEDS \
    > logs/probe3_aggregate.log 2>&1 || echo "[warn] aggregate failed"
echo "=== $(date -u) probe3 ALL DONE ==="
