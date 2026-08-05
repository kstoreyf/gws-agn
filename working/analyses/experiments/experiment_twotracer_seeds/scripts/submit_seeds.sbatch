#!/bin/bash
#SBATCH --job-name=twotracer_seeds
#SBATCH --partition=RITA-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gpus=1
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm_twotracer_seeds_%j.out
#SBATCH --error=logs/slurm_twotracer_seeds_%j.err
#SBATCH --mail-user=imhernan@andrew.cmu.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --account=phy220048p

# Catalog-realisation error bars for the deep two-tracer mock.
# CPU stage (catalog + mock + 120M-draw targeted injections) is embarrassingly
# parallel across seeds and is the long pole; the GPU stage is serialised
# because there is one device.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2

SEEDS="7301 7302 7303 7304 7305 7306 7307 7308 7309 7310 7311 7312"
MAXPAR=6

echo "=== $(date -u) CPU stage: catalog + mock + injections ==="
n=0
for s in $SEEDS; do
  ./scripts/one_realisation.sh "$s" cpu &
  n=$((n + 1))
  if [ $((n % MAXPAR)) -eq 0 ]; then wait; fi
done
wait
echo "=== $(date -u) CPU stage complete ==="

echo "=== $(date -u) GPU stage: guard + scans ==="
for s in $SEEDS; do
  ./scripts/one_realisation.sh "$s" gpu
  echo "  $(date -u) seed $s scanned"
done
echo "=== $(date -u) GPU stage complete ==="

python scripts/aggregate_seeds.py > logs/aggregate.log 2>&1 || \
  echo "[warn] aggregation failed; results are on disk, rerun by hand"
echo "=== $(date -u) ALL DONE ==="
