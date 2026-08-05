#!/bin/bash
#SBATCH --job-name=rita-darksirens-fix
#SBATCH --partition=HENON-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm_twotracer_seeds_fix_%j.out
#SBATCH --error=logs/slurm_twotracer_seeds_fix_%j.err
#SBATCH --mail-user=imhernan@andrew.cmu.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --account=phy220048p

# 12-seed rerun of the deep two-tracer mock with the FIXED generator
# (darksirens-oraclefix: observed-data detection + sequential observable
# sigma_ang, snr_ref=6.278).  Catalogs are reused from the pre-fix run; events
# and injections are regenerated per seed (CPU stage, parallel), then the
# guard + f-scan + joint scan run serially on the one GPU.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2

SEEDS="7301 7302 7303 7304 7305 7306 7307 7308 7309 7310 7311 7312"
MAXPAR=6

echo "=== $(date -u) CPU stage: mock + injections (catalogs reused) ==="
n=0
for s in $SEEDS; do
  ./scripts/one_realisation_fix.sh "$s" cpu &
  n=$((n + 1))
  if [ $((n % MAXPAR)) -eq 0 ]; then wait; fi
done
wait
echo "=== $(date -u) CPU stage complete ==="

echo "=== $(date -u) GPU stage: guard + scans ==="
for s in $SEEDS; do
  ./scripts/one_realisation_fix.sh "$s" gpu
  echo "  $(date -u) seed $s scanned"
done
echo "=== $(date -u) GPU stage complete ==="

python scripts/aggregate_seeds_fix.py > logs/aggregate_fix.log 2>&1 || \
  echo "[warn] aggregation failed; results are on disk, rerun by hand"
echo "=== $(date -u) ALL DONE ==="
