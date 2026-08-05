#!/bin/bash
#SBATCH --job-name=a3_gates
#SBATCH --account=phy220048p
#SBATCH --partition=HENON-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=02:30:00
#SBATCH --output=logs/slurm_a3_gates_%j.out
#SBATCH --error=logs/slurm_a3_gates_%j.err
#
# The three validation gates, all of them, before any ladder grid runs:
#   (a) timing    -- steady-state s/eval at every rung, to size the campaign
#   (b) continuity-- the COMPLETE pair under the analysis-3 configuration (true-n0
#                    field term instead of -24), h0scan + fscan on seed 100, to be
#                    compared against analysis 2's own h0scan_s100 / fscan_s100
#   (c) guard     -- N_eff and the total-variance budget at every rung, both lanes
#                    at the worst rung
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export DARKSIRENS_SRC=/hildafs/projects/phy230014p/magana/src/darksirens
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
PY=/hildafs/home/magana/tmp_ondemand_hildafs_phy230014p_symlink/magana/.conda/envs/jax/bin
export PATH="$PY:$PATH"

echo "[$(date -u)] === GATE (a) timing pilot ==="
PILOT_LEVELS="m21 m20 m19 m18" SEED=100 ./scripts/pilot_timing.sh

echo "[$(date -u)] === GATE (b) continuity: COMPLETE pair at true n0, seed 100 ==="
KIND=h0scan SEED=100 LEVEL=complete LANE=targeted ./scripts/run_scan.sh
KIND=fscan  SEED=100 LEVEL=complete LANE=targeted ./scripts/run_scan.sh
python -u scripts/continuity_check.py | tee logs/continuity.log

echo "[$(date -u)] === GATE (c) guard / N_eff along the ladder ==="
GUARD_LEVELS="m18 m19 m20 m21 complete" GUARD_LANES="targeted" SEED=100 ./scripts/run_guard.sh
GUARD_LEVELS="m18" GUARD_LANES="popuni" SEED=100 ./scripts/run_guard.sh

echo "[$(date -u)] === gate summary ==="
python -u scripts/gate_report.py | tee logs/gate_report.log
