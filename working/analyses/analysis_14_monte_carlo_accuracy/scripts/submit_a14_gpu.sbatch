#!/bin/bash
#SBATCH --job-name=agn-mcacc
#SBATCH --account=phy220048p
#SBATCH --partition=RITA-GPU
#SBATCH --qos=rita
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100-80:1
#SBATCH --mem=120G
#SBATCH --time=15:00:00
#SBATCH --output=/hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_14_monte_carlo_accuracy/logs/%x_%j.out
#SBATCH --error=/hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_14_monte_carlo_accuracy/logs/%x_%j.err
# Analysis 14 (run R0): Monte Carlo accuracy of the likelihood, darksirens-core e7c3007, historical settings.
# Resubmit identically to resume (finished scans in results/scans/*.json are skipped):
#   sbatch scripts/submit_a14_gpu.sbatch
# The reproduction check runs first; the job exits 3 if the stored A0/A2 posteriors are not reproduced.
# Since 2026-11-17 (b) (seed 100) and (a) (seed 103) run as separate jobs:
#   sbatch --export=ALL,A14_ARGS=--configs=b scripts/submit_a14_gpu.sbatch
#   sbatch --time=03:00:00 --export=ALL,A14_WALL_HOURS=3,A14_ARGS="--configs=a --seed-a=103 --plan-hours=2.5" scripts/submit_a14_gpu.sbatch
set -uo pipefail
A14=/hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_14_monte_carlo_accuracy
cd $A14/scripts
export PYTHONPATH= XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 JAX_PLATFORMS=cuda OMP_NUM_THREADS=4
export A14_JOB_START=$(date +%s)
# wall limit (A14_WALL_HOURS, default 15 = the --time above) minus 25 min for aggregation and slack
export A14_DEADLINE=$(( A14_JOB_START + ${A14_WALL_HOURS:-15}*3600 - 1500 ))
echo "[agn-mcacc] host=$(hostname) job=${SLURM_JOB_ID:-none} start=$(date -u +%FT%TZ)"
/hildafs/projects/phy230014p/magana/envs/darksirens-core-e7c3007/bin/python a14_mc_accuracy.py ${A14_ARGS:-}
rc=$?
echo "[agn-mcacc] driver exit $rc at $(date -u +%FT%TZ)"
if [ $rc -eq 3 ]; then
  echo "[agn-mcacc] STOPPED: the reproduction check failed (results/repro_{a,b}.json); no redraw was run"
  exit 3
fi
# aggregation and figure on CPU (scripts written alongside the job; skipped if absent)
PY=/hildafs/home/magana/tmp_ondemand_hildafs_phy230014p_symlink/magana/.conda/envs/jax/bin/python
for s in aggregate_mc_accuracy.py make_figures.py; do
  if [ -f $A14/scripts/$s ]; then
    JAX_PLATFORMS=cpu $PY $A14/scripts/$s || echo "[agn-mcacc] $s failed"
  else
    echo "[agn-mcacc] $s not present; run it by hand"
  fi
done
echo "[agn-mcacc] end=$(date -u +%FT%TZ)"
exit $rc
