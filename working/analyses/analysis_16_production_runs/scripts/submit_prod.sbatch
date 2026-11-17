#!/bin/bash
#SBATCH --job-name=agn-prod-l1
#SBATCH --account=phy220048p
#SBATCH --partition=RITA-GPU
#SBATCH --qos=rita
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100-80:1
#SBATCH --mem=120G
#SBATCH --time=48:00:00
#SBATCH --output=/hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_16_production_runs/logs/%x_%j.out
#SBATCH --error=/hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_16_production_runs/logs/%x_%j.err
# Analysis 16 production job: one GPU, one lane (LANE=1|2) or one task (TASK='<run>|<part>').
# Submitted by scripts/launch.py; a lane job submits its own successor (afterany) while work remains.
# One lane by default (A16_LANES=1): every agn job is chained behind every other pending/running agn-*
# job of the user on RITA-GPU, so this project runs one job at a time there. Submit by hand only with
# that dependency (launch.py does it): --dependency=afterany:<ids of squeue --me -p RITA-GPU -n agn-...>
#   sbatch --job-name=agn-prod-l1 --export=ALL,LANE=1 scripts/submit_prod.sbatch
#   sbatch --job-name=agn-prod-run --export=ALL,TASK='<run>|grid' scripts/submit_prod.sbatch
# A16_WALL_MINUTES must match --time when it is changed (the deadline leaves 15 min for checkpoints).
set -uo pipefail
cd /hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_16_production_runs/scripts
export PYTHONPATH= XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 JAX_PLATFORMS=cuda OMP_NUM_THREADS=4
export A16_JOB_START=$(date +%s)
export A16_DEADLINE=$(( A16_JOB_START + ${A16_WALL_MINUTES:-2880}*60 - 900 ))
PY=/hildafs/projects/phy230014p/magana/envs/darksirens-core-e7c3007/bin/python
echo "[agn-prod] host=$(hostname) job=${SLURM_JOB_ID:-none} lane=${LANE:-} task=${TASK:-} start=$(date -u +%FT%TZ) gpu=${CUDA_VISIBLE_DEVICES:-}"
if [ -n "${LANE:-}" ]; then
  $PY lane_worker.py --lane "$LANE"
else
  $PY lane_worker.py --task "${TASK:?LANE or TASK}"
fi
rc=$?
echo "[agn-prod] end=$(date -u +%FT%TZ) exit=$rc"
exit $rc
