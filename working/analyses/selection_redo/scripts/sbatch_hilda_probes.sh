#!/bin/bash
#SBATCH --job-name=fu-probes
#SBATCH --account=phy220048p
#SBATCH --partition=HENON-GPU
#SBATCH --qos=henon-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100-40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=2-00:00:00
#SBATCH --output=/hildafs/projects/phy230014p/magana/gws-agn/working/analyses/selection_redo/logs_slurm/%x-%j.log

# The two follow-up probes (fu_probes/queue/tasks.txt), sequential on one
# A100-40: KDE-window pin (~30 min) then the zero-density probe (~4-8 h).
# Sized to start immediately on henon-gpu01's free GPU (node had 120 CPU /
# 468G / 1 GPU free at submission).
set -uo pipefail

SCRIPTS_DIR=/hildafs/projects/phy230014p/magana/gws-agn/working/analyses/selection_redo/scripts
ROOT=/hildafs/projects/phy230014p/magana/gws-agn/working/analyses/selection_redo

export DATA_ROOT=/hildafs/projects/phy230014p/magana/gws-agn/working/data
export DARKSIRENS_SRC=/hildafs/projects/phy230014p/magana/src/darksirens-0c5b3db
export PY=/hildafs/projects/phy230014p/magana/.conda/envs/jax/bin/python
export FITS_DIR=${SCRIPTS_DIR}/selection_fits_truez

bash "${SCRIPTS_DIR}/run_campaign.sh" fu_probes

# The zero-density cell ran on an 80G H100 in the campaign; on 40G an OOM is
# conceivable. Batch sizes are pure chunking (numerically inert), so retry
# once with smaller blocks if -- and only if -- the failure was memory.
TAG=joint_complete_n0m24_s100
RES="$ROOT/fu_probes/results/$TAG.json"
LOG="$ROOT/fu_probes/logs/$TAG.log"
if [ ! -f "$RES" ] && [ -f "$LOG" ] && grep -qiE "RESOURCE_EXHAUSTED|out of memory|OOM" "$LOG"; then
  echo "[fallback] $TAG hit OOM -- retrying with sel_batch_size 12500, pe_event_block 5"
  export SCRIPTS="$SCRIPTS_DIR" RESULTS="$ROOT/fu_probes/results"
  export PYTHONPATH="${DARKSIRENS_SRC}${PYTHONPATH:+:${PYTHONPATH}}"
  export XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=8
  task=$(grep -- "$TAG" "$ROOT/fu_probes/queue/tasks.txt" \
         | sed 's/--sel_batch_size 50000/--sel_batch_size 12500/; s/--pe_event_block 25/--pe_event_block 5/')
  eval "$task --outdir $RESULTS" > "$LOG.retry" 2>&1 \
    && echo "[fallback] $TAG OK on retry" \
    || { echo "[fallback] $TAG FAILED on retry -- see $LOG.retry"; tail -n 15 "$LOG.retry"; }
fi
