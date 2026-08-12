#!/usr/bin/env bash
# One GPU worker: claim chunks from queue/tasks.txt until the queue is empty.
#
# The claim is `mkdir queue/claim_<i>` -- atomic on POSIX, so any number of
# workers on any number of partitions can share the queue without a lock server
# and without a static split that strands a free GPU.  A chunk whose output .h5
# already exists is skipped (so a killed worker's queue can simply be re-run after
# `rm -rf queue/claim_*`).
set -uo pipefail
cd "$(dirname "$0")/.."
Q=queue
N=$(wc -l < "$Q/tasks.txt")
ME="${SLURM_JOB_ID:-$$}.${SLURM_ARRAY_TASK_ID:-0}"
echo "[worker $ME] $(hostname) starting; $N chunks in the queue"

while true; do
  CLAIM=""
  for i in $(seq 1 "$N"); do
    if mkdir "$Q/claim_$i" 2>/dev/null; then CLAIM=$i; break; fi
  done
  [ -z "$CLAIM" ] && { echo "[worker $ME] queue empty; exiting"; break; }
  echo "$ME $(date -u +%FT%TZ)" > "$Q/claim_$CLAIM/owner"

  read -r SEED LANE CHUNK NCHUNK < <(sed -n "${CLAIM}p" "$Q/tasks.txt")
  SUF=""; [ "$LANE" = "popuni" ] && SUF="_popuni"
  TAG="joint_s${SEED}${SUF}_c${CHUNK}"
  if [ -f "results/chunks/${TAG}.h5" ]; then
    echo "[worker $ME] $TAG already present; skipping"
    touch "$Q/claim_$CLAIM/done"
    continue
  fi
  echo "[worker $ME] $(date -u +%T) -> $TAG"
  T0=$(date +%s)
  if KIND=joint SEED=$SEED LANE=$LANE CHUNK=$CHUNK NCHUNK=$NCHUNK ./scripts/run_scan.sh; then
    echo "[worker $ME] $TAG OK in $(( $(date +%s) - T0 ))s"
    touch "$Q/claim_$CLAIM/done"
  else
    echo "[worker $ME] $TAG FAILED after $(( $(date +%s) - T0 ))s"
    touch "$Q/claim_$CLAIM/failed"
  fi
done
