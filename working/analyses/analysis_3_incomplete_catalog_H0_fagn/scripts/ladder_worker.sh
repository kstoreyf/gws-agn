#!/usr/bin/env bash
# One GPU worker: claim tasks from queue/tasks.txt until the queue is empty.
#
# Task line:  KIND SEED LEVEL LANE CHUNK NCHUNK   ("-" "-" = whole grid, one task)
#
# The claim is `mkdir queue/claim_<i>` -- atomic on POSIX, so workers on any
# number of partitions share the queue without a lock server and without a static
# split that strands a free GPU.  A task whose output .h5 already exists is
# skipped, so a killed worker's queue can simply be re-run after
# `rm -rf queue/claim_*`.
#
# Workers run on SHORT walltimes so they backfill, and task cost spans 16x (a
# complete-pair H0 chunk 1.7 GPU-h against an m18 grid 0.42), so a worker does not
# claim blindly: for each candidate it looks up that rung's MEASURED s/eval in
# results/gates.json, divides by the chunk count, and claims only if its own
# remaining allocation covers the task with margin.  A worker that cannot afford
# the expensive tasks still drains the cheap ones instead of sitting idle, and no
# worker dies half way through a task leaving a claimed-but-unfinished cell behind.
set -uo pipefail
cd "$(dirname "$0")/.."
Q=queue
N=$(wc -l < "$Q/tasks.txt")
ME="${SLURM_JOB_ID:-$$}.${SLURM_ARRAY_TASK_ID:-0}"
COST_MARGIN=${COST_MARGIN:-1.5}     # the pilot ran on A100-80; A100-40 is slower
COST_OVERHEAD_S=${COST_OVERHEAD_S:-900}   # data load, JIT, write-out

remaining_s() {  # seconds left in this allocation; huge if not under SLURM
  [ -z "${SLURM_JOB_ID:-}" ] && { echo 999999; return; }
  local L
  L=$(squeue -h -j "${SLURM_JOB_ID}" -o "%L" 2>/dev/null | tr -d ' ' | head -1)
  [ -z "$L" ] && { echo 999999; return; }
  python scripts/_parse_walltime.py "$L"
}

cost_s() {  # expected wall seconds for task at rung $1, kind $2, of $3 chunks
  python scripts/_task_cost.py "$1" "$2" "${3:--}" "$COST_MARGIN" "$COST_OVERHEAD_S"
}

echo "[worker $ME] $(hostname) starting; $N tasks in the queue; remaining $(remaining_s)s"

while true; do
  REM=$(remaining_s)
  CLAIM=""
  SKIPPED_FOR_TIME=0
  for i in $(seq 1 "$N"); do
    [ -d "$Q/claim_$i" ] && continue
    read -r KIND SEED LEVEL LANE CHUNK NCHUNK < <(sed -n "${i}p" "$Q/tasks.txt")
    NEED=$(cost_s "$LEVEL" "$KIND" "$NCHUNK")
    if [ "$REM" -lt "$NEED" ]; then SKIPPED_FOR_TIME=1; continue; fi
    if mkdir "$Q/claim_$i" 2>/dev/null; then CLAIM=$i; break; fi
  done
  if [ -z "$CLAIM" ]; then
    if [ "$SKIPPED_FOR_TIME" = "1" ]; then
      echo "[worker $ME] ${REM}s left is short of every unclaimed task; exiting"
    else
      echo "[worker $ME] queue empty; exiting"
    fi
    break
  fi
  echo "$ME $(date -u +%FT%TZ)" > "$Q/claim_$CLAIM/owner"

  read -r KIND SEED LEVEL LANE CHUNK NCHUNK < <(sed -n "${CLAIM}p" "$Q/tasks.txt")
  SUF=""; [ "$LANE" = "popuni" ] && SUF="_popuni"
  CHUNKED=0; [ "$NCHUNK" != "-" ] && CHUNKED=1
  case "$KIND" in
    joint) TAG="joint_${LEVEL}_s${SEED}${SUF}" ;;
    fnull) TAG="fscan_null_${LEVEL}_s${SEED}${SUF}" ;;
    *)     TAG="${KIND}_${LEVEL}_s${SEED}${SUF}" ;;
  esac
  OUTF="results/${TAG}.h5"
  if [ "$CHUNKED" = "1" ]; then
    TAG="${TAG}_c${CHUNK}"
    OUTF="results/chunks/${TAG}.h5"
  fi
  if [ -f "$OUTF" ]; then
    echo "[worker $ME] $TAG already present; skipping"
    touch "$Q/claim_$CLAIM/done"
    continue
  fi
  echo "[worker $ME] $(date -u +%T) -> $TAG (budget $(cost_s "$LEVEL" "$KIND" "$NCHUNK")s of ${REM}s)"
  T0=$(date +%s)
  if [ "$CHUNKED" = "1" ]; then
    OK=0
    KIND=$KIND SEED=$SEED LEVEL=$LEVEL LANE=$LANE CHUNK=$CHUNK NCHUNK=$NCHUNK \
      ./scripts/run_scan.sh && OK=1
  else
    OK=0
    KIND=$KIND SEED=$SEED LEVEL=$LEVEL LANE=$LANE ./scripts/run_scan.sh && OK=1
  fi
  if [ "$OK" = "1" ]; then
    echo "[worker $ME] $TAG OK in $(( $(date +%s) - T0 ))s"
    touch "$Q/claim_$CLAIM/done"
  else
    echo "[worker $ME] $TAG FAILED after $(( $(date +%s) - T0 ))s"
    touch "$Q/claim_$CLAIM/failed"
  fi
done
