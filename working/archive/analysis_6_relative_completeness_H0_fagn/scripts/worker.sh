#!/usr/bin/env bash
# One GPU worker: claim tasks from queue/tasks.txt until the queue is empty.
# Analysis 4's worker with the arm/density fields dropped and the output tag
# widened to carry BOTH survey levels.
#
# Task line:  SEED GLEV ALEV LANE
#
# The claim is `mkdir queue/claim_<i>` — atomic on POSIX.  A task whose output
# .h5 already exists is skipped, so a killed worker's queue can simply be re-run
# after `rm -rf queue/claim_*`.
set -uo pipefail
cd "$(dirname "$0")/.."
Q=queue
N=$(wc -l < "$Q/tasks.txt")
ME="${SLURM_JOB_ID:-$$}.${SLURM_ARRAY_TASK_ID:-0}"
COST_MARGIN=${COST_MARGIN:-1.5}
COST_OVERHEAD_S=${COST_OVERHEAD_S:-900}

remaining_s() {
  [ -z "${SLURM_JOB_ID:-}" ] && { echo 999999; return; }
  local L
  L=$(squeue -h -j "${SLURM_JOB_ID}" -o "%L" 2>/dev/null | tr -d ' ' | head -1)
  [ -z "$L" ] && { echo 999999; return; }
  python scripts/_parse_walltime.py "$L"
}

echo "[worker $ME] $(hostname) starting; $N tasks in the queue; remaining $(remaining_s)s"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

while true; do
  REM=$(remaining_s)
  CLAIM=""
  SKIPPED_FOR_TIME=0
  for i in $(seq 1 "$N"); do
    [ -d "$Q/claim_$i" ] && continue
    read -r SEED GLEV ALEV LANE < <(sed -n "${i}p" "$Q/tasks.txt")
    NEED=$(python scripts/_task_cost.py "$GLEV" "$ALEV" "$COST_MARGIN" "$COST_OVERHEAD_S")
    if [ "$REM" -lt "$NEED" ]; then
      echo "[worker $ME] task $i ($GLEV/$ALEV) needs ${NEED}s, ${REM}s left; skipping"
      SKIPPED_FOR_TIME=1; continue
    fi
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

  read -r SEED GLEV ALEV LANE < <(sed -n "${CLAIM}p" "$Q/tasks.txt")
  TAG="joint_g${GLEV}_a${ALEV}_s${SEED}"
  OUTF="results/${TAG}.h5"
  if [ -f "$OUTF" ]; then
    echo "[worker $ME] $TAG already present; skipping"
    touch "$Q/claim_$CLAIM/done"
    continue
  fi
  echo "[worker $ME] $(date -u +%T) -> $TAG"
  if SEED=$SEED GLEV=$GLEV ALEV=$ALEV LANE=$LANE ./scripts/run_scan.sh; then
    touch "$Q/claim_$CLAIM/done"
    echo "[worker $ME] $(date -u +%T) done $TAG"
  else
    echo "[worker $ME] $TAG FAILED (see logs/${TAG}.log); releasing claim"
    tail -25 "logs/${TAG}.log" 2>/dev/null
    rm -rf "$Q/claim_$CLAIM"
    break
  fi
done
