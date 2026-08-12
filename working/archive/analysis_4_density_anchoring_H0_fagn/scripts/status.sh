#!/usr/bin/env bash
# Queue and results at a glance.
set -euo pipefail
cd "$(dirname "$0")/.."
N=$(wc -l < queue/tasks.txt 2>/dev/null || echo 0)
DONE=0; CLAIMED=0
for i in $(seq 1 "$N"); do
  [ -d "queue/claim_$i" ] || continue
  CLAIMED=$((CLAIMED + 1))
  [ -f "queue/claim_$i/done" ] && DONE=$((DONE + 1))
done
echo "queue: $DONE done / $CLAIMED claimed / $N total"
echo "results: $(ls results/joint_*.h5 2>/dev/null | wc -l) grids"
squeue -u "$USER" -n a4_arms -o "%.10i %.4t %.10M %.20R" 2>/dev/null || true
for L in logs/joint_*.log; do
  [ -f "$L" ] || continue
  printf "%-28s %s\n" "$(basename "$L" .log)" "$(tail -1 "$L" | cut -c1-90)"
done
