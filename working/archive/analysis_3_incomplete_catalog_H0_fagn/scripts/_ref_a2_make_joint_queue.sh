#!/usr/bin/env bash
# Build the joint-grid work queue.
#
# Six joint grids (five realisations on the targeted injection lane + the seed-100
# popuni cross-check), each H0 [50,100] x 201 X f [0,1] x 41 = 8241 K=2
# evaluations at a MEASURED 3.71 s each = 8.5 GPU-hours per grid.  Each grid is
# split into NCHUNK contiguous H0 chunks (~1 h apiece) that any free GPU can pick
# up; scripts/merge_joint.py stitches them and asserts the reassembled H0 axis
# reproduces linspace(50, 100, 201) exactly.
set -euo pipefail
cd "$(dirname "$0")/.."
NCHUNK=${NCHUNK:-8}
Q=queue
mkdir -p "$Q"
: > "$Q/tasks.txt"
for spec in "100 targeted" "101 targeted" "102 targeted" "103 targeted" \
            "105 targeted" "100 popuni"; do
  read -r SEED LANE <<< "$spec"
  for c in $(seq 0 $((NCHUNK - 1))); do
    echo "$SEED $LANE $c $NCHUNK" >> "$Q/tasks.txt"
  done
done
echo "queue: $(wc -l < "$Q/tasks.txt") chunks of $NCHUNK per grid"
cat "$Q/tasks.txt"
