#!/usr/bin/env bash
# Full v3 + D3 regeneration of the remaining seeds, serially.
#   ./run_v3_all.sh 101 102 103 105
set -euo pipefail
HERE=/hildafs/projects/phy230014p/magana/gws-agn/working/data
for S in "$@"; do
  echo "=== $(date -u +%H:%M:%S) seed $S ==="
  "$HERE/run_v3_seed.sh" "$S"
  tail -3 "$HERE/logs_gen/v3_seed${S}.log"
done
echo "=== $(date -u +%H:%M:%S) ALL DONE ==="
