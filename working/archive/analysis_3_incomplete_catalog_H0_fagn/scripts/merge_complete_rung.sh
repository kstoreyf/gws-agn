#!/usr/bin/env bash
# Stitch the complete-rung H0 chunks into rung 0 of record.
#
# Only the complete pair is chunked (6.8 GPU-h per grid, more than one 6 h worker);
# m21..m18 run whole.  scripts/merge_joint.py asserts the reassembled H0 axis
# reproduces linspace(50, 100, 201) value for value before writing anything, so a
# missing or duplicated chunk is a hard failure rather than a silent gap.
#
# Safe to re-run at any point: a grid whose chunks are not all present is skipped.
set -uo pipefail
cd "$(dirname "$0")/.."
NCHUNK=${NCHUNK:-4}
merged=0; waiting=0

for spec in "100 targeted" "101 targeted" "102 targeted" "103 targeted" \
            "105 targeted" "100 popuni"; do
  read -r SEED LANE <<< "$spec"
  SUF=""; [ "$LANE" = "popuni" ] && SUF="_popuni"
  TAG="joint_complete_s${SEED}${SUF}"
  if [ -f "results/${TAG}.h5" ]; then
    echo "  $TAG already merged"; merged=$((merged+1)); continue
  fi
  have=$(ls results/chunks/${TAG}_c*.h5 2>/dev/null | wc -l)
  if [ "$have" -ne "$NCHUNK" ]; then
    echo "  $TAG waiting: $have/$NCHUNK chunks"; waiting=$((waiting+1)); continue
  fi
  echo "  $TAG merging $have chunks"
  python -u scripts/merge_joint.py \
    --chunks results/chunks/${TAG}_c*.h5 \
    --out_tag "$TAG" --outdir results \
    --h0_grid 50.0 100.0 201 --h0_true 67.74 --f_true 0.30 \
    >> logs/merge_complete.log 2>&1 \
    && { echo "    OK"; merged=$((merged+1)); } \
    || echo "    MERGE FAILED (see logs/merge_complete.log)"
done
echo "merged $merged, waiting $waiting"
