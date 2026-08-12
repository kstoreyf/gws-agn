#!/usr/bin/env bash
# GATE (a) -- measure the steady-state K = 2 per-eval cost at both ends of the
# ladder before sizing the campaign.
#
# analysis_2 measured 3.71 s/eval on the COMPLETE GAL+AGN pair, whose catalog
# blocks are (12288, 14569) and (12288, 178).  The magnitude-limited pairs are
# far narrower -- m21 (2795, 53) down to m18 (189, 5) -- and the campaign's stored
# note of ~0.11 s/eval for K = 2 was measured on blocks like these.  Neither
# number transfers: measure both ends and interpolate the campaign from what is
# actually measured, exactly as analysis 2 did.
#
# 12 evaluations per rung (a 4 x 3 grid), which is enough for a steady-state rate
# once the first-cell compile is excluded by scan_h0f.py's own timing block.
set -euo pipefail
cd "$(dirname "$0")/.."
. scripts/env.sh
mkdir -p logs results/pilot

SEED=${SEED:-100}
EV=${DATA_ROOT}/seed${SEED}/events/events.h5
for LEVEL in ${PILOT_LEVELS:-m21 m20 m19 m18}; do
  echo "=== pilot timing: level $LEVEL, seed $SEED ==="
  python -u scripts/scan_h0f.py $(ds_common "$SEED" targeted "$EV" "$LEVEL") \
    --scan joint --h0_grid 60 75 4 --f_grid 0.2 0.4 3 \
    --outdir results/pilot --out_tag pilot_joint_${LEVEL}_s${SEED} \
    > logs/pilot_joint_${LEVEL}_s${SEED}.log 2>&1
  tail -25 logs/pilot_joint_${LEVEL}_s${SEED}.log
done
