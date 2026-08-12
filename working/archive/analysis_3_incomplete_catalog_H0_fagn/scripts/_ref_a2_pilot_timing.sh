#!/usr/bin/env bash
# Measure the steady-state K=2 per-eval cost on the COMPLETE GAL+AGN pair before
# sizing the campaign.  The campaign memory note quotes ~0.11 s/eval for K=2, but
# that was measured on the magnitude-limited survey blocks; analysis_1 measured
# 6.13 s/eval for the K=1 COMPLETE GAL block at W = 4096, which is the term that
# dominates here.  Measure, do not extrapolate.
set -euo pipefail
cd "$(dirname "$0")/.."
. scripts/env.sh
mkdir -p logs results/pilot

SEED=${SEED:-100}
EV=${DATA_ROOT}/seed${SEED}/events/events.h5
python scripts/scan_h0f.py $(ds_common "$SEED" targeted "$EV") \
  --scan joint --h0_grid 60 75 4 --f_grid 0.2 0.4 3 \
  --outdir results/pilot --out_tag pilot_joint_s${SEED} \
  > logs/pilot_joint_s${SEED}.log 2>&1
tail -40 logs/pilot_joint_s${SEED}.log
