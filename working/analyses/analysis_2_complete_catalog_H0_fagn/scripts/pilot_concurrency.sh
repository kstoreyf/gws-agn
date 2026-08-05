#!/usr/bin/env bash
# Does a second process on the same A100-40 buy throughput?
#
# The joint grid is 8241 K=2 evaluations per seed at 3.71 s each (measured, not
# extrapolated) = 8.5 GPU-hours per grid, six grids in the campaign.  If two
# concurrent processes each cost less than 2 x 3.71 s per evaluation, the whole
# campaign gets proportionally shorter, so this is worth four minutes to measure.
set -euo pipefail
cd "$(dirname "$0")/.."
. scripts/env.sh
mkdir -p logs results/pilot

SEED=${SEED:-100}
EV=${DATA_ROOT}/seed${SEED}/events/events.h5
COMMON=$(ds_common "$SEED" targeted "$EV")

( nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
    --format=csv -l 20 > logs/pilot_conc_smi.log 2>&1 ) &
SMI=$!

for i in 0 1; do
  python scripts/scan_h0f.py $COMMON \
    --scan joint --h0_grid 60 75 8 --f_grid 0.2 0.4 3 \
    --outdir results/pilot --out_tag pilot_conc_${i} \
    > logs/pilot_conc_${i}.log 2>&1 &
done
wait %2 %3 || true
kill $SMI 2>/dev/null || true

for i in 0 1; do
  echo "--- concurrent process $i ---"
  grep -E "Eval done|first eval" logs/pilot_conc_${i}.log || true
done
echo "--- peak GPU memory ---"
sort -t, -k2 -n logs/pilot_conc_smi.log | tail -3
