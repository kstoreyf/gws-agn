#!/usr/bin/env bash
# ENDGAME tail: (1) the regression check that --survey_override/--dz_scale absent
# reproduces the final sweep's oracle product bit for bit; (2) the GAL half of the
# declared-photo-z-kernel scan.
set -u
cd "$(dirname "$0")/.."
echo "=== REGRESSION: oracle, agn seed 100, no override ==="
python scripts/attr_selmu_oracle.py --tracer agn --seed 100 --tag agn_regress \
    > logs/attr_selmu_agn_regress.log 2>&1 || echo "[FAIL regress]"
python - <<'PY'
import json
a=json.load(open('results/attr_selmu_agn.json'))['dlnmu_at_truth']
b=json.load(open('results/attr_selmu_agn_regress.json'))['dlnmu_at_truth']
print('  record :',a)
print('  regress:',b)
print('  BITWISE IDENTICAL:', a==b)
PY
echo "=== GAL declared-kernel scan (x2) ==="
T=gal SCALES="2" ./scripts/run_dzscan.sh
echo "ENDGAME TAIL DONE"
