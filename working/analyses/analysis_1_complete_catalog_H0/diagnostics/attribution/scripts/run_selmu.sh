#!/bin/bash
# TASK 1 -- the exact selection-function oracle, end to end.
#
#   ./scripts/run_selmu.sh            everything
#   ./scripts/run_selmu.sh pdet       only the closed-form P_det + its validation
#   ./scripts/run_selmu.sh oracle     only the mu quadrature (both catalogs)
#   ./scripts/run_selmu.sh inj        only darksirens' injection curves
#   ./scripts/run_selmu.sh summary    only the comparison + figure
#
# One A100-80GB.  P_det validation ~7 min (CPU), each mu oracle ~25 min,
# each injection curve ~10 min (AGN) / ~40 min (GAL).
set -euo pipefail
cd "$(dirname "$0")/.."
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export DARKSIRENS_SRC=${DARKSIRENS_SRC:-/hildafs/projects/phy230014p/magana/src/darksirens}
MODE=${1:-all}

if [ "$MODE" = "all" ] || [ "$MODE" = "pdet" ]; then
  echo "[$(date -u +%H:%M:%S)] P_det closed form + brute-force validation"
  python -u scripts/attr_selmu_pdet.py --n_mc 4e7 > logs/attr_selmu_pdet.log 2>&1
fi

if [ "$MODE" = "all" ] || [ "$MODE" = "oracle" ]; then
  # the G(b) convergence battery is catalog-independent, so it runs once (AGN)
  echo "[$(date -u +%H:%M:%S)] mu oracle, AGN (+ convergence battery)"
  python -u scripts/attr_selmu_oracle.py --tracer agn --conv \
      > logs/attr_selmu_agn.log 2>&1
  echo "[$(date -u +%H:%M:%S)] mu oracle, GAL"
  python -u scripts/attr_selmu_oracle.py --tracer gal --conv_lat \
      > logs/attr_selmu_gal.log 2>&1
fi

if [ "$MODE" = "all" ] || [ "$MODE" = "inj" ]; then
  for TR in agn gal; do
    for LANE in targeted popuni; do
      echo "[$(date -u +%H:%M:%S)] injections ${TR}/${LANE}"
      python -u scripts/attr_selmu_inj.py --tracer $TR --injections $LANE \
          > logs/attr_selmu_inj_${TR}_${LANE}.log 2>&1
    done
  done
fi

if [ "$MODE" = "all" ] || [ "$MODE" = "summary" ]; then
  echo "[$(date -u +%H:%M:%S)] summary + figure"
  python -u scripts/attr_selmu_summary.py | tee logs/attr_selmu_summary.log
  python -u scripts/fig_selmu_oracle.py
fi
echo "[$(date -u +%H:%M:%S)] SELMU DONE"
