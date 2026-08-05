#!/usr/bin/env bash
# ENDGAME: the EXACT selection-function oracle (task-1 machinery) on every
# post-fix realisation, so B = d ln mu/dH0 is the exact number on that seed's own
# catalog and the injection estimator's common-mode Monte-Carlo error drops out.
set -u
cd "$(dirname "$0")/.."
for S in 101 102 103 105; do
  for T in agn gal; do
    OUT="results/attr_selmu_${T}_s${S}.json"
    if [ -f "$OUT" ]; then echo "skip $OUT"; continue; fi
    echo "=== oracle $T seed $S ==="
    python scripts/attr_selmu_oracle.py --tracer "$T" --seed "$S" \
        --tag "${T}_s${S}" > "logs/attr_selmu_${T}_s${S}.log" 2>&1 \
      || echo "[FAIL] $T $S"
  done
done
echo "ALL DONE"
