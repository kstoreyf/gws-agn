#!/usr/bin/env bash
# ENDGAME: the (A-B)/(C-A) split on every post-fix realisation, both matched
# configurations.  darksirens READ-ONLY at 2b86a2d; every run is anchored
# |Delta log mu| = 0.  SEEDS/TRACERS/EXTRA let a caller split the work across
# cards; batching cannot enter any reported statistic (all are exact sums).
set -u
cd "$(dirname "$0")/.."
EXTRA="${EXTRA:-}"
for S in ${SEEDS:-100 101 102 103 105}; do
  for T in ${TRACERS:-gal agn}; do
    OUT="results/abc_${T}_s${S}.json"
    if [ -f "$OUT" ]; then echo "skip $OUT"; continue; fi
    echo "=== abc $T seed $S ==="
    python scripts/attr_abc_split.py --seed "$S" --tracer "$T" --tag "${T}_s${S}" \
        $EXTRA > "logs/abc_${T}_s${S}.log" 2>&1 || echo "[FAIL] $T $S"
    grep -E "^ *tot |ANCHOR" "logs/abc_${T}_s${S}.log" | tail -3
  done
done
echo "ALL DONE"
