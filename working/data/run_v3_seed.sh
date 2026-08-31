#!/usr/bin/env bash
# Full v3 + D3 regeneration of one seed into an explicit output root.
#
#   ./run_v3_seed.sh <SEED> [OUTROOT]
#
# Runs every stage in order under the v3 measurement family (DESIGN_PE.md) with the
# declared photo-z realised in the catalogs (D3).  Logs to logs_gen/v3_seed<S>.log.
set -euo pipefail
HERE=/hildafs/projects/phy230014p/magana/gws-agn/working/data
cd "$HERE"
S=${1:?usage: run_v3_seed.sh SEED [OUTROOT]}
OUT=${2:-/hildafs/projects/phy220048p/magana/gws-agn-data-v3}
LOG=$HERE/logs_gen/v3_seed${S}.log
mkdir -p "$OUT" "$HERE/logs_gen"

# The record's campaign sizes -- NOT the CLI defaults (120e6 for both).  The
# analysis of record was run on 1.5e8 targeted + 4.0e8 popuni proposals; keeping
# those makes the v3 selection integrals directly comparable to the v2 record.
NDRAW_T=${NDRAW_T:-150000000}
NDRAW_P=${NDRAW_P:-400000000}
STAGES=${STAGES:-"catalogs events surveys injections validation"}

: > "$LOG"
for ST in $STAGES; do
  echo "=== $(date -u +%H:%M:%S) stage $ST seed $S ===" >> "$LOG"
  EXTRA=""
  [ "$ST" = injections ] && EXTRA="--ndraw_targeted $NDRAW_T --ndraw_popuni $NDRAW_P"
  python "$HERE/generate_dataset.py" --seed "$S" --stage "$ST" \
      --outroot "$OUT" --overwrite $EXTRA >> "$LOG" 2>&1
done
echo "=== $(date -u +%H:%M:%S) DONE seed $S ===" >> "$LOG"
