#!/bin/bash
# Post-fix attribution chain: the term-by-term score residual r on the REGENERATED
# events, then the exact host-galaxy sky oracle.  Tags carry `_postfix` so the
# pre-fix attribution products ATTRIBUTION.md cites are not overwritten.
set -euo pipefail
cd "$(dirname "$0")/.."
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export DARKSIRENS_SRC=${DARKSIRENS_SRC:-/hildafs/projects/phy230014p/magana/src/darksirens}

echo "[$(date -u +%H:%M:%S)] attr_score_terms agn (post-fix)"
python scripts/attr_score_terms.py --tracer agn --tag agn_s100_postfix \
    --pe_batch_events 70 --sel_batch 100000 > logs/attr_terms_agn_postfix.log 2>&1
echo "[$(date -u +%H:%M:%S)] attr_score_terms gal (post-fix)"
python scripts/attr_score_terms.py --tracer gal --kde_window 4096 --tag gal_s100_postfix \
    --pe_batch_events 25 --sel_batch 50000 > logs/attr_terms_gal_postfix.log 2>&1
echo "[$(date -u +%H:%M:%S)] sky oracle (production)"
./scripts/run_sky_oracle.sh prod
echo "[$(date -u +%H:%M:%S)] POSTFIX ATTR DONE"
