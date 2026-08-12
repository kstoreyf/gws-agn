#!/usr/bin/env bash
# Stage A -- the cheap gate.  Rebuild the analysis-5 K=2 closure against
# darksirens master and evaluate it at the stored analysis-3 grid cells.
#
# On 2b86a2d these agreed to 1.8e-12 (float64 round-off).  If master still
# gives ~1e-12, the likelihood is unchanged for this configuration and the 4D
# fit cannot move; if it does not, this prints exactly how far it moved and
# stage B measures what that does to the posterior.
#
#   ./scripts/run_probe.sh            # m18 (default)
#   RUNG=m20 ./scripts/run_probe.sh
set -euo pipefail
cd "$(dirname "$0")/.."
. scripts/env.sh
mkdir -p logs results

CMODE=${CMODE:-aggregate}
TAG=probe_${RUNG}_${CMODE}_s${SEED}
python -u scripts/sample_4d.py $(ds_4d_common) \
  --sampler dynesty --probe_only --wiring_nonfatal --c_mode "$CMODE" \
  --outdir results --out_tag "$TAG" \
  2>&1 | tee "logs/${TAG}.log"
