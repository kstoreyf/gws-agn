#!/usr/bin/env bash
# One scan of the analysis-4 campaign: the analysis-3 joint grid with the AGN
# density anchor swept (or, for the oracle probe, the AGN survey completed).
#
#   SEED=100 GLEV=m21 ALEV=m21 ARM=a09 N0C2=-5.045757 ./scripts/run_scan.sh
#   SEED=100 GLEV=m18 ALEV=complete ARM=oracle N0C2=-5.0 ./scripts/run_scan.sh
#
# The grid is H0 [50, 100] x 201  X  f [0, 1] x 41 = 8241 evaluations —
# analysis 2 and 3's grid, unchanged, so every arm shares their axes.
set -euo pipefail
cd "$(dirname "$0")/.."
. scripts/env.sh
mkdir -p logs results data_derived

SEED=${SEED:?set SEED}
GLEV=${GLEV:?set GLEV=m21|m20|m19|m18|complete}
ALEV=${ALEV:?set ALEV=m21|m20|m19|m18|complete}
ARM=${ARM:?set ARM tag, e.g. a09|a11|oracle}
N0C2=${N0C2:?set N0C2 = log10n0_c2 for this arm}
LANE=${LANE:-targeted}

EV=${DATA_ROOT}/seed${SEED}/events/events.h5
COMMON=$(ds_common "$SEED" "$LANE" "$EV" "$GLEV" "$ALEV" "$N0C2")

TAG=joint_${GLEV}_${ARM}_s${SEED}
python -u scripts/scan_h0f.py $COMMON \
  --scan joint --h0_grid 50.0 100.0 201 --f_grid 0.0 1.0 41 \
  --outdir results --out_tag "$TAG" \
  > "logs/${TAG}.log" 2>&1
