#!/usr/bin/env bash
# One cell of the relative-completeness surface: the analysis-3 joint (H0, f_AGN)
# grid with the GAL and AGN survey depths set independently.
#
#   SEED=100 GLEV=m20 ALEV=m18 ./scripts/run_scan.sh
#
# Grid: H0 [50,100] x 201 * f [0,1] x 41 = 8241 evaluations, analysis 2/3/4's
# grid unchanged, so every cell shares their axes and the diagonal cells already
# on disk in analysis 3 are directly comparable.
set -euo pipefail
cd "$(dirname "$0")/.."
. scripts/env.sh
mkdir -p logs results

SEED=${SEED:?set SEED}
GLEV=${GLEV:?set GLEV=complete|m21|m20|m19|m18}
ALEV=${ALEV:?set ALEV=complete|m21|m20|m19|m18}
LANE=${LANE:-targeted}

EV=${DATA_ROOT}/seed${SEED}/events/events.h5
COMMON=$(ds_common "$SEED" "$LANE" "$EV" "$GLEV" "$ALEV")

TAG=joint_g${GLEV}_a${ALEV}_s${SEED}
python -u scripts/scan_h0f.py $COMMON \
  --scan joint --h0_grid 50.0 100.0 201 --f_grid 0.0 1.0 41 \
  --outdir results --out_tag "$TAG" \
  > "logs/${TAG}.log" 2>&1
