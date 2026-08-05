#!/usr/bin/env bash
# One scan of the analysis-3 campaign.  Every scan here is the same K = 2
# configuration (scripts/env.sh); this script chooses the survey level and the axis.
#
#   KIND=joint  SEED=100 LEVEL=m21 LANE=targeted            ./scripts/run_scan.sh
#   KIND=joint  SEED=100 LEVEL=m21 CHUNK=0 NCHUNK=4         ./scripts/run_scan.sh
#   KIND=fscan  SEED=100 LEVEL=complete                     ./scripts/run_scan.sh
#   KIND=h0scan SEED=100 LEVEL=complete                     ./scripts/run_scan.sh
#   KIND=fnull  SEED=100 LEVEL=m18                          ./scripts/run_scan.sh
#
# The joint grid is H0 [50, 100] x 201  X  f [0, 1] x 41 = 8241 evaluations, the
# same grid analysis 2 used, so the ladder's rungs stay on one axis with its
# complete-catalog rung 0.  With CHUNK/NCHUNK unset the whole grid runs in one
# task; set them to split the H0 axis across GPUs (chunk boundaries are exact
# subsets of linspace(50, 100, 201), step 0.25, and scripts/merge_joint.py
# asserts the reassembled axis reproduces it).
set -euo pipefail
cd "$(dirname "$0")/.."
. scripts/env.sh
mkdir -p logs results results/chunks data_derived

KIND=${KIND:?set KIND=joint|fscan|h0scan|fnull}
SEED=${SEED:?set SEED}
LEVEL=${LEVEL:?set LEVEL=complete|m21|m20|m19|m18}
LANE=${LANE:-targeted}
SUF=""; [ "$LANE" = "popuni" ] && SUF="_popuni"

EV=${DATA_ROOT}/seed${SEED}/events/events.h5
if [ "$KIND" = "fnull" ]; then
  EV=data_derived/events_skyshuffled_s${SEED}.h5
  if [ ! -f "$EV" ]; then
    python scripts/shuffle_event_sky.py \
      --in_path "${DATA_ROOT}/seed${SEED}/events/events.h5" \
      --out_path "$EV" --seed 90210 | tee -a logs/shuffle_s${SEED}.log
  fi
fi

COMMON=$(ds_common "$SEED" "$LANE" "$EV" "$LEVEL")

case "$KIND" in
  joint)
    if [ -n "${CHUNK:-}" ]; then
      NCHUNK=${NCHUNK:?set NCHUNK with CHUNK}
      read -r LO HI NPT < <(python - "$CHUNK" "$NCHUNK" <<'PY'
import sys
import numpy as np
c, nc = int(sys.argv[1]), int(sys.argv[2])
full = np.linspace(50.0, 100.0, 201)
edges = np.linspace(0, 201, nc + 1).round().astype(int)
i0, i1 = edges[c], edges[c + 1]
sub = full[i0:i1]
print(f"{sub[0]!r} {sub[-1]!r} {sub.size}")
PY
)
      TAG=joint_${LEVEL}_s${SEED}${SUF}_c${CHUNK}
      OUT=results/chunks
    else
      LO=50.0; HI=100.0; NPT=201
      TAG=joint_${LEVEL}_s${SEED}${SUF}
      OUT=results
    fi
    python -u scripts/scan_h0f.py $COMMON \
      --scan joint --h0_grid "$LO" "$HI" "$NPT" --f_grid 0.0 1.0 41 \
      --outdir "$OUT" --out_tag "$TAG" \
      > "logs/${TAG}.log" 2>&1
    ;;
  fscan|fnull)
    TAG=fscan_${LEVEL}_s${SEED}${SUF}
    [ "$KIND" = "fnull" ] && TAG=fscan_null_${LEVEL}_s${SEED}${SUF}
    python -u scripts/scan_h0f.py $COMMON \
      --scan f --f_grid 0.0 1.0 101 --h0_fixed "$H0_TRUE" \
      --outdir results --out_tag "$TAG" \
      > "logs/${TAG}.log" 2>&1
    ;;
  h0scan)
    TAG=h0scan_${LEVEL}_s${SEED}${SUF}
    python -u scripts/scan_h0f.py $COMMON \
      --scan h0 --h0_grid 50.0 100.0 201 --f_fixed "$F_TRUE" \
      --outdir results --out_tag "$TAG" \
      > "logs/${TAG}.log" 2>&1
    ;;
  *) echo "unknown KIND=$KIND" >&2; exit 2 ;;
esac
echo "[$(date -u +%H:%M:%S)] done KIND=$KIND SEED=$SEED LEVEL=$LEVEL LANE=$LANE ${CHUNK:+CHUNK=$CHUNK}"
