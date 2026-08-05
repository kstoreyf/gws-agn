#!/usr/bin/env bash
# One scan of the analysis-2 campaign.  Every scan in this directory is the same
# K = 2 configuration (scripts/env.sh); this script only chooses the axis.
#
#   KIND=joint  SEED=100 LANE=targeted CHUNK=0 NCHUNK=8  ./scripts/run_scan.sh
#   KIND=fscan  SEED=100 LANE=targeted                   ./scripts/run_scan.sh
#   KIND=h0scan SEED=100 LANE=targeted                   ./scripts/run_scan.sh
#   KIND=fnull  SEED=100 LANE=targeted                   ./scripts/run_scan.sh
#
# The joint grid is H0 [50, 100] x 201  X  f [0, 1] x 41 = 8241 evaluations.  A K=2
# evaluation on the COMPLETE GAL block costs seconds, so the H0 axis is split into
# NCHUNK contiguous chunks that run on separate GPUs and are stitched by
# scripts/merge_joint.py.  Chunk boundaries are exact subsets of the target
# linspace (step 0.25) and the merge asserts the reassembled axis reproduces it.
set -euo pipefail
cd "$(dirname "$0")/.."
. scripts/env.sh
mkdir -p logs results results/chunks data_derived

KIND=${KIND:?set KIND=joint|fscan|h0scan|fnull}
SEED=${SEED:?set SEED}
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

COMMON=$(ds_common "$SEED" "$LANE" "$EV")

case "$KIND" in
  joint)
    CHUNK=${CHUNK:?set CHUNK}
    NCHUNK=${NCHUNK:?set NCHUNK}
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
    TAG=joint_s${SEED}${SUF}_c${CHUNK}
    python -u scripts/scan_h0f.py $COMMON \
      --scan joint --h0_grid "$LO" "$HI" "$NPT" --f_grid 0.0 1.0 41 \
      --outdir results/chunks --out_tag "$TAG" \
      > "logs/${TAG}.log" 2>&1
    ;;
  fscan|fnull)
    TAG=fscan_s${SEED}${SUF}
    [ "$KIND" = "fnull" ] && TAG=fscan_null_s${SEED}${SUF}
    python -u scripts/scan_h0f.py $COMMON \
      --scan f --f_grid 0.0 1.0 101 --h0_fixed "$H0_TRUE" \
      --outdir results --out_tag "$TAG" \
      > "logs/${TAG}.log" 2>&1
    ;;
  h0scan)
    TAG=h0scan_s${SEED}${SUF}
    python -u scripts/scan_h0f.py $COMMON \
      --scan h0 --h0_grid 50.0 100.0 201 --f_fixed "$F_TRUE" \
      --outdir results --out_tag "$TAG" \
      > "logs/${TAG}.log" 2>&1
    ;;
  *) echo "unknown KIND=$KIND" >&2; exit 2 ;;
esac
echo "[$(date -u +%H:%M:%S)] done KIND=$KIND SEED=$SEED LANE=$LANE ${CHUNK:+CHUNK=$CHUNK}"
