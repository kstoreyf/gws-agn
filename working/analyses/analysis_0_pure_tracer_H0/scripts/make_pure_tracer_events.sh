#!/usr/bin/env bash
# analysis_0_pure_tracer_H0 -- the ten independent single-tracer event draws.
#
#   ./scripts/make_pure_tracer_events.sh [SEED ...]      (default: all five)
#
# For each catalog realisation S, TWO new 1000-event sets are drawn on the SAME
# signed-off v3 catalogs and surveys:
#
#   events_puregal.h5   --f_agn 0.0  --seed_events S*1000+8
#   events_pureagn.h5   --f_agn 1.0  --seed_events S*1000+9
#
# WHY NEW DRAWS.  analysis_1's `events_{gal,agn}_hosted.h5` are SUBSETS of the
# record's 1000 mixture events (705 GAL / 295 AGN), so the two tracers are neither
# independent of one another nor of equal size, and neither is a 1000-event
# measurement.  Analysis 0 needs the constraining power of each tracer AT EQUAL N,
# from event noise independent of the record, so each set is a full independent
# draw of N = 1000 with its own RNG stream.
#
# THE SUB-SEEDS.  generate_dataset.sub_seeds() spends offsets 1-7 on the record
# (glass_field 1, magnitudes 2, events 3, injections_targeted 4, injections_popuni 5,
# validation 6, photoz 7).  Offsets 8 and 9 are unused by the generator, so the two
# draws here are independent of every recorded stream and of each other.
#
# WHAT IS NOT TOUCHED.  --events_suffix writes events<SFX>.h5 beside events.h5 and
# suppresses the META.json merge, and --overwrite is deliberately NOT passed, so the
# signed-off dataset cannot be modified.  Catalogs, surveys and both injection lanes
# are reused exactly as they are on disk.
set -euo pipefail

GEN=/hildafs/projects/phy230014p/magana/gws-agn/working/data/generate_dataset.py
OUTROOT=${OUTROOT:-/hildafs/projects/phy220048p/magana/gws-agn-data-v3}
HERE="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$HERE/logs"

SEEDS=("$@")
[ ${#SEEDS[@]} -eq 0 ] && SEEDS=(100 101 102 103 105)

for S in "${SEEDS[@]}"; do
  for SPEC in "puregal 0.0 8" "pureagn 1.0 9"; do
    set -- $SPEC
    SFX=$1; FAGN=$2; OFF=$3
    SEED_EV=$(( S * 1000 + OFF ))
    OUT="$OUTROOT/seed$S/events/events_${SFX}.h5"
    if [ -s "$OUT" ]; then
      echo "[$(date -u +%H:%M:%S)] skip seed $S $SFX (exists: $OUT)"
      continue
    fi
    echo "[$(date -u +%H:%M:%S)] seed $S  $SFX  f_agn=$FAGN  seed_events=$SEED_EV"
    python -u "$GEN" --seed "$S" --stage events --outroot "$OUTROOT" \
        --f_agn "$FAGN" --seed_events "$SEED_EV" \
        --n_events 1000 --nsamp 2000 --events_suffix "_${SFX}" \
        > "$HERE/logs/gen_s${S}_${SFX}.log" 2>&1
    echo "[$(date -u +%H:%M:%S)] wrote $OUT"
  done
done
echo "PURE-TRACER EVENT GENERATION DONE: ${SEEDS[*]}"
