#!/usr/bin/env bash
# The core-population event sets: four kinds per catalog realisation, beside the
# signed-off legacy sets, never over them.
#
#   ./make_corepop_events.sh [SEED ...]          (default: 100 101 102 103 105)
#
# kind      file                                   flags (as the legacy set of that kind)
# mixed     events_corepop.h5                      record sub-seed S*1000+3, f_AGN 0.30
# pureagn   events_pureagn_corepop.h5              --f_agn 1.0 --seed_events S*1000+9
# puregal   events_puregal_corepop.h5              --f_agn 0.0 --seed_events S*1000+8
# marked    events_marked_dmu0p10_dmuG5_corepop.h5 --dmu_chi_agn 0.10 --dmu_G_agn 5, S*1000+3
#
# every one with --population_code core.  The sub-seeds are those of the legacy file of
# the same kind, so the main RNG stream -- hosts, primary masses, spins, measurement
# noise -- is the legacy draw's proposal by proposal (see README.md, "Correspondence").
#
# Not touched: --overwrite is never passed (an existing file is skipped), the suffix
# suppresses the META.json merge, and the validation runs in its events-only mode
# (--events_suffix), which writes validation/validation_events<SFX>.json and nothing
# else.  numpy's AVX-512 loops are switched off so the floating point matches the
# record's (bitcheck_default.sh explains why).
set -uo pipefail
DATA=/hildafs/projects/phy230014p/magana/gws-agn/working/data
GEN=$DATA/generate_dataset.py
OUTROOT=${OUTROOT:-/hildafs/projects/phy230054p/magana/gws-agn-data-v3}
PIN=/hildafs/projects/phy230014p/magana/src/darksirens-2b86a2d
PY=/hildafs/home/magana/tmp_ondemand_hildafs_phy230014p_symlink/magana/.conda/envs/jax/bin/python
HERE=$(cd "$(dirname "$0")/.." && pwd)
export PYTHONPATH=$PIN JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_KNL AVX512_KNM AVX512_SKX AVX512_CLX AVX512_CNL AVX512_ICL"
[ "$(git -C $PIN rev-parse HEAD)" = 2b86a2d8d48fdb5173f0ba259b8996104187dd2d ] || { echo "darksirens pin wrong"; exit 1; }
[ -z "$(git -C $PIN status --porcelain)" ] || { echo "darksirens pin dirty"; exit 1; }

SEEDS=("$@")
[ ${#SEEDS[@]} -eq 0 ] && SEEDS=(100 101 102 103 105)

one_seed () {
  local S=$1
  for KIND in mixed pureagn puregal marked; do
    case $KIND in
      mixed)   SFX=_corepop; EXTRA="";;
      pureagn) SFX=_pureagn_corepop
               EXTRA="--f_agn 1.0 --seed_events $(( S * 1000 + 9 )) --n_events 1000 --nsamp 2000";;
      puregal) SFX=_puregal_corepop
               EXTRA="--f_agn 0.0 --seed_events $(( S * 1000 + 8 )) --n_events 1000 --nsamp 2000";;
      marked)  SFX=_marked_dmu0p10_dmuG5_corepop; EXTRA="--dmu_chi_agn 0.10 --dmu_G_agn 5";;
    esac
    OUT=$OUTROOT/seed$S/events/events${SFX}.h5
    LOG=$HERE/logs/gen_s${S}${SFX}.log
    if [ -s "$OUT" ]; then
      echo "[$(date -u +%H:%M:%S)] skip seed $S $KIND (exists: $OUT)"
    else
      echo "[$(date -u +%H:%M:%S)] seed $S $KIND -> $OUT"
      /usr/bin/time -v $PY -u "$GEN" --seed "$S" --stage events --outroot "$OUTROOT" \
          --darksirens "$PIN" --population_code core $EXTRA --events_suffix "$SFX" \
          > "$LOG" 2>&1 || { echo "GENERATION FAILED seed $S $KIND (see $LOG)"; continue; }
    fi
    VLOG=$HERE/logs/val_s${S}${SFX}.log
    /usr/bin/time -v $PY -u "$GEN" --seed "$S" --stage validation --outroot "$OUTROOT" \
        --darksirens "$PIN" --events_suffix "$SFX" > "$VLOG" 2>&1 \
      && echo "[$(date -u +%H:%M:%S)] seed $S $KIND validation PASS" \
      || echo "[$(date -u +%H:%M:%S)] seed $S $KIND validation FAILED: $(grep -o 'VALIDATION FAILED: .*\]' "$VLOG")"
  done
}

for S in "${SEEDS[@]}"; do one_seed "$S" & done
wait
echo "CORE-POPULATION EVENT GENERATION DONE: ${SEEDS[*]}"
