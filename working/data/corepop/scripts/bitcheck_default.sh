#!/usr/bin/env bash
# Bit-identity of the DEFAULT path (--population_code legacy) of generate_dataset.py.
#
#   ./bitcheck_default.sh run pre           <generator.py>  # the generator as it was (git HEAD 85f2934)
#   ./bitcheck_default.sh run post          <generator.py>  # the generator with --population_code
#   ./bitcheck_default.sh run post_noavx512 <generator.py>  # the same, numpy AVX-512 dispatch off
#   ./bitcheck_default.sh compare
#
# Four event draws exercise every code path the change touches: the record mixture
# (seeds 100 and 103, record sub-seed S*1000+3), the all-AGN set (f_agn 1, sub-seed
# S*1000+9) and the host-dependent set (dmu_chi_agn 0.10, dmu_G_agn 5: the child mass
# stream).  Each is written into a scratch outroot whose catalogs/ is a symlink to the
# signed-off catalogs, so nothing in the dataset is written.  The comparison is
# analysis_0's dataset-level gate (every HDF5 dataset byte-identical):
#
#   pre vs post                    the change itself, same machine
#   record vs post_noavx512        the signed-off files on disk
#
# WHY AVX-512 OFF.  On miko (Xeon Gold 6526Y) numpy 1.26 dispatches its transcendental
# loops to AVX-512, which rounds differently in the last bit from the AVX2 loops the
# record was generated with: today's unmodified generator then reproduces the record
# to 1 ulp (11 of 1000 primary masses), not bitwise.  NPY_DISABLE_CPU_FEATURES removes
# exactly that, and the unmodified generator reproduces the record bitwise
# (results/bitcheck_record_vs_HEAD_noavx512_mixed_s100.json).
set -euo pipefail
SCR=/hildafs/projects/phy230054p/magana/gws-agn-data-v3-scratch-corepop
DATA=/hildafs/projects/phy230054p/magana/gws-agn-data-v3
PIN=/hildafs/projects/phy230014p/magana/src/darksirens-2b86a2d
PY=/hildafs/home/magana/tmp_ondemand_hildafs_phy230014p_symlink/magana/.conda/envs/jax/bin/python
BITCHECK=/hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_0_pure_tracer_H0/scripts/bitcheck_events.py
HERE=$(cd "$(dirname "$0")/.." && pwd)
NOAVX512="AVX512F AVX512CD AVX512_KNL AVX512_KNM AVX512_SKX AVX512_CLX AVX512_CNL AVX512_ICL"
export PYTHONPATH=$PIN JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4

# name | seed | extra generator flags
CONFIGS=(
  "mixed|100|"
  "mixed|103|"
  "pureagn|100|--f_agn 1.0 --seed_events 100009 --n_events 1000 --nsamp 2000 --events_suffix _pureagn"
  "marked|100|--dmu_chi_agn 0.10 --dmu_G_agn 5 --events_suffix _marked_dmu0p10_dmuG5"
)
fname () { case $1 in mixed) echo events.h5;; pureagn) echo events_pureagn.h5;;
           marked) echo events_marked_dmu0p10_dmuG5.h5;; esac; }

case ${1:?run|compare} in
run)
  SIDE=${2:?pre|post|post_noavx512}; GEN=${3:?generator}
  [ "$(git -C $PIN rev-parse HEAD)" = 2b86a2d8d48fdb5173f0ba259b8996104187dd2d ] || { echo "pin wrong"; exit 1; }
  [ "$SIDE" = post_noavx512 ] && export NPY_DISABLE_CPU_FEATURES="$NOAVX512"
  pids=()
  for c in "${CONFIGS[@]}"; do
    IFS='|' read -r NAME S EXTRA <<< "$c"
    mkdir -p "$SCR/$SIDE/seed$S/events"
    ln -sfn "$DATA/seed$S/catalogs" "$SCR/$SIDE/seed$S/catalogs"
    LOG=$HERE/logs/bitcheck_${SIDE}_${NAME}_s${S}.log
    ( /usr/bin/time -v $PY -u "$GEN" --seed "$S" --stage events --outroot "$SCR/$SIDE" \
        --darksirens "$PIN" $EXTRA > "$LOG" 2>&1; echo "exit $?" >> "$LOG" ) &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p"; done
  echo "run $SIDE done"
  ;;
compare)
  for c in "${CONFIGS[@]}"; do
    IFS='|' read -r NAME S EXTRA <<< "$c"
    F=$(fname "$NAME")
    $PY "$BITCHECK" "$SCR/pre/seed$S/events/$F" "$SCR/post/seed$S/events/$F" \
        --out "$HERE/results/bitcheck_pre_vs_post_${NAME}_s${S}.json" > /dev/null && r1=PASS || r1=FAIL
    $PY "$BITCHECK" "$DATA/seed$S/events/$F" "$SCR/post_noavx512/seed$S/events/$F" \
        --out "$HERE/results/bitcheck_record_vs_post_noavx512_${NAME}_s${S}.json" > /dev/null && r2=PASS || r2=FAIL
    echo "$NAME seed $S ($F): pre-vs-post $r1   record-vs-post_noavx512 $r2"
  done
  ;;
esac
