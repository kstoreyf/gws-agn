#!/usr/bin/env bash
# The four core-population event sets for ONE further catalog realisation whose
# catalogs/ was pruned by the 100-seed campaign (seeds 106-199).  README.md, "Further
# seeds", is the plan this follows.
#
#   sbatch --array=0-4 submit_more_seeds.sbatch 106      (task i -> seed 106 + i)
#   ./more_seeds_job.sh SEED                              (one seed, by hand)
#
# 1. --stage catalogs into a SCRATCH outroot (the stage merges into META.json, so it
#    never runs against the dataset), in the environment the campaign used (RM, numpy
#    dispatch untouched, 8 threads, darksirens pin on PYTHONPATH).
# 2. GATE: the legacy events.h5, events_pureagn.h5 and events_puregal.h5 are
#    regenerated from those catalogs into scratch and compared with the files on disk
#    dataset by dataset (analysis_0's bitcheck_events.py).  All three byte-identical
#    proves the catalogs and the floating point are the ones the seed was made with.
#    The default numpy dispatch is tried first, then numpy with AVX-512 off; the first
#    environment that passes is used for step 3.  No pass -> nothing is written.
# 3. The four core sets (flags and sub-seeds of make_corepop_events.sh) written into
#    the seed's events/ through a scratch outroot whose events/ and validation/ are
#    symlinks to the seed's own directories and whose catalogs/ is the scratch build.
#    --overwrite is never passed; a suffixed events stage never touches META.json.
# 4. Events-only validation of each set (validation/validation_events<SFX>.json).
# 5. The scratch tree of this seed (catalogs included) is deleted.
set -uo pipefail
S=${1:?usage: more_seeds_job.sh SEED}
DATA=/hildafs/projects/phy230014p/magana/gws-agn/working/data
GEN=$DATA/generate_dataset.py
REAL=/hildafs/projects/phy230054p/magana/gws-agn-data-v3
SCR=/hildafs/projects/phy230054p/magana/gws-agn-data-v3-scratch-corepop-more
PIN=/hildafs/projects/phy230014p/magana/src/darksirens-2b86a2d
PY=/hildafs/home/magana/tmp_ondemand_hildafs_phy230014p_symlink/magana/.conda/envs/jax/bin/python
BITCHECK=/hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_0_pure_tracer_H0/scripts/bitcheck_events.py
HERE=$DATA/corepop
LOGD=$HERE/logs/more_seeds
RESD=$HERE/results/more_seeds
mkdir -p "$LOGD" "$RESD"
LOG=$LOGD/job_s$S.log
: > "$LOG"
say () { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }
NOAVX512="AVX512F AVX512CD AVX512_KNL AVX512_KNM AVX512_SKX AVX512_CLX AVX512_CNL AVX512_ICL"
export PYTHONPATH=$PIN JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
unset NPY_DISABLE_CPU_FEATURES
[ "$(git -C $PIN rev-parse HEAD)" = 2b86a2d8d48fdb5173f0ba259b8996104187dd2d ] || { say "darksirens pin wrong"; exit 1; }
[ -z "$(git -C $PIN status --porcelain)" ] || { say "darksirens pin dirty"; exit 1; }
[ -d "$REAL/seed$S/events" ] && [ -d "$REAL/seed$S/surveys" ] && [ -d "$REAL/seed$S/injections" ] \
  || { say "seed $S incomplete on disk"; exit 1; }
[ -e "$REAL/seed$S/catalogs" ] && { say "seed $S already has catalogs/; this job is for pruned seeds"; exit 1; }

say "host $(hostname) job ${SLURM_JOB_ID:-none} seed $S; cpu: $(lscpu | grep 'Model name' | sed 's/.*: *//'); avx512f: $(grep -c -m1 avx512f /proc/cpuinfo)"
T0=$(date +%s)

# ---- 1. catalogs into scratch ------------------------------------------------------
CAT=$SCR/cat
mkdir -p "$CAT"
say "catalogs -> $CAT/seed$S"
/usr/bin/time -v $PY -u "$GEN" --seed "$S" --stage catalogs --outroot "$CAT" \
    > "$LOGD/catalogs_s$S.log" 2>&1 || { say "CATALOGS FAILED"; exit 1; }
T1=$(date +%s)
say "catalogs done in $((T1 - T0)) s ($(du -sh $CAT/seed$S/catalogs | cut -f1))"

# ---- 2. the gate --------------------------------------------------------------------
gate () {   # $1 = env tag (default|noavx512); returns 0 when all three legacy files match
  local TAG=$1 G=$SCR/gate_$1 ok=0
  mkdir -p "$G/seed$S/events"
  ln -sfn "$CAT/seed$S/catalogs" "$G/seed$S/catalogs"
  for SPEC in "events.h5|" \
              "events_pureagn.h5|--f_agn 1.0 --seed_events $((S*1000+9)) --n_events 1000 --nsamp 2000 --events_suffix _pureagn" \
              "events_puregal.h5|--f_agn 0.0 --seed_events $((S*1000+8)) --n_events 1000 --nsamp 2000 --events_suffix _puregal"; do
    IFS='|' read -r F EXTRA <<< "$SPEC"
    ( [ "$TAG" = noavx512 ] && export NPY_DISABLE_CPU_FEATURES="$NOAVX512"
      $PY -u "$GEN" --seed "$S" --stage events --outroot "$G" --darksirens "$PIN" $EXTRA \
        > "$LOGD/gate_${TAG}_s${S}_${F%.h5}.log" 2>&1 ) || { say "gate $TAG: regeneration of $F failed"; ok=1; continue; }
    if $PY "$BITCHECK" "$REAL/seed$S/events/$F" "$G/seed$S/events/$F" \
        --out "$RESD/gate_${TAG}_s${S}_${F%.h5}.json" > /dev/null; then
      say "gate $TAG $F: PASS (every dataset byte-identical)"
    else
      say "gate $TAG $F: FAIL"; ok=1
    fi
  done
  return $ok
}
ENVTAG=""
for TAG in default noavx512; do
  if gate $TAG; then ENVTAG=$TAG; break; fi
done
T2=$(date +%s)
if [ -z "$ENVTAG" ]; then
  say "GATE FAILED in both environments; nothing written for seed $S (scratch kept for inspection)"
  echo "{\"seed\": $S, \"gate\": \"FAIL\"}" > "$RESD/summary_s$S.json"
  exit 2
fi
say "gate PASSED with numpy dispatch '$ENVTAG' in $((T2 - T1)) s"
[ "$ENVTAG" = noavx512 ] && export NPY_DISABLE_CPU_FEATURES="$NOAVX512"

# ---- 3./4. the four core sets and their events-only validation -----------------------
GR=$SCR/gen
mkdir -p "$GR/seed$S"
ln -sfn "$CAT/seed$S/catalogs" "$GR/seed$S/catalogs"
ln -sfn "$REAL/seed$S/events" "$GR/seed$S/events"
ln -sfn "$REAL/seed$S/validation" "$GR/seed$S/validation"
NFAIL=0
declare -A GSEC VSEC VRES
for KIND in mixed pureagn puregal marked; do
  case $KIND in
    mixed)   SFX=_corepop; EXTRA="";;
    pureagn) SFX=_pureagn_corepop
             EXTRA="--f_agn 1.0 --seed_events $(( S * 1000 + 9 )) --n_events 1000 --nsamp 2000";;
    puregal) SFX=_puregal_corepop
             EXTRA="--f_agn 0.0 --seed_events $(( S * 1000 + 8 )) --n_events 1000 --nsamp 2000";;
    marked)  SFX=_marked_dmu0p10_dmuG5_corepop; EXTRA="--dmu_chi_agn 0.10 --dmu_G_agn 5";;
  esac
  OUT=$REAL/seed$S/events/events${SFX}.h5
  if [ -e "$OUT" ]; then say "skip $KIND: $OUT exists"; continue; fi
  ta=$(date +%s)
  /usr/bin/time -v $PY -u "$GEN" --seed "$S" --stage events --outroot "$GR" --darksirens "$PIN" \
      --population_code core $EXTRA --events_suffix "$SFX" > "$LOGD/gen_s${S}${SFX}.log" 2>&1 \
    || { say "GENERATION FAILED $KIND"; NFAIL=$((NFAIL+1)); continue; }
  tb=$(date +%s)
  /usr/bin/time -v $PY -u "$GEN" --seed "$S" --stage validation --outroot "$GR" --darksirens "$PIN" \
      --events_suffix "$SFX" > "$LOGD/val_s${S}${SFX}.log" 2>&1 && V=PASS || { V=FAIL; NFAIL=$((NFAIL+1)); }
  tc=$(date +%s)
  GSEC[$KIND]=$((tb - ta)); VSEC[$KIND]=$((tc - tb)); VRES[$KIND]=$V
  say "$KIND -> $(basename $OUT): generated in $((tb - ta)) s, validation $V in $((tc - tb)) s"
done
T3=$(date +%s)

# ---- 5. delete this seed's scratch tree ------------------------------------------------
rm -rf "$GR/seed$S" "$SCR/gate_default/seed$S" "$SCR/gate_noavx512/seed$S" "$CAT/seed$S"
say "scratch for seed $S deleted"
{
  echo "{\"seed\": $S, \"host\": \"$(hostname)\", \"gate\": \"PASS\", \"numpy_dispatch\": \"$ENVTAG\","
  echo " \"seconds\": {\"catalogs\": $((T1 - T0)), \"gate\": $((T2 - T1)), \"four_sets_and_validation\": $((T3 - T2)), \"total\": $((T3 - T0))},"
  echo " \"sets\": {"
  first=1
  for KIND in mixed pureagn puregal marked; do
    [ -n "${VRES[$KIND]:-}" ] || continue
    [ $first = 1 ] || echo ","
    first=0
    printf '  "%s": {"generate_s": %s, "validate_s": %s, "validation": "%s"}' "$KIND" "${GSEC[$KIND]}" "${VSEC[$KIND]}" "${VRES[$KIND]}"
  done
  echo ""
  echo " }, \"n_failed\": $NFAIL}"
} > "$RESD/summary_s$S.json"
say "seed $S done: $NFAIL failures, $((T3 - T0)) s"
[ $NFAIL -eq 0 ]
