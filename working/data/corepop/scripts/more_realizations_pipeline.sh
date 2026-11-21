#!/usr/bin/env bash
# Third batch (2026-11-21, README "Third batch"): the pipeline that drew seeds 111-120, copied from the scratch copy that ran
# (seed_pipeline.sh, md5 e976bf0d23b689069758a7f07ef07401) with one change: the scratch directory is the variable SCR (it was a
# fixed /tmp path). It ran on miko (AVX-512, 8 threads per process): seed 111 first, then 112-114, 115-117 and 118-120 in
# three parallel streams (each `timeout 5400 run_all.sh S1 S2 S3`), one seed after another per stream, by this loop (run_all.sh):
#     for S in "$@"; do $SCR/scripts/seed_pipeline.sh $S || exit 1; done
# Usage: SCR=<scratch directory> more_realizations_pipeline.sh SEED   -> $SCR/results/sSEED/summary.json, $SCR/logs/sSEED/
# One further realization (seed S, catalogs pruned): the production event set of stage 1 (mixed, f_AGN 0.30, core
# population, suffix _corepop), following working/data/corepop/README.md "Further seeds" / "Second batch"
# (scripts/more_seeds_job.sh) with these differences, all for this node (miko, AVX-512):
#   * catalogs are kept in data/seedS/catalogs/ (as for 106-110, A16 README stage X (f)); the stage runs against an
#     outroot data/seedS/catalogs_regen_tmp (it merges into META.json), its META.json kept as catalogs/regen_META.json;
#   * glass venv (numpy 2.4): NPY_DISABLE_CPU_FEATURES="X86_V4 AVX512_ICL AVX512_SPR";
#     jax env (numpy 1.26, events stage, survey tools): the numpy-1.26 AVX-512 group (more_seeds_job.sh NOAVX512);
#   * only the mixed kind; plus the AGN nside-32 survey file rebuilt and compared (make_survey_nside.py --reproduce)
#     and the nside-64 survey files (make_survey_nside.py --nside 64, checks).
# 8 threads (the original generation's OMP_NUM_THREADS), one process at a time.
set -uo pipefail
S=${1:?usage: seed_pipeline.sh SEED}
W=/hildafs/projects/phy230014p/magana/gws-agn/working
DATA=$W/data
GEN=$DATA/generate_dataset.py
PIN=/hildafs/projects/phy230014p/magana/src/darksirens-2b86a2d
PY=/hildafs/home/magana/tmp_ondemand_hildafs_phy230014p_symlink/magana/.conda/envs/jax/bin/python
BITCHECK=$W/analyses/analysis_0_pure_tracer_H0/scripts/bitcheck_events.py
SURV=$W/analyses/analysis_16_production_runs/scripts/make_survey_nside.py
SCR=${SCR:?set SCR to a scratch directory (logs, results, gate and generation trees go there)}
LOGD=$SCR/logs/s$S; RESD=$SCR/results/s$S
mkdir -p "$LOGD" "$RESD"
LOG=$LOGD/pipeline.log; : > "$LOG"
say () { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }
NOAVX512_126="AVX512F AVX512CD AVX512_KNL AVX512_KNM AVX512_SKX AVX512_CLX AVX512_CNL AVX512_ICL"
NOAVX512_24="X86_V4 AVX512_ICL AVX512_SPR"
export PYTHONPATH=$PIN JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
SD=$DATA/seed$S
[ "$(git -C $PIN rev-parse HEAD)" = 2b86a2d8d48fdb5173f0ba259b8996104187dd2d ] || { say "darksirens pin wrong"; exit 1; }
[ -z "$(git -C $PIN status --porcelain)" ] || { say "darksirens pin dirty"; exit 1; }
[ -d "$SD/events" ] && [ -d "$SD/surveys" ] && [ -f "$SD/injections/injections_targeted.h5" ] || { say "seed $S incomplete"; exit 1; }
T0=$(date +%s)
md5_before=$(md5sum "$SD/META.json" | cut -d' ' -f1)

# ---- 1. catalogs -------------------------------------------------------------------------------------------------
if [ -e "$SD/catalogs/catalog_gal_complete.h5" ] && [ -e "$SD/catalogs/regen_META.json" ]; then
  say "catalogs present (from an earlier pass), skipped"
else
  [ -e "$SD/catalogs" ] && { say "seed $S has a partial catalogs/; stop"; exit 1; }
  TMP=$SD/catalogs_regen_tmp
  rm -rf "$TMP"; mkdir -p "$TMP"
  say "catalogs -> $TMP/seed$S (glass venv, NPY_DISABLE_CPU_FEATURES='$NOAVX512_24', $OMP_NUM_THREADS threads)"
  ( export NPY_DISABLE_CPU_FEATURES="$NOAVX512_24"
    /usr/bin/time -v $PY -u "$GEN" --seed "$S" --stage catalogs --outroot "$TMP" ) > "$LOGD/catalogs.log" 2>&1 \
    || { say "CATALOGS FAILED"; exit 1; }
  mv "$TMP/seed$S/catalogs" "$SD/catalogs" && cp -p "$TMP/seed$S/META.json" "$SD/catalogs/regen_META.json" \
    || { say "moving catalogs failed"; exit 1; }
  ls -la "$TMP/seed$S" >> "$LOG"
  rm -rf "$TMP"
  say "catalogs done in $(( $(date +%s) - T0 )) s ($(du -sh $SD/catalogs | cut -f1))"
fi
T1=$(date +%s)
export NPY_DISABLE_CPU_FEATURES="$NOAVX512_126"     # every jax-env step below

# ---- 2. bitwise checks against the seed's records ------------------------------------------------------------------
G=$SCR/gate/seed$S
rm -rf "$G"; mkdir -p "$G/seed$S/events"; ln -sfn "$SD/catalogs" "$G/seed$S/catalogs"
GATE=PASS
for SPEC in "events.h5|" \
            "events_pureagn.h5|--f_agn 1.0 --seed_events $((S*1000+9)) --n_events 1000 --nsamp 2000 --events_suffix _pureagn" \
            "events_puregal.h5|--f_agn 0.0 --seed_events $((S*1000+8)) --n_events 1000 --nsamp 2000 --events_suffix _puregal"; do
  IFS='|' read -r F EXTRA <<< "$SPEC"
  $PY -u "$GEN" --seed "$S" --stage events --outroot "$G" --darksirens "$PIN" $EXTRA > "$LOGD/gate_${F%.h5}.log" 2>&1 \
    || { say "gate: regeneration of $F failed"; GATE=FAIL; continue; }
  if $PY "$BITCHECK" "$SD/events/$F" "$G/seed$S/events/$F" --out "$RESD/gate_${F%.h5}.json" > /dev/null; then
    say "gate $F: PASS (every dataset byte-identical)"
  else
    say "gate $F: FAIL"; GATE=FAIL
  fi
done
$PY -u "$SURV" --seed "$S" --nside 32 --tracers agn --out-dir "$G/surveys_ns32" --reproduce > "$LOGD/reproduce_agn_ns32.log" 2>&1 \
  && { REPRO=PASS; say "AGN nside-32 survey rebuilt from the catalog: identical to surveys/survey_agn_complete_ns32.h5"; } \
  || { REPRO=FAIL; say "AGN nside-32 survey reproduction FAILED"; }
cp -p "$G"/surveys_ns32/*.json "$RESD/" 2>/dev/null
T2=$(date +%s)
if [ "$GATE" != PASS ] || [ "$REPRO" != PASS ]; then
  say "BITWISE CHECK FAILED for seed $S (gate $GATE, survey $REPRO): nothing written into events/ (scratch kept)"
  echo "{\"seed\": $S, \"gate\": \"$GATE\", \"agn_ns32_reproduce\": \"$REPRO\"}" > "$RESD/summary.json"
  exit 2
fi
rm -rf "$G/seed$S/events"

# ---- 3./4. the production set of stage 1 (mixed, core) and its events-only validation ----------------------------------
GR=$SCR/gen/seed$S
mkdir -p "$GR"
ln -sfn "$SD/catalogs" "$GR/catalogs"; ln -sfn "$SD/events" "$GR/events"; ln -sfn "$SD/validation" "$GR/validation"
SFX=_corepop
OUT=$SD/events/events${SFX}.h5
if [ -e "$OUT" ]; then
  say "$OUT exists: skipped"; GEN_S=0
else
  ta=$(date +%s)
  $PY -u "$GEN" --seed "$S" --stage events --outroot "$SCR/gen" --darksirens "$PIN" --population_code core \
      --events_suffix "$SFX" > "$LOGD/gen${SFX}.log" 2>&1 || { say "GENERATION FAILED"; exit 3; }
  GEN_S=$(( $(date +%s) - ta ))
  say "mixed -> $(basename $OUT): generated in $GEN_S s"
fi
tb=$(date +%s)
$PY -u "$GEN" --seed "$S" --stage validation --outroot "$SCR/gen" --darksirens "$PIN" --events_suffix "$SFX" \
    > "$LOGD/val${SFX}.log" 2>&1 && V=PASS || V=FAIL
VAL_S=$(( $(date +%s) - tb ))
say "validation $V in $VAL_S s"
ls -la "$GR" >> "$LOG"

# ---- 5. survey files at nside 64 (complete galaxy and AGN catalogs) -------------------------------------------------
tc=$(date +%s)
$PY -u "$SURV" --seed "$S" --nside 64 > "$LOGD/surveys_ns64.log" 2>&1 && SV=PASS || SV=FAIL
say "nside-64 survey files: $SV in $(( $(date +%s) - tc )) s"

md5_after=$(md5sum "$SD/META.json" | cut -d' ' -f1)
[ "$md5_before" = "$md5_after" ] && M=unchanged || M=CHANGED
T3=$(date +%s)
cat > "$RESD/summary.json" <<EOF
{"seed": $S, "gate": "$GATE", "agn_ns32_reproduce": "$REPRO", "generation_s": $GEN_S, "validation": "$V",
 "validation_s": $VAL_S, "surveys_ns64": "$SV", "META_json_md5": "$M",
 "seconds": {"catalogs": $((T1 - T0)), "checks": $((T2 - T1)), "rest": $((T3 - T2)), "total": $((T3 - T0))}}
EOF
say "seed $S done in $((T3 - T0)) s: gate $GATE, reproduce $REPRO, validation $V, surveys $SV, META.json $M"
[ "$V" = PASS ] && [ "$SV" = PASS ] && [ "$M" = unchanged ]
