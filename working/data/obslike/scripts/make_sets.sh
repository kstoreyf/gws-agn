#!/usr/bin/env bash
# Extra core-population event sets on the realisations whose catalogs are on disk
# (100, 101, 102, 103, 105), each followed by its events-only validation.
#
#   ./make_sets.sh KIND [SEED ...]        (default seeds: 100 101 102 103 105)
#
# KIND       file                                       flags                                        sub-seed
# fagn0p10   events_fagn0p10_corepop.h5                 --f_agn 0.10                                 S*1000+10
# obslike    events_obslike.h5                          --mass_spectrum gwtc3_pp                     S*1000+11
# obsmarked  events_marked_dmu0p10_dmuG5_obslike.h5     --mass_spectrum gwtc3_pp --dmu_chi_agn 0.10  S*1000+11
#                                                       --dmu_G_agn 5
# all with --population_code core, 1000 detected events x 2000 samples.  Offsets 10 and 11
# are used by no other set (1-7 the record, 8/9 the pure-tracer sets), so these draws are
# independent of every existing one; the two obslike kinds share their sub-seed exactly as
# the present family's mixed and host-dependent sets share S*1000+3.
#
# Not touched: --overwrite is never passed (an existing file is skipped), the suffix
# suppresses the META.json merge, and the validation runs in its events-only mode.
# numpy's AVX-512 loops are switched off as for the first corepop sets (a no-op on RM's
# Haswell nodes).
set -uo pipefail
KIND=${1:?usage: make_sets.sh fagn0p10|obslike|obsmarked [SEED ...]}; shift
DATA=/hildafs/projects/phy230014p/magana/gws-agn/working/data
GEN=$DATA/generate_dataset.py
OUTROOT=/hildafs/projects/phy230054p/magana/gws-agn-data-v3
PIN=/hildafs/projects/phy230014p/magana/src/darksirens-2b86a2d
PY=/hildafs/home/magana/tmp_ondemand_hildafs_phy230014p_symlink/magana/.conda/envs/jax/bin/python
LOGD=$DATA/obslike/logs/sets
mkdir -p "$LOGD"
export PYTHONPATH=$PIN JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_KNL AVX512_KNM AVX512_SKX AVX512_CLX AVX512_CNL AVX512_ICL"
[ "$(git -C $PIN rev-parse HEAD)" = 2b86a2d8d48fdb5173f0ba259b8996104187dd2d ] || { echo "darksirens pin wrong"; exit 1; }
[ -z "$(git -C $PIN status --porcelain)" ] || { echo "darksirens pin dirty"; exit 1; }
SEEDS=("$@")
[ ${#SEEDS[@]} -eq 0 ] && SEEDS=(100 101 102 103 105)
echo "host $(hostname) job ${SLURM_JOB_ID:-none} kind $KIND seeds ${SEEDS[*]}; cpu $(lscpu | grep 'Model name' | sed 's/.*: *//')"

one () {
  local S=$1 SFX EXTRA
  case $KIND in
    fagn0p10)  SFX=_fagn0p10_corepop
               EXTRA="--f_agn 0.10 --seed_events $(( S * 1000 + 10 )) --n_events 1000 --nsamp 2000";;
    obslike)   SFX=_obslike
               EXTRA="--mass_spectrum gwtc3_pp --seed_events $(( S * 1000 + 11 )) --n_events 1000 --nsamp 2000";;
    obsmarked) SFX=_marked_dmu0p10_dmuG5_obslike
               EXTRA="--mass_spectrum gwtc3_pp --dmu_chi_agn 0.10 --dmu_G_agn 5 --seed_events $(( S * 1000 + 11 )) --n_events 1000 --nsamp 2000";;
    # --- the short paper's experiment (paper_short_independent/EXPERIMENT.md) -----------
    # host-dependent sets share the sub-seed of the equal-population set of the same f
    # (+10: f 0.1, +3: f 0.3, +13: f 0.5), so hosts and galaxy-class draws are common in s.
    fagn0p50)  SFX=_fagn0p50_corepop
               EXTRA="--f_agn 0.50 --seed_events $(( S * 1000 + 13 )) --n_events 1000 --nsamp 2000";;
    m1f010)    SFX=_marked_dmu0p10_dmuG5_fagn0p10_corepop
               EXTRA="--dmu_chi_agn 0.10 --dmu_G_agn 5 --f_agn 0.10 --seed_events $(( S * 1000 + 10 )) --n_events 1000 --nsamp 2000";;
    m1f050)    SFX=_marked_dmu0p10_dmuG5_fagn0p50_corepop
               EXTRA="--dmu_chi_agn 0.10 --dmu_G_agn 5 --f_agn 0.50 --seed_events $(( S * 1000 + 13 )) --n_events 1000 --nsamp 2000";;
    m2f010)    SFX=_marked_dmu0p20_dmuG10_fagn0p10_corepop
               EXTRA="--dmu_chi_agn 0.20 --dmu_G_agn 10 --f_agn 0.10 --seed_events $(( S * 1000 + 10 )) --n_events 1000 --nsamp 2000";;
    m2f030)    SFX=_marked_dmu0p20_dmuG10_corepop
               EXTRA="--dmu_chi_agn 0.20 --dmu_G_agn 10";;
    m2f050)    SFX=_marked_dmu0p20_dmuG10_fagn0p50_corepop
               EXTRA="--dmu_chi_agn 0.20 --dmu_G_agn 10 --f_agn 0.50 --seed_events $(( S * 1000 + 13 )) --n_events 1000 --nsamp 2000";;
    obspuregal) SFX=_puregal_obslike
               EXTRA="--mass_spectrum gwtc3_pp --f_agn 0.0 --seed_events $(( S * 1000 + 14 )) --n_events 1000 --nsamp 2000";;
    obspureagn) SFX=_pureagn_obslike
               EXTRA="--mass_spectrum gwtc3_pp --f_agn 1.0 --seed_events $(( S * 1000 + 15 )) --n_events 1000 --nsamp 2000";;
    *) echo "unknown kind $KIND"; return 1;;
  esac
  local OUT=$OUTROOT/seed$S/events/events${SFX}.h5
  if [ -e "$OUT" ]; then
    echo "[$(date -u +%H:%M:%S)] skip seed $S $KIND (exists: $OUT)"
  else
    local ta=$(date +%s.%N)
    /usr/bin/time -v $PY -u "$GEN" --seed "$S" --stage events --outroot "$OUTROOT" --darksirens "$PIN" \
        --population_code core $EXTRA --events_suffix "$SFX" > "$LOGD/gen_s${S}${SFX}.log" 2>&1 \
      || { echo "GENERATION FAILED seed $S $KIND (see $LOGD/gen_s${S}${SFX}.log)"; return 1; }
    echo "[$(date -u +%H:%M:%S)] seed $S $KIND generated in $(echo "$(date +%s.%N) - $ta" | bc | cut -c1-5) s -> $OUT"
  fi
  local tb=$(date +%s.%N)
  /usr/bin/time -v $PY -u "$GEN" --seed "$S" --stage validation --outroot "$OUTROOT" --darksirens "$PIN" \
      --events_suffix "$SFX" > "$LOGD/val_s${S}${SFX}.log" 2>&1 \
    && echo "[$(date -u +%H:%M:%S)] seed $S $KIND validation PASS ($(echo "$(date +%s.%N) - $tb" | bc | cut -c1-5) s)" \
    || echo "[$(date -u +%H:%M:%S)] seed $S $KIND validation FAILED: $(grep -o 'VALIDATION FAILED: .*\]' "$LOGD/val_s${S}${SFX}.log")"
}

for S in "${SEEDS[@]}"; do one "$S" & done
wait
echo "DONE $KIND: ${SEEDS[*]}"
