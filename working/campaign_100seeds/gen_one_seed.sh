#!/usr/bin/env bash
# Generate ONE new v3 seed for the 100-seed campaign, end to end.
#
#   ./gen_one_seed.sh <SEED>
#
# Stages, in the order the record uses them (working/data/run_v3_seed.sh), with the
# campaign's injection sizes -- NOT the CLI defaults -- so the selection integrals
# stay comparable to seeds 100-105:  NDRAW_T=1.5e8 targeted, NDRAW_P=4.0e8 popuni.
#
# Then the two pure-tracer event draws analysis_0 needs (sub-seed offsets 8 and 9,
# unused by the generator, so they are independent of every recorded stream), then
# validation, then the prune.
#
# THE PRUNE.  catalogs/ is 6.7 GB of the 9.4 GB a seed occupies, and 5.7 GB of that
# is catalog_gal_complete.h5 alone.  No scan driver in analyses 0, 1 or 2 ever opens
# catalogs/ -- they read surveys/, events/ and injections/.  catalogs/ is an input to
# the surveys and events stages only, both of which have finished by this point, so
# it is deleted and the seed lands at ~2.7 GB.  It is reproducible in ~4 minutes
# (`--stage catalogs`) from the same seed if it is ever needed again; that is the
# whole reason this is safe to do at 95x scale on a filesystem at 98% full.
#
# VALIDATION is run but is NOT allowed to abort the prune, and its verdict is
# recorded rather than acted on here: see manifest/seeds.tsv and the note on V6 in
# analysis_1/README.md ("A note on seed 104") -- V6 applies a Gaussian binomial
# error to one-count Poisson bins, so it fails seeds whose end-to-end closure is
# fine.  Selection of which seeds enter the measurement happens at aggregation.
set -uo pipefail

S=${1:?usage: gen_one_seed.sh SEED}
HERE=$(cd "$(dirname "$0")" && pwd)
DATA=/hildafs/projects/phy230014p/magana/gws-agn/working/data
OUTROOT=${OUTROOT:-/hildafs/projects/phy220048p/magana/gws-agn-data-v3}
GEN=$DATA/generate_dataset.py
MANIFEST=$HERE/manifest/seeds.tsv
LOG=$HERE/logs/gen_s${S}.log

NDRAW_T=${NDRAW_T:-150000000}
NDRAW_P=${NDRAW_P:-400000000}

PY=/hildafs/home/magana/tmp_ondemand_hildafs_phy230014p_symlink/magana/.conda/envs/jax/bin
export PATH="$PY:$PATH"
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS MKL_NUM_THREADS=$OMP_NUM_THREADS

# the campaign pin: generation imports darksirens (import_gmd) in the events,
# surveys and injections stages, so the dataset carries a SHA too -- seed101 records
# 2b86a2d.  Generating new seeds on any other SHA makes them incomparable.
. "$HERE/pin.sh"
ds_pin_guard || exit 1

mkdir -p "$HERE/logs" "$HERE/manifest"
: > "$LOG"
say () { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }

record () {  # seed status detail bytes
  ( flock 9; printf '%s\t%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" "$(date -u +%FT%TZ)" >> "$MANIFEST"
  ) 9>"$MANIFEST.lock"
}

SD=$OUTROOT/seed$S
say "===== seed $S : generation start (outroot $OUTROOT) ====="

# ---- 1. the four data stages -------------------------------------------------
for ST in catalogs events surveys injections; do
  say "stage $ST"
  EXTRA=""
  [ "$ST" = injections ] && EXTRA="--ndraw_targeted $NDRAW_T --ndraw_popuni $NDRAW_P"
  if ! python -u "$GEN" --darksirens "$DS_PIN" --seed "$S" --stage "$ST" --outroot "$OUTROOT" --overwrite $EXTRA >> "$LOG" 2>&1; then
    say "STAGE $ST FAILED for seed $S"
    record "$S" GEN_FAILED "stage=$ST" 0
    exit 1
  fi
done

# ---- 2. the two pure-tracer event draws analysis_0 needs (needs catalogs/) ----
for SPEC in "puregal 0.0 8" "pureagn 1.0 9"; do
  set -- $SPEC; SFX=$1; FAGN=$2; OFF=$3
  SEED_EV=$(( S * 1000 + OFF ))
  OUT=$SD/events/events_${SFX}.h5
  if [ -s "$OUT" ]; then say "skip pure-tracer $SFX (exists)"; continue; fi
  say "pure-tracer events $SFX  f_agn=$FAGN  seed_events=$SEED_EV"
  if ! python -u "$GEN" --darksirens "$DS_PIN" --seed "$S" --stage events --outroot "$OUTROOT" \
        --f_agn "$FAGN" --seed_events "$SEED_EV" \
        --n_events 1000 --nsamp 2000 --events_suffix "_${SFX}" >> "$LOG" 2>&1; then
    say "PURE-TRACER $SFX FAILED for seed $S"
    record "$S" GEN_FAILED "pure_tracer=$SFX" 0
    exit 1
  fi
done

# ---- 3. validation: recorded, never fatal ------------------------------------
say "stage validation"
VSTATUS=PASS; VDETAIL=-
if ! python -u "$GEN" --darksirens "$DS_PIN" --seed "$S" --stage validation --outroot "$OUTROOT" --overwrite >> "$LOG" 2>&1; then
  VSTATUS=VALIDATION_FAILED
  VDETAIL=$(grep -o "VALIDATION FAILED: \[.*\]" "$LOG" | tail -1 | sed "s/VALIDATION FAILED: //; s/[][' ]//g")
  [ -z "$VDETAIL" ] && VDETAIL=unknown
  say "validation FAILED: $VDETAIL"
else
  say "validation PASS"
fi

# ---- 4. the prune ------------------------------------------------------------
BEFORE=$(du -sb "$SD" 2>/dev/null | cut -f1)
if [ -d "$SD/catalogs" ]; then
  say "pruning catalogs/ ($(du -sh "$SD/catalogs" | cut -f1))"
  rm -rf "$SD/catalogs"
fi
AFTER=$(du -sb "$SD" 2>/dev/null | cut -f1)
say "seed $S footprint $(numfmt --to=iec "$BEFORE") -> $(numfmt --to=iec "$AFTER")"

# ---- 5. the symlink the analysis drivers resolve DATAROOT through -------------
[ -e "$DATA/seed$S" ] || ln -s "$SD" "$DATA/seed$S"

# ---- 6. the inputs every downstream scan needs must exist --------------------
MISSING=""
for f in surveys/survey_gal_complete_ns32.h5 surveys/survey_agn_complete_ns32.h5 \
         events/events.h5 events/events_puregal.h5 events/events_pureagn.h5 \
         injections/injections_targeted.h5 injections/injections_popuni.h5; do
  [ -s "$SD/$f" ] || MISSING="$MISSING $f"
done
if [ -n "$MISSING" ]; then
  say "INCOMPLETE, missing:$MISSING"
  record "$S" INCOMPLETE "missing:${MISSING// /,}" "$AFTER"
  exit 1
fi

record "$S" "$VSTATUS" "$VDETAIL" "$AFTER"
say "===== seed $S DONE ($VSTATUS) ====="
