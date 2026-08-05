#!/bin/bash
# STEP 4 -- the matched-host controls against SURVEY RESOLUTION, seed 100.
#
# Everything is the analysis of record except the survey block's HEALPix nside:
# same events (the regenerated, post-(b2)+(c2) ones), same targeted injection lane,
# same estimator, grid, guard and W.  The injections carry TRUE sky positions and
# darksirens re-pixelates them at load time from the survey's own nside, so the
# selection campaign needs no regeneration.
#
# W is held at 4096 for GAL at EVERY resolution so that nothing but the pixel size
# changes along the curve; the measured requirements are 3410 (nside 32), 986
# (nside 64) and 293 (nside 128), so 4096 clears all three and the catalog KDE is
# never truncated.  A window longer than the row is the full-row path.
#
# The nside-32 point is the record scan itself (results/ctrl_{gal,agn}_matched).
set -euo pipefail
cd "$(dirname "$0")/.."
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export DARKSIRENS_SRC=${DARKSIRENS_SRC:-/hildafs/projects/phy230014p/magana/src/darksirens}

DATA=/hildafs/projects/phy230014p/magana/gws-agn/working/data/seed100
SURV=/hildafs/projects/phy220048p/magana/gws-agn-data/derived/analysis_1_complete_catalog_H0/surveys_nside
INJ=$DATA/injections/injections_targeted.h5
KDE_W=${KDE_W:-4096}

COMMON="--universe_model dark_sirens --log10n0 ${LOG10N0:--24} \
  --catalog_sky_weighting field --scan h0 --h0_grid 50.0 100.0 201 --h0_true 67.74 \
  --selection_neff_guard hard --max_likelihood_variance 1e6 \
  --sel_batch_size ${SEL_BATCH:-50000} --pe_event_block ${PE_BLOCK:-25} --outdir results"

for NS in ${NSIDES:-64 128}; do
  for T in agn gal; do
    TAG=ctrl_${T}_matched_ns${NS}
    if [ -s "results/${TAG}.h5" ] && [ "${FORCE:-0}" != "1" ]; then
      echo "[$(date -u +%H:%M:%S)] skip $TAG"; continue
    fi
    EXTRA=""
    [ "$T" = "gal" ] && EXTRA="--kde_window $KDE_W"
    echo "[$(date -u +%H:%M:%S)] scanning $TAG"
    python scripts/scan_h0f.py $COMMON \
      --survey_path "$SURV/survey_${T}_complete_ns${NS}.h5" \
      --gw_path "data_derived/events_${T}_hosted.h5" \
      --gwselection_path "$INJ" $EXTRA --out_tag "$TAG" \
      > "logs/${TAG}.log" 2>&1
    echo "[$(date -u +%H:%M:%S)] done $TAG"
  done
done
echo "NSIDE SCANS DONE"
