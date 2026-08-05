#!/bin/bash
# analysis_1_complete_catalog_H0 — the four independent single-tracer H0 scans,
# plus the two matched-host controls, all under ONE estimator.
#
#   1. h0_gal_targeted    GAL complete catalog + targeted injection lane
#   2. h0_gal_popuni      GAL complete catalog + popuni  injection lane  (lane cross-check)
#   3. h0_agn_targeted    AGN complete catalog + targeted injection lane
#   4. h0_agn_popuni      AGN complete catalog + popuni  injection lane  (lane cross-check)
#   c1. ctrl_gal_matched  GAL catalog + only the events it hosts + targeted lane
#   c2. ctrl_agn_matched  AGN catalog + only the events it hosts + targeted lane
#
# ---------------------------------------------------------------------------
# THE MODEL, AND WHY log10n0 = -24
# ---------------------------------------------------------------------------
# Every run in this campaign is carried by darksirens' GENERAL incomplete-catalog
# likelihood `dark_sirens`, driven into its complete-catalog limit by taking the
# modelled missing comoving density to zero.  That limit is not approximate:
# experiment_model_equivalence measured `dark_sirens` against the dedicated
# complete-catalog likelihood `dark_sirens_complete` on this dataset and found
# them **bit-for-bit identical in all 201 cells of all four configurations at
# log10n0 = -24** (max |delta ln L| = 0, identical float64 posterior medians).
# At the log10n0 = -12 the campaign used earlier the completion term is small but
# NOT yet off (4.1e-6 nats on the sparse AGN catalog), so -24 is the value at
# which the nesting is exact rather than merely unmeasurable.
# See working/experiments/experiment_model_equivalence/README.md.
#
# All nuisances are fixed: delta = 0, sigma_kde = 0, b_miss = 1 (the driver's
# FIXED_DEFAULTS), use_lss off.  Only H0 is scanned.
#
# `dark_sirens` requires float64 survey blocks — on float32 blocks its
# observed-density KDE returns NaN at the z = 100 row padding and the likelihood
# is -inf everywhere.  The dataset has been float64 since 2026-07-31
# (working/data/generate_dataset.py, CAT_DTYPE).
#
# Grid: H0 in [50, 100] x 201, catalog_sky_weighting = field, fixed powerlaw+peak
# population, Om0 pinned at 0.3075.
# Guard: the campaign convention — hard Neff wall, variance criterion made inert.
set -euo pipefail
cd "$(dirname "$0")/.."

DATA=/hildafs/projects/phy230014p/magana/gws-agn/working/data/seed100
GW=$DATA/events/events.h5
SUR_GAL=$DATA/surveys/survey_gal_complete_ns32.h5
SUR_AGN=$DATA/surveys/survey_agn_complete_ns32.h5
INJ_TARGETED=$DATA/injections/injections_targeted.h5
INJ_POPUNI=$DATA/injections/injections_popuni.h5
EV_GAL=data_derived/events_gal_hosted.h5
EV_AGN=data_derived/events_agn_hosted.h5

MODEL="--universe_model dark_sirens --log10n0 ${LOG10N0:--24}"

COMMON="$MODEL --catalog_sky_weighting field --scan h0 --h0_true 67.74 \
  --selection_neff_guard hard --max_likelihood_variance 1e6 \
  --sel_batch_size ${SEL_BATCH:-200000} --pe_event_block ${PE_BLOCK:-100} --outdir results"

# The complete GAL survey carries ~1.2e4 galaxies per nside-32 row, so darksirens'
# windowed catalog-KDE evaluator must be given a window at least as large as the
# number of galaxies inside the kernel support.  W is sized from
# darksirens.redshift.catalog.recommended_kde_window measured on THIS dataset's
# float64 GAL survey (scripts/kde_window_check.py; see README).  The AGN survey's
# rows are far shorter than the window and take the full-row path.
KDE_W=${KDE_W:-4096}
NGRID=${NGRID:-201}

scan () {  # tag survey events selection [extra...]
  local tag=$1 sur=$2 ev=$3 sel=$4; shift 4
  if [ -s "results/${tag}.h5" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "[$(date -u +%H:%M:%S)] skip $tag (results/${tag}.h5 exists)"; return 0
  fi
  echo "[$(date -u +%H:%M:%S)] scanning $tag"
  python scripts/scan_h0f.py $COMMON --h0_grid 50.0 100.0 "$NGRID" \
    --survey_path "$sur" --gw_path "$ev" --gwselection_path "$sel" "$@" \
    --out_tag "$tag" > "logs/${tag}.log" 2>&1
  echo "[$(date -u +%H:%M:%S)] done $tag"
}

# AGN first — the cheap configurations double as a smoke test of the model, the
# survey files and the guard before the expensive GAL grids start.
scan h0_agn_targeted  "$SUR_AGN" "$GW"     "$INJ_TARGETED"
scan h0_agn_popuni    "$SUR_AGN" "$GW"     "$INJ_POPUNI"
scan ctrl_agn_matched "$SUR_AGN" "$EV_AGN" "$INJ_TARGETED"
scan h0_gal_targeted  "$SUR_GAL" "$GW"     "$INJ_TARGETED" --kde_window "$KDE_W"
scan h0_gal_popuni    "$SUR_GAL" "$GW"     "$INJ_POPUNI"   --kde_window "$KDE_W"
scan ctrl_gal_matched "$SUR_GAL" "$EV_GAL" "$INJ_TARGETED" --kde_window "$KDE_W"

echo "ALL SCANS DONE"
