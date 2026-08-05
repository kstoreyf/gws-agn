#!/bin/bash
# analysis_1_complete_catalog_H0 — variance/guard diagnostic at H0 = truth (67.74)
# for each of the four production configurations.  Writes results/guard_<tag>.json
# holding Neff, pe_variance_sum = sum_i sigma^2_PE, the guard threshold and the
# verdict, plus the per-event sigma^2_i distribution.
#
# Same estimator as the scans: `dark_sirens` at log10n0 = -24 (the complete-catalog
# limit, bitwise equal to `dark_sirens_complete` — see run_scans.sh), field
# weighting, all nuisances fixed.
set -euo pipefail
cd "$(dirname "$0")/.."

DATA=/hildafs/projects/phy230014p/magana/gws-agn/working/data/seed100
GW=$DATA/events/events.h5
SUR_GAL=$DATA/surveys/survey_gal_complete_ns32.h5
SUR_AGN=$DATA/surveys/survey_agn_complete_ns32.h5
INJ_TARGETED=$DATA/injections/injections_targeted.h5
INJ_POPUNI=$DATA/injections/injections_popuni.h5

KDE_W=${KDE_W:-4096}
LOG10N0=${LOG10N0:--24}

diag () {  # tag survey selection
  local tag=$1 sur=$2 sel=$3
  local EXTRA=""
  case "$sur" in *survey_gal_*) EXTRA="--kde_window $KDE_W";; esac
  echo "[$(date -u +%H:%M:%S)] guard diag $tag"
  python scripts/diag_variance_guard.py \
    --universe_model dark_sirens --log10n0 "$LOG10N0" \
    --catalog_sky_weighting field \
    --survey_path "$sur" --gw_path "$GW" --gwselection_path "$sel" \
    --h0_at 67.74 --max_likelihood_variance 1e6 --capture_event_vars \
    --sel_batch_size ${SEL_BATCH:-200000} --pe_event_block ${PE_BLOCK:-100} $EXTRA \
    --out_json "results/guard_${tag}.json" > "logs/guard_${tag}.log" 2>&1
  echo "[$(date -u +%H:%M:%S)] done $tag"
}

diag h0_agn_targeted "$SUR_AGN" "$INJ_TARGETED"
diag h0_agn_popuni   "$SUR_AGN" "$INJ_POPUNI"
diag h0_gal_targeted "$SUR_GAL" "$INJ_TARGETED"
diag h0_gal_popuni   "$SUR_GAL" "$INJ_POPUNI"
echo "ALL GUARD DIAGS DONE"
