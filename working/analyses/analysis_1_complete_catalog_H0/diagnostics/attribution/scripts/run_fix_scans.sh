#!/bin/bash
# ATTRIBUTION follow-up, TASK 2 -- the before/after H0 scans.
#
# Identical to the two matched-host controls of record (ctrl_gal_matched,
# ctrl_agn_matched) in EVERY respect except the events file: the PE mass samples
# are reweighted to the exact flat-prior posterior of the generator's own
# measurement model obs ~ N(m, f m), carried through darksirens as a per-sample
# p_pe correction (scripts/make_pe_corrected_events.py).  Same model, same
# log10n0, same grid, same guard convention, same blocking, same targeted
# injection lane, same KDE window.
set -euo pipefail
cd "$(dirname "$0")/.."

DATA=/hildafs/projects/phy230014p/magana/gws-agn/working/data/seed100
SUR_GAL=$DATA/surveys/survey_gal_complete_ns32.h5
SUR_AGN=$DATA/surveys/survey_agn_complete_ns32.h5
INJ=$DATA/injections/injections_targeted.h5

MODEL="--universe_model dark_sirens --log10n0 ${LOG10N0:--24}"
COMMON="$MODEL --catalog_sky_weighting field --scan h0 --h0_true 67.74 \
  --selection_neff_guard hard --max_likelihood_variance 1e6 \
  --sel_batch_size ${SEL_BATCH:-50000} --pe_event_block ${PE_BLOCK:-25} --outdir results"
KDE_W=${KDE_W:-4096}
NGRID=${NGRID:-201}

scan () {  # tag survey events [extra...]
  local tag=$1 sur=$2 ev=$3; shift 3
  if [ -s "results/${tag}.h5" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "[$(date -u +%H:%M:%S)] skip $tag"; return 0
  fi
  echo "[$(date -u +%H:%M:%S)] scanning $tag"
  python scripts/scan_h0f.py $COMMON --h0_grid 50.0 100.0 "$NGRID" \
    --survey_path "$sur" --gw_path "$ev" --gwselection_path "$INJ" "$@" \
    --out_tag "$tag" > "logs/${tag}.log" 2>&1
  echo "[$(date -u +%H:%M:%S)] done $tag"
}

scan fix_named_defect_agn      "$SUR_AGN" data_derived/events_agn_hosted_pefix_m1m2.h5
scan fix_named_defect_agn_m1   "$SUR_AGN" data_derived/events_agn_hosted_pefix_m1.h5
scan fix_named_defect_gal      "$SUR_GAL" data_derived/events_gal_hosted_pefix_m1m2.h5 --kde_window "$KDE_W"
scan fix_named_defect_gal_m1   "$SUR_GAL" data_derived/events_gal_hosted_pefix_m1.h5   --kde_window "$KDE_W"
echo "FIX SCANS DONE"
