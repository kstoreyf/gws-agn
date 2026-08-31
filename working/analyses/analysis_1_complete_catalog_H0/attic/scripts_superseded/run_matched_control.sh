#!/bin/bash
# analysis_1_complete_catalog_H0 — MATCHED-SUBSET CONTROL (not one of the four
# production scans).  Each catalog is given ONLY the events it actually hosts:
#   ctrl_gal_matched   GAL catalog + the 720 GAL-hosted events + targeted lane
#   ctrl_agn_matched   AGN catalog + the 280 AGN-hosted events + targeted lane
# Same model, grid and guard convention as the production scans.  Its only job
# is to separate "the single-tracer mis-specification" from "something wrong
# with the configuration or the data" as the reading of the production offsets.
set -euo pipefail
cd "$(dirname "$0")/.."

DATA=/hildafs/projects/phy230014p/magana/gws-agn/working/data/seed100
SUR_GAL=$DATA/surveys/survey_gal_complete_ns32.h5
SUR_AGN=$DATA/surveys/survey_agn_complete_ns32.h5
INJ_TARGETED=$DATA/injections/injections_targeted.h5
KDE_W=${KDE_W:-4096}      # windowed catalog-KDE size for the dense GAL survey

COMMON="--universe_model dark_sirens_complete --catalog_sky_weighting field \
  --scan h0 --h0_grid 50.0 100.0 201 --h0_true 67.74 \
  --gwselection_path $INJ_TARGETED \
  --selection_neff_guard hard --max_likelihood_variance 1e6 --sel_batch_size ${SEL_BATCH:-200000} --pe_event_block ${PE_BLOCK:-100} --outdir results"

run () {  # tag survey events extra
  local tag=$1 sur=$2 ev=$3; shift 3
  echo "[$(date +%H:%M:%S)] control $tag $*"
  python scripts/scan_h0f.py $COMMON --survey_path "$sur" --gw_path "$ev" "$@" \
    --out_tag "$tag" > "logs/${tag}.log" 2>&1
  echo "[$(date +%H:%M:%S)] done $tag"
}

run ctrl_gal_matched "$SUR_GAL" data_derived/events_gal_hosted.h5 --kde_window "$KDE_W"
run ctrl_agn_matched "$SUR_AGN" data_derived/events_agn_hosted.h5
echo "CONTROLS DONE"
