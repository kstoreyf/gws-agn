#!/bin/bash
# H0 closure scans for the obsfix arm (PR #335 observable sky width).
# Same estimator, grid, guard and selection file as the obs arm; the only
# change relative to obsdet_obs_<tag> is the events file.
set -euo pipefail
cd "$(dirname "$0")/.."
DD=data_derived/obsdet
COMMON="--universe_model dark_sirens_complete --catalog_sky_weighting field \
  --scan h0 --h0_grid 58.0 78.0 161 --h0_true 67.74 \
  --selection_neff_guard hard --max_likelihood_variance 1e6 --outdir results"

scan () {  # tag survey
  local tag=$1 sur=$2
  python scripts/scan_h0f.py $COMMON \
    --survey_path "$sur" \
    --gw_path "$DD/ev_obsfix_${tag}.h5" \
    --gwselection_path "$DD/sel_obs.h5" \
    --out_tag "obsdet_fix_${tag}" > "logs/obsdet_scan_fix_${tag}.log" 2>&1
  echo "scanned $tag  $(date +%H:%M:%S)"
}

scan b     data_derived/deep_survey_z2_ns16.h5
scan s4102 data_derived/survey_s4102_ns16.h5
scan s4103 data_derived/survey_s4103_ns16.h5
scan s4104 data_derived/survey_s4104_ns16.h5
scan s4105 data_derived/survey_s4105_ns16.h5
for s in $(seq 4201 4215); do
  scan "n$s" "$DD/survey_n${s}_ns16.h5"
done
echo "ALL OBSFIX SCANS DONE"
