#!/bin/bash
# Closure scans for both detection arms across the five catalog realisations.
# Same estimator, guard and grid as the published baseline; wider H0 grid at the
# same 0.125 resolution so a larger offset cannot run off the end.
set -euo pipefail
cd "$(dirname "$0")/.."
COMMON="--universe_model dark_sirens_complete --catalog_sky_weighting field \
  --scan h0 --h0_grid 58.0 78.0 161 --h0_true 67.74 \
  --selection_neff_guard hard --max_likelihood_variance 1e6 --outdir results"
scan () {  # arm tag survey
  local arm=$1 tag=$2 sur=$3
  python scripts/scan_h0f.py $COMMON \
    --survey_path "$sur" \
    --gw_path "data_derived/obsdet/ev_${arm}_${tag}.h5" \
    --gwselection_path "data_derived/obsdet/sel_${arm}.h5" \
    --out_tag "obsdet_${arm}_${tag}" > "logs/obsdet_scan_${arm}_${tag}.log" 2>&1
  echo "done ${arm} ${tag}"
}
for arm in ctrl obs; do
  scan $arm b     data_derived/deep_survey_z2_ns16.h5
  scan $arm s4102 data_derived/survey_s4102_ns16.h5
  scan $arm s4103 data_derived/survey_s4103_ns16.h5
  scan $arm s4104 data_derived/survey_s4104_ns16.h5
  scan $arm s4105 data_derived/survey_s4105_ns16.h5
done
echo "ALL OBSDET SCANS DONE"
