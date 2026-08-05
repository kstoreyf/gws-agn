#!/bin/bash
# Events for both detection arms across the five catalog realisations of the
# published baseline.  Arms share catalog, survey, seed and every ancillary
# uncertainty model, so an A/B difference is the detection rule alone.
set -euo pipefail
cd "$(dirname "$0")/.."
SNR_OBS=6.278363879917771
run () {  # tag catalog seed
  local tag=$1 cat=$2 seed=$3
  python scripts/build_obsdet_mock.py --mode events --detection true-params \
    --catalog "$cat" --seed "$seed" --snr_ref 11.5 \
    --out_path "data_derived/obsdet/ev_ctrl_${tag}.h5" \
    --dL_fractional_uncertainty 0.10 \
    --summary_json "results/obsdet_ev_ctrl_${tag}.json" > "logs/obsdet_ev_ctrl_${tag}.log" 2>&1
  python scripts/build_obsdet_mock.py --mode events --detection observed-data \
    --catalog "$cat" --seed "$seed" --snr_ref "$SNR_OBS" \
    --out_path "data_derived/obsdet/ev_obs_${tag}.h5" \
    --dL_fractional_uncertainty 0.10 \
    --summary_json "results/obsdet_ev_obs_${tag}.json" > "logs/obsdet_ev_obs_${tag}.log" 2>&1
  echo "done $tag"
}
run b     data_derived/deep_mock_z2_big/mock_galaxy_catalog_complete.h5 4101
run s4102 data_derived/pefix_s4102/mock_galaxy_catalog_complete.h5      4102
run s4103 data_derived/pefix_s4103/mock_galaxy_catalog_complete.h5      4103
run s4104 data_derived/pefix_s4104/mock_galaxy_catalog_complete.h5      4104
run s4105 data_derived/pefix_s4105/mock_galaxy_catalog_complete.h5      4105
echo "ALL EVENTS DONE"
