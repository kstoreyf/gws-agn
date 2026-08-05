#!/bin/bash
# Brute-force test of Monte-Carlo resolution: same catalogs, same event truths,
# same injections -- only the number of PE samples per event changes.
# The delta-method correction (scan_pe_variance.py) tests only the leading term;
# with ~1 catalog host inside each event's PE window the weights are heavy-tailed,
# so higher-order MC bias is not bounded by it. If the offset scales as 1/nsamp,
# MC resolution is the mechanism; if it is flat, it is not.
set -euo pipefail
cd "$(dirname "$0")/.."
SNR_OBS=6.278363879917771
NSAMP=${1:-16000}
TAG=${2:-n16k}
declare -A CATS=(
  [b]=data_derived/deep_mock_z2_big/mock_galaxy_catalog_complete.h5
  [s4102]=data_derived/pefix_s4102/mock_galaxy_catalog_complete.h5
  [s4103]=data_derived/pefix_s4103/mock_galaxy_catalog_complete.h5
  [s4104]=data_derived/pefix_s4104/mock_galaxy_catalog_complete.h5
  [s4105]=data_derived/pefix_s4105/mock_galaxy_catalog_complete.h5
  [n4201]=data_derived/obsdet/cat_n4201.h5
  [n4202]=data_derived/obsdet/cat_n4202.h5
  [n4203]=data_derived/obsdet/cat_n4203.h5
  [n4204]=data_derived/obsdet/cat_n4204.h5
  [n4205]=data_derived/obsdet/cat_n4205.h5
)
declare -A SEEDS=([b]=4101 [s4102]=4102 [s4103]=4103 [s4104]=4104 [s4105]=4105
                  [n4201]=4201 [n4202]=4202 [n4203]=4203 [n4204]=4204 [n4205]=4205)
declare -A SURV=(
  [b]=data_derived/deep_survey_z2_ns16.h5
  [s4102]=data_derived/survey_s4102_ns16.h5
  [s4103]=data_derived/survey_s4103_ns16.h5
  [s4104]=data_derived/survey_s4104_ns16.h5
  [s4105]=data_derived/survey_s4105_ns16.h5
  [n4201]=data_derived/obsdet/survey_n4201_ns16.h5
  [n4202]=data_derived/obsdet/survey_n4202_ns16.h5
  [n4203]=data_derived/obsdet/survey_n4203_ns16.h5
  [n4204]=data_derived/obsdet/survey_n4204_ns16.h5
  [n4205]=data_derived/obsdet/survey_n4205_ns16.h5
)
KEYS="b s4102 s4103 s4104 s4105 n4201 n4202 n4203 n4204 n4205"

for k in $KEYS; do
  python scripts/build_obsdet_mock.py --mode events --detection observed-data \
    --catalog "${CATS[$k]}" --seed "${SEEDS[$k]}" --snr_ref "$SNR_OBS" \
    --nsamp "$NSAMP" --out_path "data_derived/obsdet/ev_${TAG}_${k}.h5" \
    --dL_fractional_uncertainty 0.10 \
    > "logs/obsdet_ev_${TAG}_${k}.log" 2>&1 &
done
wait
echo "EVENTS DONE"

COMMON="--universe_model dark_sirens_complete --catalog_sky_weighting field \
  --scan h0 --h0_grid 58.0 78.0 161 --h0_true 67.74 \
  --selection_neff_guard hard --max_likelihood_variance 1e6 --outdir results"
for k in $KEYS; do
  python scripts/scan_h0f.py $COMMON --survey_path "${SURV[$k]}" \
    --gw_path "data_derived/obsdet/ev_${TAG}_${k}.h5" \
    --gwselection_path data_derived/obsdet/sel_obs.h5 \
    --out_tag "obsdet_${TAG}_${k}" > "logs/obsdet_scan_${TAG}_${k}.log" 2>&1
  echo "scanned $k"
done
echo "ALL ${TAG} DONE"
