#!/bin/bash
# Extend the closure A/B from 5 to 20 catalog realisations.
# CPU stages (catalog, pixelation, events) run several seeds at a time; the GPU
# scans run serially afterwards.
set -euo pipefail
cd "$(dirname "$0")/.."
GMD=/hildafs/projects/phy230014p/magana/src/darksirens-pefix/scripts/mock_dark_sirens
SNR_OBS=6.278363879917771
DD=data_derived/obsdet

prep () {  # seed
  local s=$1 tag="n$1"
  python scripts/build_obsdet_mock.py --mode catalog --detection true-params \
    --out_path "$DD/cat_${tag}.h5" --seed "$s" --dL_fractional_uncertainty 0.10 \
    > "logs/obsdet_cat_${tag}.log" 2>&1
  python scripts/pixelate_complete_catalog.py --complete_catalog "$DD/cat_${tag}.h5" \
    --out_path "$DD/survey_${tag}_ns16.h5" --nside 16 --gmd_dir "$GMD" \
    --z_error_floor 0.003 --z_error_slope 0.0 >> "logs/obsdet_cat_${tag}.log" 2>&1
  python scripts/build_obsdet_mock.py --mode events --detection true-params \
    --catalog "$DD/cat_${tag}.h5" --seed "$s" --snr_ref 11.5 \
    --out_path "$DD/ev_ctrl_${tag}.h5" --dL_fractional_uncertainty 0.10 \
    --summary_json "results/obsdet_ev_ctrl_${tag}.json" \
    > "logs/obsdet_ev_ctrl_${tag}.log" 2>&1
  python scripts/build_obsdet_mock.py --mode events --detection observed-data \
    --catalog "$DD/cat_${tag}.h5" --seed "$s" --snr_ref "$SNR_OBS" \
    --out_path "$DD/ev_obs_${tag}.h5" --dL_fractional_uncertainty 0.10 \
    --summary_json "results/obsdet_ev_obs_${tag}.json" \
    > "logs/obsdet_ev_obs_${tag}.log" 2>&1
  echo "prepped $tag"
}

SEEDS=$(seq 4201 4215)
N=0
for s in $SEEDS; do
  prep "$s" &
  N=$((N+1))
  if [ $((N % 5)) -eq 0 ]; then wait; fi
done
wait
echo "ALL PREP DONE"

COMMON="--universe_model dark_sirens_complete --catalog_sky_weighting field \
  --scan h0 --h0_grid 58.0 78.0 161 --h0_true 67.74 \
  --selection_neff_guard hard --max_likelihood_variance 1e6 --outdir results"
for s in $SEEDS; do
  tag="n$s"
  for arm in ctrl obs; do
    python scripts/scan_h0f.py $COMMON \
      --survey_path "$DD/survey_${tag}_ns16.h5" \
      --gw_path "$DD/ev_${arm}_${tag}.h5" \
      --gwselection_path "$DD/sel_${arm}.h5" \
      --out_tag "obsdet_${arm}_${tag}" > "logs/obsdet_scan_${arm}_${tag}.log" 2>&1
  done
  echo "scanned $tag"
done
echo "ALL EXTEND DONE"
