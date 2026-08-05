#!/bin/bash
# Bound the (m1det, q)-basis Jacobian in p_pe.  Paired at the EVENT level -- the
# same events, only p_pe rewritten -- so the difference carries essentially no
# realisation noise and a few realisations bound the effect.
set -euo pipefail
cd "$(dirname "$0")/.."
COMMON="--universe_model dark_sirens_complete --catalog_sky_weighting field \
  --scan h0 --h0_grid 58.0 78.0 161 --h0_true 67.74 \
  --selection_neff_guard hard --max_likelihood_variance 1e6 --outdir results"
for spec in "s4102 data_derived/survey_s4102_ns16.h5" "n4201 data_derived/obsdet/survey_n4201_ns16.h5" "n4207 data_derived/obsdet/survey_n4207_ns16.h5"; do
  set -- $spec; tag=$1; sur=$2
  python scripts/repe_basis_jacobian.py --in_path "data_derived/obsdet/ev_obs_${tag}.h5" \
    --out_path "data_derived/obsdet/ev_obsq_${tag}.h5" > "logs/obsq_${tag}.log" 2>&1
  python scripts/scan_h0f.py $COMMON --survey_path "$sur" \
    --gw_path "data_derived/obsdet/ev_obsq_${tag}.h5" \
    --gwselection_path data_derived/obsdet/sel_obs.h5 \
    --out_tag "obsdet_obsq_${tag}" >> "logs/obsq_${tag}.log" 2>&1
  echo "done $tag"
done
echo "OBSQ PROBE DONE"
