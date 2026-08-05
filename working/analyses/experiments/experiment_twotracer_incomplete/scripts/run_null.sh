#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
GW=data_derived/events_skyshuffled.h5
COMMON="--universe_model dark_sirens --catalog_sky_weighting field --gw_path $GW \
  --log10n0 -5.806380 --log10n0_c2 -7.720033 \
  --nuisance_json {\"delta\":0.0194,\"delta_c2\":-0.0031} \
  --selection_neff_guard hard --max_likelihood_variance 1e6 \
  --outdir results --h0_true 67.74 --f_true 0.3 --scan f --f_grid 0.0 1.0 41"
for lev in complete m20.0 m18.0; do
  python scripts/scan_h0f.py $COMMON \
    --survey_path "data_derived/survey_gal_${lev}_ns32.h5" "data_derived/survey_agn_${lev}_ns32.h5" \
    --gwselection_path "data_derived/inj_${lev}.h5" \
    --out_tag "fscan_null_${lev}" > "logs/fscan_null_${lev}.log" 2>&1
  echo "null $lev"
done
echo "NULL DONE"
