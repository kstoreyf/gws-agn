#!/bin/bash
# Does the residual depend on the CATALOG kernel width?
# sigma_kde broadens the per-host redshift kernel (effective width
# sqrt(dzgals^2 + sigma_kde^2)). If the offset moves with it, the bias lives in
# the catalog-conditioned redshift prior; if it is flat, that prior is exonerated
# and the problem is in the dL<->z / selection machinery.
set -euo pipefail
cd "$(dirname "$0")/.."
COMMON="--universe_model dark_sirens_complete --catalog_sky_weighting field \
  --scan h0 --h0_grid 58.0 78.0 161 --h0_true 67.74 \
  --selection_neff_guard hard --max_likelihood_variance 1e6 --outdir results \
  --survey_path data_derived/deep_survey_z2_ns16.h5 \
  --gw_path data_derived/obsdet/ev_obs_b.h5 \
  --gwselection_path data_derived/obsdet/sel_obs.h5"
for sk in 0.000 0.003 0.010 0.020 0.040; do
  python scripts/scan_h0f.py $COMMON --nuisance_json "{\"sigma_kde\": $sk}" \
    --out_tag "skde_${sk}" > "logs/skde_${sk}.log" 2>&1
  echo "sigma_kde=$sk done"
done
echo "SKDE DONE"
