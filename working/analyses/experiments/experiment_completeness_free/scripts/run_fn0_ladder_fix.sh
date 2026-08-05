#!/bin/bash
# Post-fix rerun of the (f_AGN, log10 n0_AGN) grids on the sigma_ang-FIXED events
# (darksirens PR #335; truth/detected set bit-identical, injections/surveys reused).
# Identical to run_fn0_ladder.sh apart from the events file and the _fix tags.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-/hildafs/home/magana/tmp_ondemand_hildafs_phy230014p_symlink/magana/.conda/envs/jax/bin/python}
INC=../experiment_twotracer_incomplete
GW=../experiment_twotracer_deep/data_derived/twotracer_gw_events_fix.h5
COMMON="--universe_model dark_sirens --catalog_sky_weighting field --gw_path $GW \
  --log10n0 -5.806380 --log10n0_c2 -7.720033 \
  --nuisance_json {\"delta\":0.0194,\"delta_c2\":-0.0031} \
  --selection_neff_guard hard --max_likelihood_variance 1e6 --outdir results \
  --scan fn0 --f_grid 0.0 1.0 51 --n0c2_grid -9.6 -7.1 201 \
  --f_true 0.3 --n0c2_true -7.720033 --h0_fixed 67.74"
for lev in ${LEVELS:-complete m21.0 m20.0 m19.0 m18.0}; do
  $PY scripts/scan_h0f.py $COMMON \
    --survey_path "$INC/data_derived/survey_gal_${lev}_ns32.h5" \
                  "$INC/data_derived/survey_agn_${lev}_ns32.h5" \
    --gwselection_path "$INC/data_derived/inj_${lev}.h5" \
    --out_tag "fn0_${lev}_fix" > "logs/fn0_${lev}_fix.log" 2>&1
  echo "scanned $lev"
done
echo "ALL FN0 FIX SCANS DONE"
