#!/bin/bash
# Isotropic flux-limit ladder on BOTH tracers of the deep two-tracer mock.
# AGN inherit their host galaxy's apparent magnitude -- they ARE those galaxies --
# so one limit thins both tracers with the same C(z), keeping this a single-axis
# experiment. Events are untouched: incompleteness is an observational effect.
set -euo pipefail
cd "$(dirname "$0")/.."
GMD=/hildafs/projects/phy230014p/magana/src/darksirens-pefix/scripts/mock_dark_sirens
PIX="python scripts/pixelate_complete_catalog.py --nside 32 --gmd_dir $GMD \
     --z_error_floor 0.003 --z_error_slope 0.0 --completeness_z_ref 0.30"

for t in gal agn; do
  $PIX --complete_catalog "data_derived/catalog_${t}_complete.h5" \
       --out_path "data_derived/survey_${t}_complete_ns32.h5" \
       > "logs/pix_${t}_complete.log" 2>&1
  for m in 21.0 20.0 19.0 18.0; do
    $PIX --complete_catalog "data_derived/catalog_${t}_complete.h5" --mag_limit "$m" \
         --out_path "data_derived/survey_${t}_m${m}_ns32.h5" \
         > "logs/pix_${t}_m${m}.log" 2>&1
  done
  echo "pixelated $t"
done

# Density anchors: the best fit of the completion's model form to the TRUE host
# density, per tracer -- not the raw mean density (see measure_density_model.py).
for t in gal agn; do
  python scripts/measure_density_model.py \
    --complete_catalog "data_derived/catalog_${t}_complete.h5" --z_ref 0.30 \
    --out_json "results/density_anchor_${t}.json" > "logs/anchor_${t}.log" 2>&1
  echo "anchored $t"
done
echo "LADDER BUILD DONE"
