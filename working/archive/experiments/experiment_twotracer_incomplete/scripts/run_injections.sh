#!/bin/bash
# One catalog-targeted injection set per rung, with the weights and ndraw that
# calibrate_mixture.py solved for -- the DETECTED-ROW split is what is held fixed
# across the ladder, not the weights (see that script's docstring).
set -euo pipefail
cd "$(dirname "$0")/.."
CAL=results/mixture_calibration.json
for lev in complete m21.0 m20.0 m19.0 m18.0; do
  read -r WP WU WG WA ND <<<"$(python -c "
import json;d=json.load(open('$CAL'))['levels']['$lev'];w=d['weights']
print(w['population'],w['uniform'],w['targeted_gal'],w['targeted_agn'],d['ndraw'])")"
  python scripts/build_targeted_injections_k2.py \
    --out_path "data_derived/inj_${lev}.h5" --ndraw "$ND" --batch_size 4000000 \
    --mix_population "$WP" --mix_uniform "$WU" \
    --target "data_derived/survey_gal_${lev}_ns32.h5:$WG" \
    --target "data_derived/survey_agn_${lev}_ns32.h5:$WA" \
    --seed 74101 --validate_nsamp 300 \
    --validation_json "results/inj_${lev}_validation.json" \
    > "logs/inj_${lev}.log" 2>&1 &
done
wait
echo "ALL INJECTIONS DONE"
