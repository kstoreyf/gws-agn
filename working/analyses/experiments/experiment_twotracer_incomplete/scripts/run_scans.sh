#!/bin/bash
# (H0, f_AGN) across the completeness ladder.
# Same estimator, same events, same guard at every rung -- only the survey files,
# their matched injection set and nothing else change.
set -euo pipefail
cd "$(dirname "$0")/.."
GW=../experiment_twotracer_deep/data_derived/twotracer_gw_events.h5
N0_GAL=-5.806380
N0_AGN=-7.720033
NUIS='{"delta": 0.0194, "delta_c2": -0.0031}'
COMMON="--universe_model dark_sirens --catalog_sky_weighting field \
  --gw_path $GW --log10n0 $N0_GAL --log10n0_c2 $N0_AGN \
  --selection_neff_guard hard --max_likelihood_variance 1e6 \
  --outdir results --h0_true 67.74 --f_true 0.3"

for lev in complete m21.0 m20.0 m19.0 m18.0; do
  SUR="data_derived/survey_gal_${lev}_ns32.h5 data_derived/survey_agn_${lev}_ns32.h5"
  SEL="data_derived/inj_${lev}.h5"
  python scripts/diag_variance_guard.py --universe_model dark_sirens \
    --catalog_sky_weighting field --survey_path $SUR --gw_path $GW \
    --gwselection_path "$SEL" --log10n0 $N0_GAL --log10n0_c2 $N0_AGN \
    --f_at 0.3 --max_likelihood_variance 1e6 \
    --out_json "results/guard_${lev}.json" > "logs/guard_${lev}.log" 2>&1
  python scripts/scan_h0f.py $COMMON --survey_path $SUR --gwselection_path "$SEL" \
    --nuisance_json "$NUIS" --scan f --f_grid 0.0 1.0 41 \
    --out_tag "fscan_${lev}" > "logs/fscan_${lev}.log" 2>&1
  python scripts/scan_h0f.py $COMMON --survey_path $SUR --gwselection_path "$SEL" \
    --nuisance_json "$NUIS" --scan joint --h0_grid 58.0 78.0 81 --f_grid 0.0 1.0 41 \
    --out_tag "joint_${lev}" > "logs/joint_${lev}.log" 2>&1
  echo "scanned $lev"
done
echo "ALL SCANS DONE"
