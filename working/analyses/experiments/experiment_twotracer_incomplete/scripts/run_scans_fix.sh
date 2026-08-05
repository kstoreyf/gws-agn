#!/bin/bash
# Post-fix rerun of the completeness ladder on the sigma_ang-FIXED events
# (darksirens PR #335). Surveys, density anchors and per-rung targeted injection
# sets are reused unchanged: the fix touches only the events' PE (truth/detected
# set verified bit-identical, ../experiment_twotracer_deep/results/events_fix_check.json).
# Identical to run_scans.sh + run_null.sh apart from the events file and _fix tags.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/hildafs/home/magana/tmp_ondemand_hildafs_phy230014p_symlink/magana/.conda/envs/jax/bin/python
GW=../experiment_twotracer_deep/data_derived/twotracer_gw_events_fix.h5
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
  $PY scripts/diag_variance_guard.py --universe_model dark_sirens \
    --catalog_sky_weighting field --survey_path $SUR --gw_path $GW \
    --gwselection_path "$SEL" --log10n0 $N0_GAL --log10n0_c2 $N0_AGN \
    --f_at 0.3 --max_likelihood_variance 1e6 \
    --out_json "results/guard_${lev}_fix.json" > "logs/guard_${lev}_fix.log" 2>&1
  $PY scripts/scan_h0f.py $COMMON --survey_path $SUR --gwselection_path "$SEL" \
    --nuisance_json "$NUIS" --scan f --f_grid 0.0 1.0 41 \
    --out_tag "fscan_${lev}_fix" > "logs/fscan_${lev}_fix.log" 2>&1
  $PY scripts/scan_h0f.py $COMMON --survey_path $SUR --gwselection_path "$SEL" \
    --nuisance_json "$NUIS" --scan joint --h0_grid 58.0 78.0 81 --f_grid 0.0 1.0 41 \
    --out_tag "joint_${lev}_fix" > "logs/joint_${lev}_fix.log" 2>&1
  echo "scanned $lev"
done

# Sky-shuffle null on the fixed events (same rungs as run_null.sh).
GWN=data_derived/events_skyshuffled_fix.h5
NCOMMON="--universe_model dark_sirens --catalog_sky_weighting field --gw_path $GWN \
  --log10n0 $N0_GAL --log10n0_c2 $N0_AGN \
  --nuisance_json {\"delta\":0.0194,\"delta_c2\":-0.0031} \
  --selection_neff_guard hard --max_likelihood_variance 1e6 \
  --outdir results --h0_true 67.74 --f_true 0.3 --scan f --f_grid 0.0 1.0 41"
for lev in complete m20.0 m18.0; do
  $PY scripts/scan_h0f.py $NCOMMON \
    --survey_path "data_derived/survey_gal_${lev}_ns32.h5" "data_derived/survey_agn_${lev}_ns32.h5" \
    --gwselection_path "data_derived/inj_${lev}.h5" \
    --out_tag "fscan_null_${lev}_fix" > "logs/fscan_null_${lev}_fix.log" 2>&1
  echo "null $lev"
done
echo "LADDER FIX SCANS DONE"
