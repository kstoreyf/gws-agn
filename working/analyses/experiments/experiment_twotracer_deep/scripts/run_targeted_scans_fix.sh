#!/bin/bash
# Post-fix rerun of the deep two-tracer scans on the sigma_ang-FIXED events
# (twotracer_gw_events_fix.h5, darksirens PR #335 sequential observable sky width).
# Truth/detected set verified bit-identical to the pre-fix mock
# (results/events_fix_check.json), so the targeted injection set and survey files
# are reused unchanged. Same estimator, grids and guard as run_targeted_scans.sh;
# only the events file (and out tags, suffixed _fix) differ.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/hildafs/home/magana/tmp_ondemand_hildafs_phy230014p_symlink/magana/.conda/envs/jax/bin/python
SEL=data_derived/injections_targeted_k2.h5
SUR="data_derived/survey_gal_ns32.h5 data_derived/survey_agn_ns32.h5"
COMMON="--universe_model dark_sirens --catalog_sky_weighting field \
  --survey_path $SUR --gwselection_path $SEL \
  --log10n0 -12 --log10n0_c2 -12 \
  --selection_neff_guard hard --max_likelihood_variance 1e6 \
  --outdir results --h0_true 67.74 --f_true 0.3"

for f in 0.0 0.3 0.7 1.0; do
  $PY scripts/diag_variance_guard.py --universe_model dark_sirens \
    --catalog_sky_weighting field --survey_path $SUR \
    --gw_path data_derived/twotracer_gw_events_fix.h5 \
    --gwselection_path $SEL --log10n0 -12 --log10n0_c2 -12 \
    --f_at $f --max_likelihood_variance 1e6 \
    --out_json "results/guard_targeted_f${f}_fix.json" \
    > "logs/guard_targeted_f${f}_fix.log" 2>&1
done
echo "guards done"

$PY scripts/scan_h0f.py $COMMON --gw_path data_derived/twotracer_n80_fix.h5 \
  --scan f --f_grid 0.0 1.0 41 --out_tag tgt_fscan_n80_fix > logs/tgt_fscan_n80_fix.log 2>&1
$PY scripts/scan_h0f.py $COMMON --gw_path data_derived/twotracer_gw_events_fix.h5 \
  --scan f --f_grid 0.0 1.0 41 --out_tag tgt_fscan_n200_fix > logs/tgt_fscan_n200_fix.log 2>&1
$PY scripts/scan_h0f.py $COMMON --gw_path data_derived/twotracer_gw_events_fix.h5 \
  --scan joint --h0_grid 58.0 78.0 81 --f_grid 0.0 1.0 41 \
  --out_tag tgt_joint_n200_fix > logs/tgt_joint_n200_fix.log 2>&1
echo "DEEP FIX SCANS DONE"
