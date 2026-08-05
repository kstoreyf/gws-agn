#!/bin/bash
# Re-run the deep two-tracer scans on the catalog-targeted injection lane.
# Arm N80 is the like-for-like control against the population+uniform run
# (same 80-event subsample, only the injection set changed); the N200 arms are
# the measurement the targeting unblocks.
set -euo pipefail
cd "$(dirname "$0")/.."
SEL=data_derived/injections_targeted_k2.h5
SUR="data_derived/survey_gal_ns32.h5 data_derived/survey_agn_ns32.h5"
COMMON="--universe_model dark_sirens --catalog_sky_weighting field \
  --survey_path $SUR --gwselection_path $SEL \
  --log10n0 -12 --log10n0_c2 -12 \
  --selection_neff_guard hard --max_likelihood_variance 1e6 \
  --outdir results --h0_true 67.74 --f_true 0.3"

python scripts/scan_h0f.py $COMMON --gw_path data_derived/twotracer_n80.h5 \
  --scan f --f_grid 0.0 1.0 41 --out_tag tgt_fscan_n80 > logs/tgt_fscan_n80.log 2>&1
python scripts/scan_h0f.py $COMMON --gw_path data_derived/twotracer_gw_events.h5 \
  --scan f --f_grid 0.0 1.0 41 --out_tag tgt_fscan_n200 > logs/tgt_fscan_n200.log 2>&1
python scripts/scan_h0f.py $COMMON --gw_path data_derived/twotracer_gw_events.h5 \
  --scan joint --h0_grid 58.0 78.0 81 --f_grid 0.0 1.0 41 \
  --out_tag tgt_joint_n200 > logs/tgt_joint_n200.log 2>&1
echo "ALL SCANS DONE"
