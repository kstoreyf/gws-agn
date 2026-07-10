#!/usr/bin/env bash
# Fixed-campaign production scans — darksirens @ origin/master (post-#212), field mode.
# Primary estimator: dark_sirens_complete K=2 FIELD (matched to the complete catalogs).
# Secondary: dark_sirens K=2 FIELD at the complete-catalog n0 limit (-12) + true-n0 demo.
set -uo pipefail
cd "$(dirname "$0")"
export DARKSIRENS_ZMAX=1.5
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH=/tmp/claude-88592/-hildafs-projects-phy230014p-magana-gws-agn/89590650-74f5-413e-8311-7f0160636741/scratchpad/wt-master

LOG10N0_GAL=-5.50627668499162
LOG10N0_AGN=-7.508083961432144
INJ=../data/injections.h5
INJB=../data/injections_B.h5

run () { echo "=== $(date +%H:%M:%S) $1 ==="; T=$1; shift; python scan_darksirens.py "$@" > ../logs/${T}.log 2>&1 || echo "FAILED: $T"; }

# --- A. f scans, dsc-field K2 (primary), 41 pts, soft guard --------------------
for K in 0.0 0.3 0.7 1.0; do
  run fscan_dscf_fagn${K} --universe_model dark_sirens_complete --catalog_sky_weighting field \
    --selection_neff_guard soft --survey_path ../data/gal.h5 ../data/agn.h5 \
    --gw_path ../data/gw_fagn${K}.h5 --gwselection_path $INJ \
    --scan f --f_grid 0 1 41 --h0_fixed 67.74 --out_tag fscan_dscf_fagn${K}
done
run fscan_dscf_fagn0.3_injB --universe_model dark_sirens_complete --catalog_sky_weighting field \
  --selection_neff_guard soft --survey_path ../data/gal.h5 ../data/agn.h5 \
  --gw_path ../data/gw_fagn0.3.h5 --gwselection_path $INJB \
  --scan f --f_grid 0 1 41 --h0_fixed 67.74 --out_tag fscan_dscf_fagn0.3_injB

# --- B. f scans, ds-field K2: n0->complete-limit cross-check + true-n0 demo ----
for K in 0.3 0.7; do
  run fscan_dsf_n0low_fagn${K} --universe_model dark_sirens --catalog_sky_weighting field \
    --selection_neff_guard soft --survey_path ../data/gal.h5 ../data/agn.h5 \
    --gw_path ../data/gw_fagn${K}.h5 --gwselection_path $INJ \
    --scan f --f_grid 0 1 41 --h0_fixed 67.74 --log10n0 -12 --log10n0_c2 -12 \
    --out_tag fscan_dsf_n0low_fagn${K}
done
run fscan_dsf_n0true_fagn0.3 --universe_model dark_sirens --catalog_sky_weighting field \
  --selection_neff_guard soft --survey_path ../data/gal.h5 ../data/agn.h5 \
  --gw_path ../data/gw_fagn0.3.h5 --gwselection_path $INJ \
  --scan f --f_grid 0 1 41 --h0_fixed 67.74 --log10n0 $LOG10N0_GAL --log10n0_c2 $LOG10N0_AGN \
  --out_tag fscan_dsf_n0true_fagn0.3

# --- C. H0 scans ----------------------------------------------------------------
run h0_dscf_k2_fagn0.3 --universe_model dark_sirens_complete --catalog_sky_weighting field \
  --selection_neff_guard soft --survey_path ../data/gal.h5 ../data/agn.h5 \
  --gw_path ../data/gw_fagn0.3.h5 --gwselection_path $INJ \
  --scan h0 --h0_grid 50 100 61 --f_fixed 0.307 --h0_true 67.74 --out_tag h0_dscf_k2_fagn0.3
for R in 00 01 02 03 04 05 06 07 08 09; do
  run h0_dscf_gal_r${R} --universe_model dark_sirens_complete --catalog_sky_weighting field \
    --survey_path ../data/gal.h5 --gw_path ../data/gw_cov_gal_r${R}.h5 --gwselection_path $INJ \
    --scan h0 --h0_grid 50 100 61 --h0_true 67.74 --out_tag h0_dscf_gal_r${R}
  run h0_dscf_agn_r${R} --universe_model dark_sirens_complete --catalog_sky_weighting field \
    --survey_path ../data/agn.h5 --gw_path ../data/gw_cov_agn_r${R}.h5 --gwselection_path $INJ \
    --scan h0 --h0_grid 50 100 61 --h0_true 67.74 --out_tag h0_dscf_agn_r${R}
done
for R in 00 01 02 03 04; do
  run h0_dsf_gal_r${R} --universe_model dark_sirens --catalog_sky_weighting field \
    --survey_path ../data/gal.h5 --gw_path ../data/gw_cov_gal_r${R}.h5 --gwselection_path $INJ \
    --scan h0 --h0_grid 50 100 61 --log10n0 -12 --h0_true 67.74 --out_tag h0_dsf_gal_r${R}
  run h0_dsf_agn_r${R} --universe_model dark_sirens --catalog_sky_weighting field \
    --survey_path ../data/agn.h5 --gw_path ../data/gw_cov_agn_r${R}.h5 --gwselection_path $INJ \
    --scan h0 --h0_grid 50 100 61 --log10n0 -12 --h0_true 67.74 --out_tag h0_dsf_agn_r${R}
done

# --- D. Joint (H0, f) 61x41, dsc-field K2, soft guard ---------------------------
for K in 0.3 0.7; do
  run joint_dscf_fagn${K} --universe_model dark_sirens_complete --catalog_sky_weighting field \
    --selection_neff_guard soft --survey_path ../data/gal.h5 ../data/agn.h5 \
    --gw_path ../data/gw_fagn${K}.h5 --gwselection_path $INJ \
    --scan joint --h0_grid 50 100 61 --f_grid 0 1 41 --h0_true 67.74 --out_tag joint_dscf_fagn${K}
done

echo "=== $(date +%H:%M:%S) FIXED PRODUCTION DONE ==="
