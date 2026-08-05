#!/usr/bin/env bash
# Phase C production scans — sequential on one GPU. ~2h.
# All runs: DARKSIRENS_ZMAX=1.5 (matched to the mock's z_max), fixed fiducial population,
# Om0 pinned, nuisances at truth (log10n0 = true density, delta=0, b_miss=1, sigma_kde=0).
set -uo pipefail
cd "$(dirname "$0")"
export DARKSIRENS_ZMAX=1.5
export XLA_PYTHON_CLIENT_PREALLOCATE=false

LOG10N0_GAL=-5.50627668499162
LOG10N0_AGN=-7.508083961432144
LOG10N0_GAL_Z1=-5.51390382309138
LOG10N0_AGN_Z1=-7.5143989402634475
INJ=../data/injections.h5
INJB=../data/injections_B.h5

run () { echo "=== $(date +%H:%M:%S) $1 ==="; shift; python scan_darksirens.py "$@" || echo "FAILED: $*"; }

# --- 1. f scans, K=2, 41 pts, H0=truth, all four recovery sets -------------
for K in 0.0 0.3 0.7 1.0; do
  run "fscan_fagn${K}" --universe_model dark_sirens \
    --survey_path ../data/gal.h5 ../data/agn.h5 \
    --gw_path ../data/gw_fagn${K}.h5 --gwselection_path $INJ \
    --scan f --f_grid 0 1 41 --h0_fixed 67.74 \
    --log10n0 $LOG10N0_GAL --log10n0_c2 $LOG10N0_AGN \
    --out_tag fscan_fagn${K} > ../logs/c_fscan_${K}.log 2>&1
done

# --- 2. A/B levers on the fagn0.3 f scan -----------------------------------
run "fscan_fagn0.3_injB" --universe_model dark_sirens \
  --survey_path ../data/gal.h5 ../data/agn.h5 \
  --gw_path ../data/gw_fagn0.3.h5 --gwselection_path $INJB \
  --scan f --f_grid 0 1 41 --h0_fixed 67.74 \
  --log10n0 $LOG10N0_GAL --log10n0_c2 $LOG10N0_AGN \
  --out_tag fscan_fagn0.3_injB > ../logs/c_fscan_03_injB.log 2>&1

run "fscan_fagn0.3_zlt1" --universe_model dark_sirens \
  --survey_path ../data/gal_zlt1.h5 ../data/agn_zlt1.h5 \
  --gw_path ../data/gw_fagn0.3.h5 --gwselection_path $INJ \
  --scan f --f_grid 0 1 41 --h0_fixed 67.74 \
  --log10n0 $LOG10N0_GAL_Z1 --log10n0_c2 $LOG10N0_AGN_Z1 \
  --out_tag fscan_fagn0.3_zlt1 > ../logs/c_fscan_03_zlt1.log 2>&1

# --- 3. H0 scans, K=2 mixture, fagn0.3: at f=truth and at f=1 --------------
run "h0_k2_f0.307" --universe_model dark_sirens \
  --survey_path ../data/gal.h5 ../data/agn.h5 \
  --gw_path ../data/gw_fagn0.3.h5 --gwselection_path $INJ \
  --scan h0 --h0_grid 50 100 61 --f_fixed 0.307 \
  --log10n0 $LOG10N0_GAL --log10n0_c2 $LOG10N0_AGN --h0_true 67.74 \
  --out_tag h0_k2_fagn0.3_ftruth > ../logs/c_h0_k2_ft.log 2>&1
run "h0_k2_f1.0" --universe_model dark_sirens \
  --survey_path ../data/gal.h5 ../data/agn.h5 \
  --gw_path ../data/gw_fagn0.3.h5 --gwselection_path $INJ \
  --scan h0 --h0_grid 50 100 61 --f_fixed 1.0 \
  --log10n0 $LOG10N0_GAL --log10n0_c2 $LOG10N0_AGN --h0_true 67.74 \
  --out_tag h0_k2_fagn0.3_f1 > ../logs/c_h0_k2_f1.log 2>&1

# --- 4. Joint (H0, f) grids, K=2: fagn0.3 and fagn0.7, 61x41 ---------------
for K in 0.3 0.7; do
  run "joint_fagn${K}" --universe_model dark_sirens \
    --survey_path ../data/gal.h5 ../data/agn.h5 \
    --gw_path ../data/gw_fagn${K}.h5 --gwselection_path $INJ \
    --scan joint --h0_grid 50 100 61 --f_grid 0 1 41 \
    --log10n0 $LOG10N0_GAL --log10n0_c2 $LOG10N0_AGN --h0_true 67.74 \
    --out_tag joint_fagn${K} > ../logs/c_joint_${K}.log 2>&1
done

# --- 5. Per-tracer H0 scans, 61 pts, both models, r00-r09 ------------------
for R in 00 01 02 03 04 05 06 07 08 09; do
  run "h0_ds_gal_r${R}" --universe_model dark_sirens --survey_path ../data/gal.h5 \
    --gw_path ../data/gw_cov_gal_r${R}.h5 --gwselection_path $INJ \
    --scan h0 --h0_grid 50 100 61 --log10n0 $LOG10N0_GAL --h0_true 67.74 \
    --out_tag h0_ds_gal_r${R} > ../logs/c_h0_ds_gal_r${R}.log 2>&1
  run "h0_dsc_gal_r${R}" --universe_model dark_sirens_complete --survey_path ../data/gal.h5 \
    --gw_path ../data/gw_cov_gal_r${R}.h5 --gwselection_path $INJ \
    --scan h0 --h0_grid 50 100 61 --h0_true 67.74 \
    --out_tag h0_dsc_gal_r${R} > ../logs/c_h0_dsc_gal_r${R}.log 2>&1
  run "h0_ds_agn_r${R}" --universe_model dark_sirens --survey_path ../data/agn.h5 \
    --gw_path ../data/gw_cov_agn_r${R}.h5 --gwselection_path $INJ \
    --scan h0 --h0_grid 50 100 61 --log10n0 $LOG10N0_AGN --h0_true 67.74 \
    --out_tag h0_ds_agn_r${R} > ../logs/c_h0_ds_agn_r${R}.log 2>&1
  run "h0_dsc_agn_r${R}" --universe_model dark_sirens_complete --survey_path ../data/agn.h5 \
    --gw_path ../data/gw_cov_agn_r${R}.h5 --gwselection_path $INJ \
    --scan h0 --h0_grid 50 100 61 --h0_true 67.74 \
    --out_tag h0_dsc_agn_r${R} > ../logs/c_h0_dsc_agn_r${R}.log 2>&1
done

echo "=== $(date +%H:%M:%S) PHASE C DONE ==="
