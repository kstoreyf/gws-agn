#!/usr/bin/env bash
# ARM L — full production scans on darksirens master @ 2b86a2d with the
# post-#212 total-variance criterion made INERT (--max_likelihood_variance 1e6,
# which collapses the threshold to the legacy `Neff > 5*N_obs` floor exactly).
#
# Purpose: isolate the effect of the 295 commits between the #212-era code the
# previous run used and current master.  Same inputs, same estimators, same
# grids, same nuisance points as ../../gw_agn_darksirens_fixed/scripts/
# run_production2.sh -> the numbers are directly comparable, and any shift is
# attributable to the code, not to the guard.
#
# Tags are prefixed `L_` (legacy-equivalent budget).  ARM G (the default,
# guard-respecting budget) is run_production_default.sh.
set -uo pipefail
cd "$(dirname "$0")"
source ./env.sh

run () {
  local tag=$1; shift
  echo "=== $(date +%H:%M:%S) $tag ==="
  python scan_darksirens.py "$@" --max_likelihood_variance $LEGACY_VAR \
    > ../logs/${tag}.log 2>&1 || echo "FAILED: $tag"
  grep -hE "^Eval done" ../logs/${tag}.log
}

# --- A. f scans, dark_sirens_complete FIELD K=2, 41 pts ------------------------
for K in 0.0 0.3 0.7 1.0; do
  run L_fscan_dscf_fagn${K} --universe_model dark_sirens_complete \
    --catalog_sky_weighting field --survey_path ../data/gal.h5 ../data/agn.h5 \
    --gw_path ../data/gw_fagn${K}.h5 --gwselection_path $INJ \
    --scan f --f_grid 0 1 41 --h0_fixed 67.74 --out_tag L_fscan_dscf_fagn${K}
done
run L_fscan_dscf_fagn0.3_injB --universe_model dark_sirens_complete \
  --catalog_sky_weighting field --survey_path ../data/gal.h5 ../data/agn.h5 \
  --gw_path ../data/gw_fagn0.3.h5 --gwselection_path $INJB \
  --scan f --f_grid 0 1 41 --h0_fixed 67.74 --out_tag L_fscan_dscf_fagn0.3_injB

# --- B. f scans, dark_sirens FIELD K=2 at the complete-catalog n0 limit --------
# `dsf` is the primary estimator (matches gw_agn's field construction). The
# previous run only scanned 0.3/0.7 here; the full ladder is run now.
for K in 0.0 0.3 0.7 1.0; do
  run L_fscan_dsf_n0low_fagn${K} --universe_model dark_sirens \
    --catalog_sky_weighting field --survey_path ../data/gal.h5 ../data/agn.h5 \
    --gw_path ../data/gw_fagn${K}.h5 --gwselection_path $INJ \
    --scan f --f_grid 0 1 41 --h0_fixed 67.74 --log10n0 -12 --log10n0_c2 -12 \
    --out_tag L_fscan_dsf_n0low_fagn${K}
done
run L_fscan_dsf_n0low_fagn0.3_injB --universe_model dark_sirens \
  --catalog_sky_weighting field --survey_path ../data/gal.h5 ../data/agn.h5 \
  --gw_path ../data/gw_fagn0.3.h5 --gwselection_path $INJB \
  --scan f --f_grid 0 1 41 --h0_fixed 67.74 --log10n0 -12 --log10n0_c2 -12 \
  --out_tag L_fscan_dsf_n0low_fagn0.3_injB
# true-n0 model-misspecification demo (budgets missing AGN into empty pixels)
run L_fscan_dsf_n0true_fagn0.3 --universe_model dark_sirens \
  --catalog_sky_weighting field --survey_path ../data/gal.h5 ../data/agn.h5 \
  --gw_path ../data/gw_fagn0.3.h5 --gwselection_path $INJ \
  --scan f --f_grid 0 1 41 --h0_fixed 67.74 \
  --log10n0 $LOG10N0_GAL --log10n0_c2 $LOG10N0_AGN \
  --out_tag L_fscan_dsf_n0true_fagn0.3

# --- C. H0 scans ---------------------------------------------------------------
run L_h0_dscf_k2_fagn0.3 --universe_model dark_sirens_complete \
  --catalog_sky_weighting field --survey_path ../data/gal.h5 ../data/agn.h5 \
  --gw_path ../data/gw_fagn0.3.h5 --gwselection_path $INJ \
  --scan h0 --h0_grid 50 100 61 --f_fixed 0.307 --h0_true 67.74 \
  --out_tag L_h0_dscf_k2_fagn0.3
run L_h0_dsf_k2_fagn0.3 --universe_model dark_sirens \
  --catalog_sky_weighting field --survey_path ../data/gal.h5 ../data/agn.h5 \
  --gw_path ../data/gw_fagn0.3.h5 --gwselection_path $INJ \
  --scan h0 --h0_grid 50 100 61 --f_fixed 0.307 --h0_true 67.74 \
  --log10n0 -12 --log10n0_c2 -12 --out_tag L_h0_dsf_k2_fagn0.3
# per-tracer K=1 coverage realizations
for R in 00 01 02 03 04 05 06 07 08 09; do
  for T in gal agn; do
    run L_h0_dscf_${T}_r${R} --universe_model dark_sirens_complete \
      --catalog_sky_weighting field --survey_path ../data/${T}.h5 \
      --gw_path ../data/gw_cov_${T}_r${R}.h5 --gwselection_path $INJ \
      --scan h0 --h0_grid 50 100 61 --h0_true 67.74 --out_tag L_h0_dscf_${T}_r${R}
  done
done
for R in 00 01 02 03 04; do
  for T in gal agn; do
    run L_h0_dsf_${T}_r${R} --universe_model dark_sirens \
      --catalog_sky_weighting field --survey_path ../data/${T}.h5 \
      --gw_path ../data/gw_cov_${T}_r${R}.h5 --gwselection_path $INJ \
      --scan h0 --h0_grid 50 100 61 --log10n0 -12 --h0_true 67.74 \
      --out_tag L_h0_dsf_${T}_r${R}
  done
done

# --- D. Joint (H0, f) 61x41 ----------------------------------------------------
for K in 0.3 0.7; do
  run L_joint_dsf_fagn${K} --universe_model dark_sirens \
    --catalog_sky_weighting field --survey_path ../data/gal.h5 ../data/agn.h5 \
    --gw_path ../data/gw_fagn${K}.h5 --gwselection_path $INJ \
    --scan joint --h0_grid 50 100 61 --f_grid 0 1 41 --h0_true 67.74 \
    --log10n0 -12 --log10n0_c2 -12 --out_tag L_joint_dsf_fagn${K}
done
for K in 0.3 0.7; do
  run L_joint_dscf_fagn${K} --universe_model dark_sirens_complete \
    --catalog_sky_weighting field --survey_path ../data/gal.h5 ../data/agn.h5 \
    --gw_path ../data/gw_fagn${K}.h5 --gwselection_path $INJ \
    --scan joint --h0_grid 50 100 61 --f_grid 0 1 41 --h0_true 67.74 \
    --out_tag L_joint_dscf_fagn${K}
done

echo "=== $(date +%H:%M:%S) ARM L DONE ==="
