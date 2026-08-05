#!/usr/bin/env bash
# ARM G — scans at darksirens master's DEFAULT total-variance budget
# (max_likelihood_variance = 1.0, the GWTC-4.0/5.0 bound on sigma^2_lnL).
#
# The guard audit (../GUARD_AUDIT.md) measures which configurations the default
# budget admits on this campaign's mock.  Only the DENSE-tracer K=1 sets clear it
# (GAL, N=100: sigma^2 ~ 0.5); the sparse AGN tracer (~9) and every N=1000 K=2
# mixture (~14-48) are over budget and every grid cell returns -inf.  So Arm G
# runs the admitted configurations in full, plus three deliberately-rejected
# configurations recorded as first-class -inf outcomes for the audit trail.
#
# Note: for an ADMITTED cell the guard is a pure gate — it does not enter the
# returned logL — so Arm G and Arm L agree cell-by-cell wherever both admit.
# That is what makes the Arm-L numbers legitimate as a code-drift comparison.
set -uo pipefail
cd "$(dirname "$0")"
source ./env.sh

run () {
  local tag=$1; shift
  echo "=== $(date +%H:%M:%S) $tag ==="
  python scan_darksirens.py "$@" > ../logs/${tag}.log 2>&1 || echo "FAILED: $tag"
  grep -hE "^Eval done" ../logs/${tag}.log
}

# --- Admitted: dense-tracer (GAL) K=1 H0 coverage scans ------------------------
for R in 00 01 02 03 04 05 06 07 08 09; do
  run G_h0_dscf_gal_r${R} --universe_model dark_sirens_complete \
    --catalog_sky_weighting field --survey_path ../data/gal.h5 \
    --gw_path ../data/gw_cov_gal_r${R}.h5 --gwselection_path $INJ \
    --scan h0 --h0_grid 50 100 61 --h0_true 67.74 --out_tag G_h0_dscf_gal_r${R}
done
for R in 00 01 02 03 04; do
  run G_h0_dsf_gal_r${R} --universe_model dark_sirens \
    --catalog_sky_weighting field --survey_path ../data/gal.h5 \
    --gw_path ../data/gw_cov_gal_r${R}.h5 --gwselection_path $INJ \
    --scan h0 --h0_grid 50 100 61 --log10n0 -12 --h0_true 67.74 \
    --out_tag G_h0_dsf_gal_r${R}
done

# --- Rejected-by-design records (expected: all cells -inf) ---------------------
run G_h0_dscf_agn_r00 --universe_model dark_sirens_complete \
  --catalog_sky_weighting field --survey_path ../data/agn.h5 \
  --gw_path ../data/gw_cov_agn_r00.h5 --gwselection_path $INJ \
  --scan h0 --h0_grid 50 100 61 --h0_true 67.74 --out_tag G_h0_dscf_agn_r00
run G_fscan_dscf_fagn0.3 --universe_model dark_sirens_complete \
  --catalog_sky_weighting field --survey_path ../data/gal.h5 ../data/agn.h5 \
  --gw_path ../data/gw_fagn0.3.h5 --gwselection_path $INJ \
  --scan f --f_grid 0 1 41 --h0_fixed 67.74 --out_tag G_fscan_dscf_fagn0.3
run G_fscan_dsf_n0low_fagn0.3 --universe_model dark_sirens \
  --catalog_sky_weighting field --survey_path ../data/gal.h5 ../data/agn.h5 \
  --gw_path ../data/gw_fagn0.3.h5 --gwselection_path $INJ \
  --scan f --f_grid 0 1 41 --h0_fixed 67.74 --log10n0 -12 --log10n0_c2 -12 \
  --out_tag G_fscan_dsf_n0low_fagn0.3

echo "=== $(date +%H:%M:%S) ARM G DONE ==="
