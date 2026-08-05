#!/usr/bin/env bash
# ARM N — the guard-COMPLIANT mixture result: same mock, same estimators, master's
# DEFAULT total-variance budget (1.0), at the event count the budget admits.
#
# The audit shows the N=1000 mixture carries sigma^2_lnL ~ 21 at the fagn0.3 truth
# point, dominated by the per-event PE reweighting term (~0.016/event). That term
# is proportional to N_obs and the selection term to N_obs^2, so the same data
# becomes admissible below N ~ 57 events; N=50 and N=25 stratified subsamples
# (scripts/build_event_subsample.py, host-fraction preserving) are scanned here.
# N=50 is also the gw_agn proof-of-concept target (sigma(alpha)=0.086 forecast).
#
# The subsample truths differ from the parent by rounding and are passed with
# --f_true: N=50 -> 0.30/0.70, N=25 -> 0.32/0.72.
#
# NOTE: sigma^2_PE varies along the f axis (it grows as the mixture leans on the
# sparse AGN component), so a partial guard rejection at high f is expected and
# is a real feature of the estimator on this mock, not a scan failure. Read
# n_neginf_cells in each log and treat summaries over partially-rejected grids as
# truncated posteriors.
set -uo pipefail
cd "$(dirname "$0")"
source ./env.sh

DER=../data_derived

run () {
  local tag=$1; shift
  echo "=== $(date +%H:%M:%S) $tag ==="
  python scan_darksirens.py "$@" > ../logs/${tag}.log 2>&1 || echo "FAILED: $tag"
  grep -hE "^Eval done" ../logs/${tag}.log
}
probe () {
  local tag=$1; shift
  mkdir -p ../results/guard_audit ../logs/guard_audit
  echo "=== $(date +%H:%M:%S) audit $tag ==="
  python diag_variance_guard.py "$@" --out_json ../results/guard_audit/${tag}.json \
    > ../logs/guard_audit/${tag}.log 2>&1 || echo "FAILED: $tag"
  grep -h "^\[guard\]" ../logs/guard_audit/${tag}.log | tail -1
}

declare -A FTRUE_50=( [0.3]=0.30 [0.7]=0.70 )
declare -A FTRUE_25=( [0.3]=0.32 [0.7]=0.72 )

# --- guard probes at the subsample truth points --------------------------------
for K in 0.3 0.7; do
  probe k2_dsf_n50_fagn${K} --universe_model dark_sirens --catalog_sky_weighting field \
    --survey_path ../data/gal.h5 ../data/agn.h5 --gw_path $DER/gw_fagn${K}_n50.h5 \
    --gwselection_path $INJ --log10n0 -12 --log10n0_c2 -12 --f_at ${FTRUE_50[$K]}
  probe k2_dsf_n25_fagn${K} --universe_model dark_sirens --catalog_sky_weighting field \
    --survey_path ../data/gal.h5 ../data/agn.h5 --gw_path $DER/gw_fagn${K}_n25.h5 \
    --gwselection_path $INJ --log10n0 -12 --log10n0_c2 -12 --f_at ${FTRUE_25[$K]}
done

# --- f scans at the DEFAULT budget ---------------------------------------------
for K in 0.3 0.7; do
  run N_fscan_dsf_n50_fagn${K} --universe_model dark_sirens --catalog_sky_weighting field \
    --survey_path ../data/gal.h5 ../data/agn.h5 --gw_path $DER/gw_fagn${K}_n50.h5 \
    --gwselection_path $INJ --scan f --f_grid 0 1 41 --h0_fixed 67.74 \
    --log10n0 -12 --log10n0_c2 -12 --f_true ${FTRUE_50[$K]} \
    --out_tag N_fscan_dsf_n50_fagn${K}
  run N_fscan_dscf_n50_fagn${K} --universe_model dark_sirens_complete \
    --catalog_sky_weighting field --survey_path ../data/gal.h5 ../data/agn.h5 \
    --gw_path $DER/gw_fagn${K}_n50.h5 --gwselection_path $INJ \
    --scan f --f_grid 0 1 41 --h0_fixed 67.74 --f_true ${FTRUE_50[$K]} \
    --out_tag N_fscan_dscf_n50_fagn${K}
  run N_fscan_dsf_n25_fagn${K} --universe_model dark_sirens --catalog_sky_weighting field \
    --survey_path ../data/gal.h5 ../data/agn.h5 --gw_path $DER/gw_fagn${K}_n25.h5 \
    --gwselection_path $INJ --scan f --f_grid 0 1 41 --h0_fixed 67.74 \
    --log10n0 -12 --log10n0_c2 -12 --f_true ${FTRUE_25[$K]} \
    --out_tag N_fscan_dsf_n25_fagn${K}
done

# --- A/B: the SAME subsamples at the legacy-equivalent budget -------------------
# Isolates "what the guard removed" from "what the smaller sample cost".
for K in 0.3 0.7; do
  run NL_fscan_dsf_n50_fagn${K} --universe_model dark_sirens --catalog_sky_weighting field \
    --survey_path ../data/gal.h5 ../data/agn.h5 --gw_path $DER/gw_fagn${K}_n50.h5 \
    --gwselection_path $INJ --scan f --f_grid 0 1 41 --h0_fixed 67.74 \
    --log10n0 -12 --log10n0_c2 -12 --f_true ${FTRUE_50[$K]} \
    --max_likelihood_variance $LEGACY_VAR --out_tag NL_fscan_dsf_n50_fagn${K}
done

# --- joint (H0,f) at N=50, default budget --------------------------------------
run N_joint_dsf_n50_fagn0.3 --universe_model dark_sirens --catalog_sky_weighting field \
  --survey_path ../data/gal.h5 ../data/agn.h5 --gw_path $DER/gw_fagn0.3_n50.h5 \
  --gwselection_path $INJ --scan joint --h0_grid 50 100 61 --f_grid 0 1 41 \
  --log10n0 -12 --log10n0_c2 -12 --h0_true 67.74 --f_true 0.30 \
  --out_tag N_joint_dsf_n50_fagn0.3

echo "=== $(date +%H:%M:%S) ARM N DONE ==="
python aggregate_guard_audit.py --audit_dir ../results/guard_audit \
  --out_json ../results/guard_audit_summary.json --out_md ../GUARD_AUDIT.md
