#!/usr/bin/env bash
# Guard-budget audit: one likelihood evaluation per configuration, with the
# total-variance guard instrumented, so we learn for EVERY configuration in this
# campaign whether darksirens master @ 2b86a2d admits it at the default
# GWTC-4.0/5.0 budget (max_likelihood_variance = 1.0) and, if not, by how much
# it is over.  This is the cheap replacement for running every scan twice: the
# guard verdict is a property of (data, model, sky weighting), essentially
# constant across a scan grid.
#
# Output: ../results/guard_audit/<tag>.json (guard_records carry Neff,
# pe_variance_sum, sigma2_total, threshold, pass flags).
set -uo pipefail
cd "$(dirname "$0")"
source ./env.sh

OUT=../results/guard_audit
mkdir -p "$OUT" ../logs/guard_audit

probe () {
  local tag=$1; shift
  echo "=== $(date +%H:%M:%S) audit $tag ==="
  python diag_variance_guard.py "$@" --out_json "$OUT/${tag}.json" \
    > ../logs/guard_audit/${tag}.log 2>&1 || echo "FAILED: $tag"
  grep -h "^\[guard\]" ../logs/guard_audit/${tag}.log | tail -2
}

# --- K=2 mixture configurations (N=1000 events) -------------------------------
# Probed at the planted alpha_AGN truth of each event set (eligible-pool truth).
declare -A FTRUE=( [0.0]=0.0099 [0.3]=0.307 [0.7]=0.703 [1.0]=1.0 )
for K in 0.0 0.3 0.7 1.0; do
  probe k2_dscf_fagn${K} --universe_model dark_sirens_complete \
    --catalog_sky_weighting field --survey_path ../data/gal.h5 ../data/agn.h5 \
    --gw_path ../data/gw_fagn${K}.h5 --gwselection_path $INJ --f_at ${FTRUE[$K]}
done
for K in 0.0 0.3 0.7 1.0; do
  probe k2_dsf_n0low_fagn${K} --universe_model dark_sirens \
    --catalog_sky_weighting field --survey_path ../data/gal.h5 ../data/agn.h5 \
    --gw_path ../data/gw_fagn${K}.h5 --gwselection_path $INJ \
    --log10n0 -12 --log10n0_c2 -12 --f_at ${FTRUE[$K]}
done
probe k2_dsf_n0true_fagn0.3 --universe_model dark_sirens \
  --catalog_sky_weighting field --survey_path ../data/gal.h5 ../data/agn.h5 \
  --gw_path ../data/gw_fagn0.3.h5 --gwselection_path $INJ \
  --log10n0 $LOG10N0_GAL --log10n0_c2 $LOG10N0_AGN --f_at 0.307
probe k2_dscf_fagn0.3_injB --universe_model dark_sirens_complete \
  --catalog_sky_weighting field --survey_path ../data/gal.h5 ../data/agn.h5 \
  --gw_path ../data/gw_fagn0.3.h5 --gwselection_path $INJB --f_at 0.307

# --- K=1 per-tracer coverage configurations (N=100 events each) ---------------
for R in 00 01 02 03 04 05 06 07 08 09; do
  for T in gal agn; do
    probe k1_dscf_${T}_r${R} --universe_model dark_sirens_complete \
      --catalog_sky_weighting field --survey_path ../data/${T}.h5 \
      --gw_path ../data/gw_cov_${T}_r${R}.h5 --gwselection_path $INJ
  done
done
for R in 00 01 02 03 04; do
  for T in gal agn; do
    probe k1_dsf_${T}_r${R} --universe_model dark_sirens \
      --catalog_sky_weighting field --survey_path ../data/${T}.h5 \
      --gw_path ../data/gw_cov_${T}_r${R}.h5 --gwselection_path $INJ --log10n0 -12
  done
done

# --- Isotropic (original, non-catalog-targeted) injections, for the record ----
probe k2_dscf_fagn0.3_isoinj --universe_model dark_sirens_complete \
  --catalog_sky_weighting field --survey_path ../data/gal.h5 ../data/agn.h5 \
  --gw_path ../data/gw_fagn0.3.h5 --gwselection_path ../data/injections.h5 --f_at 0.307

echo "=== $(date +%H:%M:%S) GUARD AUDIT DONE ==="
python aggregate_guard_audit.py --audit_dir "$OUT" \
  --out_json ../results/guard_audit_summary.json --out_md ../GUARD_AUDIT.md
