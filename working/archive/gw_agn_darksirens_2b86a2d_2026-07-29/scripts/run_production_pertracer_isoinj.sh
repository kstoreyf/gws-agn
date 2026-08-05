#!/usr/bin/env bash
# ARM LI — per-tracer K=1 H0 coverage scans with the ORIGINAL ISOTROPIC injection
# set (../data/injections.h5), legacy-equivalent variance budget.
#
# WHY THIS ARM EXISTS: the #212-era run's published per-tracer H0 numbers
# (RESULTS.md: dscf GAL 65.98 +- 0.85, AGN 67.79 +- 0.83) were produced with
# injections.h5 — the catalog-targeted lane (injections_cat.h5) was introduced for
# the K>=2 sparse mixtures only, and its per-tracer scans were never rerun. Arm L
# uses injections_cat.h5 throughout for internal consistency, which changes mu(H0)
# and therefore the H0 landscape. Comparing Arm L against those published numbers
# would conflate the injection set with 295 commits of code change; this arm makes
# the comparison exact by matching the injection set.
set -uo pipefail
cd "$(dirname "$0")"
source ./env.sh

ISO=../data/injections.h5

run () {
  local tag=$1; shift
  echo "=== $(date +%H:%M:%S) $tag ==="
  python scan_darksirens.py "$@" --max_likelihood_variance $LEGACY_VAR \
    > ../logs/${tag}.log 2>&1 || echo "FAILED: $tag"
  grep -hE "^Eval done" ../logs/${tag}.log
}

for R in 00 01 02 03 04 05 06 07 08 09; do
  for T in gal agn; do
    run LI_h0_dscf_${T}_r${R} --universe_model dark_sirens_complete \
      --catalog_sky_weighting field --survey_path ../data/${T}.h5 \
      --gw_path ../data/gw_cov_${T}_r${R}.h5 --gwselection_path $ISO \
      --scan h0 --h0_grid 50 100 61 --h0_true 67.74 --out_tag LI_h0_dscf_${T}_r${R}
  done
done
for R in 00 01 02 03 04; do
  for T in gal agn; do
    run LI_h0_dsf_${T}_r${R} --universe_model dark_sirens \
      --catalog_sky_weighting field --survey_path ../data/${T}.h5 \
      --gw_path ../data/gw_cov_${T}_r${R}.h5 --gwselection_path $ISO \
      --scan h0 --h0_grid 50 100 61 --log10n0 -12 --h0_true 67.74 \
      --out_tag LI_h0_dsf_${T}_r${R}
  done
done

echo "=== $(date +%H:%M:%S) ARM LI DONE ==="
