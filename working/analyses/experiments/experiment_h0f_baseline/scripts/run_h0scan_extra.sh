#!/usr/bin/env bash
# H0 scans at the planted f for the two rungs the original set skipped
# (f = 0.0099 and 1.0) — same recipe as run_experiment.sh stage h0scan.
set -uo pipefail
cd "$(dirname "$0")"
export DARKSIRENS_WT=/hildafs/projects/phy230014p/magana/src/darksirens
export PYTHONPATH="$DARKSIRENS_WT" DARKSIRENS_SRC="$DARKSIRENS_WT"
export DARKSIRENS_ZMAX=1.5 XLA_PYTHON_CLIENT_PREALLOCATE=false
SURVEYS="../data/gal.h5 ../data/agn.h5"
INJ=../data/injections_cat.h5
COMMON="--universe_model dark_sirens --catalog_sky_weighting field
        --selection_neff_guard hard --max_likelihood_variance 1e6
        --log10n0 -12 --log10n0_c2 -12"
declare -A FTRUTH=( [0.0]=0.00989 [1.0]=1.0 )
for K in 0.0 1.0; do
  python scan_h0f.py $COMMON --survey_path $SURVEYS \
    --gw_path ../data/gw_fagn${K}.h5 --gwselection_path $INJ \
    --scan h0 --h0_grid 50 100 61 --f_fixed ${FTRUTH[$K]} \
    --h0_true 67.74 --out_tag h0scan_fagn${K} > ../logs/h0scan_fagn${K}.log 2>&1 \
    && echo "DONE $K" || echo "FAILED $K (see logs/h0scan_fagn${K}.log)"
done
