#!/usr/bin/env bash
# Fixed-campaign RERUN with catalog-targeted injections (injections_cat*.h5):
# field-mode AGN selection Neff 1.2k -> 145k, so the hard (auto) guard applies.
set -uo pipefail
cd "$(dirname "$0")"
export DARKSIRENS_ZMAX=1.5
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH=/tmp/claude-88592/-hildafs-projects-phy230014p-magana-gws-agn/89590650-74f5-413e-8311-7f0160636741/scratchpad/wt-master

INJ=../data/injections_cat.h5
INJB=../data/injections_cat_B.h5
run () { echo "=== $(date +%H:%M:%S) $1 ==="; T=$1; shift; python scan_darksirens.py "$@" > ../logs/${T}.log 2>&1 || echo "FAILED: $T"; }

# --- A. f scans, dsc-field K2 (primary), 41 pts ---------------------------------
for K in 0.0 0.3 0.7 1.0; do
  run c2_fscan_dscf_fagn${K} --universe_model dark_sirens_complete --catalog_sky_weighting field \
    --survey_path ../data/gal.h5 ../data/agn.h5 \
    --gw_path ../data/gw_fagn${K}.h5 --gwselection_path $INJ \
    --scan f --f_grid 0 1 41 --h0_fixed 67.74 --out_tag c2_fscan_dscf_fagn${K}
done
run c2_fscan_dscf_fagn0.3_injB --universe_model dark_sirens_complete --catalog_sky_weighting field \
  --survey_path ../data/gal.h5 ../data/agn.h5 \
  --gw_path ../data/gw_fagn0.3.h5 --gwselection_path $INJB \
  --scan f --f_grid 0 1 41 --h0_fixed 67.74 --out_tag c2_fscan_dscf_fagn0.3_injB

# --- B. f scans, ds-field K2 (complete-limit cross-check + true-n0 demo) --------
for K in 0.3 0.7; do
  run c2_fscan_dsf_n0low_fagn${K} --universe_model dark_sirens --catalog_sky_weighting field \
    --survey_path ../data/gal.h5 ../data/agn.h5 \
    --gw_path ../data/gw_fagn${K}.h5 --gwselection_path $INJ \
    --scan f --f_grid 0 1 41 --h0_fixed 67.74 --log10n0 -12 --log10n0_c2 -12 \
    --out_tag c2_fscan_dsf_n0low_fagn${K}
done
run c2_fscan_dsf_n0true_fagn0.3 --universe_model dark_sirens --catalog_sky_weighting field \
  --survey_path ../data/gal.h5 ../data/agn.h5 \
  --gw_path ../data/gw_fagn0.3.h5 --gwselection_path $INJ \
  --scan f --f_grid 0 1 41 --h0_fixed 67.74 --log10n0 -5.50627668499162 --log10n0_c2 -7.508083961432144 \
  --out_tag c2_fscan_dsf_n0true_fagn0.3

# --- C. K=2 H0 scan at f=truth ---------------------------------------------------
run c2_h0_dscf_k2_fagn0.3 --universe_model dark_sirens_complete --catalog_sky_weighting field \
  --survey_path ../data/gal.h5 ../data/agn.h5 \
  --gw_path ../data/gw_fagn0.3.h5 --gwselection_path $INJ \
  --scan h0 --h0_grid 50 100 61 --f_fixed 0.307 --h0_true 67.74 --out_tag c2_h0_dscf_k2_fagn0.3

# --- D. Joint (H0, f) 61x41, dsc-field K2 ---------------------------------------
for K in 0.3 0.7; do
  run c2_joint_dscf_fagn${K} --universe_model dark_sirens_complete --catalog_sky_weighting field \
    --survey_path ../data/gal.h5 ../data/agn.h5 \
    --gw_path ../data/gw_fagn${K}.h5 --gwselection_path $INJ \
    --scan joint --h0_grid 50 100 61 --f_grid 0 1 41 --h0_true 67.74 --out_tag c2_joint_dscf_fagn${K}
done

echo "=== $(date +%H:%M:%S) RERUN DONE ==="
