#!/bin/bash
# analysis_1_complete_catalog_H0 -- CLOSURE DIAGNOSTICS on seed 100.
#
# Two questions about the matched-host controls, both answerable without any new
# realisation:
#
#   (i)  Does each control give the same answer on the second injection lane?
#        The production scans agree between lanes to ~2% of a half-width; the
#        controls were only ever run on the targeted lane, so their lane
#        agreement is asserted rather than measured.
#          ctrl_gal_matched_popuni, ctrl_agn_matched_popuni
#
#   (ii) Is the likelihood width an honest description of how much the answer
#        moves under resampling?  Each host-type event set is cut into 8 disjoint
#        contiguous blocks (events are stored as_drawn, so a block is an unbiased
#        sub-realisation) and each block is scanned on its own.  sd(block
#        medians)/sqrt(8) is an empirical standard error on the full-set median
#        and is compared against the quoted 68% half-width.
#          jk_gal_b0..b7 (90 events each), jk_agn_b0..b7 (35 events each)
#
# Everything else -- model, survey, guard, population, cosmology -- is identical
# to run_matched_control.sh.  The block scans use a coarser 101-point grid on the
# same [50, 100] range, which is all the block-level precision supports.
set -euo pipefail
cd "$(dirname "$0")/.."

DATA=/hildafs/projects/phy230014p/magana/gws-agn/working/data/seed100
SUR_GAL=$DATA/surveys/survey_gal_complete_ns32.h5
SUR_AGN=$DATA/surveys/survey_agn_complete_ns32.h5
INJ_TARGETED=$DATA/injections/injections_targeted.h5
INJ_POPUNI=$DATA/injections/injections_popuni.h5
KDE_W=${KDE_W:-4096}

COMMON="--universe_model dark_sirens_complete --catalog_sky_weighting field \
  --scan h0 --h0_true 67.74 \
  --selection_neff_guard hard --max_likelihood_variance 1e6 \
  --sel_batch_size ${SEL_BATCH:-200000} --pe_event_block ${PE_BLOCK:-100} --outdir results"

run () {  # tag survey events injections grid... [extra]
  local tag=$1 sur=$2 ev=$3 inj=$4 gmin=$5 gmax=$6 gn=$7; shift 7
  echo "[$(date +%H:%M:%S)] $tag"
  python scripts/scan_h0f.py $COMMON --survey_path "$sur" --gw_path "$ev" \
    --gwselection_path "$inj" --h0_grid "$gmin" "$gmax" "$gn" "$@" \
    --out_tag "$tag" > "logs/${tag}.log" 2>&1
  echo "[$(date +%H:%M:%S)] done $tag"
}

# (i) lane cross-check of the two matched-host controls
run ctrl_gal_matched_popuni "$SUR_GAL" data_derived/events_gal_hosted.h5 \
    "$INJ_POPUNI" 50.0 100.0 201 --kde_window "$KDE_W"
run ctrl_agn_matched_popuni "$SUR_AGN" data_derived/events_agn_hosted.h5 \
    "$INJ_POPUNI" 50.0 100.0 201

# (ii) disjoint-block scatter, targeted lane (the measurement of record)
for k in 0 1 2 3 4 5 6 7; do
  run "jk_agn_b$k" "$SUR_AGN" "data_derived/blocks/agn_b$k.h5" \
      "$INJ_TARGETED" 50.0 100.0 101
done
for k in 0 1 2 3 4 5 6 7; do
  run "jk_gal_b$k" "$SUR_GAL" "data_derived/blocks/gal_b$k.h5" \
      "$INJ_TARGETED" 50.0 100.0 101 --kde_window "$KDE_W"
done
echo "CLOSURE DIAGNOSTICS DONE"
