#!/bin/bash
# analysis_1_complete_catalog_H0 -- the matched-host controls on a further
# realisation.
#
#   run_seed_controls.sh 101 [102 ...]
#
# For each seed: split that seed's events.h5 on host_type, then run both matched
# controls (GAL catalog + its own GAL-hosted events; AGN catalog + its own
# AGN-hosted events) with exactly the configuration of run_matched_control.sh --
# same model, grid, guard, population and injection lane (targeted, the
# measurement of record).  Outputs are tagged ctrl_{gal,agn}_matched_s<SEED>.
#
# The host-type event subsets are large (~160 MB per seed) and are therefore
# written to the campaign's bulk filesystem, reached through the
# data_derived/seeds symlink; only the scan outputs (a few tens of kB) land in
# this analysis directory.
set -euo pipefail
cd "$(dirname "$0")/.."

BULK=/hildafs/projects/phy220048p/magana/gws-agn-data/derived/analysis_1_complete_catalog_H0
DATAROOT=${DATAROOT:-/hildafs/projects/phy230014p/magana/gws-agn/working/data}
KDE_W=${KDE_W:-4096}

mkdir -p "$BULK"
[ -L data_derived/seeds ] || ln -s "$BULK" data_derived/seeds

# 2026-08-01: the estimator of record is `dark_sirens` at log10n0 = -24 -- the
# complete-catalog LIMIT of the general incomplete-catalog likelihood, which
# experiment_model_equivalence measured to be bitwise identical to
# `dark_sirens_complete` in all 201 cells of all four seed-100 configurations.  The
# five-seed table is now carried by the same estimator as run_scans.sh so that every
# number in the campaign comes from one likelihood.  SUFFIX lets a variant set
# (e.g. the nside study) be written without colliding with the record.
MODEL=${MODEL:-"--universe_model dark_sirens --log10n0 ${LOG10N0:--24}"}
SUFFIX=${SUFFIX:-}
COMMON="$MODEL --catalog_sky_weighting field \
  --scan h0 --h0_grid 50.0 100.0 201 --h0_true 67.74 \
  --selection_neff_guard hard --max_likelihood_variance 1e6 \
  --sel_batch_size ${SEL_BATCH:-200000} --pe_event_block ${PE_BLOCK:-100} --outdir results"

for S in "$@"; do
  DATA=$DATAROOT/seed$S
  D=$BULK/seed$S
  mkdir -p "$D"
  echo "[$(date +%H:%M:%S)] ===== seed $S ====="
  [ -f "$D/events_gal_hosted.h5" ] || python scripts/build_hosttype_subset.py \
      --in_path "$DATA/events/events.h5" --out_path "$D/events_gal_hosted.h5" --host_type 0
  [ -f "$D/events_agn_hosted.h5" ] || python scripts/build_hosttype_subset.py \
      --in_path "$DATA/events/events.h5" --out_path "$D/events_agn_hosted.h5" --host_type 1

  echo "[$(date +%H:%M:%S)] ctrl_gal_matched_s$S$SUFFIX"
  python scripts/scan_h0f.py $COMMON \
      --survey_path "$DATA/surveys/survey_gal_complete_ns32.h5" \
      --gw_path "$D/events_gal_hosted.h5" \
      --gwselection_path "$DATA/injections/injections_targeted.h5" \
      --kde_window "$KDE_W" --out_tag "ctrl_gal_matched_s$S$SUFFIX" \
      > "logs/ctrl_gal_matched_s$S$SUFFIX.log" 2>&1

  echo "[$(date +%H:%M:%S)] ctrl_agn_matched_s$S$SUFFIX"
  python scripts/scan_h0f.py $COMMON \
      --survey_path "$DATA/surveys/survey_agn_complete_ns32.h5" \
      --gw_path "$D/events_agn_hosted.h5" \
      --gwselection_path "$DATA/injections/injections_targeted.h5" \
      --out_tag "ctrl_agn_matched_s$S$SUFFIX" > "logs/ctrl_agn_matched_s$S$SUFFIX.log" 2>&1
  echo "[$(date +%H:%M:%S)] seed $S done"
done
echo "SEED CONTROLS DONE: $*"
