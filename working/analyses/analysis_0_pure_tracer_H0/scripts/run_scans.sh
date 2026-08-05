#!/usr/bin/env bash
# analysis_0_pure_tracer_H0 -- the four K=1 H0 scans of one catalog realisation.
#
#   ./scripts/run_scans.sh <SEED> [SEED ...]
#
#   h0_puregal_targeted_s<S>   GAL complete survey + the 1000 pure-GAL events + targeted
#   h0_puregal_popuni_s<S>     ... same events, popuni lane           (lane cross-check)
#   h0_pureagn_targeted_s<S>   AGN complete survey + the 1000 pure-AGN events + targeted
#   h0_pureagn_popuni_s<S>     ... same events, popuni lane           (lane cross-check)
#
# CONFIGURATION IS analysis_1's, VERBATIM.  Only the events file changes.  From
# analysis_1/scripts/{run_scans.sh,run_seed_controls.sh}:
#
#   estimator   dark_sirens at log10n0 = -24 -- the complete-catalog LIMIT of the
#               general incomplete-catalog likelihood, measured bitwise identical to
#               dark_sirens_complete in all 201 cells (experiment_model_equivalence)
#   weighting   --catalog_sky_weighting field
#   grid        H0 in [50, 100] x 201, Om0 pinned at 0.3075, truth 67.74
#   nuisances   delta = 0, sigma_kde = 0, b_miss = 1 (the driver's FIXED_DEFAULTS);
#               the powerlaw+peak population is fixed at truth
#   guard       --selection_neff_guard hard --max_likelihood_variance 1e6, i.e. the
#               historical N_eff > 5 N_obs wall with the total-variance criterion
#               made inert
#   KDE window  W = 4096 on the GAL survey only (its nside-32 rows hold ~1.2e4
#               galaxies, more than the module default 1024); the AGN survey's rows
#               are far shorter than the window and take the full-row path, so
#               analysis_1 left it at the default there and so does this script
#
# `scripts/scan_h0f.py` is analysis_1's driver copied byte-for-byte
# (md5 02acecc6f73d5ae0bd31985e2b7ac1c3); see README.md.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs results figs

DATAROOT=${DATAROOT:-/hildafs/projects/phy230014p/magana/gws-agn/working/data}
KDE_W=${KDE_W:-4096}
NGRID=${NGRID:-201}
LOG10N0=${LOG10N0:--24}

MODEL="--universe_model dark_sirens --log10n0 $LOG10N0"
COMMON="$MODEL --catalog_sky_weighting field \
  --scan h0 --h0_grid 50.0 100.0 $NGRID --h0_true 67.74 \
  --selection_neff_guard hard --max_likelihood_variance 1e6 \
  --sel_batch_size ${SEL_BATCH:-50000} --pe_event_block ${PE_BLOCK:-25} --outdir results"

scan () {  # tag survey events selection [extra...]
  local tag=$1 sur=$2 ev=$3 sel=$4; shift 4
  if [ -s "results/${tag}.h5" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "[$(date -u +%H:%M:%S)] skip $tag (results/${tag}.h5 exists)"; return 0
  fi
  echo "[$(date -u +%H:%M:%S)] scanning $tag"
  python -u scripts/scan_h0f.py $COMMON \
    --survey_path "$sur" --gw_path "$ev" --gwselection_path "$sel" "$@" \
    --out_tag "$tag" > "logs/${tag}.log" 2>&1
  echo "[$(date -u +%H:%M:%S)] done $tag"
}

for S in "$@"; do
  D=$DATAROOT/seed$S
  SUR_GAL=$D/surveys/survey_gal_complete_ns32.h5
  SUR_AGN=$D/surveys/survey_agn_complete_ns32.h5
  EV_GAL=$D/events/events_puregal.h5
  EV_AGN=$D/events/events_pureagn.h5
  INJ_T=$D/injections/injections_targeted.h5
  INJ_P=$D/injections/injections_popuni.h5
  for f in "$SUR_GAL" "$SUR_AGN" "$EV_GAL" "$EV_AGN" "$INJ_T" "$INJ_P"; do
    [ -s "$f" ] || { echo "MISSING: $f" >&2; exit 2; }
  done
  echo "[$(date -u +%H:%M:%S)] ===== seed $S ====="
  # AGN first: the cheap configurations double as a smoke test of the model, the
  # survey files and the guard before the expensive GAL grids start.
  scan "h0_pureagn_targeted_s$S" "$SUR_AGN" "$EV_AGN" "$INJ_T"
  scan "h0_pureagn_popuni_s$S"   "$SUR_AGN" "$EV_AGN" "$INJ_P"
  scan "h0_puregal_targeted_s$S" "$SUR_GAL" "$EV_GAL" "$INJ_T" --kde_window "$KDE_W"
  scan "h0_puregal_popuni_s$S"   "$SUR_GAL" "$EV_GAL" "$INJ_P" --kde_window "$KDE_W"
  echo "[$(date -u +%H:%M:%S)] seed $S done"
done
echo "ALL SCANS DONE: $*"
