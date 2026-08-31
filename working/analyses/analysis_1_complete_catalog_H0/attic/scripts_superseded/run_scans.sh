#!/bin/bash
# analysis_1_complete_catalog_H0 — the four independent single-tracer H0 scans.
#
#   1. h0_gal_targeted   GAL complete catalog + targeted injection lane
#   2. h0_gal_popuni     GAL complete catalog + popuni  injection lane   (lane cross-check)
#   3. h0_agn_targeted   AGN complete catalog + targeted injection lane
#   4. h0_agn_popuni     AGN complete catalog + popuni  injection lane   (lane cross-check)
#
# Model: dark_sirens_complete (K=1) + catalog_sky_weighting=field, fixed
# powerlaw+peak population, Om0 pinned at 0.3075, H0 grid 50..100 x 201.
# Guard: the campaign convention — hard Neff wall, variance criterion made inert.
set -euo pipefail
cd "$(dirname "$0")/.."

DATA=/hildafs/projects/phy230014p/magana/gws-agn/working/data/seed100
GW=$DATA/events/events.h5
SUR_GAL=$DATA/surveys/survey_gal_complete_ns32.h5
SUR_AGN=$DATA/surveys/survey_agn_complete_ns32.h5
INJ_TARGETED=$DATA/injections/injections_targeted.h5
INJ_POPUNI=$DATA/injections/injections_popuni.h5

COMMON="--universe_model dark_sirens_complete --catalog_sky_weighting field \
  --gw_path $GW --scan h0 --h0_true 67.74 \
  --selection_neff_guard hard --max_likelihood_variance 1e6 --sel_batch_size ${SEL_BATCH:-200000} --pe_event_block ${PE_BLOCK:-100} --outdir results"

# The v2 GAL survey carries ~1.2e4 galaxies per nside-32 row, so darksirens'
# windowed catalog-KDE evaluator must be given a window at least as large as the
# number of galaxies inside the kernel support (sized with
# darksirens.redshift.catalog.recommended_kde_window; see README).  The AGN
# survey's rows are far shorter than the window and take the full-row path.
KDE_W=${KDE_W:-4096}
NGRID=${NGRID:-201}

scan () {  # tag survey selection [extra...]
  local tag=$1 sur=$2 sel=$3; shift 3
  echo "[$(date +%H:%M:%S)] scanning $tag"
  python scripts/scan_h0f.py $COMMON --h0_grid 50.0 100.0 "$NGRID" \
    --survey_path "$sur" --gwselection_path "$sel" "$@" \
    --out_tag "$tag" > "logs/${tag}.log" 2>&1
  echo "[$(date +%H:%M:%S)] done $tag"
}

scan h0_gal_targeted "$SUR_GAL" "$INJ_TARGETED" --kde_window "$KDE_W"
scan h0_gal_popuni   "$SUR_GAL" "$INJ_POPUNI"   --kde_window "$KDE_W"
scan h0_agn_targeted "$SUR_AGN" "$INJ_TARGETED"
scan h0_agn_popuni   "$SUR_AGN" "$INJ_POPUNI"
echo "ALL SCANS DONE"
