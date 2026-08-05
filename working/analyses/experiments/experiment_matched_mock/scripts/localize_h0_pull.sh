#!/usr/bin/env bash
# Localise the H0 low-pull to a small set of events.
#
# The 10-block closure test showed BIMODAL scatter (b6/b8/b9 near -3.4, the rest near
# -0.9), which is not Gaussian GW noise about a single systematic and suggests a few
# events dominate. This sweeps DISJOINT blocks of BLOCK events and, for each, measures
# the pull toward low H0 as
#
#     dlogL = logL(H0_LO) - logL(H0_TRUE)
#
# Positive dlogL = that block prefers the low H0. Only two likelihood evaluations per
# block, so the sweep is cheap. The selection term scales with the block's event count,
# so dlogL is compared PER EVENT across equally-sized blocks.
set -uo pipefail
cd "$(dirname "$0")"

: "${DARKSIRENS_WT:=/tmp/claude-88592/-hildafs-projects-phy230014p-magana-gws-agn/6b9abc89-f874-41de-9ed3-c0ca4def231c/scratchpad/wt-2b86a2d}"
export PYTHONPATH="$DARKSIRENS_WT"
export DARKSIRENS_SRC="$DARKSIRENS_WT"
export DARKSIRENS_ZMAX=2.0
export XLA_PYTHON_CLIENT_PREALLOCATE=false

PARENT=../data_derived/deep_mock_z2_big/mock_gw_events.h5
SEL=../data_derived/deep_mock_z2_big/mock_gw_selection.h5
SURVEY=../data_derived/deep_survey_z2_dz3e3.h5
BLOCK=${1:-20}                 # events per block
NBLOCK=${2:-50}                # number of disjoint blocks
H0_LO=64.0
H0_TRUE=67.74

OUT=../results/localize_b${BLOCK}
mkdir -p "$OUT" ../logs/localize

echo "[sweep] block=$BLOCK nblocks=$NBLOCK  dlogL = logL($H0_LO) - logL($H0_TRUE)"
for ((B=0; B<NBLOCK; B++)); do
  EV=../data_derived/loc_b${BLOCK}_${B}.h5
  python build_event_subsample.py --in_path $PARENT --out_path "$EV" \
    --n_events $BLOCK --block_index $B > /dev/null 2>&1 || { echo "subsample failed $B"; continue; }
  python scan_h0f.py --universe_model dark_sirens_complete \
    --catalog_sky_weighting field --survey_path $SURVEY \
    --gw_path "$EV" --gwselection_path $SEL \
    --scan h0 --h0_grid $H0_LO $H0_TRUE 2 --h0_true $H0_TRUE \
    --out_tag loc_${BLOCK}_${B} --outdir "$OUT" \
    > ../logs/localize/loc_${BLOCK}_${B}.log 2>&1 || echo "scan failed $B"
  rm -f "$EV"
done

python aggregate_localize.py --results_dir "$OUT" --parent $PARENT \
  --block $BLOCK --nblock $NBLOCK \
  --out_json ../results/localize_summary_b${BLOCK}.json
