#!/usr/bin/env bash
# experiment_h0f_baseline — the (H0, f) baseline result set.
#
# One estimator, one selection guard, one set of inputs. Everything else is
# deliberately out of scope; later experiments build on this set.
#
#   estimator   dark_sirens, field-convention sky weighting, K=2 mixture
#               [GAL, AGN] so fcat_2 = the AGN-hosted fraction, evaluated at the
#               complete-catalog limit log10n0 -> -12 (both catalogs are complete)
#   guard       historical N_eff > 5*N_obs only (--max_likelihood_variance 1e6
#               makes master's newer total-variance criterion inert; see
#               scan_h0f.py's docstring for why `soft` does not do this)
#   population  fixed at the mock truth (powerlaw+peak + chieff), Om0 pinned
#   selection   catalog-targeted injections (required for field-mode mixtures
#               with a sparse tracer)
#   nuisances   delta = 0, sigma_kde = 0
#
# Grids: f 41 pts on [0,1]; H0 61 pts on [50,100]; joint 81 x 61 (H0 x f).
#
# Usage:  ./run_experiment.sh            (all runs)
#         ./run_experiment.sh fscan      (one stage: fscan | h0scan | joint)
set -uo pipefail
cd "$(dirname "$0")"

# --- environment ---------------------------------------------------------------
: "${DARKSIRENS_WT:=/tmp/claude-88592/-hildafs-projects-phy230014p-magana-gws-agn/6b9abc89-f874-41de-9ed3-c0ca4def231c/scratchpad/wt-2b86a2d}"
export PYTHONPATH="$DARKSIRENS_WT"
export DARKSIRENS_SRC="$DARKSIRENS_WT"
export DARKSIRENS_ZMAX=1.5          # survey depth; NOT the 5.0 default
export XLA_PYTHON_CLIENT_PREALLOCATE=false
if [ ! -d "$DARKSIRENS_WT/darksirens" ]; then
  echo "[fatal] DARKSIRENS_WT=$DARKSIRENS_WT is not a darksirens checkout" >&2
  exit 1
fi
echo "[env] darksirens $(git -C "$DARKSIRENS_WT" rev-parse --short HEAD) at $DARKSIRENS_WT"

# --- fixed configuration -------------------------------------------------------
SURVEYS="../data/gal.h5 ../data/agn.h5"     # order sets fcat_2 = AGN fraction
INJ=../data/injections_cat.h5
H0_TRUTH=67.74
N0_LIMIT=-12                                 # complete-catalog limit
COMMON="--universe_model dark_sirens --catalog_sky_weighting field
        --selection_neff_guard hard --max_likelihood_variance 1e6
        --log10n0 $N0_LIMIT --log10n0_c2 $N0_LIMIT"

# Planted AGN-hosted fraction of each event set (eligible-pool truth).
declare -A FTRUTH=( [0.0]=0.00989 [0.3]=0.307 [0.7]=0.703 [1.0]=1.0 )

STAGE="${1:-all}"

run () {
  local tag=$1; shift
  echo "=== $(date +%H:%M:%S) $tag ==="
  python scan_h0f.py "$@" > ../logs/${tag}.log 2>&1 || { echo "FAILED: $tag"; return; }
  grep -hE "^Eval done" ../logs/${tag}.log
}

# --- 1. f scans at the true H0 -------------------------------------------------
if [ "$STAGE" = all ] || [ "$STAGE" = fscan ]; then
  for K in 0.0 0.3 0.7 1.0; do
    run fscan_fagn${K} $COMMON --survey_path $SURVEYS \
      --gw_path ../data/gw_fagn${K}.h5 --gwselection_path $INJ \
      --scan f --f_grid 0 1 41 --h0_fixed $H0_TRUTH \
      --f_true ${FTRUTH[$K]} --out_tag fscan_fagn${K}
  done
fi

# --- 2. H0 scans at the true f -------------------------------------------------
if [ "$STAGE" = all ] || [ "$STAGE" = h0scan ]; then
  for K in 0.3 0.7; do
    run h0scan_fagn${K} $COMMON --survey_path $SURVEYS \
      --gw_path ../data/gw_fagn${K}.h5 --gwselection_path $INJ \
      --scan h0 --h0_grid 50 100 61 --f_fixed ${FTRUTH[$K]} \
      --h0_true $H0_TRUTH --out_tag h0scan_fagn${K}
  done
fi

# --- 3. joint (H0, f) ----------------------------------------------------------
if [ "$STAGE" = all ] || [ "$STAGE" = joint ]; then
  for K in 0.3 0.7; do
    run joint_fagn${K} $COMMON --survey_path $SURVEYS \
      --gw_path ../data/gw_fagn${K}.h5 --gwselection_path $INJ \
      --scan joint --h0_grid 50 100 81 --f_grid 0 1 61 \
      --h0_true $H0_TRUTH --f_true ${FTRUTH[$K]} --out_tag joint_fagn${K}
  done
fi

# --- 4. refined joint (H0, f) around the peak, for publication contours -------
# The wide 81x61 grid resolves the 90% region with only ~5-8 cells (step 0.625 in
# H0, 0.0167 in f, against half-widths 0.45 and 0.022), which renders as a polygon.
# These windows keep the planted value inside while giving ~10-20 cells across the
# 68% region. Windows are hard-coded from the wide-grid peaks, not auto-derived,
# so the figure never depends on a fit that could silently drift.
if [ "$STAGE" = all ] || [ "$STAGE" = jointzoom ]; then
  run jointzoom_fagn0.3 $COMMON --survey_path $SURVEYS \
    --gw_path ../data/gw_fagn0.3.h5 --gwselection_path $INJ \
    --scan joint --h0_grid 64.0 70.0 61 --f_grid 0.26 0.40 57 \
    --h0_true $H0_TRUTH --f_true ${FTRUTH[0.3]} --out_tag jointzoom_fagn0.3
  run jointzoom_fagn0.7 $COMMON --survey_path $SURVEYS \
    --gw_path ../data/gw_fagn0.7.h5 --gwselection_path $INJ \
    --scan joint --h0_grid 62.0 69.0 71 --f_grid 0.63 0.78 61 \
    --h0_true $H0_TRUTH --f_true ${FTRUTH[0.7]} --out_tag jointzoom_fagn0.7
fi

echo "=== $(date +%H:%M:%S) EXPERIMENT DONE ($STAGE) ==="
