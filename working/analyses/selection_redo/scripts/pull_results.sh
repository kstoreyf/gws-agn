#!/usr/bin/env bash
# Pull the J2 campaign's results back to hilda, into the real analysis dirs.
#
#   ./pull_results.sh            # all four
#   ./pull_results.sh a3 a6      # just these
#
# Pull, not push: J2 cannot reach hilda, so this always runs from here.
# rsync is one-way (J2 -> hilda) and never deletes, so re-running is free and a
# half-written remote file simply arrives complete on the next pass.
set -euo pipefail

REMOTE_HOST=js2h100
REMOTE_ROOT=/media/volume/darksirens-data/gws-agn-js2-data/analyses/selection_redo
HERE="$(cd "$(dirname "$0")" && pwd)"
ANALYSES="$(dirname "$(dirname "$HERE")")"          # working/analyses

# short name on J2  ->  the analysis directory of record here
declare -A DEST=(
  [a3]=analysis_3_incomplete_catalog_H0_fagn
  [a4]=analysis_4_density_anchoring_H0_fagn
  [a5]=analysis_5_free_anchors_H0_fagn
  [a6]=analysis_6_relative_completeness_H0_fagn
)

ARGS=("$@"); [ ${#ARGS[@]} -eq 0 ] && ARGS=(a3 a6 a4 a5)
for a in "${ARGS[@]}"; do
  d="${DEST[$a]:-}"
  [ -n "$d" ] || { echo "[skip] unknown $a"; continue; }
  mkdir -p "$ANALYSES/$d"/{results,logs,queue,figs}
  for sub in results logs queue; do
    rsync -a "$REMOTE_HOST:$REMOTE_ROOT/$a/$sub/" "$ANALYSES/$d/$sub/" 2>/dev/null || true
  done
  n=$(find "$ANALYSES/$d/results" -maxdepth 1 -name "*.json" 2>/dev/null | wc -l)
  echo "  $a -> $d   $n result JSON(s)"
done

# The shared drivers are provenance too: an analysis directory that cannot say
# which code produced it is not reproducible.
mkdir -p "$ANALYSES/selection_redo/scripts"
rsync -a "$REMOTE_HOST:$REMOTE_ROOT/scripts/" "$ANALYSES/selection_redo/scripts/" \
      --exclude '__pycache__' 2>/dev/null || true
echo "  scripts synced"
