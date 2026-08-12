#!/usr/bin/env bash
# Build the campaign work queue.
#
# Line format:  SEED GLEV ALEV LANE ARM N0C2
#
# THE ARMS (24 tasks).  Seed 100, rungs m21..m18, the AGN density anchor set off
# truth by a factor {0.5, 0.7, 0.9, 1.1, 1.3, 2.0}: log10n0_c2 = -5 + log10(f).
# The EXACT arm (factor 1) is analysis_3's own seed-100 grids
# (../analysis_3_incomplete_catalog_H0_fagn/results/joint_<rung>_s100.h5) —
# referenced by the aggregation, never rerun, so the reference and the arms
# share one estimator by construction.
#
# THE ORACLE PROBE (1 task).  GAL at m < 18, AGN survey COMPLETE, both densities
# at truth: does handing the model every AGN host remove the +0.084 f_AGN bias
# the sparse AGN completion is suspected of manufacturing at the faintest rung?
#
# Tasks are emitted MOST EXPENSIVE FIRST so the long pole starts on the first
# free GPU.  Workers claim by `mkdir` (atomic on POSIX), as in analysis 3.
set -euo pipefail
cd "$(dirname "$0")/.."
Q=queue
mkdir -p "$Q"
: > "$Q/tasks.txt"

# ARM tag -> log10n0_c2 = -5 + log10(factor)
ARMS="a05:-5.301030 a07:-5.154902 a09:-5.045757 a11:-4.958607 a13:-4.886057 a20:-4.698970"

for lev in m21 m20 m19; do
  for a in $ARMS; do
    echo "100 $lev $lev targeted ${a%%:*} ${a##*:}" >> "$Q/tasks.txt"
  done
done
echo "100 m18 complete targeted oracle -5.0" >> "$Q/tasks.txt"
for a in $ARMS; do
  echo "100 m18 m18 targeted ${a%%:*} ${a##*:}" >> "$Q/tasks.txt"
done

echo "queue: $(wc -l < "$Q/tasks.txt") tasks"
cat "$Q/tasks.txt"
