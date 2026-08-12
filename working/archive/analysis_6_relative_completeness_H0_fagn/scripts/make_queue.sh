#!/usr/bin/env bash
# Build the analysis-6 work queue.
#
# Line format:  SEED GLEV ALEV LANE
#
# THE SURFACE.  GAL depth x AGN depth over {complete, m20, m19, m18}^2 restricted
# to GAL in {m20, m19, m18} — the depths where the completion actually does
# something (m21 is 100 % complete inside the horizon, so a GAL m21 row would
# duplicate `complete`).
#
#            AGN complete   AGN m20      AGN m19      AGN m18
#   GAL m20      NEW         have (a3)     NEW          NEW
#   GAL m19      NEW           NEW       have (a3)      NEW
#   GAL m18   have (a4)        NEW         NEW        have (a3)
#
# The four cells already on disk are REFERENCED by the aggregation, never rerun:
#   ../analysis_3_incomplete_catalog_H0_fagn/results/joint_{m20,m19,m18}_s100.h5
#   ../analysis_4_density_anchoring_H0_fagn/results/joint_m18_oracle_s100.h5
# so the surface and its diagonal share one estimator by construction (all five
# scan_h0f.py copies are byte-identical, md5 02acecc6f73d5ae0bd31985e2b7ac1c3).
#
# 8 new cells.  Every survey file already exists — no mock regeneration, no new
# events, no new injections.  Emitted MOST EXPENSIVE FIRST (cost is set by the
# GAL block) so the long pole starts on the first free GPU.
set -euo pipefail
cd "$(dirname "$0")/.."
Q=queue
mkdir -p "$Q"
: > "$Q/tasks.txt"

emit() { echo "100 $1 $2 targeted" >> "$Q/tasks.txt"; }

# GAL m20 row (most expensive), then m19, then m18
emit m20 complete
emit m20 m19
emit m20 m18
emit m19 complete
emit m19 m20
emit m19 m18
emit m18 m20
emit m18 m19

echo "queue: $(wc -l < "$Q/tasks.txt") tasks"
cat -n "$Q/tasks.txt"
