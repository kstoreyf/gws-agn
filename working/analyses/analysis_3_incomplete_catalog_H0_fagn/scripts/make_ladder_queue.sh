#!/usr/bin/env bash
# Build the campaign work queue.
#
# Line format:  KIND SEED LEVEL LANE CHUNK NCHUNK
#               CHUNK/NCHUNK are "-" for a whole grid run in one task.
#
# RUNG 0 OF RECORD (24 tasks).  The complete-catalog rung is re-run HERE, in the
# analysis-3 configuration (out-of-catalog field term active at the mock's true
# densities), 5 seeds targeted + the seed-100 popuni cross-check.  It is no longer
# taken from analysis 2: the continuity check measured that analysis 2's
# log10n0 = -24 run is a different estimator on this data (f_AGN +0.080, 1.74 of its
# own 68 % half-widths), so a ladder quoted against it would confound completeness
# degradation with that estimator offset.  Analysis 2's grids are kept as the
# zero-missing-budget REFERENCE, reported alongside, never as rung 0.
# A complete-pair grid is 8241 x 2.97 s = 6.8 GPU-h, more than one 6 h worker holds,
# so each is split into NCHUNK = 4 contiguous H0 chunks (~1.7 h each) that
# scripts/merge_joint.py stitches, asserting the reassembled axis reproduces
# linspace(50, 100, 201) exactly.
#
# THE LADDER (25 tasks).  m21..m18 cost 0.42-2.28 GPU-h per grid, so each runs whole
# in one task.  20 record grids + 4 seed-100 popuni cross-checks + the m18
# sky-shuffle null.
#
# Tasks are emitted MOST EXPENSIVE FIRST so the long pole starts on the first free
# GPU.  Workers claim by `mkdir` (atomic on POSIX), so any number of workers on any
# number of partitions share the queue without a lock server.
set -euo pipefail
cd "$(dirname "$0")/.."
Q=queue
NCHUNK_COMPLETE=${NCHUNK_COMPLETE:-4}
mkdir -p "$Q"
: > "$Q/tasks.txt"

# --- rung 0 of record: the complete pair in THIS configuration, chunked ---
for seed in 100 101 102 103 105; do
  for c in $(seq 0 $((NCHUNK_COMPLETE - 1))); do
    echo "joint $seed complete targeted $c $NCHUNK_COMPLETE" >> "$Q/tasks.txt"
  done
done
for c in $(seq 0 $((NCHUNK_COMPLETE - 1))); do
  echo "joint 100 complete popuni $c $NCHUNK_COMPLETE" >> "$Q/tasks.txt"
done

# --- the ladder ---
for lev in m21 m20 m19 m18; do
  for seed in 100 101 102 103 105; do
    echo "joint $seed $lev targeted - -" >> "$Q/tasks.txt"
  done
done
for lev in m21 m20 m19 m18; do
  echo "joint 100 $lev popuni - -" >> "$Q/tasks.txt"
done
echo "fnull 100 m18 targeted - -" >> "$Q/tasks.txt"

echo "queue: $(wc -l < "$Q/tasks.txt") tasks"
cat "$Q/tasks.txt"
