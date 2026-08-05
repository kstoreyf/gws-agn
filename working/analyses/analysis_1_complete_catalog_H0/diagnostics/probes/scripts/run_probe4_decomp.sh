#!/bin/bash
# Probe 3's decomposition applied to probe 4's synthetic surveys: which term
# moved between the analytic continuum and the catalog's own smooth dN/dz?
set -uo pipefail
cd "$(dirname "$0")/.."
export XLA_PYTHON_CLIENT_PREALLOCATE=false
B=/hildafs/projects/phy220048p/magana/gws-agn-data/derived/analysis_1_complete_catalog_H0/probe4
run () {
  local suf=$1 sur=$2
  echo "=== $(date -u) decomposition on $suf ==="
  python -u scripts/probe3_decomposition.py --seed 100 --tracer gal \
    --h0_grid 50.0 100.0 201 --kde_window 4096 \
    --sel_batch_size 200000 --pe_event_block 100 \
    --survey_override "$sur" --tag_suffix "$suf" > "logs/probe3_gal_s100${suf}.log" 2>&1
  echo "    done $suf"
}
run _p4a "$B/survey_gal_probe4a_continuum_s100_ns32.h5"
run _p4b "$B/survey_gal_probe4b_uniform_s100_ns32.h5"
run _p4bemp "$B/survey_gal_probe4bemp_uniform_s100_ns32.h5"
echo "=== $(date -u) probe4 decomposition DONE ==="
