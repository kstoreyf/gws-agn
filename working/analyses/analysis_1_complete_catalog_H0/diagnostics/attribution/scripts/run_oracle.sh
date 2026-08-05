#!/bin/bash
# ATTRIBUTION follow-up, TASK 3 -- the quadrature oracle: production runs plus
# the quadrature convergence battery (double the nodes, shift the grids, tighten
# the sky-pixel threshold) on a fixed 150/100-event subset.
set -euo pipefail
cd "$(dirname "$0")/.."
export XLA_PYTHON_CLIENT_PREALLOCATE=false

run () { # tag tracer extra...
  local tag=$1 tr=$2; shift 2
  if [ -s "results/attr_oracle_${tag}.json" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "[$(date -u +%H:%M:%S)] skip $tag"; return 0
  fi
  echo "[$(date -u +%H:%M:%S)] oracle $tag  ($*)"
  python -u scripts/attr_oracle.py --tracer "$tr" --tag "$tag" "$@" \
      > "logs/attr_oracle_${tag}.log" 2>&1
  echo "[$(date -u +%H:%M:%S)] done $tag"
}

# production
run agn agn --n_events 0
run gal gal --n_events 0

# convergence battery (same leading events as the production runs)
run agn_conv_nz  agn --n_events 100 --n_z 1024
run agn_conv_nm  agn --n_events 100 --n_m  384
run agn_conv_sh  agn --n_events 100 --grid_shift 0.37
run agn_conv_sky agn --n_events 100 --sky_frac 1e-7 --n_gh 64
run gal_conv_nz  gal --n_events 150 --n_z 1024
run gal_conv_nm  gal --n_events 150 --n_m  384
run gal_conv_sh  gal --n_events 150 --grid_shift 0.37
run gal_conv_sky gal --n_events 150 --sky_frac 1e-7 --n_gh 64
echo "ORACLE RUNS DONE"
