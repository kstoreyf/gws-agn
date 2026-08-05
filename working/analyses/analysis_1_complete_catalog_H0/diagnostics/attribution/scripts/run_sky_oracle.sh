#!/bin/bash
# The exact host-galaxy sky oracle: production runs + the convergence battery.
#
#   ./scripts/run_sky_oracle.sh            production only
#   ./scripts/run_sky_oracle.sh conv       production + convergence
#
# Production is on EVERY matched event of seed 100 (720 GAL, 280 AGN); the
# convergence battery re-runs a 120-event subset with each knob moved, because the
# quantity being converged -- the PAIRED substitution delta_host - delta_pix -- has
# no Monte-Carlo error, so a subset resolves it as well as the full set.
set -euo pipefail
cd "$(dirname "$0")/.."
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export DARKSIRENS_SRC=${DARKSIRENS_SRC:-/hildafs/projects/phy230014p/magana/src/darksirens}

run () { # tag tracer [extra...]
  local tag=$1 tr=$2; shift 2
  echo "[$(date -u +%H:%M:%S)] sky-oracle $tag"
  python scripts/attr_sky_oracle.py --tracer "$tr" --tag "$tag" --with_kde_host \
      ${EXTRA:-} "$@" \
      > "logs/attr_sky_oracle_${tag}.log" 2>&1
  echo "[$(date -u +%H:%M:%S)] done $tag"
}

MODE=${1:-prod}
if [ "$MODE" != "convonly" ]; then
  run agn agn
  run gal gal
fi

if [ "$MODE" = "conv" ] || [ "$MODE" = "convonly" ]; then
  NS=${NS:-120}
  TRACERS=${TRACERS:-"gal agn"}
  for tr in $TRACERS; do
    run ${tr}_conv_ap4    $tr --n_events $NS --n_ap 4
    run ${tr}_conv_ap8    $tr --n_events $NS --n_ap 8
    run ${tr}_conv_sf5    $tr --n_events $NS --sky_frac 1e-5
    run ${tr}_conv_sf7    $tr --n_events $NS --sky_frac 1e-7
    run ${tr}_conv_sub4   $tr --n_events $NS --n_sub 4
    run ${tr}_conv_sub6   $tr --n_events $NS --n_sub 6
    run ${tr}_conv_nz     $tr --n_events $NS --n_z 1024
    run ${tr}_conv_nm     $tr --n_events $NS --n_m 384
    run ${tr}_conv_shift  $tr --n_events $NS --grid_shift 0.37
    run ${tr}_conv_base   $tr --n_events $NS
  done
fi
echo "SKY ORACLE DONE"
