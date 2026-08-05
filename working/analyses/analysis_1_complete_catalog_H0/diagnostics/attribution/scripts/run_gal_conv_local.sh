#!/bin/bash
# GAL sky-oracle convergence battery, on the 80 GB card.
#
# darksirens' materialised catalog-KDE state for the GAL survey is ~6.8 copies of the
# (11,266 x 14,569) block, i.e. ~8.9 GB in one allocation, which does not fit on the
# 40 GB HENON cards alongside the rest of the likelihood -- it is not batch-tunable,
# so this battery runs locally.  120 events is ample: the quantity being converged,
# the PAIRED substitution delta_host - delta_pix, carries no Monte-Carlo error.
set -euo pipefail
cd "$(dirname "$0")/.."
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export DARKSIRENS_SRC=${DARKSIRENS_SRC:-/hildafs/projects/phy230014p/magana/src/darksirens}
NS=${NS:-120}
run () { local tag=$1; shift
  echo "[$(date -u +%H:%M:%S)] $tag"
  python scripts/attr_sky_oracle.py --tracer gal --tag "$tag" --with_kde_host \
      --n_events $NS "$@" > "logs/attr_sky_oracle_${tag}.log" 2>&1
  echo "[$(date -u +%H:%M:%S)] done $tag"; }
run gal_conv_base
run gal_conv_sub4  --n_sub 4
run gal_conv_sub6  --n_sub 6
run gal_conv_sf5   --sky_frac 1e-5
run gal_conv_sf7   --sky_frac 1e-7
run gal_conv_ap4   --n_ap 4
run gal_conv_nz    --n_z 1024
run gal_conv_nm    --n_m 384
run gal_conv_shift --grid_shift 0.37
echo "GAL CONV DONE"
