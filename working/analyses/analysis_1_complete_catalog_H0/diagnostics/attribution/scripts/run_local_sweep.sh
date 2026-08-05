#!/bin/bash
# The GAL half of the final sweep, on the local A100-80GB (the GAL catalog KDE
# does not fit the 40GB HENON card -- see logs/attr_sky_oracle_gal_hostw.log).
# Ordered by how much each step contributes to the verdict.
set -euo pipefail
cd "$(dirname "$0")/.."
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export DARKSIRENS_SRC=${DARKSIRENS_SRC:-/hildafs/projects/phy230014p/magana/src/darksirens}

echo "[$(date -u +%H:%M:%S)] 1/5  mu oracle, GAL"
python -u scripts/attr_selmu_oracle.py --tracer gal --conv_lat \
    > logs/attr_selmu_gal.log 2>&1

echo "[$(date -u +%H:%M:%S)] 2/5  injections gal/targeted"
python -u scripts/attr_selmu_inj.py --tracer gal --injections targeted \
    --sel_batch 50000 > logs/attr_selmu_inj_gal_targeted.log 2>&1

echo "[$(date -u +%H:%M:%S)] 3/5  TASK 2: chi_eff clipping"
python -u scripts/attr_chieff_clip.py > logs/attr_chieff.log 2>&1

echo "[$(date -u +%H:%M:%S)] 4/5  TASK 3: sky oracle host-prior arms, GAL"
python -u scripts/attr_sky_oracle.py --tracer gal --tag gal_hostw \
    --host_prior_arms > logs/attr_sky_oracle_gal_hostw.log 2>&1

echo "[$(date -u +%H:%M:%S)] 5/5  injections gal/popuni"
python -u scripts/attr_selmu_inj.py --tracer gal --injections popuni \
    --sel_batch 50000 > logs/attr_selmu_inj_gal_popuni.log 2>&1

echo "[$(date -u +%H:%M:%S)] LOCAL SWEEP DONE"
