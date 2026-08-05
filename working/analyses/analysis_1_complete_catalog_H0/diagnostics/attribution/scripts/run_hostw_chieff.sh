#!/bin/bash
# TASKS 2 and 3.
#
#   ./scripts/run_hostw_chieff.sh          both
#   ./scripts/run_hostw_chieff.sh chieff   only the chi_eff clipping substitution
#   ./scripts/run_hostw_chieff.sh hostw    only the host-acceptance convention
#
# TASK 3 re-runs the VALIDATED sky oracle with the opt-in --host_prior_arms flag,
# which adds two extra arms and changes nothing else; every existing product is
# untouched (new tags `*_hostw`).  The substitutions are PAIRED per event and
# carry no Monte-Carlo error, so a subset resolves them as well as the full set.
set -euo pipefail
cd "$(dirname "$0")/.."
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export DARKSIRENS_SRC=${DARKSIRENS_SRC:-/hildafs/projects/phy230014p/magana/src/darksirens}
MODE=${1:-all}
NS=${NS:-0}     # 0 = every matched event

if [ "$MODE" = "all" ] || [ "$MODE" = "chieff" ]; then
  echo "[$(date -u +%H:%M:%S)] TASK 2: chi_eff clipping"
  python -u scripts/attr_chieff_clip.py > logs/attr_chieff.log 2>&1
  tail -6 logs/attr_chieff.log
fi

if [ "$MODE" = "all" ] || [ "$MODE" = "hostw" ]; then
  for TR in agn gal; do
    echo "[$(date -u +%H:%M:%S)] TASK 3: sky oracle host-prior arms, $TR"
    python -u scripts/attr_sky_oracle.py --tracer $TR --tag ${TR}_hostw \
        --host_prior_arms --n_events $NS ${EXTRA:-} \
        > logs/attr_sky_oracle_${TR}_hostw.log 2>&1
  done
  python -u scripts/attr_hostw.py | tee logs/attr_hostw.log
fi
echo "[$(date -u +%H:%M:%S)] DONE"
