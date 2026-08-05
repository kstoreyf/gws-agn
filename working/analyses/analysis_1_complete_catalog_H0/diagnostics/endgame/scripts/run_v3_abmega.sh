#!/usr/bin/env bash
# THE v3 (A - B) CAMPAIGN -- redraw the event stage many times on a FIXED catalog so
# that E[A] is measured against a single exact B and the only Monte Carlo left is the
# event draw itself.  This is the instrument CLOSURE.md 15.2 used; 500 replays give
# sem(A - B) ~ 1.4e-4, the precision the owner's gate asks for.
#
#   ./scripts/run_v3_abmega.sh [SEED] [NREPLICAS] [DATAROOT]
set -euo pipefail
cd "$(dirname "$0")/.."
S=${1:-100}
N=${2:-500}
ROOT=${3:-/hildafs/projects/phy220048p/magana/gws-agn-data-v3}
SC=${SCRATCH_V3:-/hildafs/projects/phy220048p/magana/gws-agn-data-v3/scratch_ab}
D=$ROOT/seed$S
mkdir -p "$SC" logs results

# 1. the replay IS the generator: reproduce the record bit-identically first
if [ ! -s "$SC/events_notrunc_full_s${S}.h5" ]; then
  JAX_PLATFORMS=cpu python scripts/regen_events_notrunc.py --seed "$S" --verify \
      --dataroot "$ROOT" --pe_model v3 \
      --out "$SC/events_notrunc_full_s${S}.h5" \
      > "logs/v3_regen_verify_s${S}.log" 2>&1
  grep -E "VERIFY|replay " "logs/v3_regen_verify_s${S}.log" || true
fi

# 2. the redraw campaign
if [ ! -s "$SC/events_notrunc_replicas_s${S}_n${N}.h5" ]; then
  JAX_PLATFORMS=cpu python scripts/regen_events_notrunc.py --seed "$S" --replicas "$N" \
      --dataroot "$ROOT" --pe_model v3 --rep_seed0 "${REP_SEED0:-9500000}" \
      --out "$SC/events_notrunc_replicas_s${S}_n${N}.h5" \
      > "logs/v3_regen_replicas_s${S}.log" 2>&1
fi

# 3. A on the replayed truths, per tracer
for T in gal agn; do
  KW=""; [ "$T" = gal ] && KW="--kde_window ${KDE_W:-4096}"
  python scripts/attr_abc_split.py --seed "$S" --tracer $T --dataroot "$ROOT" \
      --events "$D/events_${T}_hosted.h5" $KW --extra_only \
      --extra_truth "$SC/events_notrunc_replicas_s${S}_n${N}.h5" \
      --tag "${T}_v3_mega" > "logs/v3_abc_mega_${T}_s${S}.log" 2>&1
  tail -5 "logs/v3_abc_mega_${T}_s${S}.log"
done
