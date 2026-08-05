#!/usr/bin/env bash
# ENDGAME: does (A - B) track the survey block's DECLARED photo-z kernel width?
# The block's zgals are the galaxies' exact redshifts (bitwise, verified), so the
# likelihood's p_z(z|pix) is a KDE of width dz = DZ_SCALE (1+z) around the very
# redshifts the host draw uses.  Rescaling DZ_SCALE and re-measuring (A - B) with
# the EXACT oracle B on the same block identifies the channel.
set -u
cd "$(dirname "$0")/.."
SC=/hildafs/projects/phy220048p/magana/gws-agn-data/scratch_truncation_test
MEGA=$SC/events_notrunc_replicas_s100_n1500.h5
T=${T:-agn}
for S in ${SCALES:-0.5 2 3}; do
  TAG=$(echo "x$S" | tr '.' 'p')
  SUR=$SC/surveys_dz/survey_${T}_complete_ns32_dz${TAG}.h5
  [ -f "$SUR" ] || { echo "missing $SUR"; continue; }
  if [ ! -f "results/attr_selmu_${T}_dz${TAG}.json" ]; then
    echo "=== exact B, $T dz$TAG ==="
    python scripts/attr_selmu_oracle.py --tracer "$T" --seed 100 \
      --survey_override "$SUR" --tag "${T}_dz${TAG}" \
      > "logs/attr_selmu_${T}_dz${TAG}.log" 2>&1 || echo "[FAIL B] $T $S"
  fi
  if [ ! -f "results/abc_${T}_mega_dz${TAG}.json" ]; then
    echo "=== A on the replay, $T dz$TAG ==="
    python scripts/attr_abc_split.py --seed 100 --tracer "$T" --extra_only \
      --extra_truth "$MEGA" --survey_override "$SUR" --truth_batch 20000 \
      --tag "${T}_mega_dz${TAG}" \
      > "logs/abc_${T}_mega_dz${TAG}.log" 2>&1 || echo "[FAIL A] $T $S"
  fi
  python - "$T" "$TAG" <<'PY'
import json,sys,numpy as np
t,tag=sys.argv[1],sys.argv[2]
try:
    B=json.load(open(f'results/attr_selmu_{t}_dz{tag}.json'))['dlnmu_at_truth']['kde']
    Z=np.load(f'results/abc_{t}_mega_dz{tag}.npz'); S=Z['X_tot']; r=Z['Xg_rank']
    for nm,m in (('head',r<1000),('full',np.ones_like(r,bool))):
        x=S[m]; se=x.std(ddof=1)/np.sqrt(x.size)
        print(f'  dz{tag} {nm}: A={x.mean():.7e} B={B:.7e} A-B={x.mean()-B:+.4e} +- {se:.2e}')
except Exception as e:
    print('  [pending]',e)
PY
done
echo "DZSCAN DONE"
