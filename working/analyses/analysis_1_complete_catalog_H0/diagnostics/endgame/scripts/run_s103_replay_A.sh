#!/usr/bin/env bash
# ENDGAME: (A - B) on a SECOND catalog (seed 103), to show the offset is not a
# property of seed 100's realisation.  Wants the GPU to itself.
set -u
cd "$(dirname "$0")/.."
MEGA=/hildafs/projects/phy220048p/magana/gws-agn-data/scratch_truncation_test/events_notrunc_replicas_s103_n500.h5
for T in agn gal; do
  echo "=== A on the seed-103 replay, $T ==="
  python scripts/attr_abc_split.py --seed 103 --tracer "$T" --extra_only \
      --extra_truth "$MEGA" --truth_batch 20000 --tag "${T}_mega_s103" \
      > "logs/abc_${T}_mega_s103.log" 2>&1 || echo "[FAIL] $T"
  python - "$T" <<'PY'
import json,sys,numpy as np
t=sys.argv[1]
B=json.load(open(f'results/attr_selmu_{t}_s103.json'))['dlnmu_at_truth']['kde']
Z=np.load(f'results/abc_{t}_mega_s103.npz'); S=Z['X_tot']; r=Z['Xg_rank']
for nm,m in (('head',r<1000),('tail',r>=1000),('full',np.ones_like(r,bool))):
    x=S[m]; se=x.std(ddof=1)/np.sqrt(x.size)
    print(f'  seed103 {t} {nm}: n={m.sum():>8} A={x.mean():.7e} B={B:.7e} '
          f'A-B={x.mean()-B:+.4e} +- {se:.2e} ({(x.mean()-B)/se:+.2f} sigma)')
PY
done
echo "S103 REPLAY DONE"
