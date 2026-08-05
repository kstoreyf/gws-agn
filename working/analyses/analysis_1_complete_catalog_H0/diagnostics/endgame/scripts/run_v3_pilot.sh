#!/usr/bin/env bash
# THE v3 CLOSURE PILOT -- the gate the owner set before any full regeneration.
#
#   ./scripts/run_v3_pilot.sh [SEED] [DATAROOT]
#
# Measures, on the v3 dataset in DATAROOT (NOT the record tree):
#   * the exact selection oracle B = d ln mu/dH0 under the NEW detection rule
#     P_det = Phi((rho_opt - 8)/sigma_rho)                  [attr_selmu_oracle --pe_model v3]
#   * the closed-form P_det against the generator's own observe_v3/detect_v3
#                                                            [attr_selmu_pdet --pe_model v3]
#   * A (truth), B (injections), C (posterior-averaged) and the (A-B)/(C-A) split
#     on both matched-host controls                          [attr_abc_split]
#
# REQUIRE, before proceeding to the full regeneration:
#   (A - B) and (C - A) both consistent with zero at the pilot's precision.
set -euo pipefail
cd "$(dirname "$0")/.."
S=${1:-100}
ROOT=${2:-/hildafs/projects/phy220048p/magana/gws-agn-data-v3}
D=$ROOT/seed$S
mkdir -p logs results

# ---- matched-host subsets (the two controls) --------------------------------
for T in gal agn; do
  HT=0; [ "$T" = agn ] && HT=1
  [ -f "$D/events_${T}_hosted.h5" ] || \
    python scripts/build_hosttype_subset.py --in_path "$D/events/events.h5" \
        --out_path "$D/events_${T}_hosted.h5" --host_type $HT
done

# ---- P_det in closed form, against the generator itself ----------------------
if [ ! -s results/attr_selmu_pdet_v3.json ]; then
  JAX_PLATFORMS=cpu python scripts/attr_selmu_pdet.py --pe_model v3 \
      --n_mc "${PDET_NMC:-2e7}" > logs/v3_pdet.log 2>&1
fi

# ---- the EXACT selection oracle under the new rule ---------------------------
for T in agn gal; do
  python scripts/attr_selmu_oracle.py --tracer $T --seed "$S" --pe_model v3 \
      --dataroot "$ROOT" --events "$D/events_${T}_hosted.h5" \
      --conv_lat --tag "${T}_v3_s${S}" \
      > "logs/v3_selmu_${T}_s${S}.log" 2>&1
done

# ---- A / B / C and the split -------------------------------------------------
for T in gal agn; do
  KW=""; [ "$T" = gal ] && KW="--kde_window ${KDE_W:-4096}"
  python scripts/attr_abc_split.py --seed "$S" --tracer $T --dataroot "$ROOT" \
      --events "$D/events_${T}_hosted.h5" $KW \
      --pe_batch_events "${PE_BLOCK:-25}" --sel_batch "${SEL_BATCH:-50000}" \
      --tag "${T}_v3_s${S}" > "logs/v3_abc_${T}_s${S}.log" 2>&1
done

echo "pilot products:"
ls -la results/attr_selmu_pdet_v3.json results/attr_selmu_*_v3_s${S}.json \
       results/abc_*_v3_s${S}.json 2>/dev/null || true
