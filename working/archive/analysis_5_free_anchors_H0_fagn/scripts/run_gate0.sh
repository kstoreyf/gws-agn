#!/usr/bin/env bash
# Gate-0 pilot lanes: $1 = dynesty | emcee.  Rung m18, seed 100, targeted lane —
# the exact analysis-3/4 configuration of record with the two completion-density
# anchors freed.  Wiring refs: analysis-3 m18 exact grid (anchors at truth,
# includes the f=0/f=1 endpoint columns) + two analysis-4 off-truth arms.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

LANE=${1:?usage: run_gate0.sh dynesty|emcee}
SEED=100
DATA_ROOT=${DATA_ROOT:-/hildafs/projects/phy230014p/magana/gws-agn/working/data}
A3=../analysis_3_incomplete_catalog_H0_fagn/results
A4=../analysis_4_density_anchoring_H0_fagn/results

COMMON=(
  --universe_model dark_sirens
  --catalog_sky_weighting field
  --survey_path "${DATA_ROOT}/seed${SEED}/surveys/survey_gal_m18_ns32.h5"
                "${DATA_ROOT}/seed${SEED}/surveys/survey_agn_m18_ns32.h5"
  --gw_path "${DATA_ROOT}/seed${SEED}/events/events.h5"
  --gwselection_path "${DATA_ROOT}/seed${SEED}/injections/injections_targeted.h5"
  --selection_neff_guard hard --max_likelihood_variance 1e6
  --kde_window 4096 --kde_window_nsigma 8
  --sel_batch_size 50000 --pe_event_block 25
  --h0_true 67.74 --f_true 0.295 --n0_true -3.0 --n0c2_true -5.0
  --wiring_ref "${A3}/joint_m18_s100.h5"
  --wiring_ref "${A4}/joint_m18_a05_s100.h5"
  --wiring_ref "${A4}/joint_m18_a07_s100.h5"
  --device gpu
)

case "$LANE" in
  dynesty)
    exec python -u scripts/sample_4d.py "${COMMON[@]}" \
      --sampler dynesty --nlive 200 --dlogz 0.1 --maxcall 80000 --rstate_seed 7 \
      --out_tag gate0_m18_dynesty_s${SEED} 2>&1 | tee logs/gate0_m18_dynesty_s${SEED}.log
    ;;
  emcee)
    exec python -u scripts/sample_4d.py "${COMMON[@]}" \
      --sampler emcee --nwalkers 32 --nsteps 1500 --burn_frac 0.5 --rstate_seed 11 \
      --init_ref_h5 "${A3}/joint_m18_s100.h5" \
      --out_tag gate0_m18_emcee_s${SEED} 2>&1 | tee logs/gate0_m18_emcee_s${SEED}.log
    ;;
  *) echo "unknown lane: $LANE" >&2; exit 2 ;;
esac
