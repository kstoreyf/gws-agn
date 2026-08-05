#!/usr/bin/env bash
# GATE (c) -- the per-cell validity guard and the selection integral's N_eff at
# every rung of the ladder, at the peak (H0 = 67.74, f = 0.30).
#
# analysis_2 found the guard inert everywhere on the complete pair (0 of 41,205
# cells rejected).  The question here is whether switching the out-of-catalog
# field term on and emptying the catalog changes that.  The prototype
# (experiment_twotracer_incomplete) found N_eff RISING along the ladder, because
# the incomplete model's target is increasingly dominated by the smooth field
# term, which the population branch of the proposal covers well; this is the
# check of that on the campaign's own data and injections.
#
# diag_variance_guard.py takes delta = delta_c2 = 0 by construction, which IS the
# analysis-3 setting, so unlike the prototype there is no delta mismatch between
# the diagnostic and the scans.
set -euo pipefail
cd "$(dirname "$0")/.."
. scripts/env.sh
mkdir -p logs results/guard

SEED=${SEED:-100}
EV=${DATA_ROOT}/seed${SEED}/events/events.h5
for LEVEL in ${GUARD_LEVELS:-m18 m19 m20 m21 complete}; do
  for LANE in ${GUARD_LANES:-targeted}; do
    SUF=""; [ "$LANE" = "popuni" ] && SUF="_popuni"
    echo "=== guard: level $LEVEL, lane $LANE, seed $SEED ==="
    python -u scripts/diag_variance_guard.py \
      --universe_model dark_sirens --catalog_sky_weighting field \
      --survey_path ${DATA_ROOT}/seed${SEED}/surveys/survey_gal_${LEVEL}_ns32.h5 \
                    ${DATA_ROOT}/seed${SEED}/surveys/survey_agn_${LEVEL}_ns32.h5 \
      --gw_path "$EV" \
      --gwselection_path ${DATA_ROOT}/seed${SEED}/injections/injections_${LANE}.h5 \
      --log10n0 ${LOG10N0} --log10n0_c2 ${LOG10N0_C2} \
      --h0_at ${H0_TRUE} --f_at ${F_TRUE} \
      --max_likelihood_variance 1e6 \
      --kde_window ${KDE_W} --kde_window_nsigma 8 \
      --sel_batch_size ${SEL_BATCH} --pe_event_block ${PE_BLOCK} \
      --capture_event_vars \
      --out_json "results/guard/guard_${LEVEL}_s${SEED}${SUF}.json" \
      > "logs/guard_${LEVEL}_s${SEED}${SUF}.log" 2>&1
    tail -20 "logs/guard_${LEVEL}_s${SEED}${SUF}.log"
  done
done
