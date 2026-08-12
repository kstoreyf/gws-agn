#!/usr/bin/env bash
# Analysis-5 production campaign runner for hilda SLURM workers: runs every
# campaign tag SEQUENTIALLY, skipping tags whose result JSON already exists
# (m18/m18-r2/m19/m20 came back from the js2 H100), so only the missing rungs
# actually compute.  dynesty checkpoints every 15 min; a preempted or
# walltime-killed job resumes from the checkpoint on resubmission.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

SEED=100
DATA_ROOT=${DATA_ROOT:-/hildafs/projects/phy230014p/magana/gws-agn/working/data}
A3=../analysis_3_incomplete_catalog_H0_fagn/results

run_rung () {
  local RUNG=$1 RSTATE=$2 TAG=$3
  if [ -s "results/${TAG}.json" ]; then
    echo "${TAG} SKIP (result exists)" >> logs/campaign_progress.log
    return 0
  fi
  python -u scripts/sample_4d.py \
    --universe_model dark_sirens --catalog_sky_weighting field \
    --survey_path "${DATA_ROOT}/seed${SEED}/surveys/survey_gal_${RUNG}_ns32.h5" \
                  "${DATA_ROOT}/seed${SEED}/surveys/survey_agn_${RUNG}_ns32.h5" \
    --gw_path "${DATA_ROOT}/seed${SEED}/events/events.h5" \
    --gwselection_path "${DATA_ROOT}/seed${SEED}/injections/injections_targeted.h5" \
    --selection_neff_guard hard --max_likelihood_variance 1e6 \
    --kde_window 4096 --kde_window_nsigma 8 \
    --sel_batch_size 50000 --pe_event_block 25 \
    --h0_true 67.74 --f_true 0.295 --n0_true -3.0 --n0c2_true -5.0 \
    --n0_prior -4.0 -1.0 --n0c2_prior -6.0 -4.0 \
    --wiring_ref "${A3}/joint_${RUNG}_s100.h5" \
    --sampler dynesty --nlive 1000 --dlogz 0.1 --maxcall 500000 \
    --rstate_seed "${RSTATE}" \
    --checkpoint_file "results/${TAG}.ckpt" --checkpoint_every 900 \
    --outdir results --out_tag "${TAG}" --device gpu \
    > "logs/${TAG}.log" 2>&1 \
    && { echo "${TAG} DONE" >> logs/campaign_progress.log; rm -f "results/${TAG}.ckpt"; } \
    || echo "${TAG} FAILED rc=$?" >> logs/campaign_progress.log
}

run_rung m18 7  campaign_m18_dynesty_s100
run_rung m18 23 campaign_m18_dynesty_r2_s100
run_rung m19 7  campaign_m19_dynesty_s100
run_rung m20 7  campaign_m20_dynesty_s100
run_rung m21 7  campaign_m21_dynesty_s100
echo "CAMPAIGN_ALLDONE" >> logs/campaign_progress.log
