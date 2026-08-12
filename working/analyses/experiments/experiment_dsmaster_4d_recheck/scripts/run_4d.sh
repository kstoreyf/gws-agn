#!/usr/bin/env bash
# Stage B -- the full 4D fit on darksirens master, at the analysis-5 settings.
#
# rstate 7 is analysis 5's own m18 state, so results/fit_m18_master_s100.json
# is directly comparable to A5's campaign_m18_dynesty_s100.json cell by cell,
# and A5's rstate-23 twin gives the sampling-noise floor to judge it against.
#
# m18 cost 5731 s on the js2 H100; budget ~2-3 h on a rita A100-80.  dynesty
# checkpoints every 15 min, so a walltime kill resumes on resubmission.
set -euo pipefail
cd "$(dirname "$0")/.."
. scripts/env.sh
mkdir -p logs results

RSTATE=${RSTATE:-7}
CMODE=${CMODE:-aggregate}
TAG=fit_${RUNG}_${CMODE}_s${SEED}

# Empty for the aggregate/per_pixel arms, so their command line is unchanged.
SEL_ARGS=()
if [ -n "${SELECTION_FIT:-}" ]; then
  SEL_ARGS=(--selection_fit "${SELECTION_FIT}")
fi

python -u scripts/sample_4d.py $(ds_4d_common) \
  --sampler dynesty --nlive 1000 --dlogz 0.1 --maxcall 500000 \
  --rstate_seed "${RSTATE}" --wiring_nonfatal --c_mode "$CMODE" \
  "${SEL_ARGS[@]}" \
  --checkpoint_file "results/${TAG}.ckpt" --checkpoint_every 900 \
  --outdir results --out_tag "$TAG" \
  > "logs/${TAG}.log" 2>&1

rm -f "results/${TAG}.ckpt"
echo "${TAG} DONE"
