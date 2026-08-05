#!/bin/bash
# One end-to-end catalog realisation of the deep two-tracer mock.
#
# Every headline two-tracer number in this programme -- f_AGN 1.8 sigma low, the
# H0 2.7 sigma offset, the completeness width ladder -- comes from ONE draw
# (seed 7301).  On the K=1 side the catalog realisation was measured to
# contribute 2.2x the quoted interval width, so those error bars are very
# probably too narrow and the f_AGN offset may be noise.  This reruns the whole
# chain per seed: fresh galaxy catalog, fresh AGN subset, fresh events, fresh
# targeted injection set, same estimator and settings as
# ../experiment_twotracer_deep's headline run.
#
# Usage: one_realisation.sh <seed> <stage>     stage in {cpu, gpu}
#   cpu : catalog -> mock -> injections     (parallel-safe across seeds)
#   gpu : guard + f-scan + joint scan        (serialise; one GPU)
set -euo pipefail
SEED=$1
STAGE=$2
cd "$(dirname "$0")/.."

WT=/hildafs/projects/phy230014p/magana/src/darksirens-pefix
GMD=$WT/scripts/mock_dark_sirens
DD=data_derived/s${SEED}
mkdir -p "$DD" logs results

if [ "$STAGE" = "cpu" ]; then
  # 1. a fresh 1M-host complete galaxy catalog (gmd's own draw)
  python scripts/build_obsdet_mock.py --mode catalog --detection true-params \
    --out_path "$DD/catalog_complete.h5" --seed "$SEED" \
    --dL_fractional_uncertainty 0.10 --n_galaxies 1000000 \
    > "logs/s${SEED}_catalog.log" 2>&1

  # 2. the two-tracer mock on it: AGN subset, 140+60 events, both surveys
  python scripts/build_twotracer_mock.py --complete_catalog "$DD/catalog_complete.h5" \
    --gmd_dir "$GMD" --outdir "$DD" --seed "$SEED" \
    > "logs/s${SEED}_mock.log" 2>&1

  # 3. the catalog-targeted injection set, matching the headline recipe exactly
  python scripts/build_targeted_injections_k2.py \
    --out_path "$DD/injections_targeted_k2.h5" --ndraw 120000000 --batch_size 4000000 \
    --mix_population 0.65 --mix_uniform 0.10 \
    --target "$DD/survey_agn_ns32.h5:0.25" \
    --seed $((SEED + 60000)) --validate_nsamp 200 \
    --validation_json "results/inj_s${SEED}_validation.json" \
    > "logs/s${SEED}_inj.log" 2>&1
  echo "CPU STAGE DONE seed=$SEED"
  exit 0
fi

if [ "$STAGE" = "gpu" ]; then
  SUR="$DD/survey_gal_ns32.h5 $DD/survey_agn_ns32.h5"
  GW="$DD/twotracer_gw_events.h5"
  SEL="$DD/injections_targeted_k2.h5"
  COMMON="--universe_model dark_sirens --catalog_sky_weighting field \
    --survey_path $SUR --gw_path $GW --gwselection_path $SEL \
    --log10n0 -12 --log10n0_c2 -12 \
    --selection_neff_guard hard --max_likelihood_variance 1e6 \
    --outdir results --h0_true 67.74 --f_true 0.3"

  python scripts/diag_variance_guard.py --universe_model dark_sirens \
    --catalog_sky_weighting field --survey_path $SUR --gw_path "$GW" \
    --gwselection_path "$SEL" --log10n0 -12 --log10n0_c2 -12 --f_at 0.3 \
    --max_likelihood_variance 1e6 --out_json "results/guard_s${SEED}.json" \
    > "logs/s${SEED}_guard.log" 2>&1
  python scripts/scan_h0f.py $COMMON --scan f --f_grid 0.0 1.0 41 \
    --out_tag "fscan_s${SEED}" > "logs/s${SEED}_fscan.log" 2>&1
  python scripts/scan_h0f.py $COMMON --scan joint --h0_grid 58.0 78.0 81 \
    --f_grid 0.0 1.0 41 --out_tag "joint_s${SEED}" > "logs/s${SEED}_joint.log" 2>&1
  echo "GPU STAGE DONE seed=$SEED"
  exit 0
fi

echo "unknown stage: $STAGE" >&2
exit 2
