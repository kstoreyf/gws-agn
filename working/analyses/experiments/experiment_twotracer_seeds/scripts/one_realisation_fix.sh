#!/bin/bash
# One end-to-end realisation of the deep two-tracer mock with the FIXED
# generator (darksirens-oraclefix @ fix/mock-observable-sky-width):
#   * detection_data=observed (PR #334): detection thresholds the SNR of the ONE
#     recorded measurement the posterior conditions on (no projection latent;
#     snr_ref recalibrated to 6.278, which reproduces the historical detected
#     fraction on this catalog family to 1% -- results/fixcheck_detection_sigma_ang.json);
#   * sequential observable sky width (PR #335): sigma_ang is derived from the
#     OBSERVED masses/distance, not the latent true parameters.
#
# What is REUSED from the pre-fix run (verified, results/fixcheck_*.json):
#   * catalog_complete.h5 per seed -- the catalog draw involves no sky width;
#   * nothing else.  Surveys are regenerated (verified bit-identical to the old
#     ones: the AGN-subset draw precedes the event draws in the rng stream) and
#     the injections MUST be regenerated because the detection rule changed
#     (old-vs-new detected sets share only 48/200 host redshifts at s7301).
#     Within the observed rule detection is provably independent of sigma_ang
#     (bit-identical detection masks under sequential vs fixed 5 deg sky width).
#
# Usage: one_realisation_fix.sh <seed> <stage>     stage in {cpu, gpu}
set -euo pipefail
SEED=$1
STAGE=$2
cd "$(dirname "$0")/.."

WT=/hildafs/projects/phy230014p/magana/src/darksirens-oraclefix
GMD=$WT/scripts/mock_dark_sirens
DD=data_derived/s${SEED}          # pre-fix realisation (catalog reused from here)
DDF=data_derived/s${SEED}_fix     # fixed-generator outputs
SNR_REF=6.278
mkdir -p "$DDF" logs results

if [ "$STAGE" = "cpu" ]; then
  # Idempotent: outputs are deterministic in the seed, so completed work
  # (marked by the injection validation json, written last) is skipped.
  if [ -f "$DDF/twotracer_gw_events.h5" ] && [ -f "$DDF/injections_targeted_k2.h5" ] \
     && [ -f "results/inj_fix_s${SEED}_validation.json" ]; then
    echo "CPU STAGE ALREADY DONE seed=$SEED (fix)"
    exit 0
  fi
  # 1. catalog REUSED from the pre-fix run (sky-width independent).
  test -f "$DD/catalog_complete.h5" || { echo "missing $DD/catalog_complete.h5" >&2; exit 3; }

  # 2. the two-tracer mock with the fixed generator (same seed => same AGN
  #    subset and bit-identical surveys; events re-drawn under the fixed rule).
  python scripts/build_twotracer_mock.py --complete_catalog "$DD/catalog_complete.h5" \
    --gmd_dir "$GMD" --outdir "$DDF" --seed "$SEED" \
    --detection_data observed --snr_ref "$SNR_REF" \
    > "logs/s${SEED}_fix_mock.log" 2>&1

  # 3. targeted injections under the SAME observed-data detection rule.
  python scripts/build_targeted_injections_k2.py \
    --out_path "$DDF/injections_targeted_k2.h5" --ndraw 120000000 --batch_size 4000000 \
    --mix_population 0.65 --mix_uniform 0.10 \
    --target "$DDF/survey_agn_ns32.h5:0.25" \
    --worktree "$WT" --detection_data observed --snr_ref "$SNR_REF" \
    --seed $((SEED + 60000)) --validate_nsamp 200 \
    --validation_json "results/inj_fix_s${SEED}_validation.json" \
    > "logs/s${SEED}_fix_inj.log" 2>&1
  echo "CPU STAGE DONE seed=$SEED (fix)"
  exit 0
fi

if [ "$STAGE" = "gpu" ]; then
  # Idempotent: the joint json is written last, so its presence (with the fscan
  # json) marks a completed seed.  Lets an interrupted GPU stage resume.
  if [ -f "results/joint_fix_s${SEED}.json" ] && [ -f "results/fscan_fix_s${SEED}.json" ] \
     && [ -f "results/guard_fix_s${SEED}.json" ]; then
    echo "GPU STAGE ALREADY DONE seed=$SEED (fix)"
    exit 0
  fi
  SUR="$DDF/survey_gal_ns32.h5 $DDF/survey_agn_ns32.h5"
  GW="$DDF/twotracer_gw_events.h5"
  SEL="$DDF/injections_targeted_k2.h5"
  COMMON="--universe_model dark_sirens --catalog_sky_weighting field \
    --survey_path $SUR --gw_path $GW --gwselection_path $SEL \
    --log10n0 -12 --log10n0_c2 -12 \
    --selection_neff_guard hard --max_likelihood_variance 1e6 \
    --outdir results --h0_true 67.74 --f_true 0.3"

  python scripts/diag_variance_guard.py --universe_model dark_sirens \
    --catalog_sky_weighting field --survey_path $SUR --gw_path "$GW" \
    --gwselection_path "$SEL" --log10n0 -12 --log10n0_c2 -12 --f_at 0.3 \
    --max_likelihood_variance 1e6 --out_json "results/guard_fix_s${SEED}.json" \
    > "logs/s${SEED}_fix_guard.log" 2>&1
  python scripts/scan_h0f.py $COMMON --scan f --f_grid 0.0 1.0 41 \
    --out_tag "fscan_fix_s${SEED}" > "logs/s${SEED}_fix_fscan.log" 2>&1
  python scripts/scan_h0f.py $COMMON --scan joint --h0_grid 58.0 78.0 81 \
    --f_grid 0.0 1.0 41 --out_tag "joint_fix_s${SEED}" > "logs/s${SEED}_fix_joint.log" 2>&1
  echo "GPU STAGE DONE seed=$SEED (fix)"
  exit 0
fi

echo "unknown stage: $STAGE" >&2
exit 2
