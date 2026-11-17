#!/usr/bin/env bash
# Pilot of the dedicated obslike injection lane (seed 100, scratch only) + regression of the
# default injection path.  ./inj_pilot.sh regress | pilot TAG MIXPOP MIXUNIF MIXTGT NDRAW
set -uo pipefail
D=/hildafs/projects/phy230014p/magana/gws-agn/working/data
REAL=/hildafs/projects/phy230054p/magana/gws-agn-data-v3
S=/hildafs/projects/phy230054p/magana/gws-agn-data-v3-scratch-injobs
PIN=/hildafs/projects/phy230014p/magana/src/darksirens-2b86a2d
PY=/hildafs/home/magana/tmp_ondemand_hildafs_phy230014p_symlink/magana/.conda/envs/jax/bin/python
BITCHECK=/hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_0_pure_tracer_H0/scripts/bitcheck_events.py
export PYTHONPATH=$PIN JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1
PROP=$(cat $D/obslike/scripts/inj_proposal.json)
case $1 in
regress)
  for SIDE in head post; do
    mkdir -p $S/regress_$SIDE/seed100; ln -sfn $REAL/seed100/surveys $S/regress_$SIDE/seed100/surveys
  done
  $PY -u $S/code/generate_dataset_HEAD.py --seed 100 --stage injections --lane targeted --ndraw 6000000 \
      --jobs 8 --outroot $S/regress_head --darksirens $PIN > $D/obslike/logs/inj/regress_head.log 2>&1 &
  $PY -u $D/generate_dataset.py --seed 100 --stage injections --lane targeted --ndraw 6000000 \
      --jobs 8 --outroot $S/regress_post --darksirens $PIN > $D/obslike/logs/inj/regress_post.log 2>&1 &
  wait
  $PY $BITCHECK $S/regress_head/seed100/injections/injections_targeted.h5 \
      $S/regress_post/seed100/injections/injections_targeted.h5 \
      --out $D/obslike/results/inj_regress_default_bitcheck.json > /dev/null && echo "default injection path: PASS" || echo "default injection path: FAIL"
  ;;
pilot)
  TAG=$2; MP=$3; MU=$4; MT=$5; ND=$6
  mkdir -p $S/pilot_$TAG/seed100; ln -sfn $REAL/seed100/surveys $S/pilot_$TAG/seed100/surveys
  $PY -u $D/generate_dataset.py --seed 100 --stage injections --lane targeted --ndraw $ND --jobs 8 \
      --outroot $S/pilot_$TAG --darksirens $PIN --inj_proposal_json "$PROP" --inj_suffix _obslike \
      --inj_seed 100012 --tgt_host_zmax 0.75 --tgt_z_cap 0.75 \
      --mix_population $MP --mix_uniform $MU --mix_targeted $MT > $D/obslike/logs/inj/pilot_$TAG.log 2>&1
  tail -n 2 $D/obslike/logs/inj/pilot_$TAG.log
  ;;
esac
