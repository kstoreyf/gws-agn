#!/usr/bin/env bash
# Pilot of the dedicated peak-family lane (seed 100, scratch only).
set -uo pipefail
D=/hildafs/projects/phy230014p/magana/gws-agn/working/data
REAL=/hildafs/projects/phy230054p/magana/gws-agn-data-v3
S=/hildafs/projects/phy230054p/magana/gws-agn-data-v3-scratch-injpeak
PIN=/hildafs/projects/phy230014p/magana/src/darksirens-2b86a2d
PY=/hildafs/home/magana/tmp_ondemand_hildafs_phy230014p_symlink/magana/.conda/envs/jax/bin/python
export PYTHONPATH=$PIN JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1
mkdir -p $S/pilot/seed100; ln -sfn $REAL/seed100/surveys $S/pilot/seed100/surveys
$PY -u $D/generate_dataset.py --seed 100 --stage injections --lane targeted --ndraw 100000000 --jobs 8 \
    --outroot $S/pilot --darksirens $PIN --inj_proposal_json "$(cat $D/obslike/scripts/inj_proposal_peak_pilot.json)" \
    --inj_suffix _peak --inj_seed 100016 --mix_population 0.62 --mix_uniform 0.03 --mix_targeted 0.35 \
    > $D/obslike/logs/inj/pilot_peak.log 2>&1
tail -n 2 $D/obslike/logs/inj/pilot_peak.log
