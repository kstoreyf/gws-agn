#!/usr/bin/env bash
# One extra record-lane injection set (Analysis 14's recipe, make_injection_redraws.py) + the pdraw check.
#   ./a14_redraw_job.sh SEED SET
set -uo pipefail
S=${1:?seed}; J=${2:?set}
A14=/hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_14_monte_carlo_accuracy/scripts
D=/hildafs/projects/phy230014p/magana/gws-agn/working/data
PY=/hildafs/home/magana/tmp_ondemand_hildafs_phy230014p_symlink/magana/.conda/envs/jax/bin/python
export JAX_PLATFORMS=cpu PYTHONUNBUFFERED=1
cd $A14 && $PY make_injection_redraws.py --sets $J --jobs 8 --seed $S > $D/obslike/logs/inj/a14redraw_s${S}_set$J.log 2>&1 \
  || { echo "seed $S set $J: GENERATION FAILED"; exit 1; }
F=$(realpath $D/seed$S)/mc_accuracy/inj_set$J/seed$S/injections/injections_targeted.h5
PYTHONPATH=/hildafs/projects/phy230014p/magana/src/darksirens-2b86a2d $PY $D/obslike/scripts/check_injections.py $F \
    $D/obslike/results/inj_check_a14redraw_s${S}_set$J.json > $D/obslike/logs/inj/a14check_s${S}_set$J.log 2>&1
echo "seed $S set $J: $(python3 -c "import json;d=json.load(open('$D/obslike/results/inj_check_a14redraw_s${S}_set$J.json'));print('rows',d['n_detected'],'pass',d['pass'])")"
