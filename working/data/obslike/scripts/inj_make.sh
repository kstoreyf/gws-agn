#!/usr/bin/env bash
# The dedicated obslike injection set for ONE realisation, written beside the record's lanes:
#   seed<S>/injections/injections_targeted_obslike.h5 (+ _meta.json)
#
#   sbatch --array=0-4 submit_inj_make.sbatch      (task i -> seed in SEEDS[i])
#   ./inj_make.sh SEED NDRAW MIXPOP MIXUNIF MIXTGT
#
# Proposal: scripts/inj_proposal.json (mass/spin of the population and targeted branches),
# record's uniform branch, targeted branch with host cut and box cap 0.75, lane seed
# S*1000+12 (no other stream uses it).  --overwrite is never passed; the suffix keeps the
# record's files and META.json untouched.  Then the V6-style check and the stressed-point
# effective numbers.
set -uo pipefail
S=${1:?seed}; ND=${2:?ndraw}; MP=${3:?}; MU=${4:?}; MT=${5:?}
D=/hildafs/projects/phy230014p/magana/gws-agn/working/data
REAL=/hildafs/projects/phy230054p/magana/gws-agn-data-v3
PIN=/hildafs/projects/phy230014p/magana/src/darksirens-2b86a2d
PY=/hildafs/home/magana/tmp_ondemand_hildafs_phy230014p_symlink/magana/.conda/envs/jax/bin/python
CPY=/hildafs/projects/phy230014p/magana/envs/darksirens-core-e7c3007/bin/python
L=$D/obslike/logs/inj
mkdir -p $L
OUT=$REAL/seed$S/injections/injections_targeted_obslike.h5
[ -e "$OUT" ] && { echo "exists: $OUT"; exit 1; }
PROP=$(cat $D/obslike/scripts/inj_proposal.json)
t0=$(date +%s)
JAX_PLATFORMS=cpu PYTHONPATH=$PIN /usr/bin/time -v $PY -u $D/generate_dataset.py --seed $S --stage injections \
    --lane targeted --ndraw $ND --jobs 8 --outroot $REAL --darksirens $PIN \
    --inj_proposal_json "$PROP" --inj_suffix _obslike --inj_seed $((S*1000+12)) \
    --tgt_host_zmax 0.75 --tgt_z_cap 0.75 --mix_population $MP --mix_uniform $MU --mix_targeted $MT \
    > $L/make_s$S.log 2>&1 || { echo "GENERATION FAILED seed $S"; exit 1; }
t1=$(date +%s)
echo "seed $S generated in $((t1 - t0)) s: $(grep -o 'detected [0-9,]*/[0-9,]*' $L/make_s$S.log | tail -1)"
JAX_PLATFORMS=cpu PYTHONPATH=$PIN $PY -u $D/obslike/scripts/check_injections.py $OUT \
    $D/obslike/results/inj_check_s$S.json > $L/check_s$S.log 2>&1 && echo "seed $S check: $(grep -o '"pass": [a-z]*' $L/check_s$S.log)"
cd $D/obslike/scripts && JAX_PLATFORMS=cpu PYTHONPATH= $CPY -u inj_stress.py obslike $S $OUT \
    $D/obslike/results/inj_stress_dedicated_s$S.json > $L/stress_s$S.log 2>&1
echo "seed $S stress: $(grep '^worst' $L/stress_s$S.log | cut -c1-300)  total $(( $(date +%s) - t0 )) s"
