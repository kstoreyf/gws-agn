#!/usr/bin/env bash
# The dedicated peak-family injection lane for ONE realisation, beside the record's lanes:
#   seed<S>/injections/injections_targeted_peak.h5 (+ _meta.json)
#   ./inj_make_peak.sh SEED NDRAW MIXPOP MIXUNIF MIXTGT          (sbatch --array via submit_inj_make_peak.sbatch)
# Proposal: scripts/inj_proposal_peak.json (a mixture of legacy PopulationConfigs; exact pdraw);
# lane seed S*1000+16.  --overwrite is never passed; the suffix keeps the record and META.json
# untouched.  Then the pdraw check and the stressed-point effective numbers for s = 1 and s = 2
# (the s = 1 run also carries the s = 0 equal-population mixtures).
set -uo pipefail
S=${1:?seed}; ND=${2:?ndraw}; MP=${3:?}; MU=${4:?}; MT=${5:?}
D=/hildafs/projects/phy230014p/magana/gws-agn/working/data
REAL=/hildafs/projects/phy230054p/magana/gws-agn-data-v3
PIN=/hildafs/projects/phy230014p/magana/src/darksirens-2b86a2d
PY=/hildafs/home/magana/tmp_ondemand_hildafs_phy230014p_symlink/magana/.conda/envs/jax/bin/python
CPY=/hildafs/projects/phy230014p/magana/envs/darksirens-core-e7c3007/bin/python
L=$D/obslike/logs/inj
OUT=$REAL/seed$S/injections/injections_targeted_peak.h5
[ -e "$OUT" ] && { echo "exists: $OUT"; exit 1; }
t0=$(date +%s)
JAX_PLATFORMS=cpu PYTHONPATH=$PIN /usr/bin/time -v $PY -u $D/generate_dataset.py --seed $S --stage injections \
    --lane targeted --ndraw $ND --jobs 8 --outroot $REAL --darksirens $PIN \
    --inj_proposal_json "$(cat $D/obslike/scripts/inj_proposal_peak.json)" --inj_suffix _peak --inj_seed $((S*1000+16)) \
    --mix_population $MP --mix_uniform $MU --mix_targeted $MT > $L/make_peak_s$S.log 2>&1 || { echo "GENERATION FAILED seed $S"; exit 1; }
t1=$(date +%s)
echo "seed $S generated in $((t1 - t0)) s: $(grep -o 'detected [0-9,]*/[0-9,]*' $L/make_peak_s$S.log | tail -1)"
JAX_PLATFORMS=cpu PYTHONPATH=$PIN $PY -u $D/obslike/scripts/check_injections.py $OUT \
    $D/obslike/results/inj_check_peak_s$S.json > $L/check_peak_s$S.log 2>&1 && echo "seed $S check: $(grep -o '"pass": [a-z]*' $L/check_peak_s$S.log)"
cd $D/obslike/scripts
for M in 1 2; do
  JAX_PLATFORMS=cpu PYTHONPATH= $CPY -u inj_stress.py mock $S $OUT $D/obslike/results/inj_stress_peaklane_s${S}_x$M.json $M \
      > $L/stress_peak_s${S}_x$M.log 2>&1 &
done
wait
echo "seed $S done in $(( $(date +%s) - t0 )) s"
