#!/usr/bin/env bash
# Regressions of the mixture-proposal edit: (a) default injection path vs git HEAD (6e6
# proposals); (b) the single-proposal path vs the stored seed-100 obslike lane (its first
# two batches, which do not depend on the number of batches).
set -uo pipefail
D=/hildafs/projects/phy230014p/magana/gws-agn/working/data
REAL=/hildafs/projects/phy230054p/magana/gws-agn-data-v3
S=/hildafs/projects/phy230054p/magana/gws-agn-data-v3-scratch-injpeak
PIN=/hildafs/projects/phy230014p/magana/src/darksirens-2b86a2d
PY=/hildafs/home/magana/tmp_ondemand_hildafs_phy230014p_symlink/magana/.conda/envs/jax/bin/python
BITCHECK=/hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_0_pure_tracer_H0/scripts/bitcheck_events.py
export PYTHONPATH=$PIN JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1
mkdir -p $S/code; git -C /hildafs/projects/phy230014p/magana/gws-agn show HEAD:working/data/generate_dataset.py > $S/code/gen_HEAD.py
for SIDE in head post obs; do mkdir -p $S/r_$SIDE/seed100; ln -sfn $REAL/seed100/surveys $S/r_$SIDE/seed100/surveys; done
$PY -u $S/code/gen_HEAD.py --seed 100 --stage injections --lane targeted --ndraw 6000000 --jobs 4 --outroot $S/r_head --darksirens $PIN > $D/obslike/logs/inj/regress2_head.log 2>&1 &
$PY -u $D/generate_dataset.py --seed 100 --stage injections --lane targeted --ndraw 6000000 --jobs 4 --outroot $S/r_post --darksirens $PIN > $D/obslike/logs/inj/regress2_post.log 2>&1 &
$PY -u $D/generate_dataset.py --seed 100 --stage injections --lane targeted --ndraw 4000000 --jobs 2 --outroot $S/r_obs --darksirens $PIN \
    --inj_proposal_json "$(cat $D/obslike/scripts/inj_proposal.json)" --inj_suffix _obslike --inj_seed 100012 \
    --tgt_host_zmax 0.75 --tgt_z_cap 0.75 --mix_population 0.62 --mix_uniform 0.03 --mix_targeted 0.35 > $D/obslike/logs/inj/regress2_obs.log 2>&1 &
wait
$PY $BITCHECK $S/r_head/seed100/injections/injections_targeted.h5 $S/r_post/seed100/injections/injections_targeted.h5 \
    --out $D/obslike/results/inj_regress2_default_bitcheck.json > /dev/null && echo "default path vs HEAD: PASS" || echo "default path vs HEAD: FAIL"
$PY - <<'PYEOF'
import h5py, numpy as np, json
a = h5py.File("/hildafs/projects/phy230054p/magana/gws-agn-data-v3/seed100/injections/injections_targeted_obslike.h5", "r")
b = h5py.File("/hildafs/projects/phy230054p/magana/gws-agn-data-v3-scratch-injpeak/r_obs/seed100/injections/injections_targeted_obslike.h5", "r")
n = b["pdraw"].shape[0]
res = {k: bool(np.array_equal(a[k][:n], b[k][:])) for k in b.keys()}
ok = all(res.values())
json.dump({"n_rows_compared": int(n), "identical": res, "pass": ok},
          open("/hildafs/projects/phy230014p/magana/gws-agn/working/data/obslike/results/inj_regress2_single_proposal.json", "w"), indent=1)
print("single-proposal path vs stored obslike lane (first %d rows): %s" % (n, "PASS" if ok else "FAIL " + str(res)))
PYEOF
