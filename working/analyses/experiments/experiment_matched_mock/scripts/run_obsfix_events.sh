#!/bin/bash
# Regenerate the obs arm's EVENTS with the PR #335 sky-width fix
# (--sky_width observed): sigma_ang derived from the OBSERVED amplitude,
# drawn sequentially after dL/m1det/m2det.  Catalogs, surveys and sel_obs.h5
# are REUSED -- detection does not involve the sky, so the detected sets are
# bit-identical to the old obs arm (verified on n4201: only ra/dec/sigma_ang
# columns differ; truth/snr and every mass/distance column identical).
set -euo pipefail
cd "$(dirname "$0")/.."
export JAX_PLATFORMS=cpu
SNR_OBS=6.278363879917771
WT=/hildafs/projects/phy230014p/magana/src/darksirens-oraclefix
DD=data_derived/obsdet

run () {  # tag catalog seed
  local tag=$1 cat=$2 seed=$3
  python scripts/build_obsdet_mock.py --mode events --detection observed-data \
    --catalog "$cat" --seed "$seed" --snr_ref "$SNR_OBS" \
    --out_path "$DD/ev_obsfix_${tag}.h5" --dL_fractional_uncertainty 0.10 \
    --sky_width observed --worktree "$WT" \
    --summary_json "results/obsdet_ev_obsfix_${tag}.json" \
    > "logs/obsdet_ev_obsfix_${tag}.log" 2>&1
  echo "done $tag"
}

run b     data_derived/deep_mock_z2_big/mock_galaxy_catalog_complete.h5 4101 &
run s4102 data_derived/pefix_s4102/mock_galaxy_catalog_complete.h5      4102 &
run s4103 data_derived/pefix_s4103/mock_galaxy_catalog_complete.h5      4103 &
run s4104 data_derived/pefix_s4104/mock_galaxy_catalog_complete.h5      4104 &
run s4105 data_derived/pefix_s4105/mock_galaxy_catalog_complete.h5      4105 &
wait
N=0
for s in $(seq 4202 4215); do
  run "n$s" "$DD/cat_n$s.h5" "$s" &
  N=$((N+1))
  if [ $((N % 5)) -eq 0 ]; then wait; fi
done
wait
echo "ALL OBSFIX EVENTS DONE"
