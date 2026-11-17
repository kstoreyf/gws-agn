#!/usr/bin/env bash
# Large-N test of the planted share at f = 0.5 (scratch only): 20,000 detected events, sub-seed S*1000+90.
set -uo pipefail
D=/hildafs/projects/phy230014p/magana/gws-agn/working/data
REAL=/hildafs/projects/phy230054p/magana/gws-agn-data-v3
S=/hildafs/projects/phy230054p/magana/gws-agn-data-v3-scratch-injobs/v5test
PIN=/hildafs/projects/phy230014p/magana/src/darksirens-2b86a2d
PY=/hildafs/home/magana/tmp_ondemand_hildafs_phy230014p_symlink/magana/.conda/envs/jax/bin/python
export PYTHONPATH=$PIN JAX_PLATFORMS=cpu OMP_NUM_THREADS=2
for SEED in 100 103; do
  mkdir -p $S/seed$SEED/events; ln -sfn $REAL/seed$SEED/catalogs $S/seed$SEED/catalogs
  ( $PY -u $D/generate_dataset.py --seed $SEED --stage events --outroot $S --darksirens $PIN --population_code core \
      --f_agn 0.5 --seed_events $((SEED*1000+90)) --n_events 20000 --nsamp 10 --events_suffix _v5test \
      > $D/obslike/logs/v5_largeN_s$SEED.log 2>&1; \
    $PY -c "import h5py,numpy as np; f=h5py.File('$S/seed$SEED/events/events_v5test.h5','r'); h=f['truth/host_type'][:]; print($SEED, 'n', h.size, 'AGN share %.4f +- %.4f'%(h.mean(), np.sqrt(h.mean()*(1-h.mean())/h.size)))" ) &
done
wait
