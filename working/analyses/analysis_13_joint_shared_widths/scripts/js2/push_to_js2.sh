#!/bin/bash
# Run on hilda. Ships core bf58aa6, the A13 driver, the seed-100 inputs and the pre-flight
# reference to the VM, then builds the venv there (idempotent; safe to re-run).
set -euo pipefail
source "$(dirname "$0")/config.sh"
ssh $JS2 "mkdir -p $JS2_CORE $JS2_A13/{scripts,results,queue,logs,ref} $JS2_DATA/{surveys,events,injections} $JS2_ROOT/env"
[ "$(git -C $CORE_SRC rev-parse HEAD)" = "$CORE_SHA" ] || { echo "core worktree is not at $CORE_SHA"; exit 1; }
git -C $CORE_SRC archive --format=tar $CORE_SHA | ssh $JS2 "tar -x -C $JS2_CORE && echo $CORE_SHA > $JS2_CORE/CORE_SHA"
rsync -a $A13/scripts/a13_core_sampler.py $A13/scripts/js2/run_a13_js2.sh $JS2:$JS2_A13/scripts/
rsync -a $REF_CELLS $JS2:$JS2_A13/ref/
for f in $A13_INPUTS; do rsync -aL --partial --info=progress2 $SEED100/$f $JS2:$JS2_DATA/$f; done
# checksums: hilda vs VM
( cd $SEED100 && sha256sum $A13_INPUTS ) > /tmp/a13_js2_inputs.sha256.$$
scp -q /tmp/a13_js2_inputs.sha256.$$ $JS2:$JS2_DATA/SHA256SUMS && rm /tmp/a13_js2_inputs.sha256.$$
ssh $JS2 "cd $JS2_DATA && sha256sum -c --quiet SHA256SUMS && echo inputs verified"
ssh $JS2 bash -s <<REMOTE
set -euo pipefail
if [ ! -x $JS2_ENV/bin/python ]; then
  /usr/bin/python3.11 -m venv $JS2_ENV
  $JS2_ENV/bin/pip install -q --upgrade pip
  # core bf58aa6 pins + the CUDA-12 jax plugin and dynesty versions hilda runs (2.1.4)
  ~/qchieffenv/bin/pip freeze | grep -E '^(jax-cuda12-pjrt|jax-cuda12-plugin|nvidia-[a-z0-9-]+-cu12)==' > $JS2_ROOT/env/cuda-pins.txt
  $JS2_ENV/bin/pip install -q jax==0.4.34 jaxlib==0.4.34 numpy==1.26.4 scipy==1.12.0 h5py==3.12.1 dynesty==2.1.4 -r $JS2_ROOT/env/cuda-pins.txt
fi
# core is pure python and is imported from the shipped source (never pip-installed), so
# darksirens.__file__ carries the darksirens-core-bf58aa6 path the driver checks
PYTHONPATH=$JS2_CORE/src JAX_PLATFORMS=cuda $JS2_ENV/bin/python -c "import darksirens,jax,numpy,dynesty; print('darksirens',darksirens.__file__); print('jax',jax.__version__,jax.devices(),'numpy',numpy.__version__,'dynesty',dynesty.__version__)"
REMOTE
