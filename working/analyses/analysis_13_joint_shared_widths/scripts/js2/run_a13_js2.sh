#!/bin/bash
# Run ON THE VM (pushed to $JS2_A13/scripts/). Detached runner: survives ssh disconnects.
#   setsid nohup bash run_a13_js2.sh 1 > /dev/null 2>&1 &
# Re-running it with the same seed resumes from queue/a13core_dynesty_n200_s<seed>.save.
set -euo pipefail
SEED=${1:?seed}
ROOT=/media/volume/tbs/gws-agn-data-js2
A13=$ROOT/analysis_13_joint_shared_widths
cd $A13/scripts
if pgrep -f "bin/python a13_core_sampler.py --seed $SEED" > /dev/null; then echo "already running"; exit 1; fi
TAG=a13core_js2_s${SEED}_$(date +%Y%m%d_%H%M%S)
export A13_DATA=$ROOT/data/seed100 A13_REF_CELLS=$A13/ref/core_bf58aa6_a11_cells.json
# the vGPU has 20 GB and JAX caps itself at 75% of it (~15 GB) by default; rita peaked at 17.3 GB.
# Raise the cap; the padded layout still ran out at 18.7 GB, so also use the layout-only opt-ins
# (galaxy-list kernel normaliser, gathered missing density). The driver's pre-flight must still
# reproduce the rita reference cells to 1e-8 or the run stops before sampling.
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
# BFC caching allocator still ran out at 17-19 GB; the platform allocator frees each buffer
# when it dies (no cache, no fragmentation), at some cost in allocation speed
export XLA_PYTHON_CLIENT_ALLOCATOR=${XLA_PYTHON_CLIENT_ALLOCATOR:-platform}
export A13_KERNEL_LAYOUT=${A13_KERNEL_LAYOUT:-galaxy_list} A13_MISSING_DENSITY=${A13_MISSING_DENSITY:-gather}
export PYTHONPATH=$ROOT/src/darksirens-core-bf58aa6/src XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 JAX_PLATFORMS=cuda OMP_NUM_THREADS=4
echo "[a13core] host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) seed=$SEED start=$(date -Is)" > $A13/logs/$TAG.out
# host RSS + GPU memory every 60 s, so the 58 GB VM / 20 GB vGPU limits are visible
( while sleep 60; do echo "$(date -Is) rss_kb=$(ps -o rss= -C python 2>/dev/null | sort -n | tail -1) $(nvidia-smi --query-gpu=memory.used --format=csv,noheader) $(free -g | awk '/Mem/{print "free_g="$7}')"; done ) > $A13/logs/$TAG.mem &
MON=$!
set +e
$ROOT/env/core-bf58aa6/bin/python a13_core_sampler.py --seed $SEED >> $A13/logs/$TAG.out 2> $A13/logs/$TAG.err
RC=$?
kill $MON 2>/dev/null
echo "[a13core] exit=$RC end=$(date -Is)" >> $A13/logs/$TAG.out
