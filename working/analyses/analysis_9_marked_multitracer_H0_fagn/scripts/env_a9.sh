#!/usr/bin/env bash
# The load-bearing Analysis-9 environment.  Source it; do not inline it per job.
#
#   PYTHONPATH -> darksirens-a8 is MANDATORY.  Only that checkout carries the
#   tracer-dependent population blocks (9 references to mixture_pop_params in
#   darksirens/likelihood/core.py; ZERO in src/darksirens and
#   src/darksirens-2b86a2d).  DARKSIRENS_SRC does NOT steer the import -- the
#   editable install does -- so without PYTHONPATH the run silently becomes
#   Analysis 2 with three coordinates instead of Analysis 9.
export A9_DIR=/hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_9_marked_multitracer_H0_fagn
export DARKSIRENS_A8=/hildafs/projects/phy230014p/magana/src/darksirens-a8
export DARKSIRENS_A8_SHA=af896cae6f3f3dd1f87dec50046e3a8228f59b39
export PYTHONPATH="${DARKSIRENS_A8}${PYTHONPATH:+:${PYTHONPATH}}"
export DARKSIRENS_SRC="${DARKSIRENS_A8}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONDONTWRITEBYTECODE=1          # never write .pyc into the a8 tree
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export A9_PY=/hildafs/home/magana/tmp_ondemand_hildafs_phy230014p_symlink/magana/.conda/envs/jax/bin/python

_a9_head=$(git -C "${DARKSIRENS_A8}" rev-parse HEAD)
if [ "${_a9_head}" != "${DARKSIRENS_A8_SHA}" ]; then
  echo "[fatal] darksirens-a8 HEAD ${_a9_head} != ${DARKSIRENS_A8_SHA}" >&2
  return 1 2>/dev/null || exit 1
fi
if [ -n "$(git -C "${DARKSIRENS_A8}" status --porcelain)" ]; then
  echo "[fatal] darksirens-a8 worktree is dirty; the SHA no longer describes the code" >&2
  return 1 2>/dev/null || exit 1
fi
echo "[env] darksirens-a8 ${_a9_head} (clean)"
echo "[env] PYTHONPATH=${PYTHONPATH}"
echo "[env] python ${A9_PY}"
unset _a9_head
