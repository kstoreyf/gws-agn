#!/usr/bin/env bash
# The provenance pin for the 100-seed campaign.  Source this; do not run it.
#
# Analyses 0, 1 and 2 of record all ran on darksirens 2b86a2d (SLURM 1059xxx, early
# August 2026).  That SHA is also what generated the seed-100..105 datasets:
# seed101/META.json carries stages.events.darksirens_sha = 2b86a2d....
#
# The main checkout at src/darksirens has since moved to b324bed on the branch
# feat/likelihood-2d-scan, and the selection redo of analyses 3-7 used a third SHA,
# 0c5b3db.  Extending analyses 0-2 on any SHA but 2b86a2d would make the 95 new
# realisations differ from the 5 existing ones by code drift as well as by seed --
# which is exactly the systematic the campaign's own executive summary flags when it
# says archive-vs-new is "estimator plus drift".  The whole point of more seeds is to
# measure seed scatter, so the code has to be held fixed.
#
# ds_pin_guard aborts rather than silently producing incomparable numbers.
export DS_PIN=${DS_PIN:-/hildafs/projects/phy230014p/magana/src/darksirens-2b86a2d}
export DS_PIN_SHA=2b86a2d8d48fdb5173f0ba259b8996104187dd2d
export DARKSIRENS_SRC="$DS_PIN"
export PYTHONPATH="$DS_PIN${PYTHONPATH:+:$PYTHONPATH}"   # the egg-link in site-packages
                                                   # points at the MAIN checkout;
                                                   # only PYTHONPATH overrides it

ds_pin_guard () {
  local got
  got=$(git -C "$DS_PIN" rev-parse HEAD 2>/dev/null)
  if [ "$got" != "$DS_PIN_SHA" ]; then
    echo "FATAL: darksirens pin is wrong." >&2
    echo "  want $DS_PIN_SHA" >&2
    echo "  got  ${got:-<no worktree at $DS_PIN>}" >&2
    return 1
  fi
  if [ -n "$(git -C "$DS_PIN" status --porcelain 2>/dev/null)" ]; then
    echo "FATAL: the pinned worktree $DS_PIN is dirty; refusing to run." >&2
    return 1
  fi
  local main_head
  main_head=$(git -C /hildafs/projects/phy230014p/magana/src/darksirens rev-parse --short HEAD 2>/dev/null)
  echo "[pin] darksirens $DS_PIN_SHA (clean) at $DS_PIN"
  echo "[pin] PYTHONPATH pin active; main checkout is at ${main_head:-?} (not used)"
  return 0
}
