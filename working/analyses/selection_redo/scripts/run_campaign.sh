#!/usr/bin/env bash
# Run one analysis' queue, sequentially, on the Jetstream2 H100.
#
#   ./run_campaign.sh a3 [a6 a4 a5 ...]
#
# RESUMABLE BY DESIGN: a task whose result JSON already exists is skipped. The
# campaign is ~83 GPU-h across 41 tasks on a single GPU, so it will outlive any
# ssh session -- launch it detached and let this script's skip logic make
# re-invocation free:
#
#   setsid nohup ./run_campaign.sh a3 a6 a4 a5 > logs/campaign.out 2>&1 < /dev/null &
#
# Each task's stdout goes to <analysis>/logs/<out_tag>.log; this script's own
# stdout is a one-line-per-task ledger.
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
export SCRIPTS="$HERE"
ROOT="$(dirname "$HERE")"

# --- machine configuration ---------------------------------------------------
# Everything for this project on J2 lives under one root, per the owner.
export CAMPAIGN_ROOT="${CAMPAIGN_ROOT:-/media/volume/darksirens-data/gws-agn-js2-data}"
export DATA_ROOT="${DATA_ROOT:-${CAMPAIGN_ROOT}/data}"
export FITS_DIR="${FITS_DIR:-${SCRIPTS}/selection_fits_truez}"
# The validated clone (agent-checked 2026-08-10: SHA 0c5b3db, Tier-0 304 passed
# / 1 skipped, and the m18 selection probe reproduced to 2-4e-11). NOT a bare
# "src/darksirens" -- that path is deliberately absent so a later clone into it
# cannot fail.
export DARKSIRENS_SRC="${DARKSIRENS_SRC:-${CAMPAIGN_ROOT}/src/darksirens-0c5b3db}"
# PYTHONPATH must win: the micromamba env carries an EDITABLE install pointing
# at ~/darksirens @ ae9cd41, so anything that loses this silently runs stale
# code. Every task prints its resolved darksirens module file and records
# darksirens_git_sha in its output -- checked on task 1 below, not assumed.
export PYTHONPATH="${DARKSIRENS_SRC}${PYTHONPATH:+:${PYTHONPATH}}"
# There is no bare `python` on J2's default PATH (only /usr/bin/python3, no jax).
export PY="${PY:-/home/exouser/mm/root/envs/darksirens-ignacio/bin/python}"

# The H100 is shared with nothing else, but preallocation still has to be off:
# darksirens' guard spy places jax.debug.callback inputs on a CPU device.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

command -v "$PY" >/dev/null || { echo "[fatal] python not found: $PY" >&2; exit 1; }
[ -d "$DARKSIRENS_SRC" ] || { echo "[fatal] no darksirens at $DARKSIRENS_SRC" >&2; exit 1; }
[ -d "$FITS_DIR" ] || { echo "[fatal] no selection fits at $FITS_DIR" >&2; exit 1; }

DS_SHA="$(git -C "$DARKSIRENS_SRC" rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "=== selection-mode campaign ==="
echo "  darksirens : $DARKSIRENS_SRC @ $DS_SHA"
echo "  data       : $DATA_ROOT"
echo "  fits       : $FITS_DIR"
echo "  python     : $($PY -c 'import sys;print(sys.executable)')"
# Fail fast if PYTHONPATH lost the argument: the editable install in the env
# points at a stale ae9cd41 checkout, and running 83 GPU-h of that would be
# silent and total.
RESOLVED="$($PY -c 'import darksirens;print(darksirens.__file__)' 2>&1)"
echo "  resolves to: $RESOLVED"
case "$RESOLVED" in
  "$DARKSIRENS_SRC"/*) ;;
  *) echo "[fatal] darksirens resolves OUTSIDE $DARKSIRENS_SRC -- refusing to run." >&2
     exit 1 ;;
esac
echo

started_all=$(date +%s)
for A in "$@"; do
  Q="$ROOT/$A/queue/tasks.txt"
  [ -f "$Q" ] || { echo "[skip] no queue for $A"; continue; }
  export RESULTS="$ROOT/$A/results"
  mkdir -p "$RESULTS" "$ROOT/$A/logs"

  n=0; done_n=0; skip_n=0; fail_n=0
  total=$(grep -cve '^\s*$' "$Q")
  echo "--- $A : $total tasks ---"

  while IFS= read -r task; do
    [ -z "${task// }" ] && continue
    n=$((n+1))
    tag=$(sed -n 's/.*--out_tag \([A-Za-z0-9_]*\).*/\1/p' <<<"$task")
    [ -z "$tag" ] && { echo "  [$n/$total] SKIP (no --out_tag parsed)"; continue; }

    if [ -f "$RESULTS/$tag.json" ]; then
      skip_n=$((skip_n+1))
      echo "  [$n/$total] $tag  already done, skipping"
      continue
    fi

    log="$ROOT/$A/logs/$tag.log"
    t0=$(date +%s)
    echo "  [$n/$total] $tag  started $(date -u +%H:%M:%SZ)"
    # --outdir is appended here rather than baked into the queue so one queue
    # file works from any checkout.
    if eval "$task --outdir $RESULTS" > "$log" 2>&1; then
      dt=$(( $(date +%s) - t0 ))
      done_n=$((done_n+1))
      printf '  [%d/%d] %s  OK  %dh%02dm\n' "$n" "$total" "$tag" $((dt/3600)) $(((dt%3600)/60))
    else
      dt=$(( $(date +%s) - t0 ))
      fail_n=$((fail_n+1))
      printf '  [%d/%d] %s  FAILED after %dh%02dm -- see %s\n' \
             "$n" "$total" "$tag" $((dt/3600)) $(((dt%3600)/60)) "$log"
      tail -n 15 "$log" | sed 's/^/        | /'
      # keep going: one bad cell must not cost the rest of a multi-day queue
    fi
  done < "$Q"

  echo "--- $A done: $done_n ok, $skip_n skipped, $fail_n failed ---"
  echo
done
printf '=== campaign finished in %dh%02dm ===\n' \
       $(( ($(date +%s)-started_all)/3600 )) $(( (($(date +%s)-started_all)%3600)/60 ))
