#!/usr/bin/env bash
# Self-healing watchdog for the selection campaign. Runs ON J2, from cron.
#
# The failure that motivated it: a task computes its full 8241-cell grid, writes
# the h5, then dies in post-processing (numpy 2 removed np.trapz). Hours of GPU
# time sit on disk as a complete grid with no summary. This finds any such
# orphan -- an h5 with no matching json -- and rebuilds the summary from it,
# which costs seconds and no GPU.
#
# Deliberately conservative: it only ever ADDS a missing json next to a finished
# h5. It never deletes, never re-runs a task, and never touches a grid that
# already has its summary. Safe to run while the campaign is running, because a
# task in flight has not written its h5 yet.
#
#   */20 * * * * /path/to/heal.sh >> /path/to/logs/heal.log 2>&1
set -uo pipefail

ROOT="${CAMPAIGN_ROOT:-/media/volume/darksirens-data/gws-agn-js2-data}"
REDO="$ROOT/analyses/selection_redo"
PY="${PY:-/home/exouser/mm/root/envs/darksirens-ignacio/bin/python}"
export PYTHONPATH="$ROOT/src/darksirens-0c5b3db"
export JAX_PLATFORMS=cpu          # recovery is pure numpy; keep off the GPU

stamp() { date -u +'%Y-%m-%dT%H:%M:%SZ'; }
healed=0

for a in a3 a6 a7 a4 a5; do
  d="$REDO/$a/results"
  [ -d "$d" ] || continue
  for h5 in "$d"/*.h5; do
    [ -e "$h5" ] || continue
    json="${h5%.h5}.json"
    [ -f "$json" ] && continue
    # Do not touch a file still being written: require it stable for 60 s.
    if [ -n "$(find "$h5" -newermt '-60 seconds' 2>/dev/null)" ]; then
      echo "$(stamp) [wait] $(basename "$h5") still settling"
      continue
    fi
    echo "$(stamp) [heal] $a/$(basename "$h5") has no summary -- rebuilding"
    if "$PY" "$REDO/scripts/recover_summary_from_h5.py" "$h5"; then
      healed=$((healed+1))
    else
      echo "$(stamp) [heal] FAILED on $h5"
    fi
  done
done

# One-line health line every run, so the log is a timeline rather than silence.
tot=0; for a in a3 a6 a7 a4 a5; do
  tot=$((tot + $(find "$REDO/$a/results" -maxdepth 1 -name '*.json' 2>/dev/null | wc -l)))
done
# pgrep -c PRINTS 0 and EXITS 1 when nothing matches, so '|| echo 0' appended a
# SECOND zero and $running became "0\n0" -- which never equals "0", so the
# auto-requeue below could never fire and the campaign wedged with the GPU idle
# for 20 minutes before this was caught. wc -l always prints one number, exit 0.
running=$(pgrep -f 'scan_h0f.py|sample_4d.py' 2>/dev/null | wc -l | tr -d ' ')
gpu=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader 2>/dev/null | tr -d ' ')
echo "$(stamp) [status] results=$tot/48  healed_now=$healed  tasks_running=$running  gpu=$gpu"

# --- auto-requeue -----------------------------------------------------------
# A stopped queue with work outstanding is the one state that costs real time:
# the GPU sits idle until a human notices. Since run_campaign.sh skips anything
# that already has a summary, relaunching is free and idempotent, so do it here
# rather than wait to be told.
#
# Two safeguards against a crash-loop: only relaunch when nothing is running,
# and cap the number of automatic relaunches. If the cap is hit, the campaign
# has a real fault that re-running will not fix, and it stays stopped and loud.
LAUNCH_COUNT="$REDO/logs/.auto_requeue_count"
MAX_AUTO=6

if [ "$running" = "0" ] && [ "$tot" -lt 48 ]; then
  n_auto=$(cat "$LAUNCH_COUNT" 2>/dev/null || echo 0)
  if [ "$n_auto" -ge "$MAX_AUTO" ]; then
    echo "$(stamp) [ALERT] stopped at $tot/48 and already auto-requeued $n_auto times -- NOT relaunching; this needs a human"
  else
    echo "$((n_auto+1))" > "$LAUNCH_COUNT"
    echo "$(stamp) [requeue] stopped at $tot/48 -- relaunching (auto #$((n_auto+1))/$MAX_AUTO)"
    cd "$REDO/scripts" && \
      setsid nohup ./run_campaign.sh a7 a3 a6 a4 a5 \
        >> "$REDO/logs/campaign.out" 2>&1 < /dev/null &
  fi
elif [ "$tot" -ge 48 ]; then
  echo "$(stamp) [done] all 48 results present"
fi
