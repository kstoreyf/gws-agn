#!/usr/bin/env bash
# Emit one line per campaign event: a finished chunk, a finished 1-D scan, a
# failed chunk, a worker that died, plus a heartbeat every 30 minutes so silence
# can never be mistaken for progress.  Exits when all 48 chunks are on disk.
cd "$(dirname "$0")/.."
listing() {
  { ls results/chunks/*.h5 2>/dev/null
    ls results/fscan_s*.json results/h0scan_s*.json results/joint_s*.json 2>/dev/null
    ls queue/claim_*/failed 2>/dev/null; } | sort
}
prev=$(listing)
echo "WATCH start: $(echo "$prev" | grep -c . ) products already on disk"
hb=0
while true; do
  sleep 60
  cur=$(listing)
  comm -13 <(echo "$prev") <(echo "$cur") | sed 's/^/NEW /'
  prev=$cur
  hb=$((hb + 1))
  nch=$(ls results/chunks/*.h5 2>/dev/null | wc -l)
  nrun=$(squeue -u "$USER" -h -o "%j %t" 2>/dev/null | grep -c "a2_.* R")
  npd=$(squeue -u "$USER" -h -o "%j %t" 2>/dev/null | grep -c "a2_.* PD")
  if [ "$nch" -ge 48 ]; then
    echo "DONE all 48 chunks present"
    exit 0
  fi
  if [ "$nrun" -eq 0 ] && [ "$npd" -eq 0 ]; then
    echo "ALERT no a2 jobs running or pending; $nch/48 chunks on disk"
    exit 1
  fi
  if [ $((hb % 30)) -eq 0 ]; then
    echo "HEARTBEAT $(date -u +%H:%MZ) chunks=$nch/48 running=$nrun pending=$npd"
  fi
done
