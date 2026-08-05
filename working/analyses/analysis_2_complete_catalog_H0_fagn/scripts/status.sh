#!/usr/bin/env bash
# Compact campaign status: queue progress, running scans, finished products.
cd "$(dirname "$0")/.."
echo "=== $(date -u +%FT%TZ) ==="
squeue -u "$USER" -h -o "%.12i %.10P %.9j %.2t %.9M %R" | grep -E "a2_|^$" || true
NQ=$(wc -l < queue/tasks.txt 2>/dev/null || echo 0)
NC=$(ls -d queue/claim_* 2>/dev/null | wc -l)
ND=$(ls queue/claim_*/done 2>/dev/null | wc -l)
NF=$(ls queue/claim_*/failed 2>/dev/null | wc -l)
echo "queue: $ND done / $NC claimed / $NQ total  (failed: $NF)"
echo "chunks on disk: $(ls results/chunks/*.h5 2>/dev/null | wc -l)"
for s in 100 101 102 103 105; do
  n=$(ls results/chunks/joint_s${s}_c*.h5 2>/dev/null | wc -l)
  p=$(ls results/chunks/joint_s${s}_popuni_c*.h5 2>/dev/null | wc -l)
  printf "  seed %s: targeted %d/8  popuni %d\n" "$s" "$n" "$p"
done
echo "1-D results: $(ls results/fscan_s*.json results/h0scan_s*.json 2>/dev/null | wc -l) / 13"
echo "merged joint: $(ls results/joint_s*.json 2>/dev/null | wc -l)"
for f in logs/joint_*.log; do
  [ -f "$f" ] || continue
  last=$(grep -oE "\[eval\] [0-9]+/[0-9]+" "$f" 2>/dev/null | tail -1)
  [ -n "$last" ] && echo "  $(basename "$f" .log): $last"
done
