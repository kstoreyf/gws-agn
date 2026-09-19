#!/usr/bin/env bash
# One-screen state of the 100-seed campaign.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=/hildafs/projects/phy230014p/magana/gws-agn/working
A0=$ROOT/analyses/analysis_0_pure_tracer_H0/results
A1=$ROOT/analyses/analysis_1_complete_catalog_H0/results
A2=$ROOT/analyses/analysis_2_complete_catalog_H0_fagn/results/chunks
M=$HERE/manifest/seeds.tsv

echo "===== 100-seed campaign  $(date -u +%FT%TZ) ====="
echo
echo "-- generation (target: seeds 106-200, 95 new) --"
if [ -f "$M" ]; then
  tot=$(wc -l < "$M")
  printf "  recorded %3d / 95   " "$tot"
  for st in PASS VALIDATION_FAILED GEN_FAILED INCOMPLETE; do
    n=$(awk -F'\t' -v s="$st" '$2==s' "$M" | wc -l); printf "%s=%d  " "$st" "$n"; done; echo
  awk -F'\t' '$2=="VALIDATION_FAILED"{print "    seed "$1" failed: "$3}' "$M" | head -8
else echo "  (no manifest yet)"; fi
echo "  queued/running gen tasks: $(squeue -u magana -h -n gen100 2>/dev/null | wc -l)"
echo

echo "-- a0 + a1 on rita (2x A100-80) --"
n0=$(ls "$A0"/h0_puregal_targeted_s1*.h5 2>/dev/null | wc -l)
n1=$(ls "$A1"/ctrl_gal_matched_s1*.h5 2>/dev/null | wc -l)
echo "  a0 seeds with the GAL targeted grid : $n0   (5 pre-existing + new)"
echo "  a1 seeds with the GAL control       : $n1"
echo "  workers running: $(squeue -u magana -h -n a01_100 2>/dev/null | wc -l) / 2"
echo

echo "-- a2 on miko (1x H100 NVL), 45 new grids to reach 50 --"
if [ -f "$HERE/queue100/tasks.txt" ]; then
  tot=$(wc -l < "$HERE/queue100/tasks.txt")
  cl=$(ls -d "$HERE"/queue100/claim_* 2>/dev/null | wc -l)
  dn=$(ls "$HERE"/queue100/claim_*/done 2>/dev/null | wc -l)
  fl=$(ls "$HERE"/queue100/claim_*/failed 2>/dev/null | wc -l)
  printf "  chunks %3d/%3d done, %d claimed, %d failed\n" "$dn" "$tot" "$cl" "$fl"
  printf "  complete grids: %d / 45\n" "$(( dn / 8 ))"
fi
echo "  worker alive: $(pgrep -c -f a2_worker.sh 2>/dev/null || echo 0)"
echo

echo "-- disk (the binding limit is the phy220048p PROJECT quota, not the mount) --"
timeout 60 lfs quota -hp 553067 /hildafs 2>/dev/null | awk '/hildafs/{print "  phy220048p (data)     "$2" of "$3}'
timeout 60 lfs quota -hp 553654 /hildafs 2>/dev/null | awk '/hildafs/{print "  phy230014p (results)  "$2" of "$3}'
du -sh /hildafs/projects/phy220048p/magana/gws-agn-data-v3 2>/dev/null | awk '{print "  dataset: "$1}'
