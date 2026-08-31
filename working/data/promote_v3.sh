#!/usr/bin/env bash
# Move-aside / validate / promote the v3 dataset.
#
#   ./promote_v3.sh check      report what would happen
#   ./promote_v3.sh promote    re-point working/data/seed<N> at the v3 tree
#   ./promote_v3.sh delete     DELETE the superseded v2 tree (only after promote)
#
# The v2 dataset stays on disk until `delete` is run explicitly, so the promotion
# is reversible up to that point.
set -euo pipefail
HERE=/hildafs/projects/phy230014p/magana/gws-agn/working/data
OLD=/hildafs/projects/phy220048p/magana/gws-agn-data
NEW=/hildafs/projects/phy220048p/magana/gws-agn-data-v3
SEEDS=${SEEDS:-"100 101 102 103 105"}
ACT=${1:-check}

fail=0
for S in $SEEDS; do
  V=$NEW/seed$S/validation/validation.json
  if [ ! -s "$V" ]; then echo "[MISSING] $V"; fail=1; continue; fi
  NF=$(python -c "import json,sys; print(json.load(open('$V'))['n_failed'])")
  NC=$(python -c "import json,sys; print(json.load(open('$V'))['n_checks'])")
  PM=$(python -c "
import h5py
with h5py.File('$NEW/seed$S/events/events.h5','r') as f:
    print(f.attrs.get('pe_model','?'))")
  ZC=$(python -c "
import h5py
with h5py.File('$NEW/seed$S/surveys/survey_gal_complete_ns32.h5','r') as f:
    print(f.attrs.get('z_column','?'))")
  echo "seed $S: $NC checks, n_failed=$NF, pe_model=$PM, survey z_column=$ZC"
  [ "$NF" = "0" ] || fail=1
  [ "$PM" = "v3" ] || fail=1
  [ "$ZC" = "z_obs" ] || fail=1
done
if [ "$fail" != "0" ]; then
  echo "[FATAL] not every seed validates as v3/z_obs with n_failed = 0"
  exit 1
fi
echo "[OK] every seed validates"

case "$ACT" in
  check) echo "(check only; run 'promote' to re-point the symlinks)";;
  promote)
    for S in $SEEDS; do
      ln -sfn "$NEW/seed$S" "$HERE/seed$S"
      echo "  $HERE/seed$S -> $(readlink -f "$HERE/seed$S")"
    done
    echo "[OK] promoted.  The v2 tree is still at $OLD (run 'delete' to remove it)."
    ;;
  delete)
    for S in $SEEDS; do
      T=$(readlink -f "$HERE/seed$S")
      case "$T" in "$NEW"/*) ;; *) echo "[FATAL] seed$S does not point at the v3 tree"; exit 1;; esac
    done
    for S in $SEEDS; do
      if [ -d "$OLD/seed$S" ]; then
        echo "  deleting $OLD/seed$S"
        rm -rf "$OLD/seed$S"
      fi
    done
    echo "[OK] the superseded v2 seed trees are gone."
    ;;
  *) echo "usage: promote_v3.sh {check|promote|delete}"; exit 2;;
esac
