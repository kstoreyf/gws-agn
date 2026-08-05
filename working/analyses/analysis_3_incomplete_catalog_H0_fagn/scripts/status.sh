#!/usr/bin/env bash
# Campaign progress at a glance: which of the 25 ladder tasks are done, which are
# claimed and running, which are still unclaimed, and what SLURM is doing.
set -uo pipefail
cd "$(dirname "$0")/.."

echo "=== gates ==="
if [ -f results/gates.json ]; then
  python - <<'PY'
import json
from pathlib import Path
g = json.loads(Path("results/gates.json").read_text())
v = g.get("verdict", {})
print(f"  verdict: {'PASS' if v.get('pass') else 'FAIL'}")
for f in v.get("failures", []):
    print(f"    - {f}")
for lev in ("m21", "m20", "m19", "m18"):
    t = (g.get("timing") or {}).get(lev)
    if t:
        print(f"  {lev:5s} {t['steady_state_s_per_eval']:.3f} s/eval  "
              f"{t['gpu_hours_per_grid']:.2f} GPU-h/grid")
c = g.get("continuity")
if c:
    for n, r in c["scans"].items():
        print(f"  continuity {n:8s} shift {r['shift_median']:+.4g} "
              f"({r['shift_median_in_a2_half_widths']:+.3f} a2-hw)  "
              f"width x{r['half_width_ratio']:.4f}")
PY
else
  echo "  (not yet written)"
fi

echo
echo "=== ladder queue ==="
N=$(wc -l < queue/tasks.txt)
done_n=0; run_n=0; free_n=0; fail_n=0
for i in $(seq 1 "$N"); do
  read -r KIND SEED LEVEL LANE CHUNK NCHUNK < <(sed -n "${i}p" queue/tasks.txt)
  SUF=""; [ "$LANE" = "popuni" ] && SUF="_popuni"
  case "$KIND" in
    joint) TAG="joint_${LEVEL}_s${SEED}${SUF}" ;;
    fnull) TAG="fscan_null_${LEVEL}_s${SEED}${SUF}" ;;
    *)     TAG="${KIND}_${LEVEL}_s${SEED}${SUF}" ;;
  esac
  OUTF="results/${TAG}.h5"
  if [ "$NCHUNK" != "-" ]; then TAG="${TAG}_c${CHUNK}"; OUTF="results/chunks/${TAG}.h5"; fi
  if [ -f "$OUTF" ]; then done_n=$((done_n+1)); st="done"
  elif [ -f "queue/claim_$i/failed" ]; then fail_n=$((fail_n+1)); st="FAILED"
  elif [ -d "queue/claim_$i" ]; then run_n=$((run_n+1)); st="running"
  else free_n=$((free_n+1)); st="-"
  fi
  [ "$st" != "done" ] && printf "  %-34s %s\n" "$TAG" "$st"
done
echo "  ---- $done_n done, $run_n running, $free_n unclaimed, $fail_n failed, of $N"

echo
echo "=== merged complete-rung grids (rung 0 of record) ==="
for sd in 100 101 102 103 105; do
  [ -f "results/joint_complete_s${sd}.h5" ] && echo "  joint_complete_s${sd}  merged" \
    || echo "  joint_complete_s${sd}  not merged"
done
[ -f "results/joint_complete_s100_popuni.h5" ] && echo "  joint_complete_s100_popuni  merged" \
  || echo "  joint_complete_s100_popuni  not merged"

echo
echo "=== slurm ==="
squeue -u "$USER" -o "%.14i %.11P %.10j %.9T %.9M %.10l %.22R"
