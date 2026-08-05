#!/usr/bin/env bash
# Does coarser reduction blocking buy throughput, and does it move the answer?
#
# analysis_1 sized (sel_batch_size, pe_event_block) = (50000, 25) for an A100-40
# under K = 1.  The concurrency probe measured 14 GB per K = 2 process at that
# setting, i.e. 26 GB of headroom, and both reductions are pure summation blocking
# -- the only effect on the value is floating-point summation order.  This probe
# reports both: the steady-state cost AND the logL difference against the
# 3.71 s/eval reference point, on the same 12 cells.
set -euo pipefail
cd "$(dirname "$0")/.."
. scripts/env.sh
mkdir -p logs results/pilot

SEED=${SEED:-100}
EV=${DATA_ROOT}/seed${SEED}/events/events.h5

for CFG in "100000 50" "200000 100"; do
  read -r SB PB <<< "$CFG"
  SEL_BATCH=$SB PE_BLOCK=$PB
  echo "=== sel_batch_size=$SB pe_event_block=$PB ==="
  python scripts/scan_h0f.py \
    --universe_model dark_sirens --catalog_sky_weighting field \
    --survey_path ${DATA_ROOT}/seed${SEED}/surveys/survey_gal_complete_ns32.h5 \
                  ${DATA_ROOT}/seed${SEED}/surveys/survey_agn_complete_ns32.h5 \
    --gw_path "$EV" \
    --gwselection_path ${DATA_ROOT}/seed${SEED}/injections/injections_targeted.h5 \
    --log10n0 ${LOG10N0} --log10n0_c2 ${LOG10N0} \
    --selection_neff_guard hard --max_likelihood_variance 1e6 \
    --kde_window ${KDE_W} --kde_window_nsigma 8 \
    --sel_batch_size $SB --pe_event_block $PB \
    --h0_true ${H0_TRUE} --f_true ${F_TRUE} --device gpu \
    --scan joint --h0_grid 60 75 4 --f_grid 0.2 0.4 3 \
    --outdir results/pilot --out_tag pilot_blk_${SB}_${PB} \
    > logs/pilot_blk_${SB}_${PB}.log 2>&1 || { echo "  FAILED (see log)"; continue; }
  grep -E "Eval done" logs/pilot_blk_${SB}_${PB}.log
done

python - <<'PY'
import h5py, numpy as np, glob, os
ref = "results/pilot/pilot_joint_s100.h5"
with h5py.File(ref) as f:
    r = np.asarray(f["log_likelihood"][:]); rt = float(f.attrs["steady_state_median_seconds"])
print(f"reference (50000, 25): steady = {rt:.3f} s/eval")
for p in sorted(glob.glob("results/pilot/pilot_blk_*.h5")):
    with h5py.File(p) as f:
        v = np.asarray(f["log_likelihood"][:]); t = float(f.attrs["steady_state_median_seconds"])
    print(f"{os.path.basename(p):32s} steady = {t:6.3f} s/eval  "
          f"speedup = {rt/t:5.2f}x  max|dlogL| = {np.abs(v-r).max():.3e}")
PY
