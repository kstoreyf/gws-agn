#!/bin/bash
# Local CPU fallback for the per-seed scan stage (guard + f-scan + joint).
#
# Same commands as one_realisation_fix.sh's gpu stage with --device cpu and
# JAX_PLATFORMS=cpu; per-file skips make it resumable and complementary to any
# results that already landed via slurm.  Run from the experiment root.
# Usage: run_scans_cpu_local.sh <seed>
set -euo pipefail
SEED=$1
cd "$(dirname "$0")/.."
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16

DDF=data_derived/s${SEED}_fix
SUR="$DDF/survey_gal_ns32.h5 $DDF/survey_agn_ns32.h5"
GW="$DDF/twotracer_gw_events.h5"
SEL="$DDF/injections_targeted_k2.h5"
COMMON="--universe_model dark_sirens --catalog_sky_weighting field \
  --survey_path $SUR --gw_path $GW --gwselection_path $SEL \
  --log10n0 -12 --log10n0_c2 -12 \
  --selection_neff_guard hard --max_likelihood_variance 1e6 \
  --outdir results --h0_true 67.74 --f_true 0.3 --device cpu"

if [ ! -f "results/guard_fix_s${SEED}.json" ]; then
  python scripts/diag_variance_guard.py --universe_model dark_sirens \
    --catalog_sky_weighting field --survey_path $SUR --gw_path "$GW" \
    --gwselection_path "$SEL" --log10n0 -12 --log10n0_c2 -12 --f_at 0.3 \
    --max_likelihood_variance 1e6 --out_json "results/guard_fix_s${SEED}.json" \
    > "logs/s${SEED}_fix_guard.log" 2>&1
fi
if [ ! -f "results/fscan_fix_s${SEED}.json" ]; then
  python scripts/scan_h0f.py $COMMON --scan f --f_grid 0.0 1.0 41 \
    --out_tag "fscan_fix_s${SEED}" > "logs/s${SEED}_fix_fscan.log" 2>&1
fi
if [ ! -f "results/joint_fix_s${SEED}.json" ]; then
  python scripts/scan_h0f.py $COMMON --scan joint --h0_grid 58.0 78.0 81 \
    --f_grid 0.0 1.0 41 --out_tag "joint_fix_s${SEED}" > "logs/s${SEED}_fix_joint.log" 2>&1
fi
echo "SCANS DONE seed=$SEED (cpu-local)"
