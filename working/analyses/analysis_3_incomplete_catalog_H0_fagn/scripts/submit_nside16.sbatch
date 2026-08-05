#!/bin/bash
#SBATCH --job-name=a3_ns16
#SBATCH --account=phy220048p
#SBATCH --partition=RITA-GPU
#SBATCH --qos=rita
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=110G
#SBATCH --time=02:00:00
#
# 110G, not more: the RITA-GPU node has 250G total and a long-running
# job holds 125G of it, so a larger request is unschedulable there
# regardless of whether a GPU is free.  The degrade peaks around 25G.
#SBATCH --output=logs/slurm_a3_ns16_%j.out
#SBATCH --error=logs/slurm_a3_ns16_%j.err
#
# The confirmatory test for the continuity finding.
#
# CLAIM: the complete-catalog f_AGN shift under the true-n0 completion is driven by
# PER-PIXEL AGN Poisson noise -- an nside-32 pixel holds a mean of 7.0 AGN inside
# the horizon, so C_AGN(z) < 1 wherever a pixel fluctuates low, one-sidedly, and the
# model manufactures a missing-AGN budget on a complete catalog.
#
# TEST: regroup the SAME hosts into nside-16 pixels (4x the area, so 4x the hosts
# per pixel, so half the per-pixel Poisson error) and re-measure the shift between
# the two configurations at that pixelisation.  If the mechanism is what is claimed,
# the shift should fall by roughly 2x.
#
# The control matters: the nside-16 run at log10n0 = -24 is done too, so the
# comparison is shift-vs-shift at fixed pixelisation and the loss of sky resolution
# cancels out of it.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export DARKSIRENS_SRC=/hildafs/projects/phy230014p/magana/src/darksirens
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
PY=/hildafs/home/magana/tmp_ondemand_hildafs_phy230014p_symlink/magana/.conda/envs/jax/bin
export PATH="$PY:$PATH"
. scripts/env.sh
SEED=${SEED:-100}
EV=${DATA_ROOT}/seed${SEED}/events/events.h5
SUR="data_derived/survey_gal_complete_ns16.h5 data_derived/survey_agn_complete_ns16.h5"

echo "[$(date -u)] === build the nside-16 blocks (exact regrouping) ==="
if [ ! -f data_derived/survey_agn_complete_ns16.h5 ]; then
  python -u scripts/degrade_survey_nside.py --seed $SEED --nside_out 16 \
    --tracers gal agn --level complete
fi

echo "[$(date -u)] === the KDE window the coarser blocks need ==="
python -u scripts/kde_window_check.py --survey_path $SUR \
  --out_json results/kde_window_ns16.json 2>&1 | tail -20
W16=$(python -c "import json;print(json.load(open('results/kde_window_ns16.json'))['window_recommended_power_of_two'])")
echo "[$(date -u)] using kde_window W=$W16 for BOTH nside-16 runs"
echo "  (nside-32 runs used W=${KDE_W}; the window is a numerical requirement set by"
echo "   the densest block, so it MUST change with the pixelisation -- what has to be"
echo "   held fixed is that both nside-16 arms use the same W, which they do)"

COMMON="--universe_model dark_sirens --catalog_sky_weighting field \
--survey_path $SUR --gw_path $EV \
--gwselection_path ${DATA_ROOT}/seed${SEED}/injections/injections_targeted.h5 \
--selection_neff_guard hard --max_likelihood_variance 1e6 \
--kde_window $W16 --kde_window_nsigma 8 \
--sel_batch_size ${SEL_BATCH} --pe_event_block ${PE_BLOCK} \
--h0_true ${H0_TRUE} --f_true ${F_TRUE} --device gpu \
--scan f --f_grid 0.0 1.0 101 --h0_fixed ${H0_TRUE} --outdir results"

echo "[$(date -u)] === arm A: analysis-3 configuration (true n0) at nside 16 ==="
python -u scripts/scan_h0f.py $COMMON --log10n0 -3.0 --log10n0_c2 -5.0 \
  --out_tag fscan_complete_ns16_truen0_s${SEED} \
  > logs/fscan_complete_ns16_truen0_s${SEED}.log 2>&1
grep -E "Eval done|kde\]" logs/fscan_complete_ns16_truen0_s${SEED}.log

echo "[$(date -u)] === arm B: analysis-2 configuration (log10n0 = -24) at nside 16 ==="
python -u scripts/scan_h0f.py $COMMON --log10n0 -24 --log10n0_c2 -24 \
  --out_tag fscan_complete_ns16_n24_s${SEED} \
  > logs/fscan_complete_ns16_n24_s${SEED}.log 2>&1
grep -E "Eval done|kde\]" logs/fscan_complete_ns16_n24_s${SEED}.log

echo "[$(date -u)] === verdict ==="
python -u scripts/nside_scaling_verdict.py | tee logs/nside_scaling_verdict.log
