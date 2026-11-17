#!/bin/bash
#SBATCH --job-name=agn-prodtest
#SBATCH --account=phy220048p
#SBATCH --partition=RITA-GPU
#SBATCH --qos=rita
#SBATCH --nice=50
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100-80:1
#SBATCH --mem=120G
#SBATCH --time=00:45:00
#SBATCH --output=/hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_16_production_runs/logs/%x_closure_%j.out
#SBATCH --error=/hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_16_production_runs/logs/%x_closure_%j.err
# Test (a): closure of the production driver with Analyses 14 and 15 (seed 100, legacy events.h5,
# H0 64-74 step 0.25 x f_AGN 0.05-0.55 step 0.025 = Analysis 14's (b) and Analysis 15's cells).
#   a14_convention        A14 settings exactly: redraw [141002026, 2, 0] at 8000 samples, count-ratio
#                         completeness at log10 n0 = -24, grid to z = 5, variance cap 1e6
#                         -> must equal results/scans/b_pe_n8000_r00 of Analysis 14
#   production_a14_redraw the production convention (complete catalogs Z_k = N_obs, grid to 1.10,
#                         cap 20) on the same redraw -> compared with the same A14 scan
#   a15_convention        A15 b_complete_x1 settings: stored samples, selection completeness, log10 n0
#                         -3/-5 physical, grid to 1.10, cap 1e6 -> must equal A15's scan
#   production_stored     the production convention on the stored samples -> A15 b_complete_x1, and
#                         a15_convention (the complete model against the selection model at C = 1)
# Then tests/compare_closure.py (CPU) -> tests/results/closure.json.
set -uo pipefail
A16=/hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_16_production_runs
cd $A16
export PYTHONPATH= XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 JAX_PLATFORMS=cuda OMP_NUM_THREADS=4
export A16_DEADLINE=$(( $(date +%s) + 45*60 - 180 ))
PY=/hildafs/projects/phy230014p/magana/envs/darksirens-core-e7c3007/bin/python
JPY=/hildafs/home/magana/tmp_ondemand_hildafs_phy230014p_symlink/magana/.conda/envs/jax/bin/python
EV=/hildafs/projects/phy230014p/magana/gws-agn/working/data/seed100/events/events.h5
COMMON="--mode grid --analysis known_population --catalogs both --events-kind equal_populations --realization 100
        --events-file $EV --grid H0=64,74,0.25 --grid f_agn=0.05,0.55,0.025 --coarse none --out-root tests/closure_runs"
echo "[agn-prodtest closure] host=$(hostname) job=${SLURM_JOB_ID:-none} start=$(date -u +%FT%TZ)"
$PY scripts/prod_driver.py $COMMON --run-name a14_convention --redraw-key 141002026,2,0 --nsamp 8000 \
    --complete-model count_ratio --density given --log10n-gal -24 --log10n-agn -24 --density-units physical \
    --zmax 5.0 --max-var 1e6
echo "[closure] a14_convention exit $?"
$PY scripts/prod_driver.py $COMMON --run-name production_a14_redraw --redraw-key 141002026,2,0 --nsamp 8000
echo "[closure] production_a14_redraw exit $?"
$PY scripts/prod_driver.py $COMMON --run-name a15_convention --nsamp 0 --complete-model selection --density given \
    --log10n-gal -3 --log10n-agn -5 --density-units physical --max-var 1e6
echo "[closure] a15_convention exit $?"
$PY scripts/prod_driver.py $COMMON --run-name production_stored --nsamp 0
echo "[closure] production_stored exit $?"
JAX_PLATFORMS=cpu $JPY tests/compare_closure.py
echo "[agn-prodtest closure] end=$(date -u +%FT%TZ)"
