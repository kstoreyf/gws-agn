#!/bin/bash
#SBATCH --job-name=a5smoke_legacy
#SBATCH --account=phy220048p
#SBATCH --partition=RITA-GPU
#SBATCH --qos=rita
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100-80:1
#SBATCH --mem=100G
#SBATCH --time=06:00:00
#SBATCH --output=/hildafs/projects/phy230014p/magana/gws-agn/working/analyses/experiments/experiment_darksirens_core_equivalence/logs/%x_%j.out
#SBATCH --error=/hildafs/projects/phy230014p/magana/gws-agn/working/analyses/experiments/experiment_darksirens_core_equivalence/logs/%x_%j.err
# A5 free-anchor smoke, OLD code (0c5b3db, the commit A5 ran on): the archived A5 m<18 recipe
# (results/campaign_m18_dynesty_s100.h5 attrs) with dlogz 10. Core #48/#49 merged at 661ef3d (2026-10-02).
set -euo pipefail
R=/hildafs/projects/phy230014p/magana/gws-agn/working
D=$R/data/seed100; S=$R/analyses/selection_redo/scripts
OUT=$R/analyses/experiments/experiment_darksirens_core_equivalence/results/a5_smoke
mkdir -p $OUT
export PYTHONPATH=/hildafs/projects/phy230014p/magana/src/darksirens-0c5b3db XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1
/hildafs/home/magana/tmp_ondemand_hildafs_phy230014p_symlink/magana/.conda/envs/jax/bin/python $S/sample_4d.py \
  --survey_path $D/surveys/survey_gal_m18_ns32.h5 $D/surveys/survey_agn_m18_ns32.h5 \
  --gw_path $D/events/events.h5 --gwselection_path $D/injections/injections_targeted.h5 \
  --catalog_sky_weighting field --selection_neff_guard hard --max_likelihood_variance 1e6 \
  --sel_batch_size 50000 --pe_event_block 25 --kde_window 4096 --kde_window_nsigma 8 \
  --c_mode selection --selection_fit $S/selection_fits_truez/fit_gal_m18_schechter.json,$S/selection_fits_truez/fit_agn_m18_schechter.json \
  --h0_prior 50 100 --f_prior 0 1 --n0_prior -4 -1 --n0c2_prior -6 -4 \
  --sampler dynesty --nlive 1000 --dlogz 10 --maxcall 500000 --rstate_seed 7 --skip_wiring \
  --out_tag a5smoke_legacy_0c5b3db_m18 --outdir $OUT
