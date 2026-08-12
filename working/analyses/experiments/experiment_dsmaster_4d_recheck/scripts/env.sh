#!/usr/bin/env bash
# Shared configuration for the darksirens-master 4D re-check.
#
# The ONE thing this experiment changes relative to analysis 5 is which
# darksirens the closure is built from.  The conda env installs darksirens as a
# setuptools egg-link pointing at src/darksirens (currently parked on the
# feat/likelihood-2d-scan branch), so the switch is PYTHONPATH, which resolves
# ahead of the egg-link -- verified, not assumed:
#
#   PYTHONPATH=$DS_MASTER python -c "import darksirens; print(darksirens.__file__)"
#   -> /hildafs/projects/phy230014p/magana/src/darksirens-master/darksirens/__init__.py
#
# DARKSIRENS_SRC additionally points scan_h0f's provenance/ancestor guard at the
# same tree, so the recorded SHA is the code that actually ran.

export DS_MASTER=${DS_MASTER:-/hildafs/projects/phy230014p/magana/src/darksirens-master}
export PYTHONPATH=${DS_MASTER}${PYTHONPATH:+:${PYTHONPATH}}
export DARKSIRENS_SRC=${DS_MASTER}

export DATA_ROOT=${DATA_ROOT:-/hildafs/projects/phy230014p/magana/gws-agn/working/data}
export SEED=${SEED:-100}
export RUNG=${RUNG:-m18}

# Which survey directory ds_4d_common reads.  Default "surveys" reproduces the
# aggregate/per_pixel arms' command string byte for byte.  The selection arm
# sets SRV_SUBDIR=surveys_galprop: identical files plus the padded gal_app_mag
# column that darksirens_fit_selection needs (scripts/add_gal_app_mag.py built
# them and asserted the four original datasets came back bit-identical, so the
# switch cannot move the likelihood).
export SRV_SUBDIR=${SRV_SUBDIR:-surveys}

# darksirens 5aa90fa -- the FIRST SHA carrying K>=2 x c_mode=selection
# (PRs #342-#346).  A separate worktree from DS_MASTER (e8d5035) on purpose:
# the aggregate and per_pixel arms are finished and must stay reproducible at
# the SHA they ran on, so the selection arm gets its own tree rather than
# moving that one forward.
export DS_SEL=${DS_SEL:-/hildafs/projects/phy230014p/magana/src/darksirens-sel}

# analysis 5's stored result for this rung -- the comparison baseline, built on
# 2b86a2d.  Its rstate-23 twin (campaign_m18_dynesty_r2_s100) bounds dynesty's
# own scatter, which is the noise floor any "master changed something" claim
# has to clear.
export A5=/hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_5_free_anchors_H0_fagn
export A3=/hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_3_incomplete_catalog_H0_fagn/results

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
PY=/hildafs/home/magana/tmp_ondemand_hildafs_phy230014p_symlink/magana/.conda/envs/jax/bin
export PATH="$PY:$PATH"

# The analysis-5 invocation, verbatim apart from --survey_path's rung.  Any
# drift here invalidates the comparison, so it lives in exactly one place.
# $1 = out_tag, plus any extra flags after it.
ds_4d_common() {
  echo "--universe_model dark_sirens --catalog_sky_weighting field \
--survey_path ${DATA_ROOT}/seed${SEED}/${SRV_SUBDIR}/survey_gal_${RUNG}_ns32.h5 \
              ${DATA_ROOT}/seed${SEED}/${SRV_SUBDIR}/survey_agn_${RUNG}_ns32.h5 \
--gw_path ${DATA_ROOT}/seed${SEED}/events/events.h5 \
--gwselection_path ${DATA_ROOT}/seed${SEED}/injections/injections_targeted.h5 \
--selection_neff_guard hard --max_likelihood_variance 1e6 \
--kde_window 4096 --kde_window_nsigma 8 \
--sel_batch_size 50000 --pe_event_block 25 \
--h0_true 67.74 --f_true 0.295 --n0_true -3.0 --n0c2_true -5.0 \
--n0_prior -4.0 -1.0 --n0c2_prior -6.0 -4.0 \
--wiring_ref ${A3}/joint_${RUNG}_s100.h5 \
--device gpu"
}
