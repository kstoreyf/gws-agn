#!/usr/bin/env bash
# Shared configuration for analysis_4 (what mis-anchoring the AGN comoving
# density costs the joint (H0, f_AGN) measurement down the completeness ladder).
#
# EXACTLY analysis_3's K = 2 configuration with ONE scalar swept: the AGN
# density the completion is anchored to.
#
#   analysis_3:  log10n0_c2 = -5.0                  (the mock's TRUE AGN density)
#   analysis_4:  log10n0_c2 = -5.0 + log10(factor), factor in
#                {0.5, 0.7, 0.9, 1.1, 1.3, 2.0}
#
# The GAL density stays at truth (log10n0 = -3) in every arm: the prototype
# (experiments/experiment_completeness_free) showed the AGN anchor is the axis
# that matters, and sweeping one scalar keeps the exact arm — analysis_3's own
# seed-100 grids — a shared reference rather than a rerun.
#
# One extra grid, the ORACLE probe, changes the SURVEYS instead of the density:
# GAL at m < 18, AGN complete, both densities at truth.  If the +0.084 f_AGN
# bias at m18 is manufactured by the sparse AGN completion (5 hosts/pixel,
# 52.8 % empty pixels), handing the model every AGN host while the galaxies
# stay 10 %-complete should remove most of it.
#
# Everything else is byte-identical to analysis_3/scripts/env.sh.
export DATA_ROOT=${DATA_ROOT:-/hildafs/projects/phy230014p/magana/gws-agn/working/data}
export DARKSIRENS_SRC=${DARKSIRENS_SRC:-/hildafs/projects/phy230014p/magana/src/darksirens}
export SEL_BATCH=${SEL_BATCH:-50000}
export PE_BLOCK=${PE_BLOCK:-25}
export KDE_W=${KDE_W:-4096}
export LOG10N0=${LOG10N0:--3.0}         # GAL true comoving density, log10 Mpc^-3
export LOG10N0_C2_TRUE=-5.0             # AGN true comoving density, log10 Mpc^-3
export H0_TRUE=${H0_TRUE:-67.74}
export F_TRUE=${F_TRUE:-0.30}

# the K=2 configuration every scan in this directory shares
# $1 = seed, $2 = injection lane (targeted|popuni), $3 = events file,
# $4 = GAL survey level, $5 = AGN survey level, $6 = log10n0_c2 for this arm
ds_common() {
  local seed=$1 lane=$2 ev=$3 glev=$4 alev=$5 n0c2=$6
  echo "--universe_model dark_sirens \
--catalog_sky_weighting field \
--survey_path ${DATA_ROOT}/seed${seed}/surveys/survey_gal_${glev}_ns32.h5 \
              ${DATA_ROOT}/seed${seed}/surveys/survey_agn_${alev}_ns32.h5 \
--gw_path ${ev} \
--gwselection_path ${DATA_ROOT}/seed${seed}/injections/injections_${lane}.h5 \
--log10n0 ${LOG10N0} --log10n0_c2 ${n0c2} \
--selection_neff_guard hard --max_likelihood_variance 1e6 \
--kde_window ${KDE_W} --kde_window_nsigma 8 \
--sel_batch_size ${SEL_BATCH} --pe_event_block ${PE_BLOCK} \
--h0_true ${H0_TRUE} --f_true ${F_TRUE} --device gpu"
}
