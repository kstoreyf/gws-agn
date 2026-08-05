#!/usr/bin/env bash
# Shared configuration for analysis_2 (the joint (H0, f_AGN) measurement).
#
# The complete-catalog limit, K = 2 mixture, one estimator:
#   universe_model dark_sirens (K>=2 requires it), catalog_sky_weighting field,
#   survey_path [GAL, AGN] so fcat_2 = f_AGN,
#   log10n0 = log10n0_c2 = -24 (complete limit), delta = delta_c2 = 0,
#   sigma_kde = sigma_kde_c2 = 0, population fixed at the mock fiducial, Om0 pinned.
# Free parameters: H0 and fcat_2 only.
export DATA_ROOT=${DATA_ROOT:-/hildafs/projects/phy230014p/magana/gws-agn/working/data}
export DARKSIRENS_SRC=${DARKSIRENS_SRC:-/hildafs/projects/phy230014p/magana/src/darksirens}
export SEL_BATCH=${SEL_BATCH:-50000}
export PE_BLOCK=${PE_BLOCK:-25}
export KDE_W=${KDE_W:-4096}
export LOG10N0=${LOG10N0:--24}
export H0_TRUE=${H0_TRUE:-67.74}
export F_TRUE=${F_TRUE:-0.30}
export SEEDS=${SEEDS:-"100 101 102 103 105"}

# the K=2 configuration every scan in this directory shares
ds_common() {   # $1 = seed, $2 = injection lane (targeted|popuni), $3 = events file
  local seed=$1 lane=$2 ev=$3
  echo "--universe_model dark_sirens \
--catalog_sky_weighting field \
--survey_path ${DATA_ROOT}/seed${seed}/surveys/survey_gal_complete_ns32.h5 \
              ${DATA_ROOT}/seed${seed}/surveys/survey_agn_complete_ns32.h5 \
--gw_path ${ev} \
--gwselection_path ${DATA_ROOT}/seed${seed}/injections/injections_${lane}.h5 \
--log10n0 ${LOG10N0} --log10n0_c2 ${LOG10N0} \
--selection_neff_guard hard --max_likelihood_variance 1e6 \
--kde_window ${KDE_W} --kde_window_nsigma 8 \
--sel_batch_size ${SEL_BATCH} --pe_event_block ${PE_BLOCK} \
--h0_true ${H0_TRUE} --f_true ${F_TRUE} --device gpu"
}
