#!/usr/bin/env bash
# Shared configuration for analysis_3 (the joint (H0, f_AGN) measurement down a
# magnitude-limited completeness ladder).
#
# EXACTLY analysis_2's K = 2 configuration with ONE change: the out-of-catalog
# field term is switched on at the mock's own true comoving densities instead of
# being suppressed to nothing.
#
#   analysis_2:  log10n0 = log10n0_c2 = -24        (the complete-catalog limit)
#   analysis_3:  log10n0 = -3, log10n0_c2 = -5     (the mock's TRUE densities)
#
# Everything else is byte-identical to analysis_2/scripts/env.sh:
#   universe_model dark_sirens (K>=2 requires it), catalog_sky_weighting field,
#   survey_path [GAL, AGN] so fcat_2 = f_AGN, delta = delta_c2 = 0,
#   sigma_kde = sigma_kde_c2 = 0, population fixed at the mock fiducial, Om0
#   pinned, guard `hard` with max_likelihood_variance 1e6, W = 4096 (n_sigma 8),
#   (sel_batch_size, pe_event_block) = (50000, 25).
# Free parameters: H0 and fcat_2 only.
#
# delta = delta_c2 = 0 is NOT a default left unexamined: both tracers are drawn
# from GLASS shells at constant comoving density and the population carries
# gamma = 0, so the true evolution is exactly (1+z)^0.  scripts/measure_true_density.py
# measures it (results/true_density.json) and it is consistent with zero; keeping
# it at zero also leaves analysis 2's nuisance block untouched, so log10n0 is the
# only configuration difference between the two directories.
export DATA_ROOT=${DATA_ROOT:-/hildafs/projects/phy230014p/magana/gws-agn/working/data}
export DARKSIRENS_SRC=${DARKSIRENS_SRC:-/hildafs/projects/phy230014p/magana/src/darksirens}
export SEL_BATCH=${SEL_BATCH:-50000}
export PE_BLOCK=${PE_BLOCK:-25}
export KDE_W=${KDE_W:-4096}
export LOG10N0=${LOG10N0:--3.0}        # GAL  true comoving density, log10 Mpc^-3
export LOG10N0_C2=${LOG10N0_C2:--5.0}  # AGN  true comoving density, log10 Mpc^-3
export H0_TRUE=${H0_TRUE:-67.74}
export F_TRUE=${F_TRUE:-0.30}
export SEEDS=${SEEDS:-"100 101 102 103 105"}
export LEVELS=${LEVELS:-"m21 m20 m19 m18"}

# the K=2 configuration every scan in this directory shares
# $1 = seed, $2 = injection lane (targeted|popuni), $3 = events file,
# $4 = survey level (complete|m21|m20|m19|m18)
ds_common() {
  local seed=$1 lane=$2 ev=$3 lev=$4
  echo "--universe_model dark_sirens \
--catalog_sky_weighting field \
--survey_path ${DATA_ROOT}/seed${seed}/surveys/survey_gal_${lev}_ns32.h5 \
              ${DATA_ROOT}/seed${seed}/surveys/survey_agn_${lev}_ns32.h5 \
--gw_path ${ev} \
--gwselection_path ${DATA_ROOT}/seed${seed}/injections/injections_${lane}.h5 \
--log10n0 ${LOG10N0} --log10n0_c2 ${LOG10N0_C2} \
--selection_neff_guard hard --max_likelihood_variance 1e6 \
--kde_window ${KDE_W} --kde_window_nsigma 8 \
--sel_batch_size ${SEL_BATCH} --pe_event_block ${PE_BLOCK} \
--h0_true ${H0_TRUE} --f_true ${F_TRUE} --device gpu"
}
