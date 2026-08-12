#!/usr/bin/env bash
# Shared configuration for analysis_6 — the RELATIVE-COMPLETENESS surface.
#
# Analyses 3/4/5 all live on the DIAGONAL of a two-dimensional plane: the mock's
# two tracers share a magnitude distribution, so every rung of analysis 3's
# ladder has C_GAL = C_AGN by construction (surveys_meta.json, within the
# horizon: m20 0.8143/0.8143, m19 0.3151/0.3170, m18 0.0954/0.0960).  Real
# surveys never do — AGN are selected in different bands to different depths.
#
# Analysis 4's single off-diagonal point is the reason this directory exists:
# GAL at m < 18 with the AGN survey COMPLETE did not remove the faint-rung
# f_AGN bias, it tripled it (+0.073 -> +0.197).  Here the GAL and AGN survey
# depths are varied INDEPENDENTLY, with both completion densities at the mock's
# truth, to ask whether the bias is a function of the RATIO C_AGN / C_GAL.
#
# Configuration is byte-identical to analysis_3/scripts/env.sh and
# analysis_4/scripts/env.sh: darksirens @ 2b86a2d, K = 2 dark_sirens mixture,
# catalog_sky_weighting field, survey order [GAL, AGN] so fcat_2 = f_AGN,
# delta = delta_c2 = 0, sigma_kde = sigma_kde_c2 = 0, population fixed at the
# mock fiducial, Om0 pinned, guard `hard` with max_likelihood_variance 1e6,
# W = 4096 (n_sigma 8), (sel_batch_size, pe_event_block) = (50000, 25), grid
# H0 [50,100] x 201 * f [0,1] x 41.  scan_h0f.py is byte-identical to
# analysis 3's (md5 02acecc6f73d5ae0bd31985e2b7ac1c3).
#
# BOTH densities stay at truth in every cell.  The anchoring axis was measured
# in analysis 4 and the free-anchor cost in analysis 5; mixing either into this
# scan would confound the completeness ratio with them.
export DATA_ROOT=${DATA_ROOT:-/hildafs/projects/phy230014p/magana/gws-agn/working/data}
export DARKSIRENS_SRC=${DARKSIRENS_SRC:-/hildafs/projects/phy230014p/magana/src/darksirens}
export SEL_BATCH=${SEL_BATCH:-50000}
export PE_BLOCK=${PE_BLOCK:-25}
export KDE_W=${KDE_W:-4096}
export LOG10N0=${LOG10N0:--3.0}         # GAL true comoving density, log10 Mpc^-3
export LOG10N0_C2=${LOG10N0_C2:--5.0}   # AGN true comoving density, log10 Mpc^-3
export H0_TRUE=${H0_TRUE:-67.74}
export F_TRUE=${F_TRUE:-0.30}

# the K=2 configuration every scan in this directory shares
# $1 = seed, $2 = injection lane, $3 = events file, $4 = GAL level, $5 = AGN level
ds_common() {
  local seed=$1 lane=$2 ev=$3 glev=$4 alev=$5
  echo "--universe_model dark_sirens \
--catalog_sky_weighting field \
--survey_path ${DATA_ROOT}/seed${seed}/surveys/survey_gal_${glev}_ns32.h5 \
              ${DATA_ROOT}/seed${seed}/surveys/survey_agn_${alev}_ns32.h5 \
--gw_path ${ev} \
--gwselection_path ${DATA_ROOT}/seed${seed}/injections/injections_${lane}.h5 \
--log10n0 ${LOG10N0} --log10n0_c2 ${LOG10N0_C2} \
--selection_neff_guard hard --max_likelihood_variance 1e6 \
--kde_window ${KDE_W} --kde_window_nsigma 8 \
--sel_batch_size ${SEL_BATCH} --pe_event_block ${PE_BLOCK} \
--h0_true ${H0_TRUE} --f_true ${F_TRUE} --device gpu"
}
