#!/bin/bash
# experiment_model_equivalence — the scans.
#
# QUESTION: does darksirens' general incomplete-catalog model `dark_sirens`,
# driven into its complete-catalog limit, reproduce the dedicated K=1
# complete-catalog likelihood `dark_sirens_complete` on the same data — and
# where do the two agree BITWISE?
#
# ---------------------------------------------------------------------------
# WHY THERE ARE TWO SURVEY VARIANTS
# ---------------------------------------------------------------------------
# `dark_sirens` cannot be evaluated at all on the survey files as shipped.  Its
# completion term needs the observed-density KDE `_kde_dndz_obs`, which builds
# the truncated-kernel mass as
#
#     mass = ndtr((5 - z_i)/0.05) - ndtr(-z_i/0.05);  mass = max(mass, 1e-300)
#
# in the CATALOG'S storage dtype, while the kernel itself is promoted to the
# package zgrid's float64.  The survey files store galaxies in float32 and pad
# short rows at z = 100, so for every padded slot mass underflows to 0 (1e-300
# is not representable in float32), the float64 kernel is 0 there, and 0/0 = NaN
# — which the `* real_gal` mask cannot remove.  Every catalog row that has any
# padding comes back all-NaN; only the single row at the maximum galaxy count
# survives.  The NaN reaches the survey-global field normalizer log_Z_global,
# and the likelihood is -inf in every cell for every log10n0.
# `dark_sirens_complete` never touches that KDE, which is why it is unaffected.
#
# So the scans run on two survey variants and both are reported:
#
#   f32_* : the survey files exactly as the analysis of record uses them.
#           Establishes the blocker — the general model is -inf everywhere.
#   (none): float64 copies in data_derived/ (a pure precision widening; every
#           float32 value is exactly representable in float64, so no number
#           changed).  This is the only configuration in which the equivalence
#           question can be asked at all.
#
# ---------------------------------------------------------------------------
# ARMS on the float64 surveys, 4 configurations each
#   ({GAL, AGN} x {all 1000 events, the matched host-type subset})
# ---------------------------------------------------------------------------
#   dsc_*     REFERENCE  dark_sirens_complete, labels [H0, sigma_kde], sigma_kde = 0
#   ds_*      PRIMARY    dark_sirens at log10n0 = -12, labels
#                        [H0, log10n0, delta, sigma_kde], delta = 0, sigma_kde = 0,
#                        use_lss off — the complete-catalog limit as specified
#   dsdeep_*  PRIMARY, deeper limit: the same at log10n0 = -24.  A single-point
#                        pilot showed the -12 residual is the completion term
#                        itself (it scales with n0), so this locates where the
#                        limit becomes exact.
#   dstrue_*  SECONDARY  dark_sirens at the tracer's TRUE density (GAL
#                        log10(1e-3) = -3, AGN log10(1e-5) = -5).
#                        CHARACTERIZATION ONLY, never pass/fail.
#
# Everything except the model and log10n0 is held fixed: same survey, same
# events, same injection file (the targeted lane, the analysis of record's
# choice), same grid, same guard, same blocking, same GPU, same job.  Bit
# equality does not survive a change of device or compilation, which is why
# this is one serial job.
set -euo pipefail
cd "$(dirname "$0")/.."

DATA=/hildafs/projects/phy230014p/magana/gws-agn/working/data/seed100
A1=/hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_1_complete_catalog_H0

GW_ALL=$DATA/events/events.h5
GW_GAL=$A1/data_derived/events_gal_hosted.h5      # host_type == 0, READ ONLY
GW_AGN=$A1/data_derived/events_agn_hosted.h5      # host_type == 1, READ ONLY
SUR_GAL32=$DATA/surveys/survey_gal_complete_ns32.h5
SUR_AGN32=$DATA/surveys/survey_agn_complete_ns32.h5
SUR_GAL=data_derived/survey_gal_complete_ns32_f64.h5
SUR_AGN=data_derived/survey_agn_complete_ns32_f64.h5
INJ=$DATA/injections/injections_targeted.h5       # the SAME file for every scan

# The complete GAL survey block is (12288, 14569); darksirens'
# recommended_kde_window returns 3410 at n_sigma = 8, so W = 4096.  The AGN rows
# are far shorter than the module default and take the full-row path.
KDE_W=${KDE_W:-4096}
NGRID=${NGRID:-201}
# HENON-GPU carries A100-40s, half the memory analysis_1 was sized for
# (200000 / 100 on an A100-80), so both reductions are blocked twice as tightly.
# Identical for every scan, so it cannot enter any comparison.
SEL_BATCH=${SEL_BATCH:-50000}
PE_BLOCK=${PE_BLOCK:-25}

COMMON="--catalog_sky_weighting field --scan h0 --h0_true 67.74 \
  --gwselection_path $INJ --selection_neff_guard hard --max_likelihood_variance 1e6 \
  --sel_batch_size $SEL_BATCH --pe_event_block $PE_BLOCK --outdir results"

scan () {  # tag model survey events [extra...]
  local tag=$1 model=$2 sur=$3 ev=$4; shift 4
  if [ -s "results/${tag}.h5" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "[$(date -u +%H:%M:%S)] skip $tag (results/${tag}.h5 exists)"
    return 0
  fi
  echo "[$(date -u +%H:%M:%S)] scanning $tag  ($model)"
  python scripts/scan_h0f.py $COMMON --h0_grid 50.0 100.0 "$NGRID" \
    --universe_model "$model" --survey_path "$sur" --gw_path "$ev" "$@" \
    --out_tag "$tag" > "logs/${tag}.log" 2>&1
  echo "[$(date -u +%H:%M:%S)] done $tag"
}

point () {  # single-point evaluation at H0 = truth (blocker evidence, cheap)
  local tag=$1 model=$2 sur=$3 ev=$4; shift 4
  if [ -s "results/${tag}.h5" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "[$(date -u +%H:%M:%S)] skip $tag (exists)"; return 0
  fi
  echo "[$(date -u +%H:%M:%S)] single point $tag  ($model)"
  python scripts/scan_h0f.py $COMMON --h0_grid 67.74 67.74 1 \
    --universe_model "$model" --survey_path "$sur" --gw_path "$ev" "$@" \
    --out_tag "$tag" > "logs/${tag}.log" 2>&1
  echo "[$(date -u +%H:%M:%S)] done $tag"
}

# --------------------------------------------------------------------------- #
# Stage 0 — the float32 blocker, and the reference model's own dtype sensitivity
# --------------------------------------------------------------------------- #
echo "=== $(date -u) stage 0: survey files as shipped (float32) ==="
scan  f32_dsc_gal_matched dark_sirens_complete "$SUR_GAL32" "$GW_GAL" --kde_window "$KDE_W"
point f32_ds1pt_gal_all   dark_sirens          "$SUR_GAL32" "$GW_ALL" --kde_window "$KDE_W" --log10n0 -12

# --------------------------------------------------------------------------- #
# Stage 1 — AGN configurations on the float64 surveys (cheap; also a smoke test)
# --------------------------------------------------------------------------- #
echo "=== $(date -u) stage 1: AGN, float64 surveys ==="
scan dsc_agn_all        dark_sirens_complete "$SUR_AGN" "$GW_ALL"
scan ds_agn_all         dark_sirens          "$SUR_AGN" "$GW_ALL" --log10n0 -12
scan dsdeep_agn_all     dark_sirens          "$SUR_AGN" "$GW_ALL" --log10n0 -24
scan dstrue_agn_all     dark_sirens          "$SUR_AGN" "$GW_ALL" --log10n0 -5
scan dsc_agn_matched    dark_sirens_complete "$SUR_AGN" "$GW_AGN"
scan ds_agn_matched     dark_sirens          "$SUR_AGN" "$GW_AGN" --log10n0 -12
scan dsdeep_agn_matched dark_sirens          "$SUR_AGN" "$GW_AGN" --log10n0 -24
scan dstrue_agn_matched dark_sirens          "$SUR_AGN" "$GW_AGN" --log10n0 -5

# --------------------------------------------------------------------------- #
# Stage 2 — GAL configurations on the float64 surveys (the expensive ones)
# --------------------------------------------------------------------------- #
echo "=== $(date -u) stage 2: GAL, float64 surveys ==="
scan dsc_gal_all        dark_sirens_complete "$SUR_GAL" "$GW_ALL" --kde_window "$KDE_W"
scan ds_gal_all         dark_sirens          "$SUR_GAL" "$GW_ALL" --kde_window "$KDE_W" --log10n0 -12
scan dsdeep_gal_all     dark_sirens          "$SUR_GAL" "$GW_ALL" --kde_window "$KDE_W" --log10n0 -24
scan dsc_gal_matched    dark_sirens_complete "$SUR_GAL" "$GW_GAL" --kde_window "$KDE_W"
scan ds_gal_matched     dark_sirens          "$SUR_GAL" "$GW_GAL" --kde_window "$KDE_W" --log10n0 -12
scan dsdeep_gal_matched dark_sirens          "$SUR_GAL" "$GW_GAL" --kde_window "$KDE_W" --log10n0 -24

# --------------------------------------------------------------------------- #
# Stage 3 — the secondary arm on GAL (characterization; last, so a wall-clock
#           timeout can only cost this)
# --------------------------------------------------------------------------- #
echo "=== $(date -u) stage 3: GAL secondary arm ==="
scan dstrue_gal_all     dark_sirens "$SUR_GAL" "$GW_ALL" --kde_window "$KDE_W" --log10n0 -3
scan dstrue_gal_matched dark_sirens "$SUR_GAL" "$GW_GAL" --kde_window "$KDE_W" --log10n0 -3

echo "=== $(date -u) ALL SCANS DONE ==="
