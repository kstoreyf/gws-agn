#!/usr/bin/env bash
# Offline magnitude fits for the c_mode=selection arm -- one per catalog.
#
# These are SINGLE-catalog fits (the K>=2 restriction lives at inference time,
# not fit time), so both run today.  What does NOT run today is the fit our
# mock actually needs: see "the matched fit" below.
#
# Runs on CPU by design.  fit_selection reaches JAX through
# reference_absolute_mags -> distance_modulus, so it OOMs against a shared
# login-node GPU; JAX_PLATFORMS=cpu finishes in seconds at our sizes (8k AGN,
# 821k galaxies) and takes nothing off the queue.
set -euo pipefail
cd "$(dirname "$0")/.."
. scripts/env.sh

export PYTHONPATH=${DS_SEL}          # 5aa90fa, not DS_MASTER
export DARKSIRENS_SRC=${DS_SEL}
export JAX_PLATFORMS=${JAX_PLATFORMS:-cpu}

# CALIBRATION reads the TRUE-redshift surveys; INFERENCE reads the photo-z ones.
#
# The fit builds Mhat = m - DM(z; H0=100) from whatever redshift the survey
# carries.  On our photo-z surveys that smears the LF's sharp bright edge by
# sd(Mhat_obs - Mhat_true) = 0.0866 mag against a 0.0944 mag truncation depth --
# the edge is blurred to its own width, --m_faint_cut then drops the 3.35% that
# scattered faint-ward, and the fit misses truth by 3.2 sigma (Mstar_hat) and
# 4.3 sigma (alpha).  Not a footnote: upstream measured max|dC_sel| = 0.19 and
# the missing budget (1 - C_sel) -- the term that places missing hosts --
# mispriced up to ~2.5x in the transition zone where our events' hosts live.
#
# Calibrating theta on true z is legitimate HERE because this experiment
# isolates the completeness ESTIMATOR against a known truth; a photo-z
# calibration nuisance is not what it measures.  For real photo-z data the fix
# is forward-modelling the scatter into the fit normalization -- NOT widening
# the cut, which trades a sharp bias for a contamination bias you cannot
# propagate.
SRV=${SRV:-${DATA_ROOT}/seed${SEED}/surveys_truez}
OUT=${OUT:-results/selection_fits_truez}
MLIM=${MLIM:-18.0}
FAMILY=${FAMILY:-schechter}

# ----------------------------------------------------------------- the matched fit
# Our mock draws absolute magnitudes from a Schechter TRUNCATED AT
# x_cut = 1.0907900366549803 L*, the cut that makes the integrated number
# density come out at 1e-3 Mpc^-3 -- i.e. log10n0 = -3.0, the inference truth.
# The density anchor and the LF truncation are one construction; neither moves
# alone.  Support for it landed in darksirens PR #347.
#
# TWO flags are needed and they do different jobs:
#
#   --m_faint_offset  sets the CONSUMED curve's denominator (which galaxies the
#                     completeness answers for).  It never enters the fit
#                     likelihood -- deliberately unfitted.
#   --m_faint_cut     restricts the FIT's per-galaxy normalization to
#                     x >= max(x_lim, x_cut), i.e. tells the fitter the sample
#                     is bright-truncated.  THIS is what keeps alpha honest.
#
# Passing the offset without the cut is what produced our alpha = +3.22 against
# a truth of -1.07: the sample was truncated but the model had no faint edge.
# Upstream now refuses that pairing fail-closed, because it failed silently.
#
# Both numbers are DERIVED from the seed's recorded glass_field_meta.json by
# scripts/lf_constants.py, which also asserts the identity
# m_faint_cut - Mstar_hat == m_faint_offset.  Never retype them: --m_faint_cut
# is h-SCALED (Mhat = m - DM(z; H0=100)) while M_B_faint_limit is not, and a
# botched 5 log10 h shifts the fit support by 0.85 mag.
eval "$(python scripts/lf_constants.py --seed "${SEED}" --emit shell)"
M_FAINT_OFFSET=${M_FAINT_OFFSET:-$M_FAINT_OFFSET}
M_FAINT_CUT=${M_FAINT_CUT:-$M_FAINT_CUT}
echo "LF constants (derived from glass_field_meta.json, seed ${SEED}):"
python scripts/lf_constants.py --seed "${SEED}"
echo

mkdir -p "$OUT" logs
# AGN FIRST, deliberately.  The bright-truncated fit's cost scales with the
# catalog size and is single-core CPU-bound whatever JAX_PLATFORMS says (the
# GPU sat at 0% for two hours on the galaxy catalog), so the 8k AGN fit is the
# cheap end-to-end validation of the recipe -- offset sign, h-scaling of the
# cut, recovery of the LF -- and it lands in minutes.  Only then does the 821k
# galaxy fit, which is the expensive one, get its hours.
for T in ${TRACER_ORDER:-agn gal}; do
  echo "=== ${T} ${RUNG} family=${FAMILY} offset=${M_FAINT_OFFSET} cut=${M_FAINT_CUT} ==="
  python -u "${DS_SEL}/darksirens/cli/fit_selection.py" \
    --survey_path "${SRV}/survey_${T}_${RUNG}_ns32.h5" \
    --m_lim "${MLIM}" \
    --family "${FAMILY}" \
    --m_faint_offset "${M_FAINT_OFFSET}" \
    --m_faint_cut "${M_FAINT_CUT}" \
    --out "${OUT}/fit_${T}_${RUNG}_${FAMILY}.json" \
    2>&1 | tee "logs/fit_selection_${T}_${RUNG}_${FAMILY}.log"
done

# The acceptance test upstream asked for: at n=821k the error bars are ~1e-3
# mag, so any pull beyond ~3 sigma is a systematic (h-scaling of the cut, the
# z >= 0.01 floor drops, K-correction leakage), not noise.
python scripts/check_fit_recovery.py --fits "${OUT}" --rung "${RUNG}" \
                                     --family "${FAMILY}" --seed "${SEED}"

echo
echo "fits written to ${OUT}:"
ls -la "${OUT}"
echo
echo "inference order is GAL then AGN, matching --survey_path:"
echo "  --selection_fit ${OUT}/fit_gal_${RUNG}_${FAMILY}.json,${OUT}/fit_agn_${RUNG}_${FAMILY}.json"
