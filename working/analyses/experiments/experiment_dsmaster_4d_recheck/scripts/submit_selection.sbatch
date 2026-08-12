#!/bin/bash
#SBATCH --job-name=dsm_sel
#SBATCH --account=phy220048p
#SBATCH --partition=RITA-GPU
#SBATCH --qos=rita
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=110G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/slurm_sel_%j.out
#SBATCH --error=logs/slurm_sel_%j.err
#
# The THIRD arm: c_mode=selection, the parametric magnitude-limited
# completeness, on the same rung/seed/rstate as the two finished arms.  The
# owner's read is that this is probably the right mode, which is what makes it
# the arm that decides whether the analysis 3-6 ladder gets re-run.
#
# DO NOT SUBMIT until scripts/fit_selection.sh has produced BOTH fits with the
# matched (schechter, M_faint_offset = -0.0944) settings.  Anchoring is
# all-or-nothing upstream and a Gaussian fit of our LF is badly misspecified in
# exactly the faint end that sets the completion budget, so a run against the
# wrong fits would answer a question nobody asked.  See RESULTS.md "Next".
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
. scripts/env.sh

export CMODE=selection
export SRV_SUBDIR=surveys_galprop
# The arm runs on the selection worktree at darksirens master 0c5b3db (#342-#349:
# K>=2 x c_mode=selection, homogeneous-schechter, bright-truncated negative
# M_faint_offset, suffstats fit).  A merged SHA on purpose -- this arm decides
# whether the analysis 3-6 ladder is re-run, so its provenance has to resolve
# permanently rather than point at a branch commit that may be deleted.
export DS_MASTER=${DS_SEL}
export PYTHONPATH=${DS_SEL}
export DARKSIRENS_SRC=${DS_SEL}

# GAL then AGN, matching --survey_path's order in env.sh.  These are the
# TRUE-Z calibration fits (see scripts/fit_selection.sh); the inference itself
# still reads the photo-z surveys via SRV_SUBDIR=surveys_galprop above.
FITS=${FITS:-results/selection_fits_truez/fit_gal_${RUNG}_schechter.json,results/selection_fits_truez/fit_agn_${RUNG}_schechter.json}

for f in ${FITS//,/ }; do
  [ -f "$f" ] || { echo "missing selection fit: $f -- run scripts/fit_selection.sh first" >&2; exit 1; }
done

SELECTION_FIT="$FITS" ./scripts/run_4d.sh
