#!/usr/bin/env bash
# Shared environment for the 2026-07-29 rerun on darksirens master @ 2b86a2d.
#
# DARKSIRENS_WT is a detached worktree pinned at 2b86a2d, so the run is immune to
# the shared checkout at /hildafs/projects/phy230014p/magana/src/darksirens moving
# under it.  Both PYTHONPATH (import) and DARKSIRENS_SRC (provenance `git rev-parse`)
# point at the pin.  Recreate with:
#   git -C /hildafs/projects/phy230014p/magana/src/darksirens worktree add --detach <path> 2b86a2d
: "${DARKSIRENS_WT:=/tmp/claude-88592/-hildafs-projects-phy230014p-magana-gws-agn/6b9abc89-f874-41de-9ed3-c0ca4def231c/scratchpad/wt-2b86a2d}"

export PYTHONPATH="$DARKSIRENS_WT"
export DARKSIRENS_SRC="$DARKSIRENS_WT"
export DARKSIRENS_ZMAX=1.5              # survey depth; NOT the 5.0 default (see RESULTS)
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Count-anchored true mean densities of the two catalogs (catalog_meta.json).
LOG10N0_GAL=-5.50627668499162
LOG10N0_AGN=-7.508083961432144

# Catalog-targeted injection lane — the operative prerequisite for field-mode
# sparse-tracer mixtures (see ../../gw_agn_darksirens_fixed/RESULTS.md).
INJ=../data/injections_cat.h5
INJB=../data/injections_cat_B.h5

# Legacy-equivalent total-variance budget: makes the post-#212 sigma^2_lnL <= V
# criterion inert (threshold collapses to the legacy Neff > 5*N_obs floor) so the
# scans are directly comparable with the #212-era run.
LEGACY_VAR=1e6

if [ ! -d "$DARKSIRENS_WT/darksirens" ]; then
  echo "[fatal] DARKSIRENS_WT=$DARKSIRENS_WT is not a darksirens checkout" >&2
  exit 1
fi
echo "[env] darksirens pin: $(git -C "$DARKSIRENS_WT" rev-parse --short HEAD) at $DARKSIRENS_WT"
