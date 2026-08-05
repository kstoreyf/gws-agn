#!/usr/bin/env bash
# THE v3 ANALYSIS OF RECORD -- the six production scans, the five matched controls,
# the guard diagnostics, the single-tracer values, the closure table and the figures.
#
#   ./scripts/run_v3_analysis.sh
#
# Assumes working/data/seed{100,101,102,103,105} already point at the v3 dataset
# (the pilot gate has passed and the symlinks have been re-pointed).  Configuration
# is the campaign's, unchanged: dark_sirens at log10n0 = -24, field weighting, K = 1,
# targeted injections, H0 in [50, 100] x 201, W = 4096 (GAL), hard N_eff guard with
# max_likelihood_variance = 1e6.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs results figs

export SEL_BATCH=${SEL_BATCH:-50000}
export PE_BLOCK=${PE_BLOCK:-25}
export KDE_W=${KDE_W:-4096}
export LOG10N0=${LOG10N0:--24}
export DARKSIRENS_SRC=${DARKSIRENS_SRC:-/hildafs/projects/phy230014p/magana/src/darksirens}

echo "[$(date -u +%H:%M:%S)] recommended KDE window on the v3 survey blocks"
python scripts/kde_window_check.py > logs/v3_kde_window.log 2>&1
python - <<'PY'
import json, pathlib
d = json.loads((pathlib.Path("results") / "kde_window.json").read_text())
print("  window_required            :", d.get("window_required"))
print("  window_recommended_pow2    :", d.get("window_recommended_power_of_two"))
PY

echo "[$(date -u +%H:%M:%S)] matched-host subsets, seed 100"
DATA=/hildafs/projects/phy230014p/magana/gws-agn/working/data/seed100
for T in gal agn; do
  HT=0; [ "$T" = agn ] && HT=1
  python scripts/build_hosttype_subset.py --in_path "$DATA/events/events.h5" \
      --out_path "data_derived/events_${T}_hosted.h5" --host_type $HT \
      > "logs/v3_subset_${T}.log" 2>&1
done

echo "[$(date -u +%H:%M:%S)] the six production scans"
FORCE=1 ./scripts/run_scans.sh

if [ "${SKIP_CONTROLS:-0}" != "1" ]; then
  echo "[$(date -u +%H:%M:%S)] the four remaining matched controls"
  # run_seed_controls.sh only rebuilds a host-type subset when it is ABSENT, so the
  # stale v2 subsets on the bulk allocation must go first.
  BULK=/hildafs/projects/phy220048p/magana/gws-agn-data/derived/analysis_1_complete_catalog_H0
  for S in 101 102 103 105; do
    rm -f "$BULK/seed$S/events_gal_hosted.h5" "$BULK/seed$S/events_agn_hosted.h5"
  done
  ./scripts/run_seed_controls.sh 101 102 103 105
else
  echo "[$(date -u +%H:%M:%S)] SKIP_CONTROLS=1 -- the matched controls are running elsewhere"
fi

echo "[$(date -u +%H:%M:%S)] guard diagnostics"
./scripts/run_guard_diag.sh || echo "[warn] guard diagnostics failed"

if [ "${SKIP_AGGREGATE:-0}" = "1" ]; then
  echo "[$(date -u +%H:%M:%S)] SKIP_AGGREGATE=1 -- stopping before the aggregation"
  exit 0
fi
echo "[$(date -u +%H:%M:%S)] single-tracer values + closure table + figures"
python scripts/build_single_tracer.py       > logs/v3_build_single_tracer.log 2>&1
python scripts/aggregate_closure.py --seeds 100 101 102 103 105 \
      > logs/v3_aggregate_closure.log 2>&1
python scripts/make_figures.py              > logs/v3_make_figures.log 2>&1
python scripts/fig_closure_after_fix.py --seeds 100 101 102 103 105 \
      --before_dir results_v2postfix \
      --before_label "v2 (post-(b2)/(c2)) measurement family" \
      --after_label  "v3 all-observable family + realised photo-z" \
      --fig_tag fig_closure_v3 \
      --out_json results/closure_v3.json \
      --what "matched-host controls under the v2 post-fix measurement family (latent-width component masses, exact flat-prior mass PE, declared-but-unrealised catalog photo-z) and under the v3 all-observable family of working/data/DESIGN_PE.md (rho_obs = rho_opt + N(0,1), every width a_x*8/rho_obs, PE exact in (ln Mc, ln q, rho, chieff, sky), dL derived from the SNR) with the declared photo-z realised in the catalog (D3).  Identical analysis configuration: dark_sirens at log10n0 = -24, field weighting, K = 1, targeted injections, H0 in [50, 100] x 201, W = 4096 (GAL), campaign guard convention.  The datasets are DIFFERENT draws, so this is an unpaired comparison." \
      > logs/v3_fig_closure.log 2>&1
echo "[$(date -u +%H:%M:%S)] done"
tail -6 logs/v3_aggregate_closure.log || true
