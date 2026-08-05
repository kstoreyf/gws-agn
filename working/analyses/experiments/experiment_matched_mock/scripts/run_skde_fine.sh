#!/bin/bash
# Finer sigma_kde ladder around the threshold + second-realisation check of 0.040.
set -u
cd "$(dirname "$0")/.."
LOG=logs/chain_20260730.log
COMMON="--universe_model dark_sirens_complete --catalog_sky_weighting field \
  --scan h0 --h0_grid 58.0 78.0 161 --h0_true 67.74 \
  --selection_neff_guard hard --max_likelihood_variance 1e6 --outdir results"
B="--survey_path data_derived/deep_survey_z2_ns16.h5 --gw_path data_derived/obsdet/ev_obs_b.h5 --gwselection_path data_derived/obsdet/sel_obs.h5"
S="--survey_path data_derived/survey_s4102_ns16.h5 --gw_path data_derived/obsdet/ev_obs_s4102.h5 --gwselection_path data_derived/obsdet/sel_obs.h5"
for sk in 0.025 0.030 0.035 0.050 0.070; do
  if python scripts/scan_h0f.py $COMMON $B --nuisance_json "{\"sigma_kde\": $sk}" \
      --out_tag "skde_${sk}" > "logs/skde_${sk}.log" 2>&1; then
    m=$(python -c "import json;print(round(json.load(open('results/skde_${sk}.json'))['H0']['median']-67.74,3))" 2>/dev/null)
    echo "[CHAIN2] skde=$sk (b) offset=$m" >> "$LOG"
  else echo "[CHAIN2] skde=$sk (b) FAILED" >> "$LOG"; fi
done
for sk in 0.020 0.040; do
  if python scripts/scan_h0f.py $COMMON $S --nuisance_json "{\"sigma_kde\": $sk}" \
      --out_tag "skde_s4102_${sk}" > "logs/skde_s4102_${sk}.log" 2>&1; then
    m=$(python -c "import json;print(round(json.load(open('results/skde_s4102_${sk}.json'))['H0']['median']-67.74,3))" 2>/dev/null)
    echo "[CHAIN2] skde=$sk (s4102) offset=$m" >> "$LOG"
  else echo "[CHAIN2] skde=$sk (s4102) FAILED" >> "$LOG"; fi
done
echo "[CHAIN2] FINE LADDER DONE" >> "$LOG"
