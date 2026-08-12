#!/usr/bin/env bash
# Aggregate -> figures -> README, in that order, from whatever grids are on disk.
# Idempotent and safe mid-campaign: absent arms are recorded as missing, never
# imputed, and every figure whose inputs are absent is skipped rather than drawn.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python scripts/aggregate_arms.py "$@"
python scripts/make_figures.py
python scripts/render_readme_tables.py
echo
echo "figs/:"
ls -1 figs/
