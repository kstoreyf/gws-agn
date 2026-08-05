#!/usr/bin/env bash
# Collate whatever is on disk: stitch the complete-rung chunks, aggregate the
# ladder, render the README.  Idempotent and safe to run at any point in the
# campaign -- grids that are not finished are simply absent from the tables, and
# scripts/render_readme_tables.py says which rungs those are.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-/hildafs/home/magana/tmp_ondemand_hildafs_phy230014p_symlink/magana/.conda/envs/jax/bin/python}
mkdir -p logs

echo "[$(date -u +%FT%TZ)] === stitch the complete-rung H0 chunks (rung 0) ==="
./scripts/merge_complete_rung.sh

echo "[$(date -u +%FT%TZ)] === the nside-16 scaling verdict ==="
$PY -u scripts/nside_scaling_verdict.py | tee logs/nside_scaling_verdict.log

echo "[$(date -u +%FT%TZ)] === aggregate the ladder ==="
$PY -u scripts/aggregate_ladder.py --seeds 100 101 102 103 105 \
  | tee logs/aggregate_ladder.log

echo "[$(date -u +%FT%TZ)] === render the README ==="
$PY -u scripts/render_readme_tables.py | tee logs/render_readme.log

echo "[$(date -u +%FT%TZ)] === draw the figures ==="
$PY -u scripts/make_figures.py | tee logs/make_figures.log

echo "[$(date -u +%FT%TZ)] === status ==="
./scripts/status.sh
