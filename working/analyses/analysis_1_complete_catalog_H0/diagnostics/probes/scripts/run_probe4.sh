#!/bin/bash
set -uo pipefail
cd "$(dirname "$0")/.."
export XLA_PYTHON_CLIENT_PREALLOCATE=false
echo "=== $(date -u) probe4 scans ==="
python -u scripts/probe4_continuum_survey.py scan
echo "=== $(date -u) probe4 analyse ==="
python -u scripts/probe4_continuum_survey.py analyse
echo "=== $(date -u) probe4 DONE ==="
