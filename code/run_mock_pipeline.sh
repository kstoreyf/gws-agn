#!/usr/bin/env bash
# Wrapper to run the mock → GW samples → pixelization pipeline with one config.
# Adjust the parameter values below; they are embedded into the expected config
# filename. You can also pass an explicit config path as the first argument.

set -euo pipefail

# If invoked via "sh run_mock_pipeline.sh", re-exec under bash so that
# bash-specific syntax (process substitution, [[ .. ]]) works.
if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi

# --- User-adjustable parameters (lowercase to match other scripts) -------------
seed=42
ratio_ngal_nagn=10
bgal=1.0
bagn=1.0
fagn=0.5
lambdaagn=0.0
# -------------------------------------------------------------------------------

root_dir="/global/homes/k/kstoreyf/gws-agn"
config_dir="${root_dir}/configs/configs_data"
config_basename="config_data_seed${seed}_ratioNgalNagn${ratio_ngal_nagn}_bgal${bgal}_bagn${bagn}_fagn${fagn}_lambdaagn${lambdaagn}.yaml"
config_path="${config_dir}/${config_basename}"

# Allow overriding the config path as the first argument.
if [[ $# -ge 1 ]]; then
  config_path="$1"
  config_basename="$(basename "$config_path")"
fi

if [[ ! -f "$config_path" ]]; then
  echo "Config not found: $config_path" >&2
  exit 1
fi

log_dir="${root_dir}/logs"
mkdir -p "$log_dir"
timestamp="$(date +%Y%m%d_%H%M%S)"
log_file="${log_dir}/mock_pipeline_${config_basename%.yaml}_${timestamp}.log"
log_name="$(basename "$log_file")"

echo "Using config: $config_path"
echo "Logging to:   $log_file"
echo

# Capture stdout/stderr to both console and log file.
#exec > (tee -a "$log_file") 2>&1
{
  cd "${root_dir}/code"

  echo
  echo "[$(date)] Running make_mocks.py..."
  python make_mocks.py "$config_path"

  echo
  echo "[$(date)] Running generate_gwsamples.py..."
  python generate_gwsamples.py "$config_path"

  echo
  echo "[$(date)] Running pixelize_catalogs.py..."
  python pixelize_catalogs.py "$config_path"

  echo
  echo "[$(date)] Pipeline finished."
  echo "You can view the progress with:"
  echo "tail -f logs/${log_name}"
} 2>&1 | tee -a "$log_file"