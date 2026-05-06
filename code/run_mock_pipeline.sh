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

# Same conda env as job_inference.s unless overridden (must provide camb, glass, healpy, etc.).
: "${GWS_AGN_CONDA_ENV:=glassenv}"
if command -v module >/dev/null 2>&1; then
  module load conda 2>/dev/null || true
fi
if command -v conda >/dev/null 2>&1; then
  _conda_base="$(conda info --base 2>/dev/null)" || _conda_base=""
  if [[ -n "${_conda_base}" && -f "${_conda_base}/etc/profile.d/conda.sh" ]]; then
    # shellcheck source=/dev/null
    source "${_conda_base}/etc/profile.d/conda.sh"
  fi
  conda activate "${GWS_AGN_CONDA_ENV}"
fi

# --- User-adjustable parameters (lowercase to match other scripts) -------------
seed=1
seedgw=1001
ratio_ngal_nagn=1
bgal=1.0
bagn=1.0
#nside=256
nside=64
fagn=0.0
lambdaagn=0.0
zmaxgw=1.0
dLunc=0.1
#dLunc=0.75
# -------------------------------------------------------------------------------

#root_dir="/global/homes/k/kstoreyf/gws-agn" #perl
root_dir="/sdf/home/k/ksf/gws-agn" #s3df
config_dir="${root_dir}/configs/configs_data"
config_basename="config_data_seed${seed}_ratioNgalNagn${ratio_ngal_nagn}_bgal${bgal}_bagn${bagn}_nside${nside}_seedgw${seedgw}_fagn${fagn}_lambdaagn${lambdaagn}_zmaxgw${zmaxgw}_dLunc${dLunc}.yaml"
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