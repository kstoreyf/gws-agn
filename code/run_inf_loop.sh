#!/usr/bin/env bash
# Submit Slurm jobs: one inference run per (seed, dLunc).
#
# TAG_MOCKTYPE (env, default _glass) must match the data config / generate_configs:
#   _glass   — inference basename includes tag_cat with _bgal*_bagn*
#   _uniform — basename is config_inference_uniform_... (no bias in tag_cat)
#
# Example: TAG_MOCKTYPE=_uniform ./run_inf_loop.sh

set -euo pipefail

root_dir="/sdf/home/k/ksf/gws-agn"
code_dir="${root_dir}/code"
config_dir="${root_dir}/configs/configs_inference"
log_dir="${code_dir}/logs"
mkdir -p "${log_dir}"


# Mirror create_config_inference + data tags from generate_configs / job_inference.s
TAG_MOCKTYPE="_uniform"
RATIO_NGAL_NAGN=1
BGAL=1.0
BAGN=1.0
NSIDE=64
FAGN=0.0
LAMBDAAGN=0.0
ZMAXGW=1.0
MCMC_NW=32
MCMC_NSTEPS=500
DZ=0.03
INFERENCE_SUFFIX=betaH0_vary-H0
GWS_AGN_CONDA_ENV=glassenv

#seeds=(0 1 2 3 4 5 6 7 8 9)
seeds=(0)
#dLunc_arr=(0.0 0.25 0.5 0.75 1.0)
dLunc_arr=(0.0 0.1 0.2 0.3)

for seed in "${seeds[@]}"; do
  for dLunc in "${dLunc_arr[@]}"; do
    seedgw=$((1000 + seed))

    tag_cat="_seed${seed}_ratioNgalNagn${RATIO_NGAL_NAGN}"
    if [[ "${TAG_MOCKTYPE}" == "_glass" ]]; then
      tag_cat+="_bgal${BGAL}_bagn${BAGN}"
    fi
    tag_pix="_nside${NSIDE}"
    tag_gw="_seedgw${seedgw}_fagn${FAGN}_lambdaagn${LAMBDAAGN}_zmaxgw${ZMAXGW}"
    tag_gwsamp="_dLunclin${dLunc}"
    tag_inf="_mcmc_nw${MCMC_NW}_nsteps${MCMC_NSTEPS}"
    tag_inf_extra="_Dz${DZ}_${INFERENCE_SUFFIX}"

    if [[ "${TAG_MOCKTYPE}" == "_glass" ]]; then
      prefix_mt=""
    else
      prefix_mt="${TAG_MOCKTYPE}"
    fi

    config_basename="config_inference${prefix_mt}${tag_cat}${tag_pix}${tag_gw}${tag_gwsamp}${tag_inf}${tag_inf_extra}.yaml"
    config_path="${config_dir}/${config_basename}"

    if [[ ! -f "${config_path}" ]]; then
      echo "Skipping (missing): ${config_basename}"
      continue
    fi

    echo "Submit ${config_basename}"
    job_tag="${TAG_MOCKTYPE#_}"
    wrap_cmd=$(cat <<EOF
  source "\$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${GWS_AGN_CONDA_ENV}"
  export XLA_PYTHON_CLIENT_PREALLOCATE=false
  export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
  echo "Running on host: \$(hostname)"
  nvidia-smi || true
  python - <<'PY'
  import sys
  import jax

  devices = jax.devices()
  backend = jax.default_backend()
  print("Preflight JAX devices:", devices)
  print("Preflight JAX backend:", backend)

  if backend != "gpu":
      print("ERROR: JAX is not on GPU; aborting to avoid CPU fallback.")
      sys.exit(1)
PY
  python run_inference.py "${config_path}"
EOF
)

    sbatch \
      --job-name="inf_${job_tag}_seed${seed}_dLunc${dLunc}" \
      --output="${log_dir}/%x.out" \
      --time="1:00:00" \
      --nodes=1 \
      --ntasks=1 \
      --cpus-per-task="32" \
      --mem="10G" \
      --gres="gpu:1" \
      --account="kipac:default" \
      --partition="ampere" \
      --qos="preemptable" \
      --chdir="${code_dir}" \
      --wrap="${wrap_cmd}"
  done
done
