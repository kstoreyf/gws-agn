#!/usr/bin/env bash
# Submit Slurm jobs: one inference run per (mocktype, seed, dLunc, Dz, f/lambda).
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

RATIO_NGAL_NAGN=1
BGAL=1.0
BAGN=1.0
NSIDE=64
ZMAXGW=1.0
MCMC_NW=32
MCMC_NSTEPS=500
INFERENCE_SUFFIX=claudetry1_vary-H0-alphaagn
GWS_AGN_CONDA_ENV=glassenv

seeds=(0 1 2 3 4 5)
dLunc_arr=(0.05)
Dz_arr=(0.001)
mocktypes=(_glass)
fagn_lambda_arr=(
  "0.25 0.25"
)

for TAG_MOCKTYPE in "${mocktypes[@]}"; do
  for seed in "${seeds[@]}"; do
    for dLunc in "${dLunc_arr[@]}"; do
      for DZ in "${Dz_arr[@]}"; do
        for pair in "${fagn_lambda_arr[@]}"; do
          read -r FAGN LAMBDAAGN <<< "${pair}"
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
            --job-name="inf_${job_tag}_s${seed}_f${FAGN}_l${LAMBDAAGN}_dL${dLunc}" \
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
    done
  done
done
