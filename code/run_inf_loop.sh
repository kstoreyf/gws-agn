#!/usr/bin/env bash
# Submit one inference Slurm job per seed.

set -euo pipefail

root_dir="/sdf/home/k/ksf/gws-agn"
code_dir="${root_dir}/code"
config_dir="${root_dir}/configs/configs_inference"
log_dir="${code_dir}/logs"
mkdir -p "${log_dir}"

# Mirror config defaults from job_inference.s.
RATIO_NGAL_NAGN=1
BGAL=1.0
BAGN=1.0
NSIDE=64
FAGN=0.0
LAMBDAAGN=0.0
ZMAXGW=1.0
DLUNC=0.0
MCMC_NW=32
MCMC_NSTEPS=500
DZ=0.03
INFERENCE_SUFFIX=betaH0_vary-H0
GWS_AGN_CONDA_ENV=glassenv

#seeds=(0 1 2 3 4 5 6 7 8 9)
seeds=(0)
dLunc_arr=(0.0 0.25 0.5 0.75 1.0)
for seed in "${seeds[@]}"; do
  for dLunc in "${dLunc_arr[@]}"; do
    seedgw=$((1000 + seed))
    config_basename="config_inference_seed${seed}_ratioNgalNagn${RATIO_NGAL_NAGN}_bgal${BGAL}_bagn${BAGN}_nside${NSIDE}_seedgw${seedgw}_fagn${FAGN}_lambdaagn${LAMBDAAGN}_zmaxgw${ZMAXGW}_dLunc${dLunc}_mcmc_nw${MCMC_NW}_nsteps${MCMC_NSTEPS}_Dz${DZ}_${INFERENCE_SUFFIX}.yaml"
    config_path="${config_dir}/${config_basename}"

    if [[ ! -f "${config_path}" ]]; then
      echo "Skipping seed ${seed}: config not found (${config_path})"
      continue
    fi

    echo "Submitting seed ${seed} with config ${config_basename}"
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
  python run_inference.py --config "${config_path}"
EOF
)

    sbatch \
      --job-name="inference_mcmc_seed${seed}_dLunc${dLunc}" \
      --output="${log_dir}/%x_%j.out" \
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
