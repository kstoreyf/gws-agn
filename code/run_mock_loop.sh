#!/usr/bin/env bash
# Submit one mock pipeline Slurm job per seed.

set -euo pipefail

root_dir="/sdf/home/k/ksf/gws-agn"
code_dir="${root_dir}/code"
config_dir="${root_dir}/configs/configs_data"
log_dir="${code_dir}/logs"
mkdir -p "${log_dir}"

# Mirror defaults in run_mock_pipeline.sh.
RATIO_NGAL_NAGN=1
BGAL=1.0
BAGN=1.0
NSIDE=64
FAGN=0.0
LAMBDAAGN=0.0
ZMAXGW=1.0

#seeds=(0 1 2 3 4 5 6 7 8 9)
seeds=(0)
#seeds=(2 3 4 5 6 7 8 9)
dLunc_arr=(0.0 0.25 0.5 0.75 1.0)

for seed in "${seeds[@]}"; do
  for dLunc in "${dLunc_arr[@]}"; do
    seedgw=$((1000 + seed))
    config_basename="config_data_seed${seed}_ratioNgalNagn${RATIO_NGAL_NAGN}_bgal${BGAL}_bagn${BAGN}_nside${NSIDE}_seedgw${seedgw}_fagn${FAGN}_lambdaagn${LAMBDAAGN}_zmaxgw${ZMAXGW}_dLunc${dLunc}.yaml"
    config_path="${config_dir}/${config_basename}"

    if [[ ! -f "${config_path}" ]]; then
      echo "Skipping seed ${seed}: config not found (${config_path})"
      continue
    fi

    echo "Submitting seed ${seed} dLunc=${dLunc} with config ${config_basename}"
    wrap_cmd="bash run_mock_pipeline.sh '${config_path}'"

    sbatch \
      --job-name="mocks_seed${seed}_dLunc${dLunc}" \
      --output="${log_dir}/%x_%j.out" \
      --time="02:00:00" \
      --nodes=1 \
      --ntasks=1 \
      --cpus-per-task="4" \
      --mem="16G" \
      --account="kipac:default" \
      --partition="roma" \
      --qos="preemptable" \
      --chdir="${code_dir}" \
      --wrap="${wrap_cmd}"
  done
done
