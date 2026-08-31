#!/usr/bin/env bash
# Submit Slurm jobs: one mock pipeline run per (mocktype, seed, dLunc, f/lambda).
#
# TAG_MOCKTYPE (env, default _glass) must match generate_configs / paths.tag_mocktype:
#   _glass   — config basename includes _bgal*_bagn*
#   _uniform — basename is config_data_uniform_... (no bias segment in tag_cat)
#
# Example: TAG_MOCKTYPE=_uniform ./run_mock_loop.sh

set -euo pipefail

root="/sdf/home/k/ksf/gws-agn"
code_dir="${root}/code"
config_dir="${root}/configs/configs_data"
log_dir="${code_dir}/logs"
mkdir -p "${log_dir}"

RATIO_NGAL_NAGN=1
BGAL=1.0
BAGN=1.0
NSIDE=64
ZMAXGW=1.0

seeds=(0 1 2 3 4 5)
dLunc_arr=(0.05)
mocktypes=(_glass)
fagn_lambda_arr=(
  "0.25 0.25"
)

for TAG_MOCKTYPE in "${mocktypes[@]}"; do
  for seed in "${seeds[@]}"; do
    for dLunc in "${dLunc_arr[@]}"; do
      for pair in "${fagn_lambda_arr[@]}"; do
        read -r FAGN LAMBDAAGN <<< "${pair}"
        seed_gw=$((1000 + seed))

        tag_cat="_seed${seed}_ratioNgalNagn${RATIO_NGAL_NAGN}"
        if [[ "${TAG_MOCKTYPE}" == "_glass" ]]; then
          tag_cat+="_bgal${BGAL}_bagn${BAGN}"
        fi
        tag_pix="_nside${NSIDE}"
        tag_gw="_seedgw${seed_gw}_fagn${FAGN}_lambdaagn${LAMBDAAGN}_zmaxgw${ZMAXGW}"
        tag_gwsamp="_dLunclin${dLunc}"

        if [[ "${TAG_MOCKTYPE}" == "_glass" ]]; then
          prefix_data=""
        else
          prefix_data="${TAG_MOCKTYPE}"
        fi

        config_basename="config_data${prefix_data}${tag_cat}${tag_pix}${tag_gw}${tag_gwsamp}.yaml"
        config_path="${config_dir}/${config_basename}"

        if [[ ! -f "${config_path}" ]]; then
          echo "Skipping (missing): ${config_basename}"
          continue
        fi

        echo "Submit ${config_basename}"
        job_tag="${TAG_MOCKTYPE#_}"
        sbatch \
          --job-name="mock_${job_tag}_s${seed}_f${FAGN}_l${LAMBDAAGN}_dL${dLunc}" \
          --output="${log_dir}/%x.out" \
          --time="02:00:00" \
          --nodes=1 \
          --ntasks=1 \
          --cpus-per-task=4 \
          --mem="35G" \
          --account="kipac:default" \
          --partition="roma" \
          --qos="preemptable" \
          --chdir="${code_dir}" \
          --wrap="bash run_mock_pipeline.sh '${config_path}'"
      done
    done
  done
done
