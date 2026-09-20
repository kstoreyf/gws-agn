#!/bin/bash
#SBATCH --job-name=a9_closure
#SBATCH --account=phy220048p
#SBATCH --partition=RITA-GPU
#SBATCH --qos=rita
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100-80:1
#SBATCH --mem=100G
#SBATCH --time=04:00:00
#SBATCH --output=/hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_9_marked_multitracer_H0_fagn/logs/a9_closure_%j.out
#SBATCH --error=/hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_9_marked_multitracer_H0_fagn/logs/a9_closure_%j.err
#
# The three closure checks of specification section 9, plus the measured cost
# model.  ~200 likelihood evaluations and three builds: minutes of GPU, not
# hours.  No production grid is run here.
#
#   sbatch --export=ALL,STAGE="closure k1refs" scripts/submit_a9_closure_rita.sbatch
#
# --qos=rita is load-bearing: this account holds priority on RITA-GPU (the QOS
# preempts rita-s).  A PENDING spell is normal; poll squeue, do not resubmit and
# do not shrink the request.
set -euo pipefail
source /hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_9_marked_multitracer_H0_fagn/scripts/env_a9.sh
echo "[a9] host=$(hostname) job=${SLURM_JOB_ID:-none} stages='${STAGE:-closure}'"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
for s in ${STAGE:-closure}; do
  echo "[a9] === stage ${s} ==="
  "${A9_PY}" "${A9_DIR}/scripts/a9_closure.py" --stage "${s}"
done
echo "[a9] all stages done"
