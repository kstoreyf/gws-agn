#!/bin/bash
#SBATCH --job-name=a10_evdec
#SBATCH --account=phy220048p
#SBATCH --partition=RITA-GPU
#SBATCH --qos=rita
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100-80:1
#SBATCH --mem=100G
#SBATCH --time=01:00:00
#SBATCH --output=/hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_10_mass_spin_marked_multitracer/logs/a10_event_decomposition_%j.out
#SBATCH --error=/hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_10_mass_spin_marked_multitracer/logs/a10_event_decomposition_%j.err
#
# Analysis 10 -- the per-event decomposition under the four NESTED models at ONE
# hyperparameter point.  One build (~25 s), six production evaluations (~3 s
# each after the first), one operand capture and FOUR per-event passes over the
# 1000 events (Analysis 9 measured ~10 s build + ~30 s pass per capture on a
# rita A100-80), so ~5 minutes of GPU work inside a one-hour request.
#
#   sbatch scripts/submit_a10_event_decomposition_rita.sbatch
#   sbatch --export=ALL,A10_DECOMP_ARGS="--f 0.35 --dmu_chi 0.1075 --dmu_G 5 \
#          --tag map_..." scripts/submit_a10_event_decomposition_rita.sbatch
#
# --qos=rita contributes ZERO to the priority number, so PENDING behind the
# A10-J cube chunks is the expected state: WAIT.  Do not resubmit, do not shrink
# the request, do not drop the qos, do not relocate.  The local H100 stays free.
set -euo pipefail
source /hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_9_marked_multitracer_H0_fagn/scripts/env_a9.sh
A10_DIR=/hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_10_mass_spin_marked_multitracer
echo "[a10] host=$(hostname) job=${SLURM_JOB_ID:-none} args='${A10_DECOMP_ARGS:-<defaults>}'"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
exec "${A9_PY}" "${A10_DIR}/scripts/a10_event_decomposition.py" ${A10_DECOMP_ARGS:---f 0.350 --dmu_chi 0.1075 --dmu_G 5.0 --tag prov_f0p350_mu0p1075_G5}
