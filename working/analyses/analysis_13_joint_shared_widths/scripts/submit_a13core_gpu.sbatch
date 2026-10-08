#!/bin/bash
#SBATCH --job-name=a13core
#SBATCH --account=phy220048p
#SBATCH --partition=RITA-GPU
#SBATCH --qos=rita
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100-80:1
#SBATCH --mem=120G
#SBATCH --time=48:00:00
#SBATCH --output=/hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_13_joint_shared_widths/logs/%x_%j.out
#SBATCH --error=/hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_13_joint_shared_widths/logs/%x_%j.err
# A13 on darksirens-core bf58aa6 (a13_core_sampler.py). Resubmit identically to resume.
#   sbatch --export=ALL,SEED=1 scripts/submit_a13core_gpu.sbatch
#   rslice from the checkpoint on (2026-10-05): sbatch --export=ALL,SEED=1,RESUME_SAMPLE=rslice ...
#   guarded rerun (2026-10-07): sbatch --export=ALL,SEED=1,A13_CORE=e7c3007,A13_ARGS="--sample rslice --max-var 20" ...
set -euo pipefail
cd /hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_13_joint_shared_widths/scripts
export PYTHONPATH= XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 JAX_PLATFORMS=cuda OMP_NUM_THREADS=4
echo "[a13core] host=$(hostname) job=${SLURM_JOB_ID:-none} seed=${SEED:?} core=${A13_CORE:-bf58aa6} args=${A13_ARGS:-}"
export A13_CORE=${A13_CORE:-bf58aa6}
exec /hildafs/projects/phy230014p/magana/envs/darksirens-core-${A13_CORE}/bin/python a13_core_sampler.py --seed $SEED ${RESUME_SAMPLE:+--resume-sample $RESUME_SAMPLE} ${A13_ARGS:-}
