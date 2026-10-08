#!/bin/bash
#SBATCH --job-name=a13profile
#SBATCH --account=phy220048p
#SBATCH --partition=TWIG-GPU
#SBATCH --qos=twig
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100-40:1
#SBATCH --mem=100G
#SBATCH --time=01:30:00
#SBATCH --output=/hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_13_joint_shared_widths/logs/%x_%j.out
#SBATCH --error=/hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_13_joint_shared_widths/logs/%x_%j.err
# GPU profile of the per-call likelihood, configs A (all #52 defaults) and B (defaults, per_sample pairing).
#   sbatch --export=ALL,A13_CORE=pr57 scripts/submit_a13_core_profile_twig.sbatch
set -euo pipefail
cd /hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_13_joint_shared_widths/scripts
export PYTHONPATH= XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 JAX_PLATFORMS=cuda OMP_NUM_THREADS=4
PY=/hildafs/projects/phy230014p/magana/envs/darksirens-core-${A13_CORE:?}/bin/python
D=/hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_13_joint_shared_widths/diagnostics/profile_${A13_CORE}
mkdir -p $D
echo "[a13profile] host=$(hostname) job=${SLURM_JOB_ID:-none} core=${A13_CORE}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
for c in A B; do
  XLA_FLAGS="--xla_dump_to=$D/hlo_$c --xla_dump_hlo_as_text" $PY a13_core_profile.py --config $c || echo "[a13profile] config $c FAILED"
done
du -sh $D/* || true
