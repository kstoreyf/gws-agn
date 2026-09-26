#!/bin/bash
#SBATCH --job-name=c10_ctl
#SBATCH --account=phy220048p
#SBATCH --partition=RITA-GPU
#SBATCH --qos=rita
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100-80:1
#SBATCH --mem=100G
#SBATCH --time=05:00:00
#SBATCH --output=/hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_10_mass_spin_marked_multitracer/logs/c10_ctl_%x_%j.out
#SBATCH --error=/hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_10_mass_spin_marked_multitracer/logs/c10_ctl_%x_%j.err
#
# C10-3 single-tracer control, one arm per job (B5M, B0M, B5U):
#   sbatch --job-name=c10_ctl_B5M --export=ALL,ARM=B5M scripts/submit_c10_control_rita.sbatch
# 42 H0 nodes on [58, 78] (+67.74) x 41 f = 1,722 cells; [GAL,GAL] runs at ~6 s/cell
# (the GAL survey is loaded twice), so ~2.9 GPU-h per arm.
# Resumes from its checkpoint; a 'TaskProlog failed' death is a node fault:
# resubmit once, identical.
set -euo pipefail
source /hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_9_marked_multitracer_H0_fagn/scripts/env_a9.sh
A10_DIR=/hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_10_mass_spin_marked_multitracer
echo "[c10_ctl] host=$(hostname) job=${SLURM_JOB_ID:-none} arm=${ARM:?ARM}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
exec "${A9_PY}" "${A10_DIR}/scripts/c10_control.py" --stage run --arm "${ARM}" ${C10_CTL_EXTRA:-}
