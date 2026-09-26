#!/bin/bash
#SBATCH --job-name=c10_route
#SBATCH --account=phy220048p
#SBATCH --partition=RITA-GPU
#SBATCH --qos=rita
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100-80:1
#SBATCH --mem=100G
#SBATCH --time=02:30:00
#SBATCH --output=/hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_10_mass_spin_marked_multitracer/logs/c10_event_routing_%j.out
#SBATCH --error=/hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_10_mass_spin_marked_multitracer/logs/c10_event_routing_%j.err
#
# C10-2: the four nested per-event models at 12 H0 nodes (P1's pin), each node
# verified against live evaluations and the recorded P1 / C10-S cells, then the
# CPU readout.  ~12 x (4 passes x ~40 s + 4 live evals + capture) ~ 45 min.
# A 'TaskProlog failed' death is a node fault: resubmit once, identical.
set -euo pipefail
source /hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_9_marked_multitracer_H0_fagn/scripts/env_a9.sh
A10_DIR=/hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_10_mass_spin_marked_multitracer
echo "[c10_route] host=$(hostname) job=${SLURM_JOB_ID:-none}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
exec "${A9_PY}" "${A10_DIR}/scripts/c10_event_routing.py"
