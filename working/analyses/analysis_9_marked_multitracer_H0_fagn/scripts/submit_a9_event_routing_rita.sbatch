#!/bin/bash
#SBATCH --job-name=a9_routing
#SBATCH --account=phy220048p
#SBATCH --partition=RITA-GPU
#SBATCH --qos=rita
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100-80:1
#SBATCH --mem=100G
#SBATCH --time=02:00:00
#SBATCH --output=/hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_9_marked_multitracer_H0_fagn/logs/a9_event_routing_%j.out
#SBATCH --error=/hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_9_marked_multitracer_H0_fagn/logs/a9_event_routing_%j.err
#
# Mechanism follow-up part B: the event-level routing diagnostic and the
# per-event dlnL_i(H0) profile at the shared S9/J9 MAP node.
#
#   sbatch scripts/submit_a9_event_routing_rita.sbatch
#
# Per-event quantities are not stored in the cubes, so this is the ONE short GPU
# job the follow-up needs: ten H0 nodes x (one operand capture + one 1000-event
# pass in blocks of 25).  At the measured 3.02 s/eval and a 29 s build the whole
# thing is well under one GPU-hour; --time=02:00:00 is the ceiling, not the
# estimate.
#
# The request is the Analysis-9 request of record and is never shrunk or
# relocated.  --qos=rita contributes ZERO to the priority number, so PENDING
# behind another user is normal: WAIT.  If the step dies within seconds with
# `slurmstepd: error: TaskProlog failed status=1`, that is a node-side prolog
# fault before any of our code runs (documented twice in STATE.md); resubmit
# ONCE with the request byte-identical.
set -euo pipefail
source /hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_9_marked_multitracer_H0_fagn/scripts/env_a9.sh
echo "[a9] host=$(hostname) job=${SLURM_JOB_ID:-none} stage=event_routing"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
exec "${A9_PY}" "${A9_DIR}/scripts/a9_event_routing.py" ${A9_EXTRA:-}
