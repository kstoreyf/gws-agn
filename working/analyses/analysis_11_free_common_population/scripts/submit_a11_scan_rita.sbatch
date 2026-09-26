#!/bin/bash
#SBATCH --account=phy220048p
#SBATCH --partition=RITA-GPU
#SBATCH --qos=rita
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100-80:1
#SBATCH --mem=100G
#SBATCH --time=10:00:00
#SBATCH --output=/hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_11_free_common_population/logs/%x_%A_%a.out
#SBATCH --error=/hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_11_free_common_population/logs/%x_%A_%a.err
#
# 11A / 11B grids, one chunk per array task:
#   sbatch --job-name=a11_11A --array=0-3%1 --export=ALL,ARM=11A,N_CHUNKS=4 scripts/submit_a11_scan_rita.sbatch
# Resumable from diagnostics/_a11_{ARM}_c{k}of{n}.jsonl.  A 'TaskProlog failed'
# death within seconds is a rita node fault: resubmit that index once, identical.
set -euo pipefail
source /hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_9_marked_multitracer_H0_fagn/scripts/env_a9.sh
cd /hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_11_free_common_population
echo "[a11_scan] host=$(hostname) job=${SLURM_JOB_ID:-none} arm=${ARM:?} chunk=${SLURM_ARRAY_TASK_ID:-0}/${N_CHUNKS:?}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
exec "${A9_PY}" scripts/a11_scan.py --stage scan --arm "${ARM}" --chunk "${SLURM_ARRAY_TASK_ID:-0}" --n_chunks "${N_CHUNKS}"
