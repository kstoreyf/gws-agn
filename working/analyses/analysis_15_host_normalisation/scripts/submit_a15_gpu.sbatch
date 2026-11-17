#!/bin/bash
#SBATCH --job-name=agn-follow
#SBATCH --account=phy220048p
#SBATCH --partition=RITA-GPU
#SBATCH --qos=rita
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100-80:1
#SBATCH --mem=120G
#SBATCH --time=03:00:00
#SBATCH --output=/hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_15_host_normalisation/logs/%x_%j.out
#SBATCH --error=/hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_15_host_normalisation/logs/%x_%j.err
# Analysis 15: the host-catalog normalisation volume (darksirens-core e7c3007, historical settings).
# Four driver processes, because the redshift-grid end (DARKSIRENS_ZMAX) is fixed at import:
#   1. grid to 5:    Part 1 gates   p1_complete, p1_m21, p1_m18 (against the stored 0c5b3db grids)
#   2. grid to 1.10: (b)            b_complete_x1, b_complete_x2
#   3. grid to 5:    reference, (c) ref_complete_n0m24, c_complete_x2
#   4. grid to 1.10: (d)            d_m18_x1, d_m18_x05, d_m18_x2
# A scan whose gate (p1_complete for b/ref/c, p1_m18 for d) failed is not run. Resubmit identically
# to resume (finished scans in results/scans/*.json are skipped):
#   sbatch scripts/submit_a15_gpu.sbatch
# The aggregation and the figure run at the end on CPU.
set -uo pipefail
A15=/hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_15_host_normalisation
cd $A15/scripts
export PYTHONPATH= XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 JAX_PLATFORMS=cuda OMP_NUM_THREADS=4
START=$(date +%s)
# wall limit (A15_WALL_HOURS, default 3 = the --time above) minus 15 min for aggregation and slack
export A15_DEADLINE=$(( START + ${A15_WALL_HOURS:-3}*3600 - 900 ))
PYC=/hildafs/projects/phy230014p/magana/envs/darksirens-core-e7c3007/bin/python
ZR=1.10   # a15_common.ZMAX_R
echo "[agn-follow] host=$(hostname) job=${SLURM_JOB_ID:-none} start=$(date -u +%FT%TZ)"
block() {
  local z=$1 scans=$2 rc
  echo "[agn-follow] block grid=$z scans=$scans at $(date -u +%FT%TZ)"
  if [ "$z" = R ]; then
    env DARKSIRENS_ZMAX=$ZR $PYC a15_host_norm.py --scans "$scans"; rc=$?
  else
    env -u DARKSIRENS_ZMAX $PYC a15_host_norm.py --scans "$scans"; rc=$?
  fi
  echo "[agn-follow] block grid=$z exit $rc at $(date -u +%FT%TZ)"
}
block 5 p1_complete,p1_m21,p1_m18
block R b_complete_x1,b_complete_x2
block 5 ref_complete_n0m24,c_complete_x2
block R d_m18_x1,d_m18_x05,d_m18_x2
PY=/hildafs/home/magana/tmp_ondemand_hildafs_phy230014p_symlink/magana/.conda/envs/jax/bin/python
for s in aggregate_normalisation.py make_figures.py; do
  JAX_PLATFORMS=cpu $PY $A15/scripts/$s || echo "[agn-follow] $s failed"
done
echo "[agn-follow] end=$(date -u +%FT%TZ)"
