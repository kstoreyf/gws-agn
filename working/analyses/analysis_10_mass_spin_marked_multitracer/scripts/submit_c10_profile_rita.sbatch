#!/bin/bash
#SBATCH --job-name=c10_profile
#SBATCH --account=phy220048p
#SBATCH --partition=RITA-GPU
#SBATCH --qos=rita
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100-80:1
#SBATCH --mem=100G
#SBATCH --time=00:15:00
#SBATCH --output=/hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_10_mass_spin_marked_multitracer/logs/c10_profile_%j.out
#SBATCH --error=/hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_10_mass_spin_marked_multitracer/logs/c10_profile_%j.err
#
# C10-P -- the H0 profile pre-flight for the cosmology stage: does [63, 76]
# still bracket the H0 peak once the mass mark is live at the fixed-H0 joint
# MAP (f=0.275, dmu_chi=+0.115, dmu_G=+5.0)?  28 H0 nodes (A9's J9 window,
# bitwise-asserted against results/j9_marked.h5) x 2 points (marked, spatial)
# = 56 cells, ~3 min at the measured 3.0143 s/eval plus one build.  ONE
# worker, no array.  Auto-assembles diagnostics/c10_profile.json when the
# checkpoint is complete.
#
#   sbatch scripts/submit_c10_profile_rita.sbatch
#
# --qos=rita contributes ZERO to the priority number, so PENDING behind other
# users is normal: WAIT.  Do not resubmit, shrink or relocate.  The checkpoint
# is keyed on H0 and resumes after an eviction, so a kill costs at most one
# H0 node (two cells).  A 'TaskProlog failed' death is a node fault: resubmit
# once, identical.
set -euo pipefail
source /hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_9_marked_multitracer_H0_fagn/scripts/env_a9.sh
A10_DIR=/hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_10_mass_spin_marked_multitracer
echo "[c10] host=$(hostname) job=${SLURM_JOB_ID:-none} stage=profile"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
exec "${A9_PY}" "${A10_DIR}/scripts/c10_scan.py" --stage profile
