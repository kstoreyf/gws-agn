#!/bin/bash
#SBATCH --job-name=c10_mech
#SBATCH --account=phy220048p
#SBATCH --partition=RITA-GPU
#SBATCH --qos=rita
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100-80:1
#SBATCH --mem=100G
#SBATCH --time=02:00:00
#SBATCH --output=/hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_10_mass_spin_marked_multitracer/logs/c10_mech_%A_%a.out
#SBATCH --error=/hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_10_mass_spin_marked_multitracer/logs/c10_mech_%A_%a.err
#
# C10-mech -- the section-21 mechanism arms, 2-D (H0, f_AGN) on the 28
# (default h0_window) x 41 (full F_GRID) lattice, marks PINNED at
# --pin_dmu_chi/--pin_dmu_G (default the fixed-H0 joint MAP, +0.115/+5.0).
# One arm per invocation via ARM={P1,I0,I1}: 1,148 cells/arm, ~0.96 GPU-h at
# the measured 3.0143 s/eval; --time=02:00:00 covers that plus the ~23 s
# build and ~11 s first (compiling) evaluation with a large margin.  P0
# (marks 0, [GAL, AGN]) is a slice of results/c10_arm_S.h5 and is NOT
# computed here -- run C10-S (and s_assemble) first.
#
# DO NOT SUBMIT until the owner has reviewed C10-J (or accepted the fixed-H0
# MAP default for the pin point) and confirmed the H0 window.
#
#   sbatch --export=ALL,ARM=P1,C10_EXTRA="--h0_window 63 76 --pin_dmu_chi 0.115 --pin_dmu_G 5.0" \
#     scripts/submit_c10_mech_rita.sbatch
#   sbatch --export=ALL,ARM=I0,C10_EXTRA="--h0_window 63 76 --pin_dmu_chi 0.115 --pin_dmu_G 5.0" \
#     scripts/submit_c10_mech_rita.sbatch
#   sbatch --export=ALL,ARM=I1,C10_EXTRA="--h0_window 63 76 --pin_dmu_chi 0.115 --pin_dmu_G 5.0" \
#     scripts/submit_c10_mech_rita.sbatch
#
# --qos=rita contributes ZERO to the priority number, so PENDING behind other
# users is normal: WAIT.  Do not resubmit, shrink or relocate.  Each arm's
# worker appends to its OWN checkpoint (_c10_mech_<ARM>.jsonl) and resumes
# after an eviction, so a kill costs at most one 41-cell row.  A
# 'TaskProlog failed' death is a node fault: resubmit once, identical.
set -euo pipefail
source /hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_9_marked_multitracer_H0_fagn/scripts/env_a9.sh
A10_DIR=/hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_10_mass_spin_marked_multitracer
if [ -z "${ARM:-}" ]; then
  echo "[fatal] pass --export=ALL,ARM={P1,I0,I1}" >&2
  exit 1
fi
echo "[c10] host=$(hostname) job=${SLURM_JOB_ID:-none} arm=${ARM}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
exec "${A9_PY}" "${A10_DIR}/scripts/c10_scan.py" --stage mech --arm "${ARM}" ${C10_EXTRA:-}
