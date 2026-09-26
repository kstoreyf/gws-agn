#!/bin/bash
#SBATCH --job-name=c10gen
#SBATCH --account=phy230014p
#SBATCH --partition=RM
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#
# C10-3 single-tracer control mocks (and the default-path bitwise control).
# Same wrapper as logs/run_events.sbatch (the production-mock run), plus
# extra generator flags.  Submit with --nodelist=r008, the production mock's
# node, so the bitwise control is free of cross-node pow-kernel drift.
#
#   sbatch --nodelist=r008 scripts/run_events_c10.sbatch SUFFIX DMU_G [EXTRA FLAGS...]
set -euo pipefail
SFX=${1:?suffix}
DMUG=${2:?dmu_G_agn}
shift 2
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/hildafs/projects/phy230014p/magana/src/darksirens-a8
export JAX_PLATFORMS=cpu
PY=/hildafs/home/magana/tmp_ondemand_hildafs_phy230014p_symlink/magana/.conda/envs/jax/bin/python
echo "=== host $(hostname) job ${SLURM_JOB_ID} suffix ${SFX} dmu_G_agn ${DMUG} extra [$*] start $(date -u +%FT%TZ) ==="
T0=$(date +%s)
cd /hildafs/projects/phy230014p/magana/gws-agn/working/data
$PY generate_dataset.py --seed 100 --stage events \
    --darksirens /hildafs/projects/phy230014p/magana/src/darksirens-a8 \
    --dmu_chi_agn 0.10 --dmu_G_agn "$DMUG" --events_suffix "$SFX" "$@"
T1=$(date +%s)
echo "=== WALL_SECONDS $((T1-T0)) host $(hostname) end $(date -u +%FT%TZ) ==="
