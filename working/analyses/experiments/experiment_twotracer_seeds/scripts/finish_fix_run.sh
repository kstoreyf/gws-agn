#!/bin/bash
# Autonomous finisher for the sigma_ang-fix 12-seed rerun.
#  1. wait for the locally-running CPU stage (12 injection validation jsons);
#  2. submit the slurm job (its CPU stage skips completed seeds; GPU stage runs
#     guard + f-scan + joint per seed, then aggregates);
#  3. when seeds_summary_fix.json appears (or the job ends), aggregate if
#     needed, then write figures and SEEDS_FIX.md.
# Idempotent and safe to re-run.  Log: logs/finish_fix_run.log
set -uo pipefail
cd "$(dirname "$0")/.."
exec >> logs/finish_fix_run.log 2>&1
echo "=== finisher start $(date -u)"

# -- 1. wait for CPU stage ---------------------------------------------------
while [ "$(ls results/inj_fix_s73*_validation.json 2>/dev/null | wc -l)" -lt 12 ]; do
  sleep 60
done
echo "=== CPU stage complete $(date -u)"

# quick integrity gate: every seed must have events + injections readable
python - <<'EOF' || { echo "integrity check FAILED"; exit 1; }
import h5py
for s in range(7301, 7313):
    with h5py.File(f"data_derived/s{s}_fix/twotracer_gw_events.h5") as f:
        assert f["dL"].shape[0] == 400000, s
        assert f.attrs["detection_data"] == "observed", s
    with h5py.File(f"data_derived/s{s}_fix/injections_targeted_k2.h5") as f:
        assert f.attrs["detection_data"] == "observed", s
        assert f.attrs["ndraw"] == 120000000, s
print("integrity OK")
EOF

# -- 2. slurm job for the GPU stage -----------------------------------------
if [ ! -f results/seeds_summary_fix.json ]; then
  JID=$(sbatch --parsable scripts/submit_seeds_fix.sbatch)
  echo "submitted job $JID $(date -u)"
  # wait until the job leaves the queue
  while squeue -j "$JID" -h 2>/dev/null | grep -q .; do sleep 120; done
  echo "job $JID left queue $(date -u)"
fi

# -- 3. aggregate + figures + note -------------------------------------------
if [ ! -f results/seeds_summary_fix.json ]; then
  python scripts/aggregate_seeds_fix.py || exit 1
fi
python scripts/make_fix_figures.py
python scripts/write_seeds_fix_note.py
echo "=== finisher done $(date -u)"
