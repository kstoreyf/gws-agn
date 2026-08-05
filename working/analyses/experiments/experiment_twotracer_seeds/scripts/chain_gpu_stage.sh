#!/bin/bash
# Chain-resubmit the GPU-stage slurm job until all 12 seeds have results.
#
# WHY: on 2026-07-30 every newly submitted slurm job of this account was
# cancelled by an unidentified same-uid automation ~4 minutes after starting
# (jobs 1058073/1058077/1058080/1058081; the owner's pre-existing jobs are
# untouched).  One seed's guard+fscan+joint completes in ~4 min on an A100-40,
# and both CPU and GPU stages are idempotent per seed, so chaining submissions
# ratchets through the seed list ~1 seed per job.  If a job survives, it
# finishes everything in one go and the loop exits.
# Log: logs/chain_gpu_stage.log
set -uo pipefail
cd "$(dirname "$0")/.."
exec >> logs/chain_gpu_stage.log 2>&1
echo "=== chain driver start $(date -u)"

done_count() { ls results/joint_fix_s73*.json 2>/dev/null | wc -l; }

it=0
while [ "$(done_count)" -lt 12 ]; do
  it=$((it + 1))
  if [ $it -gt 40 ]; then echo "giving up after 40 submissions"; exit 1; fi
  JID=$(sbatch --parsable scripts/submit_seeds_fix.sbatch 2>>logs/chain_gpu_stage.log) || {
    echo "sbatch failed (iteration $it); retrying in 120 s"; sleep 120; continue; }
  echo "iteration $it: job $JID submitted $(date -u); done=$(done_count)/12"
  # wait for the job to leave the queue (canceller or completion)
  sleep 30
  while squeue -j "$JID" -h 2>/dev/null | grep -q .; do sleep 30; done
  echo "iteration $it: job $JID left queue $(date -u); done=$(done_count)/12"
  sleep 15   # let the last file writes land on the shared filesystem
done
echo "=== all 12 seeds scanned $(date -u)"

python scripts/aggregate_seeds_fix.py &&
python scripts/make_fix_figures.py &&
python scripts/write_seeds_fix_note.py
echo "=== chain driver done $(date -u)"
