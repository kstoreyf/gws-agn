# NOTE TO THE RERUN AGENT (from the orchestrator, 2026-07-30)

Slurm is hopeless today (owner's call): do NOT resubmit submit_seeds_fix.sbatch.
The 12-seed GPU stage (guard_fix/fscan_fix/joint_fix per seed, identical recipe to
one_realisation_fix.sh) is ALREADY RUNNING on a Jetstream2 H100 and its results are
rsynced into results/ here every ~4 minutes (seeds 7303/7304 already complete).
Your job: WAIT for all 12 seeds' {guard,fscan,joint}_fix_s73xx.json to appear in
results/, then run scripts/aggregate_seeds_fix.py + finish_fix_run.sh and report.
