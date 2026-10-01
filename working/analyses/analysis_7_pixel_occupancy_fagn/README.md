# Analysis 7 — pixel occupancy

**Status:** complete; verdict PARTIAL (per-pixel scales with occupancy, the selection control is not flat). Held out of the paper.

## Question

Only the pixelisation changes (nside 16, 32, 64 at m<18), under both per-pixel and selection completeness: does the estimator's f_AGN offset scale with counts per pixel?

## Provenance

Ran as `a7` in the selection-mode redo campaign (Jetstream2, `../selection_redo/scripts/run_campaign.sh`). Moved here from `../selection_redo/a7/` on 2026-10-01; its two own scripts (`scripts/make_a7_queue.py`, `scripts/analyze_a7.py`) moved with it. It is not the Analysis 7 scoped on 2026-08-08 (seed-101 replication), which was held by the owner and never run.

## Layout

| path | contents |
|---|---|
| `REPORT.md` | results |
| `scripts/` | analysis and figure scripts |
| `results/` | result files (bulk ones git-ignored) |
| `figs/` | rendered figures, PDF + PNG |
| `logs/` | job logs (git-ignored) |
| `queue/` | task queues |
