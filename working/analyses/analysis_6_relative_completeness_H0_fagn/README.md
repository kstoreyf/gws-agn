# Analysis 6 — relative completeness

**Status:** complete (selection mode); supersedes `../../archive/analysis_6_relative_completeness_H0_fagn/` (per-pixel).

## Question

Galaxy and AGN depths varied independently over a grid, to test whether the f_AGN offset follows the completeness of one tracer relative to the other.

## Provenance

Run in the selection-mode redo campaign (`c_mode=selection`, darksirens `0c5b3db`, on Jetstream2) by the shared drivers in `../selection_redo/scripts/` (`run_campaign.sh`, `make_queues.py`, `scan_h0f.py`, `sample_4d.py`); this directory holds the results, the figures and `scripts/make_figures.py`.

## Layout

| path | contents |
|---|---|
| `REPORT.md` | results |
| `scripts/` | analysis and figure scripts |
| `results/` | result files (bulk ones git-ignored) |
| `figs/` | rendered figures, PDF + PNG |
| `logs/` | job logs (git-ignored) |
| `queue/` | task queues |
