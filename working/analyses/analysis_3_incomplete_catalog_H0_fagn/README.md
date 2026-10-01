# Analysis 3 — incomplete catalogs, (H0, f_AGN)

**Status:** complete (selection mode); supersedes `../../archive/analysis_3_incomplete_catalog_H0_fagn/` (per-pixel).

## Question

Both host catalogs magnitude-limited (complete, m<21 … m<18) with the completion densities anchored at the mock's truths: what the incompleteness costs (H0, f_AGN).

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
