# Analysis 5 — free completion densities

**Status:** complete (selection mode); supersedes `../../archive/analysis_5_free_anchors_H0_fagn/` (per-pixel).

## Question

Both completion densities free under flat priors, sampled jointly with H0 and f_AGN (dynesty, nlive 1000) at every rung: whether the data alone identify the densities.

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
