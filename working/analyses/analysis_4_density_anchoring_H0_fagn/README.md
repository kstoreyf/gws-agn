# Analysis 4 — density anchoring

**Status:** complete (selection mode); supersedes `../../archive/analysis_4_density_anchoring_H0_fagn/` (per-pixel).

## Question

The AGN completion density mis-anchored by factors 0.5–2.0 at every rung, with the galaxy anchor held at truth: where the error lands, plus the oracle probe (GAL m<18 × AGN complete).

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
