# Selection-mode redo campaign (2026-08-12) and its follow-ups

Not an analysis: the campaign infrastructure that reran Analyses 3–7 with `c_mode=selection`
after the per-pixel completeness estimator was found wrong (the per-pixel versions are in
`../../archive/`). The results of each rerun live in its own analysis directory:

| campaign label | analysis directory |
|---|---|
| a3 | `../analysis_3_incomplete_catalog_H0_fagn/` |
| a4 | `../analysis_4_density_anchoring_H0_fagn/` |
| a5 | `../analysis_5_free_anchors_H0_fagn/` |
| a6 | `../analysis_6_relative_completeness_H0_fagn/` |
| a7 | `../analysis_7_pixel_occupancy_fagn/` (moved out of `a7/` on 2026-10-01) |

What stays here:

- `scripts/`: the shared drivers (`run_campaign.sh`, `make_queues.py`, `scan_h0f.py`,
  `sample_4d.py`, `heal.sh`, `pull_results.sh`, the Schechter fits, `lf_constants.py`).
- `a3/ … a6/queue/`: the task queues the campaign ran (records). `run_campaign.sh` and
  `heal.sh` still name `a7`, whose queue is now `../analysis_7_pixel_occupancy_fagn/queue/`.
- `FU_REPORT.md`, `fu_probes/`, `fu_seed101/`, `fu_seed102/`: the 2026-08-24 follow-ups (zero-density
  probe, KDE bandwidth, the a5 headline at seeds 101 and 102). The paper's number build reads
  `fu_seed101/` and `fu_seed102/` at these paths.
