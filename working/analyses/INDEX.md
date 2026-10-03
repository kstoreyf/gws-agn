# Analyses — index and conventions

Updated 2026-10-01. `EXECUTIVE_SUMMARY.md` holds the science summary. This file records where
things live and the rules the directories follow.

## The ladder

| # | directory | question | status |
|---|---|---|---|
| 0 | `analysis_0_pure_tracer_H0/` | H0 from all-galaxy and all-AGN host sets, equal N | complete |
| 1 | `analysis_1_complete_catalog_H0/` | each complete catalog's H0 on its own | complete |
| 2 | `analysis_2_complete_catalog_H0_fagn/` | (H0, f_AGN) together on complete catalogs, K = 2 mixture | complete |
| 3 | `analysis_3_incomplete_catalog_H0_fagn/` | magnitude-limited catalogs, densities anchored at truth | complete (selection mode) |
| 4 | `analysis_4_density_anchoring_H0_fagn/` | a mis-anchored AGN completion density | complete (selection mode) |
| 5 | `analysis_5_free_anchors_H0_fagn/` | both completion densities free with H0 and f_AGN | complete (selection mode) |
| 6 | `analysis_6_relative_completeness_H0_fagn/` | GAL and AGN depths varied independently | complete (selection mode) |
| 7 | `analysis_7_pixel_occupancy_fagn/` | does the f_AGN offset scale with counts per pixel? | complete; held out of the paper |
| 8 | `analysis_8_spin_marked_fagn_fixed_H0/` | spin mark: (f_AGN, Δμ_χ) at fixed H0 | complete |
| 9 | `analysis_9_spin_marked_H0_fagn/` | spin mark with H0 free; the event-routing mechanism | complete |
| 10 | `analysis_10_mass_spin_marked_multitracer/` | mass mark on top of the spin mark, fixed then free H0 | complete |
| 11 | `analysis_11_free_common_population/` | reference population inferred with the offsets; 11D H0 free | complete through 11D and §21 |
| 12 | `analysis_12_shared_width_robustness/` | one shared σ_G or σ_χ freed on top of 11D | complete |
| 13 | `analysis_13_joint_shared_widths/` | both shared widths free together (the recommended calibration model) | restarting on darksirens-core |

Other directories here:

- `selection_redo/`: the 2026-08-12 selection-mode campaign that reran Analyses 3–7 (shared
  drivers, the queues it ran, the 2026-08-24 follow-ups `FU_REPORT.md` and `fu_*`). Not an
  analysis; its `README.md` maps campaign labels to analysis directories.
- `experiments/`: the live part of the 2026-07 pre-campaign ladder (baseline, matched mock,
  two-tracer depth and seeds, the 4-D estimator recheck, model equivalence). Its per-pixel
  incomplete-catalog experiments are in `../archive/experiments/`.
- `../archive/`: superseded work (below). `../campaign_100seeds/`: the 100-seed campaign.

## Code base (owner, 2026-10-02)

New runs use **darksirens-core, at a pinned commit, with every population value (γ included) and
every catalogue-evaluation option set explicitly**. Finished analyses keep their results and the
legacy commit they ran on (2b86a2d, 0c5b3db or af896ca). Equivalence evidence:
`experiments/experiment_darksirens_core_equivalence/`.

## Data location (2026-10-02)

The mock datasets moved from phy220048p to `/hildafs/projects/phy230054p/magana/gws-agn-data-v3`
(and `gws-agn-data`); `working/data/seedNNN` symlinks point there. Run records written before the
move were rewritten to the new path on 2026-10-03.

## Directory structure

Every analysis directory has:

| path | contents |
|---|---|
| `README.md` | the question, status, inputs and layout, one screen |
| `REPORT.md` | the results, with numbers and figure references |
| `scripts/` | everything that produced the results, including a deterministic figure script |
| `results/` | result files; bulk h5/npz are git-ignored |
| `figs/` | rendered result figures, PDF + PNG |
| `logs/` | job logs, git-ignored |

and, where the work used them, `STATE.md` (running state and resume point), `GATES.md`
(criteria registered before results), `diagnostics/` (closure and check records) and `queue/`
(task lists and sampler checkpoints).

## The archive rule

Work whose method was found wrong is moved to `../archive/`, under its original directory name,
when it is rerun with the corrected method. The rerun takes the live directory and its REPORT
names what it supersedes. The per-pixel completeness estimator (darksirens `c_mode=per_pixel`,
the legacy default) is the case so far: Analyses 3–6 ran on it, were archived, and were rerun
with `c_mode=selection` (`selection_redo/`).

The pre-campaign experiments that ran on it with incomplete catalogs (2026-07, before `c_mode`
existed) and were never rerun are archived too: `../archive/experiments/experiment_completeness_anchored/`,
`experiment_completeness_free/` and `experiment_twotracer_incomplete/` (moved 2026-10-01; the frozen
`../report/` build reads them there). `experiments/experiment_dsmaster_4d_recheck/` compares the two
estimators on purpose and stays live.
Analyses 0–2 use complete catalogs, so the completeness estimator does not enter.

## Renames and moves (2026-10-01)

| old | new |
|---|---|
| `analysis_8_marked_multitracer_H0_fagn/` | `analysis_8_spin_marked_fagn_fixed_H0/` |
| `analysis_9_marked_multitracer_H0_fagn/` | `analysis_9_spin_marked_H0_fagn/` |
| `selection_redo/a7/` (+ `scripts/analyze_a7.py`, `make_a7_queue.py`) | `analysis_7_pixel_occupancy_fagn/` |
| Analysis 12 inside `analysis_11_free_common_population/` | `analysis_12_shared_width_robustness/` |
| `analysis_{0,1,2}_*/README.md` (held the results) | `REPORT.md`, with a new short `README.md` |
| `experiments/experiment_completeness_anchored/`, `experiment_completeness_free/`, `experiment_twotracer_incomplete/` | `../archive/experiments/` (same names; per-pixel era) |

Scripts, sbatch files and markdown were updated to the new paths. Recorded JSON provenance
(diagnostics and result files written before the rename) keeps the paths that were true when it
was written, so an old name inside a JSON record refers to the directory in the left column.
