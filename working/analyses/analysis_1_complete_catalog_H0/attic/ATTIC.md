# attic/ — the superseded eras

**2026-08-01.** These directories were collected under `attic/` when the analysis was
reorganised. Nothing was deleted; only the parent directory changed, so every name
below is exactly the name it had at the top level. The closed *diagnostic* campaign
is a different thing and lives under `diagnostics/` — see `diagnostics/INDEX.md`.

## The `dark_sirens_complete` / float32 era

`results_dsc_attic/`, `figs_dsc_attic/`, `logs_dsc_attic/` — outputs of the era in
which this analysis ran the dedicated complete-catalog likelihood on the float32
dataset. Superseded as analysis of record by the `dark_sirens` version. This is also
where the **probe 1–4 numeric outputs and figures** live (`probe1_*`, `probe2_*`,
`probe3_*`, `probe4*`), together with the jackknife blocks (`jk_*`) and the
`range_agn_*` scans; `PROBES.md` and `README.md` cite them from here. The probe
*code* moved to `diagnostics/probes/scripts/`.

* `scripts_superseded/` — the three run scripts as they stood in that era (moved out
  of `results_dsc_attic/` in the 2026-08-01 reorganisation). `run_scans.sh` and
  `run_guard_diag.sh` were rewritten in place for the new estimator
  (`--universe_model dark_sirens --log10n0 -24` in place of
  `--universe_model dark_sirens_complete`); `run_matched_control.sh` was folded into
  `run_scans.sh`, which now runs all six configurations.
* `vs_dsc_attic.json` — this analysis against that era, cell by cell, written by
  `scripts_onhold/compare_dsc_attic.py`.

## The pre-`(b2)`/`(c2)` scans of record

`results_prefix2/`, `figs_prefix2/` — the scans of record of 2026-07-31, before the
two generator fixes of `CLOSURE.md` §1. They are the "before" arm of
`fig_closure_after_fix`.

## The v2 post-`(b2)`/`(c2)` scans of record

`results_v2postfix/`, `figs_v2postfix/` — the scans of record after those two fixes
and before the v3 redesign. They are the "before" arm of the record figure
`figs/fig_closure_v3`, which `scripts/fig_closure_after_fix.py` reads from here
(`--before_dir attic/results_v2postfix`).

`logs_v2postfix/` — the pre-v3 stdout and SLURM logs of the record tools
(`build_single_tracer`, `kde_window_check`, `make_figures`, the v2 scan and
seed-control jobs). The v3 logs of the same tools are in `logs/`.

## Studies on hold

`scripts_onhold/` — `compare_dsc_attic.py` (this analysis against the
`dark_sirens_complete` era), `run_closure_diag.sh` (the closure/jackknife driver,
still written for `dark_sirens_complete`) and `build_event_blocks.py` (the
disjoint-block noise study). All three read archived inputs.

`data_derived_v2/blocks/` — the disjoint event blocks of the closure and
disjoint-block studies. They were built from the **float32** dataset — every event
redshift, distance and PE sample in them is ~2e-8 relative away from the dataset now
on disk — and were never rebuilt. They must be rebuilt before those studies are
rerun. `data_derived/events_{gal,agn}_hosted.h5`, the only subsets the analysis of
record reads, are current (rebuilt from the v3 events, 705 GAL / 295 AGN).

## Cruft

`figs_ipynb_checkpoints/` — Jupyter's `.ipynb_checkpoints` copies of figures, moved
out of `figs/` so that directory holds only the four record figures.
`scripts_pycache/` — the interpreter's `__pycache__` from `scripts/`.
