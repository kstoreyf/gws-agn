# gw_agn × darksirens — rerun on master @ `2b86a2d` (2026-07-29)

Third run in the reproduction campaign. Predecessors, both under `working/`:

| dir | darksirens code | outcome |
|---|---|---|
| `gw_agn` | none (bespoke estimator) | the reference result: α_AGN recoverable, ρ(H0,α)≈0 |
| `gw_agn_darksirens` | branch @ `d387b4f` (PR #195 K-mixture) | per-tracer H0 reproduced; K=2 conditional mixture rails to f=1 — estimand diagnosis |
| `gw_agn_darksirens_fixed` | master @ `8eae3ea` (PR #212 field stack) | ladder recovered: dsf medians {0.014, 0.322, 0.687, 0.974} |
| **this dir** | master @ `2b86a2d` | 295 commits later; see `RESULTS.md` and `GUARD_AUDIT.md` |

`data/` symlinks to `../gw_agn_darksirens/data` — inputs are bit-identical across
all three darksirens runs (events, distance/sky PE, catalogs, injections).

## Layout

```
scripts/env.sh                    shared env: pinned worktree, DARKSIRENS_ZMAX=1.5, paths
scripts/scan_darksirens.py        scan driver (logL grids by module import, no sampler)
scripts/diag_variance_guard.py    instruments the total-variance guard for one cell
scripts/run_guard_audit.sh        one probe per configuration -> results/guard_audit/
scripts/aggregate_guard_audit.py  -> results/guard_audit_summary.json + GUARD_AUDIT.md
scripts/run_production_legacy.sh  ARM L: full scans, variance criterion made inert
scripts/run_production_default.sh ARM G: scans at master's default budget (1.0)
scripts/build_comparison.py       figures + results/comparison_summary.json
results/  logs/  figs/            outputs; tags are prefixed L_ (arm L) or G_ (arm G)
```

## Reproducing

```bash
git -C /hildafs/projects/phy230014p/magana/src/darksirens \
    worktree add --detach /path/to/wt-2b86a2d 2b86a2d
export DARKSIRENS_WT=/path/to/wt-2b86a2d          # scripts/env.sh reads this
cd scripts
./run_guard_audit.sh          # ~10 min, 41 probes
./run_production_legacy.sh    # ~3 h on one A100
./run_production_default.sh   # ~15 min
python build_comparison.py
```

Conda env `jax` (editable darksirens install; `PYTHONPATH` overrides it with the
pin). A100; `XLA_PYTHON_CLIENT_PREALLOCATE=false`.

## What changed in the driver vs the `8eae3ea` run

1. `build_parameter_space` now derives the sampled survey block from a declarative
   registry (PR #308) and takes `use_lss` / `lss_completion_active` /
   `mark_names_by_catalog`. These are threaded exactly as `darksirens.cli.inference`
   does, so `b_miss` is no longer sampled as a phantom dimension when `use_LSS`
   is off. The resulting label sets are unchanged for `dark_sirens_complete`.
2. New `--max_likelihood_variance` flag, and a fully-rejected scan now writes a
   `all_cells_rejected` summary instead of raising on an all-NaN argmax.
