# Experiment — darksirens-core as gws-agn's darksirens (K = 1 equivalence)

**Status (2026-10-02): K = 1 grids running (rita job 1350666). Owner criterion: posterior-level
agreement (KS, 90% widths), bitwise not required. K = 2 / A5 / A8–13 wait for darksirens-work
to port the mixture, c_mode=selection and per-catalogue population blocks into core.**

Arms (each its own process, same rita A100, seed 100, K = 1 GAL m<19, conditional per-pixel,
δ = σ_kde = 0, Om0 0.3075, hard guard 1e6): legacy 2b86a2d, 0c5b3db, c042527 (core's frozen
reference); core f527b94 legacy_arith (kernel_pin off, pairing node_max), default, fast
(float32 + galaxy_list + gather). Cells: `scripts/cells.py` (g1 1-D H0 33 nodes; g2 9×7
(H0, log10n0)). Run `sbatch scripts/run_arms.sbatch`, then `python scripts/compare.py`.

Dry run (2 cells, job 1350665): 2b86a2d == 0c5b3db; c042527 is −5.34 nats off them at H0 = 60
and the H0 slope changes (legacy history, not core); core default == c042527 to 10 decimals;
core fast differs by ~2e-5; speed 0.27 s (legacy) → 0.03 s (core) → 0.02 s (fast).
core legacy_arith was OOM-killed at 100 GB host RAM; the full job asks for 140 GB.

Environment: core clone `src/darksirens-core-f527b94`, venv `envs/darksirens-core-f527b94`
(--system-site-packages on the jax env, core installed -e --no-deps); legacy worktree
`src/darksirens-c042527`. Next: A5 dynesty smoke (dlogz ~10) once the K = 2 mixture lands in core.
