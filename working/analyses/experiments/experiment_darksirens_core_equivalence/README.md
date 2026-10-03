# Experiment — darksirens-core as gws-agn's darksirens (K = 1 equivalence)

**Status (2026-10-02): K = 1 COMPLETE. Core reproduces our old code at the posterior level once the
merger-rate slope is pinned to the mock's γ = 0 (see the end). Owner criterion: posterior-level
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

## Full K = 1 result (job 1350666, 96 cells per arm; `results/comparison.json`)

| pair | bitwise | max abs dlogL | posterior (g1 H0) |
|---|---|---|---|
| 2b86a2d → 0c5b3db | 96/96 | 0 | identical |
| 0c5b3db → c042527 (legacy history) | 0/96 | 19.4 (mean −1.20, spread 6.6) | **KS 0.43, 90% width 1.14×: posterior moves** |
| c042527 → core legacy_arith | 85/96 | 1.8e-12 | identical |
| c042527 → core default | 51/96 | 3.6e-12 | identical |
| core default → core fast | 0/96 | 3.6e-5 | identical (KS 0) |

Core reproduces its frozen reference to rounding, and the opt-in speed-ups change nothing at the
posterior level. The posterior shift is the 435 legacy commits between 0c5b3db and c042527, not
core: moving gws-agn to core moves K = 1 incomplete-catalogue posteriors on this mock. Timing
(median s/call): legacy 0.266, c042527 0.043, core default 0.029, core fast 0.016 (16× legacy).

## Where the legacy-history shift comes from (bisect, job 1350701; `results/bisect/summary.json`)

Bisected the 92-merge first-parent chain 0c5b3db..c042527 at H0 = 60, 67, 75 (log10n0 = −3),
1e-3 tolerance, logL recorded at every step (13 evaluations, 5 min). Essentially all of it is one
merge: **PR #359 `review/populations`** (575464d, 2026-08-11, inside the #367 review-fixes merge):
dlogL −5.34, −5.99, +0.30. The only other change is #355 `review/selection-catalog`
(−0.002). Inside #359, **0befab7 "fiducial rate slope is the measured kappa_z, not zero"** changes
the fixed powerlaw+peak fiducial **γ from 0 to 2.5**. The mock was generated with γ = 0, so at
c042527 and in core a `fixed=True` population assumes the wrong merger-rate evolution.

## With γ pinned to 0 (job 1350706; `legacy_k1_grid.py --gamma 0`, `core_k1_grid.py --gamma 0`)

| pair | max abs dlogL | posterior |
|---|---|---|
| c042527(γ=0) → core default(γ=0) | 3.6e-12 | identical |
| core default(γ=0) → core fast(γ=0) | 4.0e-5 | identical |
| **0c5b3db → core (γ=0)** | 1.25 (mean −0.92, spread 0.20) | H0 KS 0.035 (1-D) / 0.039 (2-D), 90% width 0.99 / 0.97, median shift 0.05 half-widths; log10n0 KS 0.018; joint TV 0.039 |

The remaining 0.2-nat shape change is the other #359 population commits (7f82fc0 shared low-mass
edge, 214c2ba closed-form normalisers, 8f3826a pairing quadrature). **For gws-agn on core: pin γ
to the mock's value explicitly** (`Population("powerlaw+peak", fixed={..., "$\\gamma$": 0.0})`);
never rely on `fixed=True`.

## A5 free-anchor dynesty smoke (prepared 2026-10-02; waiting for core #48/#49 to MERGE)

Old code: `scripts/a5_smoke_legacy.sbatch` (0c5b3db, the archived A5 m<18 recipe: GAL+AGN m<18,
field weighting, c_mode=selection with the true-z Schechter fits, n0 priors [−4, −1] / [−6, −4],
nlive 1000, rstate 7, dlogz 10). Core: `ds.model(catalog=[GAL, AGN], catalog_sky_weighting="field",
completeness="selection", selection=<stripped fits>, survey_priors={log10n0: [−4, −1],
log10n0_c2: [−6, −4]})`, γ pinned to 0, same nlive/dlogz, to be written against the merged API.
Compare: logZ, per-parameter KS on the 4 posteriors, 90% widths; quick check only, not production.

### A5 smoke result (jobs 1350894 legacy, 1350895 core 661ef3d; `results/a5_smoke/comparison.json`)

| | legacy 0c5b3db | core 661ef3d (γ = 0) |
|---|---|---|
| logZ | −4187.54 ± 0.70 | −4188.11 ± 0.72 (Δ −0.57, combined error 1.0) |
| calls / iterations | 43,915 / 5,530 | 44,791 / 5,556 |
| time per call | ≈ 0.19 s | 0.034 s (5.7×) |
| wall | 2 h 25 m | 26 m |
| peak memory | — | host 5.6 GB, device 1.4 GB |

| | H0 | log10n0 | log10n0_c2 | f_AGN |
|---|---|---|---|---|
| median shift (half-widths) | −0.015 | −0.025 | +0.022 | +0.018 |
| 90% width ratio | 0.992 | 0.991 | 0.977 | 0.984 |
| KS | 0.108 | 0.092 | 0.069 | 0.104 |

Kish n_eff is 146 (legacy) and 127 (core) at dlogz 10, so the 95% KS critical value is 0.165:
every KS is inside sampling noise (the naive p-values count duplicated resamples as
independent). Max correlation difference 0.09. The legacy logZ equals the archived full A5 m<18
run (−4187.537). **Verdict: core reproduces A5 at the posterior level**, with γ pinned to 0.

## A8–A13 line: the Analysis-11 likelihood on core b47e41c against af896ca (2026-10-02)

Core build (darksirens-work's equivalent of a11_likelihood.build_a11): complete GAL + AGN
catalogues, field weighting, completeness=None at log10n0 = log10n0_c2 = −24
(allow_out_of_prior), per_catalog_population={2: [μ_G, μ_χ]}, rest of powerlaw+peak pinned
with γ = 0, hard guard 1e6, sel_batch 5000, pe_event_block 5, **kernel_window = 1e-10 set
explicitly** (kernel_layout padded, missing_density grid recorded). Scripts `core_a11_cells.py`,
`legacy_window_check.py`, `a11_grid_cells.py`, `compare_a11.py`.

1. **af896ca's own KDE window drops nothing:** W = 4096 and full rows give identical logL at
   all four cells (`results/legacy_af896ca_window_{4096,full}.json`).
2. **Core vs af896ca, 212 cells around the 11D posterior** (`results/a11_comparison.json`):

| slice | mean dlogL | spread | KS | 90% width ratio | median shift (half-widths) |
|---|---|---|---|---|---|
| H0 (33) | −3.55 | 0.18 | 0.015 | 1.007 | +0.023 |
| f_AGN (17) | −3.63 | 0.29 | 0.028 | 1.002 | +0.043 |
| (μ_G, Δμ_G) 9×9 | −3.68 | 0.31 | 0.026 / 0.044 | 1.000 / 0.999 | −0.040 / +0.067 |
| (μ_χ, Δμ_χ) 9×9 | −3.69 | 0.07 | 0.003 / 0.005 | 0.998 / 0.999 | +0.004 / +0.007 |

The ~−3.6 nat offset is constant to ±0.3 and moves no posterior; it is the legacy population
fixes after 2b86a2d that af896ca lacks (darksirens-work). Conditional slices, not full marginals.
Speed: core 0.28 s/call vs af896ca 3.43 s (12×). **Verdict: A8–A13 can move to core**, with
γ and every population value pinned explicitly and the core commit pinned.

## Post-#54 recheck on core bf58aa6 (2026-10-02, job 1351068)

darksirens-core #54 fixed a mixture bug (a failed per-catalog safety verdict returned the other
catalog's finite value instead of -inf). On bf58aa6, the last commit before the #52 default
changes, with every setting explicit (kernel_window 1e-10, padded, grid, pairing_norm per_sample,
pairing_scale analytic, γ = 0): the 4 A11 cells and all 212 A11 grid cells are bitwise equal to
b47e41c, and the A5 probe point is bitwise equal to 661ef3d. **bf58aa6 is the validated pin for new
runs** (A13 onward).
