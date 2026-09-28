# Analysis 11 state

**Stage (2026-09-26, later): closure PASS (1-ULP residuals judged), selection support PASS with one rejected corner; 11A and 11B grids RUNNING; S0 queued.**

- Timing: 3.011 s per evaluation steady state (first call 12.3 s), the same as Analysis 10.
- Grid-design profiles at the planted point (f = 0.3): local marginal sd μ_G 0.38, Δμ_G 0.72
  (ρ −0.77); μ_χ 0.0089, Δμ_χ 0.019 (ρ −0.78). The brief's suggested 0.5 M☉ / 0.01 spacings
  would put 2–3 nodes across the μ_G and μ_χ posteriors, so the one production resolution
  (`scripts/a11_grid_axes.json`) keeps the brief's coarse spacing over the full domains and
  adds a fine core (μ_G 0.125 in [34, 36], Δμ_G 0.25 in [3, 7], μ_χ 0.0025 in
  [−0.025, 0.025], Δμ_χ 0.005 in [0.07, 0.16]); f uses every 0.025 node in [0.1, 0.5] and
  coarser nodes elsewhere, all on Analysis 10's lattice.
- 11A: rita array 1339462, 26,825 cells ≈ 22.4 GPU-h. 11B: rita array 1339463,
  32,375 cells ≈ 27.1 GPU-h. Both about 25–30 h of wall time, one GPU each.
- S0 (`a11_sampler.py --stage s0`, job 1339466) queued behind them.
- tinyns: darksirens' `run_sampler` uses a FIXED isotropic rwalk step of 0.1 in the unit
  cube, several posterior widths here (unit-cube posterior sd ≈ 0.02–0.05), so with
  `min_accepts = 1` late iterations would burn up to `max_attempts` calls each. tinyns'
  own `rwalk_adaptive_step_scale` (not forwarded by darksirens) is therefore used by calling
  `tinyns.NestedSampler` directly; dynesty is run with `bound='multi', sample='unif'` and
  darksirens' checkpoint helpers. Both engines are validated in S1 before either is trusted.

- gws-agn HEAD at start: `2db436aa880563f470b0b94cba811c22a3fa9bcf` (in sync with
  origin). darksirens-a8 `af896cae6f3f3dd1f87dec50046e3a8228f59b39`, clean.
- Inputs re-hashed: events `427990378e299850a9c0708d389bc0bf`, injections
  `e8a611a27f1f0699adc1768b2a3e395a`, GAL survey `3568cf69…`, AGN survey `6dcc38d1…`
  — all equal to the Analysis-10 record.
- `a11_likelihood.py` CPU selftest: PASS (exactly the two base labels released).
- Jobs: `a11_selection_file.py` on RM (job 1339460); `a11_closure.py` on rita
  (job 1339461: C1–C6 closure, live selection support, timing, grid-design
  profiles).

## Sampler survey (2026-09-26, read-only, no runs)

darksirens-a8 `run_sampler` supports tinyns, dynesty and numpyro and can be called
with a hand-built namespace. Per nested-sampling iteration:

- tinyns `recommended` (rwalk, 5 walks × 1 chain): about 5 likelihood calls.
- darksirens' dynesty path (bound multi, sample rwalk hard-coded, walks = 20 + ndim):
  about 23–26.
- tinyns `batched_gpu`: 400; `heavy_darksirens`: 1,280; `bounded_multi`: fails at
  construction (`live-cov` proposal rejected).

At ~3 s per call only `recommended` is affordable. It requires a pure-JAX,
vmappable likelihood, so the redshift-prior optimisation barrier must be off
(`opts.redshift_prior_barrier = "off"` before `make_likelihood`). The factory
likelihood is jitted, one point per call, with no host callbacks. No measured
darksirens nested-sampling costs exist in the repository; S0/S1 will measure them.

## S0 — sampler preflight on the exact 11C likelihood (2026-09-27, job 1339466, 6 min)

- Redshift-prior barrier off vs on: logL bitwise identical at all four probe points
  (including a guard-rejected one).
- `jit(loglike)` of the sampler map: first call 41.7 s (compile), steady 2.99 s; equals
  `cell.evaluate_at` bitwise.
- vmap width 1: 3.12 s, equals the scalar value. Width 2: 6.15 s (no batching gain: the A100
  is saturated by one point), values move by 5e-12 (reduction order).
- Device memory: 10.7 GB in use, peak 28.5 GB of a 63.7 GB limit.
- darksirens `_nested_sampler_preflight` on the 11C box: 31/32 prior draws finite
  (logL in [−5049, −4344]), 96 s.
- S1 queued: dynesty (multi/unif, nlive 300; job 1339614) and tinyns (direct, adaptive
  rwalk, 5 walks, nlive 250; job 1339615) on the A10-J problem.
- 11A chunk 3 hit the rita `TaskProlog failed` start fault (0 rows); resubmitted as 1339613.

## Owner decision (2026-09-27): no tinyns

The owner excluded tinyns. The S1 tinyns job (1339615) was cancelled before it started (no
GPU time spent). S1, 11C and 11D use dynesty only (`bound='multi'`, `sample='unif'`,
darksirens checkpoint helpers); S1 dynesty job 1339614 continues.

## S1 — dynesty validated against the A10-J grid (2026-09-27, job 1339614)

- dynesty `bound='multi', sample='unif'`, nlive 300, seed 1: 3,067 iterations, 15,173
  likelihood calls, 12.7 h, logZ −4291.68 ± 0.18, 3,367 equal-weight samples.
  `results/a11_ns_a10J_dynesty_n300_s1.json`, `diagnostics/a11_s1_compare_*.json`.
- About 10.7k of the calls were spent before the first bound: dynesty 2 builds its
  first ellipsoid only once the unit-cube efficiency falls below 10%
  (`first_update` default). Production runs build the first bound after 2·nlive calls.
- Grid quantiles: the Analysis-10/11 grid code interpolates the CDF linearly between nodes.
  Where nodes are ≥ 0.6 posterior sd apart this widens the 90% ends by up to 0.22
  half-widths. Sampler-vs-grid comparisons use spline-interpolated grid quantiles, and
  the 11A/11B 90% ends quoted from the linear CDF are slightly conservative.

## 11C launched (2026-09-28)

Gate G11AB and S1 passed, so 11C runs: (f, μ_G, Δμ_G, μ_χ, Δμ_χ) at H0 = 67.74 on the
brief's domains. Two independent dynesty runs (multi/unif, nlive 200, seeds 1 and 2,
first bound after 2·nlive calls; jobs 1341008, 1341009), one per rita GPU, merged with
`dynesty.utils.merge_runs` (`a11_sampler.py --stage merge`) into an nlive-400-equivalent
posterior. The two runs also serve as a run-to-run consistency check.

Cost estimate before submission: S1 needed ≈ 10 × nlive iterations in 3-D. 5-D adds
roughly 5 nats of prior-to-posterior compression, so ≈ 15 × nlive = 3,000 iterations per
run; at 10–15% efficiency that is ≈ 20–30k calls ≈ 17–25 GPU-h per run, ≈ 35–50 GPU-h total,
about one day of wall time. That is below Analysis 10's ≈ 150 GPU-h, so no owner stop
applies.

## 11C complete, fixed-H0 owner gate PASS (2026-09-28)

Jobs 1341008 (seed 1: 2,590 iterations, 10,816 calls, 9.5 h) and 1341009 (seed 2: 2,628
iterations, 9,677 calls, 7.8 h), merged to `results/a11_11C.json`. The efficiency (22–30%)
beat the pre-run estimate, so 11C cost ≈ 17 GPU-h, not 35–50. Event assignment: job
1345022 (3 min). Comparisons: `diagnostics/a11_11C_comparisons.json`. Figure
`figs/fig_11C`. Next: 11D (H0 released), which is not started.
