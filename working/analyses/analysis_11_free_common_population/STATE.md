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
