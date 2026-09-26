# Analysis 11 state

**Stage (2026-09-26): specification written; closure + selection-support jobs running.**

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
