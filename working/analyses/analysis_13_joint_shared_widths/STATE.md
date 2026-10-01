# Analysis 13 state

**RESUME HERE (2026-10-01): closure PASS. Dynesty smoke run 1350474 RUNNING (it doubles as production seed 1). The second smoke sampler (tinyns ≈ 85–135 GPU-h vs Nautilus ≈ 17–33 GPU-h, per the tinyns session) awaits the owner. Production and calibration are owner-gated.**
smoke sampler (tinyns or Nautilus) awaits the owner's choice. Production and calibration are
owner-gated.**

## Setup (2026-10-01)

- Owner request: set up A13; consult the tinyns session on settings; smoke test dynesty against
  tinyns on one GPU, seed 100, comparing logZ and posteriors; gate the two-seed run; defer
  calibration until after the two seeds.
- Problem `13` added to `a11_sampler.BOXES` (11D + σ_G [1, 10] + σ_χ [0.01, 1], Δμ_χ to 0.30).
- tinyns session advice (2026-10-01, tinyns 0.2.3 @ 5b6da64): defaults (rwalk, live-cov
  proposal, walks = max(25, 6·ndim) = 48, no bound, one chain), nlive 200, jax_block_size 4,
  return −inf directly, checkpoint via run(checkpoint_path=, checkpoint_interval=) and
  resume(). Its cost estimate for this problem: ~100–160k likelihood calls per run (85–135 GPU-h
  at 3 s/call), 3–4× dynesty; it recommends tinyns as a cross-check only and suggests Nautilus
  (installed here: 1.0.5) as the calls-efficient alternative. walks = 25 biased logZ by
  +0.2–0.3 nats at 8-D in its sweeps.
- tinyns session, Nautilus advice (2026-10-01): `Sampler(prior, likelihood, n_dim=8, n_live=1000,
  n_networks=4, n_batch=100, vectorized=False, pass_dict=False, seed=100, filepath=<.hdf5>,
  resume=True)`; `run(f_live=0.01, n_eff=5000 (2000 to save calls), discard_exploration=True on
  the FIRST call, timeout=<s before wall limit>)`; −inf allowed (map NaN to −inf); results
  `sampler.log_z`, `sampler.posterior(equal_weight=True)`. Its estimate (extrapolated from 13-D,
  not measured at 8-D): ~20–40k calls (17–33 GPU-h), comparable to dynesty, with much tighter
  logZ; no reported logZ error (seed scatter ~0.01 nats). It advises pinning 1.0.6 in a separate
  venv; this jax env already has nautilus 1.0.5. Benchmark report:
  `/hildafs/projects/phy220048p/magana/darksirens-core-data/tinyns_h100_2026-09-30/nautilus_bench_REPORT.md`.
- Jobs: closure 1350473; dynesty smoke = seed 1 of production, 1350474 (afterok closure; it doubles
  as production seed 1 if the smoke passes).
