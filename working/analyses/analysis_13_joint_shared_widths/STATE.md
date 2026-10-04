# Analysis 13 state

**RESUME HERE (2026-10-03): A13 seed 1 RUNNING on darksirens-core bf58aa6 (post-#54; pinned, bitwise-rechecked), rita job 1351377 (`sbatch --export=ALL,SEED=1 scripts/submit_a13core_gpu.sbatch`; resubmit identically to resume from `queue/a13core_*.save`). The af896ca dynesty run (rita 1350718) was cancelled at 1,852+ iterations (checkpoint kept, not used). Second smoke sampler still DEFERRED. Production (second seed) and calibration remain owner-gated; core main e7c3007 (#52 defaults, #53) is NOT adopted — move deliberately before calibration.**

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

## Restart on darksirens-core (2026-10-02)

Owner: "Restart on core"; "New runs on core" (finished analyses keep their results with the code
version recorded); second sampler still deferred. Core b47e41c is validated for this likelihood
(A11 grids: KS ≤ 0.044, 90% widths within 0.2%, 12× faster; experiment_darksirens_core_equivalence).
The core run uses the same coordinates, boxes and dynesty settings as problem 13 and a pre-flight
that reproduces the measured core A11 cells to 1e-8 at the fiducial widths. The af896ca run's
checkpoint (`queue/a11_ns_13_dynesty_n200_s1.ckpt`, cancelled at ~30k calls, dlogz 27) is kept but
not resumed.

## Checkpoint fix and Jetstream2 (2026-10-03/04)

- rita 1351377 (core bf58aa6, seed 1) passed its pre-flight (4/4 cells |d| = 0) and then died at the
  first dynesty checkpoint (900 s): dynesty pickles the sampler, and `loglike` is a closure
  ("Can't pickle local object"). Fixed: checkpoints now go through core's
  `darksirens.inference.dynesty_checkpoint` (state-only save; the callables are rebound on restore).
  A toy run killed at iteration 302 and resumed reproduces the uninterrupted logZ exactly
  (653 iterations both).
- Owner moved the run to the Jetstream2 A100 VM (rita busy): `scripts/js2/` (README there). The
  driver takes `A13_DATA`, `A13_REF_CELLS`, `A13_KERNEL_LAYOUT`, `A13_MISSING_DENSITY` from the
  environment; the defaults are the rita values, so `submit_a13core_gpu.sbatch` is unchanged.
- The VM's GPU is a 20 GB vGPU slice. The likelihood build ran out of device memory (a 1.43 GB
  allocation in the pinned field-kernel build) with the default 75% JAX cap (15.8 GB), with the cap at
  95% (18.7 GB), and with 95% plus the galaxy-list/gather layouts (17.1 GB). Host RAM peaked at ~14 GB.
- Fourth try, platform allocator (no BFC cache) plus the layouts: it got past that allocation and then
  requested a single **27.4 GiB** buffer in the same step (`build_pinned_catalog_kernel` →
  `_state(catalog)`, the field compact view). **A13 cannot run on a 20 GB GPU with core bf58aa6**;
  it needs an 80 GB card (rita A100-80, js2h100 H100-80) or a core change that chunks that build.
  `scripts/js2/` works unchanged on a bigger VM (set `JS2`/`JS2_ROOT` in `config.sh`).
