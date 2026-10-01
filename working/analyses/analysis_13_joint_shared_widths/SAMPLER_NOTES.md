# Analysis 13 — sampler notes from the tinyns session (2026-10-01)

The owner asked to consult the tinyns session (the Claude session that develops tinyns) on how
to sample problem 13. This file records everything it said, in two replies. The choice of the
second smoke-test sampler is **deferred by the owner** (2026-10-01). Only the dynesty smoke run
(job 1350474) was launched.

## What it was told

8-D flat box prior (unit-cube transform), a jit'd pure-JAX likelihood at 3.0 s per point on one
A100, no vmap gain (width 2 takes 6.15 s), a −inf region from the selection guard, a compact
unimodal posterior (unit-cube sd ~0.01–0.05, |ρ| ≤ 0.6), logZ ≈ −4297, H ≈ 15–20 nats. dynesty
multi/unif with nlive 200 needs ~14.6k calls per run at 6-D and ~24k at 7-D, so ~35–40k is the
8-D projection.

## Reply 1 — tinyns

**Headline: for this problem tinyns costs about 3–4× more likelihood calls than our dynesty
setup. Use it as a cross-check, not as the production sampler.**

Why. tinyns 0.2.3 replaces each dead point with a live-covariance random walk of
walks = max(25, 6·ndim) = 48 proposals. It beats dynesty by avoiding host overhead (ms-per-call
likelihoods) and by not stalling at high dimension; neither helps at 3 s per call.

Cost estimate (nlive 200, H ≈ 15–20): ~200·(H + 4) ≈ 4–5k iterations × 48 proposals
≈ 190–240k proposals. Out-of-cube proposals are skipped (typically 20–50%), so ~100–160k
likelihood calls ≈ **85–135 GPU-h** at 3 s per call, against 35–40k for dynesty. Fewer walks is
not safe: at 8-D, walks = 25 biased logZ by +0.2–0.3 nats in its sweeps, and walks = 5 is badly
biased.

Settings if tinyns is run (tinyns 0.2.3 @ `5b6da64`, called directly):

```python
from tinyns import NestedSampler
sampler = NestedSampler(loglike, prior_transform, ndim=8, nlive=200, jax_block_size=4)
result = sampler.run(100, dlogz=0.1, progress=True, progress_interval=4,
                     checkpoint_path="run.ckpt.npz", checkpoint_interval=4)
```

- The defaults are the recommended path: `sample="rwalk"`, `kernel="jax"`,
  `rwalk_proposal="live-cov"` (always adaptive, target acceptance 0.25), `walks=48`,
  `bound="none"`, `replacement_chains=1`. Do **not** pass `walks=5`, `step_scale`, bounds or
  `rwalk_adaptive_step_scale`: those were the old broken path.
- `jax_block_size=4`, not the default 32. One block costs block_size × 48 × 3 s; at 32 that is
  ~77 min between progress reports and checkpoints, at 4 it is ~10 min. Host overhead does not
  matter at this cost.
- `loglike(theta) -> scalar` and `prior_transform(u) -> theta` must be JAX-traceable. Pass the
  likelihood as `jax.tree_util.Partial(fn, data)` or as a plain closure; arrays of ≥ 4096 elements
  are passed as jit arguments automatically. tinyns vmaps the likelihood over one chain, so it
  must be vmappable: keep the optimization barrier off.
- The 200 initial live points are evaluated in one compiled pass, one at a time (~10 min).
- Keep `replacement_chains=1`. More chains only burn calls: vmap gives no gain here, and the
  out-of-cube skip is disabled under vmap.
- nlive 200 and dlogz 0.1 match the dynesty runs. The reported logzerr = sqrt(H/nlive) ≈ 0.3.
- **−inf:** return −inf directly (e.g. `jnp.where(ok, logl, -jnp.inf)`); −1e300 is not needed.
  −inf live points die first with zero weight, and chains start only from points above the
  threshold when any exist. **NaN is not handled:** map NaN to −inf. Finite plateaus (ties) get
  no special treatment.
- **Checkpoint/resume:** `run(..., checkpoint_path=..., checkpoint_interval=<iterations>)` writes
  an `.npz`. To continue: `sampler.resume("run.ckpt.npz", dlogz=0.1,
  checkpoint_path_out="run.ckpt.npz")`, with a sampler built with the same arguments.
- **Results:** `result.logz`, `result.logzerr`, `result.samples` (dead + final live, in theta),
  `result.logl`, `result.logwt` (posterior weights = exp(logwt − logz)),
  `result.resample_equal(key)`, `result.save_npz` / `NestedSamplingResult.load_npz`,
  `result.to_dynesty_dict()`, `result.metadata` (ncall, wall_time_s, mean_ms_per_call,
  insertion ranks).
- **Merging two runs:** there is no merge function yet. Z = mean of the Z_i
  (logsumexp(logz_i) − log 2); pool the posterior samples with weights proportional to
  exp(logwt − logz_combined)/2.
- **Pin** 0.2.3 (`5b6da64`). Pitfalls:
  (a) logzerr can be ~25% optimistic above ~10-D; fine at 8-D.
  (b) Strongly curved posteriors need more walks than 6·ndim; our compact |ρ| ≤ 0.6 case is fine.
  (c) Multimodal posteriors: mode weights drift between seeds; ours is unimodal.
  (d) Don't go through darksirens' old `run_sampler` wrapper unless darksirens-core is at or
  past the `livecov` preset (core #33); the old `recommended` preset used a fixed 0.1 step.

Its suggestion: a better candidate for a calls-limited problem is **Nautilus**
(`nautilus-sampler`). On its 13-D GW population likelihoods Nautilus needed ~10× fewer calls
than tinyns, with accurate and much tighter evidence; the fixed ~5 min of network training is
irrelevant at 3 s per call. Use `discard_exploration=True` for the evidence. For the smoke test
it would compare all three on seed 100: dynesty multi/unif, Nautilus, tinyns defaults.

## Reply 2 — Nautilus

Caveat first: the call count below is an extrapolation from 13-D runs, not a measurement at 8-D.

Measured (nautilus-sampler 1.0.6 on js2h100, two 13-D GW population likelihoods, n_live 2000,
n_eff 1e4): 52–66k calls (cheap mock) and 86–100k calls (2.5M-injection mock); logZ within
0.2–1.0σ of an independent ground truth, seed scatter 0.001–0.012 nats; −inf returns fine
(~2k per run). Full report:
`/hildafs/projects/phy220048p/magana/darksirens-core-data/tinyns_h100_2026-09-30/nautilus_bench_REPORT.md`.

Settings:

```python
from nautilus import Sampler
sampler = Sampler(prior, likelihood, n_dim=8, n_live=1000, n_networks=4, n_batch=100,
                  vectorized=False, pass_dict=False, seed=100,
                  filepath="naut_seed100.hdf5", resume=True)
ok = sampler.run(f_live=0.01, n_eff=5000, discard_exploration=True,
                 timeout=<seconds before the wall limit>, verbose=True)
```

- `prior` is the unit-cube transform, a callable u → theta; with `pass_dict=False` the
  likelihood receives the 1-D theta array.
- `vectorized=False`, since one point saturates the GPU. `n_batch=100` is how many points
  Nautilus proposes between bound updates and checkpoints (~5 min at 3 s per call).
- `n_live=1000` (default 2000) is a reasonable cost/robustness trade for a compact unimodal 8-D
  posterior. One `n_live=2000` run is what the Nautilus docs suggest as a stability check.
- `n_eff=5000` target ESS, far more than a dynesty nlive-200 run gives; 2000 saves calls.
- `discard_exploration=True` for an unbiased evidence; it costs ~15–25% more calls. Set it on
  the **first** `run()` call: Nautilus applies it only at the end of exploration, so turning it
  on in a continued run silently does nothing. The default (False) can be biased on curved
  targets.
- **Expected calls:** roughly 20–40k at 8-D with n_live 1000, n_eff 5000 (≈ 17–33 GPU-h):
  comparable to dynesty's 35–40k, probably somewhat fewer, with much tighter evidence. Not the
  10× win it was against tinyns' random walk.
- **logZ error:** 1.0.6 reports none. Seed scatter was ~0.01 nats at n_eff 1e4; expect a few ×
  0.01 at n_eff 5000. Two seeds give an empirical check.
- **−inf** is allowed (return −inf, not −1e300). NaN: map to −inf.
- **Checkpoint:** `filepath=<.hdf5>` (needs h5py) saves state after each batch. For a 96 h
  limit pass `timeout=`; `run()` then returns False. Recreate the Sampler with the same arguments
  and `resume=True`, and call `run()` again with the same kwargs.
- **Results:** `sampler.log_z`, `sampler.n_eff`; `points, log_w, log_l = sampler.posterior()`
  (weighted); `sampler.posterior(equal_weight=True)`. Combining two seeds: average Z
  (logsumexp(log_z_i) − log 2) and pool the weighted samples with weights exp(log_w_i), scaled
  by Z_i / ΣZ.
- **Pin** nautilus-sampler == 1.0.6 (pulls scikit-learn; add h5py), installed in a separate venv
  layered on the jax env so the sklearn/numpy pins don't disturb it. In its setup
  `python -m venv --system-site-packages` was not enough; a `.pth` pointing at the base env's
  site-packages was needed. **Our jax env already has nautilus 1.0.5.**
- Pitfalls: network training is ~15 s of serial CPU per bound, negligible next to ~300 s per
  batch, so no pool; set OMP_NUM_THREADS=4 if sklearn/BLAS threads oversubscribe the 8 CPUs.
  Memory is small. A fixed seed is reproducible and a resumed run continues the same stream.
  With `vectorized=False` the likelihood must return a Python float or a 0-d array.

Its pick for the smoke test: dynesty multi/unif (the reference) plus Nautilus as above, both on
seed 100; tinyns only as an optional third cross-check, given its ~3–4× call cost here.

## Implications for Analysis 13 (agn session)

- The sampler driver (`a11_sampler.stage_run`) still has the **old** tinyns branch (walks,
  step_scale, `rwalk_adaptive_step_scale`): exactly the path the tinyns session says not to use.
  Before any tinyns run it must be rewritten to the defaults above. It has no Nautilus branch.
- The dynesty branch maps −inf to −1e300. That is fine for dynesty; tinyns and Nautilus take
  −inf directly.
- Cost per smoke-test run: dynesty ~35–40k calls (~30–35 GPU-h); Nautilus ~20–40k (17–33 GPU-h,
  extrapolated); tinyns ~100–160k (85–135 GPU-h, more than one 96 h job, so a resume).
