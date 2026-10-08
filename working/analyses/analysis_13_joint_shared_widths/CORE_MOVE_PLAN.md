# Moving A13 off core bf58aa6 before calibration (plan, 2026-10-06)

Owner: move core deliberately before calibration; no calibration runs in this step.

## What changed since the pin

core main is now **5fa48a3**; the recorded target **e7c3007** is two commits behind it.

| PR | commit | touches A13? |
|---|---|---|
| #52 speed defaults | 99de16e | yes, by default: pairing_norm `auto` (per_point), kernel_layout `galaxy_list`, missing_density `auto` (gather), kernel_window `auto`. The historical values are stated to reproduce b47e41c bit for bit. Our driver already sets kernel_window 1e-10, padded, grid, per_sample/analytic explicitly. |
| #53 per-catalog completeness | e7c3007 | mixture.py rewritten for `completeness=[...]`; we pass `completeness=None` with a single value, which the PR says "keeps today's redshift, plan and record". Must be checked, not assumed. |
| #55 README | 1c796a4 | no |
| #56 public API, dynesty default, guard report | 5fa48a3 | runtime_binding.py, public.py, results io, run fingerprint. Probably no arithmetic change; must be checked. |

## Checks (one rita GPU job, after seed 1 finishes)

0. **Environment**: `src/darksirens-core-<sha>` by git archive + `envs/darksirens-core-<sha>` with
   bf58aa6's pins. The driver's path guard takes the sha from the environment.
1. **Historical settings, bitwise**: the four A11 cells plus four A13 cells off the fiducial widths
   (σ_G, σ_χ at the seed-1 90% ends) on the new commit vs bf58aa6. Pass: |ΔlogL| ≤ 1e-8.
2. **New defaults, posterior level**: the same cells plus 500 equal-weight seed-1 posterior draws,
   new defaults vs bf58aa6. Pass: sd(ΔlogL) over the draws ≤ 0.05 nats, so reweighting moves no median or
   90% end by more than 0.05 posterior sd. A constant offset is allowed; it shifts logZ only.
   Also record the s/call (the point of #52).
3. **Guard**: the K3 rejection set (f = 1, AGN spin mean ≥ 0.237) unchanged.

Cost: ~1 h of one A100 (two builds ~6–12 min each on HildaFS, ~1,000 evaluations).

## Decision after the checks

- 1 passes, 2 passes → calibrate on the new commit **with the new defaults** (faster), recording both.
- 1 passes, 2 fails → calibrate on the new commit with the historical settings pinned explicitly.
- 1 fails → stop; take the difference to the core session before anything else.

## Status (2026-10-06)

Owner leans to **e7c3007 as planned**; asked the darksirens-core session (darksirens-work) whether
anything after it (#55, #56, in flight) benefits A13, to report back before the owner chooses.
The checks wait for the owner's review of seed 1 — they do not auto-run.

## darksirens-work's reply (2026-10-06)

- Recommends **5fa48a3**. e7c3007 is equally safe for our path; #55 is README only; #56 is additive
  (ds.infer default sampler, guard report inside ds.infer, BoundAnalysis.diagnostics as a separate
  program). bind_analysis's program is unchanged; dynesty_checkpoint is untouched.
- Bit for bit on paper with our explicit historical settings (#52: 24 cases; #53: single
  completeness value keeps the plan, fingerprint, HLO and logL). Our exact setup is untested → the
  bitwise check stays.
- New defaults: speed comes from the per-point pairing normaliser (2.5–3× on CPU in #52's benchmarks,
  never measured on A100/K=2); |ΔlogL| ≤ 4.0e-8 on the K=2 field mixture. galaxy_list will not remove
  the 27.4 GiB pinned-kernel buffer (the state stays padded) → keep the A100-80. A checkpoint written
  under the old settings resumes only with the old settings set explicitly.
- Nothing in flight to wait for; a chunked pinned-kernel build would be a separate owner request.

## Owner decision (2026-10-06)

- Target: **e7c3007** (as planned), not 5fa48a3.
- New #52 defaults: decide after timing them on the A100 in check 2.
- The checks run only after the owner has reviewed seed 1.

## Results (2026-10-07, core e7c3007, rita A100-80)

| check | result |
|---|---|
| 1 historical settings vs seed 1 (bf58aa6) | **PASS, bitwise**: 4/4 cells, 500/500 posterior points, 20/20 edge points |d| = 0; 6/6 guard rejections hold; 0.280 s/call; peak 17.3 GB (`diagnostics/a13_core_move_e7c3007_historical.json`, job 1375152) |
| 2 #52 defaults vs seed 1 | logL within 3.6e-12 (sd 1.2e-12; 317/500 bitwise); 6/6 rejections hold; **0.324 s/call (16% SLOWER)**; peak **47.9 GB**; the build OOMs at JAX's default 75% cap (one 30.4 GB buffer) and needs XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 (`..._defaults.json`, job 1375182) |

The defaults buy nothing for A13 (slower, 2.8x the memory, needs a raised cap); the historical
settings on e7c3007 are the same program as bf58aa6.

## Per-switch timing (2026-10-07, darksirens-work request; twig A100-SXM4-40GB, jobs 1375238 + 1375255)

Historical settings with ONE #52 default switched on, 100 posterior points each, peak 17.3 GB throughout:
baseline 0.331/0.336/0.332 s/call (bitwise); pairing_norm auto 0.346/0.346/0.349 (+4–5%, logL
≤ 3.6e-12); missing_density auto 0.302/0.308 (−8–9%, bitwise); kernel_window auto 0.303/0.314 (−6–9%,
bitwise); kernel_layout galaxy_list not run (~48 GB build). So galaxy_list carries the all-defaults
slowdown; missing_density + kernel_window auto are bitwise and faster — a candidate for calibration
(owner decision). darksirens-work is fixing the galaxy-list build memory (chunked loop).

## PR #57 (fix/galaxy-list-gpu-memory @ a8f104c) timing, twig job 1375275

Own worktree `src/darksirens-core-pr57` + venv `envs/darksirens-core-pr57` (diagnostic only; A13
stays on e7c3007). Default 75% cap. baseline 0.328 / 0.353 s/call (first/last); galaxy_list 0.334,
peak 17.9 GB, bitwise (the fix works, no slowdown); kernel_pin off 0.861 (2.6×), ≤ 3.6e-12.
No single switch explains the +16% all-defaults result on rita; sum of the per-switch effects ≈ −10%.
PR #57 all-defaults (twig 1375291): 0.373 s/call vs baseline 0.327/0.319 (+14–17%), 17.9 GB, ≤3.6e-12 → an interaction between switches, not allocator state; reported to darksirens-work (pairwise runs offered, would need owner OK).
PR #57 all-but-one (twig 1375300): only all-but-pairing_norm is fast (0.306 vs baselines 0.352/0.370, bitwise); every row with pairing_norm auto sits at baseline level → per_point pairing normaliser is the culprit; galaxy_list + missing_density auto + kernel_window auto with per_sample are bitwise and ~13–17% faster (calibration candidate, owner decision; needs a core with PR #57 for galaxy_list, or keep padded).
Profiles (twig 1375337, diagnostics/profile_pr57/, ~150 MB, not for git): A all-defaults 0.357 vs B per_sample 0.310 s/call with equal flops/bytes. Our driver's jax.jit(b.__call__) embeds the data as constants (generated_code ≈ 12.5 GB) — possible efficiency item (BoundAnalysis already jits; as_pytree_callable passes data as arguments); not changed for the running rerun.
