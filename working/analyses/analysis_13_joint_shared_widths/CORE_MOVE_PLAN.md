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
