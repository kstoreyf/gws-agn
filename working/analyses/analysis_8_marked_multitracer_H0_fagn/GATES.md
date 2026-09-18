# Analysis 8 gates

This file is the concise pass/fail ledger. Detailed evidence belongs in `diagnostics/`, `REPORT.md`, and `STATE.md`.

## Scope lock

- [x] Analysis defined as marked GAL/AGN multi-tracer HBI.
- [x] First mark is a GAL/AGN difference in \(\mu_{\chi_{\rm eff}}\) only.
- [x] Seed 100 only.
- [x] Complete catalogs only.
- [x] \(H_0=67.74\) fixed for this campaign.
- [x] Common mass, \(q\), redshift evolution and spin width fixed.
- [x] Multiple realizations require an explicit owner gate.

## Gate A — null/equivalence

Status: **PASS** (2026-09-18, seed 100, local H100 NVL; gws-agn `ac74b98`,
darksirens `af896ca`). Evidence: `diagnostics/null_equivalence.{json,md}`.

Headline: with `mu_chi_c2 = 0` the new path reproduces the Analysis-2 model to the
last bit of double precision. The largest disagreement anywhere in the gate is
**1.8189894035458565e-12**, exactly 2 ULP of a log-likelihood of order 4.2e3
(relative 4.36e-16); the endpoints and the selection integral are exactly zero.

Measured, per check:

| check | measured | verdict |
|---|---|---|
| A1 cell equivalence, f in {0, 0.295, 0.5, 1} | max abs diff 1.8189894035458565e-12 = 2 ULP; exactly 0.0 at f = 0 and f = 1 | PASS |
| A2 endpoint reduction vs K=1 GAL / K=1 AGN | 0.0 exactly in the total, the PE term AND the selection term, for both shapes | PASS |
| A3 selection reduction, mu(f) = (1-f) mu_GAL + f mu_AGN | max rel diff 7.90e-16; mu_GAL = 7.471462445318e-06, mu_AGN = 7.544556698731e-06 | PASS |
| A4 seed-100 f-scan reproduction, 101 cells | max abs diff 1.8189894035458565e-12 = 2 ULP; f median moves by -3.55e-15 (0.26649932068725635 -> 0.2664993206872528), MAP unchanged at 0.27 | PASS |
| guard | 216 cells, min N_eff = 378742 = 75.7x the threshold 5000; 0 rejected, 0 at -inf | PASS |

Three findings worth carrying forward:

* **The coordinate is live, and wired to the right branch.** An equivalence test
  passes for free if the new parameter is ignored — the exact defect recorded
  below. Measured: at f = 0.295, `mu_chi_c2` 0 -> +0.10 moves logL by
  **-18.2976**; at f = 0, where catalog 2 carries zero mixture weight, the same
  offset leaves logL **bitwise identical**. The reachability fix holds.
* **The tolerance in this file was in the wrong unit.** The documented
  `<= 3.552713678800501e-15` is exactly 2 ULP of a toy logL of order 8
  (`np.spacing(8.0) = 1.7763568394002505e-15`). At the seed-100 scale 1 ULP is
  9.0949470177292824e-13, so the same 1-2 ULP effect is 1.82e-12 — 500x the
  quoted absolute number and the identical effect. Quote this bound in ULP, not
  in absolute logL. The gate does not rest on it either way: re-running the
  UNCHANGED Analysis-2 configuration through the current code lands the SAME
  2 ULP from the stored array (87/101 cells exactly 0), so the residual is the
  machine's last-bit floor, not the new population path and not code drift.
* **The residual sits at the PE seam, not the selection seam.** This file
  recorded it as "isolated to the last bit of the selection term". On seed 100
  `log_mu` is bitwise identical between the two shapes at every f, so the
  selection contribution agrees exactly and the whole 2 ULP is in the per-event
  PE sum. Same effect, same size, other seam.

One caveat on A3 for the record: `mu_GAL` and `mu_AGN` differ by +0.98%, about 6x
the naive 1/sqrt(N_eff) Monte-Carlo scale. That is not a violation of the
chi_eff-independence of the v3 detection rule — the two branches carry different
spatial priors p_k(z|pix) and should not coincide. The chi_eff-independence claim
was measured separately and holds: moving the AGN branch spin mean by +0.10 changes
mu by -3.44e-04, i.e. 0.30 Monte-Carlo sigma.

### Implementation findings recorded before the gate runs

The per-tracer population block is implemented in the local darksirens worktree
`src/darksirens-a8` (branch `analysis8-marked-populations`, base SHA `2b86a2d`,
never pushed) as a new `mixture_pop_params` kwarg on `darksiren_log_likelihood`.

Measured, not argued:

* **Backward compatibility.** With `mixture_pop_params=()` the value is
  bit-identical to the pinned code: the same hex float `0x1.7701cac65c400p-6`
  with the change stashed out and in, through both `make_likelihood` and a direct
  core call, at 8 separate mixture weights. The empty case re-executes the
  historical statement verbatim because `pop_params_shared` is a static Python
  bool, so this is structural rather than coincidental.
* **Regression.** 60 passed, 1 skipped on the seven-file suite (baseline exactly
  60/1) and 214 passed, 1 skipped on the tier-0 subset. Three failures seen in a
  wider sweep were re-run on the stashed baseline and fail there with identical
  messages and digits, i.e. pre-existing at the pin.
* **Ordering (specification 0.6).** An adversarial review confirmed the branch
  sum happens only after the per-branch spatial x intrinsic product:
  `_eval_prior_branches` returns the open list `[log f_k + log p_k(z|pix_k)]`,
  `log p_pop(theta | Lambda_k)` is added inside each branch, `_mixture_logsumexp`
  collapses afterwards, and the branch-independent Jacobian and `log prior_wt`
  are subtracted once outside. `f_k` appears exactly once. PE and selection use
  the identical kernel, so the normalisation cannot disagree with the event
  weights. No spatial-times-intrinsic Bayes-factor product exists anywhere.
* **Tolerance for the explicitly-equal-vector case.** Supplying two identical
  population vectors takes the new per-branch route and differs from the shared
  route by 1-2 ULP: max 3.552713678800501e-15 absolute in logL, 1.15e-16
  relative, isolated to the last bit of the selection term and caused by the
  intended re-association `logsumexp_k[a_k] + c -> logsumexp_k[a_k + c]`. It is
  exactly 0.0 at some mixture weights and nonzero at others. Specification 4/A1
  permits this provided the tolerance is documented before acceptance; it is
  documented here, and it is far below any evidence difference that matters.

### Defect found and fixed before the gate ran

The feature was **unreachable**: `build_parameter_space` emits `_c{k}` suffixes
only for survey, stick and mark labels (`prior.py:604-610`) and hard-rejects any
other key (`prior.py:611-623`), so `has_per_catalog_pop_params` was always false,
`mixture_pop_params` always `()`, and a production run would have silently kept
the old shared-population model. Two tests failed to catch it because they
hand-built the decoder with a plain label spelling (`"mu_chi"`) that never occurs
in production, where population labels are LaTeX (`'$\mu_\chi$'`).

Required:

- [x] new tracer-dependent population path identified/implemented
      (sampled labels gain `'$\mu_\chi$_c2'`, and it moves logL by -18.30 at
      `dmu_chi = +0.10`);
- [x] \(\Delta\mu_\chi=0\) reproduces existing Analysis-2 likelihood
      (max 2 ULP = 1.819e-12);
- [x] \(f=0\) reproduces GAL endpoint (0.0 exactly, PE and selection alike);
- [x] \(f=1\) reproduces AGN endpoint (0.0 exactly, PE and selection alike);
- [x] selection term reproduces existing K=2 result (mu(f) linear to 7.90e-16);
- [x] seed-100 Analysis-2 \(f_{\rm AGN}\) posterior reproduced
      (median shifts 3.55e-15, MAP unchanged);
- [x] evidence written to `diagnostics/null_equivalence.*`.

Gate A passed, so the marked mock is unblocked.

## Gate B — marked seed-100 mock integrity

Status: **UNBLOCKED** (Gate A passed 2026-09-18); not started.

Registered mark:

\[
\Delta\mu_\chi^{\rm plant}=+0.10.
\]

Required:

- [ ] existing seed-100 LSS/catalog realization reused;
- [ ] existing signed-off files unchanged;
- [ ] one new marked seed-100 event family only;
- [ ] host fraction recorded;
- [ ] GAL/AGN truth spin means differ by the registered amount;
- [ ] masses and \(q\) have no planted channel difference;
- [ ] v3 PE contract unchanged;
- [ ] selection support validated;
- [ ] validation written to `diagnostics/marked_mock_validation.json`.

## Gate C — seed-100 marked recovery

Status: **BLOCKED ON B**

Exactly three scientific arms:

- [ ] S: spatial-only;
- [ ] I: intrinsic-only;
- [ ] J: joint spatial+intrinsic.

Required:

- [ ] joint \(f_{\rm AGN}\) recovery;
- [ ] joint \(\Delta\mu_\chi=+0.10\) recovery;
- [ ] spatial-only result;
- [ ] intrinsic-only result;
- [ ] joint result;
- [ ] selection guard valid over posterior support;
- [ ] event-level spatial/intrinsic evidence decomposition;
- [ ] production figures;
- [ ] `REPORT.md`.

## Owner gate

Status: **LOCKED**

After Gate C, stop.

Forbidden before explicit owner approval:

- seeds 101/102/103/105;
- any additional realization;
- free \(H_0\);
- mass marks;
- incompleteness;
- GP/HSGP population differences;
- GWTC data.

Completion line required from the driver:

> **OWNER GATE: seed-100 Analysis 8 is complete. I have not run additional realizations.**
