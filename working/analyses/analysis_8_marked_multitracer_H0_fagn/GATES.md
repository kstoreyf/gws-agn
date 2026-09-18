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

Status: **PASS**, 10/10 (2026-09-18). Evidence:
`diagnostics/marked_mock_validation.json`, `diagnostics/selection_support.json`.
Product: `seed100/events/events_marked_dmu0p10.h5` (md5 7dcb8bccba7f7a360a6da2741d4cf9d1),
generated with `--dmu_chi_agn 0.10 --events_suffix _marked_dmu0p10`, no `--seed_events`,
no `--f_agn`, no `--overwrite`.

**The mark is exactly where it should be.** Against a same-environment unmarked run,
46 of 50 datasets are bit-identical and the 4 that differ are all effective spin
(`chieff`, `true_chieff`, `truth/chieff`, `truth/obs_chieff`), differing on exactly the
295 AGN events and nowhere else. GAL true spin is bitwise unchanged; the AGN shift is
0.1 to within 2.8e-17. Host labels, indices, redshifts, distances, sky, masses, `q`,
SNRs and every `truth/obs_*` array are bit-identical, and so is the rejected-proposal
file. Realised: 705 GAL / 295 AGN, `f_agn` 0.295, unchanged from the record.

**Selection: REUSE.** No new injections. The existing 2.2M-injection targeted set spans
chieff [-0.99999507, +0.99997832], the whole truncation both branches need.

Three things this gate established that Gate C must carry.

* **The registered scan range collides with the guard.** The specification registers
  `dmu_chi` in [-0.20, +0.25], but the REAL selection integral falls below the hard
  N_eff floor of 5000 (likelihood returns -inf) above `mu_chi_c2` ~ +0.2273 at f = 1,
  +0.2787 at f = 0.5 and +0.3238 at f = 0.295. At f = 1, mu = +0.25 the measured N_eff
  is 2766 and the cell is rejected. Recommended scan `|mu_chi_c2| <= 0.20`, whose
  thinnest corner (f = 1, mu = +0.20) sits at 11,013, i.e. 2.2x the guard. Gate C must
  either scan the reduced range or scan the registered one and DEMONSTRATE that the
  rejected corner carries negligible posterior mass; specification C6 forbids a result
  standing behind a rejected cell either way.
* **Do not size that margin from the population-only proxy.** Check 9 of
  `marked_mock_validation.json` uses it and reads 10,044 (pass) at mu = +0.25, where the
  real selection integral is 2766 (fail); the proxy puts the crossing at 0.286 against a
  true 0.227 at f -> 1. Likewise the injection file's own `Neff` attribute (3714.98) is
  the N_eff of a FLAT target computed at generation time, two orders of magnitude below
  any selection-integral N_eff, and must never be quoted as the selection margin.
* **Seed 100 carries an unplanted GAL/AGN difference in mass ratio.** KS p = 0.0096 on
  the detected set; permuting labels within redshift quartiles gives p = 0.0066; the
  z-stratified Fisher statistic 29.57 is the largest of the 41 seeds examined (median
  7.54). It is the record's own property, present in the unmarked data and not created
  here, but it means the branch label is partly identifiable from `q` alone. The
  intrinsic arm therefore cannot be read as measuring the spin mark in isolation, and
  Gate C must report this confound rather than attribute all intrinsic information to
  `dmu_chi`.

Two caveats for readers. The `shared_spin` HDF5 attribute is still `True` in the marked
file (correct in darksirens' vocabulary, where it means one spin component shared across
the MASS components of one mixture); branch spin must be read from `dmu_chi_agn` = 0.1,
`mu_chi_gal` = 0.0, `mu_chi_agn` = 0.1. And the **realised** mark is +0.111924 +/- 0.006815,
not +0.10: the record already carried a +0.011924 branch difference from finite sampling,
identical to the last digit in the unmarked run. Score Gate C against both, as analyses
0-2 do with planted 0.30 against realised 0.295.

**On "one difference".** Against the signed-off record 38 of 50 datasets differ, not 4.
The extra difference is not the mark: it is a 1-ULP environment drift in the tapered
power-law inverse CDF (11 of 1000 primary masses, max 1.42e-14, and everything
downstream of them), because the August record was written on a node with a different
`pow` kernel. Proven by recomputation: this node reproduces the MARKED file's
`snr_true` bit-for-bit from its own stored masses and distances (0/1000 differ) but not
the RECORD's (406/1000, max 1.421e-14). The detected set, its order, and every realised
statistic are unchanged to full printed precision. Gate A independently measured the
same drift: re-running the UNCHANGED analysis-2 configuration lands 2 ULP from the
stored array. The flag itself is provably inert - pristine generator versus patched
generator at the default gives 50/50 datasets bit-identical.

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

Status: **PASS** (2026-09-18, seed 100, local H100; gws-agn `c3bd534`, darksirens
`af896ca` on the pinned base `2b86a2d`, tree clean). Evidence:
`results/arm_{J_joint,I_intrinsic,S_spatial}.{h5,json}`,
`results/event_decomposition.{h5,json}`, `diagnostics/gate_c_guard.json`,
`diagnostics/event_decomposition.json`, `REPORT.md`.

Headline: on one marked seed-100 dataset the joint arm recovers **both** planted
quantities with **both** truths — planted and realised — inside the 68% intervals,
and the information split is asymmetric: `f_AGN` is essentially all spatial
(J 0.0921 vs S 0.0915 in 68% width, intrinsic-only 4.85x worse), while the spin
offset is measured 2.38x better jointly than intrinsically alone (0.0394 vs 0.0938).

Exactly three scientific arms, one marked dataset, one likelihood build:

- [x] S: spatial-only — 101 cells, `logL_max` −4233.939503109996, `f` median 0.264977;
- [x] I: intrinsic-only — 41 x 61, `logL_max` −4249.502089201707, diagnostic construction;
- [x] J: joint spatial+intrinsic — 41 x 61, `logL_max` −4211.842592845020, MAP
      (`f` = 0.275, `dmu_chi` = +0.1075).

Measured, per criterion:

| criterion | the number that decides it | verdict |
|---|---|---|
| C1 joint recovery | `f` median 0.261653, 68% [0.216174, 0.308298], 90% [0.187170, 0.339609] — contains planted 0.30 AND realised 0.295. `dmu_chi` median 0.107386, 68% [0.088587, 0.127960], 90% [0.077621, 0.142695] — contains planted +0.100000 AND realised +0.111924 +/- 0.006815 | PASS |
| C2 null disfavoured only when warranted | `P(dmu_chi <= 0) = 4.85e-11` (arm J; arm I 3.37e-06), reported as a posterior probability under this model and grid and NOT converted to a sigma; marginal density at 0 is 3.04e-10 of peak; zero is not a grid node (neighbours −0.0050, +0.0025) | PASS |
| C3 spatial arm behaves as expected | vs the recorded unmarked Analysis-2 scan on the identical 101-node grid, `f` median moves −0.001522 (0.266499 -> 0.264977) and `logL` shifts by a nearly constant offset, mean −58.137, s.d. 0.291, range [−58.794, −57.625]; the arm has no access to the mark by construction (`dmu_chi` == 0) | PASS |
| C4 intrinsic arm behaves as expected | `dmu_chi` median +0.081723, 68% [0.041726, 0.135511], 90% [0.028444, 0.172627] — right sign, right scale, contains planted +0.100000 AND realised +0.111924. Read with the `q` confound below | PASS |
| C5 joint information is coherent | 68% widths measured, not assumed — `f`: S 0.0915, J 0.0921, I 0.4442; `dmu_chi`: J 0.0394, I 0.0938. J is 0.68% wider than S in `f`, diagnosed: J marginalises over `dmu_chi` where S fixes it, at posterior correlation −0.5544. No arm is pathologically broad and none shifts off truth | PASS |
| C6 selection validity | 23 of 2501 arm-J cells rejected, ALL at `f >= 0.750` and `dmu_chi >= +0.2350`; upper bound on the rejected posterior mass 3.86e-104. Over the cells carrying the posterior, min `N_eff`/threshold 85.5 at 90% of the mass, 70.5 at 99%, 55.3 at 99.9%, 30.7 at 99.999%. Arms I and S reject 0 (min 1.708x, 75.7x). Only the targeted lane was run, so the two-lane clause does not apply | PASS |

Three things this gate established that any later phase must carry.

* **Two truths, and they differ.** Planted `f_AGN` 0.30 against realised 0.295;
  planted `dmu_chi` +0.100000 against realised +0.111924 +/- 0.006815, of which
  +0.011924 predates the mark. The joint `dmu_chi` median falls BETWEEN them. No
  recovery statement is complete without naming which truth and which interval.
* **The `q` confound is live in the intrinsic channel.** Seed 100's detected set
  separates GAL from AGN in mass ratio (KS p = 0.0096; z-stratified Fisher 29.57,
  the largest of 41 seeds examined, median 7.54) and the model holds `q` identical
  in both branches. Arm I and the per-event `log BF_intrinsic` therefore do not
  measure the spin mark in isolation, and the C2 tail is conditional on a model
  that is wrong in this respect.
* **Arm I is a diagnostic, not a physical model**, and its thin `N_eff` floor
  (1.708x) is a consequence of that deliberate misspecification, not a warning
  about the production arm.

Required:

- [x] joint \(f_{\rm AGN}\) recovery (median 0.261653; both truths in the 68% interval);
- [x] joint \(\Delta\mu_\chi=+0.10\) recovery (median 0.107386; both truths in the 68% interval);
- [x] spatial-only result (`results/arm_S_spatial.{h5,json}`);
- [x] intrinsic-only result (`results/arm_I_intrinsic.{h5,json}`);
- [x] joint result (`results/arm_J_joint.{h5,json}`);
- [x] selection guard valid over posterior support (`diagnostics/gate_c_guard.json`);
- [x] event-level spatial/intrinsic evidence decomposition
      (`results/event_decomposition.{h5,json}`; identity verified against production
      to 0.0 absolute at the MAP);
- [x] production figures — `figs/fig_joint_f_dmu`, `fig_ablation_fagn`,
      `fig_ablation_dmu` and `fig_event_evidence_plane`, each as `.pdf` and `.png`,
      rendered by `scripts/make_figures.py`. `fig_selection_marked` (specification 11,
      optional) was not made: the selection diagnostics are reported numerically;
- [x] `REPORT.md`.

## Owner gate

Status: **REACHED — WORK STOPPED, AWAITING OWNER DECISION** (2026-09-18).

Gates A, B and C have all passed and the seed-100 package is closed. Per
specification 14 the run stops here; nothing proceeds automatically. The
completion line below has been issued, at the end of `REPORT.md`.

Still forbidden before explicit owner approval, unchanged:

- seeds 101/102/103/105;
- any additional realization, including a repeat draw of seed 100;
- effect-size ladders;
- free \(H_0\);
- mass marks;
- free common population parameters;
- incompleteness;
- GP/HSGP population differences;
- GWTC data.

Claims explicitly NOT established by this one realization (specification 15):
calibration across realizations, coverage, unbiasedness in expectation, any
sensitivity scaling with \(N\), anything about real BBH spins in AGN, any \(H_0\)
improvement, any statement about incomplete catalogs. Seed-100 closure establishes
implementation closure and proof of concept only.

Completion line required from the driver, and given:

> **OWNER GATE: seed-100 Analysis 8 is complete. I have not run additional realizations.**
