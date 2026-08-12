# RESULTS — experiment_dsmaster_4d_recheck

**Status (2026-08-10). ALL THREE ARMS COMPLETE.** `aggregate` and the legacy
`per_pixel` control (Results 3–5), and the `selection` arm on darksirens master
`0c5b3db` (Result 6). The mode question is settled: the data prefer `selection`
by ln B +12.1 over `aggregate` and +17.7 over `per_pixel`, and it is the only
one of the three that recovers the galaxy density anchor. What remains is the
owner gate on re-running analyses 3–6.

All numbers below: seed 100, rung m18, rstate 7, dynesty nlive 1000 dlogz 0.1,
darksirens `e8d5035` (origin/master), compared against analysis 5's
`campaign_m18_dynesty_s100` on `2b86a2d`. Truth: H0 67.74, log10n0 −3.0,
log10n0_c2 −5.0, realised f_AGN 0.295.

---

## Result 1 — under the LEGACY estimator, master is bit-identical

Rebuilding the analysis-5 closure on master with `c_mode="per_pixel"` and
evaluating at stored analysis-3 grid cells gives **max|diff| = 0.000e+00** at
all three probe cells, including both f = 0 and f = 1 endpoints.

So none of the fourteen completeness commits changed the legacy code path. Every
difference reported below is the **estimator**, not code drift. The reasoning
per commit is tabulated in `DESIGN.md`; the one change that does reach the
homogeneous non-LSS path (9e64001, expected-side smoothing truncated at the
survey depth) is gated on `z_depth is not None`, and none of our ten survey
files carries a `z_depth` attribute.

**Caveat on scope.** Three cells, both anchors pinned at truth. The 4D posterior
does not live there — see Result 3.

## Result 2 — switching to `aggregate` reshapes the likelihood, hugely and unevenly

Same cells, `c_mode="aggregate"`, both anchors still at truth. Reproduced on a
GPU and an independent CPU node to the last printed digit:

| f_AGN | per_pixel | aggregate | Δ log L |
|---|---|---|---|
| 0.0 (pure galaxy) | −4238.68 | −4234.24 | **+4.44** |
| 0.5 | −4229.35 | −4220.09 | **+9.26** |
| 1.0 (pure AGN) | −4274.26 | −5019.78 | **−745.52** |

The pure-AGN endpoint collapses. Mechanism: at m18 the AGN catalog is ~9.6 %
complete and AGN are intrinsically sparse (`n0 = 1e-5 Mpc^-3`), so most pixels
hold few or no catalogued AGN. `per_pixel`'s clip at 1 masks that pixel by
pixel; one sky-wide `Cbar(z)` reports it honestly, the completion must carry far
more of the budget, and the pure-AGN hypothesis is penalised hard.

## Result 3 — the 4D fit: f_AGN gets much WORSE, H0 does not move

| | truth | `per_pixel` (A5, 2b86a2d) | `aggregate` (master) | Δ | × replicate floor |
|---|---|---|---|---|---|
| H0 | 67.74 | 69.6515 ± 1.258 | 69.7178 ± 1.230 | **+0.066** | 7 |
| log10n0 (GAL) | −3.0 | −1.8081 ± 0.538 | −2.2770 ± 0.579 | **−0.469** | 100 |
| log10n0_c2 (AGN) | −5.0 | −4.8852 ± 0.265 | −4.7150 ± 0.223 | **+0.170** | 147 |
| f_AGN | 0.295 | 0.3837 ± 0.228 | 0.5939 ± 0.240 | **+0.210** | 63 |

("Replicate floor" = analysis 5's own rstate-23 twin, i.e. dynesty's scatter.)

**f_AGN.** The bias more than triples, `+0.089 → +0.299`. The naive read of
Result 2 — mid-f gains, high-f collapses, so f should fall — is **wrong**, and
Result 2's fixed-truth anchors are why. With the anchors free, the AGN anchor
moves UP to −4.715 to buy back the completion budget the sky-aggregate `Cbar`
took away, and drags f_AGN up with it. The `log10n0_c2`–`f_AGN` correlation
tightens from +0.890 to **+0.908**: the same degeneracy analysis 5 identified,
running harder.

**H0.** +0.066 on a width of 1.23 — **0.05 σ**. It clears the replicate floor
but is physically nil. H0 has now survived completeness, density anchoring,
anchor freedom, relative completeness, and a change of completeness estimator.

**The galaxy anchor stops railing.** −1.81 → −2.28, i.e. 0.47 dex back toward
truth. Analysis 5 had to widen the GAL prior from [−4,−2] to [−4,−1] because
this rung railed; the sky-anchored budget relieves exactly that pathology. The
upstream warning that `Cbar ∝ 1/n0` through a single global clip could kink the
likelihood in `n0` did **not** show up as a sampling failure: 8613 iterations,
58572 calls, **zero guard rejections**, s/eval 0.1825, wall 10723 s.

## Result 4 — the data prefer `aggregate`

log Z −4205.24 → **−4199.62**, so **ln B ≈ +5.6** for `aggregate` over
`per_pixel` on identical data and priors. The worse f_AGN bias is therefore not
the new estimator being wrong; it is the legacy per-pixel clipping having
flattered the fit. Same-SHA control (Result 5): −4205.15 → −4199.62, **ln B =
+5.53** — the cross-SHA number was not an artefact of the version change.

## Result 5 — the control: master in `per_pixel` reproduces analysis 5

Full 4D fit on master, `c_mode="per_pixel"`, same seed/rung/rstate. Wiring check
**0.000e+00**; verdict **UNCHANGED**:

| | A5 (`2b86a2d`) | master (`e8d5035`) | Δ | × replicate floor |
|---|---|---|---|---|
| H0 | 69.6515 | 69.6398 | −0.0117 | 1.2 |
| log10n0 | −1.8081 | −1.8061 | +0.0020 | 0.4 |
| log10n0_c2 | −4.8852 | −4.8866 | −0.0015 | 1.3 |
| f_AGN | 0.3837 | 0.3840 | +0.0003 | 0.1 |
| log Z | −4205.24 | −4205.15 | +0.095 | 2.8 |

Every parameter sits at or below ~1.3× dynesty's own rstate-to-rstate scatter.
The 23 commits between the two SHAs are therefore invisible to the legacy path
at the posterior level, and **the whole of Result 3 is the estimator switch** —
not code drift, not the sampler, not the data.

---

## What this costs the campaign

**Analyses 3, 4, 5 and 6 are all `per_pixel` numbers.** On this single rung the
f_AGN answer moves by 0.21 — about 3.5× the σ(f_AGN) ≈ 0.06 those analyses
quote. Analysis 6's headline relation
`offset ≈ 0.067 + 0.124 log10(C_AGN/C_GAL)` and its sign flip are measurements
of an estimator that is no longer current, and should not go into the paper
without re-running under the current mode. The H0 conclusions look safe.

Held honestly: one rung, one seed, one rstate. The direction and size are far
outside sampling noise, but the campaign-wide restatement needs the ladder.

## Next

**All three arms are complete.** What remains is owner-gated.

1. ~~Control~~ **DONE** — Result 5, UNCHANGED. Result 3 is the estimator alone.
2. ~~`selection` arm~~ **DONE** — Result 6. The mode is settled: the data prefer
   it by ln B +12.1, and it is the only one that recovers the galaxy anchor.
3. **Whether analyses 3–6 are re-run under `selection`.** Owner gate, large
   recompute. Nothing upstream blocks it: the per-catalog machinery is pinned in
   Tier-0 (#346), the homogeneous-schechter + bright-truncation path we used is
   quadrature- and closed-loop-pinned (#347/#349), and `--validate_completion`
   is now c_mode-aware (#348), giving a cheap pre-run gate per rung.

   Weigh against it: f_AGN's bias **survives** the mode change (+0.089 →
   +0.299 → +0.215), so the ladder's f_AGN conclusions may move less than the
   anchor result suggests. Weigh for it: analysis 6's sign flip and its
   `0.067 + 0.124 log10(C_AGN/C_GAL)` relation are legacy-mode measurements of
   an estimator that is now demonstrably the worst of the three.

4. **Cheap mechanism probe (~1 GPU-h, not yet approved).** Rerun the selection
   arm at m19 and m20, where the AGN catalog is deeper. If the +0.29 dex AGN
   anchor offset shrinks with completeness, it is sparsity / shot noise in the
   anchor; if it holds flat, the AGN weighting conventions are the place to
   look. This is the cheapest available test of the Result 6 attribution and
   would sharpen — or overturn — the sparsity reading before any ladder
   decision. Suggested by darksirens-dev.

---

## The selection arm — state as of 2026-08-09

**K≥2 landed.** darksirens `5aa90fa` (PRs #342–#346) gives per-catalog magnitude
fits: catalog *k* anchors its own `M0hat{sfx}` / `sigma_M{sfx}`, `m_lim` is
pinned per catalog, anchoring is all-or-nothing, field weighting required (we
already run it). The original blocker is gone.

**A second one replaced it, now also resolved (darksirens PR #347).** Our LF is
truncated BRIGHT-ward of M*.
The mock draws absolute magnitudes from a Schechter cut at
`x_cut = 1.09079 L*` — the cut that makes the integrated density come out at
`1e-3 Mpc^-3`, i.e. `log10n0 = -3.0`, the inference truth. The density anchor
and the LF truncation are one construction. So

```
M_faint_offset = M_B_faint_limit - M_B_star = -20.56435 - (-20.47) = -0.0944
```

and darksirens refused a negative offset in three places on the reasoning that
the faint cutoff lies faint-ward of M*. Ours does not. `c_sel_schechter` forms
`x_faint = 10^(-0.4 * M_faint_offset)`, which for our offset is **1.09079 —
exactly our x_cut**, and both incomplete-gamma arguments stay positive for any
real offset, so the refusal was a convention rather than a constraint. Raised
upstream and implemented in PR #347.

**Two flags, two different jobs — and the one that fixes alpha is the cut.**
`M_faint_offset` never enters the fit likelihood (it is deliberately unfitted,
`meta["m_faint_offset_constrained"] = False`); it only sets the *consumed*
curve's denominator. What inverted our alpha was the SAMPLE being
bright-truncated while the fit modelled an LF with no faint edge. The fit-side
answer is `--m_faint_cut`, which restricts the per-galaxy normalisation to
`x >= max(x_lim, x_cut)`. Upstream now refuses a negative offset without a cut,
fail-closed, precisely because that pairing failed *silently*:

```
--m_faint_offset = -2.5 log10(x_cut)              = -0.094353   [no h]
--m_faint_cut    = M_B_faint_limit - 5 log10 h    = -19.718579  [h-SCALED]
truth Mstar_hat  = M_B_star        - 5 log10 h    = -19.624226
truth alpha      =                                  -1.07
```

`scripts/lf_constants.py` derives all four from the seed's own
`glass_field_meta.json` and asserts the identity
`m_faint_cut - Mstar_hat == m_faint_offset`. Never retype them: the cut is
h-scaled and the offset is not, so a botched `5 log10 h` shifts the fit support
by 0.85 mag.

**What the un-cut fit cost, measured.** At the only setting the old guard
permitted (`--m_faint_offset 5.0`, no cut), on `5aa90fa`:

| | AGN (n=8,273) | galaxies (n=821,444) | truth |
|---|---|---|---|
| Mstar_hat | −18.61729 ± 0.02053 | −18.61163 ± 0.00206 | −19.625 (h-scaled) |
| alpha | **+3.22347** ± 0.09669 | **+3.23410** ± 0.00975 | **−1.07** |

The faint-end slope comes out *positive* — the fit explaining a truncated
sample with an untruncated model. darksirens' own diagnostic agrees on both
catalogs: *"0 galaxies lie faint-ward of it"*. A Gaussian fit is no escape
either: skew −1.07, KS D = 0.107, and a hard faint edge at M = −20.564 smeared
over ~0.4 mag at the 99th percentile — the faint end being the entire
completion budget.

Note the two fits agree to **0.006 mag in Mstar_hat and 0.011 in alpha**. Both
tracers share one LF by construction (AGN carry their host galaxy's apparent
magnitude), so the "one completeness denominator per run" pin that K≥2 assumes
is measured, not assumed, for this configuration.

**Owner decision (2026-08-09): prep only, wait for the matched Schechter.** The
Gaussian ablation is deliberately NOT being run — it would confound the c_mode
question with LF misspecification, which is the one thing this experiment
exists to separate.

### The calibration fits (darksirens master `0c5b3db`)

Both catalogs, m18, seed 100, `--family schechter`,
`--m_faint_offset -0.0943529` `--m_faint_cut -19.7185789`, fitted on the
**true-redshift** surveys:

| tracer | param | fitted | σ | truth | pull |
|---|---|---|---|---|---|
| gal (821,361) | Mstar_hat | −19.61578 | 0.00512 | −19.624226 | +1.65 |
| gal | alpha | −1.04942 | 0.01177 | −1.07 | +1.75 |
| agn (8,274) | Mstar_hat | −19.62614 | 0.05102 | −19.624226 | −0.04 |
| agn | alpha | −1.05468 | 0.11644 | −1.07 | +0.13 |

`worst |pull| = 1.75 — PASS`. The AGN catalog is an independent draw from the
same LF and lands at −0.04/+0.13, so the galaxy pull reads as this
realisation rather than a residual systematic. Both fits together take **14 s**
(they took >2 h and never converged before the upstream sufficient-statistics
rewrite; the same fix repaired an `fatol=1e-10` that was absolute while the NLL
scales with N, so large-N fits had been grinding to maxiter and would have
returned non-converged theta wearing credible error bars).

The four LF labels are **pinned** at these centres in the arm, exactly as
`delta`/`sigma_kde` are pinned in every arm, so all three arms sample the same
four parameters and a posterior difference is the estimator rather than a wider
marginalisation. Upstream bounds the alternative: `sd(Mstar_hat) = 5.1e-3` mag
moves `C_sel` by ~0.5% at the transition, ~50× below the +0.21 f_AGN effect
being measured.

### The probe: all three estimators at the same three cells

`H0 = 75`, `log10n0_c2 = −5`, both anchors at truth:

| f_AGN | per_pixel | aggregate | selection | sel − pp |
|---|---|---|---|---|
| 0.0 (pure galaxy) | −4238.68 | −4234.24 | −4217.16 | **+21.52** |
| 0.5 | −4229.35 | −4220.09 | −4206.24 | **+23.10** |
| 1.0 (pure AGN) | −4274.26 | −5019.78 | −5148.89 | **−874.63** |

`selection` follows `aggregate`'s shape but harder: the low- and mid-f cells
gain ~2× more, and the pure-AGN endpoint collapses further still (−874.6 against
−745.5).

**The same caveat applies, and it has already caught us once.** These cells pin
both anchors at truth, and the posterior does not live there. The identical
probe under `aggregate` implied f_AGN should FALL; the free-anchor fit moved it
UP by +0.21. Nothing above should be read as a prediction for the arm.

### Result 6 — the `selection` arm: the GALAXY ANCHOR COMES BACK

m18, seed 100, rstate 7, darksirens master `0c5b3db`, 10,874 iterations,
`dlogz = 9.7e-05`, zero guard rejections.

| | truth | `per_pixel` | `aggregate` | **`selection`** |
|---|---|---|---|---|
| H0 | 67.74 | 69.640 ± 1.234 | 69.718 ± 1.229 | **69.121 ± 1.111** |
| log10n0 (GAL) | −3.0 | −1.806 ± 0.539 | −2.277 ± 0.579 | **−3.126 ± 0.291** |
| log10n0_c2 (AGN) | −5.0 | −4.887 ± 0.265 | −4.715 ± 0.223 | **−4.714 ± 0.355** |
| f_AGN | 0.295 | 0.384 ± 0.226 | 0.594 ± 0.240 | **0.510 ± 0.263** |
| log Z | | −4205.145 | −4199.619 | **−4187.476** |

**The galaxy anchor is recovered.** `log10n0 = −3.126 ± 0.291` against a truth
of −3.0 — an offset of −0.126, or **−0.43σ**. The other two estimators miss it
by +1.194 dex (per_pixel, railing against its prior) and +0.723 (aggregate).
The width also halves, 0.54/0.58 → 0.29. This is the pathology that forced
analysis 5 to widen the GAL prior from [−4,−2] to [−4,−1], and under the
parametric completeness it simply is not there: the anchor was never a property
of the data, it was the non-parametric estimators' way of absorbing a
completeness they could not model.

**The data strongly prefer it.** ln B = **+12.1** over `aggregate` and
**+17.7** over `per_pixel`. The comparison is legitimate — all three arms
sample the same four parameters over identical flat priors on identical data
(the four LF labels are pinned, not sampled; `samples` is (N, 4) in every arm).

**f_AGN is still biased, +0.215**, landing between per_pixel's +0.089 and
aggregate's +0.299 and much nearer the latter. So the f_AGN bias is NOT an
artefact of the sky-aggregate estimator that a better completeness model
removes. The degeneracy that carries it is untouched:
`corr(log10n0_c2, f_AGN)` = +0.890 / +0.908 / **+0.903** across the three
modes. The AGN anchor sits at −4.714 in both `aggregate` and `selection`, to
three decimals — two very different estimators agreeing on the same wrong
answer, which points at the AGN catalog's sparsity rather than at the
completeness model.

### Result 6b — what the posterior actually looks like (figs 1–2)

`figs/fig1_corner_three_estimators.{pdf,png}` overlays the three joint
posteriors; `figs/fig2_offsets_from_truth.{pdf,png}` is the accuracy view.
Rendered by `scripts/make_figures.py`. Looking at them qualifies the table
above in a way the medians alone hide:

| arm | param | median | 68% CI | pull | truth in 68 / 90 |
|---|---|---|---|---|---|
| per_pixel | log10n0 | −1.806 | [−2.412, −1.257] | **+2.07** | **n / n** |
| aggregate | log10n0 | −2.277 | [−2.772, −1.512] | +1.15 | n / Y |
| **selection** | log10n0 | −3.126 | [−3.508, −2.902] | **−0.42** | **Y / Y** |
| per_pixel | f_AGN | 0.384 | [0.211, 0.677] | +0.38 | Y / Y |
| aggregate | f_AGN | 0.594 | [0.311, 0.859] | +1.09 | n / Y |
| selection | f_AGN | 0.510 | [0.175, 0.793] | +0.69 | Y / Y |
| per_pixel | H0 | 69.640 | [68.471, 70.827] | +1.61 | n / Y |
| aggregate | H0 | 69.718 | [68.541, 70.916] | +1.67 | n / Y |
| selection | H0 | 69.121 | [67.976, 70.165] | +1.26 | n / Y |

**Only the galaxy anchor is a statistically significant estimator effect.**
`per_pixel` puts truth outside its 90% interval — it is the one genuinely
failing measurement in the set, and the corner shows why: its `log10n0`
marginal is piled against the prior edge at −1.0, i.e. railing, not measuring.
`selection` recovers a clean interior peak straddling truth.

**f_AGN is weakly constrained in this configuration, and that governs how the
f_AGN numbers may be quoted.** σ(f_AGN) ≈ 0.23–0.26 against a flat [0,1] prior,
and fig 1's 1-D marginals run into both prior edges. All three pulls are ≤1.1σ
and truth sits inside every 90% interval. So the +0.089 → +0.299 → +0.215
sequence is a shift of medians of a barely-informative posterior; the three
arms are statistically consistent with each other AND with truth in f_AGN.
Earlier phrasing in this file — "the bias TRIPLES" — is about medians and
should not be read as a significant bias in this free-anchor configuration.

**This does NOT transfer to analyses 3–6.** Those quote σ(f_AGN) ≈ 0.06 because
they scan grids with the anchors FIXED. Freeing both anchors costs a factor ~4
in f_AGN precision, so this arm cannot say whether their f_AGN conclusions
move — only that the completeness estimator does not shift f_AGN by more than
this configuration can resolve.

**The degeneracy is estimator-independent.** The `log10n0_c2`–`f_AGN` banana in
fig 1 is visually the same curve in all three arms (+0.890 / +0.908 / +0.903),
which is the clearest single picture of why the AGN anchor and f_AGN move
together.

**H0, read differentially.** Seed 100's own complete-catalog draw is +1.48
(analysis 2's five-seed closure; the campaign-wide offset is this seed's
realisation, not a bias — quote H0 differentially, never absolutely). Against
that reference:

| | H0 offset | vs seed 100's own draw |
|---|---|---|
| per_pixel | +1.900 | +0.42 |
| aggregate | +1.978 | +0.50 |
| **selection** | **+1.381** | **−0.10** |

`selection` is the only mode with no residual incompleteness shift in H0. But
this is ~0.4σ on a 1.1 width — a direction, not a detection, and it should be
quoted as "the parametric completeness leaves no H0 residual at our precision",
not as a measured improvement.

Note this is the first estimator change to move H0 at all appreciably:
per_pixel → aggregate was 0.05σ, aggregate → selection is 0.54σ. H0's
robustness across completeness estimators is weaker than the first two arms
suggested.

### What is built and launch-ready

| piece | state |
|---|---|
| `src/darksirens-sel` worktree at **`0c5b3db`** | done — `darksirens-master` left pinned at `e8d5035` so the finished arms stay reproducible |
| `gal_app_mag` on the m18 surveys | done — `scripts/add_gal_app_mag.py`, non-destructive into `data/seed100/surveys_galprop/`, four original datasets asserted bit-identical before writing |
| `surveys_truez/` | done — same script, `--z_column z`. The bit-identity gate cannot apply (rows are z-sorted within pixels, so a different z column is a different permutation); co-indexing is by construction and checked instead by identical per-pixel counts and an identical magnitude MULTISET |
| `scripts/lf_constants.py` | done — derives offset, cut and both truths from the seed's own `glass_field_meta.json` and asserts `m_faint_cut − Mstar_hat == m_faint_offset`; nothing is retyped, which is the trap (the cut is h-scaled, the offset is not) |
| `sample_4d.py --selection_fit` | done — per-catalog paths, `_resolve_selection_fits` before `build_parameter_space` (darksirens' own ordering), `selection_prior`/`selection_family` threaded, fits and pinned theta recorded in the output JSON. Gated on `c_mode == "selection"`, so the two finished arms take byte-for-byte their existing path |
| `scripts/fit_selection.sh` | done — derives the constants, fits AGN first (cheap validation before the expensive catalog), then runs the acceptance test |
| `scripts/check_fit_recovery.py` | done — pulls vs the mock's generative truth, σ from the fit JSON's 2×2 `cov`, exits nonzero past 3σ |
| `scripts/submit_selection.sbatch` | done — RITA-GPU, refuses to start without both fits present |
| the fits | **done** — PASS, worst pull 1.75σ |
| the arm | **RUNNING** — job 1118740 on master `0c5b3db` |

Reproduce with `./scripts/fit_selection.sh && sbatch
scripts/submit_selection.sbatch`. Budget ~3 h on a RITA A100, from the
per_pixel arm's 3 h 12 m.
