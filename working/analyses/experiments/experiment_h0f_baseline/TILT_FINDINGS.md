# The mechanism of the H0 tilt in the GLASS two-tracer baseline

Diagnosis of the result: with complete GLASS catalogs (GAL b=1.2 + AGN b=2.0),
K=2 field-mode mixture scans at fixed planted f_AGN recover H0 low, with the
deficit growing in f_AGN — medians −0.92 km/s/Mpc at f=0.307 and −3.32 at
f=0.703 (truth 67.74) — while f_AGN itself recovers well.

**Verdict in one sentence.** The tilt is the *incomplete cancellation of two
spurious H0 levers created by the same model inconsistency* — the mock detects
on TRUE redshift (z ≤ 1) while the analysis likelihood expresses both its
selection integral and its host prior in luminosity distance against a catalog
whose support continues to z ≈ 1.5: the injection-based selection term acquires
a spurious slope worth **−4.0 to −4.3 km/s/Mpc** (nearly independent of f_AGN),
the per-event numerator acquires a spurious opposite pull from catalog support
above the detection horizon worth **+3.7 (f=0.307) / +1.2 (f=0.703)**, and the
f_AGN growth of the net bias is entirely the *decay of the numerator's pull* as
the mixture weight moves onto the sparse AGN catalog, whose near-delta-function
kernels anchor the numerator to the true H0. Repairing both levers recovers the
truth within statistical error at both f.

All numbers below come from a term-by-term standalone rebuild of the production
likelihood (`scripts/tilt_terms.py`) validated against the production
decomposition to 0.0008 nats (f=0.307) / 0.0013 nats (f=0.703) in curve shape
and to machine precision in ln mu — i.e. the model *is* the production
likelihood, evaluated with counterfactual switches.

## The budget

Peak shifts of the total log-likelihood in km/s/Mpc (quad-refined; H0 grid
55–80, 51 pts; `results/tilt_budget.json`):

| quantity | f = 0.307 | f = 0.703 |
|---|---|---|
| **measured total offset** (production) | **−0.92** | **−3.32** |
| model total offset (this rebuild) | −0.92 | −3.32 |
| = numerator peak offset | +3.09 | +0.95 |
| + selection-term shift | −4.01 | −4.27 |
| numerator offset: z>1 catalog-leak component | +3.72 | +1.19 |
| numerator offset: PE Monte-Carlo (delta-method) bias | −0.01 | +0.07 |
| numerator offset: residual (statistical) | −0.63 (σ≈0.22) | −0.24 (grid res.) |
| **repaired estimator** (z≤1-truncated numerator prior + H0-independent β) | **67.11 (−0.63 ± 0.22)** | **67.5 (−0.24, boundary-limited)** |

The decomposition `total = numerator + selection` is exact (the two rows sum to
the total by construction); the numerator sub-rows are counterfactual
re-peakings. The repaired-estimator row is the closure test: with the catalog
prior truncated at the detection horizon AND a flat (correct) selection
normalization, the peak returns to the truth within ~1–3σ statistical at both
planted fractions. The two levers fully account for the tilt.

## Mechanism 1 — selection: a dL-threshold μ(H0) for a z-threshold detection
(−4.0 / −4.3 km/s/Mpc; the dominant, nearly f-independent lever)

The mock's detection is a hard cut on true redshift, `z ≤ 1.0`
(`injections_cat.h5` attr `detection_rule='true_z_cut'`). For that rule the
correct per-tracer selection normalization is β = P_cat(z ≤ 1), which does not
depend on H0. The injection-based estimator instead works in (m1det, q, dL):
because injections carry exact dL(z; H0_true), the detected set is exactly
`dL ≤ dLmax = dL(1; 67.74) = 6798 Mpc`, and under the analysis model at trial
H0 that surface is `z ≤ z*(H0)` with dL(z*; H0) = dLmax:
z*(60) = 0.906, z*(67.74) = 1.000, z*(75) = 1.086. As H0 rises, z* climbs the
steeply rising GLASS dN/dz (which peaks at z ≈ 1.4), so μ(H0) rises:

- measured d ln μ/dH0 at truth: **+0.0245** (f=0.307) / **+0.0255** (f=0.703),
  entering the likelihood multiplied by −N_obs = −1000: a −24 nats per km/s/Mpc
  lever.
- The zero-free-parameter analytic model μ(H0) = Σ_k α_k F_k(z*(H0)), with F_k
  the catalog CDF weighted by the population's (1+z)^(γ−1) (γ=0), predicts
  **+0.0234 / +0.0239** — 96% of the measured slope; the whole ln μ curve is
  reproduced to ≤ 0.05 nats over its 0.36-nat range
  (`results/tilt_selection_model.json`, `figs/tilt_selection_model.*`).
  Re-peaking the numerator against the analytic −1000·ln μ_model reproduces
  −3.89 of the −4.01 selection shift at f=0.307.
- Channel split (counterfactual freezes): of the +0.0242 slope, **+0.0172 is
  the catalog-prior channel** (z(dL) sweeping the dN/dz) and **+0.0069 the
  mass/Jacobian channel** (m1src = m1det/(1+z(dL,H0)) moving through the fixed
  powerlaw+peak model and the ddL/dz·(1+z) Jacobian).

**Not an injection-coverage artifact.** The slope is proposal-independent:
recomputing μ(H0) with the *untargeted* injection set (`injections.h5`,
0.9 pop + 0.1 uniform proposal, no AGN-catalog lane; independent seed) gives a
linear-fit slope over H0∈[60,75] of 0.02228 vs 0.02232 for the catalog-targeted
set at f=0.307 — identical to 0.2% — and 0.02224 vs 0.02208 at f=0.703
(`results/tilt_mu_proposal_independence.json`,
`results/tilt_mu_injB_compare.json`). The
injection estimator is faithfully computing the model's μ(H0); the model's μ is
what is wrong for this detection rule. (The old attribution "injection-based
μ(H0) vs analytic H0-independent β" was therefore right about *where* the term
comes from but the distortion is not about injection coverage near the edge —
it is intrinsic to expressing a z-cut as a dL-cut against a catalog that keeps
rising above the horizon.)

## Mechanism 2 — numerator: catalog support above the detection horizon
(+3.7 / +1.2 km/s/Mpc, opposing; carries all of the f-dependence)

Events exist only at z ≤ 1, but the catalogs continue to z ≈ 1.5 with dN/dz
still rising (57% of both catalogs' objects lie above z = 1). The 10% dL PE
posteriors of near-horizon events map part of their mass to z(dL; H0) > 1,
where the analysis' host prior has (spurious, for detected events) support;
raising H0 maps *more* PE mass *higher up* the rising dN/dz, so the numerator
grows with H0. The mean PE posterior mass beyond z = 1 is 8.1% at truth and
17.8% at H0=75 (f=0.307 set; 5.4% → 13.2% for f=0.703;
`figs/tilt_leak.*`). Masking the catalog prior above z = 1 moves the numerator
peak by −3.72 / −1.19, i.e. it removes essentially the whole numerator bias
(+3.09 → −0.63, +0.95 → −0.24).

- Slope decomposition at truth (f=0.307): numerator slope +16.1 nats per
  km/s/Mpc = **+12.5 catalog-prior channel + 3.5 mass/Jacobian (spectral)
  channel** (additive to 0.1%). At f=0.703: +8.2 = 4.3 + 3.9.
- **The z≈1.5 catalog/grid edge itself is irrelevant in the scanned range**:
  PE mass beyond z=1.4 is ~1e-4 at truth (8e-4 at H0=75), the fraction of
  samples clipped by the z=1.5 support boundary is ≤ 3e-5 (3e-4 at 75). The
  operative boundary is the *detection horizon z=1 vs catalog support above
  it*, not the survey edge. (This reframes measured fact (1): truncating the
  catalogs at z ≤ 1 moved H0 by −4.09 because it removes the numerator's +3.7
  upward pull while leaving the −4 selection slope uncompensated — the model
  reproduces that experiment: masked-at-z≤1 totals sit at −5.28 / −3.75 from
  truth, i.e. shifts of −4.36 / −0.43 from the full-model peaks.)

## Why the deficit grows with f_AGN

The two catalogs have *identical* dN/dz shapes (57.2% vs 57.1% above z=1), so
the growth is not the AGN redshift distribution. It is the *sparseness* of the
AGN catalog in the mixture prior:

- 11,724 AGN over 49,152 pixels (0.24/pixel; 38,719 empty pixels) with
  σ_z = 0.003(1+z) kernels: for any single event the AGN branch is a handful of
  near-delta spikes, essentially a noisy counterpart prior. It anchors H0: the
  numerator of AGN-hosted events peaks at 68.79 (+1.05) at f=0.307 and 68.38
  (+0.64) at f=0.703, while GAL-hosted events' numerator peaks at **79.0
  (+11.3)** at both f (the pure-GAL smooth prior has no interior numerator peak
  below the 80 grid edge).
- As f_AGN grows the mixture weight moves onto the anchored branch (the mean
  posterior AGN-branch weight at truth is 0.32 at f=0.307 vs 0.69 at f=0.703),
  so the numerator's spurious upward pull collapses (+3.09 → +0.95) and its
  curvature doubles (total curvature −10.1 → −18.4 nats/(km/s/Mpc)²), while
  the selection lever is set by the (shape-identical) mixture dN/dz and stays
  at −4.0 to −4.3. Net: −0.92 → −3.32. **The near-recovery at f=0.307 is an
  accidental cancellation, not correctness** (consistent with the README's
  earlier warning).

## Eliminated / minor mechanisms

- **(b) PE Monte-Carlo (delta-method) variance bias**: Σ_i σ²_i = 16.3 nats at
  truth (f=0.307) and 43.8 (f=0.703; larger because of the spiky AGN KDE), but
  its H0 slope is small; the corrected-likelihood peak moves by **+0.01 /
  −0.07 km/s/Mpc**. Negligible here (as in the matched mock, −0.08).
- **(c) injection coverage near the z edge**: ruled out by proposal
  independence (above). Coverage of the detected region is complete by
  construction (proposal z ≤ 1.2 ⊃ detected z ≤ 1; model support at the scan's
  H0 extremes stays inside it).
- **z=1.5 grid/catalog-edge clipping**: ≤ 3e-5 of PE mass at truth; no
  measurable lever in 55 ≤ H0 ≤ 80.
- **Host rate weighting**: previously measured ≤ 0.16 (RESULTS.md); not
  revisited.

- **Sparse-selection MC roughness at f=0.703** (sub-effect worth flagging, but
  not the tilt): ×1000 amplification of ln μ MC wiggles makes the f=0.703 total
  surface locally rough — near-degenerate structure across 64–67 (secondary
  local maximum at 66.5, −5.2 nats below the 64.5 peak). The *bias itself is
  robust to the selection realization*: recomputing the total against three
  injection sets gives peaks of −3.32 (targeted, seed 52001), −3.48 (targeted,
  independent seed 52002) and −2.93 (untargeted proposal, no AGN lane) — a
  −3.2 ± 0.3 offset with only the shelf detail moving
  (`results/tilt_mu_injB_compare.json`).

## Files

- `scripts/tilt_terms.py` — standalone term-by-term likelihood rebuild
  (numerator + selection, counterfactual variants, per-event diagnostics);
  validated against the production decomposition curves.
- `scripts/tilt_validate_pe.py` — production per-event lnZ spy used for the
  validation (`results/tilt_validate_pe_fagn0.3.json`).
- `scripts/tilt_selection_model.py` — analytic dL-threshold μ(H0) model.
- `scripts/tilt_budget.py` — budget assembly (`results/tilt_budget.json`).
- `scripts/tilt_figures.py` — `figs/tilt_decomposition.*`,
  `figs/tilt_counterfactuals.*`, `figs/tilt_selection_model.*`,
  `figs/tilt_leak.*`.
- `results/tilt_terms_fagn0.{3,7}.h5` — curves + per-event matrices.
- `results/tilt_mu_altinj_fagn0.3.h5`, `results/tilt_mu_injB_fagn0.7.h5`,
  `results/tilt_mu_altinj_fagn0.7.h5` — μ(H0) with alternative injection sets.
- `results/tilt_repaired_estimator.json`,
  `results/tilt_predicted_selection_shift.json`,
  `results/tilt_mu_proposal_independence.json`,
  `results/tilt_mu_injB_compare.json`.

## Consequences for the paper / next steps

1. The correct fix is a selection treatment matched to the detection rule
   (analytic β = P_cat(z ≤ z_det) per tracer, H0-independent) TOGETHER WITH a
   numerator host prior truncated at the detection horizon — either alone makes
   the bias worse (flat β alone: +3.1; truncation alone: −5.3). The repaired
   pair recovers truth at both f (closure above).
2. For real-data configurations the analogue statement: if the detected-event
   horizon lies well inside the catalog's redshift support and PE distance
   posteriors are broad (σ_dL/dL ~ 10%), the numerator picks up spurious H0
   information of order the PE-mass-leak slope; the selection function must be
   consistent with the true detection variable.
3. The residual −0.63 ± 0.22 (f=0.307) after repair is 2.8σ if taken at face
   value but is a single event-set realization; the matched-mock experiment's
   −0.8 residual is the right place to chase any remaining shared mechanism.
   MC delta-method bias is not it (measured ≤ 0.07 here).
