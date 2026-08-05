# RESULTS — experiment_matched_mock (K=1 closure)

**UPDATE (exact-likelihood oracle, see ORACLE_FINDINGS.md): the 5σ residual is
explained.**  An exact-likelihood oracle (no PE samples, no Monte Carlo)
evaluated on all 20 realisations recovers H0 at −0.489 ± 0.077 itself: the
dominant cause is a THIRD mock defect of the same family as the first two —
`sigma_ang` is a deterministic function of the latent true parameters, so the
sky-noise width is an H0-sensitive observable that the fixed-width PE clouds
(and darksirens' pixel-histogram sky treatment) cannot represent.  Making the
width a function of the observed data restores closure (parametric bootstrap
−0.62 ± 0.06 → −0.06 ± 0.07; per-event score identity restored).  The
darksirens-estimator overhead on top of the exact likelihood is −0.31 ± 0.13
(Farr 1/Neff(H0) term −0.12 systematic; the rest zero-mean noise, incl. a
previously unquantified ±0.36/realisation from the shared-injection ×
catalog-KDE interaction in mu_hat).  Upstream fix branch:
`fix/mock-observable-sky-width` (PR #335).  **The real (non-bootstrap) rerun
of the 20-realisation campaign with the fix lands at −0.393 ± 0.149 — the
predicted estimator overhead alone; see the CLOSURE section below.**

## CLOSURE — 2026-07-30: the sigma_ang fix, rerun for real (not bootstrap)

The obs-arm EVENTS of all 20 realisations were regenerated with the PR #335
convention mirrored in `scripts/build_obsdet_mock.py --sky_width observed`
(gmd helpers imported from the `darksirens-oraclefix` worktree @ 853ded3):
obs_dL/obs_m1det/obs_m2det drawn FIRST, `sigma_ang = clip(35/rho_opt(obs), 1,
12)°` derived from the OBSERVED amplitude (same snr_ref_control = 11.5 scale),
then the sky offsets.  The rng stream is untouched, so **the detected sets are
bit-identical to the old obs arm in all 20 realisations** (verified dataset by
dataset: only ra/dec/obs_sigma_ang columns differ; truth/snr, masses,
distances identical; obs_sigma_ang recomputable from the stored observables to
0).  Catalogs, surveys and `sel_obs.h5` reused; same grid (58–78, 161), same
guard (`hard`, max_likelihood_variance 1e6), same estimator.

| arm | mean H0 offset (n = 20) | sd | sigma from 0 | cells rejected |
|---|---|---|---|---|
| ctrl (gmd rule) | −1.570 ± 0.184 | 0.821 | 8.5σ | 77/3220 |
| obs (PR #334, latent sky width) | −0.802 ± 0.162 | 0.722 | 5.0σ | 14/3220 |
| **fix (PR #335, observed sky width)** | **−0.393 ± 0.149** | 0.666 | 2.6σ | 14/3220 |

Paired differences (catalog-paired, so tight): fix − obs = **+0.409 ± 0.044**
(9.3σ; the oracle predicted the darksirens-visible gain at ≈ +0.39: the exact
−0.49 sky-defect removal minus the +0.10 nside-16 pixelation cancellation).
fix − ctrl = +1.177 ± 0.235.  fix − oracle(old data) = +0.097 ± 0.124.

**The closure number matches the prediction.**  The oracle predicted the fix
arm should land at the darksirens estimator overhead alone, −0.15..−0.35
(paired ds − exact on the old data: −0.31 ± 0.13); measured **−0.393 ± 0.149**,
i.e. within 0.3σ of the −0.35 band edge and 0.5σ of the −0.31 central value.
Budget of the remaining −0.39: Farr 1/Neff(H0) term −0.12 (systematic),
mass-width latents −0.05, leaving −0.22 ± 0.15 unattributed — consistent with
zero at 1.5σ and with the known zero-mean per-realisation noise terms (mu_hat
injection×catalog-KDE ±0.36, PE-MC ±0.4; both common-mode-shared across arms
but not across the oracle pairing).  **No new mechanism is required.**  The
residual 2.6σ is exactly the estimator overhead already characterised in
ORACLE_FINDINGS.md §10; the next lever is estimator-side (neutralise the Farr
term in grid-scan closures, or budget Neff properly), not the mock.

Truth in CI68 6/20, CI90 10/20 — per-seed intervals remain ~2× too narrow, as
expected: catalog sample variance is untouched by this fix.

Files: `data_derived/obsdet/ev_obsfix_<key>.h5` (events),
`results/obsdet_fix_<key>.json` (scans), `results/obsdet_fix_summary.json`
(aggregate incl. per-realisation table and all paired differences),
`figs/fig_obsfix_closure.pdf` (ladder −1.57 → −0.80 → −0.39 with
per-realisation points), `figs/fig_obsfix_budget.pdf` (waterfall).
Scripts: `scripts/run_obsfix_events.sh`, `scripts/run_obsfix_scans.sh`,
`scripts/analyze_obsdet_fix.py`; generator patch in
`scripts/build_obsdet_mock.py` (`--sky_width`, default `latent` reproduces
every pre-existing output bit-identically — verified on n4201).

**Status (superseded): two real defects found in the mock generator, together worth about half the
bias; a 5σ residual is unexplained.** Both are cases of the mock not being a draw from
the model the inference assumes.

1. **PE construction** (darksirens PR #332, open): the mock's PE samples were clouds
   centred on the TRUE parameters stored with `p_pe = 1`, i.e. mislabelled as
   flat-prior posteriors. Correcting them closes the σ → 0 limit essentially exactly
   but does not remove the bias at realistic noise.
2. **The selection did not act on the data the posterior conditions on** — measured
   below, and fixed upstream on branch `fix/mock-shared-noise-draw`. Worth
   **+0.77 ± 0.23** of the offset (3.4σ), i.e. 49% of it.

After both, **−0.80 ± 0.16 over 20 catalog realisations, still 5.0σ low.**

## The detection/PE decoupling, measured

`_draw_events_until_detected` keeps a source if `_network_snr(...) ≥ 8`, where
`_network_snr` multiplies the true amplitude by a fresh `Beta(2,5)^0.5` projection
drawn per source; `_posterior_samples` then draws a *separate* noise realisation for
the observation the PE conditions on. So detection is decided by a latent `w` that
never enters the data. Writing out the detected-set likelihood,

```
p(d, det | θ) = ∫ dw p(w) 1[det(θ,w)] p(d|θ) = P(det|θ) p(d|θ)
p({dᵢ} | Λ, det) = Πᵢ [ ∫ dθ p(dᵢ|θ) p(θ|Λ) P(det|θ) ] / μ(Λ)
```

the correct per-event integrand carries an extra `P(det|θ)` **inside** the integral.
darksirens — like every population code — computes `Πᵢ[∫ p(dᵢ|θ) p(θ|Λ) dθ]/μ^N`, which
is right only when detection is a deterministic function of the data, so that
`1[det(dᵢ)] = 1` and it drops out. The inference is not wrong; the mock is not a draw
from it.

**The A/B.** `scripts/build_obsdet_mock.py` runs two arms that share the catalog, the
event seed and every ancillary uncertainty model, so a difference is the detection rule
alone:

* **ctrl** — gmd's rule verbatim.
* **obs** — one measurement per source, `ρ_obs = snr_ref (M_c,det,obs/30)^{5/6}
  (1000/d_obs)` thresholded, and *that same measurement* handed to the PE.

The projection latent cannot be kept in the `obs` arm: keeping it would leave detection
depending on a variable absent from the data, which is a different mis-specification
rather than a fix. Dropping it raises the detected fraction 5.75×, so `snr_ref` was
calibrated (to 6.278) to reproduce the control's detected fraction — the arms' event
populations then match to ~5% in median redshift. Injections were regenerated per arm
under the same rule (120M draws each), storing TRUE parameters, as a real injection
campaign does.

| arm | mean H₀ offset (n = 20) | sd | significance | cells rejected |
|---|---|---|---|---|
| ctrl (current) | **−1.570 ± 0.184** | 0.821 | 8.5σ | 77/3220 |
| obs (fix) | **−0.802 ± 0.162** | 0.722 | 5.0σ | 14/3220 |
| paired difference | **+0.768 ± 0.226** | — | 3.4σ | — |

**The control reproduces the published five-seed baseline** (−1.610 ± 0.486) at
−1.570 ± 0.184, which is what licenses reading the difference as the rule's effect and
not as an artefact of this script.

So the decoupling is a genuine contributor worth about half the bias — and it is not
the explanation. Something else supplies the remaining −0.80.

**Ruled out for the residual: the `p_pe` basis Jacobian.** darksirens divides the
per-sample weight by `prior_wt` in the canonical basis `(m1det, q, dL, chi_eff)`
(`likelihood/core.py` → `log_sample_weight(m1det, q, ...)`; `inference/utils.py`
subtracts `log(prior_wt)`), while gmd draws flat in `(m1det, m2det, …)` and stores
`p_pe = 1` — so `p_pe` should be ∝ `m1det`. This is the open question PR #332's
docstring flags, and the selection side is already in the canonical basis
(`_selection_pdraw` carries the Jacobian explicitly), so the asymmetry made it the
leading candidate. Rewriting `p_pe → m1det` on the *same* events (paired at the event
level, so essentially noise-free) moves H₀ by **−0.039 ± 0.005**, max |Δ| = 0.050 over
four realisations — 5% of the residual and the *wrong sign*. Eliminated. The reason is
visible in the construction: within an event the `m1det` and `dL` samples are
independent draws, so reweighting by `m1det` barely couples to distance; it reaches H₀
only through `m1src = m1det/(1+z(dL;H₀))` in the mass-population term.

Same catalog and event truths throughout (verified bit-identical — both are drawn
before the PE), same 120M-draw injection set, nside 16, 1000 events,
`dark_sirens_complete` + field weighting, K=1, Om0 pinned, fixed population,
historical guard, 0/81 cells guard-rejected:

| σ (dL fractional) | truth-centred PE | corrected PE |
|---|---|---|
| 0.01 | −0.031 ± 0.087 | **−0.001 ± 0.087** |
| 0.03 | −0.118 ± 0.124 | −0.244 ± 0.165 |
| 0.10 | −1.136 ± 0.40 | −1.349 ± 0.704 |

The corrected intervals are wider because the mock now carries genuine per-event
measurement scatter, which the truth-centred version lacked entirely.

**Multi-seed test (settles it): the residual is real.** Five independent seeds at
σ = 0.10 with corrected PE, reseeding catalog, events and PE together so the result is
marginalised over catalog realisations:

| seed | H₀ | offset |
|---|---|---|
| 4101 | 66.391 | −1.349 |
| 4102 | 66.201 | −1.539 |
| 4103 | 67.773 | +0.033 |
| 4104 | 65.103 | −2.637 |
| 4105 | 65.184 | −2.556 |

**mean offset = −1.610 ± 0.486 (sem, n=5) = 3.3σ low.** 0 of 97 cells guard-rejected in
every seed. So the PE-centering fix does not remove the bias, and the earlier 4.4σ
(truth-centred, one catalog) was not an artifact of that single catalog either.

**Second result, previously untested in this program: catalog sample variance.** The
seed-to-seed scatter is sd = 1.086 km/s/Mpc against a mean per-seed 68% half-width of
0.485 — the credible intervals are ~2.2× too narrow. Quadrature-subtracting gives
≈ 0.97 km/s/Mpc of extra scatter per 1000-event realisation attributable to the host
catalog realisation, at this catalog density. Per-seed intervals are conditional on the
catalog and do not include it. This is the first measurement of the sample-variance
question left open since the gw_agn campaign ("does the multitracer combination cancel
sample variance on H₀ across catalog seeds? Requires multiple catalog realisations; not
tested here") — for K=1 it does not, and it is comparable in size to the bias itself.

**An earlier version of this file claimed the cause was identified.** That was
premature: a σ² scaling is consistent with *any* defect that scales with the noise
level, so it isolated the channel, not the mechanism.

**That candidate has now been tested and is a partial cause — see the A/B at the top of
this file.** It is worth +0.77 ± 0.23 of the offset, leaving −0.80 ± 0.16 unexplained.

## Addendum 2026-07-30 (later session): two ladders completed

**MC resolution is eliminated at full strength.** The 10-realisation paired 2k-vs-16k
PE-sample test (`results/nsamp16k_summary.json`): paired difference **−0.009 ± 0.077**.
The per-event estimator's MC noise is not the mechanism at any level that matters.

**The catalog kernel width is a THRESHOLD lever, not a linear one**
(`results/skde_summary.json`, sigma_kde broadening on ev_obs_b, effective kernel
sqrt((3e-3)² + σ_kde²)): offsets −0.92 / −1.00 / −0.82 / −1.04 / **−4.73** at
σ_kde = 0 / 0.003 / 0.010 / 0.020 / 0.040. Flat within scatter until the kernel
approaches the PE redshift width (~0.013 at the median event), then catastrophic.
The −4.7 magnitude is the same order as the GLASS z≤1 truncation lever (−4.09) and
the GLASS f=0.7 tilt (−3.32) — consistent with a shared mechanism living in the
catalog-prior support/normalization, activated when prior structure is comparable to
or broader than the PE kernel. At the physical width (3e-3 ≪ PE width) the residual
−0.80 is NOT explained by this lever's linear regime.

**What is left to try, in the order the evidence favours.** Everything cheap has been
eliminated; the survivors are all in the catalog-conditioned redshift prior, which is
the one part of the integrand no arm of this experiment has yet varied:

1. **The KDE convention itself.** `p_cat(z|pix) = g(z) Σᵢ (wᵢ/W) N(z; zᵢ, σᵢ) / Zᵢ` with
   `Zᵢ = ∫ N(z; zᵢ, σᵢ) g(z) dz`. Per-host normalisation by `Zᵢ` and the `g(z)` factor
   are convention choices; a mismatch between them and how hosts were drawn would bias
   the recovered distance scale at fixed sign, which is what is seen. Test: substitute
   an exact discrete-host prior (σ → 0, delta functions at the true host redshifts) so
   the KDE drops out of the comparison entirely.
2. **The dL → z inversion in the per-event integrand vs in μ.** The numerator
   decomposition already localised the offset to the numerator (peak +4.40, selection
   −5.54) with μ(H₀) verified analytically. A discretisation of `z(dL)` that differs
   between the two sides would show up exactly there.
3. **Catalog sample variance is a noise floor, not a bias**, and 20 realisations put
   the mean at 0.16 — so the residual is not a realisation artefact.

**Not affected: `../experiment_h0f_baseline`.** Its PE comes from the bespoke gw_agn
pipeline with `pe_centering: obs` *and* exact inverse-CDF flat-prior posterior sampling
(verified in `code/generate_gwsamples.py` and in the mock's `source_pe_path`), so the
defect fixed in PR #332 was never present there. Its H0 offsets (−1.0 at f_AGN = 0.307,
−3.4 at 0.703) are a separate open question, for which the measured lever remains the
host catalog's redshift edge (−4.1 km/s/Mpc on truncation at z ≤ 1).

## Figures

| figure | shows |
|---|---|
| `figs/fig_pe_sigma_ladder.pdf` | H₀ offset vs σ_dL, truth-centred vs corrected PE, with a ∝σ² reference. Both series decline together and their intervals overlap at σ = 0.10 — the fix does not remove the offset. |
| `figs/fig_closure_seeds.pdf` | the five-seed corrected-PE closure (−1.61 ± 0.49, 3.3σ), and the quoted 68% half-width against the realised seed-to-seed scatter (0.48 vs 1.09). |
| `figs/fig_elimination.pdf` | every lever tested and its measured ΔH₀, split into those that move H₀ and those that do not. |
| `figs/fig_obsdet_closure.pdf` | the 20-realisation detection-rule A/B: per-realisation offsets for both arms, and the two means (−1.57 ± 0.18 vs −0.80 ± 0.16). Regenerated by `scripts/analyze_obsdet.py` into `results/obsdet_summary.json`. |

All numbers in the figures are regenerated by `scripts/make_figures.py` into
`results/summary.json`; none is hand-typed.

## How the cause was isolated (what was ruled out first)

| candidate | test | verdict |
|---|---|---|
| bespoke mock's true-z selection rule | regenerate entirely with gmd (SNR detection) | **excluded** — bias persists |
| host rate weighting (γ) | force γ = 1 to match the flat host draw | **excluded** — moves H₀ ≤ 0.16 |
| catalog redshift edge | catalog to z = 2.0, events reach z ≈ 0.26 (edge 1.7 clear) | **excluded** — bias persists |
| hosts beyond the GW horizon / field normaliser | catalogs of 48.6k / 260.6k / 1M hosts | **excluded** — bit-identical (−1.136 all three); at K=1 the field-mode global Z cancels between the PE terms and −N log μ (`24ce9a9`) |
| a minority of pathological events | 50 disjoint 20-event blocks, per-event pull | **excluded** — no outlier blocks, symmetric distribution (mean −0.013, MAD 0.055) |
| selection-validity guard | nside 16 run rejects 0/65 cells | **excluded** — no guard involvement |
| sky pixelisation | nside 16 / 32 / 64 | **excluded** — −1.136 at nside 16 vs −1.74 ± 0.40 at nside 64 |
| selection integral μ(H₀) | analytic check for a distance-limited survey at z ≈ 0.14: d ln μ/d ln H₀ ≈ 3 ⇒ 0.044 | **correct** — measured 0.0406 |
| per-event numerator | decomposition: numerator alone peaks at 72.14 (+4.40), selection shifts −5.54 | **the culprit** |
| **PE construction** | σ scaling; then fixed and re-measured | real defect (PR #332), but NOT the full explanation |
| **detection/measurement noise decoupling** | 20-realisation A/B, detection moved onto the observed data | real defect (`fix/mock-shared-noise-draw`), worth +0.77 ± 0.23 — NOT the full explanation |
| `p_pe` basis Jacobian (m2det vs q) | rewrite `p_pe → m1det` on the same events, 4 realisations | **excluded** — −0.039 ± 0.005, wrong sign, 5% of the residual |
| catalog-conditioned redshift prior (KDE convention, z(dL) inversion) | not yet tested | **leading candidate for the residual** |

## Setup

Mock generated entirely by `scripts/mock_dark_sirens/generate_mock_data.py` at
master `2b86a2d`, so every ingredient is the inference's model by construction:

| ingredient | value |
|---|---|
| detection | noisy network SNR ≥ 8 (`_network_snr`); horizon z ≈ 0.27 |
| rate weighting | gmd's `(1+z)^(γ−1)` host acceptance, γ = 0 |
| masses / spins / PE | gmd's own samplers and `_posterior_samples` |
| injections | gmd `population+uniform`, 120M drawn / 343,702 detected |
| catalog | 1M hosts to z_max = 2.0, 0% empty pixels |
| survey file | gmd's **complete** catalog via gmd's own `_pixelate_catalog`, dz = 3e-3 |
| model | `dark_sirens_complete`, field weighting, K=1, Om0 pinned, fixed population |
| guard | historical `N_eff > 5·N_obs` only |

## The closure measurement

Ten **disjoint** contiguous 100-event blocks from the same 1000-event parent set.
Their scatter measures GW-realisation noise; a common offset is systematic, because
the catalog and injection set are shared and do **not** average down.

| block | H₀ median | offset |
|---|---|---|
| b0 | 66.89 | −0.85 |
| b1 | 67.34 | −0.40 |
| b2 | 66.08 | −1.66 |
| b3 | 66.95 | −0.79 |
| b4 | 67.13 | −0.61 |
| b5 | 65.89 | −1.85 |
| b6 | 64.41 | **−3.33** |
| b7 | 66.82 | −0.92 |
| b8 | 64.25 | **−3.49** |
| b9 | 64.23 | **−3.51** |

**mean H₀ = 66.000 ± 0.397 (sem, n=10) → offset −1.740 ± 0.397, i.e. 4.4σ.**

7 of 61 H₀ cells rejected in every block — the same cells, so selection-driven and
common-mode, not per-realisation.

## Caveats and follow-ups on that measurement

- **One catalog realisation and one injection set.** The 4.4σ is against
  GW-realisation noise only, not marginalised over catalog seeds or the selection MC.
  The corrected-PE multi-seed run addresses this.
- **The bimodality was a fluctuation — refuted.** b6/b8/b9 clustering near −3.4 looked
  like a minority mechanism, but a finer sweep of 50 disjoint 20-event blocks shows no
  outlier blocks at all (mean per-event pull −0.013, MAD 0.055, symmetric, no heavy
  tail: `results/localize_summary_b20.json`). The pull is spread evenly across events,
  which is what a convention/normalisation issue looks like rather than a few bad
  events.
- **Depth was tested and is not the lever here.** Catalogs of 48.6k / 260.6k / 1M
  hosts (z ≤ 0.5 / 1.0 / 2.0) give bit-identical results, because at K=1 the
  field-mode global normaliser cancels.
- The numbers above use the truth-centred PE; see the header table for the corrected-PE
  values, which supersede them for any forward-looking statement.

## Selection-MC resolution: the third occurrence, and a trap

Getting this far required navigating a resolution wall that has now appeared three
times in this campaign (catalog-targeted injections; the GWTC-4/5 variance guard;
this). Two specific findings worth carrying upstream:

1. **gmd's reported N_eff is not the N_eff the inference sees.** For the same
   injection set, gmd reported N_eff = 4067 (its *population* selection integral)
   while darksirens' *catalog-conditioned* integral saw **60** — a 68× gap, because
   injections must land on catalog hosts' KDE support. Anyone sizing an injection
   campaign from the generator's printed N_eff will under-provision by ~2 orders of
   magnitude. Also, gmd's own N_eff scales *sublinearly* in draw count (4067 → 5133
   for 3×, tail-dominated) while the catalog-conditioned one scales linearly
   (245 → 682) — so the generator's number is not even a reliable guide to scaling.
2. **The catalog KDE width must be matched to the PE resolution.** Pixelating with
   gmd's `SurveyConfig` redshift errors (dz ≈ 0.002, spectroscopic) gave N_eff = 60
   and Σσ²_PE = 135; rebuilding at dz = 3e-3 — the gw_agn campaign's validated
   choice under n·Dz/σ_z ≳ 100 — gave N_eff = 245 and Σσ²_PE = 37. This trades
   directly against signal: the catalog's redshift precision *is* the dark-siren
   information, so dz cannot simply be inflated to buy MC resolution.

Neither is available as a `population`/`uniform`/`population+uniform` proposal
choice; **catalog-targeted injections** (the `agncat` lane, previously worth 116× in
N_eff) remain necessary and gmd does not provide them. That is what caps the present
test at N_obs = 100 per block rather than N_obs = 1000.

## What this changes

The design's arm C is now the live branch: a bias that survives a matched generative
process localises the problem **inside darksirens**, not in the mock. Concretely this
experiment has become a clean, minimal reproducer — library's own generator, library's
own detection, deep complete catalog, isotropic, K=1, no bespoke code in the data path
— which is the right artifact to hand to the PR #215 follow-up ("a separate low bias
for catalogs whose dN/dz rises into a sharp z_max edge … under separate
investigation").

Next, in order: (1) depth A/B on this mock; (2) characterise what distinguishes
b6/b8/b9; (3) build the catalog-targeted injection lane so N_obs = 1000 becomes
reachable and the significance stops being limited by N_eff.
