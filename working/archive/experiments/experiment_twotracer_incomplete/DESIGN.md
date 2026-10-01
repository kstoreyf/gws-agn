# DESIGN — experiment_twotracer_incomplete

**Question: how fast does the AGN-hosted fraction lose precision as the host survey
becomes incomplete?** Not whether it stays unbiased — the programme already has a known
unresolved bias at this scale (see below) — but how the *width* degrades, which is what
decides whether a real, flux-limited survey can measure f_AGN at all.

This composes the two experiments that precede it: the K = 2 deep two-tracer mock of
`../experiment_twotracer_deep`, put through the isotropic flux-limit ladder of
`../experiment_completeness_anchored`.

## What is held fixed

Everything except the flux limit. The mock — hosts, events, PE — is the deep two-tracer
mock unchanged: incompleteness is an *observational* effect, so the events are exactly
the ones that happened.

| | |
|---|---|
| tracers | GAL 1,000,000 hosts; AGN 12,000 (a seeded subset of them) |
| events | 140 GAL-hosted + 60 AGN-hosted = 200, planted f_AGN = 0.300, z ∈ [0.023, 0.291] |
| PE | `pe_centering="observed"` (PR #332), σ_dL = 0.10 |
| pixelisation | nside 32, dz = 3 × 10⁻³ |
| estimator | `dark_sirens` (the *incomplete* model) + field sky weighting, K = 2 |
| nuisances | Om₀ pinned, population fixed, log10n₀ and δ **anchored per tracer** |
| guard | historical `N_eff > 5·N_obs` (`--selection_neff_guard hard --max_likelihood_variance 1e6`) |

The AGN tracer is a nested subset of the galaxies, so AGN inherit their host galaxy's
apparent magnitude — one flux limit thins both tracers with the same C(z), which keeps
this a single-axis experiment rather than a two-dimensional survey-design study.
`materialise_tracer_catalogs.py` reconstructs the AGN subset from the deep mock's seed
and **verifies** it against that mock's own pixelated AGN survey before writing
anything; the complete-catalog pixelation is then checked to be bit-identical to the
deep mock's survey files, so the ladder's zeroth rung is provably the same object the
previous experiment measured.

## The ladder

Isotropic flux limit on both tracers — no footprint, no hard redshift cut — so
completeness is a *consequence* of survey depth and has the declining shape a
flux-limited survey actually has. Completeness is quoted within z ≤ 0.30, the range the
events occupy.

| level | GAL hosts | GAL empty pix | AGN hosts | AGN empty pix | C(z ≤ 0.30) |
|---|---|---|---|---|---|
| complete | 1,000,000 | 0.0% | 12,000 | 37.5% | 1.00 |
| m < 21 | 54,530 | 1.1% | 689 | 94.5% | ≈ 0.94 |
| m < 20 | 20,369 | 18.7% | 268 | 97.9% | ≈ 0.75 |
| m < 19 | 7,019 | 56.2% | 75 | 99.4% | ≈ 0.40 |
| m < 18 | 2,236 | 83.3% | 28 | 99.8% | ≈ 0.17 |

The AGN column is the point of the experiment. There are only 154 AGN inside the horizon
even when the catalog is complete; by m < 19 there are 58, against 60 AGN-hosted events.
Whatever identifies f_AGN has to survive that.

## Anchoring

The completion derives its missing-host budget from the density model, so
`C_k(z) = (dN_obs/dz) / (n₀ₖ dV_c/dz (1+z)^δₖ)` is never a free function. Following
`../experiment_completeness_anchored`, n₀ and δ are held at the **best fit of the model
form to the true host density**, per tracer, not at the raw mean density — and the
residual of that fit is the experiment's noise floor:

| tracer | log10 n₀ | δ | shape residual (fit range / within z ≤ 0.30) |
|---|---|---|---|
| GAL | −5.80638 | +0.0194 | 2.6% / 6.7% |
| AGN | −7.72003 | −0.0031 | 6.2% / 8.3% |

The AGN residual within the horizon is dominated by the shot noise of its 154 hosts
there, which is a real property of a sparse tracer and not a defect of the fit.

## Injections

The selection integral is catalog-conditioned, so the proposal must cover the support the
likelihood actually conditions on — and that support *shrinks with the survey*. One
catalog-targeted injection set is therefore generated per rung, targeting that rung's own
surveys, with the mixture held fixed across the ladder so rungs stay comparable:

    0.55 population + 0.10 uniform + 0.15 GAL-targeted + 0.20 AGN-targeted

Both tracers get a branch here, unlike in `../experiment_twotracer_deep` where only AGN
did: at the faint end GAL is sparse too (83% empty pixels at m < 18).

## What this experiment can and cannot say

**Can:** how σ(f_AGN) and σ(H₀) grow as completeness falls, at fixed data; whether the
likelihood stays evaluable (guard) as the catalog empties; where the identifiability of
f_AGN dies.

**Cannot:** anything about absolute bias. `../experiment_matched_mock` has an unresolved
−0.80 ± 0.16 H₀ offset that survives both known generator fixes, and
`../experiment_twotracer_deep` shows it propagating into this very mock (H₀ 2.7σ low,
f_AGN 1.8σ low) *before* any incompleteness is imposed. Every statement here is therefore
**differential against the complete-catalog rung**, which carries that offset too.

**Also cannot:** separate completeness from anchoring error. n₀ is held fixed at the true
best fit throughout, which is the most favourable case; a real survey infers it.
`experiment_completeness_free` is where that is asked.
