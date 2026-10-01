# experiment_completeness_anchored

**Question.** With the true missing-host budget handed to it, does darksirens'
isotropic multi-tracer completion recover H₀ and f_AGN unbiased?

This isolates **machinery correctness** from **identifiability**. n0 is held fixed, so
nothing here tests whether completeness can be *inferred* — that is the next
experiment. If this one fails, nothing downstream is interpretable.

## STATUS: RUN — see RESULTS.md

Executed 2026-07-30 on the deep matched mock with PR #332's corrected PE. Two design
adaptations were forced by what the earlier experiments measured, both recorded here
rather than silently applied:

1. **The gate is DIFFERENTIAL, not absolute.** `../experiment_matched_mock` leaves an
   unresolved baseline bias (−1.61 ± 0.49) that the complete-catalog control inherits,
   so "truth inside the 68% interval" is unusable as written. The gate became "no
   departure from the complete-catalog control at the same σ".
2. **Single-tracer (K=1).** The deep mock has one tracer, so this run tests the
   completion machinery only; multi-tracer completeness needs a two-tracer deep mock
   that does not exist yet.

The luminosity function is also not mine: gmd's complete catalog already carries
`app_mag`, so an isotropic flux limit on that column gives a physically-shaped C(z)
with no footprint and no hard redshift cut — simpler than the Schechter draw this
design originally specified, and consistent with the library's own survey model.

## Dependency and original status

Needs the deep host field from `../experiment_matched_mock` (z_max ≈ 3, so the
catalog's redshift edge sits far beyond the z ≲ 1 events). **That field does not
exist yet** — matched_mock's generator is unbuilt; its design was redirected by the
edge diagnostic (a 4 km/s/Mpc H₀ shift from relocating the catalog edge with events
fixed).

Two pieces are therefore built and exercised **now, as dry runs against the existing
z_max ≈ 1.56 catalogs**, because they are independent of the depth and de-risk the
real run:

1. `measure_density_model.py` — fits the density model form to the true tracer
   density and reports the **shape residual**. This sets the n0 anchor and bounds how
   clean an "anchored" result can possibly be.
2. `truncate_catalogs.py` — the LF + flux-limit truncation, the C_k(z) curves it
   produces, and the empty-pixel fractions per completeness level.

Both must be re-run on the deep field before any inference number is quoted. The
dry-run outputs live in `results/dryrun_*` and are labelled as such.

## The anchoring subtlety (why `measure_density_model.py` exists)

The completion derives the missing budget as

```
dN_miss,k/dz  =  n0_k · dV_c/dz · (1+z)^δ_k  −  dN_obs,k/dz
```

so completeness is never a free function — it is `C_k(z) = (dN_obs,k/dz) /
(n0_k · dV_c/dz · (1+z)^δ_k)`. Anchoring n0 fixes the budget's **amplitude**, but its
**shape** is still the assumed `(1+z)^δ·dV_c/dz`. The GLASS tracer density is whatever
the lognormal field produced, not exactly that form.

Consequence: anchor n0 to the **best fit of the model form to the true density**, not
to the raw mean density. Any residual between the fitted form and the truth is
irreducible mis-specification that will present as a completion bias, so it is
measured first and quoted as the experiment's noise floor.

## Design

- **Hosts for events are drawn from the COMPLETE population.** Only the catalogs the
  inference sees are truncated. Truncating before drawing hosts would leave nothing
  missing and test nothing.
- **Truncation via luminosity function + flux limit**, per tracer: Schechter LF for
  galaxies, a brighter/steeper LF for AGN, with independent flux limits. Completeness
  is then a *consequence* with a physically-shaped C(z), not an input handed back to
  the inference.
- **The inference never sees magnitudes.** Completeness enters only through observed
  counts, so the LF's sole job is to generate a realistic C(z) shape; SED and
  K-correction details are cosmetic and are held at simple defaults.
- **Isotropic and full-sky by construction** — one flux limit per tracer over the
  whole sky, no footprint. Anisotropic completeness is a later, deliberate stress
  test, because an anisotropic truth modelled as isotropic imprints a sky-density
  contrast, which is exactly the channel that identifies f_AGN.
- **Ladder** defined by completeness retained *within the detected-event horizon*
  (z ≤ 1), which is the physically meaningful knob: ≈ {100% (control), 70%, 40%, 20%}
  per tracer, flux limits solved per target.

## Sampled parameters

| parameter | status |
|---|---|
| H₀ | sampled |
| f_AGN (`fcat_2`) | sampled |
| `log10n0_gal`, `log10n0_agn` | **fixed** at the model-form best fit |
| `delta_gal`, `delta_agn` | fixed at the fitted values |
| `sigma_kde` | fixed at 0 |

Two sampled parameters, so grid scans remain affordable and no sampler is needed —
`experiment_sampler_parity` is only required for the following experiment.

Guard: historical `N_eff > 5·N_obs` only (`--max_likelihood_variance 1e6`), matching
`../experiment_h0f_baseline` so the comparison is like-for-like.

## Gates

1. Truth inside the 68% interval for **both** H₀ and f_AGN at every completeness
   level, including the 100% control.
2. σ(f_AGN) grows monotonically with incompleteness and **gracefully** — no rail to
   f = 1. A rail means the missing-AGN budget has swallowed the estimand, the same
   failure signature as the original conditional-mixture bug.
3. Recovered values remain insensitive to catalog depth with incompleteness present
   (the missing budget integrates to the depth, so depth-insensitivity must be
   re-established here, not inherited).
4. Any bias must be smaller than the shape-residual noise floor from
   `measure_density_model.py`; otherwise it is un-attributable.

## Recorded per level (feeds the next experiment)

Empty-pixel fraction per tracer — the predictor for the (f_AGN, n0_AGN) degeneracy
severity in `experiment_completeness_free`. The AGN tracer already sits at ≈ 79%
empty when complete; truncation raises it, and a missing AGN host and an AGN-hosted
event both explain an event in an empty pixel.
