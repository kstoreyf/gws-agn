# Upstream findings for darksirens (master @ `2b86a2d`)

Three items from the gw_agn multitracer campaign, in priority order. Item 1 is a
minimal reproducer for an already-acknowledged open problem; items 2 and 3 are
usage/API traps that cost this campaign significant time and would cost any other
user the same. Drafted to be pasted as GitHub issues — not yet filed.

Evidence paths are under
`gws-agn/working/analyses/experiments/experiment_matched_mock/` unless stated.

---

## Issue 1 — `generate_mock_data._posterior_samples` builds truth-centred distance clouds and labels them flat-prior posteriors, giving every closure test an O(σ²) low bias in H₀

**The defect.** `_posterior_samples` draws

```python
dl = rng.lognormal(np.log(truth["dl"][i]) - 0.5 * frac_dl**2, frac_dl, nsamp)
```

i.e. a cloud centred on the **true** luminosity distance, and stores it with
`p_pe = 1` — declaring it to be posterior samples under a flat prior in the canonical
`(m1det, q, dL)` basis. It is not: it is a likelihood-shaped cloud about truth. The
correct construction draws a noisy observation and then samples the flat-prior
posterior given that observation (for multiplicative distance noise this has an exact
inverse-CDF form).

**Measured consequence.** Same catalog, same injection set, same events' truths; only
`--dL-fractional-uncertainty` changed. `dark_sirens_complete`, field weighting, K=1,
1000 events, nside 16, Om0 pinned, fixed population, historical guard, 0/81 cells
guard-rejected:

| σ (dL fractional) | H₀ median | offset from 67.74 | 68% half-width |
|---|---|---|---|
| 0.01 | 67.709 | **−0.031** | ±0.087 |
| 0.03 | 67.622 | −0.118 | ±0.124 |
| 0.10 | 66.604 | −1.136 | ±0.40 |

The offset grows 9.6× between σ = 0.03 and 0.10 against the 11.1× that σ² predicts,
and at σ = 0.01 it is consistent with zero — **the likelihood closes when the
distance uncertainty is small**, which is the signature of an O(σ²) bias in the data
rather than a defect in the inference.

**Why it matters.** Being a generator defect, every mock closure built on
`generate_mock_data` inherits it. The default uncertainty is
`clip(1.8/ρ, 0.08, 0.35)`, so σ ≈ 0.1–0.35 in a typical run ⇒ a predicted bias of
roughly −1 to −14 km/s/Mpc. That range covers the low H₀ medians in
`src/darksirens/runs/full` — e.g. `A1-dark-H0` (120 events, nside 16, Ndraw 150M,
Om0 free) recovers H₀ = 56.05 with 68% [47.67, 65.06], truth **outside** the 68%
interval — which had been read as acceptable because those intervals are 18–76
km/s/Mpc wide. A tighter configuration (fixed population, Om0 pinned, 1000 events,
σ(H₀) ≈ 0.1–0.4) turns the same effect into a many-σ systematic.

**Relation to PR #215's open item.** PR #215 (`d0435b6`) notes under *Not fixed here*:
"A separate low bias for catalogs whose dN/dz rises into a sharp z_max edge remains
after this fix (mock closure recovers 45.3, not 70, on the volume-limited catalog) and
is under separate investigation." This is a candidate explanation that does not
involve the z_max edge at all — worth checking against that case before attributing it
to catalog shape.

**Independent precedent.** The same defect class was diagnosed and fixed in a bespoke
pipeline in this project: "Gaussian obs-centered clouds used as if they were the
flat-prior posterior of multiplicative distance noise — an O(fac²) distance-scale
bias", measured there at +0.7–1.0 km/s/Mpc at 10% errors, fixed by exact inverse-CDF
flat-prior posterior sampling. Sign differs with the centring convention; magnitude
matches.

### What was ruled out before reaching this, on the same mock

selection rule (regenerated with gmd's SNR detection); host rate weighting γ (forcing
γ=1 moves H₀ ≤ 0.16); catalog redshift edge (catalog to z=2.0 with events only to
z≈0.26); hosts beyond the horizon and the field normaliser (48.6k/260.6k/1M-host
catalogs give bit-identical −1.136 — at K=1 the global Z cancels per `24ce9a9`); a
minority of pathological events (50 disjoint 20-event blocks show no outliers, mean
per-event pull −0.013, MAD 0.055); the guard (0/65 cells rejected); sky pixelisation
(nside 16/32/64). The decomposition then localised it: the per-event numerator alone
peaks at 72.14 (+4.40) while the selection term shifts −5.54, and μ(H₀)'s slope
matches the analytic distance-limited expectation (measured d ln μ/dH₀ = 0.0406 vs
≈0.044 predicted at z ≈ 0.14), so the numerator was the culprit.

### Reproducer

**Setup** — every ingredient is the inference's own model; no bespoke code in the data
path:

| | |
|---|---|
| generator | `scripts/mock_dark_sirens/generate_mock_data.py` @ `2b86a2d` |
| detection | noisy network SNR ≥ 8 (`_network_snr`) |
| rate weighting | gmd's `(1+z)^(γ−1)` host acceptance, γ = 0 (matches inference) |
| masses/spins/PE | gmd's own samplers and `_posterior_samples`, 10% distance errors |
| injections | `--proposal population+uniform`, 120M drawn / 343,702 detected |
| catalog | 1M hosts to **z_max = 2.0**; events reach z ≈ 1, so the edge is 1.0 clear |
| survey file | gmd's **complete** catalog, pixelated via gmd's own `_pixelate_catalog`, dz = 3e-3 |
| model | `dark_sirens_complete`, `--catalog_sky_weighting field`, K=1, Om0 pinned, fixed population |
| guard | historical `N_eff > 5·N_obs` only (`--max_likelihood_variance 1e6`) |
| pixels | 0% empty (all 49,152 occupied) — no empty-pixel policy involvement |

**Generation:**
```
python scripts/mock_dark_sirens/generate_mock_data.py \
  --outdir <out> --seed 4101 --nobs 1000 --nsamp 2000 \
  --ndraw 120000000 --nbatches 60 --nside 64 --zmax 2.0 \
  --n-galaxies 1000000 --snr-threshold 8 --gamma 0 \
  --dL-fractional-uncertainty 0.10 --proposal population+uniform \
  --H0 67.74 --Om0 0.3075
```

**Measurement.** Ten *disjoint* contiguous 100-event blocks from the one 1000-event
parent set. Block scatter measures GW-realisation noise; a common offset is
systematic, since the catalog and injection set are shared and do not average down.
61-point H₀ scans on [55, 85], flat-prior grid medians:

| block | H₀ | offset | | block | H₀ | offset |
|---|---|---|---|---|---|---|
| b0 | 66.89 | −0.85 | | b5 | 65.89 | −1.85 |
| b1 | 67.34 | −0.40 | | b6 | 64.41 | **−3.33** |
| b2 | 66.08 | −1.66 | | b7 | 66.82 | −0.92 |
| b3 | 66.95 | −0.79 | | b8 | 64.25 | **−3.49** |
| b4 | 67.13 | −0.61 | | b9 | 64.23 | **−3.51** |

**mean H₀ = 66.000 ± 0.397 (sem, n = 10) ⇒ offset −1.740 ± 0.397 = 4.4σ.**

7 of 61 H₀ cells are guard-rejected in every block — the same cells, so common-mode
and selection-driven, not per-realisation.

**Caveats.** One catalog realisation and one injection set, so the significance is
against GW noise only and is not marginalised over catalog seeds or the selection MC.
Depth insensitivity has not yet been tested on this mock.

**Lead worth pursuing first.** The block scatter is bimodal: b6/b8/b9 cluster at
≈ −3.4 while the other seven sit near −0.9. That is not Gaussian noise about a single
systematic, and it suggests a subset of events drives the excursions — candidates
being events whose host is poorly covered by the catalog KDE, or a few dominant PE
weights. Identifying what those three blocks share is probably cheaper than a general
hunt.

**Independent corroboration from the bespoke GLASS mock** (`../experiment_h0f_baseline`):
relocating that catalog's redshift edge from 1.56 to 1.0 with the events held fixed
moved H₀ by −4.09 km/s/Mpc at planted f_AGN = 0.307. H₀ there is strongly sensitive to
a catalog-construction boundary.

---

## Issue 2 — `generate_mock_data`'s reported `N_eff` understates the inference's catalog-conditioned `N_eff` by ~68×, and mis-scales

For one and the same injection set:

| quantity | value |
|---|---|
| N_eff printed by `generate_mock_data` (population selection integral) | **4067** |
| N_eff seen by `selection_log_correction` under `dark_sirens_complete` + field weighting | **60** |

The inference's selection integral is conditioned on the host catalog, so injections
only carry weight where they land on catalog hosts' KDE support; the generator's
number is computed against the smooth population and is therefore not the quantity
the guard tests.

It also scales differently, so it cannot be used to provision by extrapolation:

| draws | gmd's printed N_eff | catalog-conditioned N_eff |
|---|---|---|
| 40M | 4067 | 245 |
| 120M | 5133 (**1.26×**, sublinear — tail-dominated) | 682 (**2.8×**, linear) |

**Consequence.** A user sizing an injection campaign from the printed value
under-provisions by roughly two orders of magnitude, and then cannot tell from the
generator's output how much more to draw. In this campaign it capped the closure test
at N_obs = 100 per block instead of 1000.

**Suggested fix.** Either report the catalog-conditioned N_eff when a survey is
present, or document explicitly that the printed value is a population-only diagnostic
and state the catalog-conditioned requirement `N_eff > 5·N_obs` against it.

---

## Issue 3 — no catalog-targeted selection proposal, though field-mode inference needs one

`SELECTION_PROPOSALS` offers only `population`, `uniform`, `population+uniform`. None
place injections preferentially at catalog objects, which is what a catalog-conditioned
selection integral requires. This campaign previously had to build that lane by hand:
a `0.65 population + 0.10 uniform + 0.25 catalog-targeted` mixture with exact mixture
`pdraw` raised field-mode N_eff from ~1.2k to ~145k — **116×** — and without it every
f_AGN scan walled at the same argmax regardless of the planted truth, i.e. a pure
injection-set artifact.

Two related, already-measured constraints that belong in the same documentation:

- **The catalog KDE width must be matched to the PE resolution.** Pixelating with
  `SurveyConfig`'s redshift errors (dz ≈ 0.002, spectroscopic) gave N_eff = 60 and
  Σσ²_PE = 135; rebuilding at dz = 3e-3 gave 245 and 37. This trades directly against
  signal — the catalog's redshift precision *is* the dark-siren information — so dz
  cannot simply be inflated to buy MC resolution. The working criterion from the
  bespoke campaign was `n·Dz/σ_z ≳ 100`.
- **The total-variance guard truncates in the inferred parameter.** Since Σσ²_PE
  depends on θ, a per-cell hard rejection acts as a parameter-dependent prior rather
  than a reliability filter. Measured on a 50-event mixture subsample: the default
  budget admitted only f ≤ 0.23 (planted 0.30) and f ≤ 0.38 (planted 0.70), excluding
  the truth from the parameter space and biasing the recovered fraction to −0.108 and
  −0.333, where the same data at an inert budget gave −0.019 and +0.023. Full numbers
  in `gws-agn/working/gw_agn_darksirens_2b86a2d_2026-07-29/RESULTS.md` §5. Suggest
  applying the criterion once per run at a reference θ, or surfacing the admitted
  fraction so a truncated posterior cannot pass as converged.

**Common theme.** Selection-MC resolution has been the binding constraint three times
in this campaign (catalog-targeted injections; the GWTC-4/5 variance guard; the
catalog-conditioned N_eff above). It reads as a structural property of field-mode
inference on clustered catalogs rather than three incidents, and would be worth a
short "provisioning the selection campaign" section in the docs covering: which N_eff
to test, how it scales, the KDE-vs-PE width criterion, and the need for a
catalog-targeted proposal.
