# RESULTS — experiment_completeness_anchored

**The isotropic completion recovers the expansion rate without added bias down to 22%
survey completeness, and the credible intervals widen as they should.** Across a
completeness ladder spanning 100% → 22% within the detection horizon — with the
fraction of empty sky pixels rising from 0% to 47% — the recovered H₀ stays within
0.5σ of the complete-catalog control, and no likelihood evaluation is rejected.

darksirens master `2b86a2d` + PR #332's corrected posterior samples;
`dark_sirens` with field-convention sky weighting, K=1, Om₀ pinned, fixed population,
historical `N_eff > 5·N_obs` guard; 1000 events, nside 16, σ_dL = 0.10.

## What was imposed

An isotropic flux limit on the complete host catalog — no footprint, no hard redshift
cut — so the completeness is a *consequence* of the survey depth, with the declining
shape a flux-limited survey actually has:

| level | flux limit | hosts kept | C(z ≤ 0.27) | C(z) across the horizon | empty pixels |
|---|---|---|---|---|---|
| control | none | 1,000,000 | 100% | flat at 1 | 0% |
| 1 | m < 20 | 20,369 | 80.7% | 1.00 → 0.68 | 0.03% |
| 2 | m < 19 | 7,019 | 50.4% | 1.00 → 0.29 | 9.8% |
| 3 | m < 18 | 2,236 | 21.9% | 1.00 → 0.06 | 47.4% |

Completeness is quoted *within the detection horizon* (z ≤ 0.27), which is the range
the events occupy; over the full catalog depth the same limits read 2.0%, 0.7% and
0.2%, dominated by hosts no event could have.

## Recovery

The completion derives its missing-host budget from the density model, so n₀ and δ are
held at the best fit of that model's form to the true host density — not at the raw
mean density: **log10 n₀ = −5.8064, δ = +0.019** (`results/density_model_anchor.json`).

| level | C(z ≤ 0.27) | H₀ offset | vs control |
|---|---|---|---|
| control | 100% | −0.775 ± 0.651 | — |
| 1 | 80.7% | −1.549 ± 1.408 | −0.774 (0.5σ) |
| 2 | 50.4% | −0.599 ± 0.682 | +0.176 (0.2σ) |
| 3 | 21.9% | −1.228 ± 1.521 | −0.453 (0.3σ) |

The largest departure from the control is 0.5σ. Intervals grow by up to 2.3× as the
catalog thins, which is the expected cost of replacing observed hosts with a modelled
budget, and 0 of 65 grid cells are rejected at any level — the completion never drives
the likelihood into its validity guard, even at 47% empty sky.

## How to read the absolute offset

Every level sits ~0.6–1.5 low, including the complete-catalog control. That pedestal is
**not** a completeness effect: it is the unresolved baseline bias measured in
`../experiment_matched_mock` (−1.61 ± 0.49 over five seeds, same code, same corrected
PE, complete catalog). The comparison here is therefore differential against the
control, and it is the differential statement that is the result. An absolute closure
statement is not available until that baseline is understood.

Two further limits on the claim. The per-level precision is a single realisation
(σ = 0.65–1.52), so a bias smaller than ~0.7 would not be detected; and the host
density's shape residual against the fitted model form is 2.6% rms over the fit range
and 7.2% within the horizon — the latter dominated by the shot noise of the 9,105 hosts
inside z ≤ 0.27 — which sets a floor below which any apparent completeness bias is not
attributable.

## Scope

Single-tracer (K=1). This tests the completion machinery, which is what has to work
before identifiability can be asked about; the multi-tracer completeness the programme
needs — where a missing AGN host and an AGN-hosted event both explain an event in an
empty pixel — requires a two-tracer deep mock that does not exist yet, and is the next
build. n₀ is held fixed throughout, so nothing here bears on whether completeness can
be *inferred*; that is `experiment_completeness_free`.

## Reproducing

```
python scripts/measure_density_model.py --complete_catalog <complete catalog> \
    --out_json results/density_model_anchor.json
python scripts/pixelate_complete_catalog.py --mag_limit <m> ...   # per level
python scripts/scan_h0f.py --universe_model dark_sirens --catalog_sky_weighting field \
    --log10n0 -5.806380 --nuisance_json '{"delta":0.0194}' ...    # per level
python scripts/make_figures.py
```

Figure: `figs/fig_completeness_ladder.pdf` (imposed C(z); recovery vs completeness with
the control band). Every number regenerates into `results/summary.json`.
