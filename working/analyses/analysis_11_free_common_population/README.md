# Analysis 11 — free common (reference) population plus environmental offsets

Seed 100 only. Science exploration, not calibration. The Analysis-10 production
mock, surveys, injections, detection and PE model are reused unchanged.

## Question

Analysis 10 measured the AGN branch's departures from a reference population that
it took as known exactly (μ_G = 35 M☉, μ_χ = 0). Here the reference population is
itself inferred:

| reporting coordinate | meaning | truth | exploration domain |
|---|---|---|---|
| μ_G | reference (GAL, catalogue-1) Gaussian-peak location | 35 M☉ | [31, 39] |
| Δμ_G | μ_G,AGN − μ_G,GAL | +5 M☉ | [−4, +10] |
| μ_χ | reference spin mean | 0 | [−0.10, +0.10] |
| Δμ_χ | μ_χ,AGN − μ_χ,GAL | +0.10 | [−0.05, +0.20] |
| f_AGN | AGN-branch weight | 0.30 planted | [0, 1] |

Validity on every evaluated point: 20 < μ_G + Δμ_G < 50 (the registry prior of
`G.mu`), and both spin means inside the production support (−1, 1).

## Likelihood

`scripts/a11_likelihood.py` builds the Analysis-10 likelihood verbatim
(`a8_likelihood.build(mode='new')` under `a10_likelihood._steer(('G.mu_c2',
'mu_chi_c2'))`) and releases exactly two of the twelve pinned base slots,
`$\mu_{\rm G}$` and `$\mu_\chi$`, by steering `a8.fixed_parameter_values_for`.
darksirens' `decode_mixture` reads catalogue 1's population from the base labels
and catalogue 2's from the absolute `_c2` labels, so the reporting map

    mu_{G,c1} = mu_G,  mu_{G,c2} = mu_G + dmu_G,
    mu_{chi,c1} = mu_chi,  mu_{chi,c2} = mu_chi + dmu_chi

is applied before the production call and the physical population code is
untouched.

## Ladder

1. **Specification, closure, selection support** (`a11_closure.py`,
   `a11_selection_file.py`).
2. **11A** (H0 = 67.74, spin sector pinned at the planted values): grid over
   (f, μ_G, Δμ_G).
3. **11B** (H0 = 67.74, mass sector pinned at the planted values): grid over
   (f, μ_χ, Δμ_χ).
4. **Gate after 11A/11B.** Stop if an offset depended on the pinned baseline.
5. **Sampler validation** against the Analysis-10 A10-J grid posterior.
6. **11C**: (f, μ_G, Δμ_G, μ_χ, Δμ_χ) at fixed H0 with the validated sampler.
7. **Fixed-H0 owner gate**, then **11D**: H0 released.
8. **Analysis 12** (only if 11D closes): one-at-a-time shared σ_G and σ_χ.

Compute discipline: anything projected at several times Analysis 10's scale is
written into `STATE.md` and stops for owner approval.

## Inputs of record

| input | path | md5 |
|---|---|---|
| events | `working/data/seed100/events/events_marked_dmu0p10_dmuG5.h5` | `427990378e299850a9c0708d389bc0bf` |
| injections | `working/data/seed100/injections/injections_targeted.h5` | `e8a611a27f1f0699adc1768b2a3e395a` |
| GAL survey | `working/data/seed100/surveys/survey_gal_complete_ns32.h5` | `3568cf69344b13d13de13a7e7b1d4fc6` |
| AGN survey | `working/data/seed100/surveys/survey_agn_complete_ns32.h5` | `6dcc38d17bd5cf6b006fde1427524e3e` |

gws-agn at the start: `2db436aa880563f470b0b94cba811c22a3fa9bcf`. darksirens:
`darksirens-a8` at `af896cae6f3f3dd1f87dec50046e3a8228f59b39`, clean.
GPU work runs on rita (partition RITA-GPU, QOS `rita`).
