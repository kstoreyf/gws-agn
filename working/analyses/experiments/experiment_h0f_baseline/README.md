# experiment_h0f_baseline

The baseline (H₀, f_AGN) result set. One estimator, one guard, one set of inputs;
later experiments build up from here.

## What is fixed

| | |
|---|---|
| code | darksirens master `2b86a2d`, run from a pinned worktree (`DARKSIRENS_WT`) |
| estimator | `dark_sirens`, field-convention sky weighting, K=2 mixture `[GAL, AGN]` so `fcat_2` = f_AGN |
| nuisance point | `log10n0 = log10n0_c2 = −12` (complete-catalog limit), `delta = 0`, `sigma_kde = 0` |
| population | fixed at the mock truth (powerlaw+peak + chieff); Om0 pinned to 0.3075 |
| selection | catalog-targeted injections (`injections_cat.h5`), `DARKSIRENS_ZMAX = 1.5` |
| **guard** | **historical `N_eff > 5·N_obs` only** |
| grids | f: 41 pts on [0,1] · H₀: 61 pts on [50,100] · joint: 81 × 61 |
| events | 1000 per set, planted f_AGN ∈ {0.00989, 0.307, 0.703, 1.0} |

### The guard setting, precisely

Current master enforces two criteria: the historical Vitale floor `N_eff > 5·N_obs`
**and** the GWTC-4.0/5.0 total-variance bound
`σ²_lnL = Σᵢσ²ᵢ + N_obs²/N_eff ≤ max_likelihood_variance` (default 1.0). This
experiment wants only the first, so it runs with
`--selection_neff_guard hard --max_likelihood_variance 1e6`, which drives the
threshold `max(5·N_obs, N_obs²/budget)` to exactly `5·N_obs`.

`--selection_neff_guard soft` does **not** do this — it only replaces the hard −∞
wall with a steep smooth penalty and leaves the variance threshold in place
(`darksirens/likelihood/selection.py:311-312`; the flag's own help notes "the
Vitale 5 N_obs mean floor always applies"). On a sparse-tracer mixture the soft
penalty also sculpts spurious ridges in the joint surface, so `hard` is used here.

## Layout

```
scripts/scan_h0f.py         grid driver (module import, no sampler)
scripts/run_experiment.sh   the driver: ./run_experiment.sh [all|fscan|h0scan|joint]
scripts/make_figures.py     publication figures + results/summary.json + table_h0f.tex
results/  figs/  logs/      outputs
data -> ../../../gw_agn_darksirens/data
```

## Reproducing

```bash
export DARKSIRENS_WT=/path/to/darksirens/worktree/at/2b86a2d
cd scripts && ./run_experiment.sh && python make_figures.py
```

Conda env `jax`, one A100. Runtime ≈ 25 min (the two joint grids dominate).

## Internal notes on validity (not for reader-facing text)

- **Guard truncation is inert for this result set.** All f scans admit 41/41 cells.
  The H₀ scan at planted f = 0.703 admits 15/61 (window H₀ ∈ [62.5, 74.2]), but its
  edges sit at ΔlogL = −19.4 and −189.1 and only 2.6e−6 of the posterior mass lies
  within one grid step of them, so the posterior is unaffected. Joint-grid rejected
  counts are recorded in `results/summary.json`.
- **Known offset carried by this configuration — measured decomposition.** H₀
  recovers low and the offset grows with the planted AGN fraction. f_AGN itself
  recovers correctly at fixed H₀ and along the joint surface, so this is not a
  property of the mixture weight. `scripts/diag_h0_offset.py` splits the likelihood
  at fixed mixture weight into the per-event numerator Σᵢ ln Zᵢ(H₀) and the
  selection term −N_obs ln μ(H₀) + MC correction
  (`results/h0_decomposition_fagn{0.3,0.7}.json`):

  | planted f | full peak | numerator alone | shift from selection |
  |---|---|---|---|
  | 0.307 | 66.82 (−0.92) | 70.83 (+3.09) | −4.01 |
  | 0.703 | 64.42 (−3.32) | 68.69 (+0.95) | −4.27 |

  **Two errors of opposite sign that partially cancel.** (i) The selection term
  pulls H₀ down ~4 km/s/Mpc nearly independently of f: ln μ spans 0.36 nats over
  H₀ ∈ [60,76] (d ln μ/dH₀ = +0.025 at truth) and enters multiplied by
  N_obs = 1000, a ~360-nat lever — the mock's detection is a hard cut on TRUE
  redshift, for which the correct per-tracer β is H₀-independent, so an
  injection-based μ(H₀) contributes a spurious slope. (ii) The per-event numerator
  is biased high (+3.09 → +0.95) and carries all of the f-dependence; it holds the
  catalog redshift/sky prior and the informative powerlaw+peak mass term, which
  couples m_det/(1+z) to H₀.

  Consequence for planning: the f = 0.307 case looks accurate by cancellation, not
  by correctness — fixing only the selection normalisation would move it from
  −0.92 to +3.09. The next experiment needs a matched z-cut selection treatment
  (analytic β or z-cut-aware injections) AND an uninformative-mass arm to isolate
  the spectral-siren coupling in the numerator.
- **Boundary case.** The planted f_AGN = 1.0 set sits on the prior boundary, where
  an equal-tailed interval cannot cover the truth; the figure and summary report the
  one-sided interval for that point and label it as such.
- Numbers in the figures come from `results/summary.json`, which is generated from
  the grids — nothing is hand-typed.
