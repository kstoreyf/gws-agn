# RESULTS — experiment_h0f_baseline

The AGN-hosted fraction of gravitational-wave sources is recovered across its full
range from a two-tracer dark-siren analysis, while the inferred expansion rate carries
a systematic offset that grows with that fraction.

darksirens master `2b86a2d`; `dark_sirens` with field-convention sky weighting, K=2
mixture `[galaxies, AGN]` at the complete-catalog limit; fixed population; Om₀ pinned;
selection validity guard `N_eff > 5·N_obs`. 1000 events per set, planted AGN-hosted
fractions {0.0099, 0.307, 0.703, 1.0}.

## The AGN-hosted fraction

Recovered from 41-point scans at the true expansion rate (`figs/fig_f_recovery.pdf`):

| planted f_AGN | recovered (median, 68%) | offset |
|---|---|---|
| 0.0099 | 0.0195 [0.006, 0.039] | +0.010 |
| 0.307 | 0.3221 [0.300, 0.347] | +0.015 |
| 0.703 | 0.6872 [0.662, 0.712] | −0.016 |
| 1.0 | 0.9750, one-sided [0.966, 1] | −0.025 |

The planted value lies inside the 68% interval at every interior point. The f = 1 case
sits on the prior boundary, where an equal-tailed interval cannot cover it; its
one-sided interval does.

## The joint constraint

61×57 and 71×61 grids refined around each peak (`figs/fig_joint_h0f.pdf`):

| planted f_AGN | H₀ | f_AGN | ρ |
|---|---|---|---|
| 0.307 | 66.81 (+0.27/−0.26) | 0.325 (+0.020/−0.020) | −0.01 |
| 0.703 | 64.36 (+0.28/−0.23) | 0.711 (+0.018/−0.019) | −0.30 |

The two parameters are essentially uncorrelated at the lower fraction, so the
fraction is measured independently of the distance scale there. H₀ is recovered
**1.0 km/s/Mpc low at f_AGN = 0.307 and 3.4 low at f_AGN = 0.703**, well outside the
statistical interval in both cases, with the deficit growing as the mixture leans on
the sparse tracer.

## The expansion-rate offset

The offset is not attributable to the mixture weight: f_AGN is recovered correctly at
fixed H₀ and along the joint surface, and ρ ≈ 0 at the fiducial point. Decomposing the
likelihood at fixed mixture weight into its per-event and selection terms
(`results/h0_decomposition_*.json`) shows the selection term supplying ~4 km/s/Mpc of
peak shift almost independently of f, with the per-event term carrying all of the
f-dependence. That decomposition localises where the sensitivity lives but does not by
itself identify an error, since the two terms are expected to oppose one another and
only their sum need be unbiased.

Two candidate explanations were tested and eliminated:

- **Host rate weighting.** The events' hosts were drawn uniformly over eligible
  catalog entries, corresponding to a redshift-rate index of 1, while the analysis
  assumes 0. Repeating the scans at the matching value moves H₀ by ≤ 0.16 km/s/Mpc:
  under field-convention weighting the factor enters both the per-event host prior and
  the survey-global normalisation and cancels.
- **The catalog's redshift extent, by contrast, matters strongly.** Truncating both
  catalogs at z ≤ 1 with the events held fixed moves H₀ by −4.1 km/s/Mpc at
  f_AGN = 0.307. The host catalog's redshift boundary, not the event sample, controls
  the offset at this level.

The follow-up in `../experiment_matched_mock` shows the offset persists at 4.4σ on a
mock whose generative process matches the analysis in every ingredient, which places
its origin in the inference rather than in the simulated data.

## What these numbers cost

Fixed fiducial population (masses informative but held at truth), Om₀ and the survey
nuisances fixed, one catalog realisation, complete catalogs, and detection by a hard
cut on true redshift. Grid scans under a flat prior rather than marginalised
posteriors. The selection guard is the historical effective-sample-size floor; the
newer total-variance criterion is inactive, and at 2000 posterior samples per event
this mock does not meet it (see `../../../gw_agn_darksirens_2b86a2d_2026-07-29/`).
Guard rejections do not shape any quoted result: all f scans admit every cell, and
where H₀ cells are rejected they lie ≥ 19 log-likelihood units below the peak carrying
< 3×10⁻⁶ of the posterior mass.

## Reproducing

`scripts/run_experiment.sh [all|fscan|h0scan|joint|jointzoom]` then
`python scripts/make_figures.py`. Every number quoted above is regenerated into
`results/summary.json` and `results/table_h0f.tex` from the grids; none is hand-typed.
Internal validity notes are in `README.md`.
