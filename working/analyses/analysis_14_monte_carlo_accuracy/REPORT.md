# Analysis 14: how accurate are the Monte Carlo sums of the likelihood?

17 Nov 2026. Rita jobs 1414628 (mixed set, realization 100; stopped by hand after its planned list
and the injection redraws) and 1414642 (all-AGN-hosted set, realization 103), darksirens-core
e7c3007. Numbers: `results/mc_accuracy_summary.json`; figure: `figs/mc_accuracy.{pdf,png}`.

## Question

The likelihood is a product of Monte Carlo sums: one over the posterior samples of each merger and
one over the injections that give the selection term. Every earlier analysis used 2000 posterior
samples per merger and one injection set, with the guard on the variance of ln L switched off
(`max_likelihood_variance = 1e6`). How much do the posterior of H0 and of f_AGN move when the same
data are evaluated with a fresh draw of the samples, or of the injections?

## Method

The simulated posterior of each merger is known in closed form, so its samples can be redrawn at
any number per merger (`scripts/mc_redraw.py`, validated in `results/validation_redraw.json`). For
each set the same grid is evaluated on independent redraws at 2000, 8000 and 32000 samples per
merger with the recorded injections, and on six injection sets (the recorded one and five new ones
of 1.5e8 draws) at 8000 samples with one redraw. The spread of the posterior median over redraws,
divided by the 68% half-width of the posterior, measures the Monte Carlo error.

## Results

Mixed set (input f_AGN = 0.30, both catalogs, H0 and f_AGN; 68% half-widths 1.1 km/s/Mpc and 0.050):

| samples per merger | redraws | spread of the H0 median | of the f_AGN median | mean f_AGN median | variance of ln L (samples) | s per call |
|---|---|---|---|---|---|---|
| 2000 | 20 | 0.47 km/s/Mpc (0.43 of the half-width) | 0.0072 (0.15) | 0.266 | 7.6 | 0.25 |
| 8000 | 12 | 0.28 (0.25) | 0.0046 (0.09) | 0.275 | 2.3 | 0.40 |
| 32000 | 6 | 0.20 (0.18) | 0.0020 (0.04) | 0.277 | 0.65 | 1.03 |
| six injection sets, 8000 | 6 | 0.22 (0.19) | 0.0043 (0.09) | | selection term 1.3 | |

All-AGN-hosted set (AGN catalog, H0 only; 68% half-width 0.53 km/s/Mpc, 90% half-width 0.88):

| samples per merger | redraws | spread of the H0 median | variance of ln L (samples) |
|---|---|---|---|
| 2000 | 20 | 0.37 km/s/Mpc (0.80 of the half-width) | 27.7 |
| 8000 | 20 | 0.25 (0.52) | 8.3 |
| 32000 | 20 | 0.13 (0.29) | 2.3 |
| six injection sets, 8000 | 6 | 0.27 (0.50) | selection term 2.6 |

1. **2000 samples per merger are too few.** On the mixed set the H0 median moves by 0.43 of its 68%
   half-width between redraws, and the half-width itself by a quarter. On the all-AGN set, where
   the posterior is narrow, the samples and the injections together move the median by 0.95 of the
   half-width, so the quoted interval understates the uncertainty by a factor 1.37. This is the
   factor 1.3 to 1.4 found empirically over 93 realizations (gap A2).
2. **The low f_AGN of every earlier analysis is the bias of the logarithm of a Monte Carlo mean.**
   The mean f_AGN median rises from 0.266 at 2000 samples to 0.277 at 32000. Adding half the
   variance of each merger's sum to ln L gives 0.277 to 0.278 at every sample count (gap A1).
3. **The injections matter as much as the samples.** One injection set of 1.5e8 draws (2.2e6
   detected) moves the H0 median by 0.19 of the half-width on the mixed set and by 0.50 on the
   all-AGN set. Beyond 8000 samples per merger the injections dominate.
4. **The width is stable.** At 8000 samples the 68% width changes by 4% between injection sets and
   by less than 10% between redraws; it is the position of the posterior that the Monte Carlo
   error moves.
5. **Core and the mocks disagreed below the minimum mass** (README): core cuts Gaussian-peak
   secondaries at 5 Msun and the old generator did not, which broke the all-AGN set of
   realization 100. All production event sets are now drawn from core's population.

## Settings adopted for production

- Mixed sets: 8000 samples per merger with the half-variance correction, and the realization's own
  injection set. Monte Carlo error of the H0 median: 0.31 of the 68% half-width (samples and
  injections in quadrature), which widens the effective interval by 5%; of f_AGN: 0.13.
- All-AGN-hosted sets and the AGN-hosted part of "hosts known": 32000 samples per merger and the six
  injection sets pooled (error 0.29 and 0.20 of the half-width, 0.35 together, 6% wider). With 8000
  samples and one injection set the error is 0.72 of the half-width.
- Every realization uses its own injections, so that the error of the selection term averages
  down in posteriors combined over realizations.
- Cost per likelihood call on an 80 GB A100: 0.25, 0.40 and 1.03 s at 2000, 8000 and 32000 samples.

## Files

`scripts/a14_mc_accuracy.py` (driver), `mc_redraw.py`, `validate_redraw.py`,
`make_injection_redraws.py`, `aggregate_mc_accuracy.py`, `make_figures.py`, `diag_*.py`;
`results/scans/*.json` (per scan), `results/mc_accuracy_summary.json`, `results/repro_*.json`,
`results/diag_*.json`.
