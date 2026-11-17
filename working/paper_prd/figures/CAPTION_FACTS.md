# Caption facts (figures of 17 Nov 2026, notes/PAPER_SKELETON.md Sec. 4)

Facts a caption writer needs and cannot read off the picture. No interpretation. The first-draft figures
and their facts are in `figures/v1/` (`CAPTION_FACTS_v1.md`). Rebuild: `python scripts/make_figures.py`
(all nine) or `python scripts/make_figures.py fig6` (one), jax env python; the figure-against-values check
at the end compares every drawn median and 90% interval end with `values/values.json` (0 mismatches).

## Conventions shared by all figures

- Width 7.0 in (`figure*`). Heights: fig1 5.70 in, fig2 2.90, fig3 3.60, fig4 3.00, fig5 2.90, fig6 2.60,
  fig7 2.60, fig8 6.36, fig9 2.60.
- **Placeholders.** Every element drawn from `values/placeholder_results.json` is a placeholder: drawn in
  red tones (strong red, salmon, maroon, rose), told apart by line style (solid, dashed, dash-dot, dotted)
  and by direct labels; each panel with any placeholder carries the small red word "placeholder". Its
  source per key is `exploratory` (an earlier run of the same analysis, file named in the JSON and in
  placeholders.txt) or `invented` (PAPER_SKELETON Sec. 7). When `values/production_map.json` names an
  existing production summary for a key, the same code draws it in the final palette (both catalogs bluish
  green, galaxy catalog blue, AGN catalog vermilion, no catalog dark gray, single population reddish
  purple, flux limits the four purples) and the word goes. Black elements are final: input lines and
  markers, the curves of the normalization identity and of constant merger rate per host, Fig. 1.
- Every interval, band and contour is 90%: 1-D intervals equal-tailed, 2-D regions highest-density.
  Every 1-D curve is a normalized probability density with units on its axis. Black dashed lines or a
  black "+" are inputs: H0 = 67.74 km/s/Mpc (`\MockHzero`), f_AGN = 0.30 (`\MockfAGN`), log10 n_AGN = -5
  (`\MockLognAGN`), mass peak 35 Msun (`\MockMuG`), offsets +5 Msun and +0.10 (`\MockDmuG`, `\MockDmuChi`).
- Names in the panels are those of PAPER_SKELETON Sec. 2 (both catalogs, galaxy catalog alone, AGN catalog
  alone, no catalog; population known, single population, host-dependent populations; realization A;
  combined over realizations). Each number below names its macro; red macros are placeholders.
- "Combined over realizations": log-likelihoods summed, parameters shared. Invented combinations are
  narrower than realization A by sqrt(N), N = 10 (population known, single population, single population
  on host-dependent events, no catalog), 5 (flux limits, host-dependent populations, observed-like
  spectrum, all-AGN and all-galaxy sets), 3 (number densities inferred).

## Fig. 1 `fig1_mocks` (final; unchanged from the first draft)

- (a) Slice of the catalog pair of realization A: |delta| < 3 deg, right ascension 0 to 180 deg, true
  redshift z < 0.33, in comoving distance (flat LCDM, H0 = 67.74, Omega_m = 0.3075). Small blue dots:
  268,767 galaxies; vermilion dots: 2,709 AGN; large blue circles: 26 galaxy-hosted detected mergers;
  large vermilion diamonds: 7 AGN-hosted (true host positions). Dashed arcs z = 0.1, 0.2, 0.3; outer arc
  z = 0.33; scale bar 500 Mpc comoving.
- (b) Top: completeness C(z) of the galaxy catalog for m < 21, 20, 19, 18 (purples, light to dark), ratio
  of flux-limited to complete counts in bins of 0.005; labels are the completeness inside z <= 0.3105
  (`\MockCompTwentyOne` ... `\MockCompEighteen`: 100.0, 81.4, 31.5, 9.5%). Bottom: true host redshifts of
  the 1000 detected mergers, counts per 0.01: 705 galaxy hosts, 295 AGN hosts.
- (c) Source-frame primary mass and (d) effective spin of the 1000 detected mergers of the host-dependent
  event set (643 galaxy-hosted, 357 AGN-hosted), each class normalized to unit area; dashed lines at the
  input peak locations 35 and 40 Msun and spin means 0 and 0.10.
- Not yet in the figure (REVIEW G1, deferred by the redesign): completeness labels at the margin, the
  redshift of completeness (`\MockzComplete...`) and the completeness weighted by the detected mergers.

## Fig. 2 `fig2_anchor` (all placeholder, source exploratory)

- Key `KnownBoth` etc.: equal populations, population known, complete catalogs.
- (a) Joint posterior of H0 and f_AGN: realization A, 90% region dashed outline (`\KnownBothHzero`,
  `\KnownBothfAGN`); combined over realizations, filled with solid outline (`\KnownBothHzeroComb`,
  `\KnownBothfAGNComb`; input at the `\KnownBothHzeroCL`% and `\KnownBothfAGNCL`% levels; correlation r =
  `\JointRhoCatA` for realization A, `\JointRhoComb` combined). Black "+": input. Axes H0 [65.5, 73], f_AGN [0.13, 0.43]. Placeholder data:
  the exploratory complete-catalog grids (9 realizations combined; production combines 10).
- (b) H0 posterior of realization A with both catalogs (solid, `\KnownBothHzero`), galaxy catalog alone
  (dashed, `\KnownGalHzero`), AGN catalog alone (dash-dot, `\KnownAGNHzero`), no catalog (dotted,
  `\KnownNoneHzero`, invented: 68.6 +- 3.2). The AGN-catalog-alone posterior rises to the upper end of the
  prior at 100 (peak density 3.9 per km/s/Mpc, off the scale of the panel, marked by an arrow and text).
  Axis H0 [56, 100] (prior [50, 100]).

## Fig. 3 `fig3_flux_limits`

- (a) Medians and 90% intervals of H0 (top) and f_AGN (bottom) of realization A against the flux limit of
  both catalogs: complete (open circle, `\KnownBothHzero`, `\KnownBothfAGN`), m < 21, 20, 19, 18
  (`\FluxTwentyOneHzero` ... `\FluxEighteenfAGN`), number densities at their inputs. Upper axis: the
  completeness weighted by the detected mergers, 100, 100, 99, 90, 49% (`\MockEventCompTwentyOne` ...
  `\MockEventCompEighteen`; complete catalogs 100% by construction). Placeholder (exploratory: the stored
  flux-limited grids, hosts normalized to z = 5; production normalizes to z = 1.10 with densities from the
  catalog counts).
- (b) Median and 90% interval of f_AGN against the assumed AGN number density over its input (0.5, 1, 2;
  log axis) at m < 21 (filled circles) and m < 18 (open diamonds), points offset by 1.5% in x for
  legibility. Placeholder (exploratory). Black curves: the prediction of the normalization identity,
  logit f(x) = logit f(1) + ln[F_AGN + x(1 - F_AGN)], from the factor-1 median, with F_AGN =
  `\NormFAGNTwentyOne` (solid, m < 21) and `\NormFAGNEighteen` (dashed, m < 18), hosts normalized to
  z = 5. The exact prediction from the full posterior gives the slopes `\SlopefAGNIdentityTwentyOne` and
  `\SlopefAGNIdentityEighteen` per decade against the measured `\SlopefAGNTwentyOne`,
  `\SlopefAGNEighteen`; a doubled density moves f_AGN by `\NormDeltafTwoEighteenFive` (predicted
  `\NormDeltafTwoEighteenFivePred`). Gray dotted: factor 1. Black dashed: input f_AGN.
- When `analyses/analysis_15_host_normalisation/results/normalisation_summary.json` exists the panel adds,
  in the final palette, Analysis 15's points (complete catalogs normalized to z = 5: black filled squares;
  to z = 1.10: black open squares; m < 18 normalized to z = 1.10: dark purple open diamonds) with their
  identity predictions as black dotted curves (`\NormfCompleteFive`, `\NormfCompleteFiveTwo`,
  `\NormfCompleteProd`, `\NormfCompleteProdTwo`, `\NormfEighteenProdHalf`, `\NormfEighteenProd`,
  `\NormfEighteenProdTwo`; F_AGN `\NormFAGNComplete`, 1, `\NormFAGNEighteenProd`).

## Fig. 4 `fig4_densities` (placeholder, source exploratory; number densities inferred, realization A)

- (a) Joint posterior of f_AGN and log10 n_AGN with both number densities inferred (log10 n_gal
  marginalized): 90% regions at m < 21 (dashed) and m < 18 (solid) (`\FreeTwentyOnefAGN`,
  `\FreeTwentyOneLognAGN`, `\FreeEighteenfAGN`, `\FreeEighteenLognAGN`); 50 x 50 histograms on the prior box
  f [0, 1] x [-6, -4], smoothed. Black curve: constant merger rate per host at the input value
  `\MockRateRatio`, with n_gal and H0 at their inputs (hosts normalized to z = 5). Gray: the 90% region
  allowed by the priors once the merger rate per host is measured (2e6 draws from the priors weighted by
  the known-density likelihood of m < 18 mapped through the identity; `\FreeEighteenLognAGNPriorUL`).
  Black "+": input (0.30, -5). Axes span the prior box.
- (b) Posterior of the merger rate per AGN over that per galaxy at m < 21, 20, 19, 18 (solid, dashed,
  dash-dot, dotted; `\FreeTwentyOneRateRatio` ... `\FreeEighteenRateRatio`), computed per posterior sample
  as [f_AGN/(1 - f_AGN)] Z_gal/Z_AGN with that sample's H0, n_gal, n_AGN. Black dashed: input
  `\MockRateRatio` (42.8). With the densities at their inputs the same ratio is `\FluxEighteenRateRatio`.

## Fig. 5 `fig5_population` (placeholder; population known with both catalogs exploratory, the rest invented)

- (a) H0 posteriors of realization A: population known, both catalogs (solid, `\KnownBothHzero`);
  population known, no catalog (dotted, `\KnownNoneHzero`); single population, both catalogs (dashed,
  `\SingleBothHzero`); single population, no catalog (dash-dot, `\SingleNoneHzero`). Axis [50, 90].
  Width ratios no catalog over both catalogs: `\MassScaleShareKnown`, `\MassScaleShareSingle`.
- (b) Joint posterior of H0 and the mass-peak location, single population: no catalog (dash-dot outline,
  correlation `\CorrHzeroMuGNone`) and both catalogs (dashed outline, filled, `\CorrHzeroMuGBoth`).
  Invented: 2-D Gaussians with the stated correlations. Black "+": input (67.74, 35).

## Fig. 6 `fig6_host_dependent` and Fig. 7 `fig7_observed_like` (placeholders)

- Host-dependent events (AGN-hosted mergers: mass peak +5 Msun, spin mean +0.10), both catalogs; Fig. 6
  peak-dominated spectrum (keys `HD...`), Fig. 7 observed-like spectrum (keys `Obs...`). All curves
  combined over realizations.
- (a) H0: single population (dashed, `\HDSingleHzeroComb` / `\ObsSingleHzeroComb`; bias of the median
  `\HDBiasHzero` / `\ObsBiasHzero`) and host-dependent populations (solid, `\HDHostHzeroComb` /
  `\ObsHostHzeroComb`). Log10 Bayes factor of the second against the first: `\HDLB` / `\ObsLB` for
  realization A, `\HDHostLBComb` / `\ObsHostLBComb` combined.
- (b) Joint posterior of the AGN offsets of the mass peak and of the spin mean, host-dependent
  populations, 90% region (`\HDHostDmuGComb`, `\HDHostDmuChiComb` / `\ObsHost...`). Black "+": input
  (5, 0.10); black "x": no offset (0, 0); gray dotted lines at zero.
- (c) f_AGN: the same two analyses and line styles (`\HDSinglefAGNComb`,
  `\HDHostfAGNComb`, `\ObsSinglefAGNComb`, `\ObsHostfAGNComb`).
- Sources. Fig. 6: single population invented (Sec. 7: H0 70.9 +- 2.6 for realization A, bias kept in the
  combination); host-dependent populations combined invented from the exploratory eight-parameter run of
  realization A (its offsets: `\HDHostDmuG`, `\HDHostDmuChi`; H0 prior [60, 76] in that run), the 2-D
  region a Gaussian with that run's correlation of the two offsets. Fig. 7: all invented (Sec. 7); the
  observed-like inputs of the mass spectrum are not on disk yet.
- Input lines stop below the labels. In (b) of Fig. 6 the input lies outside the 90% region of the
  placeholder (the exploratory spin offset, carried into the invented combination).

## Fig. 8 `fig8_summary` (all placeholder)

- One row per analysis in the order of Sec. IV, grouped under bold headings: population known, complete
  catalogs (both catalogs, galaxy catalog alone, AGN catalog alone, no catalog); flux-limited catalogs
  (m < 21, 20, 19, 18 with densities known; m < 18 with number densities inferred); unknown population
  (single population with both catalogs and with no catalog); host-dependent populations (single
  population, host-dependent populations, no catalog); observed-like spectrum (the same three).
- Left: H0, right: f_AGN; marker = median, bar = 90% interval of the combination over realizations
  (`\<Key>HzeroComb`, `\<Key>fAGNComb`). Black dashed: inputs. "not inferred": f_AGN is not a parameter of
  that analysis. The AGN-catalog-alone interval lies at the upper prior edge (99.9 to 100) and is marked by
  an arrow and text. Axes H0 [60, 84], f_AGN [0, 1]. Text 10 pt at print size.

## Fig. 9 `fig9_validation` (placeholder)

- (a) H0 posteriors combined over realizations with both catalogs: population known (solid,
  `\KnownBothHzeroComb`, exploratory, input at the `\KnownBothHzeroCL`% level), single population (dashed,
  `\SingleBothHzeroComb`, invented), host-dependent populations on host-dependent events (dash-dot,
  `\HDHostHzeroComb`, invented). Axis [65.3, 70.7].
- (b) f_AGN posteriors combined over realizations for inputs 0 (solid, `\CtrlZerofAGNComb`, 90% upper
  limit `\CtrlZerofAGNCombUL`), 0.10 (dashed, `\CtrlTenthfAGNComb`) and 0.30 (dash-dot,
  `\KnownBothfAGNComb`, exploratory); black dashed lines at the three inputs. The two control sets are
  invented.
