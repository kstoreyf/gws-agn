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
  and by direct labels; no word in a panel marks a placeholder (the red colour alone carries it). Its
  source per key is `exploratory` (an earlier run of the same analysis, file named in the JSON and in
  placeholders.txt) or `invented` (PAPER_SKELETON Sec. 7). When `values/production_map.json` names an
  existing production summary for a key, the same code draws it in the final palette (both catalogs bluish
  green, galaxy catalog blue, AGN catalog vermilion, no catalog dark gray, single population reddish
  purple, flux limits the four purples). Black elements are final: input lines and
  markers, the curves of the normalization identity and of constant merger rate per host, Fig. 1.
- Every interval, band and contour is 90%: 1-D intervals equal-tailed, 2-D regions highest-density.
  Every 1-D curve is a normalized probability density with units on its axis. Black dashed lines or a
  black "+" are inputs: H0 = 67.74 km/s/Mpc (`\MockHzero`), f_AGN = 0.30 (`\MockfAGN`), log10 n_AGN = -5
  (`\MockLognAGN`), mass peak 35 Msun (`\MockMuG`), offsets +5 Msun and +0.10 (`\MockDmuG`, `\MockDmuChi`).
- Names in the panels are those of PAPER_SKELETON Sec. 2 (both catalogs, galaxy catalog alone, AGN catalog
  alone, no catalog; population known, single population, host-dependent populations; realization A;
  combined over realizations). Each number below names its macro; red macros are placeholders.
- "Combined over realizations": log-likelihoods summed, parameters shared. N per key is that of the run list
  (`analyses/analysis_16_production_runs/runs/RUNLIST.yaml`; macros `\Nreal...`): 10 for population known with
  both catalogs, single population (both catalogs and no catalog) and the single-population analysis of
  host-dependent mergers; 5 for each catalog alone and no catalog with the population known, the all-AGN and
  all-galaxy sets, the flux limits, host-dependent populations, no catalog on host-dependent mergers, the
  observed-like spectrum (Stage 5 of the run list) and the two control sets (the experiment's run list); 3 with
  the number densities inferred. Invented combinations (placeholders) are realization A combined with N - 1
  realizations of A's width centred at the expected value: narrower by sqrt(N), median input + bias +
  (median_A - input - bias)/N (since 17 Nov; before, the shift was divided by sqrt(N) and the input stayed
  as far out as in realization A). Exploratory combinations sum the exploratory realizations that are also
  production realizations (each catalog alone, all-AGN and all-galaxy sets: realizations 100, 101, 102, 103,
  105) and all nine exploratory realizations for population known with both catalogs.

## Fig. 1 `fig1_mocks` (final; every panel drawn from the production event sets of realization A since 17 Nov)

- (a) Slice of the catalog pair of realization A: |delta| < 3 deg, right ascension 0 to 180 deg, true
  redshift z < 0.33, in comoving distance (flat LCDM, H0 = 67.74, Omega_m = 0.3075). Small blue dots:
  268,767 galaxies; vermilion dots: 2,709 AGN; large blue circles: 25 galaxy-hosted detected mergers;
  large vermilion diamonds: 9 AGN-hosted (true host positions) of the equal-population event set that the
  production runs analyse (`data/seed100/events/events_corepop.h5`). Dashed arcs z = 0.1, 0.2, 0.3; outer
  arc z = 0.33; scale bar 500 Mpc comoving.
- (b) Top: completeness C(z) of the galaxy catalog for m < 21, 20, 19, 18 (purples, light to dark), ratio
  of flux-limited to complete counts in bins of 0.005; labels are the completeness inside z <= 0.3105
  (`\MockCompTwentyOne` ... `\MockCompEighteen`: 100.0, 81.4, 31.5, 9.5%). Bottom: true host redshifts of
  the 1000 detected mergers of the same set (`events_corepop.h5`), counts per 0.01: 702 galaxy hosts, 298 AGN
  hosts (fraction `\MockfAGNRealised` = 0.298).
- (c) Source-frame primary mass and (d) effective spin of the 1000 detected mergers of the host-dependent
  event set of realization A that the production runs analyse (`data/seed100/events/
  events_marked_dmu0p10_dmuG5_corepop.h5`; 640 galaxy-hosted, 360 AGN-hosted = `\MockHDNagn`, fraction
  `\MockHDfDet` = 0.360), each class normalized to unit area; dashed lines at the input peak locations 35 and
  40 Msun and spin means 0 and 0.10.
- Until 17 Nov panels (c, d) drew the older `events_marked_dmu0p10_dmuG5.h5` (643 and 357) and panels (a, b)
  the older equal-population `events.h5` (26 and 7 mergers in the slice; 705 and 295 hosts).
- Not yet in the figure (REVIEW G1, deferred by the redesign): completeness labels at the margin, the
  redshift of completeness (`\MockzComplete...`) and the completeness weighted by the detected mergers.

## Fig. 2 `fig2_anchor` (all placeholder, source exploratory)

- Key `KnownBoth` etc.: equal populations, population known, complete catalogs.
- (a) Joint posterior of H0 and f_AGN: realization A, 90% region dashed outline (`\KnownBothHzero`,
  `\KnownBothfAGN`); combined over realizations, filled with solid outline (`\KnownBothHzeroComb`,
  `\KnownBothfAGNComb`; input at the `\KnownBothHzeroCL`% and `\KnownBothfAGNCL`% levels; correlation r =
  `\KnownBothCorrHzerofAGN` (= `\JointRhoCatA`) for realization A, `\KnownBothCorrHzerofAGNComb` (= `\JointRhoComb`) combined). Black "+": input. Axes H0 [65.5, 73], f_AGN [0.13, 0.43]. Placeholder data:
  the exploratory complete-catalog grids (9 realizations combined; production combines 10).
- (b) H0 posterior of realization A with both catalogs (solid, `\KnownBothHzero`), galaxy catalog alone
  (dashed, `\KnownGalHzero`), AGN catalog alone (dash-dot, `\KnownAGNHzero`), no catalog (dotted,
  `\KnownNoneHzero`, invented: 68.6 +- 3.2). The AGN-catalog-alone posterior rises to the upper end of the
  prior at 100 (peak density 3.9 per km/s/Mpc, off the scale of the panel, marked by an arrow and text).
  Axis H0 [56, 100] (prior [50, 100]).

<!-- fig_classes: written by scripts/make_figures.py; do not edit -->
## Fig. `fig_classes` (Sec. IV B, after `fig2_anchor`; written by the figure script from values.json)

- Width 7.0 in, height 3.45 in. (a), (b): the mean over the 5 realizations of the experiment (`\NrealClasses`; realizations [100, 101, 102, 103, 105]) of the 90% half-width of H0 of each realization (its run summary), against the input f_AGN; bars from the smallest to the largest half-width over those realizations; population known, complete catalogs, 1000 detected mergers. Points of one f are spread by up to +-0.05 in f so that they do not overlap. Horizontal lines: the mean half-width (no bar). Final colours: host class unknown bluish green (circles, solid), host class known black (triangles, dotted), host-dependent populations orange (squares, dashed; twice the offsets: brown diamonds, dashed; inferred with H0: open orange square), no catalog dark gray (long dashes, labelled below the line), galaxy catalog alone blue (short dashes, labelled below the line). An element whose macro is red is drawn in red tones (same marker and line style); a connecting line is red when any of its points is. No word in a panel marks a placeholder.
- (a) peak-dominated spectrum, equal populations except the host-dependent series (AGN-hosted mergers heavier by \MockDmuG Msun, spin mean higher by \MockDmuChi). (b) observed-like spectrum, the same analyses at the fractions that were run. The open square is the host-dependent populations analysis with the populations and H0 inferred (keys HDHost, ObsHost: \NrealHD and \NrealObs realizations).
- (c) realization A, host-dependent populations, input f_AGN = 0.3: histograms (40 bins on [0, 1], each class normalized to unit area) of the probability of an AGN host of each of the 1000 mergers at the posterior median of its grid, f L_AGN / (f L_AGN + (1 - f) L_gal); galaxy-hosted filled blue, AGN-hosted dark vermilion line; top: positions alone (one population held at its input), bottom: positions with masses and spins (the two populations at their inputs); triangles at the means with their values printed; black dashed line at the input f_AGN = 0.3 (`\MockfAGN`). The key of (c) sits above it in the top margin; the key of (a, b) above them.

Series, points and their macros (each point: mean `\HW...`, bar `...Min` to `...Max`):
- (a) Host class unknown: 5 points (5 real, 0 red): f=0 `\HWUnknownZero` = 3.53 (real) [2.85, 4.75]; f=0.1 `\HWUnknownTen` = 3.49 (real) [2.38, 5.35]; f=0.3 `\HWUnknownThirty` = 2.01 (real) [1.82, 2.40]; f=0.5 `\HWUnknownFifty` = 1.36 (real) [1.03, 1.62]; f=1 `\HWUnknownHundred` = 0.82 (real) [0.80, 0.85]
- (a) Host-dependent populations, $\Delta\mu_{\rm G}=5\,M_\odot$, $\Delta\mu_\chi=0.10$: 3 points (1 real, 2 red): f=0.1 `\HWPopsTen` = 2.21 (red) [1.99, 2.43]; f=0.3 `\HWPopsThirty` = 1.64 (real) [1.44, 1.86]; f=0.5 `\HWPopsFifty` = 1.19 (red) [1.07, 1.31]
- (a) Host-dependent populations, twice those offsets: 3 points (0 real, 3 red): f=0.1 `\HWPopsTwiceTen` = 2.11 (red) [1.90, 2.32]; f=0.3 `\HWPopsTwiceThirty` = 1.40 (red) [1.26, 1.54]; f=0.5 `\HWPopsTwiceFifty` = 1.12 (red) [1.01, 1.23]
- (a) Host class known: 5 points (5 real, 0 red): f=0 `\HWKnownZero` = 3.55 (real) [2.96, 4.57]; f=0.1 `\HWKnownTen` = 1.89 (real) [1.43, 2.13]; f=0.3 `\HWKnownThirty` = 1.34 (real) [1.28, 1.40]; f=0.5 `\HWKnownFifty` = 1.10 (real) [0.98, 1.25]; f=1 `\HWKnownHundred` = 0.81 (real) [0.80, 0.83]
- (a) Host-dependent populations inferred with $H_0$: 1 point (0 real, 1 red): f=0.3 `\HWHost` = 2.10 (red) [1.89, 2.31]
- (a) No catalog (horizontal line): 1 point (1 real, 0 red): `\HWNone` = 5.57 (real) (range over the realizations 3.97 to 6.06, not drawn)
- (a) Galaxy catalog alone, $f_{\rm AGN}=0$ (horizontal line): 1 point (1 real, 0 red): `\HWKnownZero` = 3.55 (real) (range over the realizations 2.96 to 4.57, not drawn)
- (b) Host class unknown: 3 points (3 real, 0 red): f=0 `\HWObsUnknownZero` = 3.49 (real) [2.64, 4.66]; f=0.3 `\HWObsUnknownThirty` = 1.98 (real) [1.82, 2.33]; f=1 `\HWObsUnknownHundred` = 0.84 (real) [0.70, 1.06]
- (b) Host-dependent populations, $\Delta\mu_{\rm G}=5\,M_\odot$, $\Delta\mu_\chi=0.10$: 1 point (1 real, 0 red): f=0.3 `\HWObsPopsThirty` = 1.61 (real) [1.30, 1.78]
- (b) Host class known: 3 points (3 real, 0 red): f=0 `\HWObsKnownZero` = 3.49 (real) [2.57, 4.35]; f=0.3 `\HWObsKnownThirty` = 1.34 (real) [1.17, 1.47]; f=1 `\HWObsKnownHundred` = 0.83 (real) [0.68, 1.06]
- (b) Host-dependent populations inferred with $H_0$: 1 point (0 real, 1 red): f=0.3 `\HWObsHost` = 2.30 (red) [2.07, 2.53]
- (b) No catalog (horizontal line): 1 point (1 real, 0 red): `\HWObsNone` = 8.44 (real) (range over the realizations 7.15 to 9.33, not drawn)
- (b) Galaxy catalog alone, $f_{\rm AGN}=0$ (horizontal line): 1 point (1 real, 0 red): `\HWObsKnownZero` = 3.49 (real) (range over the realizations 2.57 to 4.35, not drawn)
- (c) Positions alone: real; 360 AGN-hosted and 640 galaxy-hosted mergers; printed means: 0.30 = `\ProbGalPos` = 0.30 (real), 0.37 = `\ProbAGNPos` = 0.37 (real)
- (c) Positions with masses and spins: real; 360 AGN-hosted and 640 galaxy-hosted mergers; printed means: 0.28 = `\ProbGalPops` = 0.28 (real), 0.51 = `\ProbAGNPops` = 0.51 (real)
- Numbers printed in the panels: the four means of (c) above; the legend's offsets 5 (`\MockDmuG`) and 0.10 (`\MockDmuChi`). The figure check compares every drawn point, bar end and line with its macro (value and printed text) and the printed text with the macro's text.
<!-- end fig_classes -->

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
  Invented: 2-D Gaussians with the stated correlations, on a 261 x 261 lattice reaching 5 sd (cut at the H0
  prior edge 50); both 90% regions close inside their lattices (checked when the figure is drawn; no-catalog
  region H0 57.3 to 80.7, mass peak 32.5 to 36.7 Msun). Axes fitted to the two regions. "both catalogs" is
  attached to its region by a thin line. Black "+": input (67.74, 35).

## Fig. 6 `fig6_host_dependent` and Fig. 7 `fig7_observed_like` (placeholders)

- Host-dependent events (AGN-hosted mergers: mass peak +5 Msun, spin mean +0.10), both catalogs; Fig. 6
  peak-dominated spectrum (keys `HD...`), Fig. 7 observed-like spectrum (keys `Obs...`). All curves
  combined over realizations.
- (a) H0: single population (dashed, `\HDSingleHzeroComb` / `\ObsSingleHzeroComb`; bias of the median
  `\HDBiasHzero` / `\ObsBiasHzero`) and host-dependent populations (solid, `\HDHostHzeroComb` /
  `\ObsHostHzeroComb`). Log10 Bayes factor of the second against the first: `\HDLB` / `\ObsLB` for
  realization A, `\HDHostLBComb` / `\ObsHostLBComb` combined.
- (b) Joint posterior of the AGN offsets of the mass peak and of the spin mean, host-dependent
  populations, combined over realizations like (a) and (c), 90% region (`\HDHostDmuGComb`,
  `\HDHostDmuChiComb` / `\ObsHost...`); a 2-D Gaussian with these combined summaries and the correlation of
  the two offsets in the exploratory eight-parameter run (r = -0.13). Black "+": input (5, 0.10), inside the
  region in both figures; black "x": no offset (0, 0); gray dotted lines at zero.
- (c) f_AGN: the same two analyses and line styles (`\HDSinglefAGNComb`,
  `\HDHostfAGNComb`, `\ObsSinglefAGNComb`, `\ObsHostfAGNComb`).
- Sources. Fig. 6: single population invented (Sec. 7: H0 70.9 +- 2.6 for realization A, bias kept in the
  combination, N = 10); host-dependent populations combined (N = 5) invented from the exploratory
  eight-parameter run of realization A (its offsets: `\HDHostDmuG`, `\HDHostDmuChi`; H0 prior [60, 76] in that
  run) by the rule of the conventions above. Fig. 7: all invented (Sec. 7 as revised on 17 Nov evening, N = 5:
  realization A H0 68.4 +- 2.4 single population, bias 0.7 kept in the combination, 67.9 +- 2.3 host-dependent
  populations; f_AGN 0.27 +- 0.09 and 0.30 +- 0.08; offsets 4.5 +- 3.5 Msun and 0.10 +- 0.04); population quantities of
  the `Obs...` keys are now centred on the observed-like inputs (mass peak `\ObsMuG` = 34, width
  `\ObsSigmaG` = 3.6 Msun; not drawn in this figure).
- Input lines stop below the labels. The observed-like spectrum has `\ObsPeakFrac` = 0.092 of the mergers in
  its Gaussian peak, but `\ObsDetPeakShare` = 0.42 of the detected mergers (`\MockDetPeakShare` = 0.97 for
  the peak-dominated spectrum): "few mergers in the peak" holds for the population, not for the detections.

## Fig. 8 `fig8_summary` (all placeholder)

- One row per analysis in the order of Sec. IV, grouped under the subsection titles of Sec. IV, read from
  `main.tex` when the figure is drawn (bold): "Known population and complete catalogs" (both catalogs, galaxy
  catalog alone, AGN catalog alone, no catalog); "Flux-limited catalogs" (m < 21, 20, 19, 18, number
  densities known; m < 18, number densities inferred); "An unknown population" (single population, both
  catalogs; single population, no catalog); "Populations that depend on the host" and "A mass spectrum like
  the observed one" (each: single population, both catalogs; host-dependent populations, both catalogs;
  single population, no catalog). The last two groups use mergers with host-dependent populations, the
  others mergers with equal populations. No row label repeats its heading.
- Intervals, uniformly for every row and both panels: median (marker) and 90% interval (bar) of the
  posterior combined over realizations (`\<Key>HzeroComb`, `\<Key>fAGNComb`; N per row as in the conventions
  above). For "m < 18, number densities inferred" the f_AGN placeholder of the combination is taken equal to
  realization A (it follows its prior along the curve of constant rate per host). Realization A is not drawn
  in this figure. Black dashed: inputs. "not inferred": f_AGN is not a parameter of that analysis. The
  AGN-catalog-alone interval lies at the upper prior edge (99.9 to 100) and is marked by an arrow and text.
  Axes H0 [60, 84], f_AGN [0, 1]. Text 10 pt at print size.

## Fig. 9 `fig9_validation` (placeholder)

- (a) H0 posteriors combined over realizations with both catalogs: population known (solid,
  `\KnownBothHzeroComb`, exploratory, input at the `\KnownBothHzeroCL`% level), single population (dashed,
  `\SingleBothHzeroComb`, invented), host-dependent populations on host-dependent events (dash-dot,
  `\HDHostHzeroComb`, invented). Axis [65.3, 70.7].
- (b) f_AGN posteriors combined over realizations for inputs 0 (solid, `\CtrlZerofAGNComb`, 90% upper
  limit `\CtrlZerofAGNCombUL`), 0.10 (dashed, `\CtrlTenthfAGNComb`) and 0.30 (dash-dot,
  `\KnownBothfAGNComb`, exploratory); black dashed lines at the three inputs. The two control sets are
  invented.
