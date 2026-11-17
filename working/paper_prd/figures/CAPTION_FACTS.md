# Caption facts (figures of 16 Nov 2026)

Facts a caption writer needs and cannot read off the picture. No interpretation.
Rebuild: `python scripts/make_figures.py` (all) or `python scripts/make_figures.py fig4` (one), with the
jax env python. Output is byte-identical between runs. Paths below are relative to `working/`.

## Conventions shared by all figures

- Widths: every figure is drawn at 7.0 in (`figure*`, `\textwidth`). Heights: fig1 5.70 in, fig2 3.25,
  fig3 2.75, fig4 2.60, fig5 4.30, fig6 4.20, fig7 4.50, fig8 2.15, figC1 7.00.
- Colours: blue = galaxies, galaxy catalog, galaxy-hosted mergers, or an analysis with the galaxy
  catalog alone. Vermilion = AGN, AGN catalog, AGN-hosted mergers, AGN number density. Bluish green =
  an analysis with both catalogs. Purples (light to dark) = flux limits m < 21, 20, 19, 18, that is
  completeness inside the horizon 100.0%, 81.4%, 31.5%, 9.5%. Black dashed lines or black "+" = input
  values of the simulated catalogs. Grey dotted = prior density or prior-range edge.
- Line styles in Figs. 4 and 5 (population model): solid = host-dependent model with the reference
  population fixed at its input (mu_G = 35, mu_chi = 0) and both offsets free ("reference fixed");
  dash-dot = reference peak location and spin mean also free, six parameters ("reference free");
  dotted = reference peak location, spin mean and both widths free, eight parameters ("eight
  parameters"); dashed = common population (one mass and spin distribution for both host classes,
  both offsets set to zero; its mass and spin parameters are fixed at the galaxy-class input values).
- Every interval, band and contour is 90%: 1-D intervals are equal-tailed (as in the text); 2-D
  contours are 90% highest-posterior-density regions. Shaded areas under 1-D curves are the 90%
  equal-tailed interval.
- Every 1-D curve is a normalised probability density with units on the y axis, normalised over
  the prior range of its analysis. Grid posteriors are normalised over the stored lattice, which is
  the flat prior box except where stated (the reference-fixed lattice of Figs. 4a and 5 covers
  f [0.05, 0.55], Delta mu_G [1, 9], Delta mu_chi [0.025, 0.235]; the outside carries negligible mass).
- Grid posteriors are the spline-refined posteriors of `scripts/build_values.py` (GridPost, same
  refinement factors and nodes). Nested-sampling posteriors are drawn from the stored equal-weight
  samples: 1-D by a Gaussian kernel density reflected at the prior edges (bandwidth 0.3 sample
  standard deviations; values listed per figure), 2-D by a histogram on the prior box smoothed by a
  Gaussian kernel. In Fig. 8 the lattice curve is cubic-interpolated between lattice nodes for
  drawing only; the shaded interval ends are the lattice values.
- Figure-against-values check (end of `make_figures.py`): 884 numbers compared (medians and 90%
  interval ends of every drawn posterior, per-catalog summaries, completeness labels, catalog
  counts, input lines), 0 mismatches with `values/values.json`.
- Code versions behind the results (from notes/GAPS.md A4; not caption material): Figs. 2, 3, 8 and the
  "complete" point of Fig. 6 on darksirens 2b86a2d; flux-limited points of Fig. 6 and Fig. 7 on
  0c5b3db; Fig. 4 (a, b, c), Fig. 5 solid and dash-dot curves and Fig. 5(d) on af896ca; the
  eight-parameter curves (Figs. 4a, 5, C1) on darksirens-core e7c3007 with a selection-variance cap
  of 20.

## Fig. 1 `fig1_mocks` (panels a to d; DESIGN's "(c)" is split into (c) mass and (d) spin)

- (a) Slice of the reference catalog pair: declination |delta| < 3 deg, right ascension 0 to 180 deg,
  true redshift z < 0.33, plotted in comoving distance (flat LCDM, H0 = 67.74, Omega_m = 0.3075);
  angle = right ascension. Small light-blue dots: 268,767 galaxies. Vermilion dots: 2,709 AGN. Large
  blue circles: 26 galaxy-hosted detected mergers; large vermilion diamonds: 7 AGN-hosted detected
  mergers (the hosts in the slab of the 1000 detected events of the reference catalog, true host
  positions). Dashed arcs at z = 0.1, 0.2, 0.3; outer arc z = 0.33. Scale bar 500 Mpc (comoving).
  Read: rows of `data/seed100/catalogs/catalog_{gal,agn}_complete.h5` with z < 0.33 (columns ra, dec,
  z), `data/seed100/events/events.h5` (truth/ra, dec, z, host_type).
- (b) Top: completeness C(z) of the galaxy catalog for m < 21, 20, 19, 18 (purples, light to dark),
  ratio of flux-limited to complete counts in bins of dz = 0.005 (AGN carry their host galaxy's
  apparent magnitude, so the AGN catalog has the same C(z)). Labels = completeness inside the
  horizon (z <= 0.3105): 100.0%, 81.4%, 31.5%, 9.5% (macros `\MockCompTwentyOne` ... `\MockCompEighteen`;
  the m < 21 curve is 1 over the whole plotted range). Bottom: true host redshifts of the 1000
  detected events of the reference catalog, counts per dz = 0.01: 705 galaxy hosts (blue), 295 AGN
  hosts (vermilion). Read: `catalog_gal_{complete,m21,m20,m19,m18}.h5` (z column, z < 0.34),
  `events/events.h5`; counts inside the horizon were asserted equal to `META.json`.
- (c) True source-frame primary mass, (d) true effective spin of the 1000 detected events of the
  host-dependent mock (643 galaxy-hosted, 357 AGN-hosted), each class normalised to unit area. Bins
  2.5 Msun on [5, 80] and 0.04 on [-0.5, 0.6]; no event outside. Black dashed lines: input peak
  locations 35 (galaxies) and 40 Msun (AGN) in (c); input spin means 0 and 0.10 in (d). Read:
  `data/seed100/events/events_marked_dmu0p10_dmuG5.h5` (truth/m1src, chieff, host_type).

## Fig. 2 `fig2_two_catalogs`

- Joint (H0, f_AGN) posterior, complete catalogs, two-population model with both catalogs.
  Panels A to I: one simulated catalog each (9 catalogs; A to I = files joint_s100, s101, s102,
  s103, s105, s114, s115, s116, s117). Filled green region with outline: 90% HPD region.
- Last panel ("combined"): thin green outlines = the nine 90% regions of A to I; filled region =
  90% HPD region of the combined posterior (sum of the nine log-likelihood grids, H0 and f_AGN shared).
- Black dashed lines: input H0 = 67.74 and f_AGN = 0.30.
- Axes identical in all panels: H0 [62.5, 74.5], f_AGN [0, 0.56]. Prior: H0 [50, 100], f_AGN [0, 1].
  All 90% regions lie inside the axes; at most 0.1% of any 1-D marginal falls outside (catalog I).
- Combined marginals (macros): H0 = `\JointHzeroComb` (input at the 89% level), f_AGN = `\JointfComb`
  (input at the 70% level); 2-D: input on the boundary of the 88% HPD region (`\JointCombLevelTwoD`).
- Read: `analyses/analysis_2_complete_catalog_H0_fagn/results/joint_s{100,...,117}.h5`.

## Fig. 3 `fig3_host_information`

- (a) H0 posteriors, one thin curve per simulated catalog: 93 catalogs with 1000 galaxy-hosted
  mergers analysed with the galaxy catalog (blue) and the same 93 catalogs with 1000 AGN-hosted
  mergers analysed with the AGN catalog (vermilion). Black dashed: input H0. Axis H0 [52, 84]
  (prior [50, 100]); every curve has less than 0.1% of its mass outside. Mean 90% half-widths:
  `\PureGalHalfNinety` and `\PureAGNHalfNinety`.
  Read: `analyses/analysis_0_pure_tracer_H0/results/h0_pure{gal,agn}_targeted_s<seed>.h5` for the 93
  seeds in `h0_pure_tracer_ens93.json:seeds_requested`.
- (b) H0 from all 1000 events of the 9 complete catalogs of Fig. 2. Thin blue: galaxy catalog alone
  (f_AGN = 0 column of the joint grid), one per catalog. Thin green: both catalogs (H0 marginal of the
  joint posterior), one per catalog. Thick curves with shaded 90% interval: combined over the 9
  catalogs, galaxy catalog alone (`\GalOnlyHzeroComb`, input at the 88% level) and both catalogs
  (`\JointHzeroComb`). Axis H0 [56, 84]; no curve loses more than 0.1% of its mass.
  Read: the nine `joint_s*.h5` of Fig. 2.

## Fig. 4 `fig4_host_dependent_h0` (panel c is an addition; switch `FIG4_CONTROL_PANEL` in the script)

- (a) Host-dependent mock (AGN mass peak +5 Msun, AGN spin mean +0.10; one simulated catalog), both
  catalogs. Solid: reference fixed, offsets free (`\HDFixHzero`; grid, H0 range [60, 76]). Dotted:
  eight parameters (`\HDEightHzero`; prior H0 [60, 76]; KDE bandwidth 0.38 km/s/Mpc, 4098 samples).
  Dashed: common population (`\CommonHzero`; grid over H0 [50, 100], input outside its 90% interval,
  at the 99% level). Black dashed: input H0. Axis [60, 78]; mass outside below 3e-4.
  Read: `analyses/analysis_10_mass_spin_marked_multitracer/results/c10_arm_S.h5`, `c10_arm_J.h5`;
  `analyses/analysis_13_joint_shared_widths/results/a13core_e7c3007_rslice_cap20_historical_n200_s1.npz`.
- (b) Same mock, galaxy catalog used for both host classes (no AGN catalog). Dashed: common
  population (`\PinIZeroHzero`); its density is largest at the upper end of the H0 range (76), so the
  quoted interval is bounded by the prior. Solid ("host-dependent"): offsets not free but fixed at
  (Delta mu_G, Delta mu_chi) = (5.0, 0.115) (`\PinIOneHzero`); 0.115 is the fixed-H0 maximum of the same
  data, not the input 0.10. Prior H0 [60, 76]; grey dotted line = its upper end.
  Read: `c10_mech.h5` (groups I0, I1).
- (c) A different mock: every host is a galaxy; 30% of the mergers, chosen independently of the
  host, have the AGN-like population (+5 Msun, +0.10). Galaxy catalog only. Solid ("two
  populations"): both offsets fixed at their input values (`\CtrlModHzero`; stored H0 range [54, 95]).
  Dashed: common population (`\CtrlUnmodHzero`; stored range [58, 100]; density at H0 = 100 is 0.46% of
  the peak, so the interval is bounded by the prior edge). Prior H0 [50, 100]. Axis [55, 100].
  Read: `c10_control.h5` (groups B5M, B5U); mock `data/seed100/events/events_single_b0p30_dmu0p10_dmuG5.h5`.
- Text in the panels: "both catalogs", "galaxy catalog alone", "galaxy hosts only, galaxy catalog".
  Black dashed input lines stop below the text band.

## Fig. 5 `fig5_offsets`

- (a) f_AGN, (b) mass-peak offset Delta mu_G, (c) spin-mean offset Delta mu_chi, host-dependent mock
  (one simulated catalog), both catalogs, H0 free. Solid: reference fixed (`\HDFixfAGN`, `\HDFixDmuG`,
  `\HDFixDmuChi`; 4-D grid; lattice covers f [0.05, 0.55], Delta mu_G [1, 9], Delta mu_chi [0.025, 0.235]).
  Dash-dot: reference free (`\HDFreefAGN`, `\HDFreeDmuG`, `\HDFreeDmuChi`; 6213 samples; priors f [0,1],
  Delta mu_G [-4, 10], Delta mu_chi [-0.05, 0.20]). Dotted: eight parameters (`\HDEightfAGN`, `\HDEightDmuG`,
  `\HDEightDmuChi`; 4098 samples; Delta mu_chi prior [-0.05, 0.30]). The three analyses have different
  Delta mu_chi prior ranges. Black dashed: inputs 0.30, 5 Msun, 0.10. Axes: f [0.05, 0.55],
  Delta mu_G [0, 10], Delta mu_chi [-0.05, 0.30]; mass outside below 2e-9. In (c) the input 0.10 lies
  outside the 90% interval of the dotted curve (99% level) and inside for the other two.
  Read: `c10_arm_J.h5`; `analyses/analysis_11_free_common_population/results/a11_11D.npz`; the A13 file of Fig. 4.
- (d) A different mock: AGN spin mean +0.10, no mass offset; H0 fixed at 67.74. Green: both
  catalogs, f_AGN and Delta mu_chi free ("with the catalogs", `\SpinBothDmuChi`). Blue: the galaxy
  catalog used for both host classes, so positions carry no host-class information ("spins
  alone", `\SpinIntDmuChi`). Prior Delta mu_chi [-0.2, 0.25]; the blue density at 0.25 is 1.1% of
  its peak (the curve ends at the prior edge). Axis [-0.05, 0.30].
  Read: `analyses/analysis_8_spin_marked_fagn_fixed_H0/results/arm_J_joint.h5`, `arm_I_intrinsic.h5`.

## Fig. 6 `fig6_flux_limits` (panels a to d; DESIGN's "(a)" is split into (a) f_AGN and (c) H0)

- One simulated catalog (catalog A of Fig. 2). Both catalogs flux-limited with the same limit.
- (a) f_AGN, (c) H0: median (marker) and 90% equal-tailed interval (bar) against the flux limit, with
  the number densities of the missing hosts at their inputs. Tick labels: limit and completeness
  inside the horizon. "none / complete": complete catalogs, the Fig. 2 grid of catalog A (open
  circle). Flux-limited points: `\FluxfTwentyOne` ... `\FluxfEighteen`, `\FluxHzeroTwentyOne` ...
  `\FluxHzeroEighteen`. The m < 21 point equals the complete point to the shown precision.
- (b) f_AGN, (d) H0: posterior medians against the assumed AGN number density of the missing hosts,
  in units of its input, at 0.5, 0.7, 0.9, 1, 1.1, 1.3, 2 (log axis); one curve per flux limit (purples
  and markers as in a, c; factor 1 is the point of a, c). Shaded band: 90% equal-tailed interval at
  m < 18 (9.5%) only, labelled in the panel. Grey dotted: factor 1. Black dashed: inputs. In (b) the
  four curves overlap; direct labels are in (d). Slopes (text): `\SlopefAGN*`, `\SlopeHzero*`.
- y axes shared per row: f_AGN [0.05, 0.55], H0 [65.5, 72]; no interval or band is cropped.
- Values of the band at factors other than 1 have no macro; the medians at every factor are checked
  against `values.json` (`SlopefAGN<limit>.extra.medians_f/medians_H0`).
- Read: `analyses/analysis_3_incomplete_catalog_H0_fagn/results/joint_m{21,20,19,18}_s100.h5`,
  `analyses/analysis_4_density_anchoring_H0_fagn/results/joint_m{21,20,19,18}_a{05,07,09,11,13,20}_s100.h5`,
  `analyses/analysis_2_complete_catalog_H0_fagn/results/joint_s100.h5`.

## Fig. 7 `fig7_free_densities`

- m < 18 (9.5% completeness), both number densities free with H0 and f_AGN: three simulated
  catalogs A, B, C (A = catalog A of Fig. 2; B, C = catalogs B, C of Fig. 2).
- (a) log10 n_gal, (b) log10 n_AGN posteriors, one curve per catalog, same colour (blue, vermilion);
  the three are not distinguished. Grey dotted: flat prior density (prior [-4, -1] and [-6, -4], so
  1/3 and 1/2 per dex); both axes span the full prior. Black dashed: inputs -3 and -5. KDE bandwidths
  0.08 to 0.10 dex; 11,441 to 11,941 samples per catalog. The n_gal densities at the lower prior edge
  are 13 to 16% of their peaks; the n_AGN densities at -6 are 4 to 7%.
- Lower row: (f_AGN, log10 n_AGN) 90% HPD region, one panel per catalog, identical axes = prior box
  f [0, 1] x [-6, -4]; 40 x 40 histogram smoothed by 1.5 bins (0.0375 in f, 0.075 dex). Black "+":
  input (0.30, -5). The regions reach the f_AGN = 0 edge.
- Macros: `\FreeLognGalEighteen{,CatB,CatC}`, `\FreeLognAGNEighteen{...}`, `\FreefAGNEighteen{...}`,
  `\FreeRhofAGNEighteen{...}`.
- Read: `analyses/analysis_5_free_anchors_H0_fagn/results/campaign_m18_dynesty_s100.h5`,
  `analyses/selection_redo/fu_seed101/results/campaign_m18_dynesty_s101.h5`,
  `analyses/selection_redo/fu_seed102/results/campaign_m18_dynesty_s102.h5` (+ the .json for priors).
- Not drawn: a combined posterior over A, B, C (needs one sampler run on the summed likelihood).

## Fig. 8 `fig8_validation`

- Combined posteriors over simulated catalogs (log-likelihoods summed, parameters shared), shaded
  90% equal-tailed interval, black dashed input.
- (a) H0, 93 catalogs of 1000 galaxy-hosted mergers (`\PureGalHzeroComb`, input at the 12% level).
  (b) H0, the same 93 catalogs with 1000 AGN-hosted mergers (`\PureAGNHzeroComb`; the input 67.74 lies
  just below the interval's lower end 67.741, at the 91% level). (c) H0 and (d) f_AGN, the 9 complete
  catalogs of Fig. 2, both host classes, both catalogs (`\JointHzeroComb`, 89%; `\JointfComb`, 70%).
- Axes: H0 [66.6, 68.9], [67.45, 68.2], [66.6, 70.1]; f_AGN [0.22, 0.36]; mass outside below 1e-4.
- Read: as Figs. 3(a) and 2.
- Not drawn (no results on disk; placeholders.txt): catalogs with no AGN-hosted mergers (`\Null*`),
  flux-limited combined posteriors (`\FluxfCombEighteen` etc.; one catalog exists), free-density
  combined posteriors (`\Free*Comb`), host-dependent combined posteriors (`\HDEight*Comb`,
  `\CommonHzeroComb`; one catalog exists).

## Fig. C1 `figC1_corner`

- Eight-parameter host-dependent model on the host-dependent mock (one simulated catalog), both
  catalogs: H0, f_AGN, mu_G, Delta mu_G, mu_chi, Delta mu_chi, sigma_G, sigma_chi. 4098 equal-weight
  nested-sampling samples. Diagonal: 1-D density with shaded 90% equal-tailed interval (macros
  `\HDEight*`). Off-diagonal: 90% HPD region (36 x 36 histogram, smoothing 1.7 bins). Black dashed:
  inputs (67.74, 0.30, 35, 5, 0, 0.10, 5, 0.10).
- Axes are cropped to the posterior (0.05 to 99.95 percentiles plus 8% padding, always including the
  input), not the prior: H0 [62.5, 70.9] of [60, 76]; f_AGN [0.125, 0.416] of [0, 1]; mu_G [34.2, 37.1] of
  [31, 39]; Delta mu_G [0.34, 7.63] of [-4, 10]; mu_chi [-0.042, 0.022] of [-0.1, 0.1]; Delta mu_chi
  [0.074, 0.235] of [-0.05, 0.30]; sigma_G [3.88, 5.78] of [1, 10]; sigma_chi [0.053, 0.113] of [0.01, 1].
  At most 0.07% of any marginal lies outside its axis.
- Run: darksirens-core e7c3007, selection-variance cap 20.
- Read: `analyses/analysis_13_joint_shared_widths/results/a13core_e7c3007_rslice_cap20_historical_n200_s1.{npz,json}`.
