# Macro requests of the complete draft (`main.tex` of 17 Nov 2026)

This file replaces the request list of the first draft (kept in git history, commit 36d1997; every
macro it asked for is now in `values/results_macros.tex`). It lists what the new `main.tex` uses
beyond the key scheme of `notes/PAPER_SKELETON.md` Sec. 6.

How the paper builds: `main.tex` inputs `values/results_macros.tex` and then `values/fallback_macros.tex`.
Both are written by `scripts/build_values.py` on every run; the fallback file gives a red default of two dots only to
a macro that `main.tex` uses and the chain does not define (none at the last run: every macro requested below exists).

## A. Requested: 22 macros that are not in `values/results_macros.tex`

### A1. Numbers of realizations (7). Print words for values below 13 ("ten", "five", "three").

| macro | what it is | used in |
|---|---|---|
| `\NrealKnown` | realizations combined for population known, complete catalogs (keys `KnownBoth`, `KnownGal`, `KnownNone`) | Sec. IV A, Sec. V, Table III |
| `\NrealFlux` | realizations combined for the flux-limited catalogs with densities known (`Flux...`) | Table III |
| `\NrealFree` | realizations combined with the number densities inferred (`Free...`) | Table III |
| `\NrealSingle` | realizations combined for the single-population analysis of equal-population events (`SingleBoth`, `SingleNone`) | Table III |
| `\NrealHD` | realizations combined for host-dependent events, peak-dominated spectrum. The text uses ONE number for `HDSingle`, `HDHost` and `HDNone`, because the abstract and Sec. IV D compare the two analyses on the same mergers; if the three combinations use different counts, say so and I split the macro | abstract, Sec. IV D, Table III |
| `\NrealObs` | the same for the observed-like spectrum (`ObsSingle`, `ObsHost`, `ObsNone`) | Sec. IV E, Table III |
| `\NrealCtrl` | realizations combined for the sets with no AGN-hosted mergers and with one in ten (`CtrlZero`, `CtrlTenth`) | Sec. V |

### A2. Design values of the simulation (9)

| macro | what it is | used in |
|---|---|---|
| `\ObsPeakFrac` | fraction of mergers in the Gaussian peak of the observed-like spectrum (the value fitted to the third catalog; check it in the source before it turns black) | abstract, Sec. III C, Sec. IV E, Table II |
| `\ObsMuG`, `\ObsSigmaG` | location and width of the Gaussian peak of the observed-like spectrum [Msun] | Table II |
| `\ObsAlpha` | power-law index of the observed-like spectrum, printed without sign (the table writes the minus) | Table II |
| `\ObsMmin`, `\ObsMmax` | mass range of the observed-like spectrum [Msun] | Table II |
| `\MockfAGNTenth` | input f_AGN of the set with one merger in ten hosted by an AGN (0.10; the figure prints "input 0.10") | Fig. 9 caption |
| `\MockNAGNCompleteEighteen` | number of AGN of the m < 18 catalog of realization A below `\MockzCompleteEighteen` (about 4500) | Sec. III D |
| `\MocknAGNPrecEighteen` | precision in percent to which that count gives n_AGN (Poisson, 100/sqrt(N)) | Sec. III D |

### A3. Settings and tests (6)

| macro | what it is | used in |
|---|---|---|
| `\ProdNsamp` | posterior samples per merger in the analyses of the paper (8000 in the approved plan). `\MockNsamp` (2000) is no longer used in the text | Sec. V, Appendix A |
| `\MCNsampLo`, `\MCNsampMid`, `\MCNsampHi` | the three sample counts of Analysis 14 (2000, 8000, 32000), which belong to `\MC...TwoK`, `...EightK`, `...ThirtyTwoK` | Appendix A |
| `\SkyResShiftHzero` | change of the H0 median of `KnownBoth`, realization A, when the sky pixels are half as wide [km/s/Mpc] | Sec. V |
| `\SkyResShiftfAGN` | the same for f_AGN | Sec. V |

## B. Macros of the key scheme whose use needs a word

- `\WidthRatioAGNGal` stands in one sentence beside `\AGNhostedHzeroWid` and `\GalhostedHzeroWid`
  (Sec. IV A). A reader will divide the two, so the ratio should be that of the two printed widths
  (now 6.5 / 1.6 against 4.7).
- `\HDLB`, `\ObsLB` are read as realization A; `\HDHostLBComb`, `\ObsHostLBComb` as the sum over the
  combined realizations. Table III prints the combined values, like Fig. 8.
- Table III prints the `...Comb` values of `KnownNone`, `KnownGal`, `KnownBoth`, `FluxEighteen`,
  `FreeEighteen`, `SingleNone`, `SingleBoth`, `HDNone`, `HDSingle`, `HDHost`, `ObsNone`, `ObsSingle`,
  `ObsHost`. Table IV prints realization A (`Flux<limit>`, `Free<limit>` with `LognAGNUL` and
  `LognAGNPriorUL`).
- Table I prints one prior range per parameter from `\Prior...Lo/Hi`. These are still the ranges of
  the exploratory runs (mass peak [31, 39], offset [-4, 10], spin offset up to 0.30); the production
  ranges of `runs.txt` (mass peak [20, 50], offset [-10, 10]) should replace them in the same
  macros. `\PriorHDHzeroLo/Hi`, `\PriorDmuChiHiSix`, `\PriorSpinDmuChiLo/Hi` are no longer used.
- The correlation of H0 and f_AGN in Sec. IV A is the existing `\JointRhoComb`; an alias
  `\KnownBothCorrHzerofAGN` would fit the scheme.

## C. Existing macros outside the key scheme that the text uses

Analysis 15 (black): `\NormZmax`, `\NormZmaxProd`, `\NormFAGNComplete`, `\NormFAGNEighteenProd`,
`\NormDeltafTwoCompleteProd`, `\NormDeltafTwoCompleteFive`, `\NormDeltafTwoCompleteFivePred`,
`\NormDeltafTwoEighteenProd`, `\NormDeltafTwoEighteenProdPred`, `\NormDeltaHzeroTwoEighteenProd`, and
the exploratory `\NormDeltaHzeroTwoTwentyOneFive`.
Analysis 14 (red until its summary exists): `\MCMixedHzeroSd<N>`, `\MCMixedfSd<N>`,
`\MCMixedLnLVar<N>` for `TwoK`, `EightK`, `ThirtyTwoK`; `\MCMixedfMedTwoK`,
`\MCMixedfMedThirtyTwoK`; `\MCMixedInjHzeroSd`, `\MCMixedInjfSd`; `\MCAGNHzeroSdEightK`,
`\MCAGNLnLVarEightK`.
Others: `\JointRhoComb`, `\PureGalHostsPerVol`, `\PureAGNHostsPerVol`, `\MassScaleShareKnown`,
`\MassScaleShareSingle`, `\CorrHzeroMuGNone`, `\CorrHzeroMuGBoth`, `\HDBiasHzero`, `\ObsBiasHzero`,
the `\Mock...` and `\Prior...` design values.

## D. Numbers typed by hand in the text, on purpose

None is a result or a design value of this work.

- Published values with their citation: the H0 of the fifth catalog with its 90% interval and
  236 sources; 0.21 and 0.41 to 0.61 (95% limits on f_AGN) with the luminosity 10^44.5 erg/s;
  13 of 28 counterparts and H0 = 70.5 +3.4 -2.9 (68%) of Kumar and Torres-Orjuela; 35 Msun,
  5-10% and 11-27% of Bom and Palmese; the density 10^-4.75 Mpc^-3 of the association studies;
  the redshift error 0.01 of Pouw et al.; "their Eq. (16)" of Mastrogiovanni et al.
- The credible level of the paper's convention, 90%, and 68% and 95% where a cited paper uses them.
- "|delta| < 3 degrees" in the caption of Fig. 1 (the label drawn in the figure); zero for the
  offsets held fixed in Table I and for the input of the set without AGN-hosted mergers.
- Words for numbers: "a hundred times" (ratio of the two input number densities), "seven mergers in
  ten", "one in ten", "a factor of two", "half of the detected mergers" (the percentage is
  `\MockEventCompEighteen`).

## E. Consistency pass of 17 Nov 2026 (first production results)

### E1. Requested: 1 macro that is not in `values/results_macros.tex`

| macro | what it is | used in |
|---|---|---|
| `\NrealKnownAGNEdge` | number of the `\NrealKnownAlone` realizations of key `KnownAGN` (equal populations, population known, AGN catalog alone) whose H0 posterior has no interior maximum and rises to the upper prior edge, printed as a word. Now four: realizations 100, 101, 103 and 105 have medians of 99.7 to 99.9 with all of the grid-edge mass at H0 = 100; realization 102 has an interior maximum, 76.8 [75.5, 78.1], which excludes the input. Fallback `\ph{four}` in `values/fallback_macros.tex` | Sec. IV A |

If the text is to say what the fifth realization does, it also needs that realization's median and interval
(a macro such as `\KnownAGNHzeroInterior`); the text does not quote it now.

### E2. Existing macros that the text now uses

- Realization counts split by key: `\NrealKnownAlone` (Table III, Sec. IV A), `\NrealHDSingle` and `\NrealHDNone`
  (abstract, Sec. IV D, Table III, caption of Fig. 6). `\NrealHD` is now the host-dependent analysis alone.
- Half-widths of realization A beside each bias of a combined posterior: `\HDSingleHzeroHalfWid`,
  `\ObsSingleHzeroHalfWid` (abstract, Sec. IV D, Sec. IV E), `\HDNoneHzeroHalfWid`, `\HDSinglefAGNHalfWid` (Sec. IV D).
- `\KnownBothCorrHzerofAGNComb` replaces the exploratory `\JointRhoComb` in Sec. IV A (item B, last point).
- Full widths `\KnownNoneHzeroWid`, `\KnownGalHzeroWid`, `\KnownBothHzeroWid` and the median `\KnownAGNHzeroMed`
  against `\PriorHzeroHi` in Sec. IV A.
- Share of the detected mergers in the peak: `\ObsDetPeakShare`, `\ObsHDDetPeakShare`, `\MockDetPeakShare`,
  `\MockHDDetPeakShare` (abstract, Sec. III C, Sec. IV E, caption of Fig. 7, Sec. VI). The sections on mergers with
  host-dependent populations quote the `HD` values (0.45 against 0.97); Sec. III C gives both.
- `\MCAGNHzeroSdThirtyTwoK`, `\MCAGNLnLVarThirtyTwoK` and `\MCNsampHi` in Sec. V and Appendix A: the sets with every merger hosted by an AGN
  are analyzed with 32000 posterior samples per merger (run names `_s32000_inj6`).

## F. The letter's experiment taken into the long paper (17 Nov 2026, evening)

### F1. Requested: 11 macros that are not in `values/results_macros.tex`

All eleven now exist in `values/results_macros.tex` (written by `scripts/build_values.py`; the ten limits in h^3 Mpc^-3 are
red placeholders until the free-density runs finish, `\ShareSumThirty` is black). `main.tex` has no `\providecommand`
block of its own: after `values/results_macros.tex` it inputs `values/fallback_macros.tex`, which `scripts/build_values.py`
writes on every run from `main.tex` and which gives a red default only to a macro the text uses and the chain does not
define (none at the last run).

| macro | what it is | used in |
|---|---|---|
| `\Free<limit>LognAGNULh`, limit = `TwentyOne`, `Twenty`, `Nineteen`, `Eighteen` (4) | 90% upper limit on log10 of the AGN number density in h^3 Mpc^-3, the unit the analysis samples, number densities inferred, realization A. The existing `\Free<limit>LognAGNUL` are in Mpc^-3 (converted per sample with its own H0) and are no longer used in the text | Table IV, Sec. IV C |
| `\Free<limit>LognAGNPriorULh`, the same four limits (4) | the limit the priors allow once the rate per host is measured, in h^3 Mpc^-3, beside the macro above. The existing `...PriorUL` are in Mpc^-3 and are no longer used | Table IV, Sec. IV C |
| `\FreeEighteenLognGalULh`, `\FreeEighteenLognGalPriorULh` (2) | the same two quantities for the galaxy number density at m < 18, in h^3 Mpc^-3 | Sec. IV C |
| `\ShareSumThirty` | sum, in percent, of the shares of the information on H0 of the two host classes with the host class known at input f_AGN = 0.3: `\ShareAGNThirty` plus the galaxy share computed the same way (`values.json`, `ShareAGNThirty.extra.galaxy_share_percent`); 107 today. It is the size of the caveat on `\ShareAGNThirty` | Sec. IV B |

The text now quotes the inputs of the number densities in the same unit, `\MockLognGalh` and `\MockLognAGNh`
(Table I, Sec. III A, Sec. IV C, caption of Table IV). `\MockLognGal` and `\MockLognAGN` are no longer used in the
text; the design densities `\MocknGal` and `\MocknAGN` stay in Mpc^-3 (Sec. III A, Table II).
Panel (a) of `fig4_densities` has its axis, its prior box (Table I) and its input marker in h^3 Mpc^-3, like the text.

### F2. Superseded

- `\NrealKnownAGNEdge` (Sec. E1) is replaced in the text by the generated `\KnownAGNNoPeak`; the text also quotes the
  realization with an interior maximum, `\KnownAGNPeakOneHzero` and `\KnownAGNPeakOneHzeroCL`. The sentence reads
  "as in four of the five realizations. In the fifth ...": it is worded for exactly one realization with an interior
  maximum and must be re-read if `\KnownAGNNoPeak` changes.
- `\WidthRatioAGNGal`, `\GalhostedHzeroWid` (Sec. B) are no longer used: Sec. IV B quotes the means over five
  realizations of the half-widths (`\HWKnownHundred`, `\HWKnownZero`, `\HWNone`) and no ratio. `\AGNhostedHzeroWid`
  stays in Sec. V.

### F3. Existing macros that the text now uses

- The experiment (claim `classes`): `\HWUnknown<F>` and `\HWKnown<F>` for the five fractions, with `Min` and `Max` at
  `Zero` and `Ten` (unknown) and at `Ten` (known); `\HWNone`; `\HWKnownAGNTen`, `\HWKnownAGNThirty`; `\ShareAGNTen`,
  `\ShareAGNThirty`; `\CostUnknownTen`, `Thirty`, `Fifty`; `\HWPopsThirty`, `\HWKnownPopsThirty` (each with `Min`,
  `Max`), `\CostPopsThirty`, `\CostPopsTwiceThirty`; `\HWHost`; `\HWObsNone`, `\HWObsUnknown<F>` and `\HWObsKnownThirty`,
  `\HWObsPopsThirty`, `\HWObsKnownPopsThirty` (each with `Min`, `Max`; `\HWObsPopsThirty` turned real during this pass and
  moved from an expected sentence to the opening of Sec. IV F), `\HWObsHost`; `\ProbAGNEqual`, `\ProbGalEqual`, `\ProbAGNPos`,
  `\ProbGalPos`, `\ProbAGNPops`, `\ProbGalPops`, `\ProbAGNPopsTwice`, `\ProbGalPopsTwice`; `\NrealClasses`.
- Others: `\PureGalHostsPerVolAGNEvents`, `\PureAGNHostsPerVolGalEvents` (the crossed counts of candidate hosts,
  Sec. IV B); `\MockfAGNRealised` beside `\MockHDfDet` (Sec. IV E); `\ObsDetzMedian` beside `\MockzMedian` (Sec. IV F).
- Fractions of mergers hosted by AGN other than `\MockfAGN` and `\MockfAGNTenth` are written in words (none, one in
  ten, three in ten, half, all), like "seven mergers in ten" in Sec. D; `\fagn=0` and `\fagn=1` appear as the limits of
  the likelihood.

## G. Sec. IV E on three steps (23 Nov 2026)

Why. On the mergers with host-dependent populations the single-population analysis with its mass and spin
parameters inferred is NOT biased in H0 (five realizations: 67.6, 66.7, 68.9, 70.5, 67.6; host-dependent on the same
mergers 67.5, 66.8, 68.5, 70.6). The bias the section was written for belongs to the analysis that takes the
population as known and holds the galaxy class's mass and spin distribution for both host classes (the analysis of
Sec. IV A applied to these mergers): stored grids give H0 70.9, 86.9, 71.1, 73.2, 69.9, and the upper prior edge in
all five realizations at twice the offsets. Secs. IV E, IV F, IV G, the abstract and the Conclusions are rewritten on
three steps: (1) population assumed, (2) single population inferred, (3) host-dependent populations inferred.
Reader-facing name of step 1: "population assumed" (legend, table row); never "fixed", never "held" in a legend.

### G1. New keys (same conventions as the existing keys: `<Key><Quantity>`, `Comb`, `CL`, `HalfWid`, `Med/Lo/Hi`)

| Key | Runs (pattern, `{R}` = realization) | Realizations | State |
|---|---|---|---|
| `HDAssumed` | `two_tracers_one_population/both_catalogs/complete/peak_f0p30_s1/realization_{R}` (grid) | 100, 101, 102, 103, 105 | exist |
| `HDAssumedTwice` | `.../two_tracers_one_population/both_catalogs/complete/peak_f0p30_s2/realization_{R}` (grid) | same five | exist |
| `HDPops` | `two_tracers_two_populations/both_catalogs/complete/peak_f0p30_s1/realization_{R}` (grid; the two populations at their inputs) | same five | exist |
| `HDSingleTwice` | `single_population/both_catalogs/complete/peak_f0p30_s2/realization_{R}` (nested, six parameters; stage E3) | same five | running (first on 23 Nov) |
| `ObsAssumed` | `two_tracers_one_population/both_catalogs/complete/obslike_f0p30_s1/realization_{R}` (grid) | same five | exist |
| `ObsPops` | `two_tracers_two_populations/both_catalogs/complete/obslike_f0p30_s1/realization_{R}` (grid) | same five | exist |

These are experiment runs (stages E1 to E3), the same runs that `classes.py` already reads for `Pos`, `Pops`,
`PosTwice`; the keys above add the posteriors of H0 and f_AGN of those runs.

Macros wanted:

- `HDAssumed`: `\HDAssumedHzero` (realization A, median and 90% interval), `\HDAssumedHzeroHalfWid`,
  `\HDAssumedHzeroComb`, `\HDAssumedHzeroCL`, `\HDAssumedBiasHzero` (combined median minus the input, as
  `\HDBiasHzero` is defined now), `\HDAssumedfAGNComb`, `\HDAssumedfAGNCL`, `\NrealHDAssumed`,
  `\HDAssumedNoPeak` (number of realizations, in words, whose posterior of H0 has no interior maximum by the rule
  already used for `\KnownAGNNoPeak`; say in the report which realizations and what realization 101, 86.9
  [71.5, 97.4] with f_AGN 0.07, is by that rule). If the level of the input is 100% to the printed precision, say
  what the macro prints.
- `HDAssumedTwice`: `\HDAssumedTwiceNoPeak` (count in words, as above; expected five), `\HDAssumedTwiceHzeroMed`
  (realization A's median, to be quoted against `\PriorHzeroHi`), `\HDAssumedTwicefAGNUL` (realization A, 90% upper
  limit). No interval and no combined posterior for this key (posteriors at the prior edge).
- `HDPops`: `\HDPopsHzeroComb`, `\HDPopsHzeroCL`, `\HDPopsfAGNComb`, `\HDPopsfAGNCL`.
- `HDSingleTwice`: `\HDSingleTwiceHzero` (realization A), `\HDSingleTwiceHzeroComb`, `\HDSingleTwiceHzeroCL`,
  `\HDSingleTwiceBiasHzero`, `\HDSingleTwiceMuG`, `\HDSingleTwiceMuChi` (realization A), `\HDSingleTwicefAGNComb`,
  `\NrealHDSingleTwice`. Red until the runs exist; placeholder values: H0 combined 68.6 (+0.9, -0.9), level 80%,
  bias +0.9, realization A 68.4 (+1.9, -1.9), mu_G 38.3 (+0.6, -0.6), mu_chi 0.08 (+0.01, -0.01), f_AGN 0.41
  (+0.04, -0.04).
- `ObsAssumed`: `\ObsAssumedHzero`, `\ObsAssumedHzeroComb`, `\ObsAssumedHzeroCL`, `\ObsAssumedBiasHzero`,
  `\ObsAssumedfAGNComb`, `\ObsAssumedNoPeak`, `\NrealObsAssumed`. `ObsPops`: `\ObsPopsHzeroComb`, `\ObsPopsHzeroCL`.
- Shift of the assumed-population analysis against the analysis of the SAME mergers with the two populations at
  their inputs (posterior median of H0, `HDAssumed` minus `HDPops`, per realization): `\HDAssumedShiftHzero` (median
  over the realizations, signed), `\HDAssumedShiftHzeroMin`, `\HDAssumedShiftHzeroMax`; the same for the
  observed-like family, `\ObsAssumedShiftHzero`, `Min`, `Max`. By hand: 3.72, 20.2, 2.65, 3.20, 2.19 (median 3.2)
  and 1.31, 0.67, 1.07, 1.42, 1.11 (median 1.1).

### G2. Additions to existing keys

- Bayes factor of host-dependent against single population per realization: `\HDLBMin`, `\HDLBMax` (log10, over
  the realizations common to `HDHost` and `HDSingle`; by hand 3.9 and 8.4 on four), and the same for the
  observed-like family, `\ObsLBMin`, `\ObsLBMax` (red).
- Half-width of H0 of the single-population analysis on the realizations of `\HWHost`: `\HWHDSingle`, `Min`, `Max`
  (by hand 1.93, 1.95, 1.88, 1.98, 1.75), and the factor in the number of mergers between the two,
  `\CostSingleHost`, `Min`, `Max`, defined as `\CostPopsThirty` is (per realization, then the summary used there).
  Same for the observed-like family: `\HWObsSingle`, `\CostObsSingleHost` (red).
- `\MockHDfDetMean`: mean over the realizations of `HDSingle` of the AGN share of the detected mergers (`planted.
  truth_f_agn` of the summaries; `\MockHDfDet` is realization A's), and `\ObsHDfDetMean`.
- `\HDHostMaxAbsCorrHzero`: realization A, the largest absolute correlation coefficient of H0 with the other seven
  parameters of the host-dependent analysis (by hand 0.10).
- `\HDSinglefAGNCL` exists: check that it is the level of the INPUT fraction `\MockfAGN` in the combined posterior.

### G3. Figures

- `fig6_host_dependent` and `fig7_observed_like` (one function, `fig_host`): panels (a) and (c) gain a third curve,
  the assumed-population analysis combined over its realizations (`HDAssumed`, `ObsAssumed`), dotted; legend entries
  exactly "population assumed", "single population", "host-dependent populations". Posterior densities stay
  normalized with units on the y axis; if the dotted curve of panel (a) falls outside the present x range, widen the
  range, do not crop the curve.
- `fig8_summary`: one new row "Population assumed, both catalogs" at the top of the block "Populations that depend
  on the host" (`HDAssumed`) and of the block "A mass spectrum like the observed one" (`ObsAssumed`), H0 and f_AGN.
- `figures/CAPTION_FACTS.md`: the facts of the new curves and rows (line styles, counts of realizations).

### G4. Not wanted any more (leave defined; the text stops using them)

`\HDBiasHzero`, `\ObsBiasHzero` as "the bias of the single population" (the text will quote `\HDSingleHzeroComb`
with its level).

## H. The observed-like family on the mixture redraw (23 Nov 2026, 23:30 UTC)

Replaces the request of the same afternoon (realization 106 at 32,000 samples per merger; its nine macros
`\ObsAssumedMCHzero<Comb|CL>`, `\ObsPopsMCHzero<Comb|CL>`, `\ObsAssumedMCShiftHzero[Min|Max]`, `\ObsMCOneHzero[Prod]`, the map
block `obslike_mc` and audit J7 are removed; `main.tex` used none). Every analysis of the observed-like spectrum is recomputed with the
defensive mixture redraw of the posterior samples (analysis_16 `--redraw-mixture support`, stage E2M: E2's grids again, run names
`<E2 name>_mix`; stage 5 under its own names). Keys `ObsAssumed`, `ObsPops` (with their subsets, shifts, bias and `NoPeak`) and the
observed-like `\HWObs...` macros read the E2M runs; `ObsSingle`, `ObsHost`, `ObsNone` keep their names. Until the E2M runs exist each is
red and prints the value it had from the E2 grids (`values/obslike_e2_frozen.json`); audit J8 checks the option in every run they read.
No new macro.
