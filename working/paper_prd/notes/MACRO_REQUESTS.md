# Macro requests of the complete draft (`main.tex` of 17 Nov 2026)

This file replaces the request list of the first draft (kept in git history, commit 36d1997; every
macro it asked for is now in `values/results_macros.tex`). It lists what the new `main.tex` uses
beyond the key scheme of `notes/PAPER_SKELETON.md` Sec. 6.

How the paper builds meanwhile: `main.tex` inputs `values/results_macros.tex` and then
`values/fallback_macros.tex`, which holds `\providecommand{\Name}{\ph{..}}` for every macro the text
uses (247 names). A macro defined in `values/results_macros.tex` wins; the 22 below print two red
dots until they exist. Nothing in `values/results_macros.tex`, `scripts/` or `figures/` was touched.

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
