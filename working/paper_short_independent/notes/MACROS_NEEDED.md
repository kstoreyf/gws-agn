# Macros the short paper needs (17 Nov 2026)

`main.tex` inputs `values/results_macros.tex` (shared; it defines `\ph`) and then
`values_independent.tex` (this paper). Every macro of `values_independent.tex` is a
`\providecommand`: a value file that defines the same name earlier wins, so the numbers can be
filled either by editing `values_independent.tex` or by a generated file input before it.

- Red (`\ph{...}`): a result of the experiment in `EXPERIMENT.md` or of the production list. The
  value printed now is the expected one.
- "Source of the expected value": **brief** = given in `BRIEF.md` Sec. 6; **arith** = arithmetic
  on brief values, recompute from the final numbers; **estimate** = my estimate, not in the
  brief, with its reasoning. Treat every estimate as unknown until the result exists.
- Half-width = half the width of the 90% credible interval of H0, in km/s/Mpc, from 1000
  mergers, mean over the realizations of the experiment. (f, s) = (input fraction of mergers in
  AGN, size of the population difference in units of 5 Msun and 0.10), as in `EXPERIMENT.md`.
- A sentence of `main.tex` that ends with `% EXPECTED` must be reread when its number arrives
  (32 such marks, captions and abstract included).

## 1. Results (red), 37 macros

Figure numbers are those of the paper; the file names keep the brief's numbering (Fig. 2 is
`fig1_budget`, Fig. 3 is `fig2_labels`, Fig. 4 is `fig3_bias`; Fig. 1 is `fig_concept`).

### Fig. 2(a): sharp mass scale (0.90 of the mergers in the peak), populations at their inputs

| macro | meaning | expected | source | comes from |
|---|---|---|---|---|
| `\IndHWSpec` | half-width, mass spectrum alone, no catalog (f = 0 set) | 3.0 | brief | `mass_spectrum_alone` |
| `\IndHWGal` | half-width, all mergers in galaxies, galaxy catalog | 3.5 | brief | `galaxy_catalog_alone`, f = 0 |
| `\IndHWAGN` | half-width, all mergers in AGN, AGN catalog | 0.8 | brief | f = 1 set |
| `\IndHWMix` | half-width, f = 0.30, labels unknown (both catalogs, f_AGN free, one population) | 1.7 | brief | `two_tracers_one_population`, (0.3, 0) |
| `\IndHWMark` | half-width, f = 0.30, a population for each tracer, differences (5 Msun, 0.10) | 1.5 | brief | `two_tracers_two_populations`, (0.3, 1) |
| `\IndHWMarkTwice` | the same for (10 Msun, 0.20) | 1.4 | brief | `two_tracers_two_populations`, (0.3, 2) |
| `\IndHWKnown` | half-width, f = 0.30, hosts known | 1.3 | brief | `hosts_known`, (0.3, 0) |
| `\IndHWKnownAGNPart` | half-width from the mergers in AGN alone (AGN catalog), hosts known, f = 0.30 | 1.5 | arith: `\IndHWAGN`/sqrt(0.30) = 1.46 | the AGN factor of `hosts_known`, (0.3, 0) |
| `\IndHWMixTenth` | half-width, f = 0.10, labels unknown | 2.4 | brief | `two_tracers_one_population`, (0.1, 0) |
| `\IndHWKnownTenth` | half-width, f = 0.10, hosts known | 2.0 | brief | `hosts_known`, (0.1, 0) |
| `\IndCostMix` | (`\IndHWMix`/`\IndHWKnown`)^2: how many times more mergers the analysis with labels unknown needs | 1.7 | arith (1.71) | derived |
| `\IndCostMark` | (`\IndHWMark`/`\IndHWKnown`)^2 | 1.3 | arith (1.33) | derived |
| `\IndCostMarkTwice` | (`\IndHWMarkTwice`/`\IndHWKnown`)^2 | 1.2 | arith (1.16) | derived |
| `\IndfAGNMix` | f_AGN (median, 90% interval), f = 0.30, labels unknown, one realization | 0.28 +0.08 -0.08 | estimate: Analysis 15 gives 0.275 [0.196, 0.357] at 2000 samples; Analysis 14 gives a median of 0.275 at 8000 | `two_tracers_one_population`, (0.3, 0), realization A (shared key `KnownBoth`) |

### Fig. 2(b): mass spectrum like the observed one (0.092 of the mergers in the peak)

| macro | meaning | expected | source | comes from |
|---|---|---|---|---|
| `\IndObsHWSpec` | half-width, mass spectrum alone | 10 | brief | observed-like, `mass_spectrum_alone` |
| `\IndObsHWMix` | half-width, f = 0.30, labels unknown | 1.9 | estimate: the catalog part of `\IndHWMix` with the mass-scale part replaced by 10 | observed-like, `two_tracers_one_population`, (0.3, 0) |
| `\IndObsHWAGN` | half-width, all mergers in AGN, AGN catalog | 0.8 | estimate: `\IndHWAGN` hardly uses the mass scale (0.8 and 3.0 add in inverse squares to 0.83 without it) | observed-like, f = 1 |

### Fig. 3: probability that a merger is hosted by an AGN (one realization, f = 0.30, (5 Msun, 0.10))

The probability of merger i is r_i = f L_AGN,i / [f L_AGN,i + (1 - f) L_gal,i] at the posterior
median. The four numbers are means of r_i over the mergers of one true class.

| macro | meaning | expected | source | comes from |
|---|---|---|---|---|
| `\IndProbAGNPos` | mean r_i, mergers truly in AGN, positions alone (one population) | 0.36 | estimate (below) | membership output, `two_tracers_one_population` |
| `\IndProbGalPos` | mean r_i, mergers truly in galaxies, positions alone | 0.27 | estimate | the same |
| `\IndProbAGNMark` | mean r_i, mergers truly in AGN, positions with masses and spins (a population for each tracer) | 0.41 | estimate | membership output, `two_tracers_two_populations` |
| `\IndProbGalMark` | mean r_i, mergers truly in galaxies, positions with masses and spins | 0.25 | estimate | the same |

How the estimates were made: for calibrated probabilities E[r | AGN] = f + Var(r)/f and
E[r | gal] = f - Var(r)/(1 - f), and Var(r)/[f(1 - f)] is the share of the label information
that the data hold, (sigma_f with labels known / sigma_f measured)^2. With labels known
sigma_f = sqrt(0.3 x 0.7/1000) = 0.0145; a 90% half-width of 0.08 (Analysis 15) gives a share of
0.09, and 0.06 (brief) gives 0.16. The brief gives no expectation for this figure. Section II quotes
the two "positions alone" numbers in its paragraph on equal populations; if the equal-population
catalog gives visibly different means, that paragraph needs its own two macros.

### Fig. 4: populations inferred, f = 0.30

| macro | meaning | expected | source | comes from (shared key, where one exists) |
|---|---|---|---|---|
| `\IndBias` | one population fitted to mergers that differ by (5 Msun, 0.10): median of the combined H0 posterior minus the input | 3 | brief | production stage 3 (`\HDBiasHzero`) |
| `\IndBiasTwice` | the same for (10 Msun, 0.20) | 6 | brief | the six-parameter runs at s = 2 of the experiment |
| `\IndHWOnePop` | half-width of that one-population analysis, one catalog | 2.6 | brief | stage 3, realization A (`\HDSingleHzeroHalfWid`) |
| `\IndOnePopCL` | credible level (percent) beyond which the input H0 lies in the combined one-population posterior | 99.9 | estimate: a shift of 3 against a combined half-width near 0.8 | stage 3 (`\HDSingleHzeroCL`) |
| `\IndTwoPopHzero` | H0, a population for each tracer, one catalog | 67.4 +2.1 -2.1 | brief | stage 4 (`\HDHostHzero`) |
| `\IndTwoPopHW` | its half-width (open symbol of Fig. 2a) | 2.1 | brief | stage 4 (`\HDHostHzeroHalfWid`) |
| `\IndTwoPopfAGN` | f_AGN of the same analysis | 0.30 +0.06 -0.06 | brief | stage 4 (`\HDHostfAGN`) |
| `\IndTwoPopDm` | mass-peak difference [Msun] | 5.1 +1.0 -1.0 | brief | stage 4 (`\HDHostDmuG`) |
| `\IndTwoPopDchi` | mean effective spin difference | 0.10 +0.03 -0.03 | brief | stage 4 (`\HDHostDmuChi`) |
| `\IndTwoPopCL` | credible level (percent) of the input H0 in the combined posterior of that analysis | 30 | estimate: no expectation beyond "inside the 90% interval" | stage 4 (`\HDHostHzeroCL`) |
| `\IndLB` | log10 Bayes factor, a population for each tracer over one population, one catalog | 9 | brief | stages 3 and 4 (`\HDLB`) |

### Appendix C

| macro | meaning | expected | source | comes from |
|---|---|---|---|---|
| `\IndRecHzeroComb` | H0, equal populations, f = 0.30, labels unknown, combined over `\NrealKnown` realizations | 67.9 +0.6 -0.6 | estimate: `\IndHWMix`/sqrt(10) | stage 1 (`\KnownBothHzeroComb`) |
| `\IndRecHzeroCL` | credible level (percent) of the input H0 in it | 40 | estimate: no expectation | stage 1 (`\KnownBothHzeroCL`) |
| `\IndRecfAGNComb` | f_AGN of the same combination | 0.30 +0.03 -0.03 | estimate: 0.08/sqrt(10) | stage 1 (`\KnownBothfAGNComb`) |
| `\IndRecfAGNCL` | credible level (percent) of the input f_AGN in it | 35 | estimate: no expectation | stage 1 (`\KnownBothfAGNCL`) |
| `\IndZerofAGNUL` | 90% upper limit on f_AGN when no merger is in an AGN, combined over `\NrealCtrl` realizations | 0.05 | estimate: no expectation | f = 0 set, `two_tracers_one_population` (`\CtrlZerofAGNUL`) |

## 2. Design values defined here (black), 14 macros

| macro | value | meaning |
|---|---|---|
| `\IndNreal` | five | realizations of the experiment (100, 101, 102, 103, 105); a word, used in text mode; the caption of Fig. 2 says the points are means over them |
| `\IndfList` | 0, 0.1, 0.3, 0.5 and 1 | the input fractions of Fig. 2 (math mode) |
| `\IndDmTwice` | 10 | mass-peak difference of the stronger case [Msun] |
| `\IndDchiTwice` | 0.20 | mean effective spin difference of the stronger case |
| `\IndConceptDec` | 3 | Fig. 1(a): half-thickness of the slice in declination [degrees] |
| `\IndConceptzMax` | 0.20 | Fig. 1(a): largest redshift of the slice |
| `\IndConceptScale` | 200 | Fig. 1(a): scale bar [comoving Mpc] |
| `\IndConceptNGal` | 13,398 | Fig. 1(a): objects of the galaxy catalog in the slice |
| `\IndConceptNAGN` | 126 | Fig. 1(a): objects of the AGN catalog in the slice |
| `\IndConceptzTrue` | 0.098 | Fig. 1(a): true redshift of the AGN-hosted merger whose 90% region is outlined |
| `\IndConceptVolNGal` | 225 | Fig. 1(a): galaxies inside that region (printed in the figure) |
| `\IndConceptVolNAGN` | 6 | Fig. 1(a): AGN inside that region (printed in the figure) |
| `\IndConceptNgalHosted` | 640 | Fig. 1(b, c): galaxy-hosted among the 1000 detected mergers, populations differ |
| `\IndConceptNAGNHosted` | 360 | Fig. 1(b, c): AGN-hosted among them; also quoted in Sec. IV as the detected share |

The ten `\IndConcept...` values are those given with the figure (realization A); they are
final and belong in the generated macro file once the plotting script exports them.

## 3. Final macros of `values/results_macros.tex` that the paper uses

Simulation: `\MockNev`, `\MockHzero`, `\MockOmegam`, `\MockfAGN`, `\MockfAGNTenth`, `\MocknGal`,
`\MocknAGN`, `\MockbGal`, `\MockbAGN`, `\MockLcut`, `\MockLXcut`, `\MockNGal`, `\MockNAGN`,
`\MockzCat`, `\MockDz`, `\MockNside`, `\MockShellWidth`, `\MockAlpha`, `\MockMmin`, `\MockMmax`,
`\MockPeakFrac`, `\MockDetPeakShare`, `\MockMuG`, `\MockSigmaG`, `\MockBeta`, `\MockMuChi`,
`\MockSigmaChi`, `\MockDmuG`, `\MockMuGAGN`, `\MockDmuChi`, `\MockSNRthr`,
`\MockRhoRef`, `\MockMcRef`, `\MockSigLnMc`, `\MockSigLnq`, `\MockSigChi`, `\MockSkyMin`,
`\MockSkyMax`, `\MockSigLnDL`, `\MockzHorizon`, `\MockzMedian`, `\MockNinjFound`,
`\MockNinjDrawn`, `\ObsPeakFrac`, `\ObsDetPeakShare`, `\ObsAlpha`, `\ObsMmax`, `\ObsMuG`,
`\ObsSigmaG`, `\ProdNsamp`, `\NormZmaxProd`, `\NeffFactor`.
Hosts per localization volume: `\PureGalHostsPerVol`, `\PureAGNHostsPerVol`,
`\PureAGNHostsPerVolGalEvents`.
Priors: `\PriorHzeroLo/Hi`, `\PriorfLo/Hi`, `\PriorMuGLo/Hi`, `\PriorSigmaGLo/Hi`,
`\PriorMuChiLo/Hi`, `\PriorSigmaChiLo/Hi`, `\PriorDmuGLo/Hi`, `\PriorDmuChiLo/Hi`.
Realizations: `\NrealKnown`, `\NrealCtrl`, `\NrealHD`, `\NrealHDSingle`.
Host normalization (Analysis 15): `\MockEventCompEighteen`, `\NormfEighteenProd`,
`\NormfEighteenProdTwo`, `\NormHzeroEighteenProd`, `\NormHzeroEighteenProdTwo`,
`\NormRateRatioCompleteProd`, `\MockRateRatio`, `\MocknAGNScatterEighteen`,
`\NormDeltafCountsScatterEighteen`.
Monte Carlo accuracy (Analysis 14): `\MCAGNHzeroSdEightK`, `\MCMixedHzeroSdEightK`,
`\MCMixedNEightK`, `\MCMixedInjHzeroSd`, `\MCNsampLo`, `\MCWidthChange`.

To check when the experiment's event sets are final:
- `\MockzHorizon`, `\MockzMedian`, `\PureGalHostsPerVol`,
  `\PureAGNHostsPerVol`, `\PureAGNHostsPerVolGalEvents`, `\MockRateRatio` are numbers of the
  reference catalog. The paper quotes them for the event sets of the experiment.
- `\NormfEighteenProd` and its companions come from 2000 posterior samples per merger; the
  paper uses them only as differences on the same samples.
- Typed by hand, from `PRIOR_WORK_VERIFIED.md`: 71.7 +17.7 -11.4, 236 sources and 3.5% (fifth
  catalog, version 2), 0.21 at 95% and 10^44.5 erg/s (Veronesi et al.), 0.39 +0.41 -0.32 at 90%
  (Zhu and Chen), 35 Msun (Bom and Palmese).

## 4. What the figures must show for the captions to hold

- `figures/fig_concept` (Fig. 1, exists, three panels, full width): the caption uses the legend's
  words ("galaxy catalog", "AGN catalog", "true host of the merger", "galaxy-hosted",
  "AGN-hosted") and the numbers printed in it (225 galaxies, 6 AGN, 200 Mpc, dashed lines at
  35 and 40 Msun and at 0 and 0.10). The caption sets the 225 galaxies and 6 AGN of this one
  region beside the medians over all mergers (1500 and 16).

The captions of the other three were written before the figures. Legends should use the
caption's words.

- `figures/fig1_budget` (two panels, full width): legend entries "Labels unknown", "A population
  for each tracer" (two curves, labeled by (5 Msun, 0.10) and (10 Msun, 0.20)), "Hosts known";
  horizontal lines "Mass spectrum alone" and "Galaxy catalog alone"; one open symbol at
  f_AGN = 0.30 in panel (a), a population for each tracer inferred (`\IndTwoPopHW`). Points are
  means over the realizations; if the scatter over realizations is drawn, the caption needs one
  clause for it. Panel (a) sharp mass scale, panel (b) observed-like spectrum.
- `figures/fig2_labels` (one column): r_i for the mergers truly in galaxies and truly in AGN,
  from positions alone and from positions with masses and spins; a dashed line at
  f_AGN = 0.30. If it is drawn as two panels, change the environment to `figure*`.
- `figures/fig3_bias` (two panels, full width): (a) normalized posterior density of H0, in
  (km/s/Mpc)^-1, combined over realizations: one population fitted (`\NrealHDSingle`) and a
  population for each tracer (`\NrealHD`), vertical line at the input; (b) joint posterior of
  (Delta m, Delta chi) with its 90% contour, a cross at the input and a circle at zero. The
  caption says "in the second analysis" for panel (b) and does not say whether it is one
  realization or the combination; state it once the panel exists. The axis symbols of the text
  are Delta m and Delta chi.
- The reference for HEALPix is not in `refs.bib`; Appendix A names it without a citation.
