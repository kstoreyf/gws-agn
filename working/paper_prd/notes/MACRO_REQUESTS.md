# Macro requests from the first draft of `main.tex` (17 Nov 2026)

Numbers the text needs and `values/results_macros.tex` does not define. Each is declared in the
preamble of `main.tex` with `\providecommand`, so a macro of the same name added to
`values/results_macros.tex` (which is `\input` before the preamble declarations) takes over with no
edit to the paper. Nothing in `values/`, `scripts/` or `figures/` was changed.

Two classes:

- **Red** (20): results or realised numbers. They print as red `\ph{...}` placeholders.
- **Plain** (37): design choices that the notes state but the macro file does not carry (prior
  ranges, the normalisation redshift, constants of the measurement model). They are typed once in
  the preamble through `\req{...}`, from the source named below, and print in the plain class.
  Redefining `\newcommand{\req}[1]{\ph{XX}}` turns all of them red at once.

The 23 red macros that already exist in the macro file are all used in the text and are not
repeated here (`placeholders.txt` lists them).

## A. Red: results and realised numbers

| proposed macro | meaning | where it would come from | used in |
|---|---|---|---|
| `\PureGalHostsPerVol` | mean number of cataloged galaxies inside the 90% localization volume of a detected merger (sky and redshift, at the input H0), galaxy-hosted sets | events and complete galaxy catalog of `data/seed*/`; count catalog rows inside each event's 90% posterior volume | Sec. IV A |
| `\PureAGNHostsPerVol` | the same count for the AGN catalog, AGN-hosted sets | as above with the AGN catalog | Sec. IV A |
| `\ShiftGalOnlyMinusJointComb` | median of `\GalOnlyHzeroComb` minus median of `\JointHzeroComb` (DESIGN F3 quotes 0.5) | the two combined posteriors already built in `build_values.py` | Sec. IV A |
| `\MockSigLnDL` | median fractional uncertainty of the luminosity distance of the detected events, reference catalog, in percent (README: sigma_ln dL median 0.1137) | `data/seed100/META.json`, realised numbers of the events stage | Sec. III B |
| `\MockzCompleteEighteen` | redshift below which the m < 18 catalog is complete (LIKELIHOOD_MATH Sec. 4 table: 0.11) | selection fits of the m < 18 catalogs, or the first bin of `catalog_gal_m18.h5` with completeness below one | Sec. III D |
| `\MockEventFracCompleteEighteen` | percentage of the 1000 detected mergers of catalog A with true redshift below `\MockzCompleteEighteen` (DESIGN L1: about half) | `data/seed100/events/events.h5` truth redshifts | Sec. III D |
| `\MockEventCompEighteen` | completeness of the m < 18 catalog averaged over the true redshifts of the detected mergers of catalog A, in percent | C(z) of Fig. 1(b) evaluated at the event redshifts | Sec. III D |
| `\NormFAGNMax` | largest cataloged share F_AGN = N_AGN / Z_AGN over the four flux limits at the input densities, or a round upper bound (LIKELIHOOD_MATH Sec. 8.2: 0.0115 at m < 21; DESIGN L2 quotes "< 0.02") | closed form of Z_k in LIKELIHOOD_MATH Sec. 4 | Sec. IV C |
| `\SlopefAGNIdentityTwentyOne` | d f_AGN / d log10 n_AGN at m < 21 obtained by applying the identity (Eq. 6 of the paper) to the posterior at the input density, same factors as `\SlopefAGNTwentyOne` (LIKELIHOOD_MATH Sec. 8.3, check 4: 0.434) | the `check_reparam` computation of LIKELIHOOD_MATH Sec. 8.3 moved into `build_values.py` | Sec. IV C |
| `\SlopefAGNIdentityTwenty` | the same at m < 20 (0.438) | as above | Sec. IV C |
| `\SlopefAGNDepthNorm` | d f_AGN / d log10 n_AGN when the host populations are normalized over the catalog depth | run R5b | Sec. IV C |
| `\CommonFreeHzero` | H0 of a common population with free peak location, peak width, spin mean and spin width, no offsets, host-dependent mock, both catalogs | run R8e | Sec. IV B, Table III |
| `\CommonFreefAGN` | f_AGN of the same model | run R8e | Table III |
| `\LBHDVsCommonFree` | log10 Bayes factor of the eight-parameter host-dependent model against that free common population | evidences of runs R8d and R8e on one code and one prior box | Sec. IV B |
| `\FreeLognAGNPeakOverPrior` | largest posterior density of log10 n_AGN divided by the prior density (0.5 per dex), m < 18, catalog A (DESIGN L3: about three) | the kernel density of Fig. 7(b) in `make_figures.py`, or a histogram of the stored samples | Sec. IV C |
| `\MockNinjFound` | detected simulated signals of the reference catalog (README: 2,205,380) | `data/seed100/injections/injections_targeted_meta.json` | Appendix A |
| `\MCShiftMedian` | shift of the posterior medians when the posterior samples and the simulated signals are drawn anew, in percent of the 90% half-width | run R0 | Sec. V, Appendix A |
| `\MCWidthChange` | change of the 90% widths under the same redraws, in percent | run R0 | Appendix A |
| `\MCWidthChangeAGN` | the same for the sets of AGN-hosted mergers | run R0 | Appendix A |
| `\LnLVarAGNHosted` | Monte Carlo variance of ln L at the posterior maximum for the sets of AGN-hosted mergers (GAPS A1: 54.7 from the posterior samples plus 1.3 to 2.6 from the selection term, reference catalog) | the stored variance arrays of `analysis_0_pure_tracer_H0/results/` | Appendix A |

## B. Plain: design choices typed in the preamble through `\req{}`

| proposed macro | value typed | meaning | source of the value |
|---|---|---|---|
| `\PriorHzeroLo`, `\PriorHzeroHi` | 50, 100 | prior range of H0 in the two-parameter and free-density analyses | `values.json` (`CommonHzero.source.computation`, `FreeLognGalEighteen.source.priors`), LIKELIHOOD_MATH Sec. 0 |
| `\PriorHDHzeroLo`, `\PriorHDHzeroHi` | 60, 76 | prior range of H0 in the host-dependent models | `values.json` boxes of `HDFreeHzero`, `HDEightDmuChi`; CAPTION_FACTS Fig. 4 |
| `\PriorLognGalLo`, `\PriorLognGalHi` | -4, -1 | prior range of log10 n_gal | `values.json` `FreeLognGalEighteen.source.priors` |
| `\PriorLognAGNLo`, `\PriorLognAGNHi` | -6, -4 | prior range of log10 n_AGN | same record |
| `\PriorMuGLo`, `\PriorMuGHi` | 31, 39 | prior range of the reference mass-peak location | `values.json` box of `HDEightDmuChi` |
| `\PriorDmuGLo`, `\PriorDmuGHi` | -4, 10 | prior range of the mass-peak offset | same box; `LBOffsetsVsCommon.source.computation` |
| `\PriorSigmaGLo`, `\PriorSigmaGHi` | 1, 10 | prior range of the mass-peak width | box of `HDEightDmuChi` |
| `\PriorMuChiLo`, `\PriorMuChiHi` | -0.1, 0.1 | prior range of the reference spin mean | same box |
| `\PriorDmuChiLo`, `\PriorDmuChiHi` | -0.05, 0.30 | prior range of the spin-mean offset, eight-parameter model | same box |
| `\PriorDmuChiHiSix` | 0.20 | upper end of that prior in the models with fixed widths and in `\LBOffsetsVsCommon` | box of `HDFreeHzero`; `LBOffsetsVsCommon.source.computation` |
| `\PriorSigmaChiLo`, `\PriorSigmaChiHi` | 0.01, 1 | prior range of the spin width | box of `HDEightDmuChi` |
| `\PriorSpinDmuChiLo`, `\PriorSpinDmuChiHi` | -0.2, 0.25 | prior range of the spin-mean offset in the spin-offset mock | `SpinBothDmuChi.source.computation`, meaning of `\LBSpinFixH` |
| `\NormZmax` | 5 | upper redshift of the host normalization in the flux-limited analyses | LIKELIHOOD_MATH Sec. 0, 2 and 4 |
| `\DensityFactorLo`, `\DensityFactorHi` | 0.5, 2 | range of the factors by which the assumed AGN density is off | meaning of `\SlopefAGNTwentyOne`; CAPTION_FACTS Fig. 6 |
| `\MockShellWidth` | 200 | thickness of the lognormal shells in Mpc | `working/data/README.md`, catalogs stage |
| `\MockRhoRef`, `\MockMcRef` | 6.28, 30 | signal-to-noise ratio of a 30 Msun redshifted chirp mass at 1 Gpc, and that mass | `working/data/README.md` (snr_ref = 6.278), `DESIGN_PE.md` Sec. 2.1 |
| `\MockSkyCoeff` | 19.1 | sky error in degrees times the observed signal-to-noise ratio | `DESIGN_PE.md` Sec. 1.5 (19.1069) |
| `\MockSkyClipLo`, `\MockSkyClipHi` | 1, 12 | bounds of the sky error in degrees | `working/data/README.md`, `DESIGN_PE.md` |
| `\MockNinjDrawn` | 1.5 x 10^8 | simulated signals drawn for the reference catalog | `working/data/README.md`, injections stage |
| `\NeffFactor` | 5 | required effective number of detected signals per event | LIKELIHOOD_MATH Sec. 6 |
| `\VarCapEight` | 20 | largest admitted variance of ln L in the eight-parameter run | LIKELIHOOD_MATH Sec. 7 |
| `\PinDmuG`, `\PinDmuChi` | 5.0, 0.115 | offsets held fixed in the host-dependent model of Fig. 4(b) and Table III | CAPTION_FACTS Fig. 4(b); meaning of `\PinIOneHzero` |

A point for whoever fills Table I: `runs.txt` gives other prior ranges for the production runs
(mass-peak location [20, 50] Msun, offset [-10, 10] Msun). Table I lists the ranges under which the
teal numbers were obtained, which are the ones above.

## C. Numbers typed by hand in the text, on purpose

None of them is a result of this work.

- Published values quoted with their citation: the two LIGO-Virgo-KAGRA H0 values with their 90%
  intervals and source counts (142, 236), the published limits and measurements of the AGN-hosted
  fraction (0.21; 0.39 +0.41 -0.32; 0.41 to 0.61), "25% with about 1000 events", "a quarter",
  "about 10%", the luminosity thresholds 10^44.5 and 10^45 erg/s, the density 10^-4.75 Mpc^-3 of
  the association studies, the redshift error "about 0.01" of Pouw et al., and "3.7% to 3.2%" of
  Zhu et al. 2024.
- The prior range [0, 1] of f_AGN (Table I, Table II caption, Sec. IV A).
- "0.90" in Sec. IV C: the share of a prior range that the 90% interval of a uniform prior covers.
- "|delta| < 3 degrees" in the caption of Fig. 1: the label drawn in the figure.
- Counts written as words: six and eight parameters, three intervals, four inputs, four flux limits,
  "a hundred times" for the ratio of the two input number densities.
