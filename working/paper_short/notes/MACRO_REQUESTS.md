# Macro requests of the short paper (17 Nov 2026)

Macros that `main.tex` uses and `values/results_macros.tex` does not define. Each is declared in
`fallback_macros.tex` with `\providecommand{\Name}{\ph{..}}` and prints two red dots until a macro of
the same name is added to `values/results_macros.tex`. Names follow the long paper's requests where
it asked for the same number, so one addition serves both drafts.

| macro | what it is | used in |
|---|---|---|
| `\ObsPeakFrac` | fraction of the mergers in the Gaussian peak for the observed-like mass spectrum (plain design value once the event sets exist) | Sec. III, Table III |
| `\NrealKnown` | number of realizations combined, population known (keys `Known...`) | Appendix D |
| `\NrealSingle` | number of realizations combined, single-population analyses (keys `SingleBoth`, `SingleNone`, `HDSingle`, `HDNone`) | Sec. III, Appendix D |
| `\NrealHD` | number of realizations combined, host-dependent populations analysis (key `HDHost`) | Sec. III, Appendix D |
| `\NrealObs` | number of realizations combined, observed-like spectrum (keys `Obs...`) | Appendix D |
| `\NrealFree` | number of realizations combined, number densities inferred (keys `Free...`) | Appendix C, Appendix D |
| `\NrealCtrl` | number of realizations combined, each of the two control sets (keys `CtrlZero`, `CtrlTenth`) | Appendix D |
| `\MockNAGNCompleteEighteen` | number of AGN of the m < 18 catalog inside the redshift to which it is complete (`\MockzCompleteEighteen`), realization A (Analysis 15 report: about 4500) | Sec. IV D |
| `\MocknAGNPrecEighteen` | precision, in percent, of the AGN number density measured from those counts | Sec. IV D |
| `\NormDeltafCountsEighteen` | shift of the f_AGN median at m < 18 for an AGN number density off by that precision (Analysis 15 report: about 0.01) | Sec. IV D |
| `\MCNsampHi` | largest number of posterior samples per merger in the Monte Carlo accuracy study (the `ThirtyTwoK` suffix of the Analysis 14 macros) | Appendix E |

Eleven macros. Everything else in `main.tex` is defined in `values/results_macros.tex`.

## Numbers typed by hand, on purpose

None is a result of this work; each is a published value quoted beside its citation and checked in
`../paper_prd/notes/PRIOR_WORK_VERIFIED.md` or `../paper_prd/notes/NOVELTY.md`.

- Fifth transient catalog: 235 dark sirens with GW170817, H0 = 71.7 (+17.7, -11.4) at 90% (arXiv:2605.27227v2, Table 3); 236 sources.
- Fourth transient catalog: 142 sources.
- Pouw et al.: limits 0.41 to 0.61 at 95% on the fraction hosted by luminous unobscured AGN.
- Zhu et al. 2024: H0 precision from 3.7% to 3.2% for extreme-mass-ratio inspirals.
- AGN number density 10^-4.75 per Mpc^3 of the association studies (Veronesi et al. 2022; Bom and Palmese 2024).
- The interval convention (90%).

## Notes for whoever fills the macros

- `\AGNhostedHzeroWid` (1.6), `\GalhostedHzeroWid` (6.5) and `\WidthRatioAGNGal` (4.7) come from
  different sources and do not agree with one another; Sec. IV C quotes the first and the third.
- Table I takes its Bayes factors from `\HDLB` and `\ObsLB` (realization A), beside posteriors
  combined over realizations; the caption says so.
- Sec. IV D and Appendix A use the Analysis 15 macros (black). Table IV uses the red `Flux...`
  and `Free...` macros of realization A for all four limits.
