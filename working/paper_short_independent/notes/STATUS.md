# Status of the short paper's numbers and figures (2026-11-17 13:09 UTC)

Written by `scripts/build_values.py` from `scripts/results_map.json`; do not edit. **real** = black `\newcommand` in `values_independent_results.tex` (from the run summaries); **placeholder** = the red expected value of `values_independent.tex`. Run paths are under `analyses/analysis_16_production_runs/runs/`; a run's stage is that of `RUNLIST.yaml` (1-4) or `RUNLIST_experiment.yaml` (E1-E3).

**9 of 42 result macros real.**

## Macros

| macro | status | printed | meaning | waits for |
|---|---|---|---|---|
| `\IndHWSpec` | placeholder | `6.0` | half-width, mass spectrum alone, no catalog (f = 0 set; the f = 0.3 set's stage-1 grid is the same analysis on other events) | `mass_spectrum_alone/no_catalog/complete/peak_f0p00_s0` (0/5 summaries; missing 100,101,102,103,105), stage E1 |
| `\IndHWGal` | placeholder | `3.4` | half-width, all mergers in galaxies, galaxy catalog | `galaxy_catalog_alone/galaxy_catalog/complete/peak_f0p00_s0` (0/5 summaries; missing 100,101,102,103,105), stage E1 |
| `\IndHWAGN` | **real** | `0.81` | half-width, all mergers in AGN, AGN catalog |  |
| `\IndHWMix` | placeholder | `2.0` | half-width, f = 0.30, labels unknown | `two_tracers_one_population/both_catalogs/complete/peak_f0p30_s0` (2/5 summaries; missing 100,102,105), stage E1 |
| `\IndHWMark` | placeholder | `1.5` | half-width, f = 0.30, a population for each tracer, (5 Msun, 0.10) | `two_tracers_two_populations/both_catalogs/complete/peak_f0p30_s1` (0/5 summaries; missing 100,101,102,103,105), stage E1 |
| `\IndHWMarkTwice` | placeholder | `1.4` | the same, (10 Msun, 0.20) | `two_tracers_two_populations/both_catalogs/complete/peak_f0p30_s2` (0/5 summaries; missing 100,101,102,103,105), stage E1 |
| `\IndHWKnown` | **real** | `1.3` | half-width, f = 0.30, hosts known |  |
| `\IndHWKnownAGNPart` | **real** | `1.4` | half-width of the AGN factor of hosts known (mergers in AGN alone), f = 0.30 |  |
| `\IndHWMixTenth` | placeholder | `2.4` | half-width, f = 0.10, labels unknown | `two_tracers_one_population/both_catalogs/complete/peak_f0p10_s0` (0/5 summaries; missing 100,101,102,103,105), stage E1 |
| `\IndHWKnownTenth` | placeholder | `2.0` | half-width, f = 0.10, hosts known | `hosts_known/both_catalogs/complete/peak_f0p10_s0` (3/5 summaries; missing 101,103), stage E1 |
| `\IndCostMix` | placeholder | `2.4` | mergers needed with labels unknown over hosts known | derived: (IndHWMix/IndHWKnown)**2 |
| `\IndCostMark` | placeholder | `1.3` | the same, (5 Msun, 0.10) | derived: (IndHWMark/IndHWKnown)**2 |
| `\IndCostMarkTwice` | placeholder | `1.2` | the same, (10 Msun, 0.20) | derived: (IndHWMarkTwice/IndHWKnown)**2 |
| `\IndfAGNMix` | **real** | `0.27^{+0.08}_{-0.08}` | f_AGN, f = 0.30, equal populations, labels unknown, population held at its input, realization A (the run of Fig. 2 and row 1 of Table I) |  |
| `\IndObsHWSpec` | placeholder | `10` | observed-like: half-width, mass spectrum alone | `mass_spectrum_alone/no_catalog/complete/obslike_f0p00_s0` (0/5 summaries; missing 100,101,102,103,105), stage E2 |
| `\IndObsHWMix` | placeholder | `2.1` | observed-like: half-width, f = 0.30, labels unknown | `two_tracers_one_population/both_catalogs/complete/obslike_f0p30_s0` (2/5 summaries; missing 100,102,105), stage E2 |
| `\IndObsHWAGN` | placeholder | `0.8` | observed-like: half-width, all mergers in AGN, AGN catalog | `hosts_known/both_catalogs/complete/obslike_f1p00_s0` (0/5 summaries; missing 100,101,102,103,105), stage E2 |
| `\IndProbAGNPos` | placeholder | `0.36` | mean probability of an AGN host, mergers truly in AGN, positions alone, f = 0.30, (5 Msun, 0.10) | `two_tracers_one_population/both_catalogs/complete/peak_f0p30_s1` (0/1 summaries; missing 100), stage E1 |
| `\IndProbGalPos` | placeholder | `0.27` | mean probability of an AGN host, mergers truly in galaxies, positions alone, f = 0.30, (5 Msun, 0.10) | `two_tracers_one_population/both_catalogs/complete/peak_f0p30_s1` (0/1 summaries; missing 100), stage E1 |
| `\IndProbAGNMark` | placeholder | `0.41` | mean probability of an AGN host, mergers truly in AGN, positions with masses and spins, f = 0.30, (5 Msun, 0.10) | `two_tracers_two_populations/both_catalogs/complete/peak_f0p30_s1` (0/1 summaries; missing 100), stage E1 |
| `\IndProbGalMark` | placeholder | `0.25` | mean probability of an AGN host, mergers truly in galaxies, positions with masses and spins, f = 0.30, (5 Msun, 0.10) | `two_tracers_two_populations/both_catalogs/complete/peak_f0p30_s1` (0/1 summaries; missing 100), stage E1 |
| `\IndBias` | placeholder | `3` | one population fitted, (5 Msun, 0.10): median of the combined H0 posterior minus the input | `single_population/both_catalogs/complete/host_dependent` (0/10 summaries; missing 100,101,102,103,105,106,107,108,109,110), stage 4 |
| `\IndBiasTwice` | placeholder | `6` | the same for (10 Msun, 0.20) (experiment E3) | `single_population/both_catalogs/complete/peak_f0p30_s2` (0/5 summaries; missing 100,101,102,103,105), stage E3 |
| `\IndHWOnePop` | placeholder | `2.6` | one population fitted: half-width from one catalog (mean over realizations) | `single_population/both_catalogs/complete/host_dependent` (0/10 summaries; missing 100,101,102,103,105,106,107,108,109,110), stage 4 |
| `\IndOnePopCL` | placeholder | `99.9` | credible level (percent) of the input H0 in the combined one-population posterior | `single_population/both_catalogs/complete/host_dependent` (0/10 summaries; missing 100,101,102,103,105,106,107,108,109,110), stage 4 |
| `\IndTwoPopHzero` | placeholder | `67.4^{+2.1}_{-2.1}` | a population for each tracer, inferred, one catalog: H0 | `host_dependent_populations/both_catalogs/complete/host_dependent` (0/1 summaries; missing 100), stage 4 |
| `\IndTwoPopHW` | placeholder | `2.1` | its half-width, mean over realizations (open symbol of Fig. 3a) | `host_dependent_populations/both_catalogs/complete/host_dependent` (0/5 summaries; missing 100,101,102,103,105), stage 4 |
| `\IndTwoPopfAGN` | placeholder | `0.30^{+0.06}_{-0.06}` | f_AGN of the same analysis, one catalog | `host_dependent_populations/both_catalogs/complete/host_dependent` (0/1 summaries; missing 100), stage 4 |
| `\IndTwoPopDm` | placeholder | `5.1^{+1.0}_{-1.0}` | mass-peak difference [Msun], one catalog | `host_dependent_populations/both_catalogs/complete/host_dependent` (0/1 summaries; missing 100), stage 4 |
| `\IndTwoPopDchi` | placeholder | `0.10^{+0.03}_{-0.03}` | mean effective spin difference, one catalog | `host_dependent_populations/both_catalogs/complete/host_dependent` (0/1 summaries; missing 100), stage 4 |
| `\IndTwoPopCL` | placeholder | `30` | credible level (percent) of the input H0 in the combined posterior, a population for each tracer | `host_dependent_populations/both_catalogs/complete/host_dependent` (0/5 summaries; missing 100,101,102,103,105), stage 4 |
| `\IndLB` | placeholder | `9` | log10 Bayes factor, a population for each tracer over one population, one catalog | `host_dependent_populations/both_catalogs/complete/host_dependent` (0/1 summaries; missing 100), stage 4 / `single_population/both_catalogs/complete/host_dependent` (0/1 summaries; missing 100), stage 4 |
| `\IndRecHzeroComb` | **real** | `68.2^{+0.6}_{-0.6}` | H0, equal populations, f = 0.30, labels unknown, combined over ten realizations |  |
| `\IndRecHzeroCL` | **real** | `80` | credible level (percent) of the input H0 in it |  |
| `\IndRecfAGNComb` | **real** | `0.31^{+0.03}_{-0.03}` | f_AGN of the same combination |  |
| `\IndRecfAGNCL` | **real** | `30` | credible level (percent) of the input f_AGN in it |  |
| `\IndZerofAGNUL` | placeholder | `0.05` | no merger in AGN: 90% upper limit on f_AGN, combined over the experiment's five realizations | `two_tracers_one_population/both_catalogs/complete/peak_f0p00_s0` (0/5 summaries; missing 100,101,102,103,105), stage E1 |
| `\IndAnchorHzero` | **real** | `68.4^{+1.9}_{-2.0}` | H0, equal populations, f = 0.30, labels unknown, population held at its input, realization A (Sec. IV A, row 1 of Table I, Fig. 2) |  |
| `\IndOnePopHzero` | placeholder | `70.9^{+2.6}_{-2.6}` | H0, one population inferred from mergers whose populations differ by (5 Msun, 0.10), realization A (row 2 of Table I) | `single_population/both_catalogs/complete/host_dependent` (0/1 summaries; missing 100), stage 4 |
| `\IndOnePopfAGN` | placeholder | `0.24^{+0.09}_{-0.09}` | f_AGN of the same analysis, realization A (row 2 of Table I) | `single_population/both_catalogs/complete/host_dependent` (0/1 summaries; missing 100), stage 4 |
| `\IndOnePopHzeroComb` | placeholder | `70.7^{+0.8}_{-0.8}` | H0, one population inferred, combined over the ten realizations (dashed curve of Fig. 4a; Sec. V) | `single_population/both_catalogs/complete/host_dependent` (0/10 summaries; missing 100,101,102,103,105,106,107,108,109,110), stage 4 |
| `\IndTwoPopHzeroComb` | placeholder | `67.5^{+0.9}_{-0.9}` | H0, a population for each tracer, combined over the five realizations (solid curve of Fig. 4a; Sec. V) | `host_dependent_populations/both_catalogs/complete/host_dependent` (0/5 summaries; missing 100,101,102,103,105), stage 4 |

## Figures

Fig. 2 `figures/fig_anchor`, Fig. 3 `figures/fig1_budget`, Fig. 4 `figures/fig_populations` (drawn by `scripts/make_figures.py`). **real** = every run it needs exists (final palette); **partial k/n** = drawn from the k realizations that exist, still in red; **placeholder** = expected value, red.

| figure element | status | value drawn | waits for |
|---|---|---|---|
| Fig. 2 One realization | **real** | 90% region of (H0, f_AGN); 1 of 1 realizations (100): ln L summed, 90% region; largest density on the mesh border 2.3e-05 of the peak |  |
| Fig. 2 Combination of ten realizations | **real** | 90% region of (H0, f_AGN); 10 of 10 realizations (100,101,102,103,105,106,107,108,109,110): ln L summed, 90% region; largest density on the mesh border 1.2e-16 of the peak |  |
| Fig. 3(a) Labels unknown, f = 0 | placeholder | 3.59 [3.23, 3.95] | `two_tracers_one_population/both_catalogs/complete/peak_f0p00_s0` (0/5 summaries; missing 100,101,102,103,105), stage E1 |
| Fig. 3(a) Labels unknown, f = 0.1 | placeholder | 2.40 [2.16, 2.64] | = `\IndHWMixTenth`: `two_tracers_one_population/both_catalogs/complete/peak_f0p10_s0` (0/5 summaries; missing 100,101,102,103,105), stage E1 |
| Fig. 3(a) Labels unknown, f = 0.3 | partial 2/5 | 1.89 [1.89, 1.90] | = `\IndHWMix`: `two_tracers_one_population/both_catalogs/complete/peak_f0p30_s0` (2/5 summaries; missing 100,102,105), stage E1 |
| Fig. 3(a) Labels unknown, f = 0.5 | placeholder | 1.33 [1.20, 1.46] | `two_tracers_one_population/both_catalogs/complete/peak_f0p50_s0` (0/5 summaries; missing 100,101,102,103,105), stage E1 |
| Fig. 3(a) Labels unknown, f = 1 | placeholder | 0.84 [0.76, 0.92] | `two_tracers_one_population/both_catalogs/complete/peak_f1p00_s0` (0/5 summaries; missing 100,101,102,103,105), stage E1 |
| Fig. 3(a) A population for each tracer, $\Delta m=5\,M_\odot$, $\Delta\chi=0.10$, f = 0.1 | placeholder | 2.21 [1.99, 2.43] | `two_tracers_two_populations/both_catalogs/complete/peak_f0p10_s1` (0/5 summaries; missing 100,101,102,103,105), stage E1 |
| Fig. 3(a) A population for each tracer, $\Delta m=5\,M_\odot$, $\Delta\chi=0.10$, f = 0.3 | placeholder | 1.50 [1.35, 1.65] | = `\IndHWMark`: `two_tracers_two_populations/both_catalogs/complete/peak_f0p30_s1` (0/5 summaries; missing 100,101,102,103,105), stage E1 |
| Fig. 3(a) A population for each tracer, $\Delta m=5\,M_\odot$, $\Delta\chi=0.10$, f = 0.5 | placeholder | 1.19 [1.07, 1.31] | `two_tracers_two_populations/both_catalogs/complete/peak_f0p50_s1` (0/5 summaries; missing 100,101,102,103,105), stage E1 |
| Fig. 3(a) A population for each tracer, twice those values, f = 0.1 | placeholder | 2.11 [1.90, 2.32] | `two_tracers_two_populations/both_catalogs/complete/peak_f0p10_s2` (0/5 summaries; missing 100,101,102,103,105), stage E1 |
| Fig. 3(a) A population for each tracer, twice those values, f = 0.3 | placeholder | 1.40 [1.26, 1.54] | = `\IndHWMarkTwice`: `two_tracers_two_populations/both_catalogs/complete/peak_f0p30_s2` (0/5 summaries; missing 100,101,102,103,105), stage E1 |
| Fig. 3(a) A population for each tracer, twice those values, f = 0.5 | placeholder | 1.12 [1.01, 1.23] | `two_tracers_two_populations/both_catalogs/complete/peak_f0p50_s2` (0/5 summaries; missing 100,101,102,103,105), stage E1 |
| Fig. 3(a) Hosts known, f = 0 | placeholder | 3.40 [3.06, 3.74] | = `\IndHWGal`: `galaxy_catalog_alone/galaxy_catalog/complete/peak_f0p00_s0` (0/5 summaries; missing 100,101,102,103,105), stage E1 |
| Fig. 3(a) Hosts known, f = 0.1 | partial 3/5 | 1.82 [1.44, 2.03] | = `\IndHWKnownTenth`: `hosts_known/both_catalogs/complete/peak_f0p10_s0` (3/5 summaries; missing 101,103), stage E1 |
| Fig. 3(a) Hosts known, f = 0.3 | **real** | 1.34 [1.29, 1.41] |  |
| Fig. 3(a) Hosts known, f = 0.5 | placeholder | 1.05 [0.95, 1.16] | `hosts_known/both_catalogs/complete/peak_f0p50_s0` (0/5 summaries; missing 100,101,102,103,105), stage E1 |
| Fig. 3(a) Hosts known, f = 1 | **real** | 0.81 [0.80, 0.83] |  |
| Fig. 3(a) A population for each tracer inferred with $H_0$, f = 0.3 | placeholder | 2.10 [1.89, 2.31] | = `\IndTwoPopHW`: `host_dependent_populations/both_catalogs/complete/host_dependent` (0/5 summaries; missing 100,101,102,103,105), stage 4 |
| Fig. 3(a) Mass spectrum alone, line | placeholder | 6.00 [5.40, 6.60] | = `\IndHWSpec`: `mass_spectrum_alone/no_catalog/complete/peak_f0p00_s0` (0/5 summaries; missing 100,101,102,103,105), stage E1 |
| Fig. 3(a) Galaxy catalog alone, $f_{\rm AGN}=0$, line | placeholder | 3.40 [3.06, 3.74] | = `\IndHWGal`: `galaxy_catalog_alone/galaxy_catalog/complete/peak_f0p00_s0` (0/5 summaries; missing 100,101,102,103,105), stage E1 |
| Fig. 3(b) Labels unknown, f = 0 | placeholder | 8.21 [7.39, 9.03] | `two_tracers_one_population/both_catalogs/complete/obslike_f0p00_s0` (0/5 summaries; missing 100,101,102,103,105), stage E2 |
| Fig. 3(b) Labels unknown, f = 0.3 | partial 2/5 | 1.96 [1.85, 2.07] | = `\IndObsHWMix`: `two_tracers_one_population/both_catalogs/complete/obslike_f0p30_s0` (2/5 summaries; missing 100,102,105), stage E2 |
| Fig. 3(b) Labels unknown, f = 1 | placeholder | 0.84 [0.76, 0.92] | `two_tracers_one_population/both_catalogs/complete/obslike_f1p00_s0` (0/5 summaries; missing 100,101,102,103,105), stage E2 |
| Fig. 3(b) A population for each tracer, $\Delta m=5\,M_\odot$, $\Delta\chi=0.10$, f = 0.3 | placeholder | 1.62 [1.46, 1.78] | `two_tracers_two_populations/both_catalogs/complete/obslike_f0p30_s1` (0/5 summaries; missing 100,101,102,103,105), stage E2 |
| Fig. 3(b) Hosts known, f = 0 | placeholder | 8.00 [7.20, 8.80] | `galaxy_catalog_alone/galaxy_catalog/complete/obslike_f0p00_s0` (0/5 summaries; missing 100,101,102,103,105), stage E2 |
| Fig. 3(b) Hosts known, f = 0.3 | partial 3/5 | 1.35 [1.26, 1.48] | `hosts_known/both_catalogs/complete/obslike_f0p30_s0` (3/5 summaries; missing 101,103), stage E2 |
| Fig. 3(b) Hosts known, f = 1 | placeholder | 0.80 [0.72, 0.88] | = `\IndObsHWAGN`: `hosts_known/both_catalogs/complete/obslike_f1p00_s0` (0/5 summaries; missing 100,101,102,103,105), stage E2 |
| Fig. 3(b) Mass spectrum alone, line | placeholder | 10.00 [9.00, 11.00] | = `\IndObsHWSpec`: `mass_spectrum_alone/no_catalog/complete/obslike_f0p00_s0` (0/5 summaries; missing 100,101,102,103,105), stage E2 |
| Fig. 3(b) Galaxy catalog alone, $f_{\rm AGN}=0$, line | placeholder | 8.00 [7.20, 8.80] | `galaxy_catalog_alone/galaxy_catalog/complete/obslike_f0p00_s0` (0/5 summaries; missing 100,101,102,103,105), stage E2 |
| Fig. 4(a) One population fitted to all mergers | placeholder | median 70.74, 90% [69.92, 71.56] | `single_population/both_catalogs/complete/host_dependent` (0/10 summaries; missing 100,101,102,103,105,106,107,108,109,110), stage 4; placeholder: Gaussian, median 70.74, 90% half-width 0.82 (IndHWOnePop/sqrt(10)) |
| Fig. 4(a) A population for each tracer | placeholder | median 67.52, 90% [66.58, 68.46] | `host_dependent_populations/both_catalogs/complete/host_dependent` (0/5 summaries; missing 100,101,102,103,105), stage 4; placeholder: Gaussian, median 67.52, 90% half-width 0.94 (IndTwoPopHW/sqrt(5)) |
| Fig. 4(b) A population for each tracer, one catalog | placeholder | 90% region | `host_dependent_populations/both_catalogs/complete/host_dependent` (0/1 summaries; missing 100), stage 4; placeholder: Gaussian from ['IndTwoPopDm', 'IndTwoPopDchi'] with correlation -0.13 (paper_prd placeholder key HDHost, the exploratory Analysis 13 chain) |
| Fig. 4(c) Positions alone | placeholder | means 0.360 (AGN), 0.274 (galaxies) | `two_tracers_one_population/both_catalogs/complete/peak_f0p30_s1` (0/1 summaries; missing 100), stage E1; placeholder: calibrated logit-normal (sd of the log-likelihood ratio 0.66) matched to IndProbAGNPos = 0.36; its galaxy mean 0.274 against IndProbGalPos = 0.27 |
| Fig. 4(c) Positions with masses and spins | placeholder | means 0.410 (AGN), 0.253 (galaxies) | `two_tracers_two_populations/both_catalogs/complete/peak_f0p30_s1` (0/1 summaries; missing 100), stage E1; placeholder: calibrated logit-normal (sd of the log-likelihood ratio 0.91) matched to IndProbAGNMark = 0.41; its galaxy mean 0.253 against IndProbGalMark = 0.25 |
