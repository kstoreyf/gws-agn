# Macro menu (written by scripts/build_values.py; do not edit)

Class: plain = design choice or mock input; teal = exploratory result (production replaces it); red = not yet computable (see placeholders.txt).
Intervals are medians with 90% equal-tailed intervals unless stated. Full numbers, sources and containment records: values.json.

| macro | meaning | rendered | class | claim |
|---|---|---|---|---|
| `\MockHzero` | input Hubble constant | $67.74$ | plain | mock |
| `\MockOmegam` | input matter density Omega_m (fixed in every analysis) | $0.3075$ | plain | mock |
| `\MockfAGN` | input fraction of mergers hosted by AGN | $0.30$ | plain | mock |
| `\MockfAGNRealised` | realised AGN-hosted fraction of the 1000 detected events of the reference catalog | $0.295$ | plain | mock |
| `\MocknGal` | comoving number density of the galaxy catalog | $10^{-3}$ | plain | mock |
| `\MocknAGN` | comoving number density of the AGN catalog | $10^{-5}$ | plain | mock |
| `\MockLognGal` | log10 of the galaxy number density (input of the free-abundance analysis) | $-3.0$ | plain | mock |
| `\MockLognAGN` | log10 of the AGN number density | $-5.0$ | plain | mock |
| `\MockbGal` | linear bias of the galaxies | $1.2$ | plain | mock |
| `\MockbAGN` | linear bias of the AGN | $2.0$ | plain | mock |
| `\MockLcut` | luminosity cut of the galaxy sample, L > x L* (B-band Schechter function) | $1.09$ | plain | mock |
| `\MockMBcut` | absolute B magnitude of the galaxy cut | $-20.56$ | plain | mock |
| `\MockLXcut` | AGN class: log10 L_X(2-10 keV) above this value (pinned literature density) | $43.7$ | plain | mock |
| `\MockNGal` | realised galaxies to z = 1 (151,179,870) | $1.5\times10^{8}$ | plain | mock |
| `\MockNAGN` | realised AGN to z = 1 (1,514,567) | $1.5\times10^{6}$ | plain | mock |
| `\MockzCat` | redshift depth of the catalogs | $1$ | plain | mock |
| `\MockDz` | catalog redshift error sigma_z = this value times (1+z) | $0.003$ | plain | mock |
| `\MockNside` | HEALPix nside of the catalog pixelisation | $32$ | plain | mock |
| `\MockNev` | detected mergers per simulated catalog | $1000$ | plain | mock |
| `\MockNsamp` | posterior samples per event | $2000$ | plain | mock |
| `\MockSNRthr` | detection threshold on the observed signal-to-noise ratio | $8$ | plain | mock |
| `\MockzHorizon` | largest true redshift of a detected event, reference catalog (the horizon) | $0.31$ | plain | mock |
| `\MockzMedian` | median redshift of the detected events, reference catalog | $0.13$ | plain | mock |
| `\MockzHorizonMin` | smallest horizon over the 9 catalogs of the joint analysis | $0.31$ | plain | mock |
| `\MockzHorizonMax` | largest horizon over the 9 catalogs of the joint analysis | $0.39$ | plain | mock |
| `\MockSkyMin` | smallest one-sigma sky localisation width, reference catalog | $1.0$ | plain | mock |
| `\MockSkyMax` | largest one-sigma sky localisation width, reference catalog | $2.4$ | plain | mock |
| `\MockSigLnMc` | chirp-mass error: sigma(ln Mc) = this value times 8/rho_obs | $0.08$ | plain | mock |
| `\MockSigLnq` | mass-ratio error: sigma(ln q) = this value times 8/rho_obs | $0.60$ | plain | mock |
| `\MockSigChi` | effective-spin error: sigma(chi_eff) = this value times 8/rho_obs | $0.20$ | plain | mock |
| `\MockMlimTwentyOne` | apparent-magnitude limit m < 21 | $21$ | plain | mock,C4 |
| `\MockCompTwentyOne` | completeness of the galaxy catalog inside the horizon at m < 21 (AGN catalog: 100.0%) | $100.0$ | plain | mock,C4 |
| `\MockMlimTwenty` | apparent-magnitude limit m < 20 | $20$ | plain | mock,C4 |
| `\MockCompTwenty` | completeness of the galaxy catalog inside the horizon at m < 20 (AGN catalog: 81.4%) | $81.4$ | plain | mock,C4 |
| `\MockMlimNineteen` | apparent-magnitude limit m < 19 | $19$ | plain | mock,C4 |
| `\MockCompNineteen` | completeness of the galaxy catalog inside the horizon at m < 19 (AGN catalog: 31.7%) | $31.5$ | plain | mock,C4 |
| `\MockMlimEighteen` | apparent-magnitude limit m < 18 | $18$ | plain | mock,C4 |
| `\MockCompEighteen` | completeness of the galaxy catalog inside the horizon at m < 18 (AGN catalog: 9.6%) | $9.5$ | plain | mock,C4 |
| `\MockAlpha` | power-law index of the primary mass | $2.3$ | plain | mock |
| `\MockMmin` | minimum mass | $5$ | plain | mock |
| `\MockMmax` | maximum mass | $80$ | plain | mock |
| `\MockPeakFrac` | fraction of the primary-mass distribution in the Gaussian peak | $0.90$ | plain | mock |
| `\MockMuG` | location of the Gaussian mass peak (galaxy hosts; both host classes in the common-population mocks) | $35$ | plain | mock |
| `\MockSigmaG` | width of the Gaussian mass peak | $5$ | plain | mock |
| `\MockBeta` | power-law index of the mass ratio | $1$ | plain | mock |
| `\MockMuChi` | mean effective spin (galaxy hosts) | $0$ | plain | mock |
| `\MockSigmaChi` | width of the effective-spin distribution | $0.1$ | plain | mock |
| `\MockGammaRate` | redshift evolution index of the merger rate, (1+z)^gamma | $0$ | plain | mock |
| `\MockDmuG` | input offset of the AGN mass-peak location (host-dependent mock) | $5$ | plain | mock,C7,C8 |
| `\MockMuGAGN` | AGN mass-peak location in the host-dependent mock | $40$ | plain | mock,C7,C8 |
| `\MockDmuChi` | input offset of the AGN mean effective spin | $0.10$ | plain | mock,C7,C8,C9 |
| `\MockHDNagn` | AGN-hosted events among the 1000 detected in the host-dependent mock | $357$ | plain | mock,C7 |
| `\MockHDfDet` | AGN-hosted fraction of the detected events in the host-dependent mock (heavier AGN mergers are louder; input 0.30) | $0.357$ | plain | mock,C7 |
| `\MockSpinNagn` | AGN-hosted events in the spin-offset-only mock (Analyses 8, 9) | $295$ | plain | mock,C9 |
| `\MockCtrlBranchFrac` | fraction of mergers drawn from the offset population in the galaxy-only control mock (all hosts galaxies) | $0.30$ | plain | mock,C7 |
| `\JointNcat` | simulated catalogs with a complete joint (H0, f_AGN) grid on complete catalogs | $9$ | teal | C1 |
| `\JointHzeroCatA` | H0, complete catalogs, joint (H0, f_AGN) fit, catalog A (seed 100) (median and 90% equal-tailed interval) | $69.2^{+1.6}_{-1.6}$ | teal | C1 |
| `\JointHzeroCatAMed` | median of H0, complete catalogs, joint (H0, f_AGN) fit, catalog A (seed 100) | $69.2$ | teal | C1 |
| `\JointHzeroCatALo` | lower end of the 90% interval of H0, complete catalogs, joint (H0, f_AGN) fit, catalog A (seed 100) | $67.6$ | teal | C1 |
| `\JointHzeroCatAHi` | upper end of the 90% interval of H0, complete catalogs, joint (H0, f_AGN) fit, catalog A (seed 100) | $70.8$ | teal | C1 |
| `\JointfCatA` | f_AGN, complete catalogs, joint fit, catalog A (seed 100) (median and 90% equal-tailed interval) | $0.273^{+0.078}_{-0.075}$ | teal | C1 |
| `\JointfCatAMed` | median of f_AGN, complete catalogs, joint fit, catalog A (seed 100) | $0.273$ | teal | C1 |
| `\JointfCatALo` | lower end of the 90% interval of f_AGN, complete catalogs, joint fit, catalog A (seed 100) | $0.198$ | teal | C1 |
| `\JointfCatAHi` | upper end of the 90% interval of f_AGN, complete catalogs, joint fit, catalog A (seed 100) | $0.352$ | teal | C1 |
| `\JointRhoCatA` | correlation coefficient of H0 and f_AGN, catalog A (seed 100) | $0.07$ | teal | C1 |
| `\JointHzeroCatB` | H0, complete catalogs, joint (H0, f_AGN) fit, catalog B (seed 101) (median and 90% equal-tailed interval) | $67.8^{+2.2}_{-2.0}$ | teal | C1 |
| `\JointfCatB` | f_AGN, complete catalogs, joint fit, catalog B (seed 101) (median and 90% equal-tailed interval) | $0.307^{+0.081}_{-0.077}$ | teal | C1 |
| `\JointRhoCatB` | correlation coefficient of H0 and f_AGN, catalog B (seed 101) | $0.23$ | teal | C1 |
| `\JointHzeroCatC` | H0, complete catalogs, joint (H0, f_AGN) fit, catalog C (seed 102) (median and 90% equal-tailed interval) | $66.5^{+1.9}_{-1.8}$ | teal | C1 |
| `\JointfCatC` | f_AGN, complete catalogs, joint fit, catalog C (seed 102) (median and 90% equal-tailed interval) | $0.247^{+0.085}_{-0.081}$ | teal | C1 |
| `\JointRhoCatC` | correlation coefficient of H0 and f_AGN, catalog C (seed 102) | $0.10$ | teal | C1 |
| `\JointHzeroCatD` | H0, complete catalogs, joint (H0, f_AGN) fit, catalog D (seed 103) (median and 90% equal-tailed interval) | $69.6^{+1.6}_{-1.5}$ | teal | C1 |
| `\JointfCatD` | f_AGN, complete catalogs, joint fit, catalog D (seed 103) (median and 90% equal-tailed interval) | $0.256^{+0.078}_{-0.074}$ | teal | C1 |
| `\JointRhoCatD` | correlation coefficient of H0 and f_AGN, catalog D (seed 103) | $0.03$ | teal | C1 |
| `\JointHzeroCatE` | H0, complete catalogs, joint (H0, f_AGN) fit, catalog E (seed 105) (median and 90% equal-tailed interval) | $67.6^{+1.7}_{-1.5}$ | teal | C1 |
| `\JointfCatE` | f_AGN, complete catalogs, joint fit, catalog E (seed 105) (median and 90% equal-tailed interval) | $0.352^{+0.083}_{-0.081}$ | teal | C1 |
| `\JointRhoCatE` | correlation coefficient of H0 and f_AGN, catalog E (seed 105) | $0.09$ | teal | C1 |
| `\JointHzeroCatF` | H0, complete catalogs, joint (H0, f_AGN) fit, catalog F (seed 114) (median and 90% equal-tailed interval) | $68.2^{+2.7}_{-2.7}$ | teal | C1 |
| `\JointfCatF` | f_AGN, complete catalogs, joint fit, catalog F (seed 114) (median and 90% equal-tailed interval) | $0.268^{+0.081}_{-0.077}$ | teal | C1 |
| `\JointRhoCatF` | correlation coefficient of H0 and f_AGN, catalog F (seed 114) | $0.03$ | teal | C1 |
| `\JointHzeroCatG` | H0, complete catalogs, joint (H0, f_AGN) fit, catalog G (seed 115) (median and 90% equal-tailed interval) | $67.7^{+1.9}_{-1.7}$ | teal | C1 |
| `\JointfCatG` | f_AGN, complete catalogs, joint fit, catalog G (seed 115) (median and 90% equal-tailed interval) | $0.330^{+0.082}_{-0.081}$ | teal | C1 |
| `\JointRhoCatG` | correlation coefficient of H0 and f_AGN, catalog G (seed 115) | $0.10$ | teal | C1 |
| `\JointHzeroCatH` | H0, complete catalogs, joint (H0, f_AGN) fit, catalog H (seed 116) (median and 90% equal-tailed interval) | $68.3^{+1.7}_{-1.7}$ | teal | C1 |
| `\JointfCatH` | f_AGN, complete catalogs, joint fit, catalog H (seed 116) (median and 90% equal-tailed interval) | $0.351^{+0.086}_{-0.085}$ | teal | C1 |
| `\JointRhoCatH` | correlation coefficient of H0 and f_AGN, catalog H (seed 116) | $0.15$ | teal | C1 |
| `\JointHzeroCatI` | H0, complete catalogs, joint (H0, f_AGN) fit, catalog I (seed 117) (median and 90% equal-tailed interval) | $69.3^{+2.2}_{-2.6}$ | teal | C1 |
| `\JointfCatI` | f_AGN, complete catalogs, joint fit, catalog I (seed 117) (median and 90% equal-tailed interval) | $0.154^{+0.083}_{-0.077}$ | teal | C1 |
| `\JointRhoCatI` | correlation coefficient of H0 and f_AGN, catalog I (seed 117) | $-0.05$ | teal | C1 |
| `\JointRhoMean` | mean over the 9 catalogs of the H0-f_AGN correlation coefficient | $0.08$ | teal | C1 |
| `\JointRhoMin` | smallest H0-f_AGN correlation coefficient over the 9 catalogs | $-0.05$ | teal | C1 |
| `\JointRhoMax` | largest H0-f_AGN correlation coefficient over the 9 catalogs | $0.23$ | teal | C1 |
| `\JointHzeroComb` | H0, combined posterior of the 9 complete-catalog joint analyses (median and 90% equal-tailed interval) | $68.36^{+0.63}_{-0.63}$ | teal | C1 |
| `\JointHzeroCombMed` | median of H0, combined posterior of the 9 complete-catalog joint analyses | $68.36$ | teal | C1 |
| `\JointHzeroCombLo` | lower end of the 90% interval of H0, combined posterior of the 9 complete-catalog joint analyses | $67.73$ | teal | C1 |
| `\JointHzeroCombHi` | upper end of the 90% interval of H0, combined posterior of the 9 complete-catalog joint analyses | $69.00$ | teal | C1 |
| `\JointfComb` | f_AGN, combined posterior of the 9 complete-catalog joint analyses (median and 90% equal-tailed interval) | $0.283^{+0.027}_{-0.027}$ | teal | C1 |
| `\JointfCombMed` | median of f_AGN, combined posterior of the 9 complete-catalog joint analyses | $0.283$ | teal | C1 |
| `\JointfCombLo` | lower end of the 90% interval of f_AGN, combined posterior of the 9 complete-catalog joint analyses | $0.256$ | teal | C1 |
| `\JointfCombHi` | upper end of the 90% interval of f_AGN, combined posterior of the 9 complete-catalog joint analyses | $0.310$ | teal | C1 |
| `\JointHzeroCombLevel` | two-sided credible level at which the input H0 = 67.74 lies in the combined complete-catalog posterior | $89$ | teal | C1,validation |
| `\JointfCombLevel` | two-sided credible level at which the input f_AGN = 0.30 lies in the combined complete-catalog posterior | $70$ | teal | C1,validation |
| `\JointCombLevelTwoD` | credible level of the highest-density 2-D region of the combined (H0, f_AGN) posterior whose boundary passes through the input point | $88$ | teal | C1,validation |
| `\JointRhoComb` | H0-f_AGN correlation coefficient of the combined posterior | $0.11$ | teal | C1 |
| `\JointLnLVarMax` | largest over the 9 catalogs of the Monte Carlo variance of ln L (selection plus event terms) at the grid maximum | $11.0$ | teal | C1,appendix |
| `\JointLnLVarMin` | smallest such variance over the 9 catalogs | $8.1$ | teal | C1,appendix |
| `\GalOnlyHzeroCatA` | H0 from all 1000 events with the galaxy catalog alone, catalog A (seed 100) (median and 90% equal-tailed interval) | $69.9^{+3.0}_{-2.7}$ | teal | C2 |
| `\GalOnlyHzeroCatAMed` | median of H0 from all 1000 events with the galaxy catalog alone, catalog A (seed 100) | $69.9$ | teal | C2 |
| `\GalOnlyHzeroCatALo` | lower end of the 90% interval of H0 from all 1000 events with the galaxy catalog alone, catalog A (seed 100) | $67.2$ | teal | C2 |
| `\GalOnlyHzeroCatAHi` | upper end of the 90% interval of H0 from all 1000 events with the galaxy catalog alone, catalog A (seed 100) | $72.9$ | teal | C2 |
| `\LBfZeroCatA` | log10 Bayes factor of the two-population model against galaxy hosts only, Savage-Dickey at f_AGN = 0, catalog A (seed 100) | $11.1$ | teal | C2 |
| `\LBfOneCatA` | log10 Bayes factor of the two-population model against AGN hosts only, Savage-Dickey at f_AGN = 1, catalog A (seed 100) | $357.6$ | teal | C2 |
| `\GalOnlyHzeroCatB` | H0 from all 1000 events with the galaxy catalog alone, catalog B (seed 101) (median and 90% equal-tailed interval) | $69.0^{+5.2}_{-3.1}$ | teal | C2 |
| `\LBfZeroCatB` | log10 Bayes factor of the two-population model against galaxy hosts only, Savage-Dickey at f_AGN = 0, catalog B (seed 101) | $12.6$ | teal | C2 |
| `\LBfOneCatB` | log10 Bayes factor of the two-population model against AGN hosts only, Savage-Dickey at f_AGN = 1, catalog B (seed 101) | $258.1$ | teal | C2 |
| `\GalOnlyHzeroCatC` | H0 from all 1000 events with the galaxy catalog alone, catalog C (seed 102) (median and 90% equal-tailed interval) | $66.0^{+4.1}_{-3.8}$ | teal | C2 |
| `\LBfZeroCatC` | log10 Bayes factor of the two-population model against galaxy hosts only, Savage-Dickey at f_AGN = 0, catalog C (seed 102) | $5.97$ | teal | C2 |
| `\LBfOneCatC` | log10 Bayes factor of the two-population model against AGN hosts only, Savage-Dickey at f_AGN = 1, catalog C (seed 102) | $101.0$ | teal | C2 |
| `\GalOnlyHzeroCatD` | H0 from all 1000 events with the galaxy catalog alone, catalog D (seed 103) (median and 90% equal-tailed interval) | $73.4^{+3.6}_{-4.4}$ | teal | C2 |
| `\LBfZeroCatD` | log10 Bayes factor of the two-population model against galaxy hosts only, Savage-Dickey at f_AGN = 0, catalog D (seed 103) | $11.0$ | teal | C2 |
| `\LBfOneCatD` | log10 Bayes factor of the two-population model against AGN hosts only, Savage-Dickey at f_AGN = 1, catalog D (seed 103) | $269.7$ | teal | C2 |
| `\GalOnlyHzeroCatE` | H0 from all 1000 events with the galaxy catalog alone, catalog E (seed 105) (median and 90% equal-tailed interval) | $67.2^{+3.2}_{-2.9}$ | teal | C2 |
| `\LBfZeroCatE` | log10 Bayes factor of the two-population model against galaxy hosts only, Savage-Dickey at f_AGN = 0, catalog E (seed 105) | $14.4$ | teal | C2 |
| `\LBfOneCatE` | log10 Bayes factor of the two-population model against AGN hosts only, Savage-Dickey at f_AGN = 1, catalog E (seed 105) | $184.6$ | teal | C2 |
| `\GalOnlyHzeroCatF` | H0 from all 1000 events with the galaxy catalog alone, catalog F (seed 114) (median and 90% equal-tailed interval) | $69.4^{+4.9}_{-3.6}$ | teal | C2 |
| `\LBfZeroCatF` | log10 Bayes factor of the two-population model against galaxy hosts only, Savage-Dickey at f_AGN = 0, catalog F (seed 114) | $9.45$ | teal | C2 |
| `\LBfOneCatF` | log10 Bayes factor of the two-population model against AGN hosts only, Savage-Dickey at f_AGN = 1, catalog F (seed 114) | $223.5$ | teal | C2 |
| `\GalOnlyHzeroCatG` | H0 from all 1000 events with the galaxy catalog alone, catalog G (seed 115) (median and 90% equal-tailed interval) | $69.3^{+5.3}_{-4.3}$ | teal | C2 |
| `\LBfZeroCatG` | log10 Bayes factor of the two-population model against galaxy hosts only, Savage-Dickey at f_AGN = 0, catalog G (seed 115) | $12.8$ | teal | C2 |
| `\LBfOneCatG` | log10 Bayes factor of the two-population model against AGN hosts only, Savage-Dickey at f_AGN = 1, catalog G (seed 115) | $241.6$ | teal | C2 |
| `\GalOnlyHzeroCatH` | H0 from all 1000 events with the galaxy catalog alone, catalog H (seed 116) (median and 90% equal-tailed interval) | $68.2^{+3.6}_{-4.1}$ | teal | C2 |
| `\LBfZeroCatH` | log10 Bayes factor of the two-population model against galaxy hosts only, Savage-Dickey at f_AGN = 0, catalog H (seed 116) | $15.9$ | teal | C2 |
| `\LBfOneCatH` | log10 Bayes factor of the two-population model against AGN hosts only, Savage-Dickey at f_AGN = 1, catalog H (seed 116) | $173.5$ | teal | C2 |
| `\GalOnlyHzeroCatI` | H0 from all 1000 events with the galaxy catalog alone, catalog I (seed 117) (median and 90% equal-tailed interval) | $69.8^{+2.7}_{-2.8}$ | teal | C2 |
| `\LBfZeroCatI` | log10 Bayes factor of the two-population model against galaxy hosts only, Savage-Dickey at f_AGN = 0, catalog I (seed 117) | $1.82$ | teal | C2 |
| `\LBfOneCatI` | log10 Bayes factor of the two-population model against AGN hosts only, Savage-Dickey at f_AGN = 1, catalog I (seed 117) | $203.2$ | teal | C2 |
| `\AgnOnlyNoInterior` | number of the 9 catalogs whose AGN-catalog-only H0 posterior (all events) has no interior maximum in [50, 100] (log-likelihood largest at a grid edge) | $8$ | teal | C2 |
| `\AgnOnlyHzeroCatC` | H0 from all 1000 events with the AGN catalog alone, catalog C (seed 102), the one catalog whose AGN-only posterior has an interior maximum (median and 90% equal-tailed interval) | $76.21^{+1.01}_{-0.81}$ | teal | C2 |
| `\AgnOnlyArgmaxCatA` | H0 at which the AGN-catalog-only log-likelihood (all events, catalog A) is largest; 100 is the top of the scanned range | $100.0$ | teal | C2 |
| `\LBfZeroMin` | smallest per-catalog log10 B (two populations against galaxy hosts only) over the 9 catalogs | $1.8$ | teal | C2 |
| `\LBfZeroMax` | largest per-catalog log10 B (two populations against galaxy hosts only) | $15.9$ | teal | C2 |
| `\GalOnlyHzeroComb` | H0 with the galaxy catalog alone (all events), combined over the 9 catalogs (median and 90% equal-tailed interval) | $68.9^{+1.3}_{-1.3}$ | teal | C2 |
| `\GalOnlyHzeroCombMed` | median of H0 with the galaxy catalog alone (all events), combined over the 9 catalogs | $68.9$ | teal | C2 |
| `\GalOnlyHzeroCombLo` | lower end of the 90% interval of H0 with the galaxy catalog alone (all events), combined over the 9 catalogs | $67.7$ | teal | C2 |
| `\GalOnlyHzeroCombHi` | upper end of the 90% interval of H0 with the galaxy catalog alone (all events), combined over the 9 catalogs | $70.2$ | teal | C2 |
| `\GalOnlyHzeroCombLevel` | two-sided credible level of the input H0 in the combined galaxy-catalog-only posterior | $88$ | teal | C2 |
| `\LBfZeroComb` | log10 Bayes factor of the two-population model against galaxy hosts only, combined over the 9 catalogs (Savage-Dickey at f_AGN = 0 on the summed grid) | $101.2$ | teal | C2 |
| `\LBfOneComb` | log10 Bayes factor of the two-population model against AGN hosts only, combined over the 9 catalogs (Savage-Dickey at f_AGN = 1) | $2037.9$ | teal | C2 |
| `\PureNcat` | simulated catalogs with both a 1000-event all-galaxy-host set and a 1000-event all-AGN-host set analysed | $93$ | teal | C3 |
| `\PureGalHalfSixtyEight` | mean over catalogs of the 68% half-width of H0 from 1000 galaxy-hosted mergers | $2.11$ | teal | C3 |
| `\PureAGNHalfSixtyEight` | mean over catalogs of the 68% half-width of H0 from 1000 AGN-hosted mergers | $0.45$ | teal | C3 |
| `\PureGalHalfNinety` | mean over catalogs of the 90% half-width of H0, galaxy-hosted set | $3.53$ | teal | C3 |
| `\PureAGNHalfNinety` | mean over catalogs of the 90% half-width of H0, AGN-hosted set | $0.75$ | teal | C3 |
| `\PureRatioSixtyEight` | ratio of the mean 68% half-widths, galaxy-hosted over AGN-hosted | $4.7$ | teal | C3 |
| `\PureRatioNinety` | ratio of the mean 90% half-widths, galaxy-hosted over AGN-hosted | $4.7$ | teal | C3 |
| `\PureGalHzeroComb` | H0, combined posterior of the 93 galaxy-hosted event sets (median and 90% equal-tailed interval) | $67.71^{+0.36}_{-0.35}$ | teal | C3,validation |
| `\PureGalHzeroCombMed` | median of H0, combined posterior of the 93 galaxy-hosted event sets | $67.71$ | teal | C3,validation |
| `\PureGalHzeroCombLo` | lower end of the 90% interval of H0, combined posterior of the 93 galaxy-hosted event sets | $67.35$ | teal | C3,validation |
| `\PureGalHzeroCombHi` | upper end of the 90% interval of H0, combined posterior of the 93 galaxy-hosted event sets | $68.06$ | teal | C3,validation |
| `\PureGalHzeroCombLevel` | two-sided credible level of the input H0 in the combined galaxy-hosted posterior | $12$ | teal | C3,validation |
| `\PureAGNHzeroComb` | H0, combined posterior of the 93 AGN-hosted event sets (median and 90% equal-tailed interval) | $67.820^{+0.080}_{-0.079}$ | teal | C3,validation |
| `\PureAGNHzeroCombMed` | median of H0, combined posterior of the 93 AGN-hosted event sets | $67.820$ | teal | C3,validation |
| `\PureAGNHzeroCombLo` | lower end of the 90% interval of H0, combined posterior of the 93 AGN-hosted event sets | $67.741$ | teal | C3,validation |
| `\PureAGNHzeroCombHi` | upper end of the 90% interval of H0, combined posterior of the 93 AGN-hosted event sets | $67.901$ | teal | C3,validation |
| `\PureAGNHzeroCombLevel` | two-sided credible level of the input H0 in the combined AGN-hosted posterior | $91$ | teal | C3,validation |
| `\FluxfHalfComplete` | 90% half-width of f_AGN with complete catalogs (reference catalog, C1 grid) | $0.077$ | teal | C4 |
| `\FluxHzeroHalfComplete` | 90% half-width of H0 with complete catalogs (reference catalog, C1 grid) | $1.60$ | teal | C4 |
| `\FluxHzeroTwentyOne` | H0, flux limit m < 21, densities of missing hosts at their input values (median and 90% equal-tailed interval) | $69.2^{+1.6}_{-1.6}$ | teal | C4 |
| `\FluxfTwentyOne` | f_AGN, flux limit m < 21, densities at input (median and 90% equal-tailed interval) | $0.273^{+0.078}_{-0.075}$ | teal | C4 |
| `\FluxfHalfTwentyOne` | 90% half-width of f_AGN at m < 21 | $0.077$ | teal | C4 |
| `\FluxHzeroHalfTwentyOne` | 90% half-width of H0 at m < 21 | $1.59$ | teal | C4 |
| `\FluxfRatioTwentyOne` | 90% half-width of f_AGN at m < 21 over that with complete catalogs (C1 grid, same catalog) | $1.00$ | teal | C4 |
| `\FluxHzeroRatioTwentyOne` | 90% half-width of H0 at m < 21 over that with complete catalogs (C1 grid, same catalog) | $1.00$ | teal | C4 |
| `\FluxfShiftTwentyOne` | median f_AGN at m < 21 minus the complete-catalog median (same catalog) | $0.000$ | teal | C4 |
| `\FluxHzeroShiftTwentyOne` | median H0 at m < 21 minus the complete-catalog median (same catalog) | $0.00$ | teal | C4 |
| `\FluxHzeroTwenty` | H0, flux limit m < 20, densities of missing hosts at their input values (median and 90% equal-tailed interval) | $69.2^{+1.6}_{-1.6}$ | teal | C4 |
| `\FluxfTwenty` | f_AGN, flux limit m < 20, densities at input (median and 90% equal-tailed interval) | $0.273^{+0.078}_{-0.075}$ | teal | C4 |
| `\FluxfHalfTwenty` | 90% half-width of f_AGN at m < 20 | $0.077$ | teal | C4 |
| `\FluxHzeroHalfTwenty` | 90% half-width of H0 at m < 20 | $1.60$ | teal | C4 |
| `\FluxfRatioTwenty` | 90% half-width of f_AGN at m < 20 over that with complete catalogs (C1 grid, same catalog) | $1.00$ | teal | C4 |
| `\FluxHzeroRatioTwenty` | 90% half-width of H0 at m < 20 over that with complete catalogs (C1 grid, same catalog) | $1.00$ | teal | C4 |
| `\FluxfShiftTwenty` | median f_AGN at m < 20 minus the complete-catalog median (same catalog) | $0.000$ | teal | C4 |
| `\FluxHzeroShiftTwenty` | median H0 at m < 20 minus the complete-catalog median (same catalog) | $0.00$ | teal | C4 |
| `\FluxHzeroNineteen` | H0, flux limit m < 19, densities of missing hosts at their input values (median and 90% equal-tailed interval) | $69.0^{+1.6}_{-1.6}$ | teal | C4 |
| `\FluxfNineteen` | f_AGN, flux limit m < 19, densities at input (median and 90% equal-tailed interval) | $0.275^{+0.079}_{-0.076}$ | teal | C4 |
| `\FluxfHalfNineteen` | 90% half-width of f_AGN at m < 19 | $0.078$ | teal | C4 |
| `\FluxHzeroHalfNineteen` | 90% half-width of H0 at m < 19 | $1.62$ | teal | C4 |
| `\FluxfRatioNineteen` | 90% half-width of f_AGN at m < 19 over that with complete catalogs (C1 grid, same catalog) | $1.01$ | teal | C4 |
| `\FluxHzeroRatioNineteen` | 90% half-width of H0 at m < 19 over that with complete catalogs (C1 grid, same catalog) | $1.01$ | teal | C4 |
| `\FluxfShiftNineteen` | median f_AGN at m < 19 minus the complete-catalog median (same catalog) | $0.002$ | teal | C4 |
| `\FluxHzeroShiftNineteen` | median H0 at m < 19 minus the complete-catalog median (same catalog) | $-0.26$ | teal | C4 |
| `\FluxHzeroEighteen` | H0, flux limit m < 18, densities of missing hosts at their input values (median and 90% equal-tailed interval) | $68.7^{+1.6}_{-1.8}$ | teal | C4 |
| `\FluxfEighteen` | f_AGN, flux limit m < 18, densities at input (median and 90% equal-tailed interval) | $0.286^{+0.083}_{-0.080}$ | teal | C4 |
| `\FluxfHalfEighteen` | 90% half-width of f_AGN at m < 18 | $0.082$ | teal | C4 |
| `\FluxHzeroHalfEighteen` | 90% half-width of H0 at m < 18 | $1.71$ | teal | C4 |
| `\FluxfRatioEighteen` | 90% half-width of f_AGN at m < 18 over that with complete catalogs (C1 grid, same catalog) | $1.06$ | teal | C4 |
| `\FluxHzeroRatioEighteen` | 90% half-width of H0 at m < 18 over that with complete catalogs (C1 grid, same catalog) | $1.07$ | teal | C4 |
| `\FluxfShiftEighteen` | median f_AGN at m < 18 minus the complete-catalog median (same catalog) | $0.013$ | teal | C4 |
| `\FluxHzeroShiftEighteen` | median H0 at m < 18 minus the complete-catalog median (same catalog) | $-0.55$ | teal | C4 |
| `\SlopefAGNTwentyOne` | d f_AGN / d log10 n_AGN at m < 21 (assumed AGN density of missing hosts varied by factors 0.5 to 2) | $0.43$ | teal | C5 |
| `\SlopeHzeroTwentyOne` | d H0 / d log10 n_AGN at m < 21 | $-0.02$ | teal | C5 |
| `\DeltafTwoTwentyOne` | change of f_AGN for a factor of two in the assumed AGN density at m < 21 (slope times log10 2) | $0.13$ | teal | C5 |
| `\DeltaHzeroTwoTwentyOne` | change of H0 for a factor of two in the assumed AGN density at m < 21 (slope times log10 2) | $-0.01$ | teal | C5 |
| `\SlopefAGNTwenty` | d f_AGN / d log10 n_AGN at m < 20 (assumed AGN density of missing hosts varied by factors 0.5 to 2) | $0.44$ | teal | C5 |
| `\SlopeHzeroTwenty` | d H0 / d log10 n_AGN at m < 20 | $-0.03$ | teal | C5 |
| `\DeltafTwoTwenty` | change of f_AGN for a factor of two in the assumed AGN density at m < 20 (slope times log10 2) | $0.13$ | teal | C5 |
| `\DeltaHzeroTwoTwenty` | change of H0 for a factor of two in the assumed AGN density at m < 20 (slope times log10 2) | $-0.01$ | teal | C5 |
| `\SlopefAGNNineteen` | d f_AGN / d log10 n_AGN at m < 19 (assumed AGN density of missing hosts varied by factors 0.5 to 2) | $0.47$ | teal | C5 |
| `\SlopeHzeroNineteen` | d H0 / d log10 n_AGN at m < 19 | $0.71$ | teal | C5 |
| `\DeltafTwoNineteen` | change of f_AGN for a factor of two in the assumed AGN density at m < 19 (slope times log10 2) | $0.14$ | teal | C5 |
| `\DeltaHzeroTwoNineteen` | change of H0 for a factor of two in the assumed AGN density at m < 19 (slope times log10 2) | $0.21$ | teal | C5 |
| `\SlopefAGNEighteen` | d f_AGN / d log10 n_AGN at m < 18 (assumed AGN density of missing hosts varied by factors 0.5 to 2) | $0.46$ | teal | C5 |
| `\SlopeHzeroEighteen` | d H0 / d log10 n_AGN at m < 18 | $3.21$ | teal | C5 |
| `\DeltafTwoEighteen` | change of f_AGN for a factor of two in the assumed AGN density at m < 18 (slope times log10 2) | $0.14$ | teal | C5 |
| `\DeltaHzeroTwoEighteen` | change of H0 for a factor of two in the assumed AGN density at m < 18 (slope times log10 2) | $0.97$ | teal | C5 |
| `\SlopefAGNMin` | smallest d f_AGN / d log10 n_AGN over the four flux limits | $0.43$ | teal | C5 |
| `\SlopefAGNMax` | largest d f_AGN / d log10 n_AGN over the four flux limits | $0.47$ | teal | C5 |
| `\FreeLognGalTwentyOne` | log10 galaxy number density, both densities free, m < 21, catalog A (seed 100) (median and 90% equal-tailed interval) | $-3.20^{+1.09}_{-0.73}$ | teal | C6 |
| `\FreeLognAGNTwentyOne` | log10 AGN number density, both densities free, m < 21, catalog A (seed 100) (median and 90% equal-tailed interval) | $-4.85^{+0.75}_{-0.94}$ | teal | C6 |
| `\FreefAGNTwentyOne` | f_AGN, both densities free, m < 21, catalog A (seed 100) (median and 90% equal-tailed interval) | $0.39^{+0.45}_{-0.33}$ | teal | C6 |
| `\FreeHzeroTwentyOne` | H0, both densities free, m < 21, catalog A (seed 100) (median and 90% equal-tailed interval) | $69.2^{+1.6}_{-1.6}$ | teal | C6 |
| `\FreeRhofGalTwentyOne` | correlation of f_AGN with log10 n_gal, m < 21, catalog A (seed 100) | $-0.56$ | teal | C6 |
| `\FreeRhofAGNTwentyOne` | correlation of f_AGN with log10 n_AGN, m < 21, catalog A (seed 100) | $0.47$ | teal | C6 |
| `\FreefWidthTwentyOne` | width of the f_AGN 90% interval as a fraction of its prior range [0, 1], m < 21, catalog A (seed 100) | $0.79$ | teal | C6 |
| `\FreeLnZTwentyOne` | natural-log evidence of the free-density model, m < 21, catalog A (seed 100) (dynesty) | $-4180.76\pm0.09$ | teal | C6 |
| `\FreeLognGalTwenty` | log10 galaxy number density, both densities free, m < 20, catalog A (seed 100) (median and 90% equal-tailed interval) | $-3.26^{+0.66}_{-0.65}$ | teal | C6 |
| `\FreeLognAGNTwenty` | log10 AGN number density, both densities free, m < 20, catalog A (seed 100) (median and 90% equal-tailed interval) | $-4.90^{+0.69}_{-0.90}$ | teal | C6 |
| `\FreefAGNTwenty` | f_AGN, both densities free, m < 20, catalog A (seed 100) (median and 90% equal-tailed interval) | $0.44^{+0.43}_{-0.36}$ | teal | C6 |
| `\FreeHzeroTwenty` | H0, both densities free, m < 20, catalog A (seed 100) (median and 90% equal-tailed interval) | $69.1^{+1.6}_{-1.6}$ | teal | C6 |
| `\FreeRhofGalTwenty` | correlation of f_AGN with log10 n_gal, m < 20, catalog A (seed 100) | $-0.54$ | teal | C6 |
| `\FreeRhofAGNTwenty` | correlation of f_AGN with log10 n_AGN, m < 20, catalog A (seed 100) | $0.70$ | teal | C6 |
| `\FreefWidthTwenty` | width of the f_AGN 90% interval as a fraction of its prior range [0, 1], m < 20, catalog A (seed 100) | $0.79$ | teal | C6 |
| `\FreeLnZTwenty` | natural-log evidence of the free-density model, m < 20, catalog A (seed 100) (dynesty) | $-4181.13\pm0.09$ | teal | C6 |
| `\FreeLognGalNineteen` | log10 galaxy number density, both densities free, m < 19, catalog A (seed 100) (median and 90% equal-tailed interval) | $-3.04^{+0.28}_{-0.71}$ | teal | C6 |
| `\FreeLognAGNNineteen` | log10 AGN number density, both densities free, m < 19, catalog A (seed 100) (median and 90% equal-tailed interval) | $-4.68^{+0.36}_{-0.83}$ | teal | C6 |
| `\FreefAGNNineteen` | f_AGN, both densities free, m < 19, catalog A (seed 100) (median and 90% equal-tailed interval) | $0.46^{+0.43}_{-0.39}$ | teal | C6 |
| `\FreeHzeroNineteen` | H0, both densities free, m < 19, catalog A (seed 100) (median and 90% equal-tailed interval) | $69.2^{+1.7}_{-1.6}$ | teal | C6 |
| `\FreeRhofGalNineteen` | correlation of f_AGN with log10 n_gal, m < 19, catalog A (seed 100) | $-0.92$ | teal | C6 |
| `\FreeRhofAGNNineteen` | correlation of f_AGN with log10 n_AGN, m < 19, catalog A (seed 100) | $0.89$ | teal | C6 |
| `\FreefWidthNineteen` | width of the f_AGN 90% interval as a fraction of its prior range [0, 1], m < 19, catalog A (seed 100) | $0.82$ | teal | C6 |
| `\FreeLnZNineteen` | natural-log evidence of the free-density model, m < 19, catalog A (seed 100) (dynesty) | $-4183.91\pm0.10$ | teal | C6 |
| `\FreeLognGalEighteen` | log10 galaxy number density, both densities free, m < 18, catalog A (seed 100) (median and 90% equal-tailed interval) | $-3.12^{+0.29}_{-0.68}$ | teal | C6 |
| `\FreeLognAGNEighteen` | log10 AGN number density, both densities free, m < 18, catalog A (seed 100) (median and 90% equal-tailed interval) | $-4.72^{+0.32}_{-0.82}$ | teal | C6 |
| `\FreefAGNEighteen` | f_AGN, both densities free, m < 18, catalog A (seed 100) (median and 90% equal-tailed interval) | $0.50^{+0.40}_{-0.42}$ | teal | C6 |
| `\FreeHzeroEighteen` | H0, both densities free, m < 18, catalog A (seed 100) (median and 90% equal-tailed interval) | $69.1^{+1.8}_{-1.9}$ | teal | C6 |
| `\FreeHzeroEighteenMed` | median of H0, both densities free, m < 18, catalog A (seed 100) | $69.1$ | teal | C6 |
| `\FreeHzeroEighteenLo` | lower end of the 90% interval of H0, both densities free, m < 18, catalog A (seed 100) | $67.2$ | teal | C6 |
| `\FreeHzeroEighteenHi` | upper end of the 90% interval of H0, both densities free, m < 18, catalog A (seed 100) | $70.9$ | teal | C6 |
| `\FreeRhofGalEighteen` | correlation of f_AGN with log10 n_gal, m < 18, catalog A (seed 100) | $-0.94$ | teal | C6 |
| `\FreeRhofAGNEighteen` | correlation of f_AGN with log10 n_AGN, m < 18, catalog A (seed 100) | $0.90$ | teal | C6 |
| `\FreefWidthEighteen` | width of the f_AGN 90% interval as a fraction of its prior range [0, 1], m < 18, catalog A (seed 100) | $0.82$ | teal | C6 |
| `\FreeLnZEighteen` | natural-log evidence of the free-density model, m < 18, catalog A (seed 100) (dynesty) | $-4187.54\pm0.11$ | teal | C6 |
| `\FreeLognGalEighteenCatB` | log10 galaxy number density, both densities free, m < 18, catalog B (seed 101) (median and 90% equal-tailed interval) | $-3.11^{+0.27}_{-0.67}$ | teal | C6 |
| `\FreeLognAGNEighteenCatB` | log10 AGN number density, both densities free, m < 18, catalog B (seed 101) (median and 90% equal-tailed interval) | $-4.84^{+0.31}_{-0.77}$ | teal | C6 |
| `\FreefAGNEighteenCatB` | f_AGN, both densities free, m < 18, catalog B (seed 101) (median and 90% equal-tailed interval) | $0.48^{+0.41}_{-0.40}$ | teal | C6 |
| `\FreeHzeroEighteenCatB` | H0, both densities free, m < 18, catalog B (seed 101) (median and 90% equal-tailed interval) | $67.6^{+2.1}_{-1.9}$ | teal | C6 |
| `\FreeRhofGalEighteenCatB` | correlation of f_AGN with log10 n_gal, m < 18, catalog B (seed 101) | $-0.94$ | teal | C6 |
| `\FreeRhofAGNEighteenCatB` | correlation of f_AGN with log10 n_AGN, m < 18, catalog B (seed 101) | $0.91$ | teal | C6 |
| `\FreefWidthEighteenCatB` | width of the f_AGN 90% interval as a fraction of its prior range [0, 1], m < 18, catalog B (seed 101) | $0.81$ | teal | C6 |
| `\FreeLnZEighteenCatB` | natural-log evidence of the free-density model, m < 18, catalog B (seed 101) (dynesty) | $-4244.45\pm0.11$ | teal | C6 |
| `\FreeLognGalEighteenCatC` | log10 galaxy number density, both densities free, m < 18, catalog C (seed 102) (median and 90% equal-tailed interval) | $-3.17^{+0.27}_{-0.62}$ | teal | C6 |
| `\FreeLognAGNEighteenCatC` | log10 AGN number density, both densities free, m < 18, catalog C (seed 102) (median and 90% equal-tailed interval) | $-4.80^{+0.33}_{-0.79}$ | teal | C6 |
| `\FreefAGNEighteenCatC` | f_AGN, both densities free, m < 18, catalog C (seed 102) (median and 90% equal-tailed interval) | $0.46^{+0.41}_{-0.39}$ | teal | C6 |
| `\FreeHzeroEighteenCatC` | H0, both densities free, m < 18, catalog C (seed 102) (median and 90% equal-tailed interval) | $67.7^{+2.7}_{-2.3}$ | teal | C6 |
| `\FreeRhofGalEighteenCatC` | correlation of f_AGN with log10 n_gal, m < 18, catalog C (seed 102) | $-0.93$ | teal | C6 |
| `\FreeRhofAGNEighteenCatC` | correlation of f_AGN with log10 n_AGN, m < 18, catalog C (seed 102) | $0.90$ | teal | C6 |
| `\FreefWidthEighteenCatC` | width of the f_AGN 90% interval as a fraction of its prior range [0, 1], m < 18, catalog C (seed 102) | $0.80$ | teal | C6 |
| `\FreeLnZEighteenCatC` | natural-log evidence of the free-density model, m < 18, catalog C (seed 102) (dynesty) | $-4205.51\pm0.10$ | teal | C6 |
| `\FreeHzeroSpread` | range (max minus min) of the H0 medians over the four flux limits with both densities free (catalog A) | $0.13$ | teal | C6 |
| `\CommonHzero` | H0, host-dependent mock analysed with one common population (both offsets zero), two catalogs (median and 90% equal-tailed interval) | $70.6^{+2.3}_{-1.7}$ | teal | C7 |
| `\CommonHzeroMed` | median of H0, host-dependent mock analysed with one common population (both offsets zero), two catalogs | $70.6$ | teal | C7 |
| `\CommonHzeroLo` | lower end of the 90% interval of H0, host-dependent mock analysed with one common population (both offsets zero), two catalogs | $68.9$ | teal | C7 |
| `\CommonHzeroHi` | upper end of the 90% interval of H0, host-dependent mock analysed with one common population (both offsets zero), two catalogs | $72.9$ | teal | C7 |
| `\CommonfAGN` | f_AGN, common-population model on the host-dependent mock (median and 90% equal-tailed interval) | $0.287^{+0.086}_{-0.086}$ | teal | C7 |
| `\HDFixHzero` | H0, host-dependent model (offsets free, reference population fixed at mu_G = 35, mu_chi = 0) (median and 90% equal-tailed interval) | $67.7^{+1.7}_{-2.2}$ | teal | C8 |
| `\HDFixHzeroMed` | median of H0, host-dependent model (offsets free, reference population fixed at mu_G = 35, mu_chi = 0) | $67.7$ | teal | C8 |
| `\HDFixHzeroLo` | lower end of the 90% interval of H0, host-dependent model (offsets free, reference population fixed at mu_G = 35, mu_chi = 0) | $65.4$ | teal | C8 |
| `\HDFixHzeroHi` | upper end of the 90% interval of H0, host-dependent model (offsets free, reference population fixed at mu_G = 35, mu_chi = 0) | $69.4$ | teal | C8 |
| `\HDFixfAGN` | f_AGN, host-dependent model with the reference population fixed (median and 90% equal-tailed interval) | $0.273^{+0.068}_{-0.061}$ | teal | C8 |
| `\HDFixfAGNMed` | median of f_AGN, host-dependent model with the reference population fixed | $0.273$ | teal | C8 |
| `\HDFixfAGNLo` | lower end of the 90% interval of f_AGN, host-dependent model with the reference population fixed | $0.212$ | teal | C8 |
| `\HDFixfAGNHi` | upper end of the 90% interval of f_AGN, host-dependent model with the reference population fixed | $0.342$ | teal | C8 |
| `\HDFixDmuG` | AGN mass-peak offset, host-dependent model with the reference population fixed (median and 90% equal-tailed interval) | $4.80^{+1.08}_{-0.98}$ | teal | C8 |
| `\HDFixDmuGMed` | median of AGN mass-peak offset, host-dependent model with the reference population fixed | $4.80$ | teal | C8 |
| `\HDFixDmuGLo` | lower end of the 90% interval of AGN mass-peak offset, host-dependent model with the reference population fixed | $3.81$ | teal | C8 |
| `\HDFixDmuGHi` | upper end of the 90% interval of AGN mass-peak offset, host-dependent model with the reference population fixed | $5.87$ | teal | C8 |
| `\HDFixDmuChi` | AGN spin-mean offset, host-dependent model with the reference population fixed (median and 90% equal-tailed interval) | $0.118^{+0.027}_{-0.024}$ | teal | C8 |
| `\HDFixDmuChiMed` | median of AGN spin-mean offset, host-dependent model with the reference population fixed | $0.118$ | teal | C8 |
| `\HDFixDmuChiLo` | lower end of the 90% interval of AGN spin-mean offset, host-dependent model with the reference population fixed | $0.094$ | teal | C8 |
| `\HDFixDmuChiHi` | upper end of the 90% interval of AGN spin-mean offset, host-dependent model with the reference population fixed | $0.145$ | teal | C8 |
| `\HDFixRhoHzeroDmuG` | correlation of H0 and dmuG, host-dependent model with the reference population fixed | $-0.25$ | teal | C8 |
| `\HDFixRhoHzerof` | correlation of H0 and f, host-dependent model with the reference population fixed | $0.09$ | teal | C8 |
| `\HDFixRhofDmuG` | correlation of f and dmuG, host-dependent model with the reference population fixed | $-0.59$ | teal | C8 |
| `\HDFixRhofDmuChi` | correlation of f and dmuChi, host-dependent model with the reference population fixed | $-0.53$ | teal | C8 |
| `\HDFreeHzero` | H0, host-dependent model with the reference peak location and spin mean free (six parameters) (median and 90% equal-tailed interval) | $67.2^{+2.0}_{-2.3}$ | teal | C8 |
| `\HDFreeHzeroMed` | median of H0, host-dependent model with the reference peak location and spin mean free (six parameters) | $67.2$ | teal | C8 |
| `\HDFreeHzeroLo` | lower end of the 90% interval of H0, host-dependent model with the reference peak location and spin mean free (six parameters) | $64.8$ | teal | C8 |
| `\HDFreeHzeroHi` | upper end of the 90% interval of H0, host-dependent model with the reference peak location and spin mean free (six parameters) | $69.1$ | teal | C8 |
| `\HDFreefAGN` | f_AGN, host-dependent model with the reference peak location and spin mean free (six parameters) (median and 90% equal-tailed interval) | $0.266^{+0.079}_{-0.068}$ | teal | C8 |
| `\HDFreefAGNMed` | median of f_AGN, host-dependent model with the reference peak location and spin mean free (six parameters) | $0.266$ | teal | C8 |
| `\HDFreefAGNLo` | lower end of the 90% interval of f_AGN, host-dependent model with the reference peak location and spin mean free (six parameters) | $0.198$ | teal | C8 |
| `\HDFreefAGNHi` | upper end of the 90% interval of f_AGN, host-dependent model with the reference peak location and spin mean free (six parameters) | $0.345$ | teal | C8 |
| `\HDFreeMuG` | reference mass-peak location mu_G, host-dependent model with the reference peak location and spin mean free (six parameters) (median and 90% equal-tailed interval) | $35.61^{+0.64}_{-0.65}$ | teal | C8 |
| `\HDFreeMuGMed` | median of reference mass-peak location mu_G, host-dependent model with the reference peak location and spin mean free (six parameters) | $35.61$ | teal | C8 |
| `\HDFreeMuGLo` | lower end of the 90% interval of reference mass-peak location mu_G, host-dependent model with the reference peak location and spin mean free (six parameters) | $34.96$ | teal | C8 |
| `\HDFreeMuGHi` | upper end of the 90% interval of reference mass-peak location mu_G, host-dependent model with the reference peak location and spin mean free (six parameters) | $36.25$ | teal | C8 |
| `\HDFreeDmuG` | AGN mass-peak offset Delta mu_G, host-dependent model with the reference peak location and spin mean free (six parameters) (median and 90% equal-tailed interval) | $3.9^{+1.4}_{-1.4}$ | teal | C8 |
| `\HDFreeDmuGMed` | median of AGN mass-peak offset Delta mu_G, host-dependent model with the reference peak location and spin mean free (six parameters) | $3.9$ | teal | C8 |
| `\HDFreeDmuGLo` | lower end of the 90% interval of AGN mass-peak offset Delta mu_G, host-dependent model with the reference peak location and spin mean free (six parameters) | $2.5$ | teal | C8 |
| `\HDFreeDmuGHi` | upper end of the 90% interval of AGN mass-peak offset Delta mu_G, host-dependent model with the reference peak location and spin mean free (six parameters) | $5.4$ | teal | C8 |
| `\HDFreeMuChi` | reference mean effective spin mu_chi, host-dependent model with the reference peak location and spin mean free (six parameters) (median and 90% equal-tailed interval) | $-0.003^{+0.015}_{-0.015}$ | teal | C8 |
| `\HDFreeMuChiMed` | median of reference mean effective spin mu_chi, host-dependent model with the reference peak location and spin mean free (six parameters) | $-0.003$ | teal | C8 |
| `\HDFreeMuChiLo` | lower end of the 90% interval of reference mean effective spin mu_chi, host-dependent model with the reference peak location and spin mean free (six parameters) | $-0.019$ | teal | C8 |
| `\HDFreeMuChiHi` | upper end of the 90% interval of reference mean effective spin mu_chi, host-dependent model with the reference peak location and spin mean free (six parameters) | $0.011$ | teal | C8 |
| `\HDFreeDmuChi` | AGN spin-mean offset Delta mu_chi, host-dependent model with the reference peak location and spin mean free (six parameters) (median and 90% equal-tailed interval) | $0.131^{+0.034}_{-0.034}$ | teal | C8 |
| `\HDFreeDmuChiMed` | median of AGN spin-mean offset Delta mu_chi, host-dependent model with the reference peak location and spin mean free (six parameters) | $0.131$ | teal | C8 |
| `\HDFreeDmuChiLo` | lower end of the 90% interval of AGN spin-mean offset Delta mu_chi, host-dependent model with the reference peak location and spin mean free (six parameters) | $0.097$ | teal | C8 |
| `\HDFreeDmuChiHi` | upper end of the 90% interval of AGN spin-mean offset Delta mu_chi, host-dependent model with the reference peak location and spin mean free (six parameters) | $0.164$ | teal | C8 |
| `\WidthMHzero` | H0, six-parameter model plus a shared mass-peak width (median and 90% equal-tailed interval) | $67.4^{+1.7}_{-2.4}$ | teal | C8 |
| `\WidthMHzeroMed` | median of H0, six-parameter model plus a shared mass-peak width | $67.4$ | teal | C8 |
| `\WidthMHzeroLo` | lower end of the 90% interval of H0, six-parameter model plus a shared mass-peak width | $65.0$ | teal | C8 |
| `\WidthMHzeroHi` | upper end of the 90% interval of H0, six-parameter model plus a shared mass-peak width | $69.2$ | teal | C8 |
| `\WidthMfAGN` | f_AGN, six-parameter model plus a shared mass-peak width (median and 90% equal-tailed interval) | $0.267^{+0.073}_{-0.069}$ | teal | C8 |
| `\WidthMMuG` | reference mass-peak location mu_G, six-parameter model plus a shared mass-peak width (median and 90% equal-tailed interval) | $35.60^{+0.60}_{-0.64}$ | teal | C8 |
| `\WidthMDmuG` | AGN mass-peak offset Delta mu_G, six-parameter model plus a shared mass-peak width (median and 90% equal-tailed interval) | $4.5^{+1.4}_{-1.6}$ | teal | C8 |
| `\WidthMMuChi` | reference mean effective spin mu_chi, six-parameter model plus a shared mass-peak width (median and 90% equal-tailed interval) | $-0.004^{+0.015}_{-0.015}$ | teal | C8 |
| `\WidthMDmuChi` | AGN spin-mean offset Delta mu_chi, six-parameter model plus a shared mass-peak width (median and 90% equal-tailed interval) | $0.130^{+0.033}_{-0.032}$ | teal | C8 |
| `\WidthMSigmaG` | shared mass-peak width sigma_G, six-parameter model plus a shared mass-peak width (median and 90% equal-tailed interval) | $4.70^{+0.44}_{-0.46}$ | teal | C8 |
| `\WidthChiHzero` | H0, six-parameter model plus a shared spin width (median and 90% equal-tailed interval) | $67.2^{+1.9}_{-2.3}$ | teal | C8 |
| `\WidthChiHzeroMed` | median of H0, six-parameter model plus a shared spin width | $67.2$ | teal | C8 |
| `\WidthChiHzeroLo` | lower end of the 90% interval of H0, six-parameter model plus a shared spin width | $64.8$ | teal | C8 |
| `\WidthChiHzeroHi` | upper end of the 90% interval of H0, six-parameter model plus a shared spin width | $69.0$ | teal | C8 |
| `\WidthChifAGN` | f_AGN, six-parameter model plus a shared spin width (median and 90% equal-tailed interval) | $0.264^{+0.070}_{-0.066}$ | teal | C8 |
| `\WidthChiMuG` | reference mass-peak location mu_G, six-parameter model plus a shared spin width (median and 90% equal-tailed interval) | $35.66^{+0.60}_{-0.64}$ | teal | C8 |
| `\WidthChiDmuG` | AGN mass-peak offset Delta mu_G, six-parameter model plus a shared spin width (median and 90% equal-tailed interval) | $3.9^{+1.4}_{-1.4}$ | teal | C8 |
| `\WidthChiMuChi` | reference mean effective spin mu_chi, six-parameter model plus a shared spin width (median and 90% equal-tailed interval) | $-0.012^{+0.015}_{-0.015}$ | teal | C8 |
| `\WidthChiDmuChi` | AGN spin-mean offset Delta mu_chi, six-parameter model plus a shared spin width (median and 90% equal-tailed interval) | $0.160^{+0.031}_{-0.037}$ | teal | C8 |
| `\WidthChiSigmaChi` | shared effective-spin width sigma_chi, six-parameter model plus a shared spin width (median and 90% equal-tailed interval) | $0.081^{+0.014}_{-0.014}$ | teal | C8 |
| `\HDEightHzero` | H0, eight-parameter model (reference location, spin mean, both widths and both offsets free) (median and 90% equal-tailed interval) | $67.4^{+1.7}_{-2.4}$ | teal | C8 |
| `\HDEightHzeroMed` | median of H0, eight-parameter model (reference location, spin mean, both widths and both offsets free) | $67.4$ | teal | C8 |
| `\HDEightHzeroLo` | lower end of the 90% interval of H0, eight-parameter model (reference location, spin mean, both widths and both offsets free) | $65.0$ | teal | C8 |
| `\HDEightHzeroHi` | upper end of the 90% interval of H0, eight-parameter model (reference location, spin mean, both widths and both offsets free) | $69.2$ | teal | C8 |
| `\HDEightfAGN` | f_AGN, eight-parameter model (reference location, spin mean, both widths and both offsets free) (median and 90% equal-tailed interval) | $0.263^{+0.070}_{-0.064}$ | teal | C8 |
| `\HDEightfAGNMed` | median of f_AGN, eight-parameter model (reference location, spin mean, both widths and both offsets free) | $0.263$ | teal | C8 |
| `\HDEightfAGNLo` | lower end of the 90% interval of f_AGN, eight-parameter model (reference location, spin mean, both widths and both offsets free) | $0.199$ | teal | C8 |
| `\HDEightfAGNHi` | upper end of the 90% interval of f_AGN, eight-parameter model (reference location, spin mean, both widths and both offsets free) | $0.333$ | teal | C8 |
| `\HDEightMuG` | reference mass-peak location mu_G, eight-parameter model (reference location, spin mean, both widths and both offsets free) (median and 90% equal-tailed interval) | $35.66^{+0.59}_{-0.63}$ | teal | C8 |
| `\HDEightMuGMed` | median of reference mass-peak location mu_G, eight-parameter model (reference location, spin mean, both widths and both offsets free) | $35.66$ | teal | C8 |
| `\HDEightMuGLo` | lower end of the 90% interval of reference mass-peak location mu_G, eight-parameter model (reference location, spin mean, both widths and both offsets free) | $35.02$ | teal | C8 |
| `\HDEightMuGHi` | upper end of the 90% interval of reference mass-peak location mu_G, eight-parameter model (reference location, spin mean, both widths and both offsets free) | $36.25$ | teal | C8 |
| `\HDEightDmuG` | AGN mass-peak offset Delta mu_G, eight-parameter model (reference location, spin mean, both widths and both offsets free) (median and 90% equal-tailed interval) | $4.2^{+1.4}_{-1.5}$ | teal | C8 |
| `\HDEightDmuGMed` | median of AGN mass-peak offset Delta mu_G, eight-parameter model (reference location, spin mean, both widths and both offsets free) | $4.2$ | teal | C8 |
| `\HDEightDmuGLo` | lower end of the 90% interval of AGN mass-peak offset Delta mu_G, eight-parameter model (reference location, spin mean, both widths and both offsets free) | $2.7$ | teal | C8 |
| `\HDEightDmuGHi` | upper end of the 90% interval of AGN mass-peak offset Delta mu_G, eight-parameter model (reference location, spin mean, both widths and both offsets free) | $5.6$ | teal | C8 |
| `\HDEightMuChi` | reference mean effective spin mu_chi, eight-parameter model (reference location, spin mean, both widths and both offsets free) (median and 90% equal-tailed interval) | $-0.011^{+0.015}_{-0.015}$ | teal | C8 |
| `\HDEightMuChiMed` | median of reference mean effective spin mu_chi, eight-parameter model (reference location, spin mean, both widths and both offsets free) | $-0.011$ | teal | C8 |
| `\HDEightMuChiLo` | lower end of the 90% interval of reference mean effective spin mu_chi, eight-parameter model (reference location, spin mean, both widths and both offsets free) | $-0.026$ | teal | C8 |
| `\HDEightMuChiHi` | upper end of the 90% interval of reference mean effective spin mu_chi, eight-parameter model (reference location, spin mean, both widths and both offsets free) | $0.003$ | teal | C8 |
| `\HDEightDmuChi` | AGN spin-mean offset Delta mu_chi, eight-parameter model (reference location, spin mean, both widths and both offsets free) (median and 90% equal-tailed interval) | $0.158^{+0.033}_{-0.038}$ | teal | C8 |
| `\HDEightDmuChiMed` | median of AGN spin-mean offset Delta mu_chi, eight-parameter model (reference location, spin mean, both widths and both offsets free) | $0.158$ | teal | C8 |
| `\HDEightDmuChiLo` | lower end of the 90% interval of AGN spin-mean offset Delta mu_chi, eight-parameter model (reference location, spin mean, both widths and both offsets free) | $0.121$ | teal | C8 |
| `\HDEightDmuChiHi` | upper end of the 90% interval of AGN spin-mean offset Delta mu_chi, eight-parameter model (reference location, spin mean, both widths and both offsets free) | $0.191$ | teal | C8 |
| `\HDEightSigmaG` | shared mass-peak width sigma_G, eight-parameter model (reference location, spin mean, both widths and both offsets free) (median and 90% equal-tailed interval) | $4.83^{+0.44}_{-0.42}$ | teal | C8 |
| `\HDEightSigmaGMed` | median of shared mass-peak width sigma_G, eight-parameter model (reference location, spin mean, both widths and both offsets free) | $4.83$ | teal | C8 |
| `\HDEightSigmaGLo` | lower end of the 90% interval of shared mass-peak width sigma_G, eight-parameter model (reference location, spin mean, both widths and both offsets free) | $4.41$ | teal | C8 |
| `\HDEightSigmaGHi` | upper end of the 90% interval of shared mass-peak width sigma_G, eight-parameter model (reference location, spin mean, both widths and both offsets free) | $5.27$ | teal | C8 |
| `\HDEightSigmaChi` | shared effective-spin width sigma_chi, eight-parameter model (reference location, spin mean, both widths and both offsets free) (median and 90% equal-tailed interval) | $0.082^{+0.014}_{-0.014}$ | teal | C8 |
| `\HDEightSigmaChiMed` | median of shared effective-spin width sigma_chi, eight-parameter model (reference location, spin mean, both widths and both offsets free) | $0.082$ | teal | C8 |
| `\HDEightSigmaChiLo` | lower end of the 90% interval of shared effective-spin width sigma_chi, eight-parameter model (reference location, spin mean, both widths and both offsets free) | $0.069$ | teal | C8 |
| `\HDEightSigmaChiHi` | upper end of the 90% interval of shared effective-spin width sigma_chi, eight-parameter model (reference location, spin mean, both widths and both offsets free) | $0.096$ | teal | C8 |
| `\ShiftCommonMinusFix` | H0 median of the common-population model minus that of the host-dependent model (reference fixed); both on H0 in [60, 76] | $2.93$ | teal | C7 |
| `\ShiftCommonFullMinusFix` | same, common-population model on its full H0 range [50, 100] | $2.93$ | teal | C7 |
| `\ShiftFreeMinusFix` | H0 median of the free-reference model minus the fixed-reference model | $-0.50$ | teal | C8 |
| `\ShiftEightMinusFix` | H0 median of the eight-parameter model minus the fixed-reference model | $-0.25$ | teal | C8 |
| `\ShiftEightMinusFree` | H0 median of the eight-parameter model minus the six-parameter model | $0.25$ | teal | C8 |
| `\ShiftCommonMinusEight` | H0 median of the common-population model (H0 in [60, 76]) minus the eight-parameter model | $3.18$ | teal | C7 |
| `\WRatioFixOverCommon` | 90% width of H0, fixed-reference host-dependent model over the common-population model (both on [60, 76]) | $1.01$ | teal | C8 |
| `\WRatioFreeOverFix` | 90% width of H0, free-reference model over fixed-reference model | $1.09$ | teal | C8 |
| `\WRatioEightOverFree` | 90% width of H0, eight-parameter over six-parameter model | $0.97$ | teal | C8 |
| `\WRatioEightOverFix` | 90% width of H0, eight-parameter over fixed-reference model | $1.05$ | teal | C8 |
| `\WRatioEightOverCommon` | 90% width of H0, eight-parameter over common-population model (on [60, 76]) | $1.06$ | teal | C8 |
| `\WRatioWidthMOverFree` | 90% width of H0, shared-mass-width model over six-parameter model | $0.98$ | teal | C8 |
| `\WRatioWidthChiOverFree` | 90% width of H0, shared-spin-width model over six-parameter model | $0.98$ | teal | C8 |
| `\WRatioFreeDmuGOverFix` | 90% width of the mass-peak offset, free-reference over fixed-reference model | $1.39$ | teal | C8 |
| `\WRatioFreeDmuChiOverFix` | 90% width of the spin-mean offset, free-reference over fixed-reference model | $1.31$ | teal | C8 |
| `\HDFreeRhoHzeroMuG` | correlation of H0 and the reference mass-peak location, HDFree | $-0.22$ | teal | C8 |
| `\HDFreeRhoHzeroDmuG` | correlation of H0 and the AGN mass-peak offset, HDFree | $-0.09$ | teal | C8 |
| `\HDEightRhoHzeroMuG` | correlation of H0 and the reference mass-peak location, HDEight | $-0.23$ | teal | C8 |
| `\HDEightRhoHzeroDmuG` | correlation of H0 and the AGN mass-peak offset, HDEight | $0.01$ | teal | C8 |
| `\PinPZeroHzero` | H0 on the host-dependent mock, two catalogs, both offsets zero, H0 window [60, 76] (median and 90% equal-tailed interval) | $70.6^{+2.3}_{-1.7}$ | teal | C7,C8 |
| `\PinPOneHzero` | H0 on the host-dependent mock, two catalogs, offsets pinned at (Delta mu_G, Delta mu_chi) = (5.0, 0.115), H0 window [60, 76] (median and 90% equal-tailed interval) | $67.6^{+1.7}_{-2.1}$ | teal | C7,C8 |
| `\PinIZeroHzero` | H0 on the host-dependent mock, galaxy catalog for both host classes, both offsets zero (the posterior runs into the top of the window: a lower bound), H0 window [60, 76] (median and 90% equal-tailed interval) | $75.27^{+0.67}_{-1.60}$ | teal | C7,C8 |
| `\PinIOneHzero` | H0 on the host-dependent mock, galaxy catalog for both host classes, offsets pinned at (5.0, 0.115), H0 window [60, 76] (median and 90% equal-tailed interval) | $71.0^{+3.9}_{-3.3}$ | teal | C7,C8 |
| `\WRatioPinTwoOverGalOnly` | 90% width of H0 with both catalogs over that with the galaxy catalog for both host classes, offsets pinned (P1 over I1) | $0.53$ | teal | C8 |
| `\ShiftGalOnlyPinMinusTwo` | H0 median with the galaxy catalog for both host classes minus with both catalogs, offsets pinned (I1 minus P1) | $3.39$ | teal | C8 |
| `\ShiftCommonMinusPinned` | H0 median of the common-population model (on [60, 76]) minus the two-catalog model with offsets pinned (P1) | $3.01$ | teal | C7 |
| `\CtrlModHzero` | H0, galaxy-only control mock with the AGN-like population offset by +5 Msun and +0.10, offsets modelled (pinned at input) (median and 90% equal-tailed interval) | $68.8^{+5.8}_{-3.7}$ | teal | C7 |
| `\CtrlModHzeroMed` | median of H0, galaxy-only control mock with the AGN-like population offset by +5 Msun and +0.10, offsets modelled (pinned at input) | $68.8$ | teal | C7 |
| `\CtrlModHzeroLo` | lower end of the 90% interval of H0, galaxy-only control mock with the AGN-like population offset by +5 Msun and +0.10, offsets modelled (pinned at input) | $65.1$ | teal | C7 |
| `\CtrlModHzeroHi` | upper end of the 90% interval of H0, galaxy-only control mock with the AGN-like population offset by +5 Msun and +0.10, offsets modelled (pinned at input) | $74.6$ | teal | C7 |
| `\CtrlModfBranch` | fraction of mergers in the offset population, galaxy-only control mock with the AGN-like population offset by +5 Msun and +0.10, offsets modelled (pinned at input) (median and 90% equal-tailed interval) | $0.281^{+0.065}_{-0.064}$ | teal | C7 |
| `\CtrlUnmodHzero` | H0, same mock, offsets not modelled (common population); the posterior is cut by the top of the range at 100, so it is a lower bound (median and 90% equal-tailed interval) | $88.9^{+4.7}_{-4.2}$ | teal | C7 |
| `\CtrlUnmodHzeroMed` | median of H0, same mock, offsets not modelled (common population); the posterior is cut by the top of the range at 100, so it is a lower bound | $88.9$ | teal | C7 |
| `\CtrlUnmodHzeroLo` | lower end of the 90% interval of H0, same mock, offsets not modelled (common population); the posterior is cut by the top of the range at 100, so it is a lower bound | $84.7$ | teal | C7 |
| `\CtrlUnmodHzeroHi` | upper end of the 90% interval of H0, same mock, offsets not modelled (common population); the posterior is cut by the top of the range at 100, so it is a lower bound | $93.6$ | teal | C7 |
| `\CtrlSpinOnlyHzero` | H0, galaxy-only control mock with spin offset only, spin offset modelled (pinned at input) (median and 90% equal-tailed interval) | $67.4^{+2.8}_{-2.9}$ | teal | C7 |
| `\CtrlSpinOnlyfBranch` | fraction of mergers in the offset population, galaxy-only control mock with spin offset only, spin offset modelled (pinned at input) (median and 90% equal-tailed interval) | $0.221^{+0.092}_{-0.089}$ | teal | C7 |
| `\ShiftCtrlUnmodMinusMod` | H0 median with the offsets unmodelled minus modelled, galaxy-only control mock (a lower bound: the unmodelled posterior is cut by the top of the range at 100) | $20.1$ | teal | C7 |
| `\CtrlUnmodHzeroEdge` | posterior density at H0 = 100 relative to its peak, galaxy-only control with offsets unmodelled | $0.0046$ | teal | C7 |
| `\WRatioCtrlModOverSpinOnly` | 90% width of H0 with the mass offset present and modelled over that with only the spin offset (galaxy-only control mocks) | $1.69$ | teal | C7 |
| `\FixHSpatialfAGN` | f_AGN at fixed H0, host-dependent mock, common population (offsets zero) (median and 90% equal-tailed interval) | $0.295^{+0.082}_{-0.079}$ | teal | C8,C9 |
| `\FixHSpinfAGN` | f_AGN at fixed H0, host-dependent mock, spin offset only modelled (median and 90% equal-tailed interval) | $0.304^{+0.072}_{-0.067}$ | teal | C8,C9 |
| `\FixHSpinDmuChi` | spin-mean offset at fixed H0, host-dependent mock, spin offset only modelled (median and 90% equal-tailed interval) | $0.121^{+0.029}_{-0.026}$ | teal | C8,C9 |
| `\FixHMassfAGN` | f_AGN at fixed H0, host-dependent mock, mass offset only modelled (median and 90% equal-tailed interval) | $0.294^{+0.076}_{-0.071}$ | teal | C8 |
| `\FixHMassDmuG` | mass-peak offset at fixed H0, mass offset only modelled (median and 90% equal-tailed interval) | $4.52^{+1.09}_{-0.98}$ | teal | C8 |
| `\FixHBothfAGN` | f_AGN at fixed H0, both offsets modelled (median and 90% equal-tailed interval) | $0.276^{+0.068}_{-0.061}$ | teal | C8 |
| `\FixHBothDmuG` | mass-peak offset at fixed H0, both offsets modelled (median and 90% equal-tailed interval) | $4.75^{+1.03}_{-0.95}$ | teal | C8 |
| `\FixHBothDmuChi` | spin-mean offset at fixed H0, both offsets modelled (median and 90% equal-tailed interval) | $0.117^{+0.026}_{-0.024}$ | teal | C8 |
| `\FixHFreefAGN` | f_AGN, fixed H0, reference location and spin mean free (Analysis 11C) (median and 90% equal-tailed interval) | $0.274^{+0.076}_{-0.068}$ | teal | C8 |
| `\FixHFreeMuG` | reference mass-peak location mu_G, fixed H0, reference location and spin mean free (Analysis 11C) (median and 90% equal-tailed interval) | $35.56^{+0.61}_{-0.66}$ | teal | C8 |
| `\FixHFreeDmuG` | AGN mass-peak offset Delta mu_G, fixed H0, reference location and spin mean free (Analysis 11C) (median and 90% equal-tailed interval) | $3.9^{+1.4}_{-1.4}$ | teal | C8 |
| `\FixHFreeMuChi` | reference mean effective spin mu_chi, fixed H0, reference location and spin mean free (Analysis 11C) (median and 90% equal-tailed interval) | $-0.004^{+0.014}_{-0.015}$ | teal | C8 |
| `\FixHFreeDmuChi` | AGN spin-mean offset Delta mu_chi, fixed H0, reference location and spin mean free (Analysis 11C) (median and 90% equal-tailed interval) | $0.128^{+0.032}_{-0.032}$ | teal | C8 |
| `\LBBothOffsetsFixH` | log10 Bayes factor of the host-dependent model (both offsets free) against both offsets zero, fixed H0 = 67.74 (Savage-Dickey: arm S is the cube's zero-offset slice) (grid integration: numerical uncertainty below 0.005) | $29.5$ | teal | C8 |
| `\LBSpinOnlyFixH` | log10 Bayes factor, spin offset only against no offsets, fixed H0, host-dependent mock (grid integration: numerical uncertainty below 0.005) | $14.1$ | teal | C8 |
| `\LBMassOnlyFixH` | log10 Bayes factor, mass offset only against no offsets, fixed H0, host-dependent mock (grid integration: numerical uncertainty below 0.005) | $12.7$ | teal | C8 |
| `\LBBothVsSpinFixH` | log10 Bayes factor, both offsets against spin offset only, fixed H0 (grid integration: numerical uncertainty below 0.005) | $15.4$ | teal | C8 |
| `\LBBothVsMassFixH` | log10 Bayes factor, both offsets against mass offset only, fixed H0 (grid integration: numerical uncertainty below 0.005) | $16.7$ | teal | C8 |
| `\LBFreeRefFixH` | log10 Bayes factor, reference location and spin mean free against fixed at (35, 0), fixed H0 = 67.74 (11C against the A10 cube in the 11C prior box) | $-1.20\pm0.08$ | teal | C8 |
| `\LBOffsetsVsCommon` | log10 Bayes factor of the host-dependent model (offsets free, reference fixed) against the common population, H0 free (grid integration: numerical uncertainty below 0.005) | $28.3$ | teal | C7,C8 |
| `\LnZCommon` | ln evidence of the common-population model, H0 in [60,76] (grid integration; numerical uncertainty in values.json) | $-4357.49$ | teal | C7 |
| `\LnZFix` | ln evidence of the fixed-reference host-dependent model in the 11D box (grid integration) | $-4292.30$ | teal | C8 |
| `\LBFreeRefVsFix` | log10 Bayes factor, reference location and spin mean free against fixed, H0 free (11D against C10-J) | $-1.12\pm0.08$ | teal | C8 |
| `\LBFreeRefVsCommon` | log10 Bayes factor, six-parameter host-dependent model against the common population, H0 free | $27.2\pm0.08$ | teal | C7,C8 |
| `\LBWidthMVsFree` | log10 Bayes factor, shared mass-peak width free (flat on [1, 10] Msun) against fixed at 5 Msun | $-0.98\pm0.12$ | teal | C8 |
| `\LBWidthChiVsFree` | log10 Bayes factor, shared spin width free (flat on [0.01, 1]) against fixed at 0.1 | $-0.73\pm0.13$ | teal | C8 |
| `\LnZFree` | ln evidence (dynesty, merged runs) of LnZFree | $-4294.89\pm0.19$ | teal | C8 |
| `\LnZWidthM` | ln evidence (dynesty, merged runs) of LnZWidthM | $-4297.15\pm0.21$ | teal | C8 |
| `\LnZWidthChi` | ln evidence (dynesty, merged runs) of LnZWidthChi | $-4296.57\pm0.22$ | teal | C8 |
| `\LnZFreeFixH` | ln evidence (dynesty, merged runs) of LnZFreeFixH | $-4293.40\pm0.18$ | teal | C8 |
| `\LnZEight` | ln evidence of the eight-parameter model (one dynesty run, darksirens-core e7c3007, variance cap 20, Delta mu_chi box [-0.05, 0.30]) | $-4302.90\pm0.33$ | teal | C8 |
| `\LBCtrlModVsUnmod` | log10 Bayes factor, galaxy-only control mock: population offsets at their input values against no offsets, (H0, f) free, prior H0 [50,100] (grid integration: numerical uncertainty below 0.005) | $13.9$ | teal | C7 |
| `\SpinPosfAGN` | f_AGN at fixed H0 from positions only (spin offset not modelled), spin-offset mock (median and 90% equal-tailed interval) | $0.265^{+0.077}_{-0.074}$ | teal | C9 |
| `\SpinPosfAGNMed` | median of f_AGN at fixed H0 from positions only (spin offset not modelled), spin-offset mock | $0.265$ | teal | C9 |
| `\SpinPosfAGNLo` | lower end of the 90% interval of f_AGN at fixed H0 from positions only (spin offset not modelled), spin-offset mock | $0.191$ | teal | C9 |
| `\SpinPosfAGNHi` | upper end of the 90% interval of f_AGN at fixed H0 from positions only (spin offset not modelled), spin-offset mock | $0.341$ | teal | C9 |
| `\SpinBothfAGN` | f_AGN at fixed H0 with positions and the spin offset modelled (median and 90% equal-tailed interval) | $0.262^{+0.073}_{-0.069}$ | teal | C9 |
| `\SpinBothfAGNMed` | median of f_AGN at fixed H0 with positions and the spin offset modelled | $0.262$ | teal | C9 |
| `\SpinBothfAGNLo` | lower end of the 90% interval of f_AGN at fixed H0 with positions and the spin offset modelled | $0.192$ | teal | C9 |
| `\SpinBothfAGNHi` | upper end of the 90% interval of f_AGN at fixed H0 with positions and the spin offset modelled | $0.335$ | teal | C9 |
| `\SpinBothDmuChi` | spin-mean offset at fixed H0 with positions (two catalogs) and spins (median and 90% equal-tailed interval) | $0.107^{+0.035}_{-0.029}$ | teal | C9 |
| `\SpinBothDmuChiMed` | median of spin-mean offset at fixed H0 with positions (two catalogs) and spins | $0.107$ | teal | C9 |
| `\SpinBothDmuChiLo` | lower end of the 90% interval of spin-mean offset at fixed H0 with positions (two catalogs) and spins | $0.078$ | teal | C9 |
| `\SpinBothDmuChiHi` | upper end of the 90% interval of spin-mean offset at fixed H0 with positions (two catalogs) and spins | $0.142$ | teal | C9 |
| `\SpinIntfAGN` | f_AGN at fixed H0 from spins alone (one galaxy catalog in both slots: no positional host information) (median and 90% equal-tailed interval) | $0.31^{+0.53}_{-0.20}$ | teal | C9 |
| `\SpinIntDmuChi` | spin-mean offset at fixed H0 from spins alone (median and 90% equal-tailed interval) | $0.082^{+0.091}_{-0.052}$ | teal | C9 |
| `\SpinIntDmuChiMed` | median of spin-mean offset at fixed H0 from spins alone | $0.082$ | teal | C9 |
| `\SpinIntDmuChiLo` | lower end of the 90% interval of spin-mean offset at fixed H0 from spins alone | $0.029$ | teal | C9 |
| `\SpinIntDmuChiHi` | upper end of the 90% interval of spin-mean offset at fixed H0 from spins alone | $0.172$ | teal | C9 |
| `\SpinfWidthRatio` | 90% width of f_AGN with the spin offset modelled over without (positions only), fixed H0 | $0.952$ | teal | C9 |
| `\SpinfWidthRatioSixtyEight` | 68% width of f_AGN with the spin offset modelled over without, fixed H0 | $0.952$ | teal | C9 |
| `\SpinfWidthChange` | percentage change of the f_AGN 90% width when the spin offset is added, fixed H0 | $-4.8$ | teal | C9 |
| `\SpinDmuChiWidthRatio` | 90% width of the spin-mean offset from spins alone over that with the catalogs, fixed H0 | $2.2$ | teal | C9 |
| `\SpinDmuChiWidthRatioSixtyEight` | same with 68% widths | $2.4$ | teal | C9 |
| `\SpinIntfWidthRatio` | 90% width of f_AGN from spins alone over that from positions alone, fixed H0 | $4.8$ | teal | C9 |
| `\SpinBothRho` | correlation of f_AGN and the spin-mean offset with positions and spins, fixed H0 | $-0.55$ | teal | C9 |
| `\LBSpinFixH` | log10 Bayes factor of a spin-mean offset against none (Savage-Dickey at Delta mu_chi = 0, prior flat on [-0.2, 0.25]), fixed H0, with the catalogs | $8.54$ | teal | C9 |
| `\LBSpinIntFixH` | log10 Bayes factor of a spin-mean offset against none (Savage-Dickey at Delta mu_chi = 0, prior flat on [-0.2, 0.25]), fixed H0, from spins alone | $3.97$ | teal | C9 |
| `\SpinHPosHzero` | H0 from positions only (spin offset not modelled), spin-offset mock, H0 free (window [63, 76]) (median and 90% equal-tailed interval) | $69.1^{+1.6}_{-1.6}$ | teal | C9 |
| `\SpinHPosHzeroMed` | median of H0 from positions only (spin offset not modelled), spin-offset mock, H0 free (window [63, 76]) | $69.1$ | teal | C9 |
| `\SpinHPosHzeroLo` | lower end of the 90% interval of H0 from positions only (spin offset not modelled), spin-offset mock, H0 free (window [63, 76]) | $67.5$ | teal | C9 |
| `\SpinHPosHzeroHi` | upper end of the 90% interval of H0 from positions only (spin offset not modelled), spin-offset mock, H0 free (window [63, 76]) | $70.7$ | teal | C9 |
| `\SpinHBothHzero` | H0 with positions and the spin offset modelled, spin-offset mock (median and 90% equal-tailed interval) | $69.1^{+1.5}_{-1.4}$ | teal | C9 |
| `\SpinHBothHzeroMed` | median of H0 with positions and the spin offset modelled, spin-offset mock | $69.1$ | teal | C9 |
| `\SpinHBothHzeroLo` | lower end of the 90% interval of H0 with positions and the spin offset modelled, spin-offset mock | $67.6$ | teal | C9 |
| `\SpinHBothHzeroHi` | upper end of the 90% interval of H0 with positions and the spin offset modelled, spin-offset mock | $70.6$ | teal | C9 |
| `\SpinHPosfAGN` | f_AGN from positions only, H0 free (median and 90% equal-tailed interval) | $0.270^{+0.078}_{-0.075}$ | teal | C9 |
| `\SpinHBothfAGN` | f_AGN with positions and spins, H0 free (median and 90% equal-tailed interval) | $0.264^{+0.074}_{-0.070}$ | teal | C9 |
| `\SpinHBothDmuChi` | spin-mean offset with positions and spins, H0 free (median and 90% equal-tailed interval) | $0.108^{+0.035}_{-0.029}$ | teal | C9 |
| `\SpinHzeroWidthRatio` | 90% width of H0 with the spin offset modelled over without, both on H0 in [63, 76] | $0.902$ | teal | C9 |
| `\SpinHzeroWidthRatioSixtyEight` | same with 68% widths | $0.885$ | teal | C9 |
| `\LBSpinHfree` | log10 Bayes factor of a spin-mean offset against none (Savage-Dickey at Delta mu_chi = 0), H0 free | $8.62$ | teal | C9 |
| `\NullfAGNUpper` | 90% upper limit on f_AGN from the combined posterior of simulated catalogs with no AGN-hosted mergers (input f_AGN = 0) | $\ph{0.0X}$ | red | validation |
| `\NullHzeroComb` | combined H0 posterior from those catalogs | $\ph{XX.X^{+X.X}_{-X.X}}$ | red | validation |
| `\NullHzeroLevel` | credible level of the input H0 in that combined posterior | $\ph{XX}$ | red | validation |
| `\NullLBfZero` | log10 Bayes factor of the two-population model against galaxy hosts only on those catalogs (Savage-Dickey at f_AGN = 0) | $\ph{-X.X}$ | red | validation |
| `\NullNcat` | number of simulated catalogs with no AGN-hosted mergers | $\ph{XX}$ | red | validation |
| `\FluxNcat` | simulated catalogs analysed with flux limits (only catalog A exists for the anchored-density grids) | $\ph{XX}$ | red | C4,validation |
| `\FluxfCombEighteen` | combined f_AGN posterior at m < 18 over several catalogs, densities at input | $\ph{0.XX^{+0.XX}_{-0.XX}}$ | red | C4,validation |
| `\FluxHzeroCombEighteen` | combined H0 posterior at m < 18 over several catalogs | $\ph{XX.X^{+X.X}_{-X.X}}$ | red | C4,validation |
| `\FluxfCombEighteenLevel` | credible level of the input f_AGN in that combined posterior | $\ph{XX}$ | red | C4,validation |
| `\FluxHzeroCombEighteenLevel` | credible level of the input H0 in that combined posterior | $\ph{XX}$ | red | C4,validation |
| `\FreeLognGalEighteenComb` | combined posterior of log10 n_gal at m < 18 over the three catalogs, both densities free | $\ph{-X.XX^{+0.XX}_{-0.XX}}$ | red | C6,validation |
| `\FreeLognAGNEighteenComb` | combined posterior of log10 n_AGN at m < 18 over the three catalogs | $\ph{-X.XX^{+0.XX}_{-0.XX}}$ | red | C6,validation |
| `\FreefAGNEighteenComb` | combined posterior of f_AGN at m < 18 over the three catalogs, both densities free | $\ph{0.XX^{+0.XX}_{-0.XX}}$ | red | C6,validation |
| `\FreeHzeroEighteenComb` | combined posterior of H0 at m < 18 over the three catalogs, both densities free | $\ph{XX.X^{+X.X}_{-X.X}}$ | red | C6,validation |
| `\FreeLognGalEighteenCombLevel` | credible level of the input log10 n_gal = -3 in that combined posterior | $\ph{XX}$ | red | C6,validation |
| `\HDNcat` | simulated catalogs with host-dependent masses and spins (one exists) | $\ph{XX}$ | red | C7,C8,validation |
| `\HDEightHzeroComb` | combined H0 posterior of the eight-parameter model over several host-dependent catalogs | $\ph{XX.X^{+X.X}_{-X.X}}$ | red | C8,validation |
| `\HDEightHzeroCombLevel` | credible level of the input H0 in that combined posterior | $\ph{XX}$ | red | C8,validation |
| `\HDEightDmuGCombLevel` | credible level of the input mass-peak offset (+5 Msun) in the combined posterior | $\ph{XX}$ | red | C8,validation |
| `\HDEightDmuChiCombLevel` | credible level of the input spin-mean offset (+0.10) in the combined posterior | $\ph{XX}$ | red | C8,validation |
| `\CommonHzeroComb` | combined H0 posterior of the common-population model over several host-dependent catalogs (size of the bias) | $\ph{XX.X^{+X.X}_{-X.X}}$ | red | C7,validation |
| `\LBEightVsFree` | log10 Bayes factor of the eight-parameter model against the six-parameter model (widths fixed) | $\ph{X.X\pm X.X}$ | red | C8 |
| `\LBEightVsCommon` | log10 Bayes factor of the eight-parameter model against the common population | $\ph{X.X\pm X.X}$ | red | C7,C8 |
