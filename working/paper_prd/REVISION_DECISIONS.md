REVISION 2 OF THE PRD DRAFT (decisions of 17 Nov 2026, binding for the reviser and the figure and numbers work).
The independent review is REVIEW.md. Apply every must-fix and should-fix item, and optional items where
cheap, EXCEPT where this file decides otherwise. Where this file and notes/DESIGN.md differ, this file wins.
Keep the previous version as main_v1.tex and main_v1.pdf before overwriting.

0. WHAT CHANGES IN SUBSTANCE. The review is right on its three main points (B1 and B2, B3 and B8, B4).
   The exploratory runs cannot carry the paper's headline numbers. The paper's structure stays; its lead
   comparison, its statement of what limits f_AGN, and every number that depends on the Monte Carlo accuracy
   become placeholders tied to the production runs (runs.txt, revised with this file). Nothing is claimed in
   prose that a red placeholder could overturn (C11).

1. LEAD COMPARISON (B1, B5, B13, B14, D5, G4). The paper's two models are:
   "single population": one mass and spin distribution for both host classes, with its mass-peak location and
   width and its spin mean and width inferred (H0, f_AGN and four population parameters);
   "host-dependent populations": the same plus an offset of the mass peak and of the spin mean for the AGN class
   (eight parameters). Same prior for both, H0 uniform on [50, 100], same code, several realizations, combined.
   The abstract, Sec. IV B, Fig. 4 and Table III lead with that pair and with their Bayes factor.
   The exploratory numbers (population held at the galaxy-class inputs; offsets with the reference fixed) are
   kept in one sentence and one table row each, named for what they are. Remove the two comparisons with a
   model fixed at the input (log10 B of a free against a fixed reference; the "offsets fixed" rows and the
   solid curve of Fig. 4(b)). Replace "the host catalogs do not protect it" by "reduce the bias and do not
   remove it". The eight-parameter model on one realization is reported with the credible level of each input
   and the anticorrelation of spin offset and spin width; "recovers" waits for the combined posterior.

2. THE MASS SCALE (B2, B6, B7). Say in Sec. III B that the fraction of mergers in the Gaussian peak (0.90) is a
   design choice that makes the peak a sharp ruler, with the value found for the observed mergers beside it
   (verify the number in the source before quoting). Add the analysis with no catalog (hosts uniform in comoving
   volume) as the baseline in Sec. IV A: a row of Table II and a curve of Fig. 3(b); it is a red placeholder
   (run R11). State the factor between AGN-hosted and galaxy-hosted H0 widths against that baseline, and write
   the reading of the galaxy-catalog-only result as one of two (shared density field, or the mass scale) until
   R11 separates them. The scaling of the bias with the peak fraction is given as the estimate of B2 (mean-mass
   shift times (1+z)/z), stated as an estimate. A second mock family with a smaller peak fraction is an option
   for the owner in the production plan, not a commitment of the text.

3. WHAT LIMITS f_AGN (B3, B8, B10, G6, G7). Production convention: hosts are normalized over the volume the
   analysis uses (redshift grid ending just above the reach of the detected injections, z = 1.10), and each
   class's number density is taken from its catalog counts where the catalog is complete. The result of
   Sec. IV C is Eq. (7): the mergers measure the merger rate per host of the AGN class relative to the galaxy
   class. Its consequence, Eq. (8), is written with F_AGN as a function of depth, with the measured response
   against the prediction in Fig. 6 (Analysis 15 supplies the first numbers; the rest are red). Delete "What
   limits the measurement is not the depth of the catalogs" and the abstract's 0.13 unless it survives under
   the production convention; say for which volume any such number holds. Name each flux limit by its
   magnitude and by the redshift to which the catalog is complete; give the completeness weighted by the
   detected mergers. Free densities: rewrite as B8 says (prior-induced densities drawn in Fig. 7; 90% upper
   limits with the induced limit beside them; the two deepest limits out of Table IV's last columns; f_AGN
   follows the ratio n_AGN / n_gal). The paper reports the rate per host ratio as a number wherever it reports
   f_AGN under an assumed density.

4. MONTE CARLO ACCURACY (B4, C2, C6, D6). The statement of accuracy lives in Appendix A and is written from
   Analysis 14's summary when it exists. Until the production runs: the factor between AGN-hosted and
   galaxy-hosted widths, the credible level of the input in the 93-set combination, and any sentence that
   f_AGN is recovered carry the measured Monte Carlo effect beside them or are placeholders. No digit beyond
   what the grid and the estimator support (67.741 goes; Bayes factors to two significant figures at most).

5. STATISTICS (part D): apply all. Every Bayes factor is quoted with the prior range it depends on, once, in
   the table that holds it. No counts over realizations; combined posteriors and credible levels instead.

6. NAMES (E4 to E6, C12). The unit of the simulation is a "realization" (nine realizations, realization A);
   "catalog" means only the galaxy catalog and the AGN catalog. "Single population" and "host-dependent
   populations" for the two models; "galaxy-class population" for the reference. "The fraction of mergers hosted
   by AGN". No "horizon": say "the redshift of the most distant detected merger" once, then the number.
   r for the correlation coefficient; rho stays the signal-to-noise ratio.

7. ABSTRACT (E1): rewrite on the reviewer's shape (seven sentences, about a dozen numbers, no context opener,
   each number with its condition and credible level). Its lead sentence is the pair of item 1, as placeholders
   until the production runs.

8. LENGTH (E2, E11): 14 to 15 pages. Drop the corner figure and Appendix C; Fig. 8 panels (c) and (d);
   Fig. 6 panels (a) and (c) (their content becomes one sentence with numbers and the table); Fig. 5 panel (a);
   Fig. 4 panel (b); the column of Table II and Eq. (6) the review names; about fifteen references; the list of
   limitations in the Conclusions becomes two sentences. Reference titles stay off (journal style).
   Remove "Drawn from the exploratory runs." from the captions; the colour of the numbers and one sentence in
   the header comment of main.tex carry that information.

9. PRIOR WORK (part F): apply all. Open the PDF of every paper quoted before a quotation is kept or added (the
   reviewer's quotations came through a summarizing fetch). Cite arXiv:2607.03672 and arXiv:2508.18704 where
   the review says. For the fifth-catalog H0 give the value and interval that can be opened and verified, with
   its credible level, or no interval.

10. FIGURES (part G), for the figure script: G1 to G9 with the cuts of item 8. Fig. 3(b) gains the no-catalog
    baseline when R11 exists (leave the slot). Fig. 4 becomes single population against host-dependent
    populations on [50, 100] (exploratory curves kept, labelled by what they hold fixed, until production).
    Fig. 6: the response of f_AGN and of H0 to the assumed AGN number density with the prediction of Eq. (8) as
    black curves, under both normalization volumes when Analysis 15 is in. Fig. 7: prior-induced densities, and
    the curve of constant rate-per-host ratio drawn through the (f_AGN, n_AGN) regions.

11. NUMBERS. New macros for everything in notes/MACRO_REQUESTS.md that can be computed from disk, the plain
    design values of Table I and Appendix B, the Analysis 14 and 15 summaries, the rate-per-host ratio and its
    interval for each analysis that has f_AGN, F_gal and F_AGN per configuration, the completeness weighted by
    the detected mergers, and hosts per localization volume for each class. Table III's f_AGN entry for the
    galaxy-hosts-only mock is relabelled as the fraction of the heavier population (C1).

12. NOT APPLIED. None of the review's items is rejected. Items that need runs are placeholders, listed in
    placeholders.txt with their run.
