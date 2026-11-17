# The short paper (second, independent draft; skeleton of 17 Nov 2026)

The owner's request, in their words: "The second draft you should start and maintain is the punchy
paper. short to the point multi-tracer cosmology correlated to bbh properties. Start from scratch.
Imagine the quick paper that would be useful to the gw cosmo community. maybe the one we have still
sounds like i did a, b, c, d, i stopped. Appendices on that can absorb the detail."

This draft is written from scratch and independently of the long draft in `../paper_prd/main.tex`
(do not open that file or `main_v1.tex`). It shares the numbers (`values/`, a link to
`../paper_prd/values/`), the bibliography (`refs.bib`, a link) and the marking: every result that
waits for a production run is a red placeholder macro, and figure data that are invented or
exploratory are drawn in red. The owner's style note (`../paper_prd/notes/OWNER_STYLE_DETAILS.md`)
is binding here too.

## 1. The one idea
Galaxies and AGN are two tracers of one density field, and the black holes that merge in them need
not be the same black holes. A dark siren analysis that gives every merger the same mass and spin
distribution reads the heavier AGN-hosted mergers as more distant and gets H0 wrong. A likelihood
with two tracers, each with its own merger population, removes the bias and uses the correlation:
masses and spins tell the hosts apart, the hosts tell the populations apart, and H0, the fraction
of mergers hosted by AGN and the population differences are measured together.

The reader is someone who runs or reads catalog-based H0 analyses of the gravitational-wave
transient catalogs. After two pages they should know: how large the bias is, what removes it, what
they must change in their analysis, and what the catalogs have to supply.

## 2. Claims, in the order the paper makes them (numbers are red macros; expected values in
`../paper_prd/notes/PAPER_SKELETON.md` Sec. 7)
1. The bias. With 1000 mergers of which 0.30 are hosted by AGN, AGN-hosted mergers heavier by
   5 Msun at the mass peak and with mean effective spin higher by 0.10, an analysis with one
   population is biased in H0 (`\HDBiasHzero`, `\HDSingleHzeroComb`); state the credible level at
   which the input lies. For a mass spectrum like the observed one the bias is smaller
   (`\ObsBiasHzero`).
2. The cure. The two-tracer analysis with host-dependent populations recovers H0
   (`\HDHostHzeroComb`), measures f_AGN (`\HDHostfAGNComb`) and both offsets (`\HDHostDmuGComb`,
   `\HDHostDmuChiComb`), and is preferred (`\HDLB`, against the single-population analysis of the
   same mergers, with the prior ranges).
3. Where the information is. An AGN-hosted merger has far fewer possible hosts in its localization
   volume than a galaxy-hosted one (`\PureAGNHostsPerVol` against `\PureGalHostsPerVol`), so it
   measures H0 more tightly (`\WidthRatioAGNGal`). With no catalog at all the mass scale still
   measures H0 when the population is known and much less when it is inferred (`\KnownNoneHzero`,
   `\SingleNoneHzero`); the catalogs supply the rest (`\SingleBothHzero`).
4. What the catalogs must supply. The mergers constrain the merger rate per host of one tracer
   relative to the other (`\RateRatio...`); f_AGN follows only with the ratio of host number
   densities. Flux-limited catalogs are enough when each density is measured where its catalog is
   complete; a factor of two error in the assumed AGN density moves f_AGN by the amount the
   identity predicts (Analysis 15 macros).
5. Generality, one sentence in the outlook: the same likelihood applies to any two host classes
   whose mergers differ (early and late types, stellar mass, metallicity).

## 3. Shape (REVTeX 4.2 two-column; main text at most 3500 words, 4 figures, 1 table; about 6 pages
of main text and 3 to 4 of appendices)
Title: say what is new in the reader's words; "multi-tracer" belongs in it. Working title:
"Multi-tracer dark sirens: the Hubble constant when black hole masses and spins depend on their
hosts".
Abstract: at most 150 words, 5 or 6 sentences: the assumption everyone makes; the bias when it
fails, with its number, the number of mergers and the assumed offsets in the same sentence; the
two-tracer likelihood in one clause; what it recovers and measures; what the catalogs must supply;
the implication for analyses of the coming observing runs.
I. Introduction: four paragraphs at most. No headline numbers. Last paragraph: cosmology, priors,
   90% credible intervals, where the detail is (appendices).
II. Two tracers, one likelihood. One displayed equation for the mixture over host classes with a
   population per class, and one for the rate-per-host identity. The usual analyses as limits, one
   sentence each (one catalog; no catalog; equal populations; population held fixed). Everything
   else in Appendix A.
III. Simulated mergers. One paragraph and Fig. 1; the design values in Appendix B.
IV. Results, four short subsections titled by their physics:
   A. One population biases H0 (Fig. 2a).
   B. Masses and spins tell the hosts apart (Fig. 2b, 2c; claim 2).
   C. What each assumption is worth (Fig. 3; claims 1 and 3 across known population, inferred
      population, host-dependent populations, with and without catalogs, and the observed-like
      spectrum).
   D. What the catalogs must supply (Fig. 4; claim 4).
V. Outlook: one or two paragraphs: what to change in catalog analyses now; the number of mergers and
   the AGN catalogs at which this matters; claim 5; the last sentence hands the result on.
Table I (the only table in the main text): results by analysis, at most eight rows named in words,
   columns H0, f_AGN, the two offsets, log10 B with its reference named in the caption.
Appendices (they absorb the detail; each is self-contained and short):
   A. The likelihood, its limits and the normalization identity.
   B. Simulated catalogs and mergers (design table; `figA1_mocks`).
   C. Flux-limited catalogs and inferred number densities (table; `figA2_flux_limits`).
   D. Recovery of the inputs: posteriors combined over realizations, mergers with no AGN hosts and
      with one in ten (`figA3_validation`).
   E. Accuracy of the Monte Carlo sums (Analysis 14 macros).
No Limitations section; each caveat closes the claim it limits, with a size. No record of how the
work was done, anywhere.

## 4. Figures (file names fixed; stand-ins copied from the long draft are in `figures/` now and
will be replaced by figures drawn for this paper)
- `fig1_concept` (final, to be drawn): (a) a slice of the simulated density field with the galaxy
  catalog and the AGN catalog and one merger's localization region; (b) the mass and (c) the
  effective-spin distributions of galaxy-hosted and AGN-hosted mergers.
- `fig2_headline`: host-dependent mergers. (a) H0 posterior of the single-population analysis and
  of the host-dependent analysis, combined over realizations, input marked; (b) joint posterior of
  the two AGN offsets with the input and zero marked; (c) f_AGN posterior of the two analyses.
- `fig3_assumptions`: one row per analysis, named in words, with the 90% interval of H0 and of
  f_AGN and the inputs as vertical lines; about ten rows, grouped by what the mergers are.
- `fig4_rate_per_host`: (a) joint posterior of f_AGN and log10 n_AGN with the number densities
  inferred, the curve of constant rate per host through the input, and the region the priors allow;
  (b) f_AGN against the assumed AGN number density with the prediction of the identity.
- Appendix: `figA1_mocks`, `figA2_flux_limits`, `figA3_validation`.

## 5. Names and macros
Names as in `../paper_prd/notes/PAPER_SKELETON.md` Sec. 2 (realization; the galaxy catalog, the AGN
catalog; equal and host-dependent populations; peak-dominated and observed-like spectra; population
known, single population, host-dependent populations), plus "tracer" for a host class with its
catalog, introduced once. Macros: the scheme of that file's Sec. 6 and the names in
`values/MACROS.md`. Facts about each figure: `../paper_prd/figures/CAPTION_FACTS.md` (the stand-ins
are its Figs. 6, 8, 4, 1, 3, 9).
