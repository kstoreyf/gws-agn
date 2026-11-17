# Brief, version 2: the short paper as a proper letter (17 Nov 2026)

The owner read the first draft written from `BRIEF.md` and rejected it: "you really call that a
paper? what is that abstract even? you think this is publication ready?" and: "From the literature
audit. the novel thing is proper multi-tracer darksirens, and correlated gw properties with
galaxies". The first brief was wrong in form (question headings throughout, a 130-word abstract of
clipped numbers that opened on context and closed on a slogan, no Results or Validation section,
1850 words) and in claim (it was built around an aphorism, not around what is new). This brief
replaces it. `BRIEF.md` and `main_v1.tex` are kept for the record; do not open them.

## 1. Form: the owner's style skill governs, and this brief does not override it
Load the `astro-paper-style` skill (Skill tool; else `~/.claude/skills/astro-paper-style/SKILL.md`)
and follow it to the letter: its architecture (abstract of 6 to 8 sentences in its stated order;
introduction of about six paragraphs and at most 1100 words with no headline numbers; a short
method built on one master equation with the familiar analyses as its limits; the simulated data
described as physics consequences of the likelihood; results that open with a credibility anchor;
Validation as its own section; discussion and conclusions on its closing template), its
sentence-level rules, its number rules, its figure and caption rules, its banned vocabulary and
its self-check. Section titles are nouns. At most one heading may be a question. Read
`references/style_2407.02460.md` in the skill's directory (the owner's own paper, whose voice
wins) and `references/vocabulary_gwcosmo.md` before you write, and
`../paper_prd/notes/OWNER_STYLE_DETAILS.md` for the owner's recent words.
Where the skill mentions coverage counts or realization counts, the owner's later rule wins:
Bayesian statements only (posteriors, Bayes factors with prior ranges, and the credible level at
which an input lies in the posterior combined over simulated realizations). 90% credible
intervals.
Target: a letter. REVTeX 4.2 two columns (`aps,prd,reprint`). Main text about 3500 words;
abstract at most 200 words; five figures and tables combined (four figures and one table);
appendices outside the count. A complete paper: a reader who has never seen the project must be
able to follow it from the title to the last sentence.

## 2. What is new (from the literature audit; this is what the paper claims and is titled for)
1. **A proper multi-tracer dark siren analysis.** Two catalogs, galaxies and active galactic
   nuclei (AGN), trace the same large-scale structure; each merger is hosted by one of them; each
   tracer enters the likelihood with its own catalog, its own number density and completeness, and
   the selection effects of the gravitational-wave detectors; H0 is inferred with the fraction of
   mergers hosted by each tracer. Earlier work has parts of this and not the whole
   (`../paper_prd/notes/PRIOR_WORK_VERIFIED.md`, the only source for statements about other
   papers; use its safe sentences): H0 with the fraction of mergers in AGN from an AGN catalog with
   an unclustered background for the other mergers; H0 with a mixing fraction of two weightings of
   one galaxy catalog; the AGN fraction at fixed cosmology. Do not write "first" for the joint
   measurement of H0 and the AGN fraction and do not use the verb "unify".
2. **Merger properties that correlate with the host.** The mass and spin distributions of the
   mergers may differ between the tracers and are inferred with H0 and the catalogs. Several of
   the earlier analyses name this as future work and none does it. The physical case: black holes
   that merge in AGN disks are expected to be heavier and to spin faster. The likelihood is
   general: any two classes of host whose mergers differ (galaxy type, stellar mass) fit it.
The title says this plainly, with "multi-tracer" in it and no aphorism. Candidates to improve on:
"Multi-tracer dark sirens: the Hubble constant from binary black hole mergers whose masses and
spins depend on their hosts"; "Multi-tracer dark siren cosmology with host-dependent binary black
hole populations".

## 3. What the paper shows (results; red numbers are expectations until the runs finish)
In the order the skill's architecture asks for.
- **Credibility anchor (Results opens with it).** Two tracers, one population common to both, the
  population held at its input: H0 and the fraction of mergers in AGN are recovered together.
  The familiar single-catalog analysis is the limit with no mergers in AGN.
- **What each tracer contributes.** A localization volume holds about 1500 galaxies and about 16
  AGN, so a merger in an AGN matched to the AGN catalog constrains H0 far more than a merger in a
  galaxy, whose catalog adds little to what the mass spectrum already gives. The H0 half-width
  from 1000 mergers falls steeply with the fraction of them in AGN (0, 0.1, 0.3, 0.5, 1); the
  likelihood finds the AGN-hosted mergers statistically and loses part of the information that
  known hosts would give.
- **Populations that depend on the host (the headline).** With AGN-hosted mergers heavier by
  5 Msun at the mass peak and with mean effective spin higher by 0.10: an analysis with one
  population for both tracers, fitted to the data, is biased in H0; the analysis with a
  population for each tracer recovers H0 and the AGN fraction and measures both differences; the
  Bayes factor between them; masses and spins raise the probability that a merger truly in an AGN
  is assigned to one, which moves the H0 precision toward the known-hosts value. The same for
  differences twice as large.
- **Validation (own section).** Recovery of the inputs in the posterior combined over
  realizations; mergers with no AGN hosts; a mass spectrum like the observed one in place of the
  sharp one; the Monte Carlo error carried by the quoted intervals
  (`../analyses/analysis_14_monte_carlo_accuracy/REPORT.md`).
- **What the measurement rests on (Discussion).** The AGN catalog must be complete over the volume
  of the detected mergers for H0; the fraction of mergers in AGN also needs the AGN abundance,
  because the mergers constrain the merger rate per AGN relative to that per galaxy
  (`../analyses/analysis_15_host_normalisation/REPORT.md`, final numbers); the size of the
  population differences is not known; the mass spectrum's form.
What stays out or gets one clause: the series of flux limits, inferred number densities, the
no-catalog series. A longer companion paper carries them ("a companion paper", no reference).

## 4. Figures and table (drawn by someone else; include with `\IfFileExists`; captions per the skill)
- Fig. 1 `figures/fig_concept` (final): a slice of the simulated catalogs with one merger's 90%
  localization region (225 galaxies, 6 AGN, true host marked) and the mass and effective-spin
  distributions of galaxy-hosted and AGN-hosted mergers. Facts for the caption are in
  `notes/MACROS_NEEDED.md` (the `\IndConcept...` macros).
- Fig. 2 `figures/fig_anchor` (to be drawn): joint posterior of H0 and the AGN fraction for equal
  populations, one realization and the combination, input marked.
- Fig. 3 `figures/fig1_budget`: 90% half-width of H0 from 1000 mergers against the fraction of
  mergers hosted by AGN, for labels unknown, a population for each tracer (two sizes of the
  difference), hosts known; lines for the mass spectrum alone and the galaxy catalog alone; second
  panel for the observed-like spectrum.
- Fig. 4 `figures/fig3_bias` with `figures/fig2_labels` as a third panel (to be merged into one
  file `figures/fig_populations`; until then include the two files): H0 posterior combined over
  realizations with one population and with a population for each tracer; joint posterior of the
  two population differences; probability of being AGN-hosted for mergers truly in galaxies and
  truly in AGN, from positions alone and with masses and spins.
- Table I: results by analysis, rows named in words, columns H0, AGN fraction, the two population
  differences, log10 Bayes factor with its reference named in the caption.

## 5. Facts and numbers
- The simulation and the expected results: Sec. 6 of `BRIEF.md` is the one part of that file you
  may read (lines under the heading "## 6. Facts and numbers" only). Design values are macros in
  `values/MACROS.md` (search it; do not read it whole).
- The result macros of this paper already exist: `values_independent.tex` (red `\ph{expected}`)
  with meanings in `notes/MACROS_NEEDED.md`, and `values_independent_results.tex` receives the
  real values as runs finish (`notes/STATUS.md` says which are real). Reuse these names; add new
  ones the same way and list them in `notes/MACROS_NEEDED.md`. Every number is a macro. Put
  `% EXPECTED` at the end of each sentence whose wording depends on how a run comes out. The word
  "placeholder" does not appear in the paper.
- Equations: `../paper_prd/notes/LIKELIHOOD_MATH.md`.
- Bias is always quoted beside the 90% half-width of one catalog of 1000 mergers; a posterior
  combined over realizations is named as such where it is used.

## 6. May and may not read
May: this brief; the skill and its references; `../paper_prd/notes/OWNER_STYLE_DETAILS.md`,
`LIKELIHOOD_MATH.md`, `PRIOR_WORK_VERIFIED.md`; `refs.bib`; `values/MACROS.md` (search) and
`values/results_macros.tex`; `values_independent*.tex`, `notes/MACROS_NEEDED.md`,
`notes/STATUS.md`; the figure PNGs in `figures/`; `EXPERIMENT.md`; the two analysis reports named
above; `working/data/README.md` for facts about the simulation.
May not: any `main*.tex` here or elsewhere, `../paper_short/`, any skeleton, design, review or
revision file, `working/archive/`.

## 7. Before you report
Run the skill's build gates and self-check. Word-count each section and count display equations
per section (the skill asks for both). Count the abstract from the rendered PDF. Report: title;
abstract as rendered; section list with word counts; page count; macros added; `% EXPECTED`
count; the self-check items you could not pass and why.
