# Brief for a short paper, written in a clean room (17 Nov 2026)

You are writing a short paper from nothing. There are other drafts in this project. You must not
open them, their outlines, their reviews or their figure notes, because the owner's complaint about
the last short draft was that it summarized the long one: "it sounds the same", "tell a story to the
point", "don't summarize previous drafts". Everything you need is in this file and in the few files
it names. If you find yourself listing what was done, stop and go back to the argument.

## 1. The reader and the four minutes
The reader runs or reads the dark siren analyses of the gravitational-wave transient catalogs. They
know the method, its jargon and its weak point: with thousands of galaxies in every localization
volume the catalog adds little, and the Hubble constant comes mostly from the mass spectrum. They
will give this paper four minutes: title, abstract, three figures, last paragraph. In that time they
must learn one thing they did not know and can use.

## 2. The thing they learn
**A dark siren is as good as its hosts are rare, and the black holes can say which hosts are theirs.**

The argument, each line a claim with its evidence (red numbers are expectations; Sec. 6):
1. The information a merger carries about H0 is set by how many possible hosts lie in its
   localization volume. In our simulated catalogs that is about 1500 galaxies and about 16 active
   galactic nuclei (AGN). A merger hosted by an AGN and matched to an AGN catalog sits between a dark
   siren and a bright one: 1000 such mergers measure H0 several times better than 1000 mergers in
   galaxies matched to a galaxy catalog, which do little better than the mass spectrum alone.
2. So if only some mergers happen in AGN, those few carry most of the measurement, provided we knew
   which they are. We do not. A likelihood that treats galaxies and AGN as two tracers of the same
   large-scale structure, with the fraction of mergers in AGN as a parameter, finds them
   statistically, and loses part of the information by not knowing the labels. How much it loses,
   and how the total depends on the fraction of mergers in AGN, is Fig. 1.
3. Black holes that merge in AGN disks are expected to be heavier and to spin faster. Then masses
   and spins are labels: with a mass and spin distribution for each tracer, the same likelihood
   separates the two kinds of merger better than positions alone (Fig. 2), and H0 moves from the
   no-label curve toward the known-label curve (Fig. 1).
4. The same correlation hurts an analysis that ignores it. Give every merger one mass and spin
   distribution, even a flexible one fitted to the data, and H0 is biased; give each tracer its own
   and the bias is gone, with both population differences measured (Fig. 3).
5. What it takes: an AGN catalog complete over the volume of the detected mergers, and nothing
   else for H0. The fraction of mergers in AGN needs one more input, the abundance of AGN, because
   what the mergers constrain is the merger rate per AGN relative to the rate per galaxy.

That is the paper. It is an argument about where the information is, not a list of analyses.

## 3. What is deliberately not in it
Flux-limited catalogs as a series; number densities inferred; the mass spectrum inferred without
catalogs as a series; a table of every analysis; a validation section; a discussion of numerical
accuracy in the main text; a list of caveats. Each of these gets at most one clause with one number
where the argument needs it (for example, at the shallowest flux limit an AGN abundance wrong by a
factor of two moves H0 by about 1 km/s/Mpc). A longer companion paper carries them; cite it once as
"a companion paper" without a reference number.

## 4. Form
- REVTeX 4.2, two columns (`aps,prd,reprint`). Main text at most 2500 words. Three figures. No
  table in the main text. End matter of at most one and a half pages: (A) the likelihood in full,
  (B) the simulated catalogs and mergers with their design values, (C) two or three sentences on
  how the inputs are recovered and on the accuracy of the Monte Carlo sums.
- No section called Results. Section headings are the reader's questions, in the order of Sec. 2,
  for example: "How much is a host catalog worth?", "Which mergers are in AGN?", "Can the black
  holes tell us?", "What does it take?". The first two sentences under each heading answer it with
  a number. (The one text of this kind that the owner has called "excellent" used the question as
  its heading, put each number beside the number it is compared with, stated the claim in a short
  sentence after the evidence, and stated the limit of the claim at once.)
- Method in the main text: one displayed equation (the rate of mergers as a sum over the two
  tracers, each with its host distribution and its population) and a few sentences. The familiar
  analyses are its limits, said in passing where each is used, not listed.
- Write the three figure captions first, then the first two sentences of each section, then the
  rest, then the abstract last. Each caption opens with what the figure shows and closes with the
  one sentence the reader should take away.
- Abstract: at most 130 words. What limits dark sirens; the idea; the two or three numbers that
  carry it, each with its condition in the same sentence; what to do about it.
- Title: yours. It should make the claim of Sec. 2 in the reader's words. Candidates to beat:
  "Rare hosts make sharp sirens: the Hubble constant from black hole mergers in galaxies and active
  galactic nuclei"; "Black holes that know their hosts: dark siren cosmology with two tracers".

## 5. Voice (binding; details in `../paper_prd/notes/OWNER_STYLE_DETAILS.md` and the owner's
`astro-paper-style` skill, which you load)
- Physics first, plain words, short sentences. A number never appears without the number it is
  compared with and its unit. 90% credible intervals. Bayesian statements only (posteriors, Bayes
  factors with their prior ranges, the credible level at which an input lies in a posterior
  combined over simulated realizations); no coverage counts, p-values or sigma language.
- Words that may not organize the text: assumption, relax, step, ladder, stage, first/then/next/
  finally as the spine of a paragraph, "in this section", "as described above", summary, overview,
  anchor, proof of principle, pipeline, run, rerun, campaign, seed, gate, validation as a heading.
  No em dashes. No letters or code names for analyses: say what each one does ("one population for
  all mergers", "a population for each tracer", "hosts known").
- No Limitations paragraph. A caveat is the last clause of the claim it limits, with its size.
- These are simulations: say so once, early, with the number of mergers, and do not apologize for
  it again. Where the simulation makes a choice that drives a number (0.90 of the mergers sit in
  the Gaussian peak of the mass spectrum in one family, a few percent in the other; AGN number
  density; redshift errors), the choice travels with the number.
- Prior work, from `../paper_prd/notes/PRIOR_WORK_VERIFIED.md` only (statements checked against the
  papers; use its safe sentences; cite keys from `refs.bib`, a link in this directory). Position:
  H0 has been inferred together with the fraction of mergers in AGN before, and the AGN fraction
  has been inferred at fixed cosmology; do not claim either as new, do not write "first" for them,
  do not use the verb "unify". What is new here is the accounting of where the H0 information sits
  between the two tracers, and mass and spin distributions that depend on the tracer, inferred with
  the catalogs and H0.

## 6. Facts and numbers
- Simulation: galaxies (number density 1e-3 per Mpc^3, bias 1.2) and AGN (1e-5 per Mpc^3, bias 2.0)
  drawn from one lognormal density field; redshift errors 0.003(1+z); 1000 detected binary black
  hole mergers per realization with simulated posterior samples; input H0 = 67.74 km/s/Mpc, flat
  LCDM with matter density 0.3075. Mass spectrum: power law plus Gaussian peak; one family with
  0.90 of the mergers in the peak at 35 Msun (a sharp mass scale, chosen to make the mass scale as
  informative as it can be), one family like the fit to the third transient catalog (0.09 of the
  mergers, and 0.42 of the detected ones, in the peak). Mergers in AGN, when their population
  differs, have the peak higher by 5 Msun and the mean effective spin higher by 0.10 (and twice
  those values in the stronger case). The fraction of mergers in AGN takes the values 0, 0.1, 0.3,
  0.5 and 1. Design values are macros in `values/MACROS.md` (`values/` is a link; plain macros are
  final; use them, never a bare number).
- Equations: `../paper_prd/notes/LIKELIHOOD_MATH.md` (the mixture over tracers; Sec. 8 gives the
  identity behind claim 5).
- Results: almost all the numbers of this paper come from an experiment that is being run now
  (`EXPERIMENT.md` in this directory lists it; you do not need its logistics). Write the paper for
  the results we expect, listed below, with a macro for every number: define each in
  `values_independent.tex` in this directory as `\providecommand{\Name}{\ph{value}}` with the
  expected value inside the red `\ph`, input it after `values/results_macros.tex`, and list every
  macro with one line of meaning in `notes/MACROS_NEEDED.md` so that the numbers can be filled
  when the runs finish. Put the LaTeX comment `% EXPECTED` at the end of every sentence whose
  wording would change if the number came out differently. Do not write the word "placeholder" in
  the paper.
- Expected (90% half-width of H0 from 1000 mergers, km/s/Mpc, peak-dominated family, populations
  known unless said): mass spectrum alone, no catalog, about 3; all mergers in galaxies with the
  galaxy catalog, about 3.5 (the catalog adds little); all mergers in AGN with the AGN catalog,
  about 0.8; 0.3 of the mergers in AGN: labels unknown 1.7, with populations that differ by
  (5 Msun, 0.10) 1.5, by twice that 1.4, hosts known 1.3; 0.1 in AGN: labels unknown 2.4, hosts
  known 2.0. One population fitted to mergers whose populations differ: H0 high by about 3
  (differences of 5 Msun and 0.10) and about 6 (twice that), against a half-width of about 2.6 for
  one catalog of 1000 mergers; a population for each tracer, inferred: H0 67.4 +- 2.1, AGN
  fraction 0.30 +- 0.06, differences 5.1 +- 1.0 Msun and 0.10 +- 0.03, log10 Bayes factor about 9
  over the one-population analysis. For the family like the observed one the mass spectrum alone
  gives about 10 and the catalogs matter more. Analysis 15 (final, black macros `\Norm...` in
  `values/MACROS.md`; report in `../analyses/analysis_15_host_normalisation/REPORT.md`): with
  complete catalogs no abundance enters; at the shallowest flux limit an AGN abundance wrong by a
  factor of two moves the AGN fraction by about 0.13 and H0 by about 1 km/s/Mpc.

## 7. Figures (drawn by someone else; file names fixed; include each with `\IfFileExists`)
- `figures/fig1_budget`: 90% half-width of H0 from 1000 mergers against the fraction of mergers
  hosted by AGN. Curves: one population and both catalogs (labels unknown); a population for each
  tracer, for the two sizes of the difference; hosts known. Horizontal lines: the mass spectrum
  alone; the galaxy catalog alone when no merger is in an AGN. An open symbol where the populations
  are inferred and not known. The second mass-spectrum family as a second panel.
- `figures/fig2_labels`: for one realization, the probability that each merger is hosted by an AGN,
  shown separately for mergers truly in galaxies and truly in AGN: from positions alone, and from
  positions with masses and spins.
- `figures/fig3_bias`: (a) posterior of H0, combined over realizations, when one population is
  fitted to all mergers and when each tracer has its own, input marked; (b) the joint posterior of
  the two population differences with the input and zero marked.

## 8. Before you report
Build with `~/bin/tectonic main.tex`. Count the words of the main text (at most 2500). Then read
your draft as the reader of Sec. 1, with four minutes: after the abstract and the three captions,
can you state the claim of Sec. 2 and its numbers? Does any paragraph tell what was done instead of
what is true? Would the argument survive deleting it? Fix, then report: title, abstract, word and
page count, section headings, the list of macros you defined (count), and what in this brief you
think is wrong for the paper.
