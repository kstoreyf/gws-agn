# Style brief — arXiv:1905.12669 (Fishbach & Holz, "Picky Partners")

Source read: full LaTeX source of the ApJL-accepted version (arXiv e-print tarball,
`MassPairing.tex`, 357 lines / ~7.5k words of body text), plus all five figure
captions and both appendices. Cross-checked against ar5iv HTML and the arXiv
abstract page. All quotes below are verbatim from that source (LaTeX markup
stripped; the `\added{...}` wrappers are referee-response additions in the
accepted version and are quoted without the wrapper).

---

## 1. Identity

- **Title:** *Picky Partners: The Pairing of Component Masses in Binary Black Hole Mergers*
- **Authors:** Maya Fishbach (Dept. of Astronomy & Astrophysics, U. Chicago);
  Daniel E. Holz (Enrico Fermi Institute / Physics / Astronomy & Astrophysics /
  KICP, U. Chicago). **Two authors.**
- **Venue:** *Astrophysical Journal Letters* (ApJL), DOI 10.3847/2041-8213/ab7247;
  LIGO document P1900156. Submitted 29 May 2019, accepted version 4 Feb 2020.
  Categories: astro-ph.HE, gr-qc.
- **Length:** ~7,500 words total. Main text ~5,200 words (Intro 1,079; Models 892;
  Results 2,018; Simulations 453; Conclusion 777). Appendices ~1,415 words
  (Methods 1,072; Simulated Detections 343). **5 figures, 0 tables**, 80 references.
  Two-column `aastex62`; roughly 9 journal pages.
- **Subject:** hierarchical Bayesian population inference on the 10 O1/O2 BBH
  detections, asking whether component BH masses in a binary are randomly paired
  or preferentially near-equal. A *population-inference methods + result* paper —
  i.e. structurally the same animal as a dark-siren / H0-inference paper: a
  parameterized model, a nested null hypothesis, real data, a mock-catalog
  validation section, and a forecast.

**Why it is a useful model for us:** the whole paper is organized around a single
*nested null hypothesis* (`β_q = 0` ⇔ random pairing) and a single physical
question phrased in plain English ("does the universe pair black holes at
random?"). Everything else — models, priors, VT calculation, mock generation —
is subordinated to that one question and mostly banished to appendices.

---

## 2. Global architecture

Exact sequence, with size and job:

| # | Section | Paras | Words | Job in the argument |
|---|---------|-------|-------|---------------------|
| — | Abstract | 1 (7 sentences) | 294 (incl. title block) | States question, answer, and the three headline numbers. |
| 1 | Introduction | 6 | 1,079 | Field context → what LVC already did → **the gap** → why the pairing function is physically interesting → a subtle methodological point that motivates the 2D treatment → roadmap. |
| 2 | Mass Distribution Models | 7 | 892 | Defines the null (random pairing) analytically, then the one-parameter and two-parameter deviations from it. Pure model definition — *no* inference machinery. |
| 3 | Results | 14 | 2,018 | 3.1 LVC Model (reproduction/consistency check) · 3.2 Random Pairing · 3.3 Mass Ratio Dependent Pairing (**the headline**) · 3.4 Total Mass Dependent Pairing (the null that survives) · 3.5 Comparison of Pairing Functions · 3.6 Posterior predictive distributions (turns the fit into falsifiable predictions). |
| 4 | Simulations | 3 | 453 | Mock catalogs: what 60 and 100 future events will buy. Forecast only — the mock machinery itself is in Appendix B. |
| 5 | Conclusion | 7 | 777 | Restates the method choice, restates every headline number, maps results onto formation channels, states the single global caveat, ends on a one-sentence physical claim. |
| A | Appendix A: Methods | 7 | 1,072 | Poisson likelihood, VT, event-level priors, hyper-priors, sampler, Savage–Dickey. |
| B | Appendix B: Simulated Detections | 2 | 343 | The mock measurement model (σ_M/M and σ_η prescriptions). |

Notable architectural choices:

- **Section 3.1 is a reproduction section.** Before showing anything new, the
  authors re-derive the published LVC Model B result with their own pipeline and
  print both sets of numbers side by side. This is the paper's validation gate,
  and it costs one paragraph (115 words). Directly transferable: our "does our
  pipeline reproduce the reference result" check belongs *first in Results*, not
  in an appendix and not in a methods section.
- **The negative result gets its own subsection (3.4) and is not buried.**
  Total-mass pairing is unconstrained; the paper explains *why* (degeneracy
  `2γ + β_M ≈ const`), then extracts a usable statement from the degeneracy
  anyway ("for β_M = 4 we find γ ~ −2.5, close to Salpeter").
- **No "Methods" section in the main text at all.** The main text has Models
  (what is being fit) but never Methods (how it is fit). That split is the single
  most transferable structural trick in the paper.
- **Simulations come *after* the real-data result**, not before. Mocks are used to
  forecast, not to earn the right to look at data.

---

## 3. Abstract anatomy

Verbatim (accepted version):

> We examine the relationship between individual black hole (BH) masses in merging
> binary black hole (BBH) systems. Analyzing the ten BBH detections from
> LIGO/Virgo's first two observing runs, we find that the masses of the component
> BHs comprising each binary are unlikely to be randomly drawn from the same
> underlying distribution. Instead, the two BHs of a given binary prefer to be of
> comparable mass. We show that it is ∼5 times more likely that the component BHs
> in a given binary are always equal (to within 5%) than that they are randomly
> paired. If we assume that the probability of a merger between two BHs scales
> with the mass ratio q as q^β, so that β=0 corresponds to random pairings, we
> find β>0 is favored at credibility 0.987. By modeling the mass distribution, we
> find that the median mass ratio is q_50% = 0.91^{+0.05}_{-0.17} at 90%
> credibility. While the pairing between BHs depends on their mass ratio, we find
> no evidence that it depends on the total mass of the system: it is ∼6 times more
> likely that the pairing depends purely on the mass ratio than on the total mass.
> We predict that 99% of BBHs detected by LIGO/Virgo will have mass ratios q > 0.5.
> We conclude that merging black holes do not form random pairings; instead they
> are selective about their partners, preferring to mate with black holes of a
> similar mass. The details of these selective pairings provide insight into the
> underlying formation channels of merging binaries.

Sentence by sentence:

1. *"We examine the relationship between individual black hole (BH) masses…"* —
   **Scope, in one clause, jargon-free.** No context, no literature, no motivation.
   Notice it does not say "we perform a hierarchical Bayesian analysis"; the method
   is invisible in sentence 1.
2. *"Analyzing the ten BBH detections… unlikely to be randomly drawn from the same
   underlying distribution."* — **Data + headline result, in that order, in one
   sentence.** The dataset is named ("the ten BBH detections from O1 and O2") so the
   reader can calibrate the claim immediately.
3. *"Instead, the two BHs of a given binary prefer to be of comparable mass."* —
   **Plain-English restatement of sentence 2.** 13 words, zero symbols. This is the
   sentence a non-specialist remembers.
4. *"We show that it is ∼5 times more likely…"* — **Headline number #1**, expressed
   as an odds ratio in words, with the operational definition of "equal" inlined
   ("to within 5%").
5. *"If we assume that the probability… scales with the mass ratio q as q^β… we find
   β>0 is favored at credibility 0.987."* — **Headline number #2, with its
   assumption attached in the same sentence.** The `β=0 corresponds to random
   pairings` clause defines the null inside the abstract so the number is
   interpretable without the paper.
6. *"By modeling the mass distribution, we find that the median mass ratio is
   q_50% = 0.91^{+0.05}_{-0.17} at 90% credibility."* — **Headline number #3**, the
   only full error bar in the abstract, with credibility level stated.
7. *"While the pairing… depends on their mass ratio, we find no evidence that it
   depends on the total mass… ∼6 times more likely…"* — **The negative result, given
   equal billing and its own quantitative odds.**
8. *"We predict that 99% of BBHs detected by LIGO/Virgo will have mass ratios
   q > 0.5."* — **A falsifiable prediction.** Costs one sentence; converts a fit
   into a bet.
9–10. *"We conclude that merging black holes do not form random pairings; instead
   they are selective about their partners, preferring to mate with black holes of a
   similar mass. The details of these selective pairings provide insight into the
   underlying formation channels…"* — **Physical conclusion, then implication.** The
   anthropomorphic register ("picky", "partners", "mate") is deliberate and is the
   paper's memorability device; it appears in the title, the abstract, and the last
   line of the conclusion, and nowhere else.

Anatomy in one line: **scope → data+result → plain restatement → number → number
(+assumption) → number (+uncertainty) → null result (+number) → prediction →
physical conclusion → implication.** No sentence is spent on methods, software,
or motivation.

---

## 4. Introduction anatomy (6 paragraphs, 1,079 words)

- **¶1 (338 words) — Context and the state of the art.** Detector runs and event
  counts → "The formation and history of these BBHs remains a fundamental question
  in GW astrophysics." → three formation channels, each a citation dump →
  population fitting is how you learn about them → exactly what LVC's Abbott et al.
  (2018b) fit, described concretely enough that the reader can see its shape
  ("the primary mass … follows a power-law … while the secondary mass is
  distributed with a power-law between the minimum mass and its primary mass
  partner"). Ends with a **scope restriction and its justification**:
  *"In this work, we restrict the population analysis to the ten Abbott et al.
  (2018c) BBH detections, as the detection efficiency has been previously studied
  for this sample and is well-understood… Using the wrong detection efficiency
  leads to selection biases in population inference."* Note the caveat is delivered
  as a *design decision with a physical reason*, in the first paragraph, not as an
  apology in a caveats section.
- **¶2 (126 words) — The gap and this work, in three sentences.**
  *"In this work we extend the analysis of Abbott et al. (2018b) by focusing on a
  particular aspect of the BBH mass distribution: the pairing between the two
  component BHs in the binary."* → the question in plain English → why the prior
  parameterization structurally cannot answer it ("it is not possible to fit for an
  underlying mass distribution that is common to both component BHs or quantify the
  deviation from the random-pairing scenario, as we do in this work"). The gap is
  stated as a *capability the existing model lacks*, not as a criticism.
- **¶3 (273 words) — Why the answer matters physically.** A survey of what each
  formation channel predicts for the pairing function, organized as
  channel → prediction → mechanism, including channels that predict the *opposite*
  sign. Closes with a one-line purpose statement: *"Constraining the BBH pairing
  function with GW observations allows us to test these different predictions."*
- **¶4 (84 words) — Precedent from a neighboring field** (stellar binary IMF pairing
  functions), with an honest limit on how far the analogy carries ("complicated by
  the many stages of evolution undergone by BBHs").
- **¶5 (169 words) — The methodological insight that justifies the whole paper.**
  Why 1D marginals cannot answer the question: *"a mass ratio distribution that
  favors near-unity mass ratios may simply indicate that the underlying BH mass
  distribution peaks in a narrow mass range, rather than that similar component
  masses are more likely to partner and merge."* This is the paper's intellectual
  core and it is delivered in the introduction, before any equation, as a worked
  confusion the reader might otherwise have.
- **¶6 (89 words) — Roadmap.** One sentence per section, including the appendix.
  Plain and short: *"We conclude in Section 5. Appendix A describes the details of
  the hierarchical Bayesian analysis."*

Transferable template: **context → gap-as-missing-capability → why the answer
matters physically → precedent → the subtle point that makes the analysis
necessary → roadmap.** No "the paper is organized as follows" padding beyond the
one roadmap paragraph, and no results preview in the introduction — the intro
never states the headline numbers.

---

## 5. Results-first technique

- **The headline numbers first appear in the abstract** (sentences 4–8), then next
  in **Section 3.3**, ~2,900 words into the paper. There is no "results summary"
  box, no bulleted contributions list.
- **Methods are subordinated by physical relocation, not by compression.** The main
  text contains only *models* (Section 2: what distribution is being fit) and
  *results*. Every inference ingredient — the inhomogeneous Poisson likelihood,
  ⟨VT⟩, event-level prior division, hyper-priors and their ranges, PyMC3, the
  Savage–Dickey density ratio, the mixture-model evidence trick — lives in
  Appendix A. The mock measurement model lives in Appendix B.
  Ratio: **main text 5,200 words vs. appendix 1,415 words**, i.e. ~21% of the paper
  is machinery, and it is all at the back.
- **Forward references replace exposition.** The main text repeatedly does
  "Following the methods laid out in Appendix A, we fit…" and "The mock posteriors
  are generated according to the prescription described in Appendix B."
  Each is one clause; the reader is never detained.
- **Citations replace derivations.** *"The likelihood is given by the inhomogeneous
  Poisson process likelihood (Loredo 2004; Mandel et al. 2016; Abbott et al. 2018b)"*
  — the standard machinery is cited, not re-derived, even in the appendix.
- **Results paragraphs lead with the physical statement and put the number second.**
  E.g. §3.3: "the data display a clear preference for mass ratios close to unity …
  We infer β_q = 7.0^{+4.5}_{-5.5}, and find that β_q ≤ 0 is ruled out with
  probability 0.987."
- **Section 3.1 exists purely to earn trust** before any new number is shown, and
  its output is a *comparison of two number sets* (ours vs. published), not a plot.
- The Simulations section (4) is only 453 words and contains no method at all — it
  is entirely "with N events we will measure X to Y".

---

## 6. Voice

- **Person:** first-person plural throughout. 54 sentence-initial "We", 75
  lower-case "we", 21 "our". **"We" opens 28 of 192 main-text sentences (15%)** —
  the single most common sentence opener, ahead of "The" (20) and "In" (15). No
  passive-agent dodges like "it was found that".
- **Tense:** present tense for what the paper does and finds ("we find", "we infer",
  "we predict", "the data display"); present for the physics ("many formation
  channels predict"); future for forecasts ("will become increasingly
  well-constrained", "we expect to constrain"); present perfect only in the
  Conclusion's opening ("We have fit the mass distribution…").
- **Active/passive:** ~20% of main-text sentences contain a passive construction, and
  passive is used almost exclusively for *things done to the model or the data by
  nature or by convention* ("the primary mass is defined to be the more massive
  component", "binaries formed via homogeneous chemical evolution are expected to…").
  Every action the authors take is active.
- **Rhythm:** mean 27 words/sentence, median 26, range 4–77. The rhythm is
  **long–long–short**, where the short sentence carries the claim:
  - long (54 w): the formation-channel citation dump;
  - short (13 w): *"Instead, the two BHs of a given binary prefer to be of comparable mass."*
  - short (20 w): *"This suggests that the random pairing model (β_q = 0) is strongly
    disfavored by the data."*
  Emphasis is carried by *italics*, used ~4 times in the whole paper and only on
  claim sentences.
- **Hedging:** 25% of main-text sentences hedge. Five actual formulas, verbatim:
  1. *"Some studies have suggested that dynamical evolution also tends to produce
     more mergers with equal mass components…"*
  2. *"However, other dynamical channels may mildly prefer unequal mass components."*
  3. *"It is possible that studying the pairing function for merging BBHs may shed
     light on the masses of their stellar progenitors, although the relationship
     between a BH's mass and its progenitor star's zero-age main-sequence (ZAMS)
     mass is complicated by the many stages of evolution undergone by BBHs."*
  4. *"This suggests an even stronger preference for near-equal component masses in
     the underlying population."*
  5. *"While we cannot rule this out with ten detections, it is 6 times more likely
     that the pairing function has some mass-ratio dependence rather than depending
     on total mass alone."*
  Note the pattern: **hedges attach to interpretation, never to the measurement.**
  The measurement sentences are flat declaratives with numbers.
- **Register:** deliberately colloquial *only* on the framing metaphor ("picky",
  "partners", "mate", "the universe makes… by randomly pairing up black holes").
  Everywhere else the prose is plain and technical. Jargon is defined on first use
  and then used consistently (BBH, pairing function, posterior population
  distribution, posterior predictive process).

---

## 7. Numbers

- **Precision:** two significant figures with asymmetric 90% credible intervals as
  the house style: `β_q = 7.0^{+4.5}_{-5.5}`, `m_max = 41.9^{+18.2}_{-5.7} M_⊙`,
  `γ = -1.4^{+0.9}_{-0.8}`, `q_50% = 0.91^{+0.05}_{-0.17}`. Credibility level is
  stated explicitly at least once per context ("at 90% credibility"); Appendix
  quotes another paper's 68% interval and says so.
- **Odds are quoted in words, not logs:** "∼5 times more likely", "∼6 times more
  likely", "Bayes factors ≳ 1000". No `ln B` values anywhere in the main text.
- **Probabilities of a hypothesis are quoted as credibilities:** "β_q ≤ 0 is ruled
  out with probability 0.987" — three decimal places, no sigma conversion.
- **Rounded/approximate numbers use ∼ and are clearly flagged:** "m_min ∼ 7 M_⊙",
  "γ ∼ −2.5", "constrained to ∼0.15 in this case".
- **Assumptions travel with the number, in the same sentence.** Examples:
  - *"If we assume that the probability of a merger between two BHs scales with the
    mass ratio q as q^β, so that β=0 corresponds to random pairings, we find β>0 is
    favored at credibility 0.987."*
  - *"We find that q_min = 0.95 is five times more likely than
    (β_q, q_min) = (0, m_min/m_max)"* — the comparison point is written out, so the
    odds ratio is not free-floating.
  - *"Based on the first ten detections, and assuming that all detections are
    described by the same population model assumed here, we expect that 90% of
    future detections will have recovered mass ratio posteriors that lie within the
    shaded band."*
- **Forecast numbers are always tied to an event count and a date:**
  "60 detections (similar to what we expect by the end of O3)", "100 simulated
  events gives m_max = 38.9^{+1.4}_{-0.9} M_⊙", plus a scaling law with an
  explanation: *"these constraints will improve roughly as 1/N and 1/√N for m_max
  and γ, respectively"*, footnoted with why m_max beats √N ("because m_max is a
  sharp feature").
- **Placement:** **all numbers are inline in prose. There are zero tables.** Numbers
  the reader must compare (ours vs. LVC's) are placed in adjacent sentences in the
  same paragraph. Distributions go in figures; point estimates go in sentences.
- **Reproducibility hygiene:** every quoted value is a LaTeX macro defined at the
  top of the source (e.g. `\newcommand{\qminbetabeta}{5.1^{+6.1}_{-7.8}}`), so a
  rerun updates every occurrence at once. 60 such macros. Worth copying.

---

## 8. Figures

Five figures, no tables. Types:

1. **Fig. 1** (`\figure*`, full width) — a 3×3 grid: three competing models across
   columns, three views of each (joint m1–m2 rate density; 1D marginals of single /
   primary / secondary mass; 1D mass-ratio distribution) down rows. **This is the
   paper's argument in one image**: the reader sees the null and the alternative
   side by side in identical axes.
2. **Fig. 2** (`\figure*`) — corner plot of the 5 hyper-parameters, real data.
3. **Fig. 3** — posterior population distribution in m1–m2: underlying population
   (blue) vs. detected population (orange) vs. the ten observed events (grayscale).
4. **Fig. 4** — 1D mass-ratio posterior population distribution plus the
   "posterior predictive process" band, with the ten observed posteriors overlaid.
5. **Fig. 5** (`\figure*`) — corner plot for 60 mock events with injected truths
   marked in orange lines.

Note the pattern: **two corner plots (one data, one mock, same axes), two
predictive-distribution figures, one model-comparison grid.** A dark-siren paper
maps onto this almost one-to-one: model-comparison grid, real-data corner,
mock-recovery corner with truth lines, and a predictive check.

Two captions quoted and annotated.

**Caption, Fig. 1:**
> *Top row:* Joint m1–m2 distribution as inferred from the ten BBHs assuming a mass
> distribution given by Eq. 6 with free parameters γ, m_min, m_max (left column),
> γ, m_min, m_max and β_q (middle column), and γ, m_min, m_max and β_M (right
> column). In each case, those parameters that are not free are fixed to
> β_q = β_M = 0 and q_min = m_min/m_max. The color scale indicates the median
> log10 of the merger rate density as a function of the two masses. *Middle row:*
> Marginal distributions of single BH masses (green), along with the primary masses
> (blue) and secondary masses (yellow) of component BHs in binary systems. These
> distributions are inferred by fitting the ten BBH detections to the model of the
> corresponding column. The line shows the median merger rate density as a function
> of mass, while the shaded bands show symmetric 90% credible intervals.
> *Bottom row:* Marginal distribution of the mass ratio implied by the fits to the
> three models. The solid line and dark (light) bands denote median and 50% (90%)
> credible intervals on the merger rate as a function of mass ratio.

Why it works: (a) it is navigable — italic *Top row / Middle row / Bottom row*
labels let the reader jump; (b) **every colour is bound to a meaning in
parentheses** (green / blue / yellow) rather than left to a legend; (c) it states
exactly which parameters are free and *what the others are fixed to* — the
caption is self-sufficient for reproducing the panel; (d) it says what the line
and what the band mean, and at what credibility, in every row. It contains no
interpretation at all.

**Caption, Fig. 3:**
> Posterior population distribution of the component masses in BBH binaries, as
> inferred from the mass-ratio dependent pairing model. The true masses of the
> *underlying* population are represented by the blue points and 90% credible
> region, while the orange points represent the *detected* population, accounting
> for selection effects that favor more massive systems. In grayscale are the mass
> measurements of the ten LIGO/Virgo O1 and O2 detections. The contours denote 90%
> credible intervals. All detected systems are consistent with equal component
> masses m1 = m2.

Why it works: (a) it names the quantity plotted and the model it came from in the
first clause; (b) it distinguishes *underlying* vs *detected* in italics and
explains the physical reason the two differ ("selection effects that favor more
massive systems") — the caption teaches the selection effect rather than assuming
it; (c) it ends with **one sentence of takeaway** ("All detected systems are
consistent with equal component masses") — Fishbach & Holz allow exactly one
interpretive sentence, and only at the end. Fig. 4's caption ends the same way
("Note that measurement uncertainty shifts the posteriors on the mass ratio for
individual systems to smaller values relative to the true mass ratio.").

Caption length: 60–170 words. All are full sentences. None says "see text".

---

## 9. Caveats and assumptions

Caveats are distributed, not quarantined. There is **no "Limitations" or
"Systematics" section**; instead each caveat appears at the moment the reader
could be misled, and is always paired with either a reason or a bound on its size.

Examples, verbatim:

- **Scope restriction, stated in ¶1 of the Introduction with its physical reason:**
  *"In this work, we restrict the population analysis to the ten Abbott et al.
  (2018c) BBH detections, as the detection efficiency has been previously studied
  for this sample and is well-understood… Using the wrong detection efficiency
  leads to selection biases in population inference. In future work we will extend
  our analysis to include overlapping samples with differing selection effects."*
- **Excluded data, revisited where it would help them, with the exclusion
  re-justified:** *"Although we do not include the events of Venumadhav et al. or
  Nitz et al. in our analysis (in order to avoid assuming an incorrect selection
  function and biasing our results), we note that all of their detections are also
  consistent with mass ratios of unity."*
- **A known bias, quantified and bounded in the appendix:** *"using the uncalibrated
  VT calculation leads to a slight bias in our inference of the overall-merger rate,
  with the median shifting by a factor of ∼1.7, as expected from Fig. 9 in Abbott
  et al. However, this does not affect the inferred shape of the mass distribution,
  which is our primary interest in this work."* — the template is
  **name the bias → give its size → say which of your results it does and does not
  touch.**
- **A fixed nuisance distribution defended by an argument, not asserted:** *"For
  definiteness we fix the spin distribution of both binary components to be uniform
  in spin magnitude and isotropic in spin tilt. Although this distribution is not
  necessarily favored by the data, the correlation between the inferred spin
  distribution and the inferred mass distribution is negligible, as shown in Abbott
  et al. (2018b)… In particular, despite using a different spin model, we recover
  the results of Abbott et al. (2018b) under the same mass model."*
- **A prior boundary that is an artifact, admitted as such:** *"We restrict the
  upper limit of q_min to slightly below 1 in order to avoid sampling issues… and
  this prevents q_min from being resolved arbitrarily close to q_min = 1."*
- **Forecast conservatism, flagged as conservative:** *"These projections are
  conservative because the deviations from random-pairing in the chosen mock
  population are not very large compared to the values of q_min and β_q that are
  favored by the first ten events."*
- **The one global caveat, in the penultimate paragraph of the Conclusion, with a
  test attached:** *"As usual, our results rely on the assumption that there is a
  single population of BBHs that is adequately described by our simple parameterized
  model. One way to test the validity of this assumption with future detections is
  to compare them against the posterior predictive distribution… inferred from the
  model."*

**Caveat-to-claim ratio:** ~25% of main-text sentences carry a hedge or a
qualification, but only ~7 sentences in ~5,200 words are pure caveats. Claims are
never softened; the *conditions under which the claim holds* are stated instead.
And every substantive caveat is followed by either a size estimate, a reason it
doesn't matter, or a future test. There is not a single naked "we caution that…".

---

## 10. Transitions and cadence

- **Sections open with the action, not with a preamble.** "We begin by recovering
  the results of Abbott et al…"; "Figure 1 shows the results of fitting…"; "In the
  simplest case, we consider a model in which…"; "We have fit the mass distribution
  of merging BBHs with a simple model…". No section opens with a summary of the
  previous section.
- **Sections close with a hand-off.** §2 ends: *"Following the methods laid out in
  Appendix A, we fit the pairing models discussed above to the first ten BBH
  detections in Section 3… we quantify the evidence against the random pairing and
  the total-mass dependent pairing models, and find that the mass-ratio dependent
  pairing model provides the best fit to the data."* — i.e. the models section ends
  by telling you the answer is coming and what it is.
- **Contrast connectives do the structural work.** "However," opens 9 sentences;
  "Meanwhile," opens 5; also "Instead,", "On the other hand,", "In fact,",
  "Alternatively,", "Furthermore,". Each is a genuine pivot, usually
  claim → counter-consideration.
- **Signposting verbs mark the load-bearing sentences:** "We emphasize that…",
  "We highlight that…", "We reiterate that…", "We note that…". These are reserved
  for points the authors expect to be misread (e.g. *"We emphasize that the pairing
  mechanism cannot be determined by examining any one of these one-dimensional
  distributions independently."*).
- **Conclusion structure (7 paragraphs, 777 words):**
  1. What we did, methodologically, in one sentence + why the 2D treatment was
     necessary (the intro's key insight, restated).
  2. The headline conclusion in plain language + the 5× odds number.
  3. The derived numbers (q_1% ours vs. random-pairing) and the predictions for
     detected events + the null result on total mass.
  4. **Mapping to formation channels** — which are consistent, which are excluded,
     each with a citation and a number.
  5. What a prior belief about γ would imply for β_M — i.e. how a reader with
     different priors should read the result.
  6. Forecast: "By the end of O3, the details of the pairing function will be better
     constrained (compare Fig. 2 … with Fig. 5)."
  7. **The global caveat + a test for it + a one-sentence closing claim.**
- **The last sentence is the abstract's last sentence, re-voiced:**
  *"We conclude that the universe does not assemble its black-hole binaries at
  random, and future constraints of the pairing function we have introduced above
  will yield important insights into these formation processes."* — physical claim
  first, forward look second, no new information.

---

## 11. Ten exemplary sentences

1. *"We ask whether the universe makes merging binary black hole systems by randomly
   pairing up black holes, or whether the mass of each black hole in a pair
   influences the mass of its companion."*
   — The entire paper as a single yes/no question, in words a first-year student
   understands. No symbols, no acronyms beyond the title's.

2. *"Instead, the two BHs of a given binary prefer to be of comparable mass."*
   — 13 words. Sits immediately after the technical statement of the same result and
   translates it. Every headline number in this paper has a plain-English twin.

3. *"We show that it is ∼5 times more likely that the component BHs in a given binary
   are always equal (to within 5%) than that they are randomly paired."*
   — An odds ratio made concrete by naming *both* hypotheses being compared and
   inlining the operational definition of "equal".

4. *"If we assume that the probability of a merger between two BHs scales with the
   mass ratio q as q^β, so that β=0 corresponds to random pairings, we find β>0 is
   favored at credibility 0.987."*
   — Assumption, null-hypothesis mapping, and result in one sentence; the number is
   unusable without the clause, so the clause is not allowed to drift away from it.

5. *"We emphasize that the pairing mechanism cannot be determined by examining any
   one of these one-dimensional distributions independently."*
   — A negative methodological claim, stated flatly, that justifies the paper's
   existence. Placed in the introduction, before any math.

6. *"For example, a mass ratio distribution that favors near-unity mass ratios may
   simply indicate that the underlying BH mass distribution peaks in a narrow mass
   range, rather than that similar component masses are more likely to partner and
   merge."*
   — Pre-empts the exact wrong inference a reader would make, by name. The best
   single sentence to imitate when explaining why a degeneracy matters.

7. *"This suggests that the random pairing model (β_q = 0) is strongly disfavored by
   the data."*
   — Short, italicized, immediately after the number. The claim sentence is always
   separate from the number sentence.

8. *"We do not recover significant constraints beyond the linear combination
   2γ + β_M ≈ −1.1^{+1.0}_{-0.9}."*
   — How to report a null: say precisely *what* you did measure (the degenerate
   combination), with its error bar, rather than "we find no constraint".

9. *"using the uncalibrated VT calculation leads to a slight bias in our inference of
   the overall-merger rate, with the median shifting by a factor of ∼1.7 … However,
   this does not affect the inferred shape of the mass distribution, which is our
   primary interest in this work."*
   — The model caveat sentence: name the systematic, quantify it, and delimit which
   result it touches.

10. *"We conclude that the universe does not assemble its black-hole binaries at
    random, and future constraints of the pairing function we have introduced above
    will yield important insights into these formation processes."*
    — Closing sentence: physical claim in ordinary English, then one forward look.
    No summary, no new content, no thanks to the reader.

---

## 12. DO / DON'T for a dark-siren / mock-catalog / H0-inference validation paper

**DO**

1. **Frame the paper as one yes/no physical question** and put it in the
   introduction as a single sentence with no symbols. (Ours: "does a mock catalog
   built to match the analysis assumptions still return a biased H0?")
2. **Make the null hypothesis a nested parameter value** and say so explicitly in
   the abstract ("f_AGN = 0 corresponds to no AGN correlation", "Δ = 0 corresponds
   to an unbiased estimator"). Then every result is a statement about a number, not
   about a vibe.
3. **Put the reproduction/consistency check first in Results**, one paragraph, with
   your numbers and the reference numbers printed side by side in adjacent
   sentences. This is the reader's reason to believe everything after it.
4. **Split "Models" (main text) from "Methods" (appendix).** The main text says what
   distribution is being fit and what each parameter means; the likelihood, priors,
   selection function, sampler, and convergence go to an appendix. Target ~20% of
   words in appendices.
5. **Attach the assumption to the number in the same sentence.** Never let
   "H0 = 68 ± 3 km/s/Mpc" appear without the clause naming the catalog, the
   completeness assumption, and the credibility level.
6. **Quote odds in words** ("~5 times more likely than", "Bayes factors ≳ 1000") and
   name both compared hypotheses in the same breath.
7. **Report nulls by stating what you did measure** — the degenerate combination,
   the bound, the prior you recovered — not by saying "unconstrained".
8. **Give the negative result its own subsection and its own abstract sentence** if
   it is physically interesting. A clean "no evidence that X matters" is a result.
9. **Convert the fit into a falsifiable prediction** ("we predict 99% of detected
   BBHs will have q > 0.5"). For us: predict what the next N events, or a specific
   held-out catalog, must show if the mock is right.
10. **Use mocks for forecasting and for truth-recovery, and show the mock corner plot
    on the same axes as the data corner plot with injected truths marked.** Then
    "compare Figure 2 with Figure 5" is an argument that needs no prose.
11. **Make figure captions self-sufficient**: name every colour and line style in
    parentheses, state which parameters are free and what the rest are fixed to,
    state the credibility level of every band, and allow yourself exactly one
    interpretive sentence at the end.
12. **Explain selection effects inside the caption** the first time underlying vs.
    detected populations appear side by side.
13. **Place each caveat where the reader could be misled**, with a size estimate and
    a statement of which results it does and doesn't affect. One global caveat goes
    in the penultimate paragraph of the Conclusion, with a test attached.
14. **Define every quoted number as a macro** in the manuscript source so a pipeline
    rerun propagates everywhere (they use ~60 `\newcommand`s). Directly applicable to
    our `values/` → `main.tex` flow.
15. **Signal load-bearing sentences with "We emphasize / We highlight / We
    reiterate"** and reserve them for points you expect to be misread.
16. **Close with the physical claim in ordinary English, then one forward look.**

**DON'T**

1. **Don't open the abstract with methods.** No "We perform a hierarchical Bayesian
   analysis using nested sampling…" in sentence 1. Scope first, result second.
2. **Don't preview results in the introduction.** This paper's introduction states
   the question and never the answer; the roadmap paragraph is 89 words and lists
   sections, not findings.
3. **Don't write a "the paper is organized as follows" paragraph longer than one
   short paragraph**, and don't repeat it at every section boundary.
4. **Don't put pipeline, workflow, or gate language in reader-facing text.** There is
   no mention of convergence diagnostics, wall-clock, chains, seeds, or code
   versions anywhere in the main text; the sampler is named once, in the appendix.
5. **Don't build tables of point estimates.** Zero tables here; numbers you want
   compared go in adjacent sentences, distributions go in figures.
6. **Don't quote log-Bayes factors or sigmas where an odds ratio in words works.**
7. **Don't hedge the measurement.** Hedge the interpretation ("may", "suggests",
   "tends to") and leave the number sentences flat.
8. **Don't quarantine limitations in a "Caveats" section** where they read as an
   apology and nobody reads them.
9. **Don't state a caveat without a size or a bound.** "We caution that selection
   effects may matter" is worthless; "this shifts the rate median by a factor ~1.7
   but does not affect the shape" is a result.
10. **Don't let a symbol appear before it has a plain-English gloss.** Every
    parameter in this paper is introduced as a sentence in words before it is an
    equation.
11. **Don't add a jargon acronym you use fewer than five times.**
12. **Don't bury the reproduction check in an appendix** — it is the credibility of
    the whole paper and belongs in §3.1.
13. **Don't run mocks before the real result unless the mocks *are* the result.**
    Here mocks come after, as forecast. If our paper's point is the mock validation
    itself, the mock section is the Results — but the reproduction of the reference
    analysis still comes first.
14. **Don't use the memorable metaphor more than three times** (title, abstract,
    closing line). Everywhere else the prose is plain.
15. **Don't write captions that say "see text".**
