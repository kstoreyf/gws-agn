# Style brief — arXiv:2407.02460 ("A new bump in the night")

Source read: full LaTeX source (`arxiv.org/e-print/2407.02460`, v2 = accepted version),
including every figure caption and the appendix. Nothing summarized second-hand.

---

## 1. Identity

- **Title:** *A new bump in the night: Evidence of a new feature in the binary black hole mass
  distribution at $70~M_\odot$ from gravitational-wave observations*
- **Authors:** Ignacio Magaña Hernandez, Antonella Palmese (both McWilliams Center for
  Cosmology, Carnegie Mellon University). Two authors, one institution.
- **Venue:** Phys. Rev. D **111**, 083031 (2025). arXiv v1 2024-07-02, v2 2025-02-23
  ("Matches accepted version"). REVTeX4, `prd`, two-column.
- **Length:** 14 journal pages. ~5,350 words of body text (abstract 204 words), 8 numbered
  equations, **12 figures** (9 in the main text, 3 corner plots in Appendix A), **0 tables**,
  71 references.
- **Subject:** astro-ph.HE + gr-qc. Hierarchical Bayesian population inference on the 69
  confident GWTC-3 BBH mergers with a mixture-of-power-laws mass model; discovery claim of a
  ~65–70 $M_\odot$ sub-population, plus a 100-realization mock-catalog validation of the claim.

**Why this is a useful model for us:** it is *the same argument shape* as a mock-validated
dark-siren/H0 paper — a modelling choice on real data, a headline number whose credibility
rests entirely on simulated catalogs, and an honest statement that the evidence is marginal.
The paper's central rhetorical achievement is claiming a *weak* result (a $\log_{10}\mathcal{B}$
of 0.29!) without either overselling or apologizing it away.

---

## 2. Global architecture

Exact sequence, with paragraph counts and the job each block does:

| # | Section | Paras | Words | Job in the argument |
|---|---------|-------|-------|---------------------|
| — | Abstract | 1 (6 sentences) | 204 | Whole argument, with both headline numbers |
| 1 | Introduction | 8 (short) | 480 | Field state → menu of existing models → the specific anomaly → our question → roadmap |
| 2 | Methods | 1 intro | 50 | One paragraph: "we do hierarchical Bayes, here are reviews" |
| 2.1 | Hierarchical Bayesian Inference | 5 | 495 | Likelihood, selection function, $N_{\rm eff}$ criterion, PE priors |
| 2.2 | Model Comparison | 2 | 178 | Defines the Bayes factor and, crucially, *how to read it* |
| 2.3 | Population Models | 4 | 446 | One master equation; the three models are limits of it |
| 3 | Results | 7 | 931 | GWTC-3 constraints; both Bayes factors; which events drive the feature |
| 4 | Validation | 8 | 773 | 100 mock catalogs; false-positive rate for each claim |
| 5 | "Does spin tell us anything?" | 5 | 543 | An orthogonal observable used as a consistency test of the interpretation |
| 6 | Astrophysical Implications | header | — | Splits into three sub-populations |
| 6.1 | Low-mass ~10–20 $M_\odot$ | 1 | 206 | PL1 = isolated binaries |
| 6.2 | Intermediate-mass ~20–40 $M_\odot$ | 5 | 683 | PL2 = 1g+1g dynamical; longest interpretive block |
| 6.3 | High-mass ~60–80 $M_\odot$ | 1 | 233 | The bump = 2g hierarchical mergers |
| 7 | Conclusions | 2 | 339 | Restate both results with their caveats; forecast O4 |
| — | Acknowledgements | 1 | — | |
| A | Full Posterior distributions | 0 text | — | Three corner plots, no prose |

**Structural lessons.**
1. **Methods is 1,170 words and Results+Validation is 1,700.** The evidence outweighs the
   machinery ~1.5:1, and the machinery is compressed by outsourcing to reviews
   ("we refer the reader to review articles for an in depth description").
2. **Validation is its own numbered section, not an appendix and not a subsection of
   Results.** It is 8 paragraphs — nearly as long as Results. This is the single most
   transferable structural choice for a mock-catalog paper: the null test that licenses the
   claim gets top-level billing.
3. **The appendix contains zero prose.** It holds only the three corner plots that a referee
   or reproducer would want. Nothing is hidden there that the argument needs.
4. **Interpretation is quarantined in §6**, after all quantitative claims are settled. The
   reader never has to disentangle "what we measured" from "what we think it means."
5. **No table anywhere.** Every number is either inline or in a figure.

---

## 3. Abstract anatomy

> We analyze the confident binary black hole (BBH) detections from the third
> Gravitational-Wave Transient Catalog (GWTC-3) with an alternative mass population model in
> order to capture features in the mass distribution beyond the **Powerlaw + Peak** model. We
> find that the peak of a second power law characterizes the $\sim 30-35~M_\odot$ bump, such
> that the data marginally prefers a mixture of two power laws for the mass distribution of
> binary components over a **Powerlaw + Peak** model with a Bayes Factor
> $\log_{10}\mathcal{B}$ of 0.24. This result may imply that the $\sim 30-35~M_\odot$ feature
> represents the onset of a second population of BBH mergers (e.g. from a dynamical formation
> channel) rather than a specific mass feature over a broader distribution. When an additional
> Gaussian bump is allowed within our power law mixture model, we find a new feature in the BH
> mass spectrum at $\sim65-70~M_\odot$ ($\log_{10}\mathcal{B}$ = 0.29 compared to
> **Powerlaw + Peak**). This new feature may be consistent with hierarchical mergers, and
> constitute $\sim3\%$ of the BBH population. This model also recovers a maximum mass of
> $58^{+32}_{-14}~M_\odot$ for the second power law, consistent with the onset of a
> pair-instability supernova mass gap.

Sentence by sentence:

| S | Text (opening) | Function |
|---|---|---|
| 1 | "We analyze the confident BBH detections… with an alternative mass population model in order to capture features… beyond the Powerlaw + Peak model." | **Data + method + purpose in one sentence.** No paragraph of context first. The benchmark to be beaten (Powerlaw+Peak) is named immediately, so the whole abstract has a yardstick. |
| 2 | "We find that the peak of a second power law characterizes the $\sim30-35~M_\odot$ bump… with a Bayes Factor $\log_{10}\mathcal{B}$ of 0.24." | **Result #1 with its number.** Note "marginally prefers" — the hedge is welded to the claim, not deferred. The number arrives in sentence 2 of the paper. |
| 3 | "This result may imply that the… feature represents the onset of a second population… rather than a specific mass feature over a broader distribution." | **Physical meaning of result #1**, stated as an either/or the reader can hold in their head. "may imply" flags it as interpretation. |
| 4 | "When an additional Gaussian bump is allowed… we find a new feature in the BH mass spectrum at $\sim65-70~M_\odot$ ($\log_{10}\mathcal{B}$ = 0.29…)." | **The headline result**, with the model condition stated as a subordinate clause ("When … is allowed") so the method never becomes the subject. |
| 5 | "This new feature may be consistent with hierarchical mergers, and constitute $\sim3\%$ of the BBH population." | **Interpretation + a scale for it.** The 3% tells the reader how big a deal this is, which a Bayes factor cannot. |
| 6 | "This model also recovers a maximum mass of $58^{+32}_{-14}~M_\odot$… consistent with the onset of a pair-instability supernova mass gap." | **A bonus sanity check**: an independently-known physical scale that the model reproduces. Ends the abstract on external corroboration rather than on a promise. |

**Pattern to steal:** context(0) → method-as-clause(1) → number(2) → meaning(3) → bigger
number(4) → meaning + scale(5) → independent corroboration(6). Zero sentences of "the study of
X has long been important." Zero forward references to sections. Six sentences, two Bayes
factors, one credible interval, one percentage.

---

## 4. Introduction anatomy — paragraph function map

Eight paragraphs, mean ~60 words. Several are 2–3 sentences long; the introduction reads as a
rapid stack of premises, not as an essay.

| ¶ | Opening words | Function |
|---|---|---|
| 1 | "Since the first detection of a binary black hole (BBH) merger…" | **Field state in 2 sentences.** Immediately quantitative: "69 high significance BBH mergers have been detected, and hundreds are expected… throughout the ongoing fourth… observing run." Establishes both the data set used and why the paper will matter more later. |
| 2 | "Multiple formation channels have been proposed…" | **The theory landscape**, delivered as a single sentence with ~12 citations, closed by "For in depth reviews, see…". Compresses a review into one sentence. |
| 3 | "It is becoming increasingly clear that there is no single dominating channel…" | **The consensus premise** (3 lines) that makes a *mixture* model the natural object. This is the conceptual hinge and it gets its own paragraph for emphasis. |
| 4 | "Multiple options for fitting the BBH mass distribution… have been proposed." | **The methods landscape**: what other people fit, parametric and non-parametric. Sets up "alternative model" as a legitimate move, not a novelty. |
| 5 | "Evidence for multiple BH populations includes the presence of a peak… at $\sim30-35~M_\odot$, which is seemingly too narrow or too unrealistically located to be explained by various formation channels" | **The gap, stated as a physical discomfort**, not as a literature omission. Then names the specific alternative theory (Pop III). Closes with the *this-work* sentence: "Motivated by these theories, we explore whether the $\sim30-35~M_\odot$ excess can be explained by a second power law rather than a peak…" |
| 6 | "Given that we are considering a population model for a formation channel which is possibly of dynamical origin…, we also explore…" | **The second question**, derived logically from the first. The high-mass search is presented as a *consequence* of the modelling choice, not as a separate fishing expedition. This is what makes the ~70 $M_\odot$ bump feel earned rather than found by trawling. |
| 7 | "In §II we present our method, in §III we show our results…" | **Roadmap**, one sentence, one clause per section. |
| 8 | "Uncertainties throughout this work are at the 90\% Credible Interval (CI)." | **Global numeric convention** as the last line of the introduction, so every later number can be quoted bare. |

**Notable absences:** no "the aim of this paper is"; no bulleted contributions list; no
paragraph explaining why gravitational waves are interesting; no results preview beyond what
the abstract already said.

---

## 5. Results-first technique

- **First appearance of the headline numbers:** abstract sentences 2 and 4. They then recur in
  §3 (Results) ¶3 and ¶5, and again in §7 (Conclusions). Nothing is withheld for suspense.
- **§3 opens with data selection in 5 lines** (FAR < 1/yr, 69 events, GW190814 excluded, LVK PE
  samples, LVK injection set) and then goes straight to "We show the main results of this paper
  in Fig. 1." Sample definition is treated as part of the result, not as method.
- **Methods are subordinated three ways:**
  1. *Outsourced to reviews and prior papers.* "We provide a brief summary below and we refer
     the reader to review articles for an in depth description of the framework." The fiducial
     simulation hyperparameters are not restated: "hence the fiducial choice for the
     hyper-parameter values for the simulated population described in [MaganaHernandez:2024uty]."
  2. *Compressed to one master equation.* Eq. (6) contains two power laws plus a Gaussian; the
     three models of the paper are then defined purely as limits — "if we set $f_{{\rm pl},2}=0$
     we obtain…", "we set the Gaussian component mixture weight to zero…". Three models, one
     equation, one paragraph. No separate subsection per model.
  3. *Stated as criteria, not procedures.* The selection-function treatment is 6 lines and its
     only operational content is a guard: "we make sure that $N_{\rm eff} > 4 N_{\rm obs}$ to
     provide an unbiased estimate of $\beta$." No sampler settings, no convergence diagnostics,
     no runtimes anywhere in the paper.
- **§2.2 explains how to read the statistic before any statistic is shown**: "In general if
  $\log_{10}\mathcal{B}^M_N > 0$, then model $M$ is favored… If $\log_{10}\mathcal{B}^M_N
  \approx 0$, then it is inconclusive…". Two sentences that pre-empt every misreading of a 0.24.
- **Main text vs appendix vs citation split:** main text = model definition, the numbers, the
  validation design and its false-positive rates. Appendix = corner plots only. Citation =
  hierarchical-inference formalism, PE priors, the injection campaign, the mock-catalog
  generator. Nothing that a reader needs to evaluate the claim lives outside the main text.

---

## 6. Voice

- **Person:** first-person plural throughout, and *load-bearing* — "we analyze", "we find",
  "we note", "we argue", "we consider", "we make sure". Roughly 60 occurrences of "we" in ~5,300
  words. "We find" is the workhorse for measurements; "**we argue**" is reserved for
  interpretation, a deliberate two-verb system: "we argue that PL1 is composed of isolated
  binaries, while PL2 may mostly arise from 1g+1g binaries."
- **Tense:** present tense for what the paper does and what the data say ("we find", "the model
  recovers"); present perfect only in the Conclusions opener ("we have explored"); past tense
  for events in the field ("69 high significance BBH mergers have been detected").
- **Active/passive:** dominantly active in results and interpretation. Passive appears where
  the agent is genuinely irrelevant — the mock-generation machinery ("Mass ratios are drawn from
  a power law", "Our simulations are constructed to closely follow…", "the bump is located at a
  mass too high…"). Rough count: ~85% active in §3–§5, closer to 50% in the simulation
  paragraphs of §4. Useful rule: **passive for pipeline, active for inference.**
- **Rhythm:** short declarative opener, then a long qualified sentence. E.g. §3¶3: "First, we
  note that the PLPL model from a qualitative perspective appears to be a good fit…" (24 words)
  → "More quantitatively, we can compute the Bayes factor…" (30 words) → a 60-word sentence
  carrying the numbers and their interpretation. Paragraphs are 3–6 sentences; several are 2.
- **Discourse markers do the structural work** in place of subsection headings: "First, we
  note…", "Second, we note…", "Again, we compute…", "Most interestingly…", "More
  interestingly…", "As a secondary check…", "Finally, if we extend…". A reader can follow the
  argument from these alone.
- **Hedging formulas actually used (verbatim):**
  1. "the data **marginally prefers** a mixture of two power laws"
  2. "This result **may imply** that the $\sim30-35~M_\odot$ feature represents the onset of a
     second population"
  3. "our results and validation simulations **potentially hint** that a sub-population of BBH
     mergers at around $70~M_\odot$ is real"
  4. "**conditional on our simulation assumptions** we see that we cannot reproduce our PLPLB
     GWTC-3 results"
  5. "**Although it may be tempting to identify** the feature at $\sim70~M_\odot$ as the
     pulsational PISN build-up, the bump is located at a mass too high…"
  6. (bonus) "**more data is needed** to robustly assess whether the PLPL model is indeed
     preferred"; "it is **challenging to draw definitive conclusions**".
  Note the gradation: *marginally prefers* < *potentially hints* < *we argue* < *we find*. The
  verb tells you the strength; the paper never needs the word "significant".

---

## 7. Numbers

- **Precision:** two significant figures at most, and usually one. Bayes factors to two decimals
  ($+0.24$, $+0.29$, $+0.05$), fractions as round percentages ("$\sim3\%$", "$\sim76\%$",
  "7\% of our simulations"), masses to the nearest few $M_\odot$ or as ranges
  ("$\sim65-70~M_\odot$", "$m_{\rm min,2}\sim18~M_\odot$", "$m_{\rm max,2}\sim45-100~M_\odot$").
  The tilde is used liberally and honestly — it signals "this is the scale, not the value".
- **Uncertainties:** asymmetric 90% CI superscripts, e.g. $58^{+32}_{-14}~M_\odot$,
  $32.0^{+1.9}_{-2.0}~M_\odot$, width $3.1^{+3.9}_{-1.8}~M_\odot$. **The CI convention is
  declared once, in the last line of the introduction**, and never repeated. Non-Gaussian
  constraints are quoted as intervals with an explicit confidence: "constrained within the
  range $[0.02, 0.08]$ at 90\% confidence"; one-sided statements as
  "$\mu_{\chi,1}>0$ at $>99.8$\% confidence", "$f_{\rm g}>0.01$ is at 96\% confidence".
- **Where numbers live:** inline in prose, always. There is no table in the paper. Figures carry
  distributions; the text carries the numbers that summarize them. A few numbers appear *only*
  in captions ("We find the contribution to the overall population is 1-4\%.").
- **Assumptions attached to numbers, in the same sentence:** "conditional on our simulation
  assumptions"; "given our assumptions on the simulation inputs"; "given the current size of the
  GWTC-3 catalog and its measurement uncertainty"; "Although with large uncertainties given the
  current statistics". The qualifier is never a separate sentence that could be quoted away.
- **A false-positive rate is quoted as a number, twice:** "We find that
  $\log_{10}\mathcal{B}^{\rm PLPL}_{\rm PLB}>0$ in 7\% of our simulations", and "only 1 of the
  realizations have a similar significance of $f_{\rm g}>0.01$ at 96\% confidence" (i.e. 1/100).
  This is the paper's substitute for a p-value and it is more legible than one.
- **Forecast:** exactly one, in the Conclusions, unquantified and honest: "We expect that the
  increasing number of BBH events detected through O4 and beyond will provide the observations
  required to determine whether the $70~M_\odot$ sub-population is actually present."

---

## 8. Figures

**12 figures, 0 tables.** Types:
- 4 "population distribution with 90% band" panel-pairs ($m_1$ top, $m_2$ bottom): Figs. 1
  (GWTC-3 fits vs LVK reference), 3 (per-event population-informed posteriors), 4 (simulated
  catalogs vs truth), 8 (per-component reconstructions).
- 2 corner plots in the main text (Fig. 2, the bump's $\{\mu_m,\sigma_m,f_{\rm g}\}$; Fig. 7,
  the $\chi_{\rm eff}$ hyperparameters) + 3 full corner plots in the appendix.
- 2 histogram-of-realizations figures — the validation workhorses (Fig. 5, Bayes factors;
  Fig. 6, $f_{\rm g}$ posteriors for all 100 mocks with the GWTC-3 posterior overplotted).
- 1 CDF panel used purely to make a symmetry argument legible (Fig. 9, lower panel).

Only Fig. 1 is full-width (`figure*`); everything else is single-column. **The main result is
the only figure given the whole page width** — layout itself signals priority.

Caption style: one sentence naming what is plotted and with which model/colour, plus at most
one sentence of guidance. Colour keys are inline in parentheses. Captions never interpret
beyond a single quantitative statement.

**Caption 1 (Fig. 4, validation):**
> "Primary (top panel) and secondary (lower panel) component mass distributions (90\%
> confidence interval) from the median mass distribution reconstructions for each of the 100
> simulated GWTC-3-sized catalogs described in Section IV for the PLB (pink), PLPL (blue) and
> PLB (green) models. For reference, we show the simulated GWTC-3 Powerlaw+Peak like population
> as the solid black line."

*Why it works:* it states the estimator ("median … reconstructions"), the interval (90%), the
sample size and provenance ("each of the 100 simulated GWTC-3-sized catalogs described in
Section IV"), the colour key, and — decisively — **the truth curve is named as such** ("For
reference … the solid black line"). A reader can judge recovery without reading the section.
(It also contains the paper's one visible typo: "PLB" is written twice where the third model
is PLPLB — a reminder to check macro expansion in colour keys.)

**Caption 2 (Fig. 2, the bump):**
> "Marginalized posterior distribution on the location $\mu_m$, width $\sigma_m$ and mixture
> weight $f_{\rm g}$ for the PLPLB model. We find the contribution to the overall population is
> 1-4\%."

*Why it works:* three parameters named with their symbols in plotting order, then **one number
the reader should leave with**. The caption is self-sufficient as a result. The 1–4% appears
nowhere else in that form — captions are allowed to carry a unique number, but only one.

Other captions worth copying for their economy: "The vertical lines show the corresponding
GWTC-3 results." (Fig. 5) — the data-vs-mock comparison explained in nine words. "We also show
the CDF($\chi_{\rm eff}$) to more easily demonstrate the symmetry (about $\chi_{\rm eff}=0$)
for each sub-population." (Fig. 9) — states *why* the second panel exists.

---

## 9. Caveats and assumptions

Limitations are placed **inside the sentence that makes the claim**, in its final clause,
rather than collected in a "Limitations" section (there is none). Examples:

- "we determine that the PLPL model describes the GWTC-3 observations **similarly well to the
  PLB model, if not marginally better**."
- "This shows that given the current size of the GWTC-3 catalog and its measurement
  uncertainty, **we cannot confidently claim** that PLPL is a preferred model over PLB."
- "Thus, our results and validation simulations **potentially hint** that a sub-population of
  BBH mergers at around $70~M_\odot$ is real, **rather than a spurious feature due to the small
  number statistics and the specific representation we observed with GWTC-3**."
- "We see that this can happen, **however, conditional on our simulation assumptions** we see
  that we cannot reproduce our PLPLB GWTC-3 results."
- "Note that other formation channels are also expected to produce BBHs within this population,
  **so we cannot assume that the entirety of this population … originates in the field**."
- "While such redshift evolution **may not be physical due to the limited detector horizon**
  especially at lower masses, this finding may reconcile our and previous works' observations…"
- "**However, more data in this region of parameter space is needed** to make definitive
  conclusions regarding the origin of BHs in this sub-population."

**Ratio:** roughly one caveat clause per quantitative claim — I count ~12 explicit hedges
against ~14 numeric claims, so ≈0.85:1. Critically, **the caveats are specific**: each names
the thing that could break the result (catalog size, simulation assumptions, detector horizon,
spin measurement error), never the generic "further work is needed" alone. The paper also
concedes against itself in Results: "the PLPL model shows degeneracies with $m_{\rm max,2}$ and
$f_{\rm pl,1}$ demonstrating the necessity for an extended model" — a weakness of their own
simpler model, used as motivation rather than buried.

Two robustness checks are reported in single sentences and then dropped: "The presence of the
$\sim70~M_\odot$ feature is robust with respect to the prior on the position $\mu_m$ of the
Gaussian peak," and "Finally, if we extend the PLPLB model to have a second Gaussian
component… the conclusions of this study are not changed." **A negative check costs one
sentence and buys a lot of credibility.**

---

## 10. Transitions and cadence

**How sections open.** Every section opens with a one-sentence statement of what it will do,
in the active voice, with no throat-clearing:
- §2: "We use hierarchical Bayesian population inference to simultaneously infer the parameters…"
- §3: "In this section, we show the inferred population distributions for the masses of BBH
  mergers using the models we described in Section II C."
- §4: "We validate our results using a large set of mock GW observations drawn from a known
  fiducial population model." — and the very next sentence states the null hypothesis being
  tested: "Specifically, we want to test whether a PLPL or PLPLB model can spuriously fit the
  data given a BBH population which actually follows a PLB model."
- §5: opens by *defining the new observable* ($\chi_{\rm eff}$) before using it.
- §6.1: "We start by discussing the first power law, PL1, which appears to be consistent with…"

**A section title as a question.** §5 is titled "**Does spin tell us anything?**" — informal,
honest, and it sets up an answer that is partly "not much". This is the paper's clearest
signal that it is written for readers, not for referees.

**How sections close.** On the limit of the current data, pointing at what would resolve it:
- §3 closes on a null robustness check ("the conclusions of this study are not changed").
- §4 closes on the quantified false-positive rate ("only 1 of the realizations…").
- §5 closes on the physical reading of the least constrained component.
- §6.2 closes by naming an open thread: "An involved study exploring the redshift and
  $\chi_{\rm eff}$ dependence on the mass distribution is ongoing."

**Closing paragraph structure (Conclusions ¶2), five moves in five sentences:**
1. Restate the robust finding — "We also find that adding a Gaussian component above
   $40~M_\odot$ robustly finds a sub-population located around $65-70~M_\odot$."
2. Explain *why the method could see it when others could not* — "It is possible that by
   allowing a second power law to exist with a steeper slope than the first PL, one enables the
   detection of a sub-population that was otherwise buried under a single, flatter power law."
3. Physical interpretation — "This sub-population may arise from second generation hierarchical
   mergers…"
4. The validation, in one line, then the concession — "Our simulations show that a similar
   feature cannot be recovered by chance coincidence when the underlying population follows a
   PLB. However, more data is required to confidently claim that the PLPLB model is preferred…"
5. Forecast + payoff to a *different* subfield — "If confirmed, such sub-population would [be] a
   smoking gun about the dynamical origin of high-mass LVK BBHs, and may represent a new
   valuable feature to improve cosmological parameters constraints through spectral standard
   siren measurements."

The last sentence hands the result to another community. That is the closing move to copy.

---

## 11. Ten exemplary sentences (verbatim)

1. "Motivated by these theories, we explore whether the $\sim30-35~M_\odot$ excess can be
   explained by a second power law rather than a peak over a broader distribution by using the
   confidently detected BBH observations from the GWTC-3 catalog."
   — *The whole paper in one sentence: motivation, the binary question, the data. "A rather B"
   is the cleanest possible statement of a modelling question.*

2. "Specifically, we want to test whether a PLPL or PLPLB model can spuriously fit the data
   given a BBH population which actually follows a PLB model."
   — *States the null hypothesis of the validation in plain words. Any mock-catalog section
   should contain a sentence of exactly this shape.*

3. "We find that $\log_{10}\mathcal{B}^{\rm PLPL}_{\rm PLB}>0$ in 7\% of our simulations."
   — *A false-positive rate as a bare number. No adjectives, no interpretation — the next
   sentence does that.*

4. "This shows that given the current size of the GWTC-3 catalog and its measurement
   uncertainty, we cannot confidently claim that PLPL is a preferred model over PLB."
   — *The authors argue against their own first result. Doing this early makes the second
   result believable.*

5. "Thus, our results and validation simulations potentially hint that a sub-population of BBH
   mergers at around $70~M_\odot$ is real, rather than a spurious feature due to the small
   number statistics and the specific representation we observed with GWTC-3."
   — *The headline claim, with its exact alternative hypothesis named in the same sentence.*

6. "The primary masses of five other events contribute significantly to the peak, implying that
   the results found for the PLPLB model are not only driven by a single outlier event."
   — *Pre-empts the obvious referee objection (GW190521) with a count, not a defence.*

7. "In general if $\log_{10}\mathcal{B}^M_N > 0$, then model $M$ is favored over model $N$. …
   If $\log_{10}\mathcal{B}^M_N \approx 0$, then it is inconclusive if one model is preferred
   over the other, and therefore both are equally good models at representing the underlying
   population given available data."
   — *Teaches the reader to read the statistic before showing them a small one. Disarms the
   "0.24 is nothing" reaction without special pleading.*

8. "Although it may be tempting to identify the feature at $\sim70~M_\odot$ as the pulsational
   PISN build-up, the bump is located at a mass too high to be at the lower edge of the PISN
   mass gap, which is expected to be around $45~M_\odot$ and likely below $56~M_\odot$
   considering reasonable variations in metallicity and uncertainties in the CO reaction rates."
   — *Kills the wrong interpretation first, with numbers, before offering the right one.*

9. "Note that this is different from a broken power law as the amplitude of the second power law
   is free to vary, such that this second population does not necessarily need to be connected
   to the one peaking at $\sim10~M_\odot$."
   — *One sentence distinguishing the new model from the nearest familiar one — physics, not
   parameter counting.*

10. "If confirmed, such sub-population would [be] a smoking gun about the dynamical origin of
    high-mass LVK BBHs, and may represent a new valuable feature to improve cosmological
    parameters constraints through spectral standard siren measurements."
    — *Conditional, then two distinct payoffs, one outside the paper's own subfield. The ideal
    last sentence.*

---

## 12. Transferable DO / DON'T — for a dark-siren / mock-catalog / H0-inference validation paper

**DO**

1. **Put the headline number in abstract sentence 2 or 3**, with its hedge attached in the same
   clause ("the data marginally prefers … with a Bayes Factor of 0.24"). Never make a reader
   reach §5 for the number.
2. **Declare the uncertainty convention once, as the last line of the introduction** ("Uncertainties
   throughout this work are at the 90% Credible Interval"), then quote every later number bare.
3. **Give validation its own top-level section, sized like Results.** For an H0 paper: the
   mock-catalog recovery test is not an appendix, it is the argument.
4. **Open the validation section by stating the null hypothesis in one plain sentence** — "we
   want to test whether [our method] can spuriously [produce the claimed shift] given a
   catalog which actually follows [the truth model]".
5. **Quote a false-positive / bias rate as a number out of the realization count** ("in 7% of
   our simulations", "only 1 of the realizations"). For H0: "N of 100 matched mocks reproduce a
   shift as large as the one measured."
6. **Define the whole model family as one equation, then define each variant as a limit of it**
   ("if we set $f_{\rm pl,2}=0$ we obtain…"). Three models, one equation, one paragraph.
7. **Teach the reader to read your statistic before you show them a value of it.** If the
   headline is a σ, a bias in units of σ, or a Bayes factor, spend two sentences on what
   counts as decisive *before* revealing the number.
8. **Use two verbs, consistently: "we find" for measurements, "we argue" for interpretation.**
   Never let them blur.
9. **Report the negative robustness checks in one sentence each and move on** ("robust with
   respect to the prior on $\mu_m$"; "the conclusions of this study are not changed"). Cheap
   credibility.
10. **Attach the assumption to the number in the same sentence** — "conditional on our simulation
    assumptions", "given the current size of the catalog and its measurement uncertainty".
11. **Name the truth curve in the caption of every recovery figure** ("For reference, we show the
    simulated … population as the solid black line").
12. **Let one number live only in a caption**, if it is the single takeaway of that figure
    ("We find the contribution to the overall population is 1-4%.").
13. **Quarantine astrophysical interpretation into a late section**, subdivided by the physical
    populations/regimes, after all numbers are settled.
14. **Identify which individual events / tracers drive the result**, with a stated criterion
    ("$m_1$ or $m_2$ larger than $55~M_\odot$ and … at least 50\% of posterior samples above
    our threshold"), to pre-empt the "one outlier" objection. For a dark-siren paper: which
    events / which galaxies / which redshift shell drives the H0 shift.
15. **Argue against your own weaker result before defending your stronger one.**
16. **End on a conditional plus a payoff to a neighbouring subfield**, and name the future data
    that will settle it.
17. **Use discourse markers as the skeleton** — "First… Second… Again… Most interestingly… As a
    secondary check… Finally…" — instead of proliferating subsections.
18. **Give the main result the only full-width figure.**

**DON'T**

1. **Don't open with the field's importance.** Paragraph 1 is quantitative field state (how many
   events, how many expected) in two sentences.
2. **Don't restate formalism available in review articles** — cite the reviews and give the one
   equation you actually manipulate.
3. **Don't put sampler settings, convergence diagnostics, wall-clock times, or software
   versions in the main text.** This paper has none. The only computational statement is a
   physically-motivated guard: "$N_{\rm eff} > 4N_{\rm obs}$ to provide an unbiased estimate".
4. **Don't hide anything load-bearing in the appendix.** The appendix here is three corner plots
   and zero prose.
5. **Don't build a "Limitations" section.** Put each caveat in the final clause of the claim it
   limits, so it cannot be quoted away.
6. **Don't hedge generically.** "More data is needed" only ever appears attached to a specific
   thing more data would settle.
7. **Don't over-precise the numbers.** One or two significant figures, tildes for scales, ranges
   where the posterior is broad.
8. **Don't use a table when prose will do** — this paper has none, and loses nothing.
9. **Don't inflate a marginal result with adjectives.** The word "significant" appears once, and
   about events contributing to a peak, not about evidence.
10. **Don't separate the "what we did" from "what it means" inside a paragraph** — one job per
    paragraph, and the interpretive paragraphs are visibly marked by "we argue".
11. **Don't preview results in the roadmap paragraph.** One clause per section, no numbers.
12. **Don't write about the workflow.** There is not one sentence about pipelines, gates,
    reruns, or code organization anywhere in 14 pages.
