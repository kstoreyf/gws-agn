# Style brief — arXiv:2111.06445 (Palmese, Bom, Mucesh & Hartley)

Source read: full text of `https://ar5iv.labs.arxiv.org/html/2111.06445` (arXiv v1, Nov 2021),
plus arXiv abstract page metadata. Every section, every figure/table caption, and the
acknowledgements were read. **There are no appendices in this version** — that is itself a
style datum (see §5 below): all method detail is either in the 940-word main-text Method
section or delegated to citations. Published as ApJ 943, 56 (2023), DOI
10.3847/1538-4357/aca6e3.

---

## 1. Identity

- **Title:** "A standard siren measurement of the Hubble constant using gravitational wave
  events from the first three LIGO/Virgo observing runs and the DESI Legacy Survey"
  (23 words; names the *measurement*, the *data*, and the *survey* — no method jargon, no
  colon-and-subtitle construction, no "I." / "II." series marker).
- **Authors:** A. Palmese (UC Berkeley; NASA Einstein Fellow), C. R. Bom (CBPF / CEFET-RJ),
  S. Mucesh (UCL), W. G. Hartley (UCL / Geneva). Four authors — a small-team paper, not a
  collaboration paper, and the voice reflects that.
- **Venue:** submitted 11 Nov 2021, astro-ph.CO (cross-list astro-ph.HE); published ApJ.
- **Length:** 14 pages, ~8,200 words of body text (Introduction → Conclusions), 5 figures,
  2 tables, ~110 references, 5 numbered equations.
- **Subject:** a dark-standard-siren (statistical host) measurement of H0 from 8 well-localized
  LIGO/Virgo events cross-matched to the DESI Legacy Survey photometric galaxy catalog,
  combined with the GW170817 bright siren.
- **Keywords line:** "catalogs — cosmology: observations — gravitational waves — surveys".

### Word budget by section (body only, math stripped)

| Section | words | ≈% of body | paragraphs |
|---|---|---|---|
| 1 Introduction | 1329 | 16% | 6 (incl. roadmap para) |
| 2 Data (preamble) | 46 | <1% | 0 (heading + Fig. 1 only) |
| 2.1 The LIGO/Virgo GW data | 561 | 7% | 3 + Table 1 |
| 2.2 The DESI Imaging data | 480 | 6% | 3 + bullet list |
| 2.2.1 Redshifts truth table | 532 | 7% | 3 |
| 2.2.2 Legacy survey photo-z's | 1230 | 15% | 4 + Fig. 2, Fig. 3 |
| 2.2.3 Joint z–L PDFs with Random Forests | 566 | 7% | 3 |
| 2.2.4 Galaxy fakes | 168 | 2% | 1 |
| 3 Method | 942 | 12% | 6 + Eqs. 2–5 |
| 4 Results and Discussion | 1723 | 21% | 8 + Figs. 4–5 + Table 2 |
| 5 Conclusions | 597 | 7% | 2 |

**The single most transferable structural fact:** *Data (3,540 words, 43%) is more than
three times the size of Method (942 words, 12%).* This is a paper whose novelty is the
**inputs** — a better galaxy catalog, validated photo-z's — so the input validation gets the
page count, and the inference machinery is compressed to five equations plus "a full
derivation can be found in [Chen et al. 2018]". If your novelty is the *mock catalog* and the
*validation*, that is where your words go, not in re-deriving the likelihood.

---

## 2. Global architecture

Exact sequence: **Abstract → 1 Introduction → 2 Data (2.1 GW data; 2.2 imaging data, with
2.2.1 truth table, 2.2.2 photo-z's, 2.2.3 joint z–L PDFs, 2.2.4 galaxy fakes) → 3 Method →
4 Results and Discussion → 5 Conclusions → Acknowledgements → References.**

Jobs each section does for the argument:

- **Abstract** — states the measurement, the two headline numbers, and the one comparative
  claim that makes the paper worth publishing (8 dark sirens ≈ 1 bright siren in precision).
- **§1 Introduction** — establishes that standard sirens exist and matter (H0 tension), then
  narrows to the *specific deficiency of prior dark-siren work* (incomplete galaxy catalogs,
  badly localized events) which this paper removes. The intro's real job is to make the
  *selection choices* (best-localized events + complete catalog) look inevitable rather than
  arbitrary.
- **§2 Data** — the heart of the paper. Its job is to earn trust in the redshifts: survey
  provenance → star/galaxy separation → training/truth table → photo-z validation metrics
  vs. an external benchmark (DES requirements) → an alternative photo-z method as a
  robustness lever → honest handling of the uncovered sky (fakes). Every subsection ends in a
  number a referee can check.
- **§3 Method** — deliberately thin: Bayes' theorem, the marginalized posterior, the selection
  term β(H0), the Monte Carlo that computes it, one paragraph of self-criticism about the
  priors, and the event-combination product. No derivations.
- **§4 Results and Discussion** — headline posteriors, then interpretation of *structure* in
  the posteriors (peaks = line-of-sight overdensities), then a chain of robustness checks
  (skymap choice, luminosity weighting, Ωm), then a literature comparison that argues *why*
  fewer events gave a tighter constraint, then the bright+dark combination and the final
  number.
- **§5 Conclusions** — two paragraphs: (1) restate both numbers with the one superlative claim
  and its caveat; (2) what would improve it, on a named timeline, and which systematics must
  be studied *before* the precision makes them matter.

No "Discussion" section separate from Results; no separate "Systematics" section — robustness
is interleaved into Results as short paragraphs. No appendix.

---

## 3. Abstract anatomy

Verbatim (math rendered inline):

> We present a new constraint on the Hubble constant H0 using a sample of well-localized
> gravitational wave (GW) events detected during the first three LIGO/Virgo observing runs as
> dark standard sirens. In the case of dark standard sirens, a unique host galaxy is not
> identified, and the redshift information comes from the distribution of potential host
> galaxies. From the third LIGO/Virgo observing run detections, we add the asymmetric-mass
> binary black hole GW190412, the high–confidence GW candidates S191204r, S200129m, and
> S200311bg to the sample of dark standard sirens analyzed. Our sample contains the top 20%
> (based on localization) GW events and candidates to date with significant coverage by the
> Dark Energy Spectroscopic Instrument (DESI) Legacy Survey. We combine the H0 posterior for
> eight dark siren events, finding H0 = 79.8(+19.1)(−12.8) km s^-1 Mpc^-1 (68% Highest Density
> Interval) for a prior in H0 uniform between [20,140] km s^-1 Mpc^-1. This result shows that a
> combination of 8 well-localized dark sirens combined with an appropriate galaxy catalog is
> able to provide an H0 constraint that is competitive (~20% versus 18% precision) with a
> single bright standard siren analysis (i.e. assuming the electromagnetic counterpart) using
> GW170817. When combining the posterior with that from GW170817, we obtain
> H0 = 72.77(+11.0)(−7.55) km s^-1 Mpc^-1. This result is broadly consistent with recent H0
> estimates from both the Cosmic Microwave Background and Supernovae.

Sentence-by-sentence:

1. *"We present a new constraint…"* — **claim first.** Result type, parameter, data, and the
   method label ("dark standard sirens") in one sentence. No throat-clearing about the
   importance of cosmology.
2. *"In the case of dark standard sirens, a unique host galaxy is not identified…"* —
   **one-sentence definition of the method for a non-specialist.** This is the entire
   "background" allotment of the abstract: 26 words.
3. *"From the third LIGO/Virgo observing run detections, we add…"* — **what is new relative to
   the previous paper**, named event by event. Delta-from-prior-work, not from zero.
4. *"Our sample contains the top 20% (based on localization)…"* — **the selection criterion,
   quantified.** Pre-empts "why these events?" before the reader can ask.
5. *"We combine the H0 posterior for eight dark siren events, finding H0 = 79.8(+19.1)(−12.8)…"*
   — **headline number**, with interval type (68% HDI) and prior range attached *in the same
   sentence*. Never a number without its prior in this paper.
6. *"This result shows that a combination of 8 well-localized dark sirens … is competitive
   (~20% versus 18% precision) with a single bright standard siren…"* — **the interpretive
   payload**: the number is converted into a statement about *what dark sirens are worth*.
   This is the sentence that justifies publication.
7. *"When combining the posterior with that from GW170817, we obtain H0 = 72.77(+11.0)(−7.55)…"*
   — **second headline number** (the combined one).
8. *"This result is broadly consistent with recent H0 estimates from both the Cosmic Microwave
   Background and Supernovae."* — **implication, stated with deliberate modesty.** The paper
   does not claim to resolve the tension; it reports consistency with both sides.

Shape: 8 sentences, 2 of context, 2 of scope/selection, 2 numbers, 1 comparison, 1 implication.
**Zero sentences of method detail.** Nothing about random forests, BAYESTAR, or selection
effects, even though those consume half the paper.

---

## 4. Introduction anatomy — paragraph function map

Six paragraphs, ~1,330 words. Function of each:

- **¶1 (Field context, widest zoom).** First GW detection → multi-messenger astronomy →
  GW170817 → the menu of multi-messenger physics (tests of GR, NS equation of state, primordial
  BHs) → *funnel to standard sirens*, with Schutz (1986) and a plain-language statement of
  the idea: "given a gravitational wave detection, it is possible to measure a luminosity
  distance for the event, and if that is combined with the redshift of the galaxy that hosted
  the merger, gravitational wave events can be used to probe the distance-redshift relation."
  Note: the mechanism is explained in ordinary words *before* any formalism appears.
- **¶2 (Taxonomy).** Bright vs. dark sirens defined; state of play for bright sirens; the
  GW190521 candidate-counterpart controversy dispatched in one sentence with citations on both
  sides. Establishes that bright sirens are essentially N=1.
- **¶3 (State of the art in the relevant sub-branch + the structural argument).** Lists prior
  dark-siren measurements, concedes their per-event inferiority, then flips it: dark events
  "outnumber those with a counterpart by at least a factor of 10". Closes with the paper's
  strategic thesis: "We therefore take the approach of combining dark sirens with the much
  rarer bright sirens."
- **¶4 (Why the reader outside GW should care).** The H0 tension, both camps cited, and a
  forward-looking hook (few-per-cent H0 in 5 years). This is the "stakes" paragraph and it is
  placed *fourth*, after the field has been set up — not first.
- **¶5 (This work — the design choices, with reasons).** "In this work, we present a new
  measurement of the Hubble constant using 8 dark sirens, combined with GW170817…" followed by
  three explicit numbered-in-prose justifications for restricting to well-localized,
  well-covered events ("First… Better localized events also… Lastly…").
- **¶5b/6 (The gap, named against specific papers).** The most instructive paragraph: it
  identifies *by name and by number* what prior work got wrong — GLADE "is complete out to a
  distance of ~37 Mpc … but largely incomplete at the distances of the dark sirens (all at
  >200 Mpc)" — and then states the two improvements ("First, we improve on the results in
  Palmese et al. (2020) by adding new events from O3. We also improve on the analysis in
  Finke et al. (2021) by considering a more suitable galaxy catalog."). The gap is a *number*,
  not an adjective.
- **Roadmap paragraph (3 sentences).** Section map + assumed cosmology + error-bar convention:
  "We assume a flat ΛCDM cosmology with Ωm = 0.3 and H0 values in the 20−140 km s^-1 Mpc^-1
  range. When not otherwise stated, quoted error bars represent the 68% credible interval (CI)."
  The conventions live here, once, and are never re-litigated.

Pattern to copy: **context → taxonomy → prior work's structural limitation → stakes → this work
+ why these choices → the specific quantified gap → roadmap/conventions.**

---

## 5. Results-first technique

- **Where the headline numbers first appear:** in the abstract, sentences 5 and 7. They appear
  a *second* time in §4 (¶4: "The maximum a posteriori with the 68% CI is 79.8(+19.1)(−12.8)
  km s^-1 Mpc^-1 from this combination."; ¶8: "Our final H0 constraint from this analysis is
  72.77(+11.0)(−7.55) km s^-1 Mpc^-1."), a *third* time in Table 2, and a *fourth* time in the
  first two sentences of the Conclusions. Four exposures, identical digits every time.
- **Methods are subordinated three ways:**
  1. **By citation.** "The formalism used in this work is adapted from Chen et al. (2018),
     following Soares-Santos & Palmese et al. (2019) and Palmese et al. (2020). The H0
     posterior derivation is described in detail in those works." One sentence replaces a
     derivation. Likewise: "Note this is the same as Eq. 15 in Chen et al. (2018), where a full
     derivation can be found."
  2. **By word budget.** 5 equations, 942 words, no appendix. Eq. 2 is Bayes' theorem; Eq. 3 is
     the marginalized posterior stated *without* derivation, with every symbol glossed in the
     following two sentences; Eq. 4 is β(H0); Eq. 5 is the product over events.
  3. **By outsourcing validation to metrics.** The photo-z work is presented as *numbers a
     reader can benchmark* (σ_NMAD = 0.01; bias < 0.004; σ68 < 0.02; 2σ/3σ outlier fractions
     4.5% and 0.3%; DES requirement σ68 < 0.12) rather than as a description of what was done.
- **Main text vs. appendix vs. citation split:** main text carries (a) everything a reader
  needs to judge the *inputs*, (b) the posterior equations, (c) the Monte Carlo recipe for
  selection effects at recipe granularity ("We simulate 70,000 BBH mergers and compute β(H0)
  for 20 values of H0 within our prior range… A detection is made when at least 2 detectors
  reach a single-detector SNR > 4 and the network SNR is > 12."). Nothing goes to an appendix.
  Everything derivational goes to citations. Software is named inline with citations
  (BAYESTAR, LALSuite, GALPRO, The Tractor, TOPCAT) rather than described.
- Note the *inverted* emphasis relative to a typical methods paper: the pipeline is described
  in the ~1 paragraph it takes to reproduce it, while the photometric-redshift validation —
  the thing that could actually be wrong — gets 1,230 words and two figures.

---

## 6. Voice

- **Person:** first-person plural throughout — 118 instances of "we/We" and 24 of "our" in
  ~8,200 words (≈1 per 58 words). "We present", "We select", "We use", "We find", "We note",
  "We stress", "We have tested", "We prefer to provide". The authors are visibly making
  choices; nothing is described as having happened by itself.
- **Tense:** present for what the paper does and what results say ("We use…", "We find…",
  "The most constraining event is GW190412"); present perfect for completed work reported as
  established ("We have tested that…", "Photometric redshifts have been computed by Zhou et al.
  (2020)"); past only for history ("Since the first detection of gravitational waves in 2015…",
  "In this paper, we have presented…").
- **Active/passive:** predominantly active. ~85 passive-voice markers in 254 sentences (~1 in 3
  sentences contains one), and they cluster where the actor is genuinely irrelevant or is a
  third party ("Source detection and photometry … is performed with the software package The
  Tractor"; "Selection effects are taken into account in the β(H0) term as follows"). All
  *decisions* are active and owned.
- **Sentence rhythm:** mean 28 words, median 25, 10th percentile 12, 90th percentile 48.
  16% of sentences are under 15 words and 16% over 40 — a deliberately mixed cadence: a long
  qualifying sentence is usually followed by a short declarative. Example pair: a 60-word
  sentence about multi-peaked posteriors, then "For the DESI Legacy Survey, we find
  σ_NMAD = 0.01." Short sentences almost always carry a number.
- **Jargon discipline:** every acronym is expanded on first use (GW, FAR, NMAD, PIT, copPIT,
  KDE, HDI, NGC/SGC, EOB/PN/NR). Method names are given as plain descriptions first, labels
  second: "…so they are referred to as SEOBNRv4PHM".
- **Hedging formulas actually used (verbatim):**
  1. "This result is broadly consistent with recent H0 estimates from both the Cosmic Microwave
     Background and Supernovae."
  2. "…their classification as BBH is confident (>99%), that they are very likely to be real
     GW events of astrophysical origin."
  3. "We stress that it is expected that the H0 posteriors from individual dark standard sirens
     present multiple peaks…"
  4. "We prefer to provide the more conservative result that does not include the luminosity
     weighting."
  5. "The origin of the BBH mergers that LIGO/Virgo have detected is unclear, and it is likely
     that multiple formation mechanisms are at play."
  Note the pattern: hedges attach to *interpretation and provenance*, never to the authors'
  own measured numbers. There is no "may possibly suggest" construction anywhere.

---

## 7. Numbers

- **Precision conventions.** Central values carry only the digits the posterior supports:
  79.8(+19.1)(−12.8) and 72.77(+11.0)(−7.55). Precision percentages are rounded hard —
  "~20%", "18%", "12%", "a 1% improvement on the precision", "improving the precision from
  GW170817 by 28%". Orders of magnitude are given as such: "of the order of ~10^-3 Gpc^3",
  "O(100) [events] for a O(1%) precision on H0". The tilde is used freely and honestly.
- **Uncertainties.** Always asymmetric where the posterior is asymmetric; the interval type is
  always declared — "(68% Highest Density Interval)" in the abstract, "Quoted uncertainties
  represent 68% HDI around the maximum of the posterior" in Table 2's caption, and a blanket
  statement in the roadmap paragraph. The **prior range travels with every H0 number**, in the
  abstract, in Table 2 (a dedicated "Prior" column), and in the comparison to Finke et al.:
  "We also note that our H0 prior is slightly larger than theirs ([30,140] km s^-1 Mpc^-1),
  which could be misleading when comparing the precision of measurements."
- **A precision-normalizing column.** Table 2 reports σ_H0/H0 *and* σ_H0/σ_prior. The second
  column exists purely to stop a reader from mistaking prior width for information — an
  unusually honest move worth stealing for any prior-dominated inference.
- **Inline vs. table vs. figure.** Headline H0 values: inline *and* table *and* figure. Per-event
  properties (d_L, area, volume, FAR, skymap reference): Table 1 only. Validation metrics
  (σ_NMAD, bias, σ68, outlier fractions): inline in prose, with the figure showing the
  distribution behind them. Per-event H0 posteriors: figure only — no table of eight numbers,
  because individually they are uninformative and the paper says so.
- **Assumptions attached to numbers.** Forecasts are always dated and sourced: "potentially
  reaching a few per cent uncertanty in H0 in 5 years (Borhanian et al., 2020)"; "the upcoming
  LIGO/Virgo/KAGRA observing run, expected to start in the second half of 2022"; "when a 2%
  statistical precision on H0 will become possible". Model dependence is attached to the final
  number in the Conclusions rather than left implicit.

---

## 8. Figures

Five figures, two tables. Types:

1. **Fig. 1** — sky map (Mollweide) of the 8 events' 90% CI contours with the DESI footprint
   gap shaded. *Job:* justify the sample selection visually in one glance.
2. **Fig. 2** — two-panel photo-z validation: (left) density of Δz vs z_spec with ±2σ68 lines;
   (right) binned mean bias and σ68. *Job:* the "our redshifts are good" exhibit.
3. **Fig. 3** — per-event dN/dz *with a uniform-comoving-density dN/dz subtracted*, with the
   event distance marked. *Job:* show the physical structures that will produce the posterior
   peaks. This is the paper's cleverest figure: it makes a later interpretive claim visible in
   advance.
4. **Fig. 4** — per-event H0 posteriors, one line each.
5. **Fig. 5** — the money figure: dark-siren combination, GW170817 posterior, joint posterior,
   with Planck and R21 1σ bands as vertical shaded reference regions.

**Caption 1 (Fig. 3), verbatim:**
> "Redshift distribution of galaxies in the 90% CI area of the dark siren events analyzed in
> this work. The distribution is subtracted with a dN/dz with uniform number density, in order
> to highlight the presence of overdensities and underdensities along the line of sight. The
> dashed blue line shows the distribution using the photometric redshifts point estimates from
> the DESI Legacy Survey presented in Zhou et al. (2020), the dot-dash red line shows the same
> redshifts when their uncertainty is taken into account as a Gaussian error. The grey vertical
> lines represent the luminosity distance of each GW event marginalized over the entire sky,
> assuming an H0 of 70 km s^-1 Mpc^-1, and the shaded regions are the 1σ uncertainties
> considering the same H0. These regions are only showed for reference."

Why it works: sentence 1 says *what is plotted*; sentence 2 says *what transformation was
applied and why* ("in order to highlight…") — the reader is told the purpose of a non-obvious
choice instead of having to infer it; sentences 3–4 identify every line and shading by style;
the last sentence — "These regions are only showed for reference" — pre-empts the
over-interpretation that the grey band is a measurement. Self-contained: readable without the
body text.

**Caption 2 (Fig. 5), verbatim:**
> "Hubble constant posterior distributions. The blue line shows the result from the combination
> of all dark sirens considered in this paper. The shaded grey posterior represents the GW170817
> standard siren result adapted from Nicolaou et al. (2020), which makes use of the presence of
> the electromagnetic counterpart. The black posterior represents the final result of this work,
> showing the joint constraint from both the bright (i.e. GW170817) and the dark standard sirens.
> The vertical dashed lines show the 68% region for each posterior. For reference, the 1σ Planck
> Collaboration et al. (2018) and Riess et al. (2021) (R21) constraints on H0 are also shown as
> the vertical shaded regions. Posteriors are arbitrarily rescaled only for visualization
> purposes."

Why it works: it labels one curve as "the final result of this work" so the reader cannot miss
the headline; it credits the provenance of the borrowed GW170817 posterior *inside the caption*;
it states the interval convention for the dashed lines; and it closes with the disclosure that
the vertical scaling is arbitrary — a small honesty that stops a reader from reading relative
heights as relative evidence.

Caption style rules extracted: 3–7 sentences; declarative fragments for the opener; every
line/color/shading mapped to a meaning; the *reason* for any non-obvious plotting choice given
inline; a final defensive sentence disclaiming over-reading. Table captions do the same job for
columns (Table 1 explains why FAR ranges appear and why candidate FARs are not comparable;
Table 2 defines both precision columns).

---

## 9. Caveats and assumptions

Caveats are short, specific, quantified, and placed immediately after the claim they qualify —
never quarantined in a "limitations" list. Examples verbatim:

- Model dependence, in Results: "We note that the results presented here are valid under a
  precise cosmological model, the Flat ΛCDM model. This is unlike the H0 measurement from
  GW170817 Abbott et al. (2017a), which is nearby enough to only be sensitive to the Hubble
  constant." — followed *immediately* by the test that bounds the damage: "We have tested that
  our results are not significantly affected by changes to Ωm within the 2σ interval found by
  the DES-only measurements presented in Abbott et al. (2019b)…"
- Self-criticism of their own selection function: "We note that the selection effects
  calculation could be improved by considering the same priors as those assumed in the GW
  likelihood… However, Gray et al. (2019) show that if the mass distribution takes the form of
  a power law, no prior correction is required for this difference in assumptions."
- Conservative choices declared as choices: "We find that the weighting produces a slight
  improvement on the constraints, yielding a 1% improvement on the precision. We prefer to
  provide the more conservative result that does not include the luminosity weighting."
- Honest reporting of a validation that partly failed: "The copula probability integral
  transform (copPIT) distribution and the Kendall calibration are poor compared to their
  univariate counterparts. This indicates that the joint PDFs are not as accurate as the
  marginal ones. However, we will only be using the marginal PDFs in this work."
- Incomplete coverage handled and disclosed: "For the uncovered regions of the 90% CI, we inject
  galaxy fakes that are samples from our prior distribution, in order to ensure that the
  marginalization occurs over all the possible host galaxies and that the final uncertainty on
  the Hubble constant is not underestimated."
- Even the compound final result carries its inherited assumption: "We note that the combination
  of the GW170817 bright siren, whose H0 estimate is independent of the cosmological model, with
  the dark sirens, which we derived within the assumption of a Flat ΛCDM scenario, is also tied
  to a Flat ΛCDM scenario."

**Caveat-to-claim ratio:** roughly 1:2 in §4 — of eight Results paragraphs, three are primarily
robustness/limitation and one more is a comparison that concedes a prior-width advantage to
itself. Crucially, **every caveat is paired with either a test that bounds it, a citation that
dismisses it, or an explicit statement that the conservative option was chosen.** No naked
"we caution that".

---

## 10. Transitions and cadence

- **Section openings** are orienting, one sentence, no fanfare:
  - §3: "The formalism used in this work is adapted from Chen et al. (2018)…"
  - §4: "In this Section, we show the results using the DESI Legacy Survey photo-z's, replaced
    by the spectroscopic redshifts described in § 2.2.1, where available."
  - §5: "In this paper, we have presented a new measurement of the Hubble constant using the
    best available gravitational wave events up to date and a state-of-the-art, uniform galaxy
    catalog from the DESI Legacy Survey."
- **Section closings** hand off forward or bound the scope:
  - §2.2.2 closes by connecting the galaxy structure to the eventual result: "…the distance of
    the GW events … is close to overdensities in the galaxy distribution. These overdensities
    provide peaks in the H0 posterior." (A results claim placed in the Data section, so the
    reader is primed.)
  - §3 closes with Eq. 5, the combination rule — mechanically setting up §4.
- **Cross-references are explicit and frequent** ("as described in § 2.2.2", "detailed in § 3",
  "As reported in Section 2") — the reader is never asked to remember.
- **Within-paragraph cadence:** claim → number → reason. E.g. "The most constraining event is
  GW190412, which is reasonable provided that this event has the best localization after
  GW190814…" Explanations are attached with "which is reasonable provided that", "This is
  expected since", "which is reflected in".
- **Closing paragraph structure** (Conclusions ¶2), a four-move template worth copying:
  1. Near-term improvements available with existing techniques ("spectroscopic observations of
     the host galaxies in the localization regions, higher order multipole analyses…").
  2. The step change and when it arrives ("the upcoming LIGO/Virgo/KAGRA observing run, expected
     to start in the second half of 2022").
  3. The precision that step change buys, and why the precision matters ("a 2% statistical
     precision on H0 … valuable to inform us on the Hubble constant tension").
  4. **The homework list** — named systematics that must be solved *before* that precision is
     meaningful: "the impact of a galaxy catalog depth on the constraints, based on different
     BBH formation channels; the impact of the Gaussian ansatz on the dark siren posterior."
  The paper ends on unfinished business, not on self-congratulation.

---

## 11. Ten exemplary sentences

1. > "Another interesting application of multi messenger observations is that of 'standard
   > sirens', first proposed in Schutz (1986): given a gravitational wave detection, it is
   > possible to measure a luminosity distance for the event, and if that is combined with the
   > redshift of the galaxy that hosted the merger, gravitational wave events can be used to
   > probe the distance-redshift relation."
   — Defines the entire method in one sentence of plain physics, with zero notation.

2. > "Dark standard sirens require knowledge of the position and redshift of the ensemble of
   > potential host galaxies, over which one needs to marginalize, and are therefore expected to
   > lead to less precise results than bright standard sirens on a single event basis."
   — States your own approach's weakness before anyone else can; the concession sets up the
   next sentence's counter-argument.

3. > "On the other hand, GW binary black holes, and more in general events without counterpart,
   > currently outnumber those with a counterpart by at least a factor of 10 (Abbott et al.,
   > 2021a)."
   — The counter-argument is a *ratio*, not an adjective. Weakness answered with arithmetic.

4. > "The catalog used there is a compilation of galaxies from different surveys up to 2014, and
   > it is complete out to a distance of ~37 Mpc (Dálya et al., 2018), but largely incomplete at
   > the distances of the dark sirens (all at >200 Mpc)."
   — A competitor's limitation stated as two numbers a factor of 5 apart. No pejorative needed.

5. > "Note that while we use a fiducial ΛCDM cosmology to derive these magnitudes, the H0
   > dependence is irrelevant since the threshold value and the galaxies' absolute magnitudes
   > scale with H0 in the same way."
   — Anticipates the referee's circularity objection and kills it in one clause with the reason.

6. > "We stress that it is expected that the H0 posteriors from individual dark standard sirens
   > present multiple peaks, as these correspond to multiple overdensities in redshift along the
   > line of sight."
   — Turns an alarming-looking figure into a physically expected feature *and* names the
   physical cause. Structure = interpretation, not artifact.

7. > "We find that for both events where fakes are required, the effect of their addition is
   > that of further flattening the posterior compared to the case with no fakes. This is
   > expected since it effectively corresponds to adding more galaxies for marginalization, and
   > with an uninformative redshift distribution."
   — A validation result reported with its predicted sign; the reader can check that the code
   behaved the way the physics says it must.

8. > "It is after combining a large enough number of events (namely O(100) for a O(1%) precision
   > on H0; Palmese et al. 2020) that the dark siren method becomes more powerful."
   — Sets the reader's expectation for what this measurement can and cannot be, using a scaling
   law with a citation.

9. > "We also note that our H0 prior is slightly larger than theirs ([30,140] km s^-1 Mpc^-1),
   > which could be misleading when comparing the precision of measurements."
   — Volunteers a caveat that works *against* the paper's own favourable comparison.

10. > "The constraint is largely driven by the one bright standard siren available, but the dark
    > standard sirens also provide a significant contribution by improving the precision from
    > GW170817 by 28%."
    — Refuses to overclaim the joint result, then quantifies exactly what its own contribution
    is worth. This is the single most imitable sentence in the paper.

---

## 12. DO / DON'T for a dark-siren / mock-catalog / H0-validation paper in this style

**DO**

1. **Put the headline number in abstract sentence 5, with its interval type and prior range in
   the same sentence.** Repeat the identical digits in Results, in a table, and in the
   Conclusions' first two sentences — four exposures, never a rounded variant.
2. **Spend your page count on the thing that could be wrong.** If the contribution is a mock
   catalog and a validation, Data/Mock should be ~40% of the body and Method ~12%. Derivations
   go to citations: "the derivation is described in detail in those works."
3. **Quantify the gap you are closing with a number from the prior work**, the way "complete to
   ~37 Mpc … but the sirens are all at >200 Mpc" does. For a validation paper, the analogue is a
   measured bias in σ units, stated in the Introduction.
4. **State selection criteria as thresholds, in the abstract.** "top 20% based on localization",
   ">70% of their probability covered". A reader should be able to reconstruct your sample cut
   from the abstract alone.
5. **Report every validation metric against an external benchmark.** σ_NMAD = 0.01 is meaningless
   until it is set beside "the DES science requirements include a maximum value for the scatter
   of σ68 < 0.12". Your recovery/coverage numbers need a comparable yardstick.
6. **Report the failed checks.** "The copPIT distribution and the Kendall calibration are poor …
   However, we will only be using the marginal PDFs in this work." Disclose, bound the impact,
   move on.
7. **Pair every caveat with a bounding test, a citation that dismisses it, or a conservative
   choice.** "We have tested that our results are not significantly affected by changes to Ωm
   within the 2σ interval…" is the template.
8. **Explain structure in your posteriors physically before the reader worries.** Multi-peaked
   H0 posteriors = line-of-sight overdensities; show the dN/dz that produces them (Fig. 3) in
   the *Data* section, so the Results claim is already visualized.
9. **Report the direction of a null/degrading test as a prediction confirmed.** "Adding fakes
   flattens the posterior — this is expected since it corresponds to adding uninformative
   galaxies." Sign-checks are cheap credibility.
10. **Add a prior-normalized precision column.** σ_H0/σ_prior in Table 2 is the honest metric
    for prior-dominated inference; a mock-validation paper should carry the equivalent (e.g.
    posterior width relative to prior, or bias in units of the quoted σ).
11. **Write captions that stand alone**: what is plotted → why the transformation was applied →
    every line identified → a closing sentence disclaiming over-reading ("These regions are only
    showed for reference"; "Posteriors are arbitrarily rescaled only for visualization
    purposes").
12. **Own every decision in the first person and give the reason in the same sentence.**
    "We prefer to provide the more conservative result that does not include the luminosity
    weighting."
13. **Put conventions once, in the last Introduction paragraph** (cosmology, prior range,
    interval convention) and never repeat them.
14. **End on named future systematics**, not on a summary. Two or three specific ones, phrased
    as work that must precede the next precision milestone.
15. **Expand every acronym on first use, and give the plain description before the label**
    ("…so they are referred to as SEOBNRv4PHM").

**DON'T**

1. **Don't open the abstract with importance.** No "The Hubble tension is one of the most
   pressing problems in cosmology." Sentence 1 is the result.
2. **Don't put method detail in the abstract.** This abstract contains no random forests, no
   BAYESTAR, no selection function — despite half the paper being about them.
3. **Don't write an appendix you can avoid.** If a derivation exists in a cited paper, cite the
   equation number: "Note this is the same as Eq. 15 in Chen et al. (2018)."
4. **Don't quote a number without its prior, its interval type, and its model assumption.**
   Especially not a prior-dominated H0.
5. **Don't over-round or over-precise.** 79.8(+19.1)(−12.8) and "~20% precision" coexist; no
   spurious digits, no false modesty either.
6. **Don't hedge your own measurements.** Hedge interpretation and provenance instead
   ("broadly consistent", "very likely to be real GW events"). Never "may possibly indicate".
7. **Don't hide a favourable comparison's unfair advantage** — volunteer it, as with the wider
   H0 prior.
8. **Don't create a separate "Systematics" or "Caveats" section.** Interleave 2–5 sentence
   robustness paragraphs into Results, each immediately after the claim it qualifies.
9. **Don't describe workflow, pipelines, or process.** There is no sentence in this paper of the
   form "we then ran the pipeline" or "we performed a series of checks"; each check is named,
   executed, and reported as a number in one or two sentences.
10. **Don't tabulate what is uninformative.** Eight individual dark-siren H0 values are a figure,
    not a table, because individually they barely move the prior — and the paper says exactly
    that ("Each event reduces the 68% CI of the H0 prior to its ~85%").
11. **Don't claim to resolve the tension.** "Broadly consistent with recent H0 estimates from
    both the Cosmic Microwave Background and Supernovae" is the correct posture at 20% precision.
12. **Don't let the reader infer why a plotting choice was made** — say "in order to highlight
    the presence of overdensities and underdensities along the line of sight" in the caption.
13. **Don't front-load the stakes paragraph.** The H0-tension motivation lands in Introduction
    ¶4, after the method and the state of the art are established.
14. **Don't use passive voice for decisions.** Reserve it for third-party actions and for
    machinery whose actor is irrelevant.

---

### One-line summary of the style
A short, first-person, number-dense paper that states its result in the first sentence, spends
its pages validating its *inputs* rather than re-deriving its *inference*, converts every
weakness into an arithmetic comparison, and ends on the systematics it has not yet solved.
