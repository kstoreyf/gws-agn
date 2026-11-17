# The owner's paper style: what the darksirens session knows (written for the agn session, 17 Nov 2026)

Provenance marks, so you can weigh each line:
- **[O]** the owner's own words, verbatim from my records, with the date.
- **[S]** the owner's manuscript-style skill (`astro-paper-style`), which the owner maintains; its "owner editorial rules" are dated hard requirements.
- **[D]** decisions I wrote for the two lensing drafts after the owner said "Go ahead and revise both drafts" (10 Nov). The owner received the drafts built on them and did not object; that is not the same as approving each line.
- **[M]** my reading.

**What I do not have.** The owner has never line-edited the lensing drafts. Their feedback on those drafts was about science scope (which likelihood, which priors, which test belongs in which paper), not about sentences, captions or tables. So no abstract, paragraph, caption or table of the lensing drafts is "approved as it stands", and I have no verbatim corrections of that kind. The drafts still carry about 100 placeholders each. The only reporting text of mine the owner explicitly approved is a section of the DESI results page (a web page, not a paper): see item 5.

One conflict to know about: the style skill (August) still says validation uses "coverage counts" and "realisation counts". The owner's later rule (October to November) is Bayesian only, and it wins: combined posteriors over simulated catalogs against the input, no counts.

---

## 1. Model comparisons

No statement from the owner on table against figure against prose for a sequence of nested models. What I have:

- **[S]** "Model/Method in main text is SHORT: one master equation family with variants defined as its limits ('setting X = 0 recovers ...')." This is the owner's way to present nested models: one likelihood, and each simpler analysis is a named limit of it. The models are never a list of separate things with letters.
- **[O]** (rule recorded 2 Oct, from the owner's reaction to my reports) internal labels and bare identifiers confuse them: describe each item in words. Letters A to D are exactly this.
- **[D]** A Bayes factor is quoted only between normalised models of the same data, always against a reference that the sentence or caption names, with its prior range. Never accumulated down a column.
- **[D]** Each paper gets one table of "results by analysis": rows named by their physics, the log10 Bayes factor as ONE column among the physical quantities, not the subject of the table.

The table from lensing paper 1 (draft v4, sent to the owner, not objected to, not line-approved):

```latex
\begin{table*}
\caption{Results by analysis for the \Nobs{} events of GWTC-5. $\lB$ is the
Bayes factor against the detections-only analysis, or against the background
analysis for the row that includes the background; limits on the merger rate
are 90\% upper limits in $\rateunit$. Lensing rows count every detected
image, bright or faint, as an event; the last row assigns each event the
summed magnification with one lensed term.}
\label{tab:results}
\begin{ruledtabular}
\begin{tabular}{lcccc}
Analysis & $\lB$ & $z_p$ & $\Rz(z=2)$ & $\Rz(z=3)$\\
\hline
Detections only & 0 & $\ZpNoLens$ & $<\RtwoNoLens$ & $<\RthreeNoLens$\\
Lensing, $F=\Fearly$ & $\LBEarly$ & $\ZpEarly$ & $<\RtwoEarly$ & $<\RthreeEarly$\\
Lensing, $F=\Fall$ & $\LBAll$ & $\ZpAll$ & $<\RtwoAll$ & $<\RthreeAll$\\
Lensing, $F=\Fold$ & $\LBOld$ & $\ZpOld$ & $<\RtwoOld$ & $<\RthreeOld$\\
Lensing, $F$ free (uniform prior) & $\LBFree$ & $\ZpFree$ & $<\RtwoFree$ & $<\RthreeFree$\\
Background & & $\ZpBkg$ & $<\RtwoBkg$ & $<\RthreeBkg$\\
Lensing, $F=\Fall$, and background & $\LBAllBkg$ & $\ZpBoth$ & $<\RtwoBoth$ & $<\RthreeBoth$\\
\hline
Summed magnification, $F=\Fold$ & $\LBSummedOld$ & $\ZpSummedOld$ & $<\RtwoSummedOld$ & $<\RthreeSummedOld$\\
\end{tabular}
\end{ruledtabular}
\end{table*}
```

What changed in it across versions: v1 quoted 95% limits (now 90% everywhere); two rows were called "correlated selection" and "independent selection", names invented during the work (retired; the caption now says what is counted); a row with a limit at z = 5 went (the data do not reach it); the evidence column had no stated reference (now the caption names the reference of every row).

**[M] For your A, B, C, D:** drop the letters and the accumulation. Name each analysis by the assumption it holds or frees ("population fixed", "AGN fraction free", "flux-limited catalogs", "host-dependent masses and spins"). Quote each Bayes factor in the sentence where that assumption is discussed, against the analysis just before it or one fixed reference, with the prior range. If a summary table stays, its columns are the physical results (H0, the AGN fraction, ...) with log10 B as one column, and the rows are in the order of the physics, not the order the runs were made. A figure of the H0 posterior under each set of assumptions carries the "one assumption at a time" story better than the table.

## 2. What "lab report" means

- **[O]** 10 Nov, commissioning the lensing drafts: "astrophysics is the key, not a lab notebook".
- **[O]** 9 Nov: "What is this. It just pops out of nowhere without enough context", "One std dev is how much in km/s/Mpc?", "Dont randomly slap numbers on my face which have no physical meaning or context."
- **[O]** 7 Nov, on a design of mine: "so damn overengineered".
- **[O]** 8 Nov: "ask me the questions with explicit details and no jargon other than usual ligo paper jargon".
- **[S]** "Not one sentence anywhere about workflow." Banned in reader-facing text: gate, workstream, pipeline (the workflow kind), rerun, verdict, audit, pre-registered, convergence checks, sampler settings, runtimes, seeds as process, campaign, planted (say input or true), lane (say injection set), oracle, backbone, battery, veto, "scaled to peak".
- **[S]** "Internal tests do not get reader-facing figures. ... The test is whether the figure supports a claim the paper makes, or merely shows that you checked."
- **[S]** "No Limitations/Systematics section, ever." Every caveat is the final clause of the claim it limits, with a size, bound or test.

Before and after, from my drafts and pages:

| Before | After |
|---|---|
| "correlated selection" and "independent selection" runs | "Lensing rows count every detected image, bright or faint, as an event" |
| a panel of coverage counts over simulated catalogs | one combined posterior over the simulated catalogs against the input, with the credible level at which the input lies |
| "a shift of 0.36 posterior standard deviations" | the shift in km/s/Mpc, beside the width it is compared with |
| "F = 0.05, a hundred times the expected value" | "0.05, where a one-year catalog holds a few image pairs" |
| "the O5 mock" | "simulated catalogs of about 250 events" |
| "simulated universes" (in a paper) | "simulated catalogs" |
| "lensed fraction F" | "the lensing normalization F"; "lensed fraction" only for the derived fraction of detected events |
| a list of what an earlier paper got wrong | the earlier paper as the starting point this work extends |

## 3. A paper that relaxes assumptions one at a time

- **[S]** "Results open with a credibility anchor: a reproduction of a known reference or a told-the-truth control, before any new number." Your fixed-population result is this anchor. In lensing paper 1 the anchor is two short subsections ("The population without lensing", against the LVK numbers; "The published joint analysis of GWTC-4", reproduced), each with one table and a few sentences. It is introduced as a reproduction, in the we-find register, and it is not in the title or the first half of the abstract.
- **[S]** "Organise interpretation by physical regime, with the regime in the subsection title", never by argument type or evidence type. So each later section is titled by the physics it lets vary, and its first sentence does something: it states what is now free and what is found, not what the section will contain.
- **[S]** Abstract, 6 to 8 sentences: scope, method as one clause, headline numbers with interval type AND the assumption in the same sentence, the contrast or secondary result, validation in one clause, implication. "The assumption travels with the number, in the same sentence." No headline numbers in the introduction; its last paragraph is the roadmap with cosmology, priors and the interval convention.
- **[S]** Validation is its own top-level section, sized like Results, phrased as physics tests.
- **[S]** Conclusions: what is established, the named inputs it depends on (each with the literature's current answer), near-term improvements, the step change and its date, named unsolved systematics, and a last sentence that hands the result to another community. Do not restate the body.
- **[D]** "Astrophysics first: results sections lead with the astrophysical statement and its figure; method in the main text is the master likelihood and its limits only; computational material lives in appendices or is cut."
- **[M]** Where the headline sits: with the most general analysis the data support, stated with its assumptions; the anchor and the intermediate steps show how each assumption moves it.

## 4. The owner's corrections to my drafts

On science scope, recorded as decisions (not verbatim edits):
- Paper 1 uses single images only: "No, single images only" (13 Nov). The honest headline followed: single images do not measure F at this catalog size, so the paper leads with the merger rate, not with F.
- Prior on F: uniform up to the physical maximum as the headline, log-uniform alongside; say plainly when a limit is set by the prior (11 Nov).
- A forecast number is computed from our own posterior, with the published number beside it as a check (11 Nov).
- "No image pair identified" is restricted to published searches; no citation to an unpublished collaboration paper (11 Nov).
- A large-catalog test moved from paper 1 to paper 2, where its subject lives (12 Nov).
- Effects that cannot be seen by eye on the posterior get one line and are closed (9 Nov, about results reporting).
- **[O]** 19 Oct, in my paraphrase from the record: when a setting moves the result, say so with size and direction; a hypothesis that failed one test is not "ruled out".

## 5. Text the owner approved

Only this, from the DESI results page, 16 Nov; the owner's reply was "excellent". Numbers removed here:

> The first simulated universe shows the same pattern as the real data. The catalog moves the mean by [x] of a standard deviation (real data [y]), the median by [..], and narrows the posterior by [..]%. The move is toward the true value: the mean goes from [..] to [..] km/s/Mpc. The second shows no change. [...] So a catalog shift of the size and sign seen in the real data does occur in a universe where nothing is wrong. The real shift is therefore not, by itself, a sign of a fault in the catalog or in its treatment. Two universes cannot say how often it occurs.

What it does: the question as the heading ("Is a shift of this size expected?"), each number beside the one it is compared with, the claim in a short sentence after the evidence, and the limit of the claim stated at once. For a paper, "universes" becomes "catalogs".

A caption from lensing paper 1 that follows the skill's rule (what is plotted, every line and band with its level, one closing sentence), sent and not objected to:

> The binary black hole merger rate history. (a) 90% credible bands of the merger rate per comoving volume from detections only (black outline) and with lensing at F = [..] (blue). (b) The same detections-only band, with the background data (orange) and with lensing and the background data (blue). The gray band is the 90% range of the prior and the dashed green curve is the star formation history of Ref. [..] scaled to the local merger rate. Above z of about 3 the bands follow the prior.

## 6. Naming

- **[S]** Use the field's word; coin a handle only for something the field has no word for, introduce it once ("dubbed here as ..."), never in the abstract without a gloss. One object, one name.
- **[S]** "Legend wording must match the caption wording verbatim." Legends and axis labels obey every text rule.
- **[D]** Analyses are named by what they include, the same way in text, tables and legends: "Detections only", "Lensing, F = ...", "with the background data".
- **[D]** Mocks are "simulated catalogs", described by their number of detected events; each validation set is drawn from one fixed input so that its catalogs combine into one posterior; the true parameter is the "input" value.
- No run names, seeds, letters or version tags anywhere a reader can see.

## 7. Figures and tables

- Lensing paper 1 v4: 15 pages, 8 main-text figures, 6 tables (priors; reference values and published limits; two anchor tables; results by analysis; leading events). Paper 2 v5: 16 pages, 8 figures, 6 tables. The owner did not object to these counts; I have no stated limit from them for PRD.
- **[S]** A figure earns its place by supporting a claim the paper makes. Posterior figures show normalised densities with units on the y axis; pathologies are shown, not cropped. A multi-panel figure spans the page. Corner plots go to the appendix.
- **[D]** Cut in the lensing revisions: coverage panels, likelihood-penalty panels, a rate limit at z = 5, any panel duplicated between the two papers (figures are disjoint between companion papers).

## 8. How the owner judges a draft

- They read on a phone first, so the first screen decides: the astrophysical statement and its number, in plain words.
- They stop at a number without physical meaning, a label they have to decode, machinery ahead of the simple physical choice, and a claim stronger than its test.
- They ask for derivations rather than options: settle what can be settled and bring only the physics choices.
- On the GW-AGN paper itself I have nothing from the owner beyond what you quoted to me.
