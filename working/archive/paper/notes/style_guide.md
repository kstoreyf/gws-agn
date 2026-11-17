# Consolidated style guide (from the three golden references)

Sources: notes/style_1905.12669.md (Fishbach & Holz, ApJL), notes/style_2111.06445.md
(Palmese et al., ApJ), notes/style_2407.02460.md (Magaña Hernandez & Palmese, PRD —
the owner's own voice). Where the three disagree, the owner's paper wins.
Internal document.

## Architecture for THIS paper

1. **Abstract** — 6–8 sentences: scope → method-as-clause (one sentence, no detail)
   → headline numbers with interval type and the assumption welded into the same
   sentence → the single-tracer contrast → the null/validation corroboration →
   implication. No forward references, no machinery.
2. **Introduction** (~6 ¶, ≤1100 w) — field state quantitatively → why a second
   tracer (physics: AGN channel, branching ratio) → the hinge premise (two tracers
   of one field ⇒ mixture; density+bias contrast identifies f) → gap stated as a
   number or capability the prior art lacks → this work sentence → roadmap ¶
   closing with cosmology, priors, and "68% CI unless stated". NO headline numbers
   in the intro (Fishbach rule; the owner's abstract carries them instead).
3. **Model** (short main-text section) — the mixture likelihood as ONE master
   equation family with variants defined as limits (f=0, f=1, n0→0 complete
   limit). Keep eq count low; cite Chen/Mandel/Gray for derivations. The current
   methods.tex content largely survives but tightened; selection subsection
   loses all estimator-internals language (N_eff criterion gets ONE sentence).
4. **Simulated universe** (the Data analogue — BIG, this is our input novelty;
   Palmese precedent 43%) — field+tracers, events+measurement family (v3:
   rho_obs is the datum AND the distance observable; every width a function of
   it; exact posteriors; realised photo-z), flux-limited versions. The three
   design requirements stay but stated as physics consequences of the likelihood
   being a density over observed data — never as process history, never
   mentioning that an earlier version got it wrong.
5. **Results** — 5.1 opens with a credibility anchor (matched-host controls
   recover truth — the reproduction-check-first rule) then the single-catalog
   measurements incl. the honest AGN railing; 5.2 the joint headline; 5.3
   incomplete catalogs (STUB until analysis 3). Numbers restated with identical
   digits wherever they appear.
6. **Validation** — own top-level section (owner's-paper rule), sized like
   Results: five-realisation recovery (closure means ± sems as plain recovery
   statements), the sky-shuffle null, lane/crosscheck agreement in one sentence,
   selection-MC error carried into the quoted budget. All stated as physics
   tests, zero process language.
7. **Discussion/Conclusions** — what the measurement establishes → what
   identifies f (density+bias contrast; inherited dependencies n0, bias) →
   closing template: near-term improvements → step change with date → named
   unsolved systematics → final sentence hands the result to another community.
8. **Appendix** — none for now (Palmese precedent); if analysis 0 enters later
   it goes here as corner/recovery material with zero prose beyond captions.

## Sentence-level rules

- "We find" for measurements; "we argue" for interpretation; hedges attach ONLY
  to interpretation, never to our own numbers.
- Rhythm long–long–SHORT; the short sentence carries the claim.
- Every caveat: final clause of the claim it limits, WITH a size, bound, or test.
  No Limitations/Systematics section.
- Numbers: 1–2 sig figs, asymmetric intervals, interval convention declared once
  (roadmap ¶); prior/assumption travels with every H0 quote; probabilities as
  credibilities, never sigmas.
- Discourse markers (First… Second… Finally…) instead of micro-subsections.
- BANNED in reader-facing text: gate, workstream, pipeline, rerun, verdict,
  audit, pre-registered, convergence, sampler settings, runtimes, seeds-as-
  process (realisations of the simulated universe are fine), "lab", "campaign".
- Captions self-sufficient: what is plotted → every line/colour identified →
  credibility levels → one interpretive closing sentence; defensive closer where
  a reader could over-read.
- Sections open with the action, close with a hand-off.
