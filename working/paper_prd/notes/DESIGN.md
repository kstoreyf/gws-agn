# Paper design (16 Nov 2026, revision 2 after the novelty check and the numbers chain)

This file is the brief for everyone who works on the paper in `working/paper_prd/`.
It says what the paper claims, how it is laid out, where each number comes from, and
which rules the text and figures follow. The project owner asked for a paper written
from scratch for Physical Review D. The earlier drafts in `working/archive/paper`,
`working/archive/paper_codex` and `working/archive/report` are frozen records. Do not
open them for wording, structure or figures.

Companion files: `notes/NOVELTY.md` (prior work, positioning, sentences not to write),
`notes/LIKELIHOOD_MATH.md` (the likelihood as the code evaluates it), `notes/GAPS.md`
(what the exploratory runs do not yet support), `values/MACROS.md` (the menu of
numbers), `placeholders.txt` and `runs.txt` (numbers still to be computed and the
runs that produce them), `refs_candidates.bib` (verified references).

## 1. What the paper is about

Binary black hole (BBH) mergers have no electromagnetic counterpart, so a "dark siren"
measurement of the Hubble constant H0 assigns each merger statistically to the
galaxies of a catalog. Some mergers may instead occur in the discs of active galactic
nuclei (AGN). Searches for that channel correlate merger positions with an AGN
catalog at a fixed cosmology and infer the fraction of mergers that AGN host, f_AGN.
Population studies look for an AGN component in the masses and spins, without
positions. The three use overlapping information and each fixes what the others
infer.

Prior work already has pieces: H0 inferred with an AGN-hosted fraction from an AGN
catalog (Bom & Palmese 2024), a two-component host mixture inferred with H0 on one
galaxy catalog (Cross-Parkin et al. 2026, "Hostbusters", which the owner co-authored),
and f_AGN with selection effects and catalog incompleteness at fixed cosmology (Pouw
et al. 2026). All use one BBH population for both host classes, and all three papers
name the host-dependent population as the next step.

This paper takes that step on simulated catalogs. Each merger is hosted by a galaxy
with probability 1 - f_AGN or by an AGN with probability f_AGN; the galaxy and AGN
catalogs trace one density field; H0 is free; and the mass and spin distributions may
differ between the two host classes. It then shows what limits f_AGN: the number
density of AGN that the analysis assumes or has to infer.

All results come from simulated catalogs (mocks): galaxies and AGN drawn from one
lognormal density field, and 1000 detected BBH mergers per catalog out to z ~ 0.3.

## 2. Claims

Numbers are the present values from the exploratory runs, as computed by
`scripts/build_values.py`. Every one of them reaches the text only through a macro
(section 7). The lead result is K; F is the foundation it stands on; L is what limits
the measurement.

### K. Host-dependent masses and spins (the new core; lead the abstract with it)

K1. **If mergers in AGN are heavier, an analysis with one population is biased in H0,
    and host catalogs do not protect it.** Mock: AGN mass peak 5 Msun above the galaxy
    one (40 against 35) and AGN mean effective spin 0.10 above. The common-population
    analysis with both catalogs returns H0 = 70.6 (90%: 68.9 to 72.9) for an input of
    67.74, 3.0 km/s/Mpc above the host-dependent model on the same data. With the
    galaxy catalog alone, the common-population posterior on this mock runs into the
    upper edge of the range explored (76). In a second mock, where the same two
    populations exist but every host is a galaxy (so no catalog can tell them apart),
    the common-population analysis returns H0 above 88.9, at least 20 km/s/Mpc high,
    and the two-population model recovers 68.9 (65.1 to 74.6). Always say which mock a
    number is from. The sign is that of a spectral-siren mass-scale error: an
    unmodelled heavier peak reads as a lower source-frame mass scale, hence a larger
    redshift, hence a larger H0. The host-dependent model is preferred by
    log10 B = 28. State plainly that the common population here has its mass and spin
    parameters fixed at the galaxy-class values; a common population with free
    parameters is a red placeholder (run R8e).
K2. **Letting each host class have its own mass peak and spin mean recovers H0, f_AGN
    and both offsets.** Offsets free with the reference population fixed: H0 = 67.66
    (65.39 to 69.43), f_AGN = 0.274, mass-peak offset 4.8 Msun (input 5), spin-mean
    offset 0.119 (input 0.10). With the reference peak location, spin mean and both
    widths inferred as well (eight parameters): H0 = 67.42 (65.02 to 69.16). The
    host-dependent model adds no H0 precision; it removes the bias. In the
    eight-parameter model the spin offset is 0.158 (0.121 to 0.191), with the input
    outside the 90% interval on this one catalog; say so with the interval, and leave
    the combined-catalog statement as a placeholder.
K3. **Positions and intrinsic parameters both identify the host class.** f_AGN is set
    almost entirely by position: its 90% interval narrows by 5% when the spin
    difference is modelled. The spin offset is measured 2.2 times better (90% widths)
    with the catalogs than from spins alone, because the catalogs say which events
    are likely AGN-hosted; the Bayes factor against a zero spin offset is
    log10 B = 8.5 with the catalogs and 4.0 from spins alone. The spins-alone
    posterior is cut by its prior edge at 0.25, so the factor 2.2 depends slightly on
    the prior box; say so in a clause. Qualify "independent": this is shown for the
    spin offset; mass couples to position through detection.

### F. Foundation: two catalogs on one density field, complete catalogs

F1. **H0 and f_AGN are recovered together and are nearly uncorrelated.** Combined
    posterior of 9 simulated catalogs: H0 = 68.36 (67.73 to 69.00), input 67.74 at the
    89.5% credible level; f_AGN = 0.283 (0.256 to 0.310), input 0.30 at the 70% level;
    correlation coefficient about 0.1. This agrees with the weak correlation found by
    Cross-Parkin et al. 2026 for two weightings of one catalog, now for two physically
    distinct catalogs. The two-population model is preferred to galaxy hosts only by
    log10 B = 1.8 to 15.9 per catalog.
F2. **At equal numbers of events, AGN-hosted mergers constrain H0 about 4.7 times more
    tightly than galaxy-hosted ones** (93 catalogs of 1000 events each). The direction
    is known (rare hosts are easier to associate); the paper adds a like-for-like
    number on one density field and its reason: AGN are a hundred times rarer and more
    strongly clustered, so the redshift distribution along a line of sight is a few
    narrow spikes. Give the number of hosts per localisation volume for each class.
    Never write that real AGN catalogs would deliver this gain: it holds for
    AGN-hosted mergers only.
F3. **A galaxy-only analysis of the mixed population is rejected by the evidence;
    its H0 moves by less than its own interval; an AGN-only analysis fails.** Galaxy
    catalog alone, all events: combined H0 = 68.9, input at the 88% level, 0.5 above
    the 68.36 obtained with both catalogs (both have the input near the lower edge of
    their 90% interval). The AGN catalog alone has no interior maximum in the prior range for 8
    of 9 catalogs. Reason to give: galaxies and AGN trace the same structure, so the
    galaxy catalog still marks where AGN-hosted mergers are, while the sparse AGN
    catalog cannot stand in for the galaxies. (This replaces the earlier claim of a
    bias; it agrees with Borghi et al. 2026 and Kalomenopoulos et al. 2025.)

### L. What limits the measurement: the number of hosts the catalogs miss

Read `notes/LIKELIHOOD_MATH.md` sections 2 to 4 and 8 before writing this part. The
facts it establishes, which the earlier reports had wrong:

- Each catalog's host density is normalised to unit total over the sky and
  0 < z < z_max, counting catalogued hosts plus the modelled uncatalogued hosts
  n_k [1 - C_k(z)] dV_c. In every flux-limited analysis on disk z_max = 5, far beyond
  the catalogs (z <= 1) and the events (z < 0.4), so more than 98% of each
  normalisation is modelled hosts that no event can reach.
- The modelled uncatalogued hosts are uniform in comoving volume. No density field
  modulates them.
- Because the two normalisations do not cancel in a mixture, the likelihood depends on
  f_AGN and the assumed number densities only through the combination
  logit f~ = logit f_AGN + ln(F_AGN / F_gal), where F_k = N_k / Z_k is the catalogued
  share of class-k hosts in the normalisation volume (exact wherever no event lies in
  the incomplete part of a catalog). The mergers measure f~, which is the merger rate
  per catalogued host in AGN against that in galaxies. Turning it into the fraction of
  mergers hosted by AGN requires counting the hosts the catalogs miss.

L1. **Flux limits cost little precision when the number densities are known.** At
    the shallowest limit (9.5% completeness inside the horizon) the 90% intervals are
    wider than for complete catalogs by a factor 1.06 (f_AGN) and 1.07 (H0), on one
    simulated catalog; the H0 median moves down by 0.55 km/s/Mpc, a third of its 90%
    half-width. Say that the missing hosts are uniform in comoving volume, and that
    about half of the events lie at redshifts where even the shallowest catalog is
    complete (it is complete below z = 0.11 and the median event is at z = 0.13); a
    completeness weighted by the events is a requested macro. For H0 this is in line
    with the completion literature; for f_AGN it is new.
L2. **What the mergers measure is the merger rate per host; f_AGN follows only after
    the missing hosts are counted.** State the identity above as an equation. Its
    consequence: d logit f_AGN / d ln n_AGN = 1 - F_AGN. In the analyses here
    F_AGN < 0.02 at every flux limit, so f_AGN / (1 - f_AGN) is proportional to the
    assumed n_AGN / n_gal: 0.43 to 0.47 in f_AGN per dex, a change of about 0.13 for a
    factor of two, the same at every depth. The identity reproduces the slopes of the
    numerical posteriors (0.434 and 0.438 at the two deepest limits). The degeneracy
    is known qualitatively (Zhu et al. 2024); the exact relation and the reason it does
    not weaken as the catalog deepens (the normalisation volume, not the completeness
    where the events are) are new. H0 is untouched by the normalisation: it responds
    only when events lie where the catalog is incomplete, 3.2 km/s/Mpc per dex at 9.5%
    completeness and nothing at 81% or more. Do not write that deeper catalogs fail to
    help in general: with the normalisation volume restricted to where a catalog is
    complete, F_AGN = 1 and the assumed density drops out. That case is a red
    placeholder (run R5b).
L3. **With both number densities free, the mergers bound the galaxy density and leave
    the AGN density, and with it f_AGN, to the prior.** At 9.5% completeness:
    log10 n_gal = -3.12 (+0.29, -0.68) against -3.0, a posterior with a sharp upper
    edge and a tail to the lower prior edge; the AGN density posterior is about three
    times the prior density at its peak and no more; f_AGN and n_AGN are degenerate
    (correlation +0.90), as the identity of L2 requires, and the f_AGN 90% interval
    covers about 0.8 of its prior; H0 is unchanged (0.13 km/s/Mpc across the four
    depths). Three simulated catalogs. The difference between the two densities is one
    of degree; say what constrains n_gal (the uniform missing hosts against the
    clustered catalogued ones, where events lie beyond the complete part).

What the paper does not claim: anything about real BBHs or real catalogs; a forecast
for a specific observing run; a "first" joint inference of H0 with a host fraction or
host model (section 9); that f_AGN can be measured without an external handle on the
AGN number density.

## 3. Layout (REVTeX 4.2, `aps,prd,reprint,twocolumn`)

Model the mechanics on the lensing draft at
`/hildafs/projects/phy230054p/magana/darksirens-core-data/lensing_scope_2026-11-07/paper_drafts/paper1/`
(`main.tex` preamble, `results_macros.tex`, `placeholders.txt`, `runs.txt`). Build with
`~/bin/tectonic main.tex` from `working/paper_prd/`.

I. Introduction. About six paragraphs: dark sirens and where they stand
   quantitatively; the AGN channel, the spatial searches and the population studies;
   the hinge (galaxies and AGN trace the same structure, and each analysis fixes what
   another infers); the gap as three missing capabilities (`notes/NOVELTY.md`
   section 5); this work in one sentence; roadmap paragraph ending with the cosmology,
   the 90% convention and the Bayes-factor convention. No headline numbers.

II. Method. Short, one master likelihood and its limits. Write it from
   `notes/LIKELIHOOD_MATH.md`, which gives the equations as the code evaluates them.
   A. Host mixture with host-dependent populations:
      p(theta, z, Omega) = sum_k f_k p_k(z, Omega) p_pop(theta | Lambda_k), k = galaxy,
      AGN; the selection term is the same mixture. Limits: equal populations give the
      position-only two-catalog analysis; f_AGN = 0 gives the standard galaxy-catalog
      analysis.
   B. Catalog intensity, its normalisation over the sky and 0 < z < z_max, and hosts
      below the flux limit (uniform in comoving volume). State the identity of L2
      here or in IV C, once.
   C. Parameters and priors (Table I). All priors uniform. Posterior summaries are
      medians with 90% equal-tailed intervals.

III. Mock catalogs and gravitational-wave events. This section gets the word budget.
   A. Galaxies and AGN. B. Mergers and their measurement. C. Host-dependent
   populations. D. Flux-limited catalogs. Source: `working/data/README.md`,
   `working/data/DESIGN_PE.md`, and the `\Mock...` macros. Say that the catalogs are
   spectroscopic in quality (redshift error 0.003(1+z)), that the AGN density and bias
   are design choices with their literature anchors, and that an input f_AGN of 0.30
   lies inside the current limits of Pouw et al. 2026 (NOVELTY.md, Risks 7 and 8).

IV. Results. Each subsection opens with its result, then the figure, then the reason.
   A. Two catalogs, one density field (F1, F2, F3). The credibility anchor: agreement
      with what is known, on complete catalogs. Fig. 2, Fig. 3, Table II.
   B. Host-dependent masses and spins (K1, K2, K3). The lead result. Fig. 4, Fig. 5,
      Table III.
   C. Flux limits and the number of missing hosts (L1, L2, L3). Fig. 6, Fig. 7,
      Table IV.

V. Validation. Combined posteriors over simulated catalogs against the input, with
   the credible level at which the input lies; catalogs with no AGN-hosted mergers
   (red placeholders); one clause on numerical accuracy.

VI. Conclusions. Three compact paragraphs: the result; what limits it; what real
   applications need. End by handing the result to the surveys that measure AGN
   number densities and selection.

Appendix A. Likelihood evaluation (catalog kernels, selection integral, Monte Carlo
accuracy). Appendix B. Measurement model of the mock events. Appendix C. Full
posterior of the eight-parameter model (corner figure).

## 4. Figures (all 90% regions; drawn from results, never from the macros file)

1. The mocks. (a) a slice of the light cone with galaxies, AGN and the hosts of
   detected mergers; (b) redshift distribution of the hosts with the completeness
   curves of the four flux limits; (c) mass and chi_eff distributions of the two
   host classes in the host-dependent mock.
2. Two catalogs, complete. Joint (H0, f_AGN) posterior, one small panel per simulated
   catalog with identical axes and one colour, input marked in black, plus a panel
   with the combined posterior.
3. Which hosts carry the information. (a) H0 posteriors from 1000 galaxy-hosted and
   from 1000 AGN-hosted mergers; (b) H0 from all events with the galaxy catalog alone
   against the two-catalog posterior.
4. Host-dependent populations and H0. H0 posteriors on the host-dependent mock:
   common population against host-dependent model, with two catalogs and with the
   galaxy catalog alone.
5. The offsets. Posteriors of f_AGN, mass-peak offset and spin-mean offset, with the
   reference population fixed and with it free; the spin offset with and without the
   catalogs.
6. Flux limits. (a) f_AGN and H0 intervals against completeness with the number
   densities at their inputs; (b) f_AGN and (c) H0 against the assumed AGN number
   density, one curve per completeness, labelled directly on the curves.
7. Free number densities at 9.5% completeness: galaxy and AGN number-density
   posteriors and the (f_AGN, log n_AGN) region, one panel per simulated catalog.
8. Validation. Combined posteriors over simulated catalogs for H0 and f_AGN.
Appendix. Corner of the eight-parameter model.

Figure rules (owner): no legend strip above panels and no figure titles; label curves
directly; few colours (galaxies one colour, AGN one colour, input values black, a
completeness sequence in one hue); small multiples for independent catalogs, same
colour in each; posterior densities are normalised with units on the y axis; any
figure with more than one panel is `figure*` at page width; captions of two to four
sentences that say what is plotted, the credible level and the black input lines.
Full text of these rules: `notes/owner_voice_brief_2026-10-02.md` sections 14 to 25,
and `~/.claude/skills/astro-paper-style/SKILL.md` ("Figures and captions").

## 5. Statistics: Bayesian throughout (owner's rule for these papers)

- Results and validation are posteriors under stated priors and Bayes factors.
  Quote log10 B, with its numerical uncertainty where it comes from nested sampling.
  A Bayes factor for an offset depends on the prior box of the offset; give the box.
- No coverage counts, p-values, t statistics, "sigma" language, or scatter of a
  statistic over simulations used as a yardstick.
- Validation uses combined posteriors: all simulated catalogs of a set come from one
  input, their log-likelihoods add, and the text gives the combined posterior and
  the credible level at which the input lies.
- Nested model comparisons use the Savage-Dickey ratio where the grids allow it.
- 90% credible intervals and 90% limits everywhere, declared once in the roadmap.
- Every number comes with its physical meaning: what it is compared with, in units.
- An effect that cannot be seen by eye on the posterior gets one line.
- Any sentence that says a result agrees with or recovers an input names the interval
  that contains the input; `values/values.json` holds a containment record per macro.

## 6. Prose rules

Read, in this order: `~/.claude/skills/astro-paper-style/SKILL.md` (the owner's
manuscript style; its reference files hold the evidence), then
`notes/owner_voice_brief_2026-10-02.md` sections 1 to 13 and 26 to 29 (written about
an earlier draft of this same project; the banned constructions and word lists apply
here in full), then the lensing draft `paper1/main.tex` named above as the model of
tone and structure.

In short: the paper reads as an author who understands the result explaining it, not
as a record of runs. No chronology, no "we next", no "designed to isolate", no
workflow or code words, no em dashes. State the physical choice before any machinery.
Vocabulary: "simulated catalogs" on first definition and then mostly nothing; "input
value" for the truth; "galaxy catalog" and "AGN catalog"; "completeness";
"number density" for n_gal and n_AGN; "host class"; "host-dependent mass and spin
distributions" (never "mark", "marked", "branch", "arm", "rung", "anchor", "lane",
"seed", "realisation index"). American spelling.

## 7. Numbers

Every result number in the text, tables and captions is a macro defined in
`values/results_macros.tex`, written by `scripts/build_values.py` from the analysis
result files. `values/MACROS.md` is the menu; `values/values.json` has sources and
containment records. Three classes, as in the lensing drafts:

- `\ph{...}` red: not yet computable. Listed in `placeholders.txt`.
- `\anc{...}` teal: a real number from the exploratory runs under
  `working/analyses/`; the production runs replace it.
- plain: design choices and input values of the mocks.

If the text needs a number that has no macro, do not type it: list it in
`notes/MACRO_REQUESTS.md` (name, meaning, where it would come from) and use
`\ph{XX}` in the text until it exists. Macros that contain `^`, `\pm` or `\times`
are for math mode.

Sources by claim (directories under `working/analyses/` unless stated):

| claim | source |
|---|---|
| F1 | `analysis_2_complete_catalog_H0_fagn/results/` (per-catalog joint grids) |
| F2 | `analysis_0_pure_tracer_H0/results/` |
| F3 | `analysis_1_complete_catalog_H0/results/`, the F1 grids for the Savage-Dickey ratios |
| K1, K2 | `analysis_10_mass_spin_marked_multitracer/results/`, `analysis_11_free_common_population/results/`, `analysis_12_shared_width_robustness/results/`, `analysis_13_joint_shared_widths/results/` |
| K3 | `analysis_8_spin_marked_fagn_fixed_H0/results/`, `analysis_9_spin_marked_H0_fagn/results/` |
| L1 | `analysis_3_incomplete_catalog_H0_fagn/results/`, complete reference from the F1 grid of the same catalog |
| L2 | `analysis_4_density_anchoring_H0_fagn/results/` |
| L3 | `analysis_5_free_anchors_H0_fagn/results/`, `selection_redo/fu_seed101`, `fu_seed102` |
| mock description | `working/data/README.md`, `working/data/DESIGN_PE.md`, `working/data/seed100/META.json` |

Only the results obtained with the selection-function completeness estimate are used
for L1 to L3 (the live directories). Nothing from `working/archive/` enters.

## 8. Known gaps (kept out of the text; tracked in `notes/GAPS.md`)

The exploratory runs were made with four versions of the inference code, most of the
flux-limited and all of the host-dependent results rest on one simulated catalog, and
the Monte Carlo variance of ln L was not limited to one. The production runs listed in
`runs.txt` close these. Until then the affected numbers stay teal.

## 9. Positioning (binding; details and quotations in `notes/NOVELTY.md`)

- Do not write "first" for the joint inference of H0 with f_AGN, for a host mixture
  inferred with H0, or for a hierarchical likelihood for f_AGN with selection effects.
  Cite Bom & Palmese 2024, Cross-Parkin et al. 2026, Pouw et al. 2026, Gray et al.
  2026, Agapito & Mancarella 2026 and Zhu et al. 2024 for those.
- Do not use the verb "unify" for position and population information.
- The new statements are: non-AGN mergers assigned to a galaxy catalog that shares the
  density field with the AGN catalog; host-dependent mass and spin distributions
  inferred with the catalogs and with H0 free (quote the three deferrals); the size of
  the f_AGN and H0 response to the AGN number density; number densities inferred from
  the mergers.
- Present L1 for H0, the direction of F2, the degeneracy behind L2 and the mechanism
  of K1 as agreement with or extension of named prior work, not as discoveries.
- Cite only entries of `refs_candidates.bib`. For entries whose note says "record
  only", open the paper and confirm the statement you attribute to it before citing.
  A statement that cannot be confirmed is written without the citation and listed in
  `notes/CITATION_CHECKS.md`.
