# The paper as we expect it to come out (skeleton of 17 Nov 2026)

The owner asked for a complete draft now, written for the results we expect, with every result that
waits for a production run shown as a placeholder: **numbers as red macros, figure data invented and
drawn in red, figures otherwise publication ready.** The draft will change where the runs disagree
with the expectation. This file is binding for the writer and for the numbers and figure work. Where
it differs from `notes/DESIGN.md` or from `REVISION_DECISIONS.md` items 1 to 12, this file wins.
`REVISION_DECISIONS.md` item 13 and `notes/OWNER_STYLE_DETAILS.md` are the owner's words and win over
everything.

## 0. Marking
- Final (black): design values, anything from Analysis 14 and Analysis 15, Fig. 1.
- Placeholder (red): every number and every plotted curve, point, band or contour that a production
  run will replace. Two sources, recorded per item in `values/placeholder_results.json`:
  `exploratory` (an earlier run of the same analysis under the old settings; use its value) and
  `invented` (no run exists; use the expectation of Sec. 7). Both print red. There is no teal any more:
  `\anc` and `\ph` both render red.
- A figure panel that contains placeholder data carries the small red word "placeholder" in a corner,
  and its placeholder elements are drawn in red tones told apart by line style and direct labels.
  When a production summary exists for an item the same code draws it in the final palette and the
  macro turns black. Nothing else changes.

## 1. The paper in one paragraph
Gravitational-wave mergers without counterparts measure H0 through the galaxies that could host them.
Some binary black holes may merge in the disks of active galactic nuclei (AGN), where they would be
heavier and spin faster. We write one likelihood in which each merger comes from a galaxy or from an
AGN, each class has its own catalog and its own mass and spin distribution, and H0, the fraction of
mergers hosted by AGN and the populations are inferred together. We then remove one assumption at a
time. With the population known and complete catalogs the method measures H0 and the AGN fraction
together (the anchor). With flux-limited catalogs the measurement holds, and what the mergers
constrain is the merger rate per host of one class relative to the other, so the AGN fraction is only
as good as the assumed number densities. With the population inferred, part of the H0 information
that came from the mass scale is given back, and the catalogs supply what remains. With populations
that differ between the host classes, an analysis that assumes one population is biased in H0 and the
host-dependent analysis is not; it measures the differences in mass and spin as well. With a mass
spectrum like the observed one, in which few mergers sit in the Gaussian peak, the same holds with
smaller numbers.

## 2. Names (one object, one name; the same words in text, tables, legends and captions)
- Unit of the simulation: **realization** (one density field with its two catalogs, 1000 detected
  mergers and injections). "Realization A" is the one shown in single-realization figures.
- Catalogs: **the galaxy catalog**, **the AGN catalog**; **both catalogs**, **galaxy catalog alone**,
  **AGN catalog alone**, **no catalog** (hosts uniform in comoving volume).
- Event sets: **equal populations** (both host classes draw masses and spins from one distribution);
  **host-dependent populations** (AGN-hosted mergers have the mass peak 5 Msun higher and the mean
  effective spin 0.10 higher).
- Mass spectra: **peak-dominated** (0.90 of the mergers in the Gaussian peak; a design choice that
  makes the mass scale a sharp ruler) and **observed-like** (power law plus peak with the values
  fitted to the third gravitational-wave catalog; a few percent of the mergers in the peak).
- Analyses, named by what they hold or free: **population known**; **single population** (one
  distribution for both classes, its mass-peak location and width and its spin mean and width
  inferred); **host-dependent populations** (the same plus the two AGN offsets); **number densities
  inferred**.
- "The fraction of mergers hosted by AGN", symbol f_AGN. "The merger rate per host" and its ratio
  between the classes. No letters, rungs, run names, seeds or code names anywhere a reader can see.

## 3. Layout (REVTeX 4.2, two columns, 14 to 15 pages)
Title: keep, or propose a better one that names the host-dependent populations and not the anchor.
Abstract: 6 to 8 sentences: scope; method in one clause; the headline (host-dependent events: bias of
the single-population analysis, recovery by the host-dependent analysis, the offsets) with interval
type and assumptions in the same sentence; the observed-like spectrum; the flux-limited result (rate
per host); validation in one clause; implication. The anchor is not in the first half.

I. Introduction. No headline numbers. Last paragraph: roadmap, cosmology, priors, 90% convention.

II. Method (short).
  A. One likelihood: mixture over the two host classes; each class with its catalog term (cataloged
     hosts plus hosts the catalog misses, uniform in comoving volume) and its population; selection.
     Then the limits, each one sentence: f_AGN = 0 or one catalog recovers the usual dark siren
     analysis; no catalog recovers the spectral siren analysis; equal populations; population held
     fixed.
  B. What the mergers measure: the normalization identity, the merger rate per host (cite the earlier
     single-class definition, see `notes/PRIOR_WORK_VERIFIED.md` F8).
  C. Parameters and priors (Table I).

III. Simulated catalogs and mergers. Galaxies and AGN on one density field (Fig. 1); mergers and
  their measurement; the two event sets; the two mass spectra; flux-limited catalogs (Table II).

IV. Results. Each subsection is titled by the physics it frees and opens with what is now free and
  what is found.
  A. Known population and complete catalogs (the anchor; short). Fig. 2, first rows of Table III.
  B. Flux-limited catalogs. Number densities known, then inferred; the rate per host. Figs. 3, 4,
     Table IV.
  C. An unknown population. Single population inferred; how much of H0 the mass scale carried; the
     catalogs break the degeneracy between H0 and the mass peak. Fig. 5.
  D. Populations that depend on the host. The headline. Fig. 6.
  E. A mass spectrum like the observed one. Fig. 7.
  F. One paragraph that reads Fig. 8 (H0 and f_AGN under each set of assumptions) and Table III.
  Bayes factors: one sentence each, in the subsection where the assumption is discussed, against a
  reference named in that sentence, with prior ranges from Table I.

V. Validation (physics tests, not process). Recovery of the inputs in the posterior combined over
  realizations, with the credible level at which each input lies (Fig. 9a); mergers with no AGN hosts
  and with one in ten (Fig. 9b); accuracy of the Monte Carlo sums in km/s/Mpc beside the posterior
  width (Analysis 14; details in Appendix A); sky resolution (one sentence).

VI. Conclusions. What is established; the inputs it rests on (AGN number density and completeness,
  the form of the mass spectrum, redshift errors); near-term improvement; the step change and its
  date (next observing runs, deeper AGN catalogs); the last sentence hands the result on. No
  Limitations paragraph: each caveat is the last clause of the claim it limits, with a size.

Appendix A. Evaluation of the likelihood and its Monte Carlo accuracy (Analysis 14 numbers; no
  figure). Appendix B. Measurement model of the simulated mergers.

## 4. Figures (nine; every interval, band and contour 90%; legends repeat the caption's words)
File names are fixed so the text and the figure work can proceed in parallel.
- `fig1_mocks`: as now (final).
- `fig2_anchor`: (a) joint posterior of H0 and f_AGN, realization A and the combination over
  realizations, input marked; (b) H0 posterior with both catalogs, galaxy catalog alone, AGN catalog
  alone, no catalog (population known).
- `fig3_flux_limits`: (a) 90% intervals of H0 and of f_AGN against the flux limit (complete, 21, 20,
  19, 18), with the completeness weighted by the detected mergers on the upper axis; (b) f_AGN
  against the assumed AGN number density (factor 0.5, 1, 2) for two limits and both normalization
  volumes where Analysis 15 has them, with the identity's prediction as black curves.
- `fig4_densities`: (a) joint posterior of f_AGN and log10 n_AGN with the densities inferred, two
  flux limits, the curve of constant rate-per-host ratio through the input, and the region the
  priors alone allow in gray; (b) posterior of the rate-per-host ratio at each flux limit, input
  marked.
- `fig5_population`: (a) H0 posterior: population known and single population inferred, each with
  both catalogs and with no catalog (four curves); (b) joint posterior of H0 and the mass-peak
  location, no catalog and both catalogs.
- `fig6_host_dependent`: host-dependent events, peak-dominated spectrum. (a) H0 posterior of the
  single-population analysis and of the host-dependent analysis, combined over realizations, input
  marked; (b) joint posterior of the two AGN offsets with the input and zero marked; (c) f_AGN
  posterior of the two analyses.
- `fig7_observed_like`: the same three panels for the observed-like spectrum.
- `fig8_summary`: one row per analysis in the order of Sec. IV, labeled in words, with the 90%
  interval of H0 (left panel) and of f_AGN (right panel) and the inputs as vertical lines. This
  figure tells the paper; it must read on a phone.
- `fig9_validation`: (a) posteriors of H0 combined over realizations for the three analyses with
  both catalogs (population known, single population, host-dependent populations), input marked;
  (b) f_AGN posteriors for inputs 0, 0.10 and 0.30.

## 5. Tables (four)
- Table I: parameters, priors, and the value at which each is held when not inferred.
- Table II: the simulated catalogs: number densities, biases, redshift errors, flux limits with the
  redshift to which each is complete and the completeness weighted by the detected mergers, the two
  mass spectra, the host-dependent offsets.
- Table III: results by analysis. Rows in the order of Sec. IV, named as in Sec. 2. Columns: H0,
  f_AGN, AGN offset of the mass peak, AGN offset of the spin mean, log10 B. The caption names the
  reference of each Bayes factor (the single-population analysis of the same events) and the prior
  ranges. Only rows that have a counterpart with one fewer assumption carry a Bayes factor.
- Table IV: flux-limited catalogs: per limit, the completeness, H0, f_AGN, the rate-per-host ratio,
  and with densities inferred the 90% limits on log10 n_AGN beside what the prior alone gives.

## 6. Macros: one scheme
`\<Key><Quantity>` for realization A, `\<Key><Quantity>Comb` for the combination over realizations,
`\<Key><Quantity>CL` for the credible level at which the input lies in the combination. A macro
prints median and 90% interval (`68.1^{+1.7}_{-1.6}`); `...Med`, `...Lo`, `...Hi`, `...Wid` (full
90% width) give the pieces. Quantities: `Hzero`, `fAGN`, `MuG`, `SigmaG`, `MuChi`, `SigmaChi`,
`DmuG`, `DmuChi`, `LognGal`, `LognAGN`, `RateRatio`, `LB` (log10 Bayes factor, reference stated in
`values/MACROS.md`). Keys:

| Key | Events | Analysis | Catalogs | Free |
|---|---|---|---|---|
| `KnownBoth` | equal populations | population known | both, complete | H0, f_AGN |
| `KnownGal`, `KnownAGN`, `KnownNone` | equal populations | population known | galaxy alone, AGN alone, none | H0 |
| `AGNhosted`, `Galhosted` | all AGN-hosted, all galaxy-hosted | population known | matching catalog | H0 |
| `CtrlZero`, `CtrlTenth` | no AGN-hosted mergers; one in ten | population known | both | H0, f_AGN |
| `FluxTwentyOne`, `FluxTwenty`, `FluxNineteen`, `FluxEighteen` | equal populations | population known, densities from catalog counts | both, flux-limited | H0, f_AGN |
| `FreeTwentyOne` ... `FreeEighteen` | equal populations | number densities inferred | both, flux-limited | H0, f_AGN, two densities |
| `SingleBoth`, `SingleNone` | equal populations | single population | both; none | H0, f_AGN, four population parameters |
| `HDSingle`, `HDHost`, `HDNone` | host-dependent | single population; host-dependent populations; single population | both; both; none | six; eight; five |
| `ObsSingle`, `ObsHost`, `ObsNone` | host-dependent, observed-like spectrum | as the three above | as above | as above |

Further macros: `\HDBiasHzero` and `\ObsBiasHzero` (median of the single-population analysis minus
the input, combined), `\HDLB`, `\ObsLB` (host-dependent against single population), `\WidthRatioAGNGal`
(H0 width, galaxy-hosted over AGN-hosted), `\MassScaleShare...` (H0 width with no catalog over the
width with both catalogs, for population known and for single population), `\CorrHzeroMuGNone`,
`\CorrHzeroMuGBoth`. Design values and the Analysis 14 and 15 macros keep the names in
`values/MACROS.md`. The writer may ask for more in `notes/MACRO_REQUESTS.md`.

## 7. What we expect (source of the invented placeholders; median and 90% half-width, realization A;
the combination over N realizations is narrower by about sqrt(N) and centered nearer the input)
Inputs: H0 67.74, f_AGN 0.30, mass peak 35 and width 5 Msun, spin mean 0 and width 0.10, AGN offsets
+5 Msun and +0.10. Use the exploratory value instead wherever an earlier run of the same analysis
exists (most of `Known...`, `AGNhosted`, `Galhosted`, `Flux...`, `Free...`, and `HDHost` from the
eight-parameter run of Analysis 13); record which.
- `KnownNone`: H0 68.6 +- 3.2. The mass scale alone, with the population known, already measures H0;
  the catalogs narrow it by a factor of about two.
- `CtrlZero`: f_AGN below 0.05 at 90%. `CtrlTenth`: f_AGN 0.11 +- 0.05.
- `SingleBoth`: H0 68.0 +- 2.6; f_AGN 0.29 +- 0.09; mass peak 34.9 +- 0.6, width 5.0 +- 0.4; spin
  mean 0.000 +- 0.010, width 0.100 +- 0.008. `SingleNone`: H0 69 +- 9, mass peak 34.6 +- 1.6.
  Correlation of H0 with the mass peak: -0.9 with no catalog, -0.5 with both.
- `HDSingle`: H0 70.9 +- 2.6 (biased high by about 3), f_AGN 0.24 +- 0.09, mass peak 36.4 +- 0.6,
  spin mean 0.030 +- 0.010. `HDHost`: H0 67.4 +- 2.1, f_AGN 0.30 +- 0.06, offsets 5.1 +- 1.0 Msun
  and 0.10 +- 0.03. `HDNone`: H0 73 +- 9. `\HDLB` about 9.
- `ObsSingle`: H0 68.4 +- 2.4 (biased high by about 0.7), f_AGN 0.27 +- 0.09. `ObsHost`: H0 67.9
  +- 2.3, f_AGN 0.30 +- 0.08, offsets 4.5 +- 3.5 Msun and 0.10 +- 0.04. `ObsNone`: H0 70 +- 13.
  `\ObsLB` about 3. (Revised 17 Nov evening: with the population known the observed-like
  half-width is 1.98 with both catalogs, as for the peak-dominated spectrum, and 8.44 with no
  catalog, so with both catalogs H0 rests on the hosts and the earlier expectation of +- 5 was
  far too wide; the bias of a single population, which acts through the mass scale, is
  correspondingly small.)
- Rate-per-host ratio (AGN over galaxies): input (0.30/0.70)(n_gal/n_AGN) = 43; with densities
  inferred the ratio is measured to about 30% while f_AGN and log10 n_AGN each follow their priors
  along the curve of constant ratio.

## 8. Who does what
- Numbers and figures (Opus): `scripts/placeholder_results.py` writing
  `values/placeholder_results.json` (one entry per key and quantity: median, 90% interval, a
  posterior shape for plotting, source); a map `values/production_map.json` from key to the summary
  file of the production run (empty now), which overrides the placeholder when the file exists;
  macros per Sec. 6; figures per Sec. 4; `figures/CAPTION_FACTS.md`; `placeholders.txt` regenerated.
- Text (Fable): `main.tex` per Secs. 1 to 5, the style note, the review and the verified prior work.

## 9. Addendum of 17 Nov, evening: the long paper takes in the letter's experiment
The letter's grids (Analysis 16, stages E1 and E2) answer a question the long paper did not ask:
which mergers carry the H0 constraint, and what is lost because the host class of a merger is not
known. Most of it is real now; the rest finishes on 18 Nov. This section is binding like the rest
and wins over Secs. 3 to 6 where they differ.

### 9.1 What was found (population known, complete catalogs, 1000 detected mergers, means over five
realizations of the 90% half-width of H0 in km/s/Mpc; the numbers in the text come from macros)
| fraction of mergers hosted by AGN | 0 | 0.1 | 0.3 | 0.5 | 1 |
|---|---|---|---|---|---|
| host class unknown (both catalogs, f_AGN inferred) | 3.53 | 3.49 | 2.01 | 1.36 | 0.82 |
| host class known | 3.55 | 1.89 | 1.34 | 1.10 | 0.81 |
No catalog (the mass scale alone): 5.57. Observed-like spectrum: no catalog 8.44; host class
unknown 3.49 (0), 1.98 (0.3), 0.84 (1); host class known 3.49, 1.34, 0.83. (Values of 17 Nov,
21:00 UTC, from the run summaries on the refined lattice; `values/MACROS.md` is the source.)
- With one merger in ten hosted by an AGN the analysis gains nothing from the AGN catalog (3.49
  against 3.53), although the same mergers with their host class known would give 1.89. Positions
  alone barely say which mergers are AGN-hosted: at a fraction of 0.3 and equal populations the
  mean probability of an AGN host is 0.33 for mergers truly hosted by AGN and 0.24 for mergers
  truly hosted by galaxies (0.37 and 0.30 on the mergers with host-dependent populations, of
  which more are AGN-hosted).
- With host-dependent populations (AGN-hosted mergers heavier by 5 Msun, spin mean higher by 0.10)
  and the populations known, the masses and spins move those probabilities from 0.37 and 0.30 to
  0.51 and 0.28, and the half-width at 0.3 is 1.64, between the 2.01 of the unknown and the 1.34
  of the known host class. The comparison is not between the same mergers (360 against 298 of the
  1000 detected are AGN-hosted); on the same mergers the known host class gives 1.30.
- With the host class known at a fraction of 0.3, the AGN-hosted mergers alone give 1.41 where
  all mergers give 1.34: by the ratio of the squared half-widths they carry about 90% of the
  information on H0 (72% at a fraction of 0.1). The posteriors multiply exactly, but a share read
  off half-widths is exact only for Gaussian posteriors (the two shares add to 107% at 0.3), so
  the text says "about nine tenths" and never "exactly". Do not quote a share for the analysis
  with the host class unknown; no unique one exists.
- The half-width at 0.3 with the host class unknown in this experiment (2.01, realization A 2.09)
  and the one of Sec. IV A (realization A 1.94) are the same mergers and the same analysis with
  two injection sets; the difference is the Monte Carlo error that the Validation section
  quotes. The text does not equate them and does not set them against each other.
- The AGN catalog alone, applied to mergers of which seven in ten are hosted by galaxies, has no
  interior maximum in four of five realizations; in the fifth it peaks at 76.8 [75.5, 78.0] and
  excludes the input. It is a biased analysis, not only an uninformative one, and the text says so.

### 9.2 Names (added to Sec. 2)
- **host class unknown**: the analysis of every other section (both catalogs, f_AGN inferred). Named
  only where it is contrasted with the next.
- **host class known**: each merger analyzed with the catalog and the population of its true host
  class; the two H0 posteriors multiply. Available in simulation only, and said so once.
- **probability of an AGN host** of a merger: f L_AGN / (f L_AGN + (1 - f) L_gal) at the posterior
  median, one equation in Sec. II A, read off the mixture likelihood.

### 9.3 Layout changes
- New subsection IV B, "The constraint from each host class" (`sec:classes`), between the known
  population and the flux-limited catalogs. It takes the last paragraph of IV A (candidate hosts per
  localization volume; all mergers in AGN against all in galaxies) and adds the table above as a
  figure, the fraction 0.1 result, the comparison with the host class known, the number of mergers
  the unknown host class costs, and the probability of an AGN host from positions alone. The later
  subsections move down by one letter.
- IV A: the AGN catalog alone, with the exception of 9.1.
- Populations that depend on the host: one paragraph on masses and spins as a label (probability of
  an AGN host with and without them, half-width with the populations known), with the confound in
  the next sentence (more of the detected mergers are AGN-hosted when these are heavier).
- A mass spectrum like the observed one: the half-widths of 9.1 for the population known open the
  subsection; they are real, the host-dependent runs of that subsection are not.
- Abstract: at most one sentence on this result, and the abstract stays within eight sentences.
- Table I: number densities in the units the analysis samples, h^3 Mpc^-3; the inputs in the same
  units wherever a prior range is quoted beside them.

### 9.4 Figures
- New `fig_classes` (after `fig2_anchor`), three panels, the content of the letter's `fig1_budget`
  (a, b) and `fig_populations` (c) in this paper's names and style:
  (a) 90% half-width of H0 against the fraction of mergers hosted by AGN for the peak-dominated
  spectrum, population known: host class unknown; host class known; host-dependent populations
  (the two sizes of the offsets, 5 Msun and 0.10, and twice those) with the host class unknown;
  the open symbol for host-dependent populations inferred with H0 (stage 4); horizontal lines for
  no catalog and for the galaxy catalog alone with no AGN-hosted merger. Means over realizations,
  bars the range over realizations.
  (b) the same for the observed-like spectrum.
  (c) distribution over the mergers of realization A (host-dependent populations, fraction 0.3) of
  the probability of an AGN host, for mergers truly hosted by AGN and truly hosted by galaxies,
  from positions alone (top) and from positions with masses and spins (bottom); means marked, the
  fraction as a line.
- `fig6_host_dependent` keeps its three panels.
- `fig1_mocks` panels (c, d) are redrawn from the production event sets.

### 9.5 Macros
Names follow Sec. 6 where a key exists and are otherwise chosen by the numbers work and listed in
`values/MACROS.md` with their definitions; the writer uses only what is listed there.

### 9.6 One paper (owner, 17 Nov, ~20:05 UTC)
"Lets stick to the long paper. You did a better job there. So i would archive the short paper."
The letter is frozen in `working/archive/paper_short_independent/`. This paper is the only one and
stands alone: the code it shared with the letter lives in `paper_prd/scripts/`, and the letter's
experiment (stages E1 to E3 of Analysis 16) is now this paper's Sec. IV B and part of the
host-dependent and observed-like subsections.
EOF
