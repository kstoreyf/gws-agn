# paper/ — joint $H_0$ + AGN-hosted-fraction dark standard siren paper

Manuscript source for *"A dark standard siren measurement of the Hubble
constant and the AGN-hosted fraction of compact-binary mergers"*.

All sections are written against the dataset under `../data/seed100/` and the
inference runs under `../analyses/`. The earlier manuscript in `../report/` is
a frozen reference and is never edited from here.

## Journal target

**ApJ.** There is no main-text word limit and no cap on figures or tables. The
abstract limit is 250 words and is checked mechanically at submission, so count
it from the rendered PDF rather than from the source: macros expand, and a
source abstract under the limit can typeset over it. The document class stays
`aastex631` (still accepted; `aastex7` is the current one). There is no
per-journal class option: the journal is chosen at the submission portal, so
nothing in `main.tex` names it. Front matter ApJ expects and that is easy to
forget: an ORCID for every author, `\facility`, `\software`, a data-availability
statement, and a running title of 44 characters or fewer.

## Build

```bash
cd /hildafs/projects/phy230014p/magana/gws-agn/working/paper

python scripts/build_values.py      # values/results_macros.tex + NUMBERS.md
python scripts/make_figures.py      # figures/*.pdf + *.png
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Rebuild one figure with e.g. `python scripts/make_figures.py pgm`.

## Layout

```
main.tex                     documentclass, notation macros, \input of everything
sections/
  introduction.tex           dark standard sirens, why a second host catalog
  methods.tex                mixture of catalog redshift priors, survey-wide sky
                             normalisation, incomplete catalogs, selection function
  data.tex                   the simulated universe: one lognormal field, two
                             biased tracers, events measured once, flux limits
  results.tex                one catalog at a time; the two parameters jointly;
                             each catalog at equal event counts; flux-limited
                             catalogs
  validation.tex             closure across realisations, the sky-scramble null,
                             endpoint identities, the carried Monte-Carlo error
  discussion.tex             what the measurement establishes; what identifies fAGN
figures/                     generated PDF + PNG (do not edit; regenerate)
  fig_pgm.pdf                the graphical model
  fig_single_tracer.pdf      H0 from each catalog alone
  fig_joint.pdf              the joint (H0, fAGN) posterior
  fig_pure_tracer.pdf        each catalog on the events it hosts
  fig_incomplete.pdf         the flux-limited catalogs
  fig_closure.pdf            recovery across realisations
values/results_macros.tex    generated \newcommand for every quoted number
scripts/
  build_values.py            META.json + run outputs -> macros + NUMBERS.md
  audit_values.py            re-derives every macro and compares
  figstyle.py                the one visual system: palette, rc, shared helpers
  fig_*.py                   one script per figure
  make_figures.py            runs them all
references.bib               bibliography (built with bibtex)
NUMBERS.md                   generated macro -> source audit trail
```

## Conventions

* **No hand-typed numbers.** If a number belongs in the text, register it in
  `build_values.py` with the file that fixes its value and cite the macro.
  `NUMBERS.md` is regenerated from the same registry, so the audit cannot drift
  from the values. The registry has three kinds of entry:
  * *configuration* — constants that define the simulated universe and the
    fiducial setup, read out of `../data/seed100/META.json`;
  * *dataset* — properties measured on the generated realisation (realised
    densities, the recovered bias ratio, host counts, completeness), also from
    `META.json`;
  * *results* — computed from an inference run's output files.
* Every hook degrades to `\todo{pending}` when its source file is absent, so a
  missing run cannot silently produce a plausible-looking number.
* **`\todo{...}`** marks text waiting on a result still in production, and is
  also what an unresolved macro renders as. It is bold in the PDF so it cannot
  hide. Remove the definition in `main.tex` before submission and the build
  fails loudly if any remain.
* **Signed and scientific-notation macros are wrapped in `\ensuremath`**, so
  they render with a real minus sign in text and still work inside `$...$`.
* **Reader-facing text uses standard field terminology only.** No internal
  process vocabulary in the body, the captions or the abstract.
* **Figures follow one visual system** (`figstyle.py`): colours assigned by the
  job they do, a fixed categorical order, single-hue sequential ramps, and
  palette pairs checked with the colour-vision validator rather than by eye.
  Figure 1 is a diagram rather than a chart, so it encodes node type by fill
  lightness and stroke count — legible in greyscale and under colour-vision
  deficiency — and uses hue only for the plate labels.

## State

| section | state |
|---|---|
| abstract | written |
| §1 Introduction | written |
| §2 Method | written through the two-tracer flux-limited case |
| §3 Simulated data | written against `../data/seed100/META.json` |
| §4 Results | written, four subsections, every number a macro |
| §5 Validation | written |
| §6 Discussion | written |

There is no appendix: the equal-event-count comparison that once sat in one is
now §4.3. Any macro still awaiting a run renders as `\todo{pending}` and is
listed at the foot of `NUMBERS.md`; there are none at present.

The owner's ORCID is the one outstanding front-matter item; `main.tex` carries
the commented `\author[ORCID]{...}` line waiting for it.
