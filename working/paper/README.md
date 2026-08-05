# paper/ — joint $H_0$ + AGN-hosted-fraction dark standard siren paper

Manuscript source for *"A dark standard siren measurement of the Hubble
constant and the AGN-hosted fraction of compact-binary mergers"*.

Sections 1–3 and 5 are written against the v2 dataset under
`../data/seed100/`; section 4 carries the claim structure with pending macros
until the inference runs return. The earlier manuscript in `../report/` is a
frozen reference and is never edited from here.

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
  results.tex                the three claims, numbers pending
  discussion.tex             what the measurement establishes; what identifies fAGN
figures/                     generated PDF + PNG (do not edit; regenerate)
  fig_pgm.pdf                Fig. 1, the graphical model
values/results_macros.tex    generated \newcommand for every quoted number
scripts/
  build_values.py            META.json + run outputs -> macros + NUMBERS.md
  figstyle.py                the one visual system: palette, rc, shared helpers
  fig_pgm.py                 Fig. 1
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
  * *results* — computed from an inference run's output files. All ten are
    pending: the runs against the v2 dataset have not returned.
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
| abstract | written, headline numbers pending |
| §1 Introduction | written |
| §2 Method | written through the two-tracer incomplete case |
| §3 Simulated data | written against `../data/seed100/META.json`, with Fig. 1 |
| §4 Results | claim structure written, all numbers pending |
| §5 Discussion | written, one closing paragraph pending |

Pending macros (rendered as `\todo{pending}`) are listed at the foot of
`NUMBERS.md`.
