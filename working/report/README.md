# paper/ — joint $H_0$ + AGN-hosted-fraction dark-siren methods paper

Manuscript source for *"Dark-siren inference of $H_0$ and the AGN-hosted
fraction: what a sparse, strongly clustered tracer adds, and what the host
survey must deliver"* (working title; the original brief's title was "Joint H0
and host-fraction inference with multiple galaxy tracers in dark-siren
cosmology").

Everything is built from the experiment outputs under
`../analyses/experiments/`. Nothing in the manuscript is hand-typed: prose,
captions and tables reference LaTeX macros, and those macros are computed from
the experiments' results files by `scripts/build_values.py`. `NUMBERS.md` is the
generated audit trail mapping every macro to the file it came from.

## Build

```bash
conda activate jax
cd /hildafs/projects/phy230014p/magana/gws-agn/working/paper

JAX_PLATFORMS=cpu python scripts/build_values.py     # values/ + tables/ + NUMBERS.md
JAX_PLATFORMS=cpu python scripts/make_figures.py     # figures/*.pdf
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

`build_values.py` and the figure scripts are CPU-only (they read JSON and HDF5
and plot); `JAX_PLATFORMS=cpu` is set only to keep the import of anything
JAX-adjacent off the GPU. No inference is re-run here — the grids already exist.

Rebuild a single figure with e.g. `python scripts/make_figures.py n0_degeneracy`.

## Layout

```
main.tex                     documentclass, macros, \input of everything
sections/
  intro.tex                  dark sirens, statistical host id, why a second tracer
  methods.tex                mixture likelihood, sky normalisation, completion,
                             selection, targeted injections, kernel-width tension,
                             both mock families, inference practice
  results_fraction.tex       f_AGN across its planted range; the joint plane
  results_selection.tex      the catalog-targeted proposal changes the answer
  results_completeness.tex   flux-limit ladders, one tracer and two; the null
  results_density.tex        (f_AGN, n0_AGN) degeneracy; the density requirement
  results_budget.tex         error budget of the distance scale; sample variance
  results_scatter.tex        12-realisation two-tracer scatter, pre/post repair
  discussion.tex             complete (incl. the AGN space-density answer)
  conclusions.tex            complete
figures/                     generated PDFs (do not edit; regenerate)
tables/                      generated .tex tables (do not edit; regenerate)
values/results_macros.tex    generated \newcommand for every quoted number
scripts/
  build_values.py            results files -> macros + tables + NUMBERS.md
  figstyle.py                the one visual system: palette, rc, shared helpers
  fig_*.py                   one figure each, reading only results files
  make_figures.py            runs them all
NUMBERS.md                   generated macro -> source -> experiment audit
```

## Conventions

* **No hand-typed numbers.** If a number belongs in the text, add it to
  `build_values.py` with its source path and cite the macro. `NUMBERS.md` is
  regenerated from the same registry, so the audit cannot drift from the values.
* **Signed and scientific-notation macros are wrapped in `\ensuremath`**, so
  they render with a real minus sign in text and still work inside `$...$`.
* **`\todo{...}`** marks text waiting on a result still in production. It
  renders in bold so it is impossible to miss in the PDF. Remove the
  definition in `main.tex` before submission and the build will fail loudly if
  any remain. (The former `[CITE: ...]` placeholders are gone: the
  bibliography lives in `references.bib`, built with `bibtex main`; every
  entry's journal metadata has been verified against the journal of record —
  see the per-entry `% verified` comments.)
* **Figures follow one visual system** (`figstyle.py`): colours assigned by the
  job they do, a fixed categorical order, single-hue sequential ramps, and
  palette pairs checked with the colour-vision validator rather than by eye.
  Slots that fall below 3:1 contrast on the page always carry a direct label or
  a distinct marker.

## State

Complete and written from finalized results:

| section | source experiment |
|---|---|
| §2 Method (incl. the measured kernel-width window, §2.6) | all + `experiment_matched_mock` (`kernel_width_neff.json`, `skde_summary.json`) |
| §3 the fraction across its range | `experiment_h0f_baseline` |
| §4 the selection integral: proposal validity (pre-fix exhibit, same events), the post-fix measurement, the two-lever tilt mechanism, AGN anchoring | `experiment_twotracer_deep` (`summary.json` + `summary_fix.json`), `experiment_h0f_baseline` (`tilt_*.json/h5`) |
| §5 completeness, one tracer and two | `experiment_completeness_anchored`, `experiment_twotracer_incomplete` (`summary_fix.json`, `ladder_prepost_fix.json`) |
| §6 the density requirement | `experiment_completeness_free` (`n0_arms_summary_fix.json`, `fn0_*_fix.h5`) |
| §7 error budget (closes: 2 generator defects + estimator overhead, exact-likelihood oracle, post-fix campaign endpoint) + sample variance | `experiment_matched_mock` (`obsdet_summary.json`, `oracle_summary.json`, `obsdet_fix_summary.json`) |
| §8 catalog realisation, two tracers, pre/post-repair ensembles | `experiment_twotracer_seeds` (`seeds_summary.json`, `seeds_summary_fix.json`) |
| §9 conclusions | all |
| §10 discussion, incl. the AGN space-density literature answer (§10.2) | all + verified literature (`Ananna2022`) |

Nothing is stubbed. Post-fix convention: every K=2 matched-mock measurement is
quoted from the `_fix` reruns (repaired generator: detection on observed data,
observable-derived sky width); pre-fix numbers survive only inside the bias
budget (§7), the GLASS tilt narrative (§3.2/§4.2), the fixed-events
campaign-validity exhibit of §4.1, and explicit before/after comparisons
(`IncPre*`, `Deep*`/`DeepFix*`, `Sds*`/`SdsFix*` macro pairs). The GLASS
baseline (§3) and the K=1 anchored ladder (§5.1) were not regenerated — their
stories are the tilt mechanism and a differential ladder respectively, and
neither carries the matched-mock generator defects' repair hooks.

## Known gaps to close before submission

1. No author list beyond the first author, no acknowledgements, no software
   section, no data-availability statement (the one remaining `\todo`).

## Changelog

* 2026-07-30 (final assembly): all RESULTS sections repointed to the post-fix
  (`_fix`) reruns; `build_values.py` gained the `DeepFix*`, `IncPre*`,
  `IncFacHzeroFirst`/`IncHzeroOff*`, `IncCompleteFscan*` and `SdsFixFscan*`
  macros and repointed §5.2–§6 sources; tables `tab_twotracer_ladder`,
  `tab_n0_width` and figures `fig_completeness_twotracer`,
  `fig_n0_significance`, `fig_n0_degeneracy` now read `_fix` files
  (`fig_closure_waterfall` already carried the fix-arm endpoint); the
  pre-repair "intermediate-depth improvement" claim was withdrawn (it does not
  survive the repair) and is kept only as a before/after caution; the
  bibliography's `TODO-VERIFY` entries were verified online (Gayathri2021
  confirmed; BomPalmese2023 updated to PRD 110, 083005, 2024) and `Ananna2022`
  added to back the new §10.2 AGN space-density paragraph. 801 macros, 0
  pending, audit clean.
