# Analysis 6 — owner report (selection-mode redo)

**What it asked.** Analyses 3 and 4 left one mechanism standing for the archived
`f_AGN` bias: **relative** completeness. The two tracers are completed
separately, so what should matter is not how complete either catalog is but how
complete one is *relative to the other*. Analysis 6 tests that directly by
breaking the diagonal — galaxy depth and AGN depth varied independently over a
grid — and the archived campaign found a clean law:

> `f_AGN offset = 0.068 + 0.124 · log10(C_AGN / C_GAL)`,  R² = 0.86

with the sign of the bias set by which tracer is better observed. This redo asks
whether that law is a property of the inference or of the per-pixel estimator.

8 off-diagonal cells; the 3 diagonal cells live in analysis 3 and the GAL `m<18`
× AGN complete cell is analysis 4's oracle probe. Anchors fixed at truth in
every cell, deliberately, so the surface is not confounded with the anchoring
axes of analyses 4 and 5.

**Sources.** `results/joint_g{gal}_a{agn}_s100.json`,
`results/surface_summary.json`, `figs/fig1_ratio_law.*`. darksirens `0c5b3db`,
K = 2 field mixture, 1000 events, targeted-injection lane, seed 100.

## Result

**The law is gone.**

| | slope per dex | intercept | R² | span of offsets |
|---|---|---|---|---|
| per_pixel (archived) | **+0.123** | +0.068 | 0.86 | 0.237 |
| selection (this work) | **−0.004** | −0.027 | 0.008 | 0.062 |

The slope falls by a factor 34 and the correlation vanishes. The offsets no
longer span a quarter of the `f_AGN` range across the surface; they span 0.06,
which is about one 90 % half-width of a single cell.

**And the residual structure is not the ratio.** Splitting the eight cells by
whether the AGN catalog is flux-limited or carried as *complete*:

| subset | slope | intercept | span |
|---|---|---|---|
| 6 cells, both catalogs flux-limited | +0.005 | **−0.0145** | 0.016 |
| 2 cells, AGN complete (`gm19_acomplete`, `gm20_acomplete`) | — | −0.065, −0.069 | — |

The six genuinely flux-limited cells sit at a constant −0.0145 with a span of
0.016 — indistinguishable from analysis 4's depth-independent −0.0135, measured
on a different axis. The whole appearance of structure in the selection surface
is the two AGN=complete cells. Under per_pixel the same split changes nothing:
restricted to the six flux-limited cells the law is *stronger* (slope +0.131,
R² = 0.93), so this is not a subset artifact.

**`H0` is untouched by relative completeness**, as the archived campaign also
found: slope −0.066 per dex at R² = 0.07 across a 1.86-dex span of the ratio,
with all eight cells inside 68.79–69.24 — a range of 0.45 against a per-cell
90 % half-width of ~1.6.

## Interpretation

Relative completeness was the last mechanism standing for the archived bias, and
it does not survive the estimator change. The archived law was real as a
measurement and wrong as a physical statement: what it tracked was the per-pixel
estimator's error, which grows with how empty the pixels are, and the two
tracers' depths set that emptiness. Analysis 7 measures the same thing along its
own axis and agrees — per_pixel's offset moves +0.052 per dex of pixel
occupancy, selection's +0.028.

This closes the mechanism hunt that analyses 3, 4 and 6 were built around. The
answer to "what makes `f_AGN` biased when the catalogs are incomplete" is: under
the per-pixel completeness estimator, the estimator does; under selection, at
this seed, nothing measurable does — the residual is a constant −0.0145 with no
dependence on absolute completeness (analysis 4, four rungs), on relative
completeness (here, six cells), or on pixel occupancy beyond +0.028/dex
(analysis 7).

What remains unexplained is the −0.06 that appears whenever the AGN catalog is
carried as complete: here in two cells, in analysis 3's complete rung (−0.055)
and in analysis 4's oracle probe (−0.056). Four cells, four different galaxy
catalogs, one number. It is *not* the relative-completeness law returning — the
ratio for `gm20_acomplete` is +0.089 and for `gm19_acomplete` is +0.500, nearly
a factor six apart, and they give the same offset to 0.004.

**One seed.** `σ(f_AGN)` per cell is ≈ 0.045, so neither the −0.0145 nor the
−0.06 is individually resolved. What the surface *does* resolve is that the
cells agree with each other, which is a paired statement across cells that
share events and injections, and that is what kills the law.

## Recommendation

**Main text, and it is a retraction.** The archived relative-completeness law
must not appear in the paper. If the campaign's earlier framing has already
propagated into draft text — the sign of the `f_AGN` bias being set by which
tracer is better observed — that passage needs removing, not softening.

**What replaces it.** A one-line null: over a 1.86-dex range of relative
completeness, at fixed anchors, `f_AGN` moves by −0.004 ± per dex and `H0` by
−0.07, both consistent with no dependence. That is a stronger result for the
method than the law was against it.

**Next.** The two AGN=complete cells are the only unexplained structure left in
this directory, and they are cheap to test — see analysis 3's recommendation.
Until then quote the six-cell subset, not all eight.
