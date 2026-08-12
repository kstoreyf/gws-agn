# Analysis 6 — owner report

**What it asked.** Analyses 3, 4 and 5 all sit on the diagonal of a
two-dimensional plane: the mock's two tracers share a magnitude distribution, so
every rung has `C_GAL = C_AGN` by construction. Real surveys never do. Vary the
galaxy and AGN survey depths **independently**, both completion densities at
truth, and ask whether the `f_AGN` bias is a function of the ratio
`C_AGN / C_GAL` rather than of either depth alone.

**Scope.** Seed 100, targeted lane. GAL ∈ {m20, m19, m18} × AGN ∈ {complete, m20,
m19, m18} = 12 cells; 8 new, 4 referenced from analyses 3 and 4 (never rerun, all
`scan_h0f.py` copies byte-identical). 6.57 GPU-h, zero guard rejections, one
darksirens SHA (`2b86a2d`) across all twelve.

**Sources.** `results/surface_summary.json`, `figs/fig_surface_f.*`,
`fig_surface_h0.*`, `fig_ratio_collapse.*`.

## Result 1 — relative completeness sets the *sign* of the bias

Hold the galaxy survey at `m<20` and vary only the AGN depth:

| AGN depth | C_AGN/C_GAL | f_AGN offset |
|---|---|---|
| m<18 | 0.12 | **−0.037** |
| m<19 | 0.39 | +0.031 |
| m<20 (diagonal) | 1.00 | +0.052 |
| complete | 1.23 | +0.051 |

An AGN catalog *shallower* than the galaxy catalog makes the fraction
**under**-estimated; deeper, over-estimated; matched depths give the smallest
bias. This is the useful statement for anyone designing a real measurement, and
it also reframes the whole campaign: **the diagonal analyses 3–5 happened to sit
on is the favourable ridge of this surface, not a generic point.** The bias we
have been quoting is close to the best case.

## Result 2 — it is the ratio, not either depth

Over all twelve cells, `log10(C_AGN/C_GAL)` explains the `f_AGN` offset with
**R² = 0.89**, against 0.63 for galaxy completeness alone and 0.30 for AGN
completeness alone. The single global relation is

```
f_AGN offset  ≈  0.067 + 0.124 log10(C_AGN / C_GAL)
```

## Result 3 — but the surface is genuinely two-dimensional

The global line leaves an rms of **0.024** in `f_AGN`, above this realisation's
own binomial scatter (0.0145), so it is not the whole story. Per galaxy depth:

| GAL depth | C_GAL | low-ratio slope | saturation level |
|---|---|---|---|
| m<20 | 0.814 | +0.132 / dex | not reached (max ratio 1.23) |
| m<19 | 0.315 | +0.127 / dex | +0.091 |
| m<18 | 0.095 | **+0.217 / dex** | **+0.194** |

Two structures the one-line fit hides. The response **saturates** once the AGN
catalog is a few times more complete than the galaxy catalog — the `m<18` row
runs +0.073, +0.186, +0.200, +0.197 across ratios 1.0 → 10.5, i.e. essentially
flat above ratio ≈ 3. And the **slope itself steepens at the faintest galaxy
depth**, +0.13/dex at `m<20` and `m<19` but +0.22/dex at `m<18`.

So a one-parameter correction in the ratio is good to about 0.02 in `f_AGN` and
no better — useful, but not a clean law. My earlier reading, when only the two
brighter rows were in, was that the slopes were universal (+0.087 vs +0.089); the
`m<18` row broke that, exactly where analysis 5 found the galaxy anchor railing
to ten times the true density. The two findings are the same pathology seen from
two directions.

## Result 4 — `H0` does not care at all

**R² = 0.0002** against relative completeness, over a surface where `f_AGN` moves
by 0.24. The `H0` offsets (+0.58 to +1.98 on this seed) track the galaxy depth,
not the ratio. Combined with analyses 3, 4 and 5, the campaign now has four
independent survey-modelling axes — completeness, density anchoring, anchor
freedom, relative completeness — and **`H0` is insensitive to every one of
them.** That is a strong, repeatedly-earned design statement.

## Result 5 — the detection always survives

Significance of a non-zero AGN component runs **4.6σ to 7.7σ** over every cell,
including the one where the fraction is under-estimated by 0.037. As in analysis
4: what relative completeness puts at risk is the *value* of `f_AGN`, never its
existence.

## Interpretation

The mixture weight the likelihood recovers is not the fraction of events in AGN;
it is closer to the fraction of events in AGN *that the two surveys can see*.
When one tracer's catalog is more complete than the other's, the completion
supplies the missing hosts of the shallower tracer from a smooth density field —
which is a good stand-in for a number but a poor one for individual galaxies — so
the better-observed tracer wins weight it has not earned. That predicts exactly
what is measured: the sign follows which tracer is better observed, the magnitude
follows the log of the ratio, and the effect saturates once the shallower tracer's
contribution is dominated by the completion rather than by real hosts.

The saturation is worth a sentence on its own: beyond ratio ≈ 3 there is nothing
more to gain or lose, because the shallow catalog is already almost entirely
model. And the steepening at `m<18` says that once the galaxy catalog is ~10 %
complete, the completion is doing so much of the work that the response to *any*
asymmetry is amplified — the same regime where analysis 5's galaxy anchor railed.

## Recommendation

**Main text for Results 1, 2 and 4; one paragraph for 3; appendix for 5.**

- Result 1 with `fig_surface_f` — the sign change is the single most useful
  number in this directory, and the "our ladder was the favourable ridge"
  reframing is honest and strengthens the paper rather than weakening it.
- Result 2 as the quotable relation, immediately followed by Result 3's rms as
  its stated accuracy. Do not present the relation without that bound.
- Result 4 folded into the campaign-level `H0` robustness statement alongside
  analysis 5's σ ratios.
- Result 5 as one appendix sentence.

**Caveats to state, all three:** one seed; no ratio below 1 at `m<18` (m18 is the
shallowest survey built, so the sign change is shown at `m<20` and `m<19` only);
and `m<20` never reaches the saturation regime, so the saturation level rests on
two rows.

**Follow-up this creates (not started, needs a gate).** The natural next step is
whether the correction is *invertible*: if a real analysis knows both surveys'
completeness — which it does, that is what selection functions are — can it
de-bias `f_AGN` using Result 2 and recover the diagonal answer? That is a
re-analysis of these twelve grids, no new compute. The honest doubt is Result 3:
a correction good to 0.02 against a σ(`f_AGN`) of 0.06 is a real improvement but
not a solved problem, and a second seed would be needed before quoting it as a
method.
