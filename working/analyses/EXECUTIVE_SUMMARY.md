# Executive summary — the AGN dark-siren campaign, analyses 0–7

*2026-08-12. Written at the close of the selection-mode redo (48/48 cells,
committed `a5020c1`). Covers every analysis in `working/analyses/`.*

---

## The one-paragraph version

The method works on complete catalogs and the paper's core results are safe. The
incomplete-catalog half of the campaign, however, was measuring its own
completeness estimator rather than the physics: re-running analyses 3–6 with the
parametric selection estimator instead of the legacy per-pixel one **flips the
sign of the `f_AGN` bias, erases the relative-completeness law, reverses the
oracle probe, and removes the faintward growth in anchoring sensitivity**. Three
of the campaign's four "mechanism" findings do not survive. What replaces them is
better for the method and simpler to state: at fixed density anchors, `f_AGN` and
`H0` are insensitive to completeness. Two things are newly open — `H0`'s
robustness turns out to be conditional on catalog depth, and every cell carrying
the AGN catalog as "complete" is offset by 0.06 in `f_AGN` for reasons not yet
established.

---

## Where each analysis stands

| | what it asked | status | headline |
|---|---|---|---|
| **0** | pure-tracer `H0`, one tracer at a time, matched N | done, unaffected | both tracers recover `H0`; the campaign-wide +1.4 offset is seed 100's own draw, not a bias |
| **1** | complete-catalog `H0` closure | done, unaffected | closes; both matched-host controls sit on truth |
| **2** | complete-catalog joint `(H0, f_AGN)` | done, unaffected | K=2 mixture recovers both: `+0.41 ± 0.55`, `−0.012 ± 0.020` over 5 seeds |
| **3** | cost of magnitude-limiting both catalogs | **redone** | the archived `+0.084 ± 0.019` bias was the estimator; under selection it flips sign and shrinks faintward |
| **4** | mis-anchoring the AGN completion density | **redone** | sensitivity survives (0.43–0.47/dex, now depth-independent); oracle probe swings 0.25; `H0` immunity is conditional |
| **5** | both densities free under flat priors | **redone** | selection recovers the galaxy anchor where per_pixel rails by 1.19 dex; evidence prefers it by 17.7 in ln Z |
| **6** | GAL depth × AGN depth surface | **redone** | the relative-completeness law is gone (slope +0.123 → −0.004, R² 0.86 → 0.008) |
| **7** | does the bias scale with pixel occupancy? | new, PARTIAL | selection removes the bias's level but a +0.028/dex residual survives an estimator immune to pixel counts |

Analyses 0–2 are outside the blast radius by construction: they run at
`log10 n0 = −24`, which switches the completion term off entirely, so no
completeness estimator is ever invoked. Everything the redo changed is downstream
of having an *incomplete* catalog.

---

## What the redo changed, with numbers

All figures below: seed 100, darksirens `0c5b3db`, K = 2 field mixture, 1000
events, targeted-injection lane, anchors at the mock's truths unless stated.
Truth `f_AGN = 0.295`, `H0 = 67.74`; seed 100's own complete-catalog `H0` draw is
69.22.

**The `f_AGN` bias was the estimator (analysis 3).**

| rung | C in horizon | per_pixel | selection |
|---|---|---|---|
| complete | 100 % | +0.047 | −0.055 |
| m<21 | 99.7 % | +0.047 | −0.022 |
| m<20 | 81.4 % | +0.052 | −0.022 |
| m<19 | 31.5 % | +0.066 | −0.020 |
| m<18 | 9.5 % | +0.073 | −0.009 |

The archived monotone growth becomes a monotone *shrink*, and the intervals are
~25 % tighter at every rung. The completeness axis costs σ(f) ×1.13 and σ(H0)
×1.07 over a 10× loss of completeness — cheaper than the archived ×1.22 / ×1.14.

**The relative-completeness law does not exist (analysis 6).** Over eight
off-diagonal cells spanning 1.86 dex of `C_AGN / C_GAL`:

| | slope per dex | R² | span of offsets |
|---|---|---|---|
| per_pixel | +0.123 | 0.86 | 0.237 |
| selection | −0.004 | 0.008 | 0.062 |

Restricted to the six cells where *both* catalogs are flux-limited, selection
gives a constant −0.0145 with a span of 0.016 — the same number analysis 4
measures independently as its depth-independent offset (−0.0135). Per_pixel's law
gets *stronger* on the same subset (+0.131, R² 0.93), so this is not a
subset artifact.

**Selection recovers a parameter the legacy estimator rails on (analysis 5).**
With both densities free under flat priors:

| rung | `log10 n0`, per_pixel | selection | Δ ln Z |
|---|---|---|---|
| m<21 | −3.535 | −3.198 | −0.34 |
| m<20 | −3.454 | −3.256 | −0.36 |
| m<19 | −3.038 | −3.043 | **+6.7** |
| m<18 | **−1.808** (railed) | **−3.118** | **+17.7** |

Truth is −3.0. The evidence separates the estimators exactly on the rungs where
completion matters and ties where it cannot — the cleanest internal validation in
the campaign.

**`H0` is robust, conditionally (analysis 4).** This is new and it is a
limitation, not a win:

| rung | d`H0`/dlog₁₀(density factor) | `H0` spread over factor 0.5–2.0 |
|---|---|---|
| m<21 | −0.017 | 0.010 |
| m<20 | −0.035 | 0.022 |
| m<19 | +0.709 | 0.42 |
| m<18 | **+3.211** | **1.91** |

At `C ≥ 81 %` a factor-2 error in the assumed AGN density does not move `H0` at
all. At `C = 9.5 %` it moves it by ~1 km s⁻¹ Mpc⁻¹ — over half the posterior's
own 90 % half-width — along a clean line in log(factor), R² = 0.99. `H0` remains
insensitive to *relative* completeness (slope −0.066/dex, R² 0.07).

---

## The two open items

**1. The AGN=complete offset.** Four cells, four different galaxy catalogs, one
number:

| cell | GAL catalog | `f_AGN` offset |
|---|---|---|
| a3 complete rung | complete | −0.055 |
| a4 oracle probe | m<18 | −0.056 |
| a6 `gm19_acomplete` | m<19 | −0.065 |
| a6 `gm20_acomplete` | m<20 | −0.069 |

against −0.007 to −0.023 for the six cells with both catalogs flux-limited. It is
not the relative-completeness ratio (two of these cells differ by 6× in ratio and
agree to 0.004). "Complete" is represented under selection by a nominal `m_lim`
deeper than the faintest object, which should make the missing budget identically
zero and the rung estimator-independent — and it is not: per_pixel gives +0.047
there, a gap of 0.10.

*Sharpest test, one grid, ~3.5 GPU-h:* rerun the complete rung at
`log10 n0 = −24`. Selection's `C ≡ 1` and the `−24` limit both mean "no
completion contribution", so on the same code they must agree. If they don't, the
completion path is live at `C ≡ 1` and that is the mechanism.

**2. Analysis 7's residual slope.** Selection removes the occupancy bias's level
(+0.074 → −0.001) but a +0.028/dex slope survives an estimator whose completeness
is a function of `(m_lim, M*, α)` alone and cannot depend on pixel counts. So
something else in the pipeline tracks pixelisation — most likely the per-pixel
redshift KDE bandwidth, which is set from each pixel's own occupants. *Test:*
`m<18`, `nside 64`, selection, with `kde_window` pinned at the `nside = 32` value.

---

## What must change in the paper

- **Remove the relative-completeness law.** If the framing "the sign of the
  `f_AGN` bias is set by which tracer is better observed" has reached draft text,
  it needs deleting, not softening. Replace it with the null: over 1.86 dex of
  relative completeness, `f_AGN` moves −0.004/dex and `H0` −0.07/dex.
- **Remove the faintward `f_AGN` bias** and the oracle probe's "tripled bias".
- **Add the conditional-`H0` limitation.** Any forecast quoting this method's
  `H0` robustness at survey depths near `C ≈ 10 %` must carry the +3.2/dex number.
- **Keep, and lead with, analysis 5's anchor recovery and Δ ln Z = +17.7.** It is
  the strongest single argument for the estimator, and it is a statement the data
  make, not a modelling preference.
- **State the `f_AGN` scope conditionally.** Free anchors cost a factor ~5 on
  σ(`f_AGN`) (90 % half-width 0.40 vs 0.08 pinned). With analysis 4's 0.45/dex
  sensitivity, both say the same thing: `f_AGN` is only as well determined as the
  AGN-density prior.
- **Do not report analysis 7 as a cross-code result.** PARTIAL does not support it.

---

## The honest limitation on all of it

**Every number in the redo is one seed.** σ(`f_AGN`) ≈ 0.045 per cell, so no
single absolute offset at the 0.02–0.06 level is resolved. What *is* sound is
every comparison *between* estimators and *between* cells, because those are
paired on byte-identical events, injections and catalogs. The archived analysis 3
had five seeds; this redo has one.

The missing axis is therefore seeds, not rungs. The informative repeat is `m<18`
on seeds 101 and 102 — where the anchor recovery, the evidence separation and the
occupancy residual all live. Analyses 5's `m<21`/`m<20` arms are tied in evidence
and cost 20 GPU-h between them; that budget buys two seeds at the rung that
matters instead.

One provenance caveat, stated once: the archived arms ran on three different
darksirens SHAs (`de2a8df`, `2b86a2d`, `b324bed`) against this campaign's uniform
`0c5b3db`, so archive-vs-new is estimator *plus* drift. The drift is bounded at
0.09 in ln Z by two per_pixel runs on different SHAs agreeing to that precision,
and the SHA-controlled three-arm comparison in
`experiments/experiment_dsmaster_4d_recheck` reproduces analysis 5's `m<18`
contrast to 0.013 in `f_AGN` and 0.008 dex in the anchor.

---

## Per-analysis detail

`analysis_{3,4,5,6}_*/REPORT.md` and `selection_redo/a7/REPORT.md`. Numeric
companions: `ladder_summary.json`, `arms_summary.json`,
`free_anchor_summary.json`, `surface_summary.json`, `a7_verdict.json`. Figures
render deterministically to PDF + PNG via each directory's `make_figures.py`.
