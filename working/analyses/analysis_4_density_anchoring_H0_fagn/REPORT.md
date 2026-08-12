# Analysis 4 — owner report (selection-mode redo)

**What it asked.** Analyses 2 and 3 hand the fit the mock's true completion
densities. Analysis 4 takes that away one axis at a time: mis-anchor the **AGN**
completion density by a factor 0.5–2.0 with the galaxy anchor held at truth, at
every rung of the ladder, and measure where the error lands. The archived
campaign found it lands almost entirely on `f_AGN`, with a sensitivity that
steepened sharply as the catalog thinned. This redo asks whether that
sensitivity is a property of the *inference* or of the *estimator*.

25 arms: 4 rungs × 6 mis-anchoring factors, plus the oracle probe
(GAL `m<18` × AGN complete, both densities at truth). The factor-1.0 arm is
analysis 3's own grid, referenced rather than rerun.

**Sources.** `results/joint_{rung}_{arm}_s100.json`, `results/arms_summary.json`,
`figs/fig1_anchor_response.*`. darksirens `0c5b3db`, K = 2 field mixture,
1000 events, targeted-injection lane, seed 100.

## Result

**The sensitivity survives; the runaway does not.**

`slope` is `d f_AGN / d log10(factor)` — how hard a density error pushes
`f_AGN`. `offset@1` is where the correctly-anchored arm sits relative to truth.

| rung | slope, per_pixel | slope, selection | offset@1, per_pixel | offset@1, selection |
|---|---|---|---|---|
| m<21 | +0.526 | +0.434 | +0.049 | −0.014 |
| m<20 | +0.542 | +0.438 | +0.055 | −0.014 |
| m<19 | +0.639 | +0.470 | +0.071 | −0.014 |
| m<18 | **+0.943** | **+0.455** | +0.095 | −0.014 |

Two clean separations.

1. **The steepening is the estimator.** Under per_pixel the slope nearly
   doubles down the ladder, 0.53 → 0.94. Under selection it is flat to 8 %
   across an 87 % change in completeness: 0.434, 0.438, 0.470, 0.455. A factor-2
   error in the assumed AGN density costs ≈ 0.13 in `f_AGN` at every depth, not
   0.28 at the faint end.
2. **The offset is flat, and it is not zero.** Every correctly-anchored arm
   sits at −0.0135 to −0.0140 — the same number to three decimals at all four
   rungs, against per_pixel's +0.049 → +0.095. Whatever residual `f_AGN` carries
   under selection is depth-independent, which is exactly what a completion
   error is *not*.

**New: `H0`'s immunity is conditional.** The archived campaign reported the
anchoring error lands on `f_AGN` and not on `H0`. Under selection that holds at
the bright end and fails at the faint end:

| rung | d H0 / d log10(factor) | R² | H0 spread over factor 0.5–2.0 |
|---|---|---|---|
| m<21 | −0.017 | 0.99 | 0.010 |
| m<20 | −0.035 | 0.91 | 0.022 |
| m<19 | +0.709 | 1.00 | 0.42 |
| m<18 | **+3.211** | 0.99 | **1.91** |

At `C = 99.7 %` and `81 %`, `H0` does not move at all — a spread of 0.01–0.02
km s⁻¹ Mpc⁻¹ is numerical noise on a 1.6-wide posterior. At `C = 9.5 %` the same
factor-2 density error moves `H0` by ~1 km s⁻¹ Mpc⁻¹, over half its own 90 %
half-width, and the response is a clean straight line in `log10(factor)`
(R² = 0.99), not scatter.

**The oracle probe swings by 0.25.** GAL `m<18` × AGN complete, both densities
at truth — the arm built to ask whether a sparse AGN completion was driving
analysis 3's archived bias:

| | f_AGN | offset | 90 % CI | H0 |
|---|---|---|---|---|
| per_pixel (archived) | 0.492 | **+0.197** | [0.386, 0.598] | 69.507 |
| selection (this work) | 0.239 | **−0.056** | [0.169, 0.314] | 68.991 |

The archived reading was that handing the fit a perfect AGN catalog *tripled*
the bias — a relative-completeness effect. Under selection the same arm lands
0.25 lower, and it lands on the same −0.056 that analysis 3's complete rung
shows. See below: that is now a four-cell pattern, and it is about the AGN
catalog being carried as *complete*, not about the oracle.

## Interpretation

The mechanism the archived campaign proposed — a density error is absorbed by
the mixture fraction because both are budget-like quantities — is confirmed and
survives the estimator change with its magnitude roughly intact (0.43–0.47 per
dex versus 0.53). What does *not* survive is the claim that it gets worse as the
catalog thins. The steepening was the per-pixel estimator failing harder where
pixels are emptier, which is exactly what analysis 7 measures directly along the
occupancy axis and finds: per_pixel's offset scales at +0.052 per dex of pixel
occupancy, selection's at +0.028.

The depth-independent −0.0135 is the more interesting residual. A completion
error must scale with how much is being completed; this does not scale at all.
Combined with analysis 3's complete rung and analysis 6's flat surface, the
picture is that selection leaves a small, constant `f_AGN` offset that has
nothing to do with completeness — and that at one seed, with `σ(f_AGN) ≈ 0.045`,
−0.0135 is a third of a standard deviation and is *not* established as nonzero.
It is established as *flat*, which is a statement the four rungs can support
because they are paired on identical data.

`H0`'s conditional immunity is the finding to carry forward. It says the
robustness the campaign has been quoting is a bright-catalog property. The
physical reading: at `C = 9.5 %` most of the AGN likelihood is the model-supplied
smooth field rather than catalog hosts, so the assumed density stops being a
normalisation and starts acting as the redshift prior itself — and a redshift
prior maps directly onto `H0`. The transition is between `C = 81 %` and
`C = 32 %`, the same place analysis 3's archived ladder first moved.

## Recommendation

**Main text.** The flat slope (0.43–0.47 per dex, depth-independent) is the
quotable sensitivity: it is what a reader needs to convert an AGN-density prior
width into an `f_AGN` systematic, and it no longer needs a per-rung caveat.

**Also main text, as a limitation:** `H0` is immune to AGN-density
mis-anchoring only while the catalog is bright. At `C ≈ 10 %` a factor-2 error
costs ~1 km s⁻¹ Mpc⁻¹. Any forecast that quotes this method's `H0` robustness
at survey depths near the faint rung must carry that number.

**Hold the oracle probe.** Its −0.056 is indistinguishable from analysis 3's
complete rung (−0.055) and analysis 6's two AGN=complete cells (−0.065, −0.069)
— four cells spanning four different galaxy catalogs, all agreeing to 0.014.
Either the deep-`m_lim` representation of "complete" carries a constant
`f_AGN` offset, or all four inherit one common thing that is not the oracle
hypothesis. Until that is settled the oracle arm cannot be read as a statement
about relative completeness in either direction.

**Next probe (cheap, one grid).** Re-run `joint_m18_oracle_s100` with the
nominal `m_lim` moved 25 → 26. If `f_AGN` moves, the representation is not
exact and every AGN=complete cell in analyses 3, 4 and 6 inherits it; if it
does not, the −0.06 is real and needs seeds. ~20 min on the H100, from this
arm's own timing.
