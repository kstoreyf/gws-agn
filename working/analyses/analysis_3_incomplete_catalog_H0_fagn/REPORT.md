# Analysis 3 — owner report (selection-mode redo)

**What it asked.** The archived ladder measured what magnitude-limiting both
host catalogs costs `(H0, f_AGN)`, and found an `f_AGN` bias that grew faintward
to `+0.084 ± 0.019` at `m<18`. That campaign ran the legacy per-pixel
completeness estimator. This redo asks one question of it: **how much of that
was the estimator?** Same grids, same anchors at the mock's true densities, same
seed, same events and injections — `c_mode=selection` instead of `per_pixel`.

The ladder now starts at the **complete** catalog, which the archived version
also carried but which never ran in the first selection pass. That rung is the
control: at `C = 100 %` there is nothing to complete, so whatever it shows is
the estimator's own, not a completion error.

**Sources.** `results/joint_{complete,m21,m20,m19,m18}_s100.json`,
`results/ladder_summary.json`, `figs/fig1_ladder.*`. darksirens `0c5b3db`,
K = 2 field mixture, 1000 events, targeted-injection lane, seed 100.

## Result

**The sign of the `f_AGN` bias flips, and the intervals get tighter.**

| rung | C in horizon | f_AGN offset, per_pixel | f_AGN offset, selection | H0, per_pixel | H0, selection |
|---|---|---|---|---|---|
| complete | 100 % | +0.047 | **−0.055** | 69.609 | 69.218 |
| m<21 | 99.7 % | +0.047 | −0.022 | 69.606 | 69.220 |
| m<20 | 81.4 % | +0.052 | −0.022 | 69.472 | 69.218 |
| m<19 | 31.5 % | +0.066 | −0.020 | 68.969 | 68.958 |
| m<18 | 9.5 % | +0.073 | −0.009 | 68.323 | 68.664 |

Truth is `f_AGN = 0.295`, `H0 = 67.74`; seed 100's own complete-catalog `H0`
draw is 69.22.

Three things stand out.

1. **The faintward growth is gone.** Under per_pixel the offset grows
   monotonically +0.047 → +0.073 as the catalog thins. Under selection it does
   the opposite — it *shrinks* faintward, −0.022 → −0.009 — and never exceeds
   a third of the per_pixel value. Whatever drove the archived ladder's slope
   was the completeness estimator, not the completeness.
2. **`H0` is flat and sits on the seed's own draw.** Under selection the four
   bright rungs give 69.218, 69.220, 69.218 and 68.958 against a
   complete-catalog draw of 69.22 — a total spread of 0.26 across an 87 %
   change in completeness. The archived series wanders over 1.29.
3. **The complete rung is the outlier, not the faint end.** At `C = 100 %`,
   where the model has nothing to complete, selection lands `f_AGN` **−0.055**
   from truth — larger in magnitude than any incomplete rung. The two estimators
   agree with each other to 0.0004 at `m<21` under per_pixel but differ by 0.10
   at complete.

Widths, 90 % half-width, seed 100:

| rung | hw(f), per_pixel | hw(f), selection | hw(H0), per_pixel | hw(H0), selection |
|---|---|---|---|---|
| complete | 0.103 | 0.075 | 1.54 | 1.62 |
| m<18 | 0.122 (1.19×) | 0.085 (1.13×) | 2.00 (1.30×) | 1.73 (1.07×) |

Selection is ~25 % tighter on `f_AGN` at every rung and degrades more gracefully
down the ladder: 1.13× versus 1.19× on `f_AGN`, 1.07× versus 1.30× on `H0`.

## Interpretation

The archived ladder's headline — *incompleteness biases `f_AGN` and the bias
grows as the catalog thins* — does not survive the estimator change. What
survives is the weaker and more useful statement: **the completeness axis is
cheap**, costing 13 % on `σ(f_AGN)` and 7 % on `σ(H0)` for a 10× loss of
completeness, and it is cheaper under selection than it looked.

The complete-rung result is the finding that was not in the plan. Its
construction is meant to be exact: "complete" is represented inside selection
mode by a nominal `m_lim` deeper than the population's faintest object
(`max app_mag = 23.60`; the shipped fits pin 24.0 for galaxies and 25.0 for
AGN), which makes the missing budget identically zero — verified in
`experiments/experiment_dsmaster_4d_recheck/results/verify_complete_is_estimator_independent.json`.
If that representation were exact, the complete rung would be
estimator-independent and both series would print the same number. They differ
by 0.10.

Analyses 4 and 6 corroborate this rather than contradicting it. Every cell in
the campaign that carries the AGN catalog as *complete* shows the same offset,
whatever its galaxy catalog:

| cell | GAL catalog | f_AGN offset |
|---|---|---|
| a3 complete rung | complete | −0.055 |
| a4 oracle probe | m<18 | −0.056 |
| a6 `gm19_acomplete` | m<19 | −0.065 |
| a6 `gm20_acomplete` | m<20 | −0.069 |

against a6's six cells with *both* catalogs flux-limited, which sit between
−0.007 and −0.023. Four independent cells spanning four different galaxy
catalogs agree to 0.014 — too consistent to be this seed's scatter, and
indifferent to the galaxy side, which points at the deep-`m_lim` representation
of a complete AGN catalog rather than at completeness physics.

**What one seed can and cannot support.** Every number here is seed 100 alone,
where `σ(f_AGN) ≈ 0.045`. Absolute offsets from truth at the 0.02–0.06 level
are therefore *not* individually resolved. Differences *between estimators*
are a different matter: both series consume byte-identical events, injections
and catalogs, so the per-rung difference is paired and carries none of that
scatter. Read the table as a statement about estimators, not as a closure test.

One caveat on provenance, stated once: the archived grids ran on darksirens
`de2a8df` and these on `0c5b3db`, so the comparison is estimator *plus* whatever
moved in between. Analysis 5's evidence gives a bound on that drift — the
archived `m<18` per_pixel `lnZ` and the SHA-controlled re-check's per_pixel
`lnZ` agree to 0.09 across two different SHAs — but the clean one-variable
comparison is the three-arm re-check in
`experiments/experiment_dsmaster_4d_recheck`, not this ladder.

## Recommendation

**Report the flip, not a bias.** The load-bearing statements are (i) the
`f_AGN` bias the archived campaign measured is an artifact of the per-pixel
completeness estimator, (ii) under selection the ladder is flat to `m<18` in
both parameters, and (iii) `H0` is unmoved by completeness at fixed anchors.

**Do not yet write the complete-rung number into the paper.** It is a
four-cell pattern with a plausible mechanical cause and no test yet. Two cheap
tests, in this order:

1. **The zero-density probe, and it is the sharper one.** Analyses 0, 1 and 2 run
   at `log10 n0 = −24`, which switches the completion term off by taking the
   missing-host density to zero rather than by making `C = 1`. Under selection
   the complete rung claims `C ≡ 1`, so the two configurations should contribute
   *identically nothing* and must return the same `f_AGN` on the same code and
   data. Rerun `joint_complete_s100` at `log10 n0 = log10 n0_c2 = −24` on
   `0c5b3db`. If it differs from this rung's 0.240, something in the completion
   path is live even at `C ≡ 1`, and that is the mechanism. ~3.5 GPU-h.
   (For scale, analysis 2's `−24` complete-catalog fit on this seed gave 0.273,
   but it ran in a different era and config, so it bounds nothing on its own.)
2. **The `m_lim` depth probe.** Rerun with the nominal `m_lim` moved
   (24/25 → 26 for both tracers). If `f_AGN` moves, the deep-`m_lim`
   representation of "complete" is not exact and every AGN=complete cell in
   analyses 4 and 6 inherits it.

Note what this does *not* touch: analyses 0, 1 and 2 never invoke a completeness
estimator, because at `log10 n0 = −24` there is no completion budget to
estimate. The paper's core results are outside the blast radius of the whole
selection redo.

**Seeds are the missing axis.** The archived ladder had five; this one has one.
Any statement of the form "selection recovers `f_AGN` to within X" needs the
seed-101/102 replication that is currently held.
