# Analysis 4 — owner report

**What it asked.** Analysis 3 pinned the completion's densities at the mock's
truths. Real data cannot. Set the AGN anchor deliberately wrong by a known
factor — `log10n0_c2 = -5 + log10(factor)`, factor ∈ {0.5, 0.7, 0.9, 1.1, 1.3, 2}
— at every rung of the ladder, and measure where the error lands. Plus one oracle
probe: galaxies at `m<18` but the **AGN survey complete**, both densities at
truth, to test whether analysis 3's `+0.084` `f_AGN` bias at the faintest rung is
manufactured by the sparse AGN completion rather than by galaxy incompleteness.

**Scope.** Seed 100 only, targeted lane, rungs `m21..m18`. 25 grids, all present.
The exact (factor 1.0) arm is analysis 3's own seed-100 grid, referenced not
rerun, so arms and reference share one estimator by construction.

**Sources.** `results/arms_summary.json`, `figs/fig_anchor_response.*`,
`fig_anchor_budget.*`, `fig_anchor_significance.*`, `fig_anchor_posteriors.*`,
`fig_oracle_m18.*`, README `ARM_TABLES`.

## Result 1 — the anchoring error goes almost entirely into `f_AGN`

At `m<21`, over the factor-4 range in assumed AGN density:

| factor | `f_AGN` | Δ vs exact arm | `H0` offset |
|---|---|---|---|
| 0.5 | 0.191 ± 0.040 | **−2.41σ** | +1.63 |
| 1.0 (exact) | 0.342 ± 0.063 | — | +1.87 |
| 2.0 | 0.502 ± 0.089 | **+2.56σ** | +2.36 |

`f_AGN` tracks the assumed density almost linearly — halving it halves the
recovered fraction — while `H0` moves by −0.26 to +0.53 of its own half-width
over the same range. The effect *strengthens* down the ladder: at `m<18` the same
factor-4 range spans `f_AGN` = 0.141 → 0.699, i.e. **−3.06σ to +4.45σ**.

## Result 2 — the *detection* survives

The significance of a non-zero AGN component (median / 68 % half-width) runs
4.8σ → 5.5σ → 5.6σ across the same factor-4 range at `m<21`, and 3.8σ → 5.0σ →
7.0σ at `m<18`. Because the error moves the median and the width *together*, you
cannot mis-anchor your way out of detecting an AGN component — but the value you
quote for the fraction is only as good as your density.

## Result 3 — the oracle probe settles analysis 3's faint-rung bias

Galaxies at `m<18`, AGN survey complete, densities at truth:
**`f_AGN` = 0.492 ± 0.065, offset +0.197** (against `+0.073` for the ordinary
`m<18` arm). Handing the model every AGN host did not remove the bias — it made
it 2.7× worse.

So the answer to analysis 3's open question is **no**: the `+0.084` bias at
`m<18` is not manufactured by the sparse AGN completion. Something about the
*asymmetry* between a complete AGN catalog and a 10 %-complete galaxy catalog
inflates the fraction, and the more complete the AGN side is relative to the
galaxy side, the worse it gets. `H0` in the oracle probe sits at +1.77, back with
the bright rungs rather than at `m<18`'s −0.01.

## Interpretation

The completion's density anchor is the dominant systematic on `f_AGN` and a
near-irrelevance for `H0` — the same split analysis 3 found for completeness
itself, and now with a lever we control. The mechanism is direct: `f_AGN`
multiplies the AGN tracer prior, whose normalisation *is* the assumed AGN density
in the unobserved part of the volume, so an error in `n_0^{AGN}` and an error in
`f_AGN` are nearly the same modelling error. Analysis 5 confirms this from the
other side — the posterior correlation between the two reaches +0.89.

The oracle result reframes the faint-rung bias. It is not a sparsity artifact; it
is a **relative-completeness** effect. The mixture weight the fit recovers is not
the fraction of events in AGN, it is closer to the fraction of *recoverable*
events in AGN, and the two tracers' completeness enter that ratio. A real
analysis would face exactly this asymmetry, since AGN catalogs and galaxy
catalogs are never complete to the same depth.

## Recommendation

**Main text for Results 1 and 3; appendix for Result 2.** The linear response of
`f_AGN` to the assumed density (with the ±factor-2 bracket quoted as a systematic)
is the single most useful number in this directory for a reader planning a real
measurement. The oracle probe deserves its own short paragraph because it kills
the obvious explanation of the faint-rung bias and replaces it with a better one.
The significance-survives result is reassuring but not surprising and reads as one
appendix sentence.

**Caveat to state:** one seed. The arm-to-arm *differences* are noiseless (shared
events, shared estimator), so the response slope is trustworthy; the absolute
offsets carry seed-100's own +0.047 realisation offset, which should be quoted as
such rather than folded in.

**Follow-up this analysis creates.** The relative-completeness effect is now the
best-motivated open mechanism in the campaign, and it is a two-parameter surface
(GAL depth × AGN depth) that no existing directory scans. See `../PROPOSAL_analysis_6.md`.
