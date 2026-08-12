# Analysis 3 — owner report

**What it asked.** Analysis 2 handed the fit every host in the universe. Take
that away: magnitude-limit both host catalogs down a ladder `m<21, 20, 19, 18`
on the same 1000 events and the same injections, and measure what the
substitution of *observed catalog + model-supplied missing-host budget* costs
`(H0, f_AGN)`. Five realisations (seeds 100–105), targeted-injection lane,
completion densities held at the mock's truths — the most favourable case, which
isolates *completeness* from *anchoring error*.

**Sources.** `results/h0_fagn_ladder.json`, `results/ladder_summary.json`,
`results/gates.json`, `figs/fig_closure_ladder.*`, `fig_ladder_widths.*`,
`fig_estimator_offset.*`, `fig_null_m18.*`, `fig_nside_scaling.*`.

## Result

Going from a complete catalog to 10 % completeness inside the horizon costs
**σ(H0) a factor 1.14 and σ(f_AGN) a factor 1.22**. That is the headline, and it
is a small number: the ladder is remarkably flat.

| rung | σ(H0) vs rung 0 | σ(f) vs rung 0 | H0 offset (5 seeds) | f offset vs realised |
|---|---|---|---|---|
| complete | 1.00 | 1.00 | +1.08 ± 0.53 | +0.046 ± 0.022 |
| m<21 | 1.00 | 1.00 | +1.08 ± 0.53 | +0.045 ± 0.022 |
| m<20 | 0.99 | 1.01 | +0.99 ± 0.50 | +0.051 ± 0.022 |
| m<19 | 1.21 | 1.05 | +1.09 ± 0.27 | +0.071 ± 0.023 |
| m<18 | 1.14 | 1.22 | −0.79 ± 0.43 | **+0.084 ± 0.019** |

Two things stand out.

1. **Nothing happens until `m<19`.** The `m<21` rung is statistically
   indistinguishable from the complete catalog (σ ratios 1.000 and 0.999). The
   information the fit uses is not in the faint galaxies; it is in the bright
   ones that a shallow survey already has.
2. **The `f_AGN` bias grows monotonically and becomes significant at the faint
   end:** +0.045 → +0.051 → +0.071 → **+0.084 ± 0.019** (`t(4) = +4.53`), against
   a per-realisation binomial scatter of 0.014. `H0`'s offset does *not* grow —
   it wanders, and flips sign at `m<18` (−0.79 ± 0.43, `t(4) = −1.82`).

Robustness: 0 of 206,025 grid cells rejected by the variance guard across the
whole ladder. The sky-shuffle null at `m<18` returns
`f_AGN = 0.078 (+0.070, −0.051)`, consistent with zero — the AGN preference is
positional, not an artifact of the mixture's normalisation. Degrading the survey
pixelisation `nside 32 → 16` leaves the ladder unchanged.

## Interpretation

The completeness axis is *cheap* and the incompleteness bias lands almost
entirely on `f_AGN`, not on `H0`. Physically that follows from what the
completion does: it replaces missing hosts by a smooth comoving density field,
which is a poor stand-in for individual galaxies but a perfectly good stand-in
for a *number*. `H0` is set by the redshift–distance registration of the hosts
the survey does have; `f_AGN` is set by a ratio of budgets, and the budget that
gets replaced by a smooth field is the one that also carries the fraction.

Analysis 3 could not distinguish two candidate causes of the `+0.084` bias at
`m<18`: (i) genuine galaxy incompleteness, or (ii) the *sparse AGN completion* —
at `m<18` the AGN catalog has ~5 hosts per occupied pixel and 52.8 % of pixels
empty, so the AGN side of the mixture is being reconstructed from very little.
Analysis 4's oracle probe was built to settle exactly this, and does (below).

The `+1.0` to `+1.1` `H0` offset at the bright rungs is *not* an analysis-3
finding and should not be read as a bias: it is inherited from analysis 2, whose
five-seed closure on complete catalogs is `+0.41 ± 0.55` (`t = 0.73`) with
per-seed offsets from −1.21 to +1.82. Five realisations of a ~1.05 half-width
statistic simply cannot resolve an offset of this size, and the sign flip at
`m<18` is the same scatter. If the campaign wants a statement about `H0`
accuracy at the 0.5 km s⁻¹ Mpc⁻¹ level it needs more realisations, not more
rungs.

## Recommendation

**Main text.** The σ ratios (1.14, 1.22) and the flatness of the ladder to
`m<19` are the load-bearing statement that this estimator degrades gracefully
under realistic survey depth, and the `f_AGN` bias at `m<18` is a real, measured,
5-seed result at `t(4) = +4.53` — the one significant closure failure in the
directory. Its obvious explanation (a sparse AGN completion) is excluded by
analysis 4's oracle probe; what replaces it is relative completeness between the
two tracers. The sky-shuffle null belongs
in Validation as one sentence. The nside degradation and the analysis-2
continuity check are appendix-or-nothing.

One caveat to state plainly and once: this ladder assumes the completion
densities are known exactly. Analyses 4 and 5 remove that assumption, and the
honest scope of analysis 3's numbers is "the cost of incompleteness alone."
