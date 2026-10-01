# a7 — owner report: does the bias scale with pixel occupancy?

**What it asked.** Analyses 3, 4 and 6 all end in the same place: the archived
`f_AGN` bias tracks something about how *empty* the pixels are, and the
per-pixel completeness estimator is the thing that cares. a7 tests that directly
by changing nothing but the pixelisation. The prediction was registered in
advance, from the `desi_darksirens_selection` reading that their mechanism
depends on counts per pixel and nothing else:

> if it is the same disease, the per_pixel offset must move along this axis.
> Flat under per_pixel ⇒ our bias is not their disease and the cross-code story
> is weaker. **Both outcomes reportable.**

Axis variable is `λ = N_AGN / npix` at `m<18` — the mean per pixel *including*
empties. Occupancy per *occupied* pixel is the wrong variable; it saturates near
1 as pixels empty out. Six cells: `nside` 16 / 32 / 64 × two estimators, with
the `nside = 32` midpoints taken from the two campaigns' own `m<18` grids rather
than rerun.

**Sources.** `results/joint_m18_ns{16,64}_{per_pixel,selection}_s100.json`,
`results/policyprobe_ns64_{zero,volume}_s100.json`, `results/a7_verdict.json`,
`figs/fig_a7_occupancy.*`. darksirens `0c5b3db`, seed 100.

## Result

**Verdict: PARTIAL.**

| estimator | λ = 2.69 (ns16) | λ = 0.67 (ns32) | λ = 0.17 (ns64) | slope per dex | intercept |
|---|---|---|---|---|---|
| per_pixel | +0.092 | +0.073 | +0.029 | **+0.052 ± 0.012** | +0.074 |
| selection | +0.013 | −0.009 | −0.021 | **+0.028 ± 0.004** | −0.001 |

(`f_AGN` offsets from truth 0.295, over a 1.20-dex span in λ.)

Two halves to the verdict.

1. **The prediction is confirmed for per_pixel.** The offset moves +0.052 per
   dex of occupancy, 4.3σ from flat on its own fit error, and the intercept is
   +0.074 — the archived `m<18` bias, recovered as the value this axis passes
   through at the campaign's own pixelisation.
2. **The selection control is not flat.** It should have been. Selection's
   completeness does not depend on pixel counts at all — it is a function of
   `(m_lim, M*, α)` — so a change of pixelisation should leave `f_AGN` alone.
   Instead it moves +0.028 per dex, 6.6σ from flat, a little over half the
   per_pixel slope. Its intercept, though, is −0.0009: essentially zero, against
   per_pixel's +0.074.

**The confound was closed first.** Changing `nside` also changes how many pixels
are empty, which is a different mechanism. The empty-pixel policy probe
(`zero` versus `volume`) returns `f_AGN` medians identical to machine precision
(|Δ| = 0.0 exactly), so under field sky-weighting the policy is inert and the
axis is occupancy alone.

## Interpretation

The estimator change moves the **intercept**, not the whole slope. That splits
the archived bias into two parts:

- a **level** of +0.074 that is the per-pixel estimator's own, and that selection
  removes entirely;
- a **residual slope** of +0.028 per dex that both estimators share, and that
  therefore cannot be attributed to completeness estimation at all.

The honest reading of the residual is that something else in the pipeline
depends on pixelisation. Candidates, in the order they should be checked: the
sky-weighted mixture normalisation at fixed `field` weighting; the per-pixel
redshift KDE bandwidth, which is set from the pixel's own occupants and so
narrows as pixels shrink; and the selection `N_eff` guard, which is evaluated
per pixel. None of these is the completeness estimator, and the first two are
cheap to probe by holding the KDE window fixed across `nside`.

The cross-code claim is weaker than the strong version but not empty. Against
`desi_darksirens_selection`: their mechanism, applied here, predicts the
per_pixel slope, and we measure it at +0.052 ± 0.012 with the right sign and an
intercept that reproduces the archived bias. What we cannot say is that their
mechanism accounts for *all* of our occupancy dependence, because a third of it
survives an estimator that is architecturally immune to their mechanism.

**One seed.** Three points per estimator, one realisation. The slopes' quoted
errors are fit errors on those three points, not seed errors, and the whole axis
is `m<18`. A slope this shallow measured at one seed should be described as a
trend with a sign, not as a coefficient.

## Recommendation

**Do not put a7 in the paper as a cross-code result.** PARTIAL is not the
verdict that supports the strong claim, and the interesting half — that a third
of the occupancy dependence survives an estimator that should be immune to it —
is an open diagnostic about our own pipeline, not a finding about theirs.

**Do act on the residual.** It is the only thing in the whole selection redo
that neither estimator explains, and it sits on the same axis as every other
result in the campaign. The probe is one grid: `m<18`, `nside 64`, selection,
with `kde_window` held at the `nside = 32` value. If the residual slope collapses,
it is the bandwidth and the campaign's `nside = 32` choice needs a stated
justification; if it does not, look at the guard next.

**What is safe to quote now**, if anything from a7 is quoted at all: the
empty-pixel policy is inert under field weighting (exact), and the per-pixel
estimator's `f_AGN` bias scales with pixel occupancy while selection's does not
carry its level.
