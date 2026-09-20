# Analysis 9 — owner report (marked GAL/AGN multi-tracer with `H0` free, seed 100)

**Sources.** `results/{s9_spatial,j9_marked,section_11_comparison}.json` and the paired
`.h5` cubes; `diagnostics/a9_guard.json`, `diagnostics/a9_guard_sensitivity.json`,
`diagnostics/a9_gate_a_full_slab.json`, `diagnostics/a9_h0_matched_lattice.json`;
figures `figs/fig_{s9_spatial,j9_corner,section_11}.{pdf,png}`.
darksirens `af896ca` on the pinned base `2b86a2d`, worktree clean; gws-agn `335d9a3`,
dirty only outside this analysis. Seed 100, complete catalogs, the marked mock
`events_marked_dmu0p10.h5` and the existing targeted injection set reused unchanged.
Every number below is read from those files; nothing was recomputed for this report.

## What changed from Analysis 8

One coordinate was released: `H0`. Nothing else.

Analysis 8 measured `(f_AGN, Δμ_χ)` from this marked realisation with `H0` pinned at the
planted 67.74. That pin was the strongest assumption in the result, because the spatial
channel identifies the host label through the redshift-space structure of two tracers and
that structure moves with `H0`. Here the same likelihood, the same dataset, the same
injections and the same nine-label parameter space are used with `H0` sampled. Three
coordinates are free — `H0`, `fcat_2` (`f_AGN`) and `mu_chi_c2`, the AGN branch's
**absolute** effective-spin mean, which equals `Δμ_χ` only because `μ_χ,GAL` is pinned at
the powerlaw+peak fiducial 0. Everything else stays where Analysis 8 left it:
`Ω_m = 0.3075`, twelve base population parameters pinned by name, a common mass function,
mass ratio, spin width and redshift rate in both branches, and the six survey nuisances at
their Analysis-8 values. The `f` and `Δμ_χ` axes are Analysis 8's own arrays (41 × 61
nodes), so the `H0 = 67.74` slab of the new cube is cell for cell Analysis 8's grid.

That slab is how one checks that nothing else moved, and it was checked on the whole
2,501-cell slab, not on a subset. Of the 2,478 cells accepted in both runs, **2,165
reproduce Analysis 8's total log-likelihood bitwise**; 277 differ by 2 ULP, 35 by 4 ULP and
exactly one by 6 ULP (1 ULP = 9.094947e-13 at `lnL ≈ −4.2e3`), a largest absolute
difference of 5.46e-12 and a relative difference of 1.25e-15. The 6 ULP is not a new
disagreement: the PE and the selection terms separately cap at 2 ULP each, and `lnL` is
exactly their sum on both sides (maximum residual 0.0), so the worst total is two 2-ULP
parts plus the rounding of their sum, surfacing now only because the maximum runs over 62
times more cells than the 40-cell comparison that set the expectation. Analysis 8 ran on an
H100 and this ran on an A100, and that cross-hardware difference is the whole of the
residual. Everything else is identical: `log μ` agrees to 1 ULP, `N_eff` to 63 ULP on
values of order 7.7e5 (7e-15 relative), the guard-rejected mask is the same 23 cells at
`f ≥ 0.750` and `Δμ_χ ≥ +0.2350`, `lnL_max = −4211.84259284502` at the same MAP cell
`(0.275, +0.1075)` with a difference of exactly 0.0, and the recomputed marginals agree to
≤ 5.5e-15 with `ρ(f, Δμ_χ) = −0.5543646830622379` against Analysis 8's −0.5543646830621719.

Two arms were measured on this realisation. **S9** is spatial-only, `Δμ_χ ≡ 0` exactly:
`H0` on 202 nodes over [50, 100] plus the node 67.74, times 41 `f` nodes, 8,282 cells.
**J9** is the marked joint measurement: 28 `H0` nodes over [63, 76] plus 67.74, times the
41 × 61 `(f, Δμ_χ)` plane, 70,028 cells. Every J9 `H0` node is bitwise an S9 node, so the
two arms can be compared without resampling either.

## Did the 3-D model recover the planted values

The joint maximum sits at `(H0, f_AGN, Δμ_χ) = (69.00, 0.275, +0.1075)` with
`lnL_max = −4210.586470980414`.

| coordinate | planted | realised | median | 68% | 90% |
|---|---|---|---|---|---|
| `H0` | 67.74 | — | 69.0905 | [68.1740, 69.9967] | [67.5978, 70.6777] |
| `f_AGN` | 0.300 | 0.295 | 0.2639 | [0.2183, 0.3113] | [0.1889, 0.3425] |
| `Δμ_χ` | +0.100 | +0.111924 ± 0.006815 | 0.10828 | [0.08937, 0.12878] | [0.07815, 0.14350] |

`f_AGN` and `Δμ_χ` are recovered against **both** truths at 68%. For `f_AGN` the planted
0.300 and the realised 0.295 sit inside the 68% interval, offsets −0.0361 and −0.0311. For
`Δμ_χ` the planted +0.100 and the realised +0.111924 both sit inside the 68% interval,
offsets +0.00828 and −0.00365 — the measurement falls between the two truths. The MAP cell
in `(f, Δμ_χ)` is unchanged from Analysis 8's fixed-`H0` measurement.

`H0` is the exception and it is not a failure of the model: the planted 67.74 lies outside
the 68% interval and inside the 90% one, in both arms. The next section shows why that is
this realisation's own draw.

## What happened to `H0`

| arm | median | 68% | width | 90% | width | MAP |
|---|---|---|---|---|---|---|
| S9 spatial-only | 69.0872 | [68.0865, 70.0881] | 2.0016 | [67.4375, 70.7393] | 3.3018 | 69.00 |
| J9 marked | 69.0905 | [68.1740, 69.9967] | 1.8227 | [67.5978, 70.6777] | 3.0799 | 69.00 |

**`H0` is quoted differentially, and that is not optional here.** The offset from the
planted 67.74 is +1.3472 (S9) and +1.3505 (J9), and that offset is a property of seed 100,
not of this model. Analysis 2 measured the *same* realisation, the same catalogs and the
same selection with **unmarked** events and one shared population, and recovered
`H0 = 69.2170`, 68% [68.2491, 70.1915]. J9 sits **0.1265 below** that, about a seventh of
its own 68% half-width, and S9 sits 0.1298 below it. In other words the marked two-tracer
measurement reproduces the unmarked one on this realisation; what it does not do is move
that realisation's draw back onto the planted value. Reading +1.35 as a bias would be
reading the draw, not the estimator.

Against the same unmarked reference the marked measurement is also the tighter one: J9's
`H0` interval is 0.9384 times Analysis 2's at 68% and 0.9515 times at 90%.

The comparison that isolates the mark is S9 against J9, since those two differ only by
`Δμ_χ` being free. On the matched lattice the marked interval is **0.8879** of the
spatial-only one at 68% and **0.8996** at 90% — a narrowing of 11.2% and 10.0% — with the
median moved by +0.00041 and the MAP cell unchanged at 69.00.

The matched lattice matters. S9 integrates `H0` on 202 nodes over [50, 100] and J9 on 28
nodes over [63, 76], so the naive full-axis ratios, 0.9106 at 68% and 0.9328 at 90%,
compare two different quadratures; re-marginalising S9's own cube on exactly the 28 J9
nodes widens S9 by 1.0256 and 1.0368, so the full-axis numbers *under-state* the
sharpening and 0.888 / 0.900 are the numbers to quote.

## What happened to `f_AGN`

| arm | median | 68% | width | 90% | width |
|---|---|---|---|---|---|
| S9 spatial-only | 0.26947 | [0.22195, 0.31872] | 0.096775 | [0.19026, 0.34946] | 0.159199 |
| J9 marked | 0.26393 | [0.21828, 0.31133] | 0.093051 | [0.18892, 0.34255] | 0.153631 |

The mark sharpens the host fraction by 3.8% at 68% and 3.5% at 90% (ratios 0.9615 and
0.9650) and moves the median by −0.0055. Both arms use the identical 41-node `f` axis, so
these ratios carry no quadrature caveat at all. The MAP cell is `f = 0.275` in both arms;
J9's one-dimensional marginal mode sits one node lower, at 0.250.

Freeing `H0` costs the host fraction about one percent of its width: against Analysis 8's
fixed-`H0` measurement the J9 interval is 1.0101 times as wide at 68% and 1.0078 at 90%,
with the median moved by +0.0023. Against the unmarked Analysis 2 measurement (width
0.096259 at 68%) the J9 median is 0.0094 lower.

## What happened to `Δμ_χ` when `H0` became free

Almost nothing, and that is the measurement.

| | median | 68% | width | 90% width |
|---|---|---|---|---|
| Analysis 8, `H0` fixed | 0.107386 | [0.088587, 0.127960] | 0.0393721 | 0.0650741 |
| J9, `H0` free | 0.108275 | [0.089372, 0.128782] | 0.0394102 | 0.0653427 |

The width ratios are **1.00097 at 68% and 1.00413 at 90%** — 0.10% and 0.41% — the median
moves by +0.00089 and the MAP node is unchanged at +0.1075. Both measurements use Analysis
8's own 61-node `Δμ_χ` axis, so again there is no quadrature caveat. Marginalising over a
Hubble constant that ranges over 13 units costs the spin offset less than half a percent of
its interval.

## The posterior degeneracies

| pair | J9 | reference |
|---|---|---|
| `ρ(H0, f_AGN)` | +0.01482 | S9 +0.05904; Analysis 2, unmarked, +0.06780 |
| `ρ(H0, Δμ_χ)` | +0.02466 | — |
| `ρ(f_AGN, Δμ_χ)` | −0.55936 | Analysis 8 at fixed `H0`, −0.55436 |

The posterior standard deviations are 0.9013 in `H0`, 0.04392 in `f_AGN` and 0.019518 in
`Δμ_χ`.

Two things follow. First, the only strong degeneracy in the problem is the one Analysis 8
already measured, between the host fraction and the spin offset, and freeing `H0` barely
touches it: −0.559 against −0.554. That anticorrelation is also why the `f_AGN` interval
must widen slightly when `Δμ_χ` is marginalised over rather than fixed. Second, `H0` is
very nearly orthogonal to both of the other coordinates, +0.015 and +0.025. The cosmology
and the intrinsic mark are not trading against each other in this posterior.

## Does the mark feed back into cosmology, or is the flow one-way

The flow is one-way: **mark → cosmology**, and almost nothing returns. Adding the spin mark
narrows `H0` by 11.2% at 68% and 10.0% at 90% with the median unmoved (+0.0004) and the MAP
cell unchanged; freeing `H0` costs `Δμ_χ` 0.10% at 68% and 0.41% at 90%. The return current
is two orders of magnitude smaller than the forward one at 68% (0.097% against 11.2%) and a
factor 24 smaller at 90%.

The mechanism is not the obvious one, and it is worth stating carefully, because the
natural reading — a degeneracy between the mark and the cosmology being broken — is
excluded by the correlations. `ρ(H0, Δμ_χ) = +0.025` and `ρ(H0, f_AGN) = +0.015`: the mark
is nearly orthogonal to `H0`, so there is no `H0`–`Δμ_χ` ridge for it to cut. What the mark
does is tighten `f_AGN`, and `f_AGN` sets the weighting between the two tracers whose
redshift-space structure carries `H0`. The cosmological information arrives through the
tracer weighting, not through the spin channel directly.

The supporting evidence is that `ρ(H0, f_AGN)` falls monotonically along the ladder as more
information about the host fraction enters: **+0.0678** with unmarked events and one shared
population (Analysis 2), **+0.0590** with two tracers and no mark (S9), **+0.0148** with the
mark (J9). The better `f_AGN` is known, the less `H0` has to move with it.

An 11% width change measured on one realisation is a scoping measurement, not a calibrated
gain. It says the effect exists and gives its size on this draw; it does not establish that
the gain holds in expectation, and nothing here measures its scatter.

## Selection diagnostics

The guard is the existing hard `N_eff` floor on the selection integral; a cell below it
returns `−inf`.

| arm | rejected | min `N_eff` | min `N_eff`/threshold | median `N_eff` |
|---|---|---|---|---|
| S9 | 0 / 8,282 | 231,774 | 46.35 | 652,661 |
| J9 | 809 / 70,028 (1.155%) | 1,678 | 0.336 | 378,352 |

S9 rejects nothing, with 46× margin at its worst cell. In J9 every rejected cell lies in
one corner, at `f ≥ 0.575` **and** `Δμ_χ ≥ +0.220`, and that corner exists at every `H0` in
the window. It is not a fixed set: it is smallest on and beside the anchor, where it is
exactly the 23 cells Analysis 8 found at fixed `H0` (`H0` = 67.5, 67.74, 68.0, 68.5, 69.0,
69.5), and it grows away from there, to 57 cells at `H0 = 63.0` and back up to 31 at 71.5,
thinning to 12 at 76.0. Moving `H0` moves the selection integral, which is why the corner
had to be bounded over the whole cube rather than inherited from the anchor slab.

**No result stands behind a rejected cell.** The rejected region begins 9 `f` nodes and 10
`Δμ_χ` nodes beyond the 90% credible edges (0.3425 and 0.1435); the accepted posterior mass
at `f ≥ 0.575` is 3.36e-11 and at `Δμ_χ ≥ +0.220` is 1.02e-05. Filling every rejected cell
with the largest accepted density found beside the region — 1.59e-51 of the peak, 116.97 in
log-likelihood below it, verified to be an over-estimate on 809 of 809 cells — bounds the
rejected mass at 8.42e-51.

That fill bound assumes the log-likelihood falls into the corner, and over the whole cube it
does not everywhere: **25 of 348 boundary columns rise instead of falling**, all of them at
`H0 ≤ 65.0` (6, 6, 6, 3 and 4 columns at 63.0, 63.5, 64.0, 64.5 and 65.0), where Analysis 8
at fixed `H0` had 11 of 11 columns falling. The rises are small — at most 5.28 nats — and
they sit inside the estimator's own noise there: on the 376 accepted boundary cells
`sigma2_total` runs 200 (median) to 281 (maximum) against 8.21 at the posterior peak, a
1σ log-likelihood uncertainty of about 14.1 nats, with `N_eff` at 5,007 minimum and 5,549
median against the 5,000 floor and 308 of 376 cells within 1.2× of it. Monotonicity is
simply not resolved on the boundary.

So the bound was rebuilt **with the monotonicity assumption removed**: the log-likelihood is
allowed to keep climbing at the largest one-step rise measured on the boundary shell itself
(19.35 nats, on the `H0` axis), for all 5 grid steps of the rejected region's depth. That
gives an upper bound on the posterior mass behind the guard of **8.96e-09** — non-circular,
and still two orders of magnitude clear of 1e-6. For completeness: compounding instead the
largest gradient found three shells out (20.67 nats/step, set by the `f = 1` edge and
running in `H0`, not in the direction of approach) would give 6.30e-06; that extrapolation
is not physically motivated and is not adopted.

The grids do not clip the posterior. J9's edge densities relative to the peak are 3.31e-08
and 2.18e-08 in `H0`, 3.05e-21 and 0.0 in `f`, and 3.68e-23 and 9.45e-07 in `Δμ_χ`; S9's
are 3.09e-36 and 9.76e-31 in `H0`, 1.31e-12 and 0.0 in `f`. All clear the 1e-6 criterion —
but the `Δμ_χ` top edge clears it by only 6%, which is a limitation and is listed as one.

## Limitations

**One realisation, seed 100 only.** Everything above is a single draw. Nothing here
establishes that the estimator is calibrated across realisations, that its coverage is
correct, that the measured offsets are unbiased in expectation, or that the 11% `H0`
sharpening holds on average or with any particular scatter. No additional realisation, no
repeat draw and no effect-size ladder was run.

**The mass-ratio confound is live, and inherited.** Seed 100's detected set separates GAL
from AGN in mass ratio — KS `p = 0.0096` — while the model holds `q` identical in both
branches. That is an unmodelled channel difference, so the branch label is partly
identifiable without the spin mark and any intrinsic-channel statement carries it. That
includes `P(Δμ_χ ≤ 0) = 2.94e-11`, which is a posterior probability under **this** model and
**this** grid, with 0.0 interpolated between the neighbouring nodes at −0.0050 and +0.0025.
It is **not** a sigma, and it does not account for the kind of model error the `q`
separation is.

**The population is common and fixed apart from the mark.** Twelve base population
parameters and `Ω_m = 0.3075` are pinned by name; both branches share one mass function, one
mass ratio distribution, one spin width and one redshift rate. The only intrinsic difference
the model can express is the AGN branch's spin mean.

**Complete catalogs.** Both tracer surveys are complete; nothing here speaks to incomplete
catalogs.

**One mark, one effect size.** A single planted spin offset of +0.100 (realised +0.111924).
The information split measured here is the split at that effect size.

**The `Δμ_χ` axis is at its limit.** The top edge of `mu_chi_c2` sits at 9.45e-07 of the
peak against the 1e-6 containment criterion — clearing it by 6%, where the `f` ends and the
`Δμ_χ` low end clear it by 14 orders of magnitude or more and the `H0` ends by factors of
30 to 46. With `H0` free the posterior reaches the top of Analysis 8's `Δμ_χ` range. The
axis was deliberately not widened, so that the `Δμ_χ` widths with and without `H0` free
could be compared on identical quadrature — the right trade for this measurement — but it
means the containment margin on that one axis is 6%, not orders of magnitude, and a wider
range is the first thing a successor should extend.

**Nothing here is a statement about real AGN BBHs.** The mark is planted in a mock. No claim
is made that binaries in AGN discs carry a spin offset, nor that `H0` from real dark sirens
improves when a spin mark is added.

## What extension is justified next

Three things, in the order the numbers support them.

1. **Additional realisations.** This is the single largest gap and it is what turns the
   headline into a result. The `H0` sharpening — 0.888 and 0.900 on the matched lattice — is
   one number from one draw, and the realisation's own `H0` offset of +1.35 shows how much a
   single draw can move. Replication across seeds, scored differentially against the
   unmarked measurement of each, is what would convert an 11% scoping measurement into a
   calibrated gain with a scatter attached.
2. **A wider `Δμ_χ` range.** The top edge clears containment by 6%. Any successor that keeps
   `H0` free should extend that axis before it extends anything else; the matched-axis
   requirement that pinned it here does not apply once Analysis 8 is no longer the
   comparison.
3. **Handling the `q` confound.** KS `p = 0.0096` on this seed's detected set, against a
   model that holds `q` common. Either model `q` per branch — which is a real extension of
   the population, not a reweighting — or select seeds on the confound and say so. Until one
   of those is done, the intrinsic channel cannot be read in isolation.

The guard corner does not need work: it is 9 and 10 grid nodes beyond the 90% edges and
bounded at 8.96e-09 without assuming monotonicity.

**OWNER GATE: Analysis 9 single-seed science exploration is complete. No additional realizations were run.**
