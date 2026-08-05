# analysis_3_incomplete_catalog_H0_fagn — the expansion rate and the AGN host fraction from a host survey that cannot see everything

Round 3 of the campaign on `working/data/seed{100,101,102,103,105}` (**dataset v3 +
D3, float64**). The same likelihood and the same two free parameters as
[`analysis_2`](../analysis_2_complete_catalog_H0_fagn/README.md) — the Hubble
constant `H0` and the fraction `f_AGN` of gravitational-wave sources hosted by an
AGN — but the host catalogs are now **magnitude-limited**, down a ladder
`m < 21`, `m < 20`, `m < 19`, `m < 18`, on the same 1000 events and the same
injections.

Analysis 2 handed the fit every host in the universe. No survey does that. When
hosts are missing, the two tracer priors are no longer the observed catalogs: each
is the observed catalog **plus a missing-host budget** that the model has to supply
itself, from a smooth comoving density field. The question this directory asks is
what that substitution costs — how fast `sigma(H0)` and `sigma(f_AGN)` degrade as
the survey empties, and whether either parameter stays unbiased while it happens.

<!-- RESULTS_BANNER -->
> **Taking the host survey from complete to 10 % completeness inside the horizon costs `sigma(H0)` a factor **1.14×** and `sigma(f_AGN)` a factor **1.22×**.** Over the five realisations the faintest rung returns `H0` with a mean offset of **-0.79 ± 0.43** km s⁻¹ Mpc⁻¹ from 67.74 (`t(4) = -1.82`) and the host fraction **+0.084 ± 0.019** from each realisation's own realised fraction (`t(4) = +4.53`). 0 of 206,025 grid cells were rejected across the whole ladder.
<!-- /RESULTS_BANNER -->

---

## What changes, and what does not

**One configuration line changes between analysis 2 and analysis 3.** Analysis 2
suppressed the out-of-catalog term by setting `log10n0 = log10n0_c2 = -24`, the
complete-catalog limit: a density of `1e-24 Mpc^-3` means there is nothing the
survey could have missed. Here the term is switched on at the mock's own
densities. Everything else — the events, the injections, the grid, the guard, the
blocking, the KDE window, the population, the pinned `Om0`, the survey order
`[GAL, AGN]` that makes `fcat_2` the host fraction — is analysis 2's, unchanged.

The completion derives its missing-host budget as

```
dN_miss/dz  =  n0 dV_c/dz (1+z)^delta  -  dN_obs/dz
```

so completeness is never a free function: it is
`C(z) = (dN_obs/dz) / (n0 dV_c/dz (1+z)^delta)`, and `n0` sets the amplitude of
everything the survey did not see. Both `n0` and `delta` are held at the mock's
truth here — the most favourable case, and the one that isolates *completeness*
from *anchoring error*. (What happens when the density is not known is a separate
question; the prototype's `experiment_completeness_free` shows it is the dominant
one.)

### The densities, by two independent routes

`scripts/measure_true_density.py` → `results/true_density.json`.

<!-- DENSITY_TABLE -->
| tracer | declared `log10 n0` | counted, GLASS plateau (5 seeds) | counted, inside the horizon (5 seeds) | fitted `delta`, plateau | fit shape residual | **adopted** |
|---|---|---|---|---|---|---|
| GAL | -3.0000 | -3.0017 … -3.0016 | -3.0003 … -2.9997 | -0.0017 … +0.0025 | 0.19 – 0.25 % | **-3.0** |
| AGN | -5.0000 | -5.0020 … -5.0010 | -4.9998 … -4.9973 | -0.0577 … -0.0089 | 1.80 – 2.77 % | **-5.0** |

The two routes agree to **0.0020 dex** (0.46 %) at worst over the ten (seed, tracer) combinations, so the declared value is adopted for both tracers at every rung — one number for all five realisations, because the density is a property of the mock's construction rather than of a realisation.  The counted column is evaluated on the GLASS plateau, the interior redshift range over which the shell windows are a partition of unity and `dN/dz = n0 dV_c/dz` holds exactly; outside it the windows ramp linearly to zero over a shell half-width, which is why a naive count over the whole catalog reads about 8 % low and is not the right comparison.
<!-- /DENSITY_TABLE -->

### The evolution index

`delta = delta_c2 = 0`, **exactly**, and this is a statement about the mock rather
than a default left unexamined. Both tracers are GLASS shells drawn at *constant
comoving density*, and the population carries `gamma = 0` (`META.json`
`/stages/events/population/gamma`), so the true redshift dependence of the host
density is `(1+z)^0`. Fitting the completion's own model form to each complete
catalog returns an index consistent with that (`results/true_density.json`,
`model_form_fit`). Holding it at zero also leaves analysis 2's nuisance block
untouched, so `log10n0` is the **only** configuration difference between the two
directories.

That was expected to make analysis 2's complete-catalog grids usable as this
ladder's rung 0 directly. It does not: one configuration line is still a change of
estimator, and the continuity check below measures how much of one.

---

## Configuration

| | |
|---|---|
| model | **`dark_sirens`**, **K = 2 mixture**, `catalog_sky_weighting = field`, `use_lss` off |
| survey order | **`[GAL, AGN]`**, so darksirens' `fcat_2` **is** `f_AGN` |
| **completeness** | **`log10n0 = -3`, `log10n0_c2 = -5`** — the mock's true comoving densities; the out-of-catalog field term is **active** |
| nuisances | all fixed: `delta = delta_c2 = 0`, `sigma_kde = sigma_kde_c2 = 0` (`b_miss` drops out with `use_lss` off) |
| population | powerlaw+peak, fixed at the mock's own fiducial (`fix_population`) |
| cosmology | `Om0` pinned at 0.3075, `w0 = -1`, `wa = 0` |
| **free parameters** | **`H0` and `fcat_2`, and nothing else** |
| surveys | `data/seed<N>/surveys/survey_{gal,agn}_m{21,20,19,18}_ns32.h5` (nside 32, float64) |
| events | `data/seed<N>/events/events.h5` — **all 1000 events** × 2000 PE samples, identical at every rung |
| injections | `injections_targeted.h5` (`Ndraw = 1.5e8`) = **record**; `injections_popuni.h5` (`Ndraw = 4.0e8`) = cross-check; identical at every rung |
| guard | `--selection_neff_guard hard --max_likelihood_variance 1e6` — the campaign convention |
| posterior | flat prior, trapezoid marginals; equal-tailed CIs; 2-D regions are highest-posterior-density |
| numerics | windowed catalog KDE **`W = 4096`** (`n_sigma = 8`), `sel_batch_size = 50000`, `pe_event_block = 25` |
| grid | `H0 ∈ [50, 100] × 201` × `f ∈ [0, 1] × 41` = 8241 cells — analysis 2's grid, unchanged |
| darksirens | `/hildafs/projects/phy230014p/magana/src/darksirens` @ **`2b86a2d`**, read-only |
| device | `RITA-GPU` (A100-80), `HENON-GPU` (A100-40), `MIKO` (H100); `XLA_PYTHON_CLIENT_PREALLOCATE=false`. Every scan log records its own JAX device, so which grid ran on which card is provenance rather than an assumption |

### The ladder

The flux limit is isotropic and full-sky (the dataset's own `surveys` stage), so
completeness is a *consequence* of survey depth and has the declining shape a
flux-limited survey actually has. It thins both tracers with essentially the same
`C(z)`, because AGN inherit their host galaxy's apparent magnitude. Events are
untouched: incompleteness is an observational effect, so the events are exactly
the ones that happened.

<!-- LADDER_STRUCTURE -->
Seed 100, GW horizon `z <= 0.3105`.

| rung | GAL hosts | GAL block | GAL empty pix | AGN hosts | AGN block | AGN empty pix | `C(z <= z_hor)` |
|---|---|---|---|---|---|---|---|
| complete | 151,179,870 | 14,569 | 0.0 % | 1,514,567 | 178 | 0.0 % | 1.000 |
| `m < 21` | 24,202,870 | 2,795 | 0.0 % | 242,968 | 53 | 0.0 % | **1.000** |
| `m < 20` | 8,452,893 | 1,182 | 0.0 % | 84,547 | 23 | 0.2 % | 0.814 |
| `m < 19` | 2,733,114 | 457 | 0.0 % | 27,490 | 11 | 13.2 % | 0.315 |
| `m < 18` | 821,444 | 189 | 0.0 % | 8,274 | 5 | 52.8 % | 0.095 |
<!-- /LADDER_STRUCTURE -->

Three features of this ladder are worth naming before any result is read.

**The `m < 21` rung is complete inside the horizon.** It throws away 84 % of the
catalog and (on seed 100) not one host an event could occupy: everything it removes
lies beyond `z = 0.31`, where no event is. It is therefore a control rather than a
rung — it tests whether removing hosts the events cannot use costs anything, and
the honest prediction is that it should not.

**A rung is a flux limit, not a completeness.** The five realisations do not share
a horizon — `z_hor` runs from 0.311 (seed 100) to 0.387 (seed 105), because the
furthest event is a draw — so the same magnitude cut leaves a different fraction of
the hosts an event could actually occupy. At `m < 20` the galaxy completeness
inside the horizon runs 0.52 (seed 105) to 0.81 (seed 100); at `m < 21` it runs
0.99 to 1.00. The ladder table therefore quotes the five-realisation mean
completeness with its range, and `results/ladder_summary.json` carries the per-seed
values, because averaging five realisations at one rung is averaging over a spread
in completeness as well as over noise.

**The catalog blocks shrink by 77×.** The complete GAL block is `(12288, 14569)`;
the `m < 18` block is `(12288, 189)`. Since the selection integral's cost is set by
the block width, the whole ladder is far cheaper than analysis 2's five
complete-catalog grids — see the measured per-rung cost below.

---

## Validation, before the ladder ran

<!-- GATES_TABLE -->
**Cost, measured.** A K = 2 evaluation on the complete pair costs 3.74 s (analysis 2). The magnitude-limited pairs are far cheaper, so each 8241-cell grid fits in a single GPU task and there is no chunking.

| rung | s / eval | GPU-h / grid | grids | GPU-h |
|---|---|---|---|---|
| `m < 21` | 0.997 | 2.28 | 6 | 13.7 |
| `m < 20` | 0.516 | 1.18 | 6 | 7.1 |
| `m < 19` | 0.275 | 0.63 | 6 | 3.8 |
| `m < 18` | 0.185 | 0.42 | 6 | 2.5 |

Campaign total: **27.1 GPU-h** over 24 grids, against the 51 GPU-h analysis 2's five complete grids cost.

**Continuity with analysis 2**, complete catalogs, seed 100, targeted lane: the same data under the two configurations.

| cut | parameter | analysis 2 (`log10n0 = -24`) | analysis 3 (true `n0`) | shift | in a2 half-widths | width ratio |
|---|---|---|---|---|---|---|
| `h0scan` | `H0` | 69.248 ± 0.9506 | 69.653 ± 0.9431 | +0.40487 | +0.426 | 0.9921 |
| `fscan` | `f` | 0.2665 ± 0.04588 | 0.34624 ± 0.06163 | +0.079745 | +1.738 | 1.3433 |

**The selection integral at the peak** (`H0 = 67.74`, `f = 0.30`), seed 100:

| record | `N_eff` | vs the `5 N_obs` floor | `Σ σ²_PE` | admits |
|---|---|---|---|---|
| `guard_complete_s100` | 653,865 | 131× | 6.40 | True |
| `guard_m18_s100` | 558,657 | 112× | 5.21 | True |
| `guard_m18_s100_popuni` | 438,702 | 88× | 5.21 | True |
| `guard_m19_s100` | 681,034 | 136× | 5.73 | True |
| `guard_m20_s100` | 663,680 | 133× | 6.40 | True |
| `guard_m21_s100` | 654,132 | 131× | 6.41 | True |
<!-- /GATES_TABLE -->

These three ran as one GPU task before any ladder grid, and the ladder workers were
submitted `--dependency=afterok` on it with `scripts/gate_report.py` exiting nonzero
on failure — so a failed check leaves the 24 grids unstarted rather than letting
them run and be discarded afterwards.

The **continuity** check is the one that lets the ladder be quoted against analysis
2. It re-runs the complete GAL + AGN pair under *this* directory's configuration —
the field term switched on at the true densities instead of suppressed to `-24` —
on seed 100, and compares the two 1-D cuts through the peak (`f = 0.30` `H0`
column, truth-`H0` `f` column) against analysis 2's own `h0scan_s100` and
`fscan_s100`. On complete catalogs the missing-host budget has nothing to do, so
the two configurations should agree; whatever they do not agree by is the
systematic floor under every "× rung 0" ratio in the ladder table.

---

## The estimator's own offset on complete catalogs

Checks (a) and (c) gate the campaign and both passed. The timing pilot sized the
ladder at **27.1 GPU-h**, half of what analysis 2's five complete-catalog grids
cost, and the
selection integral is healthier at every rung than it is on the complete pair —
`N_eff` between `439k` and `681k`, i.e. **88× to 136×** the `5 N_obs` admission
floor, on both injection lanes and at the faintest rung, with the total-variance
criterion inert. The single reused injection set covers the magnitude-limited
rungs, as argued above. Nothing about the ladder itself is the problem.

Check (b) is not a gate — it is a measurement, and it returned a result. It was
written on the assumption that analysis 2's complete grids could serve as rung 0.
They cannot, and finding that out is worth more than the assumption was.

On complete catalogs the missing-host budget
`dN_miss/dz = n0 dV_c/dz - dN_obs/dz` should be zero, so analysis 2's
`log10n0 = -24` and this directory's true `n0` should be the same estimator. They
are not. `f_AGN` moves **+0.080** — 1.74 of analysis 2's own 68 % half-widths — and
its interval widens by **34 %**; `H0` moves +0.40 at unchanged width.

`scripts/diag_continuity_failure.py` → `results/continuity_failure_diag.json`
measures where that comes from, on seed 100, without a likelihood evaluation.

* **All-sky, the completion is nearly right.** Above the GLASS shell ramp the
  implied completeness is `C = 0.999 ± 0.005` (GAL) and `1.006 ± 0.019` (AGN),
  the latter matching its 2.1 % per-bin Poisson error. Integrated over the sky the
  spurious missing-host budget is only 0.18 % of the observed galaxies and 0.49 %
  of the observed AGN — far too small to move `f_AGN` by 30 %.
* **Per pixel, it is not.** The completion is evaluated per HEALPix pixel, and
  inside the horizon an nside-32 pixel holds a mean of **701 galaxies but 7.0 AGN**
  (10th percentile 3; 0.22 % of pixels empty). The AGN Poisson error per pixel is
  **38 %**, not 2 %. Wherever a pixel's AGN count fluctuates low the model reads
  `C < 1` and invents missing AGN; where it fluctuates high the budget clips at
  zero. The error is therefore **one-sided**, it inflates the AGN prior relative to
  the galaxy prior, and the mixture weight that measures their ratio rises.

This is the sparse-tracer mechanism `experiment_twotracer_incomplete` recorded
(`f = 0.2353` under the complete-catalog estimator against `0.4522` under the
incomplete one with a true anchor), and the important new fact is that **it did not
go away when the AGN catalog grew from 12,000 objects to 1.5 million.** What sets
it is hosts *per pixel per redshift bin*, which this dataset did not improve: the
deep mock had 154 AGN inside its horizon over 12,288 pixels, this one has 86,185,
and 7 per pixel is still Poisson-dominated.

**What this does not mean.** It is not a defect of the ladder, of the injections,
of the density anchor, or of darksirens. Both derivation routes for `n0` agree to
0.002 dex, and `delta = 0` is the mock's own construction. It is a property of the
completion at this tracer sparsity — and since it is present at every rung
including the complete one, it is completeness-*independent* and therefore
separable from what the ladder measures.

### The confirmatory test

The mechanism makes a quantitative prediction, fixed before the answer was looked
at. Regroup the **same hosts** into nside-16 pixels — 4× the solid angle, so 4× the
AGN per pixel (7.0 → 28), so half the per-pixel Poisson error — and the one-sided
budget, and with it the shift, should fall by about **2×**:

```
shift(nside 16) / shift(nside 32)  ~  sqrt(N_32 / N_16)  =  0.50
```

`scripts/degrade_survey_nside.py` performs the regrouping exactly (a fine HEALPix
pixel lies wholly inside one coarse pixel, so no host is added, removed or altered;
the generator's RING ordering, `100.0 / 1.0 / 0.0` padding and z-sorted-within-pixel
invariant are all reproduced and asserted, and the host count is verified preserved
tracer by tracer). Both configurations are re-run at nside 16, so the comparison is
shift-against-shift at fixed pixelisation and the coarser sky's own effect on how
well `f_AGN` is measured cancels out of the ratio. The KDE window is a numerical
requirement set by the densest block, so it necessarily changes with the
pixelisation (`W = 4096` at nside 32, a re-measured `W = 16384` at nside 16); what
has to be held fixed is that both nside-16 arms share it, and they do.
`scripts/nside_scaling_verdict.py` → `results/nside_scaling.json`.

**The result, below, confirms the mechanism's direction but not its full strength.**
The offset does shrink when pixels are merged — it is a per-pixel effect — but by
less than pure Poisson counting predicts even after allowing for the
pixelisation-independent part of the budget. What that leaves open is how much of
the remainder is the redshift binning, which merging pixels does not coarsen at
all: the completion works per pixel *per redshift bin*, and only one of those two
was changed here. Answering that is a second scan, not an argument, and it is not
run here.

### How the ladder is quoted

Two references, reported side by side, because they answer different questions:

* **rung 0 of record** — the complete pair re-run *in this configuration*, five
  seeds plus the seed-100 popuni cross-check. Every `× R0` ratio down the ladder is
  against this, so it isolates **completeness degradation** with the estimator held
  fixed.
* **the analysis-2 reference** — the same complete catalogs with the
  out-of-catalog budget suppressed (`log10n0 = -24`). Rung 0 minus this, paired per
  realisation, is the **estimator's own sparse-pixel offset**, and it is
  completeness-independent.

Summing them would confound the two; `results/ladder_summary.json` keeps them apart
(`rungs.*.width.sigma_*_vs_rung0` and
`analysis_2_reference.estimator_offset_rung0_minus_analysis2`).

---

## Scans

Per realisation, all 1000 events, one joint grid per rung on the targeted
injection lane. The grid is analysis 2's, unchanged: `H0 [50,100] x 201` x
`f [0,1] x 41` = 8241 cells.

| # | scan | grid | output |
|---|---|---|---|
| 0 | **rung 0 of record**, 5 seeds, complete pair | joint, in 4 H0 chunks | `results/joint_complete_s<seed>.{h5,json}` |
| 1 | **joint**, 5 seeds × 4 rungs | joint | `results/joint_<level>_s<seed>.{h5,json}` |
| 2 | **popuni cross-check**, seed 100 × 5 rungs | joint | `results/joint_<level>_s100_popuni.{h5,json}` |
| 3 | **sky-shuffle null**, seed 100, `m < 18` | `f ∈ [0, 1] × 101` at `H0 = 67.74` | `results/fscan_null_m18_s100.{h5,json}` |
| 4 | **nside-16 scaling test**, seed 100, complete pair | `f` scan, both configurations | `results/fscan_complete_ns16_{truen0,n24}_s100.{h5,json}` |

The complete-catalog rung is run **here**, not read from analysis 2: the continuity
measurement showed the two configurations are different estimators on this data, so
rung 0 has to share an estimator with the rungs it is compared against. A
complete-pair grid costs 6.8 GPU-h, more than one worker's walltime, so each is
split into 4 contiguous `H0` chunks that `scripts/merge_joint.py` stitches after
asserting the reassembled axis reproduces `linspace(50, 100, 201)` exactly.
Analysis 2's grids are kept as the zero-missing-budget reference.

<!-- RESULTS_BODY -->
## The ladder, rung by rung

Five realisations per rung. `sigma` is the mean 68 % half-width of the marginal; `× R0` is that against the complete-catalog rung. Offsets are mean ± s.e.m. over the five realisations, `H0` against 67.74 and `f` against each realisation's own realised host fraction. Coverage counts realisations whose interval contains truth.

| rung | `C(z<=z_hor)` | `sigma(H0)` | × R0 | `H0` offset | 68 / 90 | `sigma(f)` | × R0 | `f` offset | 68 / 90 | `rho` |
|---|---|---|---|---|---|---|---|---|---|---|
| complete | 1.000 | 1.070 | 1.00 | +1.08 ± 0.53 | 2 / 3 | 0.0637 | 1.00 | +0.046 ± 0.022 | 4 / 4 | -0.067 |
| `m < 21` | 0.998 (0.992–1.000) | 1.070 | 1.00 | +1.08 ± 0.53 | 2 / 3 | 0.0637 | 1.00 | +0.045 ± 0.022 | 4 / 4 | -0.066 |
| `m < 20` | 0.699 (0.520–0.814) | 1.064 | 0.99 | +0.99 ± 0.50 | 3 / 3 | 0.0641 | 1.01 | +0.051 ± 0.022 | 4 / 4 | -0.070 |
| `m < 19` | 0.256 (0.174–0.315) | 1.299 | 1.21 | +1.09 ± 0.27 | 3 / 4 | 0.0670 | 1.05 | +0.071 ± 0.023 | 2 / 4 | -0.092 |
| `m < 18` | 0.078 (0.053–0.095) | 1.216 | 1.14 | -0.79 ± 0.43 | 3 / 5 | 0.0775 | 1.22 | +0.084 ± 0.019 | 2 / 4 | -0.017 |

### `m < 21`, realisation by realisation

| seed | AGN-hosted | realised `f` | `H0` ± 68 % | offset | in 68 / 90 | `f` ± 68 % | offset vs realised | vs 0.30 | in 68 / 90 | `rho` |
|---|---|---|---|---|---|---|---|---|---|---|
| 100 | 295 | 0.295 | 69.61 ± 0.93 | +1.87 | no / no | 0.342 ± 0.062 | +0.047 | +0.042 | yes / yes | -0.069 |
| 101 | 326 | 0.326 | 68.77 ± 1.28 | +1.03 | yes / yes | 0.378 ± 0.063 | +0.052 | +0.078 | yes / yes | +0.036 |
| 102 | 308 | 0.308 | 66.91 ± 1.07 | -0.83 | yes / yes | 0.303 ± 0.068 | -0.005 | +0.003 | yes / yes | -0.057 |
| 103 | 277 | 0.277 | 69.98 ± 0.90 | +2.24 | no / no | 0.291 ± 0.059 | +0.014 | -0.009 | yes / yes | -0.133 |
| 105 | 289 | 0.289 | 68.85 ± 1.17 | +1.11 | no / yes | 0.410 ± 0.067 | +0.121 | +0.110 | no / no | -0.108 |

### `m < 20`, realisation by realisation

| seed | AGN-hosted | realised `f` | `H0` ± 68 % | offset | in 68 / 90 | `f` ± 68 % | offset vs realised | vs 0.30 | in 68 / 90 | `rho` |
|---|---|---|---|---|---|---|---|---|---|---|
| 100 | 295 | 0.295 | 69.47 ± 0.93 | +1.73 | no / no | 0.347 ± 0.063 | +0.052 | +0.047 | yes / yes | -0.062 |
| 101 | 326 | 0.326 | 68.67 ± 1.24 | +0.93 | yes / yes | 0.385 ± 0.063 | +0.059 | +0.085 | yes / yes | +0.024 |
| 102 | 308 | 0.308 | 66.97 ± 1.12 | -0.77 | yes / yes | 0.308 ± 0.068 | +0.000 | +0.008 | yes / yes | -0.062 |
| 103 | 277 | 0.277 | 69.88 ± 0.88 | +2.14 | no / no | 0.294 ± 0.060 | +0.017 | -0.006 | yes / yes | -0.135 |
| 105 | 289 | 0.289 | 68.65 ± 1.14 | +0.91 | yes / yes | 0.415 ± 0.067 | +0.126 | +0.115 | no / no | -0.117 |

### `m < 19`, realisation by realisation

| seed | AGN-hosted | realised `f` | `H0` ± 68 % | offset | in 68 / 90 | `f` ± 68 % | offset vs realised | vs 0.30 | in 68 / 90 | `rho` |
|---|---|---|---|---|---|---|---|---|---|---|
| 100 | 295 | 0.295 | 68.97 ± 0.95 | +1.23 | no / yes | 0.361 ± 0.065 | +0.066 | +0.061 | no / yes | -0.064 |
| 101 | 326 | 0.326 | 68.98 ± 1.36 | +1.24 | yes / yes | 0.423 ± 0.067 | +0.097 | +0.123 | no / yes | -0.023 |
| 102 | 308 | 0.308 | 68.98 ± 2.14 | +1.24 | yes / yes | 0.327 ± 0.070 | +0.019 | +0.027 | yes / yes | -0.081 |
| 103 | 277 | 0.277 | 69.41 ± 0.92 | +1.67 | no / no | 0.306 ± 0.062 | +0.029 | +0.006 | yes / yes | -0.107 |
| 105 | 289 | 0.289 | 67.80 ± 1.12 | +0.06 | yes / yes | 0.436 ± 0.072 | +0.147 | +0.136 | no / no | -0.185 |

### `m < 18`, realisation by realisation

| seed | AGN-hosted | realised `f` | `H0` ± 68 % | offset | in 68 / 90 | `f` ± 68 % | offset vs realised | vs 0.30 | in 68 / 90 | `rho` |
|---|---|---|---|---|---|---|---|---|---|---|
| 100 | 295 | 0.295 | 68.32 ± 1.20 | +0.58 | yes / yes | 0.368 ± 0.074 | +0.073 | +0.068 | no / yes | -0.091 |
| 101 | 326 | 0.326 | 65.98 ± 1.04 | -1.76 | no / yes | 0.437 ± 0.077 | +0.111 | +0.137 | no / yes | -0.014 |
| 102 | 308 | 0.308 | 66.08 ± 1.34 | -1.66 | no / yes | 0.346 ± 0.084 | +0.038 | +0.046 | yes / yes | +0.051 |
| 103 | 277 | 0.277 | 67.00 ± 1.32 | -0.74 | yes / yes | 0.335 ± 0.073 | +0.058 | +0.035 | yes / yes | +0.033 |
| 105 | 289 | 0.289 | 67.37 ± 1.17 | -0.37 | yes / yes | 0.431 ± 0.079 | +0.142 | +0.131 | no / no | -0.065 |

### The selection integral along the ladder

Across all five joint grids at each rung.

| rung | `N_eff` range | worst vs the `5 N_obs` floor | max `Σ σ²_PE` | cells rejected |
|---|---|---|---|---|
| complete | 470,770 – 858,497 | 94× | 21.4 | 0 / 41,205 |
| `m < 21` | 470,819 – 856,963 | 94× | 21.4 | 0 / 41,205 |
| `m < 20` | 476,548 – 818,461 | 95× | 20.9 | 0 / 41,205 |
| `m < 19` | 279,531 – 712,176 | 56× | 17.5 | 0 / 41,205 |
| `m < 18` | 195,278 – 623,659 | 39× | 11.4 | 0 / 41,205 |

### The estimator's own offset, separated from completeness

Rung 0 (the complete pair with the true-`n0` completion active) minus analysis 2 (the same complete pair with the out-of-catalog budget suppressed), paired per realisation. A complete catalog has no missing hosts, so a correct completion would give zero here. This offset is completeness-independent and is **not** part of the `× R0` columns above.

| parameter | rung 0 − analysis 2 | in a2 68 % half-widths | `sigma` ratio |
|---|---|---|---|
| `H0` | +0.677 ± 0.176 | +0.647 | 1.023 |
| `f_AGN` | +0.0576 ± 0.0065 | +1.150 | 1.272 |

### The scaling test

The same hosts, regrouped into 4× larger pixels. If the offset is per-pixel Poisson noise, quadrupling the AGN per pixel should halve it.

| pixelisation | `f` (true `n0`) | `f` (`log10n0 = -24`) | shift | in `-24` half-widths |
|---|---|---|---|---|
| nside32 | 0.3462 ± 0.0616 | 0.2665 ± 0.0459 | +0.0797 | +1.738 |
| nside16 | 0.3511 ± 0.0666 | 0.2918 ± 0.0546 | +0.0593 | +1.086 |

The offset **shrinks**, which is the direction the mechanism requires. Pre-registered prediction `0.50` (pure `1/sqrt(N_per_pixel)`); observed **`0.743`**; allowing for the 18 % of the spurious budget that comes from the pixelisation-independent GLASS low-z ramp raises the expectation to `0.59`. That is inside the pre-registered band `[0.33, 0.75]`, but at its edge: the offset is per-pixel in origin and shrinks with coarser pixels, yet more slowly than pure Poisson counting alone would give.

### Injection lanes

The targeted lane is the record, population+uniform the cross-check: the same detection rule with different proposals, so they must agree. Seed 100.

| rung | `H0` targeted | popuni | Δ | in 68 % half-widths | `f` targeted | popuni | Δ | in 68 % half-widths |
|---|---|---|---|---|---|---|---|---|
| complete | 69.61 | 69.89 | +0.29 | +0.307 | 0.3420 | 0.3293 | -0.0127 | -0.203 |
| `m < 21` | 69.61 | 69.89 | +0.28 | +0.307 | 0.3416 | 0.3289 | -0.0127 | -0.203 |
| `m < 20` | 69.47 | 69.70 | +0.22 | +0.241 | 0.3474 | 0.3346 | -0.0128 | -0.202 |
| `m < 19` | 68.97 | 68.93 | -0.04 | -0.045 | 0.3612 | 0.3469 | -0.0143 | -0.219 |
| `m < 18` | 68.32 | 68.65 | +0.33 | +0.275 | 0.3684 | 0.3486 | -0.0199 | -0.267 |

### The sky-shuffle null, at the faintest rung

Permuting the per-event `(ra, dec)` blocks among events destroys every host association while leaving each event's distance, masses, spin and localisation area untouched, and leaving the same patches of sky occupied. Anything the mixture weight still "measures" afterwards was never host-association information.

| | median `f` | 68 % interval | 90 % interval |
|---|---|---|---|
| record (seed 100, `m < 18`) | **0.368** | ± 0.074 | — |
| sky-shuffled | **0.078** | [0.027, 0.148] | [0.009, 0.201] |

The recorded value sits **4.78 null widths** away, with the null's width 0.82× the record's.
<!-- /RESULTS_BODY -->

---

## Two places where this directory departs from the prototype

The design prototype is
[`experiments/experiment_twotracer_incomplete`](../experiments/experiment_twotracer_incomplete/DESIGN.md),
which ran this exact ladder on the older 200-event deep mock. Its conventions are
followed except in two places, both forced by the campaign dataset.

**1. The density anchor is the mock's declared density, not a least-squares fit to
the observed one.** The prototype fitted `n0` and `delta` to the true host density
because its GLASS tracer density was *"whatever the lognormal field produced, not
exactly that form"*, and it carried the fit residual (2.6 % GAL, 6.2 % AGN) as a
noise floor. Here the generator states the density it drew (`n_comoving_gal = 1e-3`,
`n_comoving_agn = 1e-5`) and the direct count reproduces it to better than 0.5 %,
so the declared value **is** the best fit and there is nothing to prefer a fit
over. The fit is still performed, as the cross-check
(`results/true_density.json`), and it is what makes `delta = 0` a measurement
rather than a default. The prototype's fitted `delta` (`+0.019` GAL, `-0.003` AGN)
also has no counterpart here: those were the residual slopes of a field that did
not quite follow the model form, and this mock's does.

**2. The injections are not rebuilt per rung.** The prototype generated one
catalog-targeted injection set per rung, on the argument that the selection
integral is catalog-conditioned and its support shrinks with the survey. This
campaign uses the dataset's single signed-off injection sets at every rung, which
is what keeps the rungs differential against analysis 2 rather than confounded
with a change of proposal. The argument that this is safe rather than merely
convenient runs the other way from the prototype's: the targeted lane's catalog
branch targets the **complete** AGN survey, whose support strictly *contains* every
magnitude-limited AGN survey's, and the out-of-catalog term the incomplete rungs
lean on is smooth and full-sky, which is exactly what the 65 % population branch
covers. Over-coverage costs efficiency, not validity — and it is measurable, so it
is measured: the `N_eff` column of the ladder table and the per-rung guard records
in `results/guard/` are the evidence, not the argument. The prototype's own result
points the same way (`N_eff` *rose* along its ladder, because the incomplete
model's target becomes dominated by the smooth field term).

A third, smaller difference: the prototype noted that its `diag_variance_guard.py`
evaluated `N_eff` at `delta = 0` while its scans used the anchored `delta`. Here
the scans use `delta = 0` too, so the diagnostic and the scans are the same
configuration and that caveat does not arise.

---

## Figures

Drawn by `scripts/make_figures.py` from `results/` only — no number is typed into a
figure — in analysis 2's palette and rcParams so the three directories read as one
system. The finalizer renders them automatically; a figure whose inputs are not on
disk is skipped rather than drawn from partial data.

| file | what it shows |
|---|---|
| `figs/fig_ladder_widths.{pdf,png}` | the headline: `sigma(H0)` and `sigma(f_AGN)` against survey depth, every realisation plus the five-realisation mean, with analysis 2's complete-limit record as a reference line |
| `figs/fig_closure_ladder.{pdf,png}` | medians ± 68 % for both parameters at every rung, against truth and against each realisation's own realised host fraction |
| `figs/fig_estimator_offset.{pdf,png}` | the offset the completion has on **complete** catalogs, beside the per-pixel occupancy that causes it (7.0 AGN vs 701 galaxies per pixel) |
| `figs/fig_nside_scaling.{pdf,png}` | the confirmatory test: the offset at two pixelisations, and predicted vs observed shift ratio |
| `figs/fig_null_m18.{pdf,png}` | the sky-shuffle null at the faintest rung |

---

## Files

```
analysis_3_incomplete_catalog_H0_fagn/
  README.md                   this file
  scripts/
    env.sh                    the one K = 2 configuration every scan here shares;
                              analysis_2/scripts/env.sh with the field term
                              switched on and a LEVEL argument added
    run_scan.sh               one scan: KIND=joint|fscan|h0scan|fnull, LEVEL=<rung>
    measure_true_density.py   the field term's amplitude, both derivation routes
    pilot_timing.sh           steady-state s/eval at every rung
    continuity_check.py       this configuration vs analysis 2 on complete catalogs
    run_guard.sh              N_eff and the validity guard at every rung
    gate_report.py            the three checks -> results/gates.json (exits nonzero
                              on failure; the ladder workers depend on it)
    make_ladder_queue.sh      build the 25-task ladder work queue
    ladder_worker.sh          one GPU worker on that queue (atomic mkdir claim)
    diag_continuity_failure.py
                              why the complete-catalog continuity check moved:
                              C(z) and per-pixel occupancy on complete catalogs
                              -> results/continuity_failure_diag.json
    degrade_survey_nside.py   exact regrouping of a survey block to a coarser
                              nside (same hosts, coarser pixels)
    nside_scaling_verdict.py  the 1/sqrt(N_per_pixel) prediction against what the
                              nside-16 arms measured -> results/nside_scaling.json
    merge_complete_rung.sh    stitch the complete-rung H0 chunks into rung 0
    aggregate_ladder.py       the per-rung x per-realisation closure table, widths
                              against rung 0, the estimator offset against analysis
                              2, lane agreement, the null
                              -> results/ladder_summary.json + h0_fagn_ladder.json
    status.sh                 campaign progress: gates, queue, SLURM
    make_figures.py           the five figures, from results/ only
    render_readme_tables.py   regenerates this README's tables from the JSON
    submit_gates.sbatch       timing + guard + the continuity measurement, one task
    submit_nside16.sbatch     the confirmatory scaling test end to end
    submit_finalize.sbatch    CPU: stitch, aggregate, render (afterany the above)
    finalize.sh               the same, runnable by hand; idempotent
    submit_ladder_{rita,henon,miko}.sbatch
                              GPU workers on the shared queue, one wrapper per
                              partition; short (6 h) walltimes so they backfill,
                              and the worker prices each candidate grid from the
                              measured per-rung s/eval so it never claims one it
                              cannot finish
    scan_h0f.py, merge_joint.py, shuffle_event_sky.py, diag_variance_guard.py,
    kde_window_check.py, aggregate_joint.py, joint_worker.sh
                              copied UNCHANGED from analysis_2/scripts (same
                              MERGE_SHA 2b86a2d, same per-cell guard spy)
    _ref_a2_*                 read-only copies of analysis_2's own drivers, kept
                              beside their adaptations so the diff is visible
  results/
    true_density.json         the adopted log10n0, both routes, all five seeds
    gates.json                timing, continuity, guard
    continuity_vs_analysis2.json
    guard/guard_<level>_s100[_popuni].json
    joint_<level>_s<seed>.{h5,json}          the ladder grids of record
    joint_<level>_s100_popuni.{h5,json}      the cross-check lane
    joint_complete_s<seed>.{h5,json}         RUNG 0 OF RECORD (stitched)
    chunks/joint_complete_s<seed>_c<k>.{h5,json}  its four H0 chunks
    fscan_null_m18_s100.{h5,json}            the sky-shuffle null
    fscan_complete_ns16_{truen0,n24}_s100.{h5,json}  the scaling test
    nside_scaling.json        prediction vs observation for the scaling test
    degrade_nside.json        the exact nside-16 regrouping's provenance
    kde_window_ns16.json      the window the coarser blocks need
    h0scan_complete_s100.{h5,json}, fscan_complete_s100.{h5,json}
                              the continuity cuts
    continuity_failure_diag.json  the diagnosis of the stop
    ladder_summary.json       the full per-rung table (currently rung 0 only)
    h0_fagn_ladder.json       the compact hook
    pilot/                    the timing probes
  logs/                       one log per scan, plus the SLURM job logs
  queue/                      the ladder work queue and its claims
  figs/
```

## Reproducing

```bash
cd working/analyses/analysis_3_incomplete_catalog_H0_fagn

# the field term's amplitude, both derivation routes (CPU, ~15 min over five seeds)
python -u scripts/measure_true_density.py --seeds 100 101 102 103 105

# timing + guard, one GPU task; the continuity measurement rides along
sbatch --partition=RITA-GPU scripts/submit_gates.sbatch

# the campaign: 24 complete-rung H0 chunks (rung 0 of record) + 24 ladder grids
# + the null.  One worker per GPU on every partition; workers price each task from
# the measured s/eval and claim only what their walltime covers.
./scripts/make_ladder_queue.sh
sbatch --array=0-3 scripts/submit_ladder_rita.sbatch
sbatch --array=0-5 scripts/submit_ladder_henon.sbatch
sbatch --array=0-3 scripts/submit_ladder_miko.sbatch

# the confirmatory scaling test (builds its own nside-16 blocks)
sbatch scripts/submit_nside16.sbatch

# collate: stitch rung 0, aggregate, render.  Idempotent; also runs itself via
# scripts/submit_finalize.sbatch with --dependency=afterany on the above.
./scripts/finalize.sh

# progress at any point
./scripts/status.sh
```

## Cost

| piece | grids | GPU-h |
|---|---|---|
| rung 0 of record (complete pair, 5 seeds + popuni) | 6 (× 4 H0 chunks) | 40.8 |
| the ladder (`m21`..`m18`, 5 seeds + popuni per rung) | 24 | 27.1 |
| the null and the nside-16 scaling test | 3 scans | ~0.3 |
| **total** | | **~68** |

Rung 0 dominates, because the complete GAL block is 77× wider than the `m < 18` one
and a K = 2 evaluation's cost is set by that width. It is the price of the ladder's
reference sharing an estimator with its rungs, which the continuity measurement
showed is not optional.
