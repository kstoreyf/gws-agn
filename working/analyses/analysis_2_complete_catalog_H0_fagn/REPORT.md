# analysis_2_complete_catalog_H0_fagn — measuring the expansion rate and the AGN host fraction together

Round 2 of the campaign on `working/data/seed{100,101,102,103,105}` (**dataset v3 +
D3, float64**). One likelihood, two free parameters: the Hubble constant `H0` and
the fraction `f_AGN` of gravitational-wave sources whose host is an AGN. Both
complete catalogs — the galaxies and the AGN — enter the same fit as a two-component
mixture, and the mixture weight *is* the host fraction.

This is the analysis that
[`analysis_1_complete_catalog_H0`](../analysis_1_complete_catalog_H0/README.md)
motivates. There, each catalog was handed all 1000 events on its own and asked for
`H0` alone. The galaxy catalog is missing the 295 AGN-hosted events, the AGN catalog
is missing the 705 galaxy-hosted ones, and neither is told so: the GAL scan lands at
**69.9⁺¹·⁷₋¹·⁶** and the AGN scan **rails against the top of the prior range** and
returns no interval at all. Only when each catalog was handed exactly the events it
actually hosts — the matched-host controls, which no real analysis can construct
because nobody knows which events those are — did both recover truth
(`+0.81 ± 0.62` GAL, `+0.42 ± 0.47` AGN over five realisations, `CLOSURE.md` §16).

The mixture fit is the analysis that does not need to be told. It lets the data
decide what share of the sources each tracer hosts, and measures that share at the
same time as the expansion rate.

<!-- RESULTS_BANNER -->
> **Both parameters are recovered.** Over 5 independent realisations the joint fit returns `H0` with a mean offset of **+0.41 ± 0.55** km s⁻¹ Mpc⁻¹ from 67.74 (`t(4) = +0.73`) and the AGN host fraction with a mean offset of **-0.012 ± 0.020** from each realisation's own realised fraction (`t(4) = -0.57`), or -0.013 ± 0.019 from the planted 0.30. Truth is inside the 68 % interval on 2 / 5 realisations for `H0` and 3 / 5 for `f`; inside the 90 % on 4 / 5 and 5 / 5. On the reference realisation the fit reads `H0` = 69.2^{+1.0}_{-1.0}, `f_AGN` = 0.273^{+0.049}_{-0.047}, with a correlation of +0.07.
<!-- /RESULTS_BANNER -->

---

## What is being measured

Hosts were planted from the mixture `(1 - f) GAL + f AGN` with **`f_AGN = 0.30`**.
AGN are a *separate* GLASS tracer painted on the same density field, not a subset of
the galaxies, so a source is hosted by one catalog or the other and never both. The
realised counts differ from realisation to realisation because 1000 events is a
finite draw:

| seed | GAL-hosted | AGN-hosted | realised `f_AGN` |
|---|---|---|---|
| 100 | 705 | 295 | 0.295 |
| 101 | 674 | 326 | 0.326 |
| 102 | 692 | 308 | 0.308 |
| 103 | 723 | 277 | 0.277 |
| 105 | 711 | 289 | 0.289 |
| | | | mean **0.299** |

**Two truth references for `f`, and this directory reports against both.**

* The **realised** per-seed fraction is the closure reference. The mixture weight is
  estimated from the events that were actually drawn; with perfect host
  identification the maximum-likelihood estimate of a mixture weight from `N` events
  *is* the realised fraction. The gap between the realised fraction and 0.30 is the
  mock's own binomial draw — a property of the dataset, not of the estimator — and
  charging the estimator for it would be charging it for noise it cannot see.
* The **planted** 0.30 is the population parameter. An offset against it carries the
  extra binomial term `sqrt(0.3 × 0.7 / 1000) = 0.0145` per realisation, `0.0065` on
  the five-realisation mean.

For `H0` there is only one truth: **67.74**.

---

## Configuration

One estimator, the campaign's, unchanged from analysis 1 except for the second
catalog and the second free parameter.

| | |
|---|---|
| model | **`dark_sirens`**, **K = 2 mixture**, `catalog_sky_weighting = field`, `use_lss` off |
| survey order | **`[GAL, AGN]`**, so darksirens' `fcat_2` **is** `f_AGN` |
| completeness | **`log10n0 = log10n0_c2 = -24`** — the complete-catalog limit in both catalogs |
| nuisances | all fixed: `delta = delta_c2 = 0`, `sigma_kde = sigma_kde_c2 = 0` (`b_miss` drops out with `use_lss` off) |
| population | powerlaw+peak, fixed at the mock's own fiducial (`fix_population`) |
| cosmology | `Om0` pinned at 0.3075, `w0 = -1`, `wa = 0` |
| **free parameters** | **`H0` and `fcat_2`, and nothing else** |
| free labels | `["H0", "log10n0", "delta", "sigma_kde", "log10n0_c2", "delta_c2", "sigma_kde_c2", "fcat_2"]`; only `H0` and `fcat_2` are scanned |
| surveys | `data/seed<N>/surveys/survey_{gal,agn}_complete_ns32.h5` (nside 32, float64) |
| events | `data/seed<N>/events/events.h5` — **all 1000 events** × 2000 PE samples |
| injections | `injections_targeted.h5` (`Ndraw = 1.5e8`) = **record**; `injections_popuni.h5` (`Ndraw = 4.0e8`) = **cross-check** |
| guard | `--selection_neff_guard hard --max_likelihood_variance 1e6` — the campaign convention: the legacy `N_eff > 5 N_obs` floor, total-variance criterion made inert |
| posterior | flat prior, trapezoid marginals; equal-tailed CIs; 2-D regions are highest-posterior-density |
| numerics | windowed catalog KDE **`W = 4096`** (`n_sigma = 8`), `sel_batch_size = 50000`, `pe_event_block = 25` |
| darksirens | `/hildafs/projects/phy230014p/magana/src/darksirens` @ **`2b86a2d`**, read-only |
| device | A100-40 (`HENON-GPU`, `TWIG-GPU`), `XLA_PYTHON_CLIENT_PREALLOCATE=false` |

### The two numerical settings, re-measured rather than assumed

* **The catalog-KDE window.** `recommended_kde_window` on the v3 survey blocks
  returns **3422** at `n_sigma = 8` with `sigma_kde = 0`, set by the dense GAL block
  (`(12288, 14569)`; the AGN block needs 56). **`W = 4096`** clears it, unchanged
  from analysis 1 — the GAL block sets the requirement and the AGN block is
  irrelevant to it (`scripts/kde_window_check.py` → `results/kde_window.json`).
* **Reduction blocking.** `(sel_batch_size, pe_event_block) = (50000, 25)` is
  analysis 1's setting. Whether coarser blocking is faster was **measured**, not
  assumed: `(100000, 50)` and `(200000, 100)` both return **1.00×** the throughput
  and change the log-likelihood by `1.8e-12` (pure summation order), so the setting
  stays exactly analysis 1's and cannot enter any comparison between the two
  directories (`scripts/pilot_blocking.sh`, `logs/pilot_blk_*.log`).

### Cost, measured

A K = 2 evaluation on the **complete** GAL + AGN pair costs **3.71 s** on one
A100-40 (`scripts/pilot_timing.sh`). The campaign's stored note of ~0.11 s/eval for
K = 2 was measured on the *magnitude-limited* survey blocks and does not transfer:
the complete GAL block is the cost, and the K = 1 complete-GAL scan of analysis 1
costs 6.13 s/eval on the same hardware. The joint grid is therefore
`201 × 41 = 8241` evaluations ≈ **8.5 GPU-hours per realisation**, not the ~15
minutes a 0.11 s/eval extrapolation would predict.

Two consequences, both structural rather than scientific:

* Each joint grid is split into **8 contiguous `H0` chunks** that run on separate
  GPUs and are stitched by `scripts/merge_joint.py`. The chunk boundaries are exact
  subsets of `linspace(50, 100, 201)` (step 0.25, a power of two), and the merge
  **asserts** that the reassembled axis reproduces that linspace value for value
  before writing anything.
* Running two evaluations concurrently on one GPU was **measured and rejected**:
  8.56 s/eval each against 3.71 s solo, i.e. a 15 % throughput *loss*
  (`scripts/pilot_concurrency.sh`). One process per GPU.

---

## Scans

Per realisation, all 1000 events, on the targeted injection lane:

| # | scan | grid | output |
|---|---|---|---|
| 1 | **joint** | `H0 ∈ [50, 100] × 201` × `f ∈ [0, 1] × 41` | `results/joint_s<seed>.{h5,json}` |
| 2 | **fscan** | `f ∈ [0, 1] × 101` at `H0 = 67.74` | `results/fscan_s<seed>.{h5,json}` |
| 3 | **h0scan** | `H0 ∈ [50, 100] × 201` at `f = 0.30` | `results/h0scan_s<seed>.{h5,json}` |

Seed 100 additionally carries the **popuni cross-check** for all three
(`*_popuni.{h5,json}`) and the **sky-shuffle null** for the `f` scan
(`results/fscan_null_s100.{h5,json}`).

<!-- RESULTS_BODY -->
## The joint fit, realisation by realisation

Medians with equal-tailed 68 % intervals from the 2-D posterior's marginals. `offset` is median − truth; for `f` the truth is that realisation's own **realised** host fraction, with the offset against the planted 0.30 alongside.

| seed | AGN-hosted | realised `f` | `H0` median ± 68 % | offset | in 68 / 90 | `f` median ± 68 % | offset vs realised | vs 0.30 | in 68 / 90 | `rho(H0, f)` |
|---|---|---|---|---|---|---|---|---|---|---|
| 100 | 295 | 0.295 | 69.22 ± 0.97 | **+1.48** | no / yes | 0.273 ± 0.048 | **-0.022** | -0.027 | yes / yes | +0.068 |
| 101 | 326 | 0.326 | 67.82 ± 1.30 | **+0.08** | yes / yes | 0.308 ± 0.050 | **-0.018** | +0.008 | yes / yes | +0.233 |
| 102 | 308 | 0.308 | 66.53 ± 1.10 | **-1.21** | no / yes | 0.247 ± 0.052 | **-0.061** | -0.053 | no / yes | +0.105 |
| 103 | 277 | 0.277 | 69.56 ± 0.91 | **+1.82** | no / no | 0.257 ± 0.049 | **-0.020** | -0.043 | yes / yes | +0.029 |
| 105 | 289 | 0.289 | 67.60 ± 0.95 | **-0.14** | yes / yes | 0.353 ± 0.051 | **+0.064** | +0.053 | no / yes | +0.089 |
| | | | | `+0.41 ± 0.55` | **2 / 4** of 5 | | `-0.012 ± 0.020` | `-0.013 ± 0.019` | **3 / 5** of 5 | +0.105 |

### Closure over the five realisations

| quantity | truth | mean offset ± s.e.m. | `t(4)` | realisation scatter | mean quoted half-width | scatter / width |
|---|---|---|---|---|---|---|
| `H0` | 67.74 | **+0.41 ± 0.55** | +0.73 | 1.24 | 1.05 | 1.19 × |
| `f_AGN` vs **realised** | per seed | **-0.012 ± 0.020** | -0.57 | 0.043 | 0.050 | 0.86 × |
| `f_AGN` vs **planted** | 0.30 | **-0.013 ± 0.019** | -0.66 | — | — | — |

The binomial term separating the two `f` references is `sqrt(0.3 × 0.7 / 1000) = 0.0145` per realisation, `0.0065` on the five-realisation mean.

### The two one-dimensional cuts

`fscan` fixes `H0` at truth and scans `f` on 101 points; `h0scan` fixes `f` at the planted 0.30 and scans `H0` on 201. They are cuts through the same likelihood, not independent measurements, and they carry no marginalisation over the other parameter.

| seed | realised `f` | `fscan` `f` ± 68 % | offset vs realised | `h0scan` `H0` ± 68 % | offset |
|---|---|---|---|---|---|
| 100 | 0.295 | 0.266 ± 0.046 | -0.029 | 69.25 ± 0.95 | +1.51 |
| 101 | 0.326 | 0.307 ± 0.047 | -0.019 | 67.76 ± 1.26 | +0.02 |
| 102 | 0.308 | 0.252 ± 0.051 | -0.056 | 66.65 ± 1.03 | -1.09 |
| 103 | 0.277 | 0.244 ± 0.045 | -0.033 | 69.58 ± 0.86 | +1.84 |
| 105 | 0.289 | 0.354 ± 0.050 | +0.065 | 67.52 ± 0.96 | -0.22 |
| | | | **-0.014 ± 0.021** | | **+0.41 ± 0.55** |

### The selection integral across the `f` axis

**0 of 41,205 grid cells were rejected** across the 5 joint grids — the guard never fires anywhere on the (`H0`, `f`) plane, at either end of the `f` axis.

At truth `H0` = 67.75, seed 100, targeted lane:

| `f` | 0.0 | 0.25 | 0.50 | 0.75 | 1.0 |
|---|---|---|---|---|---|
| `N_eff` | 514,705 | 717,098 | 750,433 | 571,438 | 378,819 |
| `Σ σ²_PE` | 5.3 | 6.6 | 10.1 | 16.5 | 54.7 |
| `× 5N_obs` floor | 103× | 143× | 150× | 114× | 76× |

Over the whole seed-100 grid `N_eff` runs 231,774 – 871,151 against a flat floor of 5 000, and `Σ σ²_PE` runs 4.2 – 110.1 against the campaign's inert `1e6` cap.

### Injection lanes

The targeted lane is the record; population+uniform is the cross-check. They are the same detection rule with different proposals, so they must agree.

| scan | parameter | targeted | popuni | difference | in 68 % half-widths |
|---|---|---|---|---|---|
| `joint` | `H0` | 69.22 | 69.27 | +0.06 | +0.058 |
| `joint` | `f` | 0.2734 | 0.2673 | -0.0061 | -0.127 |
| `fscan` | `f` | 0.2665 | 0.2622 | -0.0043 | -0.093 |
| `h0scan` | `H0` | 69.25 | 69.32 | +0.08 | +0.080 |

### The sky-shuffle null

Permuting the per-event `(ra, dec)` blocks among events destroys every host association while leaving each event's distance, masses, spin and localisation area untouched, and leaving the same patches of sky occupied. Anything the mixture weight still "measures" afterwards was never host-association information.

| | median `f` | 68 % interval | 90 % interval |
|---|---|---|---|
| record (seed 100) | **0.266** | ± 0.046 | — |
| sky-shuffled | **0.037** | [0.012, 0.076] | [0.004, 0.106] |

The weight collapses toward zero and the recorded value 0.266 lies far outside the shuffled 90 % interval. A weight pinned by the two catalogs' global normalisations would have survived the permutation unchanged; this one does not.

<!-- /RESULTS_BODY -->

---

## What this means

Two catalogs describe the same sky, and a gravitational-wave source sits in one of
them. Analysis 1 asked each catalog, on its own, to convert a thousand distances
into an expansion rate. The galaxy catalog, blind to the 295 sources hosted by
AGN, returned `H0 = 69.9⁺¹·⁷₋¹·⁶` — truth outside its 68 % interval and at the very
edge of its 90 % — and nothing in that output says anything is wrong. The AGN
catalog, blind to the other 705, failed loudly instead: its likelihood rises
monotonically across the whole scanned range, it rails against the top of the
prior, and `results/h0_single_tracer.json` carries `null` where its interval should
be. Only the matched-host controls — each catalog handed exactly the events it
actually hosts — close on truth over five realisations (`+0.81 ± 0.62` for the
galaxies, `+0.42 ± 0.47` for the AGN), and those are not analyses but answers,
because building one requires already knowing which catalog hosts each source.

What neither single-catalog fit is allowed to do is *ask* what share of the
sources each catalog hosts. Letting it ask fixes both problems at once. With the two
catalogs entering as a mixture whose weight is the AGN host fraction, and with
nothing told to the fit beyond the catalogs themselves, the expansion rate comes
back at **`+0.41 ± 0.55 km s⁻¹ Mpc⁻¹`** of the truth over five independent
realisations of the mock, and the AGN host fraction at **`−0.012 ± 0.020`** of the
fraction each realisation actually contains (`−0.013 ± 0.019` against the planted
0.30). Neither is a significant offset: `t(4) = +0.73` and `−0.57`. The AGN
catalog stops railing — it never comes near the edge of the range in any
realisation — because the mixture no longer forces it to account for sources it
does not host. On the reference realisation the fit reads
`H0 = 69.2⁺¹·⁰₋¹·⁰` and `f_AGN = 0.273⁺⁰·⁰⁴⁹₋⁰·⁰⁴⁷`.

**The two measurements barely interfere.** The correlation between them is
`rho = +0.105 ± 0.035` over the five realisations — positive, statistically
resolved, and small enough that the host fraction costs the expansion rate almost
nothing in precision. That is not obvious in advance: a mixture weight and a
distance scale could easily have traded off, because raising `f` moves the host
prior toward a sparser, differently distributed set of galaxies, which the fit
could partly compensate by moving `H0`. It does not, and the reason is visible in
the data: the two catalogs trace the same underlying density field over the same
redshift range, so changing the mixture weight changes *which* galaxies carry the
prior far more than it changes *where in redshift* they sit. The expansion rate is
set by the second, the host fraction by the first. The practical consequence is
that adding the second free parameter costs nothing: on the reference realisation
the joint fit's `H0` interval is **1.94** wide against the **3.30** of analysis 1's
single-catalog galaxy scan, which had `f` implicitly pinned at zero and got the
answer wrong.

**What the fit costs operationally is concentrated at one end of the `f` axis, and
it is the sparse tracer's end.** The selection integral's effective sample size
rises from `514,705` at `f = 0` to a maximum of `769,570` near `f = 0.4` and then
falls to `378,819` at `f = 1`, while the per-event parameter-estimation variance
climbs monotonically from `5.3` to `54.7` — a factor of ten, almost all of it in
the last stretch above `f ≈ 0.9`, where the prior leans on a catalog with 178
objects in its busiest pixel instead of 14,569. The untargeted injection lane
degrades monotonically instead, from `602,568` to `58,289` — a factor of `10.3`
across the same axis — because its proposals were never aimed at AGN hosts. Both
lanes clear the `5 N_obs` admission floor everywhere, by `46 ×` at worst on the
targeted lane and `7.2 ×` at worst on the cross-check, and **not one of
the 41,205 cells in the five joint grids was rejected**, so nothing in this
directory sits behind a guard. But the trend is the honest warning for a sparser
tracer or a deeper catalog: the cost of the mixture is paid at `f → 1`, and it is
paid in the selection integral first.

**The host fraction is measured from host associations, not from bookkeeping.**
Permuting which patch of sky each event's distance belongs to — leaving every
distance, mass, spin and localisation area untouched, and leaving the same patches
occupied — collapses the fraction from `0.266 ± 0.046` to `0.037⁺⁰·⁰³⁹₋⁰·⁰²⁶`, with
the recorded value excluded far outside the shuffled 90 % interval `[0.004, 0.106]`.
A mixture weight pinned by the two catalogs' global normalisations would have
survived that permutation unchanged. This one does not: with no event's distance
paired to its own host's redshift, the fit concludes that essentially nothing is
hosted by an AGN, which is the correct answer to the question the shuffled data
actually pose.

### Scope

The quoted intervals are the statistical width of a flat-prior likelihood on a
fixed grid, on a mock whose population, cosmology apart from `H0`, and photometric
redshift model are all handed to the fit exactly right. What is established is
that the mixture estimator is unbiased in both parameters on this dataset at this
sample size — five realisations bound the mean `H0` offset to `± 0.55` and the mean
`f` offset to `± 0.020`, so a bias smaller than about a third of one realisation's
width would not have been seen. The realisation-to-realisation scatter is
`1.19 ×` the mean quoted `H0` half-width and `0.86 ×` the `f` one, i.e. the quoted
widths are honest to the precision five realisations can test. Truth lands inside
the 68 % interval on 2 of 5 realisations for `H0` and 3 of 5 for `f`, and inside
the 90 % on 4 of 5 and 5 of 5 — consistent with the nominal rates on five draws.
Nothing here bears on incomplete catalogs, on a mis-specified population, or on
the case where the two tracers do not share a density field.

Per analysis 1's convention (`CLOSURE.md` §14.2, §16.6), the selection estimator's
own **common-mode Monte-Carlo error** on `d ln mu/dH0` is carried rather than
dropped. Converted on this analysis's own measured curvature
(`|d² lnL/dH0²| = 9.80e-4` per event from the joint grid's `H0` marginal), the
values analysis 1 measured on these same v3 injection sets give **`± 0.12`
(`f = 0` limit) to `± 0.54` (`f = 1` limit) km s⁻¹ Mpc⁻¹ per realisation**, i.e.
`± 0.05` to `± 0.24` on the five-realisation mean. The mixture's own `sigma_MC`
was not measured here; those two limits bracket it exactly, because at `f = 0` and
`f = 1` the mixture's selection integral **is** the single-tracer one — verified
bit-for-bit on all four (tracer, lane) combinations by the `N_eff` endpoints below.
Carried at its pessimistic end the closure statement reads

```
H0      +0.41 +- 0.55 (realisations) +- <=0.24 (selection MC)  =  +0.41 +- <=0.60
f_AGN   -0.012 +- 0.020 (realisations)
```

— the term is real, it is bounded on this dataset, and it does not change the
verdict.

### The mixture reproduces the single-tracer limits exactly

At `f = 0` the AGN catalog must contribute nothing and at `f = 1` the galaxy
catalog must; if the mixture is wired correctly its selection integral at those two
points is bit-for-bit the corresponding K = 1 run of analysis 1. It is, on all four
combinations:

| limit | lane | K = 2 min `N_eff` | analysis 1 K = 1 | difference |
|---|---|---|---|---|
| `f = 0` → GAL | targeted | 427,275 | 427,275 | `0` |
| `f = 1` → AGN | targeted | 231,774 | 231,774 | `0` |
| `f = 0` → GAL | popuni | 523,575 | 523,575 | `0` |
| `f = 1` → AGN | popuni | 36,237 | 36,237 | `0` |

The chunking is checked the same way: the joint grid's `f = 0.30` column reproduces
the independently-run `h0scan` to `max |Δ lnL| = 3.6e-12` (`8.5e-16` relative), the
one-ulp difference between the grid point `0.30000000000000004` and the scan's
`0.3`.

---

## Figures

| file | what it shows |
|---|---|
| `figs/fig_joint_h0fagn.{pdf,png}` | the 2-D 68 / 90 % credible regions — seed 100 filled, the other four realisations as outlines, the truth cross and each realisation's realised host fraction |
| `figs/fig_marginals.{pdf,png}` | the `H0` and `f_AGN` marginal posteriors of all five realisations, against truth |
| `figs/fig_closure_joint.{pdf,png}` | per-realisation medians ± 68 % for both parameters, against truth and the five-realisation mean |
| `figs/fig_neff_f.{pdf,png}` | the selection integral's `N_eff` and the PE variance sum against `f`, both injection lanes, against the admission floor |

---

## Files

```
analysis_2_complete_catalog_H0_fagn/
  README.md                   this file
  scripts/
    scan_h0f.py               the (H0, f) grid driver, copied unchanged from
                              analysis_1/scripts (same MERGE_SHA 2b86a2d, same
                              per-cell guard spy); --scan joint drives K = 2
    env.sh                    the one K = 2 configuration every scan here shares
    run_scan.sh               one scan: KIND=joint|fscan|h0scan|fnull
    make_joint_queue.sh       build the 48-chunk joint-grid work queue
    joint_worker.sh           one GPU worker on that queue (atomic mkdir claim)
    merge_joint.py            stitch the H0 chunks; asserts the axis reassembles
                              exactly; writes the joint summary (both marginals,
                              rho, the guard block, N_eff against f)
    shuffle_event_sky.py      the null: permute the per-event (ra, dec) blocks
                              (adapted from experiment_twotracer_incomplete, not
                              imported from it)
    aggregate_joint.py        the five-realisation closure table for BOTH
                              parameters, lane agreement, the null ->
                              results/joint_summary.json + results/h0_fagn_joint.json
    make_figures.py           the four figures
    validate_palette.py       Python twin of the dataviz skill's node validator
                              (this cluster has no node); reproduces its published
                              numbers exactly
    mu_mc_error.py            the selection estimator's common-mode Monte-Carlo
                              error, converted on this analysis's own curvature
    render_readme_tables.py   regenerates this README's result tables from
                              results/joint_summary.json (no hand-typed numbers)
    kde_window_check.py       copied from analysis_1
    diag_variance_guard.py    copied from analysis_1
    pilot_timing.sh / pilot_concurrency.sh / pilot_blocking.sh
                              the three measured sizing decisions above
    status.sh / watch.sh      campaign progress, used while the grids ran
    submit_*.sbatch           SLURM wrappers (HENON-GPU / TWIG-GPU / RITA-GPU / MIKO)
    tasks_1d.txt              the 1-D scan task table
    _ref_*                    read-only copies taken from analysis_1/scripts for
                              reference while that directory was being reorganised
  results/
    joint_s<seed>.{h5,json}            the joint grids of record
    fscan_s<seed>.{h5,json}            f at truth H0
    h0scan_s<seed>.{h5,json}           H0 at f = 0.30
    *_popuni.{h5,json}                 the seed-100 cross-check lane
    fscan_null_s100.{h5,json}          the sky-shuffle null
    chunks/joint_s<seed>_c<k>.{h5,json}  the H0 chunks the merge consumes
    joint_summary.json                 the full five-realisation table
    h0_fagn_joint.json                 the paper hook (analysis_1 conventions)
    mu_mc_error.json                   the carried selection-MC error
    kde_window.json                    the measured KDE window
    pilot/                             the three sizing probes
  figs/
    fig_joint_h0fagn.{pdf,png}   the 2-D credible regions
    fig_marginals.{pdf,png}      both marginals across realisations
    fig_closure_joint.{pdf,png}  the closing exhibit
    fig_neff_f.{pdf,png}         the operational exhibit
  logs/                         one log per scan, plus the SLURM job logs
  queue/                        the joint-grid work queue and its claims
```

## Reproducing

```bash
cd working/analyses/analysis_2_complete_catalog_H0_fagn

# the 1-D scans (13 tasks, ~2 GPU-hours)
sbatch --array=0-12 --partition=HENON-GPU scripts/submit_1d.sbatch

# the joint grids (48 chunks, ~51 GPU-hours) -- submit one worker per free GPU
./scripts/make_joint_queue.sh
sbatch --array=0-1 scripts/submit_joint_henon.sbatch
sbatch --array=0-0 scripts/submit_joint_twig.sbatch

# stitch, aggregate, draw
for s in 100 101 102 103 105; do
  python scripts/merge_joint.py --chunks results/chunks/joint_s${s}_c*.h5 \
      --out_tag joint_s${s} --outdir results
done
python scripts/merge_joint.py --chunks results/chunks/joint_s100_popuni_c*.h5 \
    --out_tag joint_s100_popuni --outdir results
python scripts/aggregate_joint.py --seeds 100 101 102 103 105
python scripts/make_figures.py
```
