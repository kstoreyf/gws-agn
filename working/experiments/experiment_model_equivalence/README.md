# experiment_model_equivalence — is the general model the complete model in the limit?

**Question.** darksirens ships two catalog likelihoods. `dark_sirens_complete` asserts
the host is in the catalog. `dark_sirens` models the galaxies the catalog is missing,
and should reduce to the complete-catalog likelihood as the modelled missing density
goes to zero. If it does, one nested likelihood can carry every run in the eventual
paper. This experiment asks whether it does, on the *current* dataset
(`working/data/seed100`), and **where the two agree bit for bit**.

The equivalence was established once before, on older data, after darksirens PR #215.
Nothing here assumes that still holds.

> **Answer: yes, and exactly.** Once the modelled missing density is taken far enough
> to zero, `dark_sirens` reproduces `dark_sirens_complete` **bit for bit — 201/201
> cells, `max |Δ ln L| = 0`, in all four configurations**. "Far enough" matters: at the
> `log10n0 = -12` that the campaign has been using, the completion term is *not* yet
> off. On the dense GAL catalog its residual is at the float64 rounding floor (8 ulps,
> `3.6e-12` nats), but on the sparse AGN catalog it reaches `4.1e-6` nats — still
> scientifically nothing (the posterior median moves by `1e-13` km s⁻¹ Mpc⁻¹) but not
> bit equality. At `log10n0 = -24` it is exactly zero everywhere.
>
> **But none of that can be reached on the survey files as shipped.** On
> `working/data/seed100`'s float32 survey files, `dark_sirens` returns `-inf` in every
> cell at every `log10n0` — a dtype underflow in darksirens' observed-density KDE
> makes the survey-global normalizer NaN. The equivalence arms above run on float64
> copies of the same files. **Before `dark_sirens` can carry every run in the paper,
> either the surveys are written in float64 or darksirens is fixed.** See
> [the blocker](#the-blocker-dark_sirens-cannot-be-evaluated-on-the-survey-files-as-shipped).

---

## Results

Ran as SLURM job **1058122** on `henon-gpu01`, one **A100-40**, 2026-07-31 02:56–05:02
UTC (2 h 05 m); 18 scans in that job plus 5 preserved from a first pass, 23 grids in
all. **Zero guard-rejected cells in any of the 16 float64 scans.** darksirens `2b86a2d`,
read-only.

### The equivalence — primary arm, `log10n0 = -12`

`ulp` is one unit in the last place of the log-likelihood being compared, i.e. the
smallest difference float64 can represent there.

| configuration | verdict | cells identical to the bit | max \|Δ ln L\| | median \|Δ ln L\| | Δ median H₀ |
|---|---|---|---|---|---|
| GAL, all 1000 events | float-level | **108 / 201** | 3.64e-12 (8 ulp) | 0 | +4.2e-13 |
| GAL, 720 GAL-hosted | float-level | **80 / 201** | 3.64e-12 (8 ulp) | 9.1e-13 | −2.8e-14 |
| AGN, all 1000 events | structurally different † | **18 / 201** | 4.11e-06 | 4.9e-10 | +1.4e-13 |
| AGN, 280 AGN-hosted | structurally different † | **10 / 201** | 4.21e-09 | 1.4e-10 | +3.5e-11 |

† by the stated `1e-9` bar, applied mechanically. It is not a structural mismatch: it
is the completion term, which at `n0 = 1e-12 Mpc⁻³` is small but not switched off. Its
shape says so. On AGN + all events the residual is `+4.1e-6` at `H0 = 50`, falls by an
order of magnitude per grid decade to a sign change at `H0 = 53.75`, reaches `-6.4e-9`
near `H0 = 55`, and then decays smoothly to the float64 rounding floor by `H0 ≈ 91`,
beyond which the remaining cells are ±1–8 ulps of hash. That is a smooth function of
`H0`, not numerical noise, and it is the missing branch: the general model's numerator is
`logaddexp(N_obs log p_cat, log dN_miss)`, and `dN_miss = (1 − C) dN_exp` is nonzero
exactly where the catalog KDE has no support — 95 % of (pixel, kernel) cells in the
sparse AGN catalog, almost none in the dense GAL one, which is why AGN's residual is
six orders of magnitude larger than GAL's on the same grid.

A single-point pilot (`results/pilot_n0_limit.json`) confirms the residual is the
completion term by taking it to zero: on AGN + all events at `H0 = 67.74`, Δ ln L runs
−1.1e-9 (`log10n0 = -12`) → −1.8e-12 (−14 … −20, the float64 re-association floor) →
**exactly 0 (−24)**.

### The equivalence — deep limit, `log10n0 = -24`

| configuration | verdict | cells identical to the bit | max \|Δ ln L\| | Δ median H₀ |
|---|---|---|---|---|
| GAL, all 1000 events | **bitwise** | **201 / 201** | **0** | **0** |
| GAL, 720 GAL-hosted | **bitwise** | **201 / 201** | **0** | **0** |
| AGN, all 1000 events | **bitwise** | **201 / 201** | **0** | **0** |
| AGN, 280 AGN-hosted | **bitwise** | **201 / 201** | **0** | **0** |

Not "agrees to 15 digits" — the same 64 bits in every one of 804 cells, and posterior
medians that are the same float64 to the last digit
(`60.10557923193964`, `62.78539066237382`, `99.7696699072826`, `67.40119359501826`).

**Recommendation for the paper**: `dark_sirens` can carry every run, including the
complete-catalog ones, provided the complete-catalog limit is taken at
`log10n0 ≲ -20`, not `-12`. At `-12` the answers are identical to well past any
scientific precision, but they are not the *same numbers*, and a paper that claims one
nested likelihood is cleanest if the nesting is exact.

### Posterior level, primary arm

| configuration | `dark_sirens_complete` median | `dark_sirens` (−12) median | `dark_sirens` (−24) median |
|---|---|---|---|
| GAL, all 1000 | 60.10557923193964 | 60.10557923194006 | 60.10557923193964 |
| GAL, 720 matched | 62.78539066237382 | 62.78539066237379 | 62.78539066237382 |
| AGN, all 1000 | 99.7696699072826 | 99.76966990728275 | 99.7696699072826 |
| AGN, 280 matched | 67.40119359501826 | 67.40119359505279 | 67.40119359501826 |

The largest posterior disagreement anywhere in the primary arm is **3.5e-11 km s⁻¹
Mpc⁻¹**, on 68 % half-widths of order 1–2.

### Reference arm reproduces the analysis of record

The `dark_sirens_complete` scans on the shipped survey files land on
`analysis_1_complete_catalog_H0`'s published medians to ≤ 3.4e-13 km s⁻¹ Mpc⁻¹
(`h0_gal_targeted` 60.1055794439, `h0_agn_targeted` 99.7696699046, `ctrl_gal_matched`
62.7853907601, `ctrl_agn_matched` 67.4011935694). The residual is the different
reduction blocking (50000/25 here against 200000/100 there), not the estimator. So the
reference arm of this experiment *is* the analysis of record.

### Secondary arm — the completion term at the tracer's true density

**Characterization, not pass/fail.** `dark_sirens` is told the tracer's true comoving
number density (GAL `log10n0 = -3`, AGN `-5`) on a catalog that is in fact complete, so
what it adds is the shot-noise missing-host budget: the model does not know the catalog
is complete and reserves probability for hosts it thinks are absent.

| configuration | max \|Δ ln L\| | median \|Δ ln L\| | shape across the grid | complete median | true-density median | shift |
|---|---|---|---|---|---|---|
| GAL, all 1000 | 6.29 | 0.36 | general below complete everywhere, non-monotone | 60.106 | 60.927 | **+0.822** |
| GAL, 720 matched | 3.87 | 0.071 | mixed sign (65 above, 136 below) | 62.785 | 62.174 | **−0.612** |
| AGN, all 1000 | 2404 | 990 | general **above** complete everywhere, falling monotonically with H₀ | 99.770 | 66.679 | **−33.091** |
| AGN, 280 matched | 53.2 | 8.3 | mixed sign (41 above, 160 below) | 67.401 | 68.360 | **+0.959** |

Every cell differs in all four — as it should; this arm is a different model, not a
limit of one.

The AGN + all-events row is worth flagging to the campaign, though it is outside this
experiment's question. Under `dark_sirens_complete` that configuration rails against the
top of the prior (median 99.77, 68 % [99.38, 99.93]) because 720 of its 1000 events have
no host in the AGN catalog and the model must place each on *some* AGN. Given a
missing-host channel at the true AGN density, **the railing disappears**: median 66.679,
68 % [65.865, 67.460], against a truth of 67.74. On the matched 280 events the same
change costs about 1 km s⁻¹ Mpc⁻¹ in the other direction (67.401 → 68.360). Neither is
a result of this experiment; both say the completion term is doing real work and deserve
their own run.

### Precision control — does the float64 widening change the reference model?

Not measurably. `dark_sirens_complete` on the shipped float32 files versus the float64
copies:

| configuration | max \|Δ ln L\| | Δ median H₀ |
|---|---|---|
| GAL, all 1000 | 1.09e-06 | +2.1e-07 |
| GAL, 720 matched | 8.37e-07 | +9.8e-08 |
| AGN, all 1000 | 2.01e-05 | −2.7e-09 |
| AGN, 280 matched | 9.41e-07 | −2.6e-08 |

No cell is bit-identical (the arithmetic really is done at a different precision), but
the largest posterior shift is 2e-7 km s⁻¹ Mpc⁻¹. The float64 copies give the same
science as the shipped files.

One caveat on this table only: three of its four float32 grids were produced by a first
job (1058121, same node, cancelled once the blocker was found) rather than by 1058122,
so in principle they carry a device/compilation difference as well as a dtype one. The
fourth pair — `gal_matched`, whose float32 and float64 grids were both produced inside
1058122 — gives the same answer (0/201 bitwise, max 8.4e-7), so the effect being
measured is the dtype. Nothing in the equivalence arms above crosses a job: all 16
float64 scans ran back to back on one GPU in 1058122.

### Figure

`figs/fig_equivalence.{pdf,png}` — left: per-cell `|Δ ln L|` across the H₀ grid for the
primary arm, on a symmetric-log axis whose linear core contains the exactly-zero cells,
with the one-ulp level marked; right: the secondary arm's posterior shifts.

### Cost

3.84 s/eval (GAL, 1000 events), 3.73 (GAL, 720), 0.19 (AGN, 1000), 0.14 (AGN, 280) — and
identical to three digits across all four arms, so the completion term costs nothing
measurable over the complete-catalog likelihood.

---

## What was actually run

Every scan is a pure `H0` likelihood grid — no sampler, no prior — evaluated by module
import through `scripts/scan_h0f.py` (a byte-identical copy of
`analyses/analysis_1_complete_catalog_H0/scripts/scan_h0f.py`, sha256 `d0da6fe…`; it
already handles both `universe_model` values and the `--log10n0` flags, so nothing
needed changing).

| | |
|---|---|
| grid | `H0` ∈ [50, 100], 201 points; truth **67.74** |
| reference model | `dark_sirens_complete`, K = 1, `catalog_sky_weighting = field`, free labels `["H0", "sigma_kde"]`, `sigma_kde = 0` |
| general model | `dark_sirens`, K = 1, field weighting, `use_lss` off, free labels `["H0", "log10n0", "delta", "sigma_kde"]`, all nuisances fixed at `delta = 0`, `sigma_kde = 0`, only `log10n0` varied between arms |
| population | powerlaw+peak, fixed at the mock's own fiducial |
| cosmology | `Om0` pinned at 0.3075, `w0 = -1`, `wa = 0` |
| events | `data/events/events.h5` (1000), and the matched host-type subsets **read** from `analyses/analysis_1_complete_catalog_H0/data_derived/events_{gal,agn}_hosted.h5` |
| injections | `injections_targeted.h5` — the analysis of record's lane, the **same file** for every scan |
| guard | `--selection_neff_guard hard --max_likelihood_variance 1e6` (the campaign convention: the legacy `N_eff > 5 N_obs` floor, total-variance criterion inert) |
| numerics | windowed catalog KDE `W = 4096` (`n_sigma = 8`) for GAL; `sel_batch_size = 50000`, `pe_event_block = 25` — half of analysis_1's, because HENON-GPU carries A100-**40**s. Identical for every scan, so it cannot enter any comparison |
| darksirens | `/hildafs/projects/phy230014p/magana/src/darksirens` @ `2b86a2d`, read-only |
| device | **one** A100-40 on `henon-gpu01`, one serial SLURM job |

Bit equality does not survive a change of device or of compilation, so all arms run on
the same physical GPU in the same job. That is the only reason this is not a job array.

### Note on the matched subsets

The task brief expected 684 / 316 matched events. The subsets on disk (and the current
`events.h5`) are **720 GAL-hosted and 280 AGN-hosted** — `realised_f_agn = 0.28` in the
events metadata. The existing files were used as instructed and nothing in
`working/analyses/` was written.

---

## The four configurations and the four arms

Configurations (the same four in every arm):

| key | catalog | events |
|---|---|---|
| `gal_all` | complete GAL survey, nside 32 | all 1000 |
| `gal_matched` | complete GAL survey | 720 GAL-hosted |
| `agn_all` | complete AGN survey, nside 32 | all 1000 |
| `agn_matched` | complete AGN survey | 280 AGN-hosted |

Arms:

| tag prefix | model | `log10n0` | role |
|---|---|---|---|
| `dsc_` | `dark_sirens_complete` | — | **reference** |
| `ds_` | `dark_sirens` | −12 | **primary**: the complete-catalog limit as specified |
| `dsdeep_` | `dark_sirens` | −24 | **primary, deeper**: added after a pilot showed the −12 residual *is* the completion term (it scales with `n0`), to locate where the limit becomes exact |
| `dstrue_` | `dark_sirens` | GAL −3, AGN −5 | **secondary**, characterization only — what the completion term does when the model is told the tracer's true density on a catalog that is in fact complete |
| `f32_dsc_` | `dark_sirens_complete` | — | precision control, on the survey files as shipped |
| `f32_ds_`, `f32_ds1pt_` | `dark_sirens` | −12 | the blocker, on the survey files as shipped |

---

## The blocker: `dark_sirens` cannot be evaluated on the survey files as shipped

Before any equivalence can be measured, this has to be said plainly: **on
`working/data/seed100`'s survey files, `dark_sirens` returns `-inf` in every cell of
every grid, at every `log10n0` from −12 to −3.** `dark_sirens_complete` is unaffected.
This is not a limit artefact and not a guard threshold — the selection integral is
*identically zero* (`N_eff = 0.0`, `pe_variance_sum = 0.0`) because the redshift prior
is NaN everywhere.

The mechanism, traced to a single line
(`darksirens/redshift/completion.py::_kde_dndz_obs`, @ 2b86a2d):

```python
mass = ndtr((_ZMAX - zs) / _SIGMA_SMOOTH) - ndtr(-zs / _SIGMA_SMOOTH)   # _ZMAX = 5, sigma = 0.05
mass = jnp.maximum(mass, 1e-300)
pdf  = jnp.exp(-0.5 * ((zgrid[:, None] - zs[None, :]) / _SIGMA_SMOOTH) ** 2) / (_SQRT2PI * _SIGMA_SMOOTH)
kern = (pdf / mass[None, :]) * real_gal[None, :].astype(pdf.dtype)
```

`mass` is computed in the **catalog's** storage dtype; `pdf` is promoted to the package
`zgrid`'s float64. The seed100 survey files store galaxies in **float32**
(`working/data/generate_dataset.py`, `CAT_DTYPE = "float32"` — a deliberate size choice
for a 12288 × 14569 block) and pad short rows at **z = 100**. For a padded slot:

```
float32:  mass raw = 0.000e+00   after max(..., 1e-300) = 0.000e+00     <-- 1e-300 is not representable
float64:  mass raw = 0.000e+00   after max(..., 1e-300) = 1.000e-300
```

so in float32 the kernel evaluates `0 / 0 = NaN`, and the `* real_gal` mask cannot
remove it (`0 * NaN = NaN`). **Every catalog row that has any padding comes back
all-NaN**; only the single row sitting at the maximum galaxy count survives (measured:
0.008 % of the AGN cache finite, and exactly one row is at `ngals = 178`). The NaN
propagates into the survey-global field normalizer `log_Z_global`, so every injection
and PE weight is NaN and the selection guard rejects every cell.
`dark_sirens_complete` never reads that KDE, which is why it runs normally on the same
files.

The padding convention is not new — the campaign's earlier survey files
(`analyses/experiments/experiment_twotracer_deep/data/survey_*_ns32.h5`) pad at z = 100
too. They are **float64**, which is why `dark_sirens` worked there. The new variable is
the float32 storage of the seed100 dataset.

**Consequence for the paper plan:** as things stand, a single nested `dark_sirens`
likelihood cannot be run on the current dataset at all. Either the survey files are
written in float64, or darksirens' `_kde_dndz_obs` computes `mass` in the grid dtype.

### The workaround used here

`scripts/run_equivalence.sh` runs the equivalence arms on **float64 copies** of the two
survey files, written to `data_derived/survey_{gal,agn}_complete_ns32_f64.h5`. This is a
pure precision widening: every float32 value is exactly representable in float64, and
the round trip back to float32 is asserted to be the identity, so **no number changed**.
darksirens itself is untouched (read-only, no patch, no runtime monkeypatch of any
numerical path — the driver's only instrumentation is analysis_1's pass-through guard
spy).

Because the widening also changes the *reference* model's arithmetic at the last bits,
the reference model is run on **both** survey variants and the difference is reported as
a precision control.

---

## Files

```
scripts/
  scan_h0f.py             the grid driver — a byte-identical copy of
                          analysis_1's (sha256 d0da6fe…); no edits were needed
  run_equivalence.sh      all the scans, in one serial pass on one GPU
  submit_equivalence.sbatch  HENON-GPU, account phy220048p, QOS henon-gpu,
                          gres=gpu:1, 8 CPUs
  compare_models.py       the comparison -> results/equivalence_summary.json
                          and figs/fig_equivalence.{pdf,png}

data_derived/
  survey_{gal,agn}_complete_ns32_f64.h5   float64 copies of the shipped surveys
                                          (see "The workaround used here")

results/
  dsc_<config>.{h5,json}      reference,  dark_sirens_complete
  ds_<config>.{h5,json}       primary,    dark_sirens at log10n0 = -12
  dsdeep_<config>.{h5,json}   primary,    dark_sirens at log10n0 = -24
  dstrue_<config>.{h5,json}   secondary,  dark_sirens at the true density
  f32_dsc_<config>.{h5,json}  precision control, shipped survey files
  f32_ds_agn_{all,matched}.{h5,json}, f32_ds1pt_gal_all.{h5,json}
                              the blocker: dark_sirens on the shipped files
  equivalence_summary.json    every number quoted in this README
  pilot_n0_limit.json         the single-point log10n0 -> 0 scaling pilot that
                              chose -24 for the deep arm (run on the LOCAL shared
                              A100-80, internally consistent, not bit-comparable
                              with the production grids)

figs/
  fig_equivalence.{pdf,png}   per-cell |Delta logL| across the H0 grid (primary
                              arm) + the secondary arm's posterior shifts

logs/                         one stdout log per scan, plus the SLURM job log
data -> working/data/seed100
```

Reproduce (conda env `jax`, one free GPU):

```bash
sbatch scripts/submit_equivalence.sbatch     # 16 scans + the comparison
# or, on a local GPU:
./scripts/run_equivalence.sh && python scripts/compare_models.py
```

---

## Scope

Standalone. Nothing here is wired into `working/paper`; nothing in
`working/analyses/` was written (the matched event subsets are read only); darksirens is
read-only at `2b86a2d`; no git commits.
