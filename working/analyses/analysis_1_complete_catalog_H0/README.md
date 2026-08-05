# analysis_1_complete_catalog_H0 — what each complete catalog delivers alone

Round 1 of the campaign on `working/data/seed100` (**dataset v3 + D3, float64**).
Four independent single-tracer `H0` scans on the complete (magnitude-unlimited) GAL
and AGN survey pair, each run against both injection lanes, plus a matched-host
control for each catalog. Nothing here is a joint `(H0, f)` analysis; these are the
baselines the later rounds have to improve on.

> **THE DATASET WAS REDESIGNED AND FULLY REGENERATED — 2026-08-01.** The mock's
> measurement family is now **v3**, the literature-standard all-observable family
> (`working/data/DESIGN_PE.md`): the observed matched-filter SNR is data,
> `rho_obs = rho_opt(theta) + N(0, 1)`, detection is `rho_obs >= 8`, and **every**
> measurement width is `a_x * (8/rho_obs)` — a function of recorded data, never of a
> latent. The PE is the exact flat-prior posterior in
> `(ln Mc_det, ln q, rho, chieff, ra, dec)` with `p_pe = rho/(dL m1det q)`, and the
> luminosity distance is *derived* from the SNR rather than measured separately.
> The catalogs additionally **realise** the photo-z error the survey blocks declare
> (`z_obs = z + N(0, 3e-3 (1+z))`, **D3**). All five realisations — catalogs, events,
> surveys, injections — were regenerated and every scan below was rerun.
> Read **`CLOSURE.md` §16** for the closure verdict and `DESIGN_PE.md` for the
> design, its literature citations and every derivation.
>
> The previous (v2 post-`(b2)`/`(c2)`) scans are archived in `attic/results_v2postfix/`
> and `attic/figs_v2postfix/`; the pre-`(b2)`/`(c2)` ones remain in
> `attic/results_prefix2/`.
> `ATTRIBUTION.md` and `CLOSURE.md` §§1–15, and the `attr_*` products they cite,
> are unchanged and remain the diagnosis that motivated the redesign.

> **SUPERSEDED (v2 history, kept for the record).** Every scan in the previous
> revision of this README predated the two
> generator fixes of `CLOSURE.md`: **(c2)** the mass PE is now the exact flat-prior
> posterior of `obs ~ N(m, f m)`, and **(b2)** the RA measurement width now comes
> from the *observed* dec. The events stage of all five realisations was regenerated
> (detected sets bit-identical; catalogs, surveys and injections untouched) and every
> scan below was rerun. The numbers moved: `h0_gal_targeted` 60.10 → **64.12**, the
> matched-GAL control 62.79 → **64.74**, and the five-realisation matched-GAL mean
> offset **−9.43 ± 1.98 → −6.30 ± 1.28**. The scans of record are in `results/`; the
> pre-fix ones archived here are in `attic/results_prefix2/`. Read **`CLOSURE.md`** for the
> current state, including the exact host-galaxy sky oracle that rules out the
> nside-32 pixelisation as the remaining term. `ATTRIBUTION.md` and the `attr_*`
> products it cites are unchanged and remain the pre-fix diagnosis.

> **One estimator now carries every run.** darksirens ships two catalog
> likelihoods: `dark_sirens_complete`, which asserts the host is in the catalog,
> and the general `dark_sirens`, which models the galaxies the catalog is missing.
> This analysis used to run the first. It now runs the **second, driven into its
> complete-catalog limit at `log10n0 = -24`** — which is not an approximation:
> `experiment_model_equivalence` measured the two against each other on this
> dataset and found them **bit-for-bit identical, 201/201 cells, `max |Δ ln L| = 0`,
> in all four configurations**, with posterior medians agreeing to the last digit
> of float64. The complete-catalog result is therefore unchanged *as a number*, and
> the campaign no longer has to justify why its complete-catalog runs and its
> incomplete-catalog runs come from different code paths. See **Why `dark_sirens`
> at `log10n0 = -24`**.

> **Closure status — CLOSED (2026-08-01, `CLOSURE.md` §16).** On the v3 + D3
> dataset the matched-host controls — each catalog handed only the events it
> actually hosts — recover **GAL 68.96 (+1.22, truth inside the 68 % interval)** and
> **AGN 68.65 (+0.91, truth inside the 90 %)**, and over five independent
> realisations of the whole mock
>
> | | v2 (post-(b2)/(c2)) | **v3 + D3** |
> |---|---|---|
> | matched GAL | `−6.30 ± 1.28`, `t(4) = −4.92`, `p = 0.008`, truth in 68 % on **0/5** | **`+0.81 ± 0.62`, `t(4) = +1.32`, `p = 0.26`, truth in 68 % on 5/5** |
> | matched AGN | `+0.71 ± 0.20`, `t(4) = +3.51`, `p = 0.025` | **`+0.42 ± 0.47`, `t(4) = +0.89`, `p = 0.42`** |
>
> **Both matched controls now sit on truth.** The underlying per-event score
> identity closes with them: `(C − A)` in the mass channel went from 11.3σ (GAL) and
> 10.1σ (AGN) to 1.39σ and 0.95σ, and `(A − B)` from 6.9σ to 0.38σ on 1.53 M redrawn
> truths. The selection estimator's own common-mode Monte-Carlo error is carried:
> `± 0.23` (GAL) and `± 0.09 km s⁻¹ Mpc⁻¹` (AGN) per realisation. See `CLOSURE.md`
> §16 and `working/data/DESIGN_PE.md`.

**Every one of these four analyses is deliberately mis-specified.** Hosts were
planted from the mixture `(1-f) GAL + f AGN` with `f_AGN = 0.30`, realised as
**705 GAL-hosted and 295 AGN-hosted** events on seed 100 (v3; v2: 720/280). AGN are a *separate* GLASS tracer
painted on the same density field, not a subset of the galaxies, so the GAL-catalog
analysis is handed 295 events whose true hosts are absent from its catalog, and the
AGN-catalog analysis is handed 705. At `log10n0 = -24` the model asserts the host
**is** in the catalog. What these scans measure is the cost of that assertion, one
tracer at a time.

---

## Why `dark_sirens` at `log10n0 = -24`

Two separate things forced this, and only one of them is about physics.

**The physics.** `dark_sirens`'s numerator is
`logaddexp(N_obs log p_cat, log dN_miss)` with `dN_miss = (1 - C) dN_exp`
proportional to the modelled missing comoving density `n0`. Sending `n0 → 0`
switches the second branch off and leaves exactly `dark_sirens_complete`. How far
"→ 0" has to go was measured rather than assumed: at the `log10n0 = -12` the
campaign had been using, the completion term is small but **not off** — its residual
is at the float64 rounding floor on the dense GAL catalog (`3.6e-12` nats) but
reaches `4.1e-6` nats on the sparse AGN one, where 95 % of (pixel, kernel) cells
have no catalog support. At `log10n0 = -24` it is **exactly zero everywhere**: 804
of 804 grid cells bit-identical to `dark_sirens_complete`, and the four posterior
medians identical as float64 (`60.10557923193964`, `62.78539066237382`,
`99.7696699072826`, `67.40119359501826`). A paper that claims one nested likelihood
is cleanest if the nesting is exact, so `-24` is the value of record.

**The dtype.** `dark_sirens` could not be evaluated on the dataset as it was
shipped. Its observed-density KDE (`darksirens/redshift/completion.py::_kde_dndz_obs`)
builds the truncated-kernel mass in the *catalog's* storage dtype and clamps it at
`1e-300`, while the kernel itself is promoted to the package `zgrid`'s float64. The
survey blocks pad short pixel rows at `z = 100`, and `1e-300` is not representable
in float32 — so every padded slot evaluated `0 / 0 = NaN`, the `* real_gal` mask
could not remove it, every catalog row carrying any padding came back all-NaN, and
the survey-global field normalizer went NaN. The likelihood was `-inf` in every cell
of every grid, at every `log10n0`. `dark_sirens_complete` never reads that KDE,
which is why the float32 dataset was fine for a complete-catalog-only campaign and
stopped being fine the moment one likelihood had to carry every run.

**The dataset is therefore stored float64** (`working/data/generate_dataset.py`,
`CAT_DTYPE`), and seed 100 was regenerated end to end on 2026-07-31 —
catalogs, events, surveys, injections, validation, META. darksirens is untouched,
read-only at `2b86a2d`.

Regenerating changes the stored *numbers*, not the model. Catalog columns are no
longer rounded to float32, so the event host draw sees unrounded redshifts and
positions. The draw itself is unchanged — `host_type` and `host_index` are
bit-identical to the float32 run, and the realised split is the same **720 / 280**
— but every stored redshift, distance and PE sample moved by about `2e-8` relative
(`max 6.3e-8`, i.e. one float32 ulp), five orders below the catalog KDE width
`dz = 3e-3 (1+z)`. All nine generator validations pass. What that `1e-7` did to the
answers is measured in **Agreement with the `dark_sirens_complete` era**.

---

## The mock these numbers describe

Dataset v2 replaced the tracer amplitudes, the catalog edge and the field resolution
of the first pass (see `working/data/README.md` for the full specification).

| | |
|---|---|
| GAL number density | `1e-3 Mpc⁻³` comoving — the `L > 1.09 L*` bright sample of the GLADE-lineage B-band Schechter function (`phi* = 1.6e-2 h³`, `alpha = -1.07`, `M_B* = -20.47`, `h = 0.7`; GLADE [arXiv:1804.05709](https://arxiv.org/abs/1804.05709), GLADE+ [arXiv:2110.06184](https://arxiv.org/abs/2110.06184)). **151 179 870** objects |
| AGN number density | `1e-5 Mpc⁻³` comoving — the luminous class `log10 L_X(2–10 keV) ≳ 43.7` of the integrated Swift-BAT/BASS X-ray luminosity function (Ananna et al.; BASS DR2 lineage). **1 514 567** objects. Ratio 99.8 |
| catalog edge | `z_max = 1.0` (realised last shell edge 1.0016), constant comoving density to 0.1% on the plateau `0.0457 ≤ z ≤ 0.9230` (realised `1.00009e-3` and `1.00183e-5`) |
| edge margin | the events' PE support, mapped through `z(dL; H0)` at every `H0` in the scanned range, reaches **0.655** at worst (`H0 = 100`) against a `0.7 z_max = 0.700` bar |
| density field | GLASS lognormal, `nside = 128`, `lmax = 256`, 200 Mpc shells, `b_GAL = 1.2`, `b_AGN = 2.0`. The planted contrast is **measured**: `b_AGN/b_GAL = 1.6845 ± 0.0090` against a planted 1.6667 (2.0σ), and 76σ away from "no contrast" |
| events | 1000 detected, `z_median = 0.132`, horizon `z_max = 0.3565`; at most 2 events share an AGN host |
| storage | float64 (`CAT_DTYPE`). Complete GAL catalog 4.7 GB, complete GAL survey block 1.65 GB; the whole seed is 7.9 GB (2.4× the float32 tree it replaced) |

The luminosity-cut arithmetic is worth writing down because the nominal label does
not survive it. On the GLADE Schechter function `n(> x L*) = phi* Γ(alpha+1, x)`
with `phi* = 5.488e-3 Mpc⁻³`, so the often-quoted `x = 0.25` cut gives
`Γ(-0.07, 0.25) = 1.08383`, i.e. `n = 5.948e-3 Mpc⁻³` — **5.9× the intended
`1e-3`**, and ~9.8e8 galaxies inside `z ≤ 1`, several times the storage budget. The
cut was therefore solved *from* the density on the same luminosity function:
`Γ(-0.07, x) = 0.182216 ⇒ x = 1.0908`, `M_B < -20.564`. That is the classic `L*`
bright-galaxy sample, which is what a `1e-3 Mpc⁻³` host catalog physically is.

---

## Configuration

| | |
|---|---|
| model | **`dark_sirens`**, K = 1, `catalog_sky_weighting = field`, `use_lss` off |
| completeness | **`log10n0 = -24`** — the complete-catalog limit, bitwise equal to `dark_sirens_complete` (see above) |
| nuisances | all fixed: `delta = 0`, `sigma_kde = 0`, `b_miss = 1` (inert with `use_lss` off). Free labels are `["H0", "log10n0", "delta", "sigma_kde"]`; only `H0` is scanned |
| population | powerlaw+peak, fixed at the mock's own fiducial (`fix_population`) |
| cosmology | `Om0` pinned at 0.3075, `w0 = -1`, `wa = 0`; `H0` free |
| grid | `H0` ∈ [50, 100], 201 points (Δ = 0.25); truth **67.74** |
| survey | `data/seed100/surveys/survey_{gal,agn}_complete_ns32.h5` (nside 32, float64) |
| events | `data/seed100/events/events.h5` — 1000 events × 2000 PE samples |
| injections | `injections_targeted.h5` (`Ndraw = 1.5e8`) and `injections_popuni.h5` (`Ndraw = 4.0e8`) |
| guard | `--selection_neff_guard hard --max_likelihood_variance 1e6` — the campaign convention: the legacy `N_eff > 5 N_obs` floor, total-variance criterion made inert |
| posterior | flat prior on `H0`, trapezoid marginal; equal-tailed CIs |
| numerics | windowed catalog KDE `W = 4096` (`n_sigma = 8`) for the GAL survey; `sel_batch_size = 50000`, `pe_event_block = 25` |
| darksirens | `/hildafs/projects/phy230014p/magana/src/darksirens` @ `2b86a2d` (read-only) |
| device | one **A100-40** on `twig` (SLURM `TWIG-GPU`, job 1058907), `XLA_PYTHON_CLIENT_PREALLOCATE=false` |

Two numerical settings are checked rather than assumed, and both were re-measured on
the float64 survey files.

* **The catalog-KDE window.** The complete GAL survey block is `(12288, 14569)` —
  a nside-32 pixel holds ~12 300 galaxies — so darksirens' windowed KDE evaluator
  must be given a window at least as large as the number of galaxies inside the
  kernel support. `recommended_kde_window` returns **3410** at `n_sigma = 8` and the
  scans' own `sigma_kde = 0`; the scans use `W = 4096`
  (`scripts/kde_window_check.py` → `results/kde_window.json`). That is the same 3410
  the float32 blocks gave — the dtype change moved no galaxy across a pixel or a
  kernel — and the survey block shapes are unchanged, `(12288, 14569)` and
  `(12288, 178)`, 0.00 % empty pixels in both.
* **Reduction blocking.** A single pass over 2.1e6 injections at `W = 4096` does not
  fit in one shot; the scans block both reductions (`sel_batch_size`,
  `pe_event_block`). `twig` carries A100-**40**s, so both are half the setting an
  A100-80 allows. Identical for every scan here, so it cannot enter any comparison
  between them.

---

## Injection sizing

The campaign convention makes the guard threshold a flat `5 N_obs = 5000` across the
whole grid, so the requirement is simply that the selection integral's `N_eff` clear
that everywhere, with margin. The target set here was **min-over-grid `N_eff` ≥ 10 000
(2× the floor)**.

The sizing was measured, not guessed. A 2e7-proposal pilot of each lane was scanned
at 9 points spanning `H0 ∈ [50, 100]` with the per-cell guard record, giving

| configuration | min `N_eff` at `Ndraw = 2e7` | `Ndraw` for 10 000 |
|---|---|---|
| GAL + targeted | 53 210 | 3.8e6 |
| GAL + popuni | 25 020 | 8.0e6 |
| AGN + targeted | 28 480 | 7.0e6 |
| AGN + popuni | **1 933** | **1.03e8** |

`N_eff` is linear in `Ndraw`, so `AGN + popuni` is the binding configuration and it
sets the popuni lane. The production sets are built at **`Ndraw = 1.5e8` (targeted)**
and **`Ndraw = 4.0e8` (popuni)** — 3.9× the binding requirement — yielding
2 095 518 and 1 175 596 detected rows (193 MB and 101 MB on disk). The float64
regeneration rebuilt both at the same `Ndraw`; the popuni lane came back with
exactly the same 1 175 596 rows and the targeted lane with 125 more (2 095 518
against 2 095 393), which is the `1e-7` perturbation moving a handful of proposals
across the detection threshold. All four configurations are guard-safe; none was
dropped.

Two changes made that possible, and both matter more than the raw draw count.

**(1) The proposal shape.** v1's catalog-targeted branch drew `z ~ TN(z_j, sigma_j)`
— it planted injections *on* the catalog kernels of the fiducial cosmology. The
likelihood re-reads a stored injection at trial `H0` as the redshift `z'` with
`dL_fid(z') = dL_fid(z)·H0/H0_FID`, so that branch overlapped the catalog prior only
near 67.74, and its `N_eff` collapsed by three orders of magnitude across the scan.
v2 replaces it with a per-host **uniform box**

```
[L_j, U_j] = [ max(0, R_LO (z_j − 4 sigma_j)),  min(0.5, R_HI (z_j + 4 sigma_j)) ]
R_LO = H0_FID/100 = 0.6774      R_HI = H0_FID/50 = 1.3548
```

whose image under *every* trial `H0` in `[50, 100]` still contains the host's kernel.
The density is a flat mixture, hence exact in closed form: the generator's V6 check
recomputes `pdraw` from the flat host list on 200 rows per lane and agrees to
better than `1e-14` relative. Hosts deeper than `z = 0.5` are dropped and the box is
capped at `z = 0.5`; neither can ever be detected, and the population/uniform
branches keep full support, so no hole is opened in `pdraw`.

**(2) The dataset itself.** At `n_AGN = 1e-5 Mpc⁻³` every nside-32 pixel is
occupied (v1: 38.5% empty), which is why the *untargeted* lane became viable at all
— v1's AGN+popuni sat at `N_eff = 25–130` and would have needed ~5e10 draws.

### Realised convergence

| # | tag | min `N_eff` over the grid | × threshold | rejected cells |
|---|---|---|---|---|
| 1 | `h0_gal_targeted` | 395 349 | 79.1 | **0 / 201** |
| 2 | `h0_gal_popuni` | 494 877 | 99.0 | **0 / 201** |
| 3 | `h0_agn_targeted` | 216 057 | 43.2 | **0 / 201** |
| 4 | `h0_agn_popuni` | 32 979 | 6.6 | **0 / 201** |
| — | `ctrl_gal_matched` | 395 349 | 109.8 | **0 / 201** |
| — | `ctrl_agn_matched` | 216 057 | 154.3 | **0 / 201** |

**Zero cells were rejected in any production scan or control, at any `H0` in
[50, 100].** No result in this directory comes from a guard bypass; the driver's
bypass pathway has been deleted (`scripts/scan_h0f.py`, `parse_args`). The binding
configuration, `AGN + popuni`, clears the **min-over-grid `N_eff` ≥ 10 000** sizing
target by 3.3×; every other configuration clears it by 20–50×. At `H0 = 67.74` the
per-configuration diagnostics (`results/guard_h0_*.json`) read

| tag | `N_eff` at truth | Σ σ²_PE | max per-event σ²_i | σ²_total | verdict |
|---|---|---|---|---|---|
| `h0_gal_targeted` | 477 688 | 1.040 | 0.054 | 3.13 | admitted (95.5×) |
| `h0_gal_popuni` | 569 300 | 1.040 | 0.054 | 2.80 | admitted (113.9×) |
| `h0_agn_targeted` | 353 541 | 35.10 | 0.9995 | 37.92 | admitted (70.7×) |
| `h0_agn_popuni` | 53 683 | 35.10 | 0.9995 | 53.72 | admitted (10.7×) |

For scale: on the same quantities the first pass reported `N_eff` of 4 618 / 5 962 /
68 729 / 55 against the same 5 000 floor, and `Σ σ²_PE` of 66.3 (GAL) and 675.7
(AGN) with 676 of the 1000 events pinned at darksirens' per-event variance ceiling.
Under v2 the GAL configurations sit at `Σ σ²_PE ≈ 1.04`, i.e. within a few percent
of admissible even under darksirens' own default `max_likelihood_variance = 1.0`.

---

## Results — the four production scans

Medians with equal-tailed 68% CIs. `offset` is median − 67.74.

| # | tag | median ± 68% | offset | 90% CI | MAP |
|---|---|---|---|---|---|
| 1 | `h0_gal_targeted` | 60.10 (+2.17 / −2.30) | **−7.64** | [56.23, 63.69] | 60.25 |
| 2 | `h0_gal_popuni` | 60.15 (+2.29 / −2.26) | **−7.59** | [56.12, 64.02] | 60.00 |
| 3 | `h0_agn_targeted` | 99.77 (+0.16 / −0.39) | **+32.03** † | [98.99, 99.98] | 100.00 |
| 4 | `h0_agn_popuni` | 99.83 (+0.11 / −0.21) | **+32.09** † | [99.41, 99.98] | 100.00 |

† **Railed against the top of the prior range, not a measurement of `H0`.** Both AGN
log-likelihoods rise monotonically across the whole grid — at truth they sit 490 nats
(targeted) and 489 (popuni) below their value at `H0 = 100`, and their MAP is the
last grid cell. The interior peak lies far outside the scanned range; a range
diagnostic run to `H0 = 250` in the previous era located it near `H0 ≈ 122`
(targeted) and `≈ 114` (popuni) (`attic/results_dsc_attic/range_agn_*.{h5,json}`), and it
has not been repeated because the only thing it establishes — that the peak is
outside [50, 100] — is already established by the monotone rise.
`results/h0_single_tracer.json` therefore carries `null` for the AGN interval and
width.

Truth is outside the 90% CI in all four cases.

Figures: `figs/fig_h0_recovery.{pdf,png}` (production posteriors and the matched-host
controls, truth marked), `figs/fig_guard.{pdf,png}` (`N_eff` against threshold, and
the per-cell admit mask).

Cost, on one A100-40 at `sel_batch_size = 50000`, `pe_event_block = 25`: 3.78 s/eval
(GAL, 1000 events), 3.68 (GAL, 720), 0.19 (AGN, 1000), 0.14 (AGN, 280) — 44 min for
all six grids. `dark_sirens` at `log10n0 = -24` costs nothing measurable over
`dark_sirens_complete`: the completion branch is evaluated and then contributes zero.

### Lane cross-check

The two injection lanes are the same detection rule with different proposals, so
they must give the same answer.

* **GAL: 60.098 vs 60.150 — a 0.052 km s⁻¹ Mpc⁻¹ disagreement**, i.e. 2% of one
  68% half-width.
* **AGN: 99.770 vs 99.833 — 0.063**, likewise negligible against the widths, though
  both are railed.

For comparison the first pass disagreed by 2.88 on GAL against ±0.6 half-widths and
could not compare the AGN lanes at all. The selection integral no longer sets any
digit of these answers.

---

## Matched-host control

Because the offsets in rows 1–4 are large enough to be a configuration fault rather
than a mis-specification, each catalog was also run on **only the events it actually
hosts**, everything else identical. Conditioning on the host-type branch gives
exactly a draw from that tracer's own catalog prior, and `mu(θ)` never saw the host
type, so this is the matched analysis.

| control | events | median ± 68% | offset | 90% CI | rejected |
|---|---|---|---|---|---|
| `ctrl_gal_matched` | 720 GAL-hosted | 62.79 (+2.35 / −2.52) | **−4.95** | [58.46, 66.57] | 0 / 201 |
| `ctrl_agn_matched` | 280 AGN-hosted | **67.39 (+0.85 / −0.88)** | **−0.35** | [65.85, 68.83] | 0 / 201 |

**The AGN control recovers truth**, at −0.35 with a ±0.86 half-width — truth inside
the 68% interval — on 280 events. So the pipeline, the survey files, the events
file, the injections and the estimator are sound, and the +32 of rows 3–4 is
produced entirely by the 720 events whose hosts the AGN catalog does not contain.

**The GAL control does not**, landing 4.95 low with a ±2.4 half-width — about 2σ,
with truth outside the 90% interval. On one realisation that is suggestive rather
than established; the five-realisation measurement that settles it as a systematic
is in **Closure across realisations**, and predates this estimator.

Note also what the two controls say about *precision*: the sparse tracer, on a
quarter as many events, delivers a 2.8× tighter `H0` than the dense one.

---

## Agreement with the `dark_sirens_complete` era

The previous analysis of record — `dark_sirens_complete` on the float32 dataset —
is preserved in `attic/results_dsc_attic/`. Two things changed at once between it and the
table above, and they are not the same size.

**The estimator did not change the answer at all.** `experiment_model_equivalence`
put the two likelihoods on identical inputs and found them bit-for-bit identical at
`log10n0 = -24` — 201/201 cells, `max |Δ ln L| = 0`, in all four configurations, with
posterior medians equal to the last float64 digit. Whatever the two eras differ by,
it is not the model.

**The data changed at the `1e-7` level**, and that is what the difference below is.
Everything is computed cell by cell by `attic/scripts_onhold/compare_dsc_attic.py` →
`results/vs_dsc_attic.json`.

| configuration | median then | median now | Δ | Δ in 68% half-widths | max \|Δ ln L\| over the grid | min `N_eff` then → now |
|---|---|---|---|---|---|---|
| `h0_gal_targeted` | 60.105579 | 60.097931 | −7.6e-03 | −0.0034 | 0.111 | 395 323 → 395 349 |
| `h0_gal_popuni` | 60.152432 | 60.149821 | −2.6e-03 | −0.0011 | 0.0059 | 494 872 → 494 877 |
| `h0_agn_targeted` | 99.769670 | 99.769681 | **+1.1e-05** | +4e-05 | 0.275 | 216 034 → 216 057 |
| `h0_agn_popuni` | 99.832554 | 99.832556 | **+2.6e-06** | +2e-05 | 0.0173 | 32 979 → 32 979 |
| `ctrl_gal_matched` | 62.785391 | 62.789051 | +3.7e-03 | +0.0015 | 0.0800 | 395 323 → 395 349 |
| `ctrl_agn_matched` | 67.401194 | 67.391388 | −9.8e-03 | −0.0113 | 0.0770 | 216 034 → 216 057 |

**The largest posterior shift anywhere is 0.0098 km s⁻¹ Mpc⁻¹, on half-widths of
0.86 to 2.5 — 1.1% of one half-width at worst, and 0.3% or less on five of the six.**
No cell is bit-identical, which is expected: the arithmetic is being done on
different numbers (and, separately, on a different GPU with different reduction
blocking than the era's A100-80 run). Every conclusion in this directory is
unchanged, digit for digit, at the precision anyone quotes.

Two rows deserve a word. The AGN production scans move by `1e-5` and `3e-6`, three
orders below the GAL ones, because they are railed: the posterior is pinned against
the edge of the prior and a perturbation of the likelihood barely moves where the
mass sits. And `max |Δ ln L| = 0.275` on `h0_agn_targeted` is the largest
log-likelihood excursion in the table while its median is the most stable — a
reminder that the two are not the same measurement. Zero cells are rejected in every
scan in both eras.

---

## Closure across realisations

> **From the `dark_sirens_complete` era, on the float32 dataset — superseded
> pending rerun.** Everything in this section was produced by the estimator and
> the dataset this analysis has now replaced. The estimator change is inert (bitwise
> equality), but the four extra realisations have not yet been regenerated in
> float64 and the controls have not been re-run on them, so no number below has been
> reproduced under the analysis of record. The rerun is deliberately out of scope
> here. Seeds 101, 102, 103 and 105 **have** been regenerated in float64
> (`working/data/seed10{1,2,3,5}`); what is missing is the twelve control scans and
> the aggregation. The per-realisation host-type subsets under
> `data_derived/seeds/` are also float32-era and must be rebuilt first.

The question this section answers is the one the campaign cannot move past: **does
each complete catalog return the true `H0` when it is handed its own hosts?** One
realisation cannot answer it, because the matched control's own 68% interval is
±2.4 (GAL) and ±0.9 (AGN) and a 2σ excursion is not rare. So the whole mock — GLASS
density field, both catalogs, the 1000 events, both injection campaigns — was
regenerated from scratch on new seeds and both controls re-run on each. Five
regenerations were run (101–105); one, seed 104, failed its own validation and was
discarded unused (see **A note on seed 104**), leaving four.

**Five realisations: seeds 100, 101, 102, 103, 105.** Every one passes all nine
validation checks. Everything else is held fixed: `dark_sirens_complete`, K = 1,
field weighting, `H0` ∈ [50, 100] × 201, the targeted injection lane, `W = 4096`,
the campaign guard convention. `recommended_kde_window` was re-measured on every
seed's GAL survey — **3410 / 3372 / 3426 / 3457 / 3291** at `n_sigma = 8` for seeds
100 / 101 / 102 / 103 / 105 — so the `W = 4096` window clears the requirement on all
of them and the catalog KDE is not truncated anywhere. **No scan rejected a single
cell**: 0/201 everywhere, min `N_eff` ≈ 4.0e5 (GAL) and 2.2e5 (AGN), i.e. 80–110×
the threshold.

### The two cases

`offset` is median − 67.74.

| | seed | events | median ± 68% | offset | truth in 68% / 90% |
|---|---|---|---|---|---|
| **GAL** | 100 | 720 | 62.79 (+2.34 / −2.51) | −4.95 | no / no |
| | 101 | 661 | 62.14 (+2.06 / −1.97) | −5.60 | no / no |
| | 102 | 698 | **51.95** (+2.78 / −1.43) | **−15.79** † | no / no |
| | 103 | 735 | 58.25 (+2.65 / −2.12) | −9.49 | no / no |
| | 105 | 699 | 56.45 (+2.99 / −2.65) | −11.29 | no / no |
| | | | **mean offset** | **−9.43 ± 1.98** | **t(4) = −4.75, p = 0.009** |
| **AGN** | 100 | 280 | 67.40 (+0.84 / −0.88) | −0.34 | yes / yes |
| | 101 | 339 | 69.29 (+1.19 / −1.46) | +1.55 | no / yes |
| | 102 | 302 | 67.65 (+1.00 / −1.02) | −0.09 | yes / yes |
| | 103 | 265 | 67.35 (+1.05 / −1.05) | −0.39 | yes / yes |
| | 105 | 301 | 67.28 (+0.93 / −0.87) | −0.46 | yes / yes |
| | | | **mean offset** | **+0.05 ± 0.38** | **t(4) = +0.14, p = 0.90** |

† **Railed.** On seed 102 the GAL likelihood has no interior maximum: its MAP sits
exactly on `H0 = 50`, the bottom of the scanned range, with 16% of the posterior
mass in the lowest grid cell. The quoted median is where the prior was cut, not
where the likelihood peaked, so −15.79 is a **lower bound on the magnitude** of
that realisation's offset. It is kept in the mean because dropping it could only
bias the result *towards* truth; excluding it anyway gives −7.84 ± 1.53,
**t(3) = −5.13** — the verdict does not depend on it.

### Verdict

* **AGN catalog — CLOSES.** Mean offset **+0.05 ± 0.38**, 0.14σ from truth. Truth
  falls inside the 68% interval on 4 of 5 realisations and inside the 90% on 5 of 5,
  which is what a calibrated interval is supposed to do. The per-realisation scatter
  (sd 0.85) is **0.83×** the mean quoted half-width, so the error bar is honest,
  very slightly conservative.
* **GAL catalog — DOES NOT CLOSE.** Mean offset **−9.43 ± 1.98**, i.e. **4.8σ low**,
  p = 0.009 on 4 degrees of freedom. Truth is outside the 90% interval in **5 of 5**
  realisations, all five offsets are negative, and none is smaller than −4.9. This
  is a systematic, not realisation noise.

Two further facts about the GAL failure, both from the numbers above. Its
realisation-to-realisation scatter (sd 4.43) is **1.89×** the mean quoted half-width
(2.35), so the posterior width does not even describe how much the answer moves
between mocks — there is extra variance beyond it. And the failure is not a
small pedestal that grows: the offsets span −4.9 to beyond −15.8, and on one
realisation in five the answer leaves the prior range entirely.

### What the diagnostics rule out

Three cheap checks were run on seed 100 before generating new realisations, and
each closes off a candidate explanation.

**The injection lane is not responsible.** Both controls were re-run on the second
(popuni) selection campaign — the same detection rule under a different proposal.

| control | targeted | popuni | difference | as a fraction of one 68% half-width |
|---|---|---|---|---|
| `ctrl_gal_matched` | 62.785 | 63.101 | +0.32 | 13% |
| `ctrl_agn_matched` | 67.401 | 67.169 | −0.23 | 27% |

Both disagreements are small against the widths, and both lanes put GAL ≈ 5 low and
AGN on truth. (The AGN popuni lane is the campaign's least-converged selection
integral — min `N_eff` 3.3e4 against 2.2e5 for targeted — which is the natural
reading of its slightly larger fractional shift.) A −9.4 offset is three decades
larger than anything the selection integral is doing.

**The likelihood width is not underestimated.** Each host-type event set was cut
into 8 disjoint contiguous blocks and scanned separately. Events are stored
`as_drawn`, so each block is an unbiased sub-realisation.

| | blocks | events/block | sd of block medians | mean block half-width | ratio |
|---|---|---|---|---|---|
| GAL | 8 (3 railed) | 90 | 4.87 | 6.16 | 0.79 |
| AGN | 8 (0 railed) | 35 | 3.73 | 3.58 | 1.04 |

Scaling the block scatter to the full set gives an empirical standard error of 2.18
against the GAL control's quoted ±2.42 (ratio 0.90) and 1.32 against the AGN's
±0.86 (ratio 1.53; at 35 events the AGN posterior is visibly non-Gaussian and
`sqrt(N)` scaling from 35 to 280 is not reliable, which is why the block-level ratio
of 1.04 is the trustworthy statement). **Resampling events within a realisation
moves the answer by about what the likelihood says it should.** The GAL error bar is
therefore not too small — which makes the −9.4 mean offset worse, not better, and
localises the extra between-mock variance in the *catalog* realisation rather than
in the events.

Note also that 3 of the 8 GAL blocks railed at 90 events while 0 of the 8 AGN blocks
railed at 35. The dense catalog loses the interior maximum first, even though it has
2.6× more events per block.

### Scope of this result

This is a closure statement, not a mechanism. The diagnostics above say what it is
**not** — not the injection lane, not an understated width, not the guard, not one
unlucky mock. Identifying what it **is** is the next piece of work and is
deliberately not attempted here. The obvious places to look, in the order the
evidence points: the dense catalog's redshift prior inside the GW horizon (the GAL
`dN/dz` is smooth on the KDE scale, so the catalog carries little localising
information and whatever residual gradient it has is not fought by anything), the
catalog-edge and low-`z` density ramp, and the `field` sky-weighting normalisation
on a catalog whose per-pixel counts are ~1000 inside the horizon. The AGN control
closing at 0.14σ under the identical estimator, survey builder, events file and
injection sets is the strongest constraint on any candidate: whatever it is, it has
to switch off for the sparse tracer.

### A note on seed 104

Seed 104 was generated and **failed** its own validation
(`V6_injections_and_detection_closure`): the two injection lanes' `P_det(z)` curves
disagreed at `max_binomial_z = 7.53` against a `< 6.0` gate. It is not used, the
gate was not relaxed, and it has **not** been regenerated in float64; seed 105 was
generated as the replacement.

The failure is worth recording because it is a property of the check rather than of
the dataset. In every seed the extremal bin is one holding a **single** detected
targeted injection near the detection horizon (`z ≈ 0.28`, `P_det ≈ 2e-4`): seeds
100/101/102/103 gave 4.00/3.43/5.69/2.69 from `n_det` = 1, 8, 1, 1. The check applies
a Gaussian binomial error to a one-count Poisson bin, so its tail is far heavier than
the nominal σ scale and the `< 6.0` bar is not the ~6σ event it reads as. Seed 104's
*end-to-end* closure — the physically meaningful quantity, predicted against measured
event detection fraction — is **0.088σ, the best of the five**, and its `pdraw`
recomputes exactly (8e-15). If the check is revisited, the fix is a floor on the
detected counts entering the comparison (the mask currently floors only the
*proposed* counts, at 2000), not a looser threshold.

Figure: `attic/figs_dsc_attic/fig_closure_seeds.{pdf,png}` — per-realisation medians with
68% intervals for both cases, truth, and the mean ± standard error band.

---

## Reading

The single-tracer answer is set by how much redshift information the catalog
actually carries inside the GW horizon, and the two tracers sit on opposite sides of
that. Inside `z ≤ 0.3565` the GAL catalog holds 12.6 million galaxies — 1023 per
nside-32 pixel, ~4.6 per pixel per KDE width at the median event redshift, only 1.7%
of (pixel, kernel) cells empty — so its redshift prior is effectively the smooth
`dN/dz`, and the 280 events whose hosts are missing are simply absorbed into it; the
result is a wide posterior (±2.2) displaced modestly low, and most of that
displacement is already present in the matched control. The AGN catalog holds
125 681 objects in the same volume — 10 per pixel, **0.047 per pixel per KDE width,
95% of cells empty** — so it is a comb of narrow spikes, and the model's assertion
that the host is in the catalog is a hard constraint. Each of the 720 orphan events
must be placed on *some* AGN inside its sky patch, and the number of reachable
spikes grows steeply with redshift (0.047 per kernel at `z = 0.13`, 0.229 at
`z = 0.30`); raising `H0` maps a fixed `d_L` to higher `z`, so the likelihood buys
support by climbing that gradient and never turns over inside the scanned range.
That is why the *same* mis-specification — 30% of events hosted by the other tracer
— costs a couple of km s⁻¹ Mpc⁻¹ on the dense catalog and drives the sparse one
clean out of the prior. It is also why the sparse catalog is the more powerful one
when it *is* right: on its own 280 events the AGN catalog recovers `H0` to ±0.9,
against ±2.4 for the GAL catalog on 720. A single-tracer complete-catalog analysis
of a two-tracer universe is therefore not merely imprecise but confidently wrong,
and the failure is worst exactly where the constraining power is best. This sets the
bar for the joint `(H0, f)` analysis, which has the mixture weight available to
explain those events instead.

Two caveats on magnitude. The effect scales with the sparseness of the catalog's
redshift support inside the horizon, so quoting "+32" as *the* AGN mis-specification
cost is meaningless — it is a lower bound set by the prior range. And the GAL row's
−7.64 cannot be read as mis-specification at all: the matched GAL control is itself
−4.95 on this realisation and −9.4 ± 2.0 across the five measured in the previous
era (**Closure across realisations**), so the dense catalog fails to return truth
even when every event it is given is one of its own. Until that is understood, only
the AGN side of this table measures a mis-specification cost.

One more thing this round changes, which is about what comes next rather than about
these numbers. The completion term is now *present in the estimator and switched off
by a parameter*, not absent from the code path. Turning it on at the tracer's true
density is a one-flag change from here, and `experiment_model_equivalence`'s
characterization arm already indicates it does real work: given a missing-host
channel at `n_AGN = 1e-5 Mpc⁻³`, the AGN production configuration's railing
disappears (median 66.7 against a truth of 67.74, in place of the +32 above). That
is a different model, not a limit of this one, and it belongs to the next round —
but it is now one flag away rather than one estimator away.

---
## Files

Reorganised 2026-08-01. The top level now holds **only the v3 analysis of record**;
the closed diagnostic campaign is under `diagnostics/`, and the superseded eras are
under `attic/`. Nothing was deleted. `results/h0_single_tracer.json` — the only file
`working/paper` reads from this analysis — is at its original path, unchanged.

```
analysis_1_complete_catalog_H0/
  README.md            this file
  CLOSURE.md           the matched-host closure investigation, through the v3 redesign
  ATTRIBUTION.md       attribution of the per-event score residual r
  PROBES.md            the four mechanism probes
  diagnostics/         THE CLOSED INVESTIGATION -- see diagnostics/INDEX.md
  attic/               the superseded eras -- see attic/ATTIC.md

  scripts/             THE ANALYSIS OF RECORD, and nothing else
    scan_h0f.py                 grid driver, adapted from
                                experiments/experiment_matched_mock/scripts/scan_h0f.py.
                                MERGE_SHA pinned to 2b86a2d; --guard_record (default
                                on) writes a per-cell (Neff, Sigma sigma^2_PE,
                                threshold, verdict) record; --kde_window configures the
                                windowed catalog-KDE evaluator; --sel_batch_size /
                                --pe_event_block bound the GPU working set.
    diag_variance_guard.py      single-point guard/variance diagnostic at H0 = truth
                                (same memory and window knobs).
    kde_window_check.py         sizes W from darksirens' recommended_kde_window on THIS
                                dataset's survey files -> results/kde_window.json.
                                Re-run whenever the surveys are rebuilt.
    build_hosttype_subset.py    splits events.h5 by host_type (the matched control).
    build_single_tracer.py      the two production measurements ->
                                results/h0_single_tracer.json, the ONLY file
                                working/paper reads from this analysis.
    aggregate_closure.py        collects the realisations into results/closure_seeds.json
                                and prints the tables.
    make_figures.py             fig_h0_recovery, fig_guard, fig_closure_seeds.
    fig_closure_after_fix.py    the before/after closure strip; --fig_tag fig_closure_v3
                                and --before_dir attic/results_v2postfix produce the v3
                                record figure.
    run_scans.sh                THE SIX SCANS OF RECORD (four production configurations
                                + the two matched-host controls).
    run_guard_diag.sh           the four guard diagnostics at H0 = truth.
    run_seed_controls.sh        the matched controls on the further realisations.
    run_v3_analysis.sh          THE v3 ANALYSIS OF RECORD, end to end: window check,
                                subsets, six scans, four controls, guards, values,
                                closure table, figures.
    submit_v3_analysis.sbatch   run_v3_analysis.sh as one serial GPU job.
    submit_v3_controls.sbatch   the four remaining matched controls as their own job.
    submit_scans.sbatch         run_scans.sh + run_guard_diag.sh + build_single_tracer.py
                                + make_figures.py, one serial GPU job.
    submit_seed_controls.sbatch run_seed_controls.sh as its own job.

  results/             the v3 scans of record (37 files)
    h0_{gal,agn}_{targeted,popuni}.{h5,json}   THE FOUR PRODUCTION SCANS.
        h5:   H0_grid, log_likelihood, guard/{rejected, Neff, pe_variance_sum,
              selection_variance_N2_over_Neff, sigma2_total, threshold, passes,
              passes_legacy_floor, legacy_floor_5N}, full provenance attrs.
        json: median / ci68 / ci90 / map / truth flags, n_rejected, and guard.cells[]
              -- one record per grid cell.
    ctrl_{gal,agn}_matched.{h5,json}           the matched-host controls, seed 100.
    ctrl_{gal,agn}_matched_s{101,102,103,105}.{h5,json}   the same, per realisation.
    guard_h0_{gal,agn}_{targeted,popuni}.json  guard diagnostic at H0 = 67.74.
    closure_seeds.json                         the five-realisation closure table.
    closure_v3.json                            the v2 -> v3 before/after comparison.
    kde_window.json                            the measured window requirement.
    v3_curvature.json                          per-event d2 ln L/dH0^2 on the v3 controls.
    h0_single_tracer.json                      THE PAPER-FACING SUMMARY.

  figs/                the four record figures
    fig_h0_recovery.{pdf,png}    the four posteriors and the two matched-host controls.
    fig_guard.{pdf,png}          Neff/threshold and the per-cell admit mask.
    fig_closure_seeds.{pdf,png}  the five-realisation closure.
    fig_closure_v3.{pdf,png}     the v2 measurement family against v3.

  logs/                one stdout log per record run, plus the SLURM job logs (26 files)

  data_derived/
    events_{gal,agn}_hosted.h5   the 705 / 295 host-type subsets (control input),
                                 rebuilt from the v3 events.
    seeds -> bulk                the per-realisation subsets and the derived surveys.

  diagnostics/         THE CLOSED INVESTIGATION, by stage.
    INDEX.md           maps every moved file old path -> new path, with a one-line
                       description and the CLOSURE/ATTRIBUTION/PROBES section that
                       cites it.  Read this first.
    probes/            the four mechanism probes + the survey-resolution (nside) study
                       (26 files; the probe 1-4 numeric outputs are in attic/).
    attribution/       the per-event score residual r, the two quadrature oracles, the
                       selection-integral sweep, the chi_eff clip and the host-
                       acceptance convention, and the v2 post-fix closure accounting
                       (308 files).
    endgame/           the (A-B)/(C-A) split, the truncation audit, the declared-
                       photo-z-kernel scan and the v3 pilot gate (108 files).
                       Each stage mirrors the root layout (scripts/, results/, figs/,
                       logs/, data_derived), so a stage's scripts resolve their own
                       products when run from that stage's directory.

  attic/               the superseded eras.  ATTIC.md explains each.
    results_dsc_attic/, figs_dsc_attic/, logs_dsc_attic/
                       the dark_sirens_complete / float32 era, incl. every probe 1-4
                       numeric output and figure, the jackknife blocks and the closure
                       runs of that era.
    scripts_superseded/  the three run scripts as they stood in that era.
    results_prefix2/, figs_prefix2/       the pre-(b2)/(c2) scans of record.
    results_v2postfix/, figs_v2postfix/   the v2 post-(b2)/(c2) scans of record
                       (the "before" arm of fig_closure_v3).
    logs_v2postfix/    the pre-v3 logs of the record tools.
    scripts_onhold/    compare_dsc_attic.py, run_closure_diag.sh, build_event_blocks.py
                       -- studies on hold, all reading archived inputs.
    data_derived_v2/blocks/   the float32-era disjoint event blocks.
    vs_dsc_attic.json  this analysis against the dark_sirens_complete era, cell by cell.
    figs_ipynb_checkpoints/, scripts_pycache/   editor and interpreter cruft.
```

The datasets themselves live on the bulk filesystem,
`/hildafs/projects/phy220048p/magana/gws-agn-data-v3/seed<N>`, reached through the
`working/data/seed<N>` symlinks. Seed 104 is present there but deliberately
unlinked — it failed validation, and it was not regenerated.

Reproduce, in order (conda env `jax`, GPU free):

```bash
# the datasets (working/data), then promote them
cd working/data
for S in 100 101 102 103 105; do ./run_v3_seed.sh $S; done
bash promote_v3.sh check && bash promote_v3.sh promote

# then this analysis, on one GPU: window check, subsets, six scans, four controls,
# guards, the paper-facing values, the closure table and the four record figures
cd ../analyses/analysis_1_complete_catalog_H0
sbatch scripts/submit_v3_analysis.sbatch
```

Or step by step, which is what `run_v3_analysis.sh` does:

```bash
python scripts/kde_window_check.py                 # confirm W before scanning
python scripts/build_hosttype_subset.py --in_path <events.h5> \
    --out_path data_derived/events_gal_hosted.h5 --host_type 0   # and --host_type 1
FORCE=1 ./scripts/run_scans.sh                     # the six scans of record
./scripts/run_seed_controls.sh 101 102 103 105     # the four remaining controls
./scripts/run_guard_diag.sh
python scripts/build_single_tracer.py
python scripts/aggregate_closure.py --seeds 100 101 102 103 105
python scripts/make_figures.py     # fig_h0_recovery, fig_guard, fig_closure_seeds
```

`fig_closure_v3` is the fourth record figure. Its labels are part of the figure, so
take the invocation verbatim from the tail of `scripts/run_v3_analysis.sh` rather
than re-typing it — `--before_dir attic/results_v2postfix`, `--fig_tag
fig_closure_v3`, `--out_json results/closure_v3.json`, plus the `--before_label`,
`--after_label` and `--what` strings.
