# Mechanism probes for the galaxy-catalog closure failure

> **2026-08-01: diagnostic products reorganized under `diagnostics/` (see
> `diagnostics/INDEX.md`); paths in this document refer to the original layout.**

> **SUPERSEDED IN ITS ATTRIBUTION — see `ATTRIBUTION.md`.** Probes 3 and 4 correctly
> isolate the scalar `r` and correctly identify the amplifier (the dense catalog's
> 21.6× smaller per-event `H0` curvature). Their *attribution* of `r` to "the effective
> per-pixel redshift prior at the sub-percent level" does not survive a term-by-term
> split of `r`: the catalog redshift-prior term carries **+1.23e-4 of a total −1.45e-3**
> (8 %, wrong sign), while the population term's **source-frame-mass** piece carries
> **−1.518e-3 (105 %)**. Probe 4's arms moved `r` by changing the posterior weights the
> mass score is averaged over, not by mis-specifying `p_z`. Everything the probes
> *measure* stands; the reading in "Combined reading" does not.

Owner-approved probes 1–4 into the failure recorded in `README.md`
(**Closure across realisations**): with each complete catalog handed only the
events it actually hosts (`dark_sirens_complete`, K = 1, `field` weighting,
targeted injections, `H0` ∈ [50, 100] × 201, `W = 4096`, the campaign guard
convention), the AGN catalog closes (mean offset **+0.05 ± 0.38**) and the GAL
catalog does not (**−9.43 ± 1.98**, 4.8σ low, all five realisations negative).

These probes only *measure*. No fix is applied, no dataset is regenerated, no
paper text is touched. `darksirens` @ `2b86a2d` and
`working/data/generate_dataset.py` are read-only throughout; instrumentation is
an import-level pass-through monkeypatch of
`darksirens.likelihood.core.selection_log_correction`, the pattern
`scan_h0f.py`'s guard record already uses.

Where each ran: probe 1 on `rita` CPU; probe 2 and probe 4 on the local A100
(`rita`, 80 GB); probe 3 on `TWIG-GPU` (`twig`, A100-40 GB, slurm job 1058123).

---

## Probe 1 — pixelation audit (CPU, `rita`)

`scripts/probe1_pixelation_audit.py` → `results/probe1_pixelation_audit.json`

**Verdict: BENIGN LAYOUT DIFFERENCE — the two pixelations agree bitwise on the
arrays the likelihood actually sees.** A stride subsample of the seed-100 GAL
catalog (**3 779 497** objects, every one of the 12 288 nside-32 pixels
occupied, z ∈ [0.0112, 1.0016]) was pixelated by both the generator's
`pixelate_catalog_vec` and darksirens' own `_pixelate_catalog`
(`scripts/mock_dark_sirens/generate_mock_data.py:462`; identical signature, `dz`
is the same per-object array), at `nside = 32` with the generator's exact
float32 `dz = 3e-3 (1+z)`. `ngals` is identical everywhere and `wgals` is
bitwise identical as written. `zgals`/`dzgals` differ in **80.9 %** of slots as
written — gmd's python loop emits *catalog* order inside a row, the vectorised
builder emits *z-sorted* order — but that is exactly the difference
`darksirens.catalogs.io.load_survey` erases: it applies `sort_survey_rows_by_z`
on **every** load (the default, and the only path `darksirens.inference.data`
uses), after which the two blocks agree **bitwise, 0 differing slots, max |Δ| =
0** on all three arrays. Both sorts are stable and both start from catalog order
among equal-z ties, so even the tie ordering matches. The vec builder's output
is already sorted (`load_survey`'s permutation is the identity on it); gmd's is
not. Timing, incidentally: 1.0 s vs 10.1 s on this subsample.

The production block passes every invariant the windowed evaluator assumes: rows
z-sorted **as written**, `ngals` ∈ [10 256, 14 569] summing to the catalog's
151 179 870 with no empty rows, real galaxies a contiguous `w > 0` prefix,
padding exactly at the (100.0, 1.0, 0.0) sentinels, and `dzgals` **bitwise**
equal to the generator's float32 `3e-3 (1+z)` on every real slot (the
float64-then-round route differs by one float32 ULP, 4.66e-10 — the only
numerical discrepancy anywhere in the audit).

One convention wart, quantified because it is real but is not a discrepancy:
`catalog_kernel_state` sets `sig_eff_row_max = max(sig_eff, axis=1)` over **all**
columns and the padded slots carry `dzgals = 1.0`, so **12 287 of 12 288** GAL
rows get a window half-width of `n_sigma × 1.0 = 8` in redshift instead of
`n_sigma × 6.0e-3 = 0.048`. Consequence: the "block fits in W" branch of
`_sorted_row_window_start` can never engage on this survey and the evaluator
always takes the nearest-neighbour branch `start = i_z − W/2`. That branch is
still correct (index order *is* z order), and probe 2 measures the cost: zero.

*Candidate mechanism: none from this probe — the survey block on disk is what
darksirens' own builder would have written, and it satisfies the windowed
evaluator's contract.*

---

## Probe 2 — catalog-KDE window sweep (GPU, local A100)

`scripts/probe2_kde_window.py` → `results/probe2_kde_window.json`,
`figs/probe2_kde_window.{png,pdf}`

**Verdict: WINDOW TRUNCATION DOES NOT CONTRIBUTE.** The seed-100 matched-GAL
scan was rerun at `W = 4096`, `8192` and `14569` (= `N_max`, i.e. the window
covers the whole row: `start` clips to `n_max − window` = 0), everything else
identical and all three arms sharing the same reduced reduction blocking
(`sel_batch_size = 50 000`, `pe_event_block = 25`, needed to fit the full-row
pass).

| W | median | offset | max abs Δ log L vs full row (shape) | s/eval |
|---|---|---|---|---|
| 4096 | 62.785390760139634 | −4.9546092398604 | 3.6e-12 | 3.07 |
| 8192 | 62.785390760139634 | −4.9546092398604 | 3.6e-12 | 3.58 |
| 14569 (full) | 62.785390760139650 | −4.9546092398603 | — | 4.15 |

`W = 8192` is **bitwise identical** to `W = 4096` (max |Δ| = 0.0 on all 201
cells). Against the full row the differences are 3.6e-12 in log L and
**1.4e-14 km s⁻¹ Mpc⁻¹** in the median — float64 summation order, nothing else.
The grid extremes are exactly zero (`Δ log L` at `H0 = 50` and `H0 = 100` are
both 0.0); the largest difference sits mid-grid at `H0 = 80.25`. 0/201 cells
rejected and min `N_eff = 3.95e5` in every arm.

The `W = 4096` arm at the reduced blocking also reproduces the production
`ctrl_gal_matched` scan (blocking 200 000/100) to max |Δ| = **1.8e-12**, median
62.785390760139634 vs 62.785390760139784 — so the reduction blocking is
inert too.

Probe 1's window-sizing measurement says why: at `sigma_kde = 0` the widest
kernel in any GAL row is `6.0e-3`, `recommended_kde_window` returns **3410** at
`n_sigma = 8`, and the number of galaxies actually inside ±8σ of a sample is
9–2057 on the densest row across `z ∈ [0.05, 1.0]` — while the `W = 4096` window
spans `z ∈ [0.035, 0.557]` at low z and ±0.09 at `z = 1`. The truncation is
never within an order of magnitude of biting. (The 14 407 that
`recommended_kde_window` returns at `sigma_kde_max = 0.05` is the requirement
for a *sampled* `sigma_kde`; these scans hold it at 0.)

*Candidate mechanism: none — the catalog KDE is converged, and the window is not
where the offset lives.*

---

## Probe 3 — numerator / selection decomposition (GPU, TWIG-GPU)

`scripts/probe3_decomposition.py`, `scripts/submit_probe3.sbatch` →
`results/probe3_decomp_{gal,agn}_s{100,101,102,103,105}.json`,
`results/probe3_decomposition.json`,
`figs/probe3_decomposition.{png,pdf}`, `figs/probe3_peaks.{png,pdf}`

The likelihood splits exactly into `log L(H0) = Σ_i ln Z_i(H0) + [−N ln µ(H0) +
N(N+3)/(2 N_eff)]`. All ten runs reproduce the stored `ctrl_*` scans of record
to max |Δ| ≤ **4e-12**, so the decomposition is anchored to the measurement, not
to a re-derivation of it.

**Verdict: the offset is not a mis-shaped GAL numerator. It is a per-event
score residual that both tracers share, amplified 21.6× by the dense catalog's
missing H0 curvature.**

First, the hypothesis as stated ("GAL numerator damped, AGN numerator anchors")
is refuted: **the numerator alone rails at `H0 = 100` in all ten runs**, GAL and
AGN alike. Neither numerator has an interior maximum; the peak is always set by
the balance between the numerator's slope and `−N d ln µ/dH0`, two quantities
that cancel to about 3 % of themselves (`d(selection)/dH0 ÷ d(total)/dH0` = 24
for AGN, 29 for GAL at seed 100).

The quantity that decides the peak is therefore the **score residual at truth**,
`r = (1/N) Σ_i d ln Z_i/dH0 − d ln µ/dH0`, which a correctly normalised
hierarchical likelihood sets to zero in expectation. Per realisation:

| | seed | N | peak | offset | score/ev | d ln µ/dH0 | r | r / (d ln µ/dH0) | d²logL/dH0² per ev |
|---|---|---|---|---|---|---|---|---|---|
| **GAL** | 100 | 720 | 62.877 | −4.863 | 0.040440 | 0.041888 | −0.001448 | −3.46 % | −3.59e-4 |
| | 101 | 661 | 62.055 | −5.685 | 0.040266 | 0.041888 | −0.001622 | −3.87 % | −1.03e-4 |
| | 102 | 698 | 50.000 † | −17.740 | 0.039575 | 0.041643 | −0.002068 | −4.97 % | −1.60e-4 |
| | 103 | 735 | 57.724 | −10.016 | 0.040103 | 0.041756 | −0.001653 | −3.96 % | −1.29e-4 |
| | 105 | 699 | 56.330 | −11.410 | 0.040686 | 0.041930 | −0.001243 | −2.96 % | −6.7e-5 |
| | | | | **−9.94 ± 2.31** | | | **−1.607e-3 ± 0.136e-3** | **−3.84 %** | **−1.64e-4** |
| **AGN** | 100 | 280 | 67.428 | −0.312 | 0.039139 | 0.040797 | −0.001658 | −4.06 % | −5.23e-3 |
| | 101 | 339 | 69.680 | +1.940 | 0.043422 | 0.041325 | +0.002097 | +5.07 % | −1.68e-3 |
| | 102 | 302 | 67.684 | −0.056 | 0.040676 | 0.040977 | −0.000302 | −0.74 % | −3.47e-3 |
| | 103 | 265 | 67.359 | −0.381 | 0.039607 | 0.040985 | −0.001377 | −3.36 % | −3.59e-3 |
| | 105 | 301 | 67.202 | −0.538 | 0.039340 | 0.041250 | −0.001910 | −4.63 % | −3.68e-3 |
| | | | | **+0.13 ± 0.46** | | | **−0.630e-3 ± 0.735e-3** | **−1.53 %** | **−3.53e-3** |

† railed at the grid edge, as in the production table.

Three readings, all arithmetic:

1. **The residual is a stable systematic for GAL**, −1.607e-3 ± 0.136e-3 over
   five realisations (t = −11.8), i.e. **−3.84 % of `d ln µ/dH0`**, negative in
   all five, spread only −2.96 % to −4.97 %. It is *smaller* than the naive
   event-sampling scatter (`sqrt(I/N)` = 4.6e-4 per realisation, observed
   seed-to-seed sd 3.1e-4), so it is not event noise.

2. **The amplifier is the curvature.** GAL's per-event H0 curvature is
   −1.64e-4 against AGN's −3.53e-3: **21.6× smaller**. Since the peak sits at
   `r/|d²|` per event, GAL's residual buys **−9.81** (measured mean offset
   −9.94) and AGN's buys **−0.18** (measured +0.13). The dense catalog carries
   essentially no localising redshift information — probe 4's audit shows a
   nside-32 GAL pixel holds **1023 ± 130** galaxies inside the GW horizon and
   **26.7 ± 10.6** below z = 0.1, so its KDE is a smooth continuum, whereas the
   AGN pixel holds ~10 — and that is exactly what the curvature measures.

3. **The AGN control cannot exclude the same residual.** GAL − AGN residual is
   −0.977e-3 ± 0.747e-3, **1.3σ** — statistically consistent. An AGN
   configuration carrying GAL's residual would sit at **−0.46**, against a
   measured AGN mean of +0.13 ± 0.46. The AGN closure at ±0.46 is simply ~22×
   less sensitive in `H0` units; it is not evidence that the residual is absent
   there.

The between-realisation scatter also localises: GAL's residual varies by 19 %
between seeds while its curvature varies by a factor **5.4** (−6.7e-5 to
−3.6e-4). The offset is `r/|d²|`, so the seed-to-seed swing in the answer
(−4.9 to −17.7) is dominated by the *denominator*, which is the same quantity
that sets the quoted half-width — which is why the between-mock scatter exceeds
the quoted width.

*Candidate mechanism: a ~3–4 % mismatch between the events' mean score and the
selection integral's `d ln µ/dH0` — small enough to be invisible in the AGN
configuration and in every convergence diagnostic — becomes a −10 offset only
because the dense catalog's per-event H0 curvature is 21.6× smaller.*

---

## Probe 4 — analytic continuum survey (GPU, local A100)

`scripts/probe4_continuum_survey.py`, `scripts/run_probe4.sh`,
`scripts/run_probe4_decomp.sh` → `results/probe4_build.json`,
`results/probe4_continuum.json`, `results/probe4{a,b,bemp}_gal_*.h5`,
`figs/probe4_continuum.{png,pdf}`

Three synthetic GAL surveys were written with the seed-100 block's on-disk
conventions **exactly** (nside 32, `dz = 3e-3 (1+z)` float32, z-sorted real
prefix, 100.0/1.0/0.0 padding, `ngals` = the number of real slots) and scanned
against the *same* 720 seed-100 GAL-hosted events, the same targeted injections,
the same grid, window and guard:

* **4a** — each pixel keeps its **real** galaxy count (so the field weight
  `N_obs[pix]/N_obs_total` is untouched) but its redshifts are replaced by the
  mid-point quantiles of the analytic `dN/dz ∝ dV_c/dz` on [0, z_max]: no
  clustering, no shot noise in z.
* **4b** — fully uniform sky: every pixel identical, 12 303 galaxies each
  (survey mean), same analytic continuum.
* **4b-emp** — the control for 4b's one known mis-specification: identical
  construction, but the quantiles come from the **catalog's own measured global
  `dN/dz`** (2000-bin CDF) instead of the analytic form. Still perfectly smooth,
  still clustering- and shot-noise-free.

(Quantiles rather than "a fine z grid with weights ∝ dV_c/dz" because darksirens
takes the per-pixel amplitude from `ngals` and masks real slots with
`arange < ngals`, so the row length *is* the pixel count; a short weighted grid
would silently rewrite every pixel's field weight.)

The analytic form tracks the real catalog's global `dN/dz` to **0.78 %** in
shape across the whole plateau and the whole event band (the ratio is a constant
1.0919 there, an overall normalisation); it differs at the two GLASS shell ends
— a deficit below z ≈ 0.046 and the partial last shell above z ≈ 0.923 — which
leaves 4b with ~8 % less prior mass inside the GW horizon than 4b-emp
(937 vs 1023 galaxies per pixel below z = 0.3565).

**Verdict: SPLIT — and it is the informative kind.**

| arm | median | offset | half-68 | truth in 68 % | score/ev | d ln µ/dH0 | r | d²/dH0² per ev |
|---|---|---|---|---|---|---|---|---|
| real catalog (`ctrl_gal_matched`) | 62.785 | **−4.95** | 2.42 | no | 0.040440 | 0.041888 | −1.448e-3 | −3.59e-4 |
| 4a per-pixel continuum | 61.606 | **−6.13** | 3.00 | no | 0.040706 | 0.041531 | −0.824e-3 | −1.12e-4 |
| 4b uniform sky, analytic dV_c/dz | 62.021 | **−5.72** | 3.00 | no | 0.040743 | 0.041536 | −0.793e-3 | −1.21e-4 |
| 4b-emp uniform sky, empirical dN/dz | 68.784 | **+1.04** | 3.04 | **yes** | 0.042086 | 0.041939 | **+0.147e-3** | −1.54e-4 |

0/201 cells rejected in every arm; min `N_eff` = 3.95e5 (real catalog) and
≥ 4.87e5 on the three synthetic arms — 80–100× the 5 N_obs = 3600 floor.

* The offset **survives** a perfectly smooth, perfectly known analytic prior with
  no catalog realisation in it at all (4b: −5.72, 115 % of the real catalog's).
  It is therefore **not** a clustering, shot-noise, pixelation or KDE artefact,
  and it is reproducible analytically. Removing the sky structure on top of that
  (4a → 4b) changes nothing material (−6.13 → −5.72).
* But it **vanishes** when the same perfectly smooth continuum is built from the
  catalog's own measured `dN/dz`: 4b-emp returns **+1.04** with truth inside the
  68 % interval, and its score residual is **+1.5e-4**, i.e. +0.35 % — zero to
  the precision of everything else here.
* The decomposition says which term moved. Between 4b and 4b-emp the selection
  slope barely changes (0.041536 → 0.041939, +0.97 %) while the **numerator's
  per-event score** moves by **+3.3 %** (0.040743 → 0.042086). The peak moves
  because the numerator's score does, not because `µ` does.

So the two smooth priors differ by <1 % in shape over the event band and by the
shell ends, and that is worth **6.8 km s⁻¹ Mpc⁻¹** of recovered `H0`. That is
probe 3's amplifier stated in prior-space: with a per-event curvature of
~1.5e-4, moving the score by 1e-3 (2.4 % of `d ln µ/dH0`) moves the answer by
~7.

*Candidate mechanism: the recovered `H0` in this configuration is set by the
effective per-pixel redshift prior at the sub-percent level — the real catalog's
per-pixel structure and a ≲1 % error in the assumed smooth `dN/dz` each perturb
the per-event score by ~1.5e-3 and each cost ~5–6 km s⁻¹ Mpc⁻¹.*

---

## Combined reading

Probes 1 and 2 close off the two mechanical suspects: the survey block on disk is
bitwise what darksirens' own pixelation would have written once the load-time
z-sort is applied, and the catalog KDE is converged — `W = 4096`, `8192` and the
full 14 569-column row give the same log-likelihood to 3.6e-12 and the same
median to 1.4e-14.

Probes 3 and 4 pin the failure to **one scalar**: the per-event score residual at
truth,

```
r = (1/N_obs) Σ_i  d ln Z_i/dH0  −  d ln µ/dH0            evaluated at H0 = 67.74
```

which the hierarchical likelihood is supposed to zero in expectation. In the
matched-GAL configuration it is **−1.61e-3 ± 0.14e-3 per event** over five
realisations — **−3.84 % of `d ln µ/dH0`**, t = −11.8 — and the recovered peak
sits at `r / |d² log L/dH0²|`. What makes GAL fail and AGN pass is **not** the
residual, which the two tracers share within 1.3σ, but the denominator: the dense
catalog delivers **21.6× less H0 curvature per event** (−1.6e-4 vs −3.5e-3),
because a nside-32 GAL pixel holds ~1000 galaxies inside the horizon and its KDE
is a smooth continuum that localises nothing, while an AGN pixel holds ~10. The
same residual costs GAL −9.8 and AGN −0.18. The AGN control at ±0.46 cannot
exclude it.

Probe 4 says what the residual is sensitive to and what it is not. It is not
clustering, not shot noise, not the sky weighting, not pixelation and not the
KDE window: it survives at full strength on a fully analytic, clustering-free,
uniform-sky continuum. It is the **effective per-pixel redshift prior at the
sub-percent level** — swap the analytic `dV_c/dz` continuum for one built from
the catalog's own measured `dN/dz` (a <1 % change in shape across the event band,
plus the two shell ends) and the residual goes to +1.5e-4 and the recovery closes
at +1.04 with truth inside 68 %. The term that responds is the numerator's score,
not `ln µ`.

Two consequences follow directly and are stated without any proposed fix:

* The dense-catalog configuration has no error budget in the redshift prior. A
  sub-percent error in `p_cat(z | pix)` — from any source, including the choice
  of smooth form — is worth several km s⁻¹ Mpc⁻¹. The estimator's quoted
  half-width (±2.4) is set by the same tiny curvature and therefore describes
  none of that sensitivity, which is why the between-mock scatter exceeds it.
* The GAL rows of the production table cannot be read as a mis-specification cost
  until this is resolved, exactly as `README.md` already states — and the AGN
  rows are not independently validated against the same residual, only
  insensitive to it.

**Scope stops here.** The probes pin the term (`Σ_i d ln Z_i/dH0` versus
`d ln µ/dH0`, at the sub-percent level of the per-pixel redshift prior) and the
amplifier (the dense catalog's 21.6× smaller per-event H0 curvature). Deciding
whether the residual is a genuine mis-specification of the mock, a convention in
the complete-catalog prior, or a normalisation of the selection integral — and
any run that would test it — is the owner's call.

Probe 4c (drawing events from a continuum catalog and analysing them against 4b)
was **not** run. It is no longer the discriminating experiment: 4b already
decorrelates the catalog from the events entirely, and 4b-emp shows the failure
is controlled by the assumed `dN/dz` shape rather than by any event–catalog
correlation. Running 4c would test a hypothesis probe 4 has already answered.

---

## Files written

```
scripts/
  probe1_pixelation_audit.py     probe 1 (CPU)
  probe2_kde_window.py           probe 2 driver (invokes scan_h0f.py per window)
  probe3_decomposition.py        probe 3 (per-run and --aggregate); --survey_override
                                 reuses it on probe 4's surveys
  submit_probe3.sbatch           TWIG-GPU submission for probe 3 (job 1058123)
  probe4_continuum_survey.py     probe 4 (build / scan / analyse)
  run_probe4.sh                  probe 4 scan+analyse driver
  run_probe4_decomp.sh           probe 3's decomposition on probe 4's three surveys

results/
  probe1_pixelation_audit.json   head-to-head, production-block invariants, window sizing
  probe2_kde_window.json         three window arms + blocking-invariance cross-check
  probe2_gal_W{4096,8192,14569}.{h5,json}
  probe3_decomp_{gal,agn}_s{100,101,102,103,105}.json    ten decompositions
  probe3_decomp_gal_s100_{p4a,p4b,p4bemp}.json           the same on probe 4's surveys
  probe3_decomposition.json      per-seed table + per-tracer summaries
  probe4_build.json              continuum construction, dN/dz fidelity, low-z per-pixel audit
  probe4_continuum.json          four arms + their decompositions + verdict
  probe4{a_gal_continuum,b_gal_uniform,bemp_gal_uniform}.{h5,json}

figs/
  probe2_kde_window.{png,pdf}    window arms and their (null) differences
  probe3_decomposition.{png,pdf} numerator / selection / total, GAL vs AGN, five seeds
  probe3_peaks.{png,pdf}         where each piece peaks
  probe4_continuum.{png,pdf}     the four posteriors

synthetic surveys (bulk filesystem, not in this directory):
  /hildafs/projects/phy220048p/magana/gws-agn-data/derived/
      analysis_1_complete_catalog_H0/probe4/survey_gal_probe4{a_continuum,
      b_uniform,bemp_uniform}_s100_ns32.h5
```

Reproduce:

```bash
python scripts/probe1_pixelation_audit.py                       # CPU, ~40 s
python scripts/probe2_kde_window.py                             # GPU, ~55 min
sbatch scripts/submit_probe3.sbatch                             # TWIG-GPU, ~65 min
python scripts/probe3_decomposition.py --aggregate --seeds 100 101 102 103 105
python scripts/probe4_continuum_survey.py build                 # CPU, ~2 min
./scripts/run_probe4.sh                                         # GPU, ~30 min
./scripts/run_probe4_decomp.sh                                  # GPU, ~30 min
python scripts/probe4_continuum_survey.py analyse
```
