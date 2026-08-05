# Attribution of the per-event score residual

> **2026-08-01: diagnostic products reorganized under `diagnostics/` (see
> `diagnostics/INDEX.md`); paths in this document refer to the original layout.**

> **Followed up on 2026-08-01 — see `CLOSURE.md`.** Both defects named below
> (the mass PE convention, and the RA measurement width of A4.5/A5.2) were
> implemented, the events stage of all five realisations regenerated and the analysis
> of record rerun; the mass repair delivered `+5.60e-4` against the `+5.65e-4 ±
> 0.33e-4` predicted here. The remaining suspect this document named — the nside-32
> pixelisation — was then measured in closed form against the real catalog's galaxy
> positions and is **not** the residual (`+9.1e-5 ± 1.9e-4`, 11 % ± 24 %). Every
> number and every file below is PRE-FIX and unchanged; the post-fix products carry
> the `_postfix` tag or live in `attr_sky_oracle_*`.

Owner-approved follow-up to `PROBES.md`. The target is the one scalar probes 3 and 4
pinned the galaxy-catalog closure failure to,

```
r  =  <d ln Z_i/dH0>_events  -  d ln mu/dH0            evaluated at H0 = 67.74
```

measured at **−1.607e-3 ± 0.136e-3 per event** in the matched-GAL configuration over
five realisations (**−1.4499e-3** on seed 100 alone), and consistent with the same
value in matched-AGN. A correctly specified detected-set likelihood sets it to zero.

Everything below is on the **analysis of record** — `dark_sirens` at
`log10n0 = −24`, field sky weighting, K = 1, targeted injections, `W = 4096` for GAL,
the campaign guard convention, seed 100, both matched-host controls. `darksirens` was
**read-only at `2b86a2d`**; the two patches used are import-level pass-throughs of the
kind `scan_h0f.py`'s guard record already uses. **No fix is implemented, no dataset is
regenerated, no closure scan is rerun.**

---

## The verdict in one line

**The residual is not the catalog redshift prior. It is the population term's
source-frame-mass piece — the spectral-siren mass channel — and it is generated in the
per-event PE/measurement model, not in the population, the redshift prior, or the
selection integral.**

`r_mass / r` = **105 %** (GAL), 102 % (AGN);  `r_pz / r` = **−8 %** (GAL), −3 % (AGN).

This **supersedes the reading in `PROBES.md`**, which attributed the residual to "the
effective per-pixel redshift prior at the sub-percent level". Probe 4's arms did move
`r`, but they moved it by changing the *posterior weights* the mass score is averaged
over, not by mis-specifying `p_z`. The redshift-prior term itself carries 8 % of the
residual, with the wrong sign.

---

## How the split is made

`darksirens`'s per-sample target density is, exactly,

```
ln p_target(theta | H0) = ln p_pop(m1src, q, z, chieff)
                        + ln p_z(z | pix)
                        - ln[d dL/dz] - ln(1+z),          z = z(dL; H0)
```

so the per-sample score `varsigma = d ln p_target/dH0` is a sum of three additive
pieces, and **both sides of `r` are averages of that one function**:

```
d ln Z_i/dH0 = E_{PE posterior i}[varsigma]      (softmax of the event's ldw)
d ln mu /dH0 = E_{injections}   [varsigma]       (softmax of the injections' ldw)
```

Under a correctly specified detected-set likelihood, `E[C_h] = E[A_h] = B_h` for
**every** function `h(theta)` — where

| symbol | meaning |
|---|---|
| `C` | `mean_i E_post_i[h]` — what the likelihood averages |
| `A` | `mean_i h(theta_i^true)` — the empirical detected-**truth** mean |
| `B` | `E_injections[h]` — the model's detected-truth mean |

so `r = C − B` splits **term by term**, and further into `r = (C − A) + (A − B)`:
`(A − B)` is a population / detection-rule mis-specification; `(C − A)` is purely the
measurement model. `(C − A)` is a *paired* per-event statistic and is therefore far
sharper than either mean alone.

`scripts/attr_score_terms.py` rebuilds `darksirens`'s per-sample weight outside the
likelihood (same prior state, same clamping, same masking, same windowed KDE) and is
checked against it: **`|Δ log mu| = 0` and `|Δ Σ_i ln Z_i| = 0`** — bit-exact, in both
configurations. Everything below is anchored to that.

---

## Stage 1 — where the residual lives

`results/attr_terms_{gal,agn}_s100.json`, `results/attr_mass_pe_{gal,agn}_s100.json`

Per event, at `H0 = 67.74`, central difference `dh = 0.5`:

| term | matched GAL | matched AGN |
|---|---|---|
| `p_pop`, **mass** (`m1src = m1det/(1+z)`) | **−1.5179e-3** | **−1.8707e-3** |
| `p_pop`, rate (`(1+z)^(gamma−1)`, gamma = 0) | −1.94e-5 | −2.19e-6 |
| catalog `p_z(z|pix)` | **+1.23e-4** | +4.97e-5 |
| Jacobian `−ln(ddL/dz) − ln(1+z)` | −3.51e-5 | −2.04e-6 |
| **total `r`** | **−1.4499e-3** | −1.8253e-3 |

`r` reproduces probe 3's independent measurement (`−1.448e-3`, GAL seed 100) to four
digits, from a completely different code path.

**The mass piece is the whole residual, in both tracers.** That is also why the two
tracers share it within 1.3σ (probe 3's observation): it is a property of the events
and the mass model, not of the catalog. What differs between them is only the
amplifier — the dense catalog's 21.6× smaller per-event `H0` curvature.

### It is the measurement model, not the population

| | `A` (truth) | `B` (model) | `A − B` |
|---|---|---|---|
| GAL, mass term | 6.1312e-3 | 6.1949e-3 | **−6.4e-5 ± 4.5e-4** (0.14σ) |
| AGN, mass term | 7.1289e-3 | 6.1367e-3 | +9.9e-4 ± 7.9e-4 (1.3σ) |

The population and the detection rule are **exactly right**: evaluating the same score
function at the events' own true parameters and averaging reproduces the injections'
mean. The entire residual is in `(C − A)` — the posterior-averaging step:

| | `(C − A)` mass, stored PE | after the exact mass PE |
|---|---|---|
| GAL | **−1.454e-3 ± 0.273e-3** (5.3σ) | −0.891e-3 ± 0.262e-3 (3.4σ) |
| AGN | −2.863e-3 ± 0.437e-3 (6.5σ) | −2.204e-3 ± 0.416e-3 (5.3σ) |

Independent corroboration that the population is not the problem: `gmd`'s own mass
sampler (which drew the events) matches `darksirens`'s `powerlaw+peak` density to
**≤ 0.5 % in shape across the detected band and 0.02 % in the mean** (33.019 vs
33.012 M⊙); the events' true `m1src` against the model-predicted detected
distribution gives **KS p = 0.22**, `chi2/dof = 0.99`.

---

## Stage 2 — the named defect

`working/data/generate_dataset.py`, `observe()` and `posterior_samples()`:

```python
sig_m1 = SIG_M1_FRAC * m1det          # 0.08 * m1det_TRUE      <- latent
obs_m1 = np.clip(rng.normal(m1det, sig_m1), 2.0, None)
...
out["m1det"].append(np.clip(rng.normal(obs["m1det"][i], obs["sig_m1"][i], nsamp), 2.0, None))
```

The PE mass samples are a **fixed-width Gaussian about the observation whose width is
computed from the LATENT true mass** (verified bit-exact on the stored file:
`sig_m1 == 0.08 * m1det_true` and `sig_m2 == 0.10 * m2det_true` to `0.0`). The
measurement model actually realised is `obs ~ N(m, f m)` with `f` constant, and its
exact flat-prior posterior is **not** a fixed-width Gaussian:

```
p_ex(m | obs)  ∝  (1 / (f m)) exp[ -(obs - m)^2 / (2 f^2 m^2) ]
```

whose mean sits `+2 f^2 = +1.28 %` above `obs` (measured on the stored samples:
**+1.323 %**, validated against 1-D quadrature to 0.1 % of `sigma`).

This is the **mass twin of the sky-width defect** the campaign already fixed upstream
as darksirens **PR #335**, and which `generate_dataset.py`'s own convention (b)
declares non-negotiable for the sky. The distance channel is exempt — its noise is
multiplicative with a *constant* log-width, so the stored lognormal really is the
exact posterior (verified: `mean(ln dL) − ln obs = 0.01004` against `s^2 = 0.01`,
`sd = 0.0997` against `s = 0.10`).

### What repairing it is worth

The repair needs no regeneration: the stored samples are reweighted from the
fixed-width proposal to `p_ex`. The weight is **`H0`-independent by construction**, so
it cannot manufacture or hide an `H0` slope — it can only correct the measure the
score is averaged over. The selection integral is untouched (injections carry true
parameters), so `d ln mu/dH0` is identical in every arm.

| arm (matched GAL) | `r` | `r_mass` | `(C − A)_mass` |
|---|---|---|---|
| stored PE (the mock) | −1.4499e-3 | −1.5179e-3 | −1.454e-3 |
| width from `obs` (PR #335 style) | −1.6958e-3 | −1.7921e-3 | −1.728e-3 |
| **exact `p(m1|obs)`** | **−8.475e-4** | −9.462e-4 | −0.882e-3 |
| **exact `p(m1,m2|obs)`** | **−9.666e-4** | −9.549e-4 | −0.891e-3 |

**The exact-posterior repair removes 33 % of `r` (GAL) and 24 % (AGN, `m1` arm)**, and
the same fraction of the underlying posterior-mass bias. (On AGN the `m1+m2` arm
degrades the *catalog* term — `r_pz` goes to `−1.25e-3` — on that sparse, spiky prior;
its `mass` term is unchanged at `−1.212e-3`, so the mass-channel conclusion is the same.
On GAL the `m2` correction is inert, as expected: `q` is `H0`-independent and enters the
mass score only through the `m2src` low-mass taper.) Note that the *minimal* PR #335-style
repair (keep the Gaussian, take the width from the data) makes it **worse** — the
`O(f^2)` shape, not the width, is what matters here.

### Closed-form confirmation

`scripts/attr_toy_masswidth.py` → `results/attr_toy_masswidth.json`. A controlled
one-parameter toy — `m ~ p_pop`, `obs ~ N(m, f m)`, a data-based detection cut,
posteriors by quadrature so there is no Monte Carlo — reproduces the identity and
isolates the convention:

| PE convention | `r_kappa` | equivalent `r_mass` |
|---|---|---|
| stored (latent width) | **+0.365 ± 0.009** | −6.3e-4 |
| width from `obs` | +0.472 ± 0.009 | −8.1e-4 |
| **exact** | **+0.011 ± 0.009** | −1.8e-5 |

The exact arm is consistent with **zero at every selection strength** tested
(detection fraction 1.00 → 0.17); the stored convention gives −5e-4 … −9e-4. That
brackets the measured removal of **−5.7e-4 (GAL)** and **−6.6e-4 (AGN)** — the toy
predicts the size of this channel independently and correctly.

---

## What is left, and what it is not

After the exact mass-PE repair, `r = −9.67e-4` (GAL) remains, i.e. `(C − A)_mass =
−0.891e-3 ± 0.262e-3` (3.4σ). Its direct signature is that **the ensemble-averaged
posterior mean of the source-frame primary mass stays low**:

```
<E_post[m1src]> - <m1src_true>  =  -1.92 % +- 0.28 %  (GAL, stored)
                                   -1.12 % +- 0.27 %  (GAL, exact PE)
                                   -3.45 % +- 0.43 %  (AGN, stored)
                                   -2.61 % +- 0.42 %  (AGN, exact PE)
```

against a redshift channel that is unbiased (`<E_post[z]> − <z_true>` = −0.65 % ±
0.36 % GAL, +0.80 % ± 0.55 % AGN). Per event this shrinkage is expected; what must
vanish and does not is the ensemble mean. The per-event residual correlates with the
**mass-noise realisation** `xi_1 = (obs_m1 − m1det_true)/sigma_1` at **+0.65** and
grows steeply with the true mass (from `+1.4e-3` in the lowest `m1src` quintile to
`−9.6e-3` in the highest) — i.e. it lives exactly where the mass function is steep and
strongly curved.

Ruled out, all measured on the analysis of record:

| candidate | measurement |
|---|---|
| catalog redshift prior | `r_pz = +1.23e-4` of a total `−1.45e-3` — 8 %, wrong sign |
| rate factor `(1+z)^(gamma−1)` | `r_rate = −1.9e-5` (GAL), `−2.2e-6` (AGN) |
| distance / Jacobian term | `r_jac = −3.5e-5` (GAL), `−2.0e-6` (AGN) |
| population mass function | sampler vs model: ≤ 0.5 % in shape across the detected band, 0.02 % in the mean; KS p = 0.22 (AGN), chi2/dof = 0.99 (GAL) |
| detection rule | `rho_obs` recomputed from the stored observation reproduces `snr_obs` **bit-exactly** on all 1000 events, all ≥ 8 |
| selection integral / `pdraw` | `(A − B)_mass = −6.4e-5 ± 4.5e-4` (0.14σ) |
| finite `nsamp` Monte Carlo | halving 2000 → 1000 moves `r_mass` by `2.4e-5` (GAL) and `5.9e-5` (AGN): the `n = 2000` bias is 2–3 % of `r` |
| PE mass **width** error | `d r_mass / d ln sigma_1 = −3.9e-3`; closing the remainder needs a **+28 %** (GAL) / **+33 %** (AGN) width error |
| redshift-posterior bias | `z` unbiased to 0.65 % ± 0.36 %; closing the remainder through `z` needs **≈ 19 %** |
| distance and sky PE | `ln dL ~ N(ln obs + s², s)` to 1e-4; `dec`/`ra` widths reproduce the data-derived `sigma_ang` to 0.3 %; `sigma_ang ∈ [1.0°, 2.39°]` |

**Statement of scope: the channel is attributed and one defect inside it is named and
quantified; the remaining ~2/3 is localised to the same mass sector of the PE but is
not yet tied to a specific convention.**

---

## What this is worth in `H0`

The peak sits at `r / |d² logL/dH0²|` per event (probe 3). On seed 100's own curvature
(`−3.587e-4` per event) the measured `r` implies **−4.04**, and after the exact
mass-PE repair **−2.69**. On the five-realisation mean curvature (`−1.64e-4`) the
measured `r = −1.607e-3` implies **−9.8**, and the same fractional repair leaves
**≈ −6.5**. So the named defect is worth roughly **+3.3 km s⁻¹ Mpc⁻¹** of the GAL
control's −9.4, and the unattributed remainder roughly **−6.5**.

The AGN control is insensitive to all of it for the reason probe 3 already gave: 21.6×
more curvature per event turns the same residual into −0.2.

---

## Minimal fix options, ranked, and where each would live

**None of these is implemented. This section is for owner sign-off only.**

1. **Generator — the mass measurement model** (`working/data/generate_dataset.py`,
   `observe()` / `posterior_samples()`). *The one defect actually named.* Either
   (a) make the mass noise width a genuine constant per event that is a function of
   the **data** and store the exact flat-prior posterior for it, or (b) keep
   `obs ~ N(m, f m)` and draw the PE from the exact posterior
   `p(m|obs) ∝ (1/(f m)) exp[−(obs−m)²/(2f²m²)]` by inverse CDF — the direct analogue
   of what PR #335 did for `sigma_ang`. **Cost: regenerating the events stage of every
   seed** (catalogs, surveys and injections are untouched — the injections never store
   an observation). Worth ≈ +3.3 km s⁻¹ Mpc⁻¹ on the GAL control, ≈ +0.12 on AGN.
   Note that the *naive* repair (Gaussian of width `f·obs`) is measurably **worse**;
   the exact posterior is required.
2. **Generator — reduce the mass measurement precision as a design choice.** The
   residual scales with how hard the mass channel is working: `sigma/m = 8–10 %`
   against a `35 ± 5 M⊙` Gaussian peak makes the spectral-siren channel a strong,
   strongly curved lever, and the dense-catalog configuration has no `H0` curvature to
   fight it. A mock with a weaker mass–redshift lever would not exhibit this at all.
   This is a *scope* decision about what the mock is for, not a bug fix.
3. **Analysis convention — quote the dense-catalog result with its curvature.** The
   GAL configuration converts any per-event score residual into `r/|d²|` with
   `|d²| ≈ 1.6e-4`, i.e. an amplification of ~6000 km s⁻¹ Mpc⁻¹ per unit `r`. Nothing
   in the estimator's quoted half-width describes that. This does not fix `r`; it
   makes the sensitivity explicit and is the honest framing if the remainder is not
   closed.
4. **Upstream `darksirens`.** *No defect found here.* The estimator reproduces the
   score identity exactly when handed a correctly specified PE (the closed-form toy's
   exact arm), the standalone rebuild matches it bit-for-bit, and every term of its
   redshift prior, selection integral and Jacobian passes. The one known estimator
   overhead in this channel — the finite-`nsamp` MC bias of `ln Ẑ_i` — is measured at
   2–3 % of `r`.
5. **Survey building.** Not implicated: `r_pz` is 8 % of `r` with the wrong sign, and
   probes 1–2 already showed the survey block is bitwise what darksirens' own
   pixelation would have written and that the KDE is converged.

**Open item for the owner:** the remaining ≈ 2/3 of `r` is a 3.4σ (GAL) / 5.3σ (AGN)
violation of `E[C] = E[A]` in the mass sector that survives an exact mass likelihood.
The decisive next test, if authorised, is the full three-parameter oracle —
`(m1, m2, dL)` with the mock's own detection rule and an analytic redshift prior,
posteriors by quadrature — which would say whether any measurement model can satisfy
the identity in this configuration, or whether a second convention is still hiding in
the events stage.

---

## Files written

```
scripts/
  attr_score_terms.py     term-by-term split of r (GPU); bit-exact anchors
  attr_mass_pe.py         mass-channel arms, truth-point split, paired (C-A)
  attr_toy_masswidth.py   closed-form toy for the mass-measurement convention (CPU)
  attr_figures.py         the figure

results/
  attr_terms_{gal,agn}_s100.{json,npz}      the three-term split + per-event arrays
  attr_mass_pe_{gal,agn}_s100.{json,npz}    7 PE arms, truth split, per-event arrays
  attr_toy_masswidth.json                   toy: stored / obswidth / exact
  attribution_summary.json                  every number quoted above

figs/
  attr_attribution.{png,pdf}                (a) where r lives, (b) the PE arms,
                                            (c) the posterior mass bias

logs/
  attr_terms_gal.log, attr_mass_pe_gal.log
```

Reproduce (one A100; ~5 min per configuration):

```bash
python scripts/attr_score_terms.py --tracer gal --kde_window 4096 \
    --pe_batch_events 25 --sel_batch 50000
python scripts/attr_score_terms.py --tracer agn --pe_batch_events 70 --sel_batch 100000
python scripts/attr_mass_pe.py     --tracer gal --kde_window 4096 \
    --pe_batch_events 25 --sel_batch 50000
python scripts/attr_mass_pe.py     --tracer agn --pe_batch_events 70 --sel_batch 100000
python scripts/attr_toy_masswidth.py          # CPU, ~1 min
python scripts/attr_figures.py
```

---
---

# Appendix — closing `r`: the sampler test, the repaired scans, and the quadrature oracle

Owner-approved continuation of the work above. Same scope in every respect:
`dark_sirens` at `log10n0 = −24`, field sky weighting, K = 1, targeted injections,
`W = 4096` (GAL), the campaign guard convention, seed 100, both matched-host
controls. **`darksirens` was READ-ONLY at `2b86a2d`**; the only patches are the
same import-level pass-throughs already used above, now factored into
`scripts/attr_ds_bridge.py` and re-anchored in every run
(**`|Δ log μ| = 0` in all four configurations**). **No generator edits, no paper
edits, no dataset regeneration.** The two matched scans were rerun on *copies* of
the events files with `p_pe` corrected; `working/data` was not touched.

**Verdict in one line: the population sampler is exonerated at the 1e-8 level, the
named mass-PE defect is worth 39.5 % of `r` and +2.15 km s⁻¹ Mpc⁻¹ on the GAL
control, the catalog's declared photo-z kernel is worth another ~10 %, and 51 %
of `r` survives a measurement model that is exact in every channel.**

---

## A1 — the population sampler is not the residual (task 1)

`scripts/attr_sampler_ratio.py` → `results/attr_sampler_ratio.{json,npz}`,
`results/attr_sampler_draws.npz`, `figs/attr_sampler_ratio.{png,pdf}`

**1.2 × 10⁸** `(m1src, q)` pairs were drawn through the *same code path*
`generate_dataset.py::stage_events` uses — `import_gmd` from the `2b86a2d`
checkout, `gmd.PopulationConfig(gamma=0)`, then
`_sample_powerlaw_peak_m1(..., return_component=True)` → `_sample_q(..., use_peak=...)`
→ `_sample_chieff` — under a spawned `SeedSequence` (24 forked workers, 18 s CPU,
master seed 20260731). Realised peak fraction **0.8999412** against the configured
0.90.

Both `gmd` samplers are *exact rejection samplers*, so their joint density is known
in closed form up to normalisation. The script builds it — `U_c` by composite
Gauss–Legendre, `V_c(m1) = m1^-(β+1) ∫₀^{m1} m₂^β S_low dm₂` by an exact cumulative
GL rule (checked against brute force to **4 × 10⁻¹⁶**) — and the 1.2 × 10⁸ draws
then serve only to *validate* it:

| check | result |
|---|---|
| 2-D bins with ≥ 50 counts (46 770 of 89 000): pull of MC vs closed form | mean **+0.021**, sd **1.001**, max \|·\| 4.63 |
| `m1src` marginal, same pull | mean +0.054, sd 0.989, max \|·\| 3.41 |
| `chieff` marginal, same pull | mean +0.034, sd 0.998 |

so the closed form **is** the sampler, and the ratio map carries no Monte-Carlo
error at all.

**Where the two densities differ.** Over the whole `(m1src, q)` plane the
probability-weighted rms of `ln[q_sampler/p_analytic]` is **1.64 × 10⁻⁴**. The
only place it is not ≲ 1e-4 is the *low-mass taper*: `+1.3 %` at `m1src = 5.5`,
`+0.12 %` at 6.0, `< 0.02 %` above 7 — the corner where `_sample_q`'s hard-`q^β`
proposal has vanishing acceptance and `m1` has essentially no probability. There
is also a uniform normalisation offset: darksirens' `powerlaw+peak` integrates to
**0.9999880749** over `[1, 200] × [0, 1]` against the sampler's `0.9999999998`,
i.e. its per-component *trapezoid* normalisations are collectively low by
**1.2 × 10⁻⁵**. Mean `m1src`: **33.0164** (sampler) vs **33.0176** (analytic),
0.0036 %.

**What that is worth in `r`.** The detected-set expectation was formed by
importance-reweighting the stored injections by `R = q_sampler/p_analytic`
evaluated at each injection's `(m1src, q)` — darksirens' own selection weights
(which already carry `pdraw`, the detection decision, `p_z` and the Jacobian)
multiplied by `R`, so nothing but the mass channel moves. Over the detected set
`R ∈ [0.99969, 1.01032]`, weighted mean **1.000005**, weighted sd 7.8 × 10⁻⁵.

| | `E_q[ς] − E_p[ς]`, mass | total |
|---|---|---|
| matched GAL | **+8.23e-9 ± 1.17e-9** | **+1.31e-8 ± 1.03e-8** |
| matched AGN | +8.51e-9 ± 1.42e-9 | −6.3e-9 ± 4.3e-8 |

An independent histogram-`R` route (bin-mass ratios from the draws) gives
+9.0e-8 (GAL, total) and −1.2e-5 (AGN, total), consistent with the closed form at
its own Monte-Carlo precision.

**The sampler-vs-pdf channel contributes 1.3 × 10⁻⁸ of the −9.7 × 10⁻⁴ that was
unexplained — 0.001 %, five orders of magnitude too small.** The population is
exonerated far beyond the 0.14σ the 720-event `(A − B)` statistic could say.

---

## A2 — what repairing the named defect is worth in `H0` (task 2)

`scripts/make_pe_corrected_events.py` → `data_derived/events_{gal,agn}_hosted_pefix_{m1m2,m1}.h5`,
`results/pe_corrected_events.json`;
`scripts/run_fix_scans.sh` → `results/fix_named_defect_{gal,agn}[_m1].{h5,json}`;
`figs/fig_before_after_fix.{png,pdf}`

The stored PE mass samples were reweighted to the exact flat-prior posterior of
the generator's own measurement model `obs ~ N(m, f m)` using **the weight
`attr_mass_pe.py` already constructs** (`log_pex`, `log_ptilde` imported, not
re-derived), and the reweighting was carried into `darksirens` without touching it
by writing `p_pe_new = p_pe_old / ρ`. Because `load_gw_samples` renormalises `p_pe`
per event, this reproduces the self-normalised reweighted evidence exactly up to a
per-event constant that **does not depend on `H0`**, so the `H0` posterior is the
reweighted one. `ρ` is `H0`-independent by construction and the injections are
untouched — confirmed in the scan record: **min `N_eff` is identical to the
record** (3.953e5 GAL, 2.161e5 AGN), 0/201 cells rejected in every arm.

| scan | median | offset | predicted from the score arithmetic |
|---|---|---|---|
| `ctrl_gal_matched` (record) | 62.789 | **−4.951** | — |
| `fix_named_defect_gal` (exact `p(m1,m2\|obs)`) | 64.941 | **−2.799** | −3.30 |
| `fix_named_defect_gal_m1` (exact `p(m1\|obs)`) | 65.052 | −2.688 | −2.89 |
| `ctrl_agn_matched` (record) | 67.391 | **−0.349** | — |
| `fix_named_defect_agn` (exact `p(m1,m2\|obs)`) | 67.258 | −0.482 | −0.47 |
| `fix_named_defect_agn_m1` (exact `p(m1\|obs)`) | 67.500 | **−0.240** | −0.26 |

**The named defect is worth +2.15 km s⁻¹ Mpc⁻¹ on the matched-GAL control**
(−4.95 → −2.80), slightly more than the linear score arithmetic predicted, and the
`m1`-only arm lands in the same place (−2.69). Truth stays outside the 68 % CI in
both GAL arms.

Two honest caveats, both quantified:

* **The reweighting costs PE Monte-Carlo precision.** `ρ`'s own ESS is 1431 (mean)
  / 4 (min) per event on GAL `m1m2`, with 26/720 events below 500; the guard's
  `pe_variance_sum` rises from 1.09 to 1.74 (GAL) and 11.2 to 13.0 (AGN), well
  inside the campaign's inert 1e6 threshold.
* **The AGN `m1m2` arm moves the wrong way (−0.349 → −0.482), and that is noise.**
  The reweighted-MC arm attributes it to the *catalog* term
  (`Δr_pz = −1.30e-3 ± 1.11e-3`, 1.2σ from zero on a sparse, spiky prior). The
  quadrature oracle of A3, which has no Monte-Carlo error at all, puts the true
  total change at **+6.23e-4 ± 4.3e-5** — i.e. the exact-mass repair helps AGN too,
  and the `m1`-only scan (−0.240) is the better reading of it.

---

## A3 — the (m1, m2, dL) quadrature oracle (task 3)

`scripts/attr_oracle.py`, `scripts/run_oracle.sh` →
`results/attr_oracle_{gal,agn}.{json,npz}` (+ eight convergence runs),
`figs/attr_oracle.{png,pdf}`

Every measurement channel of this mock is closed-form, so the per-event evidence
can be computed by direct quadrature instead of by Monte Carlo over stored PE
samples. Writing darksirens' own target density in the canonical basis and
changing variables (`m1det → m1src` at fixed `z`, `dL → z`), **every Jacobian
cancels** and

```
Z_i(H0) = Σ_p W_p ∫dz p_z(z|p) (1+z)^(γ−1) L_D(obs_dL | dL(z;H0)) M_i(z)
M_i(z)  = ∫dm1src dm2src [p_mq(m1src, m2src/m1src)/m1src]
                          L_1(obs_m1 | m1src(1+z)) L_2(obs_m2 | m2src(1+z))
```

with `W_p` the pixel mass of the exact sky posterior by Gauss–Hermite quadrature in
the PE's own `(ra, dec)` measure, `p_z` from darksirens' **own**
`eval_redshift_prior_with_state`, and `p_mq` from darksirens' **own**
`mixture.mass_q_density`. Five arms, all on **all 720 GAL and all 280 AGN events**:

| arm | redshift prior | mass measurement model |
|---|---|---|
| `kde_gauss` | darksirens' catalog KDE | stored Gaussian `N(m; obs, f·m_true)` |
| `kde_exact` | darksirens' catalog KDE | exact `N(obs; m, f·m)` |
| `delta_gauss` | zero-bandwidth catalog (no photo-z kernel) | stored Gaussian |
| `delta_exact` | zero-bandwidth catalog | exact |
| `host_exact` | `δ(z − z_host)` | exact |

### It is exact: five anchors

| anchor | GAL | AGN |
|---|---|---|
| `\|Δ log μ\|` vs darksirens | **0** | **0** |
| my reconstruction of `p_z` from the state arrays vs darksirens' evaluator | **2.8e-14** | 2.3e-13 |
| `max dN_miss` (the completion term at `log10n0 = −24`) | 4.3e-17 | 4.3e-17 |
| split-half PE score vs `attr_terms`' `ev_s_fd` | **0** | **0** |
| quadrature: `n_z` doubled / `n_m` doubled / grids shifted 0.37 cell | max \|Δscore\| **5.7e-8 / 2.8e-8 / 2.7e-7** | 6.4e-8 / 1.9e-8 / 1.4e-7 |

and the decisive one — **the oracle reproduces darksirens per event**:

```
mean[ oracle(kde_gauss) − darksirens ]  =  +1.68e-5 ± 1.27e-4   (0.13σ)   GAL
                                          −8.8e-4  ± 1.62e-3   (0.54σ)   AGN
rms of the same difference               =  3.405e-3  vs  darksirens' OWN
                                            per-event MC error 3.455e-3   GAL
                                            (ratio 0.985; Pearson r = 0.976)
```

The per-event Monte-Carlo error of `ln Ẑ_i` was measured directly by splitting each
event's 2000 samples into two disjoint halves (median 1.84e-3, rms 3.46e-3 GAL;
1.53e-2, 2.89e-2 AGN). **The oracle and darksirens differ by nothing but
darksirens' own PE Monte Carlo**, and the observed sem of the difference
(1.27e-4) equals the sem that MC error alone predicts (1.29e-4).

The only quadrature that is not converged to 1e-7 is the *sky* rule; tightening it
(`sky_frac 1e-7`, `n_gh 64`) moves the mean score by +5.4e-6 (GAL KDE arms),
+4.7e-5 (GAL delta arms) and −4.8e-4 (AGN, whose spiky per-pixel prior makes the
sky sum the hardest object here). It moves the *substitution* — the quantity the
attribution is made of — by **−7.6e-7 (GAL) / +1.9e-6 (AGN)**, 0.1 % and 0.3 %.

### The three-variant attribution

`r` per event, against the finite-difference `d ln μ/dH0` (the convention the
oracle's own `(ln Z₊ − ln Z₋)/2dh` score must be differenced against; the record's
own FD value is `−1.4491e-3` against its term-sum `−1.4499e-3`):

| arm | matched GAL | matched AGN |
|---|---|---|
| **`kde_gauss`** (the analysis of record) | **−1.4322e-3** | −2.6673e-3 |
| `kde_exact` | −0.8670e-3 | −2.0444e-3 |
| `delta_gauss` | −1.2928e-3 | −2.6014e-3 |
| **`delta_exact`** (fully exact) | **−0.7284e-3** | −1.9848e-3 |
| `host_exact` (counterpart limit) | +7.61e-3 | −1.32e-2 |

and the same thing as **paired per-event substitutions**, which carry no
Monte-Carlo error at all because both arms use the same quadrature:

| substitution | matched GAL | matched AGN |
|---|---|---|
| exact mass likelihood (KDE prior) | **+5.653e-4 ± 0.332e-4** | **+6.229e-4 ± 0.432e-4** |
| zero-bandwidth catalog prior (stored masses) | +1.394e-4 ± 3.97e-4 | +0.659e-4 ± 8.67e-4 |
| both | +7.038e-4 ± 3.98e-4 | +6.825e-4 ± 8.65e-4 |

**Cross-check of the mass repair by two independent routes on the same events.**
The oracle's `+5.653e-4 ± 0.332e-4` is to be compared with darksirens' own
reweighted-PE arm from stage 2, `+4.833e-4 ± 1.111e-4` — difference
`+8.2e-5 ± 10.1e-5`, **0.8σ** — and with that arm's *mass term* alone,
`+5.630e-4 ± 0.351e-4`, **0.05σ**. Closed-form quadrature and reweighted Monte
Carlo agree on what the named defect is worth.

`host_exact` is listed for completeness only: pinning `z` at the true host's
redshift switches the spectral-siren mass lever off by construction
(`M_i(z_host)` carries no `H0` dependence), so its score is a pure distance term
and is not comparable to the others.

### The closure of `r`, matched GAL

| step | `r` (per event) | share |
|---|---|---|
| record (`attr_terms`, FD convention) | −1.4491e-3 | 100 % |
| oracle anchor `kde_gauss` | −1.4322e-3 | (anchor offset +1.2 %, 0.13σ) |
| − population sampler vs analytic pdf (A1) | +1.3e-8 | **0.001 %** |
| − **named mass-PE defect** | +5.653e-4 | **39.5 %** |
| − catalog's declared photo-z kernel | +1.394e-4 | 9.7 % (0.35σ) |
| **remaining, fully exact measurement model** | **−0.7284e-3** | **50.9 %** |

---

## A4 — combined verdict

1. **The population is exonerated to 1e-8.** `gmd`'s mass sampler and darksirens'
   analytic `powerlaw+peak` agree to a probability-weighted rms of 1.6e-4 in log
   density and to `R ∈ [0.9997, 1.0103]` over the detected set; the resulting
   contribution to `r` is `+1.3e-8`, 0.001 % of the residual.
2. **The named defect is real, is worth 39.5 % of `r` in closed form, and is worth
   +2.15 km s⁻¹ Mpc⁻¹ on the matched-GAL control when the scan is actually rerun.**
   Two independent computations (closed-form quadrature; reweighted PE Monte Carlo)
   agree to 0.05σ on the mass channel.
3. **The catalog's declared photo-z kernel (`dz = 3e-3 (1+z)` on a catalog whose
   redshifts are exact) carries ≈ 10 % of `r`**, `+1.39e-4 ± 3.97e-4` — consistent
   with zero on this realisation, and never more than ~25 % at 1σ.
4. **Half of `r` survives a measurement model that is exact in every channel.**
   `delta_exact` — exact mass likelihood, exact lognormal distance likelihood,
   exact sky posterior by quadrature, zero-bandwidth catalog prior, darksirens'
   own population — still gives `r = −0.728e-3` on matched GAL. Scaled onto the
   five-realisation `r = −1.607e-3 ± 0.136e-3`, that is `≈ −0.82e-3`, a 6σ
   violation of `E[C] = B`. **No choice of per-event measurement model inside this
   family closes the residual.**
5. **Where it must now be.** After A1–A3 the surviving approximations of the
   `delta_exact` arm relative to the generative truth are exactly two, and both
   live in the sky channel:
   * **the nside-32 pixelisation of the catalog prior.** The likelihood uses
     `p_z(z | pix)` with sky and redshift *independent inside a pixel*, while the
     truth places the host at one galaxy with a definite `(ra, dec, z)`. With
     `σ_ang ∈ [1.0°, 2.39°]` against a 1.83° pixel the sky likelihood varies
     substantially across a pixel, so the within-pixel sky–redshift correlation
     that the model discards is not a small quantity. This is the leading suspect.
   * **the RA measurement width.** `observe()` divides `σ_ang` by
     `max(cos dec_TRUE, 0.1)` while `posterior_samples()` divides by
     `max(cos dec_obs, 0.1)` — the *sky twin of the named mass defect*, surviving
     in the RA channel only. Measured on the matched-GAL events, the stored RA
     posterior width is wrong by `|cos dec_obs/cos dec_true − 1|` = **2.2 % mean,
     4.3 % rms, 37 % max** — the one place convention (b) is not actually honoured.
   The decisive next test, if authorised, is the **exact host-galaxy oracle**:
   replace `Σ_p W_p ∫dz p_z(z|p)…` by `Σ_g w_g L_Ω(obs | Ω_g) L_D(obs | dL(z_g;H0)) M_i(z_g)`
   over the real catalog galaxies *with their sky positions*. All the machinery is
   in `attr_oracle.py`; what it needs and the survey block does not carry is the
   galaxies' `(ra, dec)`, i.e. a spatial index over
   `working/data/seed100/catalogs/catalog_{gal,agn}_complete.h5` (151 M rows for
   GAL). One run, ~1 h, no regeneration. It would settle in closed form whether the
   remaining 51 % is the pixelisation.

---

## A5 — recommended fix set, for owner sign-off

**Still nothing is implemented.** This supersedes the ranking in the body above.

1. **Generator — the mass measurement model.** Unchanged from item 1 above, now
   with a measured price: draw the PE masses from the exact flat-prior posterior
   `p(m|obs) ∝ (1/(f m)) exp[−(obs−m)²/(2 f² m²)]` by inverse CDF (the direct
   analogue of PR #335 for `sigma_ang`). **Worth 39.5 % of `r` and
   +2.15 km s⁻¹ Mpc⁻¹ on the GAL control, +0.11 on AGN.** Cost: regenerating the
   events stage of every seed. The naive repair (Gaussian of width `f·obs`) is
   measurably worse; the exact posterior is required.
2. **Generator — the RA measurement width.** One line: use the *observed* `dec` in
   the RA width in `observe()`, so convention (b) holds in the sky channel too.
   The stored RA posterior width is currently wrong by 2.2 % (mean) / 4.3 % (rms)
   / 37 % (max) on the matched-GAL events. Same regeneration as (1), so it costs
   nothing extra if (1) is taken. Its contribution to `r` is not yet measured —
   see A4.5.
3. **Survey — the declared photo-z error.** The mock's catalog redshifts are exact
   but the survey block declares `dz = 3e-3 (1+z)`, and the likelihood smooths the
   prior by a kernel that is not in the data. Either set `dz` to the true (zero /
   floor) value, or scatter the catalog redshifts by the declared error so the
   declaration is honest. **Worth ≈ 10 % of `r`** (`+1.39e-4 ± 3.97e-4`). Cost:
   re-running the survey stage only, no event regeneration.
4. **Diagnose before fixing anything else: the exact host-galaxy oracle** (A4.5).
   Half of `r` is unattributed and the pixelisation is the named suspect; it should
   be measured before any further generator change is authorised.
5. **Analysis convention — quote the dense-catalog result with its curvature.**
   Unchanged from item 3 above, and still the honest framing while item 4 is open.
6. **Upstream `darksirens`.** *Still no defect found.* The estimator now has one
   more, much stronger, certificate: an independent closed-form quadrature of the
   same per-event evidence reproduces `d ln Z_i/dH0` to `+1.7e-5 ± 12.7e-5` per
   event over 720 events, with an rms difference (3.405e-3) equal to darksirens'
   own PE Monte-Carlo error (3.455e-3). The estimator is doing exactly what it
   claims to do.

---

## Files written (appendix)

```
scripts/
  attr_ds_bridge.py         shared, anchored darksirens loader + weight rebuild
  attr_sampler_ratio.py     task 1: 1.2e8 sampler draws, closed-form density, prediction
  make_pe_corrected_events.py task 2: p_pe -> p_pe/rho corrected events copies
  run_fix_scans.sh          task 2: the four repaired matched scans
  attr_oracle.py            task 3: the (m1, m2, dL) quadrature oracle, 5 arms
  run_oracle.sh             task 3: production + 8-run convergence battery
  attr_fix_summary.py       the combined verdict JSON
  attr_fix_figures.py       the three figures

results/
  attr_sampler_ratio.{json,npz}, attr_sampler_draws.npz
  pe_corrected_events.json
  fix_named_defect_{gal,agn}.{h5,json}, fix_named_defect_{gal,agn}_m1.{h5,json}
  attr_oracle_{gal,agn}.{json,npz}
  attr_oracle_{gal,agn}_conv_{nz,nm,sh,sky}.{json,npz}
  attr_fix_summary.json     every number quoted in this appendix

data_derived/
  events_{gal,agn}_hosted_pefix_{m1m2,m1}.h5     copies; working/data untouched

figs/
  attr_sampler_ratio.{png,pdf}      the exact log-ratio map, its validation, the prediction
  fig_before_after_fix.{png,pdf}    record vs reweighted H0 posteriors, both controls
  attr_oracle.{png,pdf}             per-event validation, the arms, the closure ladder
```

Reproduce (one A100; ~20 min CPU + ~35 min GPU + ~50 min oracle):

```bash
python scripts/attr_sampler_ratio.py --ndraw 1.2e8 --nproc 24
python scripts/make_pe_corrected_events.py
./scripts/run_fix_scans.sh
./scripts/run_oracle.sh
python scripts/attr_fix_summary.py
python scripts/attr_fix_figures.py --which all
```
