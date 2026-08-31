# The campaign dataset

`generate_dataset.py` is the single entrypoint that produces **the** dataset every
subsequent gws-agn analysis reads. One master seed fixes every random stream; all
outputs for that seed live under `seed<SEED>/`.

```bash
# everything, one seed
python generate_dataset.py --seed 100

# a single stage
python generate_dataset.py --seed 100 --stage {catalogs,events,surveys,injections,validation,meta}

# another realisation
python generate_dataset.py --seed 101
```

Stages are individually runnable and idempotent (they skip existing outputs unless
`--overwrite` is given), but they must be run **in order** — `surveys` needs the GW
horizon measured by `events`, `injections` needs the pixelated AGN survey, and
`validation` needs everything.

---

## The generative model

### 1. `catalogs` — GLASS lognormal two-tracer catalogs

**v2 (2026-07-30) replaced the tracer amplitudes, the catalog edge and the field
resolution.** v1's `nbar = 1e-2 / 1e-4` arcmin⁻² were arbitrary; v1's edge sat at
`z ≈ 1.565`, far past the data but with a rising `dN/dz` into it; v1's `lmax = 64`
made the planted bias contrast unmeasurable.

CAMB angular matter spectra (`h = 0.6774`, `Om0 = 0.3075`, `Ob0 = 0.0486`, non-linear)
on 200 Mpc linear shells from `z = 0` to `z = 1.0` (16 shells, last edge `z = 1.0016`),
lognormal fields at **`nside = 128`, `lmax = 256`**. **Both tracers are painted on the
same realisation of that field** — GAL with `b = 1.2`, AGN with `b = 2.0`.

**Number densities are comoving and literature-anchored.**

* `n_GAL = 1e-3 Mpc⁻³`. The GLADE-lineage B-band Schechter function
  (`phi* = 1.6e-2 h³ Mpc⁻³`, `alpha = -1.07`, `M_B* = -20.47`, `h = 0.7`; GLADE
  [arXiv:1804.05709](https://arxiv.org/abs/1804.05709), GLADE+
  [arXiv:2110.06184](https://arxiv.org/abs/2110.06184)) gives
  `n(> x L*) = phi* Γ(alpha+1, x)` with `phi* = 5.488e-3 Mpc⁻³`. The nominal
  `x = 0.25` cut gives `Γ(-0.07, 0.25) = 1.08383`, i.e. `n = 5.948e-3 Mpc⁻³` —
  **5.9× the intended 1e-3** and ~9.8e8 rows inside `z ≤ 1`, several times the
  storage budget. The cut is therefore solved *from* the density on the same
  luminosity function: `Γ(-0.07, x) = 0.182216 ⇒ x = 1.0908`, i.e. the sample is
  **`L > 1.09 L*`, `M_B < -20.564`** — the classic `L*` bright-galaxy sample, which
  is what a `1e-3 Mpc⁻³` host catalog physically is. Both numbers are recomputed and
  recorded in `META.json`.
* `n_AGN = 1e-5 Mpc⁻³`, the luminous class `log10 L_X(2–10 keV) ≳ 43.7` from the
  integrated Swift-BAT/BASS X-ray luminosity function (Ananna et al.; BASS DR2
  lineage). Pinned, not re-integrated.
* The ratio `n_GAL/n_AGN = 100` is the same contrast v1 planted arbitrarily, now with
  a reason.

Realised: **151 179 870 GAL** and **1 514 567 AGN**.

**Constant comoving density is enforced exactly.** v1 used
`glass.partition(z_bins, nbar·dndz, shells)` plus `glass.redshifts`. Both are wrong
for a constant-density catalog: `partition`'s NNLS fit minimises the *absolute* L2
residual against `dN/dz ∝ z²`, so its relative error near `z = 0` is unbounded (the
first v2 build came out 7.3× too dense below `z = 0.05` and 21% too dense inside the
GW horizon; v1 had the opposite sign — **no galaxy at all below `z = 0.047`**), and
`glass.redshifts` then spreads a shell's objects `∝ W_i(z)`, i.e. nearly uniformly in
redshift across a 200 Mpc shell, where a constant comoving density needs
`∝ W_i(z) dV_c/dz`. Since `glass.linear_windows` is a **partition of unity** on
`[zb[1], zb[-2]]`, v2 instead sets `N_i = n ∫ W_i(z) (dV_c/dz) dz` and draws `z` inside
shell `i` from `∝ W_i(z)(dV_c/dz)(z)`, giving exactly

```
dN/dz = n (dV_c/dz) Σ_i W_i(z) = n dV_c/dz      for  0.0457 ≤ z ≤ 0.9230
```

with linear ramps over the first and last shell half-widths. Realised density on that
plateau: `1.0000e-3` and `1.001e-5` (≤0.5% in every sub-shell).

Each object carries an absolute magnitude drawn from the **same Schechter function
truncated at the cut** (v1 used a placeholder `N(-21, 1)`), by inverse-CDF on a
log-spaced luminosity grid, and the implied apparent magnitude
`m = M + 5 log10(d_L[pc]/10)` (`d_L` from a 200k-point interpolation table, exact to
<1e-9 relative). **AGN inherit their host galaxy's apparent magnitude**: an AGN lives
in a galaxy, so its absolute magnitude comes from the same galaxy luminosity model
evaluated at the AGN's own redshift. One flux limit therefore thins both tracers with
the same `C(z)`, which keeps the completeness ladder a clean single axis.

Columns are stored **float64** (`CAT_DTYPE`; see the module docstring for why the
campaign moved off float32): the complete GAL catalog is 5.7 GB compressed with the
`z_obs` column.

Adapted from `code/make_mocks.py::create_mock_catalog_glass`.

### 1b. `z_obs` — the catalog's photo-z (D3, 2026-08-01)

Every catalog row also carries

```
z_obs = z + N(0, DZ_SCALE (1+z)),      DZ_SCALE = 3e-3,   sub-seed "photoz"
```

`z` remains the **true** redshift: it drives the host draw and the event's truth.
`z_obs` is what the **survey** records, and `stage_surveys` pixelates `z_obs` with
the declared width `dz = DZ_SCALE (1+z_obs)`.

**WHY.** Before this, the survey block declared `dz = 3e-3 (1+z)` on redshifts copied
**bit-for-bit** from the catalog the hosts were drawn from — a kernel on redshifts
that carry no error at all. `CLOSURE.md` §15.4 measured that as a **7.6σ** violation
of `E[A] = B` in the catalog `p_z` channel (`+5.84e-4 ± 0.84e-4` per event). With the
error realised, darksirens' per-galaxy kernel `g(z) N(z; z_obs, σ)/Z(z_obs)` **is**
the posterior for that galaxy's true redshift given its catalog entry, so
`p_z(z|pix)` is the correct prior for the host's true redshift. Nothing is clipped
at `z_obs ≥ 0` (clipping would censor); the realised count of negative entries is
recorded in `META.json` and gated by validation **V9**.

### 2. `events` — GW sources on the COMPLETE catalogs

Hosts are drawn from the planted mixture `(1-f) GAL + f AGN` with `f_AGN = 0.30`,
uniformly within the chosen tracer, then accepted with weight `(1+z)^(gamma-1)`,
`gamma = 0` (gmd's rate weighting, verbatim). Masses and spins come from darksirens'
own `PopulationConfig` — powerlaw+peak, `v1 = 0.10` (i.e. `w_PL = 0.10`, `w_G = 0.90`),
`alpha = 2.3`, `mmin = 5`, `mmax = 80`, `dm = 3/10`, `mu_G = 35`, `sigma_G = 5`,
`beta = 1.0`, `chi_eff ~ N(0, 0.1)` truncated to `[-1, 1]`. Cosmology `H0 = 67.74`,
`Om0 = 0.3075`, `w0 = -1`, `wa = 0`.

Then **one** observed measurement per source, and both the detection decision and the
PE are built from it. Drawing continues until 1000 detected events.

Since 2026-08-01 the measurement family is **v3** (`--pe_model v3`, the default) —
the literature-standard all-observable family described in full, with its citations
and every derivation, in **`DESIGN_PE.md`**:

```
rho_obs = rho_opt(theta) + N(0, 1)                DETECTION: rho_obs >= 8
sigma_lnMc  = 0.08 * (8/rho_obs)     ln Mc_det_obs ~ N(ln Mc_det, .)
sigma_lnq   = 0.60 * (8/rho_obs)     ln q_obs      ~ N(ln q,      .)
sigma_chi   = 0.20 * (8/rho_obs)     chieff_obs    ~ N(chieff,    .)
sigma_ang   = clip(35 deg / (1.83165 rho_obs), 1, 12)   dec_obs first, then ra_obs
```

`dL` is **not measured separately**: with no projection latent (convention (a))
`rho_opt` is an exact function of `(Mc_det, dL)`, so the SNR **is** the distance
observable — `GWMockCat`'s own construction (Farah et al. 2023, arXiv:2301.00834
App. A). Recording `rho_obs` *and* measuring `dL` would leave a `θ`-dependent
likelihood factor `N(rho_obs; rho_opt(θ), 1)` that darksirens cannot represent.

The PE is the **exact flat-prior posterior** in `(ln Mc_det, ln q, rho, chieff, ra,
dec)`, truncated **only in the prior** (`q ≤ 1`, `rho > 0`, `|chieff| ≤ 1`,
`|dec| ≤ π/2`), and `p_pe = rho/(dL m1det q)` in darksirens' canonical
`(m1det, q, dL, chieff)` basis. **No recorded value is ever clipped** — clipping the
*data* censors the likelihood and gives it a `θ`-dependent normalisation, which is
the defect convention (c2) had to repair in v2.

Realised on seed 100: `rho_obs` median 10.0 (min 8.00), `sigma_lnMc` median 0.064,
`sigma_lnq` median 0.478, `sigma_ln dL` median **0.114** (v2: a flat 0.10),
`sigma_ln m1det` median **0.177** (v2: 0.08), `sigma_ang ∈ [1.0°, 2.39°]` (v2:
identical), and **0 %** of PE samples with `q > 1` (v2: 18.4 %).

**WHY the family changed.** With the exact per-event posterior of the v2 family,
exact host positions, the mock's own host prior and the exact selection function,
`(C − A)` in the mass channel was `−1.274e-3 ± 0.113e-3` — an **11.3σ** violation of
the detected-set score identity (`CLOSURE.md` §15). The v2 mass widths were `f·m_TRUE`,
i.e. functions of the **latent** parameter; no member of that family closes the
identity. In v3 *every* width is a function of **observed** data, so the generative
likelihood is exactly invertible and the identity closes by construction.

The event file also stores a sample of rejected proposals
(`events_rejected_sample.h5`), so the validation can prove the detection rule is a
two-sided deterministic function of the data rather than only checking the survivors.

### 3. `surveys` — flux limits and pixelation

Isotropic apparent-magnitude limits `m < {21, 20, 19, 18}` on the complete catalogs
produce incomplete catalog pairs alongside the complete pair. The limit is isotropic
and there is no hard redshift cut, so `C(z)` is a **consequence** of survey depth
rather than an input — an anisotropic completeness modelled as isotropic would imprint
a sky-density contrast, which is exactly the channel that identifies `f_AGN`.

Every catalog (complete + each limit, both tracers) is then pixelated into a darksirens
survey file with `nside = 32` and KDE width `dz = 3e-3 (1+z)`. The on-disk layout
(`zgals`/`dzgals`/`wgals`/`ngals`, pads `z = 100`, `dz = 1`, `w = 0`) is byte-for-byte
what darksirens expects, but v2 replaces gmd's `_pixelate_catalog` — a per-object
python loop, i.e. 1.5e8 interpreter iterations — with a `lexsort`-based
`pixelate_catalog_vec`. It also emits rows **sorted in z**, which is exactly the
invariant darksirens' windowed catalog-KDE evaluator requires. The complete GAL survey
is `(12288, 14569)`; `recommended_kde_window` returns **3410** at `n_sigma = 8`, so
analyses must configure a window of at least that (the module default 1024 would
nearest-neighbour-truncate the KDE).

### 4. `injections` — two selection campaigns

`1.5e8` proposals (targeted) and `4.0e8` (popuni), under **the same detection rule as
the events**. Injections store TRUE parameters and the exact proposal density `pdraw`
in the canonical `(m1det, q, dL)` basis, because `mu(theta)` is an integral over true
parameters — only the detection *decision* sees the measurement noise.

**Sizing.** v1's `1.2e8` sets were undersized: with `N_obs = 1000` the legacy guard
floor is `N_eff > 5·N_obs = 5000`, and v1 sat at 2 248–8 481 (GAL) and 25–130 (AGN)
across `H0 ∈ [50, 100]`. v2 sizes for **min-over-grid `N_eff ≥ 10 000`** — see
`analyses/analysis_1_complete_catalog_H0/README.md` for the realised numbers.

* **targeted** — three-branch mixture, `0.65 population + 0.10 uniform +
  0.25 AGN-object-targeted`. The targeted branch **reads the pixelated complete AGN
  survey file, not the raw catalog**: that file *is* the object the likelihood
  conditions on (its `zgals`/`dzgals` are literally the KDE centres and widths the
  target density uses), so the proposal and the target cannot drift apart through a
  pixelisation or kernel-width mismatch. Every row's `pdraw` is the exact mixture
  density, whichever branch produced it.

  **v2 makes the targeted branch H0-range covering.** v1 drew `z ~ TN(z_j, sigma_j)`,
  planting injections *on* the catalog kernels of the fiducial cosmology. But the
  likelihood re-reads a stored injection at trial `H0` as the redshift `z'` with
  `dL_fid(z') = dL_fid(z)·H0/H0_FID`, so that branch only overlaps the catalog prior
  near `H0_FID` — the measured cause of v1's three-order-of-magnitude AGN `N_eff`
  collapse away from 67.74. Each retained host now carries a **uniform box**

  ```
  [L_j, U_j] = [ max(0, R_LO (z_j - 4 sigma_j)),  min(0.5, R_HI (z_j + 4 sigma_j)) ]
  R_LO = H0_FID/100 = 0.6774,   R_HI = H0_FID/50 = 1.3548
  ```

  whose image under *every* trial `H0` in `[50, 100]` still contains the host's kernel.
  The density is the flat mixture `(1/N_kept) Σ_{j∈pix} 1[L_j ≤ z ≤ U_j]/(U_j - L_j)` —
  closed form, so `pdraw` stays exact (V6 recomputes it from the flat host list, not
  from the padded table). Hosts deeper than `z = 0.5` are dropped and the box is capped
  at `z = 0.5`; neither can ever be detected, and the population/uniform branches keep
  full support, so no hole is opened in `pdraw`.
* **popuni** — gmd's plain `population+uniform` (Bernoulli 0.9), the cross-check lane.

Batches are drawn in parallel from `SeedSequence(seed).spawn(n_batches)`, so the result
is **independent of the worker count and of completion order**.

### 5. `validation`

See below. Fails loudly.

---

## The three non-negotiable conventions

These are the campaign's three hard-won lessons. Each was a measured, multi-σ bias in
an earlier mock; each is now structural. Do not "simplify" any of them.

### (a) Detection is a deterministic function of the observed data

```
rho_obs = snr_ref * (Mc_det,obs / 30)^(5/6) * (1000 Mpc / d_obs)  >=  8
snr_ref = 6.278363879917771
```

computed from the **same recorded measurement the posterior conditions on**. No
true-redshift cut, no projection latent, no separate noise draw for the PE.

**WHY.** A population likelihood evaluates `prod_i [∫ p(d_i|θ) p(θ|Λ) dθ] / mu^N`,
which is the correct detected-set likelihood *only* when `1[det(d_i)] = 1` on the
observed set. A latent-dependent detection rule leaves an extra `P(det|θ)` **inside**
each event's integral that no population code evaluates — the inference is fine, the
mock is not a draw from it. Measured cost of getting this wrong: H0 recovered at
−1.57 ± 0.18 km/s/Mpc (8.5σ) instead of −0.80 ± 0.16 (darksirens PR #333/#334).

`snr_ref` is the value calibrated in `experiment_matched_mock` so that dropping the
projection latent leaves the detection fraction (and hence the horizon, `z ≈ 0.27–0.31`)
comparable to the historical control arm.

### (b) The sky width comes from the observed amplitude, sequentially

Measure `dL` and the masses **first**, then set
`sigma_ang = clip(35 / rho_opt(observed values), 1, 12) deg` on the `snr_ref = 11.5`
amplitude scale, and only then draw the sky offsets.

**WHY.** `sigma_ang ∝ dL / Mc_det^(5/6)` is itself an H0-sensitive observable. Freezing
it at its **latent true** value makes the recorded sky width carry distance information
that a fixed-width sky posterior cannot represent, which breaks the score identity of
the detected-set likelihood. Measured cost: −0.49 ± 0.08 km/s/Mpc **even under the
exact likelihood** (darksirens PR #335).

Note the two different `snr_ref` scales: `6.2784` sets the **detection** threshold,
`11.5` sets the **sky-width** model. That is the campaign's convention, not an
inconsistency.

### (c) PE samples are the exact flat-prior posterior of that measurement

Distance noise is multiplicative, `ln d_obs ~ N(ln dL, s)` with `s = 0.10`. The
flat-in-`dL` posterior is then **exactly**

```
ln dL ~ N(ln d_obs + s^2, s)
```

— lognormal about the **observation**, shifted `+s^2` by the volume factor. (Derivation:
`p(dL|d_obs) ∝ exp(-(ln d_obs - ln dL)^2 / 2s^2)`; substituting `u = ln dL` brings a
Jacobian `e^u`, and completing the square gives `u ~ N(ln d_obs + s^2, s)`.) The
additively measured channels have constant widths, so a Gaussian centred on the
observed value is their exact flat-prior posterior.

**WHY.** Clouds centred on **truth** stored with `p_pe = 1` are mislabelled as
flat-prior posteriors and inject an `O(sigma^2)` distance-scale — hence H0 — bias.
Measured cost: −1.14 km/s/Mpc at `sigma_dL = 0.10`, scaling as `sigma^2`
(darksirens PR #332).

The closed form makes an inverse-CDF grid unnecessary here; the campaign's inverse-CDF
machinery in `code/generate_gwsamples.py` exists for the **additive**-noise variant
`N(d_obs; dL, fac·dL)`, whose posterior is *not* lognormal. The validation stage
re-derives the posterior numerically and checks the closed form against it
(agreement `< 1e-10` in CDF).

---

## File inventory (`seed<SEED>/`)

| path | what |
|---|---|
| `META.json` | every sub-seed, config, package versions, git SHAs, realised numbers, validation outcomes, full file inventory |
| `catalogs/catalog_{gal,agn}_complete.h5` | complete GLASS tracers: `ra`,`dec` (rad), `z`, `abs_mag`, `app_mag` |
| `catalogs/catalog_{gal,agn}_m{21,20,19,18}.h5` | the same after the isotropic flux limit |
| `catalogs/glass_field_meta.json` | shell edges, GLASS/CAMB config, glass-venv package versions |
| `surveys/survey_{gal,agn}_{complete,m21,m20,m19,m18}_ns32.h5` | darksirens survey files (`zgals`,`dzgals`,`wgals`,`ngals`, attr `nside`) |
| `surveys/surveys_meta.json` | completeness table, empty-pixel fractions |
| `events/events.h5` | `gwcat-1.0`: 1000 events × 2000 PE samples, plus a `truth` group holding **both** the true parameters and the recorded observation (incl. `obs_sigma_ang`) |
| `events/events_rejected_sample.h5` | 20 000 rejected proposals with their measurement (validation aid) |
| `events/events_meta.json` | realised horizon, detected fraction, host bookkeeping |
| `injections/injections_targeted.h5` | `gwcat-selection-1.0`, three-branch targeted mixture |
| `injections/injections_popuni.h5` | `gwcat-selection-1.0`, plain population+uniform cross-check |
| `injections/injections_*_meta.json` | branch bookkeeping and the `P_det(z)` histograms |
| `validation/validation.json` | every check, pass/fail, and the numbers behind it |
| `validation/fig_*.{png,pdf}` | dN/dz + clustering overlay, PE calibration, detection, completeness |

### Events file details

`p_pe` is stored **proportional to `m1det`**, mean-1 per event. The PE prior is flat in
`(m1det, m2det, dL, chieff, ra, dec)` while the likelihood's canonical basis is
`(m1det, q = m2det/m1det, dL, chieff)`, and `dm2det = m1det dq`, so the PE proposal
density in the canonical basis carries that Jacobian. darksirens re-normalises `p_pe`
per event, so only the *shape* matters. A `p_pe_unity` column (all ones) is stored
alongside for exact comparability with the campaign's earlier runs, which used
`p_pe = 1`; the measured difference between the two conventions is
−0.039 ± 0.005 km/s/Mpc in H0 (`experiment_matched_mock/scripts/repe_basis_jacobian.py`).

Events are stored **in draw order** (`host_order = "as_drawn"`), not sorted
gal-then-agn. Any contiguous block is therefore an unbiased sub-realisation, which is
what block-jackknife and disjoint-block noise studies need. `host_type`
(0 = GAL, 1 = AGN), `host_index`, and `true_{z,m1src,m2src,chieff,dL}` are mirrored at
the top level for the campaign's subsampling tools.

---

## Validation (`--stage validation`, exits non-zero on any failure)

| check | what it proves |
|---|---|
| `V1_detection_deterministic_in_data` | **v3**: `rho_obs` is one stored number, every detected event clears the threshold, every stored **rejected** proposal fails it, `rho_true` is reproduced from the truth to `1e-12`, and the *truncated* pull `(rho_obs − rho_opt)/σ_ρ` — a normal truncated at `(8 − rho_opt)` on the detected set — has a uniform PIT. **v2**: `rho_obs` recomputed from the stored observation reproduces the stored value bitwise. Both record the Malmquist/Eddington scatter across the threshold, which only a data-space cut can produce. |
| `V2_widths_from_observed_snr` | **v3**: *every* measurement width — `sigma_lnMc`, `sigma_lnq`, `sigma_chieff`, `sigma_ang` — recomputed from the stored `rho_obs` alone matches bitwise. **v2**: `sigma_ang` recomputed from the stored observables matches bitwise and differs from the truth-derived value. |
| `V3_pe_calibration` | **v3**: the stored measurement-basis columns are the exact bijection of the storage-basis ones (`≤ 1e-11`); pooled and per-event PIT/KS of the PE against the exact (prior-truncated) posterior in every channel — `ln Mc`, `ln q`, `rho`, `chieff`, `dec`, and the RA pull; the measurement-side pulls `(obs − truth)/σ` against `N(0,1)`; and `0 %` of samples with `q > 1`. **v2**: pooled and per-event KS against the analytic flat-prior posterior (`dL`, `m1det`, `m2det`, `chieff`), plus an independent numerical re-derivation of the distance posterior CDF. |
| `V3c_p_pe_jacobian` | **v3, new.** `p_pe` recomputed from the stored samples by the closed form `ρ/(dL m1det q)` matches the stored column to `1e-10`, and that closed form matches an independent **numerical** Jacobian of `(ln Mc, ln q, ρ) ← (m1det, q, dL)` by central differences. This is the check that the PE prior darksirens divides by is the one the samples were actually drawn under. |
| `V9_photoz_realised` | **v3/D3, new.** The survey block's `zgals` are the catalog's `z_obs` (bitwise), are **not** the true redshifts, and the declared `dzgals` equal `DZ_SCALE (1+z_obs)` bitwise; the realised scatter `(z_obs − z)/(DZ_SCALE(1+z))` is `N(0,1)` by KS. |
| `V3b_generative_replication` | an independent 3e6-proposal replication of the whole event draw; two-sample KS on the noise variate and on the detected redshifts. |
| `V4_catalog_densities_and_clustering` | the realised comoving number density on the constant-density plateau against the target (gate 3%, and 10% rms across nine sub-shells), the tracer density ratio, the GAL clustering amplitude per z-shell, and the AGN/GAL bias ratio from a shot-noise-free cross-correlation with jackknife errors. (v1 compared dN/dz against an archived reference; v2's model is different, so the check is internal and stronger.) |
| `V5_planted_f_bookkeeping` | exact host-type counts and host multiplicity of the 1000 events. |
| `V6_injections_and_detection_closure` | `pdraw` independently recomputed on 200 rows per lane; the two lanes' `P_det(z)` agree; the event detection fraction is predicted from the injections' `P_det(z)` and compared. |
| `V7_darksirens_format_contract` | every events / selection / survey file loads through darksirens' own loaders. |
| `V8_catalog_edge_clears_pe_support` | every PE sample of every event is mapped through `z(dL; H0)` at 51 values of `H0` spanning the scanned range `[50, 100]`, and the **maximum** redshift so reached is required to be `< 0.7 z_max`. Also checks the GLASS shell grid reaches `z_max` and that both complete survey files extend past the support. |

---

## Environments

`glass` requires `numpy >= 2` (it calls `np.trapezoid` and the array-API
`__array_namespace__`), while the project `jax` env is pinned at numpy 1.26 / scipy 1.12
— upgrading it would break scipy and the whole inference stack. So the `catalogs` stage
**re-executes `generate_dataset.py` under a dedicated venv**, `.venv_glass/`, and every
other stage runs in the project env. Both interpreters and all package versions are
recorded in `META.json`.

Recreate the glass venv with:

```bash
python -m venv .venv_glass
./.venv_glass/bin/python -m pip install 'glass==2025.1' 'glass.ext.camb==2023.6' camb h5py
```

which pulls `numpy 2.4.6`, `scipy 1.17.1`, `healpy 1.20.0`, `cosmology 2022.10.9`,
`camb 2.0.1`, `astropy 8.0.1`.

The population samplers, cosmology grids, `_selection_pdraw` and `_pixelate_catalog` are
**imported** from darksirens' own `scripts/mock_dark_sirens/generate_mock_data.py`, so
the mock is the inference's model by construction. The checkout and its SHA are recorded
in `META.json` (`--darksirens` to point elsewhere).

---

## Realised v3 numbers (seed 100)

| | |
|---|---|
| GAL / AGN objects | 151 179 870 / 1 514 567 (ratio 99.82) — unchanged, same GLASS field |
| realised comoving density on the plateau | `1.00009e-3` / `1.00183e-5 Mpc⁻³` |
| `b_AGN/b_GAL` (shot-noise-free cross-correlation, jackknife) | as v2 — the field is unchanged |
| GW horizon | `z_max(detected) = 0.3105`, detected fraction `7.850e-3` |
| events | 1000; 705 GAL-hosted / 295 AGN-hosted (`f_AGN = 0.295`) |
| `rho_obs` min / median / max | 8.003 / 10.03 / 240.3 |
| `sigma_lnMc` / `sigma_lnq` median | 0.0638 / 0.478 |
| `sigma_ln dL` median | **0.1137** (`= 1.13/rho`; v2: a flat 0.10) |
| `sigma_ln m1det` median | **0.177** (v2: 0.080) |
| `sigma_ang` | `[1.00°, 2.39°]` — unchanged from v2 |
| PE samples with `q > 1` | **0 %** (v2: 18.4 %) |
| catalog `z_obs` pull sd (GAL / AGN) | 0.99996 / 0.99958; 1 row of 151 179 870 with `z_obs < 0` |
| max PE redshift over `H0 ∈ [50,100]` | **0.652** against the `0.7 z_max = 0.700` bar |
| survey blocks | GAL complete `(12288, 14569)`, `recommended_kde_window` = **3422** at `n_sigma = 8` ⇒ `W = 4096` |
| injections | targeted `1.5e8 → 2 205 380` detected; popuni `4.0e8 → 1 230 471` |
| validation | **12/12 PASS** on every seed (100, 101, 102, 103, 105) |
| total | 9.7 GB / seed |

Per-seed: GAL/AGN event splits 705/295, 674/326, 692/308, 723/277, 711/289;
horizons 0.3105, 0.3338, 0.3133, 0.3519, 0.3868; detected fractions
7.850, 7.415, 7.255, 7.265, 7.310 `× 10⁻³`.

## Known scope limits

* **The catalog is deep and dense, so its dark-siren information is diffuse.** At
  `1e-3 Mpc⁻³` an `nside = 32` pixel holds ~12 300 galaxies to `z = 1` (~1000 inside
  the GW horizon), i.e. ~8 per KDE width — the redshift prior is a smooth `dN/dz`,
  not the comb of narrow spikes v1's 2-galaxies-per-pixel catalog produced. That is
  the physically correct behaviour of a complete `L*` catalog, and it is why
  `Σ σ²_PE` fell from 66.3 (v1) to ~1 for GAL.
* **The AGN tracer is no longer sparse on the sky** — every `nside = 32` pixel is
  occupied — but it is still the sparse *tracer*: ~90 000 AGN inside the GW horizon
  against ~9e6 galaxies.
* **The catalog density ramps to zero over the first and last shell half-widths**
  (`z < 0.0457` and `z > 0.9230`), because `glass.linear_windows` is a partition of
  unity only between them. The low ramp holds ~0.3% of the events; the high ramp is
  far outside the PE support (max 0.655). Both are recorded in `META.json`; the
  density gate is applied on the plateau.
* **The AGN space density is a pinned literature value**, not an integral this code
  performs — the Schechter arithmetic is redone here, the X-ray LF is not.
* **The distance is not measured on its own.** Since v3 the SNR *is* the distance
  observable (`dL = 1000·snr_ref·(Mc_det/30)^{5/6}/rho`), so the realised
  `sigma_ln dL ≈ 1.13/rho_obs` is a *consequence* of `sigma_rho = 1` and
  `sigma_lnMc`, not a free constant. The old worry — that a data-dependent PE width
  is circular in observed-data mode — is resolved by the fact that every width is a
  function of the recorded `rho_obs`, which is itself data.
* **The mass ratio is barely measured**, by design: `sigma_ln q = 0.60·(8/rho_obs)`
  is `≈ 0.48` at the detected median SNR, i.e. a factor `≈ 1.6`. That is the
  literature's number (`DESIGN_PE.md` §2.3) and it is why the component masses carry
  `≈ 18 %` rather than `8 %`, and why the GAL `H0` intervals are wider than v2's.
* **The windowed catalog-KDE evaluator must be configured** for the complete GAL
  survey: `recommended_kde_window` returns 3410 at `n_sigma = 8`, above darksirens'
  module default of 1024. Analyses must also block the selection and PE reductions
  (`sel_batch_size`, `pe_event_block`); a single pass over 2.1e6 injections at
  `W = 4096` needs ~69 GB and OOMs an 80 GB A100.
