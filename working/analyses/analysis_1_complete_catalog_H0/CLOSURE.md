# Closing the matched-host control: the two generator fixes, and what survives them

> **2026-08-01: diagnostic products reorganized under `diagnostics/` (see
> `diagnostics/INDEX.md`); paths in this document refer to the original layout.**

> **CLOSED — 2026-08-01, §16.**  Sections 1–15 diagnose a residual that survived
> every measurement-model repair inside the mock's original family and localise it,
> at 11.3σ, in the posterior-averaging step of the mass channel.  §15.6 recommended
> a redesign; it was authorised, implemented and measured.  Under the v3
> all-observable measurement family (`working/data/DESIGN_PE.md`) — every width a
> function of the observed SNR, the distance derived from it, the PE exact in
> `(ln Mc, ln q, ρ, χ_eff, sky)` — and with the catalog's declared photo-z error
> actually realised (D3), the whole dataset was regenerated and the analysis rerun:
>
> * `(C − A)` in the mass channel: **11.3σ → 1.39σ** (GAL), **10.1σ → 0.95σ** (AGN);
> * `(A − B)`: **6.9σ → 0.38σ** (GAL, on 1.53 M redrawn truths);
> * **matched GAL, five realisations: `−6.30 ± 1.28` → `+0.81 ± 0.62`**, truth
>   inside the 68 % interval on **5 of 5** (was 0 of 5);
> * **matched AGN, five realisations: `+0.71 ± 0.20` → `+0.42 ± 0.47`**.
>
> **Both matched-host controls now sit on truth.**  Every number in §§1–15 describes
> the *previous* dataset and stands as the diagnosis that motivated the redesign.
> Read **§16** for the closure and `DESIGN_PE.md` for the design.


Owner-approved closure campaign, 2026-08-01.  This is the continuation of
`ATTRIBUTION.md` (body + appendix A1–A5), which named two defects in the mock's
measurement model and predicted what repairing them was worth.  Here they are
**implemented**, the events stage of all five realisations is **regenerated**, the
analysis of record is **rerun**, and the one remaining suspect — the nside-32
pixelisation of the catalog prior — is **measured in closed form** against the real
catalog's galaxy positions.

Scope, unchanged from `ATTRIBUTION.md`: `dark_sirens` at `log10n0 = -24`, field sky
weighting, K = 1, targeted injections, `H0 ∈ [50, 100] × 201`, `W = 4096` (GAL), the
campaign guard convention.  **`darksirens` was READ-ONLY at `2b86a2d`** throughout;
the only patches are the same import-level pass-throughs `attr_ds_bridge.py` already
used, re-anchored in every run (`|Δ log μ| = 0` and `|Δ Σ_i ln Ẑ_i| = 0` in every
configuration).  **No paper edits.**

---

## The verdict in one line

**Both named defects are real and both are now fixed; together they are worth
`+6.21e-4` of the per-event score residual `r` — 43 % of it — and
`+3.13 ± 0.84 km s⁻¹ Mpc⁻¹` on the five-realisation matched-GAL control, which moves
from `−9.43 ± 1.98` to `−6.30 ± 1.28`.  The GAL control still does not close.**
What is left is **not** the pixelisation.  The exact host-galaxy oracle — every one
of the 151 M catalog galaxies at its own `(ra, dec, z)`, anchored to reproduce
darksirens per event to `−2.0e-5 ± 1.2e-4` — measures the within-pixel sky–redshift
correlation the likelihood discards at **`+9.1e-5 ± 1.9e-4` per event**, i.e. 11 % ±
24 % of the `−8.29e-4` that survives the fixes.  Together with the catalog's declared
photo-z kernel (`+1.7e-4 ± 4.4e-4`) that leaves **`−5.84e-4`, two fifths of the
original residual, surviving a model that is exact in every measurement channel and
puts every galaxy at its true position.**

**Followed up by the final sweep (§§11–14).**  The three channels that were still
open are now closed and none of them is the residual.  The selection integral —
anchored in every previous run and never verified — is computed in closed form from
the mock's own detection rule and agrees with darksirens' injection estimate to
`0.32σ` of that estimator's own Monte-Carlo error on matched GAL, with the *wrong
sign* to help; the `chi_eff` clip factorises out of the score identity exactly and
never fires; the host-acceptance convention is the mock's own (`Δr = −7.8e-7`).
What the sweep does add is a previously unpropagated error: `μ̂`'s slope carries
`±1.2e-4` (GAL) to `±1.1e-3` (AGN) of Monte-Carlo noise — a **common-mode** shift
of `r` worth `±0.36` and `±0.53 km s⁻¹ Mpc⁻¹`, and about **half of the matched-AGN
`+0.71` pedestal is exactly that**.  **`−6.20e-4` per event, 43 % of the original
residual, survives a model that is exact in every measurement channel, puts every
one of 151 M catalog galaxies at its true position, uses the mock's own host prior,
and uses the exact selection function.**

**Closed by the endgame (§15).**  Redrawing the event stage 2,000 times settles the
`(A − B)` / `(C − A)` split the five realisations could not.  The `[:N_EVENTS]`
truncation and the rejection loop are **innocent** — the kept 1000 and the withheld
tail agree to `−1.1e-4 ± 1.4e-4`.  The residual is `(C − A)` **in the mass channel**:
`−1.274e-3 ± 0.113e-3`, an **11.3σ** violation against an event draw that reproduces
the model's detected-truth mean there to `−4.4e-5 ± 1.2e-5`.  A second, genuine
generator defect is named on the way — the survey block declares
`dz = 3e-3 (1+z)` on redshifts copied bit-for-bit from the catalog the hosts are
drawn from, worth `+5.8e-4` of `(A − B)` — but it cancels in `r` to
`−2.8e-5 ± 2.0e-4`.  **No per-event measurement model in this family closes the
identity; the answer is the mass channel's design.**

---

## 1. The two fixes

Both live in `working/data/generate_dataset.py` and are declared as campaign
conventions in the module docstring and in every seed's `META.json`.

### (c2) — the mass PE is the exact flat-prior posterior

`ATTRIBUTION.md` §"Stage 2 — the named defect".  The realised mass measurement is
`obs ~ N(m, f·m)` with `f` constant, so the width is set by the **latent** mass and
the flat-prior posterior is *not* a Gaussian about `obs`:

```
p(m | obs)  ∝  (1 / (f m)) exp[ −(obs − m)² / (2 f² m²) ]
```

`posterior_samples()` now draws it by inverse CDF.  The change of variables
`y = obs/m` makes the shape **obs-independent**,

```
p(y)  ∝  (1/y) N(y; 1, f)      on   1/40 ≤ y ≤ 1 + 12 f,
```

so one quantile table per `f` serves every event, and integrating it on a uniform
`ln y` grid absorbs the `1/y` exactly, leaving a plain Gaussian integrand.  The `1/m`
tail of the posterior carries `exp(−1/2f²)` = 1.1e-34 (f = 0.08) / 1.9e-22 (f = 0.10)
per e-fold, so the truncation at `m = 40·obs` removes nothing measurable.  The
closed form is exactly the one `scripts/attr_mass_pe.py::log_pex` validated.

### (b2) — the RA measurement width comes from the *observed* dec

`ATTRIBUTION.md` A4.5 / A5.2.  `observe()` divided `σ_ang` by `cos(dec_TRUE)` while
`posterior_samples()` divided by `cos(dec_obs)`, so the recorded RA posterior width
was wrong.  `observe()` now draws `dec_obs` **first**, forms
`sig_ra = σ_ang / max(cos dec_obs, 0.1)` from that recorded value, stores it as
`obs_sig_ra`, and only then draws the RA offset; `posterior_samples()` reads the same
stored number.  Convention (b) now holds in the sky channel too.

The size of the defect removed, measured on each seed as
`|cos dec_obs / cos dec_true − 1|`:

| seed | mean | rms | max |
|---|---|---|---|
| 100 | 2.15 % | 4.44 % | 45.5 % |
| 101 | 2.36 % | 4.78 % | 56.9 % |
| 102 | 2.05 % | 3.74 % | 26.3 % |
| 103 | 2.41 % | 5.03 % | 48.7 % |
| 105 | 2.17 % | 4.22 % | 33.2 % |

---

## 2. What the regeneration did and did not move

Events were regenerated for seeds **100, 101, 102, 103, 105**; catalogs, surveys and
injections were **not** touched.  The justification is recorded in each seed's
`META.json` under `conventions.why_catalogs_surveys_injections_are_untouched`:

* the catalogs precede the events stage;
* the surveys are pixelations of the catalogs and read only the realised horizon `z`
  from the events, which is a function of the **detected set** and is bit-identical;
* the injections never open `events.h5`.  They call `observe(need_sky=False)`, which
  neither fix touches — (c2) acts in `posterior_samples()`, after the event loop, and
  (b2) lives inside the `need_sky` block — and they store TRUE parameters, so
  `μ(θ)` is unchanged.

That is asserted and then *proved*, per seed, by
`working/data/verify_events_regen.py` → `seed*/validation/events_regen_bitcheck.json`:

| seed | detected set bit-identical | detection replay | `ρ_obs` old == new | realised bookkeeping identical | `obs_ra` median / max move | `obs_dec` median / max |
|---|---|---|---|---|---|---|
| 100 | yes | yes | yes | yes | 2.08° / 46.0° | 1.61° / 14.0° |
| 101 | yes | yes | yes | yes | 2.07° / 34.8° | 1.57° / 9.1° |
| 102 | yes | yes | yes | yes | 2.17° / 25.6° | 1.65° / 8.3° |
| 103 | yes | yes | yes | yes | 2.17° / 69.7° | 1.75° / 11.3° |
| 105 | yes | yes | yes | yes | 2.06° / 48.3° | 1.63° / 9.9° |

"Bit-identical" covers every column the detected set is made of and every observable
drawn before or independently of the sky block: `z, ra, dec, dl, m1src, m2src, q,
chieff, m1det, m2det, host_type, host_index, snr_obs, snr_true, obs_dL, obs_m1det,
obs_m2det, obs_sigma_dl, obs_sig_m1, obs_sig_m2, obs_sigma_ang, obs_chieff`.  Only
`obs_ra` and `obs_dec` move, which is what (b2) is: two normal blocks of equal count
reassigned inside the sky draw.  The per-seed event counts confirm it —
720/280, 661/339, 698/302, 735/265, 699/301 GAL/AGN hosts, identical to the record.

`p_pe` is unchanged in both stored conventions.  It is the PE **prior** in the
canonical `(m1det, q, dL, chieff)` basis, `∝ m1det`; (c2) changes the mass
*likelihood*'s shape, not the prior, so the samples are still exact flat-prior
posterior draws and the bookkeeping is untouched (`|p_pe / (m1det/⟨m1det⟩) − 1| = 0`
on every seed, and `p_pe_unity` still holds).

The selection integral is untouched to the last digit: `min N_eff` at the scan level
is **395,349 / 494,877 / 216,057 / 32,979** for the four production configurations
before and after, and the guard diagnostic at truth gives the identical
`N_eff` = 477,688 (GAL targeted), 569,300 (GAL popuni), 353,541 (AGN targeted),
53,683 (AGN popuni) in both.  `pe_variance_sum` moves only through the PE itself
(GAL 1.040 → 0.986, AGN 35.10 → 36.83), far inside the campaign's inert 1e6 cap.

---

## 3. Validation

Every seed passes **all ten** checks of `generate_dataset.py --stage validation`
(nine before; V2b is new).  `n_failed = 0` on 100, 101, 102, 103, 105.

### V3 — PE calibration, now including the exact mass posterior

The PIT of the stored mass samples is taken under the **exact** posterior, on a table
deliberately finer than the sampler's (grid ×2, cap ×10, `n_sig` 12 → 16), so it is
not a self-consistency test of one grid.

| seed | KS `m1det` | KS `m2det` | per-event KS uniformity `m1` / `m2` | numeric-vs-table CDF | table convergence | `⟨m1⟩/obs − 1` | `⟨m2⟩/obs − 1` | clipped |
|---|---|---|---|---|---|---|---|---|
| 100 | 0.924 | 0.290 | 0.359 / 0.171 | 4.3e-10 | 6.8e-08 | 1.313 % | 2.120 % | 0 |
| 101 | 0.404 | 0.890 | 0.752 / 0.350 | 4.3e-10 | 6.8e-08 | 1.320 % | 2.114 % | 0 |
| 102 | 0.065 | 0.801 | 0.582 / 0.435 | 4.3e-10 | 6.8e-08 | 1.320 % | 2.109 % | 0 |
| 103 | 0.632 | 0.591 | 0.257 / 0.165 | 4.3e-10 | 6.8e-08 | 1.317 % | 2.109 % | 0 |
| 105 | 0.959 | 0.150 | 0.425 / 0.018 | 4.3e-10 | 6.8e-08 | 1.320 % | 2.113 % | 0 |

against the closed-form prediction `⟨m⟩/obs = 1 + 2f² + O(f⁴)` = 1.280 % (f = 0.08)
and 2.000 % (f = 0.10); the exact quadrature values are 1.3230 % and 2.1082 %, which
the draws reproduce.  The independent numerical re-derivation builds
`p(m|obs) ∝ (1/(f m)) exp[−(obs−m)²/(2f²m²)]` on a 4e5-point grid **in `m`** — no
change of variables — and matches the `y`-space table the sampler inverts to
**4.3e-10** in CDF.  Refining the table (grid ×2), widening the cap (×10) or the
range (`n_sig` 12 → 16) moves the quantile function by at most **6.8e-8** relative
over the `u` range 2e6 draws can reach.  Nothing is clipped at the 2 / 1 M⊙ bounds
on any seed.

### V2b — the RA width (new check)

`sig_ra` recomputed from the stored `(σ_ang, dec_obs)` equals the stored
`obs_sig_ra` **bitwise** on every seed.  The measurement-side pull
`(ra_obs − ra_true)/sig_ra` and the PE-side pull are both consistent with `N(0,1)`
(pooled KS `p` = 0.31 / 0.97 / 0.93 / 0.45 / 0.21 for the PE RA samples), and the
realised PE RA width ratio sits at 1 to within `−1.30 / +1.12 / −1.01 / −1.12 /
−0.89 σ` on the five seeds — a statistic whose sensitivity is 5e-4, i.e. one that
would have shown the pre-fix 2.2 % error at ~44σ.  Events within 8 `σ_ang` of a pole
(25–45 per seed) are excluded from that pull only, because their dec posterior is
genuinely truncated by the `|dec| ≤ π/2` clip.

---

## 4. The analysis of record, rerun

Seed 100, all six configurations.  `offset` is median − 67.74.

| scan | before | after | shift |
|---|---|---|---|
| `h0_gal_targeted` | 60.098 (−7.642) | **64.115 (−3.625)** | +4.017 |
| `h0_gal_popuni` | 60.150 (−7.590) | 64.554 (−3.186) | +4.404 |
| `h0_agn_targeted` | 99.770 (railed) | 99.619 (railed) | −0.150 |
| `h0_agn_popuni` | 99.833 (railed) | 99.803 (railed) | −0.030 |
| `ctrl_gal_matched` | 62.789 (−4.951) | **64.744 (−2.996)** | +1.955 |
| `ctrl_agn_matched` | 67.391 (−0.349) | **68.451 (+0.711)** | +1.059 |

0/201 cells rejected in every scan, before and after.  The mis-specified AGN
production configurations still rail at the top of the scanned range — that is the
mis-specification (the AGN catalog is handed the 720 events it does not host), and
neither fix addresses it.

`results/h0_single_tracer.json` moves accordingly:
`gal_h0_ci` **60.1⁺²·²₋²·³ → 64.1⁺²·³₋²·¹**, `gal_h0_width` 4.47 → 4.46,
cross-check lane 60.15 → 64.55; the AGN entries stay `null` (railed).
**This is a paper-facing number**: `working/paper/scripts/build_values.py` reads this
file for the `HzeroGal` / `HzeroGalWidth` macros.  It has not been regenerated —
no paper edits were made.

---

## 5. The five-realisation closure table

Same configuration on all five mocks; `ctrl_*_matched_s*` under the estimator of
record (`run_seed_controls.sh` now uses `dark_sirens` at `log10n0 = −24` rather than
`dark_sirens_complete`, which `experiment_model_equivalence` measured to be bitwise
identical — on seed 100 the two give 62.789 and 62.785).

### GAL catalog, matched hosts

| mock | seed | events | before | after | shift | truth in 68 / 90 (after) |
|---|---|---|---|---|---|---|
| 1 | 100 | 720 | −4.951 | −2.996 | +1.955 | no / **yes** |
| 2 | 101 | 661 | −5.603 | −5.021 | +0.582 | no / no |
| 3 | 102 | 698 | −15.795 † | −10.793 | +5.001 | no / no |
| 4 | 103 | 735 | −9.487 | −6.110 | +3.377 | no / no |
| 5 | 105 | 699 | −11.294 | −6.582 | +4.712 | no / no |
| | | | **−9.43 ± 1.98** | **−6.30 ± 1.28** | **+3.13 ± 0.84** | |

`t(4) = −4.92`, `p = 0.008`.  † railed before the fix; **no realisation rails after
it** (mock 3's MAP moves off the `H0 = 50` edge, 51.95 → 56.95).

Two things improved besides the mean.  The realisation-to-realisation scatter falls
from **sd 4.44 to 2.87**, and its ratio to the mean quoted 68 % half-width falls from
**1.89 to 1.06** — i.e. the extra between-mock variance that the posterior width did
not describe is essentially gone, and the error bar is now honest.  Truth enters the
90 % interval on 1 of 5 realisations, against 0 of 5 before.

### AGN catalog, matched hosts

| mock | seed | events | before | after | shift | truth in 68 / 90 (after) |
|---|---|---|---|---|---|---|
| 1 | 100 | 280 | −0.349 | +0.711 | +1.059 | yes / yes |
| 2 | 101 | 339 | +1.553 | +0.767 | −0.787 | yes / yes |
| 3 | 102 | 302 | −0.094 | +1.059 | +1.153 | yes / yes |
| 4 | 103 | 265 | −0.390 | +1.048 | +1.438 | yes / yes |
| 5 | 105 | 301 | −0.463 | −0.047 | +0.416 | yes / yes |
| | | | **+0.05 ± 0.38** | **+0.71 ± 0.20** | **+0.66 ± 0.40** | |

`t(4) = +3.51`, `p = 0.025`.  Truth is inside the 68 % interval on **5 of 5**
realisations (4 of 5 before) and inside the 90 % on 5 of 5, and the realisation
scatter is 0.38× the mean quoted half-width — so the AGN control is calibrated in the
sense that matters for an interval.  Its *mean* offset is nevertheless now a small
positive pedestal, `+0.71 ± 0.20`, which was `+0.05 ± 0.38` before: the fixes moved
AGN up as well, by about `+0.66`, and the AGN control's 21.6× larger per-event `H0`
curvature means that pedestal corresponds to a per-event residual an order of
magnitude larger than GAL's.  See §6.

Figure: `figs/fig_closure_after_fix.{png,pdf}` — per-realisation before/after strip
with the truth line and the two realisation means.  Also updated:
`figs/fig_h0_recovery`, `figs/fig_guard`, `figs/fig_closure_seeds`,
`results/closure_seeds.json`, `results/closure_after_fix.json`.

---

## 6. The per-event score residual, term by term

`scripts/attr_score_terms.py`, rerun on the regenerated events with tag `_postfix`
(the pre-fix products `ATTRIBUTION.md` cites are untouched).  Both runs are anchored
`|Δ log μ| = 0`, `|Δ Σ_i ln Ẑ_i| = 0`.

`r = ⟨d ln Z_i/dH0⟩_events − d ln μ/dH0` at `H0 = 67.74`, per event, seed 100:

| term | GAL before | GAL after | Δ | AGN before | AGN after | Δ |
|---|---|---|---|---|---|---|
| `p_pop` (mass + rate) | −1.5373e-3 | **−9.7719e-4** | **+5.6013e-4** | −1.8729e-3 | **−1.2164e-3** | **+6.5655e-4** |
| catalog `p_z(z\|pix)` | +1.2254e-4 | +1.8384e-4 | +6.13e-5 | +4.969e-5 | +3.1892e-3 | +3.1395e-3 |
| Jacobian | −3.512e-5 | −3.549e-5 | −3.7e-7 | −2.04e-6 | −5.13e-6 | −3.1e-6 |
| **total `r`** | **−1.4499e-3** | **−8.2884e-4** | **+6.2106e-4** | −1.8253e-3 | **+1.9677e-3** | +3.7929e-3 |

**The population term's move is the (c2) prediction, met almost exactly.**  The
quadrature oracle of `ATTRIBUTION.md` A3 predicted the exact-mass substitution at
`+5.653e-4 ± 0.332e-4` (GAL) and `+6.229e-4 ± 0.432e-4` (AGN); regenerating the mock
delivers **+5.6013e-4** and **+6.5655e-4** — 0.1σ and 0.8σ.  Two entirely independent
routes (closed-form quadrature, and a fresh generative draw) agree on what the named
defect was worth.

The rest of the GAL move (`+6.1e-5`) is (b2) and it is small, as expected: the RA
width error is a *sky* mis-specification and GAL's dense catalog makes the sky channel
nearly flat within a pixel.  On the **sparse AGN catalog it is not small**: `r_pz`
moves by `+3.14e-3` there.  That is the same sensitivity `ATTRIBUTION.md` A2 flagged
when it measured `Δr_pz = −1.30e-3 ± 1.11e-3` on AGN as noise on a spiky per-pixel
prior — the AGN catalog has ~120 galaxies per nside-32 pixel against GAL's ~12,300,
so its `p_z(z|pix)` is a spiky object and moving the recorded sky position by ~2°
(the median (b2) move) genuinely re-weights it.  This is why the AGN control acquired
a `+0.71` pedestal while GAL improved.

Converted to `H0` on seed 100's own curvature (`d²logL/dH0²` = −1.578e-1 total,
−2.19e-4 per event, measured on the post-fix `ctrl_gal_matched` grid), `r = −8.29e-4`
implies an offset of **−3.78**; the scan gives **−2.996**.  The linear score
arithmetic under-predicts the improvement by about the same factor it did in
`ATTRIBUTION.md` A2 (predicted −3.30, measured −2.80).

---

## 7. The decisive test: the exact host-galaxy sky oracle

`scripts/build_catalog_skyindex.py` + `scripts/attr_sky_oracle.py` +
`scripts/run_sky_oracle.sh` → `results/attr_sky_oracle_{gal,agn}.{json,npz}`
(+ the convergence battery), `figs/fig_sky_oracle_{gal,agn}.{png,pdf}`.

`ATTRIBUTION.md` A4.5 named the nside-32 pixelisation as the leading suspect for what
survives an exact measurement model: the likelihood carries `p_z(z | pix)`, i.e. sky
and redshift **independent inside a 1.83° pixel**, while the truth puts the host at
one galaxy with a definite `(ra, dec, z)`.  With `σ_ang ∈ [1.0°, 2.39°]` the sky
likelihood varies substantially across a pixel, so the discarded within-pixel
sky–redshift correlation need not be small.  This section measures it.

### The construction

The survey blocks throw the galaxies' sky positions away.
`build_catalog_skyindex.py` puts them back **in the survey block's own row order** —
`np.lexsort((z, pix))`, exactly what `generate_dataset.pixelate_catalog_vec` used — so
galaxy `(row, column)` of darksirens' state arrays is paired with its own
`(ra, dec)`.  The index is re-verified **bitwise** against `cat_pe.zgals` on 200
random rows at the start of every oracle run (0 failures), and
`ang2pix(stored positions)` reproduces the pixel assignment for all 151,179,870 GAL
and 1,514,567 AGN rows.

Every arm writes darksirens' own per-event evidence in the canonical basis, where all
Jacobians cancel, with the **exact** mass likelihood `N(obs; m, f m)` (which since
(c2) is also exactly what the stored PE encodes), the exact lognormal distance
likelihood, and darksirens' own `p_mq`, `kw_g`, `N_obs(p)`, `g(z)` and `Z_global`.
The only thing that changes between the two decisive arms is the sky weight:

```
delta_pix    w_g = <u(Ω)>_{Ω in pixel(g)}      the pixel AVERAGE  (what the model uses)
delta_host   w_g = u(Ω_g)                      the galaxy's OWN value (the truth)

u(Ω) = N(ra; ra_obs, sig_ra) · N(dec; dec_obs, σ_ang) / cos(dec)
```

`u` is the sky posterior density **per steradian**; the `1/cos(dec)` converts from
the PE's own `(ra, dec)` product measure, in which both Gaussians are written, to
solid angle.  Both arms estimate it the same way — an equal-solid-angle average over
the `4^5 = 1024` HEALPix children of each pixel — so `delta_host − delta_pix` is
exactly and only *the within-pixel sky structure the pixelisation throws away*, on the
same galaxies, with no aperture, normalisation or quadrature difference between them.

### It is exact: the anchors

| anchor | matched GAL | matched AGN |
|---|---|---|
| `\|Δ log μ\|` vs darksirens | **0** | **0** |
| sky index vs `cat_pe.zgals`, bitwise, 200 rows | **0 bad** | **0 bad** |
| `max dN_miss` (the completion term at `log10n0 = −24`) | 4.3e-17 | 4.3e-17 |
| sub-pixel sky rule, one more refinement level | 1.5e-3 max \|ΔW/W\| | 4.1e-3 |
| sub-pixel sky rule vs 4×10⁶ draws from the PE's own sky measure | max 3.2 σ_MC | max 2.9 σ_MC |
| retained sky mass per event | ≥ 0.99998 | ≥ 0.99998 |
| every event's own true host inside the aperture | 720/720 | 280/280 |

and the decisive one — **the oracle reproduces darksirens per event**:

```
mean[ oracle(kde_pix) − darksirens ]  =  −1.96e-5 ± 1.19e-4   (0.16 σ)   GAL, 720 events
rms of the same difference            =   3.188e-3   vs darksirens' OWN
                                          per-event PE Monte-Carlo error 3.431e-3
                                          (ratio 0.93; Pearson r = 0.979)
```

The candidate aperture is **grown until it holds the whole sky posterior** (`Σ_p W_p`
is itself the coverage test), so the answer does not depend on it: moving `n_ap` from
4 to 6 to 8 changes the pixelisation substitution by **0** to five digits.  The only
sky mass the smooth rule ever misses is the generator's own `|dec| ≤ π/2` **clip** on
the PE dec samples — verified: `Σ_p W_p` equals `P(|dec| ≤ π/2 | data)` to
**4.1e-3** on every event, and only 4 of 720 GAL events (and 4 of 280 AGN) sit within
3 `σ_ang` of a pole.  Excluding them changes nothing: the GAL pixelisation term goes
from `+9.06e-5 ± 1.95e-4` to `+7.42e-5 ± 1.95e-4`.

Convergence battery, 120 events, every knob moved one at a time, quoted as the
**relative** change it makes to the pixelisation substitution:

| knob | AGN | GAL |
|---|---|---|
| `n_ap` (aperture radius before the growth loop) | **0** (4 and 8) | **0** (4) |
| `sky_frac` 1e-5 / 1e-7 | 2.0e-4 / 4.0e-6 | 2.1e-3 / 1.3e-4 |
| `n_sub` 4 / 6 (the sub-pixel sky rule) | 2.6e-4 / 8.4e-5 | 1.0e-2 / 2.5e-3 |
| `n_z` 1024 | 2.0e-6 | 9.1e-8 |
| `n_m` 384 | 2.4e-8 | 2.3e-7 |
| grids shifted 0.37 cell | 1.6e-6 | 4.0e-6 |

The largest entry, 1 % from halving the sky rule's refinement on GAL, is 5e-7
absolute against a statistical uncertainty of 3.6e-4 on the same 120 events.  The
quadrature is converged; the substitution's uncertainty is entirely the finite number
of events.

### The result

`r` per event, matched GAL, 720 events, against the finite-difference `d ln μ/dH0`:

| arm | redshift prior | sky weight | `r` |
|---|---|---|---|
| record, post-fix (darksirens) | catalog KDE | PE samples | **−8.2916e-4** |
| `kde_pix` (the oracle anchor) | catalog KDE | pixel average | −8.4874e-4 |
| `delta_pix` | zero-bandwidth catalog | pixel average | −6.7497e-4 |
| `kde_host` | catalog KDE | **exact galaxy positions** | −7.7499e-4 |
| **`delta_host`** (fully exact) | zero-bandwidth catalog | **exact galaxy positions** | **−5.8433e-4** |

and as **paired per-event substitutions**, which carry no Monte-Carlo error at all:

| substitution | matched GAL | matched AGN |
|---|---|---|
| **nside-32 pixelisation** (`delta_host − delta_pix`) | **+9.063e-5 ± 1.95e-4** | −1.558e-3 ± 0.90e-3 |
| the same with the KDE prior (`kde_host − kde_pix`) | +7.376e-5 ± 1.56e-4 | −1.537e-3 ± 0.85e-3 |
| catalog photo-z kernel (`delta_pix − kde_pix`) | +1.738e-4 ± 4.36e-4 | +1.491e-4 ± 9.12e-4 |
| both | +2.644e-4 ± 4.73e-4 | −1.409e-3 ± 1.34e-3 |

**The pixelisation is not the remaining term.**  On matched GAL it is
`+9.1e-5 ± 1.9e-4` — consistent with zero, at most `+4.8e-4` at 2σ, i.e.
**11 % ± 24 %** of the `−8.29e-4` that survives the two generator fixes.  Where it
does show up is exactly where it should: the per-event substitution is
indistinguishable from zero for `σ_ang ≳ 1.3°` and is only non-negligible for the
best-localised events, which pile up at the `σ_ang = 1.0°` clip floor
(`figs/fig_sky_oracle_gal.png`, panel b).  Those events are a small minority and
their contributions cancel in sign.

On the **sparse AGN catalog it is not small** — `−1.56e-3 ± 0.90e-3`, and `−1.69e-3 ±
0.90e-3` excluding polar events — which is the physically sensible result: the AGN
survey carries ~120 galaxies per nside-32 pixel against GAL's ~12,300, so its
`p_z(z|pix)` is a spiky object and where inside the pixel the host sits matters.  But
it has the *wrong sign* to explain a negative residual, and the AGN oracle's own
anchor is weak (`−3.29e-3 ± 2.81e-3` against darksirens, 1.2σ, on a per-event PE
Monte-Carlo error of 3.2e-2), so AGN is corroborative only.

### The closure of `r`, matched GAL

| step | `r` per event | share of the original |
|---|---|---|
| record, pre-fix (`attr_score_terms`, FD convention) | −1.4491e-3 | 100 % |
| − population sampler vs analytic pdf (`ATTRIBUTION.md` A1) | +1.3e-8 | 0.001 % |
| − **(c2) the exact mass PE** | **+5.6013e-4** | **38.7 %** |
| − **(b2) the RA width from the observed dec** | **+6.09e-5** | **4.2 %** |
| **= the analysis of record, post-fix** | **−8.2916e-4** | **57.2 %** |
| oracle anchor `kde_pix` | −8.4874e-4 | (anchor offset −2.0e-5 per event, 0.16σ) |
| − the catalog's declared photo-z kernel | +1.738e-4 | 12.0 % (0.40σ) |
| − the nside-32 pixelisation | +9.063e-5 | 6.3 % (0.47σ) |
| **remaining: exact measurement model, exact host positions** | **−5.8433e-4** | **40.3 %** |

**Two fifths of the original residual survives a model that is exact in every
measurement channel and puts every catalog galaxy at its true `(ra, dec, z)`.**  That
is the honest state of the closure: the two named defects were real and are gone, the
two remaining modelling approximations in the catalog prior are both consistent with
zero, and what is left is not attributable to any of them.

---

## 8. The residual against survey resolution

`scripts/build_nside_surveys.py` → `results/surveys_nside.json`;
`scripts/run_nside_scans.sh` → `results/ctrl_{gal,agn}_matched_ns{64,128}.{h5,json}`;
`figs/fig_nside_curve.{png,pdf}`.

**The oracle's trigger condition for this step was NOT met** — the pixelisation is
consistent with zero on matched GAL — so this is not the design measurement the step
was written for.  It is run instead as the oracle's **falsification test**: the oracle
predicts that shrinking the pixel can buy at most
`+9.06e-5 / 2.19e-4 = +0.41 ± 0.89 km s⁻¹ Mpc⁻¹`, so the curve must be flat.  It is.

Seed 100's complete catalogs were re-pixelated at nside 64 and 128 with
`generate_dataset.pixelate_catalog_vec` verbatim, the same `dz = 3e-3 (1+z)`
convention, the same padding sentinels and the same float64 dtype.  **The record
surveys at nside 32 were not replaced**; the new blocks live on the bulk allocation
under `derived/analysis_1_complete_catalog_H0/surveys_nside/`.  `W` was held at 4096
for GAL at every resolution so that nothing but the pixel size changes along the
curve; the re-measured requirements are 3410 (nside 32), **986** (64) and **293**
(128) at `n_sigma = 8`, so 4096 clears all three.  The injections carry TRUE sky
positions and darksirens re-pixelates them at load time, so no selection campaign
was regenerated.

| catalog | nside | pixel side | galaxies / pixel (max) | offset | 68 % half-width | rejected cells | min `N_eff` |
|---|---|---|---|---|---|---|---|
| GAL | 32 | 1.83° | 14,569 | **−2.996** | 2.36 | 0 | 3.95e5 |
| GAL | 64 | 0.92° | 4,023 | **−3.393** | 2.07 | 0 | 2.15e5 |
| GAL | 128 | 0.46° | 1,085 | **−3.783** | 2.11 | 0 | 8.05e4 |
| AGN | 32 | 1.83° | 178 | +0.711 | 1.22 | 0 | 2.16e5 |
| AGN | 64 | 0.92° | 63 | +1.054 | 0.84 | 0 | 5.63e4 |
| AGN | 128 | 0.46° | 24 | −1.417 † | 0.97 | 0 | 1.44e4 |

**The GAL residual does not go away as the pixel shrinks.**  Over a factor 4 in
resolution — 16× in pixel area, 13× fewer galaxies per pixel — the offset moves by
`−0.79`, in the *unhelpful* direction and small against the ±2.1–2.4 half-width, while
closure would need `+3.0`.  The oracle's own `nside → ∞` limit, `−2.58 ± 0.89`, sits
inside the band the measured curve traces.  Two independent methods — a closed-form
paired substitution on the exact galaxy positions, and four extra `H0` scans on
physically re-pixelated surveys — agree that the pixelisation is not the residual.

† The AGN nside-128 point should not be read as a resolution trend.  At 0.46° the AGN
survey holds ~12 galaxies per occupied pixel with 0.07 % of pixels empty, and the
targeted injection lane — whose proposal was designed around the nside-32 AGN pixel
support — loses a factor 15 of selection convergence (`min N_eff` 2.16e5 → 1.44e4,
still 10× the guard threshold, but no longer the well-converged object the record
uses).  The AGN curve at nside ≥ 128 measures the sparse-catalog limit, not the
pixelisation.

---

## 9. Where the residual is now, and what it is not

After this campaign the matched-GAL residual is `r = −8.29e-4` per event
(`−6.30 ± 1.28 km s⁻¹ Mpc⁻¹` over five realisations), and the following are excluded
**by measurement, on the analysis of record**:

| candidate | measurement | status |
|---|---|---|
| the mass PE convention (c2) | worth `+5.60e-4`; predicted `+5.65e-4 ± 0.33e-4` by closed-form quadrature | **fixed** |
| the RA measurement width (b2) | worth `+6.1e-5` (GAL), `+3.1e-3` (AGN `r_pz`) | **fixed** |
| the population mass sampler vs darksirens' analytic pdf | `+1.3e-8` (`ATTRIBUTION.md` A1) | 0.001 % |
| the catalog's declared photo-z kernel | `+1.738e-4 ± 4.36e-4` | 21 % of `r`, 0.40σ |
| the nside-32 **pixelisation** | `+9.06e-5 ± 1.95e-4` (oracle); `−0.79` over nside 32→128 (four scans) | **11 % ± 24 %, and the wrong sign in the scans** |
| the selection integral | `N_eff` identical to the last digit before and after; `min N_eff` 6.6–99× the threshold (79.1 / 99.0 / 43.2 / 6.6× for the four production configurations, unchanged by the fix); 0/201 cells rejected everywhere | not implicated |
| the selection integral, **measured in closed form** (§11) | `d ln μ/dH0` exact vs estimated: `−3.9e-5` = `0.32σ` of the estimator's own MC error (GAL, the record's lane), `−7.7e-4` = `1.4σ` (AGN); `P_det` validated against the generator's own `observe()` to `1e-4`, `F(z)` against `4.6e8` of its own injection draws to `0.1 %`, `μ(truth)` against a replay of the event loop to `0.3 %` | **verified — not the GAL residual, and the wrong sign** |
| the injection estimator's **own Monte-Carlo error** on `d ln μ/dH0` (§14.2) | **new**: `1.20e-4` (GAL) / `5.58e-4` (AGN targeted) / `1.13e-3` (AGN popuni) — a **common-mode** shift of `r` the per-event `sem` does not describe | `±0.36` / `±0.26` / `±0.53 km s⁻¹ Mpc⁻¹`; **half the AGN `+0.71` pedestal** |
| the `chi_eff` measurement model (§12) | the spin factor factorises out of `Z_i` and `μ` to `8.5e-14`; the clip never fires (`6.9σ` away, 0/2,000,000 samples) | **closed channel**, `Δr = +3.8e-6` |
| the host-acceptance convention (§13) | `γ_fid = GAMMA = 0`, `rate_gmax = 1`, `w_g = 1`, `N_obs = Σw`, `Z_global = ΣN_obs`; the finite-bandwidth residue is measured | **conventions match**, `Δr = −7.8e-7 ± 1.7e-6` |
| finite-`nsamp` PE Monte Carlo | darksirens' own per-event `σ` on `d ln Ẑ_i/dH0` is 3.43e-3 rms; the oracle differs from it by 3.19e-3 rms, i.e. by nothing else | 2–3 % of `r` |
| the estimator `darksirens` | an independent closed-form quadrature reproduces its per-event score to `−2.0e-5 ± 1.2e-4` over 720 events, with an rms difference *below* its own Monte-Carlo error | **no defect found** |

Two remarks on what is left.

**It is no longer localised.**  Pre-fix, `r` was 105 % the population term's
source-frame-mass piece, and the paired `(C − A)` statistic in the mass sector was a
5.3σ violation.  Post-fix the split is `r_pop = −9.77e-4`, `r_pz = +1.84e-4`,
`r_jac = −3.5e-5`, and the truth-point decomposition
`r = (C − A) + (A − B) = −4.42e-3 ± 3.16e-3 + 3.59e-3` has no discriminating power on
one realisation: `A`'s own event-to-event scatter (sem 3.2e-3) is four times `r`.
Separating the two pieces again would need the five-realisation statistic that
`ATTRIBUTION.md`'s probe 3 used.

**The oracle cannot see the catalog prior's own construction.**  Every arm above
inherits darksirens' `kw_g`, `N_obs(p)`, `Z_global` and `g(z) = dV_c/dz · (1+z)^δ`
wholesale — that inheritance is precisely what makes `kde_pix` an anchor, and it means
the oracle tests the *measurement* model and the sky/redshift *discretisation* but not
the prior's normalisation.  Part of that is already excluded by arithmetic: with
`Om0, w0, wa` pinned, `dV_c/dz(z; H0) = H0⁻³ F(z)` exactly, so in the delta arms —
where each galaxy's `z_g` is FIXED and only `dL(z_g; H0)` moves — the volume factor
contributes the *same* `−3/H0` to every event and cancels against the identical
constant in `d ln μ/dH0`.  What does not cancel is its `z`-dependence through `F(z)`,
and that already sits inside the `r_pz = +1.84e-4` the term split measures.

The decisive next tests, if authorised, are therefore **not** further measurement-model
work.  In order of expected information per unit compute:

1. **Re-measure `r` and both controls on all five realisations post-fix** with
   `probe3_decomposition.py`, to recover the five-seed `r = ⟨r⟩ ± sem` that made the
   pre-fix statement 12σ, and to re-split `(C − A)` against `(A − B)` with the
   realisation-averaged precision the one-realisation split has lost.
2. **The catalog prior's own weights.**  One extra oracle arm — the same
   `delta_host` sum with `kw_g · g(z_g)` replaced by the generative prior the mock
   actually realises (uniform over catalog galaxies, times `(1+z)^(γ−1)`) — would say
   in closed form whether darksirens' complete-catalog prior is the mock's host prior.
   `attr_sky_oracle.py` needs one flag and one ~45-minute run.
   **Done — §13: the conventions match, `Δr = −7.8e-7 ± 1.7e-6`.**
3. **The mass-channel design.**  `ATTRIBUTION.md`'s option 2 is untouched and is a
   scope decision, not a bug fix: `σ/m = 8–10 %` against a `35 ± 5 M⊙` peak makes the
   spectral-siren lever strong and strongly curved, and the dense-catalog
   configuration has 21.6× less `H0` curvature per event than AGN with which to
   fight it.

---

## 10. Recommended final configuration for the campaign

1. **Keep the regenerated dataset.**  Conventions (b2) and (c2) are now structural in
   `generate_dataset.py`, gated by validation V2b and V3, and the regeneration is
   proved to have left the detected set, the surveys and both selection campaigns
   bit-identical.  Every future seed inherits them.
2. **Keep `dark_sirens` at `log10n0 = −24`, field weighting, `W = 4096`, the campaign
   guard convention, and the targeted lane** — unchanged, and now used for the
   five-seed table as well, so one estimator carries every number in the campaign.
3. **Keep the survey at nside 32.**  The resolution study says the pixel size buys
   nothing: `−3.00 → −3.39 → −3.78` from nside 32 to 128, against a 16× smaller pixel
   and a 4× larger survey file.  Higher resolution also degrades the AGN selection
   integral badly (`min N_eff` 2.16e5 → 1.44e4) because the targeted lane's proposal
   is built on the nside-32 AGN pixel support.  If the resolution is ever raised, the
   injection lane must be rebuilt with it.
4. **Set the survey's declared photo-z error honestly.**  Unchanged from
   `ATTRIBUTION.md` A5.3 and now measured twice: the mock's catalog redshifts are
   exact while the survey block declares `dz = 3e-3 (1+z)`, and that kernel is worth
   `+1.74e-4 ± 4.36e-4`, ≈ 21 % of the remaining `r`.  Either set `dz` to the true
   value or scatter the catalog redshifts by the declared error.  Cost: the survey
   stage only — no event regeneration, no injection regeneration.  It is the cheapest
   remaining item and the only one of the three modelling terms with a sign that helps.
5. **Quote the dense-catalog result with its curvature.**  Unchanged from
   `ATTRIBUTION.md` A5.5 and still the honest framing: the GAL configuration converts
   a per-event score residual into `r / |d²|` with `|d²| ≈ 2.2e-4` per event, an
   amplification of ~4600 km s⁻¹ Mpc⁻¹ per unit `r`, and nothing in the estimator's
   quoted half-width describes that.  The five-realisation scatter is now 1.06× the
   quoted half-width (it was 1.89×), so the *width* is honest; the *mean* is not yet.
6. **Regenerate the paper's values before quoting anything.**
   `results/h0_single_tracer.json` has moved (`gal_h0_ci` 60.1 → 64.1) and
   `working/paper/scripts/build_values.py` reads it.  No paper file was touched here.
7. **Carry the selection integral's own Monte-Carlo error** (added by §14.2).  Every
   control's offset inherits a common-mode `σ_MC(d ln μ/dH0)/|d²|`: `±0.36` (GAL) and
   `±0.26` to `±0.53 km s⁻¹ Mpc⁻¹` (AGN), independent per realisation because each
   seed has its own injection campaign.  It is not in any quoted half-width.  Two
   cheap options: quote it as an extra term, or shrink it — the AGN estimate is noisy
   because its `p_z(z|pix)` is spiky, so an injection campaign a few times larger (or
   a targeted lane matched to the AGN kernel rather than to its pixel support) would
   cut it by `√N_draw`.  **Do not "fix" the estimator: it is unbiased** (§11.4).

---
---

# The final sweep — the selection function, the `chi_eff` clip, the host prior

Owner-approved, 2026-08-01.  Three channels were still open after §10: the
selection integral had only ever been *anchored* to darksirens' injection
estimate and never *verified*; the generator's `chi_eff` clip had never been
substituted for its exact censored likelihood; and the host-acceptance convention
had never been compared between the mock and the estimator on this
configuration.  Same scope throughout — `dark_sirens` at `log10n0 = −24`, field
weighting, K = 1, `H0 ∈ [50, 100]`, `W = 4096` (GAL), seed 100, both matched
controls.  **`darksirens` READ-ONLY at `2b86a2d`; no generator edits, no dataset
changes, no paper edits.**  The one code change outside `scripts/` is an
**opt-in** flag on `attr_sky_oracle.py` (`--host_prior_arms`); without it every
product of §7 is bit-identical.

## 11. The selection integral, verified

`scripts/attr_selmu_pdet.py`, `attr_selmu_oracle.py`, `attr_selmu_inj.py`,
`attr_selmu_gencheck.py`, `attr_selmu_gconv.py`, `attr_selmu_summary.py` →
`results/attr_selmu_*.{json,npz}`; `figs/fig_selmu_oracle.{png,pdf}`.

An anchor proves that a standalone rebuild evaluates the same operands as the
likelihood.  It says nothing about whether the number both of them produce is the
integral the mock realises — and `r` is a *difference* of two large numbers,
`⟨d ln Z_i/dH0⟩ = 4.10e-2` against `d ln μ/dH0 = 4.19e-2`, so a 1.4 % error in the
second would be the whole residual.  The mock's detection rule is closed form, so
the second number does not have to be estimated at all.

### 11.1 `P_det` in closed form

`observe()` records `obs_dL = dL·exp(s·N(0,1))` — i.e. `ln obs_dL ~ N(ln dL, s)`
with `s = SIGMA_DL = 0.10` and **no** `−s²/2` — and `obs_m = clip(N(m, f·m), 2/1 M⊙)`
with `f = 0.08 / 0.10`.  `detect_from_observation()` reads **only**
`obs["m1det"], obs["m2det"], obs["dL"]`, and `snr_amplitude`'s signature is
`(m1det, m2det, dl, snr_ref)`: **`chieff` never enters** (checked twice — by
signature, and by `ρ_obs` being *bit-identical* when `chieff` is moved from 0 to
0.9 on the same RNG stream).

Conditioning on the observed masses makes the distance integral an error function
exactly, and writing `obs_m1 = m1det(1+f₁x₁)`, `obs_m2 = m2det(1+f₂x₂)` the chirp
mass factorises, so

```
P_det(m1det, m2det, dL) = E_{x1,x2}[ Phi( t + (5/6) ln R(x1,x2;q)/s ) ]
t = ln(rho_true/8)/s,   R = (a1 a2)^0.6 ((1+q)/(a1 + q a2))^0.2,   ai = 1 + fi xi
```

— a function of **two** variables, `(t, q)`, with a 2-D Gauss–Hermite outer rule.

| check | result |
|---|---|
| Gauss–Hermite node doubling, `n = 12 / 24 / 48 / 96` against `192` | `2.2e-11 / 5.6e-16 / 5.6e-16 / 4.4e-16` |
| the `(t, q)` reduction: masses rescaled ×0.5 … ×4 at fixed `(t, q)` | max `\|ΔP\| = 1.8e-15` |
| probability the 2 / 1 M⊙ clip is active for `m ≥ 5 M⊙` (the population's own support) | `3.2e-14` (`m1`), `6.2e-16` (`m2`) |
| **brute force against the GENERATOR's own `observe()` + `detect_from_observation()`** — 30 `(m1det, m2det, dL)` points spanning `P_det ∈ [0.003, 0.995]`, `4×10⁷` draws each | max `\|P_MC − P_quad\| = 9.8e-5`; pulls `−0.41 ± 1.04`, max `2.95`; mean difference `+1.2e-6 ± 5.6e-6` |

### 11.2 `μ(H0)`, and two exact reductions

With the Jacobians cancelled in the canonical basis (`ATTRIBUTION.md` A3),

```
mu(H0)   = SUM_p INT dz p_z(z|p) (1+z)^(gamma-1) F(z;H0)
F(z;H0)  = INT dm1src dq p_mq(m1src,q) P_det(m1src(1+z), q m1src(1+z), dL(z;H0))
```

(the `chieff` channel integrates to 1 exactly — see §12).  Two reductions remove
every interpolation in `H0`:

* `t = a(m1src, q) + b(z, H0)` separates, so `F(z;H0) = G(b(z,H0))` with `G` the
  CDF of the **one-dimensional** `W = ε − V`; `G′` comes from the same
  construction with the normal PDF in place of the CDF, so `dμ/dH0` is
  **analytic**, not a finite difference.
* `dL(z;H0) = (H0_fid/H0)·dL(z;H0_fid)` **exactly** (checked against darksirens'
  own `dL_of_z` to `1.5e-15`), so `b(z,H0) = b(z,fid) + ln(H0/H0_fid)/s` and
  `d ln μ/dH0 = (1/(s H0))·⟨G′(b)⟩/⟨G(b)⟩` over the *same* host measure at every
  `H0`.

The volume factor cancels **exactly**: with `volume_weighted = False`,
`kw_g = (1/n_pix)/Z(z_g)` and the evaluator reapplies `g(z)` in front, so `g/Z`
carries no `H0`.  `N_obs[p] = ngals[p]` and `Z_global = Σ_p ngals[p]`, so the
per-pixel amplitude cancels the per-row kernel normalisation and the host measure
collapses to a one-dimensional object over the catalog's redshifts.  **All
151,179,870 GAL and 1,514,567 AGN galaxies enter**, on a `Δz = 10⁻⁶` lattice with
per-bin first moments.  Four host measures are carried: `kde` (darksirens' own
catalog KDE — what the likelihood conditions on), `delta` (its zero-bandwidth
limit), `unif` (uniform over catalog rows × `(1+z)^(γ−1)` — the mock's own
generative host prior) and `norate` (the rate-convention lever).

### 11.3 It is exact: the anchors

| anchor | GAL | AGN |
|---|---|---|
| `N_obs[p] − ngals[p]`, max abs | `1.3e-11` | `8.5e-14` |
| `Z_global − Σ_p ngals[p]` (and `Z_global` carries no `H0`, field convention) | `6.0e-8` | `1.4e-9` |
| `max dN_miss` (the completion term at `log10n0 = −24`) | `4.3e-17` | `4.3e-17` |
| my 1-D `log Z(z)` table against darksirens' own `log_kw`, 4000 rows | `3.1e-7` | `1.3e-7` |
| `g(z;H0)/Z(z;H0)` between `H0 = 50 / 100` and truth | `1.2e-14` | `1.2e-14` |
| `ln dL(z;H0) − ln dL(z;fid) + ln(H0/fid)` | `1.5e-15` | `1.5e-15` |
| catalog lattice halved, `10⁻⁶ → 5×10⁻⁷` | `9.0e-11` | `2.3e-9` |
| **the `G(b)` battery** — `n_m1`, `n_q`, `n_gh` doubled, the `V` lattice halved, the mass range widened to `[0.5, 190] M⊙` — quoted as the change in `d ln μ/dH0` itself | max `2.2e-10` over every knob, tracer and arm | |
| the oracle's own finite difference against its analytic derivative, `dh` halved to `0.0625` | `~1e-8` | `~1e-8` |

and two **generative** certificates, which owe nothing to darksirens:

* **`F(z)` against the mock's own injection bookkeeping.**  `stage_injections`
  records the population branch's proposed and detected counts in 1030 redshift
  bins.  Against `G(b(z))` integrated over each bin: `184,677` detected vs
  `184,895` predicted on `9.75×10⁷` proposals (`0.99882 ± 0.00232`, `−0.51σ`,
  binomial `χ² = 409/471`) in the targeted campaign, and `683,598` vs `683,847`
  on `3.60×10⁸` (`0.99964 ± 0.00121`, `−0.30σ`, `χ² = 557/541`) in the popuni
  campaign.
* **`μ` at truth against a replay of the event loop.**  `attr_selmu_gencheck.py`
  re-runs `stage_events`' proposal verbatim — hosts drawn uniformly from the real
  catalogs, masses and spin from `gmd`'s own samplers, `gmd._interp_dl`, the
  generator's `observe()` and `detect_from_observation()`, the `acc` acceptance —
  for `1.2×10⁷` draws per tracer:

  | per catalog galaxy, mixed at `f_AGN = 0.30` | oracle | brute force | |
  |---|---|---|---|
  | `⟨acc · P_det⟩` | `6.972664e-3` | `6.951150e-3 ± 1.83e-5` | `+1.18σ` |
  | `⟨P_det⟩` | `7.873444e-3` | `7.850083e-3 ± 1.95e-5` | `+1.20σ` |
  | their ratio (the rate factor alone) | `0.88559` | `0.88549` | mock realised `0.88636` |

  The mock's own `events_meta` fraction (`7.605e-3`) sits `+3.24σ` above both —
  it is shot noise on *that seed's* 200,000 proposals, and it is the largest of
  the five realisations (`7.605 / 7.21 / 6.95 / 6.75 / 7.01 ×10⁻³`).

### 11.4 The decisive comparison

`d ln μ/dH0` at truth.  The oracle is quoted analytically and at `dh = 0.5` (the
convention `r` is taken in); the injections are quoted at `dh = 0.5` and
Richardson-extrapolated from `dh = 0.25 / 0.125`, with the Monte-Carlo error of
the estimator itself from a **Poisson bootstrap over injections** (200 replicates;
the delta-method influence function agrees to 5 %).

| | matched GAL | matched AGN |
|---|---|---|
| **oracle, KDE hosts** (analytic) | **`+4.1907226e-2`** | **`+4.1554932e-2`** |
| oracle, KDE hosts, `dh = 0.5` | `+4.1908092e-2` | `+4.1555852e-2` |
| oracle, exact hosts (zero bandwidth) | `+4.1899302e-2` | `+4.1545742e-2` |
| oracle, the mock's own uniform host prior | `+4.1874722e-2` | `+4.1521283e-2` |
| oracle, no rate factor | `+4.3528621e-2` | `+4.3162573e-2` |
| **darksirens, targeted injections** (the analysis of record) | **`+4.1867891e-2 ± 1.203e-4`** | **`+4.0788772e-2 ± 5.578e-4`** |
| darksirens, popuni injections | see §14 | `+4.1850124e-2 ± 1.134e-3` |
| **Δ (targeted − oracle)** | **`−3.91e-5`  = `−0.32σ`** | **`−7.66e-4` = `−1.37σ`** |
| Δ (popuni − oracle) | see §14 | `+2.95e-4` = `+0.26σ` |

**The estimator is unbiased, and its error is now measured.**

* **matched GAL — the selection side is verified and is not the residual.**  The
  exact and the estimated slope agree to `−3.9e-5`, `0.32σ` of the estimator's own
  Monte-Carlo error.  Correcting for it moves `r` from `−8.2916e-4` to
  `−8.682e-4` and the exact-numerator remainder from `−5.8433e-4` to `−6.234e-4`:
  the selection side is worth **`4.7 %` of the residual and it moves it the wrong
  way**.  In `H0`, `+0.12 km s⁻¹ Mpc⁻¹` against a `−3.00` offset.
* **matched AGN — about half the `+0.71` pedestal is selection Monte Carlo.**  The
  record's targeted lane sits `−7.66e-4` below the exact slope, `1.4σ` of its own
  error, which inflates `r` by `+7.66e-4` — **39 % of the AGN residual**, worth
  `+0.36 km s⁻¹ Mpc⁻¹` of the `+0.71`.  A *second, independent* injection campaign
  on the same catalog sits `+2.9e-4` on the **other** side of the exact value, and
  the two are consistent with it at `1.1σ` when combined.  Nothing is biased; the
  AGN estimate is simply noisy, because that catalog's `p_z(z|pix)` is a spiky
  object (≈ 120 galaxies per nside-32 pixel against GAL's ≈ 12,300) and the
  injections sample it sparsely.
* **The sub-tests localise the difference to the estimator, not the model.**  The
  three proposal branches of the targeted campaign give
  `+4.0562e-2 / +4.0712e-2` (population / targeted-AGN) on AGN and
  `+4.2032e-2 / +4.2222e-2` on GAL, each a *self-contained* importance sampler
  with its own declared density — so `pdraw` is not the issue.  The `n_finite`
  injection count is constant across the whole `H0` grid, so no `H0`-dependent
  masking.  The finite-difference step-halving sequence converges as `O(dh²)` to
  the Richardson value.  The `ln μ` **shapes** of estimator and oracle agree to
  `0.5 %` over the entire `H0 ∈ [50, 100]` range.
* **The KDE in `μ` is negligible.**  `kde − delta` is `+7.9e-6` (GAL) and
  `+9.2e-6` (AGN) — the catalog's photo-z kernel matters in the *numerator*
  (§7), not in the selection integral.

**What is new, and what it costs.**  `r` subtracts **one** number from **every**
event, so whatever error `μ̂`'s slope carries is a **common-mode** shift of `r`
that the per-event `sem` cannot see and that does **not** average down with
`nEvents`.  It has never been propagated.  Measured here:

| configuration | `σ_MC` on `d ln μ/dH0` | in km s⁻¹ Mpc⁻¹ on that control |
|---|---|---|
| matched GAL, targeted | `1.20e-4` | **`±0.36`** |
| matched AGN, targeted | `5.58e-4` | `±0.26` |
| matched AGN, popuni | `1.13e-3` | `±0.53` |

(per-event curvatures at truth, measured on the post-fix control grids:
`−3.32e-4` GAL, `−2.15e-3` AGN.)

---

## 12. The `chi_eff` clip

`scripts/attr_chieff_clip.py` → `results/attr_chieff.json`, `attr_chieff_{gal,agn}.npz`.

`observe()` records `obs_chieff = clip(N(chi, 0.08), −1, +1)`, so the realised
measurement model is **censored**, not Gaussian —
`P(obs = +1 | chi) = 1 − Φ((1−chi)/s)` is an atom — and `posterior_samples()`
draws `clip(N(obs_chieff, s), −1, 1)`, which is **not** the exact flat-prior
posterior (that is a *truncated* normal for an interior observation and a smooth
ramp `∝ 1 − Φ((1−chi)/s)` for a censored one).

**The channel is closed twice over.**

*Analytically.*  `parametric.py::log_p_pop` is a **product**,
`log[p_mass(m1src)·p_pair(q|m1src)·p_spin(chieff)] + (γ−1)log(1+z)`; the spin
factor depends on `chieff` alone, `p_pe ∝ m1det` carries no `chieff`, and
`snr_amplitude` never reads it.  So in both the per-event evidence and `μ` the
`chieff` integral factorises into a multiplicative constant with no `H0` in it,

```
Z_i(H0) = [ INT dchi p_spin(chi) L_chi(obs_i | chi) ] x [ everything else ](H0)
```

and `d ln Z_i/dH0` is **identically** independent of the `chieff` measurement
model — as is `d ln μ/dH0`.  Measured: `log p_pop(chi = −0.4) − log p_pop(chi = 0.35)`
is constant over 4000 random `(m1det, q, dL, pix)` to **`8.5e-14`**, on both
catalogs and at every `H0`.

*Empirically.*  The clip never fires.  `|obs_chieff|` reaches `0.4477` — **6.90 σ**
from the boundary; there are **0** censored observations and **0** of `2,000,000`
PE samples at `±1`; `|chi_true|` reaches `0.333`.  The exact censored posterior and
the stored clipped Gaussian therefore coincide to **`2.5e-12`** in CDF on every
event.  (The model error is real, it is simply never exercised: a hypothetical
observation at `0.90 / 0.95 / 0.99 / 1.0` would carry a CDF distance of
`0.11 / 0.27 / 0.45 / 0.50`.)

Redrawing every `chieff` sample from the **exact** censored posterior and
re-running darksirens' own per-event score:

| substitution on `d ln Z_i/dH0` | matched GAL | matched AGN |
|---|---|---|
| exact censored posterior − stored | `+3.77e-6 ± 1.09e-4` | `−1.63e-3 ± 1.43e-3` |
| within-event permutation control (preserves the marginal exactly) | `+2.63e-5 ± 1.41e-4` | `−3.91e-5 ± 1.42e-3` |

The substitution and the permutation control carry the **same** error bar, which
is the point: what is measured is the finite-`nsamp` coupling between the
independently drawn `chieff` samples and the rest — Monte-Carlo variance, not
bias.  `Δr` = `+3.8e-6` on GAL, **0.5 % of the residual and consistent with
zero**.

**Verdict: a closed channel, and a null.**

---

## 13. The host-acceptance convention

`scripts/attr_hostw.py` + `attr_sky_oracle.py --host_prior_arms` →
`results/attr_hostw.json`, `attr_sky_oracle_{gal,agn}_hostw.{json,npz}`.

**What the mock does.**  `stage_events` draws `i ~ U{0, …, N_tracer−1}` — uniform
over catalog **rows** — and accepts with `(1+z)^(γ−1)/rate_gmax`, where `γ = GAMMA
= 0` and `rate_gmax = max(1, (1+z_grid[−1])^(γ−1)) = 1`, i.e. `∝ 1/(1+z)`.

**What darksirens does.**  In field mode with `volume_weighted = False`,
`z_depth = None`, `sigma_kde = 0`:

```
p_z(z|pix) = [N_obs(pix) p_cat(z|pix) + dN_miss] / Z_global
p_cat(z|p) = g(z) SUM_g kw_g N(z; z_g, sig_g),   kw_g = (w_g / SUM w) / Z(z_g)
```

In the zero-bandwidth limit `Z(z_g) → g(z_g)`, the front `g(z)` cancels the kernel
normalisation, and with `N_obs(p) = Σ_{g∈p} w_g` the prior collapses to
`w_g / Z_global` — **uniform over catalog rows**.  The rate factor `(1+z)^(γ_fid−1)`
comes from `p_pop`.  The two conventions agree provided five things hold, and all
five are checked against the live objects:

| condition | GAL | AGN |
|---|---|---|
| `gamma_fid == GAMMA` | `0.0 == 0.0` | `0.0 == 0.0` |
| `rate_gmax == 1` | `1.0` | `1.0` |
| `w_g == 1` on every catalog row | yes | yes |
| `N_obs[p] == Σ_{g∈p} w_g` | `1.3e-11` | `8.5e-14` |
| `Z_global == Σ_p N_obs[p]` | `6.0e-8` | `1.4e-9` |

**The conventions match.**  What does not cancel at finite bandwidth is the
`O(σ_z²)` residue `g(z_g)/Z(z_g)`; that, and the rate factor, are measured as
**paired one-term substitutions on both sides of the identity** — the numerator
from two extra arms of the exact host-galaxy sky oracle (opt-in
`--host_prior_arms`; with the flag absent every §7 product is reproduced to the
last digit — `kde_pix` `−8.48742e-4`, `delta_pix` `−6.74967e-4`, `delta_host`
`−5.84334e-4`, pixelisation `+9.06327e-5 ± 1.95e-4`), the selection side from the
`μ` oracle's `unif` and `norate` arms:

| substitution | tracer | numerator | selection | **Δr** |
|---|---|---|---|---|
| **uniform host prior − darksirens'** | GAL | `−2.5363e-5 ± 1.7e-6` | `−2.4580e-5` | **`−7.8e-7 ± 1.7e-6`** |
| | AGN | `−1.3450e-5 ± 7.4e-7` | `−2.4460e-5` | **`+1.10e-5 ± 7.4e-7`** |
| drop the rate factor − darksirens' | GAL | `+1.5375e-3 ± 1.7e-5` | `+1.6293e-3` | `−9.18e-5 ± 1.7e-5` |
| | AGN | `+1.3533e-3 ± 4.5e-5` | `+1.6168e-3` | `−2.63e-4 ± 4.5e-5` |

The first block is the whole of task 3: replacing darksirens' host prior by the
mock's own — literally uniform over the 151 M catalog rows — moves `r` by
**`−7.8e-7` on matched GAL, 0.13 % of the residual**, and `+1.1e-5` on matched
AGN, 0.6 %.  The second block is a **lever, not a defect**: it says that had
darksirens carried a different `γ` from the mock, `r` would have moved by
`−9.2e-5` (GAL) per unit of that mis-specification.  Both sides carry `γ = 0`, so
the realised contribution is zero.

**Verdict: the conventions match; a null, with the sensitivity quantified.**

---

## 14. The closing accounting, and the one test left

### 14.1 The three verdicts

| task | verdict | the number |
|---|---|---|
| **1 — the selection function** | the estimator is **unbiased and now verified**; on matched GAL the exact and the estimated `d ln μ/dH0` differ by `−3.9e-5`, `0.32σ` of the estimator's own Monte-Carlo error, and the *wrong sign* to help | GAL `−3.91e-5` (targeted) / `+1.19e-4` (popuni), combined `+4.3e-5 ± 8.3e-5`; AGN `−7.66e-4` / `+2.95e-4`, combined `−5.59e-4 ± 5.00e-4` |
| **2 — the `chi_eff` clip** | a **closed channel**: the spin factor factorises out of both `Z_i` and `μ` to `8.5e-14`, and the clip never fires (`6.9σ` away, 0 of 2,000,000 samples) | `Δr = +3.8e-6 ± 1.1e-4` (GAL), the same size as a pure permutation control |
| **3 — the host-acceptance convention** | the conventions **match** (`γ_fid = GAMMA = 0`, `rate_gmax = 1`, `w_g = 1`, `N_obs = Σw`, `Z_global = ΣN_obs`) | `Δr = −7.8e-7 ± 1.7e-6` (GAL), `+1.1e-5 ± 7.4e-7` (AGN) |

### 14.2 What is new: the selection integral carries a common-mode error

`r` subtracts **one** number from **every** event, so whatever Monte-Carlo error
`μ̂`'s slope carries is a **common-mode** shift of `r` — invisible to the per-event
`sem`, and it does **not** average down with `nEvents`.  It had never been
propagated.  Three independent routes agree on its size: a Poisson bootstrap over
injections, the delta-method influence function of the same statistic, and the
scatter of the estimator about the exact curve across the `H0` grid.

| configuration | `σ_MC` (bootstrap) | (delta method) | (scatter about the exact curve, 10 `H0` points) | in km s⁻¹ Mpc⁻¹ |
|---|---|---|---|---|
| matched GAL, targeted | `1.20e-4` | `1.27e-4` | `1.63e-4` | **`±0.36`** |
| matched GAL, popuni | `1.16e-4` | `1.14e-4` | `2.00e-4` | `±0.35` |
| matched AGN, targeted | `5.58e-4` | `5.41e-4` | `6.39e-4` | `±0.26` |
| matched AGN, popuni | `1.13e-3` | `1.11e-3` | `1.40e-3` | `±0.53` |

The AGN estimate is 5× noisier than the GAL one for a physical reason: the AGN
catalog carries ≈ 120 galaxies per nside-32 pixel against GAL's ≈ 12,300, so its
`p_z(z|pix)` is a spiky object that 2.1 M injections sample sparsely.  **About
half of the AGN control's `+0.71 ± 0.20` pedestal is a Monte-Carlo fluctuation of
its own selection integral**: the record's targeted lane sits `−7.66e-4` below the
exact slope, worth `+0.36 km s⁻¹ Mpc⁻¹`, and an independent campaign on the same
catalog sits `+2.9e-4` on the other side.

### 14.3 The closure of `r`, matched GAL — final

| step | `r` per event | share of the original |
|---|---|---|
| record, pre-fix (`attr_score_terms`, FD convention) | `−1.4491e-3` | 100 % |
| − population sampler vs analytic pdf (`ATTRIBUTION.md` A1) | `+1.3e-8` | 0.001 % |
| − **(c2) the exact mass PE** | `+5.6013e-4` | 38.7 % |
| − **(b2) the RA width from the observed dec** | `+5.98e-5` | 4.1 % |
| **= the analysis of record, post-fix** | **`−8.2916e-4`** | 57.2 % |
| oracle anchor `kde_pix` | `−8.4874e-4` | (anchor offset `−2.0e-5`, 0.16σ) |
| − the catalog's declared photo-z kernel | `+1.7378e-4` | 12.0 % |
| − the nside-32 pixelisation | `+9.063e-5` | 6.3 % |
| **= exact measurement model, exact host positions** | **`−5.8433e-4`** | 40.3 % |
| − the host-prior convention (task 3) | `−7.8e-7` | 0.05 % |
| − the `chi_eff` clip (task 2) | `+3.8e-6` | 0.3 % |
| − the selection estimator's own error (task 1) | `−3.91e-5` | −2.7 % |
| **= remaining, everything exact** | **`−6.204e-4`** | **42.8 %** |

(the `(b2)` entry is `+5.98e-5` in the finite-difference convention these endpoints
are quoted in; §7's `+6.09e-5` is the term-sum convention, and the two differ by
`1.1e-6`.)

`figs/fig_selmu_oracle.{png,pdf}` — (a) the selection slope across the scanned
range, (b) the estimator against the closed form with its own `±1σ_MC` band and
the level each catalog would need, (c) `F(z)` against the generator's own
`4.6×10⁸` injection draws, (d) this ladder.

**Three of the three channels this sweep opened are closed, none of them is the
residual, and the residual is slightly larger than it was**: `−6.20e-4` per event,
two fifths of the original, surviving a model that is exact in every measurement
channel, puts every one of 151 M catalog galaxies at its true `(ra, dec, z)`, uses
the mock's own host prior, and now also uses the exact selection function.

### 14.4 The score identity now fails with every term exact

What has been measured and excluded, cumulatively:

* the population (`1.3e-8`), the detection rule (`ρ_obs` reproduced bit-exactly),
  the mass PE convention (fixed), the RA width (fixed);
* the catalog prior's photo-z kernel (`+1.7e-4 ± 4.4e-4`), its nside-32
  pixelisation (`+9.1e-5 ± 2.0e-4`), measured against the real galaxy positions;
* the `chi_eff` measurement model (factorises out exactly);
* the host-acceptance convention (matches; residue `−7.8e-7`);
* **the selection integral** — `P_det` in closed form validated against the
  generator's own `observe()` to `1e-4`, `F(z)` against `4.6×10⁸` of its own
  injection draws to `0.1 %`, `μ(truth)` against a replay of the event loop to
  `0.3 %`, and `d ln μ/dH0` against darksirens' estimator to `0.32σ` of that
  estimator's own error;
* the estimator `darksirens` itself — an independent closed-form quadrature
  reproduces its per-event score to `−2.0e-5 ± 1.2e-4` over 720 events, with an
  rms difference *below* its own PE Monte-Carlo error.

**What has NOT been audited is the generator's event-draw bookkeeping**, i.e.
whether the 1000 stored events are distributed as `p_target(θ) P_det(θ)/μ`:

1. **the rejection loop and the `[:N_EVENTS]` truncation.**  `stage_events` draws
   fixed batches of 100,000 proposals until 1000 detections have accumulated, then
   keeps `concatenate(...)[:1000]` of the 1521 it found.  Within a batch the
   proposals are i.i.d. and `det` is a mask that preserves proposal order, so the
   kept set is exchangeable and *should* be unbiased — but that is an argument,
   not a measurement.
2. **`rate_gmax`** — `max(1, (1+z_grid[−1])^(γ−1)) = 1` at `γ = 0`, so the
   acceptance is `(1+z)^(γ−1)` un-normalised.  Verified here (§13); it would
   matter at `γ > 1`.
3. **the rejected-proposal sample** (`events_rejected_sample.h5`, the first 20,000
   rejections) — a validation product, never read by the likelihood.

**The one test to run next.**  The `(C − A)` / `(A − B)` split — "does the
posterior-averaging step fail, or is the detected *truth* set mis-distributed?" —
is the only remaining discriminator, and on one realisation it has no power:
`(C − A) = −4.42e-3 ± 3.16e-3` against `r = −6.2e-4`.  Run
`probe3_decomposition.py` **and** the `delta_host` arm of the exact host-galaxy
oracle **on all five post-fix realisations**, each with its own injection campaign
so the selection-side common-mode error (`±1.2e-4`, now known) averages down as
`1/√5`.  That converts the one-realisation `−6.20e-4` into `⟨r⟩ ± sem` with the
precision that made the pre-fix statement 12σ, and it separates the two branches:

* if **`(A − B) ≠ 0`** the mock's detected-truth set is not the model's, and after
  everything above the only place left for that is item 1 — the event-draw
  bookkeeping;
* if **`(C − A) ≠ 0`** with an exact measurement model and exact host positions,
  then no measurement model in this family can satisfy the identity in this
  configuration, and the answer is the mass-channel *design*
  (`ATTRIBUTION.md` option 2: `σ/m = 8–10 %` against a `35 ± 5 M⊙` peak is a
  strong, strongly curved spectral-siren lever, and the dense-catalog
  configuration has 6.5× less `H0` curvature per event than AGN with which to
  fight it) — a scope decision about what the mock is for, not a bug.

Nothing else in this campaign discriminates between those two, and nothing else
should be run until that split is measured.

**Measured — §15.  Both branches fire, and only one of them carries `r`.**  Item 1
is exonerated: the truncation is exchangeable and the rejection loop is clean.
`(A − B)` is nonetheless `+5.84e-4 ± 0.84e-4`, but it is confined to the catalog
`p_z` term, its cause is a *catalog* convention this list did not contain (the
survey block declares a photo-z kernel on redshifts that are exact), and it cancels
in `r` (`r_pz = −2.8e-5 ± 2.0e-4`).  In the mass channel `(A − B) = −4.4e-5 ±
1.2e-5` and `(C − A) = −1.274e-3 ± 0.113e-3` — branch 2, at 11.3σ.

---

# 15. The endgame — the split, measured

Owner-approved, 2026-08-01.  `darksirens` **READ-ONLY at `2b86a2d`** throughout;
every run below is anchored `|Δ log μ| = 0` against darksirens' own likelihood.
Nothing under `working/data` was written; the regenerated events live in
`/hildafs/projects/phy220048p/.../scratch_truncation_test/`.

## The verdict in one line

**The truncation is innocent and the split is decided both ways at once.**  The
event-draw bookkeeping §14.4 named — the `[:N_EVENTS]` truncation and the rejection
loop — is **exonerated**: the kept 1000 and the withheld 521 have the same mean
score to `−1.1e-4 ± 1.4e-4` (GAL) and `−1.8e-4 ± 2.1e-4` (AGN) over 2000 replays of
the event stage on two different catalogs.  `(A − B)` is nonetheless **nonzero at
6.9σ**, `+5.84e-4 ± 0.84e-4` per event — but it lives **entirely in the catalog
`p_z` term**, its cause is a *catalog* convention rather than an event-draw one
(the survey block declares a photo-z kernel on redshifts that are exact), and it
**cancels in `r`**: `r_pz = −2.8e-5 ± 2.0e-4`.  In the channel that carries the
residual — the population/mass term — `(A − B) = −4.4e-5 ± 1.2e-5` and
`(C − A) = −1.274e-3 ± 0.113e-3`, an **11.3σ** violation.  `r_pop = −1.138e-3 ±
0.173e-3` is **96 % of** `r_tot = −1.187e-3 ± 0.254e-3`.  **The residual is
`(C − A)` in the mass channel: no per-event measurement model in this family closes
it, and the answer is the design-scope one.**

## 15.1 Why five realisations could not do it, and what was run instead

`A` is a mean of `ς(θ_true)` whose per-event scatter is **0.086** — 5.5× the scatter
of `C` (0.0156), because the catalog KDE's log-slope at a *single* galaxy's own
redshift is a spiky object, while `C` has already averaged it over 2000 PE samples.
3,513 matched-GAL events therefore give `sem(A − B) = 1.45e-3`, **ten times** the
`1.5e-4` this test needs; reaching `1.5e-4` through realisations alone
would take ~450 of them.  The five-realisation split is reported below and is
consistent with everything that follows, but it has no power, exactly as §14.4
anticipated.

What has power is to **redraw the event stage itself**.  `regen_events_notrunc.py`
replays `stage_events`' proposal loop verbatim — the same `gmd` samplers, the same
`observe()`, the same RNG consumption order, the same `ntry = 100,000` batches and
the same stopping rule — with the `[:N_EVENTS]` truncation lifted and every
detection's position in the stream recorded.  Run 1500 times with fresh event
sub-seeds **on seed 100's own catalog**, it measures `E[A]` against a `B` that is a
single exact number for that catalog, so the only Monte Carlo left is the event draw
— the thing under audit.  500 further replays on seed 103's catalog check that the
answer is not a property of one catalog realisation.

**The replay is the generator.**  Run with seed 100's own event sub-seed it
reproduces the record **bit-identically**: all **25** `truth/` fields, **0** differing
bits, **1521** detections from **200,000** proposals, matching `events_meta.json`.

## 15.2 The truncation test

Seed 100 needs **two** batches: batch 0 yields 764 detections (all kept), batch 1
yields 757, of which the first **236** are kept and **521** are withheld; the last
kept event sits at slot 33,681 of batch 1.  Over 1500 replays the loop always takes
exactly two batches and finds `1394.1 ± 36.4` detections (the record's 1521 is a
`+3.5σ` draw of that distribution — the replay is bit-identical, so this is the
realisation, not the code).

`A` on the kept head, the withheld tail, and the union:

| catalog | group | `n` | `A` | `A − B` |
|---|---|---|---|---|
| seed 100, 1500 replays, **GAL** | kept 1000 | 1,045,857 | 4.2490777e-2 | **+5.84e-4 ± 0.84e-4** |
| | withheld | 412,123 | 4.2725540e-2 | +8.18e-4 ± 1.34e-4 |
| | all found | 1,457,980 | 4.2557137e-2 | +6.50e-4 ± 0.72e-4 |
| seed 103, 500 replays, **GAL** | kept 1000 | 347,595 | 4.2614548e-2 | +7.85e-4 ± 1.45e-4 |
| | withheld | 138,338 | 4.2354696e-2 | +5.25e-4 ± 2.30e-4 |
| seed 100, 1500 replays, **AGN** | kept 1000 | 454,143 | 4.2049080e-2 | **+4.94e-4 ± 1.29e-4** |
| | withheld | 179,003 | 4.2232761e-2 | +6.78e-4 ± 2.06e-4 |
| seed 103, 500 replays, **AGN** | kept 1000 | 152,405 | 4.1775440e-2 | +4.70e-4 ± 2.24e-4 |

**head − tail**, the truncation's own statistic:

| | seed 100 | seed 103 | combined |
|---|---|---|---|
| GAL | −2.35e-4 ± 1.59e-4 | +2.60e-4 ± 2.72e-4 | **−1.09e-4 ± 1.37e-4** (0.79σ) |
| AGN | −1.84e-4 ± 2.43e-4 | −1.51e-4 ± 4.21e-4 | **−1.76e-4 ± 2.10e-4** (0.84σ) |

The two catalogs disagree in sign and the combination is consistent with zero.  Any
truncation-induced bias is below `1.4e-4` per event, a factor **8** below
`r = −1.19e-3`.

The stream itself is clean, on 2,089,626 accepted-index gaps:

| test | result |
|---|---|
| gap mean / sd | 143.37 / 142.92 vs geometric `1/p = 143.46` / `142.96` |
| gap vs Geometric(`p̂`), χ² on 20 quantile bins | **χ²/dof = 0.608**, `p = 0.90` |
| autocorrelation of the gaps and of `z`, lags 1,2,3,5,10,25 | all `\|r\| ≤ 1.9e-3` with `sem = 6.9e-4`; largest of the 12 is `−2.7σ` |
| Spearman(within-batch slot, `z`), 3000 batches | `+3.4e-4 ± 6.9e-4` |
| Spearman(within-batch slot, `ρ_obs`) | `−9.7e-4 ± 6.9e-4` |
| Spearman(rank, `z`), 1500 replays | `+8.8e-4 ± 6.9e-4` |
| `ς(θ_true)` vs stream rank, slope over the kept 1000 | GAL `−3.5e-6 ± 2.9e-4`; AGN `+2.9e-4 ± 4.5e-4` |

(A single realisation is not enough to see this: seed 100 alone gives
Spearman(slot, `z`) `= −0.079`, `p = 0.028` in batch 0 and `−0.004`, `p = 0.91` in
batch 1 — the sort of split the pooled `±6.9e-4` shows to be noise.)

**Verdict: the proposal stream is exchangeable, the rejection loop is a clean
Bernoulli thinning, and `[:N_EVENTS]` is unbiased.  Item 1 of §14.4 is closed.**

## 15.3 The split, per term

`(A − B)` from the 1500 replays on the **kept** 1000 (the record's own truncation);
`(C − A)` and `r` pooled over the five post-fix realisations, each against **its own
catalog's exact** `d ln μ/dH0`.  Per event, at `H0 = 67.74`, term-sum convention,
`dh = 0.5`.  The `tot` row's `B` is the exact oracle; the per-term rows use the
injection estimator's split of it, whose total error is `−1.39e-5` (GAL).

**matched GAL**

| term | `(A − B)` (2.09 M replayed truths) | `(C − A)` (5 realisations, paired) | `r` |
|---|---|---|---|
| `p_pop` | `−4.38e-5 ± 1.24e-5` | **`−1.2738e-3 ± 1.13e-4`** (11.3σ) | `−1.1384e-3 ± 1.73e-4` |
| → rate | `+1.5e-6 ± 0.4e-6` | `+5.6e-6 ± 2.0e-6` | `−5.5e-6 ± 7.3e-6` |
| → mass | `−4.53e-5 ± 1.26e-5` | **`−1.2794e-3 ± 1.13e-4`** | `−1.1329e-3 ± 1.76e-4` |
| catalog `p_z` | **`+6.383e-4 ± 0.836e-4`** (7.6σ) | `−5.82e-4 ± 1.43e-3` | `−2.78e-5 ± 1.96e-4` |
| Jacobian | `+2.9e-6 ± 0.9e-6` | `+1.04e-5 ± 0.39e-5` | `−1.03e-5 ± 1.46e-5` |
| **total** | **`+5.836e-4 ± 0.844e-4`** | `−1.846e-3 ± 1.44e-3` | **`−1.1874e-3 ± 2.54e-4`** |

**matched AGN**

| term | `(A − B)` | `(C − A)` | `r` |
|---|---|---|---|
| `p_pop` | `+7.8e-6 ± 1.89e-5` | **`−1.7354e-3 ± 1.72e-4`** (10.1σ) | `−1.5131e-3 ± 2.62e-4` |
| catalog `p_z` | **`+1.0965e-3 ± 1.28e-4`** (8.6σ) | `−4.2e-5 ± 2.71e-3` | `+3.291e-3 ± 1.55e-3` |
| **total** | **`+4.942e-4 ± 1.292e-4`** | `−1.778e-3 ± 2.71e-3` | **`+1.494e-3 ± 1.57e-3`** |

(the `r` column pools events; AGN's per-event `r` is dominated by `p_z` outliers, so
its **five-seed** mean is far tighter — `+1.4911e-3 ± 4.26e-4`.  On GAL the two
agree: `−1.1874e-3 ± 2.54e-4` pooling events, `−1.1934e-3 ± 2.84e-4` pooling seeds.)

and the five-realisation split on its own, which is what §14.4 asked for and which
— as predicted — resolves nothing but corroborates everything:

| | GAL per seed (100/101/102/103/105) | pooled | AGN pooled |
|---|---|---|---|
| `(A − B)` | `+3.56e-3`, `+5.62e-3`, `−6.56e-3`, `+0.86e-3`, `−0.02e-3` (each `± 3.2e-3`) | `+6.6e-4 ± 1.45e-3` | `+3.27e-3 ± 2.21e-3` |
| `(C − A)` | `−4.40e-3`, `−6.91e-3`, `+4.32e-3`, `−1.81e-3`, `−0.62e-3` | `−1.85e-3 ± 1.44e-3` | `−1.78e-3 ± 2.71e-3` |
| `r` | `−8.43e-4`, `−1.291e-3`, `−2.248e-3`, `−9.46e-4`, `−6.39e-4` | `−1.187e-3 ± 2.54e-4` | `+1.491e-3 ± 4.26e-4` (five-seed) |

The pooled five-realisation `(A − B)` (`+6.6e-4 ± 1.45e-3` GAL, `+3.3e-3 ± 2.2e-3`
AGN) sits on top of the replay campaign's `+5.8e-4` and `+4.9e-4`: two completely
different estimators of the same expectation, one from the analysis of record and
one from 2 million redrawn events, agree.

Combining the two — `(C − A) = r − (A − B)` with the replay campaign's `(A − B)` —
gives the **total** `(C − A)` at the precision the paired five-realisation statistic
cannot reach: **`−1.771e-3 ± 2.68e-4` (GAL, 6.6σ)** and `+9.97e-4 ± 4.45e-4` (AGN,
on its five-seed `r`).

**Precision delivered against the `≤ 1.5e-4` target:** `(A − B)` `= 8.4e-5` (GAL) and
`1.29e-4` (AGN) — met; `(C − A)` in the mass channel `= 1.13e-4` (GAL) and `1.72e-4`
(AGN) — met.  The `(C − A)` *total* is limited to `2.7e-4` by `r`'s own
five-realisation error, which is a property of the five-realisation design and not of
the split.

## 15.4 `(A − B) ≠ 0` is real, and it is not the event draw

The whole of `(A − B)` sits in the catalog term, and its cause is visible in the
survey block itself.  `stage_events` draws its hosts from
`catalog_{gal,agn}_complete.h5`; `stage_surveys` builds the block from the **same**
file and writes the galaxies' redshifts through **untouched** —

```
sorted(survey zgals) vs sorted(catalog z):   max|Δ| = 0.0
  (all 151,179,870 GAL rows and all 1,514,567 AGN rows)
dzgals == DZ_SCALE (1+z),  DZ_SCALE = 3e-3:  max|Δ| = 0.0
```

so the likelihood's `p_z(z|pix)` is a KDE of width `3e-3 (1+z)` **about redshifts
that carry no error at all**.  The catalog declares an uncertainty it does not have.
The mock's host redshifts are a comb of deltas, the model's are that comb smeared,
and no positive kernel width reproduces a comb: `E[A] = B` cannot hold in this
channel for **any** member of the model family.

Measured, by rescaling **only** `dzgals` and recomputing the exact oracle `B` on the
same block (`log_kw` anchor `≤ 1.3e-7` in every run, so the oracle and darksirens
still use the identical kernel):

| declared `dz` | matched GAL `(A − B)` | matched AGN `(A − B)` |
|---|---|---|
| `0.0015 (1+z)` (×0.5) | — | `+1.446e-3 ± 1.99e-4` |
| **`0.0030 (1+z)` (the record)** | **`+5.836e-4 ± 0.844e-4`** | **`+4.942e-4 ± 1.292e-4`** |
| `0.0060 (1+z)` (×2) | `+3.669e-4 ± 0.371e-4` | `+5.940e-4 ± 0.833e-4` |
| `0.0090 (1+z)` (×3) | — | `+6.897e-4 ± 0.637e-4` |

`(A − B)` is a strong function of the declared width, which identifies the channel.
The two catalogs behave differently for a reason that is measurable in the blocks
themselves — the kernel width against the local per-pixel redshift spacing, counted
at the detected sample's median `z = 0.132` over 200 random pixels:

| | galaxies per nside-32 pixel | within ±1 kernel width of `z = 0.132` | spacing | kernel/spacing |
|---|---|---|---|---|
| GAL | 12,303 | 9.7 | 7.0e-4 | **4.8** |
| AGN | 123 | 0.10 | 6.5e-2 | **0.05** |

GAL is in the **smoothed** regime: the kernel already spans ~10 galaxies, widening it
pushes the smoothed prior further towards the comb's own density, and `(A − B)` falls
by 37 % from ×1 to ×2.  AGN is in the **isolated** regime at every width tested — the
kernel is 20× narrower than the gap to the next galaxy — so the prior is a set of
separate bumps, and `(A − B)` is minimised near the record's width rather than
removed.

**This is a real, named generator defect, and it was not on §14.4's list** — §7
measured the same declared kernel only as a substitution inside `C`
(`delta_pix − kde_pix = +1.738e-4 ± 4.36e-4`), where it is consistent with zero.
Seen from the truth side it is a 7.6σ effect.  But it does **not** carry the
residual: the shift it puts into `A` and the shift it puts into `C` very nearly
cancel, leaving `r_pz = −2.78e-5 ± 1.96e-4` on matched GAL — the catalog channel
contributes nothing to `r`.

## 15.5 What the split says

| branch of §14.4 | outcome |
|---|---|
| **1 — the event-draw bookkeeping** (`[:N_EVENTS]`, the rejection loop) | **exonerated.**  Bit-identical replay; head − tail `−1.09e-4 ± 1.37e-4` (GAL) and `−1.76e-4 ± 2.10e-4` (AGN) over 2000 replays on two catalogs; geometric, uncorrelated gaps; no score trend with stream position |
| **1′ — the detected-truth set is nonetheless not the model's** | **confirmed at 6.9σ**, `(A − B) = +5.84e-4`, entirely in `p_z`, caused by the survey block's declared photo-z kernel on exact redshifts — **and it cancels in `r`** (`r_pz = −2.8e-5 ± 2.0e-4`) |
| **2 — `(C − A) ≠ 0` with an exact measurement model** | **confirmed at 11.3σ in the mass channel**, `(C − A)_pop = −1.274e-3 ± 0.113e-3` (GAL), `−1.735e-3 ± 0.172e-3` (AGN), against `(A − B)_pop = −4.4e-5 ± 1.2e-5` and `+0.8e-5 ± 1.9e-5` |

In the channel that carries the residual the events' true parameters reproduce the
model's detected-truth mean to `4.4e-5` — one part in `10^3` of `ς` and inside the
injection estimator's own error — while the **posterior-averaged** score misses it by
`−1.27e-3`.  `r_pop = −1.138e-3` is 96 % of `r_tot = −1.187e-3`.  Seed 100's own
`(C − A)_pop = −8.87e-4 ± 2.62e-4` reproduces `ATTRIBUTION.md`'s post-fix
`(C − A)_mass = −0.891e-3 ± 0.262e-3` — computed there by reweighting the *pre-fix*
PE, here by a fresh generative draw and a separate code path.

**The named cause.**  `r` is the mass channel's posterior-averaging step.  The
population is right, the detection rule is right, the event draw is right, the
selection integral is exact, the measurement model is exact in every channel, the
host positions are exact — and the ensemble mean of `E_post[ς_mass]` still sits
`−1.27e-3` below `ς_mass(θ_true)`.  That is `ATTRIBUTION.md` option 2 and it is a
**design** statement, not a bug: `σ/m = 8–10 %` against a `35 ± 5 M⊙` peak is a
strong, strongly curved spectral-siren lever, the exact flat-prior posterior of
`obs ~ N(m, f m)` is skewed by construction, and the ensemble mean of a nonlinear
functional of it does not have to equal the functional of the mean.  The
dense-catalog configuration has 21.6× less `H0` curvature per event than AGN with
which to fight it, which is why the same `(C − A)` shows up as `−6.30 km s⁻¹ Mpc⁻¹`
on matched GAL and `+0.71` on matched AGN.

## 15.6 Recommended fix — NOT implemented

Two separable items, in order of what they buy.

**(a) The catalog's declared photo-z kernel — a real defect, worth `+5.8e-4` of
`(A − B)` and `−2.8e-5 ± 2.0e-4` of `r`.**  `stage_surveys` must not declare
`dz = 3e-3 (1+z)` on redshifts copied verbatim from the catalog.  Either

  * *realise the declared error*: store `zgals = z_true + N(0, DZ_SCALE (1+z_true))`
    in the survey block (built inside `pixelate_catalog_vec`, before the
    `lexsort((z, pix))`, so the block's row order stays the block's own), and leave
    the host draw and the sky index on `z_true`.  The likelihood's prior then *is*
    the distribution of the true host redshift given the catalog, and `E[A] = B` in
    the `p_z` channel by construction.  This is the physical reading and the one the
    campaign's photo-z language implies; it changes the catalog, so it invalidates
    the surveys, the skyindex and every scan, and it needs a full regeneration.
  * *or drop the claim*: set `DZ_SCALE = 0` and declare an exact-redshift catalog,
    accepting that darksirens' KDE prior then needs a bandwidth floor.  The
    zero-bandwidth limit is already computed — it is the oracle's `delta` arm — so
    the cost of this choice is known in advance (`d ln μ/dH0`: `kde` 4.190723e-2 vs
    `delta` 4.189930e-2 on seed 100 GAL).

  Fixing this is worth doing for the mock's honesty.  It is **not** worth doing to
  chase `r`: the channel's contribution to `r` is `−2.8e-5 ± 2.0e-4`.

**(b) The residual itself is not a bug and should not be fixed by more
measurement-model work.**  The recommendation is a scope decision on the mass
channel: either soften the lever (a wider mass peak, or `σ/m` small enough that the
posterior's `O(f²)` skew is negligible over the detected band), or accept the
matched-GAL control's offset as a property of *this* configuration and quote it as
such.  No further per-event measurement model in this family will close the identity
— that is what `(C − A)_pop = −1.274e-3 ± 0.113e-3` with `(A − B)_pop = −4.4e-5 ±
1.2e-5` means.

---

## Files

```
working/data/
  generate_dataset.py             conventions (b2) and (c2); V2b; V3 extended to the
                                  exact mass posterior; META records both fixes and
                                  why catalogs/surveys/injections are untouched
  verify_events_regen.py          NEW -- the bitwise regeneration audit
  seed{100,101,102,103,105}/
    events/                       REGENERATED (events_prefix2/ deleted after
                                  validation, per the owner's discard instruction)
    validation/validation.json    10/10 PASS on every seed
    validation/events_regen_bitcheck.json   NEW
    META.json                     refreshed

analyses/analysis_1_complete_catalog_H0/
  CLOSURE.md                      this file
  scripts/
    build_catalog_skyindex.py     NEW -- galaxy positions in the survey's row order
    attr_sky_oracle.py            NEW -- the exact host-galaxy sky oracle, 4 arms
    run_sky_oracle.sh             NEW -- production + convergence battery
    run_gal_conv_local.sh         NEW -- the GAL battery (needs the 80 GB card)
    build_nside_surveys.py        NEW -- nside 64/128 complete surveys
    run_nside_scans.sh            NEW -- the resolution study's four scans
    fig_closure_after_fix.py      NEW -- the before/after strip
    fig_sky_oracle.py             NEW -- the oracle and the nside curve
    closure_summary.py            NEW -- one JSON for this document
    run_postfix_attr.sh           NEW -- post-fix score terms + oracle chain
    submit_seed_controls.sbatch, submit_postfix_aux.sbatch,
    submit_gal_conv.sbatch        NEW -- the HENON-GPU jobs
    run_seed_controls.sh          now runs the estimator of record
  results/                        POST-FIX scans + the post-fix attribution
    ctrl_{gal,agn}_matched[_s1**][_ns{64,128}].{h5,json}
    h0_{gal,agn}_{targeted,popuni}.{h5,json}, guard_h0_*.json
    h0_single_tracer.json, closure_seeds.json, closure_after_fix.json,
    closure_summary.json, surveys_nside.json
    attr_terms_{gal,agn}_s100_postfix.{json,npz}
    attr_sky_oracle_{gal,agn}.{json,npz} + the convergence battery
    (the pre-fix attr_* products ATTRIBUTION.md cites are unchanged)
  results_prefix2/                the PRE-FIX scans of record, archived
  figs_prefix2/                   the pre-fix fig_h0_recovery, fig_guard
  figs/
    fig_h0_recovery, fig_guard, fig_closure_seeds          REGENERATED
    fig_closure_after_fix, fig_sky_oracle_{gal,agn}, fig_nside_curve   NEW

bulk (/hildafs/projects/phy220048p/magana/gws-agn-data/derived/analysis_1_.../):
  skyindex/seed100_{gal,agn}_ns32.h5        3.67 GB, the position index
  surveys_nside/survey_{gal,agn}_complete_ns{64,128}.h5   3.6 GB, the resolution study
  seed{101,102,103,105}/events_{gal,agn}_hosted.h5        rebuilt matched subsets

THE FINAL SWEEP (2026-08-01)
  scripts/
    attr_selmu_pdet.py       NEW -- P_det in closed form + the brute-force
                             validation against the generator's own observe()
    attr_selmu_oracle.py     NEW -- mu(H0) and d ln mu/dH0 by quadrature, four
                             host measures, every catalog galaxy
    attr_selmu_inj.py        NEW -- darksirens' injection log mu(H0) on an H0
                             grid, per-branch estimators, the MC-error bootstrap
    attr_selmu_gencheck.py   NEW -- the generative replay of stage_events
    attr_selmu_gconv.py      NEW -- the G(b) convergence battery (CPU, 6 tasks)
    attr_selmu_summary.py    NEW -- the task-1 verdict
    attr_chieff_clip.py      NEW -- TASK 2
    attr_hostw.py            NEW -- TASK 3
    attr_sky_oracle.py       + --host_prior_arms (OPT-IN; with the flag absent
                             every section-7 product is reproduced to the last
                             digit -- verified in the same run)
    fig_selmu_oracle.py      NEW
    run_selmu.sh, run_local_sweep.sh, run_hostw_chieff.sh,
    submit_agn_aux.sbatch, submit_gconv.sbatch, submit_gal_hostw.sbatch   NEW
  results/
    attr_selmu_pdet.{json,npz}
    attr_selmu_{gal,agn}.{json,npz}
    attr_selmu_inj_{gal,agn}_{targeted,popuni}.{json,npz}
    attr_selmu_gencheck.json, attr_selmu_gconv{,_*}.json, attr_selmu_summary.json
    attr_chieff.json, attr_chieff_{gal,agn}.npz
    attr_hostw.json, attr_sky_oracle_{gal,agn}_hostw.{json,npz}
  figs/
    fig_selmu_oracle.{png,pdf}   NEW

THE ENDGAME (2026-08-01)  --  section 15
  scripts/
    regen_events_notrunc.py  NEW -- replays stage_events' proposal loop verbatim
                             with [:N_EVENTS] lifted, recording every detection's
                             (replica, rank, batch, slot).  --verify does the
                             bitwise audit against the record; --replicas N
                             redraws the event stage N times on a fixed catalog
    attr_abc_split.py        NEW -- A (score at the events' TRUE parameters),
                             C (posterior-averaged), B (injection estimate),
                             anchored |Delta log mu| = 0; --extra_truth evaluates
                             A on the replayed truths, --survey_override on a
                             rescaled-kernel block
    make_dz_survey.py        NEW -- a survey block identical to the record's
                             except for the DECLARED photo-z width (scratch only)
    trunc_diag.py            NEW -- proposal-stream exchangeability: gap law,
                             autocorrelations, slot/rank correlations, and the
                             score in bins of stream rank
    abc_summary.py           NEW -- the five-realisation split
    endgame_summary.py       NEW -- one JSON for this section
    run_abc.sh, run_selmu_seeds.sh, run_dzscan.sh, run_endgame_tail.sh,
    run_s103_replay_A.sh, submit_abc.sbatch                             NEW
    attr_selmu_oracle.py     + --survey_override / --dz_scale (OPT-IN; with both
                             absent the final sweep's product is reproduced
                             BITWISE -- verified in results/attr_selmu_agn_regress)
  results/
    abc_{gal,agn}_s{100,101,102,103,105}.{json,npz}   the split per realisation
    abc_{gal,agn}_mega.{json,npz}                     A on 1500 replays, seed 100
    abc_{gal,agn}_mega_s103.{json,npz}                A on 500 replays, seed 103
    abc_{gal,agn}_mega_dz{x0p5,x2,x3}.{json,npz}      the declared-kernel scan
    attr_selmu_{gal,agn}_s{101,102,103,105}.{json,npz}  the EXACT B per catalog
    attr_selmu_{agn_dzx0p5,agn_dzx2,agn_dzx3,gal_dzx2}.json  exact B, scan
    attr_selmu_agn_regress.json                       the no-override regression
    abc_summary.json, trunc_diag.json, endgame_summary.json

scratch (/hildafs/projects/phy220048p/magana/gws-agn-data/scratch_truncation_test/):
  events_notrunc_full_s100.h5              the untruncated seed-100 replay (1521)
  events_notrunc_replicas_s100_n1500.h5    2,091,126 detections, 1500 replays
  events_notrunc_replicas_s103_n500.h5       699,075 detections,  500 replays
  surveys_dz/survey_{gal,agn}_complete_ns32_dz*.h5   rescaled-kernel blocks
```

Reproduce the final sweep (one A100-80GB for the GAL half; the AGN half fits a
40GB card, the `G(b)` battery is CPU-only):

```bash
./scripts/run_selmu.sh pdet                                   # ~7 min, CPU
python scripts/attr_selmu_oracle.py --tracer agn --conv_lat   # ~2 min
python scripts/attr_selmu_oracle.py --tracer gal --conv_lat   # ~4 min
sbatch scripts/submit_agn_aux.sbatch      # AGN injection curves + task 3
sbatch scripts/submit_gconv.sbatch        # the G(b) battery, 6 CPU tasks
JAX_PLATFORMS=cpu python scripts/attr_selmu_gencheck.py --ndraw 1.2e7
./scripts/run_local_sweep.sh              # GAL injections, task 2, task 3, popuni
python scripts/attr_selmu_gconv.py --collect
python scripts/attr_selmu_summary.py
python scripts/attr_hostw.py
python scripts/fig_selmu_oracle.py
```

Reproduce (one A100-80GB; ~4 min events + ~10 min validation per seed, ~2 h GPU
scans, ~50 min oracle):

```bash
cd working/data
for S in 100 101 102 103 105; do
  python generate_dataset.py --stage events     --seed $S --overwrite
  python verify_events_regen.py --seed $S
  python generate_dataset.py --stage validation --seed $S
done

cd ../analyses/analysis_1_complete_catalog_H0
python scripts/build_hosttype_subset.py --in_path .../seed100/events/events.h5 \
       --out_path data_derived/events_gal_hosted.h5 --host_type 0     # and AGN
FORCE=1 ./scripts/run_scans.sh
./scripts/run_seed_controls.sh 101 102 103 105
./scripts/run_guard_diag.sh
python scripts/build_single_tracer.py
python scripts/aggregate_closure.py --seeds 100 101 102 103 105

python scripts/build_catalog_skyindex.py --seed 100 --tracer both
./scripts/run_postfix_attr.sh          # post-fix r terms + the oracle
NS=120 ./scripts/run_gal_conv_local.sh; TRACERS=agn ./scripts/run_sky_oracle.sh convonly

python scripts/build_nside_surveys.py --nside 64 128
./scripts/run_nside_scans.sh

python scripts/closure_summary.py
python scripts/make_figures.py
python scripts/fig_closure_after_fix.py
python scripts/fig_sky_oracle.py --which all
```

Reproduce the endgame (one A100-80GB, ~2 h wall including the CPU replays):

```bash
# the exact selection oracle on every realisation -- B, to 1e-10, per catalog
./scripts/run_selmu_seeds.sh

# the split on the analysis of record, five realisations x two configurations
./scripts/run_abc.sh                       # or sbatch scripts/submit_abc.sbatch

# the truncation audit: bitwise replay, then the redraw campaigns
JAX_PLATFORMS=cpu python scripts/regen_events_notrunc.py --seed 100 --verify
JAX_PLATFORMS=cpu python scripts/regen_events_notrunc.py --seed 100 --replicas 1500
JAX_PLATFORMS=cpu python scripts/regen_events_notrunc.py --seed 103 --replicas 500 \
       --rep_seed0 7000000
SC=/hildafs/projects/phy220048p/magana/gws-agn-data/scratch_truncation_test
for T in gal agn; do
  python scripts/attr_abc_split.py --seed 100 --tracer $T --extra_only \
         --extra_truth $SC/events_notrunc_replicas_s100_n1500.h5 --tag ${T}_mega
  python scripts/attr_abc_split.py --seed 103 --tracer $T --extra_only \
         --extra_truth $SC/events_notrunc_replicas_s103_n500.h5 --tag ${T}_mega_s103
done

# the declared-photo-z-kernel scan
for S in 0.5 2 3; do python scripts/make_dz_survey.py --tracer agn --scale $S; done
python scripts/make_dz_survey.py --tracer gal --scale 2
T=agn SCALES="0.5 2 3" ./scripts/run_dzscan.sh
T=gal SCALES="2"       ./scripts/run_dzscan.sh

JAX_PLATFORMS=cpu python scripts/trunc_diag.py
JAX_PLATFORMS=cpu python scripts/abc_summary.py
JAX_PLATFORMS=cpu python scripts/endgame_summary.py
```

**Superseded by §16** — the redesign §15.6 recommended was authorised, implemented and measured.
---
---

# 16. The redesign — a measurement family in which every width is data

Owner-approved, 2026-08-01.  `CLOSURE.md` §15 ended with a measured statement and a
recommendation: the residual is `(C − A)` in the mass channel, no per-event
measurement model in the latent-width family closes the identity, and the answer is
the mass channel's **design**.  This section implements that redesign, regenerates
the whole dataset under it, and reruns the analysis of record.

`darksirens` was **READ-ONLY at `2b86a2d`** throughout; the only patches are the same
import-level pass-throughs, re-anchored in every run.  **No paper edits.**

## 16.1 What changed, and why

Two changes, both structural, both in `working/data/generate_dataset.py`, both
documented in full — with the literature check, the citations, the calibration of
every constant and the `p_pe` Jacobian derivation — in
**`working/data/DESIGN_PE.md`**.

### (v3) the measurement family

The previous family drew independent component masses with widths `f · m_TRUE`, i.e.
**functions of the latent parameter**.  `§15` measured what that costs: with the
exact per-event posterior of that family, exact host positions, the mock's own host
prior and the exact selection function, `(C − A)_pop = −1.274e-3 ± 0.113e-3`, an
**11.3σ** violation.

v3 is the literature-standard all-observable family:

```
rho_obs = rho_opt(theta) + N(0, sigma_rho)         DETECTION: rho_obs >= 8
sigma_lnMc  = A_MC  * (8/rho_obs)      ln Mc_det_obs ~ N(ln Mc_det, .)
sigma_lnq   = A_Q   * (8/rho_obs)      ln q_obs      ~ N(ln q,      .)
sigma_chi   = A_CHI * (8/rho_obs)      chieff_obs    ~ N(chieff,    .)
sigma_ang   = clip(35 deg/(1.83165 rho_obs), 1, 12)   dec_obs first, then ra_obs
```

with `sigma_rho = 1`, `A_MC = 0.08`, `A_CHI = 0.20` and the threshold 8 taken
verbatim from `GWMockCat` (Farah et al. 2023, ApJ 955, 107; arXiv:2301.00834
App. A — `posterior_utils.py::uncert_default` and `parser.py`), whose lineage is
Fishbach, Holz & Farr (2018, arXiv:1805.10270) eqs. 29–31; `A_Q = 0.60` converted
from `GWMockCat`'s `eta_uncert = 0.022` and anchored on GW150914; and the sky
constant the campaign already used.

Three consequences are worth naming.

* **`dL` is not measured on its own.**  With no projection latent (convention (a))
  `rho_opt` is an exact function of `(Mc_det, dL)`, so `(Mc_det, q, rho)` is a
  bijection of `(m1det, m2det, dL)` and **the SNR is the distance observable** —
  `GWMockCat`'s own construction.  Recording `rho_obs` *and* measuring `dL`
  separately would leave a `θ`-dependent factor `N(rho_obs; rho_opt(θ), 1)` in the
  true likelihood that darksirens cannot represent, and the identity would not
  close.  This is the one place the brief's literal reading had to be adapted, and
  the literature is what settles it (`DESIGN_PE.md` §3.1).
* **No recorded value is clipped.**  Clipping the *data* censors the likelihood and
  gives it a `θ`-dependent normalisation — the defect (c2) had to repair in v2.  The
  physical ranges `q ≤ 1`, `rho > 0`, `|chieff| ≤ 1`, `|dec| ≤ π/2` are imposed on
  the PE **prior** instead, which is free.
* **`p_pe` changes.**  With the prior flat in `(ln Mc_det, ln q, rho, chieff, ra,
  dec)`, its density in darksirens' canonical `(m1det, q, dL, chieff)` basis is
  `|∂(ln Mc, ln q, rho)/∂(m1det, q, dL)| = rho/(dL·m1det·q)` (`DESIGN_PE.md` §2.5).

### (D3) the declared photo-z is realised

`§15.4` named a second, genuine defect: the survey block declared
`dz = 3e-3 (1+z)` on redshifts copied **bit-for-bit** from the catalog the hosts are
drawn from.  The catalogs now carry `z_obs = z + N(0, DZ_SCALE (1+z))` and the
survey blocks pixelate `z_obs` with `dz = DZ_SCALE (1+z_obs)`; `z` remains the true
redshift and still drives the host draw and the event's truth.  darksirens'
per-galaxy kernel `g(z) N(z; z_obs, σ)/Z(z_obs)` is then exactly the posterior for
that galaxy's true redshift given its catalog entry, so `p_z(z|pix)` **is** the
prior for the host's true redshift.

## 16.2 The pilot — the gate, measured before anything was regenerated

Seed 100 was regenerated end to end under v3 + D3 into a separate tree
(`/hildafs/projects/phy220048p/magana/gws-agn-data-v3/seed100`) and the split was
measured there before the other four realisations were touched.  `darksirens` is
anchored `|Δ log μ| = 0` **exactly** in every run below.

### The split

`r = ⟨d ln Ẑ_i/dH0⟩ − d ln μ/dH0 = (C − A) + (A − B)`, per event, at `H0 = 67.74`,
`dh = 0.5`.  `(C − A)` is the paired per-event statistic — the measurement model.
`(A − B)` uses **B from the exact selection oracle** and **A from 1,500 redraws of
the event stage** on seed 100's own catalog (2,197,243 detections), the instrument
§15.2 built.

**matched GAL**

| term | `(A − B)` (1,534,210 replayed truths) | `(C − A)` (705 events, paired) | `r` |
|---|---|---|---|
| `p_pop` | `+9.85e-6 ± 1.29e-5` (0.76σ) | `+4.436e-4 ± 3.19e-4` (1.39σ) | `+1.396e-4` |
| → mass | `+1.028e-5 ± 1.31e-5` | `+4.488e-4 ± 3.20e-4` | `+1.432e-4` |
| catalog `p_z` | `−5.49e-5 ± 9.19e-5` (0.60σ) | `−6.990e-3 ± 3.34e-3` | `+3.694e-4` |
| Jacobian | `−9.69e-7 ± 0.90e-6` | `−1.006e-5 ± 1.07e-5` | `−4.86e-6` |
| **total** | **`−3.518e-5 ± 9.27e-5`** (0.38σ) | `−6.556e-3 ± 3.36e-3` | **`+5.150e-4 ± 7.40e-4`** |

**matched AGN**

| term | `(A − B)` (663,033 replayed truths) | `(C − A)` (295 events, paired) | `r` |
|---|---|---|---|
| `p_pop` | `−1.17e-6 ± 1.94e-5` (0.06σ) | `−4.756e-4 ± 5.00e-4` (0.95σ) | `+9.94e-4` |
| catalog `p_z` | `−1.429e-3 ± 7.13e-4` (2.0σ) | `+9.126e-3 ± 2.93e-2` | `+5.735e-3` |
| **total** | `−1.955e-3 ± 7.13e-4` | `+8.661e-3 ± 2.93e-2` | `+6.198e-3 ± 5.63e-3` |

against the v2 values `§15.3` measured on the same configuration:

| | v2 | v3 |
|---|---|---|
| GAL `(C − A)_pop` | `−1.2738e-3 ± 1.13e-4` — **11.3σ** | `+4.436e-4 ± 3.19e-4` — **1.39σ** |
| AGN `(C − A)_pop` | `−1.7354e-3 ± 1.72e-4` — **10.1σ** | `−4.756e-4 ± 5.00e-4` — **0.95σ** |
| GAL `(A − B)_pz` | `+6.383e-4 ± 0.836e-4` — **7.6σ** | `−5.49e-5 ± 9.19e-5` — **0.60σ** |
| GAL `(A − B)_tot` | `+5.836e-4 ± 0.844e-4` — **6.9σ** | `−3.518e-5 ± 9.27e-5` — **0.38σ** |
| GAL `r` (five-seed v2 / one-seed v3) | `−1.187e-3 ± 2.54e-4` | `+5.150e-4 ± 7.40e-4` |

**The gate passes.**  The 11.3σ and 10.1σ violations of `E[C] = E[A]` in the mass
channel are gone; the 6.9σ `(A − B)` the declared-but-unrealised photo-z produced is
gone.  What remains at 2σ is the AGN catalog's `p_z` term, where `A`'s own per-truth
scatter is `0.58` — the sparse catalog's spiky prior, whose sampling distribution is
manifestly non-Gaussian; in the channel that carried the residual, AGN `(A − B)_pop`
is `−0.06σ`.

### The truncation, again

The `[:N_EVENTS]` truncation and the rejection loop are re-audited under v3 and are
again clean: the replay is **bit-identical** to the record (30 truth fields, 0
differing bits, 1570 detections from 200,000 proposals), and

| | head − tail |
|---|---|
| GAL | `−1.189e-4 ± 1.64e-4` (0.72σ) |
| AGN | `+5.959e-4 ± 1.27e-3` (0.47σ) |

### The selection side, re-verified under the new rule

`P_det(θ) = Φ((ρ_opt(θ) − 8)/σ_ρ)` — one Gaussian CDF, replacing v2's
two-dimensional Gauss–Hermite average over the mass-noise latents.

| check | result |
|---|---|
| the closed form against the generator's own `observe_v3`/`detect_v3`, 30 points spanning `P_det ∈ [0.003, 0.999]`, `2e7` draws each | max `\|P_MC − P_exact\| = 1.09e-4`, max pull 2.54, **mean pull `−0.16 ± 0.20`** |
| the same, inside the generator's own validation (3e6 fresh proposals) | `0.0082793` (exact) vs `0.008281` (brute force), **`+0.03σ`** |
| the `G(b)` kernel against a direct unbinned sum over the `(m1src, q)` grid | `1e-7` where the host measure lives |
| the catalog lattice halved | `4.3e-10` (GAL), `1.3e-9` (AGN) |
| the analytic derivative vs the `dh`-halving finite difference | `1.3e-8` |
| **the injection estimator against the exact oracle, matched GAL targeted** | `−8.56e-6` = **`−0.07σ`** of the estimator's own Monte-Carlo error |
| the same, matched AGN targeted / popuni | `−6.81e-4` = `−1.28σ` / `+3.32e-4` = `+0.30σ` |

and the estimator's own **common-mode** Monte-Carlo error on `d ln μ/dH0`, which
`§14.2` named and `§10` item 7 asks to be carried, re-measured on the v3 injection
sets by a Poisson bootstrap (delta-method influence function in brackets):

| configuration | `σ_MC` on `d ln μ/dH0` | v2 |
|---|---|---|
| matched GAL, targeted | `1.197e-4` (`1.219e-4`) | `1.20e-4` |
| matched AGN, targeted | `5.334e-4` (`5.282e-4`) | `5.58e-4` |
| matched AGN, popuni | `1.101e-3` (`1.075e-3`) | `1.13e-3` |

## 16.3 The dataset, regenerated and validated

All five realisations were regenerated **end to end** — catalogs (the new `z_obs`
column), events, surveys, injections — into a separate tree
(`/hildafs/projects/phy220048p/magana/gws-agn-data-v3`), validated there, and only
then promoted by re-pointing `working/data/seed<N>`.  The superseded v2 tree is
still on disk (`promote_v3.sh delete` removes it).  Injection sizes are the
record's: `1.5e8` targeted, `4.0e8` popuni.

**Every seed passes all TWELVE checks** (nine before; V2 rewritten, V3c and V9 new):

| seed | checks | failed | GAL/AGN events | horizon `z` | detected fraction | max PE `z` (bar 0.700) | `q > 1` PE samples |
|---|---|---|---|---|---|---|---|
| 100 | 12 | **0** | 705 / 295 | 0.3105 | 7.850e-3 | 0.652 | 0 |
| 101 | 12 | **0** | 674 / 326 | 0.3338 | 7.415e-3 | 0.565 | 0 |
| 102 | 12 | **0** | 692 / 308 | 0.3133 | 7.255e-3 | 0.664 | 0 |
| 103 | 12 | **0** | 723 / 277 | 0.3519 | 7.265e-3 | 0.659 | 0 |
| 105 | 12 | **0** | 711 / 289 | 0.3868 | 7.310e-3 | 0.567 | 0 |

with, per seed, the measurement model's own certificates:

| seed | pooled PE PIT KS `p` (`ln Mc` / `ln q` / `ρ`) | truncated-`ρ` detection PIT KS `p` | `p_pe` stored vs closed form / vs numerical Jacobian | V9 photo-z pull sd (GAL) | negative `z_obs` rows |
|---|---|---|---|---|---|
| 100 | 0.920 / 0.497 / 0.749 | 0.609 | `0.0` / `6.2e-10` | 0.99996 | 1 |
| 101 | 0.966 / 0.181 / 0.983 | 0.632 | `0.0` / `5.7e-10` | 1.00001 | 0 |
| 102 | 0.844 / 0.042 / 0.310 | 0.301 | `0.0` / `5.7e-10` | 1.00004 | 1 |
| 103 | 0.647 / 0.216 / 0.650 | 0.249 | `0.0` / `6.1e-10` | 0.99998 | 0 |
| 105 | 0.534 / 0.500 / 0.742 | 0.441 | `0.0` / `5.6e-10` | 1.00001 | 3 |

**One check had to be repaired, and it is worth naming.**  `V6`'s lane comparison
required the two injection campaigns' `P_det(z)` to agree, but formed the binomial
variance from **one arm's own** `p_det`.  In a deep bin where that arm happens to
draw a near-zero count the "σ" collapses and the z-score diverges: on seed 101 a bin
at `z = 0.295` held **1** detection in 5,444 targeted proposals against 37 in 20,102
popuni ones, and the one-arm variance turned a ~2.5σ Poisson fluctuation into
**8.0σ**.  V6 now uses the standard two-proportion z test — **pooled** variance, and
a minimum of 25 detections in both arms — plus an aggregate comparison over every
bin.  Under the corrected test the same seed's 163 well-populated bins give a z
distribution of mean `−0.08`, sd `1.02`, max `|z| = 3.24`, and the aggregate
`P_det` ratio is `0.9956` (`−0.11σ`).  Across the five seeds the max `|z|` is
`2.69–3.89`.  **This is the most likely reason seed 104 was condemned by this check
in the v2 campaign**; it was a defect in the statistic, not in the data.

`recommended_kde_window` on the v3 blocks returns **3422** at `n_sigma = 8` (v2:
3410), so **`W = 4096` is unchanged** and remains correct.

## 16.4 The analysis of record, rerun

Seed 100, all six configurations, identical scan configuration throughout.
`offset` is median − 67.74.  0/201 cells rejected in every scan, before and after.

| scan | v2 (post-(b2)/(c2)) | **v3 + D3** | shift |
|---|---|---|---|
| `h0_gal_targeted` | 64.115 (−3.625) | **69.910 (+2.170)** | +5.795 |
| `h0_gal_popuni` | 64.554 (−3.186) | 69.957 (+2.217) | +5.403 |
| `h0_agn_targeted` | 99.619 (railed) | 99.797 (railed) | — |
| `h0_agn_popuni` | 99.803 (railed) | 99.833 (railed) | — |
| **`ctrl_gal_matched`** | 64.744 (−2.996) | **68.957 (+1.217)** | **+4.213** |
| **`ctrl_agn_matched`** | 68.451 (+0.711) | **68.646 (+0.906)** | +0.195 |

The mis-specified AGN production configurations still rail at the top of the scanned
range — that is the mis-specification (the AGN catalog is handed the 705 events it
does not host), and neither the redesign nor D3 addresses it.

Selection convergence is unchanged in kind: `min N_eff` over the grid is
**427,275 / 523,575 / 231,774 / 36,237** for the four production configurations
(85.5× / 104.7× / 46.4× / 7.2× the `5 N_obs` threshold; v2: 395,349 / 494,877 /
216,057 / 32,979).  `pe_variance_sum` rises with the broader PE — 0.99 → **5.34**
(GAL), 36.8 → **54.8** (AGN) — and remains far inside the campaign's inert `1e6` cap.

`results/h0_single_tracer.json` moves accordingly: `gal_h0_ci`
**64.1⁺²·³₋²·¹ → 69.9⁺¹·⁷₋¹·⁶**, `gal_h0_width` 4.46 → **3.30**, cross-check lane
64.55 → 69.96; the AGN entries stay `null` (railed).  **This is a paper-facing
number** (`working/paper/scripts/build_values.py` reads it for the `HzeroGal` /
`HzeroGalWidth` macros).  It has not been regenerated — **no paper file was
touched.**

## 16.5 The five-realisation closure table — THE VERDICT

Same configuration on all five mocks, `dark_sirens` at `log10n0 = −24`.

### GAL catalog, matched hosts

| mock | seed | events | v2 offset | **v3 offset** | 68 % half-width | truth in 68 / 90 |
|---|---|---|---|---|---|---|
| 1 | 100 | 705 | −2.996 | **+1.217** | 1.84 | **yes / yes** |
| 2 | 101 | 674 | −5.021 | **+1.251** | 2.79 | **yes / yes** |
| 3 | 102 | 692 | −10.793 | **−0.118** | 2.95 | **yes / yes** |
| 4 | 103 | 723 | −6.110 | **+2.651** | 4.02 | **yes / yes** |
| 5 | 105 | 711 | −6.582 | **−0.929** | 2.01 | **yes / yes** |
| | | | **−6.30 ± 1.28** | **+0.81 ± 0.62** | | **5 / 5** |

`t(4) = +1.32`, `p = 0.26` (v2: `t = −4.92`, `p = 0.008`).  No realisation rails.
Truth is inside the **68 %** interval on **5 of 5** realisations, against **0 of 5**
under v2.  The realisation scatter is `0.51 ×` the mean quoted half-width (v2:
`1.06 ×`), i.e. the quoted width is now, if anything, conservative.

### AGN catalog, matched hosts

| mock | seed | events | v2 offset | **v3 offset** | 68 % half-width | truth in 68 / 90 |
|---|---|---|---|---|---|---|
| 1 | 100 | 295 | +0.711 | +0.906 | 0.74 | no / yes |
| 2 | 101 | 326 | +0.767 | **+0.143** | 0.96 | yes / yes |
| 3 | 102 | 308 | +1.059 | **−0.467** | 0.57 | yes / yes |
| 4 | 103 | 277 | +1.048 | +2.012 | 0.82 | no / no |
| 5 | 105 | 289 | −0.047 | **−0.488** | 0.68 | yes / yes |
| | | | **+0.71 ± 0.20** | **+0.42 ± 0.47** | | 3 / 5, 4 / 5 |

`t(4) = +0.89`, `p = 0.42` (v2: `t = +3.51`, `p = 0.025`).

### The verdict

**Yes — both matched controls now sit on truth.**  The `+0.71 ± 0.20` AGN pedestal
(2.5σ under v2) and the `−6.30 ± 1.28` GAL deficit (4.9σ) are both gone:

| | v2 | **v3 + D3** |
|---|---|---|
| matched GAL, five realisations | `−6.30 ± 1.28`, `t(4) = −4.92`, `p = 0.008`, truth in 68 % on **0/5** | **`+0.81 ± 0.62`, `t(4) = +1.32`, `p = 0.26`, truth in 68 % on 5/5** |
| matched AGN, five realisations | `+0.71 ± 0.20`, `t(4) = +3.51`, `p = 0.025` | **`+0.42 ± 0.47`, `t(4) = +0.89`, `p = 0.42`** |

`results/closure_v3.json` records the paired-by-seed comparison: the GAL mean moves
by **`+7.12 ± 1.15`** and the AGN mean by `−0.29 ± 0.42` (the two datasets are
different draws of the whole mock, so this is an *unpaired* comparison and the
"shift" is a difference of means, not a per-event repair).

Figures: `figs/fig_closure_v3.{png,pdf}` (the before/after strip),
`figs/fig_closure_seeds`, `figs/fig_h0_recovery`, `figs/fig_guard` — all
regenerated; the v2 versions are archived in `figs_v2postfix/`.

## 16.6 What the selection estimator's own error costs, carried

`§10` item 7 asks that the common-mode Monte-Carlo error of `μ̂`'s slope be carried
rather than dropped.  Measured on the v3 injection sets and converted on the v3
per-event curvature at truth (`d² ln L/dH0²` = `−5.11e-4` per event on matched GAL,
`−6.21e-3` on matched AGN, from the post-v3 control grids):

| configuration | `σ_MC(d ln μ/dH0)` | in km s⁻¹ Mpc⁻¹ on that control |
|---|---|---|
| matched GAL, targeted | `1.197e-4` | **`± 0.23`** |
| matched GAL, popuni | `1.063e-4` | `± 0.21` |
| matched AGN, targeted | `5.334e-4` | **`± 0.09`** |
| matched AGN, popuni | `1.101e-3` | `± 0.18` |

Each realisation has its own injection campaign, so the term averages down over the
five: `± 0.10` (GAL) and `± 0.04` (AGN) on the means.  **Carried into the quoted
numbers**, the closure table reads

```
matched GAL   +0.81 +- 0.62 (realisations) +- 0.10 (selection MC)  =  +0.81 +- 0.63
matched AGN   +0.42 +- 0.47 (realisations) +- 0.04 (selection MC)  =  +0.42 +- 0.47
```

— i.e. the term is real, is now measured on this dataset, and is sub-dominant.

## 16.7 What is claimed, and what is not

**Claimed.**  Under a measurement family in which every width is a function of
recorded data, with the catalog's declared photo-z error actually realised, the
detected-set score identity closes: `(C − A)` in the mass channel — an 11.3σ (GAL)
and 10.1σ (AGN) violation under v2 — is `+1.39σ` and `−0.95σ`; `(A − B)` — a 6.9σ
violation under v2 — is `−0.38σ` on 1.53 M replayed truths; and both matched-host
`H0` controls recover truth over five independent realisations.

**Not claimed.**

* The mock is now a *different physical mock*, not a repaired one.  The component
  masses carry `σ_ln m1det ≈ 0.18` instead of `0.08`, the mass ratio is measured to
  a factor `≈ 1.6` instead of 10 %, and the distance error runs through the SNR.
  Those are the literature's numbers, but they are **not** the v2 mock's, so every
  v2 number in this document and in `ATTRIBUTION.md` describes a *different dataset*
  and none of them is invalidated — they are the diagnosis that motivated this.
* `A_Q = 0.60` is a *calibration*, not a measurement: `GWMockCat` quotes `σ_η`, and
  converting it to `σ_ln q` needs a reference mass ratio (`DESIGN_PE.md` §2.3).  The
  closure does **not** depend on it — the identity closes for any `A_Q`, because the
  width is a function of data — but the *precision* of the GAL control does.
* The AGN production configurations still rail; that is the deliberate
  mis-specification, untouched.
* `(A − B)` on matched AGN is `−1.955e-3 ± 7.13e-4` (2.7σ), entirely in the sparse
  catalog's `p_z` term, where `A`'s per-truth scatter is `0.58` and its sampling
  distribution is manifestly non-Gaussian; in the mass channel it is `−0.06σ`.  A
  sharper statement would need more replicas, not a model change.
* The five-realisation `(C − A)` / `(A − B)` split was **not** redone — only seed
  100's, which was the gate.  The five-realisation `H0` controls are the closure
  statement of record.
* The nside-32 pixelisation of the catalog prior and the finite-`nsamp` PE Monte
  Carlo are unchanged approximations, both already measured consistent with zero.

## 16.8 Files

```
working/data/
  DESIGN_PE.md                 NEW -- the literature check, the adopted family, the
                               calibration of every constant, the p_pe Jacobian
                               derivation, and the realised seed-100 numbers
  generate_dataset.py          v3 measurement family (--pe_model, default v3):
                               observe_v3 / detect_v3 / posterior_samples_v3 /
                               p_pe_v3 / v3_widths / the (Mc, q, rho) bijection;
                               D3 (--photoz_survey, default obs): the catalogs'
                               z_obs column and the survey blocks pixelated on it;
                               validation V1/V2/V2b/V3/V3b rewritten for v3, V3c
                               (the p_pe Jacobian) and V9 (the realised photo-z)
                               NEW, V6's lane test corrected to the pooled
                               two-proportion z; --n_events/--nsamp/--events_suffix
                               for pilots
  run_v3_seed.sh               NEW -- one seed, every stage, the record's ndraw
  run_v3_all.sh                NEW -- the remaining seeds, serially
  promote_v3.sh                NEW -- check / promote / delete (move-aside)
  README.md                    the v3 family, D3, and the validation table
  seed{100,101,102,103,105}    -> /hildafs/projects/phy220048p/.../gws-agn-data-v3/

analyses/analysis_1_complete_catalog_H0/
  CLOSURE.md                   this section
  README.md                    the v3 banner
  scripts/
    run_v3_pilot.sh            NEW -- the gate: P_det, the exact oracle, A/B/C
    run_v3_abmega.sh           NEW -- the (A - B) redraw campaign
    v3_pilot_summary.py        NEW -- the gate table and verdict
    attr_selmu_mcerr.py        NEW -- the lean Poisson bootstrap of sigma_MC
    run_v3_analysis.sh         NEW -- the whole analysis of record
    submit_v3_analysis.sbatch, submit_v3_controls.sbatch   NEW
    attr_selmu_pdet.py         + --pe_model v3 (P_det = Phi((rho_opt-8)/sigma_rho)
                               against the generator's own observe_v3/detect_v3)
    attr_selmu_oracle.py       + --pe_model v3 (build_G_v3), --dataroot, --events;
                               ztab extended below z = 0 and the host lattice made
                               robust to a marginally negative photo-z.  With
                               --pe_model v2 every section-11 product is unchanged
    attr_selmu_inj.py          + --dataroot / --events
    attr_abc_split.py          + --dataroot
    attr_ds_bridge.py          + dataroot / events_dir
    regen_events_notrunc.py    + --pe_model / --dataroot
    build_catalog_skyindex.py  + --dataroot / --z_column (the block's own sort key
                               is now lexsort((z_obs, pix)))
    fig_closure_after_fix.py   + --before_dir / --before_label / --after_label /
                               --what / --fig_tag  (so the same figure serves the
                               v3 comparison)
    run_seed_controls.sh       DATAROOT is now overridable
  results/                     the v3 scans of record
    h0_{gal,agn}_{targeted,popuni}.{h5,json}, ctrl_{gal,agn}_matched[_s1**].{h5,json},
    guard_h0_*.json, h0_single_tracer.json, closure_seeds.json, closure_v3.json,
    kde_window.json, v3_curvature.json
    attr_selmu_pdet_v3.json, attr_selmu_{gal,agn}_v3_s100.{json,npz},
    attr_selmu_mcerr_{gal,agn}_{targeted,popuni}_v3_s100.json,
    abc_{gal,agn}_v3_s100.{json,npz}, abc_{gal,agn}_v3_mega.{json,npz},
    v3_pilot_summary.json
  results_v2postfix/           the v2 post-(b2)/(c2) scans, archived
  figs_v2postfix/              the v2 post-(b2)/(c2) figures, archived
  figs/                        fig_h0_recovery, fig_guard, fig_closure_seeds
                               REGENERATED; fig_closure_v3 NEW

bulk:
  /hildafs/projects/phy220048p/magana/gws-agn-data-v3/seed{100,101,102,103,105}
      the v3 dataset (9.7 GB/seed)
  /hildafs/projects/phy220048p/magana/gws-agn-data-v3/scratch_ab/
      events_notrunc_full_s100.h5              the bit-identical replay
      events_notrunc_replicas_s100_n1500.h5    2,197,243 redrawn detections
  /hildafs/projects/phy220048p/magana/gws-agn-data/seed{...}
      the SUPERSEDED v2 dataset -- still on disk; `promote_v3.sh delete` removes it
```

Reproduce:

```bash
cd working/data
for S in 100 101 102 103 105; do ./run_v3_seed.sh $S; done
bash promote_v3.sh check && bash promote_v3.sh promote

cd ../analyses/analysis_1_complete_catalog_H0
./scripts/run_v3_pilot.sh 100                      # the gate: P_det, oracle, A/B/C
./scripts/run_v3_abmega.sh 100 1500                # the (A - B) redraw campaign
python scripts/v3_pilot_summary.py --seed 100      # the gate verdict

python scripts/kde_window_check.py                 # W = 4096 confirmed (req 3422)
SKIP_CONTROLS=1 SKIP_AGGREGATE=1 ./scripts/run_v3_analysis.sh   # six scans
sbatch scripts/submit_v3_controls.sbatch 101 102 103 105        # four controls
./scripts/run_guard_diag.sh
python scripts/build_single_tracer.py
python scripts/aggregate_closure.py --seeds 100 101 102 103 105
python scripts/make_figures.py
python scripts/fig_closure_after_fix.py --seeds 100 101 102 103 105 \
       --before_dir results_v2postfix --fig_tag fig_closure_v3 \
       --out_json results/closure_v3.json
for T in gal agn; do for L in targeted popuni; do
  python scripts/attr_selmu_mcerr.py --tracer $T --injections $L --seed 100 \
         --dataroot /hildafs/projects/phy220048p/magana/gws-agn-data-v3 \
         --events .../seed100/events_${T}_hosted.h5 --exact <the oracle value>
done; done
```

**Signed off — owner, 2026-08-02.**  The sign-off was conditional on the v3
dataset being the one actually consumed by analyses 1 and 2; verified before
recording it:

* `working/data/seed{100,101,102,103,105}` are symlinks into
  `/hildafs/projects/phy220048p/magana/gws-agn-data-v3` (re-pointed 2026-08-01
  08:17, before every scan in this analysis' `results/` and before analysis 2
  ran); `promote_v3.sh check` passes all 12 checks per seed
  (`pe_model=v3`, `z_column=z_obs`, `n_failed=0`).
* Analysis 1's record numbers are v3: `results/h0_single_tracer.json` carries
  `gal_h0_ci = 69.9^{+1.7}_{-1.6}` (the v3 value; v2 gave 64.115).
* Analysis 2 (`analysis_2_complete_catalog_H0_fagn`) reads the same symlinks
  through its `env.sh` `DATA_ROOT`, so the joint (H0, f_AGN) campaign is v3
  throughout.

The superseded v2 seed trees under
`/hildafs/projects/phy220048p/magana/gws-agn-data/` are already removed; only
`derived/` (analysis-1 bulk outputs, current) remains there.
