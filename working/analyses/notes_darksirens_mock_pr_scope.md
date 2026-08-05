# Scope: upstreaming the v3 measurement-family standard into `darksirens`' dark-siren mock generator

> **OWNER DECISION 2026-08-02 (supersedes the PR-split and toggle recommendations
> below where they differ): NO legacy support.** The v3 all-observable family
> becomes the generator's only behaviour; the legacy paths (latent widths,
> independent component-mass measurement, data/sample clipping, `p_pe = 1`,
> `--detection-data true`, `--pe-centering truth`) are removed, not toggled.
> Bit-identity item T12 and the `--pe-model` flag are therefore dropped; B1 is
> resolved by removal. One breaking PR: branch `feat/mock-v3-measurement-family`.

**Status: SCOPING ONLY.** Nothing was branched, edited, or opened. `darksirens` was read
read-only (`git show origin/master:…`); the pinned worktree at `2b86a2d` was not touched.

* upstream ref read: `origin/master = b122b90` (merge of #335)
* target file: `scripts/mock_dark_sirens/generate_mock_data.py` (1781 lines on master)
* docs: `docs/source/mock-data.md` (123 lines)
* mock tests on master: `tests/test_mock_detection_data.py` (20 tests),
  `tests/test_mock_pe_flat_prior_posterior.py` (5), `tests/test_mock_generator_taper.py` (13),
  `tests/test_unified_observed_mocks.py` (4), `tests/clustered_mock.py` (a fixture builder, not
  a generator test)
* validated standard: `working/data/DESIGN_PE.md` + `working/data/generate_dataset.py`
* evidence: `working/analyses/analysis_1_complete_catalog_H0/CLOSURE.md` §15, §16

Already merged upstream and **not** re-proposed: #332 (`a5efb67`, PE = flat-prior posterior of a
measurement), #334 (`f85ab7b` + `23a7d06`, threshold acts on the data the posterior conditions
on / shared noise draw), #335 (`853ded3`, sky width from observed data).

---

## 0. The one-paragraph summary

Master has already fixed the three *structural* defects (posterior-of-a-measurement, detection
deterministic in data, sky width from data). What remains is a single coherent gap: **every
remaining measurement width is still a function of the latent truth**, and master's own
`_pe_widths` docstring (L456-460) states the reason it was left that way — "a width taken from
the noisy SNR would be circular". That argument is correct *within master's data basis*, and the
resolution is not a local patch: the circularity is escaped only by drawing the SNR **first** as
its own datum (Fishbach+2018 eq. 29), which then forces `dL` to be **derived** from
`(Mc_det, rho)` rather than measured (otherwise `N(rho_obs; rho_opt(theta), sigma_rho)` is a
theta-dependent factor of the true likelihood that darksirens cannot represent). That chain —
SNR datum -> all widths `a/rho_obs` -> `dL` derived -> `(ln Mc, ln q)` instead of independent
`(m1det, m2det)` -> `p_pe = rho/(dL m1det q)` -> no clipping of any recorded value — is the v3
family, and it is *forced*, not stylistic. A minimal-diff alternative (constant log-widths) also
removes the latent dependence and is offered as A1-lite / A2-lite for a reviewer who wants a
small PR.

Recommended PR split, so no single PR is unreviewable:

* **PR-A (core, ~M):** A1, A2, A3, A5, A6, A8 — the measurement family + `p_pe`. Behind
  `--pe-model {legacy,v3}`, default `legacy` in the PR, default flip proposed separately (B1/B2).
* **PR-B (censoring, ~S):** A4 — remove every clip on recorded data and on PE samples; impose
  physical ranges on the PE prior by exact inverse-CDF truncation.
* **PR-C (catalog, ~S):** A7 + B7 + B6 — realise the declared photo-z, fix `--galaxy-density-delta`,
  guard catalog dtype.
* **PR-D (validation, ~M):** B4 — a `--validate` mode carrying the V1/V2/V3/V3b/V3c/V9 certificates,
  plus the new tests of §3.
* **PR-E (defaults/CLI, ~S):** B1, B2, B5 — the default flips and the reserved sub-seed streams.

---

## 1. The gap table, ordered by scientific importance

Legend for CLASS: **A** core measurement-family gap, upstreamable as-is; **B** upstreamable but a
design choice the owner should confirm; **C** project-specific, do NOT upstream; **D** already on
master, skip.

---

### A1 — mass measurement widths are functions of the LATENT mass
**CLASS A · size M (part of the PR-A restructure) · `_pe_widths`, `_measure`, `_posterior_samples`**

**Master does** (`generate_mock_data.py`):
```
452  def _pe_widths(m1det, m2det, dl, rho, dL_fractional_uncertainty,
...
467            "sig_m1": m1det_fractional_uncertainty * m1det,
468            "sig_m2": m2det_fractional_uncertainty * m2det}
```
called with the **true** detector-frame masses,
```
538      m1det, m2det = m1src * (1.0 + z), m2src * (1.0 + z)
539      rho_opt = _snr_from_detector_frame(m1det, m2det, dl, SNR_REF_DEFAULT)
540      widths = _pe_widths(m1det, m2det, dl, rho_opt, dL_fractional_uncertainty,
```
and the posterior is drawn with the same truth-derived widths,
```
821          sig_m1 = m1det_fractional_uncertainty * m1det      # m1det = truth["m1"][i]*(1+z)
867              m1_draws = rng.normal(m1_obs, sig_m1, nsamp)
```
Master's own `_measure` docstring (L490-492) already names this: *"The mass widths `frac *
m_det,true` share the same latent structure at a much smaller measured level (-0.05 km/s/Mpc);
they are kept and documented rather than restructured here."*

**Standard requires** (DESIGN_PE.md §0 item 1, §2.2 step 5, §1.1): every width a function of the
**recorded** `rho_obs` and nothing else — `sigma_lnMc = A_MC * (8/rho_obs)`, `A_MC = 0.08`
(GWMockCat `uncert_default["mc"]`; Fishbach, Holz & Farr 2018 eq. 30).

**Why.** `N(obs; m, f m)` carries a theta-dependent normalisation `1/(f m)`, so the flat-prior
posterior is skewed by construction (mean `+2 f^2` above `obs`) and the ensemble mean of a
non-linear functional of it need not equal the functional at the truth. Measured cost, matched
mock, exact per-event posterior of that family: `(C - A)_pop = -1.274e-3 ± 0.113e-3`, an
**11.3 sigma** violation of `E[C] = E[A]` (CLOSURE.md §15/§16.1); the same statistic under v3 is
`+4.44e-4 ± 3.19e-4`, **1.39 sigma**. Master's own `-0.05 km/s/Mpc` figure is the H0 projection
of the same defect, measured on master's own (much narrower) mass channel.

**A1-lite (minimal-diff alternative, size S).** If the reviewer will not take the basis change,
replace the multiplicative truth-scaled width with a **constant log-width**:
`ln m1det_obs ~ N(ln m1det, s_1)` with `s_1` a CLI constant. A constant width is trivially not
latent-dependent, the flat-prior posterior in `m1det` is then lognormal about the observation
shifted by `+s^2` (identical algebra to master's existing distance channel, L863), and the whole
change is ~10 lines in `_measure`/`_posterior_samples`. This does **not** fix A3 (uncorrelated
masses, no mass-distance degeneracy) — it only removes the latent dependence.

---

### A2 — the distance width is latent-dependent **by default**
**CLASS A · size S within PR-A · `_pe_widths` L461-462, `_posterior_samples` L817**

**Master does:**
```
461      frac_dl = (dL_fractional_uncertainty if dL_fractional_uncertainty is not None
462                 else np.clip(1.8 / rho, 0.08, 0.35))
...
817          frac_dl = dL_fractional_uncertainty if dL_fractional_uncertainty is not None else np.clip(1.8 / rho, 0.08, 0.35)
```
with `rho = rho_opt` (truth-derived) in `_detect_on_observation`, and — worse — with
`rho = truth["snr"][i]` in `_posterior_samples` L816, which under `detection_data="true"` is the
**projection-latent-carrying** network SNR. `--dL-fractional-uncertainty` defaults to `None`
(L1722), so the latent-dependent branch is the default path.

**Standard requires** (DESIGN_PE.md §2.2 step 5, §3.1): no free distance width at all — `dL` is
derived, and `sigma_ln dL = sqrt((5/6 · sigma_lnMc)^2 + (sigma_rho/rho)^2) = 1.133/rho`.

**Why.** Identical mechanism to A1, in the channel H0 is read from. It is also exactly the
mechanism #335 fixed for the sky (`-0.49 ± 0.08 km/s/Mpc` under the **exact** likelihood): the
width `clip(1.8/rho_opt, 0.08, 0.35)` is proportional to `dL/Mc_det^{5/6}`, i.e. it *is* a
distance observable, and a fixed-width posterior cannot represent it. The `clip(., 0.08, 0.35)`
makes the latent dependence non-smooth on top.

**A2-lite:** make `--dL-fractional-uncertainty` **required** (no `None` fallback) under
`--detection-data observed`, i.e. a data-independent constant. Size XS, one-line, and it is
already the mode our v2 dataset ran in (`SIGMA_DL = 0.10` constant) — the channel that measured
clean while the mass channel measured at 11.3 sigma.

---

### A3 — masses are measured independently in `(m1det, m2det)`, with no mass-distance correlation
**CLASS A · size M · `_measure`, `_posterior_samples`, plus a new bijection block**

**Master does:**
```
495      obs_m1det = np.clip(rng.normal(m1det, widths["sig_m1"]), 2.0, None)
496      obs_m2det = np.clip(rng.normal(m2det, widths["sig_m2"]), 1.0, None)
...
867              m1_draws = rng.normal(m1_obs, sig_m1, nsamp)
868              m2_draws = rng.normal(m2_obs, sig_m2, nsamp)
```
with defaults `0.08` / `0.10` (L1723-1724) and the distance channel drawn independently (L863).

**Standard requires** (DESIGN_PE.md §2.1-§2.2, §1.3): the measurement basis is
`(ln Mc_det, ln q, rho, chieff, ra, dec)`; `dL = 1000 · SNR_REF · (Mc_det/30)^{5/6} / rho` is
derived; `A_Q = 0.60` calibrated in §2.3 (GW150914 anchor, bracketed by the `eta_uncert = 0.022`
conversion).

**Why.** Three measured consequences (DESIGN_PE.md §0 items 2-3):
* independent 8 %/10 % component masses are an unphysically strong and *uncorrelated* mass
  measurement — real interferometry constrains `Mc` to a fraction of a percent and `q` hardly at
  all;
* **18.4 % of v2 PE samples had `q = m2det/m1det > 1`** and were discarded by the population
  prior — a fifth of the Monte Carlo spent on a null region. Under v3 it is **0 %** by
  construction (CLOSURE.md §16, DESIGN_PE.md §6);
* the real degeneracy runs through the SNR (`rho ~ Mc_det^{5/6}/dL`), so a chirp-mass error *is*
  a distance error. Breaking that link makes the spectral-siren lever against a `35 ± 5 Msun`
  peak far stronger and far more curved than any real catalog's — i.e. the mock's H0 information
  content is not the one being claimed.

---

### A4 — recorded data and PE samples are **clipped** (a censored likelihood)
**CLASS A · size S · `_measure` L495-496, L508-512; `_posterior_samples` L858-861, L872-877**

**Master does:**
```
495      obs_m1det = np.clip(rng.normal(m1det, widths["sig_m1"]), 2.0, None)
496      obs_m2det = np.clip(rng.normal(m2det, widths["sig_m2"]), 1.0, None)
508          "obs_chieff": np.clip(rng.normal(chi, widths.get("sigma_chi", 0.08)), -1.0, 1.0),
511          "obs_dec": np.clip(dec + rng.normal(0.0, sigma_ang),
512                             -0.5 * np.pi, 0.5 * np.pi),
```
and again on the **samples**:
```
873          arrays["dec"].append(np.clip(dec_centre + ddec, -0.5 * np.pi, 0.5 * np.pi))
875          arrays["m1det"].append(np.clip(m1_draws, 2.0, None))
876          arrays["m2det"].append(np.clip(m2_draws, 1.0, None))
877          arrays["chieff"].append(np.clip(chi_draws, -1.0, 1.0))
```

**Standard requires** (DESIGN_PE.md §2.2 "No observation is clipped or truncated", §2.4):
truncate the **prior**, never the data; every truncated PE draw is an exact inverse-CDF
truncated normal (`Phi^-1(Phi(a) + u(Phi(b)-Phi(a)))` via `ndtr`/`ndtri`, scipy-free).

**Why.** Clipping the data makes the measurement model *censored*: the likelihood acquires a
theta-dependent normalisation `P(obs = boundary | theta) = 1 - Phi(...)`, and the exact
flat-prior posterior is then no longer a simple (truncated) normal — so master's L863-869
posterior is not the posterior of the L494-512 measurement wherever a clip bites. Clipping the
*samples* is worse: it puts a **point mass** at the boundary, which is not a density at all and
which `p_pe` cannot describe.

**How active is it on master?** The `dec` clip is the live one: `sigma_ang` reaches the 12 deg
cap (L464/L501), and the fraction of the isotropic sky within 12 deg of a pole is ~2.2 %, so of
order 2 % of events carry a censored declination channel and a point mass at `|dec| = pi/2` in
their sky posterior. The `m1det >= 2`, `m2det >= 1`, `|chieff| <= 1` clips are inert at the
default population but are not inert under `--proposal uniform` or a stress population. (For
comparison: DESIGN_PE.md §2.2 records the analogous `eta`-boundary case in GWMockCat, where the
detected `eta` median sits **0.17 sigma** from the boundary — censoring active for the *median*
event, not a corner case.)

---

### A5 — `p_pe = 1` is the wrong density in darksirens' canonical basis
**CLASS A · size S · `_posterior_samples` L878**

**Master does:**
```
878          arrays["p_pe"].append(np.ones(nsamp))
```
with the docstring (L779-783) explicitly deferring the question: *"Whether the consuming
likelihood's canonical basis uses `q = m2det/m1det` instead of `m2det` — which would introduce an
`m1det` Jacobian into `p_pe` — is a separate convention question this change does not touch."*

**It is not a convention question — darksirens has already answered it.**
`darksirens/inference/utils.py` (module docstring, "Integration variables"):

> *"both posterior samples and detected injections are integrated in the same coordinates
> `(m1det, q, dL, chieff, sky pixel)`, not in `(m1det, m2det, dL)`. Any proposal density divided
> out by the likelihood (`p_pe` …) must be expressed per unit `m1det`, per unit `q`, per Mpc of
> `dL` … A density native to `(m1det, m2det, dL)` is converted to the canonical `(m1det, q, dL)`
> basis by multiplying by `|dm2det/dq| = m1det`."*

**Standard requires** (DESIGN_PE.md §2.5): with the prior flat in
`(ln Mc_det, ln q, rho, chieff, ra, dec)`,
`p_pe ∝ rho/(dL · m1det · q) ∝ Mc_det^{5/6}/(dL^2 · m1det · q)`. Under master's *current*
(pre-A3) basis the correct value is the simpler `p_pe ∝ m1det`.

**Why.** darksirens renormalises `p_pe` per event, so only the shape matters — but the shape is
wrong by a factor `m1det` across the samples of every event, which mis-weights the mass channel
of every mock master has ever produced. This is a **one-line fix today** (`p_pe = m1det`) that
becomes `p_pe_v3` under A3; it is the highest value-per-line item in the whole list and can ship
independently of everything else.

---

### A6 — the RA measurement width uses `cos(dec_TRUE)`; the posterior uses `cos(dec_obs)`
**CLASS A · size XS · `_measure` L509-510 vs `_posterior_samples` L864 / L853-854**

**Master does** — measurement side, latent `dec`:
```
509          "obs_ra": (ra + rng.normal(0.0, sigma_ang
510                                     / np.maximum(np.cos(dec), 0.1))) % (2.0 * np.pi),
```
posterior side, observed `dec_obs`:
```
864              dra = rng.normal(0.0, sigma_ang / max(np.cos(dec_obs), 0.1), nsamp)
```
(and the non-recorded branch L853-854 draws `ra_obs` with `cos(truth["dec"][i])` while L864 again
uses `dec_obs`).

**Standard requires** (DESIGN_PE.md §2.2 convention (b2)): `dec` is measured **first**, and the RA
width is formed from the declination **already recorded** —
`sigma_ra = sigma_ang / max(cos dec_obs, 0.1)`, on both sides.

**Why.** Two defects in one line: (i) the stored posterior is not the exact posterior of the
recorded measurement (the widths differ), so V3-style PIT calibration of the RA channel cannot
pass; (ii) the measurement width is a function of the latent `dec`, the exact defect #335 fixed
for `sigma_ang` — and it survived #335 because #335 only touched the *magnitude* of the sky
width, not the `cos dec` factor. `sigma_ang/cos dec` is `O(1)`-wrong near the poles, precisely
where the 12 deg cap and the A4 `dec` clip also bite. This is a genuine bug and is XS to fix.

---

### A7 — the catalog declares a photo-z error it never realises
**CLASS A · size S · `_generate_complete_catalog`, `write_mock_data` L1357-1364, L1550-1553**

**Master does:**
```
1358      zerr = survey.redshift_error_floor + survey.redshift_error_slope * (1.0 + complete["z"])
...
1361      pixelated = _pixelate_catalog(
1362          complete["ra"][observed], complete["dec"][observed], complete["z"][observed],
1363          zerr[observed], weights, args.nside, marks=mark_obs,
1364      )
...
1552          f.create_dataset("Z", data=complete["z"][observed], compression="gzip", shuffle=True)
1553          f.create_dataset("ZERR", data=zerr[observed], compression="gzip", shuffle=True)
```
with `redshift_error_floor = 0.0005`, `redshift_error_slope = 0.0015` (L92-93), i.e. a declared
`sigma_z ≈ 0.002-0.0035` on redshifts that are **bit-for-bit the true redshifts the GW hosts were
drawn from**. `_generate_complete_catalog` (L555-566) has no `z_obs` column at all.

**Standard requires** (DESIGN_PE.md §3.3): the catalog carries **two** redshift columns —
`z` (true; drives the host draw and the event truth) and `z_obs = z + N(0, DZ_SCALE (1+z))` —
and the survey block pixelates on `z_obs` with the declared width `dz = DZ_SCALE (1+z_obs)`.

**Why.** The declared-but-unrealised kernel makes the likelihood smooth a comb that carries no
error, so darksirens' per-galaxy kernel `g(z) N(z; z_g, sigma_g)/Z(z_g)` is **not** the Bayesian
posterior for the host's true redshift. Measured: `(A - B)_pz = +6.383e-4 ± 0.836e-4`, a
**7.6 sigma** internal inconsistency, driving `(A - B)_tot` to 6.9 sigma; after realising it,
`-5.49e-5 ± 9.19e-5`, **0.60 sigma** (CLOSURE.md §16.1-§16.2). Master is in exactly the
pre-fix state.

**Sub-item A7b (do not clip `z_obs`).** `z_obs` can go negative for a galaxy at `z ~ 0`. Do not
clip — clipping re-introduces censoring (A4). Record the realised count in metadata instead: our
seed 100 had **1 negative row out of 151,179,870**, unclipped, and V9 reports it.

---

### A8 — the truth group cannot support an exact-`P_det` oracle test
**CLASS A · size S · `_draw_events_until_detected` L724, `write_mock_data` L1588-1603**

**Master does:**
```
724              block = dict(z=z, ra=ra, dec=dec, dl=dl, m1=m1, m2=m2, q=q, chi=chi, snr=snr)
```
where under `detection_data="observed"` `snr` **is** `obs["obs_snr"]` (L698) — so the file stores
the *observed* amplitude but never the *optimal* one `rho_opt(theta)`; and under
`detection_data="true"` `snr` is the projection-carrying network SNR, with the projection latent
itself never stored. The per-event widths are stored (`obs_sigma_dl`, `obs_sigma_ang`,
`obs_sig_m1`, `obs_sig_m2`, L513-516) only in `observed` mode.

**Standard requires** (DESIGN_PE.md §2.6, §3.2): store `rho_obs` **and** `rho_true = rho_opt(theta)`
plus every width, so that "every width is recomputable from stored data" and the closed-form
selection function `P_det(theta) = Phi((rho_opt(theta) - 8)/sigma_rho)` can be checked against
the generator's own draw.

**Why.** Without `rho_true` in the file you cannot run the two checks that certify the selection
side — the exact-`P_det` oracle (V3b: `0.0082793` exact vs `0.008281` brute force, `+0.03 sigma`)
and the Malmquist observables (V1: `frac_detected_with_TRUE_snr_below_threshold` and
`frac_rejected_with_TRUE_snr_above_threshold`, both of which are **exactly 0** for a
true-parameter cut and strictly positive for a data-space cut). Those two numbers are the
cheapest possible proof that #334 is actually in effect in a given file.

**Sub-item A8b:** also write the **rejected** proposals (or a subsample) to a sidecar, as our
`stage_events` does; V1 checks that every rejected row fails the threshold on its own recorded
`rho_obs`, which is what closes the loop on "detection is a deterministic function of the data".

---

### B1 — `--detection-data` should default to `observed`
**CLASS B · size XS · `parse_args` L1704**

Master: `parser.add_argument("--detection-data", choices=DETECTION_DATA_MODES, default="true", …)`
with the help text (L1716-1717) explaining the default is `true` only because *"it changes the
event population of every existing mock"*.

The whole v3 chain is conditional on `observed`. With A1-A6 landed, the *"demonstrably more
self-consistent but not demonstrably complete"* caveat in `f85ab7b` no longer applies — the
campaign has since measured completeness (12/12 validation checks, `(C-A)` at 1.39 sigma,
`(A-B)` at 0.38 sigma). Breaking change: every existing mock's event population moves, and
`--snr-ref` needs recalibration (master's own note: the detected fraction rises **5.75x**;
`snr_ref = 6.278` reproduced the `true` arm's detected fraction on our mock — and that is
exactly the `SNR_REF_DETECT = 6.278363879917771` our dataset ships with). **Owner decision.**

---

### B2 — expose the family as `--pe-model {legacy,v3}` and choose the default
**CLASS B · size S (the flag; the body is A1-A5) · `parse_args`, `write_mock_data`**

Mirrors our `--pe_model {v2,v3}` with `PE_MODEL_DEFAULT = "v3"` (DESIGN_PE.md §4). Keeping
`legacy` bit-for-bit reproducible is the same discipline `--pe-centering truth` already
established on master (`test_truth_centering_reproduces_the_historical_draws_bit_for_bit`), and
that test is the template. Record the model and every constant in `metadata_json` and in the
events-file attrs (master already writes `pe_centering`, `detection_data`, `snr_ref` at
L1574-1576). Breaking format change: `truth/` gains `obs_rho`, `obs_lnmc`, `obs_lnq`,
`obs_sig_*`, `snr_true`; `p_pe` stops being all-ones. **Owner decision on the default.**

---

### B3 — keep `--sky-uncertainty-deg` as the data-independent escape hatch
**CLASS B · size XS · `_detect_on_observation` L543-546**

Master already routes an explicit constant through unchanged (L543-546, and
`test_explicit_sky_uncertainty_is_passed_through`). Under v3 the SNR-derived branch must move
from `clip(35/rho_obs_opt(obs masses, obs dL), 1, 12)` to `clip(35/(1.83165 rho_obs), 1, 12)`
(DESIGN_PE.md §1.5) — i.e. from *the amplitude recomputed from observed masses and distance* to
*the recorded SNR itself*. **These two are not equivalent under v3** because under v3 there is no
independently-observed `dL`. Under master's current basis they *are* both functions of recorded
data, so #335's form is already correct there — this line only changes as part of A3.

---

### B4 — a validation mode
**CLASS B · size M · new `--validate` / `--stage validation`**

Master has no in-generator validation; `run_mock_data_test.sh` only checks *ingestibility*
(`load_all_data` runs). Our generator ships 12 certificates (DESIGN_PE.md §6). The upstreamable
subset is V1, V2, V2b, V3, V3b, V3c, V9 — see §3 for the test-shaped versions. Writing a
`validation.json` next to the HDF5 products, as ours does, makes every mock self-certifying.

---

### B5 — reserved sub-seed streams and per-stage seed/knob CLI
**CLASS B · size S · `write_mock_data` L1311, `parse_args`**

Master threads **one** `np.random.default_rng(args.seed)` (L1311) through catalog -> marks ->
survey -> events -> selection in sequence. Consequence: changing `--nobs`, or any PE width under
`--detection-data observed` (the event loop's draw count changes), shifts the selection
injections' key (L1240 `jax.random.PRNGKey(int(rng.integers(...)))`) and therefore changes the
selection set. That makes A/B comparisons at fixed catalog impossible without regenerating.

Pattern to port (`analysis_0_pure_tracer_H0/results/generate_dataset_extension.diff`): a
`sub_seeds(seed)` map with **named, auditable child streams** (`seed*1000 + k`), reserved offsets
for explicitly-requested extra draws, and CLI knobs that **default to `None` so a no-flag run is
bit-identical to the record**:

```
+    _FAGN = F_AGN if getattr(args, "f_agn", None) is None else float(args.f_agn)
+    _SEED_EV = (seeds["events"] if getattr(args, "seed_events", None) is None
+                else int(args.seed_events))
```
with the metadata recording `seed_events_is_record_default` / `planted_f_agn_is_record_default`.
Upstream analogue: `--seed-catalog`, `--seed-events`, `--seed-selection`, all defaulting to the
derived value. Bit-identity of the default path is the acceptance criterion and is testable.

---

### B6 — guard the catalog storage dtype
**CLASS B · size XS · `_pixelate_catalog` L586-588, `write_mock_data` L1561-1562**

Master builds `zgals`/`dzgals`/`wgals` as **float64** (`np.full(..., 100.0)`, `np.zeros`) and
writes them with `compression="gzip", shuffle=True` without a dtype cast — so master is **not**
exposed to the trap below. This item is purely defensive.

**The trap** (`analysis_1_complete_catalog_H0/README.md` L103-113): `darksirens/redshift/
completion.py::_kde_dndz_obs` builds the truncated-kernel mass in the *catalog's* storage dtype
and clamps at `1e-300` (`mass = jnp.maximum(mass, 1e-300)`), while the kernel is promoted to the
float64 `zgrid`. Padded rows sit at `z = 100.0`, so `mass = 0` exactly — and `1e-300` **is not
representable in float32**, so every padded slot evaluates `0/0 = NaN`, the `* real_gal` mask
cannot remove it, every row carrying any padding returns all-NaN, and the survey-global
normaliser goes NaN. The likelihood was `-inf` in every cell of every grid. Add (i) an explicit
`dtype=np.float64` on the three pixelated arrays, (ii) a one-line assertion at write time, (iii)
the regression test T9 in §3. Cheap insurance against a failure mode that cost this campaign a
full dataset regeneration.

---

### B7 — `--galaxy-density-delta` changes the catalog **count** but not its **redshift distribution**
**CLASS B (a genuine bug; inert at the default) · size S · `_generate_complete_catalog` L561**

**Master does:**
```
1305  def _galaxy_count_from_density(n0: float, delta: float, grids) -> int:
1306      density_weighted_volume = jnp.trapezoid(grids["dvc_dz"] * (1.0 + grids["z"]) ** delta, grids["z"])
1307      return max(1, int(round(n0 * density_weighted_volume)))
```
but the redshifts are drawn from the **un-evolved** comoving-volume CDF:
```
111  def _sample_uniform_comoving_z(rng, grids, n):
112      return np.interp(rng.uniform(size=n), grids["vc_cdf"], grids["z"])
561      z = _sample_uniform_comoving_z(rng, grids, n_galaxies)
```
where `vc_cdf` is the normalised cumulative of `dvc_dz` alone (L106-107). So for `delta != 0` the
catalog has the right total count and the **wrong** `dN/dz`. Default is `delta = 0.0` (L93,
L1688), so the shipped fixture is unaffected; `docs/source/mock-data.md` L16-19 advertises the
knob and the validation runner mirrors it into the inference survey JSON, which means an
inference run can be handed a `log10n0`/`delta` pair that no catalog realises. Fix: build a
second CDF from `dvc_dz * (1+z)^delta` and sample from it (this is the "constant-comoving-density
sampling" class of bug the campaign hit in its own GLASS route). Also add T10.

---

### C — do NOT upstream
**CLASS C · no change**

For the record, so a future reader does not try: the AGN/GAL two-tracer mixture and `--f_agn`;
the GLASS density-field catalog builder (`_glass_build`, its dedicated venv, the Schechter and
AGN luminosity functions, `MAG_LIMITS`); the targeted-AGN injection branch
(`MIX_TARGETED_AGN = 0.25`, `p_targeted_density`, `SurveyPixelMap`, the H0-range-covering
distance kernel `TGT_R_LO/TGT_R_HI`); `nside = 32` and `DZ_SCALE = 3e-3` as *values* (the
*mechanism* of A7 is upstreamable, the constants are ours); our seed/path layout and `META.json`
schema; the `SNR_REF_DETECT = 6.278…` constant (upstream should expose `--snr-ref`, which it
already does, not hard-code ours).

The **generic** parts of the above — reserved sub-seed streams (B5), realised photo-z (A7),
non-clipping (A4) — are already broken out as A/B items and are the only pieces that should
cross the boundary.

---

### D — already on master, skip
**CLASS D**

| what | where on master | source |
|---|---|---|
| PE samples are the flat-prior posterior of a measurement, not a cloud around truth | `_posterior_samples` L835-870, `--pe-centering observed` **default** L1696 | #332 `a5efb67` |
| distance posterior algebra (`ln dL ~ N(ln d_obs + s^2, s)`) correct for a flat `dL` prior | L863 | #332 |
| detection thresholds the SNR of the recorded measurement; the **same** measurement is handed to the posterior | `_detect_on_observation` L549-552, `use_recorded_observation` L836-849 | #334 `f85ab7b` |
| the projection latent is dropped under `observed` (our convention (a) in its strongest form) | L1200-1211, docstring L643-649 | #334 |
| injections apply the same rule but store **true** parameters and an untouched `pdraw` | `_draw_selection_batch` L1007-1018, docstring L956-960 | #334 |
| numpy and JAX selection paths implement the same observed rule | `_make_selection_kernel` L1200-1211 + test | #334 |
| the recorded observation is written exactly once, with both sides asserted equal | L1597-1603 | #334 `23a7d06` |
| the sky width is a function of the recorded data | `_measure` L498-501 | #335 `853ded3` |
| `--detection-data observed` requires `--pe-centering observed` | L1468-1471 | #334 |
| catalog arrays stored float64 (see B6) | L586-588 | — |
| galaxies uniform in **comoving volume**, isotropic on the sky | L111-112, L119-123 | — |

---

## 2. Where master is BETTER than, or defensibly different from, our reference

These are genuine judgment calls. Our implementation is not automatically right.

**(i) Master's detection statistic needs no separate SNR datum — and that is arguably cleaner.**
Master computes `rho_obs = _snr_from_detector_frame(obs_m1det, obs_m2det, obs_dL, snr_ref)`
(L549-550): a *deterministic function of data already recorded*, not an extra datum. Our v3
records `rho_obs` as its own datum, which is what forces `dL` to be derived (DESIGN_PE.md §3.1 —
"the one place the brief's literal reading had to be adapted"). Master's route keeps an explicit
`dL` channel and needs no bijection, no `p_pe` Jacobian, and no `(Mc, q)` reparametrisation.
**The criterion is not which basis is prettier, it is whether every width is a function of the
recorded data in that basis.** Master's basis fails that test only because of A1/A2 — and, as
noted in §0, it *cannot* be fixed inside that basis without either (a) constant widths
(A1-lite/A2-lite, which works and is small) or (b) an SNR datum drawn first (which forces the
v3 chain). **A reviewer could reasonably prefer A1-lite + A2-lite + A4 + A5 + A6 + A7 and stop
there**; that combination is internally consistent, is a much smaller diff, and closes every
latent-dependent-width defect. What it does not buy is the *physical* mass-distance degeneracy
(A3) — a modelling-realism gain, not a correctness one. Flagging this explicitly so the owner
can choose scope.

**(ii) Master's population model is more carefully matched to the inference than ours needs to
be, and we depend on it.** `_powerlaw_pdf`/`_pair_pdf`/`_mass_spin_pdf` (L196-425) reproduce
darksirens' `powerlaw+peak` including the logistic edge tapers, the per-component pairing floors
(`_PAIR_M_LO = 1.0`, `_PAIR_DM = 0.01` for the Gaussian peak vs `(mmin, dm_min)` for the power
law), and the exact normalisation grids (`n_mass=500` on `[1,200]`, `n_q=200` on `[0,1]`) with
the row-max factoring of `base.py:221`. The comment at L410-417 records that the naive
`p_m1 * _q_pdf` form drove `p_inference/p_draw -> e^31`. Our `generate_dataset.py` **imports these
functions from upstream** (`import_gmd`) rather than reimplementing them. Nothing to change; do
not regress it.

**(iii) The `population+uniform` defensive mixture proposal is master's, and is better than a
bare population proposal.** L974-989 / L934: every row's `pdraw` is `0.9 p_pop + 0.1 p_unif`, so
the `0.1` floor bounds the importance weight and keeps `Neff` healthy under a stress population.
Our injection campaign uses the same idea with our own mixture weights. Keep.

**(iv) The JAX selection kernel evaluating `pdraw` in numpy on the *detected subset only*
(L1030-1032, L1264) is a good pattern** — the stored `pdraw` is byte-for-byte the reference
density with no second implementation to drift. Keep, and extend it to any new v3 quantity.

**(v) `truth`/observation double-write guard (L1597-1603).** The `assert np.array_equal(...)`
with the message *"detection and PE have drifted apart"* is a better structural guard than
anything in our generator, which relies on validation-stage checks instead. Worth copying **into**
our generator, not out of it.

**(vi) Divergent but equivalent: the sky-width amplitude scale.** Master uses
`clip(35/rho, 1, 12)` deg on the `SNR_REF_DEFAULT = 11.5` scale; ours is
`clip(35/(1.83165 rho_obs), 1, 12) = clip(19.1069/rho_obs, 1, 12)` because our `rho_obs` lives on
the `SNR_REF_DETECT = 6.278` detection scale (DESIGN_PE.md §1.5). Same model, different
amplitude convention; the realised width distribution is the same `[1.00, 2.39] deg`. No change
needed — but any port must carry the scale factor explicitly or the sky widths silently change by
1.83x.

**(vii) Genuinely open, not a gap: `sigma` treated as known on both sides.** Master's docstring
L775-777: *"`s` is treated as known per event … which keeps the Gaussian/lognormal conjugacy
exact; real parameter estimation infers the width from the data."* Our v3 makes the same
simplification (widths are stored, not inferred). Both are the standard mock-PE idealisation
(GWMockCat does the same). Not a defect in either; worth one sentence in the upstream docs.

---

## 3. Test coverage — what master would catch, and what the PR must add

### 3.1 Would master's tests catch a regression in each area?

| area | master's coverage | verdict |
|---|---|---|
| A1 mass widths latent | none — `test_sigma_ang_is_not_derived_from_the_latent_truth` pins **only** `sigma_ang` | **no** |
| A2 distance width latent | none | **no** |
| A3 mass basis / `q<=1` | none; no test ever inspects `m2det/m1det` | **no** |
| A4 clipping | none; no test asserts absence of boundary atoms | **no** |
| A5 `p_pe` | none; `p_pe` is never read by any mock test | **no** |
| A6 RA width consistency | none; only `obs_sigma_ang` is pinned, never `sigma_ra` | **no** |
| A7 photo-z realisation | none; no test compares survey `Z` to the true catalog `z` | **no** |
| A8 `rho_true` / Malmquist | partial — `test_every_detected_event_passes_the_threshold_on_its_own_observation` and `test_observation_is_not_the_truth` pin the *rule*, nothing pins `P_det` | **partial** |
| #332 posterior algebra | **yes** — `test_distance_samples_match_analytic_flat_prior_posterior`, `test_legacy_centring_biases_the_distance_scale_and_the_fix_removes_it` | yes |
| #334 shared draw | **yes** — `test_posterior_conditions_on_the_recorded_observation`, `test_injections_store_true_parameters_under_observed_detection`, `test_injection_numpy_and_jax_paths_agree_on_the_observed_rule` | yes |
| #335 sky width | **yes** — the three `sigma_ang` tests | yes |
| legacy bit-identity | **yes** — `test_truth_centering_reproduces_the_historical_draws_bit_for_bit`, `test_injection_true_mode_unchanged`, `test_default_is_the_historical_rule` | yes |
| population/pairing/`pdraw` | **yes** — 13 tests in `test_mock_generator_taper.py` | yes |
| B6 dtype | none | **no** |
| B7 `delta` | none | **no** |
| end-to-end CLI | `test_write_mock_data_end_to_end_observed` + `run_mock_data_test.sh` (ingestibility only) | partial |

Summary: master's mock tests are strong on *what the three merged PRs changed* and blind
everywhere else. Every A-item needs its own test.

### 3.2 New tests the PR needs

**T1 — no clipping anywhere (A4).** For a run with widths large enough to push the boundaries:
assert **zero** exact ties at each physical boundary in both `truth/obs_*` and the PE sample
arrays (`m1det == 2.0`, `m2det == 1.0`, `|chieff| == 1.0`, `|dec| == pi/2`), and assert that
recorded observables **are allowed** to fall outside the physical range. Under v3 additionally:
every `ln q` PE sample satisfies `q <= 1` **and** the `ln q` sample set has no atom at `q = 1`
(prior truncation by inverse CDF, not a clip).

**T2 — every width is a function of the recorded data (A1, A2, A6).** Generalise
`test_sigma_ang_is_a_function_of_the_recorded_observation`: recompute `sig_lnmc`, `sig_lnq`,
`sig_chieff`, `sigma_ang` from the stored `rho_obs` **alone**, bitwise; and `sig_ra` from
`sigma_ang / max(cos(obs_dec), 0.1)`, bitwise. Plus the negative control (the analogue of
`test_sigma_ang_is_not_derived_from_the_latent_truth`) for **each** width.

**T3 — the endpoint identity: observed-family detection statistics vs an exact-`P_det` oracle
(A8).** Two halves.
* *(a) closed form.* On a grid of ~30 `theta` spanning `P_det ∈ [0.003, 0.999]`, compare
  `P_det(theta) = Phi((rho_opt(theta) - rho_th)/sigma_rho)` against a brute-force Monte Carlo of
  the generator's own `observe`/`detect` (2e7 draws each): require `max |P_MC - P_exact|` at the
  Monte-Carlo floor and `mean pull` consistent with 0. Our measured values:
  `max |ΔP| = 1.09e-4`, `mean pull -0.16 ± 0.20`; in-generator variant `0.0082793` vs
  `0.008281`, `+0.03 sigma`.
* *(b) Malmquist observables.* On a generated file, assert
  `frac_detected_with_TRUE_snr_below_threshold > 0` **and**
  `frac_rejected_with_TRUE_snr_above_threshold > 0`. Both are **exactly 0** for a true-parameter
  cut, so this is a two-line test that a `detection_data="true"` regression cannot survive. (Under
  master's current basis the oracle is not a single `Phi` — the observed amplitude is a function of
  three noisy quantities — so (a) is only available after A3; (b) is available **today** and should
  ship immediately.)

**T4 — derived-`dL` consistency / the bijection (A3).** For every stored PE sample,
`rho * dL == 1000 * snr_ref * (Mc_det/30)^{5/6}` to float tolerance, and the round trip
`(m1det, m2det, dL) -> (Mc_det, q, rho) -> (m1det, m2det, dL)` is the identity. Our V3 measured
`<= 4.4e-16`. Also assert the stored diagnostic point estimates `obs_m1det/obs_m2det/obs_dL` equal
the bijection evaluated at the observation — and that the PE code path never reads them.

**T5 — the `p_pe` Jacobian, two independent routes (A5).** Assert the stored `p_pe` equals the
closed form (`m1det` in master's current basis; `rho/(dL m1det q)` under v3) **exactly**, and that
the closed form equals a **numerical** Jacobian of the bijection by central differences. Our V3c:
`0.0` and `6.2e-10` respectively. This is the test that makes A5 un-regressable.

**T6 — posterior calibration, PIT/KS (A1-A6 jointly).** Pooled PIT of the truth under each stored
posterior is uniform (KS `p` not tiny), and the measurement pulls are `N(0,1)`. For the SNR
channel use the **truncated** pull `(rho_obs - rho_opt)/sigma_rho` truncated at `(8 - rho_opt)` on
the detected set — the detected-set PIT is uniform only for the truncated reference. Our V1/V3:
KS `p = 0.61` (truncated SNR pull), and `0.92 / 0.50 / 0.75 / 0.32 / 0.49 / 0.65` for
`ln Mc / ln q / rho / chieff / dec / ra`; pull sd `1.004 / 1.010 / 0.957`.

**T7 — generative replay (structural, cheap).** Re-run `observe`/`detect` on fresh proposals with
the recorded seed and two-sample-KS the detected `rho` pull and `z` against the stored file. Our
V3b: `p = 0.55` and `0.48`. Catches any silent draw-order change — the failure mode
`--pe-centering truth`'s bit-identity test protects the legacy path from and nothing protects the
new path from.

**T8 — the photo-z is realised (A7).** The survey block's `zgals` equal the catalog's `z_obs`
bitwise; they are **not** the true redshifts (assert a nonzero max `|Δ|`); the declared `dzgals`
equal the model at `z_obs` bitwise; the realised pull `(z_obs - z_true)/sigma` has sd 1 and passes
KS; and any `z_obs < 0` row survives to the file unclipped (count recorded, not asserted zero).
Our V9: pull sd `0.99996`/`0.99958`, KS `p = 0.65`/`0.87`, 1 negative row of 151,179,870.

**T9 — the padded-row KDE dtype guard (B6).** Assert the pixelated arrays are `float64`, and run
`darksirens.redshift.completion._kde_dndz_obs` on a pixel row that contains padding, asserting the
result is finite. Direct regression against the `0/0 = NaN` trap; ~15 lines.

**T10 — `--galaxy-density-delta` shapes the redshift distribution (B7).** With `delta != 0`,
KS the realised catalog `z` against `dV_c/dz (1+z)^delta`; currently fails.

**T11 — end-to-end, extended.** Extend `test_write_mock_data_end_to_end_observed` to assert the
new `truth/` keys exist, `p_pe` is not all-ones, and `metadata_json` records the pe-model and every
constant. Keep `run_mock_data_test.sh`'s `load_all_data` ingestibility check as-is.

**T12 — default-path bit-identity (B5).** With every new seed/knob flag unset, the generated files
are byte-identical to the pre-change generator at the same `--seed`. This is the acceptance
criterion for B5 and the template already exists
(`test_truth_centering_reproduces_the_historical_draws_bit_for_bit`).

---

## 4. Documentation changes

`docs/source/mock-data.md` needs, at minimum:

* the numbered "the simulation is intentionally simple" list (L62-69) rewritten — step 6 currently
  reads *"Write posterior samples around detected truth values"*, which #332 already made false;
* a new section on the measurement family: the generative order, the `a/rho_obs` width law with the
  GWMockCat/Fishbach constants and citations, the derived-`dL` structure, and the statement that no
  recorded value is clipped;
* the `p_pe` basis statement, quoting `darksirens/inference/utils.py`'s canonical-coordinates
  paragraph;
* the two-redshift-column catalog contract (`z` vs `z_obs`), and the warning that declaring an
  error without realising it is a 7.6-sigma internal inconsistency;
* the `--snr-ref` recalibration note for `--detection-data observed` (currently only in the CLI
  help, L1718-1721);
* the `--galaxy-density-delta` semantics once B7 is fixed (L16-19 currently over-promises).

Citations to carry into both the docs and the code constants (DESIGN_PE.md §5): Fishbach, Holz &
Farr 2018 (arXiv:1805.10270) eqs. 29-31; Fishbach & Holz 2020 (arXiv:1905.12669) App. B; Farah et
al. 2023, ApJ 955, 107 (arXiv:2301.00834) App. A + the `GWMockCat` code (CC0); Fairhurst 2009 /
Berry et al. 2015 for the `Delta Omega ~ rho^-2` sky scaling; GW150914 (arXiv:1602.03840) for the
`A_Q` anchor. Note explicitly that Fishbach & Holz's `0.2 z/(1+z)` chirp-mass term is **rejected**
because it is latent-dependent — the same defect class as A1 — and that `GWMockCat` itself drops it.
