# Adversarial module review — multitracer `fagn` dark-siren pipeline

Reviewer: fresh adversarial agent (no prior authorship). Date: 2026-07-08.
Repo: `/hildafs/projects/phy230014p/magana/gws-agn` (reviewed read-only).
Scope: GOAL.md §3.2 selection-normalization SEV-1 target; exact flat-prior dL
sampler; `alpha_agn_true` vs injector; `src/darksirens_multitracer`; known-issue
verification (F-P1/F-P2); general hunt (grids, bounds, Jacobians, logsumexp,
`(1+z)^gamma`, coverage/recovery/summarize/fisher drivers).

Method: static reading of all sources plus two independent numerical checks
(fixed-z rewrite vs the direct per-tracer-β mixture; `alpha_agn_true`
interp-vs-exact-count worst case). Environment: `python` in the active conda env.

## Summary table

| ID | SEV | file:line | one-liner |
|---|---|---|---|
| F2-1 | SEV-2 | run_inference.py:1169,1197,1322-1334; core.py:217,233 | `(1+z)^gamma` weighting is inconsistent: per-pixel KDE uses weighted sum but the between-pixel W_pix and β_X use raw COUNTS → wrong field whenever gamma≠0 (F-P2 confirmed) |
| F2-2 | SEV-2 | run_inference.py (no `jax_enable_x64`); generate_gwsamples.py:68; run_inference.py:1003 | mixed precision by import side-effect: catalog+cosmology f32, samples+likelihood f64; results depend on import order (F-P1 confirmed) |
| F2-3 | SEV-2 | run_coverage.py:75-78; run_recovery.py:76-80 | drivers never run `pixelize_catalogs.py` and never check the pixelated catalog matches the current mock → silent stale-catalog inference on any config/seed change |
| F2-4 | SEV-2 | run_inference.py:1020 | event shuffle uses unseeded global RNG; with `N_gw_inf < total` it selects a NON-reproducible random event subset |
| F3-1 | SEV-3 | make_configs.py:81 | `inference_config` default `Dz=1e-4` is the known-bad delta-spike value (S0-a); only the drivers override it |
| F3-2 | SEV-3 | run_inference.py:151-152 | H0/alpha grid bounds hardcoded in `main()`, config-ignored; edge-truncated CIs are silent |
| F3-3 | SEV-3 | run_inference.py:246-247 vs 1443 | numerator truncates the KDE-smeared field at sample z; β_X uses the SHARP catalog-source-count CDF — inconsistent by ~(source density × Dz) at the cut |
| F3-4 | SEV-3 | run_inference.py:1169,1196,1239-1245; core.py:212,239-242 | `-1e10` magic sentinels instead of `-inf`; benign under logaddexp but fragile |
| F3-5 | SEV-3 | generate_gwsamples.py:526,531 | `abs(d_obs)` guard + window fallback distort the posterior for negative obs / fac≥0.119; silently active if `dL_uncertainty_fac` is raised |
| F3-6 | SEV-3 | generate_gwsamples.py:550 | PE subsample drawn WITH replacement → possible duplicate samples (negligible at n_initial=256000) |

No SEV-1 defects found. The primary SEV-1 target is verified correct (N1 below).

## Detailed findings

### F2-1 (SEV-2) — `(1+z)^gamma` tracer weighting is internally inconsistent (F-P2 confirmed)
`logpcatalog_agns/gals` compute the per-pixel normalized redshift density using the
gamma-weighted sum `wts_sum = Σ w_j (1+z_j)^gamma`, but `return log_prob, ngals`
returns the integer COUNT (`jnp.sum(valid_mask)`), not `wts_sum` (the alternative is
present but commented: `#return log_prob, wts_sum`). In `logPriorUniverse`,
`W_agn_pix = nagns` (count) and `W_agn_total = N_agn_total` (count). The field is
`p_X(z,pix) = (W_pix/W_total)·p_cat(z|pix)`. For **gamma=0** the `W_pix` cancels
cleanly (`(count_pix/count_total)·Σ(1/count_pix)N = Σ N / count_total`) and the field
is exactly the full-catalog-normalized KDE — correct. For **gamma≠0** the within-pixel
weighting is `(1+z)^gamma` but the between-pixel weighting is raw count; a properly
`(1+z)^gamma`-weighted field would use `wts_sum_pix/wts_sum_total` between pixels.
`β_X = CDF_X(z_max_gw)` is likewise an UNweighted count CDF. So the entire gamma≠0
path (field mixing weights and β both) is wrong. The MCMC bounds explicitly allow
`gamma_agn, gamma_gal ∈ [-5,5]`, so this is reachable by a documented parameter.
Current gates all run gamma=0, so recorded numbers are unaffected.
Suggested fix: return `wts_sum` from the kernels and build `W_total`/`β_X` from the
same gamma-weighted sums (or forbid gamma≠0 until done). Same defect mirrored in
`core.py` `_build_tracer_kernel` (returns `nobj`) and `W_totals` (sum of `n`).

### F2-2 (SEV-2) — mixed precision by import side-effect (F-P1 confirmed)
`run_inference.py` never calls `jax.config.update("jax_enable_x64", True)` (grep: 0
hits). `load_catalog_data`, `setup_cosmology` and `precompute_beta_cdf` therefore run
in default float32. `load_gw_samples` (line 1003) lazily `import generate_gwsamples`,
whose module-level line 68 flips x64 True; from then on GW-sample arrays and all
likelihood arithmetic are float64. The pixelized catalogs, cosmology interpolation
tables (`rs`), and the β constants are quantized to f32 (~1e-7 relative). The numerical
effect on H0/α is ~1e-6 — below gate resolution — but results are import-order
dependent and the G4 parity contract deliberately reproduces this accident
(`core.enable_x64` docstring). Suggested fix: set x64 once at the top of
`run_inference.py` and rebuild the β/cosmology tables in f64; drop the parity-to-an-
accident requirement.

### F2-3 (SEV-2) — drivers skip pixelization and never verify catalog consistency
`make_mocks.py` does mock+inject only (no pixelation). `run_coverage.one_realization`
and `run_recovery.one_set` run `make_mocks.py → generate_gwsamples.py →
run_inference.py`, never `pixelize_catalogs.py`. `run_inference` loads whatever
`cat_{gal,agn}_pixelated_nside{N}.h5` is on disk with no check that it was built from
the current `mock_catalog.h5`. The gates passed only because the pixelated files on
disk happen to match seed=101. If a user changes the mock seed/nbar/bias but a stale
pixelated file exists, `run_inference` silently infers against the WRONG catalog →
wrong H0/α with no error. Suggested fix: add a pixelize step to the drivers and stamp
a mock hash/seed into the pixelated file, asserting it at load.

### F2-4 (SEV-2) — unseeded event shuffle; non-reproducible subset when `N_gw_inf<total`
`run_inference.load_gw_samples:1020` does `np.random.permutation(N_gw_total)` with no
seed set in `run_inference`. When `N_gw_inf is None` (all gate runs) this only reorders
events and the summed logL is invariant — harmless. But `N_gw_inf` is a documented
config knob; with `N_gw_inf < N_gw_total` the retained event SUBSET is a fresh random
draw every invocation → non-reproducible posteriors. (The multitracer CLI correctly
seeds via `seed_shuffle`; legacy does not.) Suggested fix: seed the shuffle from config
in `run_inference` too.

### F3-1 (SEV-3) — factory default `Dz=1e-4` is the known-bad value
`make_configs.inference_config` hardcodes `catalog: {Dz_gal: 0.0001, Dz_agn: 0.0001}`.
That is the delta-spike KDE resolution S0-a proved biases α low (α̂=0.33 vs 0.505). The
coverage/recovery drivers override to 3e-3 after the call, but `case_s0_smoke` and any
direct use of `inference_config` inherit the bad default. Suggested fix: default to
`3e-3` (the owner-decision value).

### F3-2 (SEV-3) — grid bounds hardcoded, config-ignored
`main()` sets `H0_bounds=(50,100)`, `alpha_agn_bounds=(0,1)` as literals; the grid and
`summarize_grid` flat-prior integration span exactly these. A posterior that reached an
edge would be silently truncated (CI clipped to the grid). Interior for all gate runs.
Suggested fix: read bounds from config; assert MAP interiority in `summarize_grid`.

### F3-3 (SEV-3) — β_X truncation is sharp-count, numerator truncation is KDE-smeared
The numerator zeroes PE samples with `z(dL_s|H0) > z_max_gw` (line 1443) — a truncation
of the smeared KDE field in the integration variable (correct). But
`beta_{gal,agn}_fixed = np.interp(z_max_gw, z_sorted, cdf)` (lines 246-247) is the
fraction of catalog SOURCES with `z_j ≤ z_max_gw`, i.e. `Σ_j w_j 1{z_j≤z_max}`, whereas
the self-consistent normalization is `∫_{z≤z_max} p_X(z)dz = Σ_j w_j Φ((z_max−z_j)/dz_j)`
(KDE mass below the cut). The two differ by the KDE leakage of sources within ~Dz of the
cut. β cancels for single-tracer H0 (G2 unaffected); it only shifts the relative AGN/GAL
normalization that sets α, i.e. contributes to G3, and lies within the already-disclosed
`|Δα|≲0.03` systematic. Suggested fix: build β_X from the normal-CDF of the catalog KDE.

### F3-4 (SEV-3) — `-1e10` sentinels
Empty-pixel / zero-alpha branches return `-1e10` instead of `-inf`
(run_inference.py:1169,1196,1239-1245; core.py mirror). Under `logaddexp`/`logsumexp`
these are dominated away, but if both mixture terms hit `-1e10` the sample log-weight is
`≈-1e10` (finite, not `-inf`) — a fragile magic number that could surface if scales
change. Suggested fix: use `-jnp.inf`.

### F3-5 (SEV-3) — `abs(d_obs)` and window fallback in the dL sampler
`d_obs = abs(obs[2])` folds negative observed distances to positive, and
`hi = 20*d_obs` replaces the `d_obs/(1-8fac)` upper bound when `8fac≥0.95`
(fac≥0.11875). Both are documented as immaterial at the fiducial `fac=0.1` (GATES E1-b)
but activate silently if `dL_uncertainty_fac` is increased, reintroducing the
distance-scale bias the sampler was built to remove. Suggested fix: warn/guard when
`fac ≥ 0.1` that the window/abs approximations degrade.

### F3-6 (SEV-3) — PE subsample with replacement
`choose = np.random.randint(0, len(samples), N_samples_gw)` draws with replacement →
duplicate PE samples possible. Negligible variance at n_initial=256000/N=2000.

## Verification of the priority targets (numerical)

**Priority 1 — SEV-1 selection normalization (CORRECT).** I re-derived and numerically
checked the `fixed_z` rewrite. `logPriorUniverse` returns the field mixture
`alpha_used·p_AGN + (1−alpha_used)·p_GAL`; the per-event integral is
`N = mean_s[(dz/ddL)·1{z_s≤z_max}·(that mixture)]`; and `ll += N_gw·log(mix_norm)` with
`a_agn=alpha/β_A, a_gal=(1−alpha)/β_G, mix_norm=a_agn+a_gal, alpha_used=a_agn/mix_norm`.
Then `p(d_i) = mix_norm·N = (alpha/β_A)·N_AGN + ((1−alpha)/β_G)·N_GAL` — exactly the
GOAL §3.2 mixture with per-tracer β applied once. Synthetic check: `p_code` vs the direct
`alpha·N_AGN/β_A + (1−alpha)·N_GAL/β_G` gave **relative difference 0.0**; the identities
`mix_norm·alpha_used == alpha/β_A` and `mix_norm·(1−alpha_used) == (1−alpha)/β_G` hold;
`alpha=0`/`alpha=1` collapse exactly to the pure single-tracer β-normalized estimators.
The legacy `dl_horizon` β(H0) block is fully inert under `fixed_z` (guarded by
`selection_mode=='dl_horizon'` AND `dL_max is not None`, and `main` sets `dL_max=None`
in the fixed_z branch). β is applied exactly once.

**Priority 2 — exact flat-prior dL sampler (CORRECT at fiducial).** `logp =
-0.5((d_obs−dL)/(fac·dL))² − log(dL)` is the exact flat-prior posterior of the
multiplicative-noise likelihood `N(d_obs; dL, fac·dL)` (the `−log(dL)` term is the
dL-dependent Gaussian normalization; the dropped `−log(fac)` is constant). Data are
generated as `d_obs ~ N(dL_true, fac·dL_true)` = the same likelihood at `dL=dL_true`, so
generator and posterior are consistent. Window `[d_obs/(1+8fac), d_obs/(1−8fac)]`
covers ±8σ of the multiplicative noise. The inverse-CDF dL draws are independent of the
dec Gaussian, so the dec-filter + `randint` subsample preserve the dL marginal. Seeding
is once (`seed_samples` before mass gen) so masses+clouds are reproducible together.
Downstream the Jacobian `−log(ddL/dz)` = `+log(dz/ddL)` with `p_pe=1` is the correct
importance weight for uniform-in-dL PE samples. Correct. Residual `abs`/window issues at
fac≥0.119 are F3-5.

**Priority 3 — `alpha_agn_true` from eligible pools (negligible discrepancy).** The
injector uses exact eligible integer counts; `run_inference.main` uses
`N_total·np.interp(z_max_gw, z_sorted, cdf)`, which can differ from the exact eligible
count by <1 fractional object. At the glass_prod counts (gal 503774 / agn 5032) the
worst-case `|Δα_true|` over f_agn∈{0,0.3,0.7,1.0} is **≤ 2.0e-6** (measured), vs α CIs of
width ~0.03. Moreover `run_recovery` scores G3 against the injection file's realized
fraction `n_agn/(n_agn+n_gal)`, not this interp value, so the discrepancy never enters a
gate verdict. Noted-not-a-bug.

**Priority 4 — `src/darksirens_multitracer` (CORRECT).** (a) K-general path
`logsumexp_k[log w_k + log(n_k)−log(W_k)+logp_k]` reduces for K=2 to
`logaddexp(term_gal, term_agn)` = the pair path (test asserts allclose). (b) betas/
weights applied once: `_effective_weights` folds β into `w_eff` and `mix_norm=s`; the
prior uses `w_eff`, and `ll += N_gw·log(mix_norm)` adds `s` once; β appears nowhere else.
The pair path passes `alpha=w_eff[1]` = legacy `alpha_used` exactly, and
`weights=[1−alpha, alpha]`, `betas=[β_gal, β_agn]` map onto legacy `a_gal/a_agn` exactly.
(c) numerator truncation `jnp.where(z<=z_max_gw, …, −inf)` applied once. (d) empty-block
handling in `load_gw_samples._blocks` matches legacy (ndim check ↔ `N_gw_*>0`); the only
un-handled case is BOTH tracers empty (no events at all), degenerate. (e) The G4 parity
test uses `np.array_equal` (true exact equality; `−inf==−inf` holds, no NaN produced);
the s0_uniform fixtures (`cat_{gal,agn}_pixelated_nside32.h5`, the seedgw1007 samples)
are present on disk, so `have_data=True` and the test does NOT skip. Both sides seed
`np.random.seed(SEED)` before the single `permutation`, so event order aligns and parity
is real. The one weakness (N5) is that the test does not assert grid non-degeneracy, so
it would pass vacuously if BOTH implementations produced an all-`−inf` grid; but the
H0 window [62,74] is interior and the grid has real structure.

**Known issues:** F-P1 confirmed (F2-2). F-P2 confirmed (F2-1).

## Noted-not-a-bug list

- **N1** SEV-1 fixed-z mixture algebra — correct; β once; dl_horizon inert; verified
  rel-diff 0 (Priority 1 above).
- **N2** obs-centered exact-posterior dL sampler — correct at fiducial; exact flat-prior
  posterior of the multiplicative-noise likelihood; window/Jacobian/p_pe self-consistent
  (Priority 2).
- **N3** `alpha_agn_true` interp-vs-exact-count — worst |Δα|≤2e-6, and not used for G3
  scoring (Priority 3).
- **N4** multitracer general-vs-pair equivalence, single β application, single truncation
  — correct; parity test non-skipping (Priority 4).
- **N5** `np.array_equal` parity assertion — legitimate exact test; only fails to assert
  non-degeneracy (would need an all-`−inf` bug on BOTH sides to pass vacuously).
- **N6** G2 coverage over a single fixed catalog (seed=101, only seed_gw/seed_samples
  vary) — defensible: the galaxy survey is conditioned-on data, so conditional coverage
  is the relevant notion. Caveat: the `⟨H0med⟩` offset SE is computed from the 25
  injection realizations and omits catalog-to-catalog variance, so the reported "1.7σ"
  offset significance is conditional on this catalog, not marginal.
- **N7** hard-truncation `−inf` grid rows at high H0 — real information from the highest-z
  event (an event whose entire PE cloud implies z>z_max_gw has zero likelihood), not a
  defect; `summarize_grid` floors these via `log(max(trapz,1e-300))`.
- **N8** `summarize_grid` marginals/CIs — trapezoid flat-prior marginalization over the
  correct axes (alpha over axis=1, H0 over axis=0), equal-tailed CIs by inverse-CDF;
  correct. The alpha=1 equal-tailed-CI boundary miss is already disclosed in G3 and
  handled with a one-sided interval.
- **N9** `fisher_forecast` Hessian — `cov = inv(−[[2d,f],[f,2e]])` is the correct Laplace
  covariance of the quadratic logL fit; √(n_grid/n) iid scaling standard.

## Do any findings invalidate the GATES.md verdicts?

**No.** Every recorded verdict (E1/E1-b, S0/S0-a, G0, G1, G2, G3a, G3, G4, G5) stands.
The one SEV-1 target that would have invalidated the science — selection normalization
applied exactly once with per-tracer β — is verified **correct** (rel-diff 0), so
G2 (the √2-class selection gate), G3 (α recovery) and E1 (estimator verdict) rest on
sound math. All SEV-2 findings are LATENT with respect to the as-run configuration:
F2-1/F-P2 bites only for `gamma≠0` (all gates run gamma=0), F2-2/F-P1 changes numbers
only at the ~1e-6 level and is fixed under the recorded import order, F2-3 (missing
pixelize) did not corrupt results because the on-disk pixelated catalogs match seed=101,
and F2-4 (unseeded shuffle) is inert because every gate run uses `N_gw_inf=None`. The
SEV-3 items are hygiene/footguns. Two forward-looking caveats the verdicts should carry:
(i) **no number may be quoted for any `gamma≠0` run** until F2-1 is fixed, and (ii) a
clean reproduction must regenerate the pixelated catalogs (F2-3), since the drivers
assume they already exist and match.
