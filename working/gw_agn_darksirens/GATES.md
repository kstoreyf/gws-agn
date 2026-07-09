# Gates — gw_agn reproduction with real darksirens

Verdicts recorded by the orchestrator (Fable). Evidence = agent validation runs + files in
`data/` (`catalog_meta.json`, `gw_inputs_validation.json`, `injections_validation.json`).

## GA — data translation round-trip. **PASS** (2026-07-09)

- Survey files `gal.h5/agn.h5/(+_zlt1)` load via `darksirens.catalogs.io.load_survey`;
  ngals sums exactly 1,177,289 / 11,724 (and 503,774 / 5,032 truncated); no NaNs; padding
  exact (100/1/0); dzgals == 3e-3(1+z) exact; per-pixel row alignment spot-verified.
  log10n0_true: gal −5.5063, agn −7.5081 (z≤1.5 volume 3.7771e11 Mpc³).
- 24 gwcat-1.0 PE files load via `load_gw_samples` (mock path); dL/ra/dec bit-identical to
  gw_agn `gwsamples_*`; masses/chieff regenerated at darksirens fiducial powerlaw+peak/chieff
  (mass_seed = seed_gw + 900000); median z(dL_PE) tracks true_z unbiased.
- Injection files (Ndraw 2e6/2e6/4e5, seeds 42001/42002/42003) load via
  `load_selection_samples`; detected fraction matches analytic Vc(1.0)/Vc(1.2)=0.677638 to
  <0.05%; z_det ≤ 1.000000 verified by re-inversion; pdraw = gmd `_selection_pdraw`
  population branch, unchanged.
- **Flag (disclosed):** file-attr Neff (1/pdraw convention, i.e. flat-p_pop) is ~1–40 —
  dominated by fiducial-population tail draws. Not the operative statistic: at
  fix_population=True the likelihood weight p_pop/p_draw cancels the mass/spin factor, so
  the in-likelihood selection Neff is set by catalog-prior lumpiness. Operative evidence at
  GB (guard behavior at truth point) + injections A/B seed comparison in Phase C.

## GB — smoke evals + coarse scans. IN PROGRESS

**GB incident 1 (diagnosed, fixed at proposal level).** First smoke (11-pt f scan, K=2,
fagn0.3, injections.h5 v1) returned logL = −inf at every grid point. Instrumented diagnosis
(diag_selection.py, diag2_maxweight.py): the hard selection-Neff guard fired because ONE
injection carried importance weight e^30.7 (Neff → 1.0). Root cause is a latent
generator↔inference tail inconsistency INSIDE darksirens: the PL+G population is a
per-component mixture and the mixture code (gw/populations/parametric.py component_densities)
gives components without `m_min_spec` — i.e. the Gaussian peak — a pairing taper of
(M_LO, 0.01), effectively NO m2 taper; so p_pop(m1≈5.3, q≈0.95) is floored at
w_G·φ_G(m1)·q^β ≈ e^−18.9. The mock generator's `_mass_spin_pdf` (used for pdraw) instead
tapers the WHOLE mixture with S_low(m2; m_min=5, dm=3) ≈ e^−50 there. Ratio ≈ e^31 for any
injection with m2src just above m_min. Verified numerically at the rogue point
(ds log_p_pop −18.86 vs gmd −50.36; control point m1=35,q=0.8 agrees to 0.034 in log).
Only manifests at large Ndraw (their 42k-injection validation never drew a rogue; our 1.35M
drew several). `dark_sirens_complete` masked it by chance (catalog KDE = 0 at the rogue's
(z,pix)); the dark_sirens missing-floor exposes it.

Fix (estimator-variance only, μ stays unbiased; physics/detection unchanged): injections
rebuilt with a 90/10 defensive-mixture proposal — 90% population branch (unchanged), 10%
gmd "uniform" branch (m1det~U(2,200), q~U(0,1), chi~U(-1,1)); pdraw = 0.9·pdraw_pop +
0.1·pdraw_unif evaluated exactly for every row. The uniform component floors pdraw so no
p_pop tail can produce a rogue weight. Same seeds/Ndraw/filenames.

**Upstream note for darksirens maintainers (do not fix here):** generate_mock_data.py's
docstring claims its PopulationConfig matches get_fixed_population_params, but the pdraw
density tapers the Gaussian component's m2 while the inference model does not — any
large-Ndraw selection run with the population proposal can hit Neff collapse. Also the
mock's event m2 truths inherit the same (tiny) shape difference near m_min — negligible at
fixed population, disclosed.

**GB incident 2 (measured; a finding, not a bug in the translation).** With the fixed
injections, the K=2 f scan (fagn0.3, H0=truth, DARKSIRENS_ZMAX=1.5 matched to the mock
depth) is finite everywhere (selection Neff 135k at f=0, 25k at f=1) but rises
MONOTONICALLY to f=1 (range 184 logL units), whereas gw_agn's own conditional profile at
the same H0 node peaks sharply at alpha=0.325 (f=0 at −288.5, f=1 at ~−inf). Decomposition
via instrumented selection (diag_selection.py, zMax=1.5): selection term contributes only
+40 of the +184 tilt (ln mu_gal=1.216 vs ln mu_agn=1.192 + Neff penalty); the numerator
contributes +143. Mechanism: darksirens' K-mixture combines PER-PIXEL-CONDITIONAL z priors
under an isotropic sky (required by the K≥2 guard), and its completion gives EMPTY AGN
pixels the normalized volume prior (n0 cancels in dN_miss/N_miss). For a complete dense GAL
catalog the pixel-KDE averaged over a PE cloud ≈ that same volume prior, so GAL-hosted
events are nearly indifferent between the two priors while AGN-hosted events gain their
own-host spikes → fcat_2 pushed to 1 regardless of interior truth. gw_agn's field
convention (p_k(z,pix) ∝ n_pix,k/N_k · KDE) zeroes empty-sky AGN probability — that
number-density/sky-clustering contrast is exactly the alpha-identifying channel, and it is
structurally absent from the isotropic-sky conditional-prior mixture. DARKSIRENS_ZMAX=5
(default) additionally dilutes the catalog term by budgeting missing galaxies to z=5 for a
z≤1.5 mock (GAL pixel: N_miss≈133 vs N_obs≈24); all production scans use
DARKSIRENS_ZMAX=1.5. Falsification tests in flight: fagn0.0 f scan (predicted ~flat/weakly
rising — fcat_2 uninformative for GAL-hosted events), fagn1.0 (predicted strongly rising,
aligned with truth).

Criteria (pre-registered): finite logL at the truth point for every scan config
(dark_sirens K=2; dark_sirens K=1 gal/agn; dark_sirens_complete K=1 gal/agn); coarse 11-pt
H0 scan argmax at 67.74 ± one grid step (fagn0.3 set, K=2, f=0.307); coarse 11-pt f scan
argmax within one step of 0.307. Scan driver regression vs
`working/multitracer/runs/conditional_scans.json` to ~1e-3.

**Verdict: PASS with one criterion replaced by a measured finding (2026-07-09).**
- Scan-driver regression: PASS, bit-exact (0.0 diff) at all 9 overlapping grid points of the
  PR #195 recorded conditional scans; H0 node value bit-exact.
- Finiteness: PASS — all six smoke configs finite everywhere (n_neginf=0) after the
  defensive-mixture injection fix (incident 1).
- Coarse H0 argmax: PASS — all four K=1 configs (ds/dsc × gal/agn, 100-event r00 sets) peak
  at the node adjacent to 67.74 on the step-5 grid. dsc AGN is by far the sharpest
  (range 5192 logL units over [50,100]) — the empty-pixel-zero policy is the closest
  analogue of gw_agn's field convention for the sparse tracer.
- Coarse f argmax: CRITERION NOT MET, replaced by incident-2 finding, now CONFIRMED by its
  two falsification tests: fagn0.0 → argmax f=0 (range 86; −0.087/GAL-event toward f=1);
  fagn1.0 → argmax f=1 (range 757; +0.76/AGN-event). The ~8.7× per-event asymmetry means
  the conditional-scan argmax runs to f=1 whenever the true AGN-hosted fraction ≳ 0.1
  (hence fagn0.3 → 1). This is an estimand property of the darksirens K≥2 mixture
  (isotropic sky + per-pixel-conditional completion priors), not a data-translation bug —
  the K=1 H0 scans validate the translation independently.

## GC — reproduction vs gw_agn numbers. **PARTIAL — H0 PASS, f/(H0,f) FAIL-with-attribution (2026-07-09)**

Criteria (pre-registered): every production scan's MAP within one grid step of the gw_agn
MAP or the truth; alpha medians within ~2× Fisher sigma (0.019·2) of gw_agn medians at
N=1000; joint-grid rho consistent with ≈0 (|rho| < 0.1); A/B injection-seed shift of the
f-scan argmax < one grid step. Failure → z≤1-truncated-catalog A/B before concluding.

Verdicts (full numbers: RESULTS.md, results/comparison_summary.json):
- H0 per tracer: PASS. dsc×AGN ⟨peak⟩ 67.75±0.87 vs ref 67.67±0.33 (exact); ds×AGN −0.94
  (~1 grid step); GAL configs −1.1/−2.3 BUT truth only 1.2 logL below max (flat
  likelihoods — compatible, not significant). zlt1 A/B run per the pre-registered fallback:
  worsens GAL (→60.7–61.6) ⇒ offset attributed to the convention stack, NOT truncation.
- f scans: FAIL the alpha-median criterion — argmax f=1 for all truths ≥ 0.307 (fagn0.0
  correctly at 0). A/B injection-seed shift: zero (argmax pinned at 1; range 184→185),
  so the pre-registered A/B criterion itself passes — the failure is structural, not MC.
  Full mechanism + falsification tests: GB incident 2 and RESULTS.md.
- Joint: FAIL truth-coverage (fagn0.3 MAP (63.33, 1.000), truth outside 90% CI); rho
  criterion moot at a boundary-railed posterior (−0.119).
- Overall: the H0 half of the reproduction stands; the f half is a diagnosed estimand
  difference in the production code's K≥2 mixture — reported as the campaign's central
  finding rather than engineered away (per the program's honesty rules).

## Stack acceptance — field-mode reruns on the campaign data (2026-07-09, PR-3 branch c3fc3f8)

Field-convention sky weighting (darksirens PR #207) rerun on the fagn0.3 campaign inputs
(N=1000, H0=truth, DARKSIRENS_ZMAX=1.5):
- log10n0 at TRUE densities: argmax f=0.8 with f>=0.9 killed by the selection guard — the
  dark_sirens field model with n0=true honestly budgets ~11.8k "missing" AGN across the
  90% empty pixels (half of Z_agn); on a COMPLETE clustered catalog that is misspecified.
- Complete-catalog limit (log10n0=-12, N_miss->0, Z->N_obs = gw_agn's field): **interior
  argmax f=0.40 on BOTH independent injection seeds** (truth 0.307; d(0.30)=-23.8/-13.2);
  f->1 strongly disfavored / -inf (gw_agn signature). Conditional control on the same data
  railed at f=1.0 (+184). VERDICT: estimand fix CONFIRMED end-to-end.
- Residual +0.1 argmax offset and the f>0.45 guard behavior are selection-MC-limited:
  seed A's soft-guard penalty explodes above 0.45 while seed B is smooth (-0.3 at 0.45),
  i.e. mu_AGN under field weighting is dominated by the few occupied-pixel injections
  (isotropic injection set x narrow kernels x sparse tracer). gw_agn's analytic
  beta = CDF(z_max) had no such MC term. Clean quantitative recovery gate = the
  purpose-built clustered mock in darksirens PR-5 (+ PR-4 complete-model mixture, whose
  Z is theta-independent).
