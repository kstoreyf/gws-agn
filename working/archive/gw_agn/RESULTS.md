# RESULTS — multitracer dark sirens: joint (alpha_AGN, H0) proof of concept

2026-07-08. Orchestrator: Fable. Evidence trail: GATES.md + gates_report.json (verdicts),
BIAS_DIAGNOSIS.md (why the pipeline was biased and what changed), figs/ (verified
caption-vs-pixels). Repo commits `7f21ab9`…`804332a` on master (not pushed).
Adversarial module review: reviews/module_review_2026-07-08.md.

## The goal, and what was delivered

Build and validate a multitracer dark-siren pipeline that infers the AGN-hosted GW fraction
jointly with H0 from two tracers (galaxies + AGN) of one density field, then productionize it
tracer-pair-agnostically. Delivered: the pipeline **works and is calibrated** once four
stacked estimator defects are fixed (see below); alpha_AGN is **decisively recoverable** at
the fiducial contrast — this is a non-null result, not an information-limit report; and the
validated likelihood now exists as a reusable package (`src/darksirens_multitracer`) that
reproduces the proof-of-concept bit-for-bit (G4) and runs on a second tracer pair
config-only (G5).

## Headline numbers (all gate-verified, all on the GLASS mock described below)

- **Calibration (G2, the selection gate).** 25+25 single-tracer realizations (N=100 events,
  10% distance errors): GAL 68/90% coverage = 0.84/0.92, ⟨H0⟩ = 67.31 ± 0.25; AGN coverage
  = 0.68/0.80, ⟨H0⟩ = 67.68 ± 0.20 against truth 67.74. No detectable bias.
- **alpha_AGN recovery (G3).** At N=1000: truths {0.0099, 0.307, 0.703, 1.0} recovered as
  {0.017, 0.330, 0.687, 0.975} with 68% CI half-widths ≈ 0.02–0.04. Truth in the 68%
  interval for all four (the boundary case via its one-sided interval — see systematics).
- **Information content (G3a).** σ(alpha_AGN) = 0.019 at N=1000 ⇒ **0.086 at N=50**:
  the N=50 proof-of-concept target is NOT prior-dominated at this tracer contrast.
- **No fagn–H0 degeneracy:** ρ(H0, alpha_AGN) = −0.002 at the fiducial point. The joint
  posterior is essentially a product of its marginals here.
- **Multitracer payoff:** at equal N, AGN-hosted events constrain H0 **2.0× tighter** than
  galaxy-hosted (mean 68% half-width 0.72 vs 1.43 km/s/Mpc) — the sparse, high-bias tracer
  sharpens per-event redshift attribution. This is the concrete argument for the
  multitracer generalization.

## Why the pipeline was biased before (the result that unblocks everything else)

Four independent defects, individually measured (BIAS_DIAGNOSIS.md): (1) an H0-dependent
β(H0) division mismatched to the mock's fixed-true-z host cut — the exact normalization is
the H0-independent constant CDF_X(z_max_gw) per tracer; (2) truth-centered PE clouds;
(3) a Dz=1e-4 delta-spike catalog KDE that Jensen-biases the sparse tracer (α̂ 0.33 vs
0.505) and roughens the H0 landscape; (4) Gaussian obs-centered clouds used as if they were
the flat-prior posterior of multiplicative distance noise — an O(fac²) distance-scale bias
(+0.7–1.0 km/s/Mpc at 10% errors, measured at pipeline level by a deliberate A/B arm).
The fixes: `selection_mode: fixed_z`, `pe_centering: obs` with exact inverse-CDF posterior
sampling, Dz=3e-3 with 2000 PE samples. Every pre-fix inference result is superseded.

## What the claims cost (assumptions)

Fixed population (masses generated but unused; no SNR selection — detection is a hard
true-z cut, matched exactly in the estimator); complete catalogs from GLASS lognormal
fields with linear bias (gal 1.2, AGN 2.0; nbar 1e-2 / 1e-4 arcmin⁻²; one catalog seed —
coverage is over GW injection/noise realizations, not catalog realizations); PE = Gaussian
sky + exact-posterior distance clouds, no waveform realism; Om0, gammas fixed at truth;
alpha_AGN scored against the eligible-pool truth (never f_agn). The G2/G3 calibration
statements hold for THIS generative model; realism upgrades (SNR selection with a matched
H0-dependent β, incomplete catalogs) are the next scientific step, not a gap in the
present validation.

## Known systematics (quantified, disclosed)

- **Sky-pixelization attenuation of the sparse tracer:** nside-64 pixels (0.92°) box-smooth
  the AGN field on the PE sky scale (0.57°), pulling α̂ toward the interior by ≲0.03
  (cleanly visible only at the α=1 boundary: MAP 0.975, one-sided 68% [0.966, 1]; shown
  NOT to be MC noise by a 4× sample-count A/B). Shrinks with nside; an nside-128 A/B is the
  natural follow-up if percent-level alpha accuracy matters.
- **Estimator resolution knobs:** results hold for Dz=3e-3, N_samples=2000 (chosen by the
  resolution criterion n·Dz/σ_z ≳ 100 with 0.2% smoothing systematic; α̂ plateau verified
  over Dz ∈ [1e-3, 1e-2]). Do not run at the legacy Dz=1e-4.
- **Legacy engineering fragility (found during bit-parity work):** the pipeline is
  mixed-precision by import side effect, and the OPTION-4 mixture weights pixels by integer
  counts (valid only for unit weights, γ=0). Both are inert for the present results and
  made explicit in the package.

## Open questions

- Realistic selection: move the mock to SNR-based detection and use the matched β(H0)
  (correction and injection process must match — the central methodological lesson here).
- Does the multitracer combination cancel sample variance on H0 across catalog seeds
  (McDonald–Seljak sense)? Requires multiple catalog realizations; not tested here.
- Bias/contrast scaling of σ(alpha): the 2.0× H0 payoff and σ(alpha)=0.086@N=50 are for
  one (nbar, bias) contrast point; a small contrast grid would map the information budget.
- Percent-level alpha work needs the nside/pixel-attenuation systematic retired (finer
  pixels or a continuous-sky treatment).
