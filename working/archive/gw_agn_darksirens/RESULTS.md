# RESULTS — reproducing `working/gw_agn` with the production darksirens code

2026-07-09. Orchestrator: Fable. Gates + incident forensics: GATES.md. Interface contract:
RECON.md. Comparison numbers: results/comparison_summary.json; figures: figs/ (verified
caption-vs-pixels). Code: the real darksirens package (branch `lensing-stack-to-master`,
PR #195 K-catalog mixture merged, d387b4f), driven by module import
(`scripts/scan_darksirens.py` — validated **bit-exact** against the PR #195 recorded
conditional scans). All analyses at fixed fiducial population (powerlaw+peak + chieff ==
mock truth), Om0 pinned, survey nuisances at truth, DARKSIRENS_ZMAX=1.5, conditional grid
scans only (no samplers), same events / distance+sky PE / catalogs as gw_agn
(loader-validated, dL/ra/dec bit-identical).

## One-paragraph summary

The per-tracer H0 scans REPRODUCE gw_agn — exactly where the mock carries sharp H0
information (dark_sirens_complete × AGN: ⟨peak⟩ = 67.75 ± 0.87 vs gw_agn 67.67 ± 0.33,
truth 67.74), and compatibly elsewhere (GAL likelihoods are nearly flat, range ~50 logL
over H0∈[50,100]; their peak offsets of −1 to −2.3 km/s/Mpc put the truth only 1.2–1.4
logL below the maximum — tilts, not detections). The f and (H0,f) scans do NOT reproduce:
darksirens' K=2 mixture weight `fcat_2` is a **different estimand** from gw_agn's
alpha_AGN — its conditional scan runs to f=1 for any true AGN-hosted fraction ≳ 0.1
(fagn0.3 joint MAP (63.3, 1.00) vs truth (67.74, 0.307), dragging H0 4.4 km/s/Mpc low),
because the isotropic-sky per-pixel-conditional completion priors structurally lack the
number-density/sky-clustering contrast that identifies alpha_AGN in the field convention.
The mechanism is fully diagnosed, falsification-tested, and robust to every A/B lever
(injection seed, z≤1 catalog truncation, zMax). Two latent darksirens issues were found
and worked around along the way (pairing-taper pdraw inconsistency; DARKSIRENS_ZMAX
completion-budget sensitivity). A concrete upstream fix direction for the mixture is
identified (per-catalog pixel-count sky weighting).

## H0 scans (the part that reproduces)

61-pt scans, H0∈[50,100], coverage realizations r00–r09 (100 events each), both universe
models, quadratic-refined argmax; reference = gw_agn per-realization grid medians
(results/gw_agn_coverage_reference_r00_09.json):

| model | tracer | ⟨peak⟩ ± sem | gw_agn ref | Δ | ⟨hw68⟩ | ref hw | ΔlogL(truth) below max |
|---|---|---|---|---|---|---|---|
| dark_sirens_complete | AGN | **67.75 ± 0.87** | 67.67 ± 0.33 | **+0.08** | 0.61 | 0.74 | 18.0 ± 21.0 |
| dark_sirens | AGN | 66.81 ± 0.27 | 67.67 ± 0.33 | −0.94 | 1.01 | 0.74 | 1.4 ± 1.3 |
| dark_sirens_complete | GAL | 66.01 ± 0.84 | 67.32 ± 0.41 | −1.14 | 2.07 | 1.36 | 1.2 ± 0.9 |
| dark_sirens | GAL | 65.45 ± 1.01 | 67.32 ± 0.41 | −2.29 | 2.62 | 1.36 | 1.2 ± 1.2 |

- The multitracer payoff direction reproduces: AGN sharper than GAL in both codes (ours
  0.61 vs 2.07 within dsc; gw_agn 0.74 vs 1.36).
- GAL flatness: darksirens' isotropic-sky conditional prior carries less GAL H0
  information than gw_agn's count-weighted field (hw 2.6 vs 1.36); the −1 to −2.3 peak
  offsets are within the flatness (truth ~1.2 logL below max) — compatible, not biased at
  measurable significance per realization.
- dsc×AGN is overconfident per realization (scatter 2.56 vs hw 0.61) — an amplified
  analogue of gw_agn's own AGN undercoverage (their 90% CI covered 80%).
- **A/B attribution (zlt1):** truncating both catalogs at z=1.0 does NOT restore the GAL
  peaks — it worsens them (ds GAL 65.45→61.6; dsc GAL 66.01→60.7; AGN unchanged 67.0) ⇒
  the offset is NOT the z≤1-truncation convention; it is the convention stack (conditional
  prior information dilution + (1+z)^(γ−1) rate convention + injection-based μ(H0) vs
  gw_agn's H0-independent β). Left as a quantified, disclosed difference.
- K=2 mixture H0 at f=0.307 (fagn0.3 events, N=1000): refined peak 66.83, hw 0.62 vs
  gw_agn 67.53 [66.94, 68.10] — ~0.9 low, driven by the same stack plus the mixture's
  AGN volume-floor component.

## f and (H0,f) scans (the part that does not reproduce — and why)

41-pt f scans at H0=67.74, K=2 (`dark_sirens`, the only model the K≥2 guard allows):

| set | alpha_true | this-work argmax | range (logL) | gw_agn argmax (same events, H0 node 67.5) |
|---|---|---|---|---|
| fagn0.0 | 0.0099 | **0.000** | 86 | 0.000 |
| fagn0.3 | 0.307 | **1.000** | 184 | 0.325 |
| fagn0.7 | 0.703 | **1.000** | 515 | 0.675 |
| fagn1.0 | 1.0 | **1.000** | 757 | 0.975 |

Joint 61×41 grids: fagn0.3 MAP (63.33, 1.000) — truth (67.74, 0.307) outside the 90% CI,
H0 dragged −4.4; fagn0.7 MAP (66.67, 1.000). gw_agn on the same events: (67.5, 0.325) and
(67.5, 0.675), truth-covering.

Mechanism (diagnosed via instrumented selection + numerator decomposition, confirmed by
falsification tests): the mixture combines PER-PIXEL-CONDITIONAL completion priors under
an isotropic sky (the K≥2 guard requires it), and the completion assigns EMPTY pixels the
normalized volume prior (n0 cancels in dN_miss/N_miss). On this mock (GAL: 0% empty
pixels; AGN: 78.8% empty), a GAL-hosted event's PE-averaged GAL prior is itself close to
the volume prior ⇒ GAL-hosted events are nearly indifferent between components (−0.086
logL/event from f=0→1), while AGN-hosted events gain their own-host KDE spikes
(+0.757/event). The 8.8× asymmetry sends the argmax to f=1 whenever the true AGN fraction
≳ 0.1. Selection contributes only +40 of the fagn0.3 tilt of +184 (ln μ_gal 1.216 vs
ln μ_agn 1.192). gw_agn's field convention — p_k(z,pix) ∝ (n_pix,k/N_k)·KDE — gives
empty-sky AGN probability ≈ 0, which is exactly the number-density contrast channel that
identifies alpha_AGN (and diverges at f→1 for GAL-hosted events; their grid assigns f=1
zero probability). Robustness: independent injection seed (range 184→185), z≤1-truncated
catalogs (→209), zMax 5→1.5 (285→184) — argmax f=1 in all.

Consistency with PR #195's own validation: its uniform-catalog mock (log10n0 ≈ −3.5,
completion grid to z=5) had priors ≈ 99.99% volume floor — nearly tracer-independent — so
its end-to-end recovery was truth-compatible but nearly prior-wide, masking this estimand
property. Conditional-likelihood scans at that contrast could not have revealed it.

**Upstream fix direction:** allow per-catalog pixel-count sky weighting (field convention,
P(pix|k) ∝ n_pix,k) in the K≥2 mixture — the gw_agn OPTION-4-style weighting (valid for
unit weights, γ=0). Without a sky-density channel, `fcat_2` measures per-pixel z-shape
preference, not host-fraction.

## Latent darksirens issues found (worked around here; report upstream)

1. **PL+G pairing-taper pdraw inconsistency** (GATES.md incident 1):
   `generate_mock_data._mass_spin_pdf` tapers the whole mass mixture's m2; the inference
   PL+G gives components without `m_min_spec` (the Gaussian peak) pairing taper
   (M_LO, 0.01) — effectively none. p_pop/p_draw reaches e^31 at m2src ≳ m_min ⇒ one rogue
   injection ⇒ selection Neff → 1 ⇒ logL = −inf via the hard guard. Manifests only at
   Ndraw ≳ 10^6. Workaround: 90/10 defensive-mixture injection proposal (exact pdraw);
   max weight down 15 orders, Neff 135k (f=0) / 25k (f=1).
2. **DARKSIRENS_ZMAX (default 5.0) silently sets the missing-galaxy budget**: for a z≤1.5
   catalog it dilutes the per-pixel catalog term to N_obs/(N_obs+N_miss) ≈ 0.15 at the
   count-anchored log10n0. Set it to the survey depth (1.5 here). Amplifies but does not
   cause the f-scan tilt.
3. Cosmetic: dark_sirens_complete × sparse AGN coverage scans show −inf cells at fixed
   nodes (H0 ∈ {50.8, 51.7, 52.5, 63.3}, realization-independent, ≥3000 logL below peak)
   — a selection-side zero/guard artifact of the empty-pixel-zero policy; peaks unaffected.

## What these claims cost (conventions/assumptions)

Conditional scans at true nuisance values (log10n0 = true densities, delta=0, b_miss=1,
sigma_kde=0), not marginalized posteriors; flat-prior trapezoid grid summaries matching
gw_agn's pipeline convention; comparisons at CI level, never bitwise. Selection: gw_agn's
H0-independent β = CDF(z_max_gw) vs darksirens' injection-based μ(H0) — coincide at truth.
Masses/chieff regenerated at darksirens fiducials (events' m2 truths inherit the
generator's whole-mixture taper — negligible at fixed population); the fixed-population
mass term adds mild spectral-siren H0 information absent from gw_agn. One catalog seed;
r00–r09 subset (10 of 25 realizations) for coverage comparisons.

## Verdict against the goal

- **H0 scans (both universe models): REPRODUCED** — exactly where information is sharp,
  compatibly (within likelihood flatness) elsewhere.
- **f scans / (H0,f) scans: NOT reproduced — measured, diagnosed, and attributed**: the
  production K=2 mixture estimand structurally lacks the sky-density channel; this is the
  central actionable finding for the multitracer program (PR #195 works as designed; the
  design measures a different quantity than alpha_AGN on clustered sparse tracers).
