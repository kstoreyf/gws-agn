# Why the H0 inference was biased — diagnosis and fixes (2026-07-08)

Audience: Kate + Ignacio. Analysis by Claude (Fable) during the 2026-07-08 session; fixes are
in commit `7f21ab9` on master (not pushed). Everything here is reproducible from
`working/gw_agn/` (configs, logs, grids) and the toy script referenced below.

## TL;DR

Three independent defects stacked on top of each other, partially canceling at some settings
(which is why the bias moved around as dL_uncertainty changed and why some reverts "sort of
worked"). None of them is the mixture model itself — OPTION 4 in `logPriorUniverse` is fine.

1. **The β(H0) division is the wrong selection correction for these mocks** (dominant, H0 bias).
2. **PE clouds were truth-centered** — the commented-out "NEW METHOD ATTEMPT" block was the
   correct one (data-consistency bias, grows with dL_uncertainty).
3. **The per-pixel catalog KDE at Dz=1e-4 is a delta-spike mixture** the PE-sample Monte Carlo
   cannot resolve (variance → Jensen bias, hits the sparser tracer hardest → alpha biased low,
   jagged H0 landscapes).

## 1. The β(H0) mismatch

`inject_gw_sources` selects hosts by a hard cut on the **true** redshift: z ≤ z_max_gw. For
that detection process, P_det(z) = 1{z ≤ z_max_gw}, so the selection normalization per tracer is

    β_X = ∫ P_det(z) p_X(z) dz = CDF_X(z_max_gw)   — a constant. No H0 dependence.

The pipeline instead divided by β(H0) = CDF(z_max_det(H0)), z_max_det = z(dL_max | H0). The
in-code rationale ("catalog density rises with z, so high H0 lands in denser regions and wins")
is a plausibility argument, not likelihood math: near the truth the raw (untruncated) estimator
already has zero expected score in the dLunc→0 limit, so subtracting N·log β(H0) *injects* a
score −N·∂logβ/∂H0 ≠ 0 at the truth and pushes the MAP off it. The correction that "fixed" the
raw likelihood's high bias at large dL uncertainty was actually compensating a *different*
missing term (the numerator truncation, below) with the wrong shape — two opposite-sign errors
that cancel only at one tuning.

What the exact estimator for this mock looks like (all three pieces matter):

- numerator per event: mean over PE samples of p_X(z_s)·**1{z_s ≤ z_max_gw}**/(ddL/dz)|_s,
  with z_s = z(dL_s|H0) — the indicator is against the FIXED z_max_gw, evaluated at each trial H0;
- per-tracer constant renormalization β_X = CDF_X(z_max_gw) inside the mixture:
  α·N_AGN/β_AGN + (1−α)·N_GAL/β_GAL  (constants cancel for single-tracer H0 runs, but set the
  relative AGN/GAL normalization for α);
- **no** H0-dependent global division.

Toy evidence (catalog-free, `gwsagn_bias_toy.py`, N_gw=200, dLunc=0.1·dL, 60 realizations,
H0_true=67.74): pipeline configuration (truth-centered, no indicator, β(H0)) biased −1.55±2.04;
adding the indicator while keeping β(H0): +4.84; obs-centered with indicator and β(H0): +9.97 —
β(H0) is never right, it just changes which way you miss.

Implemented: `selection_mode: fixed_z` (new default) in `run_inference.py`; the old behavior
remains as `dl_horizon` for A/B; `none` disables everything.

If you want an H0-dependent β to be *correct*, the mock has to actually select on the
observable (e.g. keep events with dL_obs < dL_max, or an SNR threshold) — then β(H0) built from
that same P_det is the right correction. Selection correction and injection process must match;
"standard" is not a property of the formula but of the pair (mock, correction).

## 2. Truth-centered PE clouds

`generate_event_samples` drew samples ~ N(truth, σ). Real data are d_obs ~ N(truth, σ) with PE
posterior ~ N(d_obs, σ(d_obs)) under a flat prior. Truth-centered clouds make every event's
posterior peak exactly at the truth — collectively over-informative and inconsistent with the
flat-p_pe importance weights. Your own commented-out block ("NEW METHOD ATTEMPT") was right.
Implemented: `pe_centering: obs` (default), `truth` kept as labeled legacy;
`gw_samples.seed_samples` now seeds masses+clouds for reproducibility.

Interaction that hid it: with truth-centered clouds and dLunc=0, every sample sits exactly on
the host's catalog spike (see #3), so the delta-KDE artifact vanishes — that's why the
dlunc=0 configuration behaved differently from everything else during the hunt.

## 3. Delta-spike KDE (Dz = 1e-4)

`logpcatalog_*` builds p_X(z|pix) = Σ_j N(z; z_j, Dz) with default Dz=1e-4. The per-event
integral is estimated by evaluating this at ~10²–10³ PE sample redshifts spread over σ_z ≈ 0.05:
with spikes of width 1e-4 nearly all samples score ~0 and the estimate is dominated by the
handful landing on spikes. That's an unbiased-but-huge-variance MC estimate, and log(noisy)
biases logL downward per event (Jensen), *more for the sparser tracer* (AGN: ~0.1–4 objects per
pixel) → α̂ pulled low, plus a jagged spurious H0 landscape with overtight CIs.

Smoke measurement (uniform tiny mock, 60 events, α_true=0.505): Dz=1e-4 → α̂=0.33 (outside
90%); Dz ∈ {1e-3, 3e-3, 1e-2} → α̂ = 0.497/0.506/0.509 (plateau, truth inside 68%).

Working knobs (owner decision in GATES.md): **Dz=3e-3, N_samples_gw=2000**, chosen by the
resolution criterion n_samp·Dz/σ_z ≳ 100 with negligible smoothing systematic ((Dz/σ_z)²/2 ≈
0.2%). The statistically cleaner long-term fix is the galaxy-sum form (evaluate the smooth
per-event z-posterior at each catalog z_j instead of the spiky catalog KDE at sample z_s) —
that is how the darksirens library does it, and it is on the productionization path (P5).

## 4. One more scoring trap (not a code bug)

The inference parameter α ("alpha_agn") is the **total** AGN-hosted fraction. The injected
truth is α_true = f_agn + (1−f_agn)·λN_agn/(λN_agn+(1−λ)N_gal) computed on the z≤z_max_gw
**eligible** pools — not f_agn, and not the full-catalog counts. Comparing α̂ to f_agn looks
like a bias that isn't there (Δ ≈ (1−f_agn)·1% at the current densities — small here, but it
scales with nbar_agn/nbar_gal). `run_inference.py` now computes α_true from eligible counts.

## Status of the fixed pipeline (as of this writing)

Fresh GLASS mock (shared density field, nbar 1e-2/1e-4 arcmin⁻², bias 1.2/2.0, nside 64):
single-realization GAL run recovers H0 = 68.1, 68% CI [66.1, 69.6] (truth 67.74 inside); first
coverage realizations all cover. Full 25×2-realization coverage gate (G2) and α recovery gate
(G3) are running — verdicts land in `working/gw_agn/GATES.md`.

## What to distrust from before

Any H0/α numbers produced with (a) the β(H0) division, (b) truth-centered clouds, or
(c) Dz=1e-4 — i.e., effectively all prior inference runs — are superseded; do not quote.
The mock catalogs themselves and the pixelization are fine.
