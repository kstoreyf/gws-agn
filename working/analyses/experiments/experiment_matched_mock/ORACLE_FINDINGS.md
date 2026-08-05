# ORACLE_FINDINGS — exact-likelihood oracle for the −0.80 residual

**Status: MECHANISM FOUND AND VERIFIED.**  The residual is dominated by a
third mock defect of the same family as the two already fixed: the mock's
sky-localisation width `sigma_ang = clip(35/rho_opt(true params), 1°, 12°)` is
a deterministic function of the LATENT true parameters, handed to the PE as
known metadata.  Because `sigma_ang ∝ dL/Mc_det^(5/6)`, the width is itself an
H0-sensitive observable; a fixed-width sky posterior (the mock's PE clouds, and
therefore darksirens' pixelated sky treatment, and this oracle's O1..O4)
cannot represent it, and the omission breaks the score identity of the
detected-set likelihood by −0.0047 ± 0.0008 per event per km/s/Mpc — an H0
bias of **−0.49 ± 0.08 that survives even the EXACT likelihood**.  Deriving
the width from the OBSERVED masses/distance (drawn first, sequentially)
restores closure: parametric bootstrap −0.62 ± 0.06 (as-is recipe) →
**−0.06 ± 0.07** (fix), and the per-event score identity 0.0364 → 0.0406 ±
0.0007 against the required 0.0411.  The remaining darksirens-vs-exact
overhead is −0.31 ± 0.13, itself mostly the Farr 1/Neff(H0) term (−0.12
systematic) plus zero-mean noise terms quantified below.  Fix PR: darksirens
`fix/mock-observable-sky-width` (see section 9).

## 1. The generative process (obs arm, per `scripts/build_obsdet_mock.py`)

Per proposal:
1. Host `j` drawn **uniformly** over the 1M-entry complete catalog (atoms at exact
   `(alpha_j, delta_j, z_j)`); kept with probability `(1+z_j)^(gamma-1)/rate_gmax`,
   gamma = 0, rate_gmax = 1.  Net host prior `P(j) ∝ (1+z_j)^(-1)`.
2. Masses: `m1src ~ powerlaw+peak` (w_PL = 0.10, alpha 2.3, mmin 5, mmax 80,
   dm 3/10, muG 35, sigG 5), `q | m1, component` with per-component pairing
   (PL: floor (5,3); peak: floor (1,0.01)), beta = 1.  `chi ~ TN(0, 0.10, ±1)`.
   Verified: `gmd._mass_spin_pdf * (1+z)^(gamma-1)` equals darksirens'
   `log_p_pop` to 1.4e-14 max |dlog| over 4000 random points.
3. One observation, shared by detection and PE (post fix):
   - `D = dL(z_j; H0*) exp(s eps)`, s = 0.10, `dL` from astropy Flatw0waCDM via
     `gmd._cosmology_grids` (20k nodes to z=2).
   - `M1 = clip(N(m1det, a1 m1det), 2, inf)`, a1 = 0.08, `m1det = m1src (1+z_j)`
     (heteroscedastic: width prop. to the TRUE mass).
   - `M2 = clip(N(m2det, a2 m2det), 1, inf)`, a2 = 0.10.
   - `X  = clip(N(chi, 0.08), -1, 1)`.
   - `A = alpha_j + N(0, sig_i/max(cos delta_j, 0.1)) mod 2pi`,
     `B = clip(delta_j + N(0, sig_i), ±pi/2)`, with
     `sig_i = clip(35/rho_opt, 1, 12) deg`, rho_opt deterministic in the TRUE
     `(m1det, m2det, dL)` with snr_ref_control = 11.5.
4. Detection: `rho_obs = 6.278363879917771 (Mc_det_obs/30)^(5/6)(1000/D) >= 8`
   — deterministic in the observed data, so `1[det(d)] = 1` on the detected set
   and the standard detected-set likelihood applies.
5. PE: exact flat-prior posteriors GIVEN that observation: `dL` samples
   `LogNormal(ln D + s^2, s)` (verified: equals likelihood(D|dL) as a function of
   dL up to a constant e^{-s^2/2}); `m1det ~ N(M1, w1)` with the FIXED width
   `w1 = a1 * m1det_true` (stored `truth/obs_sig_m1`); likewise m2det, chi, sky.

## 2. The exact per-event marginal likelihood (numerator)

On the detected set, `L_i(Lambda) ∝ Σ_j P(j) p(d_i | j, Lambda)` with everything
except the host discrete.  Since chi and p_pe factor out (H0-independent per
event), and with Om0 pinned so `dL(z;H0) = dL(z;H0_ref) · H0_ref/H0` exactly:

    L_i(H0) = const_i × Σ_j (1+z_j)^(-1) · F_sky(i,j) · D_i(z_j;H0) · M_i(z_j)

    D_i(z;H0) = exp( -(ln D_i - ln dL(z;H0))^2 / (2 s^2) )
    F_sky(i,j) = N(A_i; alpha_j, sig_i/max(cos delta_j,0.1)) · N(B_i; delta_j, sig_i)
                 (exact generative sky likelihood; sig_i approximated by its
                  stored true value — the theta-dependence of sig is dropped, an
                  approximation whose score has zero mean at truth)
    M_i(z)   = ∫∫ rho(m1,m2) N(M1; m1 t, ·) N(M2; m2 t, ·) dm1 dm2,   t = 1+z
    rho(m1,m2) = p_PLpeak(m1) · pair(m2/m1 | m1, component) / m1   (density in masses)

Two mass-noise variants:
- **O1 (exact)**: widths `a1 m1 t`, `a2 m2 t` (heteroscedastic, incl. the 1/(a m)
  normalisation) — the true generative likelihood.
- **O2 (PE-implied)**: fixed widths `w1_i, w2_i` — the likelihood the mock's PE
  cloud actually encodes.  O2−O1 measures the mock's PE mis-specification
  (the PE pretends the noise width is a known constant equal to its true-mass
  value; the exact flat-prior posterior of the heteroscedastic likelihood is
  NOT a Gaussian centred at the observation — its mean sits ≈ a^2 lower).

Sky ablations:
- **O3**: `F_sky(i,j) → Q_i(pix_j)` where `Q_i(pix)` is the analytic mass of the
  PE sky cloud in HEALPix pixel `pix` (nside 16) — darksirens' pixelated sky.
- **O3b**: O3 with the extra `(1+z) · m1src` weight — the p_pe basis-Jacobian
  factor darksirens' estimator carries (derived below).
- **O4**: full darksirens expectation: O3b with the atoms replaced by the survey
  KDE kernels `g(z) N(z; z_k, 0.003)/Z_k` (per-kernel unit mass, volumetric
  tilt g = dV_c/dz).

## 3. What darksirens' per-event estimator converges to (derived)

darksirens computes `Zhat_i = (1/S) Σ_s w_s` with
`w_s = p_pop(m1src,q,z,chi) p_cat(z|pix) N_obs[pix]/N_tot / [(ddL/dz)(1+z) p_pe]`,
z = z(dL_s; H0).  The PE sample density is `N(m1det;M1,w1) N(m2det;M2,w2) ×
LN(dL) × (chi, sky factors)` in the (m1det, m2det, dL) product basis with
p_pe = 1.  Taking the expectation, integrating dL → z (the ddL/dz cancels), and
switching mass integration to source frame ((1+z)^2 Jacobian):

    E[Zhat_i] = const × ∫ dz (1+z)^(gamma-1) · (1+z) · W_i(z) · D_i(z;H0) · Mm_i(z)

with `W_i(z) = Σ_pix Q_i(pix) N_obs[pix] p_cat(z|pix)` and `Mm_i` the O2 mass
integral carrying an extra `m1src` factor.  The **extra `(1+z)·m1src`** relative
to the exact form is precisely the p_pe basis Jacobian (`p_pe should be m1det =
(1+z) m1src`); rewriting `p_pe → m1det` was already measured to move H0 by only
−0.039.  So the candidate mechanisms in the numerator are (a) O2−O1 (PE noise
mis-specification), (b) O3−O2 (sky pixelation), (c) O4−O3b (KDE smoothing +
volumetric kernel tilt), (d) O3b−O3 (Jacobian, known small).

## 4. The exact selection function

    mu(H0) ∝ Σ_j (1+z_j)^(-1) · Pdet(z_j; H0)
    Pdet(z;H0) = E_{m1,q ~ pop} E_{eps1,eps2} Phi( [V + zeta(z;H0)] / s )
    V = (5/6)(ln Mc_src + ln kappa(q,eps1,eps2)),
    kappa = Mc(m1(1+a1 eps1), m2(1+a2 eps2)) / Mc(m1,m2)   (scale-free; clips are
            >7 sigma away for every populated mass),
    zeta(z;H0) = ln(snr_ref·1000/8) - (5/6) ln 30 + (5/6) ln(1+z) - ln dL(z;H0).

`H0` enters only through `- ln dL = - ln dL_ref(z) + ln(H0/H0_ref)`, so
`mu(H0)` is a single 1-D profile evaluated at shifted arguments.  The
distribution of V is built by deterministic quadrature (pop grid × 2-D
Gauss–Hermite for kappa, per-q convolution on a uniform V-grid); no Monte Carlo.

darksirens' `mu_hat(H0)` (from the shared 120M-draw injection file) is
structurally exact for this rule (pdraw carries the full Jacobian; detection is
data-deterministic) — its errors are the SHARED-realisation MC noise of the
slope, the KDE smoothing, and the z(dL) table.  Note **all 20 realisations use
the same sel_obs.h5**, so selection-side MC error is common mode and does not
average down in the −0.80 ± 0.16.  darksirens additionally adds the Farr
correction `+N(N+3)/(2 Neff(H0))`, absent from the exact likelihood; its H0
tilt is measured directly.

## 5. Validation of the two instruments

- **Eager darksirens mirror** (`scripts/oracle_ds_eager.py`): reproduces the
  archived compiled GPU scan `results/obsdet_obs_b.h5` to max |diff| = 5.5e-12
  over the full 161-point grid — the darksirens side is bit-faithful.
- **Oracle quadrature**: doubling every quadrature resolution changes summed
  slopes at truth by < 0.02 nats/km (ΔH0 < 0.005).  The exact mu(H0) was
  validated two independent ways: 20M-draw fresh semi-MC of the generative
  process agrees with the quadrature slope to 2.5e-4/km, and the predicted
  detected fraction 1.668e-3 matches the realized 1000/616000 = 1.623e-3
  (within its 3.2% binomial error).
- **darksirens vs astropy distance maps**: |Δ ln dL| ≤ 6e-5 → ≤ 0.004 on H0.
  The z(dL)-inversion candidate is dead.
- gmd's mass/spin/rate density equals darksirens' `log_p_pop` to 1.4e-14.

## 6. Measurements (realisation b; ds peak 66.815, offset −0.925)

Attribution of (darksirens − exact oracle), slopes at truth (nats/km) and the
equivalent peak shift dH0 = slope / curvature (curvature 8.07):

| term | slope | dH0 | status |
|---|---|---|---|
| O2−O1: PE noise-width mis-spec (mock defect, small) | −0.38 | −0.047 | systematic, tightly measured |
| O3−O2: sky pixelation (nside 16) | +1.46 ± 1.22 | +0.180 | consistent with 0 (1.2σ), realisation-dependent |
| O3b−O3: p_pe basis Jacobian (1+z)·m1src | +0.25 | +0.031 | systematic; confirms the earlier −0.039 measurement |
| O4−O3b: KDE kernels + volumetric tilt | +0.42 ± 0.53 | +0.053 | consistent with 0 |
| ds−O4: PE-sample MC residual | +1.73 ± 1.49 | +0.214 | consistent with 0 (1.2σ); uncorrelated with event properties |
| **numerator total (ds − O1)** | **+3.47** | **+0.431** | |
| selection: −N (ln mu_hat − ln mu_exact) | −6.06 | −0.751 | **injection×catalog MC fluctuation** (see below) |
| selection: Farr term +N(N+3)/(2 Neff(H0)) | −0.95 | −0.118 | systematic (Neff(H0) rises with H0) |
| **selection total** | **−7.01** | **−0.869** | |
| **grand total** | **−3.54** | **−0.438** | matches the paired peak diff −0.485 |

Oracle exact peak for b: **67.299 (−0.441)** — i.e. about half of b's −0.925 is
ordinary realisation noise; the paired darksirens-minus-exact systematic for
this realisation is −0.485.

## 7. Campaign-level verdict (all 20 realisations)

Evaluating the exact oracle on every realisation (posterior MEDIAN, the
campaign's statistic; darksirens side = the archived compiled scans):

| quantity | mean ± sem | sd |
|---|---|---|
| darksirens offset | −0.802 ± 0.162 | 0.722 |
| **exact-oracle offset** | **−0.489 ± 0.077** | 0.344 |
| paired (darksirens − oracle) | −0.312 ± 0.125 | 0.559 |

**The exact likelihood itself recovers H0 low at 6.4σ.**  Equivalently the
score identity fails: mean d lnL/dH0 at truth = −4.80 ± 0.75 nats/km over the
20 realisations (expected 0; per-realisation score sd 3.33 matches √Fisher =
3.06).  The deficit sits in the per-event numerator: mean per-event score
0.0364 vs the d ln mu/dH0 = 0.0411 the identity demands (deficit −0.0047 ±
0.0008 per event per km/s/Mpc).

Cross-checks that localize the deficit:
- Parametric bootstrap (fresh events drawn with a re-implementation of the
  generator recipe, oracle evaluated): mean offset −0.619 ± 0.062 (n = 20) and
  the same per-event score deficit (−0.0058 ± 0.0007) — so the mock files match
  the recipe; the mismatch is between the RECIPE and the (fixed-sigma) likelihood.
- The detected data marginals match the recipe exactly: E[u], E[u|z] bin by
  bin, z-quantiles (30M-draw semi-MC vs the pooled 20k real events).
- The sampler-vs-pdf mass density agrees to 0.04% (20M draws, 404 bins); the
  mass-density log-slope error is +0.004 ± 0.002 per ln m1 — irrelevant.
- A minimal toy (hosts + lognormal distance + observed-data threshold)
  satisfies the identity — the core structure is fine.
- Toy-B (adds a sky channel with width sigma ∝ dL(z;H0)): the LIVE-sigma exact
  likelihood satisfies the identity; the FIXED-sigma (truth-frozen) likelihood
  shows a clear score deficit.  **This is the mechanism** (see section 8).

The selection-side (darksirens − exact) overhead over 20 realisations:
mu_hat excess slope mean +0.00026 ± 0.00077 (sd 0.0034) — ZERO mean, but a
per-realisation NOISE of ±0.36 km/s/Mpc on H0 (an unreported variance term
from the injection×catalog-KDE interaction, shared-injection file).  The Farr
term −N(N+3)/(2 Neff(H0)) contributes a SYSTEMATIC ≈ −0.12 (Neff rises with
H0).  Numerator-ladder means over 20: pixelation +0.978 ± 0.198 nats/km
(≈ +0.10 on H0), Jacobian +0.242 (≈ +0.026), PE-width −0.356 (≈ −0.038),
KDE +0.139 ± 0.094 (≈ +0.015).

## 8. The mechanism: the sky-noise width is a latent-dependent observable

In the mock, `sigma_ang = clip(35/rho_opt(m1det, m2det, dL_true), 1°, 12°)` —
a deterministic function of the TRUE parameters — and the PE sky cloud is
built as a fixed-width Gaussian using that truth-derived width.  The actual
observation model therefore has p(sky_obs | theta) with width sigma(theta) ∝
dL(z;H0)/Mc_det^(5/6): the sky data carry distance (hence H0) information
through the width.  The exact likelihood harvests it (Toy-B: identity holds);
any likelihood that freezes the width at its per-event true value — my oracle
O1..O4, the mock's own PE clouds, and hence darksirens' pixel-histogram sky
treatment — omits it, and the omission has a NEGATIVE mean score at truth:
the posterior-weighted E[r²/sigma²] is < 2 because events whose true host got a
large sky residual down-weight that host, so the (2 − r²/sigma²)·d ln sigma/dH0
term does not average to zero under the frozen-width likelihood.

This mechanism reproduces every measured constraint: ∝ sigma_PE² (host
ambiguity → competitor weight → the correlation term; vanishes as the true
host dominates), flat in nsamp (not MC), numerator-localized (mu has no sky),
even across events, catalog-depth invariant, and WORSE at finer nside — the
nside-16 pixelation error (+0.10 here) partially cancels the −0.5, and
refining the pixels removes the cancellation.

## 9. Fix verification and the upstream PR

Making `sigma_ang` a SEQUENTIAL function of the observed data (draw D, M1, M2
first; `sigma_ang = clip(35/rho(M1obs, M2obs, Dobs; SNR_REF_DEFAULT), 1, 12)°`;
then draw the sky offsets) makes the fixed-width sky posterior exact:

| ensemble (20 x 1000 fresh events, exact oracle) | mean offset |
|---|---|
| recipe as-is (sigma_ang from true params) | −0.619 ± 0.062 |
| sigma_ang from observables (the fix) | **−0.062 ± 0.066** |

and the pooled per-event score at truth: 0.0353 ± 0.0007 (as-is) →
0.0406 ± 0.0007 (fix) against the identity value 0.0411.  The small remaining
deficit bound (−0.0005 ± 0.0007) is consistent with the measured mass-width
latent channel (O2−O1 ≈ −0.05 on H0), documented but not restructured.

Upstream fix: darksirens branch `fix/mock-observable-sky-width` on top of
master ad71915 (post #332/#334): `_measure` draws masses/distance first and
derives the sky width from them when no explicit `sky_uncertainty_deg` is
given; `_detect_on_observation` requests that path; three new tests in
`tests/test_mock_detection_data.py` pin (i) `obs_sigma_ang` recomputable from
the stored observables, (ii) not equal to the truth-derived width, (iii) the
explicit-constant override.  NOT merged (per instructions).  **PR: https://github.com/ignaciomagana/darksirens/pull/335**

Direct per-event brute-force split on real events (masses from the
samplers, live sigma(theta)): the sampler-vs-pdf channel is null at high
precision (+0.00012 +- 0.00016 per event) and the sigma(theta) channel is
positive-trending (+0.0021 +- 0.0063 at n=10; the per-event estimator is
noisy and was cut short -- the fixed-recipe closure and the restored score
identity above are the operative proofs).

## 10. What remains open

- The darksirens-minus-exact overhead −0.31 ± 0.13 (2.5σ): Farr term −0.12
  (systematic; consider whether the 1/Neff(H0) correction belongs in
  grid-scan closures at all, or budget Neff per the GWTC variance criterion
  instead of the inert 1e6 guard), pixelation +0.10 ± 0.02, Jacobian +0.026,
  PE-width −0.038, KDE +0.015, and PE-MC / mu-MC noise (zero-mean, per-
  realisation sd ~0.4-0.5 each).  After the mock fix the residual campaign
  bias should be ≈ (−0.31) − (numerator terms already inside the oracle
  ladder…) — i.e. re-run the 20-realisation campaign with regenerated mocks to
  confirm the total closes; predicted ≈ −0.3 ± 0.15 before any estimator-side
  cleanup, ≈ −0.15 with the Farr term neutralised.
- The mu-hat injection×catalog-KDE interaction contributes ±0.36/realisation
  of unmodelled scatter (part of why per-seed intervals are ~2x too narrow);
  catalog-targeted injections or larger Neff would shrink it.
- The mass-width latent channel (−0.05) and the sigma_ang clip atoms remain in
  the mock; harmless at current precision.

**The selection excess decomposed.**  mu_hat's local slope at truth is
0.04672 ± 0.00293 vs exact 0.04109 (excess +0.0056, 1.9σ against the
delta-method/bootstrap MC error of the injection set).  Splitting the
fluctuation: reweighting the SAME detected injections with the smooth
(catalog-ensemble-mean) host density reproduces the exact slope to +2e-5 — the
injection-set-only (common-mode) fluctuation is ZERO.  The excess is therefore
an injection×catalog-KDE interaction: with dz = 0.003 kernels, mu_hat is
dominated by injections that land within a kernel width of a catalog host in
their pixel, and as H0 varies each (injection, host) coincidence slides in/out
of contact over ΔH0 ~ dz / (dz/dH0) ≈ 1.7 km/s/Mpc — producing exactly the
observed wiggles of ln mu_hat − ln mu_exact with ~0.005 amplitude and ~2 km/s/Mpc
period.  This term varies per catalog realisation (b: +0.0056; n4201: −0.0010;
ctrl injection set on b's catalog: +0.0001).

