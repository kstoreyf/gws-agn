# Analysis 12 — owner report

Seed 100, one realisation, science exploration. Same mock, surveys, injections, detection
and PE model as Analyses 10 and 11. `STATE.md` holds the run record and `GATES.md` the
criteria.

## 1. The question

Analysis 11D measured the reference population (μ_G, μ_χ) and the AGN offsets (Δμ_G, Δμ_χ)
with H0 free, holding the Gaussian-peak width σ_G and the spin width σ_χ at their fiducials
(5 M☉, 0.10). A broader common population could imitate a branch difference. Each arm frees
one width, shared by GAL and AGN, on top of 11D, and asks whether the corresponding offset
survives (brief §§22–25):

R = W(offset, width free) / W(offset, 11D), with the median shift and ρ(width, offset).

Both arms: dynesty (multi-ellipsoid, uniform), two independent runs of nlive 200 merged,
first bound after 2·nlive calls. Closure before sampling passed (`diagnostics/a12_closure.json`:
8/8 points bitwise at the fiducial widths; both widths move logL at f = 0, 0.266 and 1, so
each reaches both branches). Numbers: `diagnostics/a12_comparisons.json`
(`scripts/a12_compare.py`); figure: `figs/fig_12`.

## 2. 12M — one shared Gaussian-peak width

47,705 likelihood calls (≈ 40 GPU-h), merged logZ −4297.15 ± 0.21, 7,401 equal-weight samples
(Kish n_eff 2,708). The runs agree to ≤ 0.13 posterior sd on every median and 90% end, and
their logZ values agree (−4297.23 ± 0.29, −4297.04 ± 0.29).

| | median | 68% | 90% | planted | 11D median [90%] |
|---|---|---|---|---|---|
| σ_G [M☉] | 4.70 | [4.42, 4.96] | [4.24, 5.14] | 5 | ≡ 5 |
| Δμ_G [M☉] | 4.47 | [3.55, 5.35] | [2.91, 5.90] | 5 | 3.94 [2.52, 5.37] |
| μ_G [M☉] | 35.60 | [35.21, 35.96] | [34.96, 36.20] | 35 | 35.61 [34.96, 36.25] |
| H0 | 67.43 | [65.97, 68.59] | [64.98, 69.18] | 67.74 | 67.16 [64.83, 69.13] |
| f_AGN | 0.267 | [0.224, 0.310] | [0.198, 0.340] | 0.30 | 0.266 [0.198, 0.345] |

**A broader common mass peak does not absorb the mass offset.** The shared width comes out
4.70 M☉, slightly narrower than the fiducial and with 5 inside its 90% interval, so the data
do not reach for a broader peak. With it free, Δμ_G moves up by 0.52 M☉ (0.6 of its 11D sd),
toward the planted 5, and R_M = 1.05 at both 68% and 90%. Δμ_G = 0 lies 4.95 posterior sd
below the median, and no sample falls below zero. The width and the offset trade against
each other (ρ = −0.47): a narrower common peak needs a larger displacement to place the
AGN-branch masses. The absolute AGN peak μ_G + Δμ_G is 40.04 [38.72, 41.33] M☉ (planted 40,
11D 39.56).

Freeing σ_G costs nothing elsewhere: every other 90% width is 0.96–1.00 of 11D's, and every
other median moves by ≤ 0.2 of its 11D sd. H0 moves up by 0.26 (0.2 sd), and ρ(σ_G, H0) =
−0.15.

## 3. 12χ — one shared spin width

48,959 likelihood calls (≈ 41 GPU-h), merged logZ −4296.57 ± 0.22, 7,809 equal-weight samples
(Kish n_eff 2,673). The runs agree to ≤ 0.25 posterior sd, logZ −4296.43 ± 0.30 and
−4296.67 ± 0.31. The 7 guard-rejected closure points (σ_χ ≥ 0.2 at f = 1, ≥ 0.6 at the
posterior f, 1.0 at f = 0) lie far outside the posterior.

| | median | 68% | 90% | planted | 11D median [90%] |
|---|---|---|---|---|---|
| σ_χ | 0.081 | [0.073, 0.090] | [0.067, 0.095] | 0.10 | ≡ 0.10 |
| Δμ_χ | 0.160 | [0.138, 0.180] | [0.123, 0.191]* | +0.10 | 0.131 [0.097, 0.164] |
| μ_χ | −0.0115 | [−0.0202, −0.0028] | [−0.0265, +0.0031] | 0 | −0.0035 [−0.0188, +0.0111] |
| H0 | 67.19 | [65.82, 68.42] | [64.84, 69.04] | 67.74 | 67.16 [64.83, 69.13] |
| f_AGN | 0.265 | [0.223, 0.307] | [0.199, 0.335] | 0.30 | 0.266 [0.198, 0.345] |

\* Cut by the top of the Δμ_χ prior box (0.20); see below.

**A broader common spin distribution does not absorb the spin offset either: the width
narrows, and the offset grows.** σ_χ comes out 0.081, with the fiducial 0.10 just above its
90% interval. Δμ_χ moves up by 0.029 (1.4 of its 11D sd) and lies 7.7 posterior sd from zero.
Its width barely changes: R_χ = 1.03 (68%) and 1.01 (90%) as sampled, about 1.11 at 90% once
the prior cut is undone. The reference zero point moves down by 0.008 (0.9 sd) as the width
narrows (ρ(σ_χ, μ_χ) = +0.34), and the offset moves up (ρ(σ_χ, Δμ_χ) = −0.56). The absolute
AGN spin mean rises from 0.127 [0.099, 0.156] in 11D to 0.148 [0.117, 0.175]. The planted
offset 0.10 is now below the 90% interval (2.9 posterior sd from the median).

**The narrow width is this realisation's.** Each event's χ_eff is measured with a noise of
0.155 (sd of observed − true; pull sd 1.01 against the quoted errors), larger than the 0.10
intrinsic width, so σ_χ is a deconvolution. In this draw the GAL branch's observed χ_eff
spread is 0.176, against the 0.184 expected from its true spread and the noise. Removing the
noise by moments gives 0.082 ± 0.011 for GAL and 0.099 ± 0.013 for AGN, where the true spreads
are 0.100 and 0.102. The shared width the fit returns (0.081) sits on the GAL value, the
branch with 64% of the events. The offset's rise is not in the data the same way: the observed
AGN − GAL mean difference is 0.100 (true 0.0995). It is the width–offset trade inside the
model.

**The Δμ_χ posterior reaches the top of its prior box.** 2.2% of the samples lie within 2% of
the edge at 0.20, and the last bin holds 22% of the peak density. A normal truncated to the box
and fitted to the samples puts 4.3% of the untruncated mass above 0.20 and gives 0.161
[0.124, 0.198] without the cut, against 0.160 [0.123, 0.191] sampled. The cut can only pull
the offset down, so it does not change the conclusion: the offset is, if anything, larger and
its interval wider (R_χ ≈ 1.11). The box is the brief's domain, set before any result. The
registered rule is to extend a boundary once when it becomes relevant, which here means one
more 12χ pair with Δμ_χ on [−0.05, 0.30] (≈ 41 GPU-h). That run is an owner decision and has
not been made. The AGN spin means it would add (up to ≈ 0.29) approach the region the selection
guard rejected in 11B.

Freeing σ_χ leaves the mass sector and H0 alone: every other 90% width is 0.92–0.99 of 11D's,
H0 moves by +0.02, ρ(σ_χ, H0) = −0.04.

## 4. The decision metric

| | width | R (68%) | R (90%) | offset shift (11D sd) | ρ(width, offset) | offset from zero |
|---|---|---|---|---|---|---|
| 12M | σ_G 4.70 [4.24, 5.14] | 1.05 | 1.05 | +0.52 M☉ (+0.6) | −0.47 | 4.95 sd |
| 12χ | σ_χ 0.081 [0.067, 0.095] | 1.03 | 1.01 (≈ 1.11 uncut) | +0.029 (+1.4) | −0.56 | 7.7 sd |

Neither common width removes its mean shift. Both widths come out narrower than the fiducial,
and both offsets move away from zero. Freeing a width costs the H0 posterior nothing
(90% widths 0.98 of 11D's in both arms).

logZ relative to 11D: −2.26 ± 0.21 (12M), −1.68 ± 0.22 (12χ). This is the Occam cost of the wide
width priors and is recorded, not ranked (brief §25).

## 5. Interpretation limits

One realisation. The σ_χ shortfall and the Δμ_χ rise are measured on one draw, with a
measurement noise larger than the width being inferred. How often such a draw moves Δμ_χ by
1.4 sd, and whether the 90% intervals cover the planted values at their nominal rate, is a
calibration question. The two widths were never freed together (brief §24).
