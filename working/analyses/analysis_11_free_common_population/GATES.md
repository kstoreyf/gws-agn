# Analysis 11 gates

Registered before any Analysis-11 likelihood value exists (2026-09-26). A miss is a
FAIL to be diagnosed, never a bound to be widened. Domains are not narrowed after
the fact; if a boundary becomes relevant it is extended once and reported.

| gate | status | the number that decides it |
|---|---|---|
| 0 provenance | **PASS** | gws-agn `2db436a`; darksirens-a8 `af896ca` clean; events `427990378e29…`, injections `e8a611a27f1f…` (both re-hashed 2026-09-26) |
| L labels | **PASS** (CPU selftest) | exactly `$\mu_{\rm G}$` and `$\mu_\chi$` released; the other ten base slots stay pinned |
| C closure C1–C6 | **PASS** (C1, C2 judged at 1 ULP) | `diagnostics/a11_closure.json`, rita job 1339461. C1: 15/20 cells bitwise against both the live A10 cell and the recorded `a10_arm_J.h5`; the other 5 differ by exactly 1.82e-12 = 1 ULP of logL ≈ 1.6e4 (selection term at f = 0 in four, PE term at f = 0.3 in one). C2: 3/4 bitwise, one at 1 ULP. C3 (Δ = 0 ⇔ shared population) 4/4 bitwise; C4 (f = 0) 8/8 bitwise constant and equal to the shared cell; C5 (f = 1, only μ+Δμ enters) 9/9 bitwise constant; C6 live (logL moves by 136 over the μ_G axis, 161 over μ_χ) |
| S selection support | **PASS WITH REJECTED CORNER** | live (30 cells): 29 pass; the one rejection is (μ_χ, Δμ_χ) = (+0.1, +0.2) at f = 1 — AGN spin mean 0.30 — N_eff/threshold 0.16 (1.07 at f = 0.3). Mass extremes pass (min 1.27 at AGN mean 49, f = 1). File proxy (`diagnostics/a11_selection_file.json`): every branch point ≥ 3.5 × 5000 except AGN spin mean 0.30 (N_eff 3,894, w_max/Σw 2.1e-3). The local posterior puts the AGN spin mean at ≈ 0.12 ± 0.02; the grids bound the rejected region's mass |
| 11A | **PASS** | 26,825 cells, 22.5 GPU-h, 0 rejected, all edges ≤ 2e-12; μ_G 35.48 [34.80, 36.30], Δμ_G 3.85 [2.37, 5.23] (90%), f 0.298; ρ(μ_G, Δμ_G) −0.73; density at Δμ_G = 0 is 2.3e-4 of the peak; D_Δμ_G = 1.31 / 1.30 against A10-J; μ_G = 35 slab vs `a10_arm_J.h5` 410/525 bitwise, worst 2 ULP (`results/a11_11A.json`) |
| 11B | **PASS** | 32,375 cells, 27.2 GPU-h; μ_χ −0.0054 [−0.0202, 0.0091], Δμ_χ 0.126 [0.094, 0.158] (90%), f 0.274; ρ(μ_χ, Δμ_χ) −0.64; density at Δμ_χ = 0 is 8.5e-9; D_Δμ_χ = 1.21 / 1.24; 245 rejected cells (AGN spin mean ≥ 0.22, f ≥ 0.325), fill bound 6e-68; Δμ_χ top edge 4.3e-4 of the peak, tail mass beyond 0.20 ≈ 4e-5 (reported, not extended); μ_χ = 0 slab 258/300 bitwise, worst 2 ULP |
| G11AB gate after 11A/11B | **PASS** | all six conditions (REPORT.md, *Gate after 11A and 11B*) |
| S1 sampler validation | **PASS** (dynesty; judged against spline-interpolated grid quantiles) | dynesty multi/unif, nlive 300, seed 1 (job 1339614): 3,067 iterations, 15,173 calls, 12.7 h, logZ −4291.68 ± 0.18. Against the A10-J grid's own linear-CDF quantiles: medians ≤ 0.05, 68% ends ≤ 0.093, 90% ends up to 0.158 half-widths (all ends on the narrow side). The grid's linear CDF between coarse nodes widens its 90% ends by up to 0.22 half-widths (Δμ_G, 1 M☉ outer spacing); against cubic-spline quantiles of the same grid marginals the sampler agrees within 0.055 (medians), 0.033 (68%), 0.057 (90%), consistent with its bootstrap SE 0.024–0.035; correlations within 0.03. tinyns excluded by the owner (job 1339615 cancelled before starting) |
| 11C | **PASS** | dynesty, 2 × nlive 200 merged, 20,293 calls ≈ 17 GPU-h, seeds agree ≤ 0.17 sd; f 0.274 [0.206, 0.350], μ_G 35.56 [34.90, 36.17], Δμ_G 3.93 [2.49, 5.29], μ_χ −0.0043 [−0.0195, 0.0100], Δμ_χ 0.128 [0.096, 0.160] (90%); D vs A10-J Δμ_G 1.43/1.43, Δμ_χ 1.29/1.27; vs 11A/11B 1.00–1.04; cross-sector |ρ| ≤ 0.12; P_i(AGN) vs A10: RMS ΔP 0.028, 1 event > 0.1, Spearman 0.994 |
| fixed-H0 owner gate | **PASS on all five conditions** | REPORT.md, *Fixed-H0 owner gate*; 11D not started |
| 11D | **COMPLETE** (owner-released 2026-09-28) | dynesty, 2 × nlive 200 merged, 29,262 calls ≈ 24 GPU-h, seeds agree ≤ 0.16 sd; H0 67.16 [64.83, 69.13] (90%), planted 67.74; vs C10-J (spline) W68 1.13 / W90 1.09, median −0.50; Δμ_G 3.94 [2.52, 5.37], Δμ_χ 0.131 [0.097, 0.164], both ≥ 4.6 sd from 0; vs 11C every 90% width 1.01–1.06; ρ(H0, μ_G) −0.22 vs ρ(H0, Δμ_G) −0.09 (`diagnostics/a11_11D_comparisons.json`) |
| §21 mechanism profile | NOT RUN (owner gate) | |
| A12M / A12χ | NOT REACHED | |

## Gate C — closure (brief §11)

- **C1 baseline truth reduction.** At (μ_G, μ_χ) = (35, 0) the A11 likelihood equals
  a live Analysis-10 cell and the recorded `a10_arm_J.h5` cells, bitwise, in logL,
  logL_pe and logL_selection, at 20 (f, Δμ_χ, Δμ_G) nodes.
- **C2 free vs pinned.** At (33, −0.05) the released coordinates equal an A10-shape
  cell with those base slots pinned at the same values, bitwise.
- **C3 environmental-zero identity.** At Δμ_G = Δμ_χ = 0 the likelihood equals a
  K = 2 likelihood with one shared population at the same base values, at every f.
- **C4 f = 0.** logL is bitwise constant over (Δμ_G, Δμ_χ).
- **C5 f = 1.** logL depends on (μ_G, μ_χ) only through μ_G + Δμ_G and μ_χ + Δμ_χ.
  Constancy as the decomposition moves is the correct model geometry.
- **C6 liveness.** Both released base coordinates move logL at f = 0.3.

A non-bitwise result in C1–C5 is judged, not waived: a residual at the level of a
few ULP from a different summation path is recorded with its size; anything above
1e-6 absolute is a FAIL.

## Gate S — selection support (brief §10)

Mass points (μ_G, Δμ_G) ∈ {(31,−4), (31,+10), (35,+5), (39,−4), (39,+10)} and spin
points (μ_χ, Δμ_χ) ∈ {(−0.1,−0.05), (−0.1,+0.2), (0,+0.1), (+0.1,−0.05), (+0.1,+0.2)},
each at f ∈ {0, 0.3, 1}:

- (a) live: the hard guard passes, and N_eff / threshold is recorded (mixture
  N_eff; f = 0 and f = 1 isolate each branch);
- (b) injection-file proxy: per-branch and f = 0.3 mixture N_eff, the largest
  normalised weight, and the top-10/100/1000 shares;
- the production grids record the full rejection map.

PASS iff every live cell passes the guard; a rejection confined to a remote corner
is retained only with a bound on the posterior mass it could carry. No injections
are generated automatically.

## Gate 11A / 11B

Deterministic grids (a timing check sets one production resolution; at most one
refinement if an axis is visibly under-resolved). Report the three 1-D marginals,
the three pairwise correlations, and the density at Δμ_G = 0 (11A) or Δμ_χ = 0
(11B) against the peak. No predefined significance.

## Gate G11AB — before the joint model (brief §12)

Proceed only if: selection support is valid; both likelihood paths close; μ_G and
Δμ_G are separately informative; μ_χ and Δμ_χ are separately informative; no
posterior is prior-edge dominated; neither offset is unidentified. Otherwise STOP
and report which environmental measurement depended on the pinned baseline.

## Gate S1 — sampler validation (brief §14)

The sampler reproduces the A10-J grid posterior (f, Δμ_χ, Δμ_G at H0 = 67.74):
median, 68% and 90% interval ends within ≈ 0.1 posterior sd per coordinate, and
the pairwise correlations. Prefer tinyns if it validates, otherwise dynesty. If
both fail, STOP and report the compute blocker.

## Fixed-H0 owner gate (brief §17)

Release H0 only if: both offsets identifiable; baseline constrained away from prior
edges; selection support valid; sampler validated; no catastrophic
baseline–offset degeneracy.
