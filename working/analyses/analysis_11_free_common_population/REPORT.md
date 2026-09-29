# Analysis 11 — owner report

Seed 100, one realisation, science exploration. The mock, surveys, injections,
detection and PE model are Analysis 10's. `STATE.md` holds the running state and
`GATES.md` the registered criteria.

## 1. What changed relative to Analysis 10

Analysis 10 measured the AGN branch against a reference population it treated as known
(Gaussian-peak location 35 M☉, spin mean 0). Here the reference is inferred along with the
departure:

μ_G,GAL = μ_G, μ_G,AGN = μ_G + Δμ_G, μ_χ,GAL = μ_χ, μ_χ,AGN = μ_χ + Δμ_χ

(planted 35 M☉, +5 M☉, 0, +0.10). The likelihood is Analysis 10's with the two
reference slots unpinned. At the Analysis-10 reference it reproduces Analysis 10 to within
1 ULP. At zero offsets it is the shared-population model, bitwise. At f = 0 the offsets
drop out, and at f = 1 only the absolute AGN means enter, both bitwise
(`diagnostics/a11_closure.json`). The injections support the explored domain except for
one remote corner: an AGN spin mean of 0.30 at f = 1.

## 2. 11A — the reference mass scale free

H0 = 67.74, spin sector at the planted (0, +0.10); grid over (f, μ_G, Δμ_G), 26,825
cells, 22.5 GPU-h, no cell rejected, every axis contained (edges ≤ 2e-12 of the
peak). `results/a11_11A.{h5,json}`, `figs/fig_11A`.

| | median | 68% | 90% | planted |
|---|---|---|---|---|
| μ_G [M☉] | 35.48 | [35.07, 35.90] | [34.80, 36.30] | 35 |
| Δμ_G [M☉] | 3.85 | [2.98, 4.70] | [2.37, 5.23] | 5 |
| f_AGN | 0.298 | [0.258, 0.339] | [0.233, 0.366] | 0.30 |
| μ_G + Δμ_G [M☉] | 39.4 | [38.8, 39.9] | [38.4, 40.4] | 40 |

MAP (0.300, 35.375, 4.0). ρ(μ_G, Δμ_G) = −0.73, ρ(f, Δμ_G) = −0.32, ρ(f, μ_G) = −0.05.

**The environmental mass shift survives a free reference.** The marginal density at
Δμ_G = 0 is 2.3 × 10⁻⁴ of the peak. The reference and the offset are separately
constrained. The reference is pinned by the ~70% of events in the reference branch, and
the offset is anticorrelated with it (ρ = −0.73), because what the AGN-branch events
measure most directly is the absolute location μ_G + Δμ_G.

**The two move together along that direction.** The reference comes out 0.48 above its
planted value and the offset 1.15 below, while their sum sits 0.6 below the planted 40.
Both planted values are inside the 90% intervals; the offset is 1.3 posterior sd low. A
single realisation cannot say whether this is a draw or a pull.

**Cost against Analysis 10.** Δμ_G widths against A10-J (reference pinned, spin offset
free): D_Δμ_G = 1.31 (68%) and 1.30 (90%). The f width is unchanged (0.96×). On the
μ_G = 35 slab of this grid, Δμ_G is 4.57 [3.65, 5.54], close to A10-J's 4.76 [3.67, 5.89].
Pinning the reference is what moves the offset.

The μ_G = 35 slab reproduces the recorded A10-J cube in 410 of 525 cells bitwise, and
within 2 ULP everywhere.

## 3. 11B — the reference spin zero point free

H0 = 67.74, mass sector at the planted (35, +5); grid over (f, μ_χ, Δμ_χ), 32,375 cells,
27.2 GPU-h. `results/a11_11B.{h5,json}`, `figs/fig_11B`.

| | median | 68% | 90% | planted |
|---|---|---|---|---|
| μ_χ | −0.0054 | [−0.0143, +0.0034] | [−0.0202, +0.0091] | 0 |
| Δμ_χ | 0.126 | [0.107, 0.145] | [0.094, 0.158] | +0.10 |
| f_AGN | 0.274 | [0.239, 0.311] | [0.217, 0.335] | 0.30 |
| μ_χ + Δμ_χ | 0.120 | [0.105, 0.135] | [0.098, 0.145] | 0.10 |

MAP (0.275, −0.005, 0.125). ρ(μ_χ, Δμ_χ) = −0.64, ρ(f, μ_χ) = −0.29, ρ(f, Δμ_χ) = −0.15.

**The environmental spin shift survives a free zero point, and does not trade it away.**
The density at Δμ_χ = 0 is 8.5 × 10⁻⁹ of the peak. The reference spin mean is pinned
to ±0.009 by the reference-branch events. The offset is anticorrelated with it
(ρ = −0.64) for the same reason as in 11A: the AGN-branch events measure the absolute
AGN spin mean most directly. The zero point comes out 0.005 below 0 and the offset 0.009
above the Analysis-10 value, so the absolute AGN mean is 0.120. The planted offset is
inside the 90% interval and 1.4 posterior sd below the median.

**Cost against Analysis 10:** D_Δμ_χ = 1.21 (68%) and 1.24 (90%) against A10-J. On the
μ_χ = 0 slab of this grid Δμ_χ is 0.119 [0.095, 0.144], indistinguishable from A10-J's
0.117 [0.093, 0.144]. The f interval is narrower than A10-J's (0.86×) because the mass
offset is pinned here and free there.

**Domain edge and rejected cells.** The Δμ_χ marginal at the top of the domain (0.20) is
4.3 × 10⁻⁴ of the peak. It falls by factors of 3–7 per node towards the edge, the edge is
3.9 posterior sd above the median, and the extrapolated mass beyond it is 4 × 10⁻⁵. It is
reported, not extended. 245 cells (AGN spin mean ≥ 0.22 at f ≥ 0.325) are
guard-rejected, and a fill bound puts at most 6 × 10⁻⁶⁸ of the posterior there. The
μ_χ = 0 slab reproduces A10-J's recorded cells in 258 of 300 cases bitwise, and within
2 ULP everywhere.

## Gate after 11A and 11B: PASS

Selection support is valid over both explored regions (the only rejections are in the
AGN-spin-mean ≥ 0.22 corner, bounded at 6 × 10⁻⁶⁸). Both likelihood paths close. μ_G and
Δμ_G are separately informative (ρ = −0.73, zero offset at 2.3 × 10⁻⁴), and so are μ_χ
and Δμ_χ (ρ = −0.64, zero offset at 8.5 × 10⁻⁹). No posterior is edge-dominated, and
neither offset is lost. Neither environmental measurement depended on the pinned
reference: freeing it widens the offsets by 1.2–1.3× and moves them by about one
posterior sd.

## 4. 11C — both references and both offsets free, fixed H0

(f, μ_G, Δμ_G, μ_χ, Δμ_χ) at H0 = 67.74 on the brief's domains, with dynesty
(multi-ellipsoid, uniform; validated in S1). Two independent runs (nlive 200, seeds 1 and
2) were merged with `dynesty.utils.merge_runs`: 20,293 likelihood calls, ≈ 17 GPU-h,
merged logZ −4293.40 ± 0.18, 5,618 equal-weight samples. `results/a11_11C.json`,
`figs/fig_11C`. The two runs agree to ≤ 0.17 posterior sd on every median and 90% end,
and their logZ values (−4293.26 ± 0.24, −4293.51 ± 0.25) agree.

| | median | 68% | 90% | planted |
|---|---|---|---|---|
| f_AGN | 0.274 | [0.230, 0.321] | [0.206, 0.350] | 0.30 |
| μ_G [M☉] | 35.56 | [35.16, 35.93] | [34.90, 36.17] | 35 |
| Δμ_G [M☉] | 3.93 | [3.05, 4.75] | [2.49, 5.29] | 5 |
| μ_χ | −0.0043 | [−0.0135, +0.0044] | [−0.0195, +0.0100] | 0 |
| Δμ_χ | 0.128 | [0.109, 0.148] | [0.096, 0.160] | +0.10 |

Correlations: ρ(μ_G, Δμ_G) = −0.66 and ρ(μ_χ, Δμ_χ) = −0.54, as in 11A and 11B. Every
cross-sector coefficient is small: ρ(Δμ_G, Δμ_χ) = +0.02, ρ(μ_G, μ_χ) = +0.12,
ρ(μ_G, Δμ_χ) = +0.11, ρ(Δμ_G, μ_χ) = +0.08. f couples to both sectors: ρ(f, Δμ_G) = −0.33,
ρ(f, μ_χ) = −0.40, ρ(f, Δμ_χ) = −0.19, ρ(f, μ_G) = −0.17.

**Both references and both environmental offsets are measured at once.** Δμ_G = 0 lies
4.6 posterior sd below the median (1 of 5,618 samples below zero). Δμ_χ = 0 lies 6.6 sd
below (no sample below 0.057). Neither reference is near its prior edge: no sample lies
within 2% of any box edge, except 0.04% at the top of Δμ_χ.

**The mass and spin sectors are independent.** 11C reproduces 11A and 11B, each run with
the other sector pinned at its planted values. Every offset and reference width is within
4% of the one-sector value, and every median within 0.1 M☉ or 0.002.

## 5. How much the free references cost

Widths against Analysis 10's A10-J (references pinned, both offsets free, H0 = 67.74). The
A10-J grid quantiles are spline-interpolated, as in S1:

| | D (68%) | D (90%) | median shift A11 − A10 |
|---|---|---|---|
| Δμ_G | 1.43 | 1.43 | −0.82 M☉ |
| Δμ_χ | 1.29 | 1.27 | +0.011 |
| f_AGN | 1.15 | 1.12 | −0.002 |

Against 11A/11B (only that sector's reference free), 11C's widths are 1.00–1.04. Freeing
the second sector's reference costs nothing more.

The cost is the reference–offset trade within each sector. What the AGN-branch events
measure best is the absolute AGN location, μ_G + Δμ_G or μ_χ + Δμ_χ. Once the reference
moves, the offset inherits its uncertainty. The mass offset's −0.82 median shift is that
trade at work: the reference moves up 0.56 and the offset down, while the absolute AGN peak
moves by only −0.26.

## 6. Are the event-level AGN assignments stable?

P_i(AGN) at the 11C posterior median against the A10-J posterior median, both from the
verified per-event decomposition (each pass reproduces the production logL_pe to ≤ 2 ULP;
`diagnostics/a11_event_assignment.json`):

- RMS ΔP 0.028, median |ΔP| 0.015, max |ΔP| 0.103.
- One event moves by more than 0.1 and 98 by more than 0.05.
- 32 events cross 0.5; these are events already sitting near it.
- Spearman rank correlation 0.994. ΣP moves from 337.9 to 323.2 (f from 0.276 to 0.274).

The classification is stable: freeing the references changes which events look like AGN
events only at the margin. (For interpretation only: agreement with the true host label at
0.5 is 72.4% for A10 and 72.0% for A11.)

## Fixed-H0 owner gate: PASS on all five conditions

1. Both environmental offsets remain identifiable (4.6 and 6.6 posterior sd from zero).
2. Both references are constrained away from their prior edges: μ_G ± 0.39 inside [31, 39],
   μ_χ ± 0.009 inside [−0.10, +0.10].
3. Selection support is valid (Gate S; the grids bound the only rejected corner at 6e-68).
4. The sampler validated (S1).
5. There is no catastrophic reference–offset degeneracy: |ρ| ≤ 0.66 within each sector,
   and ≤ 0.12 across sectors.

11D (H0 released) followed, on owner release.

## 7. 11D — H0 released

(H0, f, μ_G, Δμ_G, μ_χ, Δμ_χ) with H0 on [60, 76] (C10-J's window) and the same dynesty
configuration as 11C. Two runs (seeds 1 and 2, rita jobs 1345442 and 1345443) were merged:
29,262 likelihood calls, ≈ 24 GPU-h, merged logZ −4294.89 ± 0.19, 6,213 equal-weight
samples (Kish n_eff 2,548). The runs agree to ≤ 0.16 posterior sd on every median and 90%
end, and their logZ values (−4294.71 ± 0.26, −4295.05 ± 0.26) agree. `results/a11_11D.json`,
`diagnostics/a11_11D_comparisons.json` (`scripts/a11_11D_compare.py`), `figs/fig_11D`.

| | median | 68% | 90% | planted |
|---|---|---|---|---|
| H0 | 67.16 | [65.72, 68.41] | [64.83, 69.13] | 67.74 |
| f_AGN | 0.266 | [0.223, 0.312] | [0.198, 0.345] | 0.30 |
| μ_G [M☉] | 35.61 | [35.22, 36.01] | [34.96, 36.25] | 35 |
| Δμ_G [M☉] | 3.94 | [3.11, 4.83] | [2.52, 5.37] | 5 |
| μ_χ | −0.0035 | [−0.0127, +0.0061] | [−0.0188, +0.0111] | 0 |
| Δμ_χ | 0.131 | [0.110, 0.150] | [0.097, 0.164] | +0.10 |

Both offsets stay identified: Δμ_G = 0 lies 4.6 posterior sd below the median (no sample
below 0.23) and Δμ_χ = 0 lies 6.4 sd below (none below 0.054). No sample lies within 2% of
any box edge, except 0.03% at the top of Δμ_χ.

**H0 is recovered with the references free.** The median is 0.58 below the planted value
(0.44 posterior sd). The reference is C10-J, the correctly specified Analysis-10 model with
the references pinned at (35, 0), and not the spatial-only C10-S. Against C10-J the median
moves by −0.50, and the widths are W68 1.13 and W90 1.09 (posterior sd 1.30 against 1.20).
C10-J's quantiles are spline-interpolated on its grid, as in S1; its own linear-CDF
quantiles give 1.09 and 1.06.

**Releasing H0 costs the population coordinates nothing.** Against 11C at fixed H0, every
90% width is within 7% (1.01–1.06) and every median within 0.06 M☉, 0.008 in f and 0.003
in spin. Against C10-J the environmental marks widen by Δμ_G 1.39 / 1.39 (68% / 90%;
median −0.85 M☉), Δμ_χ 1.32 / 1.33 (+0.012) and f 1.13 / 1.14 (−0.007). These are the 11C
costs against A10-J (1.43 and 1.27–1.29) again. The marks degrade because the references
are free, not because H0 is.

## 8. Does the environmental mass model still prevent the cosmological bias?

On this realisation, yes. The spatial-only model C10-S, which gives the AGN branch the GAL
mass function, sits 2.94 above C10-J (2.6 of C10-J's posterior sd). 11D sits 0.50 below
C10-J. Stated differentially: C10-J − C10-S = −2.94 and 11D − C10-J = −0.50. The mass mark
removes the offset even when the reference mass scale it is measured against is inferred.
This is one draw, not a coverage statement.

## 9. Which population coordinate H0 trades against

| | ρ(H0, ·) 11D | ρ(H0, ·) C10-J | dH0/dx (regression) |
|---|---|---|---|
| μ_G | −0.22 | pinned | −0.73 per M☉ |
| Δμ_G | −0.09 | −0.25 | −0.13 per M☉ |
| μ_χ | −0.03 | pinned | — |
| Δμ_χ | −0.10 | −0.08 | — |
| f_AGN | +0.16 | +0.09 | — |
| μ_G + Δμ_G (AGN peak) | −0.23 | — | −0.42 per M☉ |

**H0 trades against the overall mass scale, not against the environmental displacement.**
With the reference pinned, Δμ_G was the only free mass location, so it carried the
mass–distance coupling (ρ = −0.25). Once μ_G is free the coupling moves onto it (−0.22) and
ρ(H0, Δμ_G) falls to −0.09. Through m_det = (1 + z) m_src, a common shift of both mass peaks
reads as a change of the distance scale. A shift of the AGN peak relative to the reference
does not, because the ~70% of events in the reference branch fix the source-frame scale.
The spin zero point does not enter (ρ = −0.03).

**This one coupling accounts for the H0 shift and the widening.** μ_G came out 0.61 M☉
above 35, which at −0.73 per M☉ is −0.45 in H0: the observed −0.50 shift. Within the slice
μ_G ∈ [34.6, 35.2] (870 samples) H0 is 67.43 ± 1.21, against C10-J's 67.57 ± 1.20, so
fixing the reference mass scale recovers C10-J's H0 posterior. The H0 spread grows with μ_G
across the slices (sd 1.21, 1.27, 1.28, 1.32 from low to high μ_G). A linear regression on
(μ_G, μ_χ) therefore gets the shift right (E[H0 | 35, 0] = 67.56) but understates the width
(R² = 0.047, a 2% narrowing).

These statements come from the posterior samples. The section-21 H0 profile at pinned
references is the likelihood-level test of the same claim, and it has not been run.

## 10. Common-width stress tests

Not reached. Analysis 12 (a shared σ_G, then separately a shared σ_χ) waits for the
owner.

## 11. The model to calibrate

Deferred until Analysis 12 shows whether a common width absorbs either offset.
