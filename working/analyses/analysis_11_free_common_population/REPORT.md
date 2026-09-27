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

Running.
