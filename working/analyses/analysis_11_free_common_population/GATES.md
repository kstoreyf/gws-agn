# Analysis 11 gates

Registered before any Analysis-11 likelihood value exists (2026-09-26). A miss is a
FAIL to be diagnosed, never a bound to be widened. Domains are not narrowed after
the fact; if a boundary becomes relevant it is extended once and reported.

| gate | status | the number that decides it |
|---|---|---|
| 0 provenance | **PASS** | gws-agn `2db436a`; darksirens-a8 `af896ca` clean; events `427990378e29…`, injections `e8a611a27f1f…` (both re-hashed 2026-09-26) |
| L labels | **PASS** (CPU selftest) | exactly `$\mu_{\rm G}$` and `$\mu_\chi$` released; the other ten base slots stay pinned |
| C closure C1–C6 | NOT RUN | see below |
| S selection support | NOT RUN | see below |
| 11A | NOT RUN | |
| 11B | NOT RUN | |
| G11AB gate after 11A/11B | NOT REACHED | |
| S1 sampler validation | NOT REACHED | |
| 11C | NOT REACHED | |
| fixed-H0 owner gate | NOT REACHED | |
| 11D | NOT REACHED | |
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
