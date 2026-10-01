# Analysis 12 gates

| gate | status | the number that decides it |
|---|---|---|
| A12 closure K1–K3 | **PASS** (job 1348932) | K1 8/8 bitwise against A11 at the fiducial widths; K2 both widths live at f = 0, 0.266, 1 (shared); K3 7 rejections, all at σ_χ ≥ 0.2 (f = 1), ≥ 0.6 (f = 0.266) or 1.0 (f = 0), ≥ 98 nats below the fiducial (`diagnostics/a12_closure.json`) |
| 12M / 12χ | RUNNING | rita jobs 1348933/1348935 (12M s1/s2), 1348934/1348936 (12χ s1/s2) |

## Closure (before sampling)

- **K1 reduction.** At the fiducial width (σ_G = 5, σ_χ = 0.1) each cell equals the A11 cell
  in logL, logL_pe and logL_selection, within 1e-8 (bitwise expected).
- **K2 shared and live.** Moving the width moves logL at f = 0, at the 11D median f and at
  f = 1, so it reaches both branches.
- **K3 selection support.** The live guard along each width at f ∈ {0, 0.266, 1}; rejections
  are recorded with how far below the fiducial they sit.

## Decision (brief §25)

Robustness, not evidence ranking. R_M = W(Δμ_G)_{σ_G free} / W(Δμ_G)_{11D}, R_χ likewise, with
the median shifts and ρ(σ_G, Δμ_G), ρ(σ_χ, Δμ_χ). If one common width removes the
corresponding mean shift, stop and report that.
