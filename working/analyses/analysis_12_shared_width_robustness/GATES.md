# Analysis 12 gates

| gate | status | the number that decides it |
|---|---|---|
| A12 closure K1–K3 | **PASS** (job 1348932) | K1 8/8 bitwise against A11 at the fiducial widths; K2 both widths live at f = 0, 0.266, 1 (shared); K3 7 rejections, all at σ_χ ≥ 0.2 (f = 1), ≥ 0.6 (f = 0.266) or 1.0 (f = 0), ≥ 98 nats below the fiducial (`diagnostics/a12_closure.json`) |
| 12M | **PASS: offset survives** | 2 × nlive 200, 47,705 calls ≈ 40 GPU-h, seeds ≤ 0.13 sd; σ_G 4.70 [4.24, 5.14] (planted 5 inside); Δμ_G 4.47 [2.91, 5.90], R_M 1.05 / 1.05, shift +0.52 (0.6 sd), ρ(σ_G, Δμ_G) −0.47, Δμ_G = 0 at 4.95 sd; H0 90% width 0.98× 11D |
| 12χ | **PASS: offset survives; Δμ_χ prior edge relevant** | 2 × nlive 200, 48,959 calls ≈ 41 GPU-h, seeds ≤ 0.25 sd; σ_χ 0.081 [0.067, 0.095] (planted 0.10 just outside; GAL deconvolved spread of this draw 0.082 ± 0.011); Δμ_χ 0.160 [0.123, 0.191], R_χ 1.03 / 1.01 (≈ 1.11 uncut), shift +0.029 (1.4 sd), ρ −0.56, 7.7 sd from 0; 2.2% of samples within 2% of the 0.20 box top (≈ 4.3% mass cut). Extension to [−0.05, 0.30] (≈ 41 GPU-h) DEFERRED by the owner 2026-10-01: the conclusion holds with the cut (REPORT §3 note) |

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
