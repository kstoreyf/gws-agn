# Guard audit — darksirens master @ 2b86a2d

Total-variance guard (`darksirens/likelihood/selection.py`): a cell is
admitted only if `Neff > max(5·N_obs, N_obs²/(V − Σσ²_PE))` with
`V = max_likelihood_variance` (default **1.0**, the GWTC-4.0/5.0 bound on
σ²_lnL). Equivalently `σ²_total = Σσ²_PE + N_obs²/Neff ≤ V`. This criterion
did not exist in the #212-era code the previous run used, which faced only
the legacy `Neff > 5·N_obs` floor (last column).

| config | model | K | N_obs | Neff | Σσ²_PE | N²/Neff | σ²_total | ≤1.0? | legacy 5N? |
|---|---|---|---|---|---|---|---|---|---|
| k1_dscf_agn_r00 | dscf | 1 | 100 | 1.47e+05 | 9.14 | 0.0678 | 9.21 | fail | pass |
| k1_dscf_agn_r01 | dscf | 1 | 100 | 1.47e+05 | 8.14 | 0.0678 | 8.21 | fail | pass |
| k1_dscf_agn_r02 | dscf | 1 | 100 | 1.47e+05 | 8.44 | 0.0678 | 8.51 | fail | pass |
| k1_dscf_agn_r03 | dscf | 1 | 100 | 1.47e+05 | 9.39 | 0.0678 | 9.46 | fail | pass |
| k1_dscf_agn_r04 | dscf | 1 | 100 | 1.47e+05 | 10.3 | 0.0678 | 10.4 | fail | pass |
| k1_dscf_agn_r05 | dscf | 1 | 100 | 1.47e+05 | 6.72 | 0.0678 | 6.79 | fail | pass |
| k1_dscf_agn_r06 | dscf | 1 | 100 | 1.47e+05 | 9.66 | 0.0678 | 9.73 | fail | pass |
| k1_dscf_agn_r07 | dscf | 1 | 100 | 1.47e+05 | 9.4 | 0.0678 | 9.47 | fail | pass |
| k1_dscf_agn_r08 | dscf | 1 | 100 | 1.47e+05 | 12.7 | 0.0678 | 12.7 | fail | pass |
| k1_dscf_agn_r09 | dscf | 1 | 100 | 1.47e+05 | 10.6 | 0.0678 | 10.7 | fail | pass |
| k1_dscf_gal_r00 | dscf | 1 | 100 | 1.06e+05 | 0.492 | 0.0941 | 0.586 | **PASS** | pass |
| k1_dscf_gal_r01 | dscf | 1 | 100 | 1.06e+05 | 0.396 | 0.0941 | 0.49 | **PASS** | pass |
| k1_dscf_gal_r02 | dscf | 1 | 100 | 1.06e+05 | 0.407 | 0.0941 | 0.502 | **PASS** | pass |
| k1_dscf_gal_r03 | dscf | 1 | 100 | 1.06e+05 | 0.42 | 0.0941 | 0.514 | **PASS** | pass |
| k1_dscf_gal_r04 | dscf | 1 | 100 | 1.06e+05 | 0.49 | 0.0941 | 0.584 | **PASS** | pass |
| k1_dscf_gal_r05 | dscf | 1 | 100 | 1.06e+05 | 0.426 | 0.0941 | 0.521 | **PASS** | pass |
| k1_dscf_gal_r06 | dscf | 1 | 100 | 1.06e+05 | 0.414 | 0.0941 | 0.508 | **PASS** | pass |
| k1_dscf_gal_r07 | dscf | 1 | 100 | 1.06e+05 | 0.476 | 0.0941 | 0.57 | **PASS** | pass |
| k1_dscf_gal_r08 | dscf | 1 | 100 | 1.06e+05 | 0.526 | 0.0941 | 0.62 | **PASS** | pass |
| k1_dscf_gal_r09 | dscf | 1 | 100 | 1.06e+05 | 0.676 | 0.0941 | 0.77 | **PASS** | pass |
| k1_dsf_agn_r00 | dsf | 1 | 100 | 1.47e+05 | 8.14 | 0.0678 | 8.21 | fail | pass |
| k1_dsf_agn_r01 | dsf | 1 | 100 | 1.47e+05 | 8.14 | 0.0678 | 8.21 | fail | pass |
| k1_dsf_agn_r02 | dsf | 1 | 100 | 1.47e+05 | 8.44 | 0.0678 | 8.51 | fail | pass |
| k1_dsf_agn_r03 | dsf | 1 | 100 | 1.47e+05 | 8.4 | 0.0678 | 8.47 | fail | pass |
| k1_dsf_agn_r04 | dsf | 1 | 100 | 1.47e+05 | 10.3 | 0.0678 | 10.3 | fail | pass |
| k1_dsf_gal_r00 | dsf | 1 | 100 | 1.06e+05 | 0.492 | 0.0941 | 0.586 | **PASS** | pass |
| k1_dsf_gal_r01 | dsf | 1 | 100 | 1.06e+05 | 0.396 | 0.0941 | 0.49 | **PASS** | pass |
| k1_dsf_gal_r02 | dsf | 1 | 100 | 1.06e+05 | 0.407 | 0.0941 | 0.502 | **PASS** | pass |
| k1_dsf_gal_r03 | dsf | 1 | 100 | 1.06e+05 | 0.42 | 0.0941 | 0.514 | **PASS** | pass |
| k1_dsf_gal_r04 | dsf | 1 | 100 | 1.06e+05 | 0.49 | 0.0941 | 0.584 | **PASS** | pass |
| k2_dscf_fagn0.0 | dscf | 2 | 1000 | 1.08e+05 | 4.66 | 9.22 | 13.9 | fail | pass |
| k2_dscf_fagn0.3 | dscf | 2 | 1000 | 2.02e+05 | 16.3 | 4.96 | 21.3 | fail | pass |
| k2_dscf_fagn0.3_injB | dscf | 2 | 1000 | 1.98e+05 | 16.3 | 5.05 | 21.4 | fail | pass |
| k2_dscf_fagn0.3_isoinj | dscf | 2 | 1000 | 9.28e+03 | 16.3 | 108 | 124 | fail | pass |
| k2_dscf_fagn0.7 | dscf | 2 | 1000 | 2.51e+05 | 43.7 | 3.98 | 47.7 | fail | pass |
| k2_dscf_fagn1.0 | dscf | 2 | 1000 | 1.47e+05 | 93.3 | 6.78 | 100 | fail | pass |
| k2_dsf_n0low_fagn0.0 | dsf | 2 | 1000 | 1.08e+05 | 4.66 | 9.22 | 13.9 | fail | pass |
| k2_dsf_n0low_fagn0.3 | dsf | 2 | 1000 | 2.02e+05 | 16.3 | 4.96 | 21.3 | fail | pass |
| k2_dsf_n0low_fagn0.7 | dsf | 2 | 1000 | 2.51e+05 | 43.7 | 3.98 | 47.7 | fail | pass |
| k2_dsf_n0low_fagn1.0 | dsf | 2 | 1000 | 1.47e+05 | 87.4 | 6.78 | 94.1 | fail | pass |
| k2_dsf_n0true_fagn0.3 | dsf | 2 | 1000 | 3.36e+05 | 7.86 | 2.98 | 10.8 | fail | pass |
| k2_dsf_n25_fagn0.3 | dsf | 2 | 25 | 2.07e+05 | 0.368 | 0.00302 | 0.371 | **PASS** | pass |
| k2_dsf_n25_fagn0.7 | dsf | 2 | 25 | 2.46e+05 | 0.967 | 0.00254 | 0.97 | **PASS** | pass |
| k2_dsf_n30_fagn0.3 | dsf | 2 | 30 | 1.99e+05 | 0.479 | 0.00452 | 0.484 | **PASS** | pass |
| k2_dsf_n35_fagn0.3 | dsf | 2 | 35 | 2.05e+05 | 0.589 | 0.00599 | 0.595 | **PASS** | pass |
| k2_dsf_n40_fagn0.3 | dsf | 2 | 40 | 1.99e+05 | 0.436 | 0.00803 | 0.445 | **PASS** | pass |
| k2_dsf_n45_fagn0.3 | dsf | 2 | 45 | 2.03e+05 | 0.607 | 0.00996 | 0.617 | **PASS** | pass |
| k2_dsf_n50_fagn0.3 | dsf | 2 | 50 | 1.99e+05 | 1.23 | 0.0126 | 1.25 | fail | pass |
| k2_dsf_n50_fagn0.7 | dsf | 2 | 50 | 2.52e+05 | 1.71 | 0.00992 | 1.72 | fail | pass |

**21 of 49 configurations pass the default guard.**

Passing: `k1_dscf_gal_r00`, `k1_dscf_gal_r01`, `k1_dscf_gal_r02`, `k1_dscf_gal_r03`, `k1_dscf_gal_r04`, `k1_dscf_gal_r05`, `k1_dscf_gal_r06`, `k1_dscf_gal_r07`, `k1_dscf_gal_r08`, `k1_dscf_gal_r09`, `k1_dsf_gal_r00`, `k1_dsf_gal_r01`, `k1_dsf_gal_r02`, `k1_dsf_gal_r03`, `k1_dsf_gal_r04`, `k2_dsf_n25_fagn0.3`, `k2_dsf_n25_fagn0.7`, `k2_dsf_n30_fagn0.3`, `k2_dsf_n35_fagn0.3`, `k2_dsf_n40_fagn0.3`, `k2_dsf_n45_fagn0.3`

