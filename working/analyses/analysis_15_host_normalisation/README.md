# Analysis 15 — the host-catalog normalisation volume (follow-up to LIKELIHOOD_MATH §8)

**Status (2026-11-17, 05:48 UTC):** scripts written and validated on CPU (`results/validation_cpu.json`,
`results/identity_check_cpu_*.json`, `logs/dryrun*.log`). GPU job RITA-GPU **1414681** (`agn-follow`,
one a100-80, 3 h limit, expected 30–50 min) queued. REPORT.md will be written once the results exist.

## Question

In the two-catalog likelihood the assumed host densities n0_k enter only through each catalog's
normaliser, Z_k = N_obs,k + n0_k 4π ∫_0^zmax (1 − C_k) dV_c, with zmax the end of darksirens'
internal redshift grid (5). Wherever no PE sample or injection lies where a catalog is incomplete,
this gives the exact identity

    ln L(H0, f; n0) = ln L(H0, f̃; n0 → 0),   logit f̃ = logit f + ln(F_AGN / F_GAL),   F_k = N_obs,k / Z_k

(`paper_prd/notes/LIKELIHOOD_MATH.md` §8). This analysis checks it with likelihood evaluations on
darksirens-core:

1. **Part 1 (gate).** Core reproduces the stored legacy flux-limited results (Analysis 3, seed
   100, `c_mode=selection`, darksirens 0c5b3db): complete, m < 21 and m < 18, input densities
   (log10 n0 = −3 GAL, −5 AGN), grid to 5. Gate as in Analysis 14: median within 0.2 of the
   68% half-width, 68% and 90% width ratios within 0.9–1.1.
2. **Part 2 (normalisation volume).**
   - (a) complete, grid to 5, input densities: this is Part 1's complete scan (stored legacy f_AGN 0.240,
     zero density 0.273).
   - (b) complete, normalisation restricted to the catalog depth, AGN density ×1 and ×2. Prediction:
     F_k = 1, so both equal the zero-density posterior.
   - (c) complete, grid to 5, AGN density ×2. Prediction: logit f shifts by ≈ (1 − F_AGN) ln 2
     relative to (a).
   - (d) m < 18, restricted normalisation, AGN density ×0.5, ×1, ×2. Prediction: the slope of f_AGN is
     ln10 (1 − F_AGN) f(1 − f) for that volume, plus the part from events lying where the catalog
     is incomplete.

Seed 100, mixed `events.h5` (input f_AGN 0.30; it is clean with respect to core's m2 < 5 cut,
Analysis 14 README), `injections_targeted.h5`. Every scan: (H0, f_AGN) on the stored cells, H0 64–74
(step 0.25) × f (step 0.025) over a range wide enough for that scan's posterior.

## How the normalisation volume is restricted

The two routes named in LIKELIHOOD_MATH §8.4 do different things in core e7c3007:

- **A survey `z_depth`** (`catalog/io.py:67-71`, read from the survey file) does *not* restrict
  the normaliser. It sets C := 0 above the depth (`selection/catalog.py:353-355`,
  `catalog/mixture.py:174-175`), so the missing-host term becomes the full n0 dV_c from the depth
  to z = 5, in Z_k and in the numerator, and truncates each galaxy's kernel at the depth
  (`catalog/redshift.py:1373-1374`). On the complete catalogs it would give F_GAL ≈ F_AGN ≈ 0.08,
  not 1, and would put modelled hosts under the injections that lie above the depth at H0 > 68.
- **`DARKSIRENS_ZMAX`** ends the whole redshift grid (`cosmology/_grid.py:14`), the distance table
  (`cosmology/distances.py:28`) and the kernel normalisation (`catalog/redshift.py:49`) at the
  value set. Z_k is then N_obs,k + n0_k 4π ∫_0^zmax (1 − C_k) dV_c. A sample or injection beyond
  the grid end would be cut out of the likelihood.

The detected injections lie at z ≤ 0.953 at H0 = 64 and z ≤ **1.0716** at H0 = 74 (one injection
above 1.014). The PE samples lie at z ≤ 0.508 at H0 = 74. So the grid end cannot be the catalog
edge (1.014) without truncating the selection integral above H0 ≈ 68. It is set to
**zmax = 1.10**: above every injection and PE sample at every H0 of the grid, and below the
redshift where the complete catalogs' C first drops below 1 (1.16 for GAL, 1.69 for AGN). So:

| configuration | Z_k in the run |
|---|---|
| complete, grid to 1.10 | **Z_k = N_obs,k exactly** (C_k = 1 on 0 < z < 1.10, so the missing term is identically 0 at any n0): 151,179,870 GAL and 1,514,567 AGN, F_k = 1; the same as normalising at the catalog edge 1.014 |
| m < 18, grid to 1.10 | N_obs,k + n0_k 4π ∫_0^1.10 (1 − C_k) dV_c: at H0 = 67.74 and input densities 2.022×10⁸ (GAL) and 2.022×10⁶ (AGN), F = 0.00406 and 0.00409 (to the edge 1.014 instead: 16% less missing volume, F = 0.00484 and 0.00488; 1 − F_AGN changes by 8×10⁻⁴) |
| any rung, grid to 5 | N_obs,k + n0_k 4π ∫_0^5 (1 − C_k) dV_c, as the stored runs (LIKELIHOOD_MATH §4) |

The survey, event and injection files are used as they are; nothing on disk is modified or copied.
Every scan file stores core's own Z_k at each H0 row (the likelihood's `_observed` +
`_missing_total`), next to the closed form.

## CPU validation (before the job)

`results/validation_cpu.json` (`scripts/validate_cpu.py`), `results/identity_check_cpu_*.json`
(`scripts/identity_check_cpu.py`), dry-run logs `logs/dryrun_grid{5,R}.log`:

- **Where the data lie.** Detected injections: z ≤ 0.9526 at H0 = 64, 0.9975 at 67.74, 1.0716 at 74
  (one injection above 1.014, none above 1.05 or 1.10); PE samples: z ≤ 0.450 at 64, 0.508 at 74.
  Nothing lies above the restricted grid end 1.10 anywhere on H0 64–74 (at H0 = 100 there would be 9).
- **Completeness curves.** C first drops below 1 at z = 1.162 (complete GAL, m_lim 24), 1.689
  (complete AGN, m_lim 25), 0.372/0.371 (m < 21), 0.108/0.108 (m < 18): the notes' 1.16, 1.69, 0.37,
  0.11. C = 1 exactly at 1.014 and 1.10 for both complete fits.
- **Z_k, F_k at H0 = 67.74.** Closed form, grid to 5, input densities: complete 1.9423×10⁹ / 1.6283×10⁷,
  F = 0.077836 / 0.093018 (ln F_A/F_G = +0.1782); m < 21 F = 0.011415 / 0.011459; m < 18 F = 0.00038743 /
  0.00039024. All within the rounding of the notes' table (≤ 0.7%). Restricted grid: complete Z_k =
  N_obs (F = 1) at any density; m < 18 F = 0.0040625 / 0.0040920 (×1), AGN 0.0081506 (×0.5), 0.0020502 (×2).
  Grid to 5 with AGN ×2: F_AGN = 0.048777 (ln F_A/F_G = −0.4673).
- **Core's own Z_k** (the likelihood's `_observed` + `_missing_total`, evaluated in the dry run at
  every H0 row) equals the closed form to ≤ 1.7×10⁻⁶ relative (core's distance table against a fine
  quadrature); exactly N_obs on the restricted complete grid.
- **The identity in core, cell by cell** (20 events, 20,000 injections, H0 = 64, 67.75, 74,
  f = 0.15, 0.30, 0.45): complete, grid to 5, AGN density ×0.5, ×1, ×2 against zero density at the mapped
  f′: largest difference **1.7×10⁻¹³ nats** (without the f remap: up to 0.59). m < 18, grid to 1.10,
  ×0.5 and ×2 against ×1 at the mapped f′: differences up to 0.34 nats on 20 events, the in-data
  missing-host part.
- **Dry run of the driver** on every scan (20 events, 20,000 injections, cells at H0 = 64 and 74 and
  at the lowest f of each scan, f = 0 for AGN ×0.5): the diagnostics program equals the bound call;
  every ln L finite; restricted complete ×1 and ×2 identical (0.0 nats).

## Identity predictions (aggregation)

For each scan s with a reference grid r that differs from it only in Z_k, the prediction is
ln L_s(H0, f) = ln L_r(H0, f′) with logit f′ = logit f + ln(F_A,s/F_G,s) − ln(F_A,r/F_G,r), row by row
in H0 (F_k depends on H0 through n0 V_c ∝ H0⁻³), by a cubic spline of the reference ln L in logit f.
References: the zero-density complete scan (`ref_complete_n0m24`, grid to 5) for (a), (b), (c); the
grid-to-5 m < 18 scan (`p1_m18`) for (d) ×1 (same numerator, different volume: exact); the (d) ×1
scan for (d) ×0.5 and ×2 (normaliser part only; the remainder is the in-data missing hosts). The
summary records, per scan, the predicted and measured medians and intervals, and the mean, rms and
largest difference in ln L over the bulk (cells within 1e-3 of the measured peak).

## Layout

| path | contents |
|---|---|
| `scripts/a15_common.py` | scan table, paths, grids, posterior summaries, closed-form C_k and Z_k |
| `scripts/a15_host_norm.py` | the GPU driver (core e7c3007 venv); `--scans`, `--dry-run` |
| `scripts/submit_a15_gpu.sbatch` | the job: four driver blocks (grid 5, 1.10, 5, 1.10), then aggregation and figure |
| `scripts/validate_cpu.py` | CPU validation → `results/validation_cpu.json` (data extents, C_k, Z_k vs the notes table, dry-run Z_k from core) |
| `scripts/identity_check_cpu.py` | the identity cell by cell in core on a subset → `results/identity_check_cpu_{complete_grid5,m18_grid110}.json` |
| `scripts/aggregate_normalisation.py` | → `results/normalisation_summary.json` (per scan: configuration, Z_k/F_k, measured, prediction, differences; gate table; checks) |
| `scripts/make_figures.py` | → `figs/normalisation_fagn.{pdf,png}` |
| `results/scans/<scan>.{h5,json}` | per scan: grid, ln L, PE variance sum, N_eff, ln μ, selection term, normaliser per H0 row; summary |
| `results/gate_<scan>.json` | Part 1 (and reference) comparison with the stored legacy grid |
| `logs/` | job and dry-run logs (git-ignored) |

Scan names: `p1_complete`, `p1_m21`, `p1_m18` (Part 1); `b_complete_x1`, `b_complete_x2`;
`ref_complete_n0m24` (zero density, grid to 5; also compared with the stored
`fu_probes/joint_complete_n0m24_s100.h5`); `c_complete_x2`; `d_m18_x1`, `d_m18_x05`, `d_m18_x2`.
f ranges (step 0.025; H0 64–74 step 0.25 for all): 0.05–0.55 for Part 1 and (b); 0.025–0.75 for the
reference; 0.05–0.75 for (c); 0.025–0.65, 0–0.45 and 0.10–0.80 for (d) ×1, ×0.5, ×2. 9,758 cells in all
(4,141 on the complete catalogs).

## How to resume

Resubmit identically; finished scans are skipped, gates on record are reused:

    cd working/analyses/analysis_15_host_normalisation
    sbatch scripts/submit_a15_gpu.sbatch

Aggregation and figure (CPU, seconds):

    PY=/hildafs/home/magana/tmp_ondemand_hildafs_phy230014p_symlink/magana/.conda/envs/jax/bin/python
    $PY scripts/aggregate_normalisation.py && $PY scripts/make_figures.py

## Runs

| job | what |
|---|---|
| CPU dry run (miko) | every scan at 20 events, 20,000 injections, 2 × 2 cells, both grids: `logs/dryrun_grid5.log`, `logs/dryrun_gridR.log`, `logs/dryrun/` (the `_mid` logs are a first pass at middle cells) |
| CPU identity check (miko) | `logs/identity_check_*.log` → `results/identity_check_cpu_*.json` (the `_vs_zero_density` m < 18 log records that the zero-density reference does not apply there) |
| RITA-GPU 1414681 `agn-follow` | all ten scans, four blocks, then aggregation and figure; queued 05:48 UTC |
