# Analysis 15: what the host normalization does to f_AGN and H0

17 Nov 2026. Rita job 1414681 (35 min on one A100), darksirens-core e7c3007, realization 100,
equal-population events as stored (2000 posterior samples per event), (H0, f_AGN) grids.
Numbers: `results/normalisation_summary.json`; figure: `figs/normalisation_fagn.{pdf,png}`.

## Question

In the two-catalog likelihood each class's host density is normalized by
Z_k = N_obs,k + n_k 4π ∫ (1 − C_k) dV_c, the catalogued hosts plus the hosts the catalog misses out
to the end of the redshift grid. `paper_prd/notes/LIKELIHOOD_MATH.md` Sec. 8 shows that only the
catalog share F_k = N_obs,k / Z_k enters, through logit f̃ = logit f + ln(F_AGN / F_gal): the mergers
constrain the merger rate per host of one class relative to the other, and f_AGN follows from it
once the number of hosts of each class is fixed. All earlier analyses normalized to z = 5, far
beyond any detected merger. This analysis tests the identity on core and asks what changes when the
hosts are normalized over the volume the data occupy (grid ending at z = 1.10; no detected
injection and no posterior sample lies above it for H0 between 64 and 74).

## Results (median and 90% interval; inputs H0 = 67.74, f_AGN = 0.30)

| Catalogs | Normalized to | Assumed AGN density | F_gal | F_AGN | H0 | f_AGN | f_AGN from the identity |
|---|---|---|---|---|---|---|---|
| complete | no missing hosts (reference) | none | 1 | 1 | 69.26 [67.65, 70.87] | 0.275 [0.196, 0.357] | |
| complete | z = 5 | input | 0.078 | 0.093 | 69.26 [67.66, 70.87] | 0.241 [0.169, 0.319] | 0.241 |
| complete | z = 5 | 2 × input | 0.078 | 0.049 | 69.26 [67.65, 70.87] | 0.373 [0.278, 0.466] | 0.377 |
| complete | z = 1.10 | input | 1 | 1 | 69.26 [67.65, 70.87] | 0.275 [0.196, 0.357] | 0.275 |
| complete | z = 1.10 | 2 × input | 1 | 1 | 69.26 [67.65, 70.87] | 0.275 [0.196, 0.357] | 0.275 |
| m < 21 | z = 5 | input | 0.0114 | 0.0115 | 69.26 [67.66, 70.87] | 0.274 [0.196, 0.356] | |
| m < 18 | z = 5 | input | 0.0004 | 0.0004 | 68.76 [66.98, 70.40] | 0.288 [0.204, 0.374] | |
| m < 18 | z = 1.10 | input | 0.0041 | 0.0041 | 68.76 [66.98, 70.40] | 0.288 [0.204, 0.374] | 0.288 |
| m < 18 | z = 1.10 | 0.5 × input | 0.0041 | 0.0082 | 67.94 [66.05, 69.70] | 0.143 [0.091, 0.198] | 0.169 |
| m < 18 | z = 1.10 | 2 × input | 0.0041 | 0.0021 | 69.80 [68.06, 71.52] | 0.406 [0.305, 0.502] | 0.447 |

1. **The identity is exact wherever the missing hosts lie outside the data.** For complete catalogs
   the three z = 5 rows follow from the reference row with no free quantity (f_AGN medians agree to
   the fourth decimal, H0 unchanged). The offset of the old "AGN = complete" cells (f_AGN 0.241
   against 0.275) is the different nominal depth of the two catalogs, nothing else. This closes
   gap A5.
2. **With complete catalogs normalized over the volume of the data, no number density enters.**
   Doubling the assumed AGN density changes ln L by exactly zero at every grid cell.
3. **A flux-limited catalog holds a small share of the hosts whichever volume is used.** At m < 18
   the share is 0.0004 to z = 5 and 0.0041 to z = 1.10, so shortening the grid does not remove the
   dependence on the assumed densities. At the input densities the two volumes give the same
   posterior, because both classes have the same share.
4. **At m < 18 a wrong AGN density moves f_AGN and H0.** Half and twice the input density give
   f_AGN 0.143 and 0.406 (0.44 per factor of ten, against 0.45 from the identity and 0.47 from its
   small-share limit), and H0 67.94 and 69.80, a shift of −0.8 and +1.1 km/s/Mpc against a 68%
   half-width of 1.0. The H0 shift is not in the identity: at this depth half of the detected
   mergers lie where the catalog is incomplete, and the assumed density sets how much of each such
   merger is given to the uniform background of missed AGN. The identity then describes f_AGN to
   0.03 to 0.04 and misses the H0 shift.

Gates: the three z = 5 scans at the input densities reproduce the stored Analysis 3 posteriors
(median shifts below 0.05 of the 68% half-width, widths within 0.6%); core's Z_k agrees with the
closed form to 2 × 10⁻⁶.

## What this means for the paper and the production runs

- The statement of the paper is the identity: the mergers measure the merger rate per host, AGN
  against galaxies. f_AGN is that ratio times the ratio of host numbers, so its accuracy is the
  accuracy of the number densities.
- Production convention: grid to z = 1.10, and each class's number density measured from its own
  catalog in the volume where the catalog is complete. With that, a complete catalog needs no
  density at all, and a flux-limited one uses a density that the catalog itself fixes to a few
  percent (about 4500 AGN inside z = 0.11 at the input density), which moves f_AGN by about 0.01.
- The sentence "the depth of the catalogs does not limit the measurement" of the first draft is
  withdrawn. What can be said: at the input densities the posterior hardly changes down to m < 18;
  an error of a factor of two in the AGN density moves f_AGN by 0.13 and, where half the mergers
  lie beyond the complete part of the catalog, H0 by about 1 km/s/Mpc.
- All scans here use 2000 posterior samples per event without the half-variance correction
  (Analysis 14); the comparisons are differential on the same samples, so they are not affected,
  but the absolute f_AGN values are low by about 0.01.

## Files

`scripts/a15_host_norm.py` (scans), `a15_common.py`, `aggregate_normalisation.py`,
`make_figures.py`, `submit_a15_gpu.sbatch`; `results/scans/*.json` (per scan),
`results/gate_*.json`, `results/normalisation_summary.json`; CPU checks
`results/validation_cpu.json`, `results/identity_check_cpu_*.json`.
