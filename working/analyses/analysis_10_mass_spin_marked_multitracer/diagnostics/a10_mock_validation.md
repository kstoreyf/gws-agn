# Analysis 10 — mock validation record

`/hildafs/projects/phy230014p/magana/gws-agn/working/data/seed100/events/events_marked_dmu0p10_dmuG5.h5`

Twelve registered checks, 12/12 PASS. Numbers are read from `diagnostics/a10_mock_validation.json`, which `scripts/a10_validate_mock.py` writes; nothing here is retyped.

| # | check | the number that decides it | verdict |
|---|---|---|---|
| 1 | one realisation, seed 100 | `seed_events` = 100003 = SEED*1000+3; `nobs` = 1000, `nsamp` = 2000; record defaults True | **PASS** |
| 2 | same LSS and survey catalogs | 10 catalog + 10 survey files, 0 md5 changes; complete catalogs 151,179,870 GAL / 1,514,567 AGN rows, matching the events log | **PASS** |
| 3 | GAL branch generated from mu_G = 35 | `mu_G_gal` = 35.0; detected GAL median m1src = 36.813 (16/84 31.547/41.804) Msun, n = 643; sampler KS p = 0.942, chi2/dof = 1.043 | **PASS** |
| 4 | AGN branch generated from mu_G = 40 | `mu_G_agn` = 40.0, `dmu_G_agn` = 5.0; detected AGN median m1src = 41.557 (16/84 36.541/45.971) Msun, n = 357; sampler KS p = 0.942, chi2/dof = 0.849 | **PASS** |
| 5 | both branches share peak fraction 0.90 | `peak_fraction` = 0.9 in pop, pop_agn, `branch_mass` and `population` | **PASS** |
| 6 | shared PL slope, mass limits and tapers | 11/11 shared dataclass fields equal; peak_mu 35.0 -> 40.0 | **PASS** |
| 7 | shared q distribution | beta = 1.0 in both; detected-set q KS D = 0.0467, p = 0.6750 (descriptive) | **PASS** |
| 8 | spin difference +0.10 | `dmu_chi_agn` = 0.1; detected-set realised +0.099502 +/- 0.006690 | **PASS** |
| 9 | v3 PE unchanged | pe_model = v3, a_mc = 0.08, a_q = 0.6, a_chi = 0.2, sigma_rho = 1.0, sky_a_deg = 35.0: 0 keys differ from the A8 mock | **PASS** |
| 10 | observed-SNR detection unchanged | detection_rule = observed-data, snr_threshold = 8.0, snr_ref = 6.278363879917771: 0 keys differ; git diff +53/-0 over 6 hunks; 14 untouched functions byte-identical to HEAD | **PASS** |
| 11 | analyses 0-9 files unchanged | 24/24 files carry their pre-edit md5; 0 changed | **PASS** |
| 12 | control behaves | patched vs pristine HEAD on r008: 50/50 datasets bitwise identical, only the 3 new attrs added; production vs control: 48/50 datasets move | **PASS** |

## Planted against realised

| quantity | planted | realised (detected set) |
|---|---|---|
| `f_AGN` | 0.30 | 0.357 (357/1000) |
| Δμ_χ | +0.10 | +0.099502 ± 0.006690 |
| Δμ_G [M⊙] | +5.00 | median m1src shift +4.745 (a different quantity — see below) |

The mass mark is a hyperparameter of the **draw**: the AGN branch's Gaussian peak sits at 40 M⊙ against the GAL branch's 35. The detected-set median shift is not that number and must not be quoted as it — the shared 10% power-law component, the peak truncation and the SNR-weighted selection all damp it.

## Detected-set branch descriptives

| branch | n | median m1src | 16/84 | mean χ_eff | median q |
|---|---|---|---|---|---|
| GAL (μ_G = 35) | 643 | 36.813 | 31.547 / 41.804 | -0.003720 ± 0.003927 | 0.8084 |
| AGN (μ_G = 40) | 357 | 41.557 | 36.541 / 45.971 | +0.095783 ± 0.005416 | 0.7986 |

## Yield, against the Analysis-8 marked mock

| quantity | this mock | A8 marked mock |
|---|---|---|
| proposals for 1000 events | 200,000 | 200,000 |
| detected fraction | 8.460000e-03 | 7.850000e-03 |
| detected fraction, SNR only | 9.540000e-03 | 8.925000e-03 |
| horizon z_max | 0.320045 | 0.310533 |
| realised f_AGN | 0.357 | 0.295 |

## Provenance

- gws-agn HEAD `7ab5412b795e5adcda99c66987d94c82dc59670b`; `generate_dataset.py` diff +53/-0, sha256 `5a7f190f5b036639…`
- darksirens `af896cae6f3f3dd1f87dec50046e3a8228f59b39` in `/hildafs/projects/phy230014p/magana/src/darksirens-a8`, worktree clean = True; the product records `darksirens_sha` = `af896cae6f3f3dd1f87dec50046e3a8228f59b39`
- generated on r008.opa.vera.psc.edu (SLURM RM, CPU only); numpy 1.26.4, h5py 3.12.1
- AGN-branch child RNG stream: `np.random.default_rng(np.random.SeedSequence(_SEED_EV, spawn_key=(0x4D47,)))`

## The three runs behind this record

| role | SLURM job | host | wall | events stage |
|---|---|---|---|---|
| bitwise control (patched generator at the default) | 1335487 | r008 | 44 s | 6.9 s |
| pristine reference (HEAD generator, same node) | 1335488 | r008 | 23 s | 6.0 s |
| production mock | 1335489 | r008 | 24 s | 7.1 s |

The sampler check draws the same 2,000,000 primaries under both populations from one seed, so the two lanes carry the identical peak-lane count (1,800,055) and the identical dispersion (4.998591 M⊙); the mark is a pure location shift, 34.9987 -> 39.9987 M⊙.
