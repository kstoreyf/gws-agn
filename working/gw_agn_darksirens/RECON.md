# RECON — interface contract for reproducing `working/gw_agn` with the real darksirens code

2026-07-09. Orchestrator: Fable. Sources: three Explore recon reports + first-hand reads of
`scan_fcat_conditional.py`, `generate_mock_data.py`, `registry.py`. This file is the single
source of truth for the implementation agents. Plan: `~/.claude/plans/unified-discovering-lovelace.md`.

## Repos / environment

- gws-agn repo (this workspace): `/hildafs/projects/phy230014p/magana/gws-agn`
- darksirens: `/hildafs/projects/phy230014p/magana/src/darksirens` — editable-installed into
  the active conda env `jax` (jax 0.4.34 + CUDA). Branch `lensing-stack-to-master`, contains
  PR #195 merge `d387b4f` (K-catalog mixture). **Never switch its branch / never edit it.**
  Every script must record `darksirens.__file__` + `git -C <repo> rev-parse HEAD` in outputs
  and assert `git merge-base --is-ancestor d387b4f HEAD`.
- GPU: login A100 80GB, often saturated → always `XLA_PYTHON_CLIENT_PREALLOCATE=false`;
  SLURM partition MIKO available (`src/darksirens/submit_dsinfer.sh` pattern, account phy230014p).
- Mock machinery importable (standalone, no darksirens import):
  `sys.path.insert(0, "/hildafs/projects/phy230014p/magana/src/darksirens/scripts/mock_dark_sirens")`
  then `import generate_mock_data as gmd`.

## Truth / fiducials (identical between mock and darksirens defaults)

- Cosmology: H0=67.74, Om0=0.3075 (astropy Planck15; = darksirens H0_FID/OM0_FID), w0=-1, wa=0.
- Population `powerlaw+peak` fiducial == `get_fixed_population_params("powerlaw+peak")` ==
  `gmd.PopulationConfig()`: v1=0.10 (w_peak=0.90), alpha=2.3, m_min=5, m_max=80, dm_min=3,
  dm_max=10, mu_G=35, sigma_G=5, beta=1.0, mu_chi=0.0, sigma_chi=0.1, gamma=0.
- Detection rule of the gw_agn mock: hard TRUE-z cut z_max_gw = 1.0. Truth alpha values are
  the eligible-pool fractions: {0.00989, 0.307, 0.703, 1.0}.

## Source data (gw_agn, all under `working/gw_agn/data/glass_prod/`)

- `mock_catalog.h5`: datasets `ra_gal, dec_gal, z_gal` (1,177,289), `ra_agn, dec_agn, z_agn`
  (11,724). **ra/dec in DEGREES**, z to 1.565. attrs: nside=64, seed=101, z_max=1.5.
- `cat_gal_pixelated_nside64.h5`, `cat_agn_pixelated_nside64.h5`: attr `nside=64`; datasets
  `n_in_pixel` (49152,) int64 and `z` (49152, maxgals) f64 **NaN-padded** (maxgals: gal 47, agn 4).
  RING ordering, pix = hp.ang2pix(nside, pi/2 - dec, ra) with ra/dec in RADIANS.
- Event truth sets `gws_fagn{0.0,0.3,0.7,1.0}_lam0.5_seedgw{5000,5001,5002,5003}.h5`:
  datasets `i_gw_gal`, `i_gw_agn` = int indices into the mock_catalog full arrays;
  attrs f_agn, lambda_agn, n_gw=1000, seed_gw, z_max_gw=1.0.
  Coverage: `gws_cov_gal_r00..r24.h5` (seed 6000+, 100 gal-hosted events),
  `gws_cov_agn_r00..r24.h5` (seed 7000+, 100 agn-hosted events).
- PE sample sets `gwsamples_fagn*_lam0.5_seedgw*_dLunc0.1_obs.h5` and
  `gwsamples_cov_{gal,agn}_r*_dLunc0.1_obs.h5`: attrs N_gw, N_gw_agn, N_gw_gal,
  N_samples_gw=2000. Datasets per host type, shape (N_events_type, 2000):
  `dL_gal/dL_agn` [Mpc], `ra_*` [rad, 0..2pi], `dec_*` [rad], `m1det_*`, `m2det_*`
  (masses are N(35,5)-source-frame ×(1+z) — **DO NOT USE; replaced by fiducial powerlaw+peak**).
  Event order within each type matches `i_gw_gal`/`i_gw_agn` order; the combined event list
  is gal-block then agn-block (KEEP this order and record it).
  PE recipe: obs-centered; sky Gaussian sigma=0.01 rad/axis; dL = exact inverse-CDF flat-prior
  posterior of 10% multiplicative Gaussian noise. p_pe = 1 convention.

## darksirens input contract (what we must write under `working/gw_agn_darksirens/data/`)

### Survey files `{gal,agn}.h5` (+ `{gal,agn}_zlt1.h5` truncated A/B variants)
- attr `nside` (=64); datasets `zgals` (npix, maxgals) f64 padded with **100.0**;
  `dzgals` same shape padded **1.0**; `wgals` padded **0.0**; `ngals` (npix,) int.
- Real entries: zgals = catalog z per pixel; dzgals = 3e-3*(1+z) (gw_agn's validated KDE
  bandwidth; scans hold sigma_kde=0); wgals = 1.0.
- Truncated variants: drop galaxies with z > 1.0 (recompute ngals/maxgals).
- Loader to validate against: `darksirens.catalogs.io.load_survey`.

### GW PE files `gw_<set>.h5` (gwcat-1.0), one per event set
- attrs: `format_version="gwcat-1.0"`, `mock_data=True`, `nobs`, `nsamp=2000`,
  `pe_cosmology_H0=67.74`, `pe_cosmology_Om0=0.3075`, `chi_eff_in_p_pe=True`, `chi_eff_amax=0.99`.
- datasets, flat event-major length nobs*nsamp: `dL` [Mpc], `ra` [rad], `dec` [rad],
  `m1det`, `m2det`, `m1src`, `m2src`, `chieff`, `p_pe`.
- dL/ra/dec: copy from gw_agn `gwsamples_*` (concatenate gal block then agn block).
- Masses/chieff per event (seeded, reproducible): draw ONE truth per event from the fiducial
  population — m1src via `gmd._sample_powerlaw_peak_m1`, q via `gmd._sample_q`, m2src=q*m1src,
  chieff via `gmd._sample_chieff`; m_det = m_src*(1+z_true) with z_true from the host catalog
  index. PE clouds per `gmd._posterior_samples` conventions: m1det ~ N(m1det_true, 0.08*m1det_true)
  clipped ≥2; m2det ~ N(m2det_true, 0.10*m2det_true) clipped ≥1; chieff ~ N(chi_true, 0.08)
  clipped [-1,1]; p_pe = 1. m1src/m2src datasets = m1det/(1+z_pe), m2det/(1+z_pe) with
  z_pe = z(dL_sample; H0=67.74, Om0=0.3075) (only used for the chi-eff swap which mock_data
  skips — follow generate_multitracer_mock.py:253-255).
- Do NOT enforce m1det>=m2det per sample after independent scatter? gmd does not re-sort
  (clouds may overlap); follow gmd exactly.
- Loader to validate against: `darksirens.gw.utils.load_gw_samples`.

### Injection file `injections.h5` (gwcat-selection-1.0)
- Proposal: z ~ uniform-in-comoving-volume on [0, 1.2] (grids via
  `gmd._cosmology_grids(gmd._build_cosmology(67.74, 0.3075, -1.0, 0.0), zmax=1.2)`),
  sky uniform (`gmd._sample_sky`), m1src/q/chieff from fiducial population samplers.
  Detection = z <= 1.0 (replaces the SNR cut in `gmd._draw_selection_batch`).
- pdraw: EXACTLY `gmd._selection_pdraw("population", m1src, q, chi, z, grids, pop)` —
  `_mass_spin_pdf(m1src,q,chi,pop) * p_z(z) / ((1+z) * dL'(z)) / (4*pi)` with p_z the
  dvc_dz normalized over the [0,1.2] grid. Keep absolute scale.
- Store DETECTED rows only, datasets `m1det m2det m1src m2src dL chieff ra dec pdraw`
  (`gmd.SELECTION_KEYS`); attrs: `format_version="gwcat-selection-1.0"`, `ndraw`=TOTAL
  proposed, `mock_data=True`, `chi_eff_swap_applied=True`, `chi_eff_amax=0.99`,
  `cosmology_H0=67.74`, `cosmology_Om0=0.3075`, `Neff` (report (Σ1/p)²/Σ(1/p)²).
- Default Ndraw=4e5 (CLI-adjustable; ~2.7e5 detected expected). Seed fixed + recorded.
- Loader to validate against: `darksirens.gw.utils.load_selection_samples`.

## Scan-by-import recipe (proven — copy from `src/darksirens/working/multitracer/scan_fcat_conditional.py`)

```python
from darksirens.inference.data import load_all_data, validate_loaded_survey_shapes
from darksirens.likelihood.factory import make_likelihood
from darksirens.gw.populations import get_fixed_population_params
from darksirens.inference.prior import build_parameter_space
# opts = SimpleNamespace(...) copied VERBATIM from scan_fcat_conditional.py:40-73,
# then adjusted: universe_model, survey_paths (1 or 2), n_catalogs, gw_path, gwselection_path.
# labels = build_parameter_space(...)[0]; coord = physical values in label order;
# ll = float(likelihood(coord)).  The closure is pure likelihood (no prior).
```

- K=2 mixture: `universe_model="dark_sirens"` ONLY (core.py:224 guard). K=1 works for both
  `dark_sirens` and `dark_sirens_complete`.
- Labels (dark_sirens K=2, fix_population=True, fix_de=True, Om0 pinned via
  fixed_parameter_values): [H0, log10n0, delta, b_miss, sigma_kde, log10n0_c2, delta_c2,
  b_miss_c2, sigma_kde_c2, fcat_2]. `fcat_2` = weight of catalog 2 (pass AGN second → fcat_2
  = alpha_AGN). dark_sirens K=1: [H0, log10n0, delta, b_miss, sigma_kde].
  dark_sirens_complete K=1: [H0, sigma_kde].
- Nuisance scan point: log10n0 = log10(N_cat / V_c(z<=1.5)) per catalog (true density),
  delta=0, b_miss=1, sigma_kde=0. (Scan values are NOT prior-bounded — pure likelihood.)
- `selection_neff_guard="auto"`; if a config returns -inf from the Neff guard, rerun that
  config with the soft/off option (check factory for the exact accepted values) and disclose.

## Scan grids (match gw_agn)

- H0: 61 pts, linspace(50, 100) (step 0.8333). alpha/fcat_2: 41 pts, linspace(0, 1)
  (step 0.025). Joint: 61×41. 1-D f scans at H0=67.74; 1-D H0 scans at fcat_2=alpha_true.
- Output per scan: `results/<tag>.h5` (grids, logL array, all provenance attrs: git SHA,
  darksirens.__file__, file paths, nuisance point, timings) + `<tag>.json` summary in the
  gw_agn `summarize_grid.py` style (flat prior, trapezoid marginals, MAP, median, 68/90% CIs,
  truth flags).

## Reproduction targets (compare at CI level, never bitwise)

| set | alpha_true | gw_agn alpha med [68%] | gw_agn H0 med [68%] | MAP |
|---|---|---|---|---|
| fagn0.0 | 0.00989 | 0.01696 [0.00543, 0.03486] | 67.958 [67.083, 68.742] | (68.33, 0.000) |
| fagn0.3 | 0.307 | 0.32951 [0.30493, 0.35265] | 67.526 [66.937, 68.098] | (67.5, 0.325) |
| fagn0.7 | 0.703 | 0.68687 [0.66211, 0.71132] | 67.500 [66.933, 68.067] | (67.5, 0.675) |
| fagn1.0 | 1.0 | 0.97500 [0.95800, 0.99200] | 67.531 [66.943, 68.098] | (67.5, 0.975) |

Fisher @ N=1000 (fagn0.3): sigma_alpha=0.0192, sigma_H0=0.345, rho=-0.0021.
Coverage (N=100/realization): GAL <H0med>=67.311±0.247, mean 68% half-width 1.43;
AGN <H0med>=67.680±0.204, half-width 0.72.

## Known convention differences (disclose; do NOT silently "fix")

1. Selection: gw_agn beta = CDF_cat(1.0) H0-independent; darksirens injection-based mu(H0)
   = dL-machine view. Coincide at truth; widths may differ.
2. Numerator z-truncation at 1.0 (gw_agn) vs none (darksirens). A/B lever: `*_zlt1.h5` cats.
3. Pixel weighting: complete = per-pixel-normalized p_cat(z|pix), empty-pixel policy zero;
   dark_sirens = N_obs-weighted + missing floor; gw_agn = count-weighted KDE mixture.
4. Masses now informative (fixed powerlaw+peak couples m1det to z) — small spectral-siren
   H0 information absent from gw_agn.
