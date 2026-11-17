# Analysis 16 — production runs (driver, run list, submission, tests)

**Status (2026-11-17, see "Tests" for the job ids):** driver, run list, launcher and summary written;
CPU checks passed (no-catalog identity, interruption/resume bit for bit, dry runs of every
configuration class, every run of the list resolved). Three GPU test jobs (`agn-prodtest`) queued
behind the running Analyses 14/15 jobs. **No production run has been submitted.**
REPORT.md comes with the production results.

## What it runs

The approved production stages of `paper_prd/runs.txt` ("REVISION OF 17 NOV", "APPROVED BY THE
OWNER") on darksirens-core e7c3007, with the production convention:

- events drawn from core's population (`events*_corepop.h5`, `data/corepop/README.md`);
- darksirens' redshift grid ends at **z = 1.10** (`DARKSIRENS_ZMAX`, set by the driver before the
  import and checked after it); complete catalogs then have Z_k = N_obs,k (core's `completeness =
  "complete"`: no missing hosts, no density), Analysis 15's normalisation;
- flux-limited catalogs (m < 21, 20, 19, 18): `completeness = "selection"` with the Schechter fit of
  that limit (`selection_redo/scripts/selection_fits_truez/`, seed-100 fits; the luminosity function and
  limits are generator constants), `field_normalizer = "moments"`, densities in h^3 Mpc^-3
  (`n0_units = "h_scaled"`), each class's density **from its catalog counts** (below), given, or free;
- 8,000 posterior samples per event, redrawn from each event's recorded measurement through the
  generator's own functions (Analysis 14's `mc_redraw`), stream `SeedSequence([16100 2026,
  realization, kind code, family code, redraw index, block])`, recorded in every run;
- the half-variance correction of ln L **on** (Analysis 14: ln L + Σσ_i²/2 − N/(2 N_eff)); every grid
  cell and every dead point stores the uncorrected ln L, Σσ_i², N_eff, ln μ and the selection term,
  so the uncorrected posterior is a reweighting (done in the products);
- the variance guard `max_likelihood_variance = 20`;
- catalog evaluation "historical" (Analyses 13–15), set explicitly; every population value sampled or
  fixed at an explicit input (`prod_common.POPULATION_INPUTS`); Om0 0.3075; δ = σ_kde = 0.

Priors, one range per parameter in every run: H0 U[50, 100]; f_AGN U[0, 1]; μ_G U[20, 50];
σ_G U[1, 10]; μ_χ U[−0.2, 0.3]; σ_χ U[0.01, 1]; Δμ_G U[−10, 10]; Δμ_χ U[−0.05, 0.30];
log10 n_gal U[−4, −1]; log10 n_AGN U[−6, −4] (h^3 Mpc^-3; the inputs are −2.49 and −4.49).

Analyses (directory and run names): `known_population` (H0, f_AGN; H0 alone with one catalog or
none), `single_population` (+ μ_G, σ_G, μ_χ, σ_χ), `host_dependent_populations` (+ Δμ_G, Δμ_χ for the
AGN class, passed to core as absolute copies μ + Δμ), `free_densities` (H0, f_AGN, log10 n_gal,
log10 n_AGN). Run names: `<analysis>/<both_catalogs|galaxy_catalog|agn_catalog|no_catalog>/
<complete|flux_limit_NN[_densities_from_counts[_agn_density_half|_double]]>/<equal_populations|
agn_hosted|galaxy_hosted|host_dependent>[_<family>]/realization_NNN`.

### No catalog

`--catalogs none` builds `ds.model(cosmology, population)` with no catalog: core's catalog-free
(spectral-siren) likelihood with p(z) ∝ dV_c/dz normalised on the redshift grid, the population
carrying (1+z)^(γ−1) as in the catalog branches. That is hosts uniform in comoving volume.
`tests/test_nocatalog_identity.py` recomputes it as a direct sum over the posterior samples and
injections (50 events, 300k injections, seven points over H0 52–95 and the population):
with core's tabulated volume element the two agree to **≤ 1.4e-13 nats**; with an independent numpy
cosmology to ≤ 2.6e-4 nats per 50 events, all of it core's 1000-node interpolation of ln dV_c/dz
(median 5e-6, largest 1.4e-3 at z = 0.01) (`tests/results/nocatalog_identity.json`).

### Number density from catalog counts

`prod_common.density_from_counts`: n = N(z < z_c) / V_c(z < z_c), V_c = (4π/3) χ(z_c)³ over the full
sky, counted on the survey file's own (photometric) redshifts, quoted in h^3 Mpc^-3 (V_c at H0 =
100), so the density used at any H0, n (H0/100)³, is N/V_c(z_c; H0): the counts fix the density in
redshift space at every H0. z_c is the redshift to which the catalog is complete under its
Schechter selection model: C(z) = 1 exactly while DM(z; H0 = 100) ≤ m_lim − M̂* − ΔM_faint (C is
h-scaled, so z_c does not depend on H0). z_c = 0.372 / 0.250 / 0.166 / 0.108 for m < 21/20/19/18 (both
classes). Every run records z_c, N(<z_c) and the density used. For the five realizations the
densities at H0 = 67.74 are within 0.1% (m < 21) to 2.5% (m < 18) of the inputs 1e-3 and 1e-5
(`tests/results/runlist_resolve.json`). The m < 18 galaxy deficit (−2%, every realization) is the
mock: its galaxy density is 15% below the input within z < 0.054 and reaches it by z ≈ 0.3.

## Layout

| path | contents |
|---|---|
| `scripts/prod_common.py` | priors, population inputs, event kinds, files, analysis classes, run names, densities from counts, posterior summaries (no jax) |
| `scripts/prod_model.py` | the core model of one run: settings, build, redraw, bind, the diagnostics program, coordinates → core labels |
| `scripts/prod_driver.py` | the driver: `--mode grid|nested`, every option explicit (`--help`); resumable |
| `scripts/make_runlist.py` | → `runs/RUNLIST.yaml`: stages 1–4, each run's arguments, parts, estimated calls and GPU hours |
| `scripts/launch.py` | `status`, `stage N`, `run NAME`: queue tasks on the two GPU lanes, start lane jobs |
| `scripts/lane_worker.py`, `scripts/submit_prod.sbatch` | the lane job (one GPU): runs tasks until its wall time, submits its own successor |
| `scripts/prod_summary.py` | finished run → `runs/<run>/summary.json` (medians, 90% intervals, ln Z, variance, calls, GPU hours) |
| `runs/<run>/` | `config.json`; `grid/` (coarse.h5, fine.h5, result.json) or `nested_1/`, `nested_2/` (state.save, results.pkl, samples.npz, result.json); `sessions.json`, `attempts.json` per part; `summary.json` |
| `runs/_lanes/lane{1,2}.txt` | the lane queues (written by `launch.py`) |
| `tests/` | CPU checks, the three GPU test jobs and their comparison scripts; `tests/results/*.json` |
| `logs/` | job logs, dry-run logs (git-ignored) |

## The driver

    PY=/hildafs/projects/phy230014p/magana/envs/darksirens-core-e7c3007/bin/python
    $PY scripts/prod_driver.py --mode grid --analysis known_population --catalogs both --flux complete \
        --events-kind equal_populations --realization 100
    $PY scripts/prod_driver.py --mode nested --analysis single_population --catalogs both \
        --events-kind host_dependent --realization 101 --nested-index 2

Options (defaults are the production convention): `--catalogs both|galaxy|agn|none`, `--flux
complete|21|20|19|18`, `--density counts|given|free` (`--log10n-gal/--log10n-agn` for given,
`--agn-density-factor`, `--density-units h_scaled|physical`), `--events-kind`, `--events-suffix`
(another mock family; `--family-name` for its directory, `--population-family` and `--pop NAME=VALUE`
for its fixed values), `--nsamp` (0 = the file's own samples), `--redraw-index`, `--half-variance-correction
on|off`, `--max-var`, `--zmax`, `--free`, `--prior NAME=LO,HI` (tests only). Grid: a coarse pass over the
whole prior (H0 step 2 × f step 0.1; H0 step 1 in 1-D), then a fine grid of 41 × 21 (81 in 1-D) cells
centred on it, ±5 sd; the mass outside the fine grid is computed from the coarse pass and must be
< 1e-4 (else the fine grid is widened once); `--grid NAME=LO,HI,STEP` sets it explicitly. Nested:
dynesty, nlive 200, dlogz 0.1, bound multi, first bound after 400 calls, unif up to six parameters
(two independent runs, `--nested-index 1|2`, merged by the summary), rslice with slices 11, enlarge 1.25,
no bootstrap for eight (Analysis 13's guarded run; see test c); checkpoint every 10 min and at the end of
every 2000-call chunk; resumes from `nested_<k>/state.save`; stops cleanly 5 min before the job's
deadline (exit 75). The run's JSON records the core commit (checked against the worktree HEAD), every
setting, the Slurm job id and wall time of every session.

## Launching

    PY=/hildafs/home/magana/tmp_ondemand_hildafs_phy230014p_symlink/magana/.conda/envs/jax/bin/python
    $PY scripts/launch.py status [-v] [--stage N]
    $PY scripts/launch.py stage 1 [--dry-run]       # then stage 2, 3, 4 (or several at once)
    $PY scripts/launch.py run <run name> [--part nested_1] [--hours 6]
    $PY scripts/prod_summary.py --stage 1           # after the runs finish

`stage N` adds the stage's unfinished tasks with existing data to the two lane files (balanced by
estimated GPU hours; each file sorted by stage and cost, so cheap stages run first) and submits a lane
job (`agn-prod-l1`, `agn-prod-l2`, 48 h, one A100 each) where none is queued. A lane job submits its
successor (afterany) at start while work remains, re-reads its lane before every task, and skips tasks
that are finished, failed three times, locked by another live job, or lack their events file. Rerun
`stage N` when the five new realizations exist; regenerate the run list first if their seeds are not
106–110 (`make_runlist.py --realizations-10 ...`).

## Run list and cost

`runs/RUNLIST.yaml` (from `make_runlist.py`; the cost model is in its header and the file):

| stage | runs | tasks | est. calls | est. GPU-h | of which on existing data |
|---|---|---|---|---|---|
| 1 population known, complete catalogs (grids) | 35 | 35 | 14,770 | 3.3 | 2.5 |
| 2 flux-limited catalogs (grids; free densities nested) | 44 | 56 | 396,704 | 45.8 | 45.8 |
| 3 equal-population events, single population | 20 | 40 | 1,300,000 | 160.9 | 80.4 |
| 4 host-dependent events | 20 | 35 | 2,370,000 | 291.4 | 246.9 |

## Tests

CPU (done): `tests/test_nocatalog_identity.py` (above); `tests/compare_resume.py` on a CPU run killed
after 170 calls and resumed (75 iterations, through the first bound, bit for bit equal to the
uninterrupted run); `tests/cpu_dryruns.sh` (every configuration class builds, binds, the diagnostics
program equals the bound call, products written); `tests/test_runlist_resolve.py` (166 tasks);
`tests/cpu_sampler_pipeline.sh` (test c's mechanics).

GPU (`agn-prodtest`, nice 50, 4.0 GPU-h in all by their time limits), submitted 2026-11-17 ~09:00 UTC:

| job | script | writes | pass criteria |
|---|---|---|---|
| 1414826 (45 min) | `tests/submit_test_closure.sbatch` | `tests/closure_runs/{a14_convention,production_a14_redraw,a15_convention,production_stored}/grid/`, `tests/results/closure.json`, `spc_fixed` in `tests/results/timing.json` | `a14_convention` vs A14 `b_pe_n8000_r00` and `a15_convention` vs A15 `b_complete_x1`: max \|Δ ln L\| ≤ 1e-6 and medians, 68%/90% bounds equal to 1e-4 (H0) / 1e-5 (f_AGN); `production_stored` vs `a15_convention` max \|Δ ln L\| ≤ 1e-6; `production_a14_redraw` vs A14: medians within 0.02 of the 68% half-width, widths within 1% |
| 1414838 (60 min) | `tests/submit_test_nested.sbatch` | `tests/nested_runs/<run>/nested_1_{ref,resumed}/`, `tests/results/nested_tests.json`, `spc_*` in `tests/results/timing.json` | per class: one killed session (500 calls), the resumed session restored a checkpoint and reached 700 calls (past the first bound at 400); no catalog and both catalogs: resumed = uninterrupted bit for bit |
| 1414839 (2 h 15) | `tests/submit_test_sampler.sbatch` | `tests/sampler_runs/.../nested_1_phaseA_unif_b0/` (+ `state_it*.save`), `nested_1_phaseB_{rslice,rwalk,unif_b0,unif_boot5}/`, `tests/results/sampler_test.json` | a cheaper setting is recommended only if its insertion-index KS p ≥ 0.01 (Phase A blocks and its branch) and its ln L* after 100 iterations is within 3 sd of rslice's; else rslice |

After them: `make_runlist.py --timing tests/results/timing.json [--sampler-8 unif_b0|rwalk --calls-8 N]`
updates the estimates (and the eight-parameter setting) before `launch.py stage ...`.
