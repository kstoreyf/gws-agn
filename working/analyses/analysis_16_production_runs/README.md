# Analysis 16 — production runs (driver, run list, submission, tests)

**Status (2026-11-17, 10:45 UTC):** extended for the short paper's experiment
(`paper_short_independent/EXPERIMENT.md`): five new analysis types, the experiment's run list
(`runs/RUNLIST_experiment.yaml`), explicit and pooled injection lanes, Analysis 14's settings for the
all-AGN-hosted analyses, one GPU lane. Test (a) passed every criterion (`tests/results/closure.json`).
Production stage 1 and all experiment grids (E1, E2) are queued on one chained lane job
(`agn-prod-l1`); nested runs wait for tests b and c. REPORT.md comes with the production results.

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

### What realizations share

Each realization uses its own injection set (`data/seedNNN/injections/injections_targeted.h5`, drawn
with its own seed S*1000+4: 2.21-2.24 M detected injections each, different files), its own catalogs and
surveys, and its own redraw stream (the realization enters the SeedSequence). The Monte Carlo error of the
selection term is therefore independent between realizations. Shared: (i) the population inputs and the
fiducial cosmology (by design); (ii) for the flux-limited catalogs of stage 2 only, the Schechter
selection fits (`selection_redo/scripts/selection_fits_truez/`, fitted on seed 100's catalogs). The
generator draws every realization from the same luminosity function (M* = -20.47 - 5 log10 0.6774 =
-19.624 in h-scaled units, alpha = -1.07, faint cut fixed), and the seed-100 fits recover it to their
statistical errors (galaxies: M* within 0.001-0.009 mag, alpha within 0.0005-0.02, at most 1.8 sd; AGN:
M* within 0.002-0.022 mag, alpha within 0.002-0.06, at most 1 sd). The fit error is therefore a common
offset of every realization's completeness model, small but not averaged down by the five realizations;
it does not touch the complete-catalog runs (stage 1, the experiment). Using the generator's constants
instead would remove it.

### The experiment's analyses (populations at their inputs; grids)

The class populations come from the events file: the galaxy class at the family's inputs
(`POPULATION_INPUTS["corepop"|"obslike"]`, checked against the file's `mu_G_gal`, `mu_chi_gal` and
`mass_spectrum`), the AGN class shifted by the file's planted `dmu_G_agn`, `dmu_chi_agn`; the planted
f_AGN is the file's `planted_f_agn` (the f = 0.10 control is 0.10).

- `mass_spectrum_alone` (no catalog): at s = 0 core's spectral likelihood (H0). Where the classes differ,
  (H0, f_AGN) with ln L = sum_i ln[(1-f) L_i,gal + f L_i,agn] - N ln[(1-f) mu_gal + f mu_agn] (plus core's
  Monte Carlo term), evaluated by core's own reduction with two branch populations
  (`_ordinary_hierarchical_likelihood(branch_populations=...)`, hosts uniform in comoving volume in both
  branches), so the per-event variance, N_eff and guard are the mixture's. Every run checks f = 0 and
  f = 1 against core's bound spectral call with either class population.
- `galaxy_catalog_alone`: the galaxy catalog, the galaxy-class population, every merger (H0).
- `two_tracers_one_population`: both catalogs, f_AGN free, the galaxy-class population for both
  (= stage 1's `known_population` both catalogs).
- `two_tracers_two_populations`: both catalogs, f_AGN free, the AGN catalog's branch with its own
  mu_G, mu_chi (core's `per_catalog_population`, held at the planted offsets).
- `hosts_known`: the events split by `truth/host_type`; the AGN-hosted subset is its own core analysis
  (AGN catalog, AGN-class population), the galaxy-hosted subset its own (galaxy catalog, galaxy-class
  population), both bound to the realization's full injection set; ln L(H0) is the sum. The selection
  term of a subset is core's for that subset: -N_c ln mu_c + N_c (N_c + 3)/(2 N_eff,c), mu_c with the
  class-c population and class-c catalog, N_c its size (the two-class likelihood conditioned on the labels
  and on N_c, which removes f_AGN). The redraw is made for the whole set and then split, so each merger
  has the samples it has in every other run on its set. The half-variance correction is the subsets' sum
  (the stored per-cell `n_eff` is the equivalent N / sum_c N_c/N_eff,c; `sigma2_sel` = sum_c N_c^2/N_eff,c).
- Membership (`--membership`, two-tracer grids of realization 100 at f = 0.3): at the posterior median,
  r_i = f L_i,agn / (f L_i,agn + (1-f) L_i,gal), with L_i,c the same model's per-event evidence at f = 1 and
  f = 0, stored with the true host class (`grid/membership.npz`, `membership.json`).

Checks (`tests/test_experiment_analyses.py`, CPU, 40 events x 8000 samples, 40,000 injections;
`tests/results/experiment_checks.json`): A/B hosts_known on the all-AGN / all-galaxy set equals the
one-catalog run bit for bit; C each subset's ln L equals the class model on all events restricted to the
subset with its own selection term; D two populations with zero offsets equal one population; E the
two-population mixture at f = 0 / 1 equals the galaxy / AGN catalog alone with the galaxy- / AGN-class
population; F the spectral mixture with equal populations equals core's spectral likelihood; G the
membership identities (per-event mixture, and mean r = f* mu_agn/mu at the maximum-likelihood f*).

## Layout

| path | contents |
|---|---|
| `scripts/prod_common.py` | priors, population inputs, event kinds, files, analysis classes, run names, densities from counts, posterior summaries (no jax) |
| `scripts/prod_model.py` | the core model of one run: settings, build, redraw, bind, the diagnostics program, coordinates → core labels; `HostsKnownModel`, `SpectralMixtureModel` (experiment) |
| `scripts/prod_driver.py` | the driver: `--mode grid|nested`, every option explicit (`--help`); resumable |
| `scripts/make_runlist.py` | → `runs/RUNLIST.yaml` (stages 1–4) and `runs/RUNLIST_experiment.yaml` (E1–E3, from `data/experiment_sets.json`; `reused_from_stage_1`, `hold`): each run's arguments, parts, estimated calls and GPU hours |
| `scripts/launch.py` | `status`, `stage S [S ...]`, `run NAME`: queue tasks on the GPU lane (one by default; `--lanes 2`), start a chained lane job |
| `scripts/lane_worker.py`, `scripts/submit_prod.sbatch` | the lane job (one GPU): runs tasks until its wall time, submits its own successor |
| `scripts/prod_summary.py` | finished run → `runs/<run>/summary.json` (medians, 90% intervals, ln Z, selection N²/N_eff, membership, calls, GPU hours); run by the lane worker after each run |
| `runs/<run>/` | `config.json`; `grid/` (coarse.h5, fine.h5, result.json) or `nested_1/`, `nested_2/` (state.save, results.pkl, samples.npz, result.json); `sessions.json`, `attempts.json` per part; `summary.json` |
| `runs/_lanes/lane1.txt` | the lane queue (written by `launch.py`; `lane2.txt` only in two-lane mode) |
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
    $PY scripts/make_runlist.py --timing tests/results/timing.json   # both run lists, reads data/experiment_sets.json
    $PY scripts/launch.py status [-v] [--stage S]
    $PY scripts/launch.py stage 1 E1 [--dry-run]    # stages 1-4 (production), E1/E2/E3 (experiment)
    $PY scripts/launch.py run <run name> [--part nested_1] [--hours 6]

**One lane** (default; this project may run one job at a time on RITA-GPU): every job the launcher
submits, and every successor a lane job submits, depends (afterany) on all of the user's pending or running
RITA-GPU jobs named `agn-*` (a job on hold is skipped: its owner releases and chains it). The lane file
`runs/_lanes/lane1.txt` is sorted by stage order (1, E1/E2, E3, 2, 3, 4), the run's priority (experiment:
f = 0.3 sets first, then 0.1 and 0.5, then the rest; s, family) and estimated hours. The lane worker
re-reads the lane before each task, writes `runs/<run>/summary.json` when a run's parts are finished, and
skips tasks without data or marked `hold`. **Two lanes** again: `launch.py --lanes 2 stage ...` (or
`export A16_LANES=2`); each lane then chains only behind its own jobs.

**Injection lanes.** `--injections-lane` (relative to `data/seedNNN/`; default the family's,
`injections/injections_targeted.h5` or `..._obslike.h5`). The experiment's run list takes each set's
lane from `data/experiment_sets.json` (17 Nov: the peak family's `injections_targeted_peak.h5`, 2.5e8
draws, for every set but the all-AGN one, which keeps the record's lane; the observed-like family's
`injections_targeted_obslike.h5`), and every analysis of one event set uses that set's lane. A stage-1
run stands in for an experiment run only when the lanes agree (today: the all-AGN set's hosts_known is
stage 1's AGN-catalog run on it). Stage 4 takes the host-dependent set's lane from the JSON as well.
A run is marked `hold` while its set's lane is the present one at s > 0, or (all-AGN runs on the
record's lane) while its realization has fewer than six injection sets to pool.

**Monte Carlo settings of the all-AGN-hosted analyses** (Analysis 14: with 8000 samples and one
injection set their H0 median moves by 0.72 of its 68% half-width): every merger AGN-hosted and matched
to the AGN catalog (stage 1's AGN catalog on the all-AGN set; two_tracers_* on f = 1 sets) runs at
32000 samples per merger (`--nsamp 32000`), and so does the AGN-hosted subset of hosts_known
(`--agn-subset-nsamp 32000`, the galaxy-hosted subset stays at 8000). Injection sets drawn from the same
proposal as the run's lane are pooled with it (`--pool-injections` / `--agn-subset-pool`; today Analysis
14's five further sets of realizations 100 and 103, `mc_accuracy/inj_set{1..5}/seedR/injections/
injections_targeted.h5`, which share the record lane's proposal): the pool is one set of the summed
draws. The run name carries the settings (`realization_100_s32000_inj6`, hosts_known
`..._agn_s32000_inj1`). The 32000-sample redraw contains the 8000-sample one (blocks of 2000 on one
stream), so a merger's first 8000 samples are the same in every run.

## Run list and cost

`runs/RUNLIST.yaml` and `runs/RUNLIST_experiment.yaml` (from `make_runlist.py`; the cost model is in its
header and the files). Nested calls are per run: the measured references are merged PAIRS (Analysis 11D,
six parameters: 29,262 = 14,245 + 15,017; Analysis 12, seven: ~48k per pair), so six parameters are
18,000 per run (14,600 + 676 iterations for the wider production priors at 5 calls each), five 14,500,
four 7,500; eight 272,000 for the single run. Seconds per call 0.389 measured (test a, both catalogs,
8000 samples); the others 0.39/0.44 until tests b and c measure them; 0.023 h per process measured.

| stage | runs | tasks | est. calls | est. GPU-h | ready now |
|---|---|---|---|---|---|
| 1 population known, complete catalogs (grids) | 35 | 35 | 14,770 | 2.7 | 2.7 |
| 2 flux-limited catalogs (grids; free densities nested) | 44 | 56 | 216,704 | 27.3 | 27.3 |
| 3 equal-population events, single population | 20 | 40 | 650,000 | 80.4 | 80.4 |
| 4 host-dependent events (peak lane) | 20 | 35 | 1,865,000 | 228.7 | 206.5 |
| E1 experiment grids, peak family | 235 | 235 | 147,745 | 28.5 | 28.5 |
| E2 experiment grids, observed-like family | 75 | 75 | 40,350 | 8.0 | 8.0 |
| E3 experiment nested, (f = 0.3, s = 2) | 5 | 10 | 180,000 | 22.2 | 22.2 |

Queued 17 Nov (one chain, job agn-prod-l1 1414946 and its successors): stage 1, then E1 and E2 in the
order s = 0, 1, 2 and within each f = 0.3, then 0.1 and 0.5, then 0 and 1. Nested stages (2-4, E3) wait
for tests b and c.

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

Test (a) result (09:56 UTC): all five criteria pass. A14 convention vs A14: max |d ln L| 3.6e-12, summaries
equal to 1e-13; A15 convention and production on stored samples vs A15: identical (0.0); production
convention vs A14 on the same redraw: H0 median 68.950 vs 68.953 (-0.003 = -0.003 of the 68% half-width),
f_AGN 0.27420 vs 0.27421, widths within 0.03%; 0.389 s per call at 8000 samples.

Extension of 17 Nov (afternoon), CPU:
- `tests/results/driver_extension_bitwise.json`: the extended driver against the morning's on dry runs of
  existing classes (grids both / m < 18 / no catalog; nested host-dependent and no-catalog): every stored
  per-cell quantity, the diagnostics-vs-bound values and ln Z identical bit for bit.
- `tests/results/experiment_checks.json` (`tests/test_experiment_analyses.py`): A, B hosts_known on the
  all-AGN / all-galaxy set = the one-catalog run, difference 0.0; C each subset's ln L = the class model
  on all 40 events restricted to the subset with selection term -N_c ln mu_c + N_c(N_c+3)/(2 N_eff,c),
  difference <= 3e-14, ln mu and N_eff identical; D zero offsets: two populations = one population,
  difference 0.0; E two-population mixture at f = 0 / 1 = galaxy / AGN catalog alone with the galaxy- /
  AGN-class population, relative difference 5e-15; F spectral mixture: endpoints = core's bound calls
  (0.0), equal populations = core's spectral likelihood (0.0); G membership: per-event mixture identity
  4e-15, and at the maximum-likelihood f*, mean r = f* mu_agn / mu to 1e-16 (equal populations: mean r
  0.2857 at f* 0.2765, mu_agn/mu_gal 1.046; host-dependent: 0.4126 at f* 0.3299, mu_agn/mu_gal 1.43).
- `tests/results/mc_settings_checks.json` (`tests/test_mc_settings.py`, 24 events, 30,000 injections per
  set): P ln mu of the six pooled sets = ln[sum_k ndraw_k mu_k / sum_k ndraw_k] to 2e-15, pooled N_eff the
  sum of the sets'; Q hosts_known on the all-AGN set at 32000 samples with the pool = the AGN catalog alone
  at the same settings, difference 0.0; R on a mixed set the galaxy-hosted subset is unchanged by the AGN
  subset's settings and the AGN-hosted subset equals the AGN-class model at 32000 samples with the pool
  restricted to it; S the first 8000 samples of a 32000-sample redraw reproduce the 8000-sample run, 0.0.
- `tests/results/runlist_resolve.json`: all 486 tasks of both run lists resolve; names, parameters,
  priors, files, planted f_AGN and offsets of every experiment set as the run list says.
