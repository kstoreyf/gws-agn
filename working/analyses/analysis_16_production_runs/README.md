# Analysis 16 — production runs (driver, run list, submission, tests)

**Status (2026-11-17, 10:45 UTC):** extended for the short paper's experiment
(`archive/paper_short_independent/EXPERIMENT.md`): five new analysis types, the experiment's run list
(`runs/RUNLIST_experiment.yaml`), explicit and pooled injection lanes, Analysis 14's settings for the
all-AGN-hosted analyses, one GPU lane. Test (a) passed every criterion (`tests/results/closure.json`).
Production stage 1 and all experiment grids (E1, E2) are queued on one chained lane job
(`agn-prod-l1`); nested runs wait for tests b and c. REPORT.md comes with the production results.
**17 Nov, 20:00 UTC:** stage 5 (observed-like mass spectrum) added to `runs/RUNLIST.yaml`; the lanes are
ordered by an explicit `lane_rank` per run (`make_runlist.py` LANE_RANKS); `launch.py stage` takes
`--match REGEX` and `--only-lane K`; both run lists regenerated with test b's timings (not queued).
**18 Nov, 13:00 UTC:** the eight-parameter runs sample uniformly first and with rslice after (stage V8 below): driver
options `--auto-switch-sample/--auto-switch-cpi` (off by default; default path bit for bit unchanged), realization 100
continued from its stopped uniform run, realizations 101-105 with the automatic switch; both run lists regenerated.

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
| `scripts/make_runlist.py` | → `runs/RUNLIST.yaml` (stages 1–5) and `runs/RUNLIST_experiment.yaml` (E1–E3, from `data/experiment_sets.json`; `reused_from_stage_1`, `hold`): each run's arguments, parts, estimated calls and GPU hours, `lane_rank` (the LANE_RANKS table at its top); written atomically |
| `scripts/launch.py` | `status`, `stage S [S ...] [--match REGEX] [--only-lane K]`, `run NAME`: queue tasks on the GPU lanes (one by default; `--lanes 2`), start a chained lane job |
| `scripts/lane_worker.py`, `scripts/submit_prod.sbatch` | the lane job (one GPU): runs tasks until its wall time, submits its own successor |
| `scripts/make_survey_nside.py` | complete-catalog survey files at another nside, as the generator made the nside-32 ones (stage X) |
| `scripts/plateau_correction.py` | ln Z of a finished nested run corrected for the floor plateau (tests/test_plateau_evidence.py) |
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
on|off`, `--max-var`, `--zmax`, `--free`, `--prior NAME=LO,HI` (tests only), `--nside N` (18 Nov; default 32 = the
production survey files `data/seedNNN/surveys/`, unchanged; another value reads `data/seedNNN/surveys_ns<N>/` and needs a
`--variant` containing `nside<N>`, recorded as `survey_nside` in config.json only off the default). Grid: a coarse pass over the
whole prior (H0 step 2 × f step 0.1; H0 step 1 in 1-D), then a fine grid of 41 × 21 (81 in 1-D) cells
centred on it, ±5 sd; the mass outside the fine grid is computed from the coarse pass and must be
< 1e-4 (else the fine grid is widened once); `--grid NAME=LO,HI,STEP` sets it explicitly. Nested:
dynesty, nlive 200, dlogz 0.1, bound multi, first bound after 400 calls, unif up to six parameters
(two independent runs, `--nested-index 1|2`, merged by the summary), rslice with slices 11, enlarge 1.25,
no bootstrap for eight (Analysis 13's guarded run; see test c); `--init-from STATE --switch-sample M` starts a part from
another run's state (only read; recorded in `<part>/init_from.json`) with sampler M; `--auto-switch-sample rslice
--auto-switch-cpi C` (off by default) switches a unif part once to rslice (dynesty's defaults, the production setting) after
the first chunk above C calls per iteration, recorded in the state itself and in `<part>/sampler_switch.json` (stage V8);
checkpoint every 10 min and at the end of every 2000-call chunk; resumes from `nested_<k>/state.save`; stops cleanly 5 min before the job's
deadline (exit 75). The run's JSON records the core commit (checked against the worktree HEAD), every
setting, the Slurm job id and wall time of every session.

## Launching

    PY=/hildafs/home/magana/tmp_ondemand_hildafs_phy230014p_symlink/magana/.conda/envs/jax/bin/python
    $PY scripts/make_runlist.py --timing tests/results/timing.json   # both run lists, reads data/experiment_sets.json
    $PY scripts/launch.py status [-v] [--stage S]
    $PY scripts/launch.py stage 1 E1 [--dry-run]    # stages 1-5 (production), E1/E2/E3 (experiment)
    $PY scripts/launch.py --lanes 2 stage 2 4 --match '<regex>' --only-lane 2 [--dry-run [--show N]]
    $PY scripts/launch.py run <run name> [--part nested_1] [--hours 6]

`--match REGEX` queues only the runs of the named stages whose name matches (Python `re.search`; anchor
with `^`/`$`). `--only-lane K` puts every new task on lane K instead of balancing, rewrites only lane K's
file and starts a job for lane K only. `--dry-run` writes and submits nothing; it prints per lane the tasks
to do, their GPU hours, the first `--show` (30) tasks in the order the worker takes them, a count by
(rank, stage), and whether a lane job would be submitted.

**Lane order.** Each run carries a `lane_rank` (written by `make_runlist.py` from its LANE_RANKS table,
the one place to change it); a lane file is sorted by (lane_rank, priority, estimated hours per part, name,
part), and a run without a rank falls back to its stage order (1, E1/E2, E3, 2, 3, 4, 5). Priority orders
the experiment's grids (f = 0.3 sets first, then 0.1 and 0.5, then the rest; s, family).

| rank | runs |
|---|---|
| 1.0, 1.5 | stage 1; E1 and E2 |
| 2 | stage 2, the grids (known population, flux-limited) |
| 2.5 | stage V8 (eight-parameter sampler, below), realization 100: the rslice continuation |
| 2.8 | stage 2, the nested runs (free densities; rank 6 until 18 Nov) |
| 2.85 | stage X (checks of two settings on realization 100, below) |
| 2.9 | stage 4, host-dependent populations (rslice), realization 100: V8's reference |
| 3 | stage 4 on realizations 100, 101, 102, 103, 105: all three classes |
| 3.5 | stage V8, realizations 101, 102, 103, 105 (unif, then rslice automatically) |
| 4 | E3 |
| 5 | stage 4, single population with both catalogs, realizations 106-110 |
| 7 | stage 3 on realizations 100, 101, 102, 103, 105 |
| 8 | stage 5 |
| 9 | stage 3 on realizations 106-110 |

**One lane** (default; this project may run one job at a time on RITA-GPU): every job the launcher
submits, and every successor a lane job submits, depends (afterany) on all of the user's pending or running
RITA-GPU jobs named `agn-*` (a job on hold is skipped: its owner releases and chains it). The lane file
is `runs/_lanes/lane1.txt`. The lane worker re-reads the lane before each task (it re-reads the run lists
when a lane line names a run it does not know), writes `runs/<run>/summary.json` when a run's parts are
finished, and skips tasks without data or marked `hold`. **Two lanes** again: `launch.py --lanes 2 stage
...` (or `export A16_LANES=2`); each lane then chains only behind its own jobs. Lane files and run lists are
written atomically (temporary file, then rename).

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
four 7,500 (both guesses); eight 272,000 for the single run (Analysis 13's guarded rslice run, extrapolated;
test c may change it). Seconds per call at 8000 samples on the record lane: 0.389 measured by test a (both
catalogs, population fixed), and by test b (`tests/results/timing.json`, realization 100) 0.405 single
population with both catalogs, 0.235 without a catalog, 0.435 host-dependent populations, 0.096 free
densities (m < 18); one catalog or no catalog with the population fixed stay at the conservative 0.39;
0.023 h per process measured.

**Another family's lane** (stage 5): the record-lane seconds per call times the lane factor, the ratio of
the observed-like family's population-fixed grids on its lane (E2, set obslike_f0p30_s0) to stage 1's on
the record lane, median over realizations 100-105 (`make_runlist.measured_lane_factors`, recomputed at each
regeneration): 1.153 with both catalogs (two_tracers_one_population vs known_population, ratios
1.143-1.163) and 1.237 without one (mass_spectrum_alone at s = 0 vs known_population, 1.210-1.245). The
lane has 2.9e6 detected injections against the record's 2.2e6.

| stage | runs | tasks | est. calls | est. GPU-h | ready now |
|---|---|---|---|---|---|
| 1 population known, complete catalogs (grids) | 35 | 35 | 14,770 | 2.7 | 2.7 |
| 2 flux-limited catalogs (grids; free densities nested) | 44 | 56 | 216,704 | 10.0 | 10.0 |
| 3 equal-population events, single population | 20 | 40 | 650,000 | 60.4 | 60.4 |
| 4 host-dependent events (peak lane) | 20 | 35 | 1,865,000 | 215.3 | 215.3 |
| 5 observed-like spectrum, host-dependent events (obslike lane) | 15 | 25 | 1,685,000 | 225.3 | 225.3 |
| E1 experiment grids, peak family | 235 | 235 | 147,745 | 28.5 | 28.5 |
| E2 experiment grids, observed-like family | 75 | 75 | 40,350 | 8.0 | 8.0 |
| E3 experiment nested, (f = 0.3, s = 2) | 5 | 10 | 180,000 | 20.5 | 20.5 |

**Stage 5** (`paper_prd/runs.txt`, approved): stage 4's three classes on the observed-like host-dependent
set (`events_marked_dmu0p10_dmuG5_obslike.h5`, E2's `obslike_f0p30_s1`), realizations 100, 101, 102, 103,
105: `--events-kind host_dependent --events-suffix _obslike --family-name obslike --population-family
obslike --injections-lane injections/injections_targeted_obslike.h5`, run names
`<analysis>/<both_catalogs|no_catalog>/complete/host_dependent_obslike/realization_NNN`. The redraw stream
is E2's on that set. The inputs (H0 67.74, f_AGN 0.30, mu_G 34, sigma_G 3.6, mu_chi 0, sigma_chi 0.1,
dmu_G 5, dmu_chi 0.10) are inside every prior (checked by `make_runlist.py` and the resolve test).

| class | runs | parts | calls per part | s per call | GPU-h |
|---|---|---|---|---|---|
| single population, both catalogs (6 parameters) | 5 | 2 | 18,000 | 0.467 = 0.405 x 1.153 | 23.6 |
| host-dependent populations, both catalogs (8; rslice, the `--sampler-8` setting) | 5 | 1 | 272,000 | 0.502 = 0.435 x 1.153 | 189.8 |
| single population, no catalog (5) | 5 | 2 | 14,500 | 0.291 = 0.235 x 1.237 | 12.0 |

Queued 17 Nov (one chain, job agn-prod-l1 1414946 and its successors): stage 1, then E1 and E2 in the
order s = 0, 1, 2 and within each f = 0.3, then 0.1 and 0.5, then 0 and 1. Nested stages (2-4, E3) wait
for tests b and c.

**Stage V8 (the eight-parameter sampler; 18 Nov).** *Found:* stage 4's eight-parameter model on realization 100 with
the six-parameter sampler (uniform sampling in multi-ellipsoid bounds, dynesty's bootstrap 5; run
`..._realization_100_unif_boot5`) was healthy by every check of `tests/nested_diagnostics.py` and cheap for about 1,500
iterations (calls per iteration by 200-iteration block 4.9, 4.1, 11.1, 21.0, 14.2, 16.7, 24.6), then stalled near 150 calls
per iteration, against about 60 for rslice. It was stopped at iteration 1,594 (30,102 calls, 0.497 s per call) and is held
in the run list. *Now:* the eight-parameter V8 runs sample uniformly while that is cheaper and with rslice after (run names
`host_dependent_populations/both_catalogs/complete/host_dependent/realization_NNN_unif_then_rslice`, stage 4's arguments,
events, lane and redraw, one part `nested_1`, seed 1):
- realization 100 continues the stopped run: `--init-from <its nested_1/state.save> --switch-sample rslice` (rank 2.5);
- realizations 101, 102, 103, 105: `--sample unif --auto-switch-sample rslice --auto-switch-cpi 60` (rank 3.5): the driver
  switches once, after the first 2,000-call chunk above 60 calls per iteration (on realization 100's trajectory: after
  iteration 1,558, at 24,769 calls; the largest chunk before was 45.5).

Both switches give the production rslice exactly (dynesty's defaults: slices 11, enlarge 1.25, bootstrap 0, bound update
every 4,400 calls). Estimates (`make_runlist.py`): 4,690 iterations per run (272,000 calls at Analysis 13's 58 per iteration;
the stopped run's ln L(X) extrapolated as an 8-D Gaussian gives 4,970-5,100), 60 calls per iteration with rslice, 0.497 s per
call: the continuation 186k calls, 25.6 GPU-h; each automatic run 213k calls, 29.4 GPU-h (rslice from the start: 272k, 37.5).
The paper's selectors skip stage V8. Checks: `tests/test_unif_then_rslice.py` (`tests/results/unif_then_rslice_checks.json`,
below).

**Stage X (18 Nov, checks of two analysis settings, realization 100; not production results).** Three grids, last in
RUNLIST.yaml, 0.25 GPU-h in all (seconds per call: the measured value of the production run each one varies):
- `known_population/both_catalogs/complete/equal_populations/realization_100_nside64`: stage 1's two-tracer grid (same
  events, redraw index 0, 8000 samples, injections, settings; no membership output) with both complete catalogs pixelized at
  nside 64 (`--nside 64 --variant nside64`), 1147 calls, ~0.15 GPU-h (an upper bound: on CPU a call costs 0.4 of the nside-32
  one, the padded rows hold 4023 instead of 14569 galaxies). Feeds `\SkyResShiftHzero`, `\SkyResShiftfAGN`
  (`paper_prd/scripts/skyres.py`). Survey files: `data/seed100/surveys_ns64/survey_{gal,agn}_complete_ns64.h5`, made by
  `scripts/make_survey_nside.py` as `data/generate_dataset.py` (stage_surveys) made the nside-32 ones; the tool rebuilds the
  nside-32 AGN file bit for bit (`--reproduce`), and each nside-64 file passes its checks (`<file>.check.json`: identical object
  counts, the four children of every nside-32 pixel hold its objects, redshifts per parent bitwise equal).
- `known_population/both_catalogs/flux_limit_18_densities_given[_agn_density_double]/equal_populations/realization_100_units_physical`:
  stage 2's m < 18 grid with both number densities given in Mpc^-3 (`--density given --density-units physical`) at the counts'
  values converted at H0 = 67.74 (log10 n_gal = -3.010584, log10 n_AGN = -5.008901), the AGN density at that value and twice it,
  ~0.05 GPU-h each. At H0 = 67.74 these equal the production runs (CPU dry runs: 1.1e-7 nats, the rounding of the inputs); they
  isolate how the densities scale with H0 (Analysis 15's +1.05 km/s/Mpc for doubling the AGN density against the production -0.66).
The paper's selectors skip stages whose names start with V or X (`paper_prd/scripts/build_values.production_stage`).
Queue: `$PY scripts/launch.py --lanes 2 stage X` (CPU dry runs: `bash tests/cpu_dryruns.sh x_nside64 x_m18_units_physical
x_m18_units_physical_double`; default-path identity of the extended driver: `tests/results/nside_option_checks.json`).
*19 Nov:* the three sky-resolution grids (`realization_NNN_nside64`, `realization_NNN_s32000_inj6`,
`realization_NNN_nside64_s32000_inj6`, same arguments with the realization's own events, lane, five further sets and survey files)
also on realizations 101, 102, 103, 105 (`make_runlist.X_SKYRES_REALIZATIONS`: the realizations with six injection sets), appended
after realization 100's five stage-X entries, rank 2.85, held while a survey file or injection set is missing; 12 runs, 5.4 GPU-h
estimated. Survey files `data/seed{101,102,103,105}/surveys_ns64/` (`make_survey_nside.py --seed R --nside 64`, each with its
passing `<file>.check.json`). The paper's mean and scatter of the shift over the five: `paper_prd/scripts/skyres.py`
(`\SkyRes[MC]Mean...`, `\SkyRes[MC]Sd...`, `\SkyResNreal`, `\SkyResMCNsets`).
*19 Nov, second batch (the sky-resolution sequence; `make_runlist.build_checks` (e)):* which part of the likelihood carries
KnownBoth's nside-64 shift, and does it converge? On the same five realizations, appended after the entries above, 55 runs:
(A) stage 1's one-catalog grids of the keys AGNhosted (`agn_catalog/.../agn_hosted/realization_NNN_nside<N>_s32000_inj6`, 32000
samples and six pooled sets as in stage 1), Galhosted (`galaxy_catalog/.../galaxy_hosted/realization_NNN_nside<N>`) and KnownGal
(`galaxy_catalog/.../equal_populations/realization_NNN_nside<N>`) at nside 16, 64, 128; (B) KnownBoth at Analysis 14's settings at
nside 128 and 16 (`realization_NNN_nside<N>_s32000_inj6`). Every other argument is the nside-32 partner's (checked by the
generator). Rank 2.85; priority 0 (A), 1 (B, nside 128), 2 (B, nside 16) orders them on a lane. Estimated 14.4 GPU-h (A 2.1, B
128 1.5, B 16 10.9): the partner's measured seconds per call, times the ratio of the longest catalog rows below nside 32 (an
upper bound; CPU dry runs give 2.7 times the nside-32 time at nside 16, against the row ratio 3.7). Survey files
`data/seedNNN/surveys_ns{16,128}/survey_{gal,agn}_complete_ns{16,128}.h5` (`make_survey_nside.py --seed R --nside 16|128`; nside
16 is checked against its four nside-32 children per pixel). Table: `tests/skyres_sequence.py` → `tests/results/skyres_sequence.json`
(per realization and nside: medians, 90% intervals, paired shifts against nside 32, the combination over the realizations by
`paper_prd/scripts/run_posteriors.combined`, the variances at the peak). The paper does not read these runs.
*21 Nov (`make_runlist.build_checks` (f)):* is KnownBoth's nside-64 shift on the first five realizations a fluctuation or an
offset? `known_population/both_catalogs/complete/equal_populations/realization_NNN_nside64` on the five other realizations of stage
1, 106-110 (`X_SKYRES_LATER`; one injection set each, so the production settings: 8000 samples, the realization's own set), the
arguments of the stage-1 KnownBoth run plus `--nside 64 --variant nside64` (as realization 100's), appended after (e), rank 2.85,
0.14-0.15 GPU-h each. Table: `tests/skyres_sequence.py`, block `KnownBoth_production` (ten realizations at nside 32 and 64; paired
shifts and combined posteriors over all present and per group 100-105, 106-110). The paper does not read them (`skyres.REALIZATIONS`;
audit S4 selects the realizations by their nside-64 grid at 32000 samples and six sets). Survey files `data/seed{106..110}/surveys_ns64/`
with passing checks. These seeds' `catalogs/` had been pruned (`campaign_100seeds/gen_one_seed.sh`) and were regenerated
(`generate_dataset.py --stage catalogs`, same seed, darksirens pin) with the glass venv's numpy (2.4) AVX-512 dispatch off
(`NPY_DISABLE_CPU_FEATURES="X86_V4 AVX512_ICL AVX512_SPR"`): the RM nodes that generated them have AVX2 only, and with AVX-512 on
0.3% of the AGN `z_obs` differ from the record by one ulp (the checks compare every pixel's redshifts with the nside-32 file bit for
bit); with it off they are equal bit for bit. The regeneration's record is `catalogs/regen_META.json`; the seeds' `META.json` is
untouched. Held (`X_SEQ_HOLD`): (A)'s AGNhosted grids at nside 128 (`agn_catalog/.../realization_NNN_nside128_s32000_inj6`), whose
Monte Carlo sum is -inf at every H0.

## Tests

**Unif, then rslice (18 Nov; `tests/test_unif_then_rslice.py`, `tests/results/unif_then_rslice_checks.json`, all pass).**
(settings) `switch_sampler(s, "rslice")` with no option gives exactly the sampler the driver builds for eight parameters
(method, proposals, slices 11, enlarge 1.25, bootstrap 0, walks, facc, bound update every 4,400 calls, first update, kwargs), on a
toy and on a copy of the stopped realization-100 checkpoint. (continuation) The continuation's run-list arguments on that copy
(dry-run model, ln L shifted by one constant to the state's scale by `tests/driver_lnl_offset.py`), into
`tests/unif_then_rslice_runs/`: the first process restored the copy, switched, wrote part `nested_1`'s own checkpoint and ran
4 rslice iterations until a hard kill; the second resumed from that checkpoint as rslice with the production settings and
continued (iterations 1,599-1,600); the copy was only read. (two_phase) A small finished run, 150 unif iterations then rslice
to convergence: the 150 first-phase dead points are in the results bit for bit, and prod_summary's posterior and ln Z are the
same before and after its change; the summary now also lists the switch, the restored state and
`calls_including_init_state` / `gpu_hours_including_init_source` (before, `calls` and `gpu_hours` covered the second phase
only). (auto) The automatic switch on that small problem (threshold 40): one switch, at iteration 246; runs killed before and
after it and resumed end bit for bit equal to the uninterrupted one (switch at the same iteration and call). Default path:
the V8, no-catalog (stopped and converged) and two-tracer grid dry runs with the driver before and after the change are
identical bit for bit, results and bounds included (`tests/results/unif_then_rslice_default_path.json`), and prod_summary
before/after gives identical summaries of three finished production runs (copies).

**ln Z with initial points on the floor (18 Nov; `tests/test_plateau_evidence.py`, `tests/results/plateau_evidence.json`).**
Analytic likelihoods (a Gaussian, sigma 0.03, inside a box holding q = 2%, 25%, 100% of the unit cube, -1e300 outside) in 6
and 8 dimensions, run with the driver's sampler and loop (nlive 200, multi, unif with dynesty's bootstrap 5 / rslice for
eight, first bound after 400 calls, dlogz 0.1 in chunks of 2000 calls, then add_live), 150-200 seeds per cell. dynesty 2.1.4
works through the floor points as a plateau and sets the volume after it to (N + 1 - n_f)/(N + 1): right in expectation, but
ln of it is biased by about +(1 - q)/(2 N q) and scatters by sqrt((1 - q)/(N q)), which logzerr does not contain. Measured
(ln Z - truth, mean +- sem; z = that / logzerr): 6D unif q = 1: +0.016 +- 0.019 (z sd 0.77); q = 0.25: +0.073 +- 0.018 (z sd
0.88); q = 0.02: +0.209 +- 0.032 (z sd 1.50). 8D unif: +0.041 +- 0.025, +0.064 +- 0.022, +0.254 +- 0.035 (z sd 0.90, 0.92,
1.46). 8D rslice: +0.022 +- 0.050, +0.079 +- 0.049, +0.42 +- 0.08 (z sd 0.93, 0.92, 1.45; at q = 0.02 also +0.18 after the
plateau). At q = 0.02, 1-2.5% of runs cannot start (every initial point on the floor) and 1-1.5% crash in dynesty's bootstrap
of the first bound (one or two points above the floor); the driver would fail the same way, deterministically. Returning -inf
instead of -1e300 (dynesty's initialiser then draws until 100 points lie above the floor) gives +0.07 +- 0.04 and +0.08 +- 0.04
at q = 0.02 (z sd 0.8). Correction for finished runs: `scripts/plateau_correction.py` (n_f from the stored dead points; ln
X -> ln(k/N) + n_f/(2 N k), variance + n_f/(N k), N = 400 for a merged pair), tested by `tests/test_plateau_correction.py`
(`tests/results/plateau_correction_test.json`); at q = 0.02 it brings z sd to 0.80-0.98.

**Test c, read again (18 Nov).** Its Phase A is void: 198 of the 200 initial points had ln L = -inf (the
variance guard over most of the eight-dimensional prior at 2,000 samples), which the driver passes as
-1e300; dynesty counts those as valid points, works through them as a plateau, and built its first bound at
400 calls from the 3 points above the floor, a degenerate ellipsoid (ln V = -79.7) that confined every later
point to a plane (singular values of the points 6.2, 2.7, then 4e-5). rslice and rwalk leave the plane in
Phase B and find ln L 3-4 above anything Phase A reached; the unif branches stay in it. The first insertion
test also counted the floor points (first block) and the final live points twice (a converged state); both
fixed in `tests/nested_diagnostics.py` (corrected ranks, bound volume against the contour volume, flatness
of the live set; CLI). Results: `tests/results/sampler_test_corrected.json` (test c),
`tests/results/insertion_reference_runs.json` (Analyses 11-13 and the A16 runs), and the comparison tool's
check on Analysis 11D's pair, `tests/results/compare_sampler_runs_A11D_pair.json`.

CPU (done): `tests/test_nocatalog_identity.py` (above); `tests/compare_resume.py` on a CPU run killed
after 170 calls and resumed (75 iterations, through the first bound, bit for bit equal to the
uninterrupted run); `tests/cpu_dryruns.sh [NAME ...]` (every configuration class builds, binds, the
diagnostics program equals the bound call, products written; `s5_*` are stage 5's three classes);
`tests/test_runlist_resolve.py` (every task of both lists resolves to the run list's name; planted values
of production event sets and every sampled input inside its prior); `tests/cpu_sampler_pipeline.sh`
(test c's mechanics).

GPU (`agn-prodtest`, nice 50, 4.0 GPU-h in all by their time limits), submitted 2026-11-17 ~09:00 UTC:

| job | script | writes | pass criteria |
|---|---|---|---|
| 1414826 (45 min) | `tests/submit_test_closure.sbatch` | `tests/closure_runs/{a14_convention,production_a14_redraw,a15_convention,production_stored}/grid/`, `tests/results/closure.json`, `spc_fixed` in `tests/results/timing.json` | `a14_convention` vs A14 `b_pe_n8000_r00` and `a15_convention` vs A15 `b_complete_x1`: max \|Δ ln L\| ≤ 1e-6 and medians, 68%/90% bounds equal to 1e-4 (H0) / 1e-5 (f_AGN); `production_stored` vs `a15_convention` max \|Δ ln L\| ≤ 1e-6; `production_a14_redraw` vs A14: medians within 0.02 of the 68% half-width, widths within 1% |
| 1414838 (60 min) | `tests/submit_test_nested.sbatch` | `tests/nested_runs/<run>/nested_1_{ref,resumed}/`, `tests/results/nested_tests.json`, `spc_*` in `tests/results/timing.json` | per class: one killed session (500 calls), the resumed session restored a checkpoint and reached 700 calls (past the first bound at 400); no catalog and both catalogs: resumed = uninterrupted bit for bit |
| 1414839 (2 h 15) | `tests/submit_test_sampler.sbatch` | `tests/sampler_runs/.../nested_1_phaseA_unif_b0/` (+ `state_it*.save`), `nested_1_phaseB_{rslice,rwalk,unif_b0,unif_boot5}/`, `tests/results/sampler_test.json` | a cheaper setting is recommended only if its insertion-index KS p ≥ 0.01 (Phase A blocks and its branch) and its ln L* after 100 iterations is within 3 sd of rslice's; else rslice |

After them: `make_runlist.py --timing tests/results/timing.json [--sampler-8 unif_b0|rwalk --calls-8 N]`
updates the estimates (and the eight-parameter setting of stages 4 and 5) before `launch.py stage ...`.

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
- 17 Nov, 20:00 UTC, with stage 5: all 511 tasks resolve (stage 5: 25); for every production run the
  events file's planted f_AGN, AGN offsets and galaxy-class mu_G, mu_chi and spectrum are the kind's and
  family's, and every sampled parameter's input is inside its prior. CPU dry runs of the three stage-5
  classes on realization 100 (`logs/dryrun/s5_*.log`, `tests/dryrun_runs/`): built and bound on the
  obslike lane (n_draw 1.08e9), finite ln L at the inputs, diagnostics = bound call.
