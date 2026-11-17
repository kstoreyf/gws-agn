# The observed-like mock family (`--mass_spectrum gwtc3_pp`, suffix `_obslike`)

**Status (2026-11-17).** Generator option added. 20 event sets (seeds 100, 101, 102, 103,
105 x {f_AGN 0.3 equal populations, f_AGN 0.3 host-dependent, all mergers in galaxies, all
in AGN}) made and validated; redraw at 8000 samples per event works. The record's
injection lanes do not serve this spectrum (selection-term variance 57 / 104 for 1000
events), so each realisation now has a **dedicated injection lane**,
`injections/injections_targeted_obslike.h5` ("Dedicated injection set" below). Nothing is
committed.

## Why

The present mocks put 0.90 of the mergers in the Gaussian peak (core's `v1` = power-law
weight 0.10), which makes the mass scale a far sharper ruler than in the data. This family
replaces the mass spectrum alone by the power law + peak fitted to GWTC-3. Spin, redshift
evolution (gamma = 0), hosts, planted f_AGN = 0.30, the host-dependent offsets (+5 Msun in
mu_G, +0.10 in mu_chi), the measurement model and detection are the present family's.

## Design values and where they come from

Source: R. Abbott et al. (LVK), *Population of Merging Compact Binaries Inferred Using
Gravitational Waves through GWTC-3*, Phys. Rev. X 13, 011048 (2023), arXiv:2111.03634. The
text was read in the journal version and arXiv v1-v4; v3, v4 and the journal version agree.

| GWTC-3 (Eqs. B4-B7) | paper median (90%) | where | used | core name | core value |
|---|---|---|---|---|---|
| alpha | 3.5 +0.6 -0.56 | Sec. VI A | 3.5 | `alpha` | 3.5 |
| m_min | 5.0 +0.86 -1.7 | Sec. VI E (fit without GW190814, GW190917) | 5.0 | `m_min` | 5.0 |
| delta_m | 4.9 +3.4 -3.2 | Sec. VI E (same fit) | 4.9 | `dm_min` | 4.9 |
| m_max | **not quoted** (only m_99 = 44 +9.2 -5.1, Sec. VI A) | - | 87 (brief) | `m_max` | 87 |
| high-mass taper | none: P is cut sharply at m_max (Eq. B4) | App. B 1 b | - | `dm_max` | 0.01 (core's prior floor) |
| lambda_peak | 0.038 +0.058 -0.026 | Sec. VI A | 0.038 | `v1` (POWER-LAW weight) | **0.908** (w_G = 0.092) |
| mu_m | 34 +2.6 -4.0 | Sec. VI A | 34 | `mu_G` | 34 |
| sigma_m | **not quoted** ("consistent with previous inferences") | Sec. VI A | 3.6 (brief) | `sigma_G` | 3.6 |
| beta_q | 1.1 +1.7 -1.3 | Sec. VI A | 1.1 | `beta` | 1.1 |
| spin, rate | not used (GWTC-3: kappa = 2.9) | - | present family | `mu_chi`, `sigma_chi`, `gamma` | 0, 0.1, 0 |

Core theta, in core's order: `[v1 0.908, alpha 3.5, m_min 5, m_max 87, dm_min 4.9, dm_max
0.01, mu_G 34, sigma_G 3.6, beta 1.1, mu_chi 0, sigma_chi 0.1, gamma 0]`; AGN class of the
host-dependent set: mu_G 39, mu_chi 0.10. All inside core's prior boxes and Analysis 16's
priors.

**Departures from the brief, and why.**
- alpha 3.5 and delta_m 4.9, not 3.4 and 4.8: those are the published medians (3.4 is
  v1/v2's; v1/v2 also quote m_min 2.5 and delta_m 7.8 for the fit without the two low-mass
  events). m_min is 5.0 exactly either way.
- **Peak weight 0.092, not 0.04.** GWTC-3 multiplies the WHOLE mixture by S(m1; m_min,
  delta_m) and then renormalises (Eq. B4; gwpopulation does the same), while core tapers and
  normalises the power law alone. The two densities coincide when
  w_G = lambda <S>_G / ((1 - lambda) <S>_PL + lambda <S>_G). The taper removes 61% of the
  steep power law's mass (<S>_PL = 0.388; <S>_G = 1 - 3e-13), so lambda = 0.038 is
  w_G = 0.0923: in GWTC-3's fitted density 9.2% of mergers are in the peak, not 3.8%. I used
  w_G = 0.092 (implied lambda 0.0379). Taking 0.04 literally would put half as many mergers
  in the peak as the fit. If the literal reading is wanted instead, regenerating the 10 sets
  takes about 15 minutes.
- sigma_m 3.6 and m_max 87 are **not verifiable from the paper**: no version quotes them. They
  are the brief's values (said to be the data release's medians). Checks: the design
  density's 99th percentile of m1 is 42.6 Msun against the paper's m_99 = 44 +9.2 -5.1, and
  it hardly depends on them (43.8 with core's registry placeholders sigma 5.5, m_max 65). With
  alpha 3.5, the power law above 77 Msun holds 3e-4 of its mass.

**The two smoothing scales in core** (`population/parametric.py` PowerLaw, `utils.py`
sfilter_low/high, `base.py` `MixtureModel._low_mass_edge`):
- `dm_min` is the width of the Planck taper that turns the power law on above `m_min`, and
  the same taper on m2 = q m1 in the pairing of BOTH components (the peak inherits the power
  law's edge). That is GWTC-3's delta_m in Eqs. B4 and B7, so `dm_min = 4.9`.
- `dm_max` is the width of the taper that turns the power law off below `m_max`. GWTC-3 has
  none (a sharp cut), so `dm_max = 0.01`, core's prior floor (`DM_MAX` = [0.01, 20]). Core
  normalises the 0.01-wide window on its own nodes; the identity check below covers it.
- Structural difference: core's Gaussian is untapered and truncated to [1, 200], while
  GWTC-3 tapers it by S. That is immaterial here, since the peak sits 6.7 sigma above
  m_min + delta_m.

`scripts/design_values.py` -> `results/design_values.json` holds all of the above with the
numbers.

## Generator option

`generate_dataset.py --mass_spectrum gwtc3_pp` (default `mock` = the record) replaces the mass
fields of `PopulationConfig` (alpha, mmin, mmax, dm_min, dm_max, peak_fraction, peak_mu,
peak_sigma, beta) and nothing else. It refuses to run without `--population_code core`. A
non-default spectrum adds:
- `truth/pop_component_peak`, the true mixture component of each event (1 = peak);
- the attribute `mass_spectrum`;
- a `mass_spectrum` block in `metadata_json` with core's theta and the detected peak count.

The draw is the core route of `corepop/README.md` at the new population.

**Regression** (`scripts/regress_generator.sh`, seed 100, AVX-512 off on miko): the edited
generator reproduces `events.h5` (legacy default), `events_corepop.h5` and
`events_marked_dmu0p10_dmuG5_corepop.h5` byte for byte in every dataset
(`results/regress_bitcheck_*.json`). It refuses `gwtc3_pp` with the legacy code.

```bash
python generate_dataset.py --seed S --stage events --population_code core --mass_spectrum gwtc3_pp \
    --seed_events $((S*1000+11)) --n_events 1000 --nsamp 2000 --events_suffix _obslike \
    [--dmu_chi_agn 0.10 --dmu_G_agn 5 --events_suffix _marked_dmu0p10_dmuG5_obslike]
python generate_dataset.py --seed S --stage validation --events_suffix <same suffix>
# all 10:   sbatch --job-name=obslike submit_sets.sbatch obslike ; ... obsmarked   (scripts/make_sets.sh)
```

Sub-seed S*1000+11 is used by no other set, so this family is independent of every existing
draw. The two kinds share it, as the present family's mixed and host-dependent sets share
S*1000+3.

## Files (under `gws-agn-data-v3/seed<S>/`, S = 100, 101, 102, 103, 105)

`events/events_obslike.h5`, `events/events_marked_dmu0p10_dmuG5_obslike.h5`, each with
`events_rejected_sample<SFX>.h5`, `events_meta<SFX>.json` and
`validation/validation_events<SFX>.json`. That is 40 files, 1.64 GB (162.9 MB per events
file). Analysis 16 finds them as `events_path(S, "equal_populations", "_obslike")` and
`events_path(S, "host_dependent", "_obslike")`. Its `POPULATION_INPUTS` needs an `obslike`
entry with the core theta above.

For the short paper's experiment, `events/events_puregal_obslike.h5` (f_AGN 0, sub-seed
S*1000+14) and `events/events_pureagn_obslike.h5` (f_AGN 1, S*1000+15) were added
(`scripts/make_sets.sh obspuregal|obspureagn`; 40 more files, about 1.6 GB). All 10 pass
V1-V7. Smallest true m2 is 5.64. Core ln p is finite everywhere and core's loader reads every
file (`results/experiment_core_logp_truth_obslike.json`). The largest per-event fraction of
samples below m2 = 5 is 0.985 (seed 101, all-AGN), the same edge effect as above.
`working/data/experiment_sets.json` lists every set of the experiment.

## Validation

| check | result | file |
|---|---|---|
| core's ln p vs the closed form the route samples (110k points, both pairing modes, both classes) | max abs Delta ln p = 3.35e-7 outside the e^-130 taper floor, a constant offset (median 3.35e-7), including within 1 Msun of m_max; no finite/infinite mismatch | `results/density_identity.json` |
| core's p(m1, q) normalisation | 1 + 2e-6 (grid quadrature); P(m2 < 5) = 7e-21, P(m2 < 6) = 0.011, P(m2 < 8) = 0.39 | same |
| sampler: 4e6 proposals per class through `draw_source_population`, KS vs core's marginals | m1, q, m2, chi_eff: D <= 5.6e-4, p >= 0.16; peak share 0.0919 / 0.0920 (w_G 0.092); P(m2 < 5) = 0 | `results/sampler_gof.json` |
| events-only validation V1 V2 V2b V3 V3c V5 V7 | all 10 files pass; equal-population AGN-share pulls -0.48, +2.00, -1.10, +0.35, +1.86 | `seed<S>/validation/validation_events*_obslike.json` |
| no event with true m2 < 5; core ln p finite at every truth (both modes, each event under its class); core's loader | all 10 files; smallest true m2 5.60 | `results/obslike_core_logp_truth.json` |
| detected events vs core x selection (60 KS tests, m1/q/m2/chi_eff, per class for host-dependent) | p-values uniform (KS 0.60), 6.7% below 0.05, min 0.0026 (q, seed 101 equal) | `results/detected_gof.json` |
| host-dependent: selection-corrected V5 | expected AGN share 0.333-0.338; pulls -0.20, -0.35, +1.21, -0.51, +1.50 | same |
| host-dependent: class offsets | Delta mean chi_eff pooled 0.097 (planted 0.10); Delta mean m1 pooled 2.45 Msun (predicted 2.71; a +5 Msun peak shift moves the mean less when 56% of detections are power law) | same |
| detected peak share vs prediction | equal 0.410-0.433 (predicted 0.420); host-dependent 0.425-0.455 (predicted 0.445-0.446) | same |
| `mc_redraw` at 8000 samples per event, seed 100 `events_obslike.h5` (Analysis 14's five checks + Analysis 16's production redraw) | PASS: re-derived columns exact; 144 per-event KS min p 0.0065 (6.9% < 0.05); pooled KS p >= 0.28; p_pe closed form to 6e-16; 8000 contains 2000; core's loader bitwise; production redraw (key [16_100_2026, 100, 2, 413853, 0]) 1000 x 8000 in 8 s, row sums 1 to 2e-16 | `results/validation_redraw_events_obslike_s100.json` |

**Approximations to know about.**
- *Legacy power-law fallback.* `gmd._sample_q` gives up after 10,000 rejection rounds and
  draws q from the untapered q^beta. With delta_m = 4.9 this happens for m1 < 5.77 Msun, at
  4.8e-5 per proposal (1.2e-6 in the present family). The route keeps such a q (m2 >= 5
  still, finite density under core). Two detected events in the 10 files have m1 < 5.77
  (seed 102 host-dependent, m1 5.64; seed 105 host-dependent, m1 5.75). Their posterior odds
  of carrying a fallback q are about 4% and 0.3%. That is about 0.004 affected events per
  1000-event set.
- *Many events near the 5 Msun edge.* 74-105 events per set have m1 < 10. Their broad q
  posteriors put 1.5-1.9% of all PE samples below m2 = 5, where core's density is zero
  (0.1-0.2% in the present family). 4-10 events per set have more than half their samples
  there. Two events (seed 101 and seed 105 host-dependent) have more than 95%, the largest
  0.97. Those events carry few effective samples in the PE Monte Carlo, so watch the
  per-event variance (`results/low_m2_pe.json`).
- *Deeper horizon.* The detected true z reaches 0.42-0.54 (present family 0.31-0.39). The
  largest redshift any PE sample reaches at H0 = 100 is 0.77-0.93 (present 0.58-0.76). That
  is inside the catalog (edge 1.0, constant-density plateau to 0.923; only seed 103's
  host-dependent set, 0.933, reaches the ramp) and inside darksirens' production grid
  (1.10). It is past the 0.7 bar of the full validation's V8 in all 10 files; V8 is not
  part of the events-only mode.

## Injections: the record's lanes are not adequate

`scripts/injection_neff.py` -> `results/injection_neff.json`, seed 100. Changing the population
changes only the mass/spin factor of each detected injection's weight. The proxy takes the
population branch's redshift prior (uniform in comoving volume), the corepop convention. It
reproduces corepop's numbers for the present family.

| target population | lane | detected injections with w > 0 | N_eff | N^2/N_eff (N = 1000) | half of sum w^2 in |
|---|---|---|---|---|---|
| present (legacy draw) | targeted | 100% | 6.21e5 | 1.61 | 30,689 |
| present, core, GAL class | targeted | 99.87% | 6.20e5 | 1.61 | 30,153 |
| present, core, AGN class (mu_G 40) | targeted | 99.87% | 1.62e5 | 6.18 | 2,037 |
| **obslike, GAL class** | targeted | 99.77% | **1.76e4** | **56.7** | 205 |
| **obslike, AGN class (mu_G 39)** | targeted | 99.82% | **9.6e3** | **104** | 25 |
| obslike, GAL class | popuni | 99.38% | 4.86e4 | 20.6 | 1,069 |
| obslike, AGN class | popuni | 99.63% | 2.56e4 | 39.1 | 132 |

The weight sits in the low-mass power law (weighted m1 10th percentile 10 Msun), where the
proposal's population branch (90% in a 35 Msun peak) barely reaches. Scaled to Analysis
14's measured production values (targeted lane, catalog redshift prior: N_eff 7.5e5 for the
mixed set, 3.8e5 for the all-AGN set) by the proxy ratio, the obslike selection-term
variance is about 47 (GAL) / 86 (AGN class) with the mixed catalog and 93 with the AGN
catalog. **Recommendation: do not use the existing injections for this family.** A side
note for the present family: its AGN class alone (host-dependent runs) has a proxy
N^2/N_eff of about 6. The mixture over classes is better than that, but it is worth
checking in those runs.

**The first sizing (superseded).** `scripts/injection_sizing.py` estimated 7e8-1.4e9
proposals and 4-9 times today's rows. It kept the record's targeted branch proposing the
present spectrum's masses (35 Msun peak), and those rows carry almost no weight under this
target. With the obslike masses in every branch the rows are useful; a dedicated lane of
about today's size serves the family (next section).

## Dedicated injection set (`injections/injections_targeted_obslike.h5`)

**Why not N_eff = N_detected.** Matching the mass proposal to the target makes the mass
factor of each weight close to one, but the targeted lane's redshift/sky proposal (population
+ uniform + AGN-targeted branches) does not match the catalog redshift prior. Analysis 14
measured N_eff/N_det = 0.34 (mixed set) and 0.17 (all-AGN set) on today's lane, whose masses
match its target exactly. So about 3e6 detected rows are needed, not 5e5-1e6.

**Design** (`generate_dataset.py --stage injections --lane targeted --inj_suffix _obslike
--inj_proposal_json "$(cat scripts/inj_proposal.json)" --inj_seed S*1000+12 --tgt_host_zmax
0.75 --tgt_z_cap 0.75 --mix_population 0.62 --mix_uniform 0.03 --mix_targeted 0.35 --ndraw
1.08e9`; `scripts/inj_make.sh`, RM array 1414900):
- **Masses and spins** of the population and AGN-targeted branches come from a broadened
  obslike proposal, the legacy PopulationConfig {alpha 3.5, m_min 5, m_max 87, dm_min 4.9,
  dm_max 0.01, beta 1.1, peak weight 0.15 at **36.5 +- 6.5 Msun**, chi_eff ~ TN(**0.05,
  0.13**)}. The single peak covers both classes (34 and 39) and widths 2.5-6. The spin covers
  mean 0 and 0.10. pdraw stays exact (legacy samplers and density).
- **Branch weights** 0.62 / 0.03 / 0.35. The uniform branch was cut from 0.10: under this
  target its rows are mostly at m1det > 90 and carry no weight. The targeted share was raised
  for the AGN catalog. Chosen by reweighting a 1.5e8-proposal pilot over seven mixtures
  (`scripts/inj_mix_opt.py`, `results/inj_mix_opt_s100.json`): it balances the AGN class
  against the GAL class at fixed stored size.
- **Targeted branch** host cut and box cap raised from 0.5 to 0.75: obslike detections reach
  z = 0.54, and at H0 = 90 the re-read redshifts go higher.
- Own random stream: lane seed S*1000+12. The record's files and META.json are untouched. The
  default injection path is byte-identical to the record (6e6-proposal regression,
  `results/inj_regress_default_bitcheck.json`).

**Result**, per realisation:
- 1.08e9 proposals give 2.90-2.95e6 detected rows (about 270 MB per file, 1.4 GB in all),
  in 19-20 min on 8 RM CPUs.
- pdraw equals the mixture of its components exactly. 3000 rows were recomputed
  independently, the targeted density by brute force over the flat host list, to 1e-15
  (`results/inj_check_s<S>.json`).
- Core's `load_injections` reads it.

**Effective number** (`scripts/inj_stress.py`, `results/inj_stress_dedicated_s<S>.json`).
The weights are evaluated in darksirens' convention: fixed dL, re-read at trial H0, core's
density, rate factor (1+z)^-1. The redshift prior is the complete AGN survey's kernels for the
AGN class and uniform in comoving volume for the GAL class (8 galaxies per kernel width).
This proxy reproduces Analysis 14's production numbers on today's lane: 3.77e5 against 3.77e5
(all-AGN set) and 8.3e5 against 7.5e5 (mixed set).

N^2/N_eff for N = 1000, seed 100 (other seeds within 0.03 at design):

| point | GAL class | AGN class (+5, +0.10) | host-dependent mix, f 0.3 | equal populations, f 0.3 | all mergers in AGN |
|---|---|---|---|---|---|
| design | 1.98 | 3.07 | **1.29** | **1.44** | 3.68 |
| mu_G 30 | 2.63 | 3.48 | 1.61 | 1.89 | 4.87 |
| mu_G 38 | 1.63 | 3.44 | 1.22 | 1.21 | 3.11 |
| sigma_G 2.5 | 2.18 | 3.52 | 1.43 | 1.61 | 4.17 |
| sigma_G 6 | 1.82 | 2.83 | 1.19 | 1.30 | 3.28 |
| H0 55 | 2.05 | 4.21 | 1.48 | 1.66 | 5.24 |
| H0 90 | 1.93 | 2.18 | 1.15 | 1.28 | 2.46 |
| *record's lane, design* | *63.7* | *76.4* | *36.3* | *33.9* | *50.0* |

The host-dependent mixture at f = 0.1 and 0.5 stays below 2.1 at every point. The likelihood
of both obslike families sees the mixtures (1.2-1.9). The per-class columns are the
conservative view; only the all-AGN case approaches 5, at H0 = 55 with masses far from the
design.

**Side note on the present family** (`results/inj_stress_existing_mock_s100.json`,
`inj_stress_peak_existing_s<S>_x{1,2}.json`). Its own lanes give the host-dependent mixture
N^2/N_eff of 3.0 at f = 0.3 (worst stressed 8-9), 5.4-5.7 at f = 0.5, and 65-170 at s = 2.
The AGN class needs a lane of this kind there too.

## The detected sample beside the present family (realisation 100; 5-realisation mean +- sd)

`scripts/describe_detected.py` -> `results/detected_sample.json` (every row) and
**`results/obslike_paper_values.json`** (design + this table + spread, for the paper's macros).

| | present, equal | obslike, equal | present, host-dep. | obslike, host-dep. |
|---|---|---|---|---|
| peak share of the population | 0.90 | 0.092 | 0.90 | 0.092 |
| **peak share of detected mergers** | 0.967 (0.968 +- 0.002) | 0.416 (0.422 +- 0.009) | 0.971 (0.971 +- 0.003) | 0.445 (0.443 +- 0.012) |
| **median z** | 0.133 (0.132 +- 0.001) | 0.103 (0.108 +- 0.003) | 0.137 (0.138 +- 0.002) | 0.110 (0.110 +- 0.001) |
| 90th-percentile z | 0.193 | 0.196 | 0.198 | 0.195 |
| **maximum z** | 0.311 (0.339 +- 0.031) | 0.440 (0.441 +- 0.015) | 0.320 (0.334 +- 0.018) | 0.420 (0.464 +- 0.045) |
| **median m1 [Msun]** | 36.5 (36.7 +- 0.2) | 32.7 (32.8 +- 0.3) | 38.4 (38.4 +- 0.0) | 33.3 (33.7 +- 0.4) |
| **median m2 [Msun]** | 28.6 (28.7 +- 0.3) | 23.3 (23.5 +- 0.2) | 30.0 (30.0 +- 0.2) | 23.4 (24.4 +- 0.8) |
| median source chirp mass [Msun] | 28.0 | 24.0 | 29.1 | 24.2 |
| median detector-frame chirp mass [Msun] | 31.6 | 26.7 | 33.1 | 27.0 |
| detected fraction per proposal | 8.0e-3 | 1.67e-3 | 8.7e-3 | 1.73e-3 |
| largest PE redshift at H0 = 100 | 0.62 | 0.80 | 0.64 | 0.79 |

The peak share of detected mergers is the realised component flag for obslike. For the
present family (whose files do not store the flag) it is the mean membership probability
w_G G(m1)/p(m1); the two agree for obslike (0.416 / 0.416 at seed 100). Selection lifts the
peak from 9% of mergers to 42% of detections; the low-mass power law is seen only nearby, so
the median redshift falls while the heaviest systems push the horizon out.

## Timings and disk

| step | where | time |
|---|---|---|
| density identity + marginals (core venv) | RM, 8 CPUs | 4.6 min |
| sampler check (8e6 proposals) | RM | 1.9 min |
| 5 equal-population sets, in parallel | RM, 8 CPUs | 53-55 s each + validation 15-17 s |
| 5 host-dependent sets, in parallel | RM, 8 CPUs | 53-54 s each + validation 15-17 s |
| core truth check, 10 files | RM | 29 s |
| redraw validation (incl. 8000-sample production redraw) | RM | 39 s |
| injection N_eff / sizing | RM | 2.9 min / 0.5 min |
| detected-set check | RM | 2.0 min |
| dedicated injection lane, 1.08e9 proposals, per realisation (5 in parallel) | RM, 8 CPUs | 19-20 min + check and stress 2 min |
| pilots (2 x 1.5e8 proposals), branch-weight scan | RM | 3 min each, 1 min |
| experiment sets (20 peak-family + 10 obslike, 5 jobs in parallel) | RM, 8 CPUs | 35-56 s per set + validation |

Disk: 1.64 GB of event files in the dataset (plus 1.6 GB for the all-galaxy/all-AGN obslike sets, 2.5 GB for the peak-family experiment sets, 1.34 GB for the five dedicated injection lanes); `cache/core_tables_obslike.npz` 25 MB
(regenerable); results and logs under 1 MB. Scratch used for the regression (0.5 GB) was
deleted.

## Layout

| path | contents |
|---|---|
| `scripts/design_values.py` | GWTC-3 medians, the lambda -> w_G mapping, the m_99 cross-check |
| `scripts/core_family.py` | core venv: `identity` (obslike), `truth` and `points` for either family |
| `scripts/regress_generator.sh` | bit-identity of the edited generator's legacy and core paths |
| `scripts/make_sets.sh`, `scripts/submit_sets.sbatch` | the extra sets (`fagn0p10`, `obslike`, `obsmarked`) + validation |
| `scripts/sampler_check.py`, `scripts/detected_gof_obslike.py` | proposals vs core; detected events vs core x selection |
| `scripts/injection_neff.py`, `scripts/injection_sizing.py` | the injection check and the dedicated-set estimate |
| `scripts/validate_redraw_obslike.py` | Analysis 14's redraw validation + Analysis 16's production redraw |
| `scripts/describe_detected.py` | the detected-sample table and the paper's JSON |
| `scripts/run_cpu.sbatch` | generic RM CPU wrapper (core or jax env) |
| `scripts/inj_proposal.json`, `scripts/inj_make.sh`, `scripts/submit_inj_make.sbatch` | the dedicated injection lane: proposal, one realisation, the array |
| `scripts/inj_pilot.sh`, `scripts/inj_mix_opt.py` | default-path regression, pilots, branch-weight scan |
| `scripts/inj_stress.py`, `scripts/check_injections.py` | N_eff at design and stressed points (either family, offset size s); V6-style pdraw check |
| `scripts/write_experiment_sets.py`, `scripts/v5_selection_corrected.py`, `scripts/v5_largeN.sh` | the experiment's set list (`../experiment_sets.json`); selection-corrected V5; large-N share test |
| `scripts/submit_sh.sbatch` | generic RM wrapper for shell scripts |
| `results/`, `logs/`, `cache/` | every number above; job logs; regenerable tables |
