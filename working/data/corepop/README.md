# Core-population event sets (`--population_code core`)

**Status (2026-11-17):** option added to `working/data/generate_dataset.py`; 20 event sets
(seeds 100, 101, 102, 103, 105 x four kinds) generated beside the signed-off ones and
validated. The default path is bit-identical to the record. The injection sets need no
regeneration. Nothing is committed.

## Why

The generator drew masses with the samplers of legacy darksirens 2b86a2d, whose
Gaussian-peak secondaries reach down to 1 Msun. darksirens-core e7c3007 gives those
binaries zero density below m2 = 5 Msun and tapers them up to 8 Msun, so about a third of
the legacy event sets contain an event core cannot explain (Analysis 14 README, "Core and
the mocks disagree below the minimum mass"). Production runs use core, so the events are
redrawn from core's population. The legacy files stay as they are.

## Core's density, read from its code

powerlaw+peak with shared beta/spin/gamma at the mock's input,
theta = [v1 0.1, alpha 2.3, m_min 5, m_max 80, dm_min 3, dm_max 10, mu_G 35, sigma_G 5,
beta 1, mu_chi 0, sigma_chi 0.1, gamma 0] (core's `fixed=True` vector has gamma = 2.5, so
gamma is set by hand). Pointers are under `src/darksirens-core-e7c3007/src/darksirens/`.

    p(m1, q, chi, z) = [w_PL PL(m1) + w_G G(m1)] * q^beta S(q m1; m_min, dm_min) / N(m1)
                       * TN(chi; mu_chi, sigma_chi, [-1, 1]) * (1+z)^(gamma-1)

- **Primary mass.** w_PL = 0.10, w_G = 0.90 (stick weight v1, `population/registry.py:153-161`).
  PL is m^-alpha times the low and high Planck tapers on [5, 80] with widths 3 and 10,
  normalised edge-exactly (`population/parametric.py:52-122`). G is a Gaussian on [1, 200],
  closed-form normaliser and support mask (`parametric.py:213-246, 269-292`;
  `population/base.py:202-267`).
- **Pairing.** `PowerLawPairing` q^beta S(q m1; m_min, dm_min), normalised per m1 over
  (m_min/m1, 1] (`parametric.py:1312-1359`; quadrature `base.py:270-999`). **Both mass
  components pair at the power law's edge (5, 3):** the peak declares no edge and inherits
  it (`MixtureModel._low_mass_edge`, `base.py:1146-1168`, used at `:1189-1219`). This is
  the only difference from the legacy draw, whose peak lanes pair at (1, 0.01). Zero
  density for m2 <= 5 and for m1 <= 5.
- **Taper.** `sfilter_low` (`population/utils.py:829-841`) has the same Planck form as
  legacy's `_sfilter_low`. Its exponent cap is 130 instead of 500, which matters only where
  S < e^-130, i.e. within 0.023 Msun of m_min.
- **Spin.** Gaussian truncated to [-1, 1], trapezoid-normalised (`parametric.py:1424-1449`,
  `base.py:1026-1039`).
- **Redshift.** (gamma - 1) ln(1+z) (`base.py:1378-1398`): with gamma = 0 that is the
  generator's host acceptance (1+z)^-1 exactly, so the host draw is already core's.
- **Host-dependent set.** Core's second branch is the same vector with mu_G,c2 and
  mu_chi,c2 replaced (`runtime_binding.py:361-382`, `paper_prd/notes/LIKELIHOOD_MATH.md`
  §5, §7). The AGN class is therefore mu_G = 40, mu_chi = 0.10, still pairing at (5, 3).
- **No sampler.** Core ships none for this density: no `sample`/`draw` function in
  `population/` and no mock-generator script. `MixtureModel.mass_q_density` is a density
  for an external flow surrogate.

## The route: an exact coupled rejection on the pairing alone

Core's density equals the legacy draw everywhere except the conditional pairing of the
peak lanes, and that ratio has a closed form. Tabulating core or rejecting on the joint
ratio is therefore unnecessary. `core_pairing_correction` in the generator works as follows.

1. Draw everything exactly as the legacy path does, on the main stream: host, m1 and its
   component flag, legacy q, chi_eff, measurement noise.
2. On a child stream (`SeedSequence(seed_events, spawn_key=(0x4350,))`), keep the legacy q
   with probability r = S_core(m2; 5, 3) / S_lane(m2). r <= 1 wherever legacy can draw,
   and r = 1 on every power-law lane.
3. Otherwise redraw q from core's pairing by rejection: q^beta on [5/m1, 1] proposed by
   gmd's `_sample_q_hard`, accepted with S_core.
4. Drop proposals core gives zero density (m1 <= 5, |chi_eff| > 1) before detection.

The kept-or-redrawn q is distributed exactly as q^beta S_core(q m1) / N(m1) for every m1,
so (m1, q, chi) follows core's density. S_core is a float64 numpy mirror of core's
`sfilter_low` (`core_sfilter_low`). The generator never imports core. It refuses `core`
unless `--core_darksirens` is at `CORE_POP_COMMIT` = e7c3007, the commit this closed form
was checked against.

Tally over 4e6 proposals: 3.3% (GAL) / 2.5% (AGN branch) of all lanes get a redrawn q, at
most 4 redraw rounds are needed, and no lane is dropped or left unresolved. Per 1000-event
set, 21-45 detected events carry a redrawn q (`truth/core_q_redrawn`).

**Bounded approximations** (all far below what 1000 events resolve):
- Taper floor (S < e^-130, m2 within 0.023 Msun of 5): the probability under core is
  ~1e-56.
- The legacy power-law lanes' own fallback: after 10,000 failed rejection rounds,
  `_sample_q` returns a hard q^beta draw. This affects only m1 in (5, 5.47), with
  probability 1.2e-6 per proposal, the same as in the record. Core gives those binaries
  finite density.
- Shifted marked spin: chi + 0.10 can exceed 1 only in the 9-sigma tail (dropped if it
  ever happens; 0 so far). The missing core mass below chi = -0.9 is 8e-24.

**Metadata** (only a core draw carries it, so a file without it is legacy). Attributes:
`population_code = "core"`, `population_core_commit`, `population_route`,
`population_table_sha256 = "none (closed form, no table)"`. `metadata_json` carries a
`population_code` block: density, route, child-stream derivation, core worktree HEAD,
`generator_sha256`, the per-proposal tally and the number of detected events with a
redrawn q.

## Generator changes (`git diff working/data/generate_dataset.py`)

- `--population_code {legacy,core}` (default `legacy`) and `--core_darksirens`.
- New functions `core_sfilter_low`, `new_core_tally`, `core_pairing_correction` and
  `draw_source_population`. The events loop now calls the last of these, which makes the
  record's calls in the record's order.
- Validation stage: with `--events_suffix` it validates `events<SFX>.h5` alone. It runs
  V1, V2, V2b, V3, V3c, V5 and V7, writes `validation/validation_events<SFX>.json`, and
  opens no catalog, survey or injection file. `validation.json` and META.json are never
  touched (`main()` skips `finalise_meta` in that case). Without the suffix the stage is
  unchanged.
- V5 in that mode tests the file's own planted f_AGN (exact host count when f is 0 or 1).
  For a set with a mass mark (dmu_G_agn != 0) the detected share is the planted f
  re-weighted by the two classes' detection efficiencies, so V5 checks bookkeeping only
  there. The selection-corrected test is below.

## Bit-identity of the default path

`scripts/bitcheck_default.sh` uses Analysis 0's dataset-level gate on four draws covering
every touched path: mixed set (seeds 100 and 103), all-AGN set, and host-dependent set
(child mass stream).

| comparison | mixed s100 | mixed s103 | all-AGN s100 | host-dependent s100 |
|---|---|---|---|---|
| unmodified generator (git HEAD 85f2934) vs modified, same machine | PASS | PASS | PASS | PASS |
| signed-off file on disk vs modified (numpy AVX-512 off) | PASS | PASS | PASS | PASS |

All 50 datasets of each file are byte-identical (`results/bitcheck_*.json`).

**Finding: AVX-512.** On miko (Xeon Gold 6526Y), numpy 1.26 dispatches transcendental
loops to AVX-512, which rounds 1 ulp differently from the loops the record was made with.
The unmodified generator then reproduces the record only to 1 ulp (11 of 1000 primary
masses, max relative difference 2.2e-16). With `NPY_DISABLE_CPU_FEATURES` set to the
AVX-512 group it reproduces the record bitwise
(`results/bitcheck_record_vs_HEAD_noavx512_mixed_s100.json`). The corepop sets were
generated that way, so their shared events are bitwise those of the legacy files.

The full default validation stage (12 checks) was also rerun on seed 100 with both
generators, in a scratch copy whose inputs were symlinks to the signed-off seed. Both pass
12/12, and the two `validation.json` files agree in all 260 leaves except the timestamp
(`scripts/fullval_compare.py`, `results/fullval_regression_s100.json`). The scratch outputs
(`gws-agn-data-v3-scratch-corepop/`) were deleted afterwards; `bitcheck_default.sh`
recreates them.

## Files written

Under `/hildafs/projects/phy230054p/magana/gws-agn-data-v3/seed<S>/` (reached through
`working/data/seed<S>`), for S = 100, 101, 102, 103, 105:

| kind | file | flags (as the legacy file of the same kind) | sub-seed |
|---|---|---|---|
| mixed, f_AGN = 0.30 | `events/events_corepop.h5` | none | S*1000+3 |
| all hosts AGN | `events/events_pureagn_corepop.h5` | `--f_agn 1.0` | S*1000+9 |
| all hosts galaxies | `events/events_puregal_corepop.h5` | `--f_agn 0.0` | S*1000+8 |
| host-dependent | `events/events_marked_dmu0p10_dmuG5_corepop.h5` | `--dmu_chi_agn 0.10 --dmu_G_agn 5` | S*1000+3 |

Each set has 1000 events x 2000 samples and is 162.7-162.9 MB. Each comes with
`events_rejected_sample<SFX>.h5` (1.3 MB), `events_meta<SFX>.json` and
`validation/validation_events<SFX>.json`. That is 80 new files and 3.29 GB in total. No
pre-existing file changed (sizes and mtimes of all 116 files under events/, validation/
and META.json compared before and after; all five META.json md5 unchanged). Only seed 100
had a legacy host-dependent set; the other four are new kinds there.

## Correspondence with the legacy files

The main stream is untouched, so every proposal keeps its host, m1, chi_eff and noise
variates. Only peak lanes whose legacy secondary fell below about 8 Msun get a new q.

- In each set, 955-979 of the 1000 events are bit-identical to an event of the legacy
  file of the same kind: same host, z, m1, q, chi_eff and recorded measurement (all
  `truth/obs_*` identical).
- 1-8 events are the same proposal with a moved q, and 18-39 events are new to the core
  set.
- PE samples are positional, so they coincide only over the leading run of positionally
  identical events (1-85 events). From the first event that differs, the PE noise belongs
  to other events. The rejected-sample file also differs from there on.

## Validation

| seed | set | checks passed | min true m2 | max frac. samples < 5 | core ln p finite | detected KS min p | events with q redrawn | events shared with legacy |
|---|---|---|---|---|---|---|---|---|
| 100 | mixed | V1 V2 V2b V3 V3c V5 V7 | 7.33 | 0.722 | yes | 0.417 | 26 | 974 |
| 101 | mixed | V1 V2 V2b V3 V3c V5 V7 | 6.89 | 0.574 | yes | 0.189 | 30 | 970 |
| 102 | mixed | V1 V2 V2b V3 V3c V5 V7 | 6.41 | 0.324 | yes | 0.448 | 41 | 959 |
| 103 | mixed | V1 V2 V2b V3 V3c V5 V7 | 7.16 | 0.238 | yes | 0.117 | 21 | 979 |
| 105 | mixed | V1 V2 V2b V3 V3c V5 V7 | 6.76 | 0.492 | yes | 0.100 | 38 | 962 |
| 100 | all-AGN | V1 V2 V2b V3 V3c V5 V7 | 5.75 | 0.215 | yes | 0.717 | 32 | 968 |
| 101 | all-AGN | V1 V2 V2b V3 V3c V5 V7 | 5.86 | 0.354 | yes | 0.001 (chi_eff; inherited) | 35 | 965 |
| 102 | all-AGN | V1 V2 V2b V3 V3c V5 V7 | 5.72 | 0.538 | yes | 0.134 | 23 | 977 |
| 103 | all-AGN | V1 V2 V2b V3 V3c V5 V7 | 6.93 | 0.081 | yes | 0.309 | 22 | 978 |
| 105 | all-AGN | V1 V2 V2b V3 V3c V5 V7 | 6.23 | 0.581 | yes | 0.113 | 45 | 955 |
| 100 | all-GAL | V1 V2 V2b V3 V3c V5 V7 | 6.63 | 0.263 | yes | 0.139 | 32 | 968 |
| 101 | all-GAL | V1 V2 V2b V3 V3c V5 V7 | 6.35 | 0.606 | yes | 0.099 | 34 | 966 |
| 102 | all-GAL | V1 V2 V2b V3 V3c V5 V7 | 6.13 | 0.715 | yes | 0.036 | 28 | 972 |
| 103 | all-GAL | V1 V2 V2b V3 V3c V5 V7 | 6.83 | 0.351 | yes | 0.358 | 33 | 967 |
| 105 | all-GAL | V1 V2 V2b V3 V3c V5 V7 | 5.82 | 0.885 | yes | 0.293 | 30 | 970 |
| 100 | host-dep. | V1 V2 V2b V3 V3c V5 V7 | 7.32 | 0.065 | yes | 0.386 | 27 | 973 |
| 101 | host-dep. | V1 V2 V2b V3 V3c V5 V7 | 6.58 | 0.590 | yes | 0.048 | 28 | - |
| 102 | host-dep. | V1 V2 V2b V3 V3c V5 V7 | 7.08 | 0.838 | yes | 0.161 | 29 | - |
| 103 | host-dep. | V1 V2 V2b V3 V3c V5 V7 | 7.16 | 0.221 | yes | 0.148 | 26 | - |
| 105 | host-dep. | V1 V2 V2b V3 V3c V5 V7 | 6.29 | 0.405 | yes | 0.159 | 35 | - |

**Generator checks.** All 20 files pass V1, V2, V2b, V3, V3c, V5 and V7 (`validation_events<SFX>.json`).

**Low m2** (`results/low_m2.json`; Analysis 14's `diag_low_m2.table` imported unchanged).
- Corepop sets: smallest true m2 is 5.72 Msun, no event has true m2 < 5, and no event has
  more than 95% of its samples below 5. The largest such fraction is 0.886, for an event
  with true m2 = 5.8, whose broad q posterior spills below the edge.
- Legacy sets of the same seeds: 23 events with true m2 < 5, and 7 sets with an event
  above 95%.

**Core's density at the truths** (`results/core_logp_truth.json`, core venv, both pairing
modes).
- ln p is finite for every event of every corepop file, each host-dependent event under
  its class's population.
- 13 of the 16 legacy counterparts have 1-4 events at -inf.
- Every corepop file loads through core's `load_gw_store`. The normalised p_pe rows sum to
  1 within 3.3e-16, and PE-reweighting ESS/nsamp is at least 0.83.

**Density identity** (`results/density_identity.json`). Core's ln p (evaluated in core's
venv) was compared with the closed form the route samples (scipy quadrature, independent
of both codes) at 100,000 points spanning the support and the taper band.
- Outside the taper floor, max |Delta ln p| = 1.1e-7, in both the "historical"
  (per-sample pairing, as Analyses 13/14 run) and the default (per-point) mode.
- Inside the floor zone (m2 within 0.023 Msun of 5) the difference is 7.0e-4 / 2.8e-4.
  That is core's own quadrature, at relative density e^-14 or below.
- The edge values reproduce Analysis 14's table: -inf at m2 = 4.9, -31.72 at 5.1, -4.29 at
  6.0, -0.765 for (35, 30).
- Core's p(m1, q) integrates to 1 + 4e-8. P(m2 < 6) = 0.0044 and P(m2 < 8) = 0.062 (GAL)
  / 0.058 (AGN).

**Sampler** (`results/sampler_gof.json`). 4e6 proposals per branch were drawn through the
generator's own `draw_source_population` and compared by one-sample KS with core's
marginals. The KS distance resolves 7e-4 at this N, against 0.043 for 1000 events.

| branch | mode | m1 D (p) | q D (p) | m2 D (p) | chi_eff D (p) | P(m2 < 5) | P(m2 < 8), core |
|---|---|---|---|---|---|---|---|
| GAL | core | 5.1e-4 (0.26) | 3.0e-4 (0.87) | 3.1e-4 (0.83) | 3.0e-4 (0.86) | 0 | 0.0622 (0.0623) |
| GAL | legacy | 5.1e-4 (0.26) | 0.031 (0) | 0.032 (0) | 3.0e-4 (0.86) | 0.0188 | 0.0941 |
| AGN | core | 4.1e-4 (0.53) | 5.6e-4 (0.16) | 3.2e-4 (0.79) | 4.0e-4 (0.55) | 0 | 0.0579 (0.0578) |
| AGN | legacy | 4.1e-4 (0.53) | 0.024 (0) | 0.024 (0) | 4.0e-4 (0.55) | 0.0142 | 0.0821 |

**Detected sets** (`results/detected_gof.json`, figure). The detected truths were compared
by KS with core's density times the generator's detection probability, built from each
seed's own catalog redshifts.
- 100 tests in all: p-values consistent with uniform (KS of the p-values 0.20), 3% below
  0.05.
- The one low value is chi_eff of seed 101's all-AGN set (p = 0.0011). Its chi_eff draws
  are those of the legacy file (p = 0.0015 there), so it is that realisation and not the
  route.
- The legacy files' m2 also pass this test (p >= 0.12): 1000 events cannot resolve the 3%
  of events that differ, though one of them drives core's likelihood.

**Host-dependent set: selection-corrected V5 and the class offsets.**

| seed | expected detected AGN share | realised | pull | Delta mean chi_eff (AGN - GAL) | Delta mean m1 [Msun] (predicted 4.6) |
|---|---|---|---|---|---|
| 100 | 0.366 | 0.360 | -0.41 | 0.098 +- 0.007 | 4.63 +- 0.37 |
| 101 | 0.370 | 0.380 | +0.67 | 0.109 +- 0.006 | 4.07 +- 0.41 |
| 102 | 0.367 | 0.364 | -0.21 | 0.100 +- 0.007 | 4.83 +- 0.38 |
| 103 | 0.368 | 0.346 | -1.45 | 0.098 +- 0.007 | 4.35 +- 0.41 |
| 105 | 0.370 | 0.351 | -1.21 | 0.094 +- 0.007 | 5.02 +- 0.38 |
| pooled | | | | 0.0999 +- 0.0030 (planted 0.10) | 4.58 +- 0.17 (predicted 4.65) |

The planted offsets are +5 in mu_G and +0.10 in mu_chi. Detected m1 differs by less than
+5 because the shared power law and selection damp it, so it is compared with its own
prediction.

![population](figs/corepop_population.png)

`figs/corepop_population.{pdf,png}`: rows are the GAL- and AGN-hosted classes of the
host-dependent set; columns are m1, q, m2, chi_eff.
- Grey fill: 4e6 core-route proposals. Black line: core's density. Dashed: the legacy
  draw.
- Coloured step: detected events, five sets pooled. Coloured line: core's density times
  detection.

## Injections: no regeneration needed (`results/injection_check.json`, seed 100)

Changing the population code multiplies each injection's weight by
R = p_core(m1, q, chi) / p_legacy(m1, q, chi), since the host, redshift and Jacobian
factors are common.

- **Proposal density.** The stored `pdraw_population` is the legacy density to 1.1e-15,
  and `pdraw` is positive and finite on every row.
- **Support.** No point has core density where the legacy proposal is zero (grid over
  core's whole support and every detected injection).
- **Zero-weight injections.** 2895 of 2,205,380 targeted and 1544 of 1,230,471 popuni
  detected injections (0.13%) have m2 < 5; under core they get weight 0.
- **Ratio on detected injections.** R <= 1.066 (median 1.03) on every detected injection
  of both lanes, in every branch.
- **Weight bound.** On a grid R peaks at 3007 at m1 = 5.05, q = 1, but that is a ratio of
  two vanishing densities. The weight itself, w = p_pop / pdraw (uniform-in-volume
  redshift prior, as the population branch draws), is bounded over core's whole support
  by the uniform branch: w <= 1.63 (targeted; legacy 1.54) and w <= 1.18 (popuni; legacy
  1.11). The realised maxima reach those bounds.
- **Effective number.** It changes by -0.12% (targeted) and -0.35% (popuni).

## Further seeds (106-199)

The 94 seeds 106-199 kept surveys, events and injections, but `catalogs/` was deleted
(`campaign_100seeds/README.md`), and the events stage needs the complete catalogs.

Per seed:
1. `--stage catalogs` (295-318 s, median 301 s, measured on RM in September).
2. **Gate:** regenerate the legacy `events.h5` into scratch and compare it bitwise with the
   one on disk. This proves the catalogs and the floating point are reproduced; RM's CPU
   type is not exposed, so do not assume the AVX-512 setting.
3. The four core sets (about 20 s each on RM).
4. Events-only validation (about 8 s each).
5. Delete `catalogs/`.

Cost: about 7.5 min per seed with 8 CPUs and 64 GB, i.e. about 90 core-hours for all 94
seeds, or about 45 min wall time at 16 concurrent array tasks (as `submit_gen.sbatch`).
Disk retained is 0.66 GB per seed (62 GB in all), plus 6.7 GB per seed in flight for the
transient catalogs (about 107 GB peak at 16-way). phy230054p is at 4.6 of 19.5 TB.

## Reproduce

```bash
cd working/data/corepop
./scripts/bitcheck_default.sh run pre <HEAD copy of generate_dataset.py>
./scripts/bitcheck_default.sh run post ../generate_dataset.py
./scripts/bitcheck_default.sh run post_noavx512 ../generate_dataset.py
./scripts/bitcheck_default.sh compare
./scripts/make_corepop_events.sh                       # 20 sets + events-only validation (~80 s)
CPY=/hildafs/projects/phy230014p/magana/envs/darksirens-core-e7c3007/bin/python
JPY=/hildafs/home/magana/tmp_ondemand_hildafs_phy230014p_symlink/magana/.conda/envs/jax/bin/python
export JAX_PLATFORMS=cpu
PYTHONPATH= $CPY scripts/core_eval.py tables           # core venv, ~2 min
PYTHONPATH= $CPY scripts/core_eval.py truth
export PYTHONPATH=/hildafs/projects/phy230014p/magana/src/darksirens-2b86a2d
$JPY scripts/sampler_check.py && $JPY scripts/detected_gof.py && $JPY scripts/injection_check.py
$JPY scripts/low_m2.py && $JPY scripts/make_figure.py && $JPY scripts/summarize.py
```

One core-mode draw by hand:

```bash
python generate_dataset.py --seed S --stage events --population_code core \
    --darksirens <2b86a2d worktree> [--f_agn F --seed_events N] --events_suffix _<kind>_corepop
python generate_dataset.py --seed S --stage validation --events_suffix _<kind>_corepop
```

## Layout

| path | contents |
|---|---|
| `scripts/make_corepop_events.sh` | the 20 sets + their events-only validation |
| `scripts/bitcheck_default.sh` | default-path bit-identity (pre/post/record) |
| `scripts/core_eval.py` | core venv: density identity, marginals, truths, core loader, point evaluation |
| `scripts/sampler_check.py` | 4e6-proposal KS of the route against core's marginals |
| `scripts/detected_gof.py` | detected sets vs core x selection; host-dependent offsets and share; correspondence |
| `scripts/injection_check.py` | the injection support and weight check |
| `scripts/low_m2.py` | Analysis 14's `diag_low_m2.table` on the new files |
| `scripts/make_figure.py`, `figs/` | the figure |
| `scripts/summarize.py`, `results/corepop_summary.json` | one row per file |
| `results/*.json` | every number above |
| `cache/`, `logs/` | regenerable intermediates (core tables, figure arrays), job logs |
