# Analysis 14 — Monte Carlo accuracy of the likelihood (run R0)

**Status (2026-11-17, 05:20 UTC):** (b) running on seed 100 (rita 1414628); (a) moved to seed 103
and queued (rita 1414642), because seed 100's all-AGN event set contains an event that core gives
zero population density (section "Core and the mocks disagree below the minimum mass"). Redraw
sampler validated on CPU; reproduction of the stored Analyses 0/2 grids is each job's first step.
REPORT.md comes after the results.

## Question

Every stored analysis used 2000 posterior samples per event and one injection set, with the
variance limit on ln L off (`paper_prd/notes/GAPS.md` A1, A2; `paper_prd/runs.txt` R0). How far
do the posterior median and width move when the per-event samples, or the injection set, are
redrawn, against the posterior width itself? How many samples per event and injections must
production use?

Complete catalogs, darksirens-core e7c3007 with the historical evaluation settings (as Analysis 13),
K = 2 GAL + AGN field mixture, one shared population fixed at the mock's input (γ = 0),
`max_likelihood_variance` 1e6 (inert, as the stored runs; the variance is recorded instead):

- (a) all-AGN-hosted events (`events_pureagn.h5`) of **seed 103**, f_AGN = 1 (= the AGN-catalog-only
  likelihood): H0 on 64.5–70, step 0.05 (centred on the stored A0 median 67.25). Gate against
  `analysis_0_pure_tracer_H0/results/h0_pureagn_targeted_s103.h5`. Seed 100's (a) is retired.
- (b) mixed events (`events.h5`, input f_AGN 0.30) of **seed 100**: (H0, f_AGN) on 64–74 (step 0.25)
  × 0.05–0.55 (step 0.025). Gate against `analysis_2_.../results/joint_s100.h5`: passed (below).

Per-event samples are redrawn in memory at N_s = 2000, 8000, 32000 from the exact flat-prior
posterior of each event's recorded measurement, through the generator's own functions
(`scripts/mc_redraw.py`); redraw r at 8000 contains redraw r at 2000 (common random numbers),
32000 contains 8000. Streams: `SeedSequence([ROOT, set, r, block])` on seed 100 and
`[ROOT, set, r, block, seed]` elsewhere. Injection sets 1–5 are the generator's own `injections`
stage with new sub-seeds (seed·1000 + 900 + j), 1.5e8 draws each, at fixed samples (N_s = 8000,
redraw 0); they sit next to each dataset under `<seedS target>/mc_accuracy/inj_set<j>/seedS/injections/`.

## Reproduction of the stored results (validation 2)

| | core median | stored median | shift / half-width | 68% width ratio | 90% width ratio | ln L core − stored (bulk) | gate |
|---|---|---|---|---|---|---|---|
| (b) seed 100, H0 | 69.262 | 69.217 | +0.046 | 0.995 | 0.995 | −0.53 ± 0.10 | pass |
| (b) seed 100, f_AGN | 0.2746 | 0.2734 | +0.026 | 1.001 | 1.001 | | pass |
| (a) seed 100, H0 | 69.476 | 68.358 | +2.22 | 1.006 | 1.036 | −161 ± 5 | **fail** |
| (a) seed 103, H0 | (job 1414642) | 67.254 | | | | | |

## Core and the mocks disagree below the minimum mass

The seed-100 (a) failure is one event, not a setting of this driver and not a Monte Carlo effect.

- **Not the driver.** Core with the AGN catalog alone (K = 1, field weighting) gives the same ln L
  shape as the K = 2 binding at f_AGN = 1 to 1e-11 at six H0 cells; core's default conditional
  weighting differs only slightly; both disagree with the stored legacy values in the same way
  (`results/diag_repro_a_pureagn.json`; stored ln L at 67.75 is −3929.2467, core K = 1 −4092.8166).
  The two events files carry the same flags (`chi_eff_in_p_pe` True in both, same p_pe convention),
  and the stored A0 and A2 runs passed the same options (field weighting, γ fiducial, guard).
- **One event.** In core, event 495 of seed 100's all-AGN set has ln evidence −176.5 at the reference
  cell (H0 67.75, f = 1) with Monte Carlo variance 0.9995 (one sample carries it), against a median
  of −15.45 for the other events (second lowest −23.07). Its true secondary mass is 3.37 Msun
  (m1src 33.7, q 0.10, χ_eff 0.072, z 0.063); 1998 of its 2000 posterior samples have m2src < 5
  (largest 6.04).
- **The population densities** (`results/diag_low_m2.json`; powerlaw+peak at the mock's input,
  γ = 0, identical θ in both codes):

  | ln p_pop at | core e7c3007 | legacy 2b86a2d |
  |---|---|---|
  | event 495's truth (m2 = 3.37) | −∞ | −3.215 |
  | same m1, m2 = 4.9 | −∞ | −2.840 |
  | same m1, m2 = 5.1 | −31.72 | −2.800 |
  | same m1, m2 = 6.0 | −4.295 | −2.636 |
  | a typical binary (35, 30), χ 0, z 0.1 | −0.765 | −0.799 |
  | finite over event 495's 2000 samples | 2 | 1999 |
  | ln (mean p_pop over its samples) | −15.33 | −5.68 |

  Core cuts (and tapers over δm_min = 3) the secondary of the Gaussian-peak component at the
  power-law low-mass edge m_min = 5 (`population/base.py:1146-1168` in the core worktree, a fix
  made after 2b86a2d; `paper_prd/notes/LIKELIHOOD_MATH.md` §7 item 1). The mock was drawn with the
  legacy 2b86a2d samplers, which put the peak's secondary down to the quadrature floor, and the
  legacy likelihood agrees with the generator. So the mock and core describe different populations
  below 5–8 Msun in m2, and an event there drives the core likelihood.
- **How common.** Over 297 event sets on disk (seeds 100–199 × events, pureagn, puregal), 214 contain
  an event with true m2src < 5 and 117 contain at least one event with more than 95% of its samples
  below 5 (events.h5 35, pureagn 42, puregal 40). Seed 100's events.h5 has none above 95% (its worst
  event has 73.5% below 5, true m2 5.35), which is why (b) reproduces; seed 103's all-AGN set has
  smallest true m2src 6.22 and at most 8.1% of any event's samples below 5.
- **Consequence beyond this analysis:** any core rerun of the campaign's mocks (runs.txt R1–R10)
  meets this on about a third of the event sets. Either the mocks are regenerated with core's
  population, or core's likelihood is run with the generator's pairing; that choice is the owner's.

## Layout

| path | contents |
|---|---|
| `scripts/mc_redraw.py` | the redraw sampler (imports `generate_dataset.posterior_samples_v3`, `p_pe_v3`) |
| `scripts/validate_redraw.py` | validation 1 (CPU): redraws vs stored samples (KS), p_pe closed form, CRN nesting, loader → `results/validation_redraw.json` |
| `scripts/make_injection_redraws.py`, `submit_injection_redraws.sbatch` | the five injection sets per seed (CPU job; `--seed`) |
| `scripts/a14_mc_accuracy.py`, `submit_a14_gpu.sbatch` | the GPU driver: build, reproduction check (stops on failure), timing, adapted plan, scans; `--configs`, `--seed-a` |
| `scripts/a14_post.py` | grid posterior summaries (median, 68%/90% equal-tailed intervals) |
| `scripts/aggregate_mc_accuracy.py` | → `results/mc_accuracy_summary.json` |
| `scripts/make_figures.py` | → `figs/mc_accuracy.{pdf,png}` |
| `scripts/diag_repro_a.py`, `scripts/diag_low_m2.py` | the seed-100 (a) diagnosis (CPU) → `results/diag_repro_a_pureagn.json`, `results/diag_low_m2.json` |
| `results/repro_{b,a,a_s103}.json` | reproduction of the stored A0 / A2 posteriors (validation 2) |
| `results/timing[_a_s103].json`, `results/plan_{b,a_s103}.json` | measured s/call per N_s and the adapted task lists |
| `results/scans/<task>.{h5,json}` | one per scan: grid, ln L, PE variance sum, N_eff, ln μ, selection term; summary |
| `logs/` | job logs (git-ignored) |

Task names: (b) `b_pe_n<N_s>_r<r>` (sample redraw r, record injections), `b_inj<j>_n8000_r00`
(injection set j), `repro_b`; (a) on seed 103 the same with prefix `a_s103` (`a_s103_pe_n2000_r00`,
`repro_a_s103`). `repro_a` is the failed seed-100 (a) gate, kept as the record of the diagnosis.

## How to resume

Resubmit identically; finished scans (`results/scans/*.json`) are skipped, and a passed reproduction
is not repeated:

    cd working/analyses/analysis_14_monte_carlo_accuracy
    sbatch --export=ALL,A14_ARGS=--configs=b scripts/submit_a14_gpu.sbatch
    sbatch --time=03:00:00 --export=ALL,A14_WALL_HOURS=3,A14_ARGS="--configs=a --seed-a=103 --plan-hours=2.5" scripts/submit_a14_gpu.sbatch

Missing injection sets: `sbatch [--export=ALL,SEED=103] scripts/submit_injection_redraws.sbatch`
(skips sets on disk). Aggregation and figure (CPU, seconds):

    PY=/hildafs/home/magana/tmp_ondemand_hildafs_phy230014p_symlink/magana/.conda/envs/jax/bin/python
    $PY scripts/aggregate_mc_accuracy.py && $PY scripts/make_figures.py

## Runs

| job | what |
|---|---|
| HENON 1414624 `agn-mcinj` | seed-100 injection sets 1–5 (done; 1414615, the same at 48 CPUs, cancelled while pending) |
| RITA-GPU 1414623 `agn-mcacc` | both configurations; stopped itself at 04:52 (exit 3): (b) gate passed, (a) seed 100 failed |
| RITA-GPU 1414628 `agn-mcacc` | (b) only, seed 100, 15 h; running since 05:00 |
| HENON 1414641 `agn-mcinj` | seed-103 injection sets 1–5 |
| RITA-GPU 1414642 `agn-mcacc` | (a) only, seed 103, 3 h (plan 2.5 h); queued |
