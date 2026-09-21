# Analysis 10 state

## Current status

**STOPPED at specification §9, before generation** (2026-09-21).

The pre-registered mark-admissibility check fired. Under the production
`powerlaw+peak` parameterisation the Gaussian-peak mixture fraction has fiducial
\(\lambda_{\rm peak}^{\rm fid} = 0.90\), so the registered mark gives

\[
  \lambda_{\rm peak}^{\rm AGN} = 0.90 + 0.15 = 1.05 \not\in (0, 1).
\]

The specification's own instruction applies: **STOP and report the real fiducial
value; do not silently alter the registered mark.** The mark was not altered,
nothing was generated, and the analysis waits on an owner decision (`REPORT.md`
carries the options, none executed).

## The verified facts

### 1. The twelve `powerlaw+peak` slots

Read off `model.param_specs` and `pop_model_prior_parser`, not retyped
(`shared_beta = shared_spin = shared_gamma = True`):

| i | name | label | fiducial | bounds |
|---|---|---|---|---|
| 0 | `v1` | `$v_1$` | **0.1** | [0, 1] |
| 1 | `PL.alpha` | `$\alpha_{\rm PL}$` | 2.3 | [−4, 6] |
| 2 | `PL.m_min` | `$m_{\min,\rm PL}$` | 5 | [2, 10] |
| 3 | `PL.m_max` | `$m_{\max,\rm PL}$` | 80 | [50, 100] |
| 4 | `PL.dm_min` | `$\delta m_{\min,\rm PL}$` | 3 | [0.01, 10] |
| 5 | `PL.dm_max` | `$\delta m_{\max,\rm PL}$` | 10 | [0.01, 20] |
| 6 | `G.mu` | `$\mu_{\rm G}$` | 35 | [20, 50] |
| 7 | `G.sigma` | `$\sigma_{\rm G}$` | 5 | [1, 10] |
| 8 | `beta` | `$\beta$` | 1.0 | [−2, 7] |
| 9 | `mu_chi` | `$\mu_\chi$` | 0.0 | [−1, 1] |
| 10 | `sigma_chi` | `$\sigma_\chi$` | 0.1 | [0.01, 1] |
| 11 | `gamma` | `$\gamma$` | 0.0 | [−10, 10] |

**There is no \(\lambda_{\rm peak}\) slot.** The mixture fraction is not sampled
directly; slot 0 is a stick-breaking input.

### 2. The stick-breaking map

`darksirens/gw/populations/base.py::_stick_breaking_weights` maps \(k-1\) inputs
\(v\) to \(k\) weights. At \(k = 2\) in composition order (PowerLaw, Gaussian):

    v = [0.1]  ->  w = [0.1, 0.9]

The inverse, `darksirens/gw/populations/grammar.py::_w_to_v`, returns `[0.1]`
from `[0.1, 0.9]` with a round-trip error of exactly 0.0. Hence

\[
  \lambda_{\rm peak} \equiv w_{\rm G} = 1 - v_1, \qquad
  \lambda_{\rm peak}^{\rm fid} = 1 - 0.10 = \mathbf{0.90}.
\]

### 3. The composition order, established twice

**From the registry.** `darksirens/gw/populations/registry.py` carries the
curated entry

    "powerlaw+peak": Curated(
        latex="PL+G",
        weights=(0.10,),                       # w_PL=0.10, w_G=0.90

and `Curated.weights` stores *desired final fractions* (the last implied as
\(1 - \sum\)), converted to stick-breaking inputs by `_w_to_v`. So the curated
intent is \(w_{\rm PL} = 0.10\), \(w_{\rm G} = 0.90\).

**Numerically.** `model.log_p_pop` on an \(m_1\) grid (2 → 100 \(M_\odot\), 4001
nodes) at fixed \(q = 0.9\), \(z = 0.1\), \(\chi_{\rm eff} = 0\), normalised over
the grid, with the mass in a peak band [25, 45] (brackets \(\mu_{\rm G} = 35\),
\(\sigma_{\rm G} = 5\)) and a power-law band [5, 20]:

| \(v_1\) | mass in [25, 45] | mass in [5, 20] |
|---|---|---|
| 0.10 (fiducial) | **0.8132** | 0.1397 |
| 0.25 | 0.6322 | 0.3168 |
| 0.90 | 0.1146 | 0.8231 |

Raising \(v_1\) moves mass **out** of the peak and **into** the power law,
monotonically. \(v_1\) is the power-law weight; component 0 is `PowerLaw`,
component 1 is `Gaussian`. The two lines of evidence agree, and the script
asserts the monotonicity rather than assuming it.

### 4. The check, in both coordinates

| quantity | value |
|---|---|
| \(\lambda_{\rm peak}^{\rm fid}\) | 0.90 |
| \(\Delta\lambda_{\rm peak}^{\rm plant}\) | +0.15 |
| \(\lambda_{\rm peak}^{\rm AGN}\) | **1.05** |
| \(0 < 1.05 < 1\) | **False** |
| sampled coordinate \(v_{1,c2} = 1 - 1.05\) | **−0.05** |
| `v1` prior bounds | [0, 1] |
| in bounds | **False** |

The failure is the same statement twice: the registered mark asks the AGN branch
for 105% of its mass in the Gaussian peak, i.e. a *negative* power-law weight.
It is not a grid or prior-width problem that a wider axis could fix — it is
outside the simplex.

### 5. Corroboration from the seed-100 mock (read-only)

`working/data/seed100/events/events_marked_dmu0p10.h5`, dataset `true_m1src`,
1000 detected events:

| set | N | median \(m_1^{\rm src}\) | in [25, 45] | below 20 \(M_\odot\) |
|---|---|---|---|---|
| all | 1000 | 36.52 | **92.4%** | 0.6% |
| GAL (`host_type = 0`) | 705 | 36.51 | 93.2% | 0.7% |
| AGN (`host_type = 1`) | 295 | 36.56 | 90.5% | 0.3% |

The detected mass distribution is already peak-dominated, as a 0.90 peak
fraction requires; there is no room above it for +0.15. The mock's own generator
record is decisive and independent of any inference code: the `metadata_json`
attribute carries `"population": {..., "peak_fraction": 0.9, ...}`. The
seed-100 dataset every analysis in this campaign is built on **was generated at
\(\lambda_{\rm peak} = 0.90\)**.

### 6. Code readiness (informative only — it decides nothing here)

`build_parameter_space`, called exactly as
`analysis_8/scripts/a8_likelihood.py::build` calls it (`n_catalogs = 2`,
`fix_population = False`, twelve base parameters pinned by LaTeX label, no data
loaded), accepts a per-catalog mass weight:

    per_catalog_pop_params=('mu_chi_c2',)            ->  9 labels
      ['H0', 'log10n0', 'delta', 'sigma_kde', 'log10n0_c2', 'delta_c2',
       'sigma_kde_c2', '$\mu_\chi$_c2', 'fcat_2']

    per_catalog_pop_params=('v1_c2', 'mu_chi_c2')    -> 10 labels
      ['H0', 'log10n0', 'delta', 'sigma_kde', 'log10n0_c2', 'delta_c2',
       'sigma_kde_c2', '$v_1$_c2', '$\mu_\chi$_c2', 'fcat_2']

The resolver does **not** refuse: it emits `$v_1$_c2` beside `$\mu_\chi$_c2` and
`fcat_2`. The inference side is ready to carry a mass mark. The blocker is the
value of the mark, not the code — and, separately, the generator (§7).

### 7. The generator has no mass-mark hook

`working/data/generate_dataset.py` (not modified, read-only) carries exactly one
branch-dependent mark: `--dmu_chi_agn`, applied to the **shared** truncated-
Gaussian spin draw so that the RNG stream, the host labels, the masses, the sky,
the distances and the detected set are untouched. Masses come from a single
shared `PopulationConfig`; `peak_fraction` is population-wide. A mass mark
therefore needs a generator change as well as an admissible value, and it cannot
be planted by the spin mark's trick of shifting a completed draw: shifting the
mixture weight changes which component each event is drawn from, hence the
masses, hence — through \(\rho_{\rm opt} \propto \mathcal{M}_{\rm det}^{5/6}/d_L\)
— the detected set itself. That is an owner decision, not a scaffolding one.

## What was NOT done

- **No mock generated.** `working/data/**` was read, never written.
- **No injections generated.** The existing `injections_targeted.h5` was not
  read, resampled or extended.
- **No likelihood built, traced or evaluated.** No `make_likelihood`, no
  `load_all_data`, no survey loaded. Zero GPU seconds; zero SLURM jobs.
- **No darksirens edit.** The checkout is clean at `af896ca`.
- **No `generate_dataset.py` edit.**
- **No commit, no push.**
- **No figure.** `figs/` and `results/` hold only `.gitkeep`.

## Provenance — as asserted by the script, not as expected

    == provenance ==
      [OK ] darksirens_head: af896cae6f3f3dd1f87dec50046e3a8228f59b39
      [OK ] darksirens_import_path: /hildafs/projects/phy230014p/magana/src/
                                    darksirens-a8/darksirens/__init__.py
      [OK ] gws_agn_head: 408990af79afd82d403d600621aa6525e97e6d6d
      [OK ] darksirens worktree clean; base 2b86a2d is an ancestor

- darksirens `af896cae6f3f3dd1f87dec50046e3a8228f59b39` in
  `/hildafs/projects/phy230014p/magana/src/darksirens-a8`, worktree **clean**,
  one local commit on the pinned base `2b86a2d`, never pushed. Same pin as
  Analyses 8 and 9, and the SHA the seed-100 marked mock was generated under
  (`darksirens_sha` in the events file's `metadata_json`).
- gws-agn HEAD `408990af79afd82d403d600621aa6525e97e6d6d`. The tree is dirty
  only outside this analysis (two untracked archives and `working/paper_codex/`
  at the repository root).
- Environment of record:

      export PYTHONPATH=/hildafs/projects/phy230014p/magana/src/darksirens-a8
      export JAX_PLATFORMS=cpu
      export PYTHONDONTWRITEBYTECODE=1
      python = .../.conda/envs/jax/bin/python

  `PYTHONPATH` is load-bearing — `DARKSIRENS_SRC` does not steer the import, the
  editable install does — so the script asserts `darksirens.__file__` resolves
  under `darksirens-a8` before it reads a single parameter spec.
- The script sets `sys.dont_write_bytecode = True` before importing
  `a8_likelihood`, so the readiness probe leaves no `__pycache__` in the
  Analysis-8 tree. Verified afterwards: no file under
  `analysis_8_marked_multitracer_H0_fagn/**` or `working/data/**` has an mtime
  later than its close.

## How to re-run the check

    export PYTHONPATH=/hildafs/projects/phy230014p/magana/src/darksirens-a8
    export JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1
    python scripts/check_mass_mark_feasibility.py

CPU, a few seconds, no GPU and no data beyond the events file's `true_m1src`.
Exit status 0 if the mark is admissible, **2** if it is not (it is 2 today). If
the owner registers a different \(\Delta\lambda_{\rm peak}\), change
`DELTA_LAMBDA_REGISTERED` at the top of the script and re-run: that is the only
edit the check needs, and it is the gate any replacement mark must clear first.
