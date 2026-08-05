# RESULTS — gw_agn reproduction on darksirens master @ `2b86a2d` (2026-07-29)

Third darksirens run in the campaign. Predecessor: `../gw_agn_darksirens_fixed/`
(master @ `8eae3ea`, the PR #212 field-weighting integration), which recovered the
α_AGN ladder and is the reference this run is checked against. Inputs are
bit-identical across both runs (`data/` symlinks to `../gw_agn_darksirens/data`:
same events, distance/sky PE, catalogs, injections). All analyses: fixed fiducial
population (powerlaw+peak + chieff), Om0 pinned, `DARKSIRENS_ZMAX=1.5`, grid scans
by module import via `scripts/scan_darksirens.py` against a worktree pinned at
`2b86a2d` (295 commits after `8eae3ea`).

## One-paragraph summary

The primary estimator is unchanged: `dark_sirens` field mode at the
complete-catalog limit ("dsf") reproduces the `8eae3ea` run **bit-for-bit** —
f medians agree to <1e-4 across the ladder, joint (H₀,f) MAPs and correlations are
identical, and the K=1 per-tracer logL agrees to 6 decimals. One disclosed
systematic is retired: the previous run's "dscf interior low bias" (−0.07/−0.11,
attributed there to a kernel-normalization micro-convention) is gone —
`dark_sirens_complete` field ("dscf") now lands exactly on dsf, and a `git bisect`
over the 295 commits pins the change to **PR #215 / `d0435b6`**, a
complete-catalog volume double-count. One open item is unchanged: the mixture-H₀
tilt (−1.07 at fagn0.3, −3.57 at fagn0.7 km/s/Mpc). The consequential new fact is
a **guard**, not a number: master enforces the GWTC-4.0/5.0 bound
σ²_lnL = Σσ²_PE + N_obs²/N_eff ≤ 1, and on this mock the N=1000 mixture carries
σ²_lnL ≈ 14–48, so every K=2 configuration is rejected outright (logL = −∞
everywhere). Because σ²_PE depends on the parameter being inferred, at event counts
where the guard *partially* admits the grid it truncates in f and biases the
recovered fraction low by up to −0.33 — the estimator itself is unbiased on the
same data at an inert budget (+0.023). That interaction is the main finding to
carry upstream.

## Arm design (why there are four)

For an ADMITTED cell the guard is a pure gate — it does not enter the returned
logL (verified: Arm G vs Arm L max|ΔlogL| = 0 on co-admitted cells). That is what
makes a like-for-like comparison with the `8eae3ea` numbers possible at all.

| arm | budget | purpose |
|---|---|---|
| **L** | `--max_likelihood_variance 1e6` (criterion inert, guard collapses to the legacy `N_eff > 5·N_obs` floor exactly) | code-drift comparison against `8eae3ea` |
| **G** | default 1.0 | scans at master's real setting; admitted configs in full + rejected-by-design records |
| **N** | default 1.0, on stratified event subsamples | the guard-compliant mixture answer, plus its A/B at an inert budget |
| **LI** | inert, isotropic injection set | matches the injection set the `8eae3ea` run's *published per-tracer* numbers used |

## 1. The primary estimator did not move

f medians, 41-point scans at H₀ = 67.74, catalog-targeted injections:

| truth | this run (dsf) | `8eae3ea` (dsf) | Δ | gw_agn |
|---|---|---|---|---|
| 0.0099 | 0.0195 | — (not scanned) | — | 0.017 |
| 0.307 | **0.3221** | **0.3221** | −0.0000 | 0.330 |
| 0.703 | **0.6872** | **0.6872** | −0.0000 | 0.687 |
| 1.0 | 0.9750 | — (not scanned) | — | 0.975 |

This run completes the ladder at 0.0099 and 1.0, which the previous run left as
"—" (it used dscf there). Joint (H₀,f) 61×41 grids are identical too:

| run | MAP (H₀, f) | ρ | `8eae3ea` MAP, ρ |
|---|---|---|---|
| dsf fagn0.3 | (66.67, 0.325) | −0.028 | (66.67, 0.325), −0.028 |
| dsf fagn0.7 | (64.17, 0.725) | −0.291 | (64.17, 0.725), −0.291 |

Single-coordinate cross-check at the truth point (H₀=67.74, f=0.307, n0=−12):
logL_dsf = −5923.238846 (`8eae3ea`) vs −5923.238847 (`2b86a2d`).

## 2. A disclosed systematic is retired — bisected to PR #215

The `8eae3ea` run disclosed: *"dscf has −0.07/−0.11 interior low bias vs dsf
(kernel-normalization micro-convention) — dsf is primary."* On master, dscf
agrees with dsf exactly:

| truth | dscf now | dscf at `8eae3ea` | shift | dsf now |
|---|---|---|---|---|
| 0.0099 | 0.0195 | 0.0141 | +0.005 | 0.0195 |
| 0.307 | 0.3221 | 0.2364 | **+0.086** | 0.3221 |
| 0.703 | 0.6872 | 0.5906 | **+0.097** | 0.6872 |
| 1.0 | 0.9750 | 0.9740 | +0.001 | 0.9750 |

Joint MAP_f moves with it: 0.225 → 0.325 (fagn0.3), 0.625 → 0.725 (fagn0.7), and
ρ becomes dsf's.

`git bisect` on the scalar M = logL_dscf − logL_dsf at the truth coord
(`scripts/bisect_metric.py`; M = −38.27 at `8eae3ea`, +0.0007 at `2b86a2d`)
identifies **`d0435b6` — "Fix complete-catalog volume double-count: unit-mass
kernels for dark_sirens_complete" (PR #215)**. `dark_sirens_complete` built its
kernels with `volume_weighted=True`, multiplying each galaxy's weight by dV_c/dz on
top of counts that already track hosts per redshift shell, so any catalog whose
dN/dz follows comoving volume got an effective prior ∝ (dV_c/dz)². This campaign's
GLASS mock is an independent end-to-end confirmation of that fix, at both K=1
(logL −623.090557 → −615.389273, matching dsf) and K=2 (dscf medians onto dsf).

The previous campaign's "micro-convention" attribution was therefore too mild —
it was a genuine volume double-count, and it is fixed.

## 3. Unchanged open item: the mixture-H₀ tilt

Joint MAP_H₀ is 66.67 at fagn0.3 (−1.07 vs truth 67.74) and 64.17 at fagn0.7
(−3.57) — the same tilt the `8eae3ea` run measured (−1.04 / −3.5). None of the 295
commits addressed it. Its attribution stands from the earlier campaigns:
injection-based μ(H₀) vs gw_agn's H₀-independent analytic β = CDF(z_max), plus
informative powerlaw+peak masses coupling m_det/(1+z) to H₀ under a hard true-z
selection cut. Retiring it needs a matched z-cut selection treatment (analytic β
or z-cut-aware injections). Still the next darksirens work item.

## 4. The new total-variance guard, measured (`GUARD_AUDIT.md`)

Master admits a cell only if `N_eff > max(5·N_obs, N_obs²/(V − Σσ²_PE))` with
`V = max_likelihood_variance` (default 1.0), i.e. σ²_lnL ≤ V — the GWTC-4.0/5.0
criterion added in PR #217/`d46f592`, after the `8eae3ea` baseline. The previous
run faced only the legacy `N_eff > 5·N_obs` floor, which every configuration here
still passes.

Measured at the truth coordinate of each configuration:

| configuration | N_obs | N_eff | Σσ²_PE | N²/N_eff | σ²_lnL | admitted? |
|---|---|---|---|---|---|---|
| GAL K=1 (×10) | 100 | 1.06e5 | 0.40–0.68 | 0.094 | **0.49–0.77** | yes |
| AGN K=1 (×10) | 100 | 1.47e5 | 6.7–12.7 | 0.068 | 6.8–12.7 | no |
| K=2 fagn0.0 | 1000 | 1.08e5 | 4.66 | 9.22 | 13.9 | no |
| K=2 fagn0.3 | 1000 | 2.02e5 | 16.3 | 4.96 | 21.3 | no |
| K=2 fagn0.7 | 1000 | 2.51e5 | 43.7 | 3.98 | 47.7 | no |
| K=2 fagn1.0 | 1000 | 1.47e5 | 93.3 | 6.78 | 100 | no |
| K=2 fagn0.3, isotropic inj | 1000 | 9.28e3 | 16.3 | **108** | 124 | no |

Three things this says:

1. **The binding term is per-event PE reweighting variance, not the injection
   set** — Σσ²_PE dominates for every K=2 configuration. At 2000 PE samples per
   event the effective sample size is ~61 (fagn0.3), so Σσ²_PE ≈ 16 at N=1000.
2. **It scales with the sparse tracer.** Σσ²_PE runs 4.66 → 16.3 → 43.7 → 93.3
   across the ladder: the AGN component's spiky catalog KDE is what concentrates
   the PE weights. The tracer that delivers the 2× H₀ payoff is the one that
   spends the variance budget.
3. **The catalog-targeted injection requirement is independently reconfirmed**,
   now through the guard: the isotropic lane's selection term is 108 vs 4.96
   (N_eff 9.3e3 vs 2.0e5), a 22× improvement from the catalog-targeted lane.

For the admitted GAL configurations the truncation is harmless: the rejected cells
sit at ΔlogL ≤ −5.7 from the peak and carry 6.4e-4 of the posterior mass, so the
default-budget and inert-budget H₀ posteriors agree to 3 decimals.

## 5. The guard × inference interaction (the finding to carry upstream)

σ²_PE is a function of θ — it grows as the mixture leans on the sparse component.
A per-cell hard wall therefore truncates the grid **in the parameter being
inferred**, which is not a reliability filter but a parameter-dependent prior.
Measured on stratified N=50 subsamples (host-fraction preserving,
`scripts/build_event_subsample.py`), same data, only the budget differing:

| N=50 subsample | budget | cells admitted | median | truth | Δ |
|---|---|---|---|---|---|
| fagn0.3 | inert | 41/41 | 0.2813 | 0.30 | −0.019 |
| fagn0.3 | **default** | 10/41 | 0.1923 | 0.30 | **−0.108** |
| fagn0.7 | inert | 41/41 | 0.7226 | 0.70 | +0.023 |
| fagn0.7 | **default** | 16/41 | 0.3671 | 0.70 | **−0.333** |

At N=50 the estimator recovers truth on this mock (−0.019 / +0.023, consistent
with gw_agn's σ(α_AGN) = 0.086 forecast at N=50). Under the default budget the
surviving low-f region yields −0.108 / −0.333. The joint at N=50 admits 744/2501
cells and gives f median 0.2147 (truth 0.30).

At N=25 the guard admits most of the grid (32/41 and 30/41) and the recovery is
unbiased within the sample noise: 0.4446 vs truth 0.32 (+0.125) and 0.6214 vs 0.72
(−0.099), against an expected σ(α_AGN) ≈ 0.12 at N=25.

**The truncation is a hard upper cut on f that can exclude the truth.** Admitted
f ranges: [0.00, 0.23] at fagn0.3 (truth 0.30) and [0.00, 0.38] at fagn0.7 (truth
0.70). The true value is outside the admitted parameter space entirely, so the
posterior cannot cover it by construction — the median lands just below the cut.
This is why the deficit grows with the truth (−0.108 → −0.333).

### Where the variance is spent (and what would fix it)

Per-event σ²_i distribution at N=1000, fagn0.3, captured directly
(`scripts/diag_variance_guard.py --capture_event_vars`,
`results/event_variance_fagn0.3.json`):

| n | Σσ²_i | mean | median | p90 | p99 | max | top-1% share |
|---|---|---|---|---|---|---|---|
| 1000 | 16.31 | 0.0163 | 0.0110 | 0.0331 | 0.0938 | 0.536 | 11.9% |

The distribution is right-skewed (max = 49× the median) but the budget is spent
**collectively, not by a few outliers** — the worst 1% of events carry only 12%, so
removing outliers cannot rescue the run. Two consequences:

- **The remedy is global PE resolution, not targeted resampling.** σ²_i ∝ 1/n_samp,
  so Σσ²_PE = 16.3 → <1 needs ≳17× more PE samples, ≈34k per event (this campaign
  stores 2000). Alternatively fewer events.
- **Scale check on the budget:** just **3 events** are enough to exhaust the entire
  1.0 nat² budget. A 1000-event catalog at this PE resolution is nowhere near the
  GWTC-4/5 MC standard, and that is a statement about the mock's PE clouds, not
  about the estimator.

The admissible event count is therefore not a clean threshold: N = 25/30/35/40/45
all pass (σ²_lnL 0.37/0.48/0.60/0.44/0.62) while N=50 fails (1.25), because the
stratified draws select different events and the skewed σ²_i distribution makes the
sum sample-composition dependent at fixed N. On this mock the mixture is
comfortably admissible up to N ≈ 45 and marginal by N ≈ 50.

**Recommendation for darksirens:** the total-variance criterion is sound as a
run-level admissibility check, but applied cell-by-cell across a θ scan (or inside
a sampler) with a θ-dependent σ²_PE it silently reshapes the posterior. Either
apply it once per run at a reference θ, or surface the admitted fraction so that a
truncated posterior cannot be mistaken for a converged one. `scan_darksirens.py`
here records `n_neginf_cells` and `all_cells_rejected` for exactly that reason.

## 6. Per-tracer K=1 H₀ (Arm LI, injection set matched to the previous run)

61-point scans, H₀ ∈ [50,100], quadratic-refined argmax, ⟨·⟩ ± sem over
realizations; isotropic injection set, matching what the `8eae3ea` run's published
per-tracer numbers used (Arm L's per-tracer scans use the catalog-targeted lane and
are NOT comparable to them — mixing the two was an error made and corrected during
this run).

| model | tracer | n | this ⟨peak⟩ | `8eae3ea` ⟨peak⟩ | Δ | this hw | prev hw | gw_agn ref |
|---|---|---|---|---|---|---|---|---|
| dscf | GAL | 10 | 65.51 ± 0.99 | 65.98 ± 0.85 | **−0.47** | 2.27 | 2.07 | 67.32 (1.36) |
| dscf | AGN | 10 | 67.76 ± 0.85 | 67.79 ± 0.83 | −0.03 | 0.83 | 0.87 | 67.67 (0.74) |
| dsf | GAL | 5 | 65.30 ± 1.46 | 65.30 ± 1.46 | **−0.000** | 2.54 | 2.54 | 67.32 (1.36) |
| dsf | AGN | 5 | 65.80 ± 0.45 | 65.80 ± 0.45 | **−0.000** | 0.79 | 0.79 | 67.67 (0.74) |

dsf is bit-identical. dscf moves for **GAL only** (−0.47, hw +0.20), not AGN
(−0.03) — precisely what PR #215 predicts: the double count scaled with how closely
a catalog's dN/dz follows comoving volume, and the dense GAL catalog tracks volume
while the sparse clustered AGN catalog does not. The campaign's standing per-tracer
conclusion is unchanged: AGN is centered on truth and ~2.7× sharper than GAL
(hw 0.83 vs 2.27), GAL sits low within a near-flat likelihood.

## What these claims cost (conventions/assumptions)

Conditional scans at true nuisance values (delta = 0, sigma_kde = 0; log10n0 → −12
for the dsf complete-catalog limit), not marginalized posteriors; flat-prior
trapezoid grid summaries; comparisons at CI level except where bit-identity is
claimed and shown. One catalog seed. `use_LSS` off, so `b_miss` is no longer a
sampled dimension on master (PR #308's survey-block registry) — verified inert:
b_miss enters only through `(1 + α_miss·b_miss·δ_g)` with the all-zero δ_g dummy,
so the two runs compare identical models. Subsample truths differ from parent
truths by rounding and are scored against their own `n_agn/N`.

## Verdict against the goal

- **Reproduction of the `8eae3ea` result on current master: CONFIRMED, bit-level**
  for the primary estimator (dsf) across f scans, joint grids, and K=1 logL.
- **One systematic retired, attributed:** dscf's interior low bias was a
  complete-catalog volume double-count, fixed by PR #215 — independently
  confirmed here end-to-end.
- **One open item unchanged:** the mixture-H₀ tilt (−1.07 / −3.57).
- **One new blocker for this mock at N=1000:** master's total-variance guard
  rejects every K=2 configuration (σ²_lnL 14–48 vs 1.0), driven by per-event PE
  resolution (2000 samples/event, Σσ²_PE = 16.3 at fagn0.3). Running the mixture at
  GWTC-4/5 MC standards needs ≈17× more PE samples per event (~34k) or N ≲ 45
  events. The cost is collective, not outlier-driven (top 1% of events = 12%).
- **One design issue to report upstream:** because σ²_PE depends on θ, per-cell
  hard rejection acts as a parameter-dependent prior. At N=50 it admits only
  f ≤ 0.23 (truth 0.30) and f ≤ 0.38 (truth 0.70), excluding the truth from the
  parameter space and biasing the recovered fraction to −0.108 / −0.333 — where
  the same data at an inert budget gives −0.019 / +0.023.

Full evidence: `GUARD_AUDIT.md`, `results/comparison_summary.json`, `figs/`,
`logs/`; per-scan provenance in every `results/*.h5` (darksirens sha, device,
budget, labels, base coord, timings).
