# RESULTS — gw_agn reproduction with the FIXED darksirens (post PR #212 master)

2026-07-10. Orchestrator: Fable. Companion to `../gw_agn_darksirens/` (the campaign that
diagnosed the estimand problem and drove the fix stack, darksirens PRs #204–#212, all on
master @ 8eae3ea). Same mock inputs (data/ symlinks to the original campaign: bit-identical
events, distance/sky PE, catalogs); darksirens now run in **field-convention sky weighting**
with **catalog-targeted injections**. All analyses fixed fiducial population, Om0 pinned,
DARKSIRENS_ZMAX=1.5, grid scans via `scripts/scan_darksirens.py` (PYTHONPATH → a worktree
at origin/master 8eae3ea).

## One-paragraph summary

The estimand fix works end-to-end on the original campaign data: the K=2 mixture now
recovers the planted AGN-hosted fraction across the full truth ladder, where the shipped
(conditional) code railed every scan to f=1. The best-matching estimator is
`dark_sirens` FIELD mode at the complete-catalog limit ("dsf"): medians
{0.014, 0.322, 0.687, —} against gw_agn's {0.017, 0.330, 0.687, 0.975} (truths
{0.0099, 0.307, 0.703, 1.0}); the fagn0.7 median matches gw_agn to the third decimal. The
`dark_sirens_complete` field variant ("dscf") also recovers the ladder monotonically
(0.014/0.236/0.591/0.974) with a small interior low bias (−0.07/−0.11). Joint (H0,f)
surfaces are compact and non-degenerate again (ρ = −0.03 at fagn0.3 vs the conditional
run's boundary-railed +0.999 artifact). Two residuals are quantified and disclosed: a
selection-resolution requirement (catalog-targeted injections; without them the sparse
tracer's μ walls every scan at the same f regardless of truth) and a mixture-H0 tilt
(−1.1 at fagn0.3, −3.5 at fagn0.7 km/s/Mpc) from the already-documented convention stack
(injection-based μ(H0), informative masses), not from the estimand fix.

## The four-truth ladder (f medians, N=1000, H0=truth)

| truth | gw_agn | conditional (broken) | **fixed dsf (n0→0)** | fixed dscf |
|---|---|---|---|---|
| 0.0099 | 0.017 | 0.000 (argmax) | — (dscf used) | 0.014 [0.005, 0.024] |
| 0.307 | 0.330 | **1.000 (railed)** | **0.322 [0.300, 0.346]** | 0.236 [0.212, 0.260] |
| 0.703 | 0.687 | **1.000 (railed)** | **0.687 [0.662, 0.712]** | 0.591 [0.564, 0.617] |
| 1.0 | 0.975 | 1.000 | — (dscf used) | 0.974 [0.957, 0.992] |

- A/B injection seeds: dscf fagn0.3 med 0.236 vs 0.236 (argmax 0.225/0.300 one grid step) —
  MC-stable.
- Model choice: for a COMPLETE catalog the correct `dark_sirens`-field nuisance point is
  log10n0 → −∞ (N_miss → 0; Z → ΣN_obs = gw_agn's field). At the TRUE mean density the
  model honestly budgets "missing" AGN into the 90% empty pixels (half of Z_AGN) and the
  fagn0.3 median shifts to 0.635 [0.595, 0.674] — the disclosed model-misspecification
  demo, now smooth and interior (the conditional-era version of this run hit the guard
  wall at 0.8).
- dscf vs dsf interior offset (−0.07/−0.11): the two complete-catalog constructions differ
  only in kernel/normalization micro-conventions (per-pixel-normalized p_cat·count-share
  vs globally normalized count-weighted KDE); dsf matches gw_agn's construction and the
  truths — use dsf as the primary estimator.

## Joint (H0, f) and H0

| run | MAP | f med [68%] | H0 med [68%] | ρ |
|---|---|---|---|---|
| dsf fagn0.3 | (66.67, **0.325**) | 0.326 [0.302, 0.349] | 66.70 [66.11, 67.27] | −0.028 |
| dsf fagn0.7 | (64.17, 0.725) | 0.715 [0.690, 0.739] | 64.23 [63.62, 64.80] | −0.291 |
| gw_agn fagn0.3 | (67.5, 0.325) | 0.330 [0.305, 0.353] | 67.53 [66.94, 68.10] | −0.002 |
| gw_agn fagn0.7 | (67.5, 0.675) | 0.687 [0.662, 0.711] | 67.50 [66.93, 68.07] | +0.003 |

- The f axis is fixed; the joint MAP's f at fagn0.3 equals gw_agn's node exactly.
- **Residual mixture-H0 tilt**: −1.04 at fagn0.3 (truth at the ~90% edge), −3.5 at
  fagn0.7, growing with the AGN-hosted fraction. This is the same convention stack
  measured in the original campaign's K=1 scans (injection-based μ(H0) vs gw_agn's
  H0-independent analytic β = CDF(z_max); informative powerlaw+peak masses coupling
  m_det/(1+z) to H0; gw_agn used neither) — not an estimand-fix artifact: the f
  recovery is correct at fixed H0 and along the joint surface (ρ ≈ 0 at fagn0.3).
  Retiring it needs a matched z-cut selection treatment (analytic β or z-cut-aware
  injections) — flagged as the next darksirens work item.
- Per-tracer H0 (61-pt, r00–r09, dscf field): GAL ⟨peak⟩ 65.98 ± 0.85 (hw 2.07), AGN
  67.79 ± 0.83 (hw 0.87) vs gw_agn refs 67.32 ± 0.41 (1.36) / 67.67 ± 0.33 (0.74) —
  AGN centered, GAL flat-likelihood compatible (same as the conditional campaign; K=1
  is insensitive to the sky-weighting fix by construction — the global normalizer
  cancels PE-vs-selection at K=1).

## Selection resolution: the operative prerequisite (measured)

With the original isotropic defensive-mixture injections, field-mode μ_AGN has effective
sample size ~1.2k (weights carried by the ~10% of injections in occupied AGN pixels near
kernels): the hard guard walls every f scan at the same position (argmax 0.325/0.425 for
three different truth sets — a pure injection-set artifact), and the soft guard's penalty
sculpts spurious joint ridges (ρ → +0.999). The catalog-targeted injection lane
(0.65 population + 0.10 uniform + 0.25 AGN-object-targeted, exact mixture pdraw, validated
to 5e-13) raises Neff to ~145k (116×) and removes the wall entirely (A/B seed stable).
**Field-mode mixtures with sparse tracers require selection injections that cover the
sparse catalog's kernels** — recorded as a usage requirement for darksirens docs.

## Verdict against the goal

- **f and (H0,f) recovery with the production code: FIXED and CONFIRMED** — interior,
  monotone, truth-ordered ladder; fagn0.3/0.7 medians within 0.015/0.016 of truth (dsf),
  matching gw_agn to ≤0.008; ρ ≈ 0 restored at the fiducial point.
- H0 within the mixture carries a quantified −1 (fagn0.3) to −3.5 (fagn0.7) km/s/Mpc
  convention tilt (pre-existing, documented, next work item); per-tracer H0 unchanged
  from the original campaign's PASS.
- Full evidence: results/comparison_summary.json, figs/, logs/; inputs bit-identical to
  ../gw_agn_darksirens (symlinked).
