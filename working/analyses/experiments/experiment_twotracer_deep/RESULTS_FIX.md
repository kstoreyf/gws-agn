# RESULTS_FIX — the deep two-tracer mock on the sigma_ang-fixed generator

**Every number in `RESULTS.md` that depends on the events' PE is superseded by the
`_fix` results below.** darksirens PR #335 (worktree `darksirens-oraclefix`,
commit `853ded3`) fixed the generator's sky-localisation width: the old
`sigma_ang = clip(35/rho, 1, 12)` deg was a deterministic function of LATENT true
parameters — an H0-sensitive observable (`∝ dL/Mc_det^(5/6)`) that a fixed-width sky
posterior cannot represent — measured to bias H0 by −0.49 ± 0.08 even under the exact
likelihood. The fix measures distance and masses FIRST and derives sigma_ang from the
OBSERVED amplitude.

## What was regenerated, and what provably was not

`scripts/build_twotracer_mock_fix.py` (same seed 7301, same config) redraws only the
PE stage: one observation per event through the fixed `gmd._measure`
(`widths["sigma_ang"] = None`, the sequential observable sky width), then
`gmd._posterior_samples(..., use_recorded_observation=True)`. Detection is
sky-independent (noisy SNR of the true parameters), and the script verifies — and
`results/events_fix_check.json` records — that **every truth array and the host_type
labels are bit-identical to the pre-fix mock**. The detected set is therefore
unchanged, and the catalogs, survey files and the targeted injection set
`injections_targeted_k2.h5` are reused as-is.

Convention shift to keep in mind (`results/sigma_ang_prepost.json`): the pre-fix width
came from the noisy detection SNR (projection latent included), the post-fix width from
the projection-free observed amplitude at `SNR_REF_DEFAULT = 11.5` — matching the
validated closure test (`experiment_matched_mock/scripts/oracle_bootstrap.py
--sigma_from_obs`, closure −0.06 ± 0.07). Median sigma_ang shrinks 3.32° → 1.93°
(ratio 0.63), so the post-fix mock carries MORE sky information as well as an unbiased
width; absolute widths are not comparable pre/post at fixed information.

## Post-fix numbers (results/summary_fix.json)

Guard: N_eff per f identical to pre-fix (2,170 / 4,977 / 25,198 / 65,509 at
f = 0/0.3/0.7/1.0 — the selection integral never touched the events), all pass the
5·N_obs floor; the PE reweighting variance falls (16.28 → 12.74 at f = 0.3).

| run | pre-fix | post-fix |
|---|---|---|
| f-scan N=80 | 0.2157 [0.166, 0.269] | 0.2707 [0.216, 0.325] |
| f-scan N=200 | 0.2353 [0.201, 0.272], 1.8σ low, truth outside 90% | **0.2629 [0.228, 0.299], 1.0σ low, truth inside 90%** |
| joint N=200 H0 | 66.40 [65.92, 66.91], −1.34 = 2.7σ LOW, truth outside 90% | **68.31 [67.72, 68.92], +0.57 = 0.95σ, truth inside 90%** |
| joint N=200 f | 0.2466 | 0.2652 |
| joint rho(H0,f) | −0.09 | +0.07 |
| mass adjacent to rejected cells | 0.00000 | 0.00000 (1478/3321 rejected, unchanged) |

## Reading

* **The 2.7σ-low H0 was the generator defect, as predicted.** Post-fix the joint H0
  sits +0.57 on a 0.60 half-width — inside its own 68% by 1σ and consistent with the
  documented post-fix budget (darksirens estimator overhead −0.31 ± 0.13, zero-mean
  mu_hat noise ±0.36/realisation, catalog-realisation scatter ~1 km/s/Mpc measured in
  `../experiment_twotracer_seeds`). No absolute-bias claim should be hung on a single
  realisation either way.
* **The f_AGN low offset halves but does not vanish:** −0.065 (1.8σ) → −0.037 (1.0σ)
  at N=200. The earlier hypothesis that the H0 distance-scale bias was dragging events
  off the sparse tracer's support is at least half right; what remains is within 1σ.
* rho(H0, f) stays ≈ 0; the pre-fix −0.09 → +0.07 change is noise-level.

Rebuilt events: `data_derived/twotracer_gw_events_fix.h5` (+ `twotracer_n80_fix.h5`);
scans by `scripts/run_targeted_scans_fix.sh`; summary by `scripts/summarize_fix.py`.
