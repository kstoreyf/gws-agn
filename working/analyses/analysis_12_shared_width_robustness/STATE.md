# Analysis 12 state

**RESUME HERE (2026-10-01): seed 1 of both arms finished; seed 2 of both RUNNING (rita 1348935 12M, 1348936 12χ). Those two jobs were launched before this directory existed: they write `../analysis_11_free_common_population/{queue,results,logs}/a11_ns_12*_s2*` and `a12_*_s2_*` logs. Move those files here when the jobs exit, then merge with `scripts/a12_sampler.py --stage merge`. A resubmission must go through `scripts/submit_a12_gpu.sbatch` + `a12_sampler.py` after moving the s2 checkpoint here.**

## Launch (2026-09-29, owner-released)

- Analysis 12 wiring: `a11_likelihood.build_a11(shared_width=("sigma_G",)|("sigma_chi",))`
  releases the base σ label on top of the A11 pair. Catalogue 2 has no σ copy (the
  per-catalogue block is only G.mu_c2, mu_chi_c2), so the width is common to both branches.
  Sampler problems `12M` (11D + σ_G ∈ [1, 10]) and `12chi` (11D + σ_χ ∈ [0.01, 1]), the
  darksirens production bounds (GAUSS_SIGMA, CHI_SIGMA).
- Closure (job 1348932, 6 min, `diagnostics/a12_closure.json`): K1 8/8 bitwise, K2 live at
  f = 0, 0.266, 1, K3 7 guard rejections, all in the wide-σ_χ corner, ≥ 98 nats down. 3.00 s
  per evaluation, the same as A11.
- Runs: dynesty multi/unif, nlive 200, first_update_min_eff 100, 96 h limit; 12M s1 1348933,
  12χ s1 1348934 (one A100 each; rita has two), s2 1348935/1348936 queued behind them. Merge
  each arm with `--stage merge --runs ... --out 12M|12chi`. Estimate 18–25k calls per run,
  ≈ 70–85 GPU-h in all, about two days of wall time.
- The brief's "do not run 12M and 12χ simultaneously" is read as never freeing both widths
  in one fit. The two one-width fits run side by side on the owner's instruction.
- Seed 1 finished: 12M 1348933 (3,502 iterations, 25,007 calls, 21.0 h), 12χ 1348934 (3,692
  iterations, 26,400 calls, 22.1 h). Seed-1-only numbers (preliminary, not for quoting):
  σ_G 4.70 [4.24, 5.12], Δμ_G 4.48 [2.95, 5.92]; σ_χ 0.081 [0.068, 0.096], Δμ_χ 0.160
  [0.124, 0.191].
- 2026-10-01: moved here from the Analysis-11 directory (scripts, closure record, seed-1
  runs and logs). `a11_sampler.PREFIX` lets this directory's merges write `a12_*`.
