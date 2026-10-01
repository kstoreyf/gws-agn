# Analysis 12 state

**RESUME HERE (2026-10-01): Analysis 12 COMPLETE and reported (REPORT.md). Both arms merged (2 × nlive 200 each). One open owner decision: extend the 12χ Δμ_χ box to [−0.05, 0.30] (the posterior reaches the 0.20 edge; ≈ 41 GPU-h).**

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

## Post-processing, prepared 2026-10-01 (owner: "continue with all of these items when the runs land")

Ready and dry-run on seed-1-only merges (the real merges overwrite `results/a12_*.json`):
`scripts/a12_compare.py` → `diagnostics/a12_comparisons.json`; `scripts/make_figures.py` →
`figs/fig_12.{pdf,png}`. When 1348935 / 1348936 exit:

1. `mv ../analysis_11_free_common_population/{results,queue}/a11_ns_12*_s2.* results|queue/`
   and the `a12_*_s2_*` logs into `logs/`.
2. `python scripts/a12_sampler.py --stage merge --out 12M --runs results/a11_ns_12M_dynesty_n200_s{1,2}.json`
   (and 12chi).
3. `a12_compare.py`, `make_figures.py`; write REPORT.md, GATES rows, A11 REPORT §§10–11, then the
   brief's §31 stop (owner-gate line + 15-line summary with the recommended calibration model).
4. Commit and push.

Realised-draw finding (from `a12_compare.py`): the per-event chi_eff noise (sd 0.155, pull sd 1.01,
calibrated) exceeds the intrinsic width (0.10), so σ_χ is a deconvolution. In this draw the GAL
branch's observed spread is 0.176 against an expected 0.184; the moment deconvolution gives
0.082 ± 0.011 (AGN 0.099 ± 0.013). Seed 1's σ_χ = 0.081 matches this draw's noise realisation. The
observed AGN−GAL mean difference is 0.100, so the Δμ_χ rise to 0.16 is the width–offset trade
(ρ −0.56), not the draw.

## Complete (2026-10-01)

Seed 2: 12M 1348935 (3,499 iterations, 22,898 calls, 19.1 h), 12χ 1348936 (3,717 iterations,
22,759 calls, 18.9 h), both clean. Files moved here from the A11 directory, merged
(`results/a12_{12M,12chi}.{json,npz}`), compared (`diagnostics/a12_comparisons.json`), figure
`figs/fig_12` (densities clipped at the prior box, edge marked). REPORT.md written.
