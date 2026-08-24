# Follow-up campaign report (2026-08-23/24)

The three GPU items from the post-redo queue, owner-approved 2026-08-23, run on
HildaFS SLURM (jobs 1136816-18): the KDE-window pin, the zero-density probe,
and the m<18 seed replication at seeds 101 and 102. All six cells completed,
zero failures. darksirens `0c5b3db` throughout (fresh worktree,
provenance-guarded); data = the v3 family under
`/hildafs/projects/phy220048p/magana/gws-agn-data-v3`. Queues, logs, results
and figures live under `fu_probes/`, `fu_seed101/`, `fu_seed102/`.

## 1. KDE-window pin — bandwidth RULED OUT

`joint_m18_ns64_selection_kdepin_s100`: the a7 ns64 selection cell with
`kde_window_nsigma` 8 → 32. Checked before launch: the W=4096 cap never binds
at m<18 (max row occupancy 189 at ns32, 61 at ns64), so the traced half-width
`n_sigma × max_row(σ_eff)` — set from the pixel's own occupants — is the only
occupancy-coupled window quantity, and n_sigma=32 makes every ns64 window
provably wider than its parent ns32 window.

Result: identical to the campaign cell to float noise (Δf_median ≈ 1e-15,
ΔH0_median ≈ 2e-13, identical max logL), across a different GPU (A100-40 vs
J2 H100). The per-pixel redshift KDE bandwidth does **not** carry the
+0.028/dex occupancy residual. Next candidate per the a7 REPORT: the per-pixel
selection `N_eff` guard. Bonus: cross-machine float drift is bounded at ~1e-13,
so HildaFS and J2 results are interchangeable at any level that matters.

Verdict file: `fu_probes/results/kdepin_verdict.json`.

## 2. Zero-density probe — completion path LIVE at C ≡ 1

`joint_complete_n0m24_s100`: the a3 complete rung (selection) at
`log10 n0 = log10 n0_c2 = −24`. If the completion contribution were inert at
C ≡ 1 (max(1−C_sel) = 0 on this rung), the density fiducial could not move the
likelihood. It does:

| cell | f_AGN median | offset | H0 median | max logL |
|---|---|---|---|---|
| fiducial (−3, −5) | 0.240 | −0.055 | 69.218 | −4174.886 |
| zero density (−24, −24) | 0.273 | −0.022 | 69.217 | −4174.861 |

Δf = +0.033 against a float-noise floor of ~1e-13. The residual −0.022 at zero
density sits inside the flux-limited cells' −0.007..−0.023 range: the anomalous
part of the AGN=complete offset is **entirely** a completion contribution that
survives C ≡ 1. H0 is untouched. The mechanism hunt (why n0 enters the
selection-mode likelihood when C ≡ 1) is a darksirens code question —
brainstorm-then-wait per the owner gates.

Figure: `fu_probes/figs/fig_zero_density.{pdf,png}`.
Verdict file: `fu_probes/results/zerodensity_verdict.json`.

## 3. Seed replication — the a5 headline replicates at three seeds

Two dynesty arms per seed (selection / per_pixel, rstate 7, seed-100 fits held
fixed on purpose — they estimate universe-level LF parameters to σ(M*) ≈ 0.005
and fixing them isolates realisation scatter). Seed-100 rows are the same-SHA
`experiment_dsmaster_4d_recheck` arms.

| seed | arm | ln Z | GAL anchor (truth −3) | f_AGN (truth 0.295) | H0 |
|---|---|---|---|---|---|
| 100 | selection | −4187.48 ± 0.11 | −3.126 ± 0.291 | 0.510 ± 0.263 | 69.12 |
| 100 | per_pixel | −4205.15 ± 0.09 | −1.806 ± 0.539 | 0.384 ± 0.226 | 69.64 |
| 101 | selection | −4244.45 ± 0.11 | −3.110 ± 0.291 | 0.477 ± 0.259 | 67.56 |
| 101 | per_pixel | −4260.97 ± 0.09 | −1.964 ± 0.592 | 0.521 ± 0.217 | 67.47 |
| 102 | selection | −4205.51 ± 0.10 | −3.170 ± 0.274 | 0.463 ± 0.257 | 67.68 |
| 102 | per_pixel | −4217.14 ± 0.09 | −2.534 ± 0.586 | 0.469 ± 0.218 | 67.51 |

- **Δ ln Z (selection − per_pixel): +17.7, +16.5, +11.6.** Decisive at every
  seed, and now a same-SHA three-realisation statement rather than one
  cross-SHA number.
- **Anchor recovery replicates.** Selection: −3.13/−3.11/−3.17, all within
  0.6σ of truth with σ ≈ 0.28-0.29. per_pixel: biased shallow by +0.5 to
  +1.2 dex at every seed.
- **H0 is on truth differentially at every seed and both arms** (seed 100
  carries its known +1.4 draw offset; 101/102 sit within 0.15σ).
- **A coherent free-anchor f_AGN offset.** Selection medians 0.510/0.477/0.463
  → offsets +0.215/+0.182/+0.168, mean +0.19 with seed scatter 0.024. Each
  seed is individually ~0.6σ (posteriors are wide, σ ≈ 0.26), and the naive
  three-seed combination is ~1.3σ — suggestive, not resolved, and consistent
  with the standing statement that free-anchor f_AGN is only as well determined
  as the AGN-density prior. Worth one more seed only if f_AGN under free
  anchors becomes paper-facing.

Figure: `fu_seed10{1,2}/figs/fig_seed_replication.{pdf,png}`.

## What this changes downstream

1. The four AGN=complete cells are now interpretable: quote them with the
   completion-path caveat, or at the zero-density limit.
2. a7's residual slope is not the KDE bandwidth; the guard probe is the next
   (cheap) step if a7 stays in scope.
3. The paper's lead result (a5 anchor recovery + evidence) now has three-seed,
   same-SHA support — the exec summary's "every number in the redo is one seed"
   limitation no longer applies to it.
4. Paper edits (queue item 4) remain the only unlaunched item.
