# Analysis 13 state

**RESUME HERE: see the last entry of the log below (2026-10-05/06: rslice resume, rita 1361503).** Earlier header (2026-10-03): A13 seed 1 RUNNING on darksirens-core bf58aa6 (post-#54; pinned, bitwise-rechecked), rita job 1351377 (`sbatch --export=ALL,SEED=1 scripts/submit_a13core_gpu.sbatch`; resubmit identically to resume from `queue/a13core_*.save`). The af896ca dynesty run (rita 1350718) was cancelled at 1,852+ iterations (checkpoint kept, not used). Second smoke sampler still DEFERRED. Production (second seed) and calibration remain owner-gated; core main e7c3007 (#52 defaults, #53) is NOT adopted — move deliberately before calibration.**

## Setup (2026-10-01)

- Owner request: set up A13; consult the tinyns session on settings; smoke test dynesty against
  tinyns on one GPU, seed 100, comparing logZ and posteriors; gate the two-seed run; defer
  calibration until after the two seeds.
- Problem `13` added to `a11_sampler.BOXES` (11D + σ_G [1, 10] + σ_χ [0.01, 1], Δμ_χ to 0.30).
- Second sampler (tinyns / Nautilus advice from the tinyns session): REMOVED by the owner
  2026-10-06 — dynesty only. The advice is in git history (SAMPLER_NOTES.md, deleted at that date).
- Jobs: closure 1350473; dynesty smoke = seed 1 of production, 1350474 (afterok closure; it doubles
  as production seed 1 if the smoke passes).

## Restart on darksirens-core (2026-10-02)

Owner: "Restart on core"; "New runs on core" (finished analyses keep their results with the code
version recorded); second sampler still deferred. Core b47e41c is validated for this likelihood
(A11 grids: KS ≤ 0.044, 90% widths within 0.2%, 12× faster; experiment_darksirens_core_equivalence).
The core run uses the same coordinates, boxes and dynesty settings as problem 13 and a pre-flight
that reproduces the measured core A11 cells to 1e-8 at the fiducial widths. The af896ca run's
checkpoint (`queue/a11_ns_13_dynesty_n200_s1.ckpt`, cancelled at ~30k calls, dlogz 27) is kept but
not resumed.

## Checkpoint fix and Jetstream2 (2026-10-03/04)

- rita 1351377 (core bf58aa6, seed 1) passed its pre-flight (4/4 cells |d| = 0) and then died at the
  first dynesty checkpoint (900 s): dynesty pickles the sampler, and `loglike` is a closure
  ("Can't pickle local object"). Fixed: checkpoints now go through core's
  `darksirens.inference.dynesty_checkpoint` (state-only save; the callables are rebound on restore).
  A toy run killed at iteration 302 and resumed reproduces the uninterrupted logZ exactly
  (653 iterations both).
- Owner moved the run to the Jetstream2 A100 VM (rita busy): `scripts/js2/` (README there). The
  driver takes `A13_DATA`, `A13_REF_CELLS`, `A13_KERNEL_LAYOUT`, `A13_MISSING_DENSITY` from the
  environment; the defaults are the rita values, so `submit_a13core_gpu.sbatch` is unchanged.
- The VM's GPU is a 20 GB vGPU slice. The likelihood build ran out of device memory (a 1.43 GB
  allocation in the pinned field-kernel build) with the default 75% JAX cap (15.8 GB), with the cap at
  95% (18.7 GB), and with 95% plus the galaxy-list/gather layouts (17.1 GB). Host RAM peaked at ~14 GB.
- Fourth try, platform allocator (no BFC cache) plus the layouts: it got past that allocation and then
  requested a single **27.4 GiB** buffer in the same step (`build_pinned_catalog_kernel` →
  `_state(catalog)`, the field compact view). **A13 cannot run on a 20 GB GPU with core bf58aa6**;
  it needs an 80 GB card (rita A100-80, js2h100 H100-80) or a core change that chunks that build.
  `scripts/js2/` works unchanged on a bigger VM (set `JS2`/`JS2_ROOT` in `config.sh`).
- rita 1352007 (2026-10-04 00:20, core bf58aa6, padded/grid, seed 1): pre-flight |d| = 0, state-only
  checkpoints work. At 36.6 h: 2,595 it, 465k calls, dlogz 10.05. **60 iterations (~2290–2390) took
  392k calls (84%)**, up to 22k calls per iteration (multi/unif stall); before and after, ~50 calls/it.
  The likelihood holds at 0.28 s/call. Live points at the 10-05 snapshot reach Δμ_χ = 0.297 (prior
  edge 0.30). Resume job 1359937 (`--dependency=afternotok:1352007`) continues from the checkpoint if
  the 48 h limit hits.
- **2026-10-05 19:00 durable checkpoint** (owner restarting the session): it 2833 / 533,529 calls,
  sha256 4c076aaf…18a0, verified to restore, copied to
  `queue/a13core_dynesty_n200_s1.save.durable_20261005_183906` and
  `/hildafs/projects/phy230054p/magana/gws-agn-data/checkpoints/analysis_13_joint_shared_widths/`.
  1352007 keeps running on rita (wall limit 2026-10-06 00:20, it 2878 / dlogz 6.0 at 18:54). If it
  ends without finishing, 1359937 resumes from `queue/a13core_dynesty_n200_s1.save`. To resume by
  hand, `sbatch --export=ALL,SEED=1 scripts/submit_a13core_gpu.sbatch`. If the live .save is ever
  lost or corrupt, copy the durable file over it first. On success the results land in
  `results/a13core_dynesty_n200_s1.{json,npz}`.
- **2026-10-05 23:10 second stall:** 1352007 at it 2885 / 588,771 calls / dlogz ≈ 5.95 (43.9 h of 48 h).
  Since the 18:39 durable checkpoint: 52 iterations for 55k calls (~1,060 calls/it; single
  iterations up to 11.4k calls). Cause: multi/unif with dynesty 2.1.4's default bootstrap = 5 —
  dynesty warns the bootstrap enlargement factor is "very large". Live points (200): logL spread
  −4291.14 to −4285.66; Δμ_χ up to 0.2972 (edge 0.30) — still on the edge; f_AGN 0.15–0.38,
  H0 63.2–72.2. 1359937 resumes on the 00:20 wall limit with the SAME settings. Changing the
  sampler mid-run (bootstrap = 0, or rslice) is waiting on the owner.
- **2026-10-05 23:12 owner: resume with rslice.** `scripts/a13_switch.py` rewires the restored
  MultiEllipsoidSampler to rslice (slices 3 + ndim = 11, enlarge 1.25, bootstrap 0, bound update every
  2·slices·nlive = 4,400 calls; idempotent; the record goes to `queue/a13core_dynesty_n200_s1.sampler_switch.json`
  and into the results JSON as `sampler_switch`). Driver flag `--resume-sample rslice`; sbatch passes
  `RESUME_SAMPLE`. Toy check (8-D truncated Gaussian near an edge, unif to it 1500 then rslice, 3 seeds):
  pulls vs analytic logZ +0.76/−0.92/+0.32 (plain unif +1.11/−0.68/+0.65), ~55 calls/it after the switch.
  Dry run on the real checkpoint (it 2885): one ellipsoid; dropping bootstrap shrinks the bound's
  log-volume from −5.12 to −7.47, and all live points stay inside.
  Pre-switch durable copy: `queue/*.save.durable_20261005_231134_preswitch` (+ phy230054p checkpoints dir).
  1359937 cancelled; 1352007 cancelled 23:12 (7 min after its last checkpoint, mid-stall);
  **rslice resume = rita job 1361503** (`sbatch --export=ALL,SEED=1,RESUME_SAMPLE=rslice scripts/submit_a13core_gpu.sbatch`;
  resubmit the same way to continue). Expect ~1.5–2k more iterations at ~55 calls/it ≈ 8–10 h.
- **2026-10-06 owner decisions:** no second sampler (Nautilus removed; smoke gate S withdrawn);
  seed 2 HELD until seed 1 is reviewed; the Δμ_χ prior edge is decided after seed 1; seed-1 figures
  when it finishes; plan the core e7c3007 move (no calibration runs). rslice pace at it 2942:
  ~59 calls/it, 16.7 s/it, dlogz 6.04.
- **2026-10-06 03:30 Δμ_χ edge mode.** Interim posterior (checkpoint it 3595, live added, ESS 369):
  main mode H0 67.5 [65.0, 69.7], f 0.263, Δμ_χ 0.162 [0.123, 0.201], σ_G 4.81, σ_χ 0.082, all away
  from the box edges; plus a SEPARATE clump at Δμ_χ 0.29–0.30 (3.8% of the weight, nothing in
  0.23–0.28) with H0 ≈ 71.6, Δμ_G ≈ 7, σ_χ ≈ 0.12, holding the run's max logL (−4282.55, 1.1 nats
  above the main mode's best), weight dominated by one live point. Closure K3: N_eff ≈ 3e5 at the
  main mode (σ²_sel = N²/N_eff ≈ 3) vs ≈ 8e3 at Δμ_χ 0.30 (σ²_sel ≈ 120, σ ≈ 11 nats) → the spike
  regime selection/gw.py's docstring describes. The run's `max_likelihood_variance = 1e6` leaves only
  the 5·N_obs = 5000 floor. NOTE: the default cap 1 would need N_eff > 1.04e6 — it rejects the
  main mode too. Owner: let seed 1 finish; check after it. `scripts/a13_edge_check.py` (af896ca cell,
  reports N_eff): edge points, main-peak points, 300 posterior draws, cap scan → rita job 1361532
  (afterany:1361503) → `diagnostics/a13_edge_check.json`.
- **Owner 2026-10-06:** fix = variance guard on the rerun, cap chosen from a13_edge_check's scan
  (smallest cap that removes the edge clump and cuts ≤ 1% of the main-mode draws); Δμ_χ box stays
  [−0.05, 0.30]. The rerun of seed 1 waits for the check and the owner's go.
- **2026-10-06 12:23 SEED 1 FINISHED** (rita 1361503; core bf58aa6; unif to it 2885, rslice after):
  logZ −4303.18 ± 0.33, 5,604 it, 753,243 calls (0.28 s/call → ~59 GPU-h total, ~46 of them in the
  unif stalls), 4,364 −inf (guard floor), 5,804 equal-weight samples, ESS 1,570. 90% (5/50/95):
  H0 64.99/67.51/69.63; f 0.184/0.263/0.330; μ_G 34.95/35.64/36.30; Δμ_G 2.79/4.29/5.99;
  μ_χ −0.025/−0.011/0.006; Δμ_χ 0.122/0.159/0.199; σ_G 4.30/4.81/5.24; σ_χ 0.068/0.082/0.099.
  The Δμ_χ edge clump survived: 3.3% of the weight at 0.28–0.30 (nothing in 0.23–0.28), H0 there
  69.8/70.6/72.2, and the run's max logL −4277.01 sits in it (6.3 nats above the main mode's best,
  −4283.26). Main mode alone: H0 64.96/67.44/69.15. Edge check 1361532 next.
- **2026-10-06 13:10 figures + cost.** `scripts/make_figures.py` → `figs/fig_13_marginals.{pdf,png}`
  (eight marginals vs 11D, 90% strips; 24/24 drawn values match the JSON) and
  `figs/fig_13_edge.{pdf,png}` (Δμ_χ–H0 plane, 90% region + the edge clump). A11 REPORT §11 now
  carries the measured cost (753k calls ≈ 59 GPU-h; rslice from the start ≈ 25–30 GPU-h; 50
  realisations ≈ 1,250–1,500 GPU-h). Edge check 1361532 PENDING (Resources): rita's GPUs are held
  by phase12u 1361525 and rita-darksirens 1361294 (other sessions).
- **2026-10-07 07:40 edge check DONE** (rita 1361532, 1 h 31 min; `diagnostics/a13_edge_check.json`, af896ca cell,
  points from the finished seed-1 results). The edge clump's best points sit at **N_eff ≈ 5,000 — exactly
  the 5·N_obs floor, the only guard active** (σ²_sel ≈ 200, σ ≈ 14 nats); 203 of 893 points with
  Δμ_χ > 0.26 are −inf (the floor). Main-mode peak: σ²_sel 3.1–3.8. 300 posterior draws: main-mode
  σ²_sel 2.6/3.7/5.5 (5/50/95%), max 10.8; the 7 clump draws all ≥ 183. Cap scan on the draws:
  cap 5 cuts 37 (12%, main mode); cap 10 cuts the 7 clump draws + 1 main draw (Δμ_χ 0.206; 0.34% of
  the main mode); caps 20–100 cut exactly the 7 clump draws and no main draw. By the registered rule
  (smallest cap removing the clump with ≤ 1% main cut) → **cap 10**; cap 20 is the margin-safe
  alternative (zero main cut, clump ≥ 183). Side note: af896ca − core logL = +3.86 (max 4.01) at the
  main peak (an offset; the posterior-level equivalence holds), +4.5 median / up to 11 at the edge.
