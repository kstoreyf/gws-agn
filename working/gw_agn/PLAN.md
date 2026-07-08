# PLAN — multitracer dark sirens: joint (alpha_agn, H0) proof of concept → production

Date 2026-07-08. Orchestrator: Fable (paper-orchestrator discipline). Spec: `working/gw_agn/GOAL.md`.
Recon: `working/gw_agn/RECON.md` (all [VERIFY P0] resolved). Repo: gws-agn @ `f92741a` (master).

## 1. Context

- Full pipeline exists in `code/` (mock → inject → PE clouds → pixelize → JAX 2-D (H0, alpha_agn)
  grid). No `src/` package yet. Envs: `glassenv` (make_mocks only), `jax` (everything else).
- The pipeline has a **known, diagnosed SEV-1**: H0-dependent β(H0) division mismatched to the
  mock's fixed-z_max_gw host cut, missing numerator truncation, truth-centered PE clouds
  (RECON §SEV-1). This is the cause of the repo's months-long "still biased" history. It must be
  fixed and gate-verified (G2) before any fagn work — it is GOAL's √2-class gate, hit proactively.
- Toy evidence: pipeline variant biased (−1.5 @ dLunc=0.1, N=200); refined 4×4 sweep
  (dLunc × {pipeline,exact}×{truth,obs-centered}) running in background (scratchpad, task bd2onxtla).

## 2. Headline decision

Deliverable claim: **a gate-validated multitracer (GAL+AGN) dark-siren pipeline that recovers
alpha_agn jointly with H0 on GLASS mocks with calibrated coverage, plus its measured information
content σ(alpha_agn | N, contrast)**. A null (prior-dominated alpha_agn at N=50) is a reportable
outcome — G3a forecasts it before the full run; we escalate N rather than tune the mock.
Assumption costs (state in RESULTS.md): fixed population; complete catalogs; selection = hard
true-z cut (P_det=1{z≤z_max_gw}, matched exactly in the estimator — no SNR realism); lognormal
LSS with linear bias; PE = Gaussian dL clouds + Gaussian sky (no waveform realism); Om0 fixed.
Truth scored against **alpha_agn**, never f_agn.

## 3. Workstreams

- **WS1 — estimator consistency fix** (Fable implements; gate E1 = sweep verdict).
  Scope: `code/run_inference.py` (numerator indicator 1{z_s≤z_max_gw}; replace β(H0) with
  H0-independent per-tracer constants β_X=CDF_X(z_max_gw) in the mixture denominator, config
  switch `selection_mode: fixed_z | dl_horizon(legacy)`), `code/generate_gwsamples.py`
  (obs-centered clouds, config switch `pe_centering: obs | truth(legacy)`). Legacy behavior
  preserved behind flags for A/B. Est: 2–3 h.
- **WS2 — mocks + injections (P1, glassenv)**. GLASS catalog with contrast:
  nbar_gal=1e-2, nbar_agn=1e-4 arcmin⁻² (~1.5e6 vs ~1.5e4), bias_gal=1.2, bias_agn=2.0,
  z∈[0,1.5], nside=64, seed=101. Injection sets: coverage sets (f_agn=0, λ=0.5, N_gw=100,
  25 seeds; AGN-channel sets f_agn=1 for the AGN-tracer G2 runs) + recovery sets
  f_agn∈{0,0.3,0.7,1.0}, λ=0.5, N_gw=1000 (subsampled for N-scaling). Gate G1: planted AGN-host
  count == round-based expectation exactly (deterministic split) and pool respects z≤z_max_gw.
  Uniform-catalog null set as control. Est: ≲1 h wall (background).
- **WS3 — GW samples + pixelization (P2, jax)**. Obs-centered clouds, N_samples=1000/event,
  dLunc=0.1 fiducial (+0.3 robustness set), σ_ra=σ_dec=0.01 rad. Pixelize both tracers, nside 64.
  Gate G0: log 100% of injected hosts at z<z_max_gw=1.0<z_complete=1.5. Est: <1 h (background).
- **WS4 — single-tracer H0 coverage (P3, gate G2)**. GAL-only (f_agn=0 sets, alpha=0) and
  AGN-only (f_agn=1 sets, alpha=1) 1-D H0 scans on ≥20 injection realizations each (fixed
  catalog; realization = injection+noise seed). Pass: truth in 68%/90% CI at nominal rate
  (binomial err at n=25: 68%→±9%, 90%→±6%). Smoke gate S1 first: one-realization grid timing +
  posterior sanity. Fallback: bug hunt (this catches any residue WS1 missed).
- **WS5 — joint (alpha_agn, H0) grid (P4, gates G3a→G3)**. G3a: curvature/Fisher σ(alpha) vs N
  from one high-N realization BEFORE committing the batch; sets N (escalate past 50 if
  prior-dominated). G3: recovery at the four injections, coverage + posterior-narrower-than-prior
  for the non-trivial ones. Report fagn–H0 degeneracy + whether AGN tracer tightens H0.
- **WS6 — productionize (P5, gates G4/G5)**. Multitracer API in the darksirens library repo
  (tracer-pair-agnostic mixture likelihood). Opus designs API + my review; Sonnet implements.
  G4: bit-for-bit parity with WS5 numbers on same inputs. G5: second pair (GAL/GAL-subsample
  "BGS/LRG-like") config-only.
- **WS7 — close-out (P6)**. Fresh-Opus adversarial module review (SEV-1 target: β exactly once,
  per-tracer mixture denominator, weights once, Jacobians); claims-vs-numbers check (Sonnet);
  RESULTS.md (Fable); commits per workstream; memory checkpoint.

## 4. Execution order & delegation

WS1 (Fable, now) ∥ WS2 (background, launch immediately) → WS3 → S1 smoke → WS4 (G2) → WS5
(G3a→G3) → WS6 (G4,G5) → WS7. Long runs always in background; grid runs on the A100 (jax env).
Commits: per workstream, on master, descriptive messages, **never push** (user pushes).
Sonnet agents: run-scripts/coverage loops/figures. Opus: WS6 design + WS7 module review.

## 5. Key files

Modified: `code/run_inference.py`, `code/generate_gwsamples.py` (flagged, legacy-preserving).
New: `working/gw_agn/{RECON,PLAN,GATES,RESULTS}.md`, `gates_report.json`, `configs/*.yaml`,
`data/` (mocks+samples, gitignored if large), `scripts/` (coverage loop, forecast, plots),
`figs/`. P5: new `src/darksirens/` package in the darksirens repo (separate commits there).

## 6. Verification (instantiated close-out checklist)

1. Gate ladder G0,G1,E1,S1,G2,G3a,G3,G4,G5 all verdicted in GATES.md + gates_report.json
   (env freezes recorded). 2. No number quoted that a gate invalidated; superseded values
   flagged. 3. Adversarial module review findings all fixed or noted-not-a-bug. 4. One headline
   number (σ(alpha_agn) at chosen N, or a coverage rate) reproduced independently by me from raw
   grid outputs. 5. RESULTS.md states claim + assumption costs verbatim from §2. 6. Memory
   updated (final numbers, superseded flags, user-only items: review commits, push).

## 7. Risks & fallbacks

- Sweep contradicts exact-estimator prediction → stop, re-derive analytics before WS1 lands (E1).
- alpha_agn prior-dominated at N=50 → G3a escalates N (mock has 1000-event sets); if still null
  at N=1000, report information limit honestly (GOAL explicitly allows).
- GLASS/CAMB env or memory failure → bring up machinery on `_uniform` mocks (same schema),
  debug GLASS in parallel; PoC *requires* GLASS for clustering contrast, uniform is control-only.
- Grid cost blows up (2-D × realizations) → GPU, coarser alpha grid, fewer realizations for G3
  (coverage from G2 already validates calibration machinery).
- AGN pool too small for f_agn=1 × N_gw=1000 without replacement (~15k eligible) → fine (≤7%);
  if contrast redesign shrinks pool, cap N_gw or raise nbar_agn.
- Session death → durable state: this file, GATES.md, RECON.md, configs, run logs; memory
  checkpoints at every gate verdict.
