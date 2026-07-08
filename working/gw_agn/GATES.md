# GATES — multitracer fagn program

Numeric pass criteria stated BEFORE running (paper-orchestrator Phase 2). Machine-readable
mirror: `gates_report.json`. Repo: gws-agn. Program plan: `PLAN.md`; spec: `GOAL.md`.
Envs: `glassenv` (make_mocks only), `jax` (all else). Cosmology truth: H0=67.74, Om0=0.3075.

| Gate | What | Pass criterion (pre-stated) | Fallback | Verdict |
|---|---|---|---|---|
| E1 | Toy-level estimator verdict: exact fixed-z estimator (numerator indicator + constant β_X, obs-centered PE) is unbiased where the legacy pipeline (β(H0), truth-centered) is biased | exact-variant mean-MAP bias consistent with 0 within 2·SE at every dLunc ∈ {0.05,0.1,0.2,0.3}; legacy variant shows ≥3·SE bias somewhere | re-derive analytics before WS1 ships | PENDING — sweep task bd2onxtla running (scratchpad toy, NR=120) |
| S0 | Machinery bring-up: uniform tiny mock end-to-end (mock→inject→samples→pixelize→2-D grid) under `selection_mode: fixed_z` | chain exits 0; grid finite at >90% of cells; MAP within grid interior (not on H0 edge) | debug before any physics runs | **PASS(machinery)/FINDING(estimator)** 2026-07-08 — chain exit 0, 60 s/grid (56 ms/eval, 51×21, N_gw=60×500 samp). −inf cells are ONLY full rows at H0≥82: the hard truncation edge from the highest-z event (real information, not a defect). FINDING S0-a: at the legacy default Dz=1e-4 the per-object KDE is a delta-spike mixture the PE-sample MC cannot resolve → sparse-tracer (AGN) Jensen bias: α̂=0.33 vs truth 0.505 (outside 90%). Dz ladder {1e-3,3e-3,1e-2} on the same realization: α̂ plateau 0.497/0.506/0.509, truth in 68% at all three. H0 median wanders 67.1–71.8 across Dz arms with ±0.8 CIs → landscape jaggedness; calibration deferred to G2 (that is what G2 is for). |
| G0 | Completeness shell | 100% of injected hosts have true z ≤ z_max_gw=1.0 < z_complete=1.5 (logged at injection) | restrict/document | PENDING |
| G1 | Planting correctness | planted AGN-host count == N_gw − round(frac_gal·N_gw) exactly (deterministic split); frac from `compute_gw_host_fractions` on the z≤z_max_gw ELIGIBLE pools; all host z ≤ z_max_gw | build/fix planting before inference | PENDING |
| S1 | Single-realization grid smoke on GLASS catalog | grid finite, H0 MAP in interior, wall-time per grid recorded (budget input for G2 batch) | shrink grid / GPU / optimize | PENDING |
| G2 | Single-tracer H0 coverage (the √2-class selection gate) | ≥20 injection realizations per tracer (GAL at f_agn=0, AGN at f_agn=1, λ=0.5, N_gw=100, dLunc=0.1): truth-in-68% rate within 68%±2σ_binom (n=25 → [0.49,0.87]) AND truth-in-90% rate within 90%±2σ_binom ([0.78,1.0]); no systematic one-sided MAP offset >3·SE | selection-function bug hunt; do NOT proceed to joint work | PENDING |
| G3a | Information forecast before committing N | curvature/Fisher σ(alpha_agn) at truth from one N=1000 realization, scaled √(N'/N); if σ(alpha) at N=50 > 0.25 (posterior spans most of [0,1]) escalate N to where σ(alpha) ≤ 0.15 or declare null-risk explicitly | report information limit; do not tune mock | PENDING |
| G3 | alpha_agn recovery + coverage | injections f_agn ∈ {0,0.3,0.7,1.0} (λ=0.5): truth (α_true from eligible-pool counts) inside 68/90% CIs at nominal rates across realizations; posterior narrower than prior for non-trivial injections | if posterior ≈ prior: REPORT AS NULL RESULT | PENDING |
| G4 | Production parity | new `src/darksirens` multitracer API reproduces the P4 grid **bit-for-bit** (max |Δlog L| = 0 on same inputs/dtype) | no refactor ships until parity | PENDING |
| G5 | Tracer-pair generality | production API runs on a second pair (GAL/GAL-subsample) config-only, sane posterior | fix API generality | PENDING |

## Environment freeze (recorded at program start, 2026-07-08)

- gws-agn @ f92741a + WS1 edits (this program's commits).
- jax env: jax 0.4.34 (A100-80GB available); glassenv: glass 2025.1 + camb (imports verified).
- Toy sweep provenance: scratchpad `gwsagn_bias_toy.py` (v1: N_gw=200, dLunc=0.1, 60 reals:
  pipeline-config bias −1.55±2.04 … obs+indicator+β(H0) worst at +9.97 — β(H0) division is
  the dominant mechanism; v2 sweep = E1 evidence, pending).

## Owner decisions

- 2026-07-08 (Fable): `selection_mode: fixed_z` is the program default; `dl_horizon` retained
  ONLY as a labeled-legacy A/B arm. Basis: analytic score argument (fixed-true-z selection ⇒
  H0-independent normalization) + toy v1; to be countersigned by E1.
- 2026-07-08 (Fable): PE clouds obs-centered by default (`pe_centering: obs`);
  `truth` retained as labeled-legacy. The repo's own commented-out "NEW METHOD ATTEMPT" was
  the correct implementation.
- 2026-07-08 (Fable): recovery is scored against **alpha_agn computed from the z≤z_max_gw
  eligible pools** (matches `inject_gw_sources`), never f_agn, never full-catalog counts.
- 2026-07-08 (Fable): production estimator knobs **Dz_gal=Dz_agn=3e-3, N_samples_gw=2000**.
  Basis: MC-resolution criterion n_samp·Dz/σ_z(PE) ≳ 100 (per-spike ~10% MC error) with
  smoothing systematic (Dz/σ_z)²/2 ≈ 0.2% — negligible; S0 Dz-ladder α̂ plateau confirms.
  Legacy Dz=1e-4 is UNRESOLVED (S0-a finding) — do not run physics at 1e-4. If G2 coverage
  fails at these knobs, fallback = galaxy-sum estimator refactor (evaluate the smooth
  per-event z posterior AT catalog z_j instead of catalog KDE at sample z_s).
