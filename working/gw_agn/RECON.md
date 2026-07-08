# RECON — Phase 0 findings (multitracer fagn program)

Date: 2026-07-08. Orchestrator: Fable. Repo commit at recon: `f92741a` (master, pulled, up to date).
Environments verified on this node: conda envs `glassenv` (GLASS/CAMB mocks) and `jax`
(everything else) under `~/.conda/envs` (tmp_ondemand symlink path). Activation pattern:
`module load conda; conda activate glassenv` (see `code/run_mock_pipeline.sh`).

## Repo layout (actual)

- `code/` — the whole pipeline: `make_mocks.py` (GLASS or uniform mocks + GW host injection),
  `generate_gwsamples.py` (PE-cloud generation), `pixelize_catalogs.py` (HEALPix per-pixel z lists),
  `run_inference.py` (JAX likelihood; **2-D (H0, alpha_agn) grid already implemented**),
  `generate_configs.py` (YAML config factory), `plotter.py`, `utils.py`.
- **No `src/` exists.** GOAL.md's `src/darksirens/` production package is NEW work (P5).
- **"A1"** = the darksirens library repo's mock end-to-end validation machinery
  (`/hildafs/projects/phy230014p/magana/src/darksirens`, `scripts/mock_dark_sirens/` + gate A1 in
  `working/GATES.md`): full mass assignment + SNR selection + dynesty/tinyns inference. It is the
  *guidance reference* for GW-sample realism; the gws-agn PoC pipeline is deliberately simpler.
- `notebooks/inference/{complete_catalog, incomplete_catalog_radialsel_noskysel}/` — existing runs.
  Recent git history = a bias hunt: "still biased" (`d29822e`), "fix dlunc=0 issue attempt"
  (`0e2ca5a`), "fixed dL uncertainty to be propto dL" (`f92741a` = HEAD).
- Configs/data dirs (`../configs/`, `../data/` relative to `code/`) are not in this checkout —
  our program generates its own under `working/gw_agn/`.

## [VERIFY P0] resolutions

1. **Host-selection weight w_j** — uniform per object within each catalog. The field channel
   weights AGN vs galaxies by `lambda_agn`: W_agn = λ·N_agn, W_gal = (1−λ)·N_gal
   (`utils.compute_gw_host_fractions`). No luminosity weighting anywhere.
2. **fagn convention (adopt theirs)** — `f_agn` = *AGN-channel* fraction (hosts planted
   deliberately in AGN); the remaining (1−f_agn) draw from the weighted combined field, which can
   still land on AGN. The **inference parameter is `alpha_agn`** = total AGN-hosted fraction:
   `alpha_agn = f_agn + (1−f_agn)·W_agn/(W_agn+W_gal)`. **Recovery must be scored against
   alpha_agn (α_true), never against f_agn** — comparing α̂ to f_agn is a built-in pseudo-bias.
3. **Masses** — generated (Gaussian m~35±5, PE width 1.5) and written to the sample files
   (`m1det/m2det`) but **never used in the likelihood and there is NO SNR selection**. Selection
   in the mock = hard cut on TRUE host z ≤ z_max_gw (`inject_gw_sources`). So "masses only via
   P_det" holds trivially; the real selection issue is item 6 below.
4. **AGN⊂GAL vs disjoint** — disjoint by construction: two independent Poisson draws
   (`positions_from_delta`) from the SAME lognormal matter realization with separate biases.
   No double-counting; §3.2's independently-normalized-field mixture applies with α=alpha_agn.
5. **Shared density field & planting (G1 machinery)** — BOTH exist. GLASS builder materializes one
   `matter` shell list and populates galaxies and AGN from it with `bias_gal`/`bias_agn`;
   `inject_gw_sources(f_agn, lambda_agn, N_gw, seed_gw, z_max_gw)` plants hosts at known truth.
   A `_uniform` (structureless) catalog variant exists — useful as a null/control.
6. **Depth / completeness** — mock catalogs are complete by construction over z∈[0, z_max=1.5];
   GW hosts restricted to z ≤ z_max_gw = 1.0. G0 is satisfied by construction *provided* the
   likelihood's selection treatment matches the z_max_gw cut (see SEV-1 below). Current test
   configs: uniform mock, nbar_gal = nbar_agn = 1e-2 arcmin⁻² (~1.5e6 each), f_agn=0, λ=0 —
   i.e. **zero density and zero bias contrast**: fine for the H0-bias hunt, useless for fagn
   information. The PoC mocks must reintroduce contrast (rarer, more-biased AGN).

## Pre-existing SEV-1: the selection/β mismatch (root of the "still biased" history)

Carried over from the 2026-07-08 bias investigation (this session, toy at scratchpad
`gwsagn_bias_toy.py`; refined sweep running):

- The mock selects hosts on TRUE z at fixed z_max_gw ⇒ P_det = 1{z ≤ z_max_gw} ⇒ the exact
  per-tracer normalization β_X = CDF_X(z_max_gw) is **H0-independent**.
- `run_inference.py` instead divides by an H0-dependent β(H0) built from
  z_max_det(H0) = z(dL_max|H0) (`precompute_beta_cdf`, applied at line ~1406 rationale block).
  That injects score −N·∂logβ/∂H0 ≠ 0 at truth ⇒ H0 biased (toy: −1.5 at dLunc=0.1 for the
  pipeline variant; other variants up to +10).
- The numerator is ALSO missing the truncation: N_X must carry the indicator
  1{z(dL_s|H0) ≤ z_max_gw} against the SAME fixed z_max_gw used at injection.
- PE clouds are truth-centered (`generate_gwsamples.py:~495`; the obs-centered block is
  commented out under "OLD MAYBE WRONG CAUSING BIAS??" — the commented code is the RIGHT one:
  d_obs ~ N(d_true, σ), samples ~ N(d_obs, σ)).

Consistent estimator for this mock (single source of truth for P2–P4):
`N_X(d_i|H0) = mean_s[ p_X(z_s) · 1{z_s ≤ z_max_gw} / (ddL/dz)|_s ]`, `z_s = z(dL_s|H0)`, and the
mixture denominator uses the H0-independent constants `β_X = CDF_X(z_max_gw)` per GOAL §3.2
(they cancel for single-tracer H0 but set the relative AGN/GAL normalization for alpha_agn).

## Other landmines noted

- `(1+z)^γ` tracer weighting exists (γ_agn, γ_gal; default 0) — leave at 0 for PoC.
- `p_pe = 1` flat-PE-prior convention (loader sets ones) — consistent with Gaussian-in-dL clouds
  ONLY when clouds are obs-centered posterior draws under a flat prior; fine after the fix.
- Pixel weighting in the mixture: per-pixel W_pix/W_total factors (OPTION-4 in
  `compute_darksiren_log_likelihood`) — adversarial review target: weights applied exactly once.
- `N_gw_agn = N_gw − round(frac_gal·N_gw)` — deterministic split, not binomial: G1's Poisson
  check must use the deterministic expectation.
